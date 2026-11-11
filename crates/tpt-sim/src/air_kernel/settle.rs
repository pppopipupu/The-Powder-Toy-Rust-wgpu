use super::*;

/// `WL_*` ids the wall-mask pass needs, mirrored for kernel bodies.
const WL_WALLELEC_U32: u32 = tpt_core::constants::WL_WALLELEC as u32;
const WL_EWALL_U32: u32 = tpt_core::constants::WL_EWALL as u32;
const WL_WALL_U32: u32 = tpt_core::constants::WL_WALL as u32;
const WL_GRAV_U32: u32 = tpt_core::constants::WL_GRAV as u32;
const WL_BLOCKAIR_U32: u32 = tpt_core::constants::WL_BLOCKAIR as u32;
const _: () = assert!(WL_WALLELEC_U32 == 1);
const _: () = assert!(WL_EWALL_U32 == 2);
const _: () = assert!(WL_WALL_U32 == 8);
const _: () = assert!(WL_GRAV_U32 == 14);
const _: () = assert!(WL_BLOCKAIR_U32 == 16);

/// Settles the ambient-heat scratch field: `ohv -> hv`.
///
/// TPT's `update_airh` ends with `memcpy(hv, ohv, ...)` (`Air.cpp:248-249`).
/// The pressure and velocity fields are written directly by
/// [`air_smooth_advect_kernel`] and [`air_heat_convect_kernel`], so this pass
/// only mirrors the heat result.
#[cube(launch_unchecked)]
pub fn air_copy_back_kernel(src_heat: &[f32], dst_heat: &mut [f32], cells: u32) {
    let idx = ABSOLUTE_POS;
    if idx < cells as usize {
        dst_heat[idx] = src_heat[idx];
    }
}

/// Rebuilds `bmap_blockair` / `bmap_blockairh` from the wall maps, exactly like
/// the per-frame wall loop in `Simulation::BeforeSim` (`Simulation.cpp:3765-3775`).
///
/// `bmap_blockair` is `1` for `WL_WALL`, `WL_WALLELEC`, `WL_BLOCKAIR` and an
/// unpowered `WL_EWALL`; `bmap_blockairh` is `0x8` for those plus `WL_GRAV`.
///
/// The element-driven half of the reference happens too late for this rebuild:
/// `WALL.cpp:50-51` stamps `1` / `0x8` into the masks from the element
/// `Update`, one frame after the rebuild that precedes it. `elem_block_air` /
/// `elem_block_airh` carry those stamps (the modext pass clears and refills
/// them every tick) and are OR-ed in here, which puts a wall's block exactly
/// one frame behind the wall itself, the same latency the reference's ordering
/// produces. Every element update that stamps the same grids (`RUBR.cpp:76-77`,
/// `ALUM.cpp:141-142`) is folded identically.
///
/// The heat mask is a three-bit saturating counter in TPT: the wall loop resets
/// it to `0`/`0x8` every frame, then `TransitionPhase` increments it for every
/// particle that fails the conduction gate (`Simulation.cpp:2770-2776`) and only
/// bit 3 (`0x8`) means "blocks ambient heat". This pass must therefore run
/// *after* [`air_heat_convect_kernel`] (which reads the previous frame's
/// counters) and before the thermal pass (which increments them).
#[cube(launch_unchecked)]
pub fn air_block_masks_kernel(
    wall_bmap: &[u32],
    wall_emap: &mut [u32],
    block_air: &mut [u32],
    block_airh: &mut [u32],
    elem_block_air: &[u32],
    elem_block_airh: &[u32],
    cells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < cells as usize {
        let wall = wall_bmap[idx] & 0xFFu32;
        // `Simulation.cpp:3779-3780`: the wall electrical state decays by one
        // per frame before the masks read it. The device owns `emap` (sparks
        // charge it in the movement pass), so the decay has to live here; a
        // host-side decay only saw a stale shadow and re-uploaded it every
        // frame while any wall was charged.
        let raw = wall_emap[idx];
        if raw != 0u32 {
            wall_emap[idx] = raw - 1u32;
        }
        let emap = wall_emap[idx] & 0xFFu32;

        let mut blocked = 0u32;
        if wall == WL_WALL_U32
            || wall == WL_WALLELEC_U32
            || wall == WL_BLOCKAIR_U32
            || (wall == WL_EWALL_U32 && emap == 0u32)
        {
            blocked = 1u32;
        }
        // `WALL.cpp:50-51` (and the other stamping elements) assign `1` / `0x8`
        // into the mask cells themselves; folding the element grid with `|`
        // merges every stamp of the cell without disturbing the wall bits.
        block_air[idx] = blocked | elem_block_air[idx];

        let mut heat = 0u32;
        if blocked != 0u32 || wall == WL_GRAV_U32 {
            heat = 0x8u32;
        }
        block_airh[idx] = heat | elem_block_airh[idx];
    }
}

/// Adds each particle's `HotAir` to the pressure of the air cell it occupies.
///
/// TPT (`Simulation.cpp:2347-2358`):
///
/// ```text
/// if (elements[t].HotAir) {
///     if (t == PT_GAS || t == PT_NBLE) {
///         if (pv < 3.5) pv += 4 * HotAir * (3.5 - pv);
///     } else {
///         pv += 4 * HotAir;
///     }
/// }
/// ```
///
/// The reference runs this inside the per-particle update loop, after the
/// ambient-air pass and before `TransitionPhase` and the element `Update`
/// (`Simulation.cpp:2347-2358`), bounded by `parts.active` captured at entry
/// (`Simulation.cpp:2300`), so same-frame creations are never visited. The port
/// runs it once per frame at the end of [`crate::air_kernel::step_air_gpu`]
/// (`crate::SimulationEngine::step`), reading the map the previous movement
/// pass left behind; one thread owns one air cell and walks its `4 x 4` pixels,
/// which reproduces the same accumulation without atomic floating-point adds
/// (WGSL has none) and without a race.
#[cube(launch_unchecked)]
pub fn hot_air_pressure_kernel(
    pmap: &[u32],
    elem_props: &[u32],
    air_pv: &mut [f32],
    xres: usize,
    yres: usize,
    xcells: usize,
    ycells: usize,
) {
    let cell = ABSOLUTE_POS;
    if cell < xcells * ycells {
        let cx = cell % xcells;
        let cy = cell / xcells;

        for dy in 0..4usize {
            for dx in 0..4usize {
                let px = cx * 4usize + dx;
                let py = cy * 4usize + dy;
                if px < xres && py < yres {
                    let entry = pmap[py * xres + px];
                    // `pmap_pack` stores `ty` in the low 9 bits; 0 = empty.
                    if entry != 0u32 {
                        let ty = (entry & 0x1FFu32) as usize;
                        // 11 = `offset_of!(GpuElementProps, hot_air) / 4`.
                        let hot = f32::reinterpret(elem_props[ty * 32usize + 11usize]);
                        if hot != 0.0f32 {
                            if ty == 10usize || ty == 97usize {
                                // `PT_GAS` / `PT_NBLE`: approach a pressure of
                                // 3.5 (`Simulation.cpp:2350-2354`).
                                if air_pv[cell] < 3.5f32 {
                                    air_pv[cell] += 4.0f32 * hot * (3.5f32 - air_pv[cell]);
                                }
                            } else {
                                air_pv[cell] += 4.0f32 * hot;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Compile-time guards for the mirrored constants above.
const _: () =
    assert!(core::mem::offset_of!(tpt_core::elements::GpuElementProps, hot_air) == 11 * 4);
const _: () = assert!(tpt_core::constants::PT_GAS == 10);
const _: () = assert!(tpt_core::constants::PT_NBLE == 97);
