//! Light and weather of the second set: `UVRD`, `SUN`.
//!
//! Ports the UV ray's per-element reactions (`UVRD.cpp:53-108`, emitted by
//! `SUN` and travelling through `FILT`) and the sun's heat / UV emission
//! (`SUN.cpp:51-93`).
//!
//! Both `Update`s are slot parallel over the live pool. `UVRD` is a
//! `TYPE_ENERGY` particle, so it has no `pmap` entry of its own: its pixel
//! comes from `part_x`/`part_y` (`UPDATE_FUNC_ARGS` passes
//! `(int)(parts[i].x + 0.5f)`), exactly like the other energy kernels, while
//! the 3x3 scan reads the matter layer — the reference also reads its single
//! `pmap`, which only ever holds matter. Motion is the generic `TYPE_ENERGY`
//! velocity integration of the move pass, so these kernels only run the
//! neighbour reactions the reference performs.
//!
//! # Deviations
//!
//! * `SUN.cpp:83-88` sets `FLAG_SKIPMOVE` on a fresh `UVRD` when the new slot
//!   id is above the sun's. The creations are drained after this launch and
//!   their slots come from the device window, so they bear no order
//!   relationship to the sun's slot; the flag is not set and the rays move on
//!   their creation frame — the same deviation `electronics/switches.rs:480`
//!   documents for the `PCLN` photon burst.
//! * `UVRD.cpp:59` reads the neighbour's element from `pmap`, whose type bits a
//!   `part_change_type` of an earlier pass does not rewrite; the port reads the
//!   authoritative `part_ty` word instead.
//! * The reference's `create_part` lands immediately, inside the ray's own
//!   update, so a second ray reading the same pixel already sees the result.
//!   Here the `PLNT` -> `VINE` record lands in the drain after this pass, so two
//!   rays that see the same plant both queue it (request order replaces TPT's
//!   slot order) and the two rays' direct field writes are unordered. The last
//!   writer still wins the pixel.

// Collapsing a nested `if` changes what the CubeCL macro expands.
#![allow(clippy::collapsible_if)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::PassDims;
use super::super::combustion::chance;
use super::super::electronics::SPARK_LIFE;
use super::super::shared::{
    CR_SET_LIFE, CR_SET_TEMP, CR_SET_VX, CR_SET_VY, CREATE_QUEUE_WORDS, hash_u32, launch_grid,
    pmap_slot, request_create, request_replace, stasis_frozen,
};
use crate::buffers::SimBuffers;

// Element ids compared against inside the kernels (`tpt_core::constants`); a
// kernel body takes device literals, so each is mirrored here and pinned by the
// assertion below.
const PT_NONE_U32: u32 = 0;
const PT_WATR_U32: u32 = 2;
const PT_SPRK_U32: u32 = 12;
const PT_SLTW_U32: u32 = 19;
const PT_WTRV_U32: u32 = 21;
const PT_PLSM_U32: u32 = 28;
const PT_H2_U32: u32 = 32;
const PT_O2_U32: u32 = 33;
const PT_DSTW_U32: u32 = 35;
const PT_PLNT_U32: u32 = 45;
const PT_PSCN_U32: u32 = 50;
const PT_STKM_U32: u32 = 59;
const PT_CBNW_U32: u32 = 73;
const PT_VINE_U32: u32 = 123;
const PT_FILT_U32: u32 = 134;
const PT_STKM2_U32: u32 = 137;
const PT_FIGH_U32: u32 = 159;
const PT_UVRD_U32: u32 = 225;
const PT_SUN_U32: u32 = 226;
const CELL_U32: u32 = 4;

const _: () = assert!(
    PT_NONE_U32 == PT_NONE
        && PT_WATR_U32 == PT_WATR
        && PT_SPRK_U32 == PT_SPRK
        && PT_SLTW_U32 == PT_SLTW
        && PT_WTRV_U32 == PT_WTRV
        && PT_PLSM_U32 == PT_PLSM
        && PT_H2_U32 == PT_H2
        && PT_O2_U32 == PT_O2
        && PT_DSTW_U32 == PT_DSTW
        && PT_PLNT_U32 == PT_PLNT
        && PT_PSCN_U32 == PT_PSCN
        && PT_STKM_U32 == PT_STKM
        && PT_CBNW_U32 == PT_CBNW
        && PT_VINE_U32 == PT_VINE
        && PT_FILT_U32 == PT_FILT
        && PT_STKM2_U32 == PT_STKM2
        && PT_FIGH_U32 == PT_FIGH
        && PT_UVRD_U32 == PT_UVRD
        && PT_SUN_U32 == PT_SUN
        && CELL_U32 == CELL
);

/// `Simulation.cpp:1872`: the spark `create_part` gives a conductor
/// (`parts[index].life = 4`), mirrored for the kernel body.
const SPARK_LIFE_I32: i32 = 4;
const _: () = assert!(SPARK_LIFE_I32 == SPARK_LIFE);

/// `Simulation.cpp:1875`: the dense conductors stop taking the spark's 10 K
/// bump once they are within 10 K of this temperature.
const SPARK_HEAT_CAP_F32: f32 = 673.0;

/// `SUN.cpp:66-69`: the emitted plasma's temperature, `24 + 273.15f`.
const SUN_PLASMA_TEMP_F32: f32 = 24.0f32 + 273.15f32;
/// `SUN.cpp:67-68`: the emitted plasma's `life`.
const SUN_PLASMA_LIFE_I32: i32 = 200;

/// `~PMAPMASK` (`Simulation.cpp:1874`): the id field of a packed map word, with
/// the nine type bits cleared.
const PMAP_ID_MASK_U32: u32 = 0xFFFF_FE00;
const _: () = assert!(u32::MAX - PMAP_ID_MASK_U32 == pmap_type(u32::MAX));

/// Packed `pmap` word at `(x, y)`, `0` outside the canvas.
///
/// `super::super::shared::pmap_at` is the read-only form. This pass also
/// rewrites the map word of a conductor it sparks (`Simulation.cpp:1874`), so
/// it holds the map mutably and cannot hand a `&mut` slice to that helper.
#[cube]
fn pmap_word_at(pmap: &mut [u32], xres: u32, yres: u32, x: i32, y: i32) -> u32 {
    let inside = x >= 0i32 && y >= 0i32 && (x as u32) < xres && (y as u32) < yres;
    // The probe pixel is clamped so the read itself stays in range; its value
    // is replaced by `0` below unless the original pixel was inside.
    let mut cx = x;
    let mut cy = y;
    if !inside {
        cx = 0i32;
        cy = 0i32;
    }
    let mut v = pmap[((cy as u32) * xres + (cx as u32)) as usize];
    if !inside {
        v = 0u32;
    }
    v
}

/// `pmap[y][x] = (pmap[y][x] & ~PMAPMASK) | ty` — the type-bit rewrite
/// `Simulation.cpp:1874` performs when `create_part` sparks a conductor.
#[cube]
fn pmap_set_type(pmap: &mut [u32], xres: u32, yres: u32, x: i32, y: i32, ty: u32) {
    if x >= 0i32 && y >= 0i32 && (x as u32) < xres && (y as u32) < yres {
        let idx = ((y as u32) * xres + (x as u32)) as usize;
        pmap[idx] = (pmap[idx] & PMAP_ID_MASK_U32) | ty;
    }
}

/// `UVRD.cpp:53-108`: the UV ray's 3x3 reactions.
///
/// Every branch is the reference's, in its switch order:
///
/// * `PLNT` sprouts into `VINE` on one roll in forty, through
///   `create_part(ID(r), ...)` — a [`request_replace`] record that re-runs
///   `VINE`'s `DefaultProperties` over the plant's slot.
/// * The water family (`WATR`, `DSTW`, `CBNW`, `SLTW`) is electrolysed while
///   the ray's own `life` is above one: a one-in-two roll each for `H2` and
///   `O2`, both applied as `part_change_type` (the target keeps its fields, and
///   the second roll overwrites a fresh `H2`). Otherwise the water flashes to
///   `WTRV`.
/// * `PSCN` becomes a `SPRK` in place, as `create_part`'s `PT_SPRK`
///   short-circuit (`Simulation.cpp:1855-1878`) does: only an idle conductor
///   sparks, the slot keeps every other field and only takes `type`, `life = 4`
///   and `ctype = PSCN`, the pixel's map word has its type bits rewritten, and
///   the 10 K dense-conductor bump applies below the 673 K cutoff.
/// * `FILT` adds one frame to the ray's own `life`.
/// * `STKM` / `FIGH` / `STKM2` each lose five frames of `life`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn uvrd_kernel(
    pmap: &mut [u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_x: &[f32],
    part_y: &[f32],
    part_flags: &mut [u32],
    create_queue: &mut [Atomic<u32>],
    used_slots: u32,
    xres: u32,
    yres: u32,
    frame_seed: u32,
    legacy: u32,
) {
    let i = ABSOLUTE_POS;
    if i < used_slots as usize {
        if part_ty[i] == PT_UVRD_U32 {
            // `Simulation.cpp:2339-2341`: a particle frozen in an unpowered
            // stasis wall never runs its `Update`.
            if stasis_frozen(part_flags, i) {
                terminate!();
            }
            let px = i32::cast_from(part_x[i] + 0.5f32);
            let py = i32::cast_from(part_y[i] + 0.5f32);
            let seed = hash_u32(frame_seed ^ (i as u32) ^ 0x5556_5244u32);
            // `UVRD.cpp:55-57`: `rx` outer, `ry` inner, centre skipped.
            for n in 0u32..9u32 {
                let dx = (n % 3u32) as i32 - 1i32;
                let dy = (n / 3u32) as i32 - 1i32;
                if dx != 0i32 || dy != 0i32 {
                    let nv = pmap_word_at(pmap, xres, yres, px + dx, py + dy);
                    if nv != 0u32 {
                        let nslot = pmap_slot(nv);
                        // A slot killed earlier in the tick keeps its stale
                        // `pmap` word with `part_ty == PT_NONE`; TPT's
                        // `kill_part` would have cleared the map instead.
                        let nty = part_ty[nslot];
                        if nty != PT_NONE_U32 {
                            let nseed = hash_u32(seed ^ n);
                            if nty == PT_PLNT_U32 {
                                // `UVRD.cpp:62-67`.
                                if chance(nseed, 1u32, 40u32) {
                                    request_replace(
                                        create_queue,
                                        nslot as u32,
                                        PT_VINE_U32,
                                        px + dx,
                                        py + dy,
                                        0u32,
                                        0i32,
                                        0u32,
                                        0.0f32,
                                        0i32,
                                        0i32,
                                        0i32,
                                        0i32,
                                        0u32,
                                        0u32,
                                        0.0f32,
                                        0.0f32,
                                    );
                                }
                            } else if nty == PT_WATR_U32
                                || nty == PT_DSTW_U32
                                || nty == PT_CBNW_U32
                                || nty == PT_SLTW_U32
                            {
                                // `UVRD.cpp:68-87`. `life` is the ray's own
                                // counter, which a `FILT` neighbour of the same
                                // scan may already have raised this frame.
                                if part_life[i] > 1i32 {
                                    if chance(nseed ^ 0x01u32, 1u32, 2u32) {
                                        part_ty[nslot] = PT_H2_U32;
                                    }
                                    if chance(nseed ^ 0x02u32, 1u32, 2u32) {
                                        part_ty[nslot] = PT_O2_U32;
                                    }
                                } else {
                                    part_ty[nslot] = PT_WTRV_U32;
                                }
                            } else if nty == PT_PSCN_U32 {
                                // `UVRD.cpp:88-92` calls `create_part` on the
                                // conductor, which short-circuits into the
                                // `PT_SPRK` path of `Simulation.cpp:1855-1878`
                                // instead of the `DefaultProperties` fill: only
                                // an idle conductor sparks, the slot keeps every
                                // other field, and only the pixel's map word has
                                // its type bits rewritten.
                                if part_life[nslot] == 0i32 {
                                    part_ty[nslot] = PT_SPRK_U32;
                                    part_life[nslot] = SPARK_LIFE_I32;
                                    part_ctype[nslot] = PT_PSCN_U32;
                                    pmap_set_type(pmap, xres, yres, px + dx, py + dy, PT_SPRK_U32);
                                    // `PSCN` is one of the dense conductors and
                                    // `legacy_enable` skips the bump.
                                    let temp = part_temp[nslot];
                                    if legacy == 0u32 && temp + 10.0f32 < SPARK_HEAT_CAP_F32 {
                                        part_temp[nslot] = temp + 10.0f32;
                                    }
                                }
                            } else if nty == PT_FILT_U32 {
                                // `UVRD.cpp:93-97`.
                                part_life[i] += 1i32;
                            } else if nty == PT_STKM_U32
                                || nty == PT_FIGH_U32
                                || nty == PT_STKM2_U32
                            {
                                // `UVRD.cpp:98-104`: five frames off the
                                // stickman. Two rays adjacent to the same
                                // stickman race on the word, where TPT's slot
                                // order subtracts twice.
                                part_life[nslot] -= 5i32;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `SUN.cpp:51-93`: the star's `tmp` pulse, its plasma and UV emission.
///
/// The `tmp` counter (`SUN.cpp:52-58`) counts down and re-arms to 100 on one
/// roll in two hundred. One roll in three hundred writes `+2.0f` into the cell
/// pressure — the reference's `= +2.0f` is a write of the positive literal, not
/// an increment — and emits one `PLSM` above and below the sun, both at
/// `297.15 K` with `life = 200` (`SUN.cpp:61-70`). The `1/90` emission
/// (`SUN.cpp:72-90`) then gives every one of the eight neighbours an
/// independent roll to spawn a `UVRD` along the offset at three pixels per
/// frame. Both creation shapes leave the field defaults to the drain, which
/// reproduces the `UVRD` `Create` hook (`life = 5`, a random 2 px/frame
/// velocity) before the velocity override lands (`UVRD.cpp:129-135`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sun_kernel(
    part_ty: &[u32],
    part_tmp: &mut [i32],
    part_x: &[f32],
    part_y: &[f32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    used_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < used_slots as usize {
        if part_ty[i] == PT_SUN_U32 {
            // `Simulation.cpp:2339-2341`.
            if stasis_frozen(part_flags, i) {
                terminate!();
            }
            let px = i32::cast_from(part_x[i] + 0.5f32);
            let py = i32::cast_from(part_y[i] + 0.5f32);
            if px >= 0i32 && py >= 0i32 && (px as u32) < xres && (py as u32) < yres {
                let seed = hash_u32(frame_seed ^ (i as u32) ^ 0x5355_4E00u32);

                // `SUN.cpp:52-58`: the countdown, then the re-arm; both can
                // fire in the same frame, and the re-arm wins.
                if part_tmp[i] > 0i32 {
                    part_tmp[i] -= 1i32;
                }
                if chance(seed ^ 0x01u32, 1u32, 200u32) {
                    part_tmp[i] = 100i32;
                }

                // `SUN.cpp:61-70`.
                if chance(seed ^ 0x02u32, 1u32, 300u32) {
                    let col = (px as u32) / CELL_U32;
                    let row = (py as u32) / CELL_U32;
                    // `sim->pv[...] = +2.0f` writes the positive literal: the
                    // cell is set, not incremented.
                    air_pv[(row * xcells + col) as usize] = 2.0f32;
                    // `SUN.cpp:64-69`: the two plasma particles at `y + 1` and
                    // `y - 1`; the occupied pixels are rejected by the drain,
                    // which is `create_part(-1, ...)` returning `-1`.
                    request_create(
                        create_queue,
                        PT_PLSM_U32,
                        px,
                        py + 1i32,
                        CR_SET_TEMP | CR_SET_LIFE,
                        SUN_PLASMA_LIFE_I32,
                        0u32,
                        SUN_PLASMA_TEMP_F32,
                        0i32,
                        0i32,
                        0i32,
                        0i32,
                        0u32,
                        0u32,
                        0.0f32,
                        0.0f32,
                    );
                    request_create(
                        create_queue,
                        PT_PLSM_U32,
                        px,
                        py - 1i32,
                        CR_SET_TEMP | CR_SET_LIFE,
                        SUN_PLASMA_LIFE_I32,
                        0u32,
                        SUN_PLASMA_TEMP_F32,
                        0i32,
                        0i32,
                        0i32,
                        0i32,
                        0u32,
                        0u32,
                        0.0f32,
                        0.0f32,
                    );
                }

                // `SUN.cpp:72-90`: one independent roll per neighbour. The
                // `rx`/`ry` casts are C++'s `(float)(rx * 3)`.
                for n in 0u32..9u32 {
                    let dx = (n % 3u32) as i32 - 1i32;
                    let dy = (n / 3u32) as i32 - 1i32;
                    if dx != 0i32 || dy != 0i32 {
                        if chance(seed ^ (0x40u32 + n), 1u32, 90u32) {
                            request_create(
                                create_queue,
                                PT_UVRD_U32,
                                px + dx,
                                py + dy,
                                CR_SET_VX | CR_SET_VY,
                                0i32,
                                0u32,
                                0.0f32,
                                0i32,
                                0i32,
                                0i32,
                                0i32,
                                0u32,
                                0u32,
                                (dx * 3i32) as f32,
                                (dy * 3i32) as f32,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Launches the weather group's two passes for one reaction step.
///
/// The rays run before the suns. Either order is equivalent for this frame:
/// both creation records drain after the reaction passes, so a fresh ray or
/// plasma first acts on the next frame.
pub(crate) fn launch(bufs: &SimBuffers, dims: &PassDims, seed: u32) {
    if dims.used_slots == 0 || dims.capacity == 0 || dims.pixels == 0 {
        return;
    }
    let (cubes, dim) = launch_grid(dims.used_slots);
    let used = dims.used_slots as u32;
    // `Simulation.cpp:1875` gates the spark's heat bump on `!legacy_enable`;
    // the port models legacy mode as "heat conduction off" (as the nuclear
    // pass does) because `SimulationConfig` carries no separate flag.
    let legacy = u32::from(!bufs.config.heat_enable);
    unsafe {
        uvrd_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            used,
            dims.xres,
            dims.yres,
            seed,
            legacy,
        );
        sun_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            used,
            dims.xres,
            dims.yres,
            dims.xcells,
            seed,
        );
    }
}
