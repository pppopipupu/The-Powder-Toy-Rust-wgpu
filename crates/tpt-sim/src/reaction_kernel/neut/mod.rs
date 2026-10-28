//! `NEUT` — neutrons: fission, transmutation and Cherenkov radiation.
//!
//! Ports the whole neutron `Update` (`NEUT.cpp:56-244`) plus the two neutron
//! behaviours TPT keeps outside it: the `GLAS`/`BGLA` Cherenkov photon from
//! `Simulation::try_move` (`Simulation.cpp:1357-1361`,
//! `Simulation::create_cherenkov_photon` at `2052-2084`) and the `PLUT` fission
//! chain that `NEUT.cpp:77-104` drives.
//!
//! # Design
//!
//! Neutrons are `TYPE_ENERGY` particles, so the pass is pixel parallel over the
//! `photons` layer: one thread per neutron, and every write goes to a
//! neighbour's `part_*` slot. `Simulation::create_part` (`Simulation.cpp:1842`)
//! rebuilds a replaced slot from the target's `DefaultProperties` row;
//! [`rebuild_type_in_place`] copies the same row from `gpu_spawn_defaults` (and
//! defers the scratch words to [`neut_row_reset_kernel`]), and the two `Create`
//! hooks the targets need (`NEUT`'s random lifetime, `LAVA`'s burn timer) are
//! written by the caller.
//!
//! `create_part(-3, x, y, t)` skips the map checks and overwrites the pixel's
//! `photons`/`pmap` word, so every creation that TPT stacks on one pixel — the
//! `DeutExplosion` burst, the Cherenkov photon on the neutron itself — is a
//! queue record with `CR_NO_OCCUPANCY` at that exact pixel.
//!
//! # Port notes
//!
//! * The `photons` map holds one energy particle per pixel, so when TPT stacks
//!   several energy particles on one pixel only the last one keeps the map
//!   entry; the port reproduces that through the drain's record order. The
//!   unmapped particles stay alive in the pool and keep moving, exactly like
//!   TPT's.
//! * The map rebuild picks one of several same-pixel particles per frame, which
//!   is the only visible difference from TPT's "last created owns the word"
//!   bookkeeping.
//! * `Element_FIRE_update(UPDATE_FUNC_SUBCALL_ARGS)` after a successful fission
//!   runs with the *neutron* as the subject. For `t == PT_NEUT` its switch falls
//!   through to `default`, the `THRM`/`COAL`/`LAVA` ring branches are gated on
//!   `t`, and `updateLegacy` only runs in legacy mode (which this port does not
//!   model for `FIRE`, see `combustion::fire`), so the subcall is exactly the
//!   5x5 flammable-ignition branch (`FIRE.cpp:303-321`); [`fire_update_subcall`]
//!   is that branch. It runs after the fission's type change and pressure bump,
//!   like the reference.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::PassDims;
use super::combustion::WORD_DEFAULT_TEMP;
use super::shared::{
    CR_NO_OCCUPANCY, CR_SET_CTYPE, CR_SET_TEMP, CR_SET_VX, CR_SET_VY, CREATE_QUEUE_WORDS,
    WORD_EXPLOSIVE, WORD_FLAMMABLE, elem_word, hash_u32, launch_grid, mix_index, pmap_at,
    pmap_slot, request_create,
};
use crate::buffers::SimBuffers;

/// `u32` words per row of `gpu_spawn_defaults` (mirrored for kernel bodies).
const SPAWN_DEFAULT_WORDS_U32: u32 = 8;
const _: () = assert!(tpt_core::elements::SPAWN_DEFAULT_WORDS == 8);

/// `DeutExplosion` clamps its product count to `1..=340` (`NEUT.cpp:236-241`).
const DEUT_MAX_BURST: u32 = 340;

/// `create_cherenkov_photon`: the photon's wavelength mask
/// (`Simulation.cpp:2072`).
const CHERENKOV_WAVELENGTH: u32 = 0x0000_0F80;
/// `create_cherenkov_photon`: photons travel at `1.269` px/frame
/// (`Simulation.cpp:2081-2083`).
const CHERENKOV_SPEED: f32 = 1.269;

/// `restrict_flt(v, lo, hi)`.
#[cube]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut r = v;
    if r < lo {
        r = lo;
    }
    if r > hi {
        r = hi;
    }
    r
}

/// TPT's `rng.chance(n, d)`: `roll % d < n`, false for a non-positive `d`.
#[cube]
fn chance(roll: u32, n: u32, d: u32) -> bool {
    let mut hit = false;
    if d > 0u32 {
        hit = (roll % d) < n;
    }
    hit
}

/// [`chance`] for the `int(<float> + pressure)` forms, which truncate toward
/// zero and can therefore be non-positive.
#[cube]
fn chance_f32(roll: u32, n: f32, d: u32) -> bool {
    let mut hit = false;
    if n > 0.0f32 && d > 0u32 {
        hit = (roll % d) < (n as u32);
    }
    hit
}

/// Seed a device local from a runtime anchor (AGENTS rule 4).
#[cube]
#[allow(clippy::erasing_op)]
fn seed_u32(anchor: u32, value: u32) -> u32 {
    anchor * 0u32 + value
}

/// A slot rebuilt by [`rebuild_type_in_place`] still needs its `tmp2`..`tmp4`
/// and `dcolour` restored from the element's defaults; the conversion kernel
/// cannot bind those four arrays on top of everything else it reads, so it
/// parks this bit in `part_flags` and [`neut_row_reset_kernel`] finishes the
/// row in a second, slot-parallel launch.
///
/// `Simulation.cpp:1941` (`parts[i] = elements[t].DefaultProperties`) resets
/// the whole row, `flags` included. `FLAG_STASIS` is preserved because it is
/// the port's per-frame copy of the cell state, not a particle field.
pub(super) const FLAG_ROW_REBUILT: u32 = 0x0000_0040;

/// `Simulation::create_part(i, x, y, t)` on an existing slot: the fields every
/// conversion writes straight from the target's `DefaultProperties` row
/// (`Simulation.cpp:1941`).
///
/// This is the shape `combustion::fire` uses; the nuclear/neutron conversions
/// call [`rebuild_type_in_place`], which also marks the slot for the deferred
/// scratch-word reset.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn change_type_in_place(
    spawn_defaults: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    nid: usize,
    ty: u32,
) {
    let db = (ty * SPAWN_DEFAULT_WORDS_U32) as usize;
    part_ty[nid] = ty;
    part_temp[nid] = f32::reinterpret(spawn_defaults[db]);
    part_life[nid] = spawn_defaults[db + 1usize] as i32;
    part_ctype[nid] = spawn_defaults[db + 2usize];
    part_tmp[nid] = spawn_defaults[db + 3usize] as i32;
    part_vx[nid] = 0.0f32;
    part_vy[nid] = 0.0f32;
}

/// [`change_type_in_place`] with `create_part`'s whole-row reset: `flags` is
/// rebuilt (keeping the cell-level `FLAG_STASIS`) and the slot is marked with
/// [`FLAG_ROW_REBUILT`] so [`neut_row_reset_kernel`] restores `tmp2`..`tmp4` and
/// `dcolour` from the element's defaults in the same step.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn rebuild_type_in_place(
    spawn_defaults: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_flags: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    nid: usize,
    ty: u32,
) {
    let db = (ty * SPAWN_DEFAULT_WORDS_U32) as usize;
    part_ty[nid] = ty;
    part_temp[nid] = f32::reinterpret(spawn_defaults[db]);
    part_life[nid] = spawn_defaults[db + 1usize] as i32;
    part_ctype[nid] = spawn_defaults[db + 2usize];
    part_tmp[nid] = spawn_defaults[db + 3usize] as i32;
    part_flags[nid] = FLAG_ROW_REBUILT | (part_flags[nid] & FLAG_STASIS);
    part_vx[nid] = 0.0f32;
    part_vy[nid] = 0.0f32;
}

/// Finish the row reset [`rebuild_type_in_place`] defers: `tmp2`..`tmp4` from
/// the element's `DefaultProperties`, `dcolour = 0` and the
/// [`FLAG_ROW_REBUILT`] marker cleared (`Simulation.cpp:1941-1961`).
///
/// Slot parallel; it touches only the slots whose conversion ran this frame.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn neut_row_reset_kernel(
    spawn_defaults: &[u32],
    part_ty: &[u32],
    part_flags: &mut [u32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_dcolour: &mut [u32],
    num_slots: u32,
) {
    let id = ABSOLUTE_POS;
    if id < num_slots as usize && (part_flags[id] & FLAG_ROW_REBUILT) != 0u32 {
        let db = (part_ty[id] * SPAWN_DEFAULT_WORDS_U32) as usize;
        part_tmp2[id] = spawn_defaults[db + 4usize] as i32;
        part_tmp3[id] = spawn_defaults[db + 5usize] as i32;
        part_tmp4[id] = spawn_defaults[db + 6usize] as i32;
        part_dcolour[id] = 0u32;
        part_flags[id] &= !FLAG_ROW_REBUILT;
    }
}

/// Launch [`neut_row_reset_kernel`] for one frame.
pub(super) fn launch_row_reset(bufs: &SimBuffers, dims: &PassDims) {
    let (cubes, dim) = launch_grid(dims.used_slots);
    unsafe {
        neut_row_reset_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(
                bufs.gpu_spawn_defaults.clone(),
                PT_NUM * tpt_core::elements::SPAWN_DEFAULT_WORDS,
            ),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            dims.used_slots as u32,
        );
    }
}

/// `Neighbourhood::surround_space > 0` (`Simulation.cpp:2283`): at least one
/// empty `pmap` cell in the 3x3 ring around the subject.
#[cube]
fn has_surround_space(pmap: &[u32], xres: u32, yres: u32, x: i32, y: i32) -> bool {
    let mut space = false;
    for n in 0u32..9u32 {
        let dx = (n % 3u32) as i32 - 1i32;
        let dy = (n / 3u32) as i32 - 1i32;
        if (dx != 0i32 || dy != 0i32) && !space && pmap_at(pmap, xres, yres, x + dx, y + dy) == 0u32
        {
            space = true;
        }
    }
    space
}

/// `Element_FIRE_update` (`FIRE.cpp:57-321`) evaluated with the neutron as the
/// burning subject: the `switch (parts[i].type)` falls through to `default` for
/// `PT_NEUT`, and of the 5x5 ring only the flammable-ignition branch
/// (`FIRE.cpp:303-321`) is reachable, so that branch *is* the subcall.
///
/// The `t != PT_SPRK` / `t != PT_PHOT` exclusions are true for a neutron and the
/// `THRM`/`COAL`/`LAVA` branches are gated on the subject's type, so they cannot
/// match. `surround_space` is the `Neighbourhood` value the core computed for
/// the neutron before its update (`Simulation.cpp:2283`), i.e. "at least one
/// empty `pmap` pixel around the neutron".
#[cube]
#[allow(clippy::too_many_arguments)]
fn fire_update_subcall(
    pmap: &[u32],
    elem_words: &[u32],
    air_pv: &mut [f32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    xres: u32,
    yres: u32,
    xcells: u32,
    px: i32,
    py: i32,
    roll: u32,
) {
    let surround = has_surround_space(pmap, xres, yres, px, py);
    let fire_temp = f32::reinterpret(elem_word(elem_words, PT_FIRE, WORD_DEFAULT_TEMP));
    let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
    for n in 0u32..25u32 {
        let dx = (n % 5u32) as i32 - 2i32;
        let dy = (n / 5u32) as i32 - 2i32;
        if dx != 0i32 || dy != 0i32 {
            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
            if nv != 0u32 {
                let nid = pmap_slot(nv);
                let nty = part_ty[nid];
                if nty != PT_NONE {
                    let flam = elem_word(elem_words, nty, WORD_FLAMMABLE) as i32;
                    let explosive = (elem_word(elem_words, nty, WORD_EXPLOSIVE) as i32) > 0i32;
                    if flam > 0i32 && (surround || explosive) {
                        let ncidx = (((py + dy) as u32 / CELL) * xcells + ((px + dx) as u32 / CELL))
                            as usize;
                        let pv = air_pv[ncidx];
                        let roll_n = roll ^ mix_index(n, 0x2545_F491u32);
                        if chance_f32(roll_n, flam as f32 + pv * 10.0f32, 1000u32)
                            && (nty != PT_SPNG || part_life[nid] == 0i32)
                        {
                            part_ty[nid] = PT_FIRE;
                            part_temp[nid] =
                                clampf(fire_temp + ((flam / 2) as f32), MIN_TEMP, MAX_TEMP);
                            part_life[nid] =
                                180i32 + ((hash_u32(roll_n ^ (nid as u32)) % 80u32) as i32);
                            part_tmp[nid] = 0i32;
                            part_ctype[nid] = 0u32;
                            if explosive {
                                air_pv[cidx] += 0.25f32;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `Simulation::create_cherenkov_photon` (`Simulation.cpp:2052-2084`): a fast
/// neutron inside glass emits a photon with the glass's temperature.
///
/// `create_part(-3, nx, ny, PT_PHOT)` skips the map checks and writes the
/// photon's word into `photons[ny][nx]`, i.e. on the neutron's own pixel; the
/// neutron keeps its slot but loses its map entry. The record opts out of the
/// occupancy test for exactly that reason. `parts[i].x/y = parts[pp].x/y` is the
/// neutron's float position, which a creation record cannot carry, so the
/// photon starts on the pixel; the next movement step takes over from there.
#[cube]
pub(super) fn cherenkov_photon(
    create_queue: &mut [Atomic<u32>],
    px: i32,
    py: i32,
    vx: f32,
    vy: f32,
    temp: f32,
    roll: u32,
) {
    // `lr` is -1 or 1; the photon leaves at +-2.5x the neutron's velocity,
    // renormalised to the speed of light in pixels per frame.
    let lr = (roll % 2u32) as i32 * 2i32 - 1i32;
    let raw_vx = vx - (lr as f32) * 2.5f32 * vy;
    let raw_vy = vy + (lr as f32) * 2.5f32 * vx;
    let speed = (raw_vx * raw_vx + raw_vy * raw_vy).sqrt();
    let scale = CHERENKOV_SPEED / speed;
    request_create(
        create_queue,
        PT_PHOT,
        px,
        py,
        CR_NO_OCCUPANCY | CR_SET_CTYPE | CR_SET_TEMP | CR_SET_VX | CR_SET_VY,
        0i32,
        CHERENKOV_WAVELENGTH,
        temp,
        0i32,
        0i32,
        0i32,
        0i32,
        0u32,
        0u32,
        raw_vx * scale,
        raw_vy * scale,
    );
}

mod seed;
mod update;

pub(super) use seed::neut_seed_kernel;
pub(super) use update::neut_reaction_kernel;

/// Launch the neutron pass for one reaction step.
pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        neut_reaction_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(
                bufs.gpu_spawn_defaults.clone(),
                PT_NUM * tpt_core::elements::SPAWN_DEFAULT_WORDS,
            ),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }
    // `create_part` rebuilds the converted slot's whole row
    // (`Simulation.cpp:1941`); the scratch words the conversion kernel cannot
    // bind are finished here, before any later pass reads them.
    launch_row_reset(bufs, dims);
    unsafe {
        neut_seed_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host mirror of the kernel's `(3i32 + pv as i32) as u32`.
    fn pressure_factor_host(pv: f32) -> u32 {
        (3i32 + pv as i32) as u32
    }

    /// Host mirror of the `DeutExplosion` product count.
    fn deut_count_host(life: i32) -> u32 {
        let mut n = life / 50;
        if n < 1 {
            n = 1;
        } else if n > DEUT_MAX_BURST as i32 {
            n = DEUT_MAX_BURST as i32;
        }
        n as u32
    }

    #[test]
    fn pressure_factor_wraps_like_the_cxx_unsigned_sum() {
        // `NEUT.cpp:60` assigns `3 + (int)pv` to an `unsigned int`.
        assert_eq!(pressure_factor_host(0.0), 3);
        assert_eq!(pressure_factor_host(10.9), 13, "(int)10.9 truncates to 10");
        assert_eq!(pressure_factor_host(-3.0), 0);
        assert_eq!(pressure_factor_host(-3.5), 0, "(int)-3.5 truncates to -3");
        assert!(
            pressure_factor_host(-10.0) > 1000,
            "a vacuum must make every fission roll certain"
        );
    }

    #[test]
    fn deut_burst_count_matches_the_tpt_clamp() {
        assert_eq!(deut_count_host(0), 1);
        assert_eq!(deut_count_host(49), 1);
        assert_eq!(deut_count_host(50), 1);
        assert_eq!(deut_count_host(100), 2);
        assert_eq!(deut_count_host(200), 4);
        assert_eq!(deut_count_host(340 * 50), DEUT_MAX_BURST);
        assert_eq!(deut_count_host(1_000_000), DEUT_MAX_BURST);
    }

    /// Host mirror of the burst's creation records: every product is a
    /// `create_part(-3, x, y, PT_NEUT)` on the DEUT's own pixel
    /// (`NEUT.cpp:271-289`), so a burst of any size keeps all of its neutrons on
    /// that one pixel instead of spreading or capping them at a ring size.
    fn deut_burst_pixels(life: i32, x: i32, y: i32) -> Vec<(i32, i32)> {
        vec![(x, y); deut_count_host(life) as usize]
    }

    #[test]
    fn deut_burst_stacks_every_product_on_the_deut_pixel() {
        // The full 1..=340 range lands on one pixel.
        for life in [0, 49, 50, 100, 200, 340 * 50, 1_000_000] {
            let pixels = deut_burst_pixels(life, 40, 41);
            assert_eq!(pixels.len(), deut_count_host(life) as usize);
            assert!(
                pixels.iter().all(|&p| p == (40, 41)),
                "life {life}: the burst must stay on the exploding pixel"
            );
        }
        // A 340-product burst is kept whole: nothing is dropped for exceeding a
        // spread window.
        assert_eq!(deut_burst_pixels(1_000_000, 0, 0).len(), 340);
    }
}
