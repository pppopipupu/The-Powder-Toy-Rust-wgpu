//! Force elements (`SC_FORCE`): `ACEL`, `DCEL`, `RPEL`, `FRAY`, `HEAC` and
//! `DMG`.
//!
//! # Submodules
//!
//! * [`accumulators`]: `ACEL`/`DCEL` velocity multipliers and `HEAC` rapid heat conduction.
//! * [`magnets`]: `RPEL` repellers/attractors and `FRAY` force emitter beams.
//! * [`projectiles`]: `DMG` damage charges and blast-cell stamping.
//!
//! # TPT references
//!
//! * `ACEL.cpp:55-102` — accelerator: multiplies the velocity of its four
//!   orthogonal neighbours by `1.1` (or `1 + clamp(life, 0, 1000) / 100`) and
//!   renormalises the result above `MAX_VELOCITY`; its own `tmp` is the glow
//!   flag.
//! * `DCEL.cpp:55-99` — decelerator: the same scan with `1 / 1.1` (or
//!   `1 - clamp(life, 0, 100) / 100`) and no renormalisation.
//! * `HEAC.cpp:119-179` — rapid heat conductor: sets its own temperature to the
//!   heat-capacity weighted mean of the nine pixels at `rad = 4`, then copies
//!   that value onto those neighbours, skipping anything a heat insulator
//!   blocks (`CheckLine`, `HEAC.cpp:70-124`).
//! * `RPEL.cpp:41-73` — repeller/attractor: eleven random offsets in
//!   `[-10, 10]²` around the source, each pushing the particle there by
//!   `(temp - 273.15) / 10` along the offset's sign, filtered by `ctype`.
//! * `FRAY.cpp:49-88` — force emitter: a `SPRK` next to a `FRAY` fires a beam
//!   of up to `tmp` (default 10) pixels in the opposite direction, pushing
//!   every non-solid particle along the beam by `(temp - 273.15) / 10`.
//! * `DMG.cpp:51-117` — damage charge: a `DMG` with a non-immune neighbour
//!   kills itself and detonates, pushing every particle in a radius-25 disc by
//!   a constant `7` pixels/step away from itself, adding one unit of pressure
//!   per covered particle and applying the element table's high-pressure
//!   transition (plus a few hard-coded pairings).
//!
//! # Kernels
//!
//! | kernel | submodule | bindings | work |
//! |--------|-----------|----------|------|
//! | [`blast_clear_kernel`] | [`projectiles`] | 1 | zero the transient `DMG` blast grid |
//! | [`dmg_blast_kernel`] | [`projectiles`] | 4 | `DMG` trigger, blast stamping, `pv`, self-kill |
//! | [`dmg_apply_kernel`] | [`projectiles`] | 7 | blast impulse and type transitions per particle |
//! | [`accelerator_kernel`] | [`accumulators`] | 8 | `ACEL` / `DCEL` |
//! | [`repeller_kernel`] | [`magnets`] | 8 | `RPEL` |
//! | [`fray_range_clear_kernel`] | [`magnets`] | 1 | zero the per-frame `FRAY` beam bound |
//! | [`fray_range_kernel`] | [`magnets`] | 3 | longest `FRAY` beam of the frame |
//! | [`fray_kernel`] | [`magnets`] | 9 | `FRAY` |
//! | [`heac_kernel`] | [`accumulators`] | 6 | `HEAC` |
//!
//! # Gather instead of scatter
//!
//! TPT updates one particle at a time and *adds* to the velocity of an
//! arbitrary neighbour, which a slot-parallel GPU pass cannot do: two threads
//! adding to the same `vx` lose one of the contributions, and the movement
//! kernel's arbitration assumes a single writer per slot. Every kernel here
//! therefore inverts the relation — a particle resolves the force elements
//! around *itself* and writes only its own slot:
//!
//! * `ACEL`/`DCEL` — the target scans its four orthogonal neighbours for an
//!   accelerator and applies each multiplier in a fixed direction order
//!   (TPT applies them in slot order, so only the f32 rounding of two adjacent
//!   accelerators can differ). The `tmp` glow flag is a second, exact gather:
//!   the accelerator scans its own four neighbours and sets its own `tmp`.
//! * `RPEL` — TPT draws eleven offsets *from the source*; the gather draws
//!   eleven offsets from the *target* and looks for an `RPEL` there. Both give
//!   the same expected number of pushes per (source, target) pair
//!   (`11 / 441`), but the distributions differ: TPT caps the pushes *per
//!   source* at eleven while the gather caps them *per target*, so a target in
//!   a dense `RPEL` field receives at most eleven pushes per frame (TPT can
//!   receive more from different sources), and a source surrounded by targets
//!   pushes fewer. `FRAY`-style beam interactions do not arise because `RPEL`
//!   has no line-of-sight test.
//! * `FRAY` — the target walks the eight rays that pass through it and looks
//!   for `[FRAY][SPRK]` immediately behind it, out to the frame's longest beam
//!   (`fray_range_kernel`), so a 10-pixel beam costs at most 88 `pmap` reads
//!   per non-solid particle and a scene without emitters costs one word read.
//! * `HEAC` — a particle finds the first `HEAC` at `rad = 4` that can see it
//!   and takes that conductor's freshly computed mean. The mean itself is a
//!   gather of the conductor's nine `rad = 4` positions, so one dispatch does
//!   both halves.
//! * `DMG` — a blast is a *scatter* by nature (one charge writes every particle
//!   in its disc), so it is split into a scatter into a per-cell blast grid
//!   ([`dmg_blast_kernel`], one thread per charge) and a gather from that grid
//!   ([`dmg_apply_kernel`], one thread per particle). The grid holds
//!   `(strength << 24) | (x << 12) | y` for the strongest blast covering the
//!   cell, so a cell's particles learn where the blast came from and recompute
//!   their exact offset. This is the one pass here that needs cross-thread
//!   state; `Atomic<u32>::fetch_max` makes the stamping race-free.

pub(super) mod accumulators;
pub(super) mod magnets;
pub(super) mod projectiles;

pub(super) use accumulators::{accelerator_kernel, heac_kernel};
pub(super) use magnets::{
    fray_kernel, fray_range_clear_kernel, fray_range_kernel, repeller_kernel,
};
pub(super) use projectiles::{blast_clear_kernel, dmg_apply_kernel, dmg_blast_kernel};

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::PassDims;
use super::shared::{WORD_PROPERTIES, elem_word, launch_grid, pmap_at, pmap_slot};
use crate::buffers::SimBuffers;

/// Low bits of a packed spatial-map word (the element id).
pub(super) const PMAP_TYPE: u32 = 0x1FF;

/// [`target_slot`] result for a pixel that holds no particle.
pub(super) const NO_TARGET: u32 = 0xFFFF_FFFF;

/// `TYPE_PART | TYPE_LIQUID | TYPE_GAS | TYPE_ENERGY`: the elements that
/// `ACEL.cpp:76` / `DCEL.cpp:73` move.
pub(super) const MOVABLE_MASK: u32 = TYPE_PART | TYPE_LIQUID | TYPE_GAS | TYPE_ENERGY;

/// TPT `MAX_VELOCITY` (px per step), mirroring the movement kernel's clamp.
pub(super) const MAX_VELOCITY: f32 = 16.0;

/// `|v|` for floats (CubeCL has no `f32::abs`).
#[cube]
pub(super) fn fabsf(v: f32) -> f32 {
    let mut r = v;
    if v < 0.0f32 {
        r = -v;
    }
    r
}

/// `max(a, b)` (CubeCL has no `f32::max`).
#[cube]
pub(super) fn maxf(a: f32, b: f32) -> f32 {
    let mut r = a;
    if b > a {
        r = b;
    }
    r
}

/// `clamp(v, lo, hi)` (CubeCL has no `f32::clamp`).
#[cube]
pub(super) fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut r = v;
    if r < lo {
        r = lo;
    }
    if r > hi {
        r = hi;
    }
    r
}

/// `clamp(v, lo, hi)` for signed integers (CubeCL has no `i32::clamp`).
#[cube]
pub(super) fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    let mut r = v;
    if r < lo {
        r = lo;
    }
    if r > hi {
        r = hi;
    }
    r
}

/// `isign(v)` as a float: `-1`, `0` or `1`.
#[cube]
pub(super) fn isign_f(v: i32) -> f32 {
    let mut r = 0.0f32;
    if v > 0i32 {
        r = 1.0f32;
    }
    if v < 0i32 {
        r = -1.0f32;
    }
    r
}

/// The four orthogonal offsets of `ACEL.cpp:66-68` (`n = 0..4`).
#[cube]
pub(super) fn ortho_dir(n: u32) -> (i32, i32) {
    let axis = n / 2u32;
    let mut sgn = 1i32;
    if (n & 1u32) == 0u32 {
        sgn = -1i32;
    }
    let mut dx = 0i32;
    let mut dy = sgn;
    if axis != 0u32 {
        dx = sgn;
        dy = 0i32;
    }
    (dx, dy)
}

/// One of the nine offsets of a 3x3 scan (`n == 4` is the centre).
#[cube]
pub(super) fn ring_dir(n: u32) -> (i32, i32) {
    ((n % 3u32) as i32 - 1i32, (n / 3u32) as i32 - 1i32)
}

/// `true` when the particle at `(x, y)` has one of the properties `ACEL` /
/// `DCEL` move. TPT reads `pmap` first and falls back to `photons`.
#[cube]
pub(super) fn pixel_movable(
    pmap: &[u32],
    photons: &[u32],
    words: &[u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> bool {
    let mut v = pmap_at(pmap, xres, yres, x, y);
    if v == 0u32 {
        v = pmap_at(photons, xres, yres, x, y);
    }
    let mut movable = false;
    if v != 0u32 {
        movable = (elem_word(words, v & PMAP_TYPE, WORD_PROPERTIES) & MOVABLE_MASK) != 0u32;
    }
    movable
}

/// Slot of the particle a gather acts on: `pmap` first, `photons` second.
///
/// [`NO_TARGET`] means the pixel holds no particle at all. The pool never
/// exceeds `0x7F_FFFF` slots (`pmap` stores `id + 1` in 23 bits), so the
/// sentinel cannot collide with a real slot.
#[cube]
pub(super) fn target_slot(pmap: &[u32], photons: &[u32], idx: usize) -> u32 {
    if pmap[idx] != 0u32 {
        pmap_slot(pmap[idx]) as u32
    } else if photons[idx] != 0u32 {
        pmap_slot(photons[idx]) as u32
    } else {
        NO_TARGET
    }
}

/// Run the force-element pass for one frame.
///
/// Order: the `DMG` blast first (its clear/stamp/apply triple has to bracket
/// itself, and the impulse then feeds the accelerator multipliers below), then
/// the accelerators (they only touch velocities), then the two velocity
/// gathers, then the heat conductor, which reads temperatures none of the other
/// passes write. TPT applies all of these in slot order, so only a particle that
/// is both blasted and accelerated by an adjacent `ACEL` sees a different f32
/// rounding (`ACEL` multiplies the blast velocity here).
pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    if dims.pixels == 0 || dims.capacity == 0 || dims.cells == 0 || dims.elem_len == 0 {
        return;
    }
    if bufs.pmap.len() < dims.pixels
        || bufs.photons.len() < dims.pixels
        || bufs.air_pv.len() < dims.cells
        || bufs.blast_cell.len() < dims.cells
        || bufs.part_ty.len() < dims.capacity
        || bufs.part_vx.len() < dims.capacity
        || bufs.part_vy.len() < dims.capacity
        || bufs.part_ctype.len() < dims.capacity
        || bufs.part_temp.len() < dims.capacity
    {
        return;
    }

    let (cubes, dim) = launch_grid(dims.pixels);

    let (cell_cubes, cell_dim) = launch_grid(dims.cells);
    unsafe {
        blast_clear_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cell_cubes, 1, 1),
            CubeDim::new_1d(cell_dim),
            BufferArg::from_raw_parts(bufs.gpu_blast_cell.clone(), dims.cells),
            dims.cells as u32,
        );
    }

    unsafe {
        dmg_blast_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_blast_cell.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
        );
    }

    unsafe {
        dmg_apply_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_blast_cell.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.xcells,
        );
    }

    unsafe {
        accelerator_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
        );
    }

    unsafe {
        repeller_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }

    // The gather only needs to look as far as the longest beam in the scene;
    // without that bound it walked every ray to the canvas edge and dominated
    // particle-heavy frames.
    let (slot_cubes, slot_dim) = launch_grid(dims.used_slots);
    unsafe {
        fray_range_clear_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_fray_range.clone(), 1),
        );
    }
    unsafe {
        fray_range_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_fray_range.clone(), 1),
            dims.used_slots as u32,
        );
    }

    unsafe {
        fray_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_fray_range.clone(), 1),
            dims.xres,
            dims.yres,
        );
    }

    unsafe {
        heac_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_directions_are_the_four_neighbours() {
        let dirs: Vec<(i32, i32)> = (0u32..4u32).map(ortho_dir).collect();
        assert_eq!(dirs, vec![(0, -1), (0, 1), (-1, 0), (1, 0)]);
        // `ring_dir` is the 3x3 scan with the centre at 4.
        let ring: Vec<(i32, i32)> = (0u32..9u32).map(ring_dir).collect();
        assert_eq!(
            ring,
            vec![
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1)
            ]
        );
        assert_eq!(ring_dir(4), (0, 0));
    }

    #[test]
    fn sign_helper_is_three_valued() {
        assert_eq!(isign_f(-7), -1.0f32);
        assert_eq!(isign_f(0), 0.0f32);
        assert_eq!(isign_f(3), 1.0f32);
    }
}
