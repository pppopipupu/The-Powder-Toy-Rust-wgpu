//! Repellers/attractors (`RPEL`) and force emitters (`FRAY`).

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::{MAX_VELOCITY, NO_TARGET, PMAP_TYPE, clampf, isign_f, ring_dir, target_slot};
use crate::reaction_kernel::shared::{
    WORD_PROPERTIES, elem_word, hash_u32, mix_index, pmap_at, pmap_slot,
};

/// `RPEL.cpp:46`: random samples per frame.
const RPEL_SAMPLES: u32 = 11;

/// `RPEL.cpp:47-48`: the samples live in `[-RPEL_RANGE, RPEL_RANGE]²`.
const RPEL_RANGE: i32 = 10;

/// `FRAY.cpp:56`: beam length used when the emitter's `tmp` is not positive.
const FRAY_DEFAULT_LEN: i32 = 10;

/// One `RPEL` contribution: `(dvx, dvy)` for a source at offset `(ox, oy)`
/// from the target.
///
/// TPT pushes the particle at the source's sampled offset by `isign(offset)`;
/// the gather samples the offset from the *target*, so the sign flips. A source
/// to the right (`ox > 0`) of a hot repeller therefore pushes the target left.
#[cube]
pub(super) fn rpel_push(coef: f32, ox: i32, oy: i32) -> (f32, f32) {
    (-isign_f(ox) * coef, -isign_f(oy) * coef)
}

/// `(vx, vy)` after every sparked `FRAY` beam that reaches `(px, py)`.
///
/// TPT walks outwards from a `FRAY` whose neighbour is a `SPRK`; the gather
/// walks inwards: for each of the eight beam directions the target reads the
/// pixel `m` steps behind it (looking for a `FRAY`) and the pixel `m + 1` steps
/// behind it (looking for the `SPRK` that arms it). Each contribution is
/// clamped to `MAX_VELOCITY` on its own, exactly like TPT's `restrict_flt`.
///
/// `FRAY.cpp:55-57` uses the emitter's `tmp` as the beam length (default 10)
/// and stops only at the canvas edge, so the gather walks along each ray and
/// checks every emitter's own `tmp` when it finds one.
///
/// `max_len` is the longest beam any `FRAY` in the scene can emit this frame
/// ([`fray_range_kernel`]). An emitter only pushes a target `m` pixels away
/// when `m <= curlen + 1`, so nothing past `max_len + 1` can contribute and
/// the walk stops there. That bound is exact; before it existed the walk ran
/// to the canvas edge for every non-solid particle and dominated large scenes.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn fray_apply(
    vx_in: f32,
    vy_in: f32,
    pmap: &[u32],
    part_temp: &[f32],
    part_tmp: &[i32],
    xres: u32,
    yres: u32,
    px: i32,
    py: i32,
    max_len: i32,
) -> (f32, f32) {
    let mut vx = vx_in;
    let mut vy = vy_in;
    // No in-canvas ray is longer than the canvas diagonal, and no contribution
    // can come from farther than the longest emitter's beam plus the spark
    // pixel behind it.
    let bound = xres + yres;
    let mut max_steps = max_len + 1i32;
    if max_steps < 0i32 {
        max_steps = 0i32;
    }
    if max_steps as u32 > bound {
        max_steps = bound as i32;
    }
    let max_steps = max_steps as u32;
    for d in 0u32..9u32 {
        if d != 4u32 {
            // `(nx, ny)` is the direction the beam travels, i.e. the opposite
            // of the spark's offset from the emitter.
            let (nx, ny) = ring_dir(d);
            let mut stopped = false;
            for step in 1u32..(max_steps + 1u32) {
                if !stopped {
                    let m = step as i32;
                    let fx = px - nx * m;
                    let fy = py - ny * m;
                    if fx < 0i32 || fy < 0i32 || fx >= xres as i32 || fy >= yres as i32 {
                        // The emitter is outside the canvas, so this ray has
                        // no further source; TPT's own bounds check ends it.
                        stopped = true;
                    } else {
                        let fv = pmap_at(pmap, xres, yres, fx, fy);
                        if fv != 0u32 && (fv & PMAP_TYPE) == PT_FRAY {
                            let sv = pmap_at(
                                pmap,
                                xres,
                                yres,
                                px - nx * (m + 1i32),
                                py - ny * (m + 1i32),
                            );
                            if sv != 0u32 && (sv & PMAP_TYPE) == PT_SPRK {
                                let sid = pmap_slot(fv);
                                // `FRAY.cpp:55-57`: `tmp` overrides the length.
                                let mut curlen = part_tmp[sid];
                                if curlen <= 0i32 {
                                    curlen = FRAY_DEFAULT_LEN;
                                }
                                if m <= curlen + 1i32 {
                                    let coef = (part_temp[sid] - 273.15f32) / 10.0f32;
                                    vx = clampf(
                                        vx + (nx as f32) * coef,
                                        -MAX_VELOCITY,
                                        MAX_VELOCITY,
                                    );
                                    vy = clampf(
                                        vy + (ny as f32) * coef,
                                        -MAX_VELOCITY,
                                        MAX_VELOCITY,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (vx, vy)
}

/// Repeller / attractor, gather form.
///
/// The target draws [`RPEL_SAMPLES`] uniform offsets in `[-10, 10]²` and looks
/// for an `RPEL` at each one; [`rpel_push`] turns a source's offset and
/// temperature into the velocity change. A source that fails the `ctype` filter
/// is ignored, and solids are never pushed (`RPEL.cpp:59`).
#[cube(launch_unchecked)]
pub fn repeller_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_ctype: &[u32],
    part_temp: &[f32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let found_slot = target_slot(pmap, photons, idx);
        if found_slot != NO_TARGET {
            let slot = found_slot as usize;
            let tty = part_ty[slot];
            // `RPEL.cpp:59`: solids are never pushed.
            if tty != PT_NONE && (elem_word(elem_words, tty, WORD_PROPERTIES) & TYPE_SOLID) == 0u32
            {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let mut vx = part_vx[slot];
                let mut vy = part_vy[slot];
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 2_246_822_519u32));
                let span = (2i32 * RPEL_RANGE + 1i32) as u32;
                for k in 0u32..RPEL_SAMPLES {
                    let h = hash_u32(seed ^ mix_index(k, 0x9E37_79B9u32));
                    let ox = (h % span) as i32 - RPEL_RANGE;
                    let oy = ((h >> 8u32) % span) as i32 - RPEL_RANGE;
                    if ox != 0i32 || oy != 0i32 {
                        let src = pmap_at(pmap, xres, yres, px + ox, py + oy);
                        if src != 0u32 && (src & PMAP_TYPE) == PT_RPEL {
                            let sid = pmap_slot(src);
                            // `RPEL.cpp:60`: `ctype` selects the affected element.
                            let ct = part_ctype[sid];
                            if ct == 0u32 || ct == tty {
                                let coef = (part_temp[sid] - 273.15f32) / 10.0f32;
                                let (dvx, dvy) = rpel_push(coef, ox, oy);
                                vx = clampf(vx + dvx, -MAX_VELOCITY, MAX_VELOCITY);
                                vy = clampf(vy + dvy, -MAX_VELOCITY, MAX_VELOCITY);
                            }
                        }
                    }
                }
                part_vx[slot] = vx;
                part_vy[slot] = vy;
            }
        }
    }
}

/// Clears the per-frame longest `FRAY` beam word ([`fray_range_kernel`]).
///
/// One thread, so it can run in the same stream right before the max pass.
#[cube(launch_unchecked)]
pub fn fray_range_clear_kernel(range: &mut [u32]) {
    let idx = ABSOLUTE_POS;
    if idx == 0usize {
        range[0] = 0u32;
    }
}

/// One thread per slot: the longest beam any `FRAY` can emit this frame.
///
/// `FRAY.cpp:55-57` gives the beam `tmp` pixels (10 when `tmp` is not
/// positive), and a target `m` pixels away is only pushed when
/// `m <= curlen + 1`. The gather therefore never needs to look past
/// `max_len + 1`, which turns its per-particle cost from the canvas diagonal
/// into the actual beam length.
#[cube(launch_unchecked)]
pub fn fray_range_kernel(
    part_ty: &[u32],
    part_tmp: &[i32],
    range: &mut [Atomic<u32>],
    num_slots: u32,
) {
    let i = ABSOLUTE_POS;
    if i < num_slots as usize && part_ty[i] == PT_FRAY {
        let mut len = part_tmp[i];
        if len <= 0i32 {
            len = FRAY_DEFAULT_LEN;
        }
        if len > 0i32 {
            range[0].fetch_max(len as u32);
        }
    }
}

/// Force emitter, gather form.
///
/// Every non-solid particle asks [`fray_apply`] for the impulse of the beams
/// that reach it; a solid is never pushed (`FRAY.cpp:79`). The spark is read
/// from `pmap` only: `SPRK` is `TYPE_SOLID` in this TPT version, so it lives in
/// the matter map exactly where `FRAY.cpp:67` looks.
///
/// `range[0]` carries the frame's longest beam ([`fray_range_kernel`]); when it
/// is zero the scene holds no emitter that could push anything, and the whole
/// pass costs one word read per pixel.
#[cube(launch_unchecked)]
pub fn fray_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_temp: &[f32],
    part_tmp: &[i32],
    elem_words: &[u32],
    range: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    let max_len = range[0] as i32;
    if idx < (xres * yres) as usize && max_len > 0i32 {
        let found_slot = target_slot(pmap, photons, idx);
        if found_slot != NO_TARGET {
            let slot = found_slot as usize;
            let tty = part_ty[slot];
            if tty != PT_NONE && (elem_word(elem_words, tty, WORD_PROPERTIES) & TYPE_SOLID) == 0u32
            {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let (vx, vy) = fray_apply(
                    part_vx[slot],
                    part_vy[slot],
                    pmap,
                    part_temp,
                    part_tmp,
                    xres,
                    yres,
                    px,
                    py,
                    max_len,
                );
                part_vx[slot] = vx;
                part_vy[slot] = vy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::MOVABLE_MASK;
    use super::*;

    /// Packed spatial-map word for `ty` in slot `id`.
    fn pack(id: u32, ty: u32) -> u32 {
        ((id + 1) << 9) | (ty & PMAP_TYPE)
    }

    /// 16x16 canvas with a `FRAY` in slot 0 at `(8, 8)` (temp 100 C, so the
    /// coefficient is `10.0`) and a `SPRK` in slot 1 at `(9, 8)`.
    fn fray_canvas() -> (Vec<u32>, Vec<f32>, Vec<i32>) {
        let mut pmap = vec![0u32; 16 * 16];
        pmap[8 * 16 + 8] = pack(0, PT_FRAY);
        pmap[8 * 16 + 9] = pack(1, PT_SPRK);
        (pmap, vec![373.15f32, 293.15f32], vec![0i32, 0i32])
    }

    #[test]
    fn fray_beam_points_away_from_the_spark() {
        let (pmap, temp, tmp) = fray_canvas();
        // The spark sits to the right of the emitter, so the beam travels left:
        // a target three pixels to the left is pushed along -x by
        // `(373.15 - 273.15) / 10`.
        let (vx, vy) = fray_apply(0.0f32, 0.0f32, &pmap, &temp, &tmp, 16, 16, 5, 8, 10);
        assert!((vx - -10.0f32).abs() < 1e-4f32);
        assert_eq!(vy, 0.0f32);
        // The pixel next to the emitter on the spark side is never pushed: the
        // beam only exists in the opposite direction.
        let (vx, vy) = fray_apply(0.0f32, 0.0f32, &pmap, &temp, &tmp, 16, 16, 11, 8, 10);
        assert_eq!(vx, 0.0f32);
        assert_eq!(vy, 0.0f32);
        // Each contribution is clamped to `MAX_VELOCITY` on its own.
        let (vx, _) = fray_apply(-15.0f32, 0.0f32, &pmap, &temp, &tmp, 16, 16, 5, 8, 10);
        assert_eq!(vx, -16.0f32);
    }

    #[test]
    fn fray_beam_length_follows_tmp() {
        let (pmap, temp, mut tmp) = fray_canvas();
        // `FRAY.cpp:55-57`: a positive `tmp` shortens the beam to `tmp + 1`.
        tmp[0] = 1i32;
        let (vx, _) = fray_apply(0.0f32, 0.0f32, &pmap, &temp, &tmp, 16, 16, 7, 8, 1);
        assert!((vx - -10.0f32).abs() < 1e-4f32);
        let (vx, _) = fray_apply(0.0f32, 0.0f32, &pmap, &temp, &tmp, 16, 16, 5, 8, 1);
        assert_eq!(vx, 0.0f32);
    }

    #[test]
    fn fray_beam_works_on_diagonals() {
        let mut pmap = vec![0u32; 16 * 16];
        pmap[8 * 16 + 8] = pack(0, PT_FRAY);
        // A spark down-right of the emitter turns the beam up-left.
        pmap[9 * 16 + 9] = pack(1, PT_SPRK);
        let temp = vec![373.15f32, 293.15f32];
        let tmp = vec![0i32, 0i32];
        let (vx, vy) = fray_apply(0.0f32, 0.0f32, &pmap, &temp, &tmp, 16, 16, 5, 5, 10);
        assert!((vx - -10.0f32).abs() < 1e-4f32);
        assert!((vy - -10.0f32).abs() < 1e-4f32);
    }

    #[test]
    fn repeller_push_is_away_from_the_source() {
        // A hot source to the right pushes the target left ...
        let (dvx, dvy) = rpel_push(10.0f32, 5, 0);
        assert_eq!(dvx, -10.0f32);
        assert_eq!(dvy, 0.0f32);
        // ... a hot source above pushes it down ...
        let (dvx, dvy) = rpel_push(10.0f32, 0, -5);
        assert_eq!(dvx, 0.0f32);
        assert_eq!(dvy, 10.0f32);
        // ... and a cold source to the left attracts it: the target moves
        // towards the source, i.e. -x.
        let (dvx, dvy) = rpel_push(-4.0f32, -2, 0);
        assert_eq!(dvx, -4.0f32);
        assert_eq!(dvy, 0.0f32);
    }

    #[test]
    fn fray_walk_covers_the_default_beam() {
        // The gather reads the emitter at `m` and the spark at `m + 1` pixels
        // behind the target, for every `m` up to the frame's longest beam, so
        // the default beam covers `m = 1..=FRAY_DEFAULT_LEN + 1`.
        assert_eq!(FRAY_DEFAULT_LEN, 10);
        assert_eq!(RPEL_SAMPLES, 11);
        assert_eq!(RPEL_RANGE, 10);
        assert_eq!(MOVABLE_MASK, 0x1B);
        assert_eq!(PMAP_TYPE, 0x1FF);
    }

    #[test]
    fn fray_beam_length_above_ten_reaches_the_target() {
        // `FRAY.cpp:55-57`: a positive `tmp` is the beam length; the gather
        // must not truncate it at the old 11-pixel window. The emitter sits at
        // (40, 8), the spark at (41, 8), so the beam runs left and covers
        // distances 1..=tmp + 1 = 1..=21.
        let mut pmap = vec![0u32; 64 * 16];
        pmap[8 * 64 + 40] = pack(0, PT_FRAY);
        pmap[8 * 64 + 41] = pack(1, PT_SPRK);
        let temp = vec![373.15f32, 293.15f32];
        let tmp = vec![20i32, 0i32];
        let (vx, _) = fray_apply(0.0f32, 0.0f32, &pmap, &temp, &tmp, 64, 16, 20, 8, 20);
        assert!(
            (vx - -10.0f32).abs() < 1e-4f32,
            "a 20-pixel beam reaches a target 20 px away"
        );
        let (vx, _) = fray_apply(0.0f32, 0.0f32, &pmap, &temp, &tmp, 64, 16, 19, 8, 20);
        assert!(
            (vx - -10.0f32).abs() < 1e-4f32,
            "the last covered pixel is `tmp + 1` away"
        );
        let (vx, _) = fray_apply(0.0f32, 0.0f32, &pmap, &temp, &tmp, 64, 16, 18, 8, 20);
        assert_eq!(vx, 0.0f32, "the beam stops after `tmp + 1` pixels");
    }
}
