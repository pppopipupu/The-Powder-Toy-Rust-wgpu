//! Accelerators (`ACEL`/`DCEL`) and rapid heat conductors (`HEAC`).

use core::mem::offset_of;

use cubecl::prelude::*;
use tpt_core::constants::*;
use tpt_core::elements::GpuElementProps;

use super::{
    MAX_VELOCITY, MOVABLE_MASK, NO_TARGET, PMAP_TYPE, clampf, clampi, fabsf, maxf, ortho_dir,
    pixel_movable, ring_dir, target_slot,
};
use crate::reaction_kernel::shared::{WORD_PROPERTIES, elem_word, iabs, pmap_at, pmap_slot};

/// `u32` word offset of [`GpuElementProps::heat_conduct`].
pub(super) const WORD_HEAT_CONDUCT: u32 = 15;

const _: () = assert!(offset_of!(GpuElementProps, heat_conduct) == WORD_HEAT_CONDUCT as usize * 4);

/// `HEAC.cpp:120`: the neighbourhood radius of the heat conductor.
const HEAC_RAD: i32 = 4;

/// `HSWC` conducts heat only at exactly this `life` (`SimulationData.cpp:338`).
const HSWC_ON_LIFE: i32 = 10;

/// `ACEL.cpp:55-63`: `1.1`, or `1 + clamp(life, 0, 1000) / 100`.
///
/// Written as one expression: CubeCL cannot assign a computed value to a local
/// that was seeded from a constant expression, so there is no accumulator.
#[cube]
pub(super) fn acel_multiplier(life: i32) -> f32 {
    if life != 0i32 {
        1.0f32 + (clampi(life, 0i32, 1000i32) as f32) / 100.0f32
    } else {
        1.1f32
    }
}

/// `DCEL.cpp:57-62`: `1 / 1.1`, or `1 - clamp(life, 0, 100) / 100`.
#[cube]
pub(super) fn dcel_multiplier(life: i32) -> f32 {
    if life != 0i32 {
        1.0f32 - (clampi(life, 0i32, 100i32) as f32) / 100.0f32
    } else {
        1.0f32 / 1.1f32
    }
}

/// `SimulationData::IsHeatInsulator` (`SimulationData.cpp:338-341`).
///
/// `PIPE`/`PPIP` insulate unless `tmp` carries `PFLAG_CAN_CONDUCT`; this port
/// has no pipe pass, so a pipe's `tmp` is always zero and the test reduces to
/// "always an insulator". `HSWC` conducts only while latched on.
#[cube]
pub(super) fn heat_insulator(words: &[u32], ty: u32, life: i32) -> bool {
    let mut insulator = elem_word(words, ty, WORD_HEAT_CONDUCT) == 0u32;
    if ty == PT_PIPE || ty == PT_PPIP {
        insulator = true;
    }
    if ty == PT_HSWC && life != HSWC_ON_LIFE {
        insulator = true;
    }
    insulator
}

/// `SimulationData::HeatCapacityOf` (`SimulationData.cpp:343-349`).
///
/// The TPT default is `1.0f` (`Element.cpp:33`); `BASE` overrides it to `1.5f`
/// (`BASE.cpp:33`) and is the only override in the shipped table. `PIPE` adds
/// its cargo's capacity, which this port never has.
#[cube]
pub(super) fn heat_capacity(ty: u32) -> f32 {
    let mut capacity = 1.0f32;
    if ty == PT_BASE {
        capacity = 1.5f32;
    }
    capacity
}

/// `1` when the matter pixel at `(x, y)` holds a heat insulator.
#[cube]
fn insulator_pixel(
    pmap: &[u32],
    part_ty: &[u32],
    part_life: &[i32],
    words: &[u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> u32 {
    let v = pmap_at(pmap, xres, yres, x, y);
    let mut hit = 0u32;
    if v != 0u32 {
        let id = pmap_slot(v);
        if heat_insulator(words, part_ty[id], part_life[id]) {
            hit = 1u32;
        }
    }
    hit
}

/// TPT's `CheckLine` (`HEAC.cpp:70-124`) for the eight `HEAC_RAD` offsets.
///
/// `(dx, dy)` is always a multiple of `HEAC_RAD` with `max(|dx|, |dy|) ==
/// HEAC_RAD`, so the DDA's `de = dy / dx` is `0` (axis aligned) or `1`
/// (diagonal) and the walk visits a fixed pattern: `rad + 1` primary pixels
/// plus, on a diagonal, one "staircase" pixel per step. The staircase is what
/// makes TPT's test asymmetric — the `(+4, +4)` line samples `(0, 1)` and
/// `(1, 2)` while `(-4, +4)` samples `(-4, 3)` and `(-3, 2)` — and it is
/// reproduced here so an insulator one pixel off the ideal line still blocks.
// `core::mem::swap` is not a CubeCL intrinsic, so the three swaps are written
// out by hand.
#[allow(clippy::manual_swap)]
#[cube]
pub(super) fn heac_line_blocked(
    pmap: &[u32],
    part_ty: &[u32],
    part_life: &[i32],
    words: &[u32],
    xres: u32,
    yres: u32,
    x0: i32,
    y0: i32,
    dx: i32,
    dy: i32,
) -> bool {
    let mut x1 = x0;
    let mut y1 = y0;
    let mut x2 = x0 + dx;
    let mut y2 = y0 + dy;
    // TPT swaps the axes when the line is more vertical than horizontal and
    // then always walks towards +x.
    let reverse = iabs(y2 - y1) > iabs(x2 - x1);
    if reverse {
        let tx = x1;
        x1 = y1;
        y1 = tx;
        let ty = x2;
        x2 = y2;
        y2 = ty;
    }
    if x1 > x2 {
        let tx = x1;
        x1 = x2;
        x2 = tx;
        let ty = y1;
        y1 = y2;
        y2 = ty;
    }
    let span = x2 - x1;
    let rise = iabs(y2 - y1);
    let mut sy = 1i32;
    if y2 < y1 {
        sy = -1i32;
    }
    let diagonal = rise == span && span > 0i32;
    let mut steps = (span + 1i32) as u32;
    if diagonal {
        steps = (2i32 * span + 1i32) as u32;
    }
    // The minor coordinate only moves on a diagonal: `de = rise / span` is 1
    // there and 0 on an axis-aligned line.
    let mut minor = 0i32;
    if diagonal {
        minor = sy;
    }

    let mut hits = 0u32;
    for k in 0u32..9u32 {
        // Even steps are the primary pixel, odd steps the staircase pixel that
        // TPT's `e >= 0.5f` advance adds after each primary pixel.
        let a = (k / 2u32) as i32;
        let even = (k & 1u32) == 0u32;
        let bx = x1 + a;
        let by = if even {
            y1 + a * minor
        } else {
            y1 + (a + 1i32) * minor
        };
        let px = if reverse { by } else { bx };
        let py = if reverse { bx } else { by };
        let ok = k < steps && bx <= x2;
        let hit = if ok {
            insulator_pixel(pmap, part_ty, part_life, words, xres, yres, px, py)
        } else {
            0u32
        };
        hits += hit;
    }
    hits > 0u32
}

/// `HEAC.cpp:126-152`: `(sum(temp * heat capacity), sum(heat capacity))` over
/// the nine `HEAC_RAD` positions around `(x, y)`.
///
/// A position is skipped when a heat insulator blocks the line from `(x, y)`
/// (the centre is always included), and an individual particle is skipped when
/// it is a heat insulator itself. Both maps contribute, so a matter and an
/// energy particle in the same pixel are both averaged.
#[cube]
pub(super) fn heac_average(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_temp: &mut [f32],
    part_life: &[i32],
    words: &[u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> (f32, f32) {
    let mut c_heat = 0.0f32;
    let mut hc_total = 0.0f32;
    for n in 0u32..9u32 {
        let (dx, dy) = ring_dir(n);
        let rx = dx * HEAC_RAD;
        let ry = dy * HEAC_RAD;
        let mut visible = true;
        if dx != 0i32 || dy != 0i32 {
            visible = !heac_line_blocked(pmap, part_ty, part_life, words, xres, yres, x, y, rx, ry);
        }

        let v = if visible {
            pmap_at(pmap, xres, yres, x + rx, y + ry)
        } else {
            0u32
        };
        let vid = pmap_slot(v);
        let vok = v != 0u32 && !heat_insulator(words, part_ty[vid], part_life[vid]);
        let vhc = if vok {
            heat_capacity(part_ty[vid])
        } else {
            0.0f32
        };
        let vadd = if vok { part_temp[vid] * vhc } else { 0.0f32 };
        c_heat += vadd;
        hc_total += vhc;

        let f = if visible {
            pmap_at(photons, xres, yres, x + rx, y + ry)
        } else {
            0u32
        };
        let fid = pmap_slot(f);
        let fok = f != 0u32 && !heat_insulator(words, part_ty[fid], part_life[fid]);
        let fhc = if fok {
            heat_capacity(part_ty[fid])
        } else {
            0.0f32
        };
        let fadd = if fok { part_temp[fid] * fhc } else { 0.0f32 };
        c_heat += fadd;
        hc_total += fhc;
    }
    (c_heat, hc_total)
}

/// Accelerators and decelerators, both halves in one dispatch.
///
/// The first half is the accelerator's own `tmp` glow flag: it is set when one
/// of the four orthogonal neighbours is a movable element, which is exactly
/// the condition under which TPT would have written a neighbour's velocity.
/// The second half is the gather: this pixel's particle is multiplied by every
/// adjacent accelerator and decelerator, with `ACEL`'s `MAX_VELOCITY`
/// renormalisation applied after each `ACEL` like TPT does.
#[cube(launch_unchecked)]
pub fn accelerator_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_life: &[i32],
    part_tmp: &mut [i32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let px = (idx as u32 % xres) as i32;
        let py = (idx as u32 / xres) as i32;

        // `ACEL.cpp:61` / `DCEL.cpp:59`: the flag is the accelerator's own.
        if pmap[idx] != 0u32 {
            let aid = pmap_slot(pmap[idx]);
            let aty = part_ty[aid];
            if aty == PT_ACEL || aty == PT_DCEL {
                let mut flag = 0i32;
                for n in 0u32..4u32 {
                    let (dx, dy) = ortho_dir(n);
                    if pixel_movable(pmap, photons, elem_words, xres, yres, px + dx, py + dy) {
                        flag = 1i32;
                    }
                }
                part_tmp[aid] = flag;
            }
        }

        // The gather: every adjacent accelerator moves this particle.
        let found_slot = target_slot(pmap, photons, idx);
        if found_slot != NO_TARGET {
            let slot = found_slot as usize;
            let tty = part_ty[slot];
            if tty != PT_NONE
                && (elem_word(elem_words, tty, WORD_PROPERTIES) & MOVABLE_MASK) != 0u32
            {
                let mut vx = part_vx[slot];
                let mut vy = part_vy[slot];
                for n in 0u32..4u32 {
                    let (dx, dy) = ortho_dir(n);
                    let mut src = pmap_at(pmap, xres, yres, px + dx, py + dy);
                    if src == 0u32 {
                        src = pmap_at(photons, xres, yres, px + dx, py + dy);
                    }
                    if src != 0u32 {
                        let st = src & PMAP_TYPE;
                        let sid = pmap_slot(src);
                        if st == PT_ACEL {
                            let m = acel_multiplier(part_life[sid]);
                            vx *= m;
                            vy *= m;
                            // "Restrict velocity and try to preserve direction"
                            // (`ACEL.cpp:79-85`).
                            let mv = maxf(fabsf(vx), fabsf(vy));
                            if mv > MAX_VELOCITY {
                                vx *= MAX_VELOCITY / mv;
                                vy *= MAX_VELOCITY / mv;
                            }
                        } else if st == PT_DCEL {
                            let m = dcel_multiplier(part_life[sid]);
                            vx *= m;
                            vy *= m;
                        }
                    }
                }
                part_vx[slot] = vx;
                part_vy[slot] = vy;
            }
        }
    }
}

/// Rapid heat conductor, gather form.
///
/// The first `HEAC` at `rad = 4` that can see this pixel publishes the mean of
/// its own nine positions here, which reproduces both of TPT's writes (the
/// conductor's own temperature and the copy onto its neighbours) in a single
/// dispatch. A particle that is itself a `HEAC` first takes its own mean and is
/// then overwritten by the neighbour's, matching TPT's "last writer wins"
/// behaviour for two conductors four pixels apart.
#[cube(launch_unchecked)]
pub fn heac_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_temp: &mut [f32],
    part_life: &[i32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        let f = photons[idx];
        if v != 0u32 || f != 0u32 {
            let px = (idx as u32 % xres) as i32;
            let py = (idx as u32 / xres) as i32;

            let mut found = false;
            let mut sx = px;
            let mut sy = py;
            for n in 0u32..9u32 {
                if n != 4u32 {
                    let (dx, dy) = ring_dir(n);
                    let nx = px + dx * HEAC_RAD;
                    let ny = py + dy * HEAC_RAD;
                    let nv = pmap_at(pmap, xres, yres, nx, ny);
                    if !found && nv != 0u32 && (nv & PMAP_TYPE) == PT_HEAC {
                        found = true;
                        sx = nx;
                        sy = ny;
                    }
                }
            }

            // Seeded from the pixel's own particle temperature so the local is
            // a runtime value; it is only consumed when `src_ok` is set.
            let mut src_ok = false;
            let mut src_temp = part_temp[pmap_slot(v)];
            let clear = if found {
                !heac_line_blocked(
                    pmap,
                    part_ty,
                    part_life,
                    elem_words,
                    xres,
                    yres,
                    px,
                    py,
                    sx - px,
                    sy - py,
                )
            } else {
                false
            };
            if clear {
                let (sum, total) = heac_average(
                    pmap, photons, part_ty, part_temp, part_life, elem_words, xres, yres, sx, sy,
                );
                if total > 0.0f32 {
                    src_temp = clampf(sum / total, MIN_TEMP, MAX_TEMP);
                    src_ok = true;
                }
            }

            if v != 0u32 {
                let id = pmap_slot(v);
                let ty = part_ty[id];
                if ty != PT_NONE && !heat_insulator(elem_words, ty, part_life[id]) {
                    let mut temp = part_temp[id];
                    if ty == PT_HEAC {
                        let (sum, total) = heac_average(
                            pmap, photons, part_ty, part_temp, part_life, elem_words, xres, yres,
                            px, py,
                        );
                        if total > 0.0f32 {
                            temp = clampf(sum / total, MIN_TEMP, MAX_TEMP);
                        }
                    }
                    if src_ok {
                        temp = src_temp;
                    }
                    part_temp[id] = temp;
                }
            }

            // Energy target: `HEAC.cpp:170-176` writes `photons` too, but an
            // energy element is never a `HEAC`, so it has no mean of its own.
            if f != 0u32 {
                let id = pmap_slot(f);
                let ty = part_ty[id];
                if ty != PT_NONE && !heat_insulator(elem_words, ty, part_life[id]) && src_ok {
                    part_temp[id] = src_temp;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reaction_kernel::force::projectiles::WORD_HIGH_PRESSURE_TRANSITION;
    use crate::reaction_kernel::shared::ELEM_PROP_WORDS;
    use tpt_core::elements::ElementRegistry;

    /// Packed spatial-map word for `ty` in slot `id`.
    fn pack(id: u32, ty: u32) -> u32 {
        ((id + 1) << 9) | (ty & PMAP_TYPE)
    }

    /// `gpu_elem_props` mirror with the words this pass reads.
    fn words_table() -> Vec<u32> {
        let props = ElementRegistry::new_default().to_gpu_props();
        let mut words = vec![0u32; PT_NUM * ELEM_PROP_WORDS];
        for (i, p) in props.iter().enumerate() {
            let base = i * ELEM_PROP_WORDS;
            words[base + WORD_PROPERTIES as usize] = p.properties;
            words[base + WORD_HEAT_CONDUCT as usize] = p.heat_conduct;
            words[base + WORD_HIGH_PRESSURE_TRANSITION as usize] =
                p.high_pressure_transition as u32;
        }
        words
    }

    #[test]
    fn accelerator_multipliers_match_tpt() {
        // `ACEL.cpp:55-63`: 1.1 without a life, `1 + clamp(life, 0, 1000)/100`.
        assert!((acel_multiplier(0) - 1.1f32).abs() < 1e-6f32);
        assert!((acel_multiplier(1) - 1.01f32).abs() < 1e-6f32);
        assert!((acel_multiplier(100) - 2.0f32).abs() < 1e-6f32);
        assert!((acel_multiplier(1000) - 11.0f32).abs() < 1e-6f32);
        assert!((acel_multiplier(5000) - 11.0f32).abs() < 1e-6f32);
        assert!((acel_multiplier(-5) - 1.0f32).abs() < 1e-6f32);

        // `DCEL.cpp:57-62`: 1/1.1 without a life, `1 - clamp(life, 0, 100)/100`.
        assert!((dcel_multiplier(0) - 1.0f32 / 1.1f32).abs() < 1e-6f32);
        assert!((dcel_multiplier(50) - 0.5f32).abs() < 1e-6f32);
        assert!((dcel_multiplier(100) - 0.0f32).abs() < 1e-6f32);
        assert!((dcel_multiplier(500) - 0.0f32).abs() < 1e-6f32);
        assert!((dcel_multiplier(-3) - 1.0f32).abs() < 1e-6f32);
    }

    #[test]
    fn heat_capacity_and_insulator_follow_the_element_table() {
        let words = words_table();
        // `Element.cpp:33` defaults to 1.0, `BASE.cpp:33` is the only override.
        assert_eq!(heat_capacity(PT_BASE), 1.5f32);
        assert_eq!(heat_capacity(PT_WATR), 1.0f32);
        assert_eq!(heat_capacity(PT_PIPE), 1.0f32);

        // `SimulationData.cpp:338`: `HeatConduct == 0` plus the two switches.
        assert!(heat_insulator(&words, PT_INSL, 0));
        assert!(!heat_insulator(&words, PT_WATR, 0));
        assert!(!heat_insulator(&words, PT_HEAC, 0));
        assert!(heat_insulator(&words, PT_PIPE, 0));
        assert!(heat_insulator(&words, PT_HSWC, 0));
        assert!(!heat_insulator(&words, PT_HSWC, HSWC_ON_LIFE));
    }

    #[test]
    fn check_line_walks_tpt_staircase() {
        // 16x16 canvas: a `HEAC` in slot 0 at (8, 8), an `INSL` in slot 1 at
        // (8, 9) — the staircase pixel of the (+4, +4) line.
        let xres = 16u32;
        let yres = 16u32;
        let mut pmap = vec![0u32; (xres * yres) as usize];
        let mut ty = vec![PT_NONE; 4];
        ty[0] = PT_HEAC;
        ty[1] = PT_INSL;
        pmap[8 * 16 + 8] = pack(0, PT_HEAC);
        pmap[9 * 16 + 8] = pack(1, PT_INSL);
        let life = vec![0i32; 4];
        let words = words_table();

        // The staircase pixel blocks the down-right diagonal ...
        assert!(heac_line_blocked(
            &pmap, &ty, &life, &words, xres, yres, 8, 8, 4, 4
        ));
        // ... and the vertical line, which runs straight through it.
        assert!(heac_line_blocked(
            &pmap, &ty, &life, &words, xres, yres, 8, 8, 0, 4
        ));
        // The down-right *up* diagonal does not sample it: TPT's DDA hugs the
        // other side of the ideal line there.
        assert!(!heac_line_blocked(
            &pmap, &ty, &life, &words, xres, yres, 8, 8, 4, -4
        ));
        // Nor does the up-left diagonal.
        assert!(!heac_line_blocked(
            &pmap, &ty, &life, &words, xres, yres, 8, 8, -4, -4
        ));
        // The horizontal line is clear as well.
        assert!(!heac_line_blocked(
            &pmap, &ty, &life, &words, xres, yres, 8, 8, 4, 0
        ));

        // An insulator on the ideal diagonal blocks it.
        let mut pmap2 = pmap.clone();
        pmap2[10 * 16 + 10] = pack(1, PT_INSL);
        assert!(heac_line_blocked(
            &pmap2, &ty, &life, &words, xres, yres, 8, 8, 4, 4
        ));
    }

    #[test]
    fn heac_average_weights_by_heat_capacity() {
        let xres = 16u32;
        let yres = 16u32;
        let mut pmap = vec![0u32; (xres * yres) as usize];
        let mut ty = vec![PT_NONE; 4];
        let mut temp = vec![0.0f32; 4];
        let life = vec![0i32; 4];
        let words = words_table();

        // Centre: the conductor itself at 350 K.
        ty[0] = PT_HEAC;
        temp[0] = 350.0f32;
        pmap[8 * 16 + 8] = pack(0, PT_HEAC);
        // Right neighbour: water at 300 K, weight 1.0.
        ty[1] = PT_WATR;
        temp[1] = 300.0f32;
        pmap[8 * 16 + 12] = pack(1, PT_WATR);
        // Left neighbour: base at 400 K, weight 1.5 (`BASE.cpp:33`).
        ty[2] = PT_BASE;
        temp[2] = 400.0f32;
        pmap[8 * 16 + 4] = pack(2, PT_BASE);

        let photons = vec![0u32; (xres * yres) as usize];
        let (sum, total) = heac_average(
            &pmap, &photons, &ty, &mut temp, &life, &words, xres, yres, 8, 8,
        );
        assert!((total - 3.5f32).abs() < 1e-5f32);
        assert!((sum - 1250.0f32).abs() < 1e-4f32);
        assert!((sum / total - 357.142_86f32).abs() < 1e-3f32);

        // A heat insulator next door is skipped, and so is its temperature.
        ty[1] = PT_INSL;
        let (sum, total) = heac_average(
            &pmap, &photons, &ty, &mut temp, &life, &words, xres, yres, 8, 8,
        );
        assert!((total - 2.5f32).abs() < 1e-5f32);
        assert!((sum - 950.0f32).abs() < 1e-4f32);
    }
}
