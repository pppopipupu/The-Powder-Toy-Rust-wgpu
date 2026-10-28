use cubecl::prelude::*;
use tpt_core::constants::*;

use super::{PMAP_TYPE, WORD_HIGH_TEMP, chance1, clampf};
use crate::reaction_kernel::shared::{elem_word, hash_u32, iabs, mix_index, pmap_at, pmap_slot};

/// `TUNG` melting, `O2` detonation and the `tmp3` pressure-brittle path
/// (`TUNG.cpp:57-104`), plus the `QRTZ` pressure rupture (`QRTZ.cpp:57-67`).
///
/// Pixel parallel: both elements compare the current coarse-cell pressure
/// against the value they remembered in `tmp3` on the previous frame, and
/// `TUNG` also needs the spatial map for its `O2` detonation.
///
/// The `TUNG.cpp:78-80` pressure pulse is a plain read-modify-write of the
/// coarse cell, so two melting pixels in the same cell can lose one add where
/// TPT serialises them; `pv` has no atomic float add in WGSL. The velocity kick
/// of the same branch is applied per particle, so it has no such race.
#[cube(launch_unchecked)]
pub(super) fn special_pressure_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp3: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_pv: &mut [f32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty != PT_NONE {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 2_654_435_761u32));

                if ty == PT_TUNG {
                    // `TUNG.cpp:55`: the melting point comes from the element
                    // table so a Lua override still applies.
                    let melting = f32::reinterpret(elem_word(elem_words, PT_TUNG, WORD_HIGH_TEMP));
                    // `TUNG.cpp:60-75`: oxygen next to tungsten above 2400 K
                    // detonates it without the 1-in-20 melt roll.
                    let mut splode = false;
                    if part_temp[id] > 2400.0f32 {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32)
                                && (pmap_at(pmap, xres, yres, px + dx, py + dy) & PMAP_TYPE)
                                    == PT_O2
                            {
                                splode = true;
                            }
                        }
                    }
                    if (part_temp[id] > melting && chance1(seed, 20u32)) || splode {
                        // `TUNG.cpp:76-90`: one branch in fifty puffs pressure,
                        // one in a hundred burns into fire, the rest melts.
                        let roll = hash_u32(seed ^ mix_index(id as u32, 0x9E37_79B9u32));
                        let roll2 = hash_u32(roll ^ mix_index(id as u32, 0x85EB_CA6Bu32));
                        if roll.is_multiple_of(50u32) {
                            // `TUNG.cpp:78`: the reference adds the 50 units
                            // raw; the air solver clamps the cell on its next
                            // pass. Clamping here would lose the overshoot
                            // that pushes neighbouring gas.
                            air_pv[cidx] += 50.0f32;
                            if splode {
                                let bump = 200i32 + (roll2 % 600u32) as i32;
                                part_temp[id] = clampf(melting + bump as f32, MIN_TEMP, MAX_TEMP);
                            }
                            // `TUNG.cpp:78-80`: the pressure pulse also kicks
                            // the particle in both axes.
                            let kick_x = (roll2 % 101u32) as i32 - 50i32;
                            let kick_y = ((roll2 >> 8u32) % 101u32) as i32 - 50i32;
                            part_vx[id] += kick_x as f32;
                            part_vy[id] += kick_y as f32;
                        } else if roll2.is_multiple_of(100u32) {
                            part_ty[id] = PT_FIRE;
                            part_life[id] = (hash_u32(roll) % 500u32) as i32;
                        } else {
                            part_ty[id] = PT_LAVA;
                            part_ctype[id] = PT_TUNG;
                        }
                    } else {
                        // `TUNG.cpp:92-103`: a pressure jump of more than 32
                        // (in 1/64 units) shatters the tungsten.
                        let press = (air_pv[cidx] * 64.0f32) as i32;
                        // `TUNG.cpp:145`: the element's `create` hook seeds
                        // `tmp3` from the ambient pressure, so a freshly drawn
                        // particle must not shatter on its first frame. We
                        // encode the recorded pressure as `(press << 1) | 1`
                        // so 0 represents "unseeded" without conflating 0 pressure.
                        let val = part_tmp3[id];
                        if val == 0i32 {
                            part_tmp3[id] = (press << 1i32) | 1i32;
                        } else {
                            let remembered = val >> 1i32;
                            let diff = press - remembered;
                            if iabs(diff) > 32i32 {
                                part_ty[id] = PT_BRMT;
                                part_ctype[id] = PT_TUNG;
                            } else {
                                part_tmp3[id] = (press << 1i32) | 1i32;
                            }
                        }
                    }
                } else if ty == PT_QRTZ {
                    // `QRTZ.cpp:57-67`: a pressure jump beyond the temperature
                    // scaled tolerance cracks the crystal into `PQRT`, which
                    // pins it for five frames.
                    let press = (air_pv[cidx] * 64.0f32) as i32;
                    // `QRTZ.cpp:171-175`: the `create` hook seeds `tmp3` from
                    // the ambient pressure, so the first frame must not crack
                    // a freshly drawn crystal on a stale zero reference.
                    // `QRTZ.cpp:171-175`: the `create` hook seeds `tmp3` from
                    // the ambient pressure, encoded as `(press << 1) | 1`.
                    let val = part_tmp3[id];
                    if val == 0i32 {
                        part_tmp3[id] = (press << 1i32) | 1i32;
                    } else {
                        let remembered = val >> 1i32;
                        let tol = part_temp[id] * 1.0666f32;
                        let diff = (press - remembered) as f32;
                        if diff > tol || diff < -tol {
                            part_ty[id] = PT_PQRT;
                            part_life[id] = 5i32;
                        }
                        part_tmp3[id] = (press << 1i32) | 1i32;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn tung_brittleness_threshold_is_32_sixtyfourths() {
        // `TUNG.cpp:92-103`: `press = int(pv*64)`; a jump beyond ±32 shatters
        // the tungsten into `BRMT` instead of remembering the new pressure.
        let encode_press = |p: i32| (p << 1) | 1;
        let brittle = |pv: f32, encoded: i32| {
            let press = (pv * 64.0f32) as i32;
            if encoded == 0 {
                return None;
            }
            let remembered = encoded >> 1;
            let diff = press - remembered;
            (!(-32i32..=32i32).contains(&diff)).then_some(press)
        };
        assert_eq!(
            brittle(0.0f32, encode_press(0)),
            None,
            "a steady cell holds"
        );
        assert_eq!(
            brittle(0.5f32, encode_press(0)),
            None,
            "exactly 32 is still within range"
        );
        assert_eq!(
            brittle(0.6f32, encode_press(0)),
            Some(38),
            "38 units shatter it"
        );
        assert_eq!(
            brittle(0.0f32, encode_press(33)),
            Some(0),
            "a drop of 33 shatters it"
        );
        assert_eq!(
            brittle(-0.6f32, encode_press(0)),
            Some(-38),
            "the vacuum side is symmetric"
        );
        // Pressure jump from zero:
        assert_eq!(brittle(10.0f32, encode_press(0)), Some(640));
    }

    #[test]
    fn qrtz_rupture_tolerance_scales_with_temperature() {
        // `QRTZ.cpp:60-67`: `tol = temp * 1.0666` against `press - tmp3`.
        let encode_press = |p: i32| (p << 1) | 1;
        let cracks = |pv: f32, encoded: i32, temp: f32| {
            let press = (pv * 64.0f32) as i32;
            if encoded == 0 {
                return false;
            }
            let remembered = encoded >> 1;
            let tol = temp * 1.0666f32;
            let diff = (press - remembered) as f32;
            diff > tol || diff < -tol
        };
        assert!(!cracks(0.0f32, encode_press(0), 293.15f32));
        assert!(!cracks(4.0f32, encode_press(0), 293.15f32), "256 < 312.7");
        assert!(cracks(10.0f32, encode_press(0), 293.15f32), "640 > 312.7");
        assert!(
            cracks(-10.0f32, encode_press(0), 293.15f32),
            "the vacuum side cracks too"
        );
        assert!(
            !cracks(10.0f32, encode_press(0), 1000.0f32),
            "a hot crystal tolerates more"
        );
        assert!(cracks(10.0f32, encode_press(0), 293.15f32));
    }
}
