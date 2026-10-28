//! Energy particles: electrons.
//!
//! Slot parallel: energy particles live in the `photons` layer, so their
//! position has to come from `part_x`/`part_y` rather than from `pmap`.
//!
//! Sparks are handled by [`super::electronics`] (`SPRK.cpp`: staged conduction,
//! discharge bookkeeping and the per-conductor rules) and neutrons by
//! [`super::neut`] (`NEUT.cpp`). This pass keeps `ELEC` and the `BIZR` family's
//! deco-colour painting (`BIZR.cpp:52-89`, shared by `BIZRG`/`BIZRS`).

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::neut::change_type_in_place;
use super::shared::{
    CR_SET_LIFE, CR_SET_TEMP, CR_SET_TMP, CR_SET_VX, CR_SET_VY, WORD_PROPERTIES, elem_word,
    hash_u32, mix_index, pmap_at, pmap_slot, request_create,
};

/// `BIZR.cpp:50`: `BLEND` divides the per-step colour difference.
const BIZR_BLEND: f32 = 21.0;

/// `ELEC.cpp:125`: `NBLE` only accepts a spark below this temperature.
const NBLE_SPARK_TEMP: f32 = 2273.15;
/// `create_part`'s spark path (`Simulation.cpp:1871-1872`): the good
/// conductors heat by 10 K while below 673 K.
const SPARK_HEAT_CAP: f32 = 673.0;
/// `ELEC.cpp:93`: the ember explosion around glass keeps 80 % of the
/// electron's temperature.
const GLAS_EMBR_TEMP_SCALE: f32 = 0.8;

/// TPT `rng.chance(n, d)`: `roll % d < n`.
#[cube]
fn chance(roll: u32, n: u32, d: u32) -> bool {
    let mut hit = false;
    if d > 0u32 {
        hit = (roll % d) < n;
    }
    hit
}

/// `rng.between(lo, hi)` for small inclusive ranges.
#[cube]
fn between(roll: u32, lo: i32, hi: i32) -> i32 {
    let span = (hi - lo + 1i32) as u32;
    lo + (roll % span) as i32
}

/// `Simulation::create_part`'s special `PT_SPRK` path
/// (`Simulation.cpp:1851-1873`): a conductor can be replaced by a spark in
/// place, with `life = 4`, the replaced type in `ctype`, and the 10 K heat
/// bump for the dense conductors.
#[cube]
#[allow(clippy::too_many_arguments)]
fn spark_conductor(
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    nid: usize,
    nty: u32,
    legacy: u32,
) {
    part_ty[nid] = PT_SPRK;
    part_life[nid] = 4i32;
    part_ctype[nid] = nty;
    if legacy == 0u32
        && part_temp[nid] + 10.0f32 < SPARK_HEAT_CAP
        && (nty == PT_METL
            || nty == PT_BMTL
            || nty == PT_BRMT
            || nty == PT_PSCN
            || nty == PT_NSCN
            || nty == PT_ETRD
            || nty == PT_NBLE
            || nty == PT_IRON)
    {
        part_temp[nid] += 10.0f32;
    }
}

/// `std::round(v / BLEND)` with C's round-half-away-from-zero, which is what
/// `int(std::round(...))` in `BIZR.cpp:77-80` computes.
#[cube]
fn blend_round(diff: i32) -> i32 {
    let q = diff as f32 / BIZR_BLEND;
    let mut r = (q + 0.5f32) as i32;
    if q < 0.0f32 {
        r = (q - 0.5f32) as i32;
    }
    r
}

/// `BIZR.cpp:77-80`: one colour channel takes one step of `BLEND` towards the
/// paint, plus the unit step in the same direction.
#[cube]
fn blend_channel(target: u32, paint: u32) -> u32 {
    let t = target as i32;
    let m = paint as i32;
    let mut step = 0i32;
    if m > t {
        step = 1i32;
    } else if m < t {
        step = -1i32;
    }
    (t + step + blend_round(m - t)) as u32
}

/// `ELEC` sparking and electrolysis; `BIZR`/`BIZRG`/`BIZRS` deco painting.
///
/// Slot parallel: energy particles live in the `photons` layer, so their
/// position has to come from `part_x`/`part_y` rather than from `pmap`.
///
/// The `BIZR` update (`BIZR.cpp:52-89`) walks the radius-2 square and pulls
/// every non-`BIZR` neighbour one `BLEND` step towards the painter's `dcolour`.
/// It is inverted here: a painted particle resolves the `BIZR` sources around
/// it and rewrites its own `dcolour`, so no thread touches a neighbour.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn energy_reaction_kernel(
    part_x: &[f32],
    part_y: &[f32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_temp: &mut [f32],
    part_dcolour: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    pmap: &[u32],
    photons: &[u32],
    elem_words: &[u32],
    create_queue: &mut [Atomic<u32>],
    spawn_defaults: &[u32],
    n_slots: u32,
    xres: u32,
    yres: u32,
    frame_seed: u32,
    legacy: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize {
        let ty = part_ty[i];
        let px = part_x[i] as i32;
        let py = part_y[i] as i32;
        let in_bounds = px >= 0i32 && py >= 0i32 && (px as u32) < xres && (py as u32) < yres;
        if ty == PT_ELEC {
            if in_bounds {
                // `ELEC.cpp:56-135`: the electron walks its radius-2 square,
                // sampling `pmap` first and the energy layer second, and
                // returns as soon as it sparks, explodes glass, electrolyses
                // water or reacts with a neutron/deuteron.
                let seed = hash_u32(frame_seed ^ mix_index(i as u32, 0x2545_F491u32));
                let mut dead = false;
                for n in 0u32..25u32 {
                    if !dead {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        let mut nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv == 0u32 {
                            nv = pmap_at(photons, xres, yres, px + dx, py + dy);
                        }
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            let nty = part_ty[nid];
                            let r = hash_u32(seed ^ mix_index(n, 0x9E37_79B9u32));
                            if nty == PT_GLAS {
                                // `ELEC.cpp:64-84`: every in-bounds pixel of
                                // the glass pixel's 3x3 gets an ember with 80 %
                                // of the electron's temperature and a +-10
                                // px/frame kick; `create_part(-1)` refuses the
                                // occupied pixels, which the queue's default
                                // occupancy test reproduces.
                                for ry2 in 0u32..3u32 {
                                    for rx2 in 0u32..3u32 {
                                        let gx = px + dx + rx2 as i32 - 1i32;
                                        let gy = py + dy + ry2 as i32 - 1i32;
                                        if gx >= 0i32
                                            && gy >= 0i32
                                            && (gx as u32) < xres
                                            && (gy as u32) < yres
                                        {
                                            let rr = hash_u32(
                                                r ^ mix_index(ry2 * 3u32 + rx2, 0x85EB_CA6Bu32),
                                            );
                                            request_create(
                                                create_queue,
                                                PT_EMBR,
                                                gx,
                                                gy,
                                                CR_SET_LIFE
                                                    | CR_SET_TEMP
                                                    | CR_SET_TMP
                                                    | CR_SET_VX
                                                    | CR_SET_VY,
                                                50i32,
                                                0u32,
                                                part_temp[i] * GLAS_EMBR_TEMP_SCALE,
                                                0i32,
                                                0i32,
                                                0i32,
                                                0i32,
                                                0u32,
                                                0u32,
                                                between(rr, -10i32, 10i32) as f32,
                                                between(rr >> 8u32, -10i32, 10i32) as f32,
                                            );
                                        }
                                    }
                                }
                                dead = true;
                            } else if nty == PT_LCRY {
                                // `ELEC.cpp:85-87`: the laser charge timer is
                                // set, the electron lives on.
                                part_tmp2[nid] = between(r, 5i32, 9i32);
                            } else if nty == PT_WATR
                                || nty == PT_DSTW
                                || nty == PT_SLTW
                                || nty == PT_CBNW
                            {
                                // `ELEC.cpp:88-98`: the water slot restarts as
                                // O2 on one roll in three and H2 otherwise; the
                                // electron is consumed either way.
                                if chance(r, 1u32, 3u32) {
                                    change_type_in_place(
                                        spawn_defaults,
                                        part_ty,
                                        part_life,
                                        part_ctype,
                                        part_temp,
                                        part_tmp,
                                        part_vx,
                                        part_vy,
                                        nid,
                                        PT_O2,
                                    );
                                } else {
                                    change_type_in_place(
                                        spawn_defaults,
                                        part_ty,
                                        part_life,
                                        part_ctype,
                                        part_temp,
                                        part_tmp,
                                        part_vx,
                                        part_vy,
                                        nid,
                                        PT_H2,
                                    );
                                }
                                dead = true;
                            } else if nty == PT_PROT || nty == PT_NEUT {
                                // `ELEC.cpp:99-105`: a proton only reacts when
                                // it has not captured an electron (`tmp2 & 1`),
                                // then the nucleon becomes H2 and the electron
                                // is consumed. `part_change_type` keeps the
                                // fields apart from the two cleared words.
                                if nty != PT_PROT || (part_tmp2[nid] & 1i32) == 0i32 {
                                    part_ty[nid] = PT_H2;
                                    part_life[nid] = 0i32;
                                    part_ctype[nid] = 0u32;
                                    dead = true;
                                }
                            } else if nty == PT_DEUT {
                                // `ELEC.cpp:106-111`: the deuteron charges its
                                // 6000-frame timer and cools to 0 K.
                                if part_life[nid] < 6000i32 {
                                    part_life[nid] += 1i32;
                                }
                                part_temp[nid] = 0.0f32;
                                dead = true;
                            } else if nty == PT_EXOT {
                                // `ELEC.cpp:112-116`: exotic matter charges and
                                // resets its collapse timer; the electron
                                // survives.
                                part_tmp2[nid] += 5i32;
                                part_life[nid] = 1000i32;
                            } else if nty == PT_RSST {
                                // `ELEC.cpp:117-126`: only the electron's own
                                // pixel (an RSST underneath it) annihilates
                                // both.
                                if dx == 0i32 && dy == 0i32 {
                                    part_ty[nid] = PT_NONE;
                                    dead = true;
                                }
                            } else if nty != PT_NONE
                                && nty != PT_ELEC
                                && nty != PT_SPRK
                                && (elem_word(elem_words, nty, WORD_PROPERTIES) & PROP_CONDUCTS)
                                    != 0u32
                                && part_life[nid] == 0i32
                                && (nty != PT_NBLE || part_temp[i] < NBLE_SPARK_TEMP)
                            {
                                // `ELEC.cpp:127-134` + `Simulation.cpp:1851-1873`:
                                // the conductor is sparked in place and the
                                // electron is consumed. `PT_NBLE` only conducts
                                // while the electron is below 2273.15 K.
                                spark_conductor(
                                    part_ty, part_life, part_ctype, part_temp, nid, nty, legacy,
                                );
                                dead = true;
                            }
                        }
                    }
                }
                if dead {
                    part_ty[i] = PT_NONE;
                }
            }
        } else if ty != PT_NONE
            && ty != PT_BIZR
            && ty != PT_BIZRG
            && ty != PT_BIZRS
            && (elem_word(elem_words, ty, WORD_PROPERTIES) & TYPE_ENERGY) == 0u32
        {
            // `BIZR.cpp:54-82`, inverted: this particle resolves the paints
            // around it. Only this thread writes its `dcolour`. TPT's
            // `UpdateParticles` walks the *slot* array, so a target touched by
            // several painters receives the blends in ascending slot order;
            // the gather therefore applies the painters by repeated
            // minimum-slot selection instead of in ring order. (A painter's
            // own colour is never blended — `BIZR.cpp:66` skips the whole
            // `BIZR` family — so the painter order is stable for the frame.)
            let bx = (part_x[i] + 0.5f32) as i32;
            let by = (part_y[i] + 0.5f32) as i32;
            let mut col = part_dcolour[i];
            let mut painted = false;
            let mut next_slot = 0u32;
            for _rank in 0u32..25u32 {
                let mut best_slot = 0xFFFF_FFFFu32;
                let mut best_paint = 0u32;
                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, bx + dx, by + dy);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            let nty = part_ty[nid];
                            if (nty == PT_BIZR || nty == PT_BIZRG || nty == PT_BIZRS)
                                && (nid as u32) >= next_slot
                            {
                                let paint = part_dcolour[nid];
                                if paint != 0u32 && (nid as u32) < best_slot {
                                    best_slot = nid as u32;
                                    best_paint = paint;
                                }
                            }
                        }
                    }
                }
                if best_slot == 0xFFFF_FFFFu32 {
                    // No further painter touches this pixel.
                    break;
                }
                let mr = (best_paint >> 16u32) & 0xFFu32;
                let mg = (best_paint >> 8u32) & 0xFFu32;
                let mb = best_paint & 0xFFu32;
                let ma = (best_paint >> 24u32) & 0xFFu32;
                let tr = (col >> 16u32) & 0xFFu32;
                let tg = (col >> 8u32) & 0xFFu32;
                let tb = col & 0xFFu32;
                let ta = (col >> 24u32) & 0xFFu32;
                let nr = blend_channel(tr, mr);
                let ng = blend_channel(tg, mg);
                let nb = blend_channel(tb, mb);
                let na = blend_channel(ta, ma);
                col = (nr << 16u32) | (ng << 8u32) | nb | (na << 24u32);
                painted = true;
                next_slot = best_slot + 1u32;
            }
            if painted {
                part_dcolour[i] = col;
            }
        }
    }
}
