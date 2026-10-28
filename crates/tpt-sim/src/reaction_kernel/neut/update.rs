//! `NEUT.cpp:56-244`: the neutron's 3x3 transmutation table.
//!
//! Split from the parent module because the table is long; the shared
//! `#[cube]` helpers live in [`super`].

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{
    CR_NO_OCCUPANCY, CR_SET_TEMP, WORD_CTYPE_IN, elem_word, hash_u32, mix_index, pmap_at,
    pmap_slot, request_create,
};
use super::{DEUT_MAX_BURST, chance, clampf, fire_update_subcall, rebuild_type_in_place, seed_u32};

/// `NEUT.cpp:56-244`: the 3x3 transmutation table, `PLUT` fission, `DEUT`
/// detonation and Cherenkov radiation.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn neut_reaction_kernel(
    photons: &[u32],
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_flags: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    elem_words: &[u32],
    spawn_defaults: &[u32],
    air_pv: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = photons[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_NEUT {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                // `unsigned int pressureFactor = 3 + (int)pv`: the C++ sum wraps
                // through `unsigned`, so a strongly negative pressure makes every
                // roll succeed. `as u32` reproduces that wrap.
                let pf = (3i32 + air_pv[cidx] as i32) as u32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 2_654_435_761u32));
                let mut dead = false;

                for n in 0u32..9u32 {
                    if !dead {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            let nty = part_ty[nid];
                            let nroll = roll ^ mix_index(n, 0x9E37_79B9u32);

                            if nty == PT_WATR {
                                // `NEUT + WATR -> NEUT + DSTW`, then the
                                // fall-through into the ice/snow damping.
                                if chance(nroll, 3u32, 20u32) {
                                    part_ty[nid] = PT_DSTW;
                                }
                                part_vx[id] *= 0.995f32;
                                part_vy[id] *= 0.995f32;
                            } else if nty == PT_ICEI || nty == PT_SNOW {
                                part_vx[id] *= 0.995f32;
                                part_vy[id] *= 0.995f32;
                            } else if nty == PT_PLUT {
                                // `NEUT + PLUT -> NEUT + URAN/LAVA(PLUT)` (1 in 3)
                                // or `-> 2xNEUT` (2 in 3), then the fission heat
                                // and a `FIRE` subcall.
                                if chance(nroll, pf, 1000u32) {
                                    if chance(nroll ^ 0x51ED_2701u32, 1u32, 3u32) {
                                        if chance(nroll ^ 0x2545_F491u32, 2u32, 3u32) {
                                            // LAVA(PLUT): `Create` sets a
                                            // 240..359 burn timer
                                            // (`LAVA.cpp:73-76`).
                                            rebuild_type_in_place(
                                                spawn_defaults,
                                                part_ty,
                                                part_life,
                                                part_ctype,
                                                part_temp,
                                                part_tmp,
                                                part_flags,
                                                part_vx,
                                                part_vy,
                                                nid,
                                                PT_LAVA,
                                            );
                                            part_life[nid] =
                                                240i32 + ((hash_u32(nroll) % 120u32) as i32);
                                            part_ctype[nid] = PT_PLUT;
                                            part_tmp[nid] = 100i32;
                                        } else {
                                            rebuild_type_in_place(
                                                spawn_defaults,
                                                part_ty,
                                                part_life,
                                                part_ctype,
                                                part_temp,
                                                part_tmp,
                                                part_flags,
                                                part_vx,
                                                part_vy,
                                                nid,
                                                PT_URAN,
                                            );
                                        }
                                        part_temp[nid] = MAX_TEMP;
                                    } else {
                                        // 2xNEUT: the fresh neutron keeps the
                                        // source neutron's velocity
                                        // (`NEUT.cpp:95-97`) and gets the
                                        // `NEUT.cpp:252-258` create hook.
                                        rebuild_type_in_place(
                                            spawn_defaults,
                                            part_ty,
                                            part_life,
                                            part_ctype,
                                            part_temp,
                                            part_tmp,
                                            part_flags,
                                            part_vx,
                                            part_vy,
                                            nid,
                                            PT_NEUT,
                                        );
                                        part_life[nid] = 480i32
                                            + ((hash_u32(nroll ^ 0x27D4_EB2Fu32) % 480u32) as i32);
                                        // `NEUT.cpp:95-97`: the create hook
                                        // leaves `vx = r*cos(a)` before the
                                        // update blends a quarter of it with
                                        // the parent's velocity.
                                        let h = hash_u32(nroll ^ 0x27D4_EB2Fu32);
                                        let angle = (h & 0xFFFFu32) as f32
                                            * (core::f32::consts::TAU / 65_535.0f32);
                                        let radius =
                                            (128.0f32 + ((h >> 16u32) & 0x7Fu32) as f32) / 127.0f32;
                                        part_vx[nid] = 0.25f32 * angle.cos() * radius + part_vx[id];
                                        part_vy[nid] = 0.25f32 * angle.sin() * radius + part_vy[id];
                                    }
                                    air_pv[cidx] += 10.0f32;
                                    fire_update_subcall(
                                        pmap, elem_words, air_pv, part_ty, part_life, part_ctype,
                                        part_temp, part_tmp, xres, yres, xcells, px, py, nroll,
                                    );
                                }
                            } else if nty == PT_DEUT {
                                let dl = part_life[nid];
                                let dt = part_temp[nid];
                                let mut extra = seed_u32(nroll, 0u32);
                                if dl > 0i32 {
                                    extra = (dl / 100i32) as u32;
                                }
                                if chance(nroll, pf + 1u32 + extra, 1000u32) {
                                    // `DeutExplosion` (`NEUT.cpp:271-289`):
                                    // `n = life/50` clamped to 1..=340 and every
                                    // product is `create_part(-3, x, y, PT_NEUT)`
                                    // on the DEUT's own pixel, so the burst stacks
                                    // there and only the last allocation keeps the
                                    // map word. `CR_NO_OCCUPANCY` is that `-3`; no
                                    // product is dropped beyond the queue's own
                                    // capacity.
                                    let mut count = seed_u32(nroll, 1u32);
                                    if dl / 50i32 > 1i32 {
                                        count = (dl / 50i32) as u32;
                                    }
                                    if count > DEUT_MAX_BURST {
                                        count = DEUT_MAX_BURST;
                                    }
                                    let burst_temp =
                                        clampf(dt + (dl as f32) * 500.0f32, MIN_TEMP, MAX_TEMP);
                                    let mut c = seed_u32(count, 0u32);
                                    for _ in 0u32..DEUT_MAX_BURST {
                                        if c < count {
                                            request_create(
                                                create_queue,
                                                PT_NEUT,
                                                px + dx,
                                                py + dy,
                                                CR_NO_OCCUPANCY | CR_SET_TEMP,
                                                0i32,
                                                0u32,
                                                burst_temp,
                                                0i32,
                                                0i32,
                                                0i32,
                                                0i32,
                                                0u32,
                                                0u32,
                                                0.0f32,
                                                0.0f32,
                                            );
                                            c += 1u32;
                                        }
                                    }
                                    let dcidx = (((py + dy) as u32 / CELL) * xcells
                                        + ((px + dx) as u32 / CELL))
                                        as usize;
                                    air_pv[dcidx] += 6.0f32 * (count as f32);
                                    part_ty[nid] = PT_NONE;
                                }
                            } else if nty == PT_GUNP {
                                if chance(nroll, 3u32, 200u32) {
                                    part_ty[nid] = PT_DUST;
                                }
                            } else if nty == PT_DYST {
                                if chance(nroll, 3u32, 200u32) {
                                    part_ty[nid] = PT_YEST;
                                }
                            } else if nty == PT_YEST {
                                part_ty[nid] = PT_DYST;
                            } else if nty == PT_C4 {
                                // `NEUT.cpp` `case PT_n` — TPT's internal name
                                // for C-4, which the port keeps as `PT_C4`.
                                if chance(nroll, 3u32, 200u32) {
                                    part_ty[nid] = PT_GOO;
                                }
                            } else if nty == PT_NITR {
                                if chance(nroll, 3u32, 200u32) {
                                    part_ty[nid] = PT_DESL;
                                }
                            } else if nty == PT_PLNT {
                                if chance(nroll, 1u32, 20u32) {
                                    rebuild_type_in_place(
                                        spawn_defaults,
                                        part_ty,
                                        part_life,
                                        part_ctype,
                                        part_temp,
                                        part_tmp,
                                        part_flags,
                                        part_vx,
                                        part_vy,
                                        nid,
                                        PT_WOOD,
                                    );
                                }
                            } else if nty == PT_DESL || nty == PT_OIL {
                                if chance(nroll, 3u32, 200u32) {
                                    part_ty[nid] = PT_GAS;
                                }
                            } else if nty == PT_COAL {
                                if chance(nroll, 1u32, 20u32) {
                                    rebuild_type_in_place(
                                        spawn_defaults,
                                        part_ty,
                                        part_life,
                                        part_ctype,
                                        part_temp,
                                        part_tmp,
                                        part_flags,
                                        part_vx,
                                        part_vy,
                                        nid,
                                        PT_WOOD,
                                    );
                                }
                            } else if nty == PT_BCOL {
                                if chance(nroll, 1u32, 20u32) {
                                    rebuild_type_in_place(
                                        spawn_defaults,
                                        part_ty,
                                        part_life,
                                        part_ctype,
                                        part_temp,
                                        part_tmp,
                                        part_flags,
                                        part_vx,
                                        part_vy,
                                        nid,
                                        PT_SAWD,
                                    );
                                }
                            } else if nty == PT_DUST {
                                if chance(nroll, 1u32, 20u32) {
                                    part_ty[nid] = PT_FWRK;
                                }
                            } else if nty == PT_FWRK {
                                if chance(nroll, 1u32, 20u32) {
                                    part_ctype[nid] = PT_DUST;
                                }
                            } else if nty == PT_ACID {
                                if chance(nroll, 1u32, 20u32) {
                                    rebuild_type_in_place(
                                        spawn_defaults,
                                        part_ty,
                                        part_life,
                                        part_ctype,
                                        part_temp,
                                        part_tmp,
                                        part_flags,
                                        part_vx,
                                        part_vy,
                                        nid,
                                        PT_ISOZ,
                                    );
                                }
                            } else if nty == PT_TTAN {
                                if chance(nroll, 1u32, 20u32) {
                                    dead = true;
                                }
                            } else if nty == PT_EXOT {
                                if chance(nroll, 1u32, 20u32) {
                                    part_life[nid] = 1500i32;
                                }
                            } else if nty == PT_RFRG {
                                if chance(nroll, 1u32, 2u32) {
                                    rebuild_type_in_place(
                                        spawn_defaults,
                                        part_ty,
                                        part_life,
                                        part_ctype,
                                        part_temp,
                                        part_tmp,
                                        part_flags,
                                        part_vx,
                                        part_vy,
                                        nid,
                                        PT_GAS,
                                    );
                                } else {
                                    rebuild_type_in_place(
                                        spawn_defaults,
                                        part_ty,
                                        part_life,
                                        part_ctype,
                                        part_temp,
                                        part_tmp,
                                        part_flags,
                                        part_vx,
                                        part_vy,
                                        nid,
                                        PT_CAUS,
                                    );
                                }
                            } else if nty == PT_RSSS {
                                // Only the neutron's own pixel (`!rx && !ry`).
                                if dx == 0i32 && dy == 0i32 {
                                    let ct = part_ctype[nid];
                                    let stored = part_tmp[nid];
                                    if ct > 0u32 && ct < PT_NUM as u32 {
                                        rebuild_type_in_place(
                                            spawn_defaults,
                                            part_ty,
                                            part_life,
                                            part_ctype,
                                            part_temp,
                                            part_tmp,
                                            part_flags,
                                            part_vx,
                                            part_vy,
                                            nid,
                                            ct,
                                        );
                                        // Restore the stored `ctype` only for
                                        // elements that carry a type there.
                                        if stored > 0i32
                                            && (stored as u32) < PT_NUM as u32
                                            && elem_word(elem_words, stored as u32, WORD_CTYPE_IN)
                                                != 0u32
                                        {
                                            part_ctype[nid] = stored as u32;
                                        }
                                    } else {
                                        part_ty[nid] = PT_RSST;
                                    }
                                    dead = true;
                                }
                            } else if nty == PT_BASE
                                && part_temp[nid] > 323.15f32
                                && chance(nroll, 1u32, 35u32)
                            {
                                rebuild_type_in_place(
                                    spawn_defaults,
                                    part_ty,
                                    part_life,
                                    part_ctype,
                                    part_temp,
                                    part_tmp,
                                    part_flags,
                                    part_vx,
                                    part_vy,
                                    nid,
                                    PT_LRBD,
                                );
                            }
                        }
                    }
                }

                if dead {
                    part_ty[id] = PT_NONE;
                }
            }
        }
    }
}
