//! Carbonated water: degassing, the `tmp` blast timer and rubber ignition.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::*;

/// `CBNW.cpp:126`: soda water ignites rubber above `273.15 + 12` K.
pub(super) const CBNW_RUBBER_IGNITE: f32 = 285.15f32;

/// `CBNW.cpp:51-145`: the whole carbonated-water `Update`.
///
/// Pixel parallel: `pmap` names the pixel's own particle, so the 3x3
/// neighbourhood is a strided read of the same map. The pass runs after the
/// base combustion kernel, exactly like the other per-element updates that
/// follow it, so a neighbour the sponge turned into `CO2` or the fire killed is
/// already visible in the live `part_ty`.
///
/// # Cross-slot state
///
/// * `CBNW.cpp:106-123` shares the blast charge between touching soda pixels by
///   writing the neighbour's `tmp`. TPT resolves a pair in slot order; the
///   parallel pass applies the same transfer from whichever side scans first,
///   and two simultaneous halves keep one value.
/// * `CBNW.cpp:62,81,104` add to the coarse-cell `pv`. The add is a plain
///   read-modify-write, so two soda pixels in one cell can lose one bump;
///   `pv` has no atomic float add in WGSL.
#[cube(launch_unchecked)]
pub(crate) fn carbonated_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
    legacy: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_CBNW {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 2_246_822_519u32));
                // TPT re-reads the cell for every branch; the local copy keeps
                // those reads ordered inside this thread.
                let mut pv = air_pv[cidx];

                // `CBNW.cpp:55-64`: below 3 pressure the dissolved gas can
                // escape. A vacuum (<= -0.5) outgasses every frame, otherwise
                // one roll in 4000.
                if pv <= 3.0f32 && (pv <= -0.5f32 || chance(seed, 1u32, 4000u32)) {
                    // `CBNW.cpp:59-62`: `part_change_type` keeps the
                    // fields; only the badge and the pressure change.
                    part_ty[id] = PT_CO2;
                    part_ctype[id] = 5u32;
                    pv += 0.5f32;
                    air_pv[cidx] = pv;
                }

                // `CBNW.cpp:65-71`: the graphics speckle timer drifts to 20,
                // then re-rolls to 0..=39 once every 200 frames.
                if part_tmp2[id] != 20i32 {
                    if part_tmp2[id] > 20i32 {
                        part_tmp2[id] -= 1i32;
                    } else {
                        part_tmp2[id] += 1i32;
                    }
                } else if chance(seed ^ 0x01u32, 1u32, 200u32) {
                    part_tmp2[id] = rand_range(seed ^ 0x02u32, 0i32, 39i32);
                }

                // `CBNW.cpp:73-84`: a charged pixel counts down; the last unit
                // pops into `CO2` three times out of four.
                if part_tmp[id] > 0i32 {
                    if part_tmp[id] == 1i32 && chance(seed ^ 0x03u32, 3u32, 4u32) {
                        part_ty[id] = PT_CO2;
                        part_ctype[id] = 5u32;
                        pv += 0.2f32;
                        air_pv[cidx] = pv;
                    }
                    part_tmp[id] -= 1i32;
                }

                // `CBNW.cpp:85-144`: the 3x3 neighbourhood.
                for n in 0u32..9u32 {
                    let rx = (n % 3u32) as i32 - 1i32;
                    let ry = (n / 3u32) as i32 - 1i32;
                    if rx != 0i32 || ry != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            let nty = part_ty[nid];
                            // A neighbour killed earlier in this pass is gone
                            // from TPT's `pmap`; skip the stale map word.
                            if nty != PT_NONE {
                                let nprops = elem_word(elem_words, nty, WORD_PROPERTIES);
                                let nroll = hash_u32(seed ^ mix_index(n, 0x9E37_79B9u32));

                                // `CBNW.cpp:94-98`: any powder seeds the blast
                                // timer, one neighbour in 83.
                                if (nprops & TYPE_PART) != 0u32
                                    && part_tmp[id] == 0i32
                                    && chance(nroll, 1u32, 83u32)
                                {
                                    part_tmp[id] = rand_range(nroll ^ 0x10u32, 0i32, 24i32);
                                } else if (nprops & TYPE_SOLID) != 0u32
                                    && nty != PT_DMND
                                    && nty != PT_GLAS
                                    && part_tmp[id] == 0i32
                                {
                                    // `CBNW.cpp:99-105`: a solid (but not
                                    // diamond or glass) also nucleates the
                                    // gas. `int(2 - pv)` can be negative, and
                                    // TPT's `chance` never fires then.
                                    let num = (2.0f32 - pv) as i32;
                                    if num > 0i32 && chance(nroll ^ 0x11u32, num as u32, 6667u32) {
                                        part_ty[id] = PT_CO2;
                                        part_ctype[id] = 5u32;
                                        pv += 0.2f32;
                                        air_pv[cidx] = pv;
                                    }
                                }

                                // `CBNW.cpp:106-123`: a pair of soda pixels
                                // shares the blast charge.
                                if nty == PT_CBNW {
                                    if part_tmp[id] == 0i32 {
                                        if part_tmp[nid] != 0i32 {
                                            part_tmp[id] = part_tmp[nid];
                                            if nid > id {
                                                // The higher slot has not run
                                                // its countdown yet.
                                                part_tmp[id] -= 1i32;
                                            }
                                        }
                                    } else if part_tmp[nid] == 0i32 {
                                        part_tmp[nid] = part_tmp[id];
                                        if nid > id {
                                            part_tmp[nid] += 1i32;
                                        }
                                    }
                                } else if nty == PT_RBDM || nty == PT_LRBD {
                                    // `CBNW.cpp:124-133`: warm soda water
                                    // turns into a short-lived water flame next
                                    // to rubber, one frame in 166.
                                    if (legacy != 0u32 || part_temp[id] > CBNW_RUBBER_IGNITE)
                                        && chance(nroll ^ 0x12u32, 1u32, 166u32)
                                    {
                                        part_ty[id] = PT_FIRE;
                                        part_life[id] = 4i32;
                                        part_ctype[id] = PT_WATR;
                                    }
                                } else if nty == PT_FIRE && part_ctype[nid] != PT_WATR {
                                    // `CBNW.cpp:134-141`: soda water puts a
                                    // flame out, and one time in fifty the
                                    // water is spent too (`return 1`).
                                    part_ty[nid] = PT_NONE;
                                    if chance(nroll ^ 0x13u32, 1u32, 50u32) {
                                        part_ty[id] = PT_NONE;
                                        terminate!();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
