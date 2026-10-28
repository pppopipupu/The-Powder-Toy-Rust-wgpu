//! Pressure gases: ANAR, BOYL, the C5 cold trigger and the `O2` graviton
//! burst.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::*;
use crate::reaction_kernel::neut::change_type_in_place;

/// Low nine bits of a `pmap` word hold the element id (`PMAPMASK`).
const PMAP_TYPE: u32 = 0x1FF;

/// `pv[cx, cy]` pulled towards `limit` when it is below it (`BOYL.cpp:57-70`).
#[cube]
fn pv_raise(pv: &mut [f32], xcells: u32, ycells: u32, cx: i32, cy: i32, limit: f32) {
    if cx >= 0i32 && cy >= 0i32 && (cx as u32) < xcells && (cy as u32) < ycells {
        let i = ((cy as u32) * xcells + (cx as u32)) as usize;
        if pv[i] < limit {
            pv[i] += 0.001f32 * (limit - pv[i]);
        }
    }
}

/// `pv[cx, cy] += 0.001 * (limit - pv)` without the below-limit guard
/// (`BOYL.cpp:63-70`).
#[cube]
fn pv_relax(pv: &mut [f32], xcells: u32, ycells: u32, cx: i32, cy: i32, limit: f32) {
    if cx >= 0i32 && cy >= 0i32 && (cx as u32) < xcells && (cy as u32) < ycells {
        let i = ((cy as u32) * xcells + (cx as u32)) as usize;
        pv[i] += 0.001f32 * (limit - pv[i]);
    }
}

/// `ANAR`, `BOYL` and the `C5` cold trigger.
///
/// These elements only need the air pressure, the type and the temperature, so
/// they fit in their own six-binding kernel instead of overflowing the
/// explosive pass.
#[cube(launch_unchecked)]
pub(crate) fn pressure_gas_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    air_pv: &mut [f32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
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
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0x85EB_CA6Bu32));

                if ty == PT_ANAR {
                    // `ANAR.cpp:52-69`: anti-air dust freezes into a cold flame.
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                if part_ty[nid] == PT_HFLM && chance(seed ^ n, 1u32, 4u32) {
                                    part_ty[id] = PT_HFLM;
                                    part_life[id] =
                                        50i32 + rand_range(seed ^ n ^ 0x77u32, 0i32, 149i32);
                                    part_temp[id] = 0.0f32;
                                    part_temp[nid] = 0.0f32;
                                    air_pv[cidx] -= 0.5f32;
                                }
                            }
                        }
                    }
                } else if ty == PT_BOYL {
                    // `BOYL.cpp:55-92`: Boyle gas pulls the 3x3 coarse cells
                    // towards `temp / 100` and then reacts with water and O2.
                    let limit = part_temp[id] / 100.0f32;
                    let cx = px / (CELL as i32);
                    let cy = py / (CELL as i32);
                    pv_raise(air_pv, xcells, ycells, cx, cy, limit);
                    pv_raise(air_pv, xcells, ycells, cx, cy + 1i32, limit);
                    pv_raise(air_pv, xcells, ycells, cx, cy - 1i32, limit);
                    pv_relax(air_pv, xcells, ycells, cx + 1i32, cy, limit);
                    pv_relax(air_pv, xcells, ycells, cx + 1i32, cy + 1i32, limit);
                    pv_relax(air_pv, xcells, ycells, cx - 1i32, cy, limit);
                    pv_relax(air_pv, xcells, ycells, cx - 1i32, cy - 1i32, limit);

                    let mut reacted = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !reacted {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if nty == PT_WATR && chance(seed ^ n, 1u32, 30u32) {
                                    part_ty[nid] = PT_FOG;
                                } else if nty == PT_O2 && chance(seed ^ n ^ 0x1u32, 1u32, 9u32) {
                                    part_ty[nid] = PT_NONE;
                                    part_ty[id] = PT_WATR;
                                    air_pv[cidx] += 4.0f32;
                                    reacted = true;
                                }
                            }
                        }
                    }
                } else if ty == PT_C5 {
                    // `C5.cpp:58-81`: any cold, non-insulating neighbour or a
                    // cold flame turns the explosive into a cold flame.
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if nty != PT_NONE {
                                    let mut cold = false;
                                    if nty != PT_C5
                                        && part_temp[nid] < 100.0f32
                                        && !heat_insulator(elem_words, nty)
                                    {
                                        cold = true;
                                    }
                                    if nty == PT_HFLM {
                                        cold = true;
                                    }
                                    if cold && chance(seed ^ n, 1u32, 6u32) {
                                        part_ty[id] = PT_HFLM;
                                        part_temp[id] = 0.0f32;
                                        part_temp[nid] = 0.0f32;
                                        part_life[id] =
                                            50i32 + rand_range(seed ^ n ^ 0x31u32, 0i32, 149i32);
                                        air_pv[cidx] += 1.5f32;
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

/// `O2.cpp:80-123`: a hot, over-pressured oxygen pixel sitting in a strong
/// Newtonian gravity field collapses into a broken-metal pellet and sprays
/// energy out of the same pixel.
///
/// This runs as its own pass because the shared combustion kernel already
/// binds sixteen storage buffers and the burst has to read the solved
/// `gravOut.forceX/Y` field.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn o2_burst_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_ctype: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    create_queue: &mut [Atomic<u32>],
    spawn_defaults: &[u32],
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
            // The shared combustion kernel may have turned this pixel into
            // fire earlier in the same frame; `pmap` is only refreshed after
            // the passes, so the element type is the authority here.
            if part_ty[id] == PT_O2 {
                // A frozen particle does not run its `Update` (`stasis`).
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                // `O2.cpp:80`: only white-hot oxygen under extreme pressure
                // fuses; the temperature and the cell pressure are read once.
                if part_temp[id] > 9973.15f32 && air_pv[cidx] > 250.0f32 {
                    let gravx = grav_fx[cidx];
                    let gravy = grav_fy[cidx];
                    // `O2.cpp:82`: |force| > 20 in TPT's cell units.
                    let prand = hash_u32(frame_seed ^ mix_index(idx as u32, 2_654_435_761u32));
                    if gravx * gravx + gravy * gravy > 400.0f32 && chance(prand, 1u32, 5u32) {
                        // `O2.cpp:85`: `create_part(i, x, y, PT_BRMT)` restarts
                        // the slot from the `BRMT` defaults.
                        change_type_in_place(
                            spawn_defaults,
                            part_ty,
                            part_life,
                            part_ctype,
                            part_temp,
                            part_tmp,
                            part_vx,
                            part_vy,
                            id,
                            PT_BRMT,
                        );
                        // `create_part` copies the whole default row, so the
                        // broken metal starts with `tmp2 = 0` as well.
                        part_tmp2[id] = 0i32;
                        // `O2.cpp:87-92`: the same-pixel energy products. One
                        // photons word exists per pixel, so a later record
                        // overwrites the previous map entry just like TPT's
                        // successive `create_part(-3, ...)` calls; the slots
                        // themselves all stay alive in the particle pool.
                        request_create(
                            create_queue,
                            PT_NEUT,
                            px,
                            py,
                            CR_NO_OCCUPANCY | CR_SET_TEMP,
                            0i32,
                            0u32,
                            MAX_TEMP,
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
                            PT_PHOT,
                            px,
                            py,
                            CR_NO_OCCUPANCY | CR_SET_TEMP | CR_SET_TMP,
                            0i32,
                            0u32,
                            MAX_TEMP,
                            1i32,
                            0i32,
                            0i32,
                            0i32,
                            0u32,
                            0u32,
                            0.0f32,
                            0.0f32,
                        );
                        // `O2.cpp:94-105`: the plasma lands in a random
                        // neighbour, gated by `can_move[PT_PLSM][rt] ||
                        // rt == PT_O2`. The device table only admits empty
                        // pixels plus the explicit oxygen case, exactly like
                        // the hydrogen fusion burst in `combustion/fire.rs`;
                        // the burst's own pixel is skipped because it now
                        // belongs to the broken metal.
                        let rx = px + (hash_u32(prand ^ 0x85EB_CA6Bu32) % 3u32) as i32 - 1i32;
                        let ry = py + (hash_u32(prand ^ 0xC2B2_AE35u32) % 3u32) as i32 - 1i32;
                        let rt = pmap_at(pmap, xres, yres, rx, ry) & PMAP_TYPE;
                        if (rx != px || ry != py) && (rt == PT_NONE || rt == PT_O2) {
                            request_create(
                                create_queue,
                                PT_PLSM,
                                rx,
                                ry,
                                CR_NO_OCCUPANCY | CR_SET_TEMP | CR_SET_TMP,
                                0i32,
                                0u32,
                                MAX_TEMP,
                                4i32,
                                0i32,
                                0i32,
                                0i32,
                                0u32,
                                0u32,
                                0.0f32,
                                0.0f32,
                            );
                        }
                        // `O2.cpp:106-109`: the graviton is created last, so
                        // its word owns the photons layer at this pixel.
                        request_create(
                            create_queue,
                            PT_GRVT,
                            px,
                            py,
                            CR_NO_OCCUPANCY | CR_SET_TEMP,
                            0i32,
                            0u32,
                            MAX_TEMP,
                            0i32,
                            0i32,
                            0i32,
                            0i32,
                            0u32,
                            0u32,
                            0.0f32,
                            0.0f32,
                        );
                        // `O2.cpp:110-111`: the pellet and the cell go to the
                        // caps.
                        part_temp[id] = MAX_TEMP;
                        air_pv[cidx] = MAX_PRESSURE;
                    }
                }
            }
        }
    }
}
