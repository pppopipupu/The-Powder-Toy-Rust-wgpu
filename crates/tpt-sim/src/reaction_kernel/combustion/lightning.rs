//! Lightning heating, propagation and C5 photon kernels.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::*;

/// `LIGH` heating, ignition and conductor sparking (`LIGH.cpp:70-240`).
///
/// The `STKM` branch subtracts the rig's life unconditionally: the port has no
/// player-held element, so `player.elem != PT_LIGH` always holds.
#[cube(launch_unchecked)]
pub(crate) fn ligh_heat_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_ctype: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_hv: &mut [f32],
    air_pv: &mut [f32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
    aheat: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_LIGH {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0x27D4_EB2Fu32));
                let powderful = (part_temp[id]
                    * (1.0f32 + part_life[id] as f32 / 40.0f32)
                    * LIGHTING_POWER) as i32;

                if aheat != 0u32 {
                    let mut hv = air_hv[cidx] + powderful as f32 / LIGH_HV_DIV;
                    if hv > MAX_TEMP || hv < 0.0f32 {
                        hv = MAX_TEMP;
                    }
                    air_hv[cidx] = hv;
                }

                let mut surround = false;
                for n in 0u32..9u32 {
                    let dx = (n % 3u32) as i32 - 1i32;
                    let dy = (n / 3u32) as i32 - 1i32;
                    if (dx != 0i32 || dy != 0i32)
                        && pmap_at(pmap, xres, yres, px + dx, py + dy) == 0u32
                    {
                        surround = true;
                    }
                }

                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            let nty = part_ty[nid];
                            if nty != PT_NONE {
                                let ncidx = ((((py + dy) as u32) / CELL) * xcells
                                    + (((px + dx) as u32) / CELL))
                                    as usize;
                                let nflam = elem_word(elem_words, nty, WORD_FLAMMABLE) as i32;
                                let nexp = elem_word(elem_words, nty, WORD_EXPLOSIVE) as i32;
                                let mut prob_num = nflam + (air_pv[ncidx] * 10.0f32) as i32;
                                if prob_num < 0i32 {
                                    prob_num = 0i32;
                                }
                                let spng_ok = nty != PT_SPNG || part_life[nid] == 0i32;
                                if (surround || nexp != 0i32)
                                    && spng_ok
                                    && nflam != 0i32
                                    && chance(seed ^ n, prob_num as u32, 1000u32)
                                {
                                    part_ty[nid] = PT_FIRE;
                                    part_temp[nid] = clampf(
                                        default_temp_of(elem_words, PT_FIRE)
                                            + nflam as f32 / 2.0f32,
                                        MIN_TEMP,
                                        MAX_TEMP,
                                    );
                                    part_life[nid] =
                                        180i32 + rand_range(seed ^ n ^ 0x1u32, 0i32, 79i32);
                                    part_tmp[nid] = 0i32;
                                    part_ctype[nid] = 0u32;
                                    if nexp != 0i32 {
                                        air_pv[cidx] += 0.25f32;
                                    }
                                }

                                if nty == PT_LIGH || nty == PT_TESC {
                                    // Plasma keeps its state; nothing to do.
                                } else if nty == PT_CLNE
                                    || nty == PT_THDR
                                    || nty == PT_DMND
                                    || nty == PT_FIRE
                                {
                                    part_temp[nid] = clampf(
                                        part_temp[nid] + powderful as f32 / 10.0f32,
                                        MIN_TEMP,
                                        MAX_TEMP,
                                    );
                                } else if nty == PT_DEUT || nty == PT_PLUT {
                                    part_temp[nid] = clampf(
                                        part_temp[nid] + powderful as f32,
                                        MIN_TEMP,
                                        MAX_TEMP,
                                    );
                                    air_pv[cidx] = clampf(
                                        air_pv[cidx] + powderful as f32 / 35.0f32,
                                        MIN_PRESSURE,
                                        MAX_PRESSURE,
                                    );
                                    if chance(seed ^ n ^ 0x2u32, 1u32, 3u32) {
                                        part_ty[nid] = PT_NEUT;
                                        part_life[nid] =
                                            480i32 + rand_range(seed ^ n ^ 0x3u32, 0i32, 479i32);
                                        part_vx[nid] =
                                            rand_range(seed ^ n ^ 0x4u32, -5i32, 5i32) as f32;
                                        part_vy[nid] =
                                            rand_range(seed ^ n ^ 0x5u32, -5i32, 5i32) as f32;
                                    }
                                } else if nty == PT_COAL || nty == PT_BCOL {
                                    if part_life[nid] > 100i32 {
                                        part_life[nid] = 99i32;
                                    }
                                } else if nty == PT_STKM || nty == PT_STKM2 {
                                    // `LIGH.cpp:139-147`: the rig loses life
                                    // unless the player is holding lightning.
                                    part_life[nid] -= powderful / 100i32;
                                } else if nty == PT_HEAC {
                                    part_temp[nid] = clampf(
                                        part_temp[nid] + powderful as f32 / 10.0f32,
                                        MIN_TEMP,
                                        MAX_TEMP,
                                    );
                                    let high = f32::reinterpret(elem_word(
                                        elem_words,
                                        PT_HEAC,
                                        WORD_HIGH_TEMP,
                                    ));
                                    if part_temp[nid] > high {
                                        part_ty[nid] = PT_LAVA;
                                        part_ctype[nid] = PT_HEAC;
                                    }
                                }

                                let nprops = elem_word(elem_words, nty, WORD_PROPERTIES);
                                if ((nprops & PROP_CONDUCTS) != 0u32 || nty == PT_INST)
                                    && part_life[nid] == 0i32
                                {
                                    part_ctype[nid] = nty;
                                    part_ty[nid] = PT_SPRK;
                                    part_life[nid] = 4i32;
                                }
                                air_pv[cidx] = clampf(
                                    air_pv[cidx] + powderful as f32 / 400.0f32,
                                    MIN_PRESSURE,
                                    MAX_PRESSURE,
                                );
                                if !heat_insulator(elem_words, nty) {
                                    part_temp[nid] = clampf(
                                        part_temp[nid] + powderful as f32 / 1.3f32,
                                        MIN_TEMP,
                                        MAX_TEMP,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `LIGH.cpp:236-300` `create_line_par` + `create_LIGH`: rasterise one bolt
/// segment from `(x1, y1)` to `(x2, y2)` and queue every free pixel.
///
/// The reference calls `create_part(-1, ...)` per pixel; an occupied pixel
/// makes the call fail and the line continues, while `VOID`/powered `PVOD`/
/// black holes eat the bolt and stop it. A line pixel outside the canvas also
/// stops it (`create_LIGH`'s out-of-bounds branch). `last` marks the endpoint,
/// which inherits a shorter life and the deferred branch/bend state.
///
/// `p > i` in `create_part`'s return always holds for the port's allocator
/// (new slots come from the top of the used window), so the `+1` deferral is
/// applied unconditionally when the parent was not a first pixel.
#[cube]
#[allow(clippy::too_many_arguments)]
fn queue_ligh_line(
    pmap: &[u32],
    // The bolt reads these four arrays only, but the kernel that owns them
    // hands out `&mut` slots (`part_ty`/`part_tmp`/`part_ctype` are written by
    // the same pass), and CubeCL's `Into` argument conversion has no
    // `&mut -> &` reborrow. They stay mutable exactly like `create_queue`.
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    temp: f32,
    life: i32,
    angle: i32,
    parent_tmp2: i32,
    dcolour: u32,
    seed: u32,
) {
    let reverse_xy = iabs(y2 - y1) > iabs(x2 - x1);
    let mut ax1 = x1;
    let mut ay1 = y1;
    let mut ax2 = x2;
    let mut ay2 = y2;
    if reverse_xy {
        ax1 = y1;
        ay1 = x1;
        ax2 = y2;
        ay2 = x2;
    }
    let back = ax1 > ax2;
    let dx = ax2 - ax1;
    let dy = iabs(ay2 - ay1);
    let mut de = 0.0f32;
    if dx != 0i32 {
        de = dy as f32 / dx as f32;
    }
    let mut y = ay1;
    let ystep = if ay1 < ay2 { 1i32 } else { -1i32 };
    let mut e = 0.0f32;
    let span = iabs(ax2 - ax1);
    let mut stopped = false;
    for k in 0u32..(span as u32 + 1u32) {
        if !stopped {
            let x = if back { ax1 - k as i32 } else { ax1 + k as i32 };
            let gx = if reverse_xy { y } else { x };
            let gy = if reverse_xy { x } else { y };
            let last = x == ax2;
            // `create_LIGH` stops the line outside the canvas.
            let mut stop = gx < 0i32 || gy < 0i32 || gx >= xres as i32 || gy >= yres as i32;
            if !stop {
                let nv = pmap_at(pmap, xres, yres, gx, gy);
                if nv != 0u32 {
                    let nid = pmap_slot(nv);
                    let nty = part_ty[nid];
                    let mut eaten = false;
                    if nty == PT_VOID || nty == PT_BHOL || nty == PT_NBHL {
                        eaten = true;
                    } else if nty == PT_PVOD && part_life[nid] == 10i32 {
                        // `LIGH.cpp:229`: an open powered void eats the bolt.
                        let open = part_ctype[nid] == 0u32
                            || (part_ctype[nid] == PT_LIGH) != ((part_tmp[nid] & 1i32) != 0i32);
                        if open {
                            eaten = true;
                        }
                    }
                    if eaten {
                        stop = true;
                    }
                    // Any other occupant makes `create_part(-1)` fail: the line
                    // still advances to the next pixel.
                } else {
                    let r = hash_u32(seed ^ mix_index(k, 0x27D4_EB2Fu32));
                    let mut seg_life = life;
                    let mut seg_tmp2 = 8i32;
                    if last {
                        // `LIGH.cpp:206-218`:
                        // `int(life/1.5 - rng.between(0,1))`.
                        let jitter = (hash_u32(r ^ 0x01u32) % 2u32) as f32;
                        let next_life = (life as f32 / 1.5f32 - jitter) as i32;
                        seg_life = next_life;
                        if next_life > 1i32 {
                            let branch = chance(hash_u32(r ^ 0x02u32), 7u32, 10u32);
                            let mut base = 0i32;
                            if branch {
                                base = 2i32;
                            }
                            // `(p > i) && tmp2 != 4`: the allocator hands out a
                            // slot above the parent's, so the deferred +1
                            // applies unless the parent was a first pixel.
                            if parent_tmp2 != 4i32 {
                                base += 1i32;
                            }
                            seg_tmp2 = base;
                        }
                    }
                    request_create(
                        create_queue,
                        PT_LIGH,
                        gx,
                        gy,
                        CR_SET_LIFE | CR_SET_TEMP | CR_SET_TMP | CR_SET_TMP2 | CR_SET_DCOLOUR,
                        seg_life,
                        0u32,
                        temp,
                        angle,
                        seg_tmp2,
                        0i32,
                        0i32,
                        dcolour,
                        0u32,
                        0.0f32,
                        0.0f32,
                    );
                }
            }
            if stop {
                stopped = true;
            }
            e += de;
            if back {
                if e <= -0.5f32 {
                    y += ystep;
                    e += 1.0f32;
                }
            } else if e >= 0.5f32 {
                y += ystep;
                e -= 1.0f32;
            }
        }
    }
}
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ligh_prop_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &[f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_ctype: &mut [u32],
    part_dcolour: &[u32],
    part_flags: &mut [u32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty == PT_LIGH {
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0x1656_67B1u32));
                let mut tmp2 = part_tmp2[id];
                let mut dead = false;
                if tmp2 == 1i32 || tmp2 == 3i32 || tmp2 == 6i32 || tmp2 == 7i32 || tmp2 == 8i32 {
                    // `LIGH.cpp:165-173`: the deferred-angle states count down.
                    // `tmp2 >= 9` is a console-set value the reference resets
                    // to 7 before the decrement.
                    if tmp2 >= 9i32 {
                        tmp2 = 7i32;
                    } else {
                        tmp2 -= 1i32;
                    }
                    part_tmp2[id] = tmp2;
                } else if tmp2 == 5i32 || part_life[id] <= 1i32 {
                    dead = true;
                } else {
                    // `LIGH.cpp:179-190`: the head casts one (or, when
                    // branching, two) Bresenham line(s) of `LIGH` segments.
                    let px = (idx as u32 % xres) as i32;
                    let py = (idx as u32 / xres) as i32;
                    let angle = (part_tmp[id] + rand_range(seed, -30i32, 30i32)) % 360i32;
                    let multipler = (part_life[id] as f32 * 1.5f32) as i32
                        + rand_range(seed ^ 0x1u32, 0i32, part_life[id]);
                    let rad = angle as f32 * (core::f32::consts::PI / 180.0f32);
                    let rx = (rad.cos() * multipler as f32) as i32;
                    let ry = (-rad.sin() * multipler as f32) as i32;
                    let parent_tmp2 = tmp2;
                    let temp = part_temp[id];
                    let life = part_life[id];
                    let dcolour = part_dcolour[id];
                    queue_ligh_line(
                        pmap,
                        part_ty,
                        part_life,
                        part_ctype,
                        part_tmp,
                        create_queue,
                        xres,
                        yres,
                        px,
                        py,
                        px + rx,
                        py + ry,
                        temp,
                        life,
                        angle,
                        parent_tmp2,
                        dcolour,
                        seed,
                    );
                    if parent_tmp2 == 2i32 {
                        // `LIGH.cpp:184-190`: a branching head casts a second
                        // line in a wide random direction from the same start.
                        let angle2 = (angle + rand_range(seed ^ 0x2u32, -100i32, 100i32)) % 360i32;
                        let rad2 = angle2 as f32 * (core::f32::consts::PI / 180.0f32);
                        let rx2 = (rad2.cos() * multipler as f32) as i32;
                        let ry2 = (-rad2.sin() * multipler as f32) as i32;
                        queue_ligh_line(
                            pmap,
                            part_ty,
                            part_life,
                            part_ctype,
                            part_tmp,
                            create_queue,
                            xres,
                            yres,
                            px,
                            py,
                            px + rx2,
                            py + ry2,
                            temp,
                            life,
                            angle2,
                            parent_tmp2,
                            dcolour,
                            seed ^ 0x3u32,
                        );
                    }
                    part_tmp2[id] = 7i32;
                }
                if dead {
                    part_ty[id] = PT_NONE;
                }
            } else if ty == PT_C5 {
                // `C5.cpp:83-98`: a charged C5 emits its photon once.
                if part_ctype[id] != 0u32 && part_life[id] == 0i32 {
                    part_tmp[id] = PT_PHOT as i32;
                    part_flags[id] |= FLAG_SPAWN_TMP;
                    part_ctype[id] = 0u32;
                    part_tmp2[id] = 0i32;
                }
            }
        }
    }
}
