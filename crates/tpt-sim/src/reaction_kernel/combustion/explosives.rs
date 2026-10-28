//! Explosives and fuse kernels.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::*;

/// Element ids that a `BOMB` contact test ignores (`BOMB.cpp:76-80`).
#[cube]
fn bomb_trigger_immune(ty: u32) -> bool {
    ty == PT_BOMB
        || ty == PT_EMBR
        || ty == PT_DMND
        || ty == PT_CLNE
        || ty == PT_PCLN
        || ty == PT_BCLN
        || ty == PT_VIBR
}

/// Element ids a `BOMB` blast leaves alive (`BOMB.cpp:92-96`).
#[cube]
fn bomb_blast_immune(ty: u32) -> bool {
    ty == PT_DMND || ty == PT_CLNE || ty == PT_PCLN || ty == PT_BCLN || ty == PT_VIBR
}

/// Rounds of the connected-`BANG` flood per frame.
///
/// One round carries the arming mark a single 4-connected hop, so the constant
/// is the reach per frame; the mark is monotone and persists on armed pixels,
/// so a charge that is wider than the reach still converges after
/// `ceil(width / BANG_FLOOD_ROUNDS)` frames. 16 arms a 32-pixel-wide charge in
/// two frames and costs two `pipe::FLOOD_ROUNDS` cascades (8 rounds each), every
/// one of them an early-exit pass over the pixel grid.
pub(crate) const BANG_FLOOD_ROUNDS: u32 = 16;

/// Sentinel written by [`bang_flood_kernel`] into a run pixel of a charge that
/// the persistent arming wave reached this frame.
///
/// TPT's `flood_prop` writes the final value `2` into the whole run in one
/// call, but this port runs the flood *before* [`explosive_reaction_kernel`]: a
/// pixel that received `2` here would advance to `3` in the same frame and
/// detonate one frame ahead of the seed that armed it. Storing the sentinel and
/// letting the explosive pass turn it into `2` keeps every pixel of the charge
/// on the same frame of the `2 -> 3 -> boom` countdown.
pub(crate) const BANG_ARMED_NEXT_FRAME: i32 = -2;

/// Sentinel written into a run pixel that a fresh `tmp == 1` seed reached.
///
/// `Simulation::flood_prop` (`Simulation.cpp:520-570`) calls
/// `changeProperty.Set` on every pixel of the connected run without looking at
/// the current value, so a charge that already counted down to `tmp == 3` is
/// reset to `2` when a new seed ignites inside it. `-1` carries that "re-armed
/// this frame" mark through the run; the explosive pass restarts the countdown
/// from `2` so the reset pixel cannot jump straight back to `3`.
pub(crate) const BANG_REARMED: i32 = -1;

/// `tmp` of the `BANG` pixel at `(x, y)`, or `0` when the pixel holds anything
/// else.
#[cube]
fn bang_tmp_at(
    pmap: &[u32],
    part_ty: &[u32],
    part_tmp: &mut [i32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> i32 {
    let mut tmp = 0i32;
    let nv = pmap_at(pmap, xres, yres, x, y);
    if nv != 0u32 {
        let ns = pmap_slot(nv);
        if part_ty[ns] == PT_BANG {
            tmp = part_tmp[ns];
        }
    }
    tmp
}

/// One round of the connected-`BANG` arming flood (`BANG.cpp:74-80`).
///
/// TPT does not advance `tmp` one pixel at a time: the first `BANG` that sees
/// fire (`tmp == 1`) calls `flood_prop(x, y, AccessProperty{ FIELD_TMP, 2 })`,
/// which writes `2` into the `tmp` of **every** pixel of the run — the
/// reference expands the horizontal run and then pushes the rows above and
/// below, i.e. a 4-connected flood (`Simulation.cpp:520-570`). The whole charge
/// therefore detonates as one mass.
///
/// A dispatch can only carry the mark one hop, so this kernel runs
/// [`BANG_FLOOD_ROUNDS`] times per frame:
///
/// * an unarmed pixel (`tmp == 0`) takes [`BANG_ARMED_NEXT_FRAME`] from any
///   4-connected `BANG` neighbour that carries a mark;
/// * a pixel already in the countdown (`2` or `3`) takes [`BANG_REARMED`] from a
///   neighbour that is a fresh seed (`1`) or already re-armed, mirroring the
///   unconditional `changeProperty.Set` of the reference flood.
///
/// Both sentinels are outside the countdown, so the explosive pass can turn
/// them into `2` before it advances the pixels that were already armed. Marked
/// pixels are read-only for the remaining rounds; the neighbours still at zero
/// pick the wave up on the next dispatch. Every thread writes only its own
/// slot.
#[cube(launch_unchecked)]
pub(crate) fn bang_flood_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_tmp: &mut [i32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 {
        terminate!();
    }
    let id = pmap_slot(v);
    if part_ty[id] != PT_BANG {
        terminate!();
    }
    let tmp = part_tmp[id];
    if tmp == BANG_ARMED_NEXT_FRAME || tmp == BANG_REARMED {
        terminate!();
    }
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;
    let mut reached = false;
    let mut rearm = false;
    for n in 0u32..4u32 {
        let mut dx = 0i32;
        let mut dy = 0i32;
        if n == 0u32 {
            dx = -1i32;
        } else if n == 1u32 {
            dx = 1i32;
        } else if n == 2u32 {
            dy = -1i32;
        } else {
            dy = 1i32;
        }
        let nt = bang_tmp_at(pmap, part_ty, part_tmp, xres, yres, px + dx, py + dy);
        if nt != 0i32 {
            reached = true;
        }
        if nt == 1i32 || nt == BANG_REARMED {
            rearm = true;
        }
    }
    if tmp == 0i32 {
        if reached {
            part_tmp[id] = BANG_ARMED_NEXT_FRAME;
        }
    } else if (tmp == 2i32 || tmp == 3i32) && rearm {
        part_tmp[id] = BANG_REARMED;
    }
}

/// `BOMB`, `DEST`, `BANG` and `EMBR`.
///
/// Pixel parallel. `air_pv` carries the pressure pulses; the spatial map is
/// only read. `BOMB`'s two ember discs queue their creations, so the pass
/// carries the creation queue; the cords that also create (`FUSE`, `FSEP`,
/// `IGNT`) live in [`fuse_kernel`], which trades `elem_words` for
/// `part_ctype`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn explosive_reaction_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp2: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_pv: &mut [f32],
    elem_words: &[u32],
    create_queue: &mut [Atomic<u32>],
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
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0x9E37_79B9u32));

                if ty == PT_BOMB {
                    // `BOMB.cpp:62-80`: anything that is not another bomb, an
                    // ember, diamond or a clone family member sets it off.
                    let mut trigger = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nt = part_ty[pmap_slot(nv)];
                                if nt != PT_NONE && !bomb_trigger_immune(nt) {
                                    trigger = true;
                                }
                            }
                        }
                    }
                    if trigger {
                        // `BOMB.cpp:82-139`: the inner radius-8 disc leaves a
                        // lit ember (`tmp = 2`, `life = 2`) on every pixel that
                        // is not blast-immune and adds 0.1 pressure there; the
                        // outer radius-9 ring fills only empty pixels with a
                        // long-lived ember (`tmp = 0`, `life = 50`) kicked at
                        // `rng.between(-20, 20)` on both axes. TPT kills the
                        // bomb with `kill_part` and lets the centre ember come
                        // from the inner disc; the queue's
                        // `CR_NO_OCCUPANCY` reproduces the `create_part(-3)`
                        // map-check bypass.
                        for oy in 0u32..17u32 {
                            let dy = oy as i32 - 8i32;
                            for ox in 0u32..17u32 {
                                let dx = ox as i32 - 8i32;
                                if dx * dx + dy * dy <= 64i32 {
                                    let nx = px + dx;
                                    let ny = py + dy;
                                    if nx >= 0i32
                                        && ny >= 0i32
                                        && nx < xres as i32
                                        && ny < yres as i32
                                    {
                                        let mut skip = false;
                                        let nv = pmap_at(pmap, xres, yres, nx, ny);
                                        if nv != 0u32 {
                                            let nid = pmap_slot(nv);
                                            let nt = part_ty[nid];
                                            if nt != PT_NONE {
                                                if bomb_blast_immune(nt) {
                                                    skip = true;
                                                } else {
                                                    part_ty[nid] = PT_NONE;
                                                }
                                            }
                                        }
                                        if !skip {
                                            let ncidx = (((ny as u32) / CELL) * xcells
                                                + ((nx as u32) / CELL))
                                                as usize;
                                            air_pv[ncidx] += 0.1f32;
                                            request_create(
                                                create_queue,
                                                PT_EMBR,
                                                nx,
                                                ny,
                                                CR_NO_OCCUPANCY
                                                    | CR_SET_LIFE
                                                    | CR_SET_TEMP
                                                    | CR_SET_TMP,
                                                2i32,
                                                0u32,
                                                MAX_TEMP,
                                                2i32,
                                                0i32,
                                                0i32,
                                                0i32,
                                                0u32,
                                                0u32,
                                                0.0f32,
                                                0.0f32,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        for oy in 0u32..19u32 {
                            let dy = oy as i32 - 9i32;
                            for ox in 0u32..19u32 {
                                let dx = ox as i32 - 9i32;
                                if dx * dx + dy * dy <= 81i32 {
                                    let nx = px + dx;
                                    let ny = py + dy;
                                    if nx >= 0i32
                                        && ny >= 0i32
                                        && nx < xres as i32
                                        && ny < yres as i32
                                        && pmap_at(pmap, xres, yres, nx, ny) == 0u32
                                    {
                                        let rr = hash_u32(
                                            seed ^ mix_index(oy * 19u32 + ox, 0x85EB_CA6Bu32),
                                        );
                                        request_create(
                                            create_queue,
                                            PT_EMBR,
                                            nx,
                                            ny,
                                            CR_NO_OCCUPANCY
                                                | CR_SET_LIFE
                                                | CR_SET_TEMP
                                                | CR_SET_TMP
                                                | CR_SET_VX
                                                | CR_SET_VY,
                                            50i32,
                                            0u32,
                                            MAX_TEMP,
                                            0i32,
                                            0i32,
                                            0i32,
                                            0i32,
                                            0u32,
                                            0u32,
                                            rand_range(rr, -20i32, 20i32) as f32,
                                            rand_range(rr >> 8u32, -20i32, 20i32) as f32,
                                        );
                                    }
                                }
                            }
                        }
                    }
                } else if ty == PT_DEST {
                    // `DEST.cpp:58-105`: one random neighbour per frame.
                    let rx = rand_range(seed ^ 0x1234_5678u32, -2i32, 2i32);
                    let ry = rand_range(seed ^ 0x9ABC_DEF0u32, -2i32, 2i32);
                    let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                    if nv != 0u32 {
                        let nid = pmap_slot(nv);
                        let rt = part_ty[nid];
                        let immune = rt == PT_NONE
                            || rt == PT_DEST
                            || rt == PT_DMND
                            || rt == PT_BCLN
                            || rt == PT_CLNE
                            || rt == PT_PCLN
                            || rt == PT_PBCN;
                        if !immune {
                            let mut life = part_life[id];
                            if life <= 0i32 || life > 37i32 {
                                life = rand_range(seed ^ 0x5A5A_5A5Au32, 30i32, 49i32);
                                air_pv[cidx] += 60.0f32;
                            }
                            if rt == PT_PLUT || rt == PT_DEUT {
                                air_pv[cidx] += 20.0f32;
                                if chance(seed ^ 0x11u32, 1u32, 2u32) {
                                    part_ty[nid] = PT_NEUT;
                                    part_temp[nid] = MAX_TEMP;
                                    air_pv[cidx] += 10.0f32;
                                    life -= 4i32;
                                }
                            } else if rt == PT_INSL {
                                part_ty[nid] = PT_PLSM;
                            } else if chance(seed ^ 0x22u32, 1u32, 3u32) {
                                part_ty[nid] = PT_NONE;
                                let solid = (elem_word(elem_words, rt, WORD_PROPERTIES)
                                    & TYPE_SOLID)
                                    != 0u32;
                                if solid {
                                    life -= 12i32;
                                } else {
                                    life -= 4i32;
                                }
                                if life <= 0i32 {
                                    life = 1i32;
                                }
                            } else if !heat_insulator(elem_words, rt) {
                                part_temp[nid] = MAX_TEMP;
                            }
                            part_life[id] = life;
                            part_temp[id] = MAX_TEMP;
                            air_pv[cidx] =
                                clampf(air_pv[cidx] + 80.0f32, MIN_PRESSURE, MAX_PRESSURE);
                        }
                    }
                } else if ty == PT_BANG {
                    // `BANG.cpp:50-122`. The `tmp == 1` mass flood is
                    // [`bang_flood_kernel`], which runs [`BANG_FLOOD_ROUNDS`]
                    // rounds before this pass and marks every 4-connected
                    // `BANG` of the run, so a charge arms as one mass instead
                    // of one 3x3 layer per frame.
                    let mut tmp = part_tmp[id];
                    if tmp == BANG_ARMED_NEXT_FRAME || tmp == BANG_REARMED {
                        // The flood reached this pixel earlier this frame.
                        // TPT's `changeProperty.Set` writes the run's `2` in
                        // the same call, so the pixel joins the countdown here
                        // and only advances to `3` on a later frame — the
                        // sentinel exists exactly to keep it level with the
                        // seed that armed it.
                        tmp = 2i32;
                    } else if tmp == 0i32 {
                        let mut ignite = part_temp[id] >= TNT_IGNITE_TEMP;
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if dx != 0i32 || dy != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nty = part_ty[pmap_slot(nv)];
                                    if nty == PT_FIRE
                                        || nty == PT_PLSM
                                        || nty == PT_SPRK
                                        || nty == PT_LIGH
                                    {
                                        ignite = true;
                                    }
                                }
                            }
                        }
                        if ignite {
                            tmp = 1i32;
                        }
                    } else if tmp == 1i32 {
                        // `BANG.cpp:74-80`: the flood already armed the run;
                        // the seed joins the countdown here.
                        tmp = 2i32;
                    } else if tmp == 2i32 {
                        tmp = 3i32;
                    } else {
                        air_pv[cidx] += 0.5f32;
                        tmp = 0i32;
                        let otemp = part_temp[id] - 273.15f32;
                        // Every outcome below replaces the slot through
                        // `create_part`, which resets `ctype`, `tmp2` and the
                        // velocity from `DefaultProperties` before the
                        // element's `Create` hook runs.
                        part_ctype[id] = 0u32;
                        part_tmp2[id] = 0i32;
                        part_vx[id] = 0.0f32;
                        part_vy[id] = 0.0f32;
                        if chance(seed ^ 0x33u32, 1u32, 3u32) {
                            let fire_temp = clampf(DEST_FIRE_TEMP + otemp, MIN_TEMP, MAX_TEMP);
                            if chance(seed ^ 0x44u32, 1u32, 2u32) {
                                // `BANG.cpp:95-98`: `create_part(i, x, y,
                                // PT_FIRE)` runs `FIRE`'s `Create` hook, so the
                                // new fire gets its own 120..169 life; the
                                // explosion then overrides the temperature.
                                part_ty[id] = PT_FIRE;
                                part_life[id] =
                                    120i32 + (hash_u32(seed ^ 0x7A7A_7A7Au32) % 50u32) as i32;
                                part_temp[id] = fire_temp;
                            } else {
                                part_ty[id] = PT_SMKE;
                                part_life[id] = 500i32 + rand_range(seed ^ 0x55u32, 0i32, 49i32);
                                part_temp[id] = fire_temp;
                            }
                        } else if chance(seed ^ 0x66u32, 1u32, 15u32) {
                            // `BANG.cpp:110-118`: the ember leaves with a
                            // `between(-10, 10)` kick on both axes; the port's
                            // `part_vx`/`part_vy` are already in TPT pixel
                            // units, so the roll is copied verbatim.
                            part_ty[id] = PT_EMBR;
                            part_life[id] = 50i32;
                            part_temp[id] = clampf(DEST_EMBR_TEMP + otemp, MIN_TEMP, MAX_TEMP);
                            part_vx[id] = rng_kick(seed ^ 0x1B87_3593u32);
                            part_vy[id] = rng_kick(seed ^ 0xC2B2_AE35u32);
                        } else {
                            part_ty[id] = PT_NONE;
                        }
                    }
                    part_tmp[id] = tmp;
                } else if ty == PT_EMBR {
                    // `EMBR.cpp:56-70`: an ember dies on any matter that does
                    // not carry `PROP_SPARKSETTLE`.
                    let mut hit = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nty = part_ty[pmap_slot(nv)];
                                if nty != PT_NONE {
                                    let props = elem_word(elem_words, nty, WORD_PROPERTIES);
                                    let matter =
                                        (props & (TYPE_SOLID | TYPE_PART | TYPE_LIQUID)) != 0u32;
                                    let settle = (props & PROP_SPARKSETTLE) != 0u32;
                                    if matter && !settle {
                                        hit = true;
                                    }
                                }
                            }
                        }
                    }
                    if hit {
                        part_ty[id] = PT_NONE;
                    }
                }
            }
        }
    }
}

/// `FUSE`, `FSEP` and `IGNT` (`FUSE.cpp:65-92`, `FSEP.cpp:58-84`,
/// `IGNT.cpp:58-91`).
///
/// Split from the explosive pass because the cord bookkeeping needs
/// `part_ctype` as a scratch word (TPT's `create_part` resets the particle and
/// the port cannot), which would be a ninth binding there. The plasma created
/// by a burnt fuse takes `MAX_TEMP`, the `PLSM` default (`PLSM.cpp:33`).
#[cube(launch_unchecked)]
pub(crate) fn fuse_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    air_pv: &mut [f32],
    create_queue: &mut [Atomic<u32>],
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
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0xB529_7A4Du32));

                if ty == PT_FUSE {
                    // `FUSE.cpp:65-92`.
                    let mut tmp = part_tmp[id];
                    let life = part_life[id];
                    if life <= 0i32 {
                        part_ty[id] = PT_PLSM;
                        part_life[id] = 50i32;
                        part_temp[id] = MAX_TEMP;
                    } else {
                        if life < 40i32 {
                            part_life[id] = life - 1i32;
                            if chance(seed, 1u32, 100u32) {
                                // `FUSE.cpp:66-71`: one `PLSM` in a random
                                // adjacent pixel with life 50; the queue gives it
                                // `PLSM`'s own default temperature (9999 K).
                                let ox = px + rng_offset(seed ^ 0x9E37_79B9u32);
                                let oy = py + rng_offset(seed ^ 0x85EB_CA6Bu32);
                                request_create(
                                    create_queue,
                                    PT_PLSM,
                                    ox,
                                    oy,
                                    CR_SET_LIFE,
                                    50i32,
                                    0u32,
                                    0.0f32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0u32,
                                    0u32,
                                    0.0f32,
                                    0.0f32,
                                );
                            }
                        }
                        if air_pv[cidx] > FUSE_QUENCH_PRESSURE && tmp > 40i32 {
                            tmp = 39i32;
                        } else if tmp <= 0i32 {
                            part_ty[id] = PT_FSEP;
                            part_tmp[id] = tmp;
                        } else if tmp < 40i32 {
                            tmp -= 1i32;
                        }
                        if part_ty[id] == PT_FUSE {
                            part_tmp[id] = tmp;
                            for n in 0u32..25u32 {
                                let dx = (n % 5u32) as i32 - 2i32;
                                let dy = (n / 5u32) as i32 - 2i32;
                                if dx != 0i32 || dy != 0i32 {
                                    let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                    if nv != 0u32 {
                                        let nty = part_ty[pmap_slot(nv)];
                                        let hot = part_temp[id] >= FUSE_IGNITE_TEMP
                                            && chance(seed ^ n, 1u32, 20u32);
                                        if (nty == PT_SPRK || hot) && part_life[id] > 40i32 {
                                            part_life[id] = 39i32;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if ty == PT_FSEP {
                    // `FSEP.cpp:58-84`.
                    let life = part_life[id];
                    if life <= 0i32 {
                        part_ty[id] = PT_PLSM;
                        part_life[id] = 50i32;
                        part_temp[id] = MAX_TEMP;
                    } else if life < 40i32 {
                        part_life[id] = life - 1i32;
                        if chance(seed, 1u32, 10u32) {
                            // `FSEP.cpp:59-66`: same random adjacent `PLSM`.
                            let ox = px + rng_offset(seed ^ 0x9E37_79B9u32);
                            let oy = py + rng_offset(seed ^ 0x85EB_CA6Bu32);
                            request_create(
                                create_queue,
                                PT_PLSM,
                                ox,
                                oy,
                                CR_SET_LIFE,
                                50i32,
                                0u32,
                                0.0f32,
                                0i32,
                                0i32,
                                0i32,
                                0i32,
                                0u32,
                                0u32,
                                0.0f32,
                                0.0f32,
                            );
                        }
                    } else {
                        for n in 0u32..25u32 {
                            let dx = (n % 5u32) as i32 - 2i32;
                            let dy = (n / 5u32) as i32 - 2i32;
                            if dx != 0i32 || dy != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nty = part_ty[pmap_slot(nv)];
                                    // `FSEP.cpp:79-82`: the ignition roll is
                                    // gated on `life > 40`, so a cord that just
                                    // reached 40 waits one more frame.
                                    if part_life[id] > 40i32
                                        && (nty == PT_SPRK || part_temp[id] >= FSEP_IGNITE_TEMP)
                                        && chance(seed ^ n, 1u32, 15u32)
                                    {
                                        part_life[id] = 39i32;
                                    }
                                }
                            }
                        }
                    }
                } else if ty == PT_IGNT {
                    // `IGNT.cpp:58-91`.
                    let mut tmp = part_tmp[id];
                    if tmp == 0i32 {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if dx != 0i32 || dy != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    if nty == PT_FIRE
                                        || nty == PT_PLSM
                                        || nty == PT_SPRK
                                        || nty == PT_LIGH
                                        || (nty == PT_IGNT && part_life[nid] == 1i32)
                                    {
                                        tmp = 1i32;
                                    }
                                }
                            }
                        }
                        part_tmp[id] = tmp;
                    } else if part_life[id] > 0i32 {
                        // `IGNT.cpp:74-90`: two thirds of the time a hot `EMBR`,
                        // otherwise a `FIRE`, both in a random adjacent pixel. The
                        // ember keeps the igniter's heat offset by 126.85 K and a
                        // random `+-10` velocity kick.
                        let ox = px + rng_offset(seed ^ 0x9E37_79B9u32);
                        let oy = py + rng_offset(seed ^ 0x85EB_CA6Bu32);
                        if chance(seed, 2u32, 3u32) {
                            let ember_temp =
                                clampf(part_temp[id] - 273.15f32 + 400.0f32, MIN_TEMP, MAX_TEMP);
                            request_create(
                                create_queue,
                                PT_EMBR,
                                ox,
                                oy,
                                CR_SET_LIFE | CR_SET_TEMP | CR_SET_VX | CR_SET_VY,
                                30i32,
                                0u32,
                                ember_temp,
                                0i32,
                                0i32,
                                0i32,
                                0i32,
                                0u32,
                                0u32,
                                rng_kick(seed ^ 0xC2B2_AE35u32),
                                rng_kick(seed ^ 0x7FEB_352Du32),
                            );
                        } else {
                            request_create_plain(create_queue, PT_FIRE, ox, oy);
                        }
                        part_life[id] -= 1i32;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{hash_u32_host, rng_kick_host};
    use super::*;

    /// Host mirror of one frame of the `BANG` pipeline: the flood rounds run
    /// before the explosive pass, exactly like `launch_combustion_pass`.
    ///
    /// `ignite` is the pixel that sees fire and enters `tmp == 1`; every other
    /// pixel waits for the flood.
    fn tnt_frame(tmp: &mut [i32], types: &mut [u32], width: i32, ignite: Option<usize>) {
        let height = tmp.len() as i32 / width;
        for _ in 0..BANG_FLOOD_ROUNDS {
            let prev = tmp.to_vec();
            for i in 0..tmp.len() {
                if types[i] != PT_BANG {
                    continue;
                }
                let t = prev[i];
                if t == BANG_ARMED_NEXT_FRAME || t == BANG_REARMED {
                    continue;
                }
                let x = i as i32 % width;
                let y = i as i32 / width;
                let mut reached = false;
                let mut rearm = false;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= width || ny >= height {
                        continue;
                    }
                    let j = (ny * width + nx) as usize;
                    if types[j] != PT_BANG {
                        continue;
                    }
                    if prev[j] != 0 {
                        reached = true;
                    }
                    if prev[j] == 1 || prev[j] == BANG_REARMED {
                        rearm = true;
                    }
                }
                if t == 0 {
                    if reached {
                        tmp[i] = BANG_ARMED_NEXT_FRAME;
                    }
                } else if (t == 2 || t == 3) && rearm {
                    tmp[i] = BANG_REARMED;
                }
            }
        }
        for i in 0..tmp.len() {
            if types[i] != PT_BANG {
                continue;
            }
            let t = tmp[i];
            if t == BANG_ARMED_NEXT_FRAME || t == BANG_REARMED {
                tmp[i] = 2;
            } else if t == 0 {
                if ignite == Some(i) {
                    tmp[i] = 1;
                }
            } else if t == 1 {
                tmp[i] = 2;
            } else if t == 2 {
                tmp[i] = 3;
            } else {
                // `BANG.cpp:87-122`: the slot is replaced by fire, smoke, an
                // ember or nothing at all.
                types[i] = PT_FIRE;
                tmp[i] = 0;
            }
        }
    }

    #[test]
    fn tnt_run_detonates_as_one_mass() {
        let width = 12i32;
        let mut tip = vec![0i32; width as usize];
        let mut types = vec![PT_BANG; width as usize];

        // Frame 0: only the seed that sees the fire moves `0 -> 1`.
        tnt_frame(&mut tip, &mut types, width, Some(0));
        assert_eq!(tip[0], 1);
        assert!(tip[1..].iter().all(|&t| t == 0));

        // Frame 1: the flood marks the run and the explosive pass turns the
        // marks into `2` while the seed runs `1 -> 2`, so every pixel leaves
        // the frame at `2`.
        tnt_frame(&mut tip, &mut types, width, None);
        assert_eq!(
            tip,
            vec![2i32; width as usize],
            "the flood must not advance the front past the seed"
        );

        // Frame 2: the whole charge counts down together.
        tnt_frame(&mut tip, &mut types, width, None);
        assert_eq!(tip, vec![3i32; width as usize]);

        // Frame 3: the front and the seed detonate in the same frame.
        tnt_frame(&mut tip, &mut types, width, None);
        assert!(
            types.iter().all(|&ty| ty == PT_FIRE),
            "the flood front must not outrun the seed: {types:?}"
        );
    }

    #[test]
    fn a_fresh_seed_rearms_a_countdown() {
        // `Simulation.cpp:520-570`: `flood_prop` writes the run's `2` with
        // `changeProperty.Set`, so an already armed `tmp == 3` pixel is reset
        // when a new seed ignites inside the charge.
        let width = 4i32;
        let mut tip = vec![1i32, 3, 3, 3];
        let mut types = vec![PT_BANG; 4];

        tnt_frame(&mut tip, &mut types, width, None);
        assert_eq!(
            tip,
            vec![2i32; 4],
            "the re-arm wave must reset the countdown"
        );
        assert!(types.iter().all(|&ty| ty == PT_BANG));

        tnt_frame(&mut tip, &mut types, width, None);
        assert_eq!(tip, vec![3i32; 4]);

        tnt_frame(&mut tip, &mut types, width, None);
        assert!(types.iter().all(|&ty| ty == PT_FIRE));
    }

    #[test]
    fn tnt_explosion_uses_the_create_hook_life_and_ember_kick() {
        // `BANG.cpp:95-118`: `create_part(i, x, y, PT_FIRE)` runs
        // `FIRE.cpp:423-426` (life 120..169) and the ember leaves with
        // `rng.between(-10, 10)` on both axes.
        let mut min_life = i32::MAX;
        let mut max_life = i32::MIN;
        let mut kicks = 0u32;
        for seed in 0u32..4096u32 {
            let life = 120i32 + (hash_u32_host(seed ^ 0x7A7A_7A7Au32) % 50u32) as i32;
            assert!(
                (120..=169).contains(&life),
                "fire life {life} outside the Create hook range"
            );
            min_life = min_life.min(life);
            max_life = max_life.max(life);
            let vx = rng_kick_host(seed ^ 0x1B87_3593u32);
            let vy = rng_kick_host(seed ^ 0xC2B2_AE35u32);
            assert!((-10.0..=10.0).contains(&vx), "ember vx {vx}");
            assert!((-10.0..=10.0).contains(&vy), "ember vy {vy}");
            if vx != 0.0 && vy != 0.0 {
                kicks += 1;
            }
        }
        assert_eq!((min_life, max_life), (120, 169));
        assert!(kicks > 0, "the kick must not collapse to zero");
        // The `FIRE.cpp:315` ignition range (180..259) belongs to the generic
        // ignition gate, never to a `create_part` product.
        assert!(max_life < 180);
    }
}
