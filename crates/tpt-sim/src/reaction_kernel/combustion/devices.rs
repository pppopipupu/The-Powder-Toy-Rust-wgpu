//! TRON movement and EMP pulse devices kernels.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::*;

/// `x` step of a `TRON` direction (`TRON.cpp:130-131`).
#[cube]
fn tron_dir_x(dir: i32) -> i32 {
    let mut v = 0i32;
    if dir == 0i32 {
        v = -1i32;
    } else if dir == 2i32 {
        v = 1i32;
    }
    v
}

/// `y` step of a `TRON` direction (`TRON.cpp:130-131`).
#[cube]
fn tron_dir_y(dir: i32) -> i32 {
    let mut v = 0i32;
    if dir == 1i32 {
        v = -1i32;
    } else if dir == 3i32 {
        v = 1i32;
    }
    v
}

/// `TRON.cpp:255-275` `canmovetron`: empty pixels, live `SWCH`, visible
/// `INVIS` and short-lived life-kill elements are passable.
#[cube]
fn tron_can_move(
    pmap: &[u32],
    wall: &[u32],
    ty: &mut [u32],
    life: &mut [i32],
    tmp2: &mut [i32],
    words: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    x: i32,
    y: i32,
    len: i32,
) -> bool {
    let mut ok = false;
    if x >= CELL as i32
        && y >= CELL as i32
        && x < xres as i32 - CELL as i32
        && y < yres as i32 - CELL as i32
    {
        let v = pmap_at(pmap, xres, yres, x, y);
        if v == 0u32 {
            ok = true;
        } else {
            let r = pmap_slot(v);
            let rt = ty[r];
            if (rt == PT_SWCH && life[r] >= 10i32) || (rt == PT_INVIS && tmp2[r] == 1i32) {
                ok = true;
            } else {
                let props = elem_word(words, rt, WORD_PROPERTIES);
                let kill = (props & PROP_LIFE_KILL) != 0u32;
                let kill_dec = (props & PROP_LIFE_KILL_DEC) != 0u32;
                let dec = (props & PROP_LIFE_DEC) != 0u32;
                if ((kill_dec && life[r] > 0i32) || (kill && dec)) && life[r] < len {
                    ok = true;
                }
            }
        }
        if ok {
            let ci = (((y as u32) / CELL) * xcells + ((x as u32) / CELL)) as usize;
            if ci < (xcells * ycells) as usize && wall[ci] != 0u32 {
                ok = false;
            }
        }
    }
    ok
}

/// `TRON.cpp:230-260` `trymovetron`: how many steps the head can see ahead.
///
/// The straight lane is counted first; every clear step then probes the two
/// perpendicular lanes (`tron_ry[dir], tron_rx[dir]`) up to `len - k - 1`
/// cells out and adds each clear one to the score, which is what makes an
/// open corridor beat a blind alley of the same depth. TPT's `return len + 1`
/// ("a safe path exists") is unreachable for axis-aligned directions: the lane
/// bound is `j < len - k` and the test is `j == len - k`, so only the counting
/// side of the probe has any effect.
#[cube]
fn tron_look(
    pmap: &[u32],
    wall: &[u32],
    ty: &mut [u32],
    life: &mut [i32],
    tmp2: &mut [i32],
    words: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    x: i32,
    y: i32,
    dir: i32,
    len: i32,
) -> i32 {
    let mut count = 0i32;
    let mut blocked = false;
    let dx = tron_dir_x(dir);
    let dy = tron_dir_y(dir);
    // Perpendicular axis: TPT walks `(tron_ry[dir], tron_rx[dir])`.
    let sx = dy;
    let sy = dx;
    for k in 0i32..len {
        if !blocked {
            let cx = x + dx * (k + 1i32);
            let cy = y + dy * (k + 1i32);
            if tron_can_move(
                pmap, wall, ty, life, tmp2, words, xres, yres, xcells, ycells, cx, cy, k,
            ) {
                count += 1i32;
                let span = len - k;
                for side in 0i32..2i32 {
                    let sign = side * 2i32 - 1i32;
                    let mut open = true;
                    for j in 1i32..span {
                        if open {
                            let tx = cx + sx * sign * j;
                            let tyy = cy + sy * sign * j;
                            if tron_can_move(
                                pmap,
                                wall,
                                ty,
                                life,
                                tmp2,
                                words,
                                xres,
                                yres,
                                xcells,
                                ycells,
                                tx,
                                tyy,
                                j + k - 1i32,
                            ) {
                                count += 1i32;
                            } else {
                                open = false;
                            }
                        }
                    }
                }
            } else {
                blocked = true;
            }
        }
    }
    count
}

/// `TRON` head movement, tail growth and tail death (`TRON.cpp:118-200`).
///
/// The new head is a spawn request ([`FLAG_SPAWN_SELF`]) expanded on the
/// device, so the child lands on the free pixel chosen by `spawn::free_neighbor`
/// rather than on the pixel the direction logic picked. To keep the chain coherent the head leaves its
/// direction in `tmp` when it becomes a tail, and a freshly created `TRON`
/// inherits direction, hue and tail length from the adjacent tail with the
/// largest `life`. The `TRON_DEATH` flare is a render rule
/// (`render_kernel/particles/pixel_mode.rs`), so the crash only has to latch
/// the flag here.
#[cube(launch_unchecked)]
pub(crate) fn tron_kernel(
    pmap: &[u32],
    wall_bmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_flags: &mut [u32],
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
            if part_ty[id] == PT_TRON {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0x3C6E_F372u32));
                let mut tmp = part_tmp[id];
                let mut tmp2 = part_tmp2[id];
                let mut life = part_life[id];

                if tmp == 0i32 && life == 0i32 {
                    // `TRON.cpp:315-325` `Create`: become a head. TPT picks a
                    // random direction and hue; a spawned child inherits them
                    // from the tail it grew out of.
                    // Seeded from an expand value so the locals are device
                    // registers; a literal initialiser would make them plain
                    // host `i32`s that cannot take a device value.
                    let mut best = tmp2 - 1i32;
                    let mut bits = tmp;
                    let mut inherited = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                if part_ty[nid] == PT_TRON
                                    && (part_tmp[nid] & TRON_HEAD) == 0
                                    && part_life[nid] > 0i32
                                    && part_life[nid] > best
                                {
                                    best = part_life[nid];
                                    bits = part_tmp[nid];
                                    tmp2 = part_tmp2[nid];
                                    inherited = true;
                                }
                            }
                        }
                    }
                    if inherited {
                        tmp = TRON_HEAD | (bits & TRON_KEEP_MASK);
                    } else {
                        tmp = TRON_HEAD
                            | (rand_range(seed, 0i32, 3i32) << 5i32)
                            | (rand_range(seed ^ 0x1u32, 0i32, 31i32) << 11i32);
                        tmp2 = TRON_TAIL_LEN;
                    }
                    life = TRON_CREATE_LIFE;
                }

                if (tmp & TRON_WAIT) != 0i32 {
                    tmp ^= TRON_WAIT;
                } else if (tmp & TRON_HEAD) != 0i32 {
                    let mut dir = (tmp >> 5i32) & 3i32;
                    let orig = dir;
                    let roll = rand_range(seed ^ 0x2u32, 0i32, 339i32);
                    if (roll == 1i32 || roll == 3i32) && (tmp & TRON_NORANDOM) == 0i32 {
                        dir = (dir + roll) % 4i32;
                    }
                    let len = clampi(tmp2, 1i32, 32i32);
                    let first = tron_look(
                        pmap, wall_bmap, part_ty, part_life, part_tmp2, elem_words, xres, yres,
                        xcells, ycells, px, py, dir, len,
                    );
                    let mut second = dir;
                    let mut last = dir;
                    if first < len {
                        if (tmp & TRON_NORANDOM) != 0i32 {
                            second = (dir + 1i32) % 4i32;
                            last = (dir + 3i32) % 4i32;
                        } else if orig != dir {
                            second = orig;
                            last = (dir + 2i32) % 4i32;
                        } else {
                            second =
                                (dir + rand_range(seed ^ 0x3u32, 0i32, 1i32) * 2i32 + 1i32) % 4i32;
                            last = (second + 2i32) % 4i32;
                        }
                    }
                    let sc = tron_look(
                        pmap, wall_bmap, part_ty, part_life, part_tmp2, elem_words, xres, yres,
                        xcells, ycells, px, py, second, len,
                    );
                    let lc = tron_look(
                        pmap, wall_bmap, part_ty, part_life, part_tmp2, elem_words, xres, yres,
                        xcells, ycells, px, py, last, len,
                    );
                    if sc > first {
                        dir = second;
                    }
                    if lc > sc && lc > first {
                        dir = last;
                    }
                    // `TRON.cpp:150-165`: the old head becomes the first tail
                    // segment and asks for a new head. A head boxed in on all
                    // three candidate lanes cannot move, so it crashes
                    // (`TRON.cpp:160-165` sets `TRON_DEATH`).
                    if first > 0i32 || sc > 0i32 || lc > 0i32 {
                        part_flags[id] |= FLAG_SPAWN_SELF;
                    } else {
                        tmp |= TRON_DEATH;
                    }
                    if life >= TRON_GROW_LIFE && (tmp & TRON_NOGROW) == 0i32 {
                        tmp2 += 1i32;
                    }
                    life = tmp2;
                    tmp &= TRON_KEEP_MASK;
                    tmp |= dir << 5i32;
                } else if (tmp & TRON_NODIE) != 0i32 {
                    life += 1i32;
                }

                part_tmp[id] = tmp;
                part_tmp2[id] = tmp2;
                part_life[id] = life;
            }
        }
    }
}

/// `EMP` pulse (`EMP.cpp:99-260`) with `triggerCount == 1`.
///
/// TPT's EMP is a tool that runs one global pulse per click over every sparked
/// electronic particle; there is no `EMP` `Update`. This port latches a pulse
/// on a particle's `tmp`: a freshly placed `EMP` (`tmp == 0`) fires once, then
/// stays inert. The binomial trigger distributions of `Probability.h` collapse
/// to single Bernoulli draws at `triggerCount == 1`, so
/// `binomial_gte1(1, p) == p` and the capped `DeltaTempGenerator` is a
/// `+3000 K` / `+2000 K` step at the same odds. `prob_breakMETLMore` is zero for
/// one trigger.
#[cube(launch_unchecked)]
pub(crate) fn emp_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_x: &[f32],
    part_y: &[f32],
    xres: u32,
    yres: u32,
    n_slots: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize && part_ty[i] == PT_EMP && part_tmp[i] == 0i32 {
        part_tmp[i] = 1i32;
        // Every fresh `EMP` latches, but only the lowest slot of a painted
        // blob runs the scan: the pulse is global, so one thread per blob is
        // enough and keeps a large brush from stalling the frame.
        let px = part_x[i] as i32;
        let py = part_y[i] as i32;
        let mut lowest = true;
        for n in 0u32..9u32 {
            let dx = (n % 3u32) as i32 - 1i32;
            let dy = (n / 3u32) as i32 - 1i32;
            if (dx != 0i32 || dy != 0i32) && lowest {
                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                if nv != 0u32 {
                    let nid = pmap_slot(nv);
                    if part_ty[nid] == PT_EMP && nid < i {
                        lowest = false;
                    }
                }
            }
        }
        if lowest {
            let seed = hash_u32(frame_seed ^ mix_index(i as u32, 0x5BD1_E995u32));
            let temp_max = MAX_TEMP as i32 - 1i32;
            for r in 0u32..n_slots {
                let rt = part_ty[r as usize];
                if rt != PT_NONE {
                    let rx = part_x[r as usize] as i32;
                    let ry = part_y[r as usize] as i32;
                    let spark = rt == PT_SPRK
                        || (rt == PT_SWCH
                            && part_life[r as usize] != 0i32
                            && part_life[r as usize] != 10i32)
                        || (rt == PT_WIRE && part_ctype[r as usize] > 0u32);
                    if spark {
                        let ct = part_ctype[r as usize];
                        let is_elec = ct == PT_PSCN
                            || ct == PT_NSCN
                            || ct == PT_PTCT
                            || ct == PT_NTCT
                            || ct == PT_INST
                            || ct == PT_SWCH
                            || rt == PT_WIRE
                            || rt == PT_SWCH;
                        let s0 = hash_u32(seed ^ r);
                        if is_elec {
                            // `temp_center`: +3000 K at 1/100, then the
                            // 1/48 break into `BREC`/`NTCT`.
                            if chance(s0, 1u32, 100u32) {
                                part_temp[r as usize] =
                                    clampf(part_temp[r as usize] + 3000.0f32, MIN_TEMP, MAX_TEMP);
                            }
                            if chance(hash_u32(s0 ^ 0x1u32), 1u32, 48u32) {
                                if chance(hash_u32(s0 ^ 0x2u32), 2u32, 5u32) {
                                    part_ty[r as usize] = PT_BREC;
                                } else {
                                    part_ty[r as usize] = PT_NTCT;
                                }
                            }
                        }
                        for oy in 0u32..5u32 {
                            for ox in 0u32..5u32 {
                                let dx = ox as i32 - 2i32;
                                let dy = oy as i32 - 2i32;
                                if dx != 0i32 || dy != 0i32 {
                                    let nv = pmap_at(pmap, xres, yres, rx + dx, ry + dy);
                                    if nv != 0u32 {
                                        let nid = pmap_slot(nv);
                                        let ntype = part_ty[nid];
                                        let nseed = mix_index(oy * 5u32 + ox, 0x27D4_EB2Fu32);
                                        let ns = hash_u32(s0 ^ nseed);
                                        if is_elec {
                                            // `METL` -> `BMTL` -> `BRMT`
                                            // (`EMP.cpp:150-180`). One trigger
                                            // means `prob_breakMETLMore` is 0.
                                            if ntype == PT_METL {
                                                if chance(ns, 1u32, 280u32) {
                                                    part_temp[nid] = clampf(
                                                        part_temp[nid] + 3000.0f32,
                                                        MIN_TEMP,
                                                        MAX_TEMP,
                                                    );
                                                }
                                                if chance(ns ^ 0x3u32, 1u32, 300u32) {
                                                    part_ty[nid] = PT_BMTL;
                                                }
                                            } else if ntype == PT_BMTL {
                                                if chance(ns, 1u32, 280u32) {
                                                    part_temp[nid] = clampf(
                                                        part_temp[nid] + 3000.0f32,
                                                        MIN_TEMP,
                                                        MAX_TEMP,
                                                    );
                                                }
                                                if chance(ns ^ 0x4u32, 1u32, 160u32) {
                                                    part_ty[nid] = PT_BRMT;
                                                    part_temp[nid] = clampf(
                                                        part_temp[nid] + 1000.0f32,
                                                        MIN_TEMP,
                                                        MAX_TEMP,
                                                    );
                                                }
                                            } else if ntype == PT_WIFI {
                                                if chance(ns, 1u32, 8u32) {
                                                    part_temp[nid] =
                                                        rand_range(ns ^ 0x5u32, 0i32, temp_max)
                                                            as f32;
                                                }
                                                if chance(ns ^ 0x6u32, 1u32, 16u32) {
                                                    part_ty[nid] = PT_BREC;
                                                    part_temp[nid] = clampf(
                                                        part_temp[nid] + 1000.0f32,
                                                        MIN_TEMP,
                                                        MAX_TEMP,
                                                    );
                                                }
                                            }
                                        }
                                        if ntype == PT_SWCH {
                                            if chance(ns ^ 0x7u32, 1u32, 100u32) {
                                                part_ty[nid] = PT_BREC;
                                            }
                                            if chance(ns ^ 0x8u32, 1u32, 100u32) {
                                                part_temp[nid] = clampf(
                                                    part_temp[nid] + 2000.0f32,
                                                    MIN_TEMP,
                                                    MAX_TEMP,
                                                );
                                            }
                                        } else if ntype == PT_ARAY {
                                            if chance(ns ^ 0x9u32, 1u32, 60u32) {
                                                part_ty[nid] = PT_BREC;
                                                part_temp[nid] = clampf(
                                                    part_temp[nid] + 1000.0f32,
                                                    MIN_TEMP,
                                                    MAX_TEMP,
                                                );
                                            }
                                        } else if ntype == PT_DLAY
                                            && chance(ns ^ 0xAu32, 1u32, 70u32)
                                        {
                                            part_temp[nid] = rand_range(ns ^ 0xBu32, 0i32, 255i32)
                                                as f32
                                                + 273.15f32;
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
}
