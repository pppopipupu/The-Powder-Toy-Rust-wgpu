use cubecl::prelude::*;

use super::fighter::figh_ai_select_command;
use super::is_obstacle;
use super::{
    C_FLAG_MOVABLE, C_PROP_DEADLY, C_PROP_RADIOACTIVE, C_PT_DUST, C_PT_FIGH, C_PT_GRVT, C_PT_NONE,
    C_PT_NUM, C_PT_PLNT, C_PT_PLUT, C_PT_SPAWN, C_PT_SPAWN2, C_PT_STKM, C_PT_STKM2, C_PT_STOR,
    C_STKM_PLAYER1, C_STKM_PLAYER2,
};
use crate::reaction_kernel::shared::{
    GPU_ALLOC_NONE, WORD_DEFAULT_LIFE, WORD_DEFAULT_TEMP, WORD_PROPERTIES, gpu_alloc_slot,
};

/// Initialise stickman leg positions and zero accelerations & metadata.
#[cube]
pub(super) fn init_legs(
    stkm_legs: &mut [f32],
    stkm_accs: &mut [f32],
    stkm_meta: &mut [u32],
    sid: usize,
    hx: f32,
    hy: f32,
    elem: u32,
) {
    let lb = sid * 16usize;
    stkm_legs[lb + 0usize] = hx - 1.0f32;
    stkm_legs[lb + 1usize] = hy + 6.0f32;
    stkm_legs[lb + 2usize] = hx - 1.0f32;
    stkm_legs[lb + 3usize] = hy + 6.0f32;

    stkm_legs[lb + 4usize] = hx - 3.0f32;
    stkm_legs[lb + 5usize] = hy + 12.0f32;
    stkm_legs[lb + 6usize] = hx - 3.0f32;
    stkm_legs[lb + 7usize] = hy + 12.0f32;

    stkm_legs[lb + 8usize] = hx + 1.0f32;
    stkm_legs[lb + 9usize] = hy + 6.0f32;
    stkm_legs[lb + 10usize] = hx + 1.0f32;
    stkm_legs[lb + 11usize] = hy + 6.0f32;

    stkm_legs[lb + 12usize] = hx + 3.0f32;
    stkm_legs[lb + 13usize] = hy + 12.0f32;
    stkm_legs[lb + 14usize] = hx + 3.0f32;
    stkm_legs[lb + 15usize] = hy + 12.0f32;

    let ab = sid * 8usize;
    stkm_accs[ab + 0usize] = 0.0f32;
    stkm_accs[ab + 1usize] = 0.0f32;
    stkm_accs[ab + 2usize] = 0.0f32;
    stkm_accs[ab + 3usize] = 0.0f32;
    stkm_accs[ab + 4usize] = 0.0f32;
    stkm_accs[ab + 5usize] = 0.0f32;
    stkm_accs[ab + 6usize] = 0.0f32;
    stkm_accs[ab + 7usize] = 0.0f32;

    let mb = sid * 8usize;
    stkm_meta[mb + 0usize] = 1u32; // spwn
    stkm_meta[mb + 1usize] = 0u32; // comm
    stkm_meta[mb + 2usize] = 0u32; // pcomm
    stkm_meta[mb + 3usize] = elem; // elem
    stkm_meta[mb + 4usize] = 0u32; // frames
    stkm_meta[mb + 5usize] = 0u32; // rocket_boots
    stkm_meta[mb + 6usize] = 0u32; // fan
    // `spawn_id` (`mb + 7`) is `player.spawnID`, which
    // `Element_STKM_init_legs` leaves alone: it belongs to the spawn point, not
    // to the rig.
}

#[cube(launch_unchecked)]
#[allow(
    clippy::manual_clamp,
    clippy::manual_range_contains,
    clippy::too_many_arguments,
    clippy::approx_constant
)]
pub fn stickman_kernel(
    part_ty: &mut [u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_flags: &mut [u32],
    stkm_legs: &mut [f32],
    stkm_accs: &mut [f32],
    stkm_meta: &mut [u32],
    free_slots: &mut [Atomic<u32>],
    pmap: &[u32],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
    capacity: u32,
    elem_count: u32,
    comm1: u32,
    pcomm1: u32,
    comm2: u32,
    pcomm2: u32,
) {
    let id = ABSOLUTE_POS as u32;
    if id >= capacity {
        terminate!();
    }

    // Player spawn points (`SPAWN.cpp:52-67`, `Simulation.cpp:3911-3914`).
    //
    // TPT keeps one `SPAWN`/`SPAWN2` slot per player in `player.spawnID` and
    // recreates the stickman at that pixel whenever it is gone. Threads 0 and 1
    // own the two players, so the bookkeeping runs even while no stickman
    // exists; the death branch below only clears `spwn`.
    if id < 2u32 {
        let p = id as usize;
        let mb = (C_STKM_PLAYER1 + p) * 8usize;
        let spawn_ty = if p == 0 { C_PT_SPAWN } else { C_PT_SPAWN2 };
        let player_ty = if p == 0 { C_PT_STKM } else { C_PT_STKM2 };
        let recorded = stkm_meta[mb + 7usize];
        // A recorded point whose particle was destroyed or replaced is stale.
        let mut valid = recorded != 0u32 && (recorded as usize) <= capacity as usize;
        if valid && part_ty[recorded as usize - 1usize] != spawn_ty {
            valid = false;
        }
        if !valid {
            stkm_meta[mb + 7usize] = 0u32;
        }
        if valid && stkm_meta[mb + 0usize] == 0u32 {
            let spawn_slot = recorded as usize - 1usize;
            let new_slot = gpu_alloc_slot(free_slots);
            if new_slot != GPU_ALLOC_NONE {
                let ns = new_slot as usize;
                let base = (player_ty * 32u32) as usize;
                let hx = part_x[spawn_slot];
                let hy = part_y[spawn_slot];
                part_ty[ns] = player_ty;
                part_x[ns] = hx;
                part_y[ns] = hy;
                part_vx[ns] = 0.0f32;
                part_vy[ns] = 0.0f32;
                part_life[ns] =
                    f32::reinterpret(elem_props[base + WORD_DEFAULT_LIFE as usize]) as i32;
                part_ctype[ns] = 0u32;
                part_temp[ns] = f32::reinterpret(elem_props[base + WORD_DEFAULT_TEMP as usize]);
                part_tmp[ns] = p as i32;
                part_flags[ns] = C_FLAG_MOVABLE;
                init_legs(
                    stkm_legs,
                    stkm_accs,
                    stkm_meta,
                    C_STKM_PLAYER1 + p,
                    hx,
                    hy,
                    C_PT_DUST,
                );
            }
        }
    }

    let ty = part_ty[id as usize];
    if ty != C_PT_FIGH && ty != C_PT_STKM && ty != C_PT_STKM2 {
        terminate!();
    }

    let sid = if ty == C_PT_STKM {
        C_STKM_PLAYER1
    } else if ty == C_PT_STKM2 {
        C_STKM_PLAYER2
    } else {
        let tmp = part_tmp[id as usize];
        if tmp >= 0i32 && tmp < 100i32 {
            tmp as usize
        } else {
            (id as usize) % 100usize
        }
    };

    let mb = sid * 8usize;
    if stkm_meta[mb + 0usize] == 0u32 {
        // `STKM.cpp:75-79`: creating a stickman also creates the spawn point it
        // returns to, and records its slot in `player.spawnID`.
        if stkm_meta[mb + 7usize] == 0u32 && (sid == C_STKM_PLAYER1 || sid == C_STKM_PLAYER2) {
            let spawn_ty = if sid == C_STKM_PLAYER1 {
                C_PT_SPAWN
            } else {
                C_PT_SPAWN2
            };
            let spawn_slot = gpu_alloc_slot(free_slots);
            if spawn_slot != GPU_ALLOC_NONE {
                let ss = spawn_slot as usize;
                let base = (spawn_ty * 32u32) as usize;
                part_ty[ss] = spawn_ty;
                part_x[ss] = part_x[id as usize];
                part_y[ss] = part_y[id as usize];
                part_vx[ss] = 0.0f32;
                part_vy[ss] = 0.0f32;
                part_life[ss] =
                    f32::reinterpret(elem_props[base + WORD_DEFAULT_LIFE as usize]) as i32;
                part_ctype[ss] = 0u32;
                part_temp[ss] = f32::reinterpret(elem_props[base + WORD_DEFAULT_TEMP as usize]);
                part_tmp[ss] = 0i32;
                part_flags[ss] = 0u32;
                stkm_meta[mb + 7usize] = spawn_slot + 1u32;
            }
        }
        part_tmp[id as usize] = sid as i32;
        let life = part_life[id as usize];
        part_life[id as usize] = if life > 0i32 { life } else { 100i32 };
        let ctype = part_ctype[id as usize];
        let elem = if ctype > 0u32 && ctype < C_PT_NUM {
            ctype
        } else {
            C_PT_DUST
        };
        init_legs(
            stkm_legs,
            stkm_accs,
            stkm_meta,
            sid,
            part_x[id as usize],
            part_y[id as usize],
            elem,
        );
    }

    // `GameController.cpp:611-730` owns `player.comm`/`player2.comm` on the
    // host; the port passes both words as kernel scalars. Only the AI-driven
    // FIGH words still live in the device table (`FIGH.cpp:150-152`).
    let (comm, pcomm) = if ty == C_PT_FIGH {
        let fcomm = figh_ai_select_command(
            pmap,
            xres,
            yres,
            part_x[id as usize],
            part_y[id as usize],
            stkm_legs,
            stkm_meta,
            sid,
        );
        stkm_meta[mb + 1usize] = fcomm;
        (fcomm, stkm_meta[mb + 2usize])
    } else if sid == C_STKM_PLAYER1 {
        (comm1, pcomm1)
    } else if sid == C_STKM_PLAYER2 {
        (comm2, pcomm2)
    } else {
        (stkm_meta[mb + 1usize], stkm_meta[mb + 2usize])
    };
    let rocket_boots = stkm_meta[mb + 5usize] != 0u32;

    stkm_meta[mb + 4usize] += 1u32;

    // Temperature handling (`STKM.cpp:140-144`).
    let temp = part_temp[id as usize];
    if temp < 243.0f32 {
        part_life[id as usize] -= 1i32;
    } else if temp < 309.6f32 {
        part_temp[id as usize] += 1.0f32;
    }
    if temp >= 323.0f32 {
        part_life[id as usize] -= 1i32;
    }

    // Downward gravity & command velocities (`STKM.cpp:207-209, 258-400`).
    let _dt = 0.9f32;
    let dt2 = 0.81f32;

    if !rocket_boots {
        part_vy[id as usize] += 0.5f32;
    }

    let lb = sid * 16usize;
    let ab = sid * 8usize;

    if (comm & 0x01u32) != 0u32 {
        if rocket_boots {
            part_vx[id as usize] -= 0.35f32;
            stkm_accs[ab + 2usize] -= 0.15f32;
            stkm_accs[ab + 6usize] -= 0.15f32;
        } else {
            part_vx[id as usize] -= 0.5f32;
            stkm_accs[ab + 2usize] -= 3.0f32;
            stkm_accs[ab + 0usize] -= 1.0f32;
        }
    }

    if (comm & 0x02u32) != 0u32 {
        if rocket_boots {
            part_vx[id as usize] += 0.35f32;
            stkm_accs[ab + 2usize] += 0.15f32;
            stkm_accs[ab + 6usize] += 0.15f32;
        } else {
            part_vx[id as usize] += 0.5f32;
            stkm_accs[ab + 2usize] += 3.0f32;
            stkm_accs[ab + 0usize] += 1.0f32;
        }
    }

    if (comm & 0x04u32) != 0u32 {
        if rocket_boots {
            part_vy[id as usize] -= 0.45f32;
            stkm_accs[ab + 3usize] -= 0.45f32;
            stkm_accs[ab + 7usize] -= 0.45f32;
        } else {
            let foot1_ground = is_obstacle(
                pmap,
                xres,
                yres,
                stkm_legs[lb + 4usize] as i32,
                (stkm_legs[lb + 5usize] + 1.0f32) as i32,
            );
            let foot2_ground = is_obstacle(
                pmap,
                xres,
                yres,
                stkm_legs[lb + 12usize] as i32,
                (stkm_legs[lb + 13usize] + 1.0f32) as i32,
            );
            if foot1_ground || foot2_ground {
                part_vy[id as usize] -= 4.0f32;
                stkm_accs[ab + 3usize] -= 1.0f32;
                stkm_accs[ab + 7usize] -= 1.0f32;
            }
        }
    }

    // Verlet leg integration (`STKM.cpp:211-238`).
    let gvy = if rocket_boots { 0.0f32 } else { 1.0f32 };

    let pp0 =
        2.0f32 * stkm_legs[lb + 0usize] - stkm_legs[lb + 2usize] + stkm_accs[ab + 0usize] * dt2;
    stkm_legs[lb + 2usize] = stkm_legs[lb + 0usize];
    stkm_legs[lb + 0usize] = pp0;
    let pp1 =
        2.0f32 * stkm_legs[lb + 1usize] - stkm_legs[lb + 3usize] + stkm_accs[ab + 1usize] * dt2;
    stkm_legs[lb + 3usize] = stkm_legs[lb + 1usize];
    stkm_legs[lb + 1usize] = pp1;

    let pp4 =
        2.0f32 * stkm_legs[lb + 4usize] - stkm_legs[lb + 6usize] + stkm_accs[ab + 2usize] * dt2;
    stkm_legs[lb + 6usize] = stkm_legs[lb + 4usize];
    stkm_legs[lb + 4usize] = pp4;
    let pp5 = 2.0f32 * stkm_legs[lb + 5usize] - stkm_legs[lb + 7usize]
        + (stkm_accs[ab + 3usize] + gvy) * dt2;
    stkm_legs[lb + 7usize] = stkm_legs[lb + 5usize];
    stkm_legs[lb + 5usize] = pp5;

    let pp8 =
        2.0f32 * stkm_legs[lb + 8usize] - stkm_legs[lb + 10usize] + stkm_accs[ab + 4usize] * dt2;
    stkm_legs[lb + 10usize] = stkm_legs[lb + 8usize];
    stkm_legs[lb + 8usize] = pp8;
    let pp9 =
        2.0f32 * stkm_legs[lb + 9usize] - stkm_legs[lb + 11usize] + stkm_accs[ab + 5usize] * dt2;
    stkm_legs[lb + 11usize] = stkm_legs[lb + 9usize];
    stkm_legs[lb + 9usize] = pp9;

    let pp12 =
        2.0f32 * stkm_legs[lb + 12usize] - stkm_legs[lb + 14usize] + stkm_accs[ab + 6usize] * dt2;
    stkm_legs[lb + 14usize] = stkm_legs[lb + 12usize];
    stkm_legs[lb + 12usize] = pp12;
    let pp13 = 2.0f32 * stkm_legs[lb + 13usize] - stkm_legs[lb + 15usize]
        + (stkm_accs[ab + 7usize] + gvy) * dt2;
    stkm_legs[lb + 15usize] = stkm_legs[lb + 13usize];
    stkm_legs[lb + 13usize] = pp13;

    stkm_accs[ab + 0usize] = 0.0f32;
    stkm_accs[ab + 1usize] = 0.0f32;
    stkm_accs[ab + 2usize] = 0.0f32;
    stkm_accs[ab + 3usize] = 0.0f32;
    stkm_accs[ab + 4usize] = 0.0f32;
    stkm_accs[ab + 5usize] = 0.0f32;
    stkm_accs[ab + 6usize] = 0.0f32;
    stkm_accs[ab + 7usize] = 0.0f32;

    // Distance constraint solver (`STKM.cpp:539-562`).
    let hx = part_x[id as usize];
    let hy = part_y[id as usize];

    let d1 = 25.0f32
        / ((stkm_legs[lb + 0usize] - stkm_legs[lb + 4usize])
            * (stkm_legs[lb + 0usize] - stkm_legs[lb + 4usize])
            + (stkm_legs[lb + 1usize] - stkm_legs[lb + 5usize])
                * (stkm_legs[lb + 1usize] - stkm_legs[lb + 5usize])
            + 25.0f32)
        - 0.5f32;
    stkm_legs[lb + 4usize] -= (stkm_legs[lb + 0usize] - stkm_legs[lb + 4usize]) * d1;
    stkm_legs[lb + 5usize] -= (stkm_legs[lb + 1usize] - stkm_legs[lb + 5usize]) * d1;
    stkm_legs[lb + 0usize] += (stkm_legs[lb + 0usize] - stkm_legs[lb + 4usize]) * d1;
    stkm_legs[lb + 1usize] += (stkm_legs[lb + 1usize] - stkm_legs[lb + 5usize]) * d1;

    let d2 = 25.0f32
        / ((stkm_legs[lb + 8usize] - stkm_legs[lb + 12usize])
            * (stkm_legs[lb + 8usize] - stkm_legs[lb + 12usize])
            + (stkm_legs[lb + 9usize] - stkm_legs[lb + 13usize])
                * (stkm_legs[lb + 9usize] - stkm_legs[lb + 13usize])
            + 25.0f32)
        - 0.5f32;
    stkm_legs[lb + 12usize] -= (stkm_legs[lb + 8usize] - stkm_legs[lb + 12usize]) * d2;
    stkm_legs[lb + 13usize] -= (stkm_legs[lb + 9usize] - stkm_legs[lb + 13usize]) * d2;
    stkm_legs[lb + 8usize] += (stkm_legs[lb + 8usize] - stkm_legs[lb + 12usize]) * d2;
    stkm_legs[lb + 9usize] += (stkm_legs[lb + 9usize] - stkm_legs[lb + 13usize]) * d2;

    let d3 = 36.0f32
        / ((stkm_legs[lb + 0usize] - hx) * (stkm_legs[lb + 0usize] - hx)
            + (stkm_legs[lb + 1usize] - hy) * (stkm_legs[lb + 1usize] - hy)
            + 36.0f32)
        - 0.5f32;
    part_vx[id as usize] -= (stkm_legs[lb + 0usize] - hx) * d3;
    part_vy[id as usize] -= (stkm_legs[lb + 1usize] - hy) * d3;
    stkm_legs[lb + 0usize] += (stkm_legs[lb + 0usize] - hx) * d3;
    stkm_legs[lb + 1usize] += (stkm_legs[lb + 1usize] - hy) * d3;

    let d4 = 36.0f32
        / ((stkm_legs[lb + 8usize] - hx) * (stkm_legs[lb + 8usize] - hx)
            + (stkm_legs[lb + 9usize] - hy) * (stkm_legs[lb + 9usize] - hy)
            + 36.0f32)
        - 0.5f32;
    part_vx[id as usize] -= (stkm_legs[lb + 8usize] - hx) * d4;
    part_vy[id as usize] -= (stkm_legs[lb + 9usize] - hy) * d4;
    stkm_legs[lb + 8usize] += (stkm_legs[lb + 8usize] - hx) * d4;
    stkm_legs[lb + 9usize] += (stkm_legs[lb + 9usize] - hy) * d4;

    // Foot obstacle collisions (`STKM.cpp:563-588`).
    if is_obstacle(
        pmap,
        xres,
        yres,
        stkm_legs[lb + 4usize] as i32,
        stkm_legs[lb + 5usize] as i32,
    ) {
        stkm_legs[lb + 4usize] = stkm_legs[lb + 6usize];
        stkm_legs[lb + 5usize] = stkm_legs[lb + 7usize];
        if part_vy[id as usize] > 0.0f32 {
            part_vy[id as usize] = 0.0f32;
        }
    }
    if is_obstacle(
        pmap,
        xres,
        yres,
        stkm_legs[lb + 12usize] as i32,
        stkm_legs[lb + 13usize] as i32,
    ) {
        stkm_legs[lb + 12usize] = stkm_legs[lb + 14usize];
        stkm_legs[lb + 13usize] = stkm_legs[lb + 15usize];
        if part_vy[id as usize] > 0.0f32 {
            part_vy[id as usize] = 0.0f32;
        }
    }

    // Head element absorption (`STKM.cpp:411-448`).
    let hxi = hx as i32;
    let hyi = hy as i32;
    for ry in -2i32..=2i32 {
        for rx in -2i32..=2i32 {
            if rx != 0i32 || ry != 0i32 {
                let nx = hxi + rx;
                let ny = hyi + ry;
                if nx >= 0i32 && nx < xres as i32 && ny >= 0i32 && ny < yres as i32 {
                    let packed = pmap[(ny as u32 * xres + nx as u32) as usize];
                    let nty = packed & 0x1FFu32;
                    if nty != 0u32 {
                        if nty == C_PT_PLNT && part_life[id as usize] < 100i32 {
                            let cl = part_life[id as usize];
                            part_life[id as usize] = if cl <= 95i32 { cl + 5i32 } else { 100i32 };
                        } else if nty == C_PT_PLUT || nty == C_PT_GRVT {
                            stkm_meta[mb + 5usize] = 1u32; // rocket_boots
                        } else if nty == C_PT_STOR {
                            stkm_meta[mb + 6usize] = 1u32; // fan
                        } else if nty < elem_count
                            && nty != C_PT_STKM
                            && nty != C_PT_STKM2
                            && nty != C_PT_FIGH
                        {
                            stkm_meta[mb + 3usize] = nty;
                            part_ctype[id as usize] = nty;
                        }

                        // Deadly / radioactive damage
                        let props = elem_props[(nty * 32u32 + WORD_PROPERTIES) as usize];
                        if (props & (C_PROP_DEADLY | C_PROP_RADIOACTIVE)) != 0u32 {
                            part_life[id as usize] -= 1i32;
                        }
                    }
                }
            }
        }
    }

    // Foot deadly / radioactive contact damage
    let f1x = stkm_legs[lb + 4usize] as i32;
    let f1y = stkm_legs[lb + 5usize] as i32;
    if f1x >= 0i32 && f1x < xres as i32 && f1y >= 0i32 && f1y < yres as i32 {
        let p1 = pmap[(f1y as u32 * xres + f1x as u32) as usize];
        let t1 = p1 & 0x1FFu32;
        if t1 != 0u32 && t1 < elem_count {
            let props = elem_props[(t1 * 32u32 + WORD_PROPERTIES) as usize];
            if (props & (C_PROP_DEADLY | C_PROP_RADIOACTIVE)) != 0u32 {
                part_life[id as usize] -= 1i32;
            }
        }
    }

    let f2x = stkm_legs[lb + 12usize] as i32;
    let f2y = stkm_legs[lb + 13usize] as i32;
    if f2x >= 0i32 && f2x < xres as i32 && f2y >= 0i32 && f2y < yres as i32 {
        let p2 = pmap[(f2y as u32 * xres + f2x as u32) as usize];
        let t2 = p2 & 0x1FFu32;
        if t2 != 0u32 && t2 < elem_count {
            let props = elem_props[(t2 * 32u32 + WORD_PROPERTIES) as usize];
            if (props & (C_PROP_DEADLY | C_PROP_RADIOACTIVE)) != 0u32 {
                part_life[id as usize] -= 1i32;
            }
        }
    }

    let frames = stkm_meta[mb + 4usize];
    if (comm & 0x08u32) != 0u32 && frames >= 8u32 {
        let new_slot = gpu_alloc_slot(free_slots);
        if new_slot != GPU_ALLOC_NONE {
            let mut dx = 1.0f32;
            let dy = 0.0f32;
            if (pcomm & 0x01u32) != 0u32 {
                dx = -1.0f32;
            } else if (pcomm & 0x02u32) != 0u32 {
                dx = 1.0f32;
            }

            let elem = stkm_meta[mb + 3usize];
            let ns = new_slot as usize;
            part_ty[ns] = elem;
            part_x[ns] = hx + dx * 3.0f32;
            part_y[ns] = hy + dy * 3.0f32 - 1.0f32;
            part_vx[ns] = dx * 5.0f32;
            part_vy[ns] = dy * 5.0f32;
            part_life[ns] = 30i32;
            part_temp[ns] = 293.15f32;
            part_ctype[ns] = 0u32;
            part_flags[ns] = C_FLAG_MOVABLE;
        }
        stkm_meta[mb + 4usize] = 0u32; // reset frames
    }

    if part_life[id as usize] <= 0i32 {
        part_ty[id as usize] = C_PT_NONE;
        stkm_meta[mb + 0usize] = 0u32; // spwn = 0
        terminate!();
    }

    // `FIGH.cpp:150`: a fighter saves its command as `pcomm` for the next AI
    // pass. Player `pcomm` comes from the host (snapshotted on a left/right
    // key release, `GameController.cpp:704-705`), so it is never overwritten.
    if ty == C_PT_FIGH {
        stkm_meta[mb + 2usize] = comm;
    }
}

#[cfg(test)]
pub(crate) fn init_legs_host(
    stkm_legs: &mut [f32],
    stkm_accs: &mut [f32],
    stkm_meta: &mut [u32],
    sid: usize,
    hx: f32,
    hy: f32,
    elem: u32,
) {
    let lb = sid * 16;
    stkm_legs[lb + 0] = hx - 1.0;
    stkm_legs[lb + 1] = hy + 6.0;
    stkm_legs[lb + 2] = hx - 1.0;
    stkm_legs[lb + 3] = hy + 6.0;

    stkm_legs[lb + 4] = hx - 3.0;
    stkm_legs[lb + 5] = hy + 12.0;
    stkm_legs[lb + 6] = hx - 3.0;
    stkm_legs[lb + 7] = hy + 12.0;

    stkm_legs[lb + 8] = hx + 1.0;
    stkm_legs[lb + 9] = hy + 6.0;
    stkm_legs[lb + 10] = hx + 1.0;
    stkm_legs[lb + 11] = hy + 6.0;

    stkm_legs[lb + 12] = hx + 3.0;
    stkm_legs[lb + 13] = hy + 12.0;
    stkm_legs[lb + 14] = hx + 3.0;
    stkm_legs[lb + 15] = hy + 12.0;

    let ab = sid * 8;
    for k in 0..8 {
        stkm_accs[ab + k] = 0.0;
    }

    let mb = sid * 8;
    stkm_meta[mb + 0] = 1;
    stkm_meta[mb + 1] = 0;
    stkm_meta[mb + 2] = 0;
    stkm_meta[mb + 3] = elem;
    stkm_meta[mb + 4] = 0;
    stkm_meta[mb + 5] = 0;
    stkm_meta[mb + 6] = 0;
    stkm_meta[mb + 7] = 0;
}
