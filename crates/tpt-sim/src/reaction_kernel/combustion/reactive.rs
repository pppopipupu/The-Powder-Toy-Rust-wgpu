//! Reactive liquids: LITH, EXOT and SING.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::CR_NO_OCCUPANCY;
use super::*;

/// `MAX_TEMP / 2` (`SING.cpp:130`): the temperature of every pop product.
const SING_POP_TEMP: f32 = 4999.5;
/// `std::numbers::pi_v<float>` (`SING.cpp:100`), written as a literal because a
/// `#[cube]` body cannot reference a host `const` path; the value is
/// bit-identical to `std::f32::consts::PI`.
#[allow(clippy::approx_constant)]
const SING_POP_PI: f32 = 3.141_592_7;
/// `2 * pi` for the random velocity direction (`SING.cpp:131`).
const SING_POP_TAU: f32 = 6.283_185_5;
const _: () = assert!(SING_POP_TEMP == MAX_TEMP / 2.0);
const _: () = assert!(SING_POP_TAU == 2.0 * SING_POP_PI);

/// `u32` words per row of `gpu_spawn_defaults` (mirrored for kernel bodies).
const SPAWN_DEFAULT_WORDS_U32: u32 = 8;
const _: () = assert!(tpt_core::elements::SPAWN_DEFAULT_WORDS == 8);

/// `Simulation::create_part(i, x, y, ty)`'s whole-row reset, for a kernel
/// whose `ctype` buffer is atomic (`LITH`'s neighbour energy trade needs it).
///
/// `EXOT.cpp:132-139` and `EXOT.cpp:176-183` convert with `create_part(i, ...)`
/// instead of `part_change_type`, so the slot restarts from the target's
/// `DefaultProperties`: temperature, life, `ctype`, `tmp`, flags and velocity
/// here, with `tmp2`..`tmp4` and `dcolour` finished by `neut_row_reset_kernel`
/// later in the same reaction step (`FLAG_ROW_REBUILT`). The `Create` hook the
/// reference runs afterwards is applied at the call site when it has fields
/// worth reproducing (the `CFLM` life roll), exactly like the other in-place
/// transmutes.
#[cube]
#[allow(clippy::too_many_arguments)]
fn rebuild_slot_with_atomic_ctype(
    spawn_defaults: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [Atomic<u32>],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_flags: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    nid: usize,
    ty: u32,
) {
    let db = (ty * SPAWN_DEFAULT_WORDS_U32) as usize;
    part_ty[nid] = ty;
    part_temp[nid] = f32::reinterpret(spawn_defaults[db]);
    part_life[nid] = spawn_defaults[db + 1usize] as i32;
    part_ctype[nid].store(spawn_defaults[db + 2usize]);
    part_tmp[nid] = spawn_defaults[db + 3usize] as i32;
    part_flags[nid] =
        crate::reaction_kernel::neut::FLAG_ROW_REBUILT | (part_flags[nid] & FLAG_STASIS);
    part_vx[nid] = 0.0f32;
    part_vy[nid] = 0.0f32;
}

/// `pv[cx, cy]` pulled towards `target` by 10 % (`SING.cpp:58-70`).
#[cube]
fn pv_raise_to(pv: &mut [f32], xcells: u32, ycells: u32, cx: i32, cy: i32, target: f32) {
    if cx >= 0i32 && cy >= 0i32 && (cx as u32) < xcells && (cy as u32) < ycells {
        let i = ((cy as u32) * xcells + (cx as u32)) as usize;
        if pv[i] < target {
            pv[i] += 0.1f32 * (target - pv[i]);
        }
    }
}

/// `pv[cx, cy] = clamp(pv + 10 % of (target - pv))` (`SING.cpp:72-77`).
#[cube]
fn pv_raise_clamped(pv: &mut [f32], xcells: u32, ycells: u32, cx: i32, cy: i32, target: f32) {
    if cx >= 0i32 && cy >= 0i32 && (cx as u32) < xcells && (cy as u32) < ycells {
        let i = ((cy as u32) * xcells + (cx as u32)) as usize;
        pv[i] = clampf(
            pv[i] + 0.1f32 * (target - pv[i]),
            MIN_PRESSURE,
            MAX_PRESSURE,
        );
    }
}

/// `LITH`, `EXOT` and `SING`.
///
/// All three need `tmp`/`tmp2`/`ctype` plus the air pressure, so they share one
/// kernel. The per-slot state is kept in locals and written back only when the
/// element did not convert into another type this frame. `SING`'s pop pushes
/// one creation record per product (`SING.cpp:106-140`), so the velocity pair
/// and the device queue are bound here as well.
#[cube(launch_unchecked)]
pub(crate) fn reactive_liquid_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    // `ctype` is atomic here: two lithiums trade stored energy into each
    // other's slot from parallel lanes, so a plain read-modify-write lost
    // updates and the pile leaked energy (the serial reference never does).
    part_ctype: &mut [Atomic<u32>],
    air_pv: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    spawn_defaults: &[u32],
    part_flags: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
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
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0xC2B2_AE35u32));

                if ty == PT_LITH {
                    // `LITH.cpp:64-260`.
                    //
                    // `storedEnergy` / `burnTimer` / `hydrogenationFactor` /
                    // `carbonationFactor` are C++ references straight into
                    // `parts[i]`, so every assignment is visible to the rest of
                    // the frame. Caching them in locals and flushing at the end
                    // let a lithium's own write-back undo the energy a
                    // neighbouring lithium had just traded into it, so a pile
                    // never accumulated charge (and the busier the trade loop,
                    // the worse it got). The fields are read and written in
                    // place here instead, exactly like the reference.
                    if (part_ctype[id].load() as i32) < 0i32 {
                        part_ctype[id].store(0u32);
                    }
                    let mut charged = false;
                    let mut discharged = false;
                    let mut converted = false;

                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if (dx != 0i32 || dy != 0i32) && !converted {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv == 0u32 {
                                // `LITH.cpp:86-91`: burning lithium spits a
                                // `FIRE` into the empty cell of the 5x5 ring.
                                // `create_part(-1, ...)` leaves every other
                                // field on the registry defaults, which is what
                                // `request_create_plain` asks the drain for.
                                if part_life[id] > LITH_FIRE_SPIT_BURN
                                    && chance(seed ^ n, 1u32, 10u32)
                                {
                                    request_create_plain(create_queue, PT_FIRE, px + dx, py + dy);
                                }
                            } else {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if nty != PT_NONE {
                                    let blocked =
                                        insulated(pmap, xres, yres, px, py, px + dx, py + dy);
                                    if nty == PT_SLTW
                                        || nty == PT_WTRV
                                        || nty == PT_WATR
                                        || nty == PT_DSTW
                                        || nty == PT_CBNW
                                    {
                                        if part_life[id] > LITH_WATER_BURN {
                                            //@ LITH + SLTW/WATR/DSTW/CBNW -> LITH + WTRV
                                            part_ty[nid] = PT_WTRV;
                                            part_temp[nid] = 440.0f32;
                                        } else if part_tmp[id] + part_tmp2[id] < 10i32 {
                                            if part_temp[id] > 440.0f32 {
                                                //@ LITH + water -> LITH + H2
                                                let stored = part_ctype[id].load() as i32;
                                                let mut next = 1024i32 + stored * 4i32;
                                                if stored > 24i32 {
                                                    next = 1120i32;
                                                }
                                                part_life[id] = next;
                                                part_ty[nid] = PT_H2;
                                                part_tmp[id] = 10i32;
                                            } else {
                                                let stored = part_ctype[id].load() as i32;
                                                part_temp[id] = clampf(
                                                    part_temp[id]
                                                        + 20.365f32
                                                        + (stored as f32)
                                                            * (stored as f32)
                                                            * 1.5f32,
                                                    MIN_TEMP,
                                                    MAX_TEMP,
                                                );
                                                part_ty[nid] = PT_H2;
                                                part_tmp[id] += 1i32;
                                            }
                                        }
                                    } else if nty == PT_CO2 {
                                        if part_tmp[id] + part_tmp2[id] < 10i32 {
                                            part_ty[nid] = PT_NONE;
                                            part_tmp2[id] += 1i32;
                                        }
                                    } else if nty == PT_SPRK {
                                        if !blocked
                                            && part_tmp[id] + part_tmp2[id] < 5i32
                                            && part_ctype[nid].load() == PT_PSCN
                                            && part_life[nid] == 3i32
                                            && !charged
                                            && part_life[id] == 0i32
                                        {
                                            charged = true;
                                        }
                                    } else if nty == PT_NSCN {
                                        if !blocked
                                            && part_life[nid] == 0i32
                                            && (part_ctype[id].load() as i32) > 0i32
                                            && part_life[id] == 0i32
                                        {
                                            part_ty[nid] = PT_SPRK;
                                            part_life[nid] = 4i32;
                                            part_ctype[nid].store(PT_NSCN);
                                            discharged = true;
                                        }
                                    } else if nty == PT_FIRE {
                                        if part_temp[id] > 440.0f32
                                            && part_tmp[id] < 6i32
                                            && chance(seed ^ n, 1u32, 40u32)
                                        {
                                            part_life[id] = LITH_BURN_START;
                                            part_tmp[id] += 1i32;
                                        }
                                    } else if nty == PT_O2
                                        && part_life[id] > 1000i32
                                        && chance(seed ^ n, 1u32, 10u32)
                                    {
                                        //@ LITH + O2 -> 2xPLSM
                                        part_ty[id] = PT_PLSM;
                                        part_ty[nid] = PT_PLSM;
                                        air_pv[cidx] += 4.0f32;
                                        converted = true;
                                    }
                                }
                            }
                        }
                    }

                    if !converted {
                        if charged {
                            part_ctype[id].fetch_add(1u32);
                            part_life[id] = 8i32;
                        }
                        if discharged {
                            // Atomic `-1`: the wrapping negation keeps the
                            // decrement visible to a concurrent trade.
                            part_ctype[id].fetch_add((!1u32) + 1u32);
                            part_life[id] = 8i32;
                        }

                        // `LITH.cpp:203-235`: nine random energy trades. The
                        // transfer is written straight into both particles,
                        // which is what lets a pile equalise.
                        let mut traded = false;
                        for trade in 0u32..9u32 {
                            let rx = rand_range(
                                hash_u32(seed ^ mix_index(trade, 0x1234_5678u32)),
                                -3i32,
                                3i32,
                            );
                            let ry = rand_range(
                                hash_u32(seed ^ mix_index(trade, 0x9ABC_DEF0u32)),
                                -3i32,
                                3i32,
                            );
                            if (rx != 0i32 || ry != 0i32) && !traded {
                                let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    if part_ty[nid] == PT_LITH {
                                        let stored = part_ctype[id].load() as i32;
                                        let other = part_ctype[nid].load() as i32;
                                        if part_life[id] < 1000i32
                                            && stored > 90i32
                                            && part_life[nid] > 1000i32
                                        {
                                            part_life[id] = 1024i32;
                                        }
                                        if stored > other {
                                            let mut transfer = stored - other;
                                            transfer -= transfer / 2i32;
                                            // Both halves are atomic so a trade
                                            // that races another one still
                                            // conserves the total stored energy.
                                            part_ctype[nid].fetch_add(transfer as u32);
                                            part_ctype[id].fetch_add((!(transfer as u32)) + 1u32);
                                            traded = true;
                                        }
                                    }
                                }
                            }
                        }

                        // Overcharged - begin explosion.
                        if part_life[id] < 1000i32
                            && (part_ctype[id].load() as i32) >= LITH_OVERCHARGE
                        {
                            part_life[id] = 1024i32;
                        }
                        if part_life[id] == 1000i32 {
                            let stored = part_ctype[id].load() as i32;
                            part_life[id] = 0i32;
                            part_ty[id] = PT_LAVA;
                            if part_tmp2[id] < 3i32 {
                                part_temp[id] =
                                    clampf(500.0f32 + stored as f32 * 10.0f32, MIN_TEMP, MAX_TEMP);
                                //@ LITH -> LAVA(LITH)
                                part_ctype[id].store(PT_LITH);
                            } else {
                                part_temp[id] =
                                    clampf(2000.0f32 + stored as f32 * 10.0f32, MIN_TEMP, MAX_TEMP);
                                //@ LITH -> LAVA(GLAS)
                                part_ctype[id].store(PT_GLAS);
                            }
                        }
                    }
                } else if ty == PT_EXOT {
                    // `EXOT.cpp:56-215`.
                    let mut life = part_life[id];
                    let mut tmp = part_tmp[id];
                    let mut tmp2 = part_tmp2[id];
                    let mut ctype = part_ctype[id].load();
                    let mut converted = false;

                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if (dx != 0i32 || dy != 0i32) && !converted {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if nty != PT_NONE {
                                    if nty == PT_WARP {
                                        if part_tmp2[nid] > 2000i32
                                            && chance(seed ^ n, 1u32, 100u32)
                                        {
                                            tmp2 += 100i32;
                                        }
                                    } else if nty == PT_EXOT {
                                        if part_ctype[nid].load() == PT_PROT {
                                            ctype = PT_PROT;
                                        }
                                        if part_life[nid] == 1500i32
                                            && chance(seed ^ n ^ 0x1u32, 1u32, 1000u32)
                                        {
                                            life = 1500i32;
                                        }
                                    } else if nty == PT_LAVA {
                                        let nct = part_ctype[nid].load();
                                        if nct == PT_TTAN || nct == PT_GOLD {
                                            if chance(seed ^ n ^ 0x2u32, 1u32, 10u32) {
                                                part_ctype[nid].store(PT_VIBR);
                                                part_ty[id] = PT_NONE;
                                                converted = true;
                                            }
                                        } else if nct == PT_VIBR
                                            && chance(seed ^ n ^ 0x3u32, 1u32, 1000u32)
                                        {
                                            part_ty[id] = PT_NONE;
                                            converted = true;
                                        }
                                    } else if nty == PT_ELEC || nty == PT_SPRK {
                                        tmp2 += 5i32;
                                        life = 1000i32;
                                        if nty == PT_ELEC {
                                            part_ty[nid] = PT_NONE;
                                        }
                                    }
                                    if !converted && tmp > 245i32 && life > 1337i32 {
                                        let absorb = nty != PT_EXOT
                                            && nty != PT_BREC
                                            && nty != PT_DMND
                                            && nty != PT_CLNE
                                            && nty != PT_PRTI
                                            && nty != PT_PRTO
                                            && nty != PT_PCLN
                                            && nty != PT_VOID
                                            && nty != PT_NBHL
                                            && nty != PT_WARP;
                                        if absorb {
                                            // `EXOT.cpp:132-139`: the
                                            // absorption is a `create_part(i,
                                            // x, y, rt)`, so the slot restarts
                                            // from `rt`'s default row.
                                            rebuild_slot_with_atomic_ctype(
                                                spawn_defaults,
                                                part_ty,
                                                part_life,
                                                part_ctype,
                                                part_temp,
                                                part_tmp,
                                                part_flags,
                                                part_vx,
                                                part_vy,
                                                id,
                                                nty,
                                            );
                                            converted = true;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !converted {
                        tmp -= 1i32;
                        if tmp <= 0i32 {
                            tmp = 250i32;
                        }
                        tmp2 -= 1i32;
                        if tmp2 < 1i32 {
                            tmp2 = 1i32;
                        } else if tmp2 > EXOT_WARP_TMP2 {
                            tmp2 = 10000i32;
                            if life < 1001i32 {
                                part_ty[id] = PT_WARP;
                                converted = true;
                            }
                        } else if life < 1001i32 {
                            air_pv[cidx] += (tmp2 as f32) / 160_000.0f32;
                        }
                        if !converted
                            && air_pv[cidx] > 200.0f32
                            && part_temp[id] > 9000.0f32
                            && tmp2 > 200i32
                        {
                            tmp2 = EXOT_WARP_TMP2;
                            part_ty[id] = PT_WARP;
                            converted = true;
                        }
                        if !converted && tmp2 > 100i32 {
                            let mut traded = false;
                            for trade in 0u32..9u32 {
                                let rx = rand_range(
                                    hash_u32(seed ^ mix_index(trade, 0x51ED_2701u32)),
                                    -2i32,
                                    2i32,
                                );
                                let ry = rand_range(
                                    hash_u32(seed ^ mix_index(trade, 0x7F4A_7C15u32)),
                                    -2i32,
                                    2i32,
                                );
                                if (rx != 0i32 || ry != 0i32) && !traded {
                                    let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                                    if nv != 0u32 {
                                        let nid = pmap_slot(nv);
                                        if part_ty[nid] == PT_EXOT {
                                            let other = part_tmp2[nid];
                                            if tmp2 > other && other >= 0i32 {
                                                let tym = tmp2 - other;
                                                if tym == 1i32 {
                                                    part_tmp2[nid] = other + 1i32;
                                                    tmp2 -= 1i32;
                                                    traded = true;
                                                } else if tym > 0i32 {
                                                    part_tmp2[nid] = other + tym / 2i32;
                                                    tmp2 -= tym / 2i32;
                                                    traded = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if !converted {
                            if ctype == PT_PROT {
                                if part_temp[id] < 50.0f32 {
                                    // `EXOT.cpp:176-183`: `create_part(i, x, y,
                                    // PT_CFLM)` copies the whole default row
                                    // and then the `CFLM` create hook sets
                                    // `life = rng.between(50, 199)`.
                                    rebuild_slot_with_atomic_ctype(
                                        spawn_defaults,
                                        part_ty,
                                        part_life,
                                        part_ctype,
                                        part_temp,
                                        part_tmp,
                                        part_flags,
                                        part_vx,
                                        part_vy,
                                        id,
                                        PT_HFLM,
                                    );
                                    part_life[id] = 50i32 + rand_range(seed, 0i32, 149i32);
                                    converted = true;
                                } else {
                                    part_temp[id] -= 1.0f32;
                                }
                            } else if part_temp[id] < 273.15f32 {
                                air_pv[cidx] -= 0.01f32;
                                tmp -= 1i32;
                            }
                        }
                    }

                    if !converted {
                        part_life[id] = life;
                        part_tmp[id] = tmp;
                        part_tmp2[id] = tmp2;
                        part_ctype[id].store(ctype);
                    }
                } else if ty == PT_SING {
                    // `SING.cpp:50-152`.
                    let mut life = part_life[id];
                    let mut tmp = part_tmp[id];
                    let mut converted = false;
                    let singularity = -(life as f32);
                    let cx = px / (CELL as i32);
                    let cy = py / (CELL as i32);
                    pv_raise_to(air_pv, xcells, ycells, cx, cy, singularity);
                    pv_raise_to(air_pv, xcells, ycells, cx, cy + 1i32, singularity);
                    pv_raise_to(air_pv, xcells, ycells, cx, cy - 1i32, singularity);
                    pv_raise_clamped(air_pv, xcells, ycells, cx + 1i32, cy, singularity);
                    pv_raise_clamped(air_pv, xcells, ycells, cx + 1i32, cy + 1i32, singularity);
                    pv_raise_clamped(air_pv, xcells, ycells, cx - 1i32, cy, singularity);
                    pv_raise_clamped(air_pv, xcells, ycells, cx - 1i32, cy - 1i32, singularity);

                    if life < 1i32 {
                        // `SING.cpp:66-77`: the 3x3 pressure burst.
                        for oy in 0u32..3u32 {
                            for ox in 0u32..3u32 {
                                let ccx = cx + ox as i32 - 1i32;
                                let ccy = cy + oy as i32 - 1i32;
                                if ccx >= 0i32
                                    && ccy >= 0i32
                                    && (ccx as u32) < xcells
                                    && (ccy as u32) < ycells
                                {
                                    let ci = ((ccy as u32) * xcells + (ccx as u32)) as usize;
                                    air_pv[ci] =
                                        clampf(air_pv[ci] + tmp as f32, MIN_PRESSURE, MAX_PRESSURE);
                                }
                            }
                        }
                        // `SING.cpp:79-111`: the pop spawns
                        // `int((abs(tmp)/8)^2 * pi)` particles — 3019 when the
                        // stored count is over 255 — and every one of them
                        // starts with a random `life`, half the maximum
                        // temperature and 0..5 px/frame in a random direction
                        // (`SING.cpp:99-105`).
                        // One queue record per product; the drain drops the
                        // overflow when the queue or the slot window fills,
                        // which is TPT's `create_part` returning `-1`.
                        let abs_tmp = iabs(tmp);
                        let k = (abs_tmp / 8i32) as f32;
                        let mut spawncount = (k * k * SING_POP_PI) as i32;
                        if abs_tmp > 255i32 {
                            spawncount = 3019i32;
                        }
                        for j in 0u32..(spawncount as u32) {
                            let r = hash_u32(seed ^ mix_index(j, 0x9E37_79B9u32));
                            let product_life = (hash_u32(r ^ 0x85EB_CA6Bu32) % 300u32) as i32;
                            let angle = rand_unit(hash_u32(r ^ 0xC2B2_AE35u32)) * SING_POP_TAU;
                            let speed = rand_unit(hash_u32(r ^ 0x27D4_EB2Fu32)) * 5.0f32;
                            let pop_mask =
                                CR_NO_OCCUPANCY | CR_SET_LIFE | CR_SET_TEMP | CR_SET_VX | CR_SET_VY;
                            let pick = r % 3u32;
                            if pick == 0u32 {
                                // `SING.cpp:85-88`: `SING -> PHOT`.
                                request_create(
                                    create_queue,
                                    PT_PHOT,
                                    px,
                                    py,
                                    pop_mask,
                                    product_life,
                                    0u32,
                                    SING_POP_TEMP,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0u32,
                                    0u32,
                                    speed * angle.cos(),
                                    speed * angle.sin(),
                                );
                            } else if pick == 1u32 {
                                // `SING.cpp:89-92`: `SING -> NEUT`.
                                request_create(
                                    create_queue,
                                    PT_NEUT,
                                    px,
                                    py,
                                    pop_mask,
                                    product_life,
                                    0u32,
                                    SING_POP_TEMP,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0u32,
                                    0u32,
                                    speed * angle.cos(),
                                    speed * angle.sin(),
                                );
                            } else {
                                // `SING.cpp:93-96`: `SING -> ELEC`.
                                request_create(
                                    create_queue,
                                    PT_ELEC,
                                    px,
                                    py,
                                    pop_mask,
                                    product_life,
                                    0u32,
                                    SING_POP_TEMP,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0u32,
                                    0u32,
                                    speed * angle.cos(),
                                    speed * angle.sin(),
                                );
                            }
                        }
                        part_ty[id] = PT_NONE;
                        converted = true;
                    } else {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32) && !converted {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    if nty != PT_DMND
                                        && nty != PT_NONE
                                        && chance(seed ^ n, 1u32, 3u32)
                                    {
                                        let mut absorbed = false;
                                        if nty == PT_SING && part_life[nid] > 10i32 {
                                            if life + part_life[nid] <= 255i32 {
                                                life += part_life[nid];
                                                absorbed = true;
                                            }
                                        } else if life + 3i32 > 255i32 {
                                            if nty != PT_SING
                                                && chance(seed ^ n ^ 0x1u32, 1u32, 1000u32)
                                            {
                                                part_ty[nid] = PT_SING;
                                                part_life[nid] = 60i32
                                                    + rand_range(seed ^ n ^ 0x2u32, 0i32, 49i32);
                                            }
                                        } else {
                                            life += 3i32;
                                            tmp += 1i32;
                                            absorbed = true;
                                        }
                                        if absorbed {
                                            part_temp[id] = clampf(
                                                part_temp[nid] + part_temp[id],
                                                MIN_TEMP,
                                                MAX_TEMP,
                                            );
                                            part_ty[nid] = PT_NONE;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !converted {
                        part_life[id] = life;
                        part_tmp[id] = tmp;
                    }
                }
            }
        }
    }
}

/// `EXOT.cpp:200-210`: a cold `EXOT` particle stops moving.
///
/// Split from [`reactive_liquid_kernel`], which already carries the creation
/// queue and cannot also carry the velocity pair.
#[cube(launch_unchecked)]
pub(crate) fn exot_freeze_kernel(
    part_ty: &[u32],
    part_temp: &[f32],
    part_ctype: &[u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    n_slots: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize
        && part_ty[i] == PT_EXOT
        && part_ctype[i] != PT_PROT
        && part_temp[i] < 273.15f32
    {
        part_vx[i] = 0.0f32;
        part_vy[i] = 0.0f32;
    }
}
