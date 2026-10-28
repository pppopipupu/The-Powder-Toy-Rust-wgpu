//! Water, ice, salt, clones and voids.
//!
//! Pixel parallel: every type change is written to `part_ty`, so the pass only
//! *reads* the spatial maps and stays free of cross-thread map races. Neighbour
//! conversions (`WATR` + `SALT`, `ICEI` + `SALT`, `FRZW` spreading, ...) use the
//! same last-writer-wins type store as [`super::corrosion`].
//!
//! The pass binds everything the reference `update` functions read, so no
//! reaction has to be approximated on the host:
//!
//! * `part_temp` — the salt-water melting gate of `ICEI.cpp:71`/`SNOW.cpp:67`
//!   (`temp > SLTW.low_temp`), the `-200 K` of `FRZW.cpp:75` and the `-1 K` a
//!   freeze-water ice sheds per frame (`ICEI.cpp:56-59`).
//! * `part_life` — `IRON`'s refractory test (`IRON.cpp:50-51`), `FRZW`'s
//!   lifetime-scaled crystallisation chance (`FRZW.cpp:70`) and the `life = 100`
//!   written by `FRZZ.cpp:64`.
//! * `part_vx` / `part_vy` — `WATR`'s `ROCK` erosion, which needs
//!   `|vx| + |vy| >= 0.5` (`WATR.cpp:89`).
//! * `photons` — `CLNE` latches the energy layer before the particle map
//!   (`CLNE.cpp:60-62`), so it can copy `PHOT`/`NEUT`/`ELEC`.
//! * `part_tmp` — `CLNE.cpp:72-73` stores the latched payload's `ctype` in the
//!   clone's `tmp` for the `LIFE` / `LAVA` payload cases, and `IRON.cpp:95-96`
//!   seeds the rust progress of a fresh `BMTL`.
//! * `part_vx` / `part_vy` are also written: `CLST.cpp:78-91` pushes clay dust
//!   away from every neighbouring clay pixel.
//! * `spawn_defaults` — the two `CLST.cpp:63-76` conversions use
//!   `create_part`, which rebuilds the slot from the target's
//!   `DefaultProperties` row (see [`change_type_in_place`]).
//!
//! The `RBDM`/`LRBD` ignition and the `SMKE`/`PLNT` neighbour cases of
//! `WATR.cpp:68-76` / `DSTW.cpp:86-110` / `SLTW.cpp:64-80` live here because
//! these are their only consumers; the combustion pass does not see them.

// Clippy's rewrite for these lints uses methods the device IR does not have
// (`is_multiple_of`, `clamp`, `Range::contains`, `saturating_sub`), and
// collapsing a nested `if` changes what the CubeCL macro expands, so the plain
// operators are deliberate.
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_clamp)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::shared::{
    FLAG_SPAWN_CLONE, hash_u32, mix_index, pmap_at, pmap_slot, seed_f32, stasis_frozen,
};
use crate::reaction_kernel::neut::change_type_in_place;

/// `SLTW`'s `LowTemperature` (Kelvin): salt-water ice only dissolves into
/// salt water above this point (`ICEI.cpp:71`, `SNOW.cpp:67`). `tpt-core`
/// carries the same 252.05 in `elements/data/liquids.rs`.
const SALTWATER_MELT_K: f32 = 252.05;
/// Plain ice melts at 273.15 K (`ICEI`/`SNOW` `high_temp`); the mirror above
/// must stay colder or "difficult to freeze" salt water would be a lie.
const ICE_MELT_K: f32 = 273.15;
const _: () = assert!(SALTWATER_MELT_K < ICE_MELT_K);
/// `IRON` rusts only while its `life` is zero (`IRON.cpp:50-51`).
const IRON_REFRACTORY_LIFE: i32 = 0;
/// `FRZW`'s registry `DefaultProperties.life` (`FRZW.cpp:46`), i.e. the value
/// from which the crystallisation ramp `(100 - life) / 50000` starts.
const FRZW_FULL_LIFE: i32 = 100;
/// Lifetime handed to water converted by `FRZZ` (`FRZZ.cpp:64`).
const FRZZ_FROZEN_LIFE: i32 = 100;
/// Temperature `FRZW` loses when it crystallises (`FRZW.cpp:75`).
const FRZW_CRYSTAL_DROP_K: f32 = 200.0;
/// Temperature a freeze-water ice loses per frame (`ICEI.cpp:58`).
const FRZW_ICE_COOL_K: f32 = 1.0;
/// `WATR` erodes `ROCK` only while it is moving (`WATR.cpp:89`).
const ROCK_ERODE_SPEED: f32 = 0.5;
/// `WATR.cpp:68` / `SLTW.cpp:72` ignition gate: `273.15 + 12`.
const WATER_RBDM_TEMP: f32 = 285.15;
/// `DSTW.cpp:86` compares against the literal `12.0f` (a TPT quirk, not a
/// Kelvin temperature), so distilled water ignites at any realistic value.
const DSTW_RBDM_TEMP: f32 = 12.0;
/// `DSTW.cpp:95-110`: both the smoke and the water must sit in this window.
const DISTILLED_WATER_SMOKE_MIN: f32 = 40.0 + 273.15;
/// See [`DISTILLED_WATER_SMOKE_MIN`].
const DISTILLED_WATER_SMOKE_MAX: f32 = 60.0 + 273.15;
/// Lifetime of the fire born from `RBDM`/`LRBD` (`WATR.cpp:72` / `SLTW.cpp:76`).
const WATER_IGNITION_FIRE_LIFE: i32 = 4;
/// `IRON.cpp:95-96`: `rng.between(20, 29)` seeds the rust progress.
const IRON_RUST_TMP_BASE: i32 = 20;
/// `CLST.cpp:78-91`: velocity kick per neighbouring clay pixel, by temperature.
const CLST_KICK_COLD: f32 = 0.05;
/// See [`CLST_KICK_COLD`].
const CLST_KICK_WARM: f32 = 0.015;
/// See [`CLST_KICK_COLD`].
const CLST_KICK_HOT: f32 = 0.01;
/// See [`CLST_KICK_COLD`].
const CLST_KICK_VERY_HOT: f32 = 0.005;
/// `PT_NUM` as a `u32` for the kernel body: `CLNE.cpp:66-69` rejects latched
/// ids at or past the end of the element table.
const PT_NUM_K: u32 = 200;
const _: () = assert!(PT_NUM_K as usize == PT_NUM);

// RNG stream salts: every reaction draws its rolls from its own stream, so two
// chances taken in the same neighbour step are independent draws like TPT's
// successive `sim->rng.chance` calls.
const SALT_WATR: u32 = 0x2545_F491;
const SALT_DSTW: u32 = 0x9E37_79B9;
const SALT_SLTW: u32 = 0x85EB_CA6B;
const SALT_ICE: u32 = 0xC2B2_AE35;
const SALT_IRON: u32 = 2_246_822_519;
const SALT_FRZW: u32 = 1_103_515_245;
const SALT_FRZZ: u32 = 0x27D4_EB2F;
const SALT_CLNE: u32 = 0x1656_67B1;
const SALT_CLST: u32 = 0x4B3A_89C1;
/// Second, independent roll derived from a neighbour's first one.
const ROLL_DERIVE: u32 = 0x517C_C1B7;

/// `f32` absolute value (CubeCL has no `f32::abs`).
#[cube]
fn fabs_f32(v: f32) -> f32 {
    let mut r = v;
    if v < 0.0f32 {
        r = -v;
    }
    r
}

/// The reference `restrict_flt(v, MIN_TEMP, MAX_TEMP)`.
#[cube]
fn restrict_temp(v: f32) -> f32 {
    let mut r = v;
    if v < MIN_TEMP {
        r = MIN_TEMP;
    }
    if v > MAX_TEMP {
        r = MAX_TEMP;
    }
    r
}

/// One deterministic roll for the reaction stream `salt` at neighbour `n`.
///
/// TPT draws from a shared stream, so the result depends on the update order;
/// the device pass has to hash the pixel and the neighbour slot instead. That
/// keeps every neighbour step an independent draw, exactly like the
/// `sim->rng.chance` call sitting inside the reference neighbour loop.
#[cube]
fn roll(frame_seed: u32, idx: u32, salt: u32, n: u32) -> u32 {
    hash_u32(frame_seed ^ mix_index(idx, salt) ^ mix_index(n, ROLL_DERIVE))
}

/// Water family, ice/snow, `CLNE` and `VOID`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn fluid_reaction_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    part_flags: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    photons: &[u32],
    spawn_defaults: &[u32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
    legacy: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty != PT_NONE {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let mut new_ty = ty;

                if ty == PT_WATR {
                    // `WATR.cpp:55-97`.
                    let speed = fabs_f32(part_vx[id]) + fabs_f32(part_vy[id]);
                    let mut dead = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !dead {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let r = roll(frame_seed, idx as u32, SALT_WATR, n);
                                if nty == PT_SALT && (r % 50u32) == 0u32 {
                                    // `WATR.cpp:60-67`: the roll is taken per
                                    // neighbour, the water becomes salt water
                                    // and the salt follows one try in three.
                                    new_ty = PT_SLTW;
                                    if (hash_u32(r ^ 0x51ED_270Bu32) % 3u32) == 0u32 {
                                        part_ty[nid] = PT_SLTW;
                                    }
                                } else if (nty == PT_RBDM || nty == PT_LRBD)
                                    && (legacy != 0u32 || part_temp[id] > WATER_RBDM_TEMP)
                                    && (r % 100u32) == 0u32
                                {
                                    // `WATR.cpp:68-74`: rubber ignites the
                                    // water on a 1/100 roll; `part_change_type`
                                    // keeps the temperature and velocity, and
                                    // the `ctype` records what burned.
                                    new_ty = PT_FIRE;
                                    part_life[id] = WATER_IGNITION_FIRE_LIFE;
                                    part_ctype[id] = PT_WATR;
                                } else if nty == PT_FIRE && part_ctype[nid] != PT_WATR {
                                    // `WATR.cpp:75-83`: fire that is not the
                                    // product of burning water (`ctype` left at
                                    // `WATR` by the `RBDM` reaction) is put out;
                                    // one extinguishing in thirty also deletes
                                    // the water pixel.
                                    part_ty[nid] = PT_NONE;
                                    if (hash_u32(r ^ 0x2C1B_3E4Du32) % 30u32) == 0u32 {
                                        dead = true;
                                    }
                                } else if nty == PT_SLTW && (r % 2000u32) == 0u32 {
                                    // `WATR.cpp:84-88`.
                                    new_ty = PT_SLTW;
                                } else if nty == PT_ROCK
                                    && speed >= ROCK_ERODE_SPEED
                                    && (r % 1000u32) == 0u32
                                {
                                    // `WATR.cpp:89-96`: moving water grinds rock
                                    // into sand one time in three, stone else.
                                    if (hash_u32(r ^ 0x7F4A_7C15u32) % 3u32) == 0u32 {
                                        part_ty[nid] = PT_SAND;
                                    } else {
                                        part_ty[nid] = PT_STNE;
                                    }
                                }
                            }
                        }
                    }
                    if dead {
                        new_ty = PT_NONE;
                    }
                } else if ty == PT_DSTW {
                    // `DSTW.cpp:51-119`.
                    let mut dead = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !dead {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let r = roll(frame_seed, idx as u32, SALT_DSTW, n);
                                if nty == PT_SALT && (r % 50u32) == 0u32 {
                                    // `DSTW.cpp:60-69`.
                                    new_ty = PT_SLTW;
                                    if (hash_u32(r ^ 0x51ED_270Bu32) % 3u32) == 0u32 {
                                        part_ty[nid] = PT_SLTW;
                                    }
                                } else if nty == PT_SLTW {
                                    // `DSTW.cpp:70-76`: the `PT_SLTW` case falls
                                    // through into the `PT_WATR` one in C++, so
                                    // a failed 1/2000 roll still gets the 1/100
                                    // chance to become plain water.
                                    if (r % 2000u32) == 0u32 {
                                        new_ty = PT_SLTW;
                                    } else if (hash_u32(r ^ 0x3AE9_1F27u32) % 100u32) == 0u32 {
                                        new_ty = PT_WATR;
                                    }
                                } else if nty == PT_WATR && (r % 100u32) == 0u32 {
                                    // `DSTW.cpp:77-83`.
                                    new_ty = PT_WATR;
                                } else if (nty == PT_RBDM || nty == PT_LRBD)
                                    && (legacy != 0u32 || part_temp[id] > DSTW_RBDM_TEMP)
                                    && (r % 100u32) == 0u32
                                {
                                    // `DSTW.cpp:84-93`: the literal `12.0f`
                                    // gate is deliberate; `part_change_type`
                                    // keeps the fields and no `ctype` is set
                                    // (distilled water does not mark its fire).
                                    new_ty = PT_FIRE;
                                    part_life[id] = WATER_IGNITION_FIRE_LIFE;
                                } else if nty == PT_FIRE {
                                    // `DSTW.cpp:94-102`: distilled water always
                                    // puts the fire out, then may be deleted.
                                    part_ty[nid] = PT_NONE;
                                    if (hash_u32(r ^ 0x2C1B_3E4Du32) % 30u32) == 0u32 {
                                        dead = true;
                                    }
                                } else if nty == PT_SMKE {
                                    // `DSTW.cpp:103-115`: both pixels need to
                                    // sit in the 40..60 C window, then 1 in
                                    // 100 turns the water into base and spends
                                    // the smoke.
                                    let ntemp = part_temp[nid];
                                    if ntemp > DISTILLED_WATER_SMOKE_MIN
                                        && ntemp < DISTILLED_WATER_SMOKE_MAX
                                        && part_temp[id] > DISTILLED_WATER_SMOKE_MIN
                                        && part_temp[id] < DISTILLED_WATER_SMOKE_MAX
                                        && (r % 100u32) == 0u32
                                    {
                                        new_ty = PT_BASE;
                                        part_life[id] = 1i32;
                                        part_ty[nid] = PT_NONE;
                                    }
                                }
                            }
                        }
                    }
                    if dead {
                        new_ty = PT_NONE;
                    }
                } else if ty == PT_SLTW {
                    // `SLTW.cpp:50-97`.
                    let mut dead = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !dead {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let r = roll(frame_seed, idx as u32, SALT_SLTW, n);
                                if nty == PT_SALT && (r % 2000u32) == 0u32 {
                                    // `SLTW.cpp:59-63`: salt dissolves into the
                                    // salt water it touches.
                                    part_ty[nid] = PT_SLTW;
                                } else if nty == PT_PLNT && (r % 40u32) == 0u32 {
                                    // `SLTW.cpp:64-67`: salt water kills the
                                    // plant it touches.
                                    part_ty[nid] = PT_NONE;
                                } else if (nty == PT_RBDM || nty == PT_LRBD)
                                    && (legacy != 0u32 || part_temp[id] > WATER_RBDM_TEMP)
                                    && (r % 100u32) == 0u32
                                {
                                    // `SLTW.cpp:72-78`: same ignition as fresh
                                    // water, `ctype` included.
                                    new_ty = PT_FIRE;
                                    part_life[id] = WATER_IGNITION_FIRE_LIFE;
                                    part_ctype[id] = PT_WATR;
                                } else if nty == PT_FIRE && part_ctype[nid] != PT_WATR {
                                    // `SLTW.cpp:78-88`.
                                    part_ty[nid] = PT_NONE;
                                    if (hash_u32(r ^ 0x2C1B_3E4Du32) % 30u32) == 0u32 {
                                        dead = true;
                                    }
                                }
                            }
                        }
                    }
                    if dead {
                        new_ty = PT_NONE;
                    }
                } else if ty == PT_ICEI || ty == PT_SNOW {
                    // `ICEI.cpp:52-89` / `SNOW.cpp:51-76`.
                    let mut temp = part_temp[id];
                    if part_ctype[id] == PT_FRZW {
                        // `ICEI.cpp:56-59` / `SNOW.cpp:53-56`: ice frozen out of
                        // `FRZW` keeps sinking in temperature, which is what
                        // eventually lets it melt back into (salt) water.
                        temp = restrict_temp(temp - FRZW_ICE_COOL_K);
                        part_temp[id] = temp;
                    }
                    let div = if ty == PT_ICEI { 200u32 } else { 333u32 };
                    let mut done = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !done {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let r = roll(frame_seed, idx as u32, SALT_ICE, n);
                                // `ICEI.cpp:69-78` forces the ice above the
                                // salt-water freezing point; `SNOW.cpp:66-71`
                                // has no temperature gate at all, so snow melts
                                // into brine even at its default 243 K.
                                if (nty == PT_SALT || nty == PT_SLTW)
                                    && (ty == PT_SNOW || temp > SALTWATER_MELT_K)
                                    && (r % div) == 0u32
                                {
                                    // `ICEI.cpp:69-78` / `SNOW.cpp:67-71`: both
                                    // the ice and the salt it touched become
                                    // salt water, and the update returns.
                                    new_ty = PT_SLTW;
                                    part_ty[nid] = PT_SLTW;
                                    done = true;
                                } else if ty == PT_ICEI && nty == PT_FRZZ && (r % 200u32) == 0u32 {
                                    // `ICEI.cpp:79-84`: ice seeds freeze powder,
                                    // which turns into freeze-water ice.
                                    part_ty[nid] = PT_ICEI;
                                    part_ctype[nid] = PT_FRZW;
                                }
                            }
                        }
                    }
                } else if ty == PT_IRON {
                    // `IRON.cpp:50-51`: refractory iron does not rust.
                    if part_life[id] == IRON_REFRACTORY_LIFE {
                        // `IRON.cpp:52-92`: salt, salt water, water and oxygen
                        // each attack the iron with their own odds; liquid
                        // oxygen dissolves it outright.
                        //
                        // `IRON.cpp:95-96` also seeds the rust progress in `tmp`
                        // (20..=29) on the first successful contact.
                        let roll = hash_u32(frame_seed ^ mix_index(idx as u32, SALT_IRON));
                        let mut rusted = false;
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32) && !rusted {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    let rusty = if nty == PT_SALT {
                                        (roll % 47u32) == 0u32
                                    } else if nty == PT_SLTW {
                                        (roll % 67u32) == 0u32
                                    } else if nty == PT_WATR {
                                        (roll % 1200u32) == 0u32
                                    } else if nty == PT_O2 {
                                        (roll % 250u32) == 0u32
                                    } else {
                                        nty == PT_LO2
                                    };
                                    if rusty {
                                        new_ty = PT_BMTL;
                                        part_tmp[id] = IRON_RUST_TMP_BASE
                                            + (hash_u32(roll ^ 0x9E37_79B9u32) % 10u32) as i32;
                                        rusted = true;
                                    }
                                }
                            }
                        }
                    }
                } else if ty == PT_FRZZ {
                    // `FRZZ.cpp:50-68`: freeze powder converts the water it
                    // touches into freeze water and is consumed by the reaction.
                    let mut frozen = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !frozen {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let r = roll(frame_seed, idx as u32, SALT_FRZZ, n);
                                if part_ty[nid] == PT_WATR && (r % 20u32) == 0u32 {
                                    // `FRZZ.cpp:60-67`: the water becomes freeze
                                    // water with a full lifetime, the powder is
                                    // killed.
                                    part_ty[nid] = PT_FRZW;
                                    part_life[nid] = FRZZ_FROZEN_LIFE;
                                    frozen = true;
                                }
                            }
                        }
                    }
                    if frozen {
                        new_ty = PT_NONE;
                    }
                } else if ty == PT_FRZW {
                    // `FRZW.cpp:51-77`.
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let r = roll(frame_seed, idx as u32, SALT_FRZW, n);
                                if part_ty[nid] == PT_WATR && (r % 14u32) == 0u32 {
                                    // `FRZW.cpp:62-66`.
                                    part_ty[nid] = PT_FRZW;
                                }
                            }
                        }
                    }
                    let life = part_life[id];
                    // A stream distinct from the neighbour-0 spread roll above.
                    let r =
                        hash_u32(frame_seed ^ mix_index(idx as u32, SALT_FRZW) ^ 0x94D0_49BBu32);
                    // `FRZW.cpp:70-71`: a spent lifetime uses the flat 1/192 roll,
                    // an active one the `(100 - life) / 50000` ramp. `chance`
                    // never fires for a non-positive numerator, so `life >= 100`
                    // contributes nothing.
                    let mut freeze =
                        life == 0i32 && (hash_u32(r ^ 0x5BD1_E995u32) % 192u32) == 0u32;
                    if !freeze && life >= 0i32 && life <= FRZW_FULL_LIFE {
                        let num = (FRZW_FULL_LIFE - life) as u32;
                        freeze = (hash_u32(r ^ 0x68E3_1DA4u32) % 50_000u32) < num;
                    }
                    if freeze {
                        // `FRZW.cpp:72-76`: the ice remembers what froze and
                        // arrives `FRZW_CRYSTAL_DROP_K` colder.
                        new_ty = PT_ICEI;
                        part_ctype[id] = PT_FRZW;
                        part_temp[id] = restrict_temp(part_temp[id] - FRZW_CRYSTAL_DROP_K);
                    }
                } else if ty == PT_CLNE {
                    let ctype = part_ctype[id];
                    if ctype == PT_NONE {
                        // `CLNE.cpp:54-77`: an empty clone latches onto the first
                        // real neighbour it sees. The energy layer is sampled
                        // first because `PHOT`/`NEUT`/`ELEC` live in `photons`,
                        // not `pmap`, and the exclusion list stops at the clone
                        // and stickman types — `VOID` is a legal payload.
                        let mut searching = 1u32;
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32) && searching == 1u32 {
                                let mut nv = pmap_at(photons, xres, yres, px + dx, py + dy);
                                if nv == 0u32 {
                                    nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                }
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    if nty != PT_NONE
                                        && nty != PT_CLNE
                                        && nty != PT_PCLN
                                        && nty != PT_BCLN
                                        && nty != PT_STKM
                                        && nty != PT_PBCN
                                        && nty != PT_STKM2
                                        && nty < PT_NUM_K
                                    {
                                        part_ctype[id] = nty;
                                        // `CLNE.cpp:72-73`: the payload's
                                        // `ctype` becomes the clone's `tmp` for
                                        // the `LIFE` and `LAVA` payload cases.
                                        if nty == PT_LIFE || nty == PT_LAVA {
                                            part_tmp[id] = part_ctype[nid] as i32;
                                        }
                                        searching = 0u32;
                                    }
                                }
                            }
                        }
                    } else {
                        // `CLNE.cpp:78-90`: a lightning payload only clones on
                        // one frame in thirty; a `LIFE` or `LAVA` payload copies
                        // its stored `tmp` into the child (the host applies it
                        // from `part_tmp`). Every other payload clones every
                        // frame. The host picks the free neighbouring pixel.
                        let mut spawn = true;
                        if ctype == PT_LIGH {
                            let r = hash_u32(frame_seed ^ mix_index(idx as u32, SALT_CLNE));
                            spawn = (r % 30u32) == 0u32;
                        }
                        if spawn {
                            part_flags[id] |= FLAG_SPAWN_CLONE;
                        }
                    }
                } else if ty == PT_VOID {
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if nty != PT_NONE && nty != PT_VOID {
                                    part_ty[nid] = PT_NONE;
                                }
                            }
                        }
                    }
                } else if ty == PT_CLST {
                    // `CLST.cpp:52-96`: clay dust turns to paste with water, or
                    // dynamite with nitroglycerin, and pushes itself away from
                    // neighbouring clay.
                    //
                    // C++ keeps scanning after a conversion: the water branch
                    // only changes the subject, so a later nitroglycerin
                    // neighbour can still replace it with dynamite (and the
                    // `CLST` kick keeps accumulating on the pixel). The loop is
                    // reproduced in reference order, and each conversion runs
                    // the same slot reset as `create_part` through
                    // [`change_type_in_place`].
                    let roll = hash_u32(frame_seed ^ mix_index(idx as u32, SALT_CLST));
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if nty == PT_WATR && hash_u32(roll ^ n).is_multiple_of(1500u32) {
                                    // `CLST.cpp:63-68`.
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
                                        PT_PSTS,
                                    );
                                    part_ty[nid] = PT_NONE;
                                } else if nty == PT_NITR {
                                    // `CLST.cpp:69-75`: both pixels restart as
                                    // dynamite.
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
                                        PT_BANG,
                                    );
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
                                        PT_BANG,
                                    );
                                } else if part_ty[nid] == PT_CLST {
                                    // `CLST.cpp:76-91`: the kick scales with the
                                    // clay's temperature, read live because an
                                    // earlier conversion may have reset it.
                                    let t = part_temp[id];
                                    // Seeded from the live temperature: a
                                    // host-literal `let mut` expands to a plain
                                    // `f32` and rejects the device assignments
                                    // below (`expected f32, found
                                    // NativeExpand<f32>`).
                                    let mut cxy = seed_f32(t, CLST_KICK_VERY_HOT);
                                    if t < 195.0f32 {
                                        cxy = CLST_KICK_COLD;
                                    } else if t < 295.0f32 {
                                        cxy = CLST_KICK_WARM;
                                    } else if t < 350.0f32 {
                                        cxy = CLST_KICK_HOT;
                                    }
                                    part_vx[id] += cxy * dx as f32;
                                    part_vy[id] += cxy * dy as f32;
                                }
                            }
                        }
                    }
                    // A conversion already wrote the slot through the shared
                    // reset helper; carry that type into the commit below.
                    new_ty = part_ty[id];
                }

                if new_ty != ty {
                    part_ty[id] = new_ty;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clst_reaction_host(roll: u32, nty: u32) -> Option<(u32, u32)> {
        if nty == PT_WATR && roll.is_multiple_of(1500) {
            Some((PT_PSTS, PT_NONE))
        } else if nty == PT_NITR {
            Some((PT_BANG, PT_BANG))
        } else {
            None
        }
    }

    #[test]
    fn clst_water_paste_reaction() {
        assert_eq!(clst_reaction_host(0, PT_WATR), Some((PT_PSTS, PT_NONE)));
        assert_eq!(clst_reaction_host(1500, PT_WATR), Some((PT_PSTS, PT_NONE)));
        assert_eq!(clst_reaction_host(1, PT_WATR), None);
    }

    #[test]
    fn clst_nitr_dynamite_reaction() {
        assert_eq!(clst_reaction_host(42, PT_NITR), Some((PT_BANG, PT_BANG)));
    }
}
