//! Fire, plasma, lava, hydrogen and oxygen chemistry.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::*;
use crate::reaction_kernel::neut::change_type_in_place;
use crate::reaction_kernel::shared::iabs;

/// `pmap` type bits; the 23-bit slot id sits above them.
const PMAP_TYPE: u32 = 0x1FF;

/// `FIRE` / `PLSM` / `LAVA` heating and 5x5 ignition, plus `H2` / `O2` chemistry.
///
/// Pixel parallel: the thread reads its own `pmap` entry, so the 5x5
/// neighbourhood is just a strided read of the same map.
///
/// `part_vx`/`part_vy` and the coarse `air_vx`/`air_vy` field are needed by
/// `GOO` (`GOO.cpp:57-58` drags the gel with the air), and the creation queue
/// by `SPNG` (`SPNG.cpp:112-125,181-200` fills every free neighbour with water
/// or steam, which is more than one request per frame), `part_tmp2` by
/// `COAL`'s peak-temperature tracking and `WOOD`'s `BCOL` pyrolysis defaults
/// (`COAL.cpp:73-74`, `WOOD.cpp:56-61`), and `spawn_defaults` by the `H2`
/// fusion burst's `create_part(i, ...)` slot restart (`H2.cpp:108-109`). That
/// brings the pass to 16 storage buffers, the hard kernel ceiling.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn combustion_reaction_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_ctype: &mut [u32],
    part_flags: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_pv: &mut [f32],
    air_vx: &[f32],
    air_vy: &[f32],
    create_queue: &mut [Atomic<u32>],
    elem_words: &[u32],
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
            let ty = part_ty[id];
            if ty != PT_NONE {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let temp = part_temp[id];
                // Per-particle, per-frame roll shared by the element updates.
                let prand = hash_u32(frame_seed ^ mix_index(idx as u32, 2_654_435_761u32));

                if ty == PT_FIRE || ty == PT_PLSM || ty == PT_LAVA {
                    // Fire and plasma are exothermic reactions, so they may
                    // release heat into their neighbourhood for free. Lava is
                    // merely hot material: it must transfer heat through
                    // ordinary conduction (`heat_conduct`) and `hot_air` only.
                    // Letting it radiate into 24 neighbours every frame while
                    // losing nothing turns any lava pool into an infinite heat
                    // source (measured: total system heat doubled in 300 ticks,
                    // so a lava lake could never freeze).
                    let exothermic = ty == PT_FIRE || ty == PT_PLSM;

                    if ty == PT_LAVA {
                        // `FIRE.cpp:112-158`: the melt's own `ctype` decides
                        // what a pressurised lava pocket crystallises into.
                        let pres = air_pv[cidx];
                        let ctype_now = part_ctype[id];
                        let lroll = hash_u32(prand ^ 0x4C41_5641u32);
                        if ctype_now == PT_ROCK {
                            if pres <= -9.0f32 {
                                // `FIRE.cpp:117-122`: vacuum boils the rock
                                // back into stone.
                                part_ctype[id] = PT_STNE;
                            } else if pres >= 25.0f32 && chance(lroll, 1u32, 12_500u32) {
                                let mut new_ctype = ctype_now;
                                if pres <= 50.0f32 {
                                    // `FIRE.cpp:128-137`: gneiss or concrete.
                                    if chance(lroll ^ 0x01u32, 1u32, 2u32) {
                                        new_ctype = PT_BRMT;
                                    } else {
                                        new_ctype = PT_CNCT;
                                    }
                                } else if pres <= 75.0f32 {
                                    // `FIRE.cpp:138-147`: gold near the upper
                                    // edge of the band, quartz otherwise.
                                    if pres >= 73.0f32 || chance(lroll ^ 0x02u32, 1u32, 8u32) {
                                        new_ctype = PT_GOLD;
                                    } else {
                                        new_ctype = PT_QRTZ;
                                    }
                                } else if pres <= 100.0f32 && temp >= 5000.0f32 {
                                    // `FIRE.cpp:148-155`: titanium or iron.
                                    if chance(lroll ^ 0x03u32, 1u32, 5u32) {
                                        new_ctype = PT_TTAN;
                                    } else {
                                        new_ctype = PT_IRON;
                                    }
                                } else if temp >= 5000.0f32 && chance(lroll ^ 0x04u32, 1u32, 5u32) {
                                    // `FIRE.cpp:156-166`: uranium, plutonium or
                                    // tungsten, each with its own inner roll.
                                    if chance(lroll ^ 0x05u32, 1u32, 5u32) {
                                        new_ctype = PT_URAN;
                                    } else if chance(lroll ^ 0x06u32, 1u32, 5u32) {
                                        new_ctype = PT_PLUT;
                                    } else {
                                        new_ctype = PT_TUNG;
                                    }
                                }
                                part_ctype[id] = new_ctype;
                            }
                        } else if ctype_now == PT_GOLD
                            && pres < -200.0f32
                            && temp
                                > f32::reinterpret(elem_word(elem_words, PT_PTNM, WORD_HIGH_TEMP))
                            && chance(lroll ^ 0x07u32, 1u32, 20_000u32)
                        {
                            // `FIRE.cpp:159-165`: platinum precipitates out of
                            // a strongly expanding gold melt.
                            part_ctype[id] = PT_PTNM;
                            air_pv[cidx] += 2.0f32;
                        } else if (ctype_now == PT_STNE || ctype_now == PT_NONE)
                            && pres >= 30.0f32
                            && (temp
                                > f32::reinterpret(elem_word(elem_words, PT_ROCK, WORD_HIGH_TEMP))
                                || pres
                                    < f32::reinterpret(elem_word(
                                        elem_words,
                                        PT_ROCK,
                                        WORD_HIGH_PRESSURE,
                                    )))
                        {
                            // `FIRE.cpp:166-172`: compressed lava freezes into
                            // rock; `tmp2` keeps the colour noise seed.
                            part_tmp2[id] = (hash_u32(lroll ^ 0x08u32) % 11u32) as i32;
                            part_ctype[id] = PT_ROCK;
                        }
                    }
                    // `Simulation.cpp:2278-2285`: the burner's `surround_space`
                    // is the number of empty pixels among its eight neighbours,
                    // computed once before the element update. The ignition
                    // gate below uses it so a flame buried in matter cannot set
                    // its neighbours alight unless they are explosive.
                    let mut surround_space = 0u32;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32)
                            && pmap_at(pmap, xres, yres, px + dx, py + dy) == 0u32
                        {
                            surround_space += 1u32;
                        }
                    }
                    // 5x5 neighbourhood: heat everything, ignite what is flammable,
                    // and pressurise the cell when the neighbour is explosive.
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if nty != PT_NONE {
                                    let ntemp = part_temp[nid];
                                    // `FIRE.cpp:181-270`: the lava suite. A
                                    // killed or converted neighbour skips the
                                    // generic ignition of the same step, like
                                    // the reference's `continue`.
                                    let mut skip_ignite = false;
                                    let nroll = hash_u32(prand ^ mix_index(n, 0x9E37_79B9u32));

                                    // `FIRE.cpp:188-216`: coal next to a flame
                                    // loses its long burn timer; lava instead
                                    // refines itself and eats the coal.
                                    if nty == PT_COAL || nty == PT_BCOL {
                                        if ty == PT_FIRE || ty == PT_PLSM {
                                            if part_life[nid] > 100i32
                                                && chance(nroll, 1u32, 500u32)
                                            {
                                                part_life[nid] = 99i32;
                                            }
                                        } else if ty == PT_LAVA {
                                            if part_ctype[id] == PT_IRON
                                                && chance(nroll ^ 0x10u32, 1u32, 500u32)
                                            {
                                                part_ctype[id] = PT_METL;
                                                part_ty[nid] = PT_NONE;
                                                skip_ignite = true;
                                            }
                                            if !skip_ignite
                                                && (part_ctype[id] == PT_STNE
                                                    || part_ctype[id] == PT_NONE)
                                                && chance(nroll ^ 0x11u32, 1u32, 60u32)
                                            {
                                                part_ctype[id] = PT_SLCN;
                                                part_ty[nid] = PT_NONE;
                                                skip_ignite = true;
                                            }
                                        }
                                    }

                                    if ty == PT_LAVA && !skip_ignite {
                                        // `FIRE.cpp:218-270`: the pair
                                        // reactions of a lava pixel. Every
                                        // branch reads the live `ctype` of the
                                        // subject, because an earlier
                                        // neighbour may already have changed
                                        // it.
                                        let lc = part_ctype[id];
                                        if lc == PT_QRTZ
                                            && nty == PT_LAVA
                                            && part_ctype[nid] == PT_CLST
                                        {
                                            // `FIRE.cpp:219-229`: clay melts
                                            // into ceramic under a hot enough
                                            // quartz melt.
                                            let lpres = max(air_pv[cidx] * 10.0f32, 0.0f32);
                                            let crmc = f32::reinterpret(elem_word(
                                                elem_words,
                                                PT_CRMC,
                                                WORD_HIGH_TEMP,
                                            ));
                                            if temp >= lpres + crmc + 50.0f32 {
                                                part_ctype[id] = PT_CRMC;
                                                part_ctype[nid] = PT_CRMC;
                                            }
                                        } else if nty == PT_O2 && lc == PT_SLCN {
                                            // `FIRE.cpp:230-250`: oxidised
                                            // silicon becomes sand, clay, a
                                            // quartz crystal or stone.
                                            let pick = nroll % 3u32;
                                            if pick == 0u32 {
                                                part_ctype[id] = PT_SAND;
                                            } else if pick == 1u32 {
                                                let pq = f32::reinterpret(elem_word(
                                                    elem_words,
                                                    PT_PQRT,
                                                    WORD_HIGH_TEMP,
                                                ));
                                                if temp >= pq * 3.0f32 {
                                                    part_ctype[id] = PT_PQRT;
                                                } else {
                                                    part_ctype[id] = PT_CLST;
                                                }
                                            } else {
                                                part_ctype[id] = PT_STNE;
                                            }
                                            part_tmp[id] = 0i32;
                                            part_ty[nid] = PT_NONE;
                                            skip_ignite = true;
                                        } else if nty == PT_LAVA
                                            && (part_ctype[nid] == PT_METL
                                                || part_ctype[nid] == PT_BMTL)
                                            && lc == PT_SLCN
                                        {
                                            // `FIRE.cpp:251-257`: silicon
                                            // dopes into the n- and p-doped
                                            // pair.
                                            part_tmp[id] = 0i32;
                                            part_ctype[id] = PT_NSCN;
                                            part_ctype[nid] = PT_PSCN;
                                        } else if lc == PT_SLCN
                                            && nty == PT_LAVA
                                            && part_ctype[nid] == PT_SALT
                                        {
                                            // `FIRE.cpp:258-268`: lithium out
                                            // of a hot salt/silicon melt.
                                            let lith = f32::reinterpret(elem_word(
                                                elem_words,
                                                PT_LITH,
                                                WORD_HIGH_TEMP,
                                            ));
                                            if temp > lith && chance(nroll ^ 0x12u32, 1u32, 1000u32)
                                            {
                                                part_ctype[id] = PT_LITH;
                                                part_tmp[id] = 0i32;
                                                part_tmp2[id] = 0i32;
                                                part_life[id] = 0i32;
                                                part_ty[nid] = PT_NONE;
                                                skip_ignite = true;
                                            }
                                        } else if nty == PT_HEAC && lc == PT_HEAC {
                                            // `FIRE.cpp:269-276`: hot molten
                                            // heat conductor joins the lava.
                                            let heac = f32::reinterpret(elem_word(
                                                elem_words,
                                                PT_HEAC,
                                                WORD_HIGH_TEMP,
                                            ));
                                            if part_temp[nid] > heac {
                                                // `part_change_type`: fields
                                                // survive.
                                                part_ty[nid] = PT_LAVA;
                                                part_ctype[nid] = PT_HEAC;
                                            }
                                        } else if lc == PT_ROCK
                                            && nty == PT_LAVA
                                            && part_ctype[nid] == PT_GOLD
                                            && part_tmp[nid] == 0i32
                                            && air_pv[cidx] >= 50.0f32
                                            && chance(nroll ^ 0x13u32, 1u32, 10_000u32)
                                        {
                                            // `FIRE.cpp:277-284`: gold veins
                                            // grow vertically.
                                            part_ctype[id] = PT_GOLD;
                                            if iabs(dx) > 1i32 {
                                                part_tmp[id] = 1i32;
                                            }
                                        } else if lc == PT_SALT
                                            && nty == PT_GLAS
                                            && part_life[nid] < 234i32 * 120i32
                                        {
                                            // `FIRE.cpp:285-289`: molten salt
                                            // slowly toughens glass.
                                            part_life[nid] += 1i32;
                                        }
                                    }

                                    if !skip_ignite {
                                        if exothermic && temp > ntemp {
                                            let mut heated = ntemp + (temp - ntemp) * 0.05f32;
                                            if heated > 9999.0f32 {
                                                heated = 9999.0f32;
                                            }
                                            part_temp[nid] = heated;
                                        }

                                        let nflam =
                                            elem_word(elem_words, nty, WORD_FLAMMABLE) as i32;
                                        let nexp = (elem_word(elem_words, nty, WORD_EXPLOSIVE)
                                            as i32)
                                            != 0i32;
                                        // `FIRE.cpp:304-319`: a flame only lights a
                                        // neighbour when it has open space to burn
                                        // in (`surround_space`), or the neighbour is
                                        // explosive and carries its own oxygen; a
                                        // wet sponge is exempt, like `SPNG.cpp`'s
                                        // life check.
                                        let mut ignitable =
                                            nflam > 0i32 && (surround_space > 0u32 || nexp);
                                        if nty == PT_SPNG && part_life[nid] > 0i32 {
                                            ignitable = false;
                                        }
                                        if ignitable {
                                            // The roll reads the *target's* coarse
                                            // cell: `FIRE.cpp:307` uses
                                            // `pv[(y+ry)/CELL][(x+rx)/CELL]`, and the
                                            // chance argument is truncated exactly
                                            // like `int(Flammable + pv*10)`.
                                            let ncidx = (((py + dy) as u32 / CELL) * xcells
                                                + ((px + dx) as u32 / CELL))
                                                as usize;
                                            let pv = air_pv[ncidx];
                                            let num = (nflam as f32 + pv * 10.0f32) as i32;
                                            let mut hit = false;
                                            if num >= 1000i32 {
                                                hit = true;
                                            } else if num > 0i32 {
                                                hit = chance(
                                                    frame_seed
                                                        ^ ((nid as u32) * 2_654_435_761u32)
                                                        ^ n,
                                                    num as u32,
                                                    1000u32,
                                                );
                                            }
                                            if hit {
                                                part_ty[nid] = PT_FIRE;
                                                // `FIRE.cpp:312-316`: the ignition
                                                // range (180..259), the `FIRE`
                                                // default temperature plus half the
                                                // target's flammability, and a clean
                                                // `tmp`/`ctype`.
                                                part_life[nid] = 180i32
                                                    + ((hash_u32((nid as u32) ^ frame_seed) % 80u32)
                                                        as i32);
                                                part_temp[nid] = clampf(
                                                    default_temp_of(elem_words, PT_FIRE)
                                                        + (nflam / 2i32) as f32,
                                                    MIN_TEMP,
                                                    MAX_TEMP,
                                                );
                                                part_tmp[nid] = 0i32;
                                                part_ctype[nid] = 0u32;
                                                if nexp {
                                                    air_pv[cidx] += 0.25f32;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if ty == PT_H2 || ty == PT_O2 {
                    // `H2.cpp:50-146`, `O2.cpp:54-124`:
                    // A flame, a plasma or hot lava in the 5x5 neighbourhood
                    // detonates hydrogen / oxygen. The flame is tagged with bit
                    // 1 (H2) or bit 2 (O2) so a stoichiometric mix
                    // (tmp & 3 == 3) condenses into water vapour later in
                    // `life.rs`.
                    //
                    // `H2.cpp:70-74`: above 45 pressure the hydrogen is packed
                    // too densely for the flame reactions, so no neighbour can
                    // set it off at all. (`O2.cpp` has no such gate.)
                    let pv = air_pv[cidx];
                    let ignite_allowed = ty != PT_H2 || pv <= 45.0f32;
                    let mut ignited = false;
                    let mut desl_switch = false;
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        // `H2.cpp:63-69`: pressurised hydrogen next to diesel
                        // swaps into oil + water and the update returns without
                        // any flame chemistry.
                        // `H2.cpp:77-97` returns on the first reaction;
                        // `O2.cpp:54-85` keeps walking the 5x5, so every
                        // burning `FIRE` neighbour gets its own heat bump.
                        if (dx != 0i32 || dy != 0i32) && !desl_switch {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if ty == PT_H2 && pv > 8.0f32 && nty == PT_DESL {
                                    // `part_change_type` keeps the fields, so
                                    // only the two types change.
                                    part_ty[nid] = PT_WATR;
                                    part_ty[id] = PT_OIL;
                                    desl_switch = true;
                                } else if ignite_allowed && (!ignited || ty == PT_O2) {
                                    let mut reaction = false;
                                    if nty == PT_FIRE {
                                        reaction = true;
                                        if ty == PT_H2 {
                                            // `H2.cpp:77-89`: `tmp |= 1` on the
                                            // fire and on the gas, and a fire
                                            // that already carries the oxygen
                                            // bit locks to 3473 K (2473.15 K
                                            // otherwise).
                                            part_tmp[nid] |= 1i32;
                                            if (part_tmp[nid] & 2i32) != 0i32 {
                                                part_temp[nid] = 3473.0f32;
                                            } else {
                                                part_temp[nid] = 2473.15f32;
                                            }
                                        } else {
                                            // `O2.cpp:63-74`: the fire climbs
                                            // by 0..99 K, or locks to 3473 K
                                            // once the hydrogen bit is set.
                                            let bump = (hash_u32(frame_seed ^ (id as u32) ^ n)
                                                % 100u32)
                                                as f32;
                                            part_temp[nid] += bump;
                                            if (part_tmp[nid] & 1i32) != 0i32 {
                                                part_temp[nid] = 3473.0f32;
                                            }
                                            part_tmp[nid] |= 2i32;
                                        }
                                    } else if nty == PT_PLSM && (part_tmp[nid] & 4i32) == 0i32 {
                                        // `tmp & 4` marks the plasma thrown
                                        // out by a hydrogen fusion, which does
                                        // not light the gas (`H2.cpp:90`).
                                        reaction = true;
                                    } else if nty == PT_LAVA && part_ctype[nid] != PT_BMTL {
                                        // `H2.cpp:90`: lava ignites both
                                        // gases, but molten tungsten is too
                                        // inert.
                                        reaction = true;
                                    }
                                    if reaction {
                                        ignited = true;
                                        if ty == PT_H2 {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if ignited {
                        let delta_t = (hash_u32(frame_seed ^ (id as u32)) % 100u32) as f32;
                        let mut mark = 2i32;
                        if ty == PT_H2 {
                            mark = 1i32;
                        }
                        // `create_part(i, x, y, PT_FIRE)` restarts the slot
                        // from `FIRE`'s defaults plus its `Create` hook (life
                        // 120..169), then the update adds the 0..99 K roll and
                        // the gas-mix bit. The old gas's temperature, life and
                        // velocity are not carried over.
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
                            PT_FIRE,
                        );
                        part_life[id] =
                            120i32 + ((hash_u32((id as u32) ^ frame_seed) % 50u32) as i32);
                        part_temp[id] += delta_t;
                        part_tmp[id] = mark;
                        part_tmp2[id] = 0i32;
                    } else if ty == PT_H2 && !desl_switch {
                        // `H2.cpp:102-142`: above 2273.15 K and 50 pressure
                        // units a hot, dense hydrogen pocket fuses one time in
                        // five: the atom becomes `NBLE` and throws off a
                        // neutron, a photon and (one burst in ten) an
                        // electron, plus a `tmp |= 4` plasma in a free
                        // neighbour. The burst is the same
                        // `create_part(-3, x, y, ...)` sequence the `NBLE`
                        // ionisation runs (`nuclear/decay.rs`), so the
                        // same-pixel products are queued with
                        // `CR_NO_OCCUPANCY`.
                        if temp > 2273.15f32 && pv > 50.0f32 && chance(prand, 1u32, 5u32) {
                            let burst_temp = temp;
                            // `H2.cpp:108-109`: `create_part(i, x, y, PT_NBLE)`
                            // restarts the slot from the `NBLE` defaults, then
                            // `tmp = 0x1` marks the nucleus.
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
                                PT_NBLE,
                            );
                            part_tmp[id] = 1i32;
                            part_tmp2[id] = 0i32;
                            // `H2.cpp:112-114`: `create_part(-3, x, y, PT_NEUT)`
                            // keeps the pre-fusion temperature.
                            request_create(
                                create_queue,
                                PT_NEUT,
                                px,
                                py,
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
                            if chance(hash_u32(prand ^ 0x9E37_79B9u32), 1u32, 10u32) {
                                // `H2.cpp:115-121`: the one-in-ten electron.
                                request_create(
                                    create_queue,
                                    PT_ELEC,
                                    px,
                                    py,
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
                            }
                            // `H2.cpp:122-128`: the photon carries the burst
                            // temperature, the fixed wavelength and `tmp = 1`.
                            request_create(
                                create_queue,
                                PT_PHOT,
                                px,
                                py,
                                CR_NO_OCCUPANCY | CR_SET_CTYPE | CR_SET_TEMP | CR_SET_TMP,
                                0i32,
                                0x7C_0000u32,
                                burst_temp,
                                1i32,
                                0i32,
                                0i32,
                                0i32,
                                0u32,
                                0u32,
                                0.0f32,
                                0.0f32,
                            );
                            // `H2.cpp:129-139`: a plasma lands in a random
                            // neighbour, gated by `can_move[PT_PLSM][rt] ||
                            // rt == PT_H2`. The device table only admits empty
                            // pixels (every `pmap` element weighs at least as
                            // much as the weight-1 plasma), plus the explicit
                            // hydrogen case; the `(0, 0)` offset hits the pixel
                            // this burst already turned into `NBLE`, which the
                            // port skips because it keeps one matter particle
                            // per pixel.
                            let rx = px + (hash_u32(prand ^ 0x85EB_CA6Bu32) % 3u32) as i32 - 1i32;
                            let ry = py + (hash_u32(prand ^ 0xC2B2_AE35u32) % 3u32) as i32 - 1i32;
                            let rt = pmap_at(pmap, xres, yres, rx, ry) & PMAP_TYPE;
                            if (rx != px || ry != py) && (rt == PT_NONE || rt == PT_H2) {
                                request_create(
                                    create_queue,
                                    PT_PLSM,
                                    rx,
                                    ry,
                                    CR_NO_OCCUPANCY | CR_SET_TEMP | CR_SET_TMP,
                                    0i32,
                                    0u32,
                                    burst_temp,
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
                            // `H2.cpp:140-141`: the nucleus heats by
                            // 750..1249 K and the cell gains 30 pressure.
                            part_temp[id] = burst_temp
                                + 750.0f32
                                + (hash_u32(prand ^ 0x27D4_EB2Fu32) % 500u32) as f32;
                            air_pv[cidx] += 30.0f32;
                        }
                    }
                } else if ty == PT_GOO {
                    // `GOO.cpp:51-60` — pressure starts the countdown that
                    // makes goo deform and finally disappear
                    // (`PROP_LIFE_KILL_DEC` turns the last frame into a kill),
                    // and while the countdown runs the gel is dragged along by
                    // the air at `ADVECTION = 0.1` of the coarse cell velocity
                    // (`GOO.cpp:49,57-58`).
                    //
                    // The port's velocity pass runs after this one and applies
                    // the element's `Loss` to the whole velocity, where TPT
                    // damps first and adds the air drag afterwards; the added
                    // term is therefore damped once per frame here. That is the
                    // same order every update-driven velocity change in this
                    // port follows, so the coupling is kept at the reference's
                    // `0.1` factor instead of compensating for the pass order.
                    let mut goo_life = part_life[id];
                    if goo_life == 0i32 && air_pv[cidx] > 1.0f32 {
                        goo_life = 300i32 + (prand % 80u32) as i32;
                    }
                    if goo_life != 0i32 {
                        part_vx[id] += 0.1f32 * air_vx[cidx];
                        part_vy[id] += 0.1f32 * air_vy[cidx];
                    }
                    part_life[id] = goo_life;
                } else if ty == PT_YEST {
                    // `YEST.cpp:48-70` — dead yeast turns live yeast, and warm
                    // yeast (303..317 K) buds one new cell per frame. The bud
                    // runs on the conversion frame too (`YEST.cpp:67-69` is
                    // after the conversion loop), so the growth test is not
                    // gated on the conversion having happened.
                    let mut dead = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !dead {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32
                                && part_ty[pmap_slot(nv)] == PT_DYST
                                && prand.is_multiple_of(6u32)
                            {
                                part_ty[id] = PT_DYST;
                                dead = true;
                            }
                        }
                    }
                    if temp > 303.0f32 && temp < 317.0f32 {
                        if dead {
                            // The slot carries `DYST` by now, so asking for a
                            // copy of the particle's own element would bud
                            // dead yeast; `create_part(-1, ..., PT_YEST)` names
                            // the live type, which `ctype` carries for the
                            // clone request.
                            part_ctype[id] = PT_YEST;
                            part_flags[id] |= FLAG_SPAWN_CLONE;
                        } else {
                            part_flags[id] |= FLAG_SPAWN_SELF;
                        }
                    }
                } else if ty == PT_COAL || ty == PT_BCOL {
                    // `COAL.cpp:54-74` — `BCOL.cpp` reuses
                    // `Element_COAL_update` verbatim, so broken coal smoulders
                    // down on the same `life` schedule. Only the `COAL`-specific
                    // `tmp` pressure memory and the `COAL -> BCOL` crumble are
                    // skipped for `BCOL` (`COAL.cpp:61`).
                    //
                    // The slow burn drops a `FIRE` into a free neighbour
                    // (`create_part(-1, x + rand, y + rand)`); the request is
                    // raised through `ctype` and lands on the next frame
                    // boundary, like every other spawn this pass cannot
                    // allocate inline.
                    let coal_life = part_life[id];
                    if coal_life <= 0i32 {
                        // `create_part(i, x, y, PT_FIRE)` restarts the slot
                        // from `FIRE`'s defaults and runs its `Create` hook:
                        // life 120..169, temperature 695.15 K, and
                        // `tmp`/`tmp2`/`ctype`/velocity cleared. The old code
                        // left the coal's four-frame life on the fire, which
                        // burned out almost immediately.
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
                            PT_FIRE,
                        );
                        part_life[id] =
                            120i32 + ((hash_u32((id as u32) ^ frame_seed) % 50u32) as i32);
                        part_tmp2[id] = 0i32;
                    } else {
                        if coal_life < 100i32 {
                            part_life[id] = coal_life - 1i32;
                            part_ctype[id] = PT_FIRE;
                            part_flags[id] |= FLAG_SPAWN_CLONE;
                        }
                        let mut crumbled = false;
                        if ty == PT_COAL {
                            let coal_tmp = part_tmp[id];
                            if air_pv[cidx] > 4.3f32 && coal_tmp > 40i32 {
                                part_tmp[id] = 39i32;
                            } else if coal_tmp < 40i32 && coal_tmp > 0i32 {
                                part_tmp[id] = coal_tmp - 1i32;
                            } else if coal_tmp <= 0i32 {
                                part_ty[id] = PT_BCOL;
                                crumbled = true;
                            }
                        }
                        // `COAL.cpp:73-74`: `tmp2` tracks the peak
                        // temperature for the glow; the update is skipped on
                        // the frames the function returns early (`life <= 0`,
                        // the `COAL -> BCOL` crumble).
                        if !crumbled && part_temp[id] > part_tmp2[id] as f32 {
                            part_tmp2[id] = part_temp[id] as i32;
                        }
                    }
                } else if ty == PT_SPNG {
                    // `SPNG.cpp:update` — a sponge soaks up neighbouring water
                    // while it is below capacity, in the pressure band it can
                    // hold, and below boiling.
                    let pv = air_pv[cidx];
                    let mut wet = part_life[id];
                    // TPT keeps the absorption band closed: `|pv| <= 3`.
                    let mut absorbing = wet < SPONGE_CAPACITY && temp <= 374.0f32;
                    if pv > 3.0f32 {
                        absorbing = false;
                    }
                    if pv < -3.0f32 {
                        absorbing = false;
                    }
                    if absorbing {
                        // TPT: absorbChanceDenom = life*10000/50 + 500.
                        let denom = wet * 10_000i32 / SPONGE_CAPACITY + 500i32;
                        let d = denom as u32;
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if dx != 0i32 || dy != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 && wet < SPONGE_CAPACITY {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    if nty == PT_WATR || nty == PT_DSTW || nty == PT_FRZW {
                                        if prand % d < 500u32 {
                                            wet += 1i32;
                                            part_ty[nid] = PT_NONE;
                                        }
                                    } else if nty == PT_SLTW {
                                        if prand % d < 50u32 {
                                            wet += 1i32;
                                            if !prand.is_multiple_of(4u32) {
                                                part_ty[nid] = PT_NONE;
                                            } else {
                                                part_ty[nid] = PT_SALT;
                                            }
                                        }
                                    } else if nty == PT_CBNW {
                                        // `SPNG.cpp:84-91`: soda water releases CO2 gas.
                                        if prand % d < 100u32 {
                                            wet += 1i32;
                                            part_ty[nid] = PT_CO2;
                                        }
                                    } else if nty == PT_PSTE {
                                        // `SPNG.cpp:92-98`: sponge absorbs liquid from paste, leaving CLST.
                                        if prand % d < 20u32 {
                                            wet += 1i32;
                                            part_ty[nid] = PT_CLST;
                                        }
                                    }
                                }
                            }
                        }
                        // Pressure diffusion with one random sponge in the 5x5.
                        let ox = (prand % 5u32) as i32 - 2i32;
                        let oy = ((prand >> 8u32) % 5u32) as i32 - 2i32;
                        if ox != 0i32 || oy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + ox, py + oy);
                            if nv != 0u32 && wet > 0i32 {
                                let nid = pmap_slot(nv);
                                let other = part_life[nid];
                                if part_ty[nid] == PT_SPNG && other < wet {
                                    let half = if wet - other == 1i32 {
                                        1i32
                                    } else {
                                        (wet - other) / 2i32
                                    };
                                    part_life[nid] = other + half;
                                    wet -= half;
                                }
                            }
                        }
                    } else if wet >= 1i32 {
                        // `SPNG.cpp:112-127`: outside the soak regime the
                        // sponge wrings out one `WATR` into **every** free
                        // neighbour, not one per frame. TPT's
                        // `create_part(-1, x+rx, y+ry, PT_WATR)` refuses an
                        // occupied pixel and decrements `life` only when it
                        // succeeds, so the free-pixel test and the store decide
                        // how many requests are made; a full queue drops the
                        // rest, exactly like a failed `create_part`.
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            let nx = px + dx;
                            let ny = py + dy;
                            if (dx != 0i32 || dy != 0i32)
                                && wet >= 1i32
                                && nx >= 0i32
                                && ny >= 0i32
                                && nx < xres as i32
                                && ny < yres as i32
                                && pmap_at(pmap, xres, yres, nx, ny) == 0u32
                            {
                                request_create(
                                    create_queue,
                                    PT_WATR,
                                    nx,
                                    ny,
                                    0u32,
                                    0i32,
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
                                wet -= 1i32;
                            }
                        }
                    }

                    // `SPNG.cpp:153-206`: fire next to a wet sponge is damped;
                    // a sponge with fire on it, or one at boiling point, boils
                    // its store off into `WTRV` — again one pixel per free
                    // neighbour, each taking the sponge's current temperature
                    // and costing 20 K.
                    let mut fires = 0i32;
                    if wet > 0i32 {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if dx != 0i32 || dy != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    if part_ty[nid] == PT_FIRE {
                                        fires += 1i32;
                                        let fl = part_life[nid];
                                        if fl > 60i32 {
                                            part_life[nid] = fl - fl / 60i32;
                                        } else if fl > 2i32 {
                                            part_life[nid] = fl - 1i32;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if fires > 0i32 && wet > 3i32 {
                        wet -= wet / 3i32;
                    }
                    if fires > 1i32 {
                        fires /= 2i32;
                    }
                    let mut spng_temp = temp;
                    if fires > 0i32 || temp >= 374.0f32 {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            let nx = px + dx;
                            let ny = py + dy;
                            if (dx != 0i32 || dy != 0i32)
                                && wet >= 1i32
                                && nx >= 0i32
                                && ny >= 0i32
                                && nx < xres as i32
                                && ny < yres as i32
                                && pmap_at(pmap, xres, yres, nx, ny) == 0u32
                            {
                                // `SPNG.cpp:193-199`: the child takes the
                                // sponge's temperature before the 20 K drop,
                                // and each pixel spends one unit of the store
                                // and one unit of the fire count.
                                request_create(
                                    create_queue,
                                    PT_WTRV,
                                    nx,
                                    ny,
                                    CR_SET_TEMP,
                                    0i32,
                                    0u32,
                                    spng_temp,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0i32,
                                    0u32,
                                    0u32,
                                    0.0f32,
                                    0.0f32,
                                );
                                fires -= 1i32;
                                wet -= 1i32;
                                spng_temp = max(spng_temp - 20.0f32, MIN_TEMP);
                            }
                        }
                    }
                    if fires > 0i32 {
                        if wet > fires {
                            wet -= fires;
                        } else {
                            wet = 0i32;
                        }
                    }
                    if spng_temp != temp {
                        part_temp[id] = spng_temp;
                    }
                    if wet != part_life[id] {
                        part_life[id] = wet;
                    }
                } else if ty == PT_WTRV {
                    // `WTRV.cpp:50-76`: steam ignites rubber it touches, and
                    // steam left over from burning water sheds heat. The port
                    // has no `legacy_enable` mode, so that guard is always
                    // satisfied (same treatment as `URAN` self-heating).
                    let temp = part_temp[id];
                    let mut ignited = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !ignited {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if (nty == PT_RBDM || nty == PT_LRBD)
                                    && temp > STEAM_IGNITE_K
                                    && chance(prand ^ n, 1u32, 100u32)
                                {
                                    //@ WTRV + RBDM/LRBD -> FIRE + RBDM/LRBD
                                    part_ty[id] = PT_FIRE;
                                    part_life[id] = 4i32;
                                    part_ctype[id] = PT_WATR;
                                    ignited = true;
                                }
                            }
                        }
                    }
                    if !ignited && temp > STEAM_COOL_K && part_ctype[id] == PT_FIRE {
                        part_temp[id] = temp - temp / 1000.0f32;
                    }
                } else if ty == PT_FOG {
                    // `FOG.cpp:update` — fog freezes onto solid matter into
                    // rime, a spark charges it, and GAS acidifies it.
                    let mut fog_life = part_life[id];
                    let mut rime = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !rime {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let solid = (elem_word(elem_words, nty, WORD_PROPERTIES)
                                    & TYPE_SOLID)
                                    != 0u32;
                                if solid
                                    && fog_life == 0i32
                                    && prand.is_multiple_of(10u32)
                                    && nty != PT_CLNE
                                    && nty != PT_PCLN
                                    && nty != PT_BCLN
                                    && nty != PT_PBCN
                                {
                                    part_ty[id] = PT_RIME;
                                    rime = true;
                                } else if nty == PT_SPRK {
                                    fog_life += (prand % 20u32) as i32;
                                } else if nty == PT_GAS && part_tmp[id] < 10i32 {
                                    // `FOG.cpp:74-79`: the acid dissolves
                                    // distilled-water fog without adding to its
                                    // acidity counter.
                                    part_ty[nid] = PT_NONE;
                                    if part_ctype[id] == PT_DSTW {
                                        part_ctype[id] = PT_NONE;
                                    } else {
                                        part_tmp[id] += 1i32;
                                    }
                                }
                            }
                        }
                    }
                    if !rime {
                        part_life[id] = fog_life;
                    }
                } else if ty == PT_RIME {
                    // `RIME.cpp:update` — a spark melts rime into fog, a
                    // neighbouring fog seed takes over, and GAS acidifies it.
                    let mut rime_life = part_life[id];
                    let mut melted = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !melted {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if nty == PT_SPRK {
                                    part_ty[id] = PT_FOG;
                                    rime_life = 60i32 + (prand % 60u32) as i32;
                                    melted = true;
                                } else if nty == PT_GAS && part_tmp[id] < 10i32 {
                                    // `RIME.cpp:69-75`: same distilled-water rule.
                                    part_ty[nid] = PT_NONE;
                                    if part_ctype[id] == PT_DSTW {
                                        part_ctype[id] = PT_NONE;
                                    } else {
                                        part_tmp[id] += 1i32;
                                    }
                                } else if nty == PT_FOG && part_life[nid] > 0i32 {
                                    part_ty[id] = PT_FOG;
                                    rime_life = part_life[nid];
                                    melted = true;
                                }
                            }
                        }
                    }
                    if melted {
                        part_life[id] = rime_life;
                    }
                } else if ty == PT_WOOD {
                    // `WOOD.cpp:51-66`: peak temperature tracking in `tmp`.
                    // Pyrolysis: vacuum/negative pressure (pv <= -10) and high temp (> 773 K)
                    // converts wood to charcoal (BCOL).
                    let mut tmp = part_tmp[id];
                    if temp > 450.0f32 && temp > (tmp as f32) {
                        tmp = temp as i32;
                        part_tmp[id] = tmp;
                    }
                    if temp > 773.0f32 && air_pv[cidx] <= -10.0f32 {
                        // `WOOD.cpp:56-61` uses `create_part(i, x, y, PT_BCOL)`,
                        // which resets the slot to `BCOL`'s `DefaultProperties`
                        // (life 110, tmp/tmp2/ctype/velocity 0) before the wood
                        // temperature is put back. Leaving the wood's
                        // peak-temperature `tmp` behind would make the new
                        // charcoal look pre-heated.
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
                            PT_BCOL,
                        );
                        part_temp[id] = temp;
                        part_tmp2[id] = 0i32;
                    }
                } else if ty == PT_CO2 {
                    // `CO2.cpp:48-115`: heavy gas that extinguishes fire, carbonates water,
                    // and breaks down into oxygen and neutron at extreme temperature and pressure.
                    let mut extinguished = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !extinguished {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                if nty == PT_FIRE {
                                    // `CO2.cpp:67-75`: CO2 kills fire; 1 in 30 chance self dies
                                    part_ty[nid] = PT_NONE;
                                    if prand.is_multiple_of(30u32) {
                                        part_ty[id] = PT_NONE;
                                        extinguished = true;
                                    }
                                } else if (nty == PT_WATR || nty == PT_DSTW)
                                    && prand.is_multiple_of(50u32)
                                {
                                    // `CO2.cpp:76-91`: carbonates water into CBNW
                                    part_ty[nid] = PT_CBNW;
                                    part_ty[id] = PT_NONE;
                                    extinguished = true;
                                }
                            }
                        }
                    }
                    if !extinguished
                        && temp > 9773.15f32
                        && air_pv[cidx] > 200.0f32
                        && prand.is_multiple_of(5u32)
                    {
                        // `CO2.cpp:95-115`: CO2 -> O2 + NEUT at MAX_TEMP
                        part_ty[id] = PT_O2;
                        part_temp[id] = 9999.0f32;
                        part_flags[id] |= FLAG_SPAWN_NEUT;
                        air_pv[cidx] += 100.0f32;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `FIRE.cpp:35`: `R_TEMP + 400.0f + 273.15f` with `R_TEMP = 22`.
    const FIRE_DEFAULT_TEMP: f32 = 695.15;
    /// `BCOL.cpp:45`: `DefaultProperties.life = 110`.
    const BCOL_DEFAULT_LIFE: i32 = 110;

    fn clamp_host(v: f32, lo: f32, hi: f32) -> f32 {
        if v < lo {
            lo
        } else if v > hi {
            hi
        } else {
            v
        }
    }

    /// Host mirror of the `COAL` / `BCOL` branch (`COAL.cpp:54-74`).
    ///
    /// Returns `(type, life, tmp, tmp2, temp, ctype, fire_request)`.
    #[allow(clippy::too_many_arguments)]
    fn coal_step(
        ty: u32,
        life: i32,
        tmp: i32,
        tmp2: i32,
        temp: f32,
        pv: f32,
        roll: u32,
    ) -> (u32, i32, i32, i32, f32, u32, bool) {
        let mut out_ty = ty;
        let mut out_life = life;
        let mut out_tmp = tmp;
        let mut out_tmp2 = tmp2;
        let mut out_temp = temp;
        let mut out_ctype = 0u32;
        let mut fire_request = false;
        if out_life <= 0 {
            // `create_part(i, x, y, PT_FIRE)`: `FIRE` defaults plus the
            // `Create` hook, then no temperature override in `COAL.cpp`.
            out_ty = PT_FIRE;
            out_life = 120 + (roll % 50) as i32;
            out_temp = FIRE_DEFAULT_TEMP;
            out_tmp = 0;
            out_ctype = 0;
        } else {
            if out_life < 100 {
                out_life -= 1;
                out_ctype = PT_FIRE;
                fire_request = true;
            }
            let mut crumbled = false;
            if out_ty == PT_COAL {
                if pv > 4.3 && out_tmp > 40 {
                    out_tmp = 39;
                } else if out_tmp < 40 && out_tmp > 0 {
                    out_tmp -= 1;
                } else if out_tmp <= 0 {
                    out_ty = PT_BCOL;
                    crumbled = true;
                }
            }
            if !crumbled && out_temp > out_tmp2 as f32 {
                out_tmp2 = out_temp as i32;
            }
        }
        (
            out_ty,
            out_life,
            out_tmp,
            out_tmp2,
            out_temp,
            out_ctype,
            fire_request,
        )
    }

    #[test]
    fn broken_coal_burns_out_into_a_create_hook_fire() {
        // `BCOL.cpp` reuses `Element_COAL_update`, so broken coal reaches the
        // same burnout branch as coal.
        // `COAL.cpp:54-56`: the burnout test is `life <= 0`; a `life == 1`
        // frame only spends the last point and drops a neighbouring fire.
        let (ty, life, tmp, tmp2, temp, ctype, _) = coal_step(PT_BCOL, 0, 0, 0, 300.0, 0.0, 7);
        assert_eq!(ty, PT_FIRE, "BCOL must burn out into FIRE");
        assert!(
            (120..=169).contains(&life),
            "the fire must take FIRE's Create life, got {life}"
        );
        assert_eq!(temp, FIRE_DEFAULT_TEMP);
        assert_eq!(tmp, 0);
        assert_eq!(ctype, 0);
        assert_eq!(tmp2, 0);
    }

    #[test]
    fn broken_coal_smoulders_without_crumbling() {
        // `COAL.cpp:61`: the `tmp` pressure memory and `COAL -> BCOL` crumble
        // are gated on the type being `COAL`.
        let (ty, life, tmp, _, _, ctype, fire) = coal_step(PT_BCOL, 60, 0, 0, 300.0, 0.0, 1);
        assert_eq!(ty, PT_BCOL, "BCOL must never crumble into itself");
        assert_eq!(life, 59);
        assert_eq!(tmp, 0);
        assert_eq!(ctype, PT_FIRE, "the slow burn asks for a neighbouring fire");
        assert!(fire);
    }

    #[test]
    fn coal_crumbles_into_broken_coal() {
        // `COAL.cpp:67-71`: `tmp` counts down and then the block crumbles.
        let (ty, _, tmp, _, _, _, _) = coal_step(PT_COAL, 110, 1, 0, 300.0, 0.0, 0);
        assert_eq!(ty, PT_COAL);
        assert_eq!(tmp, 0);
        let (ty, _, _, _, _, _, _) = coal_step(PT_COAL, 110, tmp, 0, 300.0, 0.0, 0);
        assert_eq!(ty, PT_BCOL);

        // `COAL.cpp:63-64`: pressure above 4.3 caps the memory at 39.
        let (_, _, tmp, _, _, _, _) = coal_step(PT_COAL, 110, 50, 0, 300.0, 5.0, 0);
        assert_eq!(tmp, 39);
    }

    #[test]
    fn coal_tracks_its_peak_temperature() {
        // `COAL.cpp:73-74`: `tmp2` is the highest temperature seen so far.
        let (_, _, _, tmp2, _, _, _) = coal_step(PT_COAL, 110, 50, 0, 800.0, 0.0, 0);
        assert_eq!(tmp2, 800);
        let (_, _, _, tmp2, _, _, _) = coal_step(PT_COAL, 110, 50, 800, 600.0, 0.0, 0);
        assert_eq!(tmp2, 800, "a colder frame must not lower the peak");
    }

    /// Host mirror of `WOOD.cpp:51-66`.
    fn wood_step(temp: f32, pv: f32, tmp: i32, life: i32, tmp2: i32) -> (u32, i32, i32, i32, f32) {
        let mut ty = PT_WOOD;
        let mut out_tmp = tmp;
        let mut out_life = life;
        let mut out_tmp2 = tmp2;
        if temp > 450.0 && temp > out_tmp as f32 {
            out_tmp = temp as i32;
        }
        if temp > 773.0 && pv <= -10.0 {
            ty = PT_BCOL;
            out_life = BCOL_DEFAULT_LIFE;
            out_tmp = 0;
            out_tmp2 = 0;
        }
        (ty, out_life, out_tmp, out_tmp2, temp)
    }

    #[test]
    fn wood_pyrolysis_resets_the_broken_coal_defaults() {
        // `WOOD.cpp:56-61`: `create_part(i, x, y, PT_BCOL)` resets the slot to
        // BCOL's defaults and the wood temperature is written back afterwards.
        let (ty, life, tmp, tmp2, temp) = wood_step(800.0, -15.0, 700, 5, 650);
        assert_eq!(ty, PT_BCOL);
        assert_eq!(life, BCOL_DEFAULT_LIFE);
        assert_eq!(tmp, 0, "the wood peak-temperature scratch must be cleared");
        assert_eq!(tmp2, 0);
        assert_eq!(temp, 800.0, "the wood temperature survives the conversion");

        // No pyrolysis without the vacuum: the text and its scratch survive.
        let (ty, life, tmp, tmp2, _) = wood_step(800.0, 5.0, 700, 5, 650);
        assert_eq!(ty, PT_WOOD);
        assert_eq!(life, 5);
        assert_eq!(tmp, 800);
        assert_eq!(tmp2, 650);
    }

    /// Host mirror of the `FIRE.cpp:304-319` ignition roll.
    fn fire_ignition_hit(nflam: i32, nexp: bool, surround_space: u32, pv: f32, roll: u32) -> bool {
        if nflam <= 0 || (surround_space == 0 && !nexp) {
            return false;
        }
        let num = (nflam as f32 + pv * 10.0) as i32;
        if num >= 1000 {
            return true;
        }
        if num <= 0 {
            return false;
        }
        (roll % 1000) < num as u32
    }

    #[test]
    fn fire_ignition_gate_needs_space_or_an_explosive() {
        // A flame buried in matter cannot light a plain flammable...
        assert!(!fire_ignition_hit(600, false, 0, 0.0, 0));
        // ...but an explosive carries its own oxygen, so it goes up anyway.
        assert!(fire_ignition_hit(600, true, 0, 0.0, 0));
        // Empty space around the flame also permits ignition.
        assert!(fire_ignition_hit(600, false, 1, 0.0, 0));
        // A non-flammable target never ignites, even when explosive.
        assert!(!fire_ignition_hit(0, true, 8, 0.0, 0));
    }

    #[test]
    fn fire_ignition_rolls_on_the_target_cell_pressure() {
        // `FIRE.cpp:307` reads `pv[(y+ry)/CELL][(x+rx)/CELL]`, the target's
        // cell. At 60 pressure the argument saturates even for a cold roll.
        assert!(!fire_ignition_hit(600, false, 1, 0.0, 999));
        assert!(fire_ignition_hit(600, false, 1, 60.0, 999));
        // Negative pressure can suppress the roll entirely.
        assert!(!fire_ignition_hit(100, false, 1, -20.0, 0));
    }

    #[test]
    fn fire_ignition_temperature_is_the_default_plus_half_flammability() {
        // `FIRE.cpp:315`: `DefaultProperties.temp + (Flammable/2)`, with the
        // integer `Flammable` divided as an integer.
        let temp = clamp_host(
            FIRE_DEFAULT_TEMP + (600i32 / 2i32) as f32,
            MIN_TEMP,
            MAX_TEMP,
        );
        assert_eq!(temp, FIRE_DEFAULT_TEMP + 300.0);
        assert!((temp - 995.15).abs() < 0.01);
        let odd = clamp_host(
            FIRE_DEFAULT_TEMP + (33i32 / 2i32) as f32,
            MIN_TEMP,
            MAX_TEMP,
        );
        assert_eq!(odd, FIRE_DEFAULT_TEMP + 16.0);
        // The old literal was 673.15 (the ignition trigger, not the created
        // fire temperature).
        assert_ne!(temp, 673.15 + 300.0);
    }
}
