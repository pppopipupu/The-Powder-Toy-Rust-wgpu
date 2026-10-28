// Clippy's rewrite for these lints uses methods the device IR does not have
// (`is_multiple_of`, `clamp`, `Range::contains`, `saturating_sub`), and
// collapsing a nested `if` changes what the CubeCL macro expands, so the plain
// operators are deliberate.
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_range_contains)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::{chance, chance1, clampf, clampi};
use crate::reaction_kernel::shared::{
    CR_SET_TEMP, WORD_HARDNESS, WORD_PROPERTIES, elem_word, hash_u32, mix_index, pmap_at,
    pmap_slot, request_create, request_create_plain,
};

/// Word offset of [`GpuElementProps::heat_conduct`]
/// (`SimulationData::IsHeatInsulator`, `SimulationData.cpp:338`).
const WORD_HEAT_CONDUCT: u32 = 15;
const _: () = assert!(
    core::mem::offset_of!(tpt_core::elements::GpuElementProps, heat_conduct)
        == WORD_HEAT_CONDUCT as usize * 4
);

/// `SimulationData::IsHeatInsulator`: `HeatConduct == 0` (`VIBR.cpp:108`).
#[cube]
fn heat_insulator(words: &[u32], ty: u32) -> bool {
    elem_word(words, ty, WORD_HEAT_CONDUCT) == 0u32
}

/// `orbit_rx` (`VIBR.cpp:49`): `{-1, 0, 1, 1, 1, 0, -1, -1}`.
#[cube]
fn vibr_orbit_x(k: i32) -> i32 {
    // The signed differences keep the kernel free of
    // `RangeInclusive::contains`, which CubeCL does not expand.
    let mut rx = 0i32;
    if ((k - 2i32) as u32) < 3u32 {
        rx = 1i32;
    }
    if k == 0i32 || k >= 6i32 {
        rx = -1i32;
    }
    rx
}

/// `orbit_ry` (`VIBR.cpp:50`): `{-1, -1, -1, 0, 1, 1, 1, 0}`.
#[cube]
fn vibr_orbit_y(k: i32) -> i32 {
    let mut ry = 0i32;
    if ((k - 4i32) as u32) < 3u32 {
        ry = 1i32;
    }
    if k <= 2i32 {
        ry = -1i32;
    }
    ry
}

/// The one `sim->rng.gen()` draw of an exploding `VIBR`/`BVBR`
/// (`VIBR.cpp:88`).
///
/// Both releases are derived from this word: the spark takes `r & 7` as its
/// orbit index, the heat hand-off takes the next six bits. The receiver has to
/// reproduce the draw from the source's pixel and slot, so the derivation lives
/// here and is used by both sides.
#[cube]
fn vibr_roll(idx: u32, slot: u32, frame_seed: u32) -> u32 {
    let seed = hash_u32(frame_seed ^ mix_index(idx, 2_654_435_761u32));
    hash_u32(seed ^ mix_index(slot, 0x7F4A_7C15u32))
}

/// The `(rx, ry)` cell `VIBR.cpp:105-106` samples for the heat hand-off.
///
/// `rndstore` is shifted right by 3 before the sample when the spark branch ran
/// (`life < 300`), so the two windows read different bits of the same draw; the
/// sample itself is `rndstore % 7 - 3` on both axes.
#[cube]
fn vibr_heat_pick(r: u32, life: i32) -> (i32, i32) {
    let mut v = r;
    if life < 300i32 {
        v = r >> 3u32;
    }
    ((v % 7u32) as i32 - 3i32, ((v >> 3u32) % 7u32) as i32 - 3i32)
}

/// Store a heat offer in `tmp2`: `-(ry + 3) * 7 - (rx + 3) - 1`.
///
/// The encoding is negative so it cannot be mistaken for the `0..1000`
/// discharge marker `VIBR.cpp:131` parks in the same word.
#[cube]
fn vibr_offer(rx: i32, ry: i32) -> i32 {
    -((ry + 3i32) * 7i32 + (rx + 3i32) + 1i32)
}

/// `RFRG`/`RFGL`, `BASE`, `BREC`, the pressure branch of `SEED` and the `VIBR`
/// detonation, together with the two `VIBR.cpp:87-113` releases.
///
/// Pixel parallel: the pressure cell is a strided read at `(x/CELL, y/CELL)`.
/// The spark and heat releases rewrite a neighbour, so they are resolved by the
/// target: the end of the kernel lets every particle look for the exploding
/// `VIBR`/`BVBR` whose sampled cell is itself and write only its own slot. The
/// heat hand-off spends the source's `tmp` exactly once through a
/// publish/receive/clear pipeline clocked by the source's `life` (see the
/// `PT_VIBR` branch).
#[cube(launch_unchecked)]
pub(super) fn special_air_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [Atomic<i32>],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_temp: &mut [f32],
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
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 2_654_435_761u32));

                if ty == PT_RFRG || ty == PT_RFGL {
                    // `RFRG.cpp:22-36`: temp *= (new+257)/(old+257), with the
                    // previous pressure bit-cast into `tmp` exactly like TPT.
                    let new_pressure = air_pv[cidx];
                    let old = f32::reinterpret(part_tmp[id]);
                    if old.is_nan() {
                        part_tmp[id] = i32::reinterpret(new_pressure);
                    } else {
                        let scaled = part_temp[id] * ((new_pressure + 257.0f32) / (old + 257.0f32));
                        part_temp[id] = clampf(scaled, 0.0f32, MAX_TEMP);
                        part_tmp[id] = i32::reinterpret(new_pressure);
                    }
                } else if ty == PT_BASE {
                    // `BASE.cpp:56-215`. Spark reset + concentration clamp.
                    part_tmp[id] = 0i32;
                    let mut life = clampi(part_life[id].load(), 1i32, 100i32);
                    let pres = air_pv[cidx];
                    let mut done = false;

                    // Evaporation into BOYL keeps total concentration constant.
                    if life < 100i32
                        && pres < 10.0f32
                        && part_temp[id] > 393.15f32
                        && chance1(seed, 20u32)
                    {
                        let roll = hash_u32(seed ^ mix_index(id as u32, 0x27D4_EB2Fu32));
                        if chance(roll, 1u32, (life as u32) + 1u32) {
                            part_ty[id] = PT_BOYL;
                            done = true;
                        } else {
                            life += 1i32;
                            part_temp[id] -= 20.0f32 / (life as f32);
                        }
                    }

                    // The freezing point drops with the concentration.
                    if !done && part_temp[id] < 273.15f32 - (life as f32) / 4.0f32 {
                        part_ty[id] = PT_ICEI;
                        part_ctype[id] = PT_BASE;
                        life = 0i32;
                        done = true;
                    }

                    if !done {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32) && !done {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    let rt = part_ty[nid];
                                    let skip = rt == PT_BASE
                                        || rt == PT_SALT
                                        || rt == PT_SLTW
                                        || rt == PT_BOYL
                                        || rt == PT_MERC
                                        || rt == PT_BMTL
                                        || rt == PT_BRMT
                                        || rt == PT_SOAP
                                        || rt == PT_CLNE
                                        || rt == PT_PCLN
                                        || (rt == PT_ICEI && part_ctype[nid] == PT_BASE)
                                        || (rt == PT_SNOW && part_ctype[nid] == PT_BASE)
                                        || (rt == PT_SPRK
                                            && (part_ctype[nid] == PT_BMTL
                                                || part_ctype[nid] == PT_BRMT));
                                    if !skip {
                                        let nseed =
                                            hash_u32(seed ^ mix_index(nid as u32, 0x85EB_CA6Bu32));
                                        if life > 1i32
                                            && (rt == PT_WATR || rt == PT_DSTW || rt == PT_CBNW)
                                        {
                                            // Water dilutes the base.
                                            if chance1(nseed, 20u32) {
                                                let saturh = life / 2i32;
                                                part_ty[nid] = PT_BASE;
                                                part_life[nid].store(saturh);
                                                part_temp[nid] += (saturh as f32) / 10.0f32;
                                                life -= saturh;
                                            }
                                        } else if rt == PT_ACID {
                                            // `BASE.cpp:246-257`: the base
                                            // consumes one acid life per step.
                                            // The read-then-decrement must be
                                            // one atomic step, or two bases
                                            // touching the same acid would
                                            // both pass the `> 50` test.
                                            let mut cur = part_life[nid].load();
                                            let mut dec = false;
                                            for _ in 0u32..100u32 {
                                                if !dec && cur > 50i32 && life > 0i32 {
                                                    let prev = part_life[nid]
                                                        .compare_exchange_weak(cur, cur - 1i32);
                                                    if prev == cur {
                                                        dec = true;
                                                        life -= 1i32;
                                                    } else {
                                                        cur = prev;
                                                    }
                                                }
                                            }
                                            if part_life[nid].load() <= 50i32 {
                                                part_ty[nid] = PT_SLTW;
                                            }
                                            if life <= 0i32 {
                                                part_ty[id] = PT_SLTW;
                                                done = true;
                                            }
                                        } else if rt == PT_CAUS {
                                            let mut cur = part_life[nid].load();
                                            let mut dec = false;
                                            for _ in 0u32..100u32 {
                                                if !dec && cur > 50i32 && life > 0i32 {
                                                    let prev = part_life[nid]
                                                        .compare_exchange_weak(cur, cur - 1i32);
                                                    if prev == cur {
                                                        dec = true;
                                                        life -= 1i32;
                                                    } else {
                                                        cur = prev;
                                                    }
                                                }
                                            }
                                            if part_life[nid].load() <= 50i32 {
                                                part_ty[nid] = PT_NONE;
                                            }
                                            if life <= 0i32 {
                                                part_ty[id] = PT_SLTW;
                                                done = true;
                                            }
                                        } else if life >= 70i32 && rt == PT_OIL {
                                            part_ty[id] = PT_SOAP;
                                            part_ty[nid] = PT_NONE;
                                            done = true;
                                        } else if life > 1i32 && rt == PT_GOO {
                                            part_ty[nid] = PT_GEL;
                                            life -= 1i32;
                                        } else if life > 1i32 && rt == PT_BCOL {
                                            part_ty[nid] = PT_GUNP;
                                            life -= 1i32;
                                        } else if rt == PT_LAVA
                                            && part_ctype[nid] == PT_ROCK
                                            && pres >= 10.0f32
                                            && chance1(nseed, 1000u32)
                                        {
                                            part_ty[id] = PT_MERC;
                                            part_ty[nid] = PT_NONE;
                                            done = true;
                                        } else {
                                            let nprops = elem_word(elem_words, rt, WORD_PROPERTIES);
                                            let solid_conductor = (nprops
                                                & (TYPE_SOLID | PROP_CONDUCTS))
                                                == (TYPE_SOLID | PROP_CONDUCTS);
                                            let hardness =
                                                elem_word(elem_words, rt, WORD_HARDNESS) as i32;
                                            if life >= 10i32
                                                && solid_conductor
                                                && chance1(nseed, 10u32)
                                            {
                                                // Rust the conductive solid.
                                                part_ty[nid] = PT_BMTL;
                                                part_tmp[nid] =
                                                    20i32 + ((hash_u32(nseed) % 10u32) as i32);
                                                life -= 1i32;
                                                part_tmp[id] = 1i32;
                                            } else if hardness > 0i32
                                                && hardness < 50i32
                                                && life >= 2i32 * hardness
                                                && chance(nseed, (50i32 - hardness) as u32, 1000u32)
                                            {
                                                part_ty[nid] = PT_NONE;
                                                life -= 2i32;
                                                part_tmp[id] = 1i32;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    part_life[id].store(life);
                    if !done {
                        // `BASE.cpp:219-241`: two random radius-1 trades
                        // equalise the concentration. Both halves are atomic
                        // updates of the neighbour's `life`, so two bases
                        // trading into the same pixel conserve the total
                        // concentration instead of losing one side of the
                        // transfer.
                        for trade in 0u32..2u32 {
                            let rx = (hash_u32(seed ^ mix_index(trade, 0x1B87_3593u32)) % 3u32)
                                as i32
                                - 1i32;
                            let ry = (hash_u32(seed ^ mix_index(trade, 0x2545_F491u32)) % 3u32)
                                as i32
                                - 1i32;
                            if rx != 0i32 || ry != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    if part_ty[nid] == PT_BASE {
                                        let mine = part_life[id].load();
                                        let other = part_life[nid].load();
                                        if mine > other && mine > 1i32 {
                                            let diff = mine - other;
                                            let mut delta = 1i32;
                                            if diff > 1i32 {
                                                delta = diff / 2i32;
                                            }
                                            part_life[id].fetch_sub(delta);
                                            part_life[nid].fetch_add(delta);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if ty == PT_BREC {
                    // `BREC.cpp:62-74`.
                    if part_life[id].load() != 0i32 {
                        let pv = air_pv[cidx];
                        if pv > 10.0f32 {
                            if part_temp[id] > 9000.0f32 && pv > 30.0f32 && chance1(seed, 200u32) {
                                part_ty[id] = PT_EXOT;
                                part_life[id].store(1000i32);
                            }
                            part_temp[id] = clampf(part_temp[id] + pv / 8.0f32, MIN_TEMP, MAX_TEMP);
                        }
                    }
                } else if ty == PT_SEED {
                    // `SEED.cpp:65-88`: hot, pressurised seeds melt into wax.
                    // TPT recreates the particle with `create_part`, then
                    // zeroes `ctype`/`life`/`tmp`..`tmp4` explicitly; the
                    // `MWAX` create defaults leave the remaining words zeroed.
                    if part_temp[id] > 320.0f32 && air_pv[cidx] > 50.0f32 && chance1(seed, 150u32) {
                        part_ty[id] = PT_MWAX;
                        part_ctype[id] = 0u32;
                        part_life[id].store(0i32);
                        part_tmp[id] = 0i32;
                        part_tmp2[id] = 0i32;
                        part_tmp3[id] = 0i32;
                        part_tmp4[id] = 0i32;
                    }
                } else if ty == PT_VIBR || ty == PT_BVBR {
                    // `VIBR.cpp:114-143`: the stored energy detonates on the
                    // last countdown frame. `life` is `PROP_LIFE_DEC`, and the
                    // life pass decrements before this one, so the port
                    // detonates one frame earlier than TPT.
                    //
                    // `VIBR.cpp:87-113` also runs while the countdown lasts:
                    // a spark goes into a random conductor neighbour below
                    // `life == 300`, and the stored `tmp * 3` heat goes into a
                    // random non-insulating neighbour below `life == 500`, which
                    // zeroes `tmp`. Both rewrite a neighbour, so both are
                    // inverted and resolved by the target at the end of this
                    // kernel. The heat hand-off needs to spend `tmp` exactly
                    // once, which a same-dispatch read cannot express; the
                    // source therefore publishes the picked cell in its own
                    // `tmp2` one frame before the receiver reads it, and drops
                    // the offer (and `tmp`) one frame after. `life` decrements
                    // by exactly one per frame, so `life % 3` is the clock and
                    // the read frame never overlaps a write of the same word.
                    let vibr_life = part_life[id].load();
                    if vibr_life == 1i32 {
                        // A pending offer must not be mistaken for the
                        // "discharged by another explosion" marker below.
                        if part_tmp2[id] < 0i32 {
                            part_tmp2[id] = 0i32;
                            part_tmp[id] = 0i32;
                        }
                        if part_tmp2[id] == 0i32 {
                            let r = vibr_roll(idx as u32, id as u32, frame_seed);
                            // `orbit_rx`/`orbit_ry` at `r & 7`, `(r >> 3) & 7`
                            // and `(r >> 6) & 7`.
                            for step in 0u32..3u32 {
                                let k = ((r >> (step * 3u32)) & 7u32) as i32;
                                let rx = vibr_orbit_x(k);
                                let ry = vibr_orbit_y(k);
                                // `create_part(-3, ...)` for the energy pair and
                                // `-1` for the `BREC`; the queue drops a request
                                // whose pixel is taken instead of displacing it.
                                let cty = if step == 0u32 {
                                    PT_ELEC
                                } else if step == 1u32 {
                                    PT_PHOT
                                } else {
                                    PT_BREC
                                };
                                request_create(
                                    create_queue,
                                    cty,
                                    px + rx,
                                    py + ry,
                                    CR_SET_TEMP,
                                    0i32,
                                    0u32,
                                    7000.0f32,
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
                            part_ty[id] = PT_EXOT;
                            part_life[id].store(1000i32);
                            part_ctype[id] = 0u32;
                            part_tmp[id] = 244i32;
                            part_tmp2[id] = ((r >> 9u32) % 1000u32) as i32;
                            part_temp[id] = 9000.0f32;
                            air_pv[cidx] =
                                clampf(air_pv[cidx] + 50.0f32, MIN_PRESSURE, MAX_PRESSURE);
                        } else {
                            // `VIBR.cpp:137-142`: a discharge triggered by
                            // another explosion only vents the stored energy.
                            part_tmp2[id] = 0i32;
                            part_temp[id] = 273.15f32;
                            part_tmp[id] = 0i32;
                        }
                    } else if vibr_life > 1i32 && vibr_life < 500i32 {
                        let phase = vibr_life % 3i32;
                        if phase == 0i32 {
                            // `VIBR.cpp:102-113`: publish the cell the heat is
                            // offered to. TPT only dumps when the sampled pixel
                            // holds matter that is neither `VIBR`/`BVBR` nor a
                            // heat insulator, and retries with a new roll on the
                            // next frame otherwise; the same test here decides
                            // whether an offer is made at all.
                            if part_tmp[id] != 0i32 && part_tmp2[id] >= 0i32 {
                                let r = vibr_roll(idx as u32, id as u32, frame_seed);
                                let (rx, ry) = vibr_heat_pick(r, vibr_life);
                                let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                                if nv != 0u32 {
                                    let nty = part_ty[pmap_slot(nv)];
                                    if nty != PT_VIBR
                                        && nty != PT_BVBR
                                        && !heat_insulator(elem_words, nty)
                                    {
                                        part_tmp2[id] = vibr_offer(rx, ry);
                                    }
                                }
                            }
                        } else if phase == 1i32 {
                            // The receiver has had its frame: drop the offer and
                            // spend the energy. `VIBR.cpp:111` zeroes `tmp` in
                            // the same branch that writes the neighbour.
                            if part_tmp2[id] < 0i32 {
                                part_tmp2[id] = 0i32;
                                part_tmp[id] = 0i32;
                            }
                        }
                    }
                } else if ty == PT_MORT {
                    // `MORT.cpp:52-55`: one `SMKE` per frame into the pixel
                    // above. `create_part(-1, x, y-1, PT_SMKE)` refuses an
                    // occupied pixel, so the free-pixel test stands in for its
                    // `eval_move`; walls are not bound here.
                    if pmap_at(pmap, xres, yres, px, py - 1) == 0u32 {
                        request_create_plain(create_queue, PT_SMKE, px, py - 1);
                    }
                }

                // `VIBR.cpp:87-101`: an exploding `VIBR`/`BVBR` turns one
                // random conductor neighbour into `SPRK` (`life = 4`,
                // `ctype` = the conductor's element) while `life < 300`.
                // Inverted: the conductor recomputes the source's roll from its
                // pixel and slot, checks that the sampled orbit cell is itself,
                // and rewrites only its own slot.
                let cur_ty = part_ty[id];
                if cur_ty != PT_NONE
                    && cur_ty != PT_BREC
                    && part_life[id].load() == 0i32
                    && (elem_word(elem_words, cur_ty, WORD_PROPERTIES) & PROP_CONDUCTS) != 0u32
                {
                    let mut sparked = false;
                    for k in 0u32..8u32 {
                        if !sparked {
                            let ox = vibr_orbit_x(k as i32);
                            let oy = vibr_orbit_y(k as i32);
                            let nv = pmap_at(pmap, xres, yres, px + ox, py + oy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let nlife = part_life[nid].load();
                                // `life == 1` is the detonation frame, where
                                // the source rewrites its own type; leaving it
                                // out keeps this read out of that write.
                                if (nty == PT_VIBR || nty == PT_BVBR)
                                    && nlife > 1i32
                                    && nlife < 300i32
                                {
                                    let nidx = ((py + oy) as u32) * xres + (px + ox) as u32;
                                    let r = vibr_roll(nidx, nid as u32, frame_seed);
                                    let pick = (r & 7u32) as i32;
                                    if vibr_orbit_x(pick) == -ox && vibr_orbit_y(pick) == -oy {
                                        part_ctype[id] = cur_ty;
                                        part_ty[id] = PT_SPRK;
                                        part_life[id].store(4i32);
                                        sparked = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // `VIBR.cpp:102-113`: the `tmp * 3` heat hand-off. The source
                // published the sampled cell in its `tmp2` on the frame its
                // `life` was `3k`, so the receiver reads the offer only while
                // the source's `life` is `3k - 1` — a frame in which the source
                // writes neither `tmp2` nor `tmp`, which is why this is a read
                // of a frozen word and not a cross-slot update.
                let heat_ty = part_ty[id];
                if heat_ty != PT_NONE
                    && heat_ty != PT_VIBR
                    && heat_ty != PT_BVBR
                    && !heat_insulator(elem_words, heat_ty)
                {
                    let mut donated = 0.0f32;
                    for n in 0u32..49u32 {
                        let rx = (n % 7u32) as i32 - 3i32;
                        let ry = (n / 7u32) as i32 - 3i32;
                        if rx != 0i32 || ry != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px - rx, py - ry);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let nlife = part_life[nid].load();
                                if (nty == PT_VIBR || nty == PT_BVBR)
                                    && nlife > 1i32
                                    && nlife < 500i32
                                    && nlife % 3i32 == 2i32
                                {
                                    let offer = part_tmp2[nid];
                                    if offer < 0i32 {
                                        let cell = -offer - 1i32;
                                        if cell >= 0i32 && cell < 49i32 {
                                            if cell % 7i32 - 3i32 == rx && cell / 7i32 - 3i32 == ry
                                            {
                                                donated += (part_tmp[nid] as f32) * 3.0f32;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if donated != 0.0f32 {
                        part_temp[id] = clampf(part_temp[id] + donated, MIN_TEMP, MAX_TEMP);
                    }
                }
            }
        }
    }
}
