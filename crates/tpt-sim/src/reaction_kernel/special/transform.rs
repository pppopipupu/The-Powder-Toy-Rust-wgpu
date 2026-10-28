use cubecl::prelude::*;
use tpt_core::constants::*;

use super::force::merc_maxtmp;
use super::{
    PLNT_COLOUR, PLNT_DIR, PLNT_LIFE, PLNT_PHASE, chance, chance1, clampi, dir_x, dir_y,
    stasis_frozen,
};
use crate::reaction_kernel::shared::{
    CR_SET_CTYPE, CR_SET_DCOLOUR, CR_SET_LIFE, CR_SET_TEMP, CR_SET_TMP, CR_SET_TMP2, CR_SET_TMP3,
    CR_SET_TMP4, CR_SET_VX, CR_SET_VY, FLAG_SPAWN_TMP, hash_u32, mix_index, pmap_at, pmap_slot,
    request_create, request_create_plain,
};

/// `BRMT.cpp:53-55`: reaction rate denominator for `BREC + BRMT -> THRM`.
#[cube]
fn brec_brmt_temp_factor(temp: f32) -> u32 {
    let tf = 1000i32 - ((523.15f32 - temp) * 2.0f32) as i32;
    clampi(tf, 2i32, 100_000i32) as u32
}

/// `BRMT.cpp:53-55`: reaction rate denominator for `BREC + BRMT -> THRM` on host.
#[cfg(test)]
pub fn brec_brmt_temp_factor_host(temp: f32) -> u32 {
    let tf = 1000i32 - ((523.15f32 - temp) * 2.0f32) as i32;
    if tf < 2i32 { 2u32 } else { tf as u32 }
}

/// `GOLD`, `RSST`/`RSSS`, `VINE`, `SEED`, `PLNT`, `SLCN`, `CRAY`,
/// `VIBR` (anti-air) and the `MERC`/`POLO` branches that need a map.
///
/// Pixel parallel: all spatial reads go through `pmap`; cross-slot writes are
/// limited to type changes and deaths, exactly like [`super::fluid`], and
/// creations (the `PLNT` branches) go through the device creation queue. The
/// queue keeps a creation invisible to the rest of the frame until the driver
/// drains it, so a branch cannot see a sibling branch's pixel; every branch
/// that found the pixel empty when the pass ran is emitted.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn special_transform_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_temp: &mut [f32],
    part_flags: &mut [u32],
    part_dcolour: &[u32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    frame_seed: u32,
    down_dir: u32,
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
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 2_246_822_519u32));

                if ty == PT_GOLD {
                    // `GOLD.cpp:78-113`: reverse corrosion on rusted BMTL in
                    // the four axial directions four pixels away.
                    for j in 0u32..8u32 {
                        let roll = hash_u32(seed ^ mix_index(id as u32, 0x9E37_79B9u32) ^ j);
                        let rx = (roll % 9u32) as i32 - 4i32;
                        let ry = ((roll >> 4u32) % 9u32) as i32 - 4i32;
                        if (rx == 0i32 && ry != 0i32) || (rx != 0i32 && ry == 0i32) {
                            let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                if part_ty[nid] == PT_BMTL && part_tmp[nid] != 0i32 {
                                    part_ty[nid] = PT_IRON;
                                    part_tmp[nid] = 0i32;
                                }
                            }
                        }
                    }
                    // Spark re-emission: GOLD conducts like any other metal.
                    if part_life[id] == 0i32 {
                        for j in 0u32..4u32 {
                            let nx = px + gold_dx(j);
                            let ny = py + gold_dy(j);
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nlife = part_life[nid];
                                if part_ty[nid] == PT_SPRK && nlife > 0i32 && nlife < 4i32 {
                                    part_ty[id] = PT_SPRK;
                                    part_life[id] = 4i32;
                                    part_ctype[id] = PT_GOLD;
                                }
                            }
                        }
                    }
                } else if ty == PT_RSST || ty == PT_RSSS {
                    // `RSST.cpp:57-99`, `RSSS.cpp:27-60`.
                    let mut done = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if !done {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let rt = part_ty[nid];
                                if ty == PT_RSST && rt == PT_GUNP {
                                    part_ty[id] = PT_FIRW;
                                    part_ty[nid] = PT_NONE;
                                    done = true;
                                } else if ty == PT_RSST && rt == PT_BCOL {
                                    part_ty[id] = PT_FSEP;
                                    part_life[id] = 50i32;
                                    part_ty[nid] = PT_NONE;
                                    done = true;
                                } else if rt == PT_CLNE || rt == PT_PCLN {
                                    let c = part_ctype[nid];
                                    if c != ty {
                                        part_ctype[id] = c;
                                    }
                                } else if rt == PT_BCLN || rt == PT_PBCN {
                                    let c = part_ctype[nid];
                                    if c != ty {
                                        part_tmp[id] = c as i32;
                                    }
                                }
                            }
                        }
                    }
                } else if ty == PT_VINE {
                    // `VINE.cpp:55-77`: one random neighbour per frame. The
                    // 1-in-15 roll turns the vine into `PLNT` in place, and
                    // otherwise a free neighbour grows a new `VINE` (which
                    // inherits the parent temperature) while this pixel
                    // becomes `PLNT`.
                    let roll = hash_u32(seed ^ mix_index(id as u32, 0x2545_F491u32));
                    let rx = (roll % 3u32) as i32 - 1i32;
                    let r1 = roll >> 2u32;
                    let ry = (r1 % 3u32) as i32 - 1i32;
                    let r2 = r1 >> 2u32;
                    if rx != 0i32 || ry != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                        if r2.is_multiple_of(15u32) {
                            part_ty[id] = PT_PLNT;
                            part_tmp[id] = 1i32;
                        } else if nv == 0u32 {
                            // `VINE.cpp:66-68`: the child takes the parent
                            // temperature before the parent becomes PLNT.
                            request_create(
                                create_queue,
                                PT_VINE,
                                px + rx,
                                py + ry,
                                CR_SET_TEMP,
                                0i32,
                                0u32,
                                part_temp[id],
                                0i32,
                                0i32,
                                0i32,
                                0i32,
                                0u32,
                                0u32,
                                0.0f32,
                                0.0f32,
                            );
                            part_ty[id] = PT_PLNT;
                            part_tmp[id] = 1i32;
                        }
                    }
                    if part_temp[id] > 350.0f32 && part_temp[id] > (part_tmp2[id] as f32) {
                        part_tmp2[id] = part_temp[id] as i32;
                    }
                } else if ty == PT_SEED {
                    seed_update(
                        pmap, part_ty, part_life, part_ctype, part_temp, xres, yres, down_dir, id,
                        px, py,
                    );
                } else if ty == PT_PLNT {
                    plnt_update(
                        pmap,
                        part_ty,
                        part_life,
                        part_ctype,
                        part_tmp,
                        part_tmp2,
                        part_tmp3,
                        part_tmp4,
                        part_temp,
                        part_flags,
                        create_queue,
                        xres,
                        yres,
                        down_dir,
                        seed,
                        id,
                        px,
                        py,
                    );
                } else if ty == PT_SLCN {
                    // `SLCN.cpp:58-99`: sparkle state machine.
                    let mut tmp = part_tmp[id];
                    if tmp == 0i32 {
                        tmp = 0x0010_0000i32
                            + ((hash_u32(seed ^ mix_index(id as u32, 0x68E3_1DA4u32))
                                % 0x0090_0000u32) as i32);
                    }
                    let phase = (tmp & 0x0FFFi32) + ((tmp >> 20i32) & 0x0FFFi32);
                    if (phase & 0x1000i32) != 0i32 {
                        let next =
                            (hash_u32(seed ^ mix_index(id as u32, 0x27D4_EB2Fu32)) & 0xFu32) as i32;
                        tmp = (tmp & -1_048_576i32)
                            | (phase & 0x0FFFi32)
                            | (next << 16i32)
                            | ((tmp >> 4i32) & 0xF000i32);
                    } else {
                        tmp = (tmp & -4_096i32) | phase;
                    }
                    part_tmp[id] = tmp;

                    if part_life[id] == 0i32 && part_temp[id] < 373.15f32 {
                        // `SLCN.cpp` probes for a spark four pixels away on the
                        // two axes (the `check_coords` table is {-4,4,0,0}).
                        for j in 0u32..4u32 {
                            let nx = px + gold_dx(j);
                            let ny = py + gold_dy(j);
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nlife = part_life[nid];
                                if part_ty[nid] == PT_SPRK && nlife != 0i32 && nlife < 4i32 {
                                    part_ty[id] = PT_SPRK;
                                    part_life[id] = 4i32;
                                    part_ctype[id] = PT_SLCN;
                                }
                            }
                        }
                    }
                } else if ty == PT_CRAY {
                    // `CRAY.cpp:59-79`: latch the element to emit from a
                    // neighbour that is not part of the beam machinery. The
                    // reference samples the energy layer (`photons`) first and
                    // falls back to `pmap`, so a laser next to a photon/neutron
                    // latches the energy element. `ElementDef::enabled` is a
                    // host-only flag; a disabled padding slot can never occupy a
                    // live map word, so the id-range test is the device proxy
                    // (see `pipe::pump`).
                    let ctype = part_ctype[id];
                    if ctype == 0u32 || ctype >= PT_NUM as u32 {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            let mut nv = pmap_at(photons, xres, yres, px + dx, py + dy);
                            if nv == 0u32 {
                                nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            }
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let rt = part_ty[nid];
                                if rt != PT_CRAY
                                    && rt != PT_PSCN
                                    && rt != PT_INST
                                    && rt != PT_METL
                                    && rt != PT_SPRK
                                    && rt < PT_NUM as u32
                                {
                                    part_ctype[id] = rt;
                                    part_temp[id] = part_temp[nid];
                                }
                            }
                        }
                    }
                } else if ty == PT_VIBR || ty == PT_BVBR {
                    // `VIBR.cpp:88-95`: anti-air breaks vibranium.
                    // `VIBR.cpp:177-182`: VIBR/BVBR + EXOT -> 2xEXOT (1 in 25 chance).
                    let mut done_vibr = false;
                    if part_life[id] == 0i32 {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32) && !done_vibr {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    let rt = part_ty[nid];
                                    let nseed =
                                        hash_u32(seed ^ mix_index(nid as u32, 0x1B87_3593u32));
                                    if rt == PT_EXOT && chance1(nseed, 25u32) {
                                        part_ty[id] = PT_EXOT;
                                        done_vibr = true;
                                    } else if ty == PT_VIBR && rt == PT_ANAR {
                                        part_ty[id] = PT_BVBR;
                                        done_vibr = true;
                                    }
                                }
                            }
                        }
                    } else {
                        // `VIBR.cpp:158-163`: a cold flame next to an
                        // exploding crystal triggers the early discharge.
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if dx != 0i32 || dy != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 && part_ty[pmap_slot(nv)] == PT_CFLM {
                                    part_tmp2[id] = 1i32;
                                    part_tmp[id] = 0i32;
                                }
                            }
                        }
                    }
                } else if ty == PT_MERC {
                    // `MERC.cpp:100-146`: condense neighbours below the target
                    // volume, spill into free space above it.
                    let maxtmp = merc_maxtmp(part_temp[id], seed);
                    let mut amt = part_tmp[id];
                    if amt < maxtmp {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32) && amt < maxtmp {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    if part_ty[nid] == PT_MERC
                                        && chance1(
                                            hash_u32(seed ^ mix_index(nid as u32, 0x51ED_270Bu32)),
                                            3u32,
                                        )
                                        && amt + part_tmp[nid] < maxtmp
                                    {
                                        amt += part_tmp[nid] + 1i32;
                                        part_ty[nid] = PT_NONE;
                                    }
                                }
                            }
                        }
                    } else {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32) && amt > maxtmp && amt >= 1i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv == 0u32 {
                                    // `MERC.cpp:100-118`: every free neighbour
                                    // gets one spilled pixel while the volume
                                    // is above the target. The child takes the
                                    // parent's temperature and colour, and its
                                    // `tmp` is zeroed (`create_part` would
                                    // otherwise seed the element default 10).
                                    request_create(
                                        create_queue,
                                        PT_MERC,
                                        px + dx,
                                        py + dy,
                                        CR_SET_TEMP | CR_SET_TMP | CR_SET_DCOLOUR,
                                        0i32,
                                        0u32,
                                        part_temp[id],
                                        0i32,
                                        0i32,
                                        0i32,
                                        0i32,
                                        part_dcolour[id],
                                        0u32,
                                        0.0f32,
                                        0.0f32,
                                    );
                                    amt -= 1i32;
                                }
                            }
                        }
                    }
                    part_tmp[id] = amt;
                } else if ty == PT_BREC || ty == PT_BRMT {
                    // `BRMT.cpp:51-78`, `BREC.cpp:51-78`: BREC + hot BRMT (> 250 C) react into THRM.
                    // Hot BREC (> 250 C) also initiates the conversion to THRM.
                    let mut reacted = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !reacted {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let rt = part_ty[nid];
                                let pair_match = (ty == PT_BREC && rt == PT_BRMT)
                                    || (ty == PT_BRMT && rt == PT_BREC);
                                if pair_match {
                                    let btemp = if part_temp[id] > part_temp[nid] {
                                        part_temp[id]
                                    } else {
                                        part_temp[nid]
                                    };
                                    let should_eval = if ty == PT_BREC {
                                        part_temp[id] > 523.15f32 || id < nid
                                    } else {
                                        id < nid
                                    };
                                    if btemp > 523.15f32 && should_eval {
                                        let tf = brec_brmt_temp_factor(btemp);
                                        let nseed =
                                            hash_u32(seed ^ mix_index(nid as u32, 0x3C6E_F35Fu32));
                                        if chance1(nseed, tf) {
                                            let roll = hash_u32(nseed ^ 0x51ED_270Bu32);
                                            if chance1(roll, 2u32) {
                                                part_ty[id] = PT_THRM;
                                            } else {
                                                part_ty[nid] = PT_THRM;
                                            }
                                            reacted = true;
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

/// `SEED.cpp:145-244`: the breeding eligibility of the *initiating* seed.
///
/// The reference tests the water count, the breeding bit and the temperature
/// range before it enters the neighbour loop; the partner only has to be a
/// `SEED` without its breeding bit.
#[cube]
fn seed_can_breed(part_ty: &[u32], part_ctype: &mut [u32], part_temp: &[f32], id: usize) -> bool {
    part_ty[id] == PT_SEED
        && (part_ctype[id] & 1u32) == 0u32
        && ((part_ctype[id] >> PLNT_LIFE) & 0xFFu32) > 0u32
        && part_temp[id] <= 343.15f32
        && part_temp[id] >= 278.15f32
}

/// The breeding chance of the pair `(id, nid)` at neighbour index `n`.
///
/// TPT draws once from the shared RNG per candidate neighbour; a pixel-parallel
/// pass keys the draw by the pair and the scan index instead, so every seed
/// that replays the pair computes the same answer.
#[cube]
fn seed_breed_roll(frame_seed: u32, id: usize, nid: usize, n: u32) -> u32 {
    hash_u32(
        frame_seed
            ^ mix_index(id as u32, 0x9E37_79B9u32)
            ^ mix_index(nid as u32, 0x85EB_CA6Bu32)
            ^ mix_index(n, 0xC2B2_AE35u32),
    )
}

/// `SEED.cpp:141-244`: the first eligible neighbour an initiating seed breeds
/// with, in the reference's `rx`/`ry` scan order.
///
/// Returns `(dx, dy, found)`. `n` counts the flattened scan so the chance draw
/// is stable when the same seed is replayed from a neighbour.
#[cube]
#[allow(clippy::too_many_arguments)]
fn seed_choice(
    pmap: &[u32],
    part_ty: &[u32],
    part_ctype: &mut [u32],
    part_temp: &[f32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
    id: usize,
    px: i32,
    py: i32,
) -> (i32, i32, bool) {
    let mut cdx = 0i32;
    let mut cdy = 0i32;
    let mut found = false;
    if seed_can_breed(part_ty, part_ctype, part_temp, id) {
        for n in 0u32..9u32 {
            let dx = (n % 3u32) as i32 - 1i32;
            let dy = (n / 3u32) as i32 - 1i32;
            if (dx != 0i32 || dy != 0i32) && !found {
                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                if nv != 0u32 {
                    let nid = pmap_slot(nv);
                    if part_ty[nid] == PT_SEED
                        && (part_ctype[nid] & 1u32) == 0u32
                        && chance1(seed_breed_roll(frame_seed, id, nid, n), 10u32)
                    {
                        cdx = dx;
                        cdy = dy;
                        found = true;
                    }
                }
            }
        }
    }
    (cdx, cdy, found)
}

/// `true` when a lower-slot seed already chose `id` as its partner.
///
/// TPT processes the slot array in order, so the earlier seed of a pair claims
/// it first and the later seed sees the breeding bit when its own turn comes.
/// A seed with a smaller slot that chose this one therefore suppresses its
/// initiation.
#[cube]
#[allow(clippy::too_many_arguments)]
fn seed_claimed(
    pmap: &[u32],
    part_ty: &[u32],
    part_ctype: &mut [u32],
    part_temp: &[f32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
    id: usize,
    px: i32,
    py: i32,
) -> bool {
    let mut claimed = false;
    for n in 0u32..9u32 {
        let dx = (n % 3u32) as i32 - 1i32;
        let dy = (n / 3u32) as i32 - 1i32;
        if (dx != 0i32 || dy != 0i32) && !claimed {
            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
            if nv != 0u32 {
                let nid = pmap_slot(nv);
                if nid < id && part_ty[nid] == PT_SEED {
                    let (kdx, kdy, kfound) = seed_choice(
                        pmap,
                        part_ty,
                        part_ctype,
                        part_temp,
                        xres,
                        yres,
                        frame_seed,
                        nid,
                        px + dx,
                        py + dy,
                    );
                    if kfound && px + dx + kdx == px && py + dy + kdy == py {
                        claimed = true;
                    }
                }
            }
        }
    }
    claimed
}

/// The winning initiator for the seed at `(px, py)`: the lowest-slot unclaimed
/// seed whose chosen partner is this pixel. `0xFFFF_FFFF` means none.
///
/// Restricting the initiator to a lower slot reproduces TPT's slot order for
/// the pair; the winner is unique, so the partner's genome has a single writer.
#[cube]
#[allow(clippy::too_many_arguments)]
fn seed_winner(
    pmap: &[u32],
    part_ty: &[u32],
    part_ctype: &mut [u32],
    part_temp: &[f32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
    px: i32,
    py: i32,
) -> u32 {
    let mut best = 0xFFFF_FFFFu32;
    for n in 0u32..9u32 {
        let dx = (n % 3u32) as i32 - 1i32;
        let dy = (n / 3u32) as i32 - 1i32;
        if dx != 0i32 || dy != 0i32 {
            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
            if nv != 0u32 {
                let nid = pmap_slot(nv);
                if part_ty[nid] == PT_SEED
                    && (nid as u32) < best
                    && !seed_claimed(
                        pmap,
                        part_ty,
                        part_ctype,
                        part_temp,
                        xres,
                        yres,
                        frame_seed,
                        nid,
                        px + dx,
                        py + dy,
                    )
                {
                    let (kdx, kdy, kfound) = seed_choice(
                        pmap,
                        part_ty,
                        part_ctype,
                        part_temp,
                        xres,
                        yres,
                        frame_seed,
                        nid,
                        px + dx,
                        py + dy,
                    );
                    if kfound && px + dx + kdx == px && py + dy + kdy == py {
                        best = nid as u32;
                    }
                }
            }
        }
    }
    best
}

/// One meiosis draw of the breeding swap (`SEED.cpp:184-243`).
#[cube]
fn breed_draw(seed: u32, k: u32) -> bool {
    (hash_u32(seed ^ mix_index(k, 0x27D4_EB2Fu32)) & 1u32) != 0u32
}

/// The pair seed of a breeding swap: keyed by the two slots so both parents
/// compute the same draw sequence.
#[cube]
fn breed_pair_seed(frame_seed: u32, a: usize, b: usize) -> u32 {
    let mut lo = a;
    let mut hi = b;
    if hi < lo {
        lo = b;
        hi = a;
    }
    hash_u32(
        frame_seed ^ mix_index(lo as u32, 0x9E37_79B9u32) ^ mix_index(hi as u32, 0x85EB_CA6Bu32),
    )
}

/// `SEED.cpp:184-216`: the colour half of the child genome.
///
/// `ac` is the initiator's `ctype`, `bc` the partner's. `row` selects which of
/// the two children this seed becomes (the reference fills `bits[0]` for the
/// initiator and `bits[1]` for the partner); `water` is restored on the
/// initiator only, exactly as its own update does after the swap.
#[cube]
fn breed_ctype(seed: u32, ac: u32, bc: u32, row: u32, water: u32) -> u32 {
    let mut out = 1u32;
    if row == 0u32 {
        out |= water << PLNT_LIFE;
    }
    for ind in 0u32..3u32 {
        for kid in 0u32..2u32 {
            let k = 2u32 * (2u32 * ind + kid);
            let first = breed_draw(seed, k);
            let second = breed_draw(seed, k + 1u32);
            if kid == row {
                let a0 = (ac >> (2u32 * ind + PLNT_COLOUR)) & 1u32;
                let a1 = (ac >> (1u32 + 2u32 * ind + PLNT_COLOUR)) & 1u32;
                let b0 = (bc >> (2u32 * ind + PLNT_COLOUR)) & 1u32;
                let b1 = (bc >> (1u32 + 2u32 * ind + PLNT_COLOUR)) & 1u32;
                let allele_a = if first { a0 } else { a1 };
                let allele_b = if second { b0 } else { b1 };
                out |= allele_a << (2u32 * ind + PLNT_COLOUR);
                out |= allele_b << (1u32 + 2u32 * ind + PLNT_COLOUR);
            }
        }
    }
    out
}

/// `SEED.cpp:218-243`: one `tmp`..`tmp4` word of the child genome.
///
/// The branching bitmask is inherited bit by bit, the phase pairs two bits at a
/// time; `field` selects the draw block of the word. TPT ORs the inherited
/// bits into the existing value, so the old word is the starting point.
#[cube]
fn breed_tmp_word(seed: u32, old_i: i32, old_r: i32, row: u32, field: u32) -> i32 {
    let mut out = if row == 0u32 { old_i } else { old_r };
    for ind in 0u32..5u32 {
        let k = 12u32 + 2u32 * (4u32 * ind + field);
        let first = breed_draw(seed, k);
        let second = breed_draw(seed, k + 1u32);
        let bit = 1i32 << (ind as i32);
        let src = if row == 0u32 {
            if first { old_i & bit } else { old_r & bit }
        } else if second {
            old_i & bit
        } else {
            old_r & bit
        };
        out |= src;
    }
    for j in 0u32..5u32 {
        let ind = 5u32 + 2u32 * j;
        let k = 12u32 + 40u32 + 2u32 * (4u32 * j + field);
        let first = breed_draw(seed, k);
        let second = breed_draw(seed, k + 1u32);
        let mask = 3i32 << (ind as i32);
        let src = if row == 0u32 {
            if first { old_i & mask } else { old_r & mask }
        } else if second {
            old_i & mask
        } else {
            old_r & mask
        };
        out |= src;
    }
    out
}

/// `SEED.cpp:145-244`: the breeding swap, staged after
/// [`special_transform_kernel`].
///
/// TPT's seed update breeds inside the neighbour loop: it resets both `ctype`
/// words to `1`, ORs the two children's colour alleles and `tmp` bits, and
/// restores the initiator's water at the end of its own update. That is a
/// cross-slot read-modify-write of five words in both seeds. The port resolves
/// the pair from the post-update words instead and keeps the reference's
/// **lower-slot priority**: a seed is an initiator only when no lower-slot seed
/// chose it and it is the lowest-slot unclaimed chooser of its target. That
/// unique initiator writes row 0 of its own slot and row 1 of the partner's
/// slot, so every slot has exactly one writer and no `Atomic` is needed. A
/// claimed partner writes nothing. Both roles derive the same meiosis draws
/// from the pair seed.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn seed_breed_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_temp: &[f32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_SEED {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                // A lower-slot chooser owns this seed's row 1, so this slot
                // must not initiate; the initiator writes both slots.
                if !seed_claimed(
                    pmap, part_ty, part_ctype, part_temp, xres, yres, frame_seed, id, px, py,
                ) {
                    let (cdx, cdy, has_choice) = seed_choice(
                        pmap, part_ty, part_ctype, part_temp, xres, yres, frame_seed, id, px, py,
                    );
                    if has_choice {
                        let tx = px + cdx;
                        let ty = py + cdy;
                        let tslot = pmap_slot(pmap_at(pmap, xres, yres, tx, ty));
                        // Only the earlier seed of a pair initiates; the
                        // winner check rejects a lower unclaimed chooser of
                        // the same partner.
                        if tslot > id
                            && seed_winner(
                                pmap, part_ty, part_ctype, part_temp, xres, yres, frame_seed, tx,
                                ty,
                            ) == id as u32
                        {
                            let seed = breed_pair_seed(frame_seed, id, tslot);
                            // Read both pre-swap genomes before writing either
                            // slot: the partner thread never writes its own.
                            let ac = part_ctype[id];
                            let bc = part_ctype[tslot];
                            let water = (ac >> PLNT_LIFE) & 0xFFu32;
                            let c0 = breed_ctype(seed, ac, bc, 0u32, water);
                            let c1 = breed_ctype(seed, ac, bc, 1u32, 0u32);
                            let w0a =
                                breed_tmp_word(seed, part_tmp[id], part_tmp[tslot], 0u32, 0u32);
                            let w0b =
                                breed_tmp_word(seed, part_tmp2[id], part_tmp2[tslot], 0u32, 1u32);
                            let w0c =
                                breed_tmp_word(seed, part_tmp3[id], part_tmp3[tslot], 0u32, 2u32);
                            let w0d =
                                breed_tmp_word(seed, part_tmp4[id], part_tmp4[tslot], 0u32, 3u32);
                            let w1a =
                                breed_tmp_word(seed, part_tmp[id], part_tmp[tslot], 1u32, 0u32);
                            let w1b =
                                breed_tmp_word(seed, part_tmp2[id], part_tmp2[tslot], 1u32, 1u32);
                            let w1c =
                                breed_tmp_word(seed, part_tmp3[id], part_tmp3[tslot], 1u32, 2u32);
                            let w1d =
                                breed_tmp_word(seed, part_tmp4[id], part_tmp4[tslot], 1u32, 3u32);
                            part_ctype[id] = c0;
                            part_ctype[tslot] = c1;
                            part_tmp[id] = w0a;
                            part_tmp2[id] = w0b;
                            part_tmp3[id] = w0c;
                            part_tmp4[id] = w0d;
                            part_tmp[tslot] = w1a;
                            part_tmp2[tslot] = w1b;
                            part_tmp3[tslot] = w1c;
                            part_tmp4[tslot] = w1d;
                        }
                    }
                }
            }
        }
    }
}

/// `GOLD.cpp` spark probe x offsets: `{-4, 4, 0, 0}`.
#[cube]
fn gold_dx(j: u32) -> i32 {
    if j == 0u32 {
        -4i32
    } else if j == 1u32 {
        4i32
    } else {
        0i32
    }
}

/// `GOLD.cpp` spark probe y offsets: `{0, 0, -4, 4}`.
#[cube]
fn gold_dy(j: u32) -> i32 {
    if j == 2u32 {
        -4i32
    } else if j == 3u32 {
        4i32
    } else {
        0i32
    }
}

/// `SEED.cpp:106-281`: drink water, grow into `PLNT`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn seed_update(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    xres: u32,
    yres: u32,
    down_dir: u32,
    id: usize,
    px: i32,
    py: i32,
) {
    let mut ctype = part_ctype[id];
    let mut life = part_life[id];

    if part_temp[id] > 343.15f32 || part_temp[id] < 278.15f32 {
        // Too hot or too cold: lose the stored water.
        ctype &= !(0xFFu32 << PLNT_LIFE);
        life = 0i32;
    } else {
        let mut water = (ctype >> PLNT_LIFE) & 0xFFu32;
        let mut grown = false;

        if water > 3u32 {
            // Use the gravity direction (8 = undecided -> TPT falls back to 4).
            let mut down = down_dir;
            if down > 7u32 {
                down = 4u32;
            }
            let up = (down + 4u32) % 8u32;
            let dv = pmap_at(pmap, xres, yres, px + dir_x(down), py + dir_y(down));
            let uv = pmap_at(pmap, xres, yres, px + dir_x(up), py + dir_y(up));
            let mut grounded = false;
            if dv != 0u32 {
                let dt = part_ty[pmap_slot(dv)];
                if dt == PT_SAND || dt == PT_SPNG {
                    grounded = true;
                }
            }
            if grounded && uv == 0u32 {
                if life > 200i32 {
                    // Grow: hand the seed over to the plant genome.
                    part_ty[id] = PT_PLNT;
                    ctype &= !((7u32 << PLNT_DIR) | (3u32 << PLNT_PHASE));
                    ctype |= ((up & 7u32) << PLNT_DIR) | 1u32;
                    life = 15i32 * (water as i32);
                    grown = true;
                } else {
                    life += 1i32;
                }
            } else {
                life = 0i32;
            }
        }

        if !grown {
            for n in 0u32..9u32 {
                let dx = (n % 3u32) as i32 - 1i32;
                let dy = (n / 3u32) as i32 - 1i32;
                if dx != 0i32 || dy != 0i32 {
                    let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                    if nv != 0u32 {
                        let nid = pmap_slot(nv);
                        let rt = part_ty[nid];
                        let mut drank = false;
                        // `SEED.cpp:106-140`: water, heavy water, carbonated and
                        // distilled water all feed the seed, each with its own
                        // capacity.
                        if (rt == PT_WATR && water < 31u32)
                            || (rt == PT_DEUT && water < 255u32)
                            || ((rt == PT_CBNW || rt == PT_DSTW) && water < 3u32)
                        {
                            part_ty[nid] = PT_NONE;
                            water += 1u32;
                            drank = true;
                        } else if rt == PT_SLTW && water > 0u32 {
                            part_ty[nid] = PT_NONE;
                            water -= 1u32;
                            drank = true;
                        } else if rt == PT_SPNG && water < 15u32 && part_life[nid] > 0i32 {
                            part_life[nid] -= 1i32;
                            water += 1u32;
                            drank = true;
                        }
                        if drank {
                            life = 0i32;
                        }
                        // Breeding two SEED pixels would rewrite both genomes
                        // (`tmp`..`tmp4`, `ctype`); not expressible per slot.
                    }
                }
            }
            ctype = (ctype & !(0xFFu32 << PLNT_LIFE)) | ((water & 0xFFu32) << PLNT_LIFE);
        }
    }

    part_ctype[id] = ctype;
    part_life[id] = life;
}
/// `PLNT.cpp:61-346`, `SLTW.cpp:64-67`: plant tree growth, seeds, vines and salt-water interactions.
#[cube]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_range_contains)] // `contains` is not available inside kernels
fn plnt_update(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_temp: &mut [f32],
    part_flags: &mut [u32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    down_dir: u32,
    seed: u32,
    id: usize,
    px: i32,
    py: i32,
) {
    let ctype = part_ctype[id];
    let temp = part_temp[id];

    if (ctype & 1u32) != 0u32 {
        // Tree growth mode (`PLNT.cpp:61-262`)
        if temp > 373.15f32 || temp < 223.15f32 {
            // Extreme temperature kills tree growth (`PLNT.cpp:65-75`)
            part_ctype[id] = 0u32;
            part_tmp[id] = 0i32;
            part_tmp2[id] = 0i32;
            part_tmp3[id] = 0i32;
            part_tmp4[id] = 0i32;
        } else if temp <= 343.15f32 && temp >= 278.15f32 {
            // Forward growth parks its spawn request in `life` (`< 0`) and
            // `FLAG_SPAWN_SELF`, because the neighbour helper picks the free
            // pixel and the child only exists after the drain. Branching never
            // comes back here: it creates directly through the queue above.
            if part_life[id] < 0i32 {
                let mut found_child = false;
                for n in 0u32..9u32 {
                    let dx = (n % 3u32) as i32 - 1i32;
                    let dy = (n / 3u32) as i32 - 1i32;
                    if (dx != 0i32 || dy != 0i32) && !found_child {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            if part_ty[nid] == PT_PLNT && (part_ctype[nid] & 1u32) == 0u32 {
                                let vigor = (ctype >> PLNT_LIFE) & 0xFFu32;
                                let is_diag = dx != 0i32 && dy != 0i32;
                                let scale = if is_diag { 7i32 } else { 10i32 };
                                part_life[nid] = if part_life[id] == -1i32 {
                                    scale * (vigor as i32)
                                } else {
                                    (-part_life[id]) - 1i32
                                };
                                part_ctype[nid] = ctype;
                                part_tmp[nid] = part_tmp[id];
                                part_tmp2[nid] = part_tmp2[id];
                                part_tmp3[nid] = part_tmp3[id];
                                part_tmp4[nid] = part_tmp4[id];

                                let phase = (ctype >> PLNT_PHASE) & 3u32;
                                part_ty[id] = if phase != 0u32 || vigor < 12u32 {
                                    PT_WOOD
                                } else {
                                    PT_GOO
                                };
                                part_ctype[id] = 0u32;
                                part_life[id] = 0i32;
                                part_tmp[id] = 0i32;
                                part_tmp2[id] = 0i32;
                                part_tmp3[id] = 0i32;
                                part_tmp4[id] = 0i32;
                                found_child = true;
                            }
                        }
                    }
                }
                if !found_child {
                    // Spawn failed or neighbor occupied: stop growing (`PLNT.cpp:230-260`)
                    if chance1(seed, 10u32) {
                        part_tmp[id] = PT_SEED as i32;
                        part_flags[id] |= FLAG_SPAWN_TMP;
                    }
                    part_ctype[id] = (ctype & (0x3Fu32 << 6u32)) | (7u32 << PLNT_DIR);
                    part_life[id] = 0i32;
                    part_tmp[id] = 0i32;
                    part_tmp2[id] = 0i32;
                }
            } else if !chance(seed, 9u32, 10u32) {
                // Growth tick (`PLNT.cpp:82-83`: throttled 1-in-10)
                let phase = (ctype >> PLNT_PHASE) & 3u32;
                let dir = (ctype >> PLNT_DIR) & 7u32;
                let vigor = (ctype >> PLNT_LIFE) & 0xFFu32;
                let prog = if phase == 0u32 {
                    part_tmp[id]
                } else if phase == 1u32 {
                    part_tmp2[id]
                } else if phase == 2u32 {
                    part_tmp3[id]
                } else {
                    part_tmp4[id]
                };
                let colour_bits = ctype & (0x3Fu32 << 6u32);

                let mut down = down_dir;
                if down > 7u32 {
                    down = 8u32;
                }

                let mut stopped = false;

                if vigor > 0u32 {
                    if part_life[id] <= 0i32 {
                        // Branching (`PLNT.cpp:110-165`)
                        let bmask = (prog & 0x1Fi32) as u32;
                        let mut phases = ((prog >> 5i32) & 0x3FFi32) as u32;

                        if bmask != 0u32 {
                            // `PLNT.cpp:120-165`: one child per set bit whose
                            // target pixel is free. TPT's `create_part` calls
                            // see each other, so every branch that finds a free
                            // pixel exists in the same frame; here each branch
                            // pushes its own queue record and the drain
                            // allocates them in the same order.
                            for phi in 0u32..5u32 {
                                if ((bmask >> phi) & 1u32) != 0u32 {
                                    let ndir = (dir + phi + 6u32) % 8u32;
                                    let nphase = phases & 3u32;
                                    let mut nlife = (2u32 * vigor) / 3u32;
                                    if down < 8u32 {
                                        if ndir == down
                                            || ndir == (down + 1u32) % 8u32
                                            || ndir == (down + 7u32) % 8u32
                                        {
                                            nlife = vigor / 3u32;
                                        } else if vigor > 4u32
                                            && (ndir == (down + 2u32) % 8u32
                                                || ndir == (down + 6u32) % 8u32)
                                        {
                                            nlife = vigor / 2u32;
                                        }
                                    }

                                    let nx = px + dir_x(ndir);
                                    let ny = py + dir_y(ndir);
                                    if pmap_at(pmap, xres, yres, nx, ny) == 0u32 {
                                        // `PLNT.cpp:144-155`: the child keeps the
                                        // colour bits and the programme words,
                                        // and a diagonal branch (`ndir % 2 == 1`)
                                        // is shortened by 1/sqrt(2).
                                        let child_ctype = ((nlife & 0xFFu32) << PLNT_LIFE)
                                            | colour_bits
                                            | ((ndir & 7u32) << PLNT_DIR)
                                            | ((nphase & 3u32) << PLNT_PHASE)
                                            | 1u32;
                                        let child_life = if ndir % 2u32 == 1u32 {
                                            7i32 * (nlife as i32)
                                        } else {
                                            10i32 * (nlife as i32)
                                        };
                                        request_create(
                                            create_queue,
                                            PT_PLNT,
                                            nx,
                                            ny,
                                            CR_SET_CTYPE
                                                | CR_SET_LIFE
                                                | CR_SET_TMP
                                                | CR_SET_TMP2
                                                | CR_SET_TMP3
                                                | CR_SET_TMP4,
                                            child_life,
                                            child_ctype,
                                            0.0f32,
                                            part_tmp[id],
                                            part_tmp2[id],
                                            part_tmp3[id],
                                            part_tmp4[id],
                                            0u32,
                                            0u32,
                                            0.0f32,
                                            0.0f32,
                                        );
                                    }
                                }
                                phases >>= 2u32;
                            }
                            // `PLNT.cpp:185-215`: a branch that fired turns
                            // into a stem (WOOD, or GOO for a thick phase-0
                            // trunk). TPT does this even when every target was
                            // occupied, so `stopped` stays false here.
                            part_ty[id] = if phase != 0u32 || vigor < 12u32 {
                                PT_WOOD
                            } else {
                                // `PLNT.cpp:200-215`: the thick trunk places a
                                // WOOD pixel on each side.
                                let mut left_dir = (dir + 6u32) % 8u32;
                                let mut right_dir = (dir + 2u32) % 8u32;
                                if dir % 2u32 == 1u32 {
                                    left_dir = (dir + 5u32) % 8u32;
                                    right_dir = (dir + 3u32) % 8u32;
                                }
                                request_create_plain(
                                    create_queue,
                                    PT_WOOD,
                                    px + dir_x(left_dir),
                                    py + dir_y(left_dir),
                                );
                                request_create_plain(
                                    create_queue,
                                    PT_WOOD,
                                    px + dir_x(right_dir),
                                    py + dir_y(right_dir),
                                );
                                PT_GOO
                            };
                            part_ctype[id] = 0u32;
                            part_life[id] = 0i32;
                            part_tmp[id] = 0i32;
                            part_tmp2[id] = 0i32;
                            part_tmp3[id] = 0i32;
                            part_tmp4[id] = 0i32;
                        } else {
                            stopped = true;
                        }
                    } else {
                        // Grow forward (`PLNT.cpp:166-215`). The child keeps
                        // the parent's growth timer and genome. The parent
                        // always becomes a stem, even when the target pixel is
                        // occupied: TPT only sets `stopped` when the growth
                        // timer has run out.
                        let nx = px + dir_x(dir);
                        let ny = py + dir_y(dir);
                        if pmap_at(pmap, xres, yres, nx, ny) == 0u32 {
                            let child_ctype = ((vigor & 0xFFu32) << PLNT_LIFE)
                                | colour_bits
                                | ((dir & 7u32) << PLNT_DIR)
                                | ((phase & 3u32) << PLNT_PHASE)
                                | 1u32;
                            request_create(
                                create_queue,
                                PT_PLNT,
                                nx,
                                ny,
                                CR_SET_CTYPE
                                    | CR_SET_LIFE
                                    | CR_SET_TMP
                                    | CR_SET_TMP2
                                    | CR_SET_TMP3
                                    | CR_SET_TMP4,
                                part_life[id],
                                child_ctype,
                                0.0f32,
                                part_tmp[id],
                                part_tmp2[id],
                                part_tmp3[id],
                                part_tmp4[id],
                                0u32,
                                0u32,
                                0.0f32,
                                0.0f32,
                            );
                        }
                        if phase != 0u32 || vigor < 12u32 {
                            part_ty[id] = PT_WOOD;
                        } else {
                            // `PLNT.cpp:200-215`: a thick trunk also places a
                            // WOOD pixel on each side of the growth direction.
                            part_ty[id] = PT_GOO;
                            let mut left_dir = (dir + 6u32) % 8u32;
                            let mut right_dir = (dir + 2u32) % 8u32;
                            if dir % 2u32 == 1u32 {
                                left_dir = (dir + 5u32) % 8u32;
                                right_dir = (dir + 3u32) % 8u32;
                            }
                            request_create_plain(
                                create_queue,
                                PT_WOOD,
                                px + dir_x(left_dir),
                                py + dir_y(left_dir),
                            );
                            request_create_plain(
                                create_queue,
                                PT_WOOD,
                                px + dir_x(right_dir),
                                py + dir_y(right_dir),
                            );
                        }
                        part_ctype[id] = 0u32;
                        part_life[id] = 0i32;
                        part_tmp[id] = 0i32;
                        part_tmp2[id] = 0i32;
                        part_tmp3[id] = 0i32;
                        part_tmp4[id] = 0i32;
                    }
                } else {
                    stopped = true;
                }

                if stopped {
                    // Shoot out a seed two pixels along the growth direction
                    // and inherit the genome (`PLNT.cpp:233-249`).
                    if chance1(seed, 10u32) {
                        let sdx = dir_x(dir);
                        let sdy = dir_y(dir);
                        request_create(
                            create_queue,
                            PT_SEED,
                            px + 2i32 * sdx,
                            py + 2i32 * sdy,
                            CR_SET_CTYPE
                                | CR_SET_TMP
                                | CR_SET_TMP2
                                | CR_SET_TMP3
                                | CR_SET_TMP4
                                | CR_SET_VX
                                | CR_SET_VY,
                            0i32,
                            colour_bits,
                            0.0f32,
                            part_tmp[id],
                            part_tmp2[id],
                            part_tmp3[id],
                            part_tmp4[id],
                            0u32,
                            0u32,
                            sdx as f32,
                            sdy as f32,
                        );
                    }
                    part_ctype[id] = colour_bits | (7u32 << PLNT_DIR);
                    part_life[id] = 0i32;
                    part_tmp[id] = 0i32;
                    part_tmp2[id] = 0i32;
                    part_tmp3[id] = 0i32;
                    part_tmp4[id] = 0i32;
                }
            }
        }
    } else {
        // Usual PLNT code (`PLNT.cpp:264-346`, `SLTW.cpp:64-67`)
        let mut killed = false;
        for n in 0u32..9u32 {
            let dx = (n % 3u32) as i32 - 1i32;
            let dy = (n / 3u32) as i32 - 1i32;
            if (dx != 0i32 || dy != 0i32) && !killed {
                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                if nv != 0u32 {
                    let nid = pmap_slot(nv);
                    let rt = part_ty[nid];
                    let nseed = hash_u32(seed ^ mix_index(nid as u32, 0x1B87_3593u32));

                    if rt == PT_SLTW && chance1(nseed, 40u32) {
                        part_ty[id] = PT_NONE;
                        killed = true;
                    } else if rt == PT_WATR && chance1(nseed, 50u32) {
                        part_ty[nid] = PT_PLNT;
                        part_ctype[nid] = part_ctype[id];
                        part_life[nid] = 0i32;
                    } else if rt == PT_LAVA && chance1(nseed, 50u32) {
                        part_ty[id] = PT_FIRE;
                        part_life[id] = 4i32;
                        killed = true;
                    } else if (rt == PT_SMKE || rt == PT_CO2) && chance1(nseed, 50u32) {
                        // `PLNT.cpp:292-300`: the plant eats the smoke and
                        // refills its life with a fresh 60..119 timer.
                        part_ty[nid] = PT_NONE;
                        part_life[id] = 60i32 + (nseed % 60u32) as i32;
                    } else if rt == PT_WOOD
                        && part_tmp[id] == 1i32
                        && chance1(nseed, 4u32)
                        && (part_flags[id] & FLAG_SPAWN_TMP) == 0u32
                    {
                        part_tmp[id] = PT_VINE as i32;
                        part_flags[id] |= FLAG_SPAWN_TMP;
                    }
                }
            }
        }

        if !killed {
            if part_life[id] == 2i32 && (part_flags[id] & FLAG_SPAWN_TMP) == 0u32 {
                part_tmp[id] = PT_O2 as i32;
                part_flags[id] |= FLAG_SPAWN_TMP;
                part_life[id] = 0i32;
            }
            if part_temp[id] > 350.0f32 && part_temp[id] > (part_tmp2[id] as f32) {
                part_tmp2[id] = part_temp[id] as i32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brec_brmt_reaction_temp_factor() {
        // `BRMT.cpp:53-55`: tf = int(1000 - ((523.15 - temp) * 2)), clamped >= 2.
        // The update only runs above 523.15 K, so the clamp only guards the
        // boundary itself.
        assert_eq!(brec_brmt_temp_factor_host(523.15f32), 1000);
        assert_eq!(brec_brmt_temp_factor_host(524.15f32), 1002);
        assert_eq!(brec_brmt_temp_factor_host(600.0f32), 1153);
        assert_eq!(brec_brmt_temp_factor_host(1100.0f32), 2153);
        assert_eq!(brec_brmt_temp_factor_host(520.0f32), 994);
    }

    #[test]
    fn vibr_bvbr_react_with_exot() {
        // `VIBR.cpp:177-182`: VIBR and BVBR touching EXOT when life == 0 have
        // a 1-in-25 chance to turn into EXOT.
        let reacts_exot = |ty: u32, life: i32, nty: u32, roll_seed: u32| {
            (ty == PT_VIBR || ty == PT_BVBR)
                && life == 0
                && nty == PT_EXOT
                && chance1(roll_seed, 25)
        };
        assert!(reacts_exot(PT_VIBR, 0, PT_EXOT, 50));
        assert!(reacts_exot(PT_BVBR, 0, PT_EXOT, 75));
        assert!(
            !reacts_exot(PT_VIBR, 1, PT_EXOT, 50),
            "exploding VIBR does not convert"
        );
        assert!(!reacts_exot(PT_VIBR, 0, PT_NONE, 50));
    }

    #[test]
    fn plnt_genome_extraction() {
        // ctype layout: bit 0: tree mode, bits 1-2: phase, bits 3-5: dir, bits 6-11: colour, bits 12-19: life
        let ctype = 1u32
            | (2u32 << PLNT_PHASE)
            | (4u32 << PLNT_DIR)
            | (0x1Fu32 << 6)
            | (20u32 << PLNT_LIFE);
        assert_eq!(ctype & 1, 1);
        assert_eq!((ctype >> PLNT_PHASE) & 3, 2);
        assert_eq!((ctype >> PLNT_DIR) & 7, 4);
        assert_eq!((ctype >> 6) & 0x3F, 0x1F);
        assert_eq!((ctype >> PLNT_LIFE) & 0xFF, 20);
    }
}
