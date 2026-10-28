//! Shield mechanics (`SHLD1`..`SHLD4`) and proton reactions (`PROT`).

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{
    CR_SET_LIFE, CR_SET_TEMP, GPU_ALLOC_NONE, WORD_EXPLOSIVE, WORD_FLAMMABLE, WORD_PROPERTIES,
    elem_word, gpu_alloc_slot, hash_u32, mix_index, pmap_at, pmap_slot, request_create,
};

use super::{PMAP_TYPE, chance, clampf, init_spawned_slot, seed_u32};

/// `Element_PROT::create` (`PROT.cpp:236-242`): `life = 680` and a random
/// 2 px/frame velocity along one of 36 directions.
#[cube]
fn prot_create_velocity(seed: u32, part_vx: &mut [f32], part_vy: &mut [f32], slot: usize) {
    let angle = (seed % 36u32) as f32 * 0.17453f32;
    part_vx[slot] = 2.0f32 * angle.cos();
    part_vy[slot] = 2.0f32 * angle.sin();
}

/// `PROT.cpp:180-200`: the element a proton turns into when its `tmp`
/// accumulator (the energy it absorbed from collisions) is above `tmp`.
///
/// Written as one expression: a `let mut` seeded from a host constant would be
/// typed as a plain `u32` and reject the cube-valued branches.
#[cube]
fn prot_transmute_target(tmp: i32, under: u32) -> u32 {
    if tmp > 500000i32 {
        if under == PT_TUNG { PT_AMTR } else { PT_SING }
    } else if tmp > 700i32 {
        PT_PLUT
    } else if tmp > 420i32 {
        PT_URAN
    } else if tmp > 310i32 {
        PT_POLO
    } else if tmp > 250i32 {
        PT_PLSM
    } else if tmp > 100i32 {
        PT_O2
    } else if tmp > 50i32 {
        PT_CO2
    } else {
        PT_NBLE
    }
}

/// `SHLD1`..`SHLD4`: spark growth, level upgrades and slow decay.
///
/// Pixel parallel over the neighbour ring. A spark grows a shield in *every*
/// empty pixel of its own 3x3 ring (`SHLD1.cpp:66-76` and the `SHLD2`..`SHLD4`
/// equivalents), so the creations go through the device queue — one record per
/// empty pixel — instead of one `FLAG_SPAWN_CLONE` per frame. `SHLD2`..`SHLD4`
/// growth seeds the plain shields with `life = 7`; a `SHLD1` grows them with the
/// registry default, exactly as on the reference side. The queue is drained
/// after the reaction step, so the new shields are live in the same frame.
#[cube(launch_unchecked)]
pub(crate) fn shield_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
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
            if ty == PT_SHLD1 || ty == PT_SHLD2 || ty == PT_SHLD3 || ty == PT_SHLD4 {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 2_654_435_761u32));
                // TPT rewrites `parts[i].type` / `life` mid-loop, so the later
                // neighbours of the same ring see the new state.
                let mut my_ty = ty;
                let mut my_life = part_life[id];

                for n in 0u32..9u32 {
                    let dx = (n % 3u32) as i32 - 1i32;
                    let dy = (n / 3u32) as i32 - 1i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv == 0u32 {
                            // `SHLD2.cpp:57-62`: a charged level-2 shield seeds
                            // a plain shield in every empty neighbour; 3 and 4
                            // decay into one and drop back a level
                            // (`SHLD3.cpp:57-66`, `SHLD4.cpp:57-66`).
                            if my_ty == PT_SHLD2 && my_life > 0i32 {
                                request_create(
                                    create_queue,
                                    PT_SHLD1,
                                    px + dx,
                                    py + dy,
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
                            } else if (my_ty == PT_SHLD3 && chance(roll, 1u32, 2500u32))
                                || (my_ty == PT_SHLD4 && chance(roll, 1u32, 5500u32))
                            {
                                request_create(
                                    create_queue,
                                    PT_SHLD1,
                                    px + dx,
                                    py + dy,
                                    CR_SET_LIFE,
                                    7i32,
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
                                my_ty = PT_SHLD2;
                            }
                        } else {
                            let nid = pmap_slot(nv);
                            let nty = part_ty[nid];
                            if nty == PT_SPRK && my_life == 0i32 {
                                // `SHLD1.cpp:59-76`: a spark hardens the shield
                                // and grows a fresh ring around it.
                                let mut hardened = 0u32;
                                if my_ty == PT_SHLD1 && chance(roll, 11u32, 40u32) {
                                    hardened = PT_SHLD2;
                                } else if my_ty == PT_SHLD2 && chance(roll, 1u32, 8u32) {
                                    hardened = PT_SHLD3;
                                } else if my_ty == PT_SHLD3 && chance(roll, 3u32, 500u32) {
                                    hardened = PT_SHLD4;
                                }
                                if hardened != 0u32 {
                                    my_ty = hardened;
                                    my_life = 7i32;
                                }
                                // `SHLD1.cpp:66-76`: every empty pixel of the
                                // spark's 3x3 ring gets a plain shield.
                                // `SHLD1` only: the registry default `life`.
                                let mut grow_life = 0i32;
                                let mut grow_mask = 0u32;
                                if my_ty != PT_SHLD1 {
                                    grow_life = 7i32;
                                    grow_mask = CR_SET_LIFE;
                                }
                                for m in 0u32..9u32 {
                                    let sdx = (m % 3u32) as i32 - 1i32;
                                    let sdy = (m / 3u32) as i32 - 1i32;
                                    let gx = px + dx + sdx;
                                    let gy = py + dy + sdy;
                                    if pmap_at(pmap, xres, yres, gx, gy) == 0u32 {
                                        request_create(
                                            create_queue,
                                            PT_SHLD1,
                                            gx,
                                            gy,
                                            grow_mask,
                                            grow_life,
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
                            } else if my_ty == PT_SHLD1
                                && nty == PT_SHLD3
                                && chance(roll, 2u32, 5u32)
                            {
                                // `SHLD1.cpp:78-83`: a level-3 shield recharges
                                // the level-1 shield it touches.
                                my_ty = PT_SHLD2;
                                my_life = 7i32;
                            } else if my_ty == PT_SHLD2
                                && nty == PT_SHLD4
                                && chance(roll, 2u32, 5u32)
                            {
                                // `SHLD2.cpp:83-88`.
                                my_ty = PT_SHLD3;
                                my_life = 7i32;
                            } else if my_ty == PT_SHLD3 && nty == PT_SHLD1 && my_life > 3i32 {
                                // `SHLD3.cpp:68-73`: a charged level-3 shield
                                // upgrades the level-1 shield next to it.
                                part_ty[nid] = PT_SHLD2;
                                part_life[nid] = 7i32;
                            } else if my_ty == PT_SHLD4 && nty == PT_SHLD2 && my_life > 3i32 {
                                // `SHLD4.cpp:69-74`.
                                part_ty[nid] = PT_SHLD3;
                                part_life[nid] = 7i32;
                            }
                        }
                    }
                }

                part_ty[id] = my_ty;
                part_life[id] = my_life;
            }
        }
    }
}

/// `PROT` reactions with the particle it sits on (`PROT.cpp:66-209`).
///
/// Pixel parallel over the `photons` layer. The `tmp` accumulator (energy
/// absorbed from collisions) asks the creation queue for the heavier element at
/// a random neighbour and kills the proton, exactly like `create_part(-1, ...)`
/// does. The `DEUT` arm of the same switch needs `air_pv` and a spawn request,
/// so it is [`prot_implode_kernel`]; the `SEED` genome swap is
/// [`prot_seed_kernel`].
#[cube(launch_unchecked)]
pub(crate) fn prot_under_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = photons[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_PROT {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 2_654_435_761u32));
                let under = pmap_at(pmap, xres, yres, px, py);
                // `under == 0` packs to type 0 (`PT_NONE`), so no extra guard
                // is needed for the empty case.
                let mut utype = under & PMAP_TYPE;
                let under_id = pmap_slot(under);
                // TPT returns early from the two kill branches, so the `tmp`
                // transmutation below must not resurrect the slot.
                let mut alive = true;

                if utype == PT_SPRK {
                    // `PROT.cpp:74-85`: `part_change_type` returns `false` for
                    // every accepted target and `true` only for `PT_NONE`,
                    // which it kills. The caller adds 44 frames of life and
                    // clears `ctype` on the `false` path, so a restored
                    // conductor comes back as a *live* spark.
                    let sparked = part_ctype[under_id];
                    if sparked == PT_NONE {
                        part_ty[under_id] = PT_NONE;
                        utype = 0u32;
                    } else {
                        if sparked < PT_NUM as u32 {
                            part_ty[under_id] = sparked;
                        }
                        part_life[under_id] += 44i32;
                        part_ctype[under_id] = 0u32;
                    }
                } else if utype == PT_LCRY {
                    // `PROT.cpp:104-111`: a powered liquid crystal converts the
                    // proton into a long-lived photon.
                    if part_life[under_id] > 5i32 && chance(roll, 1u32, 10u32) {
                        part_ty[id] = PT_PHOT;
                        part_life[id] *= 2i32;
                        part_ctype[id] = 0x3FFF_FFFFu32;
                    }
                } else if utype == PT_EXOT {
                    // `PROT.cpp:113-115`: exotic matter is charged by the proton.
                    part_ctype[under_id] = PT_PROT;
                } else if utype == PT_WIFI {
                    // `PROT.cpp:117-127`: a proton is a temperature remote for
                    // wireless transmitters.
                    let t = part_temp[id];
                    let change = if t < 173.15f32 {
                        -1000.0f32
                    } else if t < 273.15f32 {
                        -100.0f32
                    } else if t > 473.15f32 {
                        1000.0f32
                    } else if t > 373.15f32 {
                        100.0f32
                    } else {
                        0.0f32
                    };
                    part_temp[under_id] = clampf(part_temp[under_id] + change, MIN_TEMP, MAX_TEMP);
                } else if utype == PT_RSSS {
                    // `PROT.cpp:129-134`: protons annihilate solid residue.
                    part_ty[under_id] = PT_NONE;
                    part_ty[id] = PT_NONE;
                    alive = false;
                } else if utype == PT_NONE {
                    // `PROT.cpp:144-153`: outside any element the proton burns
                    // one life per frame.
                    if part_life[id] != 0i32 {
                        part_life[id] -= 1i32;
                        if part_life[id] == 0i32 {
                            part_ty[id] = PT_NONE;
                            alive = false;
                        }
                    }
                }

                if alive && part_tmp[id] != 0i32 {
                    // `PROT.cpp:180-200`: the accumulated collision energy turns
                    // the proton into a heavier element. TPT does not transmute
                    // in place: `create_part(-1, x + rng.between(-1, 1), ...)`
                    // puts a fresh particle next to the proton, gives it
                    // `temp = 100 * tmp` and kills the proton, so the product
                    // starts from its own defaults and the proton's slot frees.
                    let element = prot_transmute_target(part_tmp[id], utype);
                    let roll = hash_u32(roll ^ 0x27D4_EB2Fu32);
                    let dx = (roll % 3u32) as i32 - 1i32;
                    let dy = ((roll / 3u32) % 3u32) as i32 - 1i32;
                    request_create(
                        create_queue,
                        element,
                        px + dx,
                        py + dy,
                        CR_SET_TEMP,
                        0i32,
                        0u32,
                        clampf(100.0f32 * (part_tmp[id] as f32), MIN_TEMP, MAX_TEMP),
                        0i32,
                        0i32,
                        0i32,
                        0i32,
                        0u32,
                        0u32,
                        0.0f32,
                        0.0f32,
                    );
                    part_ty[id] = PT_NONE;
                    part_tmp[id] = 0i32;
                }
            }
        }
    }
}

/// `PROT.cpp:78-85` + `PROT.cpp:211-227`: a proton sitting on `DEUT` rolls the
/// implosion trigger and the deuterium becomes the first proton of the burst.
///
/// Pixel parallel over the `photons` layer. The branch belongs to the switch of
/// [`prot_under_kernel`], but the trigger reads `air_pv` and the burst spawns
/// from the device window, which would push that kernel over the storage-buffer
/// budget, so it lives in its own launch. It runs after [`prot_heat_kernel`]
/// (and after [`prot_fusion_kernel`]), so the heat pump still equalises the
/// deuterium's temperature before the slot is reused; TPT's write to the killed
/// slot is invisible, and this ordering keeps it that way. The gate uses the
/// type stored in the `photons` word, not `part_ty`, because TPT's switch runs
/// before the proton can transmute.
///
/// The burst itself is [`prot_burst_kernel`]: this pass only parks
/// `n = life/50` in the deuterium slot's `ctype` and the burst temperature in
/// its `tmp2`, the two values `DeutImplosion` hands to `create_part`.
#[cube(launch_unchecked)]
#[allow(clippy::manual_clamp, clippy::too_many_arguments)]
pub(crate) fn prot_implode_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_ctype: &mut [u32],
    part_tmp2: &mut [i32],
    air_pv: &mut [f32],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = photons[idx];
        if (v & PMAP_TYPE) == PT_PROT {
            let px = (idx as u32 % xres) as i32;
            let py = (idx as u32 / xres) as i32;
            let under = pmap_at(pmap, xres, yres, px, py);
            let deut = pmap_slot(under);
            // The maps are rebuilt once per frame, so an earlier pass may have
            // killed the deuterium (absorption, a phase change); TPT's live
            // `pmap` would already read empty at this pixel.
            if (under & PMAP_TYPE) == PT_DEUT && part_ty[deut] == PT_DEUT {
                let life = part_life[deut];
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                // `chance(-((int)pv - 4) + life/100, 200)`: `int()` truncates
                // towards zero and a non-positive numerator never hits.
                let odds = 4i32 - (air_pv[cidx] as i32) + (life / 100i32);
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 0x27D4_EB2Fu32));
                if odds > 0i32 && chance(roll, odds as u32, 200u32) {
                    // `DeutImplosion`: `n = life/50`, clamped to `1..340`.
                    let mut n = life / 50i32;
                    if n < 1i32 {
                        n = 1i32;
                    }
                    if n > 340i32 {
                        n = 340i32;
                    }
                    // `restrict_flt(deut.temp + deut.life * 500, MIN_TEMP,
                    // MAX_TEMP)`, evaluated before the pixel is reused.
                    let burst_temp = clampf(
                        part_temp[deut] + (life as f32) * 500.0f32,
                        MIN_TEMP,
                        MAX_TEMP,
                    );
                    // `create_part(-3, x, y, PT_PROT)` overwrites the pixel and
                    // `Element_PROT::create` seeds the new proton: the deuterium's
                    // slot carries the first proton of the burst. `tmp`/`tmp3`/
                    // `tmp4`/`dcolour`/`flags` are left for the burst pass, which
                    // rewrites the whole row.
                    part_ty[deut] = PT_PROT;
                    part_ctype[deut] = n as u32;
                    part_tmp2[deut] = u32::reinterpret(burst_temp) as i32;
                    // `pv -= (6.0f * CFDS) * n`, and `CFDS == 1` in this port.
                    air_pv[cidx] = clampf(
                        air_pv[cidx] - 6.0f32 * (n as f32),
                        MIN_PRESSURE,
                        MAX_PRESSURE,
                    );
                }
            }
        }
    }
}

/// `DeutImplosion`'s protons (`PROT.cpp:211-227`), spawned in the same frame
/// from the device spawn window.
///
/// Slot parallel. [`prot_implode_kernel`] leaves `n` in the deuterium slot's
/// `ctype` and the burst temperature in `tmp2`; this pass rewrites that slot as
/// the first proton and allocates the other `n - 1` from the window. TPT calls
/// `create_part(-3, x, y, PT_PROT)` `n` times, so every product starts from the
/// `PROT` defaults plus `Element_PROT::create`'s `life = 680` and 2 px/frame
/// velocity, and each call overwrites the pixel's `photons` word — the last
/// allocation owns it. The window allocations happen in this thread in that
/// order, so the port writes the word for the last proton explicitly instead of
/// leaving the winner to drain thread order.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prot_burst_kernel(
    photons: &mut [u32],
    part_ty: &mut [u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_dcolour: &mut [u32],
    part_flags: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    free_slots: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    num_slots: u32,
    frame_seed: u32,
) {
    let id = ABSOLUTE_POS;
    if id < num_slots as usize && part_ty[id] == PT_PROT && part_ctype[id] > 0u32 {
        let n = part_ctype[id];
        let temp = f32::reinterpret(part_tmp2[id] as u32);
        let px = i32::cast_from(part_x[id] + 0.5f32);
        let py = i32::cast_from(part_y[id] + 0.5f32);
        let seed = hash_u32(frame_seed ^ mix_index(id as u32, 0x1656_67B1u32));
        // The deuterium's slot carries the first proton of the burst.
        init_spawned_slot(
            PT_PROT,
            px,
            py,
            temp,
            0u32,
            0i32,
            seed,
            id,
            part_ty,
            part_x,
            part_y,
            part_life,
            part_ctype,
            part_temp,
            part_tmp,
            part_tmp2,
            part_tmp3,
            part_tmp4,
            part_dcolour,
            part_flags,
            part_vx,
            part_vy,
        );
        // The first proton is the marker slot; every successful window
        // allocation replaces it as the pixel's owner, so the last one wins
        // exactly like TPT's last `create_part(-3, ...)`.
        let mut last = seed_u32(n, id as u32);
        let mut c = seed_u32(n, 1u32);
        while c < n {
            let slot = gpu_alloc_slot(free_slots);
            if slot != GPU_ALLOC_NONE {
                let h = hash_u32(seed ^ mix_index(c, 0x9E37_79B9u32));
                init_spawned_slot(
                    PT_PROT,
                    px,
                    py,
                    temp,
                    0u32,
                    0i32,
                    h,
                    slot as usize,
                    part_ty,
                    part_x,
                    part_y,
                    part_life,
                    part_ctype,
                    part_temp,
                    part_tmp,
                    part_tmp2,
                    part_tmp3,
                    part_tmp4,
                    part_dcolour,
                    part_flags,
                    part_vx,
                    part_vy,
                );
                last = slot;
            }
            c += 1u32;
        }
        if px >= 0i32 && py >= 0i32 && (px as u32) < xres && (py as u32) < yres {
            let word = ((last + 1u32) << 9u32) | PT_PROT;
            photons[(py as u32 * xres + px as u32) as usize] = word;
        }
    }
}

/// Swap two genome words of a seed: `tmp`, `tmp2`, `tmp3`, `tmp4` in that order
/// (`PROT.cpp:150-160`'s `ptrs_tmp`).
///
/// Both words are read before either is written, so `j == k` degenerates to a
/// no-op exactly like TPT's pointer swap.
#[cube]
fn seed_swap(
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    slot: usize,
    j: u32,
    k: u32,
) {
    let mut vj = 0i32;
    if j == 0u32 {
        vj = part_tmp[slot];
    } else if j == 1u32 {
        vj = part_tmp2[slot];
    } else if j == 2u32 {
        vj = part_tmp3[slot];
    } else if j == 3u32 {
        vj = part_tmp4[slot];
    }
    let mut vk = 0i32;
    if k == 0u32 {
        vk = part_tmp[slot];
    } else if k == 1u32 {
        vk = part_tmp2[slot];
    } else if k == 2u32 {
        vk = part_tmp3[slot];
    } else if k == 3u32 {
        vk = part_tmp4[slot];
    }
    if j == 0u32 {
        part_tmp[slot] = vk;
    } else if j == 1u32 {
        part_tmp2[slot] = vk;
    } else if j == 2u32 {
        part_tmp3[slot] = vk;
    } else if j == 3u32 {
        part_tmp4[slot] = vk;
    }
    if k == 0u32 {
        part_tmp[slot] = vj;
    } else if k == 1u32 {
        part_tmp2[slot] = vj;
    } else if k == 2u32 {
        part_tmp3[slot] = vj;
    } else if k == 3u32 {
        part_tmp4[slot] = vj;
    }
}

/// `PROT.cpp:150-160`: a proton mutates the `SEED` it sits on by swapping two of
/// its four genome words.
///
/// Pixel parallel over the `photons` layer, and split out of
/// [`prot_under_kernel`] because the swap needs all four `tmp` arrays. The seed
/// under a proton is written only by that proton (no other thread of this pass
/// owns the slot), so the swap stays single-writer even though it is a
/// cross-slot update. It runs after [`prot_under_kernel`], so a proton that
/// transmuted this frame no longer matches and skips the swap.
#[cube(launch_unchecked)]
pub(crate) fn prot_seed_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = photons[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_PROT {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let under = pmap_at(pmap, xres, yres, px, py);
                if (under & PMAP_TYPE) == PT_SEED {
                    let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 0x2545_F491u32));
                    if chance(roll, 1u32, 2u32) {
                        let slot = pmap_slot(under);
                        let j = hash_u32(roll ^ 0x9E37_79B9u32) % 4u32;
                        let k = hash_u32(roll ^ 0x85EB_CA6Bu32) % 4u32;
                        seed_swap(part_tmp, part_tmp2, part_tmp3, part_tmp4, slot, j, k);
                    }
                }
            }
        }
    }
}

/// `PROT` ignition, temperature equalisation and conductor de-sparking
/// (`PROT.cpp:96-180`).
///
/// Pixel parallel over the `photons` layer. Every proton also bleeds a little
/// pressure out of its cell (`PROT.cpp:96`).
#[cube(launch_unchecked)]
pub(crate) fn prot_heat_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    elem_words: &[u32],
    air_pv: &mut [f32],
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = photons[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_PROT {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let under = pmap_at(pmap, xres, yres, px, py);
                air_pv[cidx] -= 0.003f32;
                if under != 0u32 {
                    let under_id = pmap_slot(under);
                    let utype = part_ty[under_id];
                    if utype != PT_NONE {
                        // `PROT.cpp:160-166`: everything the proton touches is
                        // pulled a quarter of the way to its own temperature
                        // (except wireless transmitters).
                        if utype != PT_WIFI {
                            let ut = part_temp[under_id];
                            part_temp[under_id] =
                                clampf(ut - (ut - part_temp[id]) / 4.0f32, MIN_TEMP, MAX_TEMP);
                        }
                        // `PROT.cpp:155-175`: the default branch of the switch —
                        // hot protons set off explosives and un-spark quiet
                        // conductors.
                        let handled = utype == PT_SPRK
                            || utype == PT_DEUT
                            || utype == PT_LCRY
                            || utype == PT_EXOT
                            || utype == PT_WIFI
                            || utype == PT_RSSS
                            || utype == PT_SEED;
                        if !handled {
                            let props = elem_word(elem_words, utype, WORD_PROPERTIES);
                            let flam = elem_word(elem_words, utype, WORD_FLAMMABLE) as i32;
                            let expl = elem_word(elem_words, utype, WORD_EXPLOSIVE) as i32;
                            if part_temp[id] > 773.15f32
                                && (flam > 0i32 || expl > 0i32 || utype == PT_BANG)
                            {
                                part_ty[under_id] = PT_FIRE;
                                // `restrict_flt(flammable * 5, MIN_TEMP, MAX_TEMP)`.
                                let add = clampf((flam * 5) as f32, MIN_TEMP, MAX_TEMP);
                                part_temp[under_id] += add;
                                air_pv[cidx] += 1.0f32;
                            } else if (props & PROP_CONDUCTS) != 0u32 && part_life[under_id] <= 4i32
                            {
                                part_life[under_id] += 40i32;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `PROT.cpp:191-206`: two protons that share a pixel collide head-on, and the
/// proton that lost the `photons` word adds both squared speeds to the
/// winner's `tmp` before dying.
///
/// Slot parallel over the pool: the displaced proton has no map entry, so the
/// pixel-parallel passes never reach it, while the winner's own thread reads
/// its own word back and skips the branch. Every loser adds into the winner's
/// `tmp` with an atomic `fetch_add`, so a third proton sharing the pixel cannot
/// lose an update to scheduling (the reference's per-particle updates can).
/// The accumulated energy is what [`prot_under_kernel`] turns into a heavier
/// element on the next frame, i.e. TPT's "collide with other protons to make
/// heavier materials"; the reference creates no particle here, so no spawn
/// request is raised.
#[cube(launch_unchecked)]
#[allow(clippy::approx_constant, clippy::too_many_arguments)]
pub(crate) fn prot_fusion_kernel(
    photons: &[u32],
    part_ty: &mut [u32],
    part_tmp: &mut [Atomic<i32>],
    part_x: &[f32],
    part_y: &[f32],
    part_vx: &[f32],
    part_vy: &[f32],
    xres: u32,
    yres: u32,
    num_slots: u32,
) {
    let id = ABSOLUTE_POS;
    if id < num_slots as usize {
        let ty = part_ty[id];
        if ty == PT_PROT {
            // Same pixel rounding as `scatter_spatial_maps_kernel`.
            let xi = i32::cast_from(part_x[id] + 0.5f32);
            let yi = i32::cast_from(part_y[id] + 0.5f32);
            if xi >= 0i32 && yi >= 0i32 && (xi as u32) < xres && (yi as u32) < yres {
                let ahead_word = photons[((yi as u32) * xres + (xi as u32)) as usize];
                let ahead = pmap_slot(ahead_word);
                // `ID(ahead) != i && TYP(ahead) == PT_PROT`. The map word is
                // what the pass sees; the live type also has to agree because
                // the maps are only rebuilt once per frame.
                if ahead != id && (ahead_word & PMAP_TYPE) == PT_PROT && part_ty[ahead] == PT_PROT {
                    let vx1 = part_vx[id];
                    let vy1 = part_vy[id];
                    let vx2 = part_vx[ahead];
                    let vy2 = part_vy[ahead];
                    let velocity1 = vx1 * vx1 + vy1 * vy1;
                    let velocity2 = vx2 * vx2 + vy2 * vy2;
                    // `atan2(-vy, vx)` in both directions, wrapped into
                    // `0..2pi` exactly like the reference.
                    let ny1 = -vy1;
                    let ny2 = -vy2;
                    let direction1 = ny1.atan2(vx1);
                    let direction2 = ny2.atan2(vx2);
                    let mut difference = direction1 - direction2;
                    if difference < 0.0f32 {
                        difference += 6.28319f32;
                    }
                    if difference > 3.12659f32
                        && difference < 3.15659f32
                        && velocity1 + velocity2 > 10.0f32
                    {
                        // The reference adds into `parts[ahead].tmp` while
                        // `ahead`'s own update may be running, so two losers
                        // sharing a pixel can lose an update. The atomic add
                        // keeps every contribution, which is the outcome TPT
                        // reports on the next frame.
                        part_tmp[ahead].fetch_add((velocity1 + velocity2) as i32);
                        part_ty[id] = PT_NONE;
                    }
                }
            }
        }
    }
}
