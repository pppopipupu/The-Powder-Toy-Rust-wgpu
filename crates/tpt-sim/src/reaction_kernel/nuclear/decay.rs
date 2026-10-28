//! Isotope decay (`ISOZ`/`ISZS`), noble-gas ionization (`NBLE`), virus
//! infection/cure (`VIRS`), and platinum catalysis (`PTNM`).

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{
    CR_NO_OCCUPANCY, CR_SET_CTYPE, CR_SET_TEMP, CR_SET_TMP, GPU_ALLOC_NONE, gpu_alloc_slot,
    hash_u32, mix_index, pmap_at, pmap_slot, rand_unit, request_create,
};
use super::{CHANNELS, PMAP_TYPE, chance, init_spawned_slot, seed_u32};
use crate::reaction_kernel::neut::rebuild_type_in_place;

/// `ISOZ`/`ISZS` decay into `PHOT`; `PRTI`/`PRTO` maintain their channel index.
///
/// Pixel parallel. The decay replaces the particle in place (`create_part(i)`),
/// so it is a type change plus the velocity TPT sets afterwards.
#[cube(launch_unchecked)]
pub(crate) fn isotope_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_pv: &[f32],
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
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 1_103_515_245u32));

                if ty == PT_ISOZ || ty == PT_ISZS {
                    // `ISOZ.cpp:69-77` / `ISZS.cpp:60-68`: a negative pressure
                    // accelerates the decay; the emitted photon keeps a random
                    // radial velocity of `128..355 / 127`.
                    let scaled = (-4.0f32 * air_pv[cidx]) as i32;
                    let mut n = 0u32;
                    if scaled > 0i32 {
                        n = scaled as u32;
                    }
                    if chance(roll, 1u32, 200u32)
                        && chance(hash_u32(roll ^ 0x85EB_CA6Bu32), n, 1000u32)
                    {
                        part_ty[id] = PT_PHOT;
                        let rr = (128u32 + (hash_u32(roll) % 228u32)) as f32 / 127.0f32;
                        // Degrees to radians; written as one literal so the
                        // kernel body needs no host constant.
                        let ang =
                            (hash_u32(roll ^ 0xC2B2_AE35u32) % 360u32) as f32 * 0.017_453_293f32;
                        part_vx[id] = rr * ang.cos();
                        part_vy[id] = rr * ang.sin();
                    }
                } else if ty == PT_PRTI || ty == PT_PRTO {
                    // `PRTI.cpp:64-68` / `PRTO.cpp:63-67`: one channel per
                    // 100 K above 73.15 K. The queue transfer itself lives in
                    // `reaction_kernel/portals/channels.rs`.
                    let mut ch = ((part_temp[id] - 73.15f32) / 100.0f32 + 1.0f32) as i32;
                    if ch >= CHANNELS {
                        ch = CHANNELS - 1i32;
                    } else if ch < 0i32 {
                        ch = 0i32;
                    }
                    part_tmp[id] = ch;
                }
            }
        }
    }
}

/// `NBLE.cpp:64-79`'s `create_part(-3, ...)` products are spawned by
/// [`nble_burst_kernel`] in the reference's order; this marker (parked in the
/// `CO2` slot's `tmp2`) tells that pass the burst is pending. `tmp3` carries the
/// burst temperature and `tmp4` the one-in-25 electron flag.
const NBLE_BURST_MARKER: i32 = 0x4E42;
/// `NBLE.cpp:76`: the photon's wavelength mask (`0xF800000`).
const NBLE_PHOT_CTYPE: u32 = 0x0F80_0000;

/// `NBLE`: a hot noble gas under pressure ionizes and bursts
/// (`NBLE.cpp:50-93`).
///
/// Pixel parallel. The burst is TPT's `create_part(-3, x, y, ...)` sequence:
/// the gas itself becomes `CO2` (a slot restart, so
/// [`change_type_in_place`] applies the `CO2` defaults) and the `NEUT`, the
/// one-in-25 `ELEC` and the `PHOT` are queued on the same pixel with
/// [`CR_NO_OCCUPANCY`], exactly as `create_part(-3, ...)` skips the map checks.
/// The reference's three same-pixel energy creations overwrite one
/// `photons` word, so only the last one it allocates stays mapped; the drain's
/// record threads run in parallel, so the port keeps the same particle set and
/// leaves the surviving map word to the drain (see the module deviations).
/// The optional `PLSM` lands in a random neighbour, gated by
/// `can_move[PT_PLSM][rt]`; on the device that table reduces to "empty or
/// `NBLE`" for every element that can appear in `pmap` (see below), plus the
/// `rt == PT_NBLE` case the reference spells out.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn nble_kernel(
    pmap: &[u32],
    spawn_defaults: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_flags: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
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
            if part_ty[id] == PT_NBLE {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 3_266_489_909u32));

                // `NBLE.cpp:54`: above 5273.15 K and 100 pressure units the gas
                // ionizes; `tmp & 1` is the flag `SPRK.cpp` reads.
                if part_temp[id] > 5273.15f32 && air_pv[cidx] > 100.0f32 {
                    part_tmp[id] |= 1i32;
                    if chance(roll, 1u32, 5u32) {
                        // `NBLE.cpp:58-62`: the gas becomes `CO2`; the slot
                        // restarts from the `CO2` defaults and only `temp` is
                        // written back afterwards.
                        let burst_temp = part_temp[id];
                        rebuild_type_in_place(
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
                            PT_CO2,
                        );
                        // `NBLE.cpp:64-79`: the `NEUT`, the optional `ELEC` and
                        // the `PHOT` are `create_part(-3, x, y, ...)` products
                        // that all overwrite the pixel's `photons` word. They
                        // are spawned from the device window by
                        // [`nble_burst_kernel`] in the reference's order, which
                        // parks the burst parameters here.
                        let mut elec = 0i32;
                        if chance(hash_u32(roll ^ 0x9E37_79B9u32), 1u32, 25u32) {
                            elec = 1i32;
                        }
                        part_tmp2[id] = NBLE_BURST_MARKER;
                        part_tmp3[id] = u32::reinterpret(burst_temp) as i32;
                        part_tmp4[id] = elec;
                        // `NBLE.cpp:80-88`: `can_move[PT_PLSM][rt]` is only
                        // true for an empty pixel (every element in `pmap`
                        // weighs at least as much as the weight-1 plasma) and
                        // for the `NBLE` special case, whose pixel the burst
                        // has already turned into `CO2` above.
                        let rx = px + (hash_u32(roll ^ 0x85EB_CA6Bu32) % 3u32) as i32 - 1i32;
                        let ry = py + (hash_u32(roll ^ 0xC2B2_AE35u32) % 3u32) as i32 - 1i32;
                        let rt = pmap_at(pmap, xres, yres, rx, ry) & PMAP_TYPE;
                        // The `(0, 0)` offset hits the pixel that already
                        // became `CO2`, which `can_move` rejects; the port's
                        // `pmap` word still names `NBLE`, so it is skipped here
                        // instead.
                        if (rx != px || ry != py) && (rt == PT_NONE || rt == PT_NBLE) {
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
                        // `NBLE.cpp:90-91`: the burst heats the gas by
                        // 1750..2249 K and raises the cell pressure by 50.
                        part_temp[id] = burst_temp
                            + 1750.0f32
                            + (hash_u32(roll ^ 0x27D4_EB2Fu32) % 500u32) as f32;
                        air_pv[cidx] += 50.0f32;
                    }
                }
            }
        }
    }
}

/// `NBLE.cpp:64-79`: the burst's `NEUT`, optional `ELEC` and `PHOT`, spawned
/// in the same frame from the device spawn window.
///
/// Slot parallel over the pool. [`nble_kernel`] parks the burst in the `CO2`
/// slot (`tmp2` marker, `tmp3` temperature bits, `tmp4` electron flag); this
/// pass allocates the products in the reference's call order. TPT's
/// `create_part(-3, x, y, ...)` overwrites the pixel's `photons` word for each
/// product, so the last successful one — the `PHOT` — owns it. Allocating in
/// one thread makes that order a fact of the kernel instead of a race between
/// the drain's record threads: the winner is fixed by creation order (and thus
/// by slot allocation order), not by dispatch order.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn nble_burst_kernel(
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
    if id < num_slots as usize && part_ty[id] == PT_CO2 && part_tmp2[id] == NBLE_BURST_MARKER {
        let temp = f32::reinterpret(part_tmp3[id] as u32);
        let elec = part_tmp4[id] != 0i32;
        let px = i32::cast_from(part_x[id] + 0.5f32);
        let py = i32::cast_from(part_y[id] + 0.5f32);
        let seed = hash_u32(frame_seed ^ mix_index(id as u32, 0x4E42_0001u32));
        // The last successful product owns the pixel's energy word.
        let mut last = seed_u32(id as u32, GPU_ALLOC_NONE);
        let mut last_ty = seed_u32(id as u32, PT_NONE);
        // `NBLE.cpp:64-66`: `create_part(-3, x, y, PT_NEUT)`.
        let neut = gpu_alloc_slot(free_slots);
        if neut != GPU_ALLOC_NONE {
            init_spawned_slot(
                PT_NEUT,
                px,
                py,
                temp,
                0u32,
                0i32,
                seed,
                neut as usize,
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
            last = neut;
            last_ty = PT_NEUT;
        }
        if elec {
            // `NBLE.cpp:67-72`: one burst in 25 also frees an electron.
            let el = gpu_alloc_slot(free_slots);
            if el != GPU_ALLOC_NONE {
                init_spawned_slot(
                    PT_ELEC,
                    px,
                    py,
                    temp,
                    0u32,
                    0i32,
                    hash_u32(seed ^ 0x9E37_79B9u32),
                    el as usize,
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
                last = el;
                last_ty = PT_ELEC;
            }
        }
        // `NBLE.cpp:73-79`: the photon carries the burst temperature, the
        // fixed wavelength and `tmp = 1`.
        let phot = gpu_alloc_slot(free_slots);
        if phot != GPU_ALLOC_NONE {
            init_spawned_slot(
                PT_PHOT,
                px,
                py,
                temp,
                NBLE_PHOT_CTYPE,
                1i32,
                hash_u32(seed ^ 0x85EB_CA6Bu32),
                phot as usize,
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
            last = phot;
            last_ty = PT_PHOT;
        }
        if last != GPU_ALLOC_NONE
            && px >= 0i32
            && py >= 0i32
            && (px as u32) < xres
            && (py as u32) < yres
        {
            photons[(py as u32 * xres + px as u32) as usize] = ((last + 1u32) << 9u32) | last_ty;
        }
        // The burst is done; the row reset pass restores the CO2 defaults.
        part_tmp2[id] = 0i32;
    }
}

/// `VIRS`/`VRSS`/`VRSG` infection, cure countdown and death timer
/// (`VIRS.cpp:55-149`; `VRSS`/`VRSG` share the update).
///
/// Pixel parallel. `tmp2` stores the original type, `tmp3` the remaining cure
/// frames and `tmp4` the remaining life of the infection. The only cross-slot
/// writes are the infection itself (`tmp2`/`tmp3`/`tmp4` plus the type of a
/// neighbour that is *not* a virus, so no other thread of this pass owns it)
/// and the `SOAP` kill.
#[cube(launch_unchecked)]
pub(crate) fn virus_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_temp: &mut [f32],
    air_pv: &[f32],
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
            if ty == PT_VIRS || ty == PT_VRSS || ty == PT_VRSG {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let mut rndstore = hash_u32(frame_seed ^ mix_index(idx as u32, 2_246_822_507u32));

                // `VIRS.cpp:57-71`: a cured virus counts down and turns back
                // into the element it was, and never reaches the ring below.
                if part_tmp3[id] != 0i32 {
                    if (rndstore & 0x1u32) == 0u32 {
                        part_tmp3[id] -= 1i32;
                    }
                    if part_tmp3[id] == 0i32 {
                        let mut orig = 0u32;
                        if part_tmp2[id] > 0i32 && (part_tmp2[id] as u32) < PT_NUM as u32 {
                            orig = part_tmp2[id] as u32;
                        }
                        part_ty[id] = orig;
                        part_tmp2[id] = 0i32;
                        part_tmp4[id] = 0i32;
                    }
                } else {
                    // `VIRS.cpp:73-81`: the infection slowly dies out. The
                    // pre-decrement only happens on one roll in eight, and a
                    // timer that reaches zero kills the virus outright.
                    let mut dead = false;
                    if part_tmp4[id] != 0i32 {
                        if (rndstore & 0x7u32) == 0u32 {
                            part_tmp4[id] -= 1i32;
                            if part_tmp4[id] <= 0i32 {
                                part_ty[id] = PT_NONE;
                                dead = true;
                            }
                        }
                        rndstore >>= 3u32;
                    }

                    if !dead {
                        // `surround_space`: the number of empty neighbours the
                        // plasma branch needs.
                        let mut space = 0u32;
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32)
                                && pmap_at(pmap, xres, yres, px + dx, py + dy) == 0u32
                            {
                                space += 1u32;
                            }
                        }

                        // TPT returns from the whole update on the first three
                        // branches below, so `stop` freezes the ring.
                        let mut stop = false;
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if dx != 0i32 || dy != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 && !stop {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    let nvirus = nty == PT_VIRS || nty == PT_VRSS || nty == PT_VRSG;
                                    if part_tmp3[nid] != 0i32 && nvirus {
                                        // `VIRS.cpp:92-96`: the cure spreads out
                                        // from a virus that is already curing.
                                        let add = if (rndstore & 0x3u32) != 0u32 {
                                            2i32
                                        } else {
                                            1i32
                                        };
                                        part_tmp3[id] = part_tmp3[nid] + add;
                                        stop = true;
                                    } else if nty == PT_SOAP {
                                        // `VIRS.cpp:97-102`: soap cures the
                                        // virus and is usually consumed.
                                        part_tmp3[id] += 10i32;
                                        if (rndstore & 0x3u32) == 0u32 {
                                            part_ty[nid] = PT_NONE;
                                        }
                                        stop = true;
                                    } else if nty == PT_PLSM {
                                        // `VIRS.cpp:107-115`: plasma burns the
                                        // virus out and spreads into the
                                        // vacated pixel. `int(pv)` truncates
                                        // towards zero, so a strong vacuum makes
                                        // `chance` impossible exactly as in TPT.
                                        let odds = 10i32 + air_pv[cidx] as i32;
                                        if space > 0u32
                                            && odds > 0i32
                                            && chance(rndstore, odds as u32, 100u32)
                                        {
                                            part_ty[id] = PT_PLSM;
                                            stop = true;
                                        }
                                    } else if !nvirus && nty != PT_DMND && nty != PT_BASE {
                                        // `VIRS.cpp:117-131`: one neighbour in
                                        // eight is converted; `tmp2` keeps what
                                        // it was so the cure can restore it and
                                        // `tmp4` carries the parent's timer.
                                        if (rndstore & 0x7u32) == 0u32 {
                                            part_tmp2[nid] = nty as i32;
                                            part_tmp3[nid] = 0i32;
                                            let mut child4 = 0i32;
                                            if part_tmp4[id] != 0i32 {
                                                child4 = part_tmp4[id] + 1i32;
                                            }
                                            part_tmp4[nid] = child4;
                                            let ntemp = part_temp[nid];
                                            if ntemp < 305.0f32 {
                                                part_ty[nid] = PT_VRSS;
                                            } else if ntemp > 673.0f32 {
                                                part_ty[nid] = PT_VRSG;
                                            } else {
                                                part_ty[nid] = PT_VIRS;
                                            }
                                        }
                                        rndstore >>= 3u32;
                                    } else if (pmap_at(photons, xres, yres, px + dx, py + dy)
                                        & PMAP_TYPE)
                                        == PT_PROT
                                    {
                                        // `VIRS.cpp:134-138`: protons make the
                                        // infection last forever.
                                        part_tmp4[id] = 0i32;
                                    }
                                }
                            } else {
                                // `VIRS.cpp:141-143`: the roll is re-seeded
                                // once, at the centre of the ring.
                                rndstore =
                                    hash_u32(rndstore ^ mix_index(idx as u32, 3_266_489_909u32));
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `PTNM` fast conduction, `WTRV` catalysis and the single-neighbour reactions
/// (`PTNM.cpp:143-235`).
///
/// Pixel parallel. TPT keeps the first `WTRV` neighbour in the `wtrv1_id` local
/// across the two passes; the port parks it in `tmp3` (unused by `PTNM`) for the
/// duration of this kernel and clears it before the single-neighbour loop, so no
/// other pass ever sees the scratch value.
#[cube(launch_unchecked)]
pub(crate) fn ptnm_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp3: &mut [i32],
    air_pv: &[f32],
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
            if ty == PT_PTNM {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 2_246_822_519u32));

                // `PTNM.cpp:148-162`: fast conduction four pixels out, like
                // `GOLD`; the spark is absorbed and platinum becomes the spark.
                if part_life[id] == 0i32 {
                    for j in 0u32..4u32 {
                        let rx = if j == 0u32 {
                            -4i32
                        } else if j == 1u32 {
                            4i32
                        } else {
                            0i32
                        };
                        let ry = if j == 2u32 {
                            -4i32
                        } else if j == 3u32 {
                            4i32
                        } else {
                            0i32
                        };
                        let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            if part_ty[nid] == PT_SPRK
                                && part_life[nid] != 0i32
                                && part_life[nid] < 4i32
                            {
                                part_ty[id] = PT_SPRK;
                                part_life[id] = 4i32;
                                part_ctype[id] = PT_PTNM;
                            }
                        }
                    }
                }

                if part_ty[id] == PT_PTNM {
                    // `PTNM.cpp:52-74` `wtrv_reactions`: a hot `BCOL` next to a
                    // hot `WTRV` under pressure turns into `OIL` and eats the
                    // steam. The first `WTRV` neighbour is remembered in `tmp3`
                    // (0 = none, otherwise slot + 1).
                    part_tmp3[id] = 0i32;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 && (nv & PMAP_TYPE) == PT_WTRV && part_tmp3[id] == 0i32 {
                                part_tmp3[id] = (pmap_slot(nv) + 1usize) as i32;
                            }
                        }
                    }
                    if part_tmp3[id] != 0i32 {
                        let wtrv = (part_tmp3[id] - 1i32) as usize;
                        if part_ty[wtrv] == PT_WTRV && part_temp[wtrv] > 473.15f32 {
                            for n in 0u32..9u32 {
                                let dx = (n % 3u32) as i32 - 1i32;
                                let dy = (n / 3u32) as i32 - 1i32;
                                if (dx != 0i32 || dy != 0i32) && part_ty[wtrv] == PT_WTRV {
                                    let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                    if nv != 0u32 {
                                        let nid = pmap_slot(nv);
                                        if part_ty[nid] == PT_BCOL
                                            && part_temp[nid] > 473.15f32
                                            && air_pv[((((py + dy) as u32) / CELL) * xcells
                                                + (((px + dx) as u32) / CELL))
                                                as usize]
                                                > 7.0f32
                                        {
                                            part_ty[nid] = PT_OIL;
                                            part_ty[wtrv] = PT_NONE;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    part_tmp3[id] = 0i32;

                    // `PTNM.cpp:166-235`: the single-neighbour reactions. The
                    // probability ramps quadratically from 0 % at 0 C to 100 %
                    // at 1500 C.
                    let mut prob = part_temp[id] / (273.15f32 + 1500.0f32);
                    if prob > 1.0f32 {
                        prob = 1.0f32;
                    }
                    prob *= prob;

                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let nlife = part_life[nid];
                                let npv = air_pv[((((py + dy) as u32) / CELL) * xcells
                                    + (((px + dx) as u32) / CELL))
                                    as usize];

                                // `PTNM.cpp:176-190`: an uncharged shield in
                                // contact grows one level, whatever its own
                                // update would have done.
                                if nlife == 0i32
                                    && (nty == PT_SHLD1 || nty == PT_SHLD2 || nty == PT_SHLD3)
                                {
                                    let next = if nty == PT_SHLD1 {
                                        PT_SHLD2
                                    } else if nty == PT_SHLD2 {
                                        PT_SHLD3
                                    } else {
                                        PT_SHLD4
                                    };
                                    part_ty[nid] = next;
                                    part_life[nid] = 7i32;
                                } else if nty == PT_ISZS || nty == PT_ISOZ {
                                    // `PTNM.cpp:192-197`: the isotope becomes
                                    // plutonium and flashes a photon at its own
                                    // pixel (`create_part(-3, ...)`), one per
                                    // isotope neighbour.
                                    part_ty[nid] = PT_PLUT;
                                    request_create(
                                        create_queue,
                                        PT_PHOT,
                                        px + dx,
                                        py + dy,
                                        CR_NO_OCCUPANCY,
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
                                } else if rand_unit(hash_u32(roll ^ n)) <= prob {
                                    if nty == PT_GAS {
                                        // `PTNM.cpp:220-228`: hot gas under
                                        // pressure condenses into insulation.
                                        if part_temp[nid] >= 473.15f32 && npv > 2.0f32 {
                                            part_ty[nid] = PT_INSL;
                                            part_temp[id] += 60.0f32;
                                        }
                                    } else if nty == PT_BREC {
                                        if part_temp[nid] > 1273.15f32 && npv > 50.0f32 {
                                            part_ty[nid] = PT_EXOT;
                                            part_temp[nid] -= 30.0f32;
                                            part_temp[id] -= 30.0f32;
                                        }
                                    } else if nty == PT_SMKE {
                                        part_ty[nid] = PT_CO2;
                                    } else if nty == PT_RSST {
                                        part_ty[nid] = PT_BIZR;
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

/// `PTNM` hydrogen catalysis (`PTNM.cpp:88-133`).
///
/// Pixel parallel. The reactions need a second pass so the first hydrogen
/// neighbour (`hygn1_id`) can be found without blowing the buffer budget; the
/// port parks it in `tmp3` (unused by `PTNM`) and clears it when a reaction
/// fires. The products TPT creates with `create_part(-3, x+rx, y+ry, ...)` go
/// through the queue at that exact pixel, and the cold-fusion pressure bump is
/// booked in the neighbour's cell like the reference.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ptnm_hydrogen_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp3: &mut [i32],
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
            if part_ty[id] == PT_PTNM {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 1_103_515_245u32));

                part_tmp3[id] = 0i32;
                for n in 0u32..9u32 {
                    let dx = (n % 3u32) as i32 - 1i32;
                    let dy = (n / 3u32) as i32 - 1i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 && (nv & PMAP_TYPE) == PT_H2 && part_tmp3[id] == 0i32 {
                            part_tmp3[id] = (pmap_slot(nv) + 1usize) as i32;
                        }
                    }
                }

                if part_tmp3[id] != 0i32 {
                    let hygn = (part_tmp3[id] - 1i32) as usize;
                    if part_ty[hygn] == PT_H2 {
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32) && part_tmp3[id] != 0i32 {
                                let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                if nv != 0u32 {
                                    let nid = pmap_slot(nv);
                                    let nty = part_ty[nid];
                                    if nid != hygn {
                                        if nty == PT_DESL {
                                            // `PTNM.cpp:88-93`: diesel + hydrogen
                                            // over platinum gives water + oil.
                                            part_ty[nid] = PT_WATR;
                                            part_ty[hygn] = PT_OIL;
                                            part_tmp3[id] = 0i32;
                                        } else if nty == PT_O2 && part_life[id] == 0i32 {
                                            // `PTNM.cpp:95-108`: hydrogen + oxygen
                                            // on platinum detonates into heavy
                                            // water and sparks the platinum.
                                            part_ty[nid] = PT_DSTW;
                                            part_ty[hygn] = PT_DSTW;
                                            part_temp[nid] += 5.0f32;
                                            part_temp[hygn] += 5.0f32;
                                            part_ctype[id] = PT_PTNM;
                                            part_life[id] = 4i32;
                                            part_ty[id] = PT_SPRK;
                                            part_tmp3[id] = 0i32;
                                        } else if nty == PT_H2
                                            && chance(roll, 1u32, 1000u32)
                                            && part_temp[nid] > 773.15f32
                                            && part_temp[hygn] > 773.15f32
                                        {
                                            // `PTNM.cpp:110-133`: cold fusion.
                                            part_ty[nid] = PT_NBLE;
                                            part_ty[hygn] = PT_NEUT;
                                            part_temp[nid] += 1000.0f32;
                                            part_temp[hygn] += 1000.0f32;
                                            // `sim->pv[(y+ry)/CELL][(x+rx)/CELL]
                                            // += 10.0f`, booked in the fused
                                            // neighbour's cell.
                                            let ncidx = (((py + dy) as u32 / CELL) * xcells
                                                + ((px + dx) as u32 / CELL))
                                                as usize;
                                            air_pv[ncidx] += 10.0f32;
                                            // `create_part(-3, x+rx, y+ry,
                                            // PT_PHOT)` with `ctype = 0x7C0000`,
                                            // `temp` and `tmp = 1`.
                                            request_create(
                                                create_queue,
                                                PT_PHOT,
                                                px + dx,
                                                py + dy,
                                                CR_NO_OCCUPANCY
                                                    | CR_SET_CTYPE
                                                    | CR_SET_TEMP
                                                    | CR_SET_TMP,
                                                0i32,
                                                0x007C_0000u32,
                                                part_temp[nid],
                                                1i32,
                                                0i32,
                                                0i32,
                                                0i32,
                                                0u32,
                                                0u32,
                                                0.0f32,
                                                0.0f32,
                                            );
                                            if chance(hash_u32(roll), 1u32, 10u32) {
                                                // `create_part(-3, x+rx, y+ry,
                                                // PT_ELEC)` at the same pixel.
                                                request_create(
                                                    create_queue,
                                                    PT_ELEC,
                                                    px + dx,
                                                    py + dy,
                                                    CR_NO_OCCUPANCY | CR_SET_TEMP,
                                                    0i32,
                                                    0u32,
                                                    part_temp[nid],
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
                                            part_tmp3[id] = 0i32;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Host mirror of [`nble_burst_kernel`]'s product order
    /// (`NBLE.cpp:64-79`): the `NEUT` first, the one-in-25 `ELEC` next, the
    /// `PHOT` last. TPT's `create_part(-3, ...)` overwrites the pixel's
    /// `photons` word for each product, so the last one — the photon — owns it.
    fn nble_products(elec: bool) -> Vec<u32> {
        let mut out = vec![PT_NEUT];
        if elec {
            out.push(PT_ELEC);
        }
        out.push(PT_PHOT);
        out
    }

    #[test]
    fn nble_burst_maps_the_photon_last() {
        assert_eq!(nble_products(false), vec![PT_NEUT, PT_PHOT]);
        assert_eq!(nble_products(true), vec![PT_NEUT, PT_ELEC, PT_PHOT]);
        // The map-word winner is the last product, whatever the dispatch order
        // of the old per-record drain would have been.
        assert_eq!(*nble_products(false).last().unwrap(), PT_PHOT);
        assert_eq!(*nble_products(true).last().unwrap(), PT_PHOT);
        assert_eq!(NBLE_PHOT_CTYPE, 0x0F80_0000, "`NBLE.cpp:76` wavelength");
    }

    /// Host mirror of [`ptnm_kernel`]'s isotope branch (`PTNM.cpp:192-197`):
    /// every `ISZS`/`ISOZ` neighbour becomes `PLUT` and gets its own photon
    /// record on its own pixel.
    fn ptnm_isotope_photons(neighbours: &[u32]) -> Vec<(u32, usize)> {
        neighbours
            .iter()
            .enumerate()
            .filter(|&(_, &ty)| ty == PT_ISZS || ty == PT_ISOZ)
            .map(|(i, _)| (PT_PHOT, i))
            .collect()
    }

    #[test]
    fn ptnm_flashes_one_photon_per_isotope_neighbour() {
        let photons = ptnm_isotope_photons(&[PT_ISZS, PT_DUST, PT_ISOZ, PT_ISZS]);
        assert_eq!(photons.len(), 3, "one photon per isotope neighbour");
        assert_eq!(
            photons,
            vec![(PT_PHOT, 0), (PT_PHOT, 2), (PT_PHOT, 3)],
            "each photon stays on its own neighbour's pixel"
        );
        assert!(ptnm_isotope_photons(&[PT_DUST, PT_WATR]).is_empty());
    }

    #[test]
    fn ptnm_cold_fusion_bumps_pressure_by_ten() {
        // `PTNM.cpp:120`: `sim->pv[(y+ry)/CELL][(x+rx)/CELL] += 10.0f`.
        let mut pv = 3.0f32;
        pv += 10.0f32;
        assert_eq!(pv, 13.0f32);
    }

    #[test]
    fn portal_channels_follow_the_temperature() {
        // `PRTI.cpp:64-68`: `(temp - 73.15)/100 + 1`, clamped to `0..CHANNELS-1`.
        assert_eq!(CHANNELS, 101);
        assert_eq!(channel_of(0.0f32), 0);
        assert_eq!(channel_of(73.15f32), 1);
        assert_eq!(channel_of(173.15f32), 2);
        assert_eq!(channel_of(10073.15f32), CHANNELS - 1);
        assert_eq!(channel_of(-500.0f32), 0);
    }

    fn channel_of(temp: f32) -> i32 {
        let mut ch = ((temp - 73.15f32) / 100.0f32 + 1.0f32) as i32;
        if ch >= CHANNELS {
            ch = CHANNELS - 1;
        } else if ch < 0 {
            ch = 0;
        }
        ch
    }

    /// Host mirror of the `tmp3` cure countdown and the `tmp4` death timer in
    /// [`virus_kernel`] (`VIRS.cpp:57-81`).
    ///
    /// Returns `(tmp3, tmp4, dead)`.
    fn virus_timers(tmp3: i32, tmp4: i32, roll: u32) -> (i32, i32, bool) {
        if tmp3 != 0 {
            let mut t3 = tmp3;
            if roll & 1 == 0 {
                t3 -= 1;
            }
            return (t3, tmp4, false);
        }
        let mut t4 = tmp4;
        let mut dead = false;
        if t4 != 0 && roll & 7 == 0 {
            t4 -= 1;
            if t4 <= 0 {
                dead = true;
            }
        }
        (0, t4, dead)
    }

    #[test]
    fn virus_cure_counts_down_and_cures() {
        // A cured virus drops one frame on every even roll...
        assert_eq!(virus_timers(4, 250, 0), (3, 250, false));
        // ...and holds on an odd one.
        assert_eq!(virus_timers(4, 250, 1), (4, 250, false));
        // Reaching zero is the cure: `tmp4` is cleared with it and the virus
        // never reaches the spreading ring that frame.
        assert_eq!(virus_timers(1, 250, 2), (0, 250, false));
    }

    #[test]
    fn virus_death_timer_only_ticks_one_in_eight() {
        // The pre-decrement is gated on the low three bits.
        assert_eq!(virus_timers(0, 5, 8), (0, 4, false));
        assert_eq!(virus_timers(0, 5, 9), (0, 5, false));
        // A timer that reaches zero kills the virus.
        assert_eq!(virus_timers(0, 1, 0), (0, 0, true));
        // A proton resets `tmp4` to 0, which means "never dies".
        assert_eq!(virus_timers(0, 0, 0), (0, 0, false));
    }
}
