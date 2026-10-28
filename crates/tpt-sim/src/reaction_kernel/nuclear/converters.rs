//! `CONV` and `WARP` converters.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{WORD_PROPERTIES, elem_word, hash_u32, mix_index, pmap_at, pmap_slot};
use super::PMAP_TYPE;

/// `CONV` latches onto the first element it touches, then converts everything
/// in its ring into that element (`CONV.cpp:51-102`).
///
/// Pixel parallel. `tmp` restricts the conversion to (or away from) one
/// element, `tmp2` picks which of the two, and the packed `ctype` carries the
/// `LIFE` extra type in its id field.
#[cube(launch_unchecked)]
pub(crate) fn converter_kernel(
    photons: &[u32],
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    _frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty == PT_CONV {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let raw = part_ctype[id];
                let ctype = raw & PMAP_TYPE;
                // `sd.IsElement(ctype)`: a defined, non-empty slot.
                let mut valid = false;
                if ctype > 0u32
                    && ctype < PT_NUM as u32
                    && ctype != PT_CONV
                    && elem_word(elem_words, ctype, WORD_PROPERTIES) != 0u32
                {
                    valid = true;
                }

                if !valid {
                    // `CONV.cpp:55-74`: latch onto the last real neighbour.
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let mut r = pmap_at(photons, xres, yres, px + dx, py + dy);
                            if r == 0u32 {
                                r = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            }
                            if r != 0u32 {
                                let rt = r & PMAP_TYPE;
                                if rt != PT_CLNE
                                    && rt != PT_PCLN
                                    && rt != PT_BCLN
                                    && rt != PT_STKM
                                    && rt != PT_PBCN
                                    && rt != PT_STKM2
                                    && rt != PT_CONV
                                    && rt < PT_NUM as u32
                                {
                                    let mut packed = rt;
                                    if rt == PT_LIFE {
                                        // `PMAPID(parts[ID(r)].ctype)`.
                                        let nid = pmap_slot(r);
                                        packed |= part_ctype[nid] << 9u32;
                                    }
                                    part_ctype[id] = packed;
                                }
                            }
                        }
                    }
                } else {
                    // `CONV.cpp:78-100`: convert the ring, honouring the
                    // `tmp`/`tmp2` restriction.
                    let tmp = part_tmp[id];
                    let mut restrict = 0u32;
                    if tmp > 0
                        && (tmp as u32) < PT_NUM as u32
                        && elem_word(elem_words, tmp as u32, WORD_PROPERTIES) != 0u32
                    {
                        restrict = tmp as u32;
                    }
                    let want_restricted = part_tmp2[id] == 1i32;

                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let mut r = pmap_at(photons, xres, yres, px + dx, py + dy);
                            let mut skip = false;
                            if restrict != 0u32 && r != 0u32 {
                                skip = ((r & PMAP_TYPE) == restrict) == want_restricted;
                            }
                            if r == 0u32 || skip {
                                r = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            }
                            if r != 0u32 {
                                let rt = r & PMAP_TYPE;
                                let mut drop = false;
                                if restrict != 0u32 {
                                    drop = (rt == restrict) == want_restricted;
                                }
                                if !drop && rt != PT_CONV && rt != PT_DMND && rt != ctype {
                                    let nid = pmap_slot(r);
                                    part_ty[nid] = ctype;
                                    if ctype == PT_LIFE {
                                        part_ctype[nid] = raw >> 9u32;
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

/// `WARP` swaps places with a random neighbour (`WARP.cpp:64-85`).
///
/// Pixel parallel. The `pmap` swap TPT performs is redundant here: the
/// movement pass rebuilds both maps from `part_x`/`part_y` every frame, so
/// swapping the positions (and writing the displaced particle's velocity) is
/// enough.
#[cube(launch_unchecked)]
pub(crate) fn warp_kernel(
    pmap: &[u32],
    part_life: &mut [i32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if (v & PMAP_TYPE) == PT_WARP {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 1_103_515_245u32));
                let mut traded = 0u32;

                for trade in 0u32..5u32 {
                    let rx =
                        (hash_u32(roll ^ mix_index(trade, 2_246_822_519u32)) % 3u32) as i32 - 1i32;
                    let ry =
                        (hash_u32(roll ^ mix_index(trade, 3_266_489_909u32)) % 3u32) as i32 - 1i32;
                    if (rx != 0i32 || ry != 0i32) && traded == 0u32 {
                        let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            let nty = nv & PMAP_TYPE;
                            if nty != PT_WARP
                                && nty != PT_STKM
                                && nty != PT_STKM2
                                && nty != PT_DMND
                                && nty != PT_CLNE
                                && nty != PT_BCLN
                                && nty != PT_PCLN
                            {
                                // TPT moves the warp onto the neighbour's
                                // exact float position but snaps the neighbour
                                // to the warp's integer pixel.
                                part_x[id] = part_x[nid];
                                part_y[id] = part_y[nid];
                                part_x[nid] = px as f32;
                                part_y[nid] = py as f32;
                                // `rng.between(-2, 1) + 0.5` / `rng.between(-2, 1)`.
                                part_vx[nid] = (hash_u32(roll ^ mix_index(trade, 0x85EB_CA6Bu32))
                                    % 4u32) as f32
                                    - 1.5f32;
                                part_vy[nid] = (hash_u32(roll ^ mix_index(trade, 0xC2B2_AE35u32))
                                    % 4u32) as f32
                                    - 2.0f32;
                                part_life[id] += 4i32;
                                traded = 1u32;
                            }
                        }
                    }
                }
            }
        }
    }
}
