use cubecl::prelude::*;

use super::temp_channel;
use super::{
    FLAG_PORTAL_HELD_U32, MOVABLE_MASK, PROP_CONDUCTS_U32, PT_FIGH_U32, PT_INST_U32, PT_NONE_U32,
    PT_NUM_U32, PT_PRTI_U32, PT_PRTO_U32, PT_SOAP_U32, PT_SPRK_U32, PT_STKM_U32, PT_STKM2_U32,
    PT_STOR_U32, PT_WIFI_U32,
};
use crate::reaction_kernel::shared::{
    WORD_PROPERTIES, elem_word, gpu_alloc_slot, hash_u32, mix_index, pmap_at, pmap_slot,
    soap_detach,
};

/// Portal neighbor offsets in clockwise ring order (`PRTI.h:3-4`).
///
/// | count | `rx` | `ry` |
/// |-------|------|------|
/// | 0     | -1   | -1   |
/// | 1     |  0   | -1   |
/// | 2     |  1   | -1   |
/// | 3     |  1   |  0   |
/// | 4     |  1   |  1   |
/// | 5     |  0   |  1   |
/// | 6     | -1   |  1   |
/// | 7     | -1   |  0   |
#[cube]
pub fn portal_offset(dir: u32) -> (i32, i32) {
    let mut rx = 0i32;
    let mut ry = 0i32;
    if dir == 0u32 {
        rx = -1i32;
        ry = -1i32;
    } else if dir == 1u32 {
        rx = 0i32;
        ry = -1i32;
    } else if dir == 2u32 {
        rx = 1i32;
        ry = -1i32;
    } else if dir == 3u32 {
        rx = 1i32;
        ry = 0i32;
    } else if dir == 4u32 {
        rx = 1i32;
        ry = 1i32;
    } else if dir == 5u32 {
        rx = 0i32;
        ry = 1i32;
    } else if dir == 6u32 {
        rx = -1i32;
        ry = 1i32;
    } else if dir == 7u32 {
        rx = -1i32;
        ry = 0i32;
    }
    (rx, ry)
}

/// Pack a donated slot and element type into a portal queue header word.
#[cube]
pub fn portal_pack_ty(slot_plus_1: u32, ty: u32) -> u32 {
    ((slot_plus_1 & 0x7F_FFFFu32) << 9u32) | (ty & 0x1FFu32)
}

/// Unpack the 0-based donated slot from a portal header word, or `0xFFFF_FFFF` if none.
#[cube]
pub fn portal_unpack_slot(packed: u32) -> u32 {
    let raw = packed >> 9u32;
    let mut slot = 0xFFFF_FFFFu32;
    if raw > 0u32 {
        slot = raw - 1u32;
    }
    slot
}

/// Unpack the element type from a portal header word.
#[cube]
pub fn portal_unpack_ty(packed: u32) -> u32 {
    packed & 0x1FFu32
}

/// Unpack one orbital particle's distance and angle (`orbitalparts.h:3-14`).
#[cube]
pub fn orbital_get(b1: u32, b2: u32, r: u32) -> (i32, i32) {
    let shift = r * 8u32;
    let d = ((b1 >> shift) & 0xFFu32) as i32;
    let l = ((b2 >> shift) & 0xFFu32) as i32;
    (d, l)
}

/// Pack one orbital particle's distance and angle (`orbitalparts.h:16-33`).
#[cube]
pub fn orbital_set(d: i32, l: i32, r: u32) -> (u32, u32) {
    let shift = r * 8u32;
    let b1 = ((d as u32) & 0xFFu32) << shift;
    let b2 = ((l as u32) & 0xFFu32) << shift;
    (b1, b2)
}

/// Step inward swirl animation for `PRTI` (`PRTI.cpp:123-148`).
#[cube]
pub fn orbital_step_prti(life: i32, ctype: u32, seed: u32) -> (i32, u32) {
    let mut b1 = life as u32;
    let mut b2 = ctype;
    if b1 == 0u32 {
        b1 = hash_u32(seed ^ 0x9E37_79B9u32);
        if b1 == 0u32 {
            b1 = 1u32;
        }
    }
    if b2 == 0u32 {
        b2 = hash_u32(seed ^ 0x85EB_CA6Bu32);
        if b2 == 0u32 {
            b2 = 1u32;
        }
    }
    let mut out_b1 = 0u32;
    let mut out_b2 = 0u32;
    for r in 0u32..4u32 {
        let (mut d, mut l) = orbital_get(b1, b2, r);
        let rnd = hash_u32(seed ^ (r * 0x1000u32) ^ (d as u32) ^ ((l as u32) << 4u32));
        if d > 1i32 {
            d -= 12i32;
            if d < 1i32 {
                d = 128i32 + ((rnd & 127u32) as i32);
                l = ((rnd >> 8u32) % 255u32) as i32;
            } else {
                l = (l + 2i32) % 255i32;
            }
        } else {
            d = 128i32 + ((rnd & 127u32) as i32);
            l = ((rnd >> 8u32) % 255u32) as i32;
        }
        let (sb1, sb2) = orbital_set(d, l, r);
        out_b1 |= sb1;
        out_b2 |= sb2;
    }
    (out_b1 as i32, out_b2)
}

/// Step outward swirl animation for `PRTO` (`PRTO.cpp:143-178`).
#[cube]
pub fn orbital_step_prto(life: i32, ctype: u32, seed: u32) -> (i32, u32) {
    let mut b1 = life as u32;
    let mut b2 = ctype;
    if b1 == 0u32 {
        b1 = hash_u32(seed ^ 0x9E37_79B9u32);
        if b1 == 0u32 {
            b1 = 1u32;
        }
    }
    if b2 == 0u32 {
        b2 = hash_u32(seed ^ 0x85EB_CA6Bu32);
        if b2 == 0u32 {
            b2 = 1u32;
        }
    }
    let mut out_b1 = 0u32;
    let mut out_b2 = 0u32;
    for r in 0u32..4u32 {
        let (mut d, mut l) = orbital_get(b1, b2, r);
        let rnd = hash_u32(seed ^ (r * 0x1000u32) ^ (d as u32) ^ ((l as u32) << 4u32));
        if d < 254i32 {
            d += 16i32;
            if d > 254i32 {
                d = 0i32;
                l = ((rnd >> 8u32) % 255u32) as i32;
            } else {
                l = (l + 1i32) % 255i32;
            }
        } else {
            d = 0i32;
            l = ((rnd >> 8u32) % 255u32) as i32;
        }
        let (sb1, sb2) = orbital_set(d, l, r);
        out_b1 |= sb1;
        out_b2 |= sb2;
    }
    (out_b1 as i32, out_b2)
}

/// Portal IN: absorbs matter, energy, or sparks into the channel queue.
#[allow(
    clippy::manual_clamp,
    clippy::too_many_arguments,
    clippy::approx_constant
)]
#[cube(launch_unchecked)]
pub fn prti_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &[f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_vx: &[f32],
    part_vy: &[f32],
    part_flags: &mut [u32],
    part_dcolour: &[u32],
    portal_ty: &mut [Atomic<u32>],
    portal_data: &mut [u32],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
    n_slots: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 {
        terminate!();
    }
    let ty = v & 0x1FFu32;
    if ty != PT_PRTI_U32 {
        terminate!();
    }
    let s = pmap_slot(v);
    if part_ty[s] != PT_PRTI_U32 {
        terminate!();
    }
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;
    let ch = temp_channel(part_temp[s]);
    part_tmp[s] = ch as i32;

    let mut fe = 0u32;
    for count in 0u32..8u32 {
        let (rx, ry) = portal_offset(count);
        let nx = px + rx;
        let ny = py + ry;
        if nx >= 0i32 && nx < xres as i32 && ny >= 0i32 && ny < yres as i32 {
            let r_pmap = pmap_at(pmap, xres, yres, nx, ny);
            let mut count_fe = 0u32;
            if r_pmap == 0u32 || (r_pmap & 0x1FFu32) == PT_STOR_U32 {
                count_fe = 1u32;
            }

            let mut r = r_pmap;
            let mut nt = r & 0x1FFu32;
            let mut valid_matter = false;
            if r != 0u32 {
                if nt == PT_SPRK_U32 || nt == PT_STOR_U32 {
                    valid_matter = true;
                } else {
                    let props = elem_word(elem_props, nt, WORD_PROPERTIES);
                    if (props & MOVABLE_MASK) != 0u32 {
                        valid_matter = true;
                    }
                }
            }
            if !valid_matter {
                r = pmap_at(photons, xres, yres, nx, ny);
                if r != 0u32 {
                    nt = r & 0x1FFu32;
                    let props = elem_word(elem_props, nt, WORD_PROPERTIES);
                    if (props & MOVABLE_MASK) != 0u32 {
                        valid_matter = true;
                    }
                }
            }

            if valid_matter
                && nt != PT_STKM_U32
                && nt != PT_STKM2_U32
                && nt != PT_FIGH_U32
                && nt != PT_PRTI_U32
                && nt != PT_PRTO_U32
                && nt != PT_WIFI_U32
            {
                let ns = pmap_slot(r);
                if part_ty[ns] == nt {
                    // `PRTI.cpp:93-94`: the ring detaches a soap bubble before
                    // it queues or consumes it, whether or not a portal lane
                    // is free.
                    if nt == PT_SOAP_U32 {
                        soap_detach(part_ty, part_ctype, part_tmp, part_tmp2, ns, n_slots);
                    }
                    let base_lane = (ch * 8u32 + count) * 80u32;
                    if nt == PT_STOR_U32 {
                        let stored_raw = part_tmp[ns];
                        if stored_raw > 0i32 && (stored_raw as u32) < PT_NUM_U32 {
                            let stored_ty = stored_raw as u32;
                            let sprops = elem_word(elem_props, stored_ty, WORD_PROPERTIES);
                            if (sprops & MOVABLE_MASK) != 0u32 {
                                let mut claimed = false;
                                for nnx in 0u32..80u32 {
                                    if !claimed {
                                        let pidx = (base_lane + nnx) as usize;
                                        if portal_ty[pidx].compare_exchange_weak(0u32, stored_ty)
                                            == 0u32
                                        {
                                            let db = pidx * 8usize;
                                            portal_data[db] = part_tmp2[ns] as u32;
                                            portal_data[db + 1usize] = part_tmp4[ns] as u32;
                                            portal_data[db + 2usize] =
                                                u32::reinterpret(part_temp[ns]);
                                            portal_data[db + 3usize] = part_tmp3[ns] as u32;
                                            portal_data[db + 4usize] = 0u32;
                                            portal_data[db + 5usize] = 0u32;
                                            portal_data[db + 6usize] = 0u32;
                                            portal_data[db + 7usize] = part_dcolour[ns];

                                            part_tmp[ns] = 0i32;
                                            part_tmp2[ns] = 0i32;
                                            part_tmp3[ns] = 0i32;
                                            part_tmp4[ns] = 0i32;

                                            claimed = true;
                                        }
                                    }
                                }
                                if claimed {
                                    count_fe = 1u32;
                                }
                            }
                        }
                    } else if nt == PT_SPRK_U32 {
                        let mut claimed = false;
                        for nnx in 0u32..80u32 {
                            if !claimed {
                                let pidx = (base_lane + nnx) as usize;
                                if portal_ty[pidx].compare_exchange_weak(0u32, PT_SPRK_U32) == 0u32
                                {
                                    let cond = part_ctype[ns];
                                    if cond > 0u32 && cond < PT_NUM_U32 {
                                        part_ty[ns] = cond;
                                        part_ctype[ns] = 0u32;
                                        part_life[ns] = 0i32;
                                    } else {
                                        part_ty[ns] = PT_NONE_U32;
                                    }
                                    claimed = true;
                                }
                            }
                        }
                        if claimed {
                            count_fe = 1u32;
                        }
                    } else {
                        let packed = (((ns as u32) + 1u32) << 9u32) | (nt & 0x1FFu32);
                        let mut claimed = false;
                        for nnx in 0u32..80u32 {
                            if !claimed {
                                let pidx = (base_lane + nnx) as usize;
                                if portal_ty[pidx].compare_exchange_weak(0u32, packed) == 0u32 {
                                    let db = pidx * 8usize;
                                    portal_data[db] = part_life[ns] as u32;
                                    portal_data[db + 1usize] = part_ctype[ns];
                                    portal_data[db + 2usize] = u32::reinterpret(part_temp[ns]);
                                    portal_data[db + 3usize] = part_tmp[ns] as u32;
                                    portal_data[db + 4usize] = part_tmp2[ns] as u32;
                                    portal_data[db + 5usize] = u32::reinterpret(part_vx[ns]);
                                    portal_data[db + 6usize] = u32::reinterpret(part_vy[ns]);
                                    portal_data[db + 7usize] = part_dcolour[ns];

                                    part_flags[ns] |= FLAG_PORTAL_HELD_U32;
                                    part_ty[ns] = PT_NONE_U32;

                                    claimed = true;
                                }
                            }
                        }
                        if claimed {
                            count_fe = 1u32;
                        }
                    }
                }
            }

            if count_fe != 0u32 {
                fe = 1u32;
            }
        }
    }

    if fe != 0u32 {
        let seed = hash_u32(frame_seed ^ mix_index(s as u32, 0x9E37_79B9u32));
        let (new_life, new_ctype) = orbital_step_prti(part_life[s], part_ctype[s], seed);
        part_life[s] = new_life;
        part_ctype[s] = new_ctype;
    } else {
        part_life[s] = 0i32;
        part_ctype[s] = 0u32;
    }
}

/// Portal OUT: emits particles or sparks from the channel queue.
#[allow(
    clippy::manual_clamp,
    clippy::too_many_arguments,
    clippy::approx_constant
)]
#[cube(launch_unchecked)]
pub fn prto_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &mut [u32],
    part_dcolour: &mut [u32],
    portal_ty: &mut [Atomic<u32>],
    portal_data: &[u32],
    free_slots: &mut [Atomic<u32>],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
    capacity: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 {
        terminate!();
    }
    let ty = v & 0x1FFu32;
    if ty != PT_PRTO_U32 {
        terminate!();
    }
    let s = pmap_slot(v);
    if part_ty[s] != PT_PRTO_U32 {
        terminate!();
    }
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;
    let ch = temp_channel(part_temp[s]);
    part_tmp[s] = ch as i32;

    let mut fe = 0u32;
    for count in 0u32..8u32 {
        let (rx, ry) = portal_offset(count);
        let nx = px + rx;
        let ny = py + ry;
        if nx >= 0i32 && nx < xres as i32 && ny >= 0i32 && ny < yres as i32 {
            let mv = pmap_at(pmap, xres, yres, nx, ny);
            let pv = pmap_at(photons, xres, yres, nx, ny);
            if mv == 0u32 && pv == 0u32 {
                fe = 1u32;
                let mut emitted = false;
                for nnx in 0u32..80u32 {
                    if !emitted {
                        let rnd = hash_u32(
                            frame_seed
                                ^ mix_index(s as u32, 0x1B87_3593u32)
                                ^ (count << 8u32)
                                ^ nnx,
                        );
                        let jitter = ((rnd % 3u32) as i32) - 1i32;
                        let randomness = ((count as i32 + jitter + 4i32) & 7i32) as u32;
                        let base_lane = (ch * 8u32 + randomness) * 80u32;
                        let pidx = (base_lane + nnx) as usize;

                        let old = portal_ty[pidx].load();
                        if old != 0u32 && portal_ty[pidx].compare_exchange_weak(old, 0u32) == old {
                            let emit_ty = old & 0x1FFu32;
                            if emit_ty == PT_SPRK_U32 {
                                for sn in 0u32..9u32 {
                                    let sdx = (sn % 3u32) as i32 - 1i32;
                                    let sdy = (sn / 3u32) as i32 - 1i32;
                                    if sdx != 0i32 || sdy != 0i32 {
                                        let scx = px + sdx;
                                        let scy = py + sdy;
                                        let scv = pmap_at(pmap, xres, yres, scx, scy);
                                        if scv != 0u32 {
                                            let scty = scv & 0x1FFu32;
                                            let cn = pmap_slot(scv);
                                            if part_ty[cn] == scty
                                                && scty != PT_SPRK_U32
                                                && part_life[cn] == 0i32
                                                && ((elem_word(elem_props, scty, WORD_PROPERTIES)
                                                    & PROP_CONDUCTS_U32)
                                                    != 0u32
                                                    || scty == PT_INST_U32)
                                            {
                                                part_ctype[cn] = scty;
                                                part_ty[cn] = PT_SPRK_U32;
                                                part_life[cn] = 4i32;
                                            }
                                        }
                                    }
                                }
                                emitted = true;
                            } else {
                                let donated = old >> 9u32;
                                let mut dst_slot = 0xFFFF_FFFFu32;
                                if donated > 0u32 {
                                    let cand = donated - 1u32;
                                    if cand < capacity && part_ty[cand as usize] == PT_NONE_U32 {
                                        dst_slot = cand;
                                        part_flags[cand as usize] &= !FLAG_PORTAL_HELD_U32;
                                    }
                                }
                                if dst_slot == 0xFFFF_FFFFu32 {
                                    dst_slot = gpu_alloc_slot(free_slots);
                                    if dst_slot != 0xFFFF_FFFFu32 {
                                        part_flags[dst_slot as usize] &= !FLAG_PORTAL_HELD_U32;
                                    }
                                }
                                if dst_slot == 0xFFFF_FFFFu32 {
                                    portal_ty[pidx].store(old);
                                    emitted = true;
                                } else {
                                    let dst = dst_slot as usize;
                                    let db = pidx * 8usize;
                                    part_ty[dst] = emit_ty;
                                    part_x[dst] = nx as f32;
                                    part_y[dst] = ny as f32;
                                    part_life[dst] = portal_data[db] as i32;
                                    part_ctype[dst] = portal_data[db + 1usize];
                                    part_temp[dst] = f32::reinterpret(portal_data[db + 2usize]);
                                    part_tmp[dst] = portal_data[db + 3usize] as i32;
                                    part_tmp2[dst] = portal_data[db + 4usize] as i32;
                                    part_vx[dst] = f32::reinterpret(portal_data[db + 5usize]);
                                    part_vy[dst] = f32::reinterpret(portal_data[db + 6usize]);
                                    part_dcolour[dst] = portal_data[db + 7usize];
                                    emitted = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if fe != 0u32 {
        let seed = hash_u32(frame_seed ^ mix_index(s as u32, 0x85EB_CA6Bu32));
        let (new_life, new_ctype) = orbital_step_prto(part_life[s], part_ctype[s], seed);
        part_life[s] = new_life;
        part_ctype[s] = new_ctype;
    } else {
        part_life[s] = 0i32;
        part_ctype[s] = 0u32;
    }
}
