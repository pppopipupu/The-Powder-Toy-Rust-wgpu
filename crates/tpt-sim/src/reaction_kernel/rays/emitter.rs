use cubecl::prelude::*;

use super::{
    C_ELEM_SLOTS, C_GPU_ALLOC_NONE, C_PROP_CONDUCTS, C_PT_ARAY, C_PT_BRAY, C_PT_DRAY, C_PT_FILT,
    C_PT_INST, C_PT_INWR, C_PT_LIFE, C_PT_NONE, C_PT_PSCN, C_PT_SPRK, C_PT_STKM, C_PT_STKM2,
    C_PT_STOR, C_PT_SWCH, C_PT_WIFI, C_TYPE_ENERGY, C_WORD_PROPERTIES,
};
use crate::reaction_kernel::shared::{elem_word, gpu_alloc_slot, mix_index, pmap_at, pmap_slot};

/// TPT `Element_FILT_getWavelengths`: `ctype` when non-zero, otherwise temperature bin.
#[cube]
#[allow(clippy::manual_clamp)]
pub fn filt_wavelengths(ctype: u32, temp: f32) -> u32 {
    if (ctype & 0x3FFF_FFFFu32) != 0u32 {
        ctype & 0x3FFF_FFFFu32
    } else {
        let mut bin = ((temp - 273.0f32) * 0.025f32) as i32;
        if bin < 0i32 {
            bin = 0i32;
        }
        if bin > 25i32 {
            bin = 25i32;
        }
        0x1Fu32 << (bin as u32)
    }
}

/// TPT `Element_FILT_interactWavelengths`: computes wavelength after passing through `FILT`.
#[cube]
#[allow(clippy::manual_checked_ops)]
pub fn interact_wavelengths(filt_tmp: i32, filt_ctype: u32, filt_temp: f32, orig_wl: u32) -> u32 {
    let mask = 0x3FFF_FFFFu32;
    let filt_wl = filt_wavelengths(filt_ctype, filt_temp);
    if filt_tmp == 0i32 {
        filt_wl
    } else if filt_tmp == 1i32 {
        orig_wl & filt_wl
    } else if filt_tmp == 2i32 {
        orig_wl | filt_wl
    } else if filt_tmp == 3i32 {
        orig_wl & (!filt_wl)
    } else if filt_tmp == 4i32 {
        let mut shift = ((filt_temp - 273.0f32) * 0.025f32) as i32;
        if shift <= 0i32 {
            shift = 1i32;
        }
        (orig_wl << (shift as u32)) & mask
    } else if filt_tmp == 5i32 {
        let mut shift = ((filt_temp - 273.0f32) * 0.025f32) as i32;
        if shift <= 0i32 {
            shift = 1i32;
        }
        (orig_wl >> (shift as u32)) & mask
    } else if filt_tmp == 6i32 {
        orig_wl
    } else if filt_tmp == 7i32 {
        orig_wl ^ filt_wl
    } else if filt_tmp == 8i32 {
        (!orig_wl) & mask
    } else if filt_tmp == 10i32 {
        let lsb = filt_wl & ((!filt_wl) + 1u32);
        mix_index(orig_wl, lsb) & mask
    } else if filt_tmp == 11i32 {
        let lsb = filt_wl & ((!filt_wl) + 1u32);
        if lsb != 0u32 {
            (orig_wl / lsb) & mask
        } else {
            orig_wl
        }
    } else {
        filt_wl
    }
}

/// `ARAY` ray emitter update pass (`ARAY.cpp`).
#[cube(launch_unchecked)]
#[allow(
    clippy::manual_clamp,
    clippy::too_many_arguments,
    clippy::approx_constant,
    unused_assignments
)]
pub fn aray_kernel(
    pmap: &[u32],
    free_slots: &mut [Atomic<u32>],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &mut [u32],
    part_dcolour: &mut [u32],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= xres * yres {
        terminate!();
    }
    let p = pmap[idx as usize];
    if (p & 0x1FFu32) != C_PT_ARAY {
        terminate!();
    }
    let i = pmap_slot(p);
    let px = (idx % xres) as i32;
    let py = (idx / xres) as i32;

    let short_bray_life = if part_life[i] > 0i32 {
        part_life[i]
    } else {
        30i32
    };
    let long_bray_life = if part_life[i] > 0i32 {
        part_life[i]
    } else {
        1020i32
    };
    let aray_temp = part_temp[i];

    for d in 0u32..8u32 {
        let mut rx = 0i32;
        let mut ry = 0i32;
        if d == 0u32 {
            rx = -1;
            ry = -1;
        } else if d == 1u32 {
            rx = 0;
            ry = -1;
        } else if d == 2u32 {
            rx = 1;
            ry = -1;
        } else if d == 3u32 {
            rx = -1;
            ry = 0;
        } else if d == 4u32 {
            rx = 1;
            ry = 0;
        } else if d == 5u32 {
            rx = -1;
            ry = 1;
        } else if d == 6u32 {
            rx = 0;
            ry = 1;
        } else {
            rx = 1;
            ry = 1;
        }

        let sprk_p = pmap_at(pmap, xres, yres, px + rx, py + ry);
        if (sprk_p & 0x1FFu32) == C_PT_SPRK {
            let sprk = pmap_slot(sprk_p);
            if part_life[sprk] == 3i32 {
                let destroy = part_ctype[sprk] == C_PT_PSCN;
                let nostop = part_ctype[sprk] == C_PT_INST;
                let mut colored = 0u32;
                let mut is_black_deco = false;
                let nxi = -rx;
                let nyi = -ry;
                let mut docontinue = true;

                for step in 0u32..256u32 {
                    if !docontinue {
                        break;
                    }
                    let st = (step + 1u32) as i32;
                    let tx = px + nxi * st;
                    let ty = py + nyi * st;
                    if tx < 0i32 || ty < 0i32 || (tx as u32) >= xres || (ty as u32) >= yres {
                        break;
                    }
                    let target_p = pmap_at(pmap, xres, yres, tx, ty);
                    let rt = target_p & 0x1FFu32;
                    let r = pmap_slot(target_p);

                    if rt == C_PT_NONE {
                        let nr = gpu_alloc_slot(free_slots);
                        if nr != C_GPU_ALLOC_NONE {
                            let s = nr as usize;
                            part_ty[s] = C_PT_BRAY;
                            part_x[s] = tx as f32;
                            part_y[s] = ty as f32;
                            part_vx[s] = 0.0f32;
                            part_vy[s] = 0.0f32;
                            part_temp[s] = aray_temp;
                            part_flags[s] = 0u32;
                            part_tmp2[s] = 0i32;
                            part_dcolour[s] = if is_black_deco { 0xFF00_0000u32 } else { 0u32 };
                            if destroy {
                                part_tmp[s] = 2i32;
                                part_life[s] = 2i32;
                                part_ctype[s] = 0u32;
                            } else {
                                part_tmp[s] = 0i32;
                                part_life[s] = short_bray_life;
                                part_ctype[s] = colored;
                            }
                        }
                    } else if !destroy {
                        if rt == C_PT_BRAY {
                            if part_tmp[r] == 0i32 {
                                part_life[r] = long_bray_life;
                                part_tmp[r] = 1i32;
                                if part_ctype[r] == 0u32 {
                                    part_ctype[r] = colored;
                                }
                                docontinue = false;
                            } else if part_tmp[r] == 1i32 {
                                part_life[r] = long_bray_life;
                            } else {
                                docontinue = false;
                            }
                            if is_black_deco {
                                part_dcolour[r] = 0xFF00_0000u32;
                            }
                        } else if rt == C_PT_FILT {
                            if part_tmp[r] != 6i32 {
                                colored = interact_wavelengths(
                                    part_tmp[r],
                                    part_ctype[r],
                                    part_temp[r],
                                    colored,
                                );
                                if colored == 0u32 {
                                    docontinue = false;
                                    break;
                                }
                            }
                            is_black_deco = part_dcolour[r] == 0xFF00_0000u32;
                            part_life[r] = 4i32;
                        } else if rt == C_PT_STOR {
                            if part_tmp[r] > 0i32 {
                                let mut released = false;
                                for dry in -1i32..=1i32 {
                                    if released {
                                        break;
                                    }
                                    for drx in -1i32..=1i32 {
                                        if released {
                                            break;
                                        }
                                        if drx != 0i32 || dry != 0i32 {
                                            let npx = tx + drx;
                                            let npy = ty + dry;
                                            if npx >= 0i32
                                                && npy >= 0i32
                                                && (npx as u32) < xres
                                                && (npy as u32) < yres
                                                && pmap_at(pmap, xres, yres, npx, npy) == 0u32
                                            {
                                                let np = gpu_alloc_slot(free_slots);
                                                if np != C_GPU_ALLOC_NONE {
                                                    let cargo_ty = (part_tmp[r] as u32) & 0x1FFu32;
                                                    let s = np as usize;
                                                    part_ty[s] = cargo_ty;
                                                    part_x[s] = npx as f32;
                                                    part_y[s] = npy as f32;
                                                    part_vx[s] = 0.0f32;
                                                    part_vy[s] = 0.0f32;
                                                    part_temp[s] = part_temp[r];
                                                    part_life[s] = part_tmp2[r];
                                                    part_tmp[s] = part_tmp3[r];
                                                    part_ctype[s] = part_tmp4[r] as u32;
                                                    part_flags[s] = 0u32;
                                                    part_dcolour[s] = 0u32;
                                                }
                                                released = true;
                                            }
                                        }
                                    }
                                }
                                part_tmp[r] = 0i32;
                                part_life[r] = 10i32;
                            } else {
                                part_life[r] = 10i32;
                            }
                        } else if rt != C_PT_INWR
                            && !(rt == C_PT_SPRK && part_ctype[r] == C_PT_INWR)
                            && rt != C_PT_ARAY
                            && rt != C_PT_WIFI
                            && !(rt == C_PT_SWCH && part_life[r] >= 10i32)
                        {
                            let cond = (elem_word(elem_props, rt, C_WORD_PROPERTIES)
                                & C_PROP_CONDUCTS)
                                != 0u32;
                            if cond && part_life[r] == 0i32 {
                                part_ctype[r] = rt;
                                part_ty[r] = C_PT_SPRK;
                                part_life[r] = 4i32;
                            }
                            let spark_cond =
                                (elem_word(elem_props, part_ctype[r], C_WORD_PROPERTIES)
                                    & C_PROP_CONDUCTS)
                                    != 0u32;
                            docontinue = nostop && rt == C_PT_SPRK && spark_cond;
                        }
                    } else {
                        // destroy == true
                        if rt == C_PT_BRAY {
                            part_tmp[r] = 2i32;
                            part_life[r] = 1i32;
                            if is_black_deco {
                                part_dcolour[r] = 0xFF00_0000u32;
                            }
                            docontinue = true;
                        } else if rt == C_PT_STOR
                            || rt == C_PT_INWR
                            || (rt == C_PT_SPRK && part_ctype[r] == C_PT_INWR)
                            || rt == C_PT_ARAY
                            || rt == C_PT_WIFI
                            || rt == C_PT_FILT
                            || (rt == C_PT_SWCH && part_life[r] >= 10i32)
                        {
                            if rt == C_PT_STOR {
                                part_tmp[r] = 0i32;
                                part_life[r] = 0i32;
                            } else if rt == C_PT_FILT {
                                is_black_deco = part_dcolour[r] == 0xFF00_0000u32;
                                part_life[r] = 2i32;
                            }
                            docontinue = true;
                        } else {
                            docontinue = false;
                        }
                    }
                }
            }
        }
    }
}

/// `DRAY` duplicator ray update pass (`DRAY.cpp`).
#[cube(launch_unchecked)]
#[allow(
    clippy::manual_clamp,
    clippy::too_many_arguments,
    clippy::approx_constant,
    unused_assignments
)]
pub fn dray_kernel(
    pmap: &[u32],
    photons: &[u32],
    free_slots: &mut [Atomic<u32>],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &mut [u32],
    part_dcolour: &mut [u32],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= xres * yres {
        terminate!();
    }
    let p = pmap[idx as usize];
    if (p & 0x1FFu32) != C_PT_DRAY {
        terminate!();
    }
    let i = pmap_slot(p);
    let px = (idx % xres) as i32;
    let py = (idx / xres) as i32;

    let ctype = part_ctype[i] & 0x1FFu32;
    let ctype_extra = part_ctype[i] >> 9u32;
    let mut copy_length = part_tmp[i];
    if copy_length < 0i32 {
        copy_length = 0i32;
    }
    let mut copy_spaces = part_tmp2[i];
    if copy_spaces < 0i32 {
        copy_spaces = 0i32;
    }
    if copy_length > 0i32 {
        copy_spaces += 1i32;
    }

    for d in 0u32..8u32 {
        let mut rx = 0i32;
        let mut ry = 0i32;
        if d == 0u32 {
            rx = -1;
            ry = -1;
        } else if d == 1u32 {
            rx = 0;
            ry = -1;
        } else if d == 2u32 {
            rx = 1;
            ry = -1;
        } else if d == 3u32 {
            rx = -1;
            ry = 0;
        } else if d == 4u32 {
            rx = 1;
            ry = 0;
        } else if d == 5u32 {
            rx = -1;
            ry = 1;
        } else if d == 6u32 {
            rx = 0;
            ry = 1;
        } else {
            rx = 1;
            ry = 1;
        }

        let sprk_p = pmap_at(pmap, xres, yres, px + rx, py + ry);
        if (sprk_p & 0x1FFu32) == C_PT_SPRK {
            let sprk = pmap_slot(sprk_p);
            if part_life[sprk] == 3i32
                && !(part_ctype[sprk] == C_PT_INWR && rx != 0i32 && ry != 0i32)
            {
                let overwrite = part_ctype[sprk] == C_PT_PSCN;
                let x_step = -rx;
                let y_step = -ry;

                let mut found_particle = false;
                let mut is_energy = false;
                let mut parts_remaining = copy_length;
                let mut local_copy_length = copy_length;
                let mut x_copy_to = px;
                let mut y_copy_to = py;
                let mut found_stop = false;

                let ctype_is_energy = if ctype != 0u32 && ctype < C_ELEM_SLOTS {
                    (elem_word(elem_props, ctype, C_WORD_PROPERTIES) & C_TYPE_ENERGY) != 0u32
                } else {
                    false
                };

                for step in 0u32..256u32 {
                    let st = (step + 1u32) as i32;
                    let x_cur = px + x_step * st;
                    let y_cur = py + y_step * st;
                    if x_cur < 0i32
                        || y_cur < 0i32
                        || (x_cur as u32) >= xres
                        || (y_cur as u32) >= yres
                    {
                        break;
                    }

                    if !found_particle {
                        let mut rr_check = pmap_at(pmap, xres, yres, x_cur, y_cur);
                        if rr_check == 0u32 {
                            rr_check = pmap_at(photons, xres, yres, x_cur, y_cur);
                            if rr_check != 0u32 {
                                found_particle = true;
                                is_energy = true;
                            }
                        } else {
                            found_particle = true;
                        }
                    }

                    let rr = if ctype_is_energy || is_energy {
                        pmap_at(photons, xres, yres, x_cur, y_cur)
                    } else {
                        pmap_at(pmap, xres, yres, x_cur, y_cur)
                    };
                    let r_ty = rr & 0x1FFu32;
                    let r_id = pmap_slot(rr);

                    parts_remaining -= 1i32;
                    let at_stop = (local_copy_length == 0i32
                        && r_ty == ctype
                        && (ctype != C_PT_LIFE || part_ctype[r_id] == ctype_extra))
                        || parts_remaining == 0i32;

                    if at_stop {
                        local_copy_length -= parts_remaining;
                        x_copy_to = x_cur + x_step * copy_spaces;
                        y_copy_to = y_cur + y_step * copy_spaces;
                        found_stop = true;
                        break;
                    }
                }

                if found_stop && local_copy_length > 0i32 {
                    let mut cur_dst_x = x_copy_to;
                    let mut cur_dst_y = y_copy_to;
                    for step in 0u32..256u32 {
                        if (step as i32) >= local_copy_length {
                            break;
                        }
                        let st = (step + 1u32) as i32;
                        let x_cur = px + x_step * st;
                        let y_cur = py + y_step * st;

                        if cur_dst_x < 0i32
                            || cur_dst_y < 0i32
                            || (cur_dst_x as u32) >= xres
                            || (cur_dst_y as u32) >= yres
                        {
                            break;
                        }

                        let p_src = if is_energy {
                            pmap_at(photons, xres, yres, x_cur, y_cur)
                        } else {
                            pmap_at(pmap, xres, yres, x_cur, y_cur)
                        };
                        let p_dst = if is_energy {
                            pmap_at(photons, xres, yres, cur_dst_x, cur_dst_y)
                        } else {
                            pmap_at(pmap, xres, yres, cur_dst_x, cur_dst_y)
                        };

                        if p_src != 0u32 {
                            let src_id = pmap_slot(p_src);
                            let src_ty = part_ty[src_id];
                            if src_ty != C_PT_STKM && src_ty != C_PT_STKM2 && src_ty != C_PT_NONE {
                                if overwrite && p_dst != 0u32 {
                                    let dst_id = pmap_slot(p_dst);
                                    part_ty[dst_id] = C_PT_NONE;
                                }
                                if p_dst == 0u32 || overwrite {
                                    let p_new = gpu_alloc_slot(free_slots);
                                    if p_new != C_GPU_ALLOC_NONE {
                                        let s = p_new as usize;
                                        part_ty[s] = src_ty;
                                        part_life[s] = part_life[src_id];
                                        part_ctype[s] = part_ctype[src_id];
                                        part_temp[s] = part_temp[src_id];
                                        part_tmp[s] = part_tmp[src_id];
                                        part_tmp2[s] = part_tmp2[src_id];
                                        part_tmp3[s] = part_tmp3[src_id];
                                        part_tmp4[s] = part_tmp4[src_id];
                                        part_vx[s] = part_vx[src_id];
                                        part_vy[s] = part_vy[src_id];
                                        part_dcolour[s] = part_dcolour[src_id];
                                        part_flags[s] = 0u32;
                                        part_x[s] = cur_dst_x as f32;
                                        part_y[s] = cur_dst_y as f32;
                                    }
                                }
                            }
                        }

                        cur_dst_x += x_step;
                        cur_dst_y += y_step;
                    }
                }
            }
        }
    }
}
