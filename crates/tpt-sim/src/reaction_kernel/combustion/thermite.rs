//! Thermite burning and molten lava kernels.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::*;

/// `THRM` burning and the `LAVA(THRM)` / `LAVA(PLUT)` molten state.
///
/// `FIRE.cpp:176-190`: a fire / plasma / lava pixel that touches thermite turns
/// it into 3500 K lava that remembers `THRM` (or `BMTL` for the 1/500 flash
/// variant) and carries `tmp = 20`. `Simulation.cpp:2731-2739`: that `tmp`
/// holds the lava at 3500 K for 20 frames. `Simulation.cpp:2623-2655`: molten
/// thermite only freezes below `BMTL`'s 1273 K and then becomes `BMTL`.
///
/// The sub-973 K branch belongs to the thermal kernel, which freezes
/// `LAVA(THRM)` into `THRM` before this pass runs; only the 973..1273 K window
/// is corrected here.
#[cube(launch_unchecked)]
pub(crate) fn thermite_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_ctype: &mut [u32],
    air_pv: &mut [f32],
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
            if ty == PT_FIRE || ty == PT_PLSM || ty == PT_LAVA {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0x7FEB_352Du32));
                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            if part_ty[nid] == PT_THRM {
                                if chance(seed ^ n, 1u32, 500u32) {
                                    part_ty[nid] = PT_LAVA;
                                    part_ctype[nid] = PT_BMTL;
                                    part_temp[nid] = THRM_BURN_TEMP;
                                    let ncidx = ((((py + dy) as u32) / CELL) * xcells
                                        + (((px + dx) as u32) / CELL))
                                        as usize;
                                    air_pv[ncidx] += 50.0f32;
                                } else {
                                    part_ty[nid] = PT_LAVA;
                                    part_life[nid] = 400i32;
                                    part_ctype[nid] = PT_THRM;
                                    part_temp[nid] = THRM_BURN_TEMP;
                                    part_tmp[nid] = 20i32;
                                }
                            }
                        }
                    }
                }
            } else if ty == PT_LAVA {
                let ct = part_ctype[id];
                if ct == PT_THRM {
                    if part_tmp[id] > 0i32 {
                        part_tmp[id] -= 1i32;
                        part_temp[id] = THRM_BURN_TEMP;
                    } else if part_temp[id] < BMTL_HIGH_TEMP {
                        // `LAVA(THRM) -> BMTL`, not back into thermite.
                        part_ty[id] = PT_BMTL;
                        part_ctype[id] = 0u32;
                    }
                } else if ct == PT_PLUT && part_tmp[id] > 0i32 {
                    part_tmp[id] -= 1i32;
                    part_temp[id] = MAX_TEMP;
                }
            } else if ty == PT_BRMT {
                let temp = part_temp[id];
                if temp > 523.15f32 {
                    // `BRMT.cpp:51-78`: tempFactor = int(1000 - ((523.15f - parts[i].temp) * 2));
                    let tf = 1000i32 - (((523.15f32 - temp) * 2.0f32) as i32);
                    let mut tempfactor = tf as u32;
                    if tf < 2i32 {
                        tempfactor = 2u32;
                    }
                    let px = (idx as u32 % xres) as i32;
                    let py = (idx as u32 / xres) as i32;
                    let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0x51ED_2701u32));
                    let mut converted = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if (dx != 0i32 || dy != 0i32) && !converted {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                if part_ty[nid] == PT_BREC && chance(seed ^ n, 1u32, tempfactor) {
                                    if chance(seed ^ n ^ 0x1u32, 1u32, 2u32) {
                                        part_ty[nid] = PT_THRM;
                                    } else {
                                        part_ty[id] = PT_THRM;
                                        converted = true;
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
