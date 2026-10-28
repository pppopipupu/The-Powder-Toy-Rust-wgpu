use cubecl::prelude::*;
use tpt_core::constants::*;

use super::{PMAP_TYPE, chance1};
use crate::reaction_kernel::shared::{FLAG_SPAWN_NEUT, hash_u32, mix_index, pmap_slot};

/// `GOLD.cpp:91-97` and `POLO.cpp:55-103`: the two matter elements that read
/// and kill the energy particle on their own pixel.
///
/// Slot parallel: every case touches only its own slot plus the single photon
/// it is standing on, so no two threads write the same word.
///
/// * `GOLD` absorbs a neutron standing on it with a 1-in-7 chance.
/// * `POLO` releases a neutron (1-in-10000 on its own, 1-in-100 when a photon
///   is present), folds the released particle's temperature into its own,
///   absorbs `PROT` into `tmp2`, decays into `PLUT` at `tmp2 >= 10` and warms
///   itself by 0.2 K per frame below 388.15 K.
#[cube(launch_unchecked)]
pub(super) fn special_photon_kernel(
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_temp: &mut [f32],
    part_flags: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    slot_count: u32,
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let id = ABSOLUTE_POS;
    if id < slot_count as usize {
        let ty = part_ty[id];
        if ty == PT_GOLD || ty == PT_POLO {
            let x = part_x[id] as i32;
            let y = part_y[id] as i32;
            if x >= 0i32 && x < xres as i32 && y >= 0i32 && y < yres as i32 {
                let idx = (y as u32 * xres + x as u32) as usize;
                let pv = photons[idx];
                let seed = hash_u32(frame_seed ^ mix_index(id as u32, 0x27D4_EB2Fu32));

                if ty == PT_GOLD {
                    // `GOLD.cpp:91-97`: neutron shielding.
                    if pv != 0u32 && (pv & PMAP_TYPE) == PT_NEUT && chance1(seed, 7u32) {
                        part_ty[pmap_slot(pv)] = PT_NONE;
                    }
                } else {
                    // `POLO.cpp:58-95`: the whole update lives here because it
                    // needs the energy layer.
                    let mut temp = part_temp[id];
                    if part_tmp[id] < 5i32 && part_life[id] == 0i32 {
                        let own = hash_u32(seed ^ mix_index(id as u32, 0x1656_67B1u32));
                        if own.is_multiple_of(10_000u32) && part_tmp[id] == 0i32 {
                            // `create_part(-3, x, y, PT_NEUT)` on the free pixel.
                            part_flags[id] |= FLAG_SPAWN_NEUT;
                            part_tmp3[id] = 0i32;
                            part_life[id] = 15i32;
                            part_tmp[id] += 1i32;
                            temp = (temp + DEFAULT_TEMP_K + 600.0f32) / 2.0f32;
                        }
                        if pv != 0u32 {
                            // The photon's temperature is folded in twice and
                            // the new neutron inherits the photon's velocity.
                            let pslot = pmap_slot(pv);
                            if chance1(
                                hash_u32(seed ^ mix_index(id as u32, 0x68E3_1DA9u32)),
                                100u32,
                            ) {
                                part_flags[id] |= FLAG_SPAWN_NEUT;
                                part_tmp3[id] = pslot as i32 + 1i32;
                                let pt = part_temp[pslot];
                                temp = (((temp + pt) + pt) + 600.0f32) / 3.0f32;
                                part_life[id] = 15i32;
                                part_tmp[id] += 1i32;
                                part_temp[pslot] = temp;
                            }
                        }
                    }
                    if part_tmp2[id] >= 10i32 {
                        // `POLO.cpp:96-102`: `POLO -> PLUT`.
                        part_ty[id] = PT_PLUT;
                        part_temp[id] = (temp + 600.0f32) / 2.0f32;
                    } else {
                        if pv != 0u32 && (pv & PMAP_TYPE) == PT_PROT {
                            // `POLO.cpp:103-108`: absorb the proton.
                            part_tmp2[id] += 1i32;
                            part_ty[pmap_slot(pv)] = PT_NONE;
                        }
                        if temp < 388.15f32 {
                            temp += 0.2f32;
                        }
                        part_temp[id] = temp;
                    }
                }
            }
        }
    }
}
