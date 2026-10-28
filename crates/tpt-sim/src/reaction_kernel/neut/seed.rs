//! `NEUT.cpp:209-233`: the `SEED` gene flip.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{hash_u32, mix_index, pmap_slot};
use super::super::special::{PLNT_COLOUR, PLNT_LIFE, PLNT_TOTAL_TMP};

/// `NEUT.cpp:209-233`: a neutron resting on a `SEED` flips one random gene bit.
///
/// Split from [`neut_reaction_kernel`] because the genome lives in `tmp2`..
/// `tmp4`, which would push the main kernel over the storage-buffer budget.
#[cube(launch_unchecked)]
pub(crate) fn neut_seed_kernel(
    photons: &[u32],
    pmap: &[u32],
    part_ty: &mut [u32],
    part_ctype: &mut [u32],
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
            if part_ty[id] == PT_NEUT {
                let under = pmap[idx];
                if under != 0u32 {
                    let nid = pmap_slot(under);
                    if part_ty[nid] == PT_SEED {
                        let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 0x27D4_EB2Fu32));
                        let which = roll % 10u32;
                        if which == 0u32 {
                            let bit = PLNT_COLOUR + (hash_u32(roll) % (PLNT_LIFE - PLNT_COLOUR));
                            part_ctype[nid] ^= 1u32 << bit;
                        } else if which == 1u32 {
                            let bit = hash_u32(roll ^ 0x11u32) % PLNT_TOTAL_TMP;
                            part_tmp[nid] ^= (1u32 << bit) as i32;
                        } else if which == 2u32 {
                            let bit = hash_u32(roll ^ 0x22u32) % PLNT_TOTAL_TMP;
                            part_tmp2[nid] ^= (1u32 << bit) as i32;
                        } else if which == 3u32 {
                            let bit = hash_u32(roll ^ 0x33u32) % PLNT_TOTAL_TMP;
                            part_tmp3[nid] ^= (1u32 << bit) as i32;
                        } else if which == 4u32 {
                            let bit = hash_u32(roll ^ 0x44u32) % PLNT_TOTAL_TMP;
                            part_tmp4[nid] ^= (1u32 << bit) as i32;
                        }
                    }
                }
            }
        }
    }
}
