//! `STOR` particle storage.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{
    FLAG_SPAWN_STOR, WORD_PROPERTIES, elem_word, pmap_at, pmap_slot, soap_detach,
};
use super::SPARK_LIFE;

const PT_STOR_U32: u32 = 74;
const _: () = assert!(PT_STOR == 74);
const PT_SPRK_U32: u32 = 12;
const _: () = assert!(PT_SPRK == 12);
const PT_PSCN_U32: u32 = 50;
const _: () = assert!(PT_PSCN == 50);
const PT_NONE_U32: u32 = 0;
const _: () = assert!(PT_NONE == 0);
const PT_NUM_U32: u32 = 200;
const _: () = assert!(PT_NUM == 200);

/// `STOR.cpp`: release cooldown after a discharge.
const STOR_RELEASE_LIFE: i32 = 10;

/// `STOR.cpp`: capture one non-solid particle, release it on a `PSCN` spark.
///
/// Captured particle fields are stored directly into `STOR`'s own SoA slots:
/// `tmp = type`, `temp = temp`, `tmp2 = life`, `tmp3 = tmp`, `tmp4 = ctype`.
/// Both `pmap` (matter) and `photons` (energy particles) are captured.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn storage_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_flags: &mut [u32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    n_slots: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_STOR_U32 {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let mut stored = part_tmp[id];
                // A `tmp` that does not name a valid element is cleared.
                if stored < 0i32 || (stored as u32) >= PT_NUM_U32 {
                    stored = 0i32;
                    part_tmp[id] = 0i32;
                }
                if part_life[id] != 0i32 && stored == 0i32 {
                    part_life[id] -= 1i32;
                }
                let mut releasing = false;
                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nx = px + dx;
                        let ny = py + dy;
                        let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                        if nv == 0u32
                            && nx >= 0i32
                            && ny >= 0i32
                            && (nx as u32) < xres
                            && (ny as u32) < yres
                        {
                            nv = photons[((ny as u32) * xres + (nx as u32)) as usize];
                        }
                        if nv != 0u32 {
                            let nslot = pmap_slot(nv);
                            let nty = part_ty[nslot];
                            if nty != PT_NONE_U32 {
                                if stored == 0i32 {
                                    let ctype = part_ctype[id];
                                    if part_life[id] == 0i32
                                        && nty != PT_STOR_U32
                                        && (elem_word(elem_words, nty, WORD_PROPERTIES)
                                            & TYPE_SOLID)
                                            == 0u32
                                        && (ctype == PT_NONE_U32 || nty == ctype)
                                    {
                                        // `STOR.cpp:75-76`: a soap bubble is
                                        // unlinked before its five fields are
                                        // captured.
                                        if nty == PT_SOAP {
                                            soap_detach(
                                                part_ty, part_ctype, part_tmp, part_tmp2, nslot,
                                                n_slots,
                                            );
                                        }
                                        stored = nty as i32;
                                        part_tmp[id] = stored;
                                        part_temp[id] = part_temp[nslot];
                                        part_tmp2[id] = part_life[nslot];
                                        part_tmp3[id] = part_tmp[nslot];
                                        part_tmp4[id] = part_ctype[nslot] as i32;
                                        part_ty[nslot] = PT_NONE_U32;
                                    }
                                } else if !releasing {
                                    let nctype = part_ctype[nslot];
                                    if nty == PT_SPRK_U32
                                        && nctype == PT_PSCN_U32
                                        && part_life[nslot] > 0i32
                                        && part_life[nslot] < SPARK_LIFE
                                    {
                                        part_flags[id] |= FLAG_SPAWN_STOR;
                                        part_life[id] = STOR_RELEASE_LIFE;
                                        releasing = true;
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
