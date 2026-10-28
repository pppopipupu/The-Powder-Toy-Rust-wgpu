//! `THDR.cpp` — the lightning particle.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{WORD_PROPERTIES, elem_word, hash_u32, pmap_at, pmap_slot};
use super::SPARK_LIFE;

/// `THDR.cpp:51-93`: lightning transfers its current and then vanishes.
///
/// Every neighbour in the radius-2 box is examined:
///
/// * a conducting, idle particle (not `WATR`/`SLTW`, and not already recording
///   a spark in `ctype`) is turned into a four-frame spark remembering its own
///   type,
/// * any other neighbour that is not `CLNE`/`THDR`/`SPRK`/`DMND`/`FIRE` adds
///   100 pressure to the lightning's own air cell.
///
/// Either outcome sets `kill`, so the lightning disappears after the loop. In
/// legacy (no-heat) mode the pressure branch instead rolls `1/200` to leave a
/// burning `FIRE` behind, which does not set `kill` by itself.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn thunder_update_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    air_pv: &mut [f32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
    legacy_enable: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_THDR {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                // `THDR.cpp:55`: `bool kill=false`. A legacy fire conversion
                // leaves it false, exactly like the reference.
                let mut kill = false;

                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nslot = pmap_slot(nv);
                            let nty = part_ty[nslot];
                            // A pass earlier in the frame can kill a slot
                            // without clearing `pmap` (death is `part_ty =
                            // PT_NONE`); the reference clears the map in
                            // `kill_part`, so a dead entry is an empty pixel.
                            if nty != PT_NONE {
                                if (elem_word(elem_words, nty, WORD_PROPERTIES) & PROP_CONDUCTS)
                                    != 0u32
                                    && part_life[nslot] == 0i32
                                    && nty != PT_WATR
                                    && nty != PT_SLTW
                                    && part_ctype[nslot] != PT_SPRK
                                {
                                    // `THDR.cpp:66-72`: `part_change_type` to
                                    // a spark that remembers the conductor
                                    // type; `THDR.cpp:70` then sets the
                                    // four-frame life. Unlike `create_part`
                                    // this does not heat the conductor.
                                    part_ctype[nslot] = nty;
                                    part_ty[nslot] = PT_SPRK;
                                    part_life[nslot] = SPARK_LIFE;
                                    kill = true;
                                } else if nty != PT_CLNE
                                    && nty != PT_THDR
                                    && nty != PT_SPRK
                                    && nty != PT_DMND
                                    && nty != PT_FIRE
                                {
                                    // `THDR.cpp:75`: `restrict_flt(pv + 100,
                                    // MIN_PRESSURE, MAX_PRESSURE)` on the
                                    // lightning's own cell, once per
                                    // qualifying neighbour.
                                    let pv =
                                        (air_pv[cidx] + 100.0f32).clamp(MIN_PRESSURE, MAX_PRESSURE);
                                    air_pv[cidx] = pv;

                                    // `THDR.cpp:76-81`: the legacy-only
                                    // `1/200` conversion into a burning fire.
                                    if legacy_enable != 0u32
                                        && hash_u32(frame_seed ^ (id as u32) ^ 0x5448_4452u32)
                                            .is_multiple_of(200u32)
                                    {
                                        // `rng.between(120, 169)`.
                                        let life = 120i32
                                            + (hash_u32(
                                                frame_seed ^ (id as u32) ^ n ^ 0x4649_5245u32,
                                            ) % 50u32)
                                                as i32;
                                        part_ty[id] = PT_FIRE;
                                        part_life[id] = life;
                                    } else {
                                        kill = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // `THDR.cpp:88-91`: `kill_part(i)`.
                if kill {
                    part_ty[id] = PT_NONE;
                }
            }
        }
    }
}
