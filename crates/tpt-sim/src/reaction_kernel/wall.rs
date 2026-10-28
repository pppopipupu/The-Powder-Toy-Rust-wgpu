//! Wall conduction: charging and sparking.
//!
//! Conductive walls charge from a nearby spark; an electrified wall sparks the
//! conductor sitting on it.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::shared::{
    EMAP_CHARGED, EMAP_SPARKING, W_ALLOWALLELEC, W_ALLOWLIQUID, W_DETECT, W_EHOLE, W_EWALL,
    W_WALLELEC, WORD_PROPERTIES, elem_word, iabs, pmap_slot,
};

/// Conductive walls charge from a nearby spark; an electrified wall sparks the
/// conductor sitting on it.
#[cube(launch_unchecked)]
pub(super) fn wall_conduction_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    wall_bmap: &[u32],
    wall_emap: &mut [u32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let px = (idx as u32 % xres) as i32;
        let py = (idx as u32 / xres) as i32;
        let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];

            if ty == PT_SPRK && part_life[id] == 3i32 {
                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if iabs(dx) + iabs(dy) < 4i32 {
                        let nx = px + dx;
                        let ny = py + dy;
                        if nx >= 0i32 && ny >= 0i32 && (nx as u32) < xres && (ny as u32) < yres {
                            let ncidx =
                                (((ny as u32) / CELL) * xcells + ((nx as u32) / CELL)) as usize;
                            let w = wall_bmap[ncidx] & 0xFFu32;
                            if w == W_WALLELEC
                                || w == W_EWALL
                                || w == W_DETECT
                                || w == W_ALLOWALLELEC
                                || w == W_ALLOWLIQUID
                                || w == W_EHOLE
                            {
                                wall_emap[ncidx] = EMAP_CHARGED;
                            }
                        }
                    }
                }
            }

            if (wall_emap[cidx] & 0xFFu32) == EMAP_SPARKING
                && ty != PT_NONE
                && ty != PT_SPRK
                && part_life[id] == 0i32
                && (elem_word(elem_words, ty, WORD_PROPERTIES) & PROP_CONDUCTS) != 0u32
            {
                part_ty[id] = PT_SPRK;
                part_ctype[id] = ty;
                part_life[id] = 4i32;
            }
        }
    }
}
