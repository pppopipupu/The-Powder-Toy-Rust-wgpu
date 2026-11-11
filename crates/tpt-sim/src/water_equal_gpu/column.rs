//! Pass 1: Column analysis for water equalisation.
//!
//! One thread per canvas column `x in 0..xres`.
//! Scans down column `x` from `y = 0` to `y = yres - 1` to find:
//! - Topmost liquid row `top_y`
//! - Bottommost liquid row `bot_y`
//! - Whether `(x, top_y - 1)` is free (empty and unblocked by wall).
//!
//! Stores `col_surf[x] = (top_y << 16) | (is_free << 15) | (bot_y & 0x7FFF)`.
//! Also clears `col_target[x].store(0)`.

use cubecl::prelude::*;
use tpt_core::constants::{PT_NUM, TYPE_LIQUID};

use crate::move_kernel::wall_blocks_particle;

const PT_NUM_U32: u32 = PT_NUM as u32;
const _: () = assert!(PT_NUM_U32 as usize == PT_NUM);
const _: () = assert!(TYPE_LIQUID == 2);

#[cube(launch_unchecked)]
pub fn water_column_analyze_kernel(
    col_surf: &mut [u32],
    col_target: &mut [Atomic<u32>],
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let x = ABSOLUTE_POS as u32;
    if x >= xres {
        terminate!();
    }

    col_target[x as usize].store(0u32);

    let mut found_liq = false;
    let mut min_y = yres;
    let mut max_y = 0u32;
    let mut y = 0u32;
    while y < yres {
        let idx = (y as usize) * (xres as usize) + (x as usize);
        let p = pmap[idx];
        if p != 0u32 {
            let ty = p & 0x1FFu32;
            if ty < PT_NUM_U32 {
                let falldown = elem_props[(ty as usize) * 32usize + 3usize];
                if falldown == 2u32 {
                    if !found_liq {
                        found_liq = true;
                        min_y = y;
                    }
                    max_y = y;
                }
            }
        }
        y += 1u32;
    }

    if found_liq {
        let mut free = false;
        if min_y > 0u32 {
            let above_y = min_y - 1u32;
            let above_idx = (above_y as usize) * (xres as usize) + (x as usize);
            if pmap[above_idx] == 0u32 {
                let wcell = (((above_y / 4u32) * xcells) + (x / 4u32)) as usize;
                let wall = wall_bmap[wcell];
                let emap = wall_emap[wcell];
                if !wall_blocks_particle(wall, emap, 0x2u32) {
                    free = true;
                }
            }
        }
        let free_bit = if free { 1u32 << 15u32 } else { 0u32 };
        col_surf[x as usize] = (min_y << 16u32) | free_bit | (max_y & 0x7FFFu32);
    } else {
        col_surf[x as usize] = 0xFFFF_FFFFu32;
    }
}
