//! Pass 2: Row scan and horizontal connectivity for water equalisation.
//!
//! One thread per canvas row `y in 0..yres`.
//! Finds contiguous spans of liquid across row `y` (stopping at walls or empty space).
//! Within each span, identifies the highest surface (`min_top_y`) and lowest free
//! surface (`max_free_dest_y`). If `max_free_dest_y > min_top_y`, assigns destination
//! targets to the higher columns distributed across the free receiving columns.

use cubecl::prelude::*;
use tpt_core::constants::{PT_NUM, TYPE_LIQUID};

use crate::move_kernel::wall_blocks_particle;

const PT_NUM_U32: u32 = PT_NUM as u32;
const _: () = assert!(PT_NUM_U32 as usize == PT_NUM);
const _: () = assert!(TYPE_LIQUID == 2);

#[cube]
fn mix_index(index: u32, salt: u32) -> u32 {
    index * salt
}

#[cube(launch_unchecked)]
pub fn water_row_connect_kernel(
    col_surf: &[u32],
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    elem_props: &[u32],
    col_target: &mut [Atomic<u32>],
    water_winner: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let y = ABSOLUTE_POS as u32;
    if y >= yres {
        terminate!();
    }

    let mut x = 0u32;
    while x < xres {
        let idx = (y as usize) * (xres as usize) + (x as usize);
        let p = pmap[idx];
        let mut is_liq = false;
        if p != 0u32 {
            let ty = p & 0x1FFu32;
            if ty < PT_NUM_U32 {
                let falldown = elem_props[(ty as usize) * 32usize + 3usize];
                if falldown == 2u32 {
                    is_liq = true;
                }
            }
        }

        if !is_liq {
            x += 1u32;
        } else {
            let x_start = x;
            let mut x_end = x_start;
            let mut next_x = x_start + 1u32;
            // The walk must not use `break`: a `break`-terminated `while` loses
            // its loop-carried updates on exit in the CubeCL frontend, which
            // left `x_end == x_start` and turned every span into one column.
            // Stop by moving `next_x` past `xres` instead.
            while next_x < xres {
                let nidx = (y as usize) * (xres as usize) + (next_x as usize);
                let np = pmap[nidx];
                let mut n_liq = false;
                if np != 0u32 {
                    let nty = np & 0x1FFu32;
                    if nty < PT_NUM_U32 {
                        let n_falldown = elem_props[(nty as usize) * 32usize + 3usize];
                        if n_falldown == 2u32 {
                            n_liq = true;
                        }
                    }
                }
                let wcell = (((y / 4u32) * xcells) + (next_x / 4u32)) as usize;
                let wall = wall_bmap[wcell];
                let emap = wall_emap[wcell];
                if n_liq && !wall_blocks_particle(wall, emap, 0x2u32) {
                    x_end = next_x;
                    next_x += 1u32;
                } else {
                    next_x = xres;
                }
            }

            let mut min_top_y = yres;
            let mut max_free_dest_y = 0u32;
            let mut dest_x_min = xres;
            let mut dest_x_max = 0u32;

            let mut k = x_start;
            while k <= x_end {
                let s = col_surf[k as usize];
                if s != 0xFFFF_FFFFu32 {
                    let top_y = s >> 16u32;
                    let free = (s & (1u32 << 15u32)) != 0u32;
                    let bot_y = s & 0x7FFFu32;
                    if y >= top_y && y <= bot_y {
                        if top_y < min_top_y {
                            min_top_y = top_y;
                        }
                        if free {
                            if top_y > max_free_dest_y {
                                max_free_dest_y = top_y;
                                dest_x_min = k;
                                dest_x_max = k;
                            } else if top_y == max_free_dest_y {
                                if k < dest_x_min {
                                    dest_x_min = k;
                                }
                                if k > dest_x_max {
                                    dest_x_max = k;
                                }
                            }
                        }
                    }
                }
                k += 1u32;
            }

            if max_free_dest_y > min_top_y + 1u32 {
                let dest_row = max_free_dest_y - 1u32;
                let span_len = (dest_x_max - dest_x_min) + 1u32;

                k = x_start;
                while k <= x_end {
                    let s = col_surf[k as usize];
                    if s != 0xFFFF_FFFFu32 {
                        let top_y = s >> 16u32;
                        if top_y < dest_row {
                            let dest_x = dest_x_min + ((k + frame_seed) % span_len);
                            let dest_pixel = (dest_row * xres) + dest_x;
                            let packed = dest_pixel + 1u32;
                            col_target[k as usize].store(packed);
                            water_winner[dest_pixel as usize].store(0xFFFF_FFFFu32);
                        }
                    }
                    k += 1u32;
                }
            }

            x = x_end + 1u32;
        }
    }
}
