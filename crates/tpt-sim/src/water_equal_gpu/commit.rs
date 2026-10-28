//! Pass 4: Commit stage for water equalisation.
//!
//! One thread per particle slot `id in 0..num_slots`.
//! Checks if this particle won its destination pixel:
//! `water_winner[target] == id + 1`.
//! If won, updates `part_x` and `part_y`, clears the old pixel in `pmap`,
//! and claims the new pixel in `pmap`.

use cubecl::prelude::*;
use tpt_core::constants::PT_NUM;

const PT_NUM_U32: u32 = PT_NUM as u32;
const _: () = assert!(PT_NUM == 200);

#[cube(launch_unchecked)]
pub fn water_commit_kernel(
    part_ty: &[u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    pmap: &mut [u32],
    col_target: &mut [Atomic<u32>],
    water_winner: &mut [Atomic<u32>],
    num_slots: u32,
    xres: u32,
    yres: u32,
) {
    let id = ABSOLUTE_POS as u32;
    if id >= num_slots {
        terminate!();
    }
    let ty = part_ty[id as usize];
    if ty == 0u32 || ty >= PT_NUM_U32 {
        terminate!();
    }
    let px = part_x[id as usize];
    let py = part_y[id as usize];
    let xi = i32::cast_from(px + 0.5f32);
    let yi = i32::cast_from(py + 0.5f32);
    if xi < 0i32 || xi >= (xres as i32) || yi < 0i32 || yi >= (yres as i32) {
        terminate!();
    }
    let old_idx = (yi as usize) * (xres as usize) + (xi as usize);

    let packed = col_target[xi as usize].fetch_max(0u32);
    if packed == 0u32 {
        terminate!();
    }
    let target = packed - 1u32;
    let dest_y = target / xres;
    if (yi as u32) < dest_y {
        let winner = water_winner[target as usize].fetch_max(0u32);
        if winner == (id + 1u32) {
            let dest_x = (target % xres) as f32;
            let dest_yf = dest_y as f32;
            if (pmap[old_idx] >> 9u32) == (id + 1u32) {
                pmap[old_idx] = 0u32;
            }
            pmap[target as usize] = ((id + 1u32) << 9u32) | (ty & 0x1FFu32);
            part_x[id as usize] = dest_x;
            part_y[id as usize] = dest_yf;
        }
    }
}
