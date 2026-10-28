#![allow(clippy::assign_op_pattern, clippy::manual_range_contains)]

//! TNT connected flood (`BANG.cpp:58-88`).
//!
//! `Element_BANG_update` stage 0 only sets `tmp = 1` on the bomb that saw fire.
//! The following update (`Simulation.cpp:3160-3175`, `Simulation::FloodParts`)
//! walks every connected BANG particle and sets `tmp = 2`; stage 2 then sets
//! `tmp = 3` on the next update and stage 3 explodes on the one after. The 3D
//! port keeps the same three updates:
//!
//! * tick T: reaction sets `tmp = 1`, `tmp2 = T`, and appends the cell to the
//!   flood seed list;
//! * tick T+1: reaction sees stage 1 and leaves it; the flood pass (after the
//!   reaction pass) visits every connected BANG and sets `tmp = 2`;
//! * tick T+2: reaction sets `tmp = 3`;
//! * tick T+3: reaction explodes.
//!
//! The flood is a single-thread breadth-first search over the seed list. TPT's
//! flood is also serial, so this is both exact and simple; only seeds are
//! stored between ticks, which keeps the persistent queue tiny. The visited
//! bitset is cleared at the end of the pass, so it always starts empty.
//!
//! Seeds whose `tmp2 == current tick` are kept for the next tick; that is the
//! one-update delay the C++ stage machine gets from its `else if` chain.

use cubecl::prelude::*;

use super::Voxel3dBuffers;
use super::engine::Voxel3dEngine;
use super::step::{C_LIFE_SHIFT, C_PT_BANG, C_TY_MASK, seed_u32};

/// World edge length.
const WORLD: i32 = 256;
/// Number of flood-queue entries (`VOXEL_COUNT`).
const CAPACITY: u32 = 16_777_216;

const _: () = assert!(CAPACITY as usize == tpt_core::voxel3d::VOXEL_COUNT);

/// Bit operations on the visited bitset.
#[cube]
fn mark_bit(index: u32) -> u32 {
    1u32 << (index & 31u32)
}

/// One serial BFS over the pending seed list.
///
/// `state[0]` is unused (kept for the shared count-buffer shape); `state[1]`
/// holds the number of pending seeds. The BFS work list starts at
/// `seed_count` so the seed list is never overwritten while it is being
/// compacted.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn flood_kernel(
    voxel_a: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &[i32],
    flood_queue: &mut [u32],
    flood_marks: &mut [Atomic<u32>],
    flood_state: &mut [Atomic<u32>],
    tick: u32,
) {
    if CUBE_POS_X == 0u32 && UNIT_POS_X == 0u32 {
        let seed_count = flood_state[1].load();
        let mut keep = seed_u32(tick, 0u32);
        let mut b = seed_u32(tick, 0u32);
        while b < seed_count {
            let index = flood_queue[b as usize];
            let word = voxel_a[index as usize];
            let ty = word & C_TY_MASK;
            let stage = aux_tmp[index as usize];
            let stamp = aux_tmp2[index as usize];
            if ty == C_PT_BANG && stage == 1i32 && stamp != tick as i32 {
                // Compact the seed list in place, then leave room for the BFS
                // work list after it.
                flood_queue[keep as usize] = index;
                keep = keep + 1u32;
            } else if ty == C_PT_BANG && stage == 1i32 {
                // Triggered this tick: process it on the next tick so the
                // stage machine keeps TPT's three-update delay.
                flood_queue[keep as usize] = index;
                keep = keep + 1u32;
            }
            b = b + 1u32;
        }

        let mut tail = seed_count;
        let mut head = seed_count;
        // The compacted seed list is [0, keep); BFS appends from seed_count.
        // Walk the kept seeds and enqueue each connected component.
        let mut si = seed_u32(tick, 0u32);
        while si < keep {
            let seed_index = flood_queue[si as usize];
            let word = voxel_a[seed_index as usize];
            if (word & C_TY_MASK) == C_PT_BANG && aux_tmp[seed_index as usize] == 1i32 {
                let bit = mark_bit(seed_index);
                let old = flood_marks[(seed_index >> 5u32) as usize].fetch_or(bit);
                if (old & bit) == 0u32 {
                    if tail < CAPACITY {
                        flood_queue[tail as usize] = seed_index;
                        tail = tail + 1u32;
                    }
                    // FloodParts sets tmp = 2 on every connected BANG.
                    aux_tmp[seed_index as usize] = 2i32;
                }
            }
            si = si + 1u32;
        }

        while head < tail {
            let index = flood_queue[head as usize];
            head = head + 1u32;
            let x = index & 255u32;
            let y = (index >> 16u32) & 255u32;
            let z = (index >> 8u32) & 255u32;
            for dx in -1i32..=1i32 {
                for dy in -1i32..=1i32 {
                    for dz in -1i32..=1i32 {
                        if dx != 0i32 || dy != 0i32 || dz != 0i32 {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            let nz = z as i32 + dz;
                            if nx >= 0
                                && nx < WORLD
                                && ny >= 0
                                && ny < WORLD
                                && nz >= 0
                                && nz < WORLD
                            {
                                let ni = ((ny as u32) << 16u32) | ((nz as u32) << 8u32) | nx as u32;
                                if (voxel_a[ni as usize] & C_TY_MASK) == C_PT_BANG {
                                    let bit = mark_bit(ni);
                                    let old = flood_marks[(ni >> 5u32) as usize].fetch_or(bit);
                                    if (old & bit) == 0u32 {
                                        if tail < CAPACITY {
                                            flood_queue[tail as usize] = ni;
                                            tail = tail + 1u32;
                                        }
                                        aux_tmp[ni as usize] = 2i32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Clear the visited bits of every cell the BFS queued.
        let mut ci = seed_count;
        while ci < tail {
            let index = flood_queue[ci as usize];
            flood_marks[(index >> 5u32) as usize].fetch_and(!mark_bit(index));
            ci = ci + 1u32;
        }
        let _ = C_LIFE_SHIFT;
        flood_state[0].store(0u32);
        flood_state[1].store(keep);
    }
}

/// Runs the pending TNT flood.
pub(crate) fn apply(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        flood_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp2.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.flood_queue.clone(), CAPACITY as usize),
            BufferArg::from_raw_parts(bufs.flood_marks.clone(), super::buffers::FLOOD_MARK_WORDS),
            BufferArg::from_raw_parts(bufs.flood_state.clone(), 2),
            engine.tick as u32,
        );
    }
}
