//! Device creation/mutation queue and drain pass.
//!
//! Element updates run in parallel over the whole world, so a branch that has
//! to create, replace or kill *another* voxel cannot write that cell directly
//! without racing its owner. TPT's `create_part`/`part_change_type`/`kill_part`
//! are immediately visible to later particles in the sequential update loop;
//! the 3D port records the same intent here and applies it in one ordered pass
//! after the movement commit.
//!
//! # Record layout
//!
//! ```text
//! word 0 target linear voxel index
//! word 1 mode (CREATE = only if empty, REPLACE = overwrite, KILL = zero)
//! word 2 packed voxel word to write
//! word 3 ctype
//! word 4 tmp   (i32 bits)
//! word 5 tmp2  (i32 bits)
//! word 6 tmp3  (i32 bits)
//! word 7 reserved
//! ```
//!
//! Producers only append with `atomicAdd` on `mutation_count[0]`. A full queue
//! drops the request, the same observable outcome as `create_part` failing to
//! allocate a particle (`Simulation.cpp:2991-3000`).
//!
//! # Arbitration
//!
//! Two producers may target the same cell (for example two fires igniting one
//! grain). TPT resolves by particle iteration order. The parallel port cannot
//! reproduce that order without a sort, so the drain uses a documented,
//! deterministic priority: `REPLACE` > `KILL` > `CREATE`, ties broken by queue
//! slot. Element branches whose source order matters within one cell (TNT
//! flood, H2/O2 neighbours) are written into the target's own record and are
//! not affected by this rule.
//!
//! The drain owns all writes in its pass: it marks leaf/coarse/brick bits and
//! clears the four auxiliary words whenever it writes an empty word.

use cubecl::prelude::*;

use super::Voxel3dBuffers;
use super::engine::Voxel3dEngine;
use super::step::{
    C_BRICK_COUNT, C_NONE, C_TY_MASK, brick_bit, brick_of_index, mark_brick_bits, mark_leaf_coarse,
    seed_u32,
};
use tpt_core::voxel3d::VOXEL_COUNT;

/// Number of records the queue holds.
pub const MUTATION_CAPACITY: u32 = 65_535;
/// Words per record.
pub const MUTATION_STRIDE: u32 = 8;
/// Only create when the target is empty (`create_part(-1, ...)`).
pub const MODE_CREATE: u32 = 0;
/// Replace the target (`create_part(i, ...)` / `part_change_type`).
pub const MODE_REPLACE: u32 = 1;
/// Kill the target (`kill_part`).
pub const MODE_KILL: u32 = 2;

const _: () = assert!(MUTATION_CAPACITY == super::buffers::MUTATION_CAPACITY as u32);
const _: () = assert!(MUTATION_STRIDE == super::buffers::MUTATION_STRIDE as u32);

/// Appends one mutation record; producers only ever append.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn push(
    queue: &mut [u32],
    count: &mut [Atomic<u32>],
    target: u32,
    mode: u32,
    word: u32,
    ctype: u32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
) {
    let slot = count[0].fetch_add(1u32);
    if slot < MUTATION_CAPACITY {
        let base = (slot * MUTATION_STRIDE) as usize;
        queue[base] = target;
        queue[base + 1usize] = mode;
        queue[base + 2usize] = word;
        queue[base + 3usize] = ctype;
        queue[base + 4usize] = u32::reinterpret(tmp);
        queue[base + 5usize] = u32::reinterpret(tmp2);
        queue[base + 6usize] = u32::reinterpret(tmp3);
        queue[base + 7usize] = 0u32;
    }
}

/// Priority key written into `spawn_claim[target]`; larger wins.
#[cube]
fn claim_key(mode: u32, slot: u32) -> u32 {
    let mut key = seed_u32(slot, 0x2000_0000u32 | slot);
    if mode == MODE_REPLACE {
        key = 0x8000_0000u32 | slot;
    } else if mode == MODE_KILL {
        key = 0x4000_0000u32 | slot;
    }
    key
}

/// One thread per record: publishes its priority for the target cell.
#[cube(launch_unchecked)]
fn arbitrate_kernel(queue: &[u32], count: &[Atomic<u32>], spawn_claim: &mut [Atomic<u32>]) {
    let slot = CUBE_POS_X;
    if slot < count[0].load() {
        let base = (slot * MUTATION_STRIDE) as usize;
        let target = queue[base];
        let mode = queue[base + 1usize];
        if target != C_NONE {
            spawn_claim[target as usize].fetch_max(claim_key(mode, slot));
        }
    }
}

/// One thread per record: the winner applies its record.
///
/// Winner priority follows TPT's operation classes: a replacement wins over a
/// kill, which wins over a conditional create, because a later `create_part`
/// sees the earlier operations. Ties use the queue slot.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn apply_kernel(
    queue: &[u32],
    count: &[Atomic<u32>],
    spawn_claim: &mut [Atomic<u32>],
    voxel_a: &mut [u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    leaf_masks: &mut [Atomic<u32>],
    coarse_masks: &mut [Atomic<u32>],
    brick_occupied: &mut [Atomic<u32>],
    dirty_bricks: &mut [Atomic<u32>],
) {
    let slot = CUBE_POS_X;
    if slot < count[0].load() {
        let base = (slot * MUTATION_STRIDE) as usize;
        let target = queue[base];
        let mode = queue[base + 1usize];
        if target != C_NONE && target < VOXEL_COUNT as u32 {
            let key = claim_key(mode, slot);
            if spawn_claim[target as usize].load() == key {
                let mut write = seed_u32(slot, 1u32);
                if mode == MODE_CREATE && voxel_a[target as usize] != 0u32 {
                    // `create_part(-1, ...)` returns -1 on an occupied cell.
                    write = 0u32;
                }
                if write != 0u32 {
                    let word = if mode == MODE_KILL {
                        0u32
                    } else {
                        queue[base + 2usize]
                    };
                    voxel_a[target as usize] = word;
                    if word == 0u32 {
                        aux_ctype[target as usize] = 0u32;
                        aux_tmp[target as usize] = 0i32;
                        aux_tmp2[target as usize] = 0i32;
                        aux_tmp3[target as usize] = 0i32;
                    } else {
                        aux_ctype[target as usize] = queue[base + 3usize];
                        aux_tmp[target as usize] = i32::reinterpret(queue[base + 4usize]);
                        aux_tmp2[target as usize] = i32::reinterpret(queue[base + 5usize]);
                        aux_tmp3[target as usize] = i32::reinterpret(queue[base + 6usize]);
                    }
                    let brick = brick_of_index(target);
                    if word != 0u32 && (word & C_TY_MASK) != 0u32 {
                        mark_leaf_coarse(leaf_masks, coarse_masks, target);
                        mark_brick_bits(brick_occupied, dirty_bricks, brick);
                    } else {
                        // A cleared cell only needs its masks rebuilt; the
                        // rebuild pass clears leaf/coarse from occupancy.
                        let (word_index, bit) = brick_bit(brick);
                        dirty_bricks[word_index as usize].fetch_or(bit);
                    }
                }
                // The winner releases the cell for the next tick. Losers can
                // observe zero and simply do nothing.
                spawn_claim[target as usize].store(0u32);
            }
        }
    }
}

/// Applies the tick's mutation queue into `voxel_a`.
pub(crate) fn drain(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        arbitrate_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(MUTATION_CAPACITY, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(
                bufs.mutation_queue.clone(),
                (MUTATION_CAPACITY * MUTATION_STRIDE) as usize,
            ),
            BufferArg::from_raw_parts(bufs.mutation_count.clone(), 3),
            BufferArg::from_raw_parts(bufs.spawn_claim.clone(), VOXEL_COUNT),
        );
        apply_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(MUTATION_CAPACITY, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(
                bufs.mutation_queue.clone(),
                (MUTATION_CAPACITY * MUTATION_STRIDE) as usize,
            ),
            BufferArg::from_raw_parts(bufs.mutation_count.clone(), 3),
            BufferArg::from_raw_parts(bufs.spawn_claim.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_ctype.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp2.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp3.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.leaf_masks.clone(), tpt_core::voxel3d::LEAF_WORD_COUNT),
            BufferArg::from_raw_parts(bufs.coarse_masks.clone(), super::buffers::COARSE_MASK_WORDS),
            BufferArg::from_raw_parts(bufs.brick_occupied.clone(), super::buffers::BRICK_BIT_WORDS),
            BufferArg::from_raw_parts(bufs.dirty_bricks.clone(), super::buffers::BRICK_BIT_WORDS),
        );
    }
    let _ = C_BRICK_COUNT;
}
