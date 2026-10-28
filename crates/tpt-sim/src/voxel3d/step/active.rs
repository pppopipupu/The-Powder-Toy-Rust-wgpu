//! Active-brick snapshot, compaction and previous-frame bookkeeping.
//!
//! The world is only simulated where `brick_occupied` says there is at least
//! one voxel. Each tick:
//!
//! 1. `active_bits = brick_occupied`, `sim_bits = active | prev | dirty`,
//!    then `dirty` is cleared (plan §4.7),
//! 2. every set bit of `sim_bits` is appended to `sim_list` and `sim_count[0]`
//!    ends up holding the number of appended bricks, so later passes can use
//!    `CubeCount::Dynamic(sim_count.binding())`,
//! 3. after the passes, `prev_active_bits = active_bits`.
//!
//! The compaction is not order stable: two bricks share a slot only through
//! `atomicAdd`, so the list order depends on thread scheduling. No pass
//! depends on the order.

use cubecl::prelude::*;

use super::super::Voxel3dBuffers;
use super::super::engine::Voxel3dEngine;
use super::C_BRICK_COUNT;
use super::C_BRICK_WORDS;

/// Threads of the fixed 1,024-word bitset passes.
const BITSET_THREADS: u32 = 256;
/// Workgroups of the fixed 1,024-word bitset passes.
const BITSET_CUBES: u32 = C_BRICK_WORDS / BITSET_THREADS;

const _: () = assert!(BITSET_CUBES * BITSET_THREADS == C_BRICK_WORDS);

/// Writes the indirect dispatch arguments `[0, 1, 1]` into both counters.
///
/// The two buffers are independent: `sim_count` drives the brick passes and
/// `explosion_count` the blast pass. Both are `[x, y, z]` workgroup counts, so
/// `y` and `z` stay one and only `x` grows.
#[cube(launch_unchecked)]
fn reset_counts_kernel(
    sim_count: &mut [Atomic<u32>],
    explosion_count: &mut [Atomic<u32>],
    mutation_count: &mut [Atomic<u32>],
    flood_state: &mut [Atomic<u32>],
) {
    if ABSOLUTE_POS == 0usize {
        sim_count[0].store(0u32);
        sim_count[1].store(1u32);
        sim_count[2].store(1u32);
        explosion_count[0].store(0u32);
        explosion_count[1].store(1u32);
        explosion_count[2].store(1u32);
        mutation_count[0].store(0u32);
        mutation_count[1].store(1u32);
        mutation_count[2].store(1u32);
        flood_state[0].store(0u32);
        flood_state[1].store(0u32);
    }
}

/// Writes `[0, 1, 1]` into one dispatch-argument buffer.
#[cube(launch_unchecked)]
fn reset_one_count_kernel(count: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        count[0].store(0u32);
        count[1].store(1u32);
        count[2].store(1u32);
    }
}

/// Snapshots the occupancy bitset and folds the dirty set into the tick's
/// simulation set.
///
/// One thread per `u32` word. `dirty` is read before it is cleared, so a brick
/// written directly to `voxel_a` by the previous tick's blast or rigid pass is
/// processed once more even when it has no occupied bit.
#[cube(launch_unchecked)]
fn snapshot_kernel(
    brick_occupied: &[u32],
    prev_active_bits: &[u32],
    active_bits: &mut [u32],
    sim_bits: &mut [u32],
    dirty_bricks: &mut [Atomic<u32>],
) {
    let word = ABSOLUTE_POS;
    if word < C_BRICK_WORDS as usize {
        let occupied = brick_occupied[word];
        let dirty = dirty_bricks[word].load();
        active_bits[word] = occupied;
        sim_bits[word] = occupied | prev_active_bits[word] | dirty;
        dirty_bricks[word].store(0u32);
    }
}

/// Appends every set bit of `bits` to `list`.
///
/// `count[0]` is the atomic write cursor; after the launch it holds the number
/// of appended bricks. The capacity guard makes a corrupted mask bitset
/// harmless instead of an out-of-bounds write.
#[cube(launch_unchecked)]
fn compact_kernel(bits: &[u32], list: &mut [u32], count: &mut [Atomic<u32>]) {
    let word = ABSOLUTE_POS;
    if word < C_BRICK_WORDS as usize {
        let value = bits[word];
        let base = word as u32 * 32u32;
        for bit in 0..32u32 {
            if (value & (1u32 << bit)) != 0u32 {
                let slot = count[0].fetch_add(1u32);
                if slot < C_BRICK_COUNT {
                    list[slot as usize] = base + bit;
                }
            }
        }
    }
}

/// Appends every set bit of `a | b` to `list`.
///
/// Used for the mask rebuild, which must cover the tick's simulation bricks
/// *and* every brick a direct write (commit, blast, rigid displacement)
/// touched during the same tick.
#[cube(launch_unchecked)]
fn compact_union_kernel(a: &[u32], b: &[u32], list: &mut [u32], count: &mut [Atomic<u32>]) {
    let word = ABSOLUTE_POS;
    if word < C_BRICK_WORDS as usize {
        let value = a[word] | b[word];
        let base = word as u32 * 32u32;
        for bit in 0..32u32 {
            if (value & (1u32 << bit)) != 0u32 {
                let slot = count[0].fetch_add(1u32);
                if slot < C_BRICK_COUNT {
                    list[slot as usize] = base + bit;
                }
            }
        }
    }
}

/// Copies the tick-start activity bitset into the previous-frame bitset.
#[cube(launch_unchecked)]
fn snapshot_prev_kernel(active_bits: &[u32], prev_active_bits: &mut [u32]) {
    let word = ABSOLUTE_POS;
    if word < C_BRICK_WORDS as usize {
        prev_active_bits[word] = active_bits[word];
    }
}

/// Clears both dispatch counters to `[0, 1, 1]`.
pub(crate) fn reset_counts(engine: &Voxel3dEngine) {
    unsafe {
        reset_counts_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(engine.bufs.sim_count.clone(), 3),
            BufferArg::from_raw_parts(engine.explosion_count.clone(), 3),
            BufferArg::from_raw_parts(engine.bufs.mutation_count.clone(), 3),
            BufferArg::from_raw_parts(engine.bufs.flood_state.clone(), 2),
        );
    }
}

/// Snapshots `brick_occupied` into the tick's activity sets.
pub(crate) fn snapshot(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        snapshot_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(BITSET_CUBES, 1, 1),
            CubeDim::new_1d(BITSET_THREADS),
            BufferArg::from_raw_parts(bufs.brick_occupied.clone(), C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(bufs.prev_active_bits.clone(), C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(bufs.active_bits.clone(), C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(bufs.sim_bits.clone(), C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(bufs.dirty_bricks.clone(), C_BRICK_WORDS as usize),
        );
    }
}

/// Compacts `sim_bits` into `sim_list`; `sim_count[0]` becomes the brick count.
pub(crate) fn compact(engine: &Voxel3dEngine) {
    let bufs = &engine.bufs;
    unsafe {
        compact_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(BITSET_CUBES, 1, 1),
            CubeDim::new_1d(BITSET_THREADS),
            BufferArg::from_raw_parts(bufs.sim_bits.clone(), C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(bufs.sim_count.clone(), 3),
        );
    }
}

/// Builds the mask-rebuild list from `sim_bits | dirty_bricks`.
///
/// The list lands in [`Voxel3dEngine::mask_list`] and the count in
/// [`Voxel3dEngine::mask_count`] so the shared `sim_list` stays untouched
/// until the next snapshot. `dirty_bricks` is *not* cleared here: the next
/// snapshot consumes and clears it.
pub(crate) fn compact_mask_list(engine: &Voxel3dEngine) {
    unsafe {
        reset_one_count_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(engine.mask_count.clone(), 3),
        );
        compact_union_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(BITSET_CUBES, 1, 1),
            CubeDim::new_1d(BITSET_THREADS),
            BufferArg::from_raw_parts(engine.bufs.sim_bits.clone(), C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(engine.bufs.dirty_bricks.clone(), C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(engine.mask_list.clone(), C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(engine.mask_count.clone(), 3),
        );
    }
}

/// Stores this tick's activity snapshot as the previous-frame bitset.
pub(crate) fn snapshot_prev(engine: &Voxel3dEngine) {
    let bufs = &engine.bufs;
    unsafe {
        snapshot_prev_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(BITSET_CUBES, 1, 1),
            CubeDim::new_1d(BITSET_THREADS),
            BufferArg::from_raw_parts(bufs.active_bits.clone(), C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(bufs.prev_active_bits.clone(), C_BRICK_WORDS as usize),
        );
    }
}
