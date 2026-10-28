//! Leaf and coarse mask rebuild (plan §4.6/§3.4).
//!
//! The renderer skips empty space with the 2³ leaf bits and the six-level
//! coarse pyramid. Over-reporting is allowed (the ray just descends further);
//! under-reporting is not, because it would cull real geometry. The rebuild is
//! therefore split into three ordered passes:
//!
//! 1. [`leaf_rebuild_kernel`] scans the eight voxels of every 2³ block of a
//!    brick and sets or clears its leaf bit exactly,
//! 2. [`coarse_l0_kernel`] derives a 4³ cell from the eight leaf bits just
//!    written, so the 4³ level is exact as well,
//! 3. [`coarse_l1_kernel`] derives the 8³ bit from the eight 4³ bits, sets
//!    `brick_occupied` exactly and only ever *sets* the 16³…128³ bits.
//!
//! Levels 2…5 are monotone: clearing one needs a scan of the whole 16³…128³
//! region, which no thread owns. The plan explicitly permits conservative
//! coarse masks, and the leaf and 8³ levels stop the ray at an empty brick.
//!
//! The pass runs over [`super::super::engine::Voxel3dEngine::mask_list`], which
//! contains `sim_bits | dirty_bricks`; the dirty half covers the bricks a
//! commit, blast or rigid-body displacement wrote during this tick, before the
//! next snapshot folds them into the simulation set.

use cubecl::prelude::*;

use super::super::Voxel3dBuffers;
use super::super::engine::Voxel3dEngine;
use super::{brick_bit, coarse_cell, seed_u32};

/// Threads per brick for the leaf pass: one per 2³ block.
const BLOCK_DIM: u32 = 4;
/// Voxels along one brick edge.
const BRICK: u32 = 8;
/// Voxels along one leaf block edge.
const LEAF: u32 = 2;
/// Coarse levels whose bits are set but never cleared by this pass.
const CONSERVATIVE_FIRST: u32 = 2;

const _: () = assert!(BLOCK_DIM * BLOCK_DIM * BLOCK_DIM == 64);
const _: () = assert!(BRICK as usize == tpt_core::voxel3d::BRICK_DIM as usize);
const _: () = assert!(LEAF as usize == tpt_core::voxel3d::LEAF_BLOCK as usize);

/// Reads one leaf bit of `brick`.
#[cube]
fn leaf_get(leaf_masks: &[u32], brick: u32, block: u32) -> u32 {
    let word = brick * 2u32 + (block >> 5u32);
    let mask = 1u32 << (block & 31u32);
    let mut out = seed_u32(block, 0u32);
    if (leaf_masks[word as usize] & mask) != 0u32 {
        out = 1u32;
    }
    out
}

/// Rebuilds the leaf bit of one 2³ block from the authoritative voxels.
///
/// One 4x4x4 workgroup per brick; each thread owns one block. The block's
/// eight voxels are read explicitly because CubeCL mis-expands a mutable
/// accumulator declared outside a loop.
#[cube(launch_unchecked)]
fn leaf_rebuild_kernel(voxel_a: &[u32], list: &[u32], leaf_masks: &mut [Atomic<u32>]) {
    let brick = list[CUBE_POS_X as usize];
    let bx = brick & 31u32;
    let bz = (brick >> 5u32) & 31u32;
    let by = (brick >> 10u32) & 31u32;
    let tx = UNIT_POS_X;
    let tz = UNIT_POS_Y;
    let ty = UNIT_POS_Z;
    let x0 = bx * BRICK + tx * LEAF;
    let y0 = by * BRICK + ty * LEAF;
    let z0 = bz * BRICK + tz * LEAF;

    let i000 = (y0 << 16u32) | (z0 << 8u32) | x0;
    let i100 = i000 + 1u32;
    let i010 = ((y0 + 1u32) << 16u32) | (z0 << 8u32) | x0;
    let i110 = i010 + 1u32;
    let i001 = (y0 << 16u32) | ((z0 + 1u32) << 8u32) | x0;
    let i101 = i001 + 1u32;
    let i011 = ((y0 + 1u32) << 16u32) | ((z0 + 1u32) << 8u32) | x0;
    let i111 = i011 + 1u32;

    let mut nonempty = seed_u32(brick, 0u32);
    if voxel_a[i000 as usize] != 0u32
        || voxel_a[i100 as usize] != 0u32
        || voxel_a[i010 as usize] != 0u32
        || voxel_a[i110 as usize] != 0u32
        || voxel_a[i001 as usize] != 0u32
        || voxel_a[i101 as usize] != 0u32
        || voxel_a[i011 as usize] != 0u32
        || voxel_a[i111 as usize] != 0u32
    {
        nonempty = 1u32;
    }

    let block = (ty << 4u32) | (tz << 2u32) | tx;
    let word = brick * 2u32 + (block >> 5u32);
    let mask = 1u32 << (block & 31u32);
    if nonempty != 0u32 {
        leaf_masks[word as usize].fetch_or(mask);
    } else {
        leaf_masks[word as usize].fetch_and(0xFFFF_FFFFu32 ^ mask);
    }
}

/// Rebuilds one 4³ coarse bit from the eight leaf bits it contains.
#[cube(launch_unchecked)]
fn coarse_l0_kernel(list: &[u32], leaf_masks: &[u32], coarse_masks: &mut [Atomic<u32>]) {
    let brick = list[CUBE_POS_X as usize];
    let slot = (UNIT_POS_Z * 16u32) | (UNIT_POS_Y * 4u32) | UNIT_POS_X;
    if slot < 8u32 {
        let bx = brick & 31u32;
        let bz = (brick >> 5u32) & 31u32;
        let by = (brick >> 10u32) & 31u32;
        let lx = (slot & 1u32) * 4u32;
        let lz = ((slot >> 1u32) & 1u32) * 4u32;
        let ly = ((slot >> 2u32) & 1u32) * 4u32;

        let b0 = ((ly >> 1u32) << 4u32) | ((lz >> 1u32) << 2u32) | (lx >> 1u32);
        let b1 = b0 + 1u32;
        let b2 = b0 + 4u32;
        let b3 = b0 + 5u32;
        let b4 = b0 + 16u32;
        let b5 = b0 + 17u32;
        let b6 = b0 + 20u32;
        let b7 = b0 + 21u32;
        let any = leaf_get(leaf_masks, brick, b0)
            | leaf_get(leaf_masks, brick, b1)
            | leaf_get(leaf_masks, brick, b2)
            | leaf_get(leaf_masks, brick, b3)
            | leaf_get(leaf_masks, brick, b4)
            | leaf_get(leaf_masks, brick, b5)
            | leaf_get(leaf_masks, brick, b6)
            | leaf_get(leaf_masks, brick, b7);

        let x = bx * BRICK + lx;
        let y = by * BRICK + ly;
        let z = bz * BRICK + lz;
        let (word, mask) = coarse_cell(0u32, x, y, z);
        if any != 0u32 {
            coarse_masks[word as usize].fetch_or(mask);
        } else {
            coarse_masks[word as usize].fetch_and(0xFFFF_FFFFu32 ^ mask);
        }
    }
}

/// Reads one coarse occupancy bit through the atomic binding.
#[cube]
fn coarse_get(coarse_masks: &mut [Atomic<u32>], level: u32, x: u32, y: u32, z: u32) -> u32 {
    let (word, mask) = coarse_cell(level, x, y, z);
    let mut out = seed_u32(x, 0u32);
    if (coarse_masks[word as usize].fetch_max(0u32) & mask) != 0u32 {
        out = 1u32;
    }
    out
}

/// Rebuilds the 8³ bit and `brick_occupied`, and sets the higher levels.
#[cube(launch_unchecked)]
fn coarse_l1_kernel(
    list: &[u32],
    coarse_masks: &mut [Atomic<u32>],
    brick_occupied: &mut [Atomic<u32>],
) {
    if UNIT_POS_X == 0u32 {
        let brick = list[CUBE_POS_X as usize];
        let bx = brick & 31u32;
        let bz = (brick >> 5u32) & 31u32;
        let by = (brick >> 10u32) & 31u32;
        let x = bx * BRICK;
        let y = by * BRICK;
        let z = bz * BRICK;

        let any = coarse_get(coarse_masks, 0u32, x, y, z)
            | coarse_get(coarse_masks, 0u32, x + 4u32, y, z)
            | coarse_get(coarse_masks, 0u32, x, y, z + 4u32)
            | coarse_get(coarse_masks, 0u32, x + 4u32, y, z + 4u32)
            | coarse_get(coarse_masks, 0u32, x, y + 4u32, z)
            | coarse_get(coarse_masks, 0u32, x + 4u32, y + 4u32, z)
            | coarse_get(coarse_masks, 0u32, x, y + 4u32, z + 4u32)
            | coarse_get(coarse_masks, 0u32, x + 4u32, y + 4u32, z + 4u32);

        let (word, mask) = coarse_cell(1u32, x, y, z);
        if any != 0u32 {
            coarse_masks[word as usize].fetch_or(mask);
            for level in CONSERVATIVE_FIRST..6u32 {
                let (word_hi, mask_hi) = coarse_cell(level, x, y, z);
                coarse_masks[word_hi as usize].fetch_or(mask_hi);
            }
        } else {
            coarse_masks[word as usize].fetch_and(0xFFFF_FFFFu32 ^ mask);
        }

        let (bword, bbit) = brick_bit(brick);
        if any != 0u32 {
            brick_occupied[bword as usize].fetch_or(bbit);
        } else {
            brick_occupied[bword as usize].fetch_and(0xFFFF_FFFFu32 ^ bbit);
        }
    }
}

/// Rebuilds the leaf and coarse masks for `sim_bits | dirty_bricks`.
pub(crate) fn rebuild(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    let cubes = CubeCount::Dynamic(engine.mask_count.clone().binding());
    let block_dim = CubeDim::new_3d(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    unsafe {
        leaf_rebuild_kernel::launch_unchecked(
            &engine.client,
            cubes.clone(),
            block_dim,
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.mask_list.clone(), super::C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(bufs.leaf_masks.clone(), tpt_core::voxel3d::LEAF_WORD_COUNT),
        );
        coarse_l0_kernel::launch_unchecked(
            &engine.client,
            cubes.clone(),
            block_dim,
            BufferArg::from_raw_parts(engine.mask_list.clone(), super::C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(bufs.leaf_masks.clone(), tpt_core::voxel3d::LEAF_WORD_COUNT),
            BufferArg::from_raw_parts(bufs.coarse_masks.clone(), super::super::COARSE_MASK_WORDS),
        );
        coarse_l1_kernel::launch_unchecked(
            &engine.client,
            cubes,
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(engine.mask_list.clone(), super::C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(bufs.coarse_masks.clone(), super::super::COARSE_MASK_WORDS),
            BufferArg::from_raw_parts(bufs.brick_occupied.clone(), super::C_BRICK_WORDS as usize),
        );
    }
}
