//! TPT `PROP_LIFE_DEC` / `PROP_LIFE_KILL` pass.
//!
//! `Simulation::RecalcFreeParticles(true)` runs at the start of every tick
//! (`Simulation.cpp:3378-3425`): a particle with `PROP_LIFE_DEC` and
//! `life > 0` loses one life, and `PROP_LIFE_KILL`/`PROP_LIFE_KILL_DEC`
//! removes it when the counter is exhausted. The 3D engine applies the same
//! update in place on `voxel_a` after the air solve and before the reaction
//! pass, so the reaction branches observe the post-decrement life exactly as
//! `Element_*_update` does in TPT.
//!
//! The pass only touches bricks in the tick's simulation list, and clearing a
//! voxel also clears its four auxiliary words: a dead particle must not leave
//! `ctype`/`tmp` behind for the next element painted into the cell.

use cubecl::prelude::*;

use super::super::Voxel3dBuffers;
use super::super::engine::Voxel3dEngine;
use super::{
    C_BRICK_COUNT, C_ELEM_TPT_PROPS, C_ELEM_WORDS, C_LIFE_MASK, C_LIFE_SHIFT, C_TPT_LIFE_DEC,
    C_TPT_LIFE_KILL, C_TPT_LIFE_KILL_DEC, C_TY_MASK, seed_u32,
};

/// Voxels along one brick edge.
const BRICK: u32 = 8;
/// Threads per brick: one per voxel.
const CUBE_DIM: u32 = 8;

/// One thread per voxel of every simulated brick.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn life_kernel(
    voxel_a: &mut [u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    sim_list: &[u32],
    table: &[u32],
) {
    let brick = sim_list[CUBE_POS_X as usize];
    let x = (brick & 31u32) * BRICK + UNIT_POS_X;
    let y = ((brick >> 10u32) & 31u32) * BRICK + UNIT_POS_Y;
    let z = ((brick >> 5u32) & 31u32) * BRICK + UNIT_POS_Z;
    let index = (y << 16u32) | (z << 8u32) | x;
    let word = voxel_a[index as usize];
    let ty = word & C_TY_MASK;
    if ty != 0u32 {
        let base = (ty as usize) * C_ELEM_WORDS;
        let props = table[base + C_ELEM_TPT_PROPS];
        let life = (word >> C_LIFE_SHIFT) & C_LIFE_MASK;
        let mut dead = seed_u32(index, 0u32);
        let mut next_life = seed_u32(index, life);
        if (props & C_TPT_LIFE_DEC) != 0u32 && life > 0u32 {
            next_life = life - 1u32;
            if next_life == 0u32 && (props & (C_TPT_LIFE_KILL | C_TPT_LIFE_KILL_DEC)) != 0u32 {
                dead = 1u32;
            }
        } else if life == 0u32 && (props & C_TPT_LIFE_KILL) != 0u32 {
            dead = 1u32;
        }
        if dead != 0u32 {
            voxel_a[index as usize] = 0u32;
            aux_ctype[index as usize] = 0u32;
            aux_tmp[index as usize] = 0i32;
            aux_tmp2[index as usize] = 0i32;
            aux_tmp3[index as usize] = 0i32;
        } else if next_life != life {
            voxel_a[index as usize] =
                (word & !(C_LIFE_MASK << C_LIFE_SHIFT)) | (next_life << C_LIFE_SHIFT);
        }
    }
}

/// Applies `PROP_LIFE_DEC` / `PROP_LIFE_KILL` to the simulated bricks.
pub(crate) fn apply(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        life_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Dynamic(bufs.sim_count.clone().binding()),
            CubeDim::new_3d(CUBE_DIM, CUBE_DIM, CUBE_DIM),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_ctype.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp2.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp3.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
        );
    }
}
