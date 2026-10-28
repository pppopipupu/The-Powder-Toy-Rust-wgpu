//! BOMB detonation fill (plan §4.3/§6.4).
//!
//! The reaction pass appends one request per detonating BOMB voxel to the
//! explosion queue; this pass runs once per request after the movement commit.
//! TPT's `Element_BOMB_update` (`BOMB.cpp:73-113`) kills the bomb and then
//! fills a radius-8 disc with EMBR (`tmp = 2`, `life = 2`, `temp = MAX_TEMP`,
//! `pv += 0.1` per cell) and a radius-9 shell with EMPTY-only EMBR
//! (`tmp = 0`, `life = 50`, `temp = MAX_TEMP`). The 3D lift replaces the 2D
//! discs with spheres of the same radii; the shell is a CREATE record so it
//! still never overwrites anything, exactly like the `if (TYP(pmap)) continue`
//! guard in the C++ shell loop.
//!
//! Requests are records in the mutation queue, so the drain owns every write
//! and all mask bits are marked in one place. Pressure impulses are applied
//! here because they do not change voxel state and are atomic.

//! Clippy's `Range::contains` and `collapsible_if` rewrites use constructs the
//! CubeCL IR does not expand, so both lints are disabled for this module.
#![allow(clippy::collapsible_if, clippy::manual_range_contains)]

use cubecl::prelude::*;

use super::super::Voxel3dBuffers;
use super::super::engine::Voxel3dEngine;
use super::super::mutation::{MODE_CREATE, MODE_REPLACE, push};
use super::{
    C_LIFE_SHIFT, C_NONE, C_PT_BOMB, C_PT_EMBR, C_TEMP_MASK, C_TEMP_SHIFT, C_WORLD_DIM, seed_u32,
};

/// Radius of the EMBR disc (`BOMB.cpp:74`).
const RADIUS: i32 = 8;
/// Radius of the outer shell (`BOMB.cpp:98`).
const SHELL_RADIUS: i32 = RADIUS + 1;
/// Words per explosion request.
const STRIDE: u32 = 4;
/// TPT `MAX_TEMP`; the packed word saturates.
const MAX_TEMP_K: f32 = 8191.875;
/// World edge length.
const WORLD: i32 = C_WORLD_DIM as i32;

const _: () = assert!(STRIDE == 4u32);
const _: () = assert!(RADIUS + 1 == SHELL_RADIUS);

/// Air-cell index of a world coordinate (`CELL = 4`).
#[cube]
fn air_cell(x: u32, y: u32, z: u32) -> u32 {
    ((y >> 2u32) << 12u32) | ((z >> 2u32) << 6u32) | (x >> 2u32)
}

/// Packed word for an EMBR at `temp_k` with `life`.
#[cube]
fn embr_word(temp_k: f32, life: u32) -> u32 {
    let mut q = (temp_k * 8.0f32).round() as u32;
    if q > 65_535u32 {
        q = 65_535u32;
    }
    C_PT_EMBR | (q << C_TEMP_SHIFT) | (life << C_LIFE_SHIFT)
}

/// Applies every queued explosion.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn blast_kernel(
    explosion_queue: &[u32],
    explosion_count: &[Atomic<u32>],
    voxel_a: &[u32],
    pv: &mut [Atomic<i32>],
    mutation_queue: &mut [u32],
    mutation_count: &mut [Atomic<u32>],
) {
    let slot = CUBE_POS_X;
    if slot < explosion_count[0].load() {
        let entry = (slot * STRIDE) as usize;
        let cx = explosion_queue[entry];
        let cy = explosion_queue[entry + 1usize];
        let cz = explosion_queue[entry + 2usize];
        let radius = f32::reinterpret(explosion_queue[entry + 3usize]);
        // Only BOMB's sentinel (-1) is produced in this wave. A positive
        // radius would be GUNP's, which is deferred with the explosion table.
        if radius < 0.0f32 {
            let xi = cx as i32;
            let yi = cy as i32;
            let zi = cz as i32;
            for dx in -RADIUS..=RADIUS {
                for dy in -RADIUS..=RADIUS {
                    for dz in -RADIUS..=RADIUS {
                        let d2 = dx * dx + dy * dy + dz * dz;
                        if d2 <= RADIUS * RADIUS {
                            let nx = xi + dx;
                            let ny = yi + dy;
                            let nz = zi + dz;
                            if nx >= 0
                                && nx < WORLD
                                && ny >= 0
                                && ny < WORLD
                                && nz >= 0
                                && nz < WORLD
                            {
                                let target =
                                    ((ny as u32) << 16u32) | ((nz as u32) << 8u32) | nx as u32;
                                // `pv[..] += 0.1f` per disc cell.
                                pv[air_cell(nx as u32, ny as u32, nz as u32) as usize]
                                    .fetch_add((0.1f32 * 1024.0f32) as i32);
                                push(
                                    mutation_queue,
                                    mutation_count,
                                    target,
                                    MODE_REPLACE,
                                    embr_word(MAX_TEMP_K, 2u32),
                                    0u32,
                                    2i32,
                                    0i32,
                                    0i32,
                                );
                            }
                        }
                    }
                }
            }
            for dx in -SHELL_RADIUS..=SHELL_RADIUS {
                for dy in -SHELL_RADIUS..=SHELL_RADIUS {
                    for dz in -SHELL_RADIUS..=SHELL_RADIUS {
                        let d2 = dx * dx + dy * dy + dz * dz;
                        if d2 <= SHELL_RADIUS * SHELL_RADIUS && d2 > RADIUS * RADIUS {
                            let nx = xi + dx;
                            let ny = yi + dy;
                            let nz = zi + dz;
                            if nx >= 0
                                && nx < WORLD
                                && ny >= 0
                                && ny < WORLD
                                && nz >= 0
                                && nz < WORLD
                            {
                                let target =
                                    ((ny as u32) << 16u32) | ((nz as u32) << 8u32) | nx as u32;
                                if voxel_a[target as usize] == 0u32 {
                                    push(
                                        mutation_queue,
                                        mutation_count,
                                        target,
                                        MODE_CREATE,
                                        embr_word(MAX_TEMP_K, 50u32),
                                        0u32,
                                        0i32,
                                        0i32,
                                        0i32,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        let _ = seed_u32(slot, C_NONE);
    }
}

/// Applies the tick's explosion queue.
pub(crate) fn apply(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        blast_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(super::super::engine::EXPLOSION_CAPACITY as u32, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(
                engine.explosion_queue.clone(),
                super::super::engine::EXPLOSION_CAPACITY * super::super::engine::EXPLOSION_STRIDE,
            ),
            BufferArg::from_raw_parts(engine.explosion_count.clone(), 3),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.air.pv.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(
                bufs.mutation_queue.clone(),
                (super::super::mutation::MUTATION_CAPACITY
                    * super::super::mutation::MUTATION_STRIDE) as usize,
            ),
            BufferArg::from_raw_parts(bufs.mutation_count.clone(), 3),
        );
    }
    let _ = C_PT_BOMB;
    let _ = C_TEMP_MASK;
}
