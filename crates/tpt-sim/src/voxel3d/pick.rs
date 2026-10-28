//! GPU voxel picking for the 3D UI (plan §1.1).
//!
//! A click casts one ray into the world: a single-thread kernel walks it in
//! half-voxel steps (2,048 steps cover 1,024 voxels, four times the world
//! edge) and writes the first non-empty hit into four words. The host then
//! reads those four words back — this is the only read-back on the edit path
//! and it runs once per click, never inside `step`.
//!
//! Stepping by `0.5` cannot skip a 1-voxel cube: the chord of a unit cube along
//! any direction is at least one voxel long. The normal is the difference
//! between the previous and the hit sample cell on the axis the ray crossed;
//! when the ray starts inside the voxel (or grazes along a face), the direction
//! sign is used instead.

use cubecl::prelude::*;

use super::Voxel3dEngine;
use super::step::{C_WORLD_DIM, seed_i32};

/// Number of half-voxel steps a pick may take.
pub const PICK_STEPS: u32 = 2_048;
/// Words written by the pick kernel.
pub const PICK_WORDS: usize = 4;

/// A picked voxel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickHit {
    /// Integer voxel coordinates.
    pub voxel: [u32; 3],
    /// Outward face normal, one component is `±1`.
    pub normal: [i32; 3],
    /// Element id.
    pub ty: u32,
    /// Temperature decoded from the fixed-point field.
    pub temp_k: f32,
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pick_kernel(
    voxel_a: &[u32],
    out: &mut [u32],
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
) {
    out[0] = 0u32;
    out[1] = 0u32;
    out[2] = 0u32;
    out[3] = 0u32;

    for step in 0..PICK_STEPS {
        if out[0] == 0u32 {
            let t = step as f32 * 0.5f32;
            let px = ox + dx * t;
            let py = oy + dy * t;
            let pz = oz + dz * t;
            let ix = px.floor() as i32;
            let iy = py.floor() as i32;
            let iz = pz.floor() as i32;
            if ix >= 0i32
                && ix < C_WORLD_DIM as i32
                && iy >= 0i32
                && iy < C_WORLD_DIM as i32
                && iz >= 0i32
                && iz < C_WORLD_DIM as i32
            {
                let index = ((iy as u32) << 16u32) | ((iz as u32) << 8u32) | ix as u32;
                let word = voxel_a[index as usize];
                if word != 0u32 {
                    let mut nx = seed_i32(ix, 0i32);
                    let mut ny = seed_i32(iy, 0i32);
                    let mut nz = seed_i32(iz, 0i32);
                    if step > 0u32 {
                        let previous = t - 0.5f32;
                        let qx = (ox + dx * previous).floor() as i32;
                        let qy = (oy + dy * previous).floor() as i32;
                        let qz = (oz + dz * previous).floor() as i32;
                        nx = qx - ix;
                        ny = qy - iy;
                        nz = qz - iz;
                    } else {
                        if dx > 0.0f32 {
                            nx = -1i32;
                        } else if dx < 0.0f32 {
                            nx = 1i32;
                        }
                        if dy > 0.0f32 {
                            ny = -1i32;
                        } else if dy < 0.0f32 {
                            ny = 1i32;
                        }
                        if dz > 0.0f32 {
                            nz = -1i32;
                        } else if dz < 0.0f32 {
                            nz = 1i32;
                        }
                    }
                    if nx == 0i32 && ny == 0i32 && nz == 0i32 {
                        if dx > 0.0f32 {
                            nx = -1i32;
                        } else if dx < 0.0f32 {
                            nx = 1i32;
                        }
                        if dy > 0.0f32 {
                            ny = -1i32;
                        } else if dy < 0.0f32 {
                            ny = 1i32;
                        }
                        if dz > 0.0f32 {
                            nz = -1i32;
                        } else if dz < 0.0f32 {
                            nz = 1i32;
                        }
                    }
                    out[0] = 1u32;
                    out[1] = (ix as u32) | ((iy as u32) << 8u32) | ((iz as u32) << 16u32);
                    out[2] = ((nx + 1i32) as u32)
                        | (((ny + 1i32) as u32) << 2u32)
                        | (((nz + 1i32) as u32) << 4u32);
                    out[3] = word;
                    break;
                }
            }
        }
    }
}

/// Casts a ray from `origin` along `dir` and returns the first voxel.
pub(crate) fn pick(engine: &Voxel3dEngine, origin: [f32; 3], dir: [f32; 3]) -> Option<PickHit> {
    let length = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if !length.is_finite() || length < 1.0e-6f32 {
        return None;
    }
    let dx = dir[0] / length;
    let dy = dir[1] / length;
    let dz = dir[2] / length;

    let zeros = [0u32; PICK_WORDS];
    let out = engine
        .client
        .create_from_slice(bytemuck::cast_slice(&zeros));
    unsafe {
        pick_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(engine.bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(out.clone(), PICK_WORDS),
            origin[0],
            origin[1],
            origin[2],
            dx,
            dy,
            dz,
        );
    }

    let bytes = engine.client.read(vec![out]).ok()?;
    let raw = bytes.first()?;
    if raw.len() < PICK_WORDS * 4 {
        return None;
    }
    let mut words = [0u32; PICK_WORDS];
    for (slot, word) in words.iter_mut().enumerate() {
        let start = slot * 4;
        *word = u32::from_le_bytes(raw[start..start + 4].try_into().ok()?);
    }
    if words[0] == 0u32 {
        return None;
    }

    let packed = words[1];
    let normal = words[2];
    let word = words[3];
    let voxel = [
        packed & 0xFFu32,
        (packed >> 8u32) & 0xFFu32,
        (packed >> 16u32) & 0xFFu32,
    ];
    let normal = [
        (normal & 3u32) as i32 - 1i32,
        ((normal >> 2u32) & 3u32) as i32 - 1i32,
        ((normal >> 4u32) & 3u32) as i32 - 1i32,
    ];
    let ty = word & 0xFFu32;
    let temp_q = (word >> 8u32) & 0xFFFFu32;
    Some(PickHit {
        voxel,
        normal,
        ty,
        temp_k: temp_q as f32 * 0.125f32,
    })
}

const _: () = assert!(PICK_WORDS == 4);
const _: () = assert!(C_WORLD_DIM == 256u32);
