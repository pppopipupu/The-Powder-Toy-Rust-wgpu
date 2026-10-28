//! Device-side zeroing for the 256^3 voxel arrays.
//!
//! `Client::empty` returns recycled pages, so the three large arrays must be
//! cleared once before the world is first used. Clearing on the device avoids
//! uploading 192 MiB of zeroes from the host and keeps the allocation path
//! independent of host memory.

use cubecl::prelude::*;
use tpt_core::voxel3d::VOXEL_COUNT;

use super::Voxel3dBuffers;

/// Clears `voxel_a`, `voxel_b`, `claim`, the four auxiliary arrays and the
/// mutation claim in one pass.
///
/// The 256x256 cube grid times 256 threads covers exactly [`VOXEL_COUNT`]
/// words. The kernel writes all eight arrays with the same index, so the
/// buffers always start in the same empty state.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn clear_world_kernel(
    voxel_a: &mut [u32],
    voxel_b: &mut [u32],
    claim: &mut [u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    spawn_claim: &mut [u32],
    len: u32,
) {
    let index = (CUBE_POS_Y * 256u32 + CUBE_POS_X) * 256u32 + UNIT_POS_X;
    if index < len {
        voxel_a[index as usize] = 0u32;
        voxel_b[index as usize] = 0u32;
        claim[index as usize] = 0u32;
        aux_ctype[index as usize] = 0u32;
        aux_tmp[index as usize] = 0i32;
        aux_tmp2[index as usize] = 0i32;
        aux_tmp3[index as usize] = 0i32;
        spawn_claim[index as usize] = 0u32;
    }
}

impl Voxel3dBuffers {
    /// Zeroes the authoritative state, scratch copy and claim buffer.
    ///
    /// Called by [`Voxel3dBuffers::new`] before the world is exposed.
    pub fn clear_world(&self) {
        unsafe {
            clear_world_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(256, 256, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(self.voxel_a.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(self.voxel_b.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(self.claim.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(self.aux_ctype.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(self.aux_tmp.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(self.aux_tmp2.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(self.aux_tmp3.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(self.spawn_claim.clone(), VOXEL_COUNT),
                VOXEL_COUNT as u32,
            );
        }
    }
}
