//! Device-side initialisation of the particle pool.
//!
//! A fresh (or resized) canvas needs every slot in its neutral state:
//! `part_ty == PT_NONE` (the slot is free), zero velocity, `equil == -1`. The
//! host shadows are filled by [`SimBuffers::new`]/[`SimBuffers::resize`]; this
//! module writes the same values straight into the device buffers instead of
//! uploading them.
//!
//! The pool is `capacity` slots of 15 arrays, twice the pixel count, so a
//! 1600x900 canvas would otherwise push ~170 MiB over the bus before the first
//! frame. Two device fills of that size take about a millisecond, and the host
//! and the device then hold identical neutral state, which is exactly what a
//! whole-array upload would have produced.

use cubecl::prelude::*;

use super::SimBuffers;

/// Temperature of a freshly created particle (`Element.cpp:54`).
///
/// A kernel body cannot read a host constant, so the value is repeated here and
/// guarded below.
const DEFAULT_TEMP_K: f32 = 295.15;
const _: () = assert!(DEFAULT_TEMP_K == tpt_core::constants::DEFAULT_TEMP_K);

/// Words (or floats) per kernel invocation.
const POOL_THREADS: u32 = 256;

/// Zeroes the integer fields of every slot.
///
/// `part_ty` is what marks a slot free, and the remaining words are the
/// TPT `life`/`ctype`/`flags`/`tmp` fields plus the decoration colour: a free
/// slot carries zeroes in all of them.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pool_clear_words_kernel(
    part_ty: &mut [u32],
    part_life: &mut [u32],
    part_ctype: &mut [u32],
    part_flags: &mut [u32],
    part_tmp: &mut [u32],
    part_tmp2: &mut [u32],
    part_tmp3: &mut [u32],
    part_tmp4: &mut [u32],
    len: u32,
) {
    let index = ABSOLUTE_POS;
    if index < len as usize {
        part_ty[index] = 0u32;
        part_life[index] = 0u32;
        part_ctype[index] = 0u32;
        part_flags[index] = 0u32;
        part_tmp[index] = 0u32;
        part_tmp2[index] = 0u32;
        part_tmp3[index] = 0u32;
        part_tmp4[index] = 0u32;
    }
}

/// Zeroes position, velocity and decoration, and reseeds the temperature and
/// equilibrium fields with their neutral values.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pool_clear_floats_kernel(
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_temp: &mut [f32],
    part_equil: &mut [f32],
    part_dcolour: &mut [u32],
    len: u32,
) {
    let index = ABSOLUTE_POS;
    if index < len as usize {
        part_x[index] = 0.0f32;
        part_y[index] = 0.0f32;
        part_vx[index] = 0.0f32;
        part_vy[index] = 0.0f32;
        part_temp[index] = DEFAULT_TEMP_K;
        part_equil[index] = -1.0f32;
        part_dcolour[index] = 0u32;
    }
}

impl SimBuffers {
    /// Writes the neutral slot state into the freshly allocated device pool.
    ///
    /// Callers must leave the host shadows in the same state (they are filled
    /// right before the pool is reallocated) and clear
    /// [`SimBuffers::host_particles_dirty`] afterwards, so the next
    /// [`SimBuffers::sync_host_to_gpu`] does not repeat the upload.
    fn clear_pool_device(&self) {
        let len = self.capacity as u32;
        if len == 0 {
            return;
        }
        let threads = POOL_THREADS;
        let cubes = len.div_ceil(threads);
        unsafe {
            pool_clear_words_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(threads),
                BufferArg::from_raw_parts(self.gpu_part_ty.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_life.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_ctype.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_flags.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_tmp.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_tmp2.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_tmp3.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_tmp4.clone(), self.capacity),
                len,
            );
            pool_clear_floats_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(threads),
                BufferArg::from_raw_parts(self.gpu_part_x.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_y.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_vx.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_vy.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_temp.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_equil.clone(), self.capacity),
                BufferArg::from_raw_parts(self.gpu_part_dcolour.clone(), self.capacity),
                len,
            );
        }
    }

    /// Initialises a freshly allocated pool on the device and marks both sides
    /// current.
    pub(super) fn initialise_pool_device(&mut self) {
        self.clear_pool_device();
        self.host_dirty_slots.clear();
        self.host_particles_dirty = false;
        self.gpu_particles_dirty = false;
    }
}
