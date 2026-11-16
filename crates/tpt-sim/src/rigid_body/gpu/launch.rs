//! Kernel launch orchestration for rigid-body passes.

use cubecl::prelude::*;
use cubecl::server::{BufferBinding, Handle};

use tpt_core::constants::PT_NUM;

use crate::buffers::SimBuffers;

use super::contact_kernels::*;
use super::layout::*;
use super::motion_kernels::*;
use super::raster_kernels::*;
use super::state::RigidGpu;

/// Builds an indirect-dispatch `BufferBinding` pointing at `slot` inside `dispatch_counts`.
#[inline]
fn dispatch_binding(handle: &Handle, slot: usize) -> BufferBinding {
    let slot_bytes = (DISPATCH_SLOT_WORDS * 4) as u64;
    let start = slot as u64 * slot_bytes;
    let end = (DISPATCH_SLOTS - slot - 1) as u64 * slot_bytes;
    handle.clone().offset_start(start).offset_end(end).binding()
}

impl RigidGpu {
    pub(super) fn run_passes(&mut self, bufs: &mut SimBuffers, dt: f32) {
        let (Some(bodies_handle), Some(pixels_handle)) = (self.bodies.clone(), self.pixels.clone())
        else {
            return;
        };
        let frames = if dt.is_finite() && dt > 0.0 {
            dt.min(super::MAX_FRAMES_PER_STEP)
        } else {
            1.0
        };
        if frames <= 0.0 {
            return;
        }
        let total = self.pixel_count;
        let body_count = self.body_count;
        if total == 0 || body_count == 0 {
            return;
        }

        if self.owner_capacity != bufs.capacity || self.owner.is_none() {
            self.ensure_owner(&bufs.client.clone(), bufs.capacity);
        }
        let num_pixels = bufs.config.num_pixels();
        if self.claims_pixels != num_pixels || self.claims.is_none() {
            self.ensure_claims(&bufs.client.clone(), num_pixels);
        }
        let (
            Some(claims_handle),
            Some(owner_handle),
            Some(surface_handle),
            Some(surface_list_handle),
            Some(dispatch_handle),
            Some(stage_handle),
        ) = (
            self.claims.clone(),
            self.owner.clone(),
            self.surface.clone(),
            self.surface_list.clone(),
            self.dispatch_counts.clone(),
            self.force_stage.clone(),
        )
        else {
            return;
        };
        let surface_words = self.pixel_capacity.div_ceil(32);

        let client = bufs.client.clone();
        let capacity = bufs.capacity;
        let xres = bufs.config.xres;
        let yres = bufs.config.yres;
        let xcells = bufs.config.xcells();
        let ycells = bufs.config.ycells();
        let gravity_mode = bufs.config.gravity_mode as u32;
        let air_mode = bufs.config.air_mode as u32;
        let custom_gx = bufs.config.custom_gravity_x;
        let custom_gy = bufs.config.custom_gravity_y;
        let newton_grav = if bufs.config.newton_grav {
            1.0f32
        } else {
            0.0f32
        };
        let needs_forces = air_mode != 3u32 || newton_grav != 0.0f32;
        let grav_cells = xcells as usize * ycells as usize;
        let body_cubes = (body_count as u32).div_ceil(64).max(1);
        let pixel_cubes = (total as u32).div_ceil(PIXEL_CUBE).max(1);
        let cell_cubes = (num_pixels as u32).div_ceil(PIXEL_CUBE).max(1);

        // Thermal / chemical detachment on the device.
        unsafe {
            reconcile_kernel::launch_unchecked(
                &client,
                CubeCount::Static(pixel_cubes, 1, 1),
                CubeDim::new_1d(PIXEL_CUBE),
                BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                BufferArg::from_raw_parts(pixels_handle.clone(), self.pixel_capacity * P_STRIDE),
                BufferArg::from_raw_parts(surface_handle.clone(), surface_words),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
                BufferArg::from_raw_parts(owner_handle.clone(), self.owner_capacity),
                total as u32,
                xres,
                yres,
            );
        }

        // Rebuild mass properties and compacted surface_list when geometry is dirty.
        unsafe {
            prepare_geometry_kernel::launch_unchecked(
                &client,
                CubeCount::Static(body_count as u32, 1, 1),
                CubeDim::new_1d(WG_SIZE),
                BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                BufferArg::from_raw_parts(pixels_handle.clone(), self.pixel_capacity * P_STRIDE),
                BufferArg::from_raw_parts(surface_handle, surface_words),
                BufferArg::from_raw_parts(surface_list_handle.clone(), self.pixel_capacity),
                BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), PT_NUM * 32),
                BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
                BufferArg::from_raw_parts(owner_handle.clone(), self.owner_capacity),
                body_count as u32,
            );
        }

        if needs_forces {
            unsafe {
                prepare_forces_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(pixel_cubes, 1, 1),
                    CubeDim::new_1d(PIXEL_CUBE),
                    BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                    BufferArg::from_raw_parts(pixels_handle.clone(), self.pixel_capacity * P_STRIDE),
                    BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), PT_NUM * 32),
                    BufferArg::from_raw_parts(
                        bufs.gpu_air_pv.clone(),
                        xcells as usize * ycells as usize,
                    ),
                    BufferArg::from_raw_parts(
                        bufs.gpu_air_vx.clone(),
                        xcells as usize * ycells as usize,
                    ),
                    BufferArg::from_raw_parts(
                        bufs.gpu_air_vy.clone(),
                        xcells as usize * ycells as usize,
                    ),
                    BufferArg::from_raw_parts(bufs.gpu_grav_fx.clone(), grav_cells),
                    BufferArg::from_raw_parts(bufs.gpu_grav_fy.clone(), grav_cells),
                    BufferArg::from_raw_parts(stage_handle.clone(), self.pixel_capacity * 3),
                    total as u32,
                    xres,
                    yres,
                    xcells,
                    ycells,
                    air_mode,
                    newton_grav,
                );
            }
        }

        // Parallel force reduction, gravity, sub-step sizing and sub-step 0 integration.
        unsafe {
            prepare_finish_kernel::launch_unchecked(
                &client,
                CubeCount::Static(body_count as u32, 1, 1),
                CubeDim::new_1d(WG_SIZE),
                BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                BufferArg::from_raw_parts(stage_handle, self.pixel_capacity * 3),
                body_count as u32,
                xres,
                yres,
                gravity_mode,
                custom_gx,
                custom_gy,
                frames,
                u32::from(needs_forces),
            );
        }

        if body_count == 1 {
            // Single-body fast path: no pair contacts can occur, so run every
            // sub-step inside a single 256-thread workgroup dispatch.
            unsafe {
                static_all_substeps_256_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(SINGLE_WG_SIZE),
                    BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                    BufferArg::from_raw_parts(pixels_handle.clone(), self.pixel_capacity * P_STRIDE),
                    BufferArg::from_raw_parts(surface_list_handle, self.pixel_capacity),
                    BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
                    BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
                    BufferArg::from_raw_parts(
                        bufs.gpu_wall_bmap.clone(),
                        xcells as usize * ycells as usize,
                    ),
                    BufferArg::from_raw_parts(
                        bufs.gpu_wall_emap.clone(),
                        xcells as usize * ycells as usize,
                    ),
                    BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), PT_NUM * 32),
                    BufferArg::from_raw_parts(owner_handle.clone(), self.owner_capacity),
                    1u32,
                    xres,
                    yres,
                    xcells,
                    ycells,
                    gravity_mode,
                    frames,
                );
            }
        } else {
            // Multi-body path: compute broadphase overlap and indirect dispatch counts.
            unsafe {
                broadphase_dispatch_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(WG_SIZE),
                    BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                    BufferArg::from_raw_parts(
                        dispatch_handle.clone(),
                        DISPATCH_SLOTS * DISPATCH_SLOT_WORDS,
                    ),
                    body_count as u32,
                    pixel_cubes,
                    body_cubes,
                    cell_cubes,
                );
                clear_claims_kernel::launch_unchecked(
                    &client,
                    CubeCount::Dynamic(dispatch_binding(&dispatch_handle, DISPATCH_CLEAR_LIVE_SLOT)),
                    CubeDim::new_1d(PIXEL_CUBE),
                    BufferArg::from_raw_parts(claims_handle.clone(), num_pixels),
                    num_pixels as u32,
                );
                static_all_substeps_64_kernel::launch_unchecked(
                    &client,
                    CubeCount::Dynamic(dispatch_binding(&dispatch_handle, DISPATCH_STATIC_ALL_SLOT)),
                    CubeDim::new_1d(WG_SIZE),
                    BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                    BufferArg::from_raw_parts(pixels_handle.clone(), self.pixel_capacity * P_STRIDE),
                    BufferArg::from_raw_parts(surface_list_handle.clone(), self.pixel_capacity),
                    BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
                    BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
                    BufferArg::from_raw_parts(
                        bufs.gpu_wall_bmap.clone(),
                        xcells as usize * ycells as usize,
                    ),
                    BufferArg::from_raw_parts(
                        bufs.gpu_wall_emap.clone(),
                        xcells as usize * ycells as usize,
                    ),
                    BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), PT_NUM * 32),
                    BufferArg::from_raw_parts(owner_handle.clone(), self.owner_capacity),
                    body_count as u32,
                    xres,
                    yres,
                    xcells,
                    ycells,
                    gravity_mode,
                    frames,
                );
            }

            let max_substeps = super::MAX_SUBSTEPS;
            for substep in 0..max_substeps {
                let scatter_cnt = dispatch_binding(&dispatch_handle, substep);
                let body_wg_cnt = dispatch_binding(&dispatch_handle, 16 + substep);
                let apply_cnt = dispatch_binding(&dispatch_handle, 32 + substep);
                let substep_u32 = substep as u32;
                let generation = substep_u32 + 1u32;

                unsafe {
                    scatter_live_kernel::launch_unchecked(
                        &client,
                        CubeCount::Dynamic(scatter_cnt),
                        CubeDim::new_1d(PIXEL_CUBE),
                        BufferArg::from_raw_parts(
                            bodies_handle.clone(),
                            self.body_capacity * B_STRIDE,
                        ),
                        BufferArg::from_raw_parts(
                            pixels_handle.clone(),
                            self.pixel_capacity * P_STRIDE,
                        ),
                        BufferArg::from_raw_parts(claims_handle.clone(), num_pixels),
                        total as u32,
                        xres,
                        yres,
                        generation,
                    );
                    contact_pair_kernel::launch_unchecked(
                        &client,
                        CubeCount::Dynamic(body_wg_cnt.clone()),
                        CubeDim::new_1d(WG_SIZE),
                        BufferArg::from_raw_parts(
                            bodies_handle.clone(),
                            self.body_capacity * B_STRIDE,
                        ),
                        BufferArg::from_raw_parts(
                            pixels_handle.clone(),
                            self.pixel_capacity * P_STRIDE,
                        ),
                        BufferArg::from_raw_parts(surface_list_handle.clone(), self.pixel_capacity),
                        BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
                        BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
                        BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
                        BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
                        BufferArg::from_raw_parts(owner_handle.clone(), self.owner_capacity),
                        BufferArg::from_raw_parts(claims_handle.clone(), num_pixels),
                        body_count as u32,
                        substep_u32,
                        xres,
                        yres,
                        0u32,
                        generation,
                    );
                    apply_body_impulses_kernel::launch_unchecked(
                        &client,
                        CubeCount::Dynamic(apply_cnt),
                        CubeDim::new_1d(WG_SIZE),
                        BufferArg::from_raw_parts(
                            bodies_handle.clone(),
                            self.body_capacity * B_STRIDE,
                        ),
                        body_count as u32,
                    );
                    contact_pair_kernel::launch_unchecked(
                        &client,
                        CubeCount::Dynamic(body_wg_cnt.clone()),
                        CubeDim::new_1d(WG_SIZE),
                        BufferArg::from_raw_parts(
                            bodies_handle.clone(),
                            self.body_capacity * B_STRIDE,
                        ),
                        BufferArg::from_raw_parts(
                            pixels_handle.clone(),
                            self.pixel_capacity * P_STRIDE,
                        ),
                        BufferArg::from_raw_parts(surface_list_handle.clone(), self.pixel_capacity),
                        BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
                        BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
                        BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
                        BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
                        BufferArg::from_raw_parts(owner_handle.clone(), self.owner_capacity),
                        BufferArg::from_raw_parts(claims_handle.clone(), num_pixels),
                        body_count as u32,
                        substep_u32,
                        xres,
                        yres,
                        1u32,
                        generation,
                    );
                    contact_static_kernel::launch_unchecked(
                        &client,
                        CubeCount::Dynamic(body_wg_cnt),
                        CubeDim::new_1d(WG_SIZE),
                        BufferArg::from_raw_parts(
                            bodies_handle.clone(),
                            self.body_capacity * B_STRIDE,
                        ),
                        BufferArg::from_raw_parts(
                            pixels_handle.clone(),
                            self.pixel_capacity * P_STRIDE,
                        ),
                        BufferArg::from_raw_parts(surface_list_handle.clone(), self.pixel_capacity),
                        BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
                        BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
                        BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
                        BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
                        BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
                        BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
                        BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
                        BufferArg::from_raw_parts(
                            bufs.gpu_wall_bmap.clone(),
                            xcells as usize * ycells as usize,
                        ),
                        BufferArg::from_raw_parts(
                            bufs.gpu_wall_emap.clone(),
                            xcells as usize * ycells as usize,
                        ),
                        BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), PT_NUM * 32),
                        BufferArg::from_raw_parts(owner_handle.clone(), self.owner_capacity),
                        body_count as u32,
                        substep_u32,
                        xres,
                        yres,
                        xcells,
                        ycells,
                        gravity_mode,
                        frames,
                    );
                }
            }
        }

        // Free the previous placement, then re-rasterise.
        unsafe {
            clear_rigid_kernel::launch_unchecked(
                &client,
                CubeCount::Static(pixel_cubes, 1, 1),
                CubeDim::new_1d(PIXEL_CUBE),
                BufferArg::from_raw_parts(pixels_handle.clone(), self.pixel_capacity * P_STRIDE),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
                BufferArg::from_raw_parts(owner_handle.clone(), self.owner_capacity),
                total as u32,
                xres,
                yres,
            );
            clear_claims_kernel::launch_unchecked(
                &client,
                CubeCount::Static(cell_cubes, 1, 1),
                CubeDim::new_1d(PIXEL_CUBE),
                BufferArg::from_raw_parts(claims_handle.clone(), num_pixels),
                num_pixels as u32,
            );
            claim_cells_kernel::launch_unchecked(
                &client,
                CubeCount::Static(pixel_cubes, 1, 1),
                CubeDim::new_1d(PIXEL_CUBE),
                BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                BufferArg::from_raw_parts(pixels_handle.clone(), self.pixel_capacity * P_STRIDE),
                BufferArg::from_raw_parts(claims_handle.clone(), num_pixels),
                total as u32,
                xres,
                yres,
            );
            rasterise_kernel::launch_unchecked(
                &client,
                CubeCount::Static(pixel_cubes, 1, 1),
                CubeDim::new_1d(PIXEL_CUBE),
                BufferArg::from_raw_parts(bodies_handle, self.body_capacity * B_STRIDE),
                BufferArg::from_raw_parts(pixels_handle, self.pixel_capacity * P_STRIDE),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), num_pixels),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
                BufferArg::from_raw_parts(
                    bufs.gpu_wall_bmap.clone(),
                    xcells as usize * ycells as usize,
                ),
                BufferArg::from_raw_parts(
                    bufs.gpu_wall_emap.clone(),
                    xcells as usize * ycells as usize,
                ),
                BufferArg::from_raw_parts(claims_handle, num_pixels),
                BufferArg::from_raw_parts(owner_handle, self.owner_capacity),
                total as u32,
                xres,
                yres,
                xcells,
                ycells,
            );
        }

        bufs.gpu_particles_dirty = true;
        bufs.gpu_maps_dirty = true;
        bufs.gpu_dirty = true;
    }
}
