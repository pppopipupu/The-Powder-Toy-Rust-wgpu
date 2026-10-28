//! Kernel launch orchestration for rigid-body passes.

use cubecl::prelude::*;

use tpt_core::constants::PT_NUM;

use crate::buffers::SimBuffers;

use super::layout::*;
use super::motion_kernels::*;
use super::raster_kernels::*;
use super::state::RigidGpu;

impl RigidGpu {
    pub(super) fn run_passes(&mut self, bufs: &mut SimBuffers, dt: f32) {
        let (Some(bodies_handle), Some(pixels_handle)) = (self.bodies.clone(), self.pixels.clone())
        else {
            return;
        };
        // `xres` / `yres` are cell-aligned, so the default clamps are positive.
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

        // The particle pool can grow between spawns; make sure the owner map
        // covers every slot before rasterising into it.
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
            Some(stage_handle),
            Some(manifolds_handle),
        ) = (
            self.claims.clone(),
            self.owner.clone(),
            self.surface.clone(),
            self.force_stage.clone(),
            self.manifolds.clone(),
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
        // `GetGravityField` adds `newtonGrav * gravOut.force[cell]` on top of
        // the mode's own field (`Simulation.cpp:2120-2124`); the rigid pass
        // samples it per pixel so the body accelerates with the field instead
        // of ignoring Newtonian gravity entirely.
        let newton_grav = if bufs.config.newton_grav {
            1.0f32
        } else {
            0.0f32
        };
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

        // Mass properties first: a fracture shifts the centroid and the origin,
        // and the air force below must use the rebuilt pose exactly like the
        // former single pass did.
        unsafe {
            prepare_geometry_kernel::launch_unchecked(
                &client,
                CubeCount::Static(body_cubes, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                BufferArg::from_raw_parts(pixels_handle.clone(), self.pixel_capacity * P_STRIDE),
                BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), PT_NUM * 32),
                BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
                BufferArg::from_raw_parts(owner_handle.clone(), self.owner_capacity),
                body_count as u32,
            );
        }

        // One thread per rigid pixel: stage the air pressure / wind force and
        // torque contribution `prepare_finish_kernel` sums below. With the air
        // simulation off the kernel writes zeros, so the fold never reads a
        // stale word.
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

        // Gravity, the staged air force, torque, sub-steps and the pose.
        unsafe {
            prepare_finish_kernel::launch_unchecked(
                &client,
                CubeCount::Static(body_cubes, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                BufferArg::from_raw_parts(stage_handle.clone(), self.pixel_capacity * 3),
                body_count as u32,
                xres,
                yres,
                gravity_mode,
                custom_gx,
                custom_gy,
                frames,
            );
        }

        // Sub-steps: integrate (prediction) then resolve contacts.
        let max_substeps = super::MAX_SUBSTEPS as u32;
        // The sub-step loop reuses the claim grid as a live rigid-cell map (the
        // pair phase falls back to it when the frame-start `pmap` cell holds no
        // rigid pixel), so start from the empty marker. The rasterise phase
        // clears it again before the claim pass.
        unsafe {
            clear_claims_kernel::launch_unchecked(
                &client,
                CubeCount::Static(cell_cubes, 1, 1),
                CubeDim::new_1d(PIXEL_CUBE),
                BufferArg::from_raw_parts(claims_handle.clone(), num_pixels),
                num_pixels as u32,
            );
        }
        for substep in 0..max_substeps {
            // The actual per-body `substeps` and `sdt` are read from the body
            // record inside the kernels; the host only bounds the loop.
            unsafe {
                integrate_substep_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(body_cubes, 1, 1),
                    CubeDim::new_1d(64),
                    BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                    body_count as u32,
                    substep,
                    frames,
                );
                // Publish the predicted pose's rigid cells for the pair phase.
                // The generation tag (the sub-step number + 1) separates this
                // sub-step's entries from the previous one's without a clear.
                scatter_live_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(pixel_cubes, 1, 1),
                    CubeDim::new_1d(PIXEL_CUBE),
                    BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                    BufferArg::from_raw_parts(
                        pixels_handle.clone(),
                        self.pixel_capacity * P_STRIDE,
                    ),
                    BufferArg::from_raw_parts(claims_handle.clone(), num_pixels),
                    total as u32,
                    xres,
                    yres,
                    substep + 1u32,
                );
                // Two pair iterations: each one hands the load one link
                // further down a stack (top -> middle -> bottom), so a third
                // body's weight reaches the ground support inside the same
                // sub-step instead of lagging by a sub-step and letting the
                // stack creep into itself. Both shares are queued, never
                // applied by the phase itself, so the second iteration is a
                // constraint iteration on the same deterministic state.
                for _pair_iter in 0..2u32 {
                    contact_kernel::launch_unchecked(
                        &client,
                        CubeCount::Static(body_cubes, 1, 1),
                        CubeDim::new_1d(64),
                        BufferArg::from_raw_parts(
                            bodies_handle.clone(),
                            self.body_capacity * B_STRIDE,
                        ),
                        BufferArg::from_raw_parts(
                            pixels_handle.clone(),
                            self.pixel_capacity * P_STRIDE,
                        ),
                        BufferArg::from_raw_parts(surface_handle.clone(), surface_words),
                        BufferArg::from_raw_parts(
                            manifolds_handle.clone(),
                            self.body_capacity * MANIFOLDS * M_STRIDE,
                        ),
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
                        BufferArg::from_raw_parts(claims_handle.clone(), num_pixels),
                        body_count as u32,
                        substep,
                        xres,
                        yres,
                        xcells,
                        ycells,
                        gravity_mode,
                        frames,
                        0u32,
                        _pair_iter,
                        substep + 1u32,
                    );
                    // Fold the queued impulses into the senders and receivers
                    // before the next phase.
                    apply_body_impulses_kernel::launch_unchecked(
                        &client,
                        CubeCount::Static(body_cubes, 1, 1),
                        CubeDim::new_1d(64),
                        BufferArg::from_raw_parts(
                            bodies_handle.clone(),
                            self.body_capacity * B_STRIDE,
                        ),
                        body_count as u32,
                    );
                }
                contact_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(body_cubes, 1, 1),
                    CubeDim::new_1d(64),
                    BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                    BufferArg::from_raw_parts(
                        pixels_handle.clone(),
                        self.pixel_capacity * P_STRIDE,
                    ),
                    BufferArg::from_raw_parts(surface_handle.clone(), surface_words),
                    BufferArg::from_raw_parts(
                        manifolds_handle.clone(),
                        self.body_capacity * MANIFOLDS * M_STRIDE,
                    ),
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
                    BufferArg::from_raw_parts(claims_handle.clone(), num_pixels),
                    body_count as u32,
                    substep,
                    xres,
                    yres,
                    xcells,
                    ycells,
                    gravity_mode,
                    frames,
                    1u32,
                    0u32,
                    substep + 1u32,
                );
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
            // Publish every desired cell before any pixel is placed: the
            // `fetch_min` arbitration only settles once all keys are in, and
            // the rasterise pass reads the result back to learn who won.
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
                BufferArg::from_raw_parts(bodies_handle.clone(), self.body_capacity * B_STRIDE),
                BufferArg::from_raw_parts(pixels_handle.clone(), self.pixel_capacity * P_STRIDE),
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

        // Particles and maps are now newer than the host shadows, but no host
        // buffer changed: leave the upload flags alone and let the next
        // explicit `ensure_host_current` / save read them back.
        bufs.gpu_particles_dirty = true;
        bufs.gpu_maps_dirty = true;
        bufs.gpu_dirty = true;
    }
}
