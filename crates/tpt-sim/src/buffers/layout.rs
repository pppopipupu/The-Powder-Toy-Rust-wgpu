//! Buffer sizing and allocation: the initial device/host layout of
//! [`SimBuffers`], canvas resizes and the growing particle pool.
//!
//! Device memory is owned through [`ParticleHandles`] (sized by the pool
//! capacity) and [`GridHandles`] (sized by the pixel / cell counts), so adding
//! a new array means touching one struct instead of three `client.empty` call
//! sites.

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::server::Handle;

use tpt_core::constants::*;
use tpt_core::elements::{ElementRegistry, GpuElementProps};
use tpt_core::particle::{EdgeMode, SimulationConfig};

use super::gpu_sync::upload;
use super::{MIN_PARTICLE_CAPACITY, SimBuffers, WORD};

impl SimBuffers {
    /// Largest particle pool the packed ids can address: `0x7FFFFF`
    /// (`8 388 607` slots).
    ///
    /// Both [`tpt_core::constants::pmap_pack`] and the movement winner key spend
    /// exactly `23` bits on `part_id + 1`, so the largest addressable id is
    /// `2^23 - 2` and `id + 1 == 0x7FFFFF` still fits the field. Every buffer
    /// that is sized by the capacity also stays far below the `i32::MAX` byte
    /// limit: `13` slot-sized `u32` arrays cost
    /// `0x7FFFFF * 13 * 4 = 436 207 564` bytes in total (see the const
    /// assertions at the bottom of this file).
    pub const MAX_PARTICLE_CAPACITY: usize = 0x7F_FFFF;

    /// Largest canvas side the GPU layout supports: `2896` pixels
    /// (`floor(sqrt(0x7FFFFF))` rounded down to a multiple of [`CELL`]).
    ///
    /// The pool must be able to hold at least one particle per pixel, and
    /// [`Self::MAX_PARTICLE_CAPACITY`] caps `xres * yres` at `8 388 607`
    /// pixels, so no canvas side may exceed this. The UI validates a requested
    /// size with [`SimBuffers::max_supported_canvas`] before resizing.
    pub const MAX_SUPPORTED_CANVAS_SIDE: u32 = max_canvas_side(Self::MAX_PARTICLE_CAPACITY);

    pub fn new(client: Client, config: SimulationConfig, registry: &ElementRegistry) -> Self {
        let capacity = Self::initial_capacity(&config);
        let xres = config.xres as usize;
        let yres = config.yres as usize;
        let num_pixels = config.num_pixels();
        let num_cells = config.num_cells();
        let ambient = config.ambient_air_temp;
        let pool = ParticleHandles::alloc(&client, capacity);
        let grid = GridHandles::alloc(&client, num_pixels, num_cells);
        let (gpu_elem_props, gpu_can_move, gpu_spawn_defaults) =
            Self::alloc_element_tables(&client, registry);

        let mut bufs = Self {
            client,
            config,
            capacity,
            active_count: 0,
            next_slot: 0,
            free_list: Vec::new(),
            host_dirty_slots: Vec::new(),
            host_dirty: true,
            host_particles_dirty: true,
            host_maps_dirty: true,
            host_air_dirty: true,
            host_walls_dirty: true,
            host_wall_state_dirty: true,
            gpu_dirty: false,
            gpu_particles_dirty: false,
            gpu_maps_dirty: false,
            gpu_air_dirty: false,
            gpu_spawn_used: false,
            patch_buf: None,
            patch_words: 0,

            // Particle SoA: every host shadow is followed by its device buffer.
            part_ty: vec![PT_NONE; capacity],
            gpu_part_ty: pool.ty,
            part_life: vec![0; capacity],
            gpu_part_life: pool.life,
            part_ctype: vec![0; capacity],
            gpu_part_ctype: pool.ctype,
            part_x: vec![0.0; capacity],
            gpu_part_x: pool.x,
            part_y: vec![0.0; capacity],
            gpu_part_y: pool.y,
            part_vx: vec![0.0; capacity],
            gpu_part_vx: pool.vx,
            part_vy: vec![0.0; capacity],
            gpu_part_vy: pool.vy,
            part_temp: vec![DEFAULT_TEMP_K; capacity],
            gpu_part_temp: pool.temp,
            part_equil: vec![-1.0; capacity],
            gpu_part_equil: pool.equil,
            part_flags: vec![0; capacity],
            gpu_part_flags: pool.flags,
            part_tmp: vec![0; capacity],
            gpu_part_tmp: pool.tmp,
            part_tmp2: vec![0; capacity],
            gpu_part_tmp2: pool.tmp2,
            part_tmp3: vec![0; capacity],
            gpu_part_tmp3: pool.tmp3,
            part_tmp4: vec![0; capacity],
            gpu_part_tmp4: pool.tmp4,
            part_dcolour: vec![0; capacity],
            gpu_part_dcolour: pool.dcolour,

            // Spatial maps (`y * xres + x`), indexed by pixel.
            pmap: vec![0; num_pixels],
            gpu_pmap: grid.pmap,
            photons: vec![0; num_pixels],
            gpu_photons: grid.photons,
            pmap_count: vec![0; num_pixels],
            gpu_pmap_count: grid.pmap_count,
            intent_target: pool.intent_target,
            cell_winner: grid.cell_winner,

            // Air, fan and wall grids, indexed by `cy * xcells + cx`.
            air_vx: vec![0.0; num_cells],
            gpu_air_vx: grid.air_vx,
            air_vy: vec![0.0; num_cells],
            gpu_air_vy: grid.air_vy,
            air_pv: vec![0.0; num_cells],
            gpu_air_pv: grid.air_pv,
            air_hv: vec![ambient; num_cells],
            gpu_air_hv: grid.air_hv,
            air_ovx: vec![0.0; num_cells],
            gpu_air_ovx: grid.air_ovx,
            air_ovy: vec![0.0; num_cells],
            gpu_air_ovy: grid.air_ovy,
            air_opv: vec![0.0; num_cells],
            gpu_air_opv: grid.air_opv,
            air_ohv: vec![ambient; num_cells],
            gpu_air_ohv: grid.air_ohv,
            fan_vx: vec![0.0; num_cells],
            gpu_fan_vx: grid.fan_vx,
            fan_vy: vec![0.0; num_cells],
            gpu_fan_vy: grid.fan_vy,
            wall_bmap: vec![0; num_cells],
            gpu_wall_bmap: grid.wall_bmap,
            wall_emap: vec![0; num_cells],
            gpu_wall_emap: grid.wall_emap,
            block_air: vec![0; num_cells],
            gpu_block_air: grid.block_air,
            block_airh: vec![0; num_cells],
            gpu_block_airh: grid.block_airh,

            // Rendering: fire glow per cell, packed framebuffer per pixel.
            fire_r: vec![0; num_cells],
            gpu_fire_r: grid.fire_r,
            fire_g: vec![0; num_cells],
            gpu_fire_g: grid.fire_g,
            fire_b: vec![0; num_cells],
            gpu_fire_b: grid.fire_b,
            fire_layer: vec![0; num_pixels],
            gpu_fire_layer: grid.fire_layer,
            fx_cell: vec![0; num_cells],
            gpu_fx_cell: grid.fx_cell,
            fx_axis: vec![0; xres + yres],
            gpu_fx_axis: grid.fx_axis,
            framebuf_rgba: vec![0xFF00_0000; num_pixels],
            gpu_framebuf: grid.framebuf,
            emp_decor: 0,
            sim_tick: 0,
            fire_tick: 0,

            // Newtonian gravity grids.
            grav_mass: vec![0.0; num_cells],
            gpu_grav_mass: grid.grav_mass,
            grav_mass_prev: vec![0.0; num_cells],
            gpu_grav_mass_prev: grid.grav_mass_prev,
            grav_tool_mass: vec![0.0; num_cells],
            gpu_grav_tool_mass: grid.grav_tool_mass,
            grav_tool_mass_dirty: false,
            grav_mask: vec![u32::MAX; num_cells],
            gpu_grav_mask: grid.grav_mask,
            grav_fx: vec![0.0; num_cells],
            gpu_grav_fx: grid.grav_fx,
            grav_fy: vec![0.0; num_cells],
            gpu_grav_fy: grid.grav_fy,
            grav_mask_dirty: true,
            grav_mass_dirty: true,
            grav_shadow_dirty: false,
            grav_force_stale: false,

            // Transient `DMG` blast grid (zeroed on the device every frame).
            blast_cell: vec![0; num_cells],
            gpu_blast_cell: grid.blast_cell,
            gpu_fray_range: grid.fray_range,
            gpu_water_col_surf: grid.water_col_surf,
            gpu_water_col_target: grid.water_col_target,
            gpu_water_winner: grid.water_winner,

            // Zero-copy device SoA: free-slot pool, creation queue, portals,
            // wireless & stickmen.
            gpu_free_slots: grid.free_slots,
            gpu_live_count: grid.live_count,
            gpu_emap_gate: grid.emap_gate,
            gpu_edit_points: grid.edit_points,
            gpu_edit_rolls: grid.edit_rolls,
            gpu_move_lut: grid.move_lut,
            gpu_elem_phys: grid.elem_phys,
            gpu_photon_mask: grid.photon_mask,
            move_tables_stale: true,
            gpu_create_queue: grid.create_queue,
            gpu_portal_ty: grid.portal_ty,
            gpu_portal_data: grid.portal_data,
            gpu_wireless: grid.wireless,
            gpu_stkm_legs: grid.stkm_legs,
            gpu_stkm_accs: grid.stkm_accs,
            gpu_stkm_meta: grid.stkm_meta,

            // Element tables: the host copies are filled by
            // `refresh_element_tables` right below.
            gpu_elem_props,
            gpu_can_move,
            gpu_spawn_defaults,
            elem_props_host: Vec::new(),
            can_move_host: Vec::new(),
            elem_is_energy: Vec::new(),
            spawn_defaults_host: Vec::new(),

            rigid_bodies: Vec::new(),
            next_rigid_id: 0,
        };

        bufs.refresh_element_tables(registry);
        // `Simulation::Load` stamps the border ring from the saved edge mode
        // (`Simulation.cpp:1073` `SetEdgeMode(edgeMode)`).
        bufs.apply_edge_mode();
        // The pool starts neutral on both sides; a device fill is orders of
        // magnitude cheaper than uploading `capacity * 15` words.
        bufs.initialise_pool_device();
        bufs.sync_host_to_gpu();
        bufs
    }

    /// Stamp the coarse border ring from [`SimulationConfig::edge_mode`].
    ///
    /// `Simulation::SetEdgeMode` (`Simulation.cpp:765-799`):
    ///
    /// ```text
    /// case EDGE_VOID:
    /// case EDGE_LOOP:
    ///     for (i = 0; i < XCELLS; i++) { bmap[0][i] = 0; bmap[YCELLS-1][i] = 0; }
    ///     for (i = 1; i < YCELLS-1; i++) { bmap[i][0] = 0; bmap[i][XCELLS-1] = 0; }
    ///     break;
    /// case EDGE_SOLID:
    ///     ... bmap[...] = WL_WALL ...
    /// ```
    ///
    /// The reference calls this at setup and on every mode change, never per
    /// tick, so a wall the user paints on the border ring under `EDGE_VOID` /
    /// `EDGE_LOOP` survives until the mode changes again. `EDGE_SOLID`'s
    /// `WL_WALL` ring is what `IsWallBlocking` turns into a blocked step, so a
    /// particle stops on the border instead of running out and being killed by
    /// `Simulation::move`'s margin check.
    ///
    /// Returns `true` when the ring changed, so the caller knows whether the
    /// derived air/heat masks and the device copy must be refreshed. The
    /// derived masks are always refreshed by [`Self::rebuild_wall_masks`] when
    /// this returns `true`.
    pub fn apply_edge_mode(&mut self) -> bool {
        let xcells = self.config.xcells() as usize;
        let ycells = self.config.ycells() as usize;
        if xcells == 0 || ycells == 0 {
            return false;
        }
        let want = if self.config.edge_mode == EdgeMode::Solid {
            WL_WALL as u32
        } else {
            0u32
        };
        let mut changed = false;
        for x in 0..xcells {
            for cy in [0usize, ycells - 1] {
                let idx = cy * xcells + x;
                if self.wall_bmap[idx] != want {
                    self.wall_bmap[idx] = want;
                    changed = true;
                }
            }
        }
        for y in 1..ycells.saturating_sub(1) {
            for cx in [0usize, xcells - 1] {
                let idx = y * xcells + cx;
                if self.wall_bmap[idx] != want {
                    self.wall_bmap[idx] = want;
                    changed = true;
                }
            }
        }
        if changed {
            self.rebuild_wall_masks();
        }
        changed
    }

    /// Device buffers for the element property table, the can-move matrix and the
    /// spawn-default table.
    ///
    /// All three are sized from the registry; `upload_element_tables` only writes
    /// into them, so their size must not change until the registry does.
    fn alloc_element_tables(
        client: &Client,
        registry: &ElementRegistry,
    ) -> (Handle, Handle, Handle) {
        let props = registry.to_gpu_props();
        let can_move = registry.to_gpu_can_move_u32();
        let defaults = registry.to_gpu_spawn_defaults_u32();
        let props_bytes = (props.len() * core::mem::size_of::<GpuElementProps>()).max(WORD);
        let can_move_bytes = (can_move.len() * WORD).max(WORD);
        let defaults_bytes = (defaults.len() * WORD).max(WORD);
        (
            client.empty(props_bytes),
            client.empty(can_move_bytes),
            client.empty(defaults_bytes),
        )
    }

    /// Pool size a fresh canvas starts with:
    /// `clamp(xres * yres * 2, 65 536, MAX_PARTICLE_CAPACITY)`.
    ///
    /// A `2048 x 2048` canvas asks for `2 * 4 194 304 = 8 388 608` slots, one
    /// more than the widest id the packed maps can hold, so the pool saturates
    /// at [`Self::MAX_PARTICLE_CAPACITY`] (`8 388 607`). That is still more
    /// slots than the canvas has pixels, so the cap never limits occupancy in
    /// practice.
    pub fn initial_capacity(config: &SimulationConfig) -> usize {
        (config.num_pixels() * 2).clamp(MIN_PARTICLE_CAPACITY, Self::MAX_PARTICLE_CAPACITY)
    }

    /// Largest square canvas the GPU layout can address, `(xres, yres)`.
    ///
    /// Derived from `u64` byte math over the buffer layout:
    ///
    /// * the packed ids limit the pool to [`Self::MAX_PARTICLE_CAPACITY`]
    ///   slots, and the pool must hold one particle per pixel, so
    ///   `xres * yres <= 8 388 607` (`134 MiB` of `u32` pixel buffers at `16`
    ///   bytes per pixel, well inside `i32::MAX`);
    /// * at that cap the `13` slot-sized arrays total
    ///   `8 388 607 * 13 * 4 = 436 207 564` bytes, which is the documented
    ///   `capacity * 13 * 4 <= i32::MAX` budget.
    ///
    /// `floor(sqrt(8 388 607)) = 2896`, already a multiple of [`CELL`].
    pub fn max_supported_canvas(&self) -> (u32, u32) {
        let side = Self::MAX_SUPPORTED_CANVAS_SIDE;
        (side, side)
    }

    /// Re-create every device buffer for a new canvas size and drop all state.
    pub fn resize(&mut self, xres: u32, yres: u32) {
        let cell = CELL;
        let nx = (xres.max(64) / cell) * cell;
        let ny = (yres.max(64) / cell) * cell;
        self.config.xres = nx;
        self.config.yres = ny;

        let capacity = Self::initial_capacity(&self.config);
        self.capacity = capacity;
        self.active_count = 0;
        self.next_slot = 0;
        self.free_list.clear();
        self.rigid_bodies.clear();
        self.next_rigid_id = 0;

        self.part_ty = vec![PT_NONE; capacity];
        self.part_life = vec![0; capacity];
        self.part_ctype = vec![0; capacity];
        self.part_x = vec![0.0; capacity];
        self.part_y = vec![0.0; capacity];
        self.part_vx = vec![0.0; capacity];
        self.part_vy = vec![0.0; capacity];
        self.part_temp = vec![DEFAULT_TEMP_K; capacity];
        self.part_equil = vec![-1.0; capacity];
        self.part_flags = vec![0; capacity];
        self.part_tmp = vec![0; capacity];
        self.part_tmp2 = vec![0; capacity];
        self.part_tmp3 = vec![0; capacity];
        self.part_tmp4 = vec![0; capacity];
        self.part_dcolour = vec![0; capacity];

        self.gpu_part_ty = self.client.empty(capacity * WORD);
        self.gpu_part_life = self.client.empty(capacity * WORD);
        self.gpu_part_ctype = self.client.empty(capacity * WORD);
        self.gpu_part_x = self.client.empty(capacity * WORD);
        self.gpu_part_y = self.client.empty(capacity * WORD);
        self.gpu_part_vx = self.client.empty(capacity * WORD);
        self.gpu_part_vy = self.client.empty(capacity * WORD);
        self.gpu_part_temp = self.client.empty(capacity * WORD);
        self.gpu_part_equil = self.client.empty(capacity * WORD);
        self.gpu_part_flags = self.client.empty(capacity * WORD);
        self.gpu_part_tmp = self.client.empty(capacity * WORD);
        self.gpu_part_tmp2 = self.client.empty(capacity * WORD);
        self.gpu_part_tmp3 = self.client.empty(capacity * WORD);
        self.gpu_part_tmp4 = self.client.empty(capacity * WORD);
        self.gpu_part_dcolour = self.client.empty(capacity * WORD);
        self.intent_target = self.client.empty(capacity * WORD);

        self.reallocate_grid_handles();
        // `clear_sim` re-stamps the border ring after a resize
        // (`Simulation.cpp:1073`).
        self.apply_edge_mode();
        self.mark_all_dirty();
        // Both host and device hold a neutral pool now; uploading the whole
        // SoA here was the single largest cost of starting a canvas.
        self.initialise_pool_device();
        self.gpu_dirty = false;
        self.gpu_particles_dirty = false;
        self.gpu_maps_dirty = false;
        self.gpu_air_dirty = false;
        self.gpu_spawn_used = false;
    }

    /// Re-create the pixel/cell sized device buffers and clear their host shadows.
    fn reallocate_grid_handles(&mut self) {
        let num_pixels = self.config.num_pixels();
        let num_cells = self.config.num_cells();
        let xres = self.config.xres as usize;
        let yres = self.config.yres as usize;
        let ambient = self.config.ambient_air_temp;

        self.pmap = vec![0; num_pixels];
        self.gpu_pmap = self.client.empty(num_pixels * WORD);
        self.photons = vec![0; num_pixels];
        self.gpu_photons = self.client.empty(num_pixels * WORD);
        self.pmap_count = vec![0; num_pixels];
        self.gpu_pmap_count = self.client.empty(num_pixels * WORD);
        self.cell_winner = self.client.empty(num_pixels * WORD);

        self.air_vx = vec![0.0; num_cells];
        self.gpu_air_vx = self.client.empty(num_cells * WORD);
        self.air_vy = vec![0.0; num_cells];
        self.gpu_air_vy = self.client.empty(num_cells * WORD);
        self.air_pv = vec![0.0; num_cells];
        self.gpu_air_pv = self.client.empty(num_cells * WORD);
        self.air_hv = vec![ambient; num_cells];
        self.gpu_air_hv = self.client.empty(num_cells * WORD);
        self.air_ovx = vec![0.0; num_cells];
        self.gpu_air_ovx = self.client.empty(num_cells * WORD);
        self.air_ovy = vec![0.0; num_cells];
        self.gpu_air_ovy = self.client.empty(num_cells * WORD);
        self.air_opv = vec![0.0; num_cells];
        self.gpu_air_opv = self.client.empty(num_cells * WORD);
        self.air_ohv = vec![ambient; num_cells];
        self.gpu_air_ohv = self.client.empty(num_cells * WORD);
        self.fan_vx = vec![0.0; num_cells];
        self.gpu_fan_vx = self.client.empty(num_cells * WORD);
        self.fan_vy = vec![0.0; num_cells];
        self.gpu_fan_vy = self.client.empty(num_cells * WORD);
        self.wall_bmap = vec![0; num_cells];
        self.gpu_wall_bmap = self.client.empty(num_cells * WORD);
        self.wall_emap = vec![0; num_cells];
        self.gpu_wall_emap = self.client.empty(num_cells * WORD);
        self.block_air = vec![0; num_cells];
        self.gpu_block_air = self.client.empty(num_cells * WORD);
        self.block_airh = vec![0; num_cells];
        self.gpu_block_airh = self.client.empty(num_cells * WORD);

        self.fire_r = vec![0; num_cells];
        self.gpu_fire_r = self.client.empty(num_cells * WORD);
        self.fire_g = vec![0; num_cells];
        self.gpu_fire_g = self.client.empty(num_cells * WORD);
        self.fire_b = vec![0; num_cells];
        self.gpu_fire_b = self.client.empty(num_cells * WORD);
        self.fire_layer = vec![0; num_pixels];
        self.gpu_fire_layer = self.client.empty(num_pixels * WORD);
        self.fx_cell = vec![0; num_cells];
        self.gpu_fx_cell = self.client.empty(num_cells * WORD);
        self.fx_axis = vec![0; xres + yres];
        self.gpu_fx_axis = self.client.empty((xres + yres) * WORD);
        self.framebuf_rgba = vec![0xFF00_0000; num_pixels];
        self.gpu_framebuf = self.client.empty(num_pixels * WORD);

        self.grav_mass = vec![0.0; num_cells];
        self.gpu_grav_mass = self.client.empty(num_cells * WORD);
        self.grav_mass_prev = vec![0.0; num_cells];
        self.gpu_grav_mass_prev = self.client.empty(num_cells * WORD);
        self.grav_tool_mass = vec![0.0; num_cells];
        self.gpu_grav_tool_mass = self.client.empty(num_cells * WORD);
        self.grav_tool_mass_dirty = false;
        self.grav_mask = vec![u32::MAX; num_cells];
        self.gpu_grav_mask = self.client.empty(num_cells * WORD);
        self.grav_fx = vec![0.0; num_cells];
        self.gpu_grav_fx = self.client.empty(num_cells * WORD);
        self.grav_fy = vec![0.0; num_cells];
        self.gpu_grav_fy = self.client.empty(num_cells * WORD);
        self.grav_mask_dirty = true;
        self.grav_mass_dirty = true;
        // Both sides were just reseeded from zero below.
        self.grav_force_stale = false;
        self.grav_shadow_dirty = false;

        self.blast_cell = vec![0; num_cells];
        self.gpu_blast_cell = self.client.empty(num_cells * WORD);
        self.gpu_fray_range = self.client.empty(WORD);
        self.gpu_water_col_surf = self.client.empty(num_pixels * WORD);
        self.gpu_water_col_target = self.client.empty(num_pixels * WORD);
        self.gpu_water_winner = self.client.empty(num_pixels * WORD);
        self.emp_decor = 0;
        // The fire cells were just reseeded from zero, so the next render must
        // run the accumulation and decay pass again.
        self.fire_tick = self.sim_tick.wrapping_sub(1);

        // Device-only arrays: no host pass uploads them, so `empty` would leave
        // whatever the allocator returned. A canvas resize then starts the next
        // frame with garbage in the fire-glow, gravity and blast buffers, which
        // the fancy renderer spreads over the whole canvas as a red wash.
        // Seed them from their host shadows so a resize is pixel-deterministic.
        upload(&self.client, &self.gpu_fire_r, &self.fire_r);
        upload(&self.client, &self.gpu_fire_g, &self.fire_g);
        upload(&self.client, &self.gpu_fire_b, &self.fire_b);
        upload(&self.client, &self.gpu_fire_layer, &self.fire_layer);
        upload(&self.client, &self.gpu_fx_cell, &self.fx_cell);
        upload(&self.client, &self.gpu_fx_axis, &self.fx_axis);
        upload(&self.client, &self.gpu_framebuf, &self.framebuf_rgba);
        upload(&self.client, &self.gpu_grav_mass, &self.grav_mass);
        upload(&self.client, &self.gpu_grav_tool_mass, &self.grav_tool_mass);
        upload(&self.client, &self.gpu_grav_mask, &self.grav_mask);
        upload(&self.client, &self.gpu_grav_fx, &self.grav_fx);
        upload(&self.client, &self.gpu_grav_fy, &self.grav_fy);
        upload(&self.client, &self.gpu_blast_cell, &self.blast_cell);
        self.reset_device_soa();
    }

    /// Zero the device-resident portal, wireless, stickman, device-allocator and
    /// creation-queue SoA buffers.
    pub fn reset_device_soa(&mut self) {
        let free_words = Self::GPU_SPAWN_COUNT + 2;
        self.gpu_free_slots = alloc_zeroed(&self.client, free_words);
        self.gpu_live_count = alloc_zeroed(&self.client, 1);
        self.gpu_emap_gate = alloc_zeroed(&self.client, 1);
        self.gpu_edit_points = alloc_zeroed(&self.client, Self::EDIT_POINT_CAPACITY);
        self.gpu_edit_rolls = alloc_zeroed(&self.client, Self::tool_rolls_capacity());
        self.gpu_move_lut = self
            .client
            .empty(crate::move_kernel::build_move_lut_len() * WORD);
        self.gpu_elem_phys = self
            .client
            .empty(crate::move_kernel::elem_phys_len() * WORD);
        self.gpu_photon_mask = self
            .client
            .empty(crate::move_kernel::photon_mask_len() * WORD);
        self.move_tables_stale = true;
        self.gpu_create_queue = alloc_zeroed(
            &self.client,
            Self::GPU_CREATE_COUNT * Self::CREATE_WORDS + 1,
        );
        self.gpu_portal_ty = alloc_zeroed(&self.client, PORTAL_CAPACITY);
        self.gpu_portal_data = alloc_zeroed(&self.client, PORTAL_CAPACITY * PORTAL_WORDS);
        self.gpu_wireless = alloc_zeroed(&self.client, PORTAL_CHANNELS * 2);
        self.gpu_stkm_legs = alloc_zeroed(&self.client, STKM_SLOTS * STKM_LEGS_STRIDE);
        self.gpu_stkm_accs = alloc_zeroed(&self.client, STKM_SLOTS * STKM_ACCS_STRIDE);
        self.gpu_stkm_meta = alloc_zeroed(&self.client, STKM_SLOTS * STKM_META_STRIDE);
        self.gpu_spawn_used = false;
    }

    /// Grow the particle pool so it holds at least `min_capacity` slots,
    /// saturating at [`Self::MAX_PARTICLE_CAPACITY`].
    ///
    /// Doubling growth; every GPU handle is reallocated and re-uploaded, so the
    /// caller never observes a partially initialised pool. Requests above the
    /// cap are clamped instead of growing past the `23`-bit id field of
    /// `pmap_pack` / the movement winner key: a saturated pool simply stays at
    /// `8 388 607` slots. Callers that need a slot must therefore treat a full
    /// pool as "no room" rather than indexing past `capacity`.
    pub fn ensure_particle_capacity(&mut self, min_capacity: usize) {
        let min_capacity = min_capacity.min(Self::MAX_PARTICLE_CAPACITY);
        if min_capacity <= self.capacity {
            return;
        }
        if self.gpu_dirty {
            self.sync_gpu_to_host();
        }
        let mut new_capacity = self.capacity.max(1);
        while new_capacity < min_capacity {
            new_capacity = new_capacity.saturating_mul(2);
        }
        self.reallocate_particle_pool(new_capacity.min(Self::MAX_PARTICLE_CAPACITY));
        self.mark_particles_dirty();
    }

    /// Reallocate every particle-sized buffer at `new_capacity`, keeping data.
    fn reallocate_particle_pool(&mut self, new_capacity: usize) {
        self.part_ty.resize(new_capacity, PT_NONE);
        self.part_life.resize(new_capacity, 0);
        self.part_ctype.resize(new_capacity, 0);
        self.part_x.resize(new_capacity, 0.0);
        self.part_y.resize(new_capacity, 0.0);
        self.part_vx.resize(new_capacity, 0.0);
        self.part_vy.resize(new_capacity, 0.0);
        self.part_temp.resize(new_capacity, DEFAULT_TEMP_K);
        self.part_equil.resize(new_capacity, -1.0);
        self.part_flags.resize(new_capacity, 0);
        self.part_tmp.resize(new_capacity, 0);
        self.part_tmp2.resize(new_capacity, 0);
        self.part_tmp3.resize(new_capacity, 0);
        self.part_tmp4.resize(new_capacity, 0);
        self.part_dcolour.resize(new_capacity, 0);

        self.gpu_part_ty = self.client.empty(new_capacity * WORD);
        self.gpu_part_life = self.client.empty(new_capacity * WORD);
        self.gpu_part_ctype = self.client.empty(new_capacity * WORD);
        self.gpu_part_x = self.client.empty(new_capacity * WORD);
        self.gpu_part_y = self.client.empty(new_capacity * WORD);
        self.gpu_part_vx = self.client.empty(new_capacity * WORD);
        self.gpu_part_vy = self.client.empty(new_capacity * WORD);
        self.gpu_part_temp = self.client.empty(new_capacity * WORD);
        self.gpu_part_equil = self.client.empty(new_capacity * WORD);
        self.gpu_part_flags = self.client.empty(new_capacity * WORD);
        self.gpu_part_tmp = self.client.empty(new_capacity * WORD);
        self.gpu_part_tmp2 = self.client.empty(new_capacity * WORD);
        self.gpu_part_tmp3 = self.client.empty(new_capacity * WORD);
        self.gpu_part_tmp4 = self.client.empty(new_capacity * WORD);
        self.gpu_part_dcolour = self.client.empty(new_capacity * WORD);
        self.intent_target = self.client.empty(new_capacity * WORD);

        self.capacity = new_capacity;
        self.upload_particles();
    }

    pub fn mark_all_dirty(&mut self) {
        self.host_dirty = true;
        self.host_particles_dirty = true;
        self.host_maps_dirty = true;
        self.host_air_dirty = true;
        self.host_walls_dirty = true;
    }

    /// Marks the host particle SoA as changed.
    ///
    /// The whole array is uploaded, not just the per-slot dirty list: external
    /// host passes (rigid bodies, water equalisation, reaction spawn requests)
    /// write particle fields directly and then call this method, so the
    /// [`super::FULL_PARTICLE_UPLOAD`] sentinel must be pushed.
    pub fn mark_particles_dirty(&mut self) {
        self.host_dirty = true;
        self.host_particles_dirty = true;
        self.host_dirty_slots.push(super::FULL_PARTICLE_UPLOAD);
    }

    /// Marks the host `pmap` / `photons` as changed.
    pub fn mark_maps_dirty(&mut self) {
        self.host_dirty = true;
        self.host_maps_dirty = true;
    }

    /// Marks the host air grids as changed.
    pub fn mark_air_dirty(&mut self) {
        self.host_dirty = true;
        self.host_air_dirty = true;
    }

    /// Marks the host wall grids as changed (also invalidates the gravity mask).
    pub fn mark_walls_dirty(&mut self) {
        self.host_dirty = true;
        self.host_walls_dirty = true;
        self.host_wall_state_dirty = true;
        self.grav_mask_dirty = true;
    }

    /// Marks only the wall electrical state and derived masks as changed.
    pub fn mark_wall_state_dirty(&mut self) {
        self.host_dirty = true;
        self.host_wall_state_dirty = true;
    }

    /// Re-upload `GpuElementProps` and the can-move matrix from `registry`.
    pub fn refresh_element_tables(&mut self, registry: &ElementRegistry) {
        self.elem_props_host = registry.to_gpu_props();
        self.can_move_host = registry.to_gpu_can_move_u32();
        self.spawn_defaults_host = registry.to_gpu_spawn_defaults_u32();
        self.elem_is_energy = registry
            .elements
            .iter()
            .map(|def| def.properties & TYPE_ENERGY != 0)
            .collect();
        self.upload_element_tables();
        self.move_tables_stale = true;
    }

    /// Push the current host copies of the element tables to the device.
    pub fn upload_element_tables(&mut self) {
        if !self.elem_props_host.is_empty() {
            self.client.write(
                &self.gpu_elem_props,
                Bytes::from_bytes_vec(bytemuck::cast_slice(&self.elem_props_host).to_vec()),
            );
        }
        if !self.can_move_host.is_empty() {
            upload(&self.client, &self.gpu_can_move, &self.can_move_host);
        }
        if !self.spawn_defaults_host.is_empty() {
            upload(
                &self.client,
                &self.gpu_spawn_defaults,
                &self.spawn_defaults_host,
            );
        }
    }

    /// `true` when element `ty` lives in the `photons` layer (`TYPE_ENERGY`).
    pub fn is_energy_type(&self, ty: u32) -> bool {
        self.elem_is_energy
            .get(ty as usize)
            .copied()
            .unwrap_or(false)
    }
}

/// Device handles sized by the particle pool capacity.
struct ParticleHandles {
    ty: Handle,
    life: Handle,
    ctype: Handle,
    x: Handle,
    y: Handle,
    vx: Handle,
    vy: Handle,
    temp: Handle,
    equil: Handle,
    flags: Handle,
    tmp: Handle,
    tmp2: Handle,
    tmp3: Handle,
    tmp4: Handle,
    dcolour: Handle,
    intent_target: Handle,
}

impl ParticleHandles {
    /// Allocate one `u32`-word array per particle-sized buffer.
    fn alloc(client: &Client, capacity: usize) -> Self {
        let bytes = capacity * WORD;
        Self {
            ty: client.empty(bytes),
            life: client.empty(bytes),
            ctype: client.empty(bytes),
            x: client.empty(bytes),
            y: client.empty(bytes),
            vx: client.empty(bytes),
            vy: client.empty(bytes),
            temp: client.empty(bytes),
            equil: client.empty(bytes),
            flags: client.empty(bytes),
            tmp: client.empty(bytes),
            tmp2: client.empty(bytes),
            tmp3: client.empty(bytes),
            tmp4: client.empty(bytes),
            dcolour: client.empty(bytes),
            intent_target: client.empty(bytes),
        }
    }
}

/// Device handles sized by the canvas (`num_pixels` / `num_cells`).
struct GridHandles {
    pmap: Handle,
    photons: Handle,
    pmap_count: Handle,
    cell_winner: Handle,
    air_vx: Handle,
    air_vy: Handle,
    air_pv: Handle,
    air_hv: Handle,
    air_ovx: Handle,
    air_ovy: Handle,
    air_opv: Handle,
    air_ohv: Handle,
    fan_vx: Handle,
    fan_vy: Handle,
    wall_bmap: Handle,
    wall_emap: Handle,
    block_air: Handle,
    block_airh: Handle,
    fire_r: Handle,
    fire_g: Handle,
    fire_b: Handle,
    fire_layer: Handle,
    fx_cell: Handle,
    fx_axis: Handle,
    framebuf: Handle,
    grav_mass: Handle,
    grav_mass_prev: Handle,
    grav_tool_mass: Handle,
    grav_mask: Handle,
    grav_fx: Handle,
    grav_fy: Handle,
    blast_cell: Handle,
    fray_range: Handle,
    water_col_surf: Handle,
    water_col_target: Handle,
    water_winner: Handle,
    free_slots: Handle,
    live_count: Handle,
    emap_gate: Handle,
    edit_points: Handle,
    edit_rolls: Handle,
    move_lut: Handle,
    elem_phys: Handle,
    photon_mask: Handle,
    create_queue: Handle,
    portal_ty: Handle,
    portal_data: Handle,
    wireless: Handle,
    stkm_legs: Handle,
    stkm_accs: Handle,
    stkm_meta: Handle,
}

impl GridHandles {
    /// Allocate the pixel-sized and cell-sized device arrays.
    fn alloc(client: &Client, num_pixels: usize, num_cells: usize) -> Self {
        let pixel_bytes = num_pixels * WORD;
        let cell_bytes = num_cells * WORD;
        Self {
            pmap: client.empty(pixel_bytes),
            photons: client.empty(pixel_bytes),
            pmap_count: client.empty(pixel_bytes),
            cell_winner: client.empty(pixel_bytes),
            air_vx: client.empty(cell_bytes),
            air_vy: client.empty(cell_bytes),
            air_pv: client.empty(cell_bytes),
            air_hv: client.empty(cell_bytes),
            air_ovx: client.empty(cell_bytes),
            air_ovy: client.empty(cell_bytes),
            air_opv: client.empty(cell_bytes),
            air_ohv: client.empty(cell_bytes),
            fan_vx: client.empty(cell_bytes),
            fan_vy: client.empty(cell_bytes),
            wall_bmap: client.empty(cell_bytes),
            wall_emap: client.empty(cell_bytes),
            block_air: client.empty(cell_bytes),
            block_airh: client.empty(cell_bytes),
            fire_r: client.empty(cell_bytes),
            fire_g: client.empty(cell_bytes),
            fire_b: client.empty(cell_bytes),
            fire_layer: client.empty(pixel_bytes),
            fx_cell: client.empty(cell_bytes),
            fx_axis: client.empty(2 * pixel_bytes),
            framebuf: client.empty(pixel_bytes),
            grav_mass: client.empty(cell_bytes),
            grav_mass_prev: client.empty(cell_bytes),
            grav_tool_mass: client.empty(cell_bytes),
            grav_mask: client.empty(cell_bytes),
            grav_fx: alloc_zeroed(client, num_cells),
            grav_fy: alloc_zeroed(client, num_cells),
            blast_cell: client.empty(cell_bytes),
            fray_range: alloc_zeroed(client, 1),
            water_col_surf: client.empty(pixel_bytes),
            water_col_target: client.empty(pixel_bytes),
            water_winner: client.empty(pixel_bytes),
            free_slots: alloc_zeroed(client, SimBuffers::GPU_SPAWN_COUNT + 2),
            live_count: alloc_zeroed(client, 1),
            emap_gate: alloc_zeroed(client, 1),
            edit_points: alloc_zeroed(client, SimBuffers::EDIT_POINT_CAPACITY),
            edit_rolls: alloc_zeroed(client, SimBuffers::tool_rolls_capacity()),
            move_lut: client.empty(crate::move_kernel::build_move_lut_len() * WORD),
            elem_phys: client.empty(crate::move_kernel::elem_phys_len() * WORD),
            photon_mask: client.empty(crate::move_kernel::photon_mask_len() * WORD),
            create_queue: alloc_zeroed(
                client,
                SimBuffers::GPU_CREATE_COUNT * SimBuffers::CREATE_WORDS + 1,
            ),
            portal_ty: alloc_zeroed(client, PORTAL_CAPACITY),
            portal_data: alloc_zeroed(client, PORTAL_CAPACITY * PORTAL_WORDS),
            wireless: alloc_zeroed(client, PORTAL_CHANNELS * 2),
            stkm_legs: alloc_zeroed(client, STKM_SLOTS * STKM_LEGS_STRIDE),
            stkm_accs: alloc_zeroed(client, STKM_SLOTS * STKM_ACCS_STRIDE),
            stkm_meta: alloc_zeroed(client, STKM_SLOTS * STKM_META_STRIDE),
        }
    }
}

/// Allocate a device buffer of `words` `u32` words and initialise it to zero.
fn alloc_zeroed(client: &Client, words: usize) -> Handle {
    let zeros = vec![0u32; words.max(1)];
    let handle = client.empty(zeros.len() * WORD);
    upload(client, &handle, &zeros);
    handle
}

/// `floor(sqrt(max_pixels))` rounded down to a multiple of [`CELL`].
///
/// Derives [`SimBuffers::MAX_SUPPORTED_CANVAS_SIDE`] from the pool cap at
/// compile time, so the documented canvas limit can never drift from the
/// packed id width.
const fn max_canvas_side(max_pixels: usize) -> u32 {
    let limit = max_pixels as u64;
    let mut side = 0u32;
    while ((side + CELL) as u64) * ((side + CELL) as u64) <= limit {
        side += CELL;
    }
    side
}

// `capacity * 13 * 4 <= i32::MAX`: the `13` particle-sized `u32` arrays
// (`ty`, `life`, `ctype`, `x`, `y`, `vx`, `vy`, `temp`, `flags`, `tmp`,
// `tmp2`, `dcolour`, `intent_target`) must fit the `i32` byte budget.
const _: () = assert!(
    (SimBuffers::MAX_PARTICLE_CAPACITY as u64) * 13 * 4 <= i32::MAX as u64,
    "the particle pool cap exceeds the i32 byte budget of the slot buffers"
);

// The canvas grid buffers (`pmap`, `photons`, `cell_winner`, `framebuf`) cost
// `16` bytes per pixel and must fit the same budget at the largest canvas.
const _: () = assert!(
    (SimBuffers::MAX_SUPPORTED_CANVAS_SIDE as u64)
        * (SimBuffers::MAX_SUPPORTED_CANVAS_SIDE as u64)
        * 16
        <= i32::MAX as u64,
    "the max canvas exceeds the i32 byte budget of the grid buffers"
);

// The air/wall grid is `CELL` pixels wide, so a canvas edge must be a multiple
// of `CELL` for `max_supported_canvas` to be a valid `SimulationConfig` size.
const _: () = assert!(SimBuffers::MAX_SUPPORTED_CANVAS_SIDE.is_multiple_of(CELL));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_cap_keeps_ids_addressable() {
        // `2048 x 2048` asks for `2^23` slots, exactly one past the 23-bit id
        // field shared by `pmap_pack` and the movement winner key.
        let config = SimulationConfig::new(2048, 2048);
        assert_eq!(config.num_pixels(), 4_194_304);
        assert_eq!(SimBuffers::initial_capacity(&config), 0x7F_FFFF);

        let max_id = SimBuffers::MAX_PARTICLE_CAPACITY - 1;
        assert!(max_id < 0x7F_FFFF);
        assert_eq!(pmap_id(pmap_pack(max_id as u32, 1)), Some(max_id as u32));
    }

    #[test]
    fn max_canvas_fits_pool_and_byte_budget() {
        let side = SimBuffers::MAX_SUPPORTED_CANVAS_SIDE;
        assert_eq!(side, 2896);
        let pixels = u64::from(side) * u64::from(side);
        assert!(pixels <= SimBuffers::MAX_PARTICLE_CAPACITY as u64);
        assert!(pixels * 16 <= i32::MAX as u64);
        assert!((SimBuffers::MAX_PARTICLE_CAPACITY as u64) * 13 * 4 <= i32::MAX as u64);
        // One `CELL` step wider would need more pixels than the pool can hold.
        let next = u64::from(side + CELL);
        assert!(next * next > SimBuffers::MAX_PARTICLE_CAPACITY as u64);
    }
}
