//! Host <-> device synchronisation.
//!
//! Dirty tracking decides which side wins:
//!
//! | flag         | meaning                                                         |
//! |--------------|-----------------------------------------------------------------|
//! | `host_dirty` | the host shadow changed and must be uploaded before kernels run  |
//! | `gpu_dirty`  | a kernel ran and the host shadow must be read back first         |
//!
//! Only the categories a caller actually touched are transferred, and every
//! read-back tolerates a device failure by leaving the host shadow untouched.

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::server::Handle;
use tpt_core::constants::{PT_NONE, pmap_id};
use tpt_core::particle::GpuParticle;

use super::{FULL_PARTICLE_UPLOAD, PixelProbe, SimBuffers, WORD};

impl SimBuffers {
    /// Upload the host shadows whose dirty flags are set.
    ///
    /// Only the categories touched by the caller are transferred: painting a
    /// brush no longer re-uploads the air grid, and rigid-body rasterisation no
    /// longer re-uploads the walls.
    ///
    /// Particle edits are uploaded per slot (see [`Self::upload_particles`]).
    /// The device keeps whatever it produced for the other slots, so a brush
    /// stroke can no longer rewind the whole pool to the last read-back.
    pub fn sync_host_to_gpu(&mut self) {
        let any = self.host_particles_dirty
            || self.host_maps_dirty
            || self.host_air_dirty
            || self.host_walls_dirty
            || self.host_wall_state_dirty;
        if !any && self.host_dirty {
            // Defensive: an unclassified host edit uploads everything.
            self.upload_all_particles();
            self.upload_grids();
            self.host_dirty_slots.clear();
            self.gpu_particles_dirty = false;
            self.gpu_maps_dirty = false;
            self.gpu_air_dirty = false;
        } else {
            if self.host_particles_dirty {
                self.upload_particles();
            }
            if self.host_maps_dirty {
                self.upload_maps();
                self.gpu_maps_dirty = false;
            }
            if self.host_air_dirty {
                self.upload_air();
                self.gpu_air_dirty = false;
            }
            if self.host_walls_dirty {
                self.upload_walls();
            }
            if self.host_wall_state_dirty {
                self.upload_wall_state();
            }
        }
        self.host_dirty = false;
        self.host_particles_dirty = false;
        self.host_maps_dirty = false;
        self.host_air_dirty = false;
        self.host_walls_dirty = false;
        self.host_wall_state_dirty = false;
        // `upload_particles` may have uploaded only a few slots, and the map
        // words of those pixels; every other host shadow still lags the device.
        self.gpu_dirty = self.gpu_particles_dirty || self.gpu_maps_dirty || self.gpu_air_dirty;
    }

    /// Upload the host particle shadow.
    ///
    /// Only the slots recorded in `host_dirty_slots` are transferred, each as
    /// one `offset_start` / `offset_end` sub-range per array, with the map words
    /// of their pixels. The whole pool is uploaded when the SoA was replaced
    /// wholesale (an empty list: creation / reset / save restore), when a host
    /// pass wrote arbitrary fields ([`FULL_PARTICLE_UPLOAD`]), or when a canvas
    /// resize emptied the pool (`next_slot == 0`).
    ///
    /// A read-back does *not* force a whole-pool upload: it makes the host the
    /// device's equal, so the slots the caller touched are the only difference.
    /// Treating "just read back" as authoritative made the first brush frame
    /// after each undo snapshot re-send every array (173 MB on a 1600x900
    /// canvas) for a handful of changed slots.
    ///
    /// This is what keeps painting from rewinding the world: after the
    /// mouse-down undo snapshot the host shadow is frozen at the click state,
    /// so uploading untouched slots would roll the device back every frame.
    ///
    /// Allocating into a slot the host believes is free is safe even while the
    /// host shadow lags: the device only ever creates inside
    /// `GPU_SPAWN_START..GPU_SPAWN_END`, which the host snapshot keeps free, and
    /// `reconcile_particle_pool` rebuilds `free_list` / `next_slot` from the
    /// per-frame read-back that also publishes that window.
    pub(super) fn upload_particles(&mut self) {
        let mut slots = core::mem::take(&mut self.host_dirty_slots);
        let out_of_range = slots
            .iter()
            .any(|&id| id != FULL_PARTICLE_UPLOAD && id as usize >= self.capacity);
        // An empty list is the wholesale-replacement sentinel (creation, reset,
        // save restore) and `FULL_PARTICLE_UPLOAD` the direct-field writer;
        // `next_slot == 0` marks a pool reset whose recorded slot ids belong to
        // the old buffers.
        let full = slots.is_empty()
            || self.next_slot == 0
            || out_of_range
            || slots.contains(&FULL_PARTICLE_UPLOAD);

        if full {
            self.upload_all_particles();
        }
        slots.sort_unstable();
        slots.dedup();
        // One staging upload plus two scatter kernels: a brush frame touches
        // thousands of scattered slots, and one `write_buffer` per run and array
        // was what made painting crawl (see `particle_patch`).
        //
        // The patch also carries the `pmap`/`photons` words of the touched
        // pixels. `create_part` and `kill_at` never set `host_maps_dirty`, so a
        // stroke that goes through the whole-array upload still needs them.
        self.upload_particle_patch(&slots);

        self.host_dirty_slots.clear();
        // A full upload synchronised every host slot to the device. A sparse
        // patch only wrote the slots the caller touched (plus their map
        // words), so it preserves whether other slots on the device already
        // diverged (`gpu_particles_dirty`) instead of forcing paused canvases
        // to treat themselves as dirty on the next snapshot.
        if full {
            self.gpu_particles_dirty = false;
        }
    }

    /// Upload every particle array in full.
    pub(super) fn upload_all_particles(&self) {
        let c = &self.client;
        upload(c, &self.gpu_part_ty, &self.part_ty);
        upload(c, &self.gpu_part_life, &self.part_life);
        upload(c, &self.gpu_part_ctype, &self.part_ctype);
        upload(c, &self.gpu_part_x, &self.part_x);
        upload(c, &self.gpu_part_y, &self.part_y);
        upload(c, &self.gpu_part_vx, &self.part_vx);
        upload(c, &self.gpu_part_vy, &self.part_vy);
        upload(c, &self.gpu_part_temp, &self.part_temp);
        upload(c, &self.gpu_part_flags, &self.part_flags);
        upload(c, &self.gpu_part_tmp, &self.part_tmp);
        upload(c, &self.gpu_part_tmp2, &self.part_tmp2);
        upload(c, &self.gpu_part_tmp3, &self.part_tmp3);
        upload(c, &self.gpu_part_tmp4, &self.part_tmp4);
        upload(c, &self.gpu_part_dcolour, &self.part_dcolour);
    }

    fn upload_maps(&self) {
        let c = &self.client;
        upload(c, &self.gpu_pmap, &self.pmap);
        upload(c, &self.gpu_photons, &self.photons);
    }

    fn upload_air(&self) {
        let c = &self.client;
        upload(c, &self.gpu_air_vx, &self.air_vx);
        upload(c, &self.gpu_air_vy, &self.air_vy);
        upload(c, &self.gpu_air_pv, &self.air_pv);
        upload(c, &self.gpu_air_hv, &self.air_hv);
        upload(c, &self.gpu_air_ovx, &self.air_ovx);
        upload(c, &self.gpu_air_ovy, &self.air_ovy);
        upload(c, &self.gpu_air_opv, &self.air_opv);
        upload(c, &self.gpu_air_ohv, &self.air_ohv);
    }

    fn upload_walls(&self) {
        let c = &self.client;
        upload(c, &self.gpu_fan_vx, &self.fan_vx);
        upload(c, &self.gpu_fan_vy, &self.fan_vy);
        upload(c, &self.gpu_wall_bmap, &self.wall_bmap);
        upload(c, &self.gpu_wall_emap, &self.wall_emap);
        upload(c, &self.gpu_block_air, &self.block_air);
        upload(c, &self.gpu_block_airh, &self.block_airh);
    }

    fn upload_wall_state(&self) {
        let c = &self.client;
        upload(c, &self.gpu_wall_emap, &self.wall_emap);
        upload(c, &self.gpu_block_air, &self.block_air);
        upload(c, &self.gpu_block_airh, &self.block_airh);
    }

    /// Uploads every host shadow, regardless of the dirty flags.
    pub fn upload_all_host_arrays(&mut self) {
        self.upload_all_particles();
        self.upload_grids();
        self.host_dirty_slots.clear();
        self.gpu_particles_dirty = false;
        self.gpu_maps_dirty = false;
        self.gpu_air_dirty = false;
        self.gpu_dirty = false;
        self.host_dirty = false;
        self.host_particles_dirty = false;
        self.host_maps_dirty = false;
        self.host_air_dirty = false;
        self.host_walls_dirty = false;
        self.host_wall_state_dirty = false;
    }

    fn upload_grids(&self) {
        self.upload_maps();
        self.upload_air();
        self.upload_walls();
    }

    /// Read every device buffer back into the host shadows.
    ///
    /// Call this before inspecting particle state on the CPU after a kernel pass
    /// (save capture, Lua queries, undo snapshots). The per-frame pipeline uses
    /// the narrower [`Self::sync_particles_gpu_to_host`] and
    /// [`Self::sync_maps_gpu_to_host`] instead.
    pub fn sync_gpu_to_host(&mut self) {
        // Only the used prefix of the particle pool can hold data, so the fast
        // path trims the particle handles instead of copying the whole (often
        // mostly free) pool back from the device every frame. If the device
        // rejects the trimmed read, fall back to the full buffers.
        if !self.read_back_device(true) && !self.read_back_device(false) {
            // A failed read leaves the host shadow untouched; the next frame will
            // simply re-upload it. Never panic on a device hiccup.
            return;
        }

        // Kernels may have freed slots on the device (ty == PT_NONE) without the
        // host knowing: settle the pool bookkeeping so those slots are reusable.
        self.reconcile_particle_pool();

        self.host_dirty_slots.clear();
        self.gpu_dirty = false;
        self.gpu_particles_dirty = false;
        self.gpu_maps_dirty = false;
        self.gpu_air_dirty = false;
        self.host_dirty = false;
        self.host_particles_dirty = false;
        self.host_maps_dirty = false;
        self.host_air_dirty = false;
        self.host_walls_dirty = false;
        self.host_wall_state_dirty = false;
    }

    /// Bring the host particle SoA up to date with the device.
    ///
    /// Returns `false` when the device read failed (host shadows untouched).
    pub fn sync_particles_gpu_to_host(&mut self) -> bool {
        if !self.read_particles_device(true) && !self.read_particles_device(false) {
            return false;
        }
        self.reconcile_particle_pool();
        // The read-back replaced every host slot, so pending slot edits are moot.
        self.host_dirty_slots.clear();
        self.gpu_particles_dirty = false;
        self.gpu_dirty = self.gpu_particles_dirty || self.gpu_maps_dirty || self.gpu_air_dirty;
        self.host_particles_dirty = false;
        self.host_dirty = self.host_particles_dirty
            || self.host_maps_dirty
            || self.host_air_dirty
            || self.host_walls_dirty;
        true
    }

    /// Bring the host `pmap` / `photons` up to date with the device.
    pub fn sync_maps_gpu_to_host(&mut self) -> bool {
        let handles = vec![self.gpu_pmap.clone(), self.gpu_photons.clone()];
        let Ok(readback) = self.client.read(handles) else {
            return false;
        };
        let mut it = readback.iter();
        read_back(it.next(), &mut self.pmap);
        read_back(it.next(), &mut self.photons);
        self.gpu_maps_dirty = false;
        self.gpu_dirty = self.gpu_particles_dirty || self.gpu_maps_dirty || self.gpu_air_dirty;
        self.host_maps_dirty = false;
        self.host_dirty = self.host_particles_dirty
            || self.host_maps_dirty
            || self.host_air_dirty
            || self.host_walls_dirty;
        true
    }

    /// Refreshes the host gravity shadows (`grav_mass`, `grav_fx`, `grav_fy`).
    ///
    /// The Poisson solve runs entirely on the device, so the shadows are stale
    /// after every tick. Only the Lua grid getters and save capture ask for
    /// them; the per-frame path never does.
    pub fn sync_gravity_gpu_to_host(&mut self) -> bool {
        // The read-back mirrors `gravOut`, which TPT keeps zero while the model
        // is off (`Simulation.cpp:3975-3983`); a field left by the last solve
        // must never leak into Lua or a save.
        if !self.config.newton_grav {
            self.zero_gravity_forces();
            return true;
        }
        if !self.grav_shadow_dirty {
            return true;
        }
        let cells = self.config.num_cells();
        let handles = vec![
            self.gpu_grav_mass.clone(),
            self.gpu_grav_fx.clone(),
            self.gpu_grav_fy.clone(),
        ];
        let Ok(readback) = self.client.read(handles) else {
            return false;
        };
        let mut it = readback.iter();
        if let Some(bytes) = it.next() {
            let raw: &[u8] = bytes;
            for (i, chunk) in raw.as_chunks::<4>().0.iter().take(cells).enumerate() {
                let quantised = i32::from_le_bytes(*chunk);
                self.grav_mass[i] = quantised as f32 * crate::gravity_kernel::MASS_SCALE_INV;
            }
        }
        read_back(it.next(), &mut self.grav_fx);
        read_back(it.next(), &mut self.grav_fy);
        self.grav_shadow_dirty = false;
        true
    }

    /// Zeroes the force grids on both sides (`gravOut = {}`).
    ///
    /// `Simulation::EnableNewtonianGravity(false)` resets the whole gravity
    /// output (`Simulation.cpp:3975-3983`), and `Renderer::RenderSimulation`
    /// lenses that field whenever `DISPLAY_WARP` is set regardless of the model
    /// (`Renderer.cpp:42-47`). The solve is the only device writer, so the two
    /// uploads only happen once after a solve; while
    /// [`Self::grav_force_stale`] is false the device buffers are already zero.
    pub fn zero_gravity_forces(&mut self) {
        if !self.grav_force_stale {
            return;
        }
        let cells = self.config.num_cells();
        if cells == 0 || self.grav_fx.len() < cells || self.grav_fy.len() < cells {
            return;
        }
        self.grav_fx[..cells].fill(0.0);
        self.grav_fy[..cells].fill(0.0);
        upload(&self.client, &self.gpu_grav_fx, &self.grav_fx[..cells]);
        upload(&self.client, &self.gpu_grav_fy, &self.grav_fy[..cells]);
        self.grav_force_stale = false;
        self.grav_shadow_dirty = false;
    }

    /// Bring the host air grids up to date with the device.
    pub fn sync_air_gpu_to_host(&mut self) -> bool {
        // `air_ovx/ovy/opv/ohv` are internal scratch buffers overwritten on
        // every air pass (`air_kernel`), so only the four authoritative fields
        // cross back to the host.
        let handles = vec![
            self.gpu_air_vx.clone(),
            self.gpu_air_vy.clone(),
            self.gpu_air_pv.clone(),
            self.gpu_air_hv.clone(),
        ];
        let Ok(readback) = self.client.read(handles) else {
            return false;
        };
        let mut it = readback.iter();
        read_back(it.next(), &mut self.air_vx);
        read_back(it.next(), &mut self.air_vy);
        read_back(it.next(), &mut self.air_pv);
        read_back(it.next(), &mut self.air_hv);
        self.gpu_air_dirty = false;
        self.gpu_dirty = self.gpu_particles_dirty || self.gpu_maps_dirty || self.gpu_air_dirty;
        self.host_air_dirty = false;
        self.host_dirty = self.host_particles_dirty
            || self.host_maps_dirty
            || self.host_air_dirty
            || self.host_walls_dirty;
        true
    }

    /// Bring the host wall grids up to date with the device.
    pub fn sync_walls_gpu_to_host(&mut self) -> bool {
        let handles = vec![
            self.gpu_wall_bmap.clone(),
            self.gpu_wall_emap.clone(),
            self.gpu_fan_vx.clone(),
            self.gpu_fan_vy.clone(),
        ];
        let Ok(readback) = self.client.read(handles) else {
            return false;
        };
        let mut it = readback.iter();
        read_back(it.next(), &mut self.wall_bmap);
        read_back(it.next(), &mut self.wall_emap);
        read_back(it.next(), &mut self.fan_vx);
        read_back(it.next(), &mut self.fan_vy);
        self.refresh_wall_masks();
        self.host_walls_dirty = false;
        self.host_dirty = self.host_particles_dirty
            || self.host_maps_dirty
            || self.host_air_dirty
            || self.host_walls_dirty;
        true
    }

    /// Read every device buffer into the host shadows.
    ///
    /// `trim_particles` limits the particle copy to the used prefix of the pool.
    /// Returns `false` when the device read failed (host shadows untouched).
    fn read_back_device(&mut self, trim_particles: bool) -> bool {
        let has_particles = self.next_slot > 0 || self.gpu_spawn_used;
        let used_slots = self.used_slot_bound();
        let trim = if trim_particles {
            ((self.capacity - used_slots) * WORD) as u64
        } else {
            0
        };

        // Only authoritative buffers cross back: `gpu_framebuf`, `gpu_fire_*`
        // (render-only) and `gpu_air_o*` (air-kernel scratch) stay on the
        // device, and `gpu_wall_bmap` / `gpu_fan_*` are host-authored. When no
        // particle exists on either side, `part_*` and `pmap`/`photons` are
        // already empty on the host and do not need a multi-megabyte transfer.
        let mut handles: Vec<Handle> = Vec::with_capacity(21);
        if has_particles {
            handles.extend([
                self.gpu_part_ty.clone().offset_end(trim),
                self.gpu_part_life.clone().offset_end(trim),
                self.gpu_part_ctype.clone().offset_end(trim),
                self.gpu_part_x.clone().offset_end(trim),
                self.gpu_part_y.clone().offset_end(trim),
                self.gpu_part_vx.clone().offset_end(trim),
                self.gpu_part_vy.clone().offset_end(trim),
                self.gpu_part_temp.clone().offset_end(trim),
                self.gpu_part_flags.clone().offset_end(trim),
                self.gpu_part_tmp.clone().offset_end(trim),
                self.gpu_part_tmp2.clone().offset_end(trim),
                self.gpu_part_tmp3.clone().offset_end(trim),
                self.gpu_part_tmp4.clone().offset_end(trim),
                self.gpu_part_dcolour.clone().offset_end(trim),
                self.gpu_pmap.clone(),
                self.gpu_photons.clone(),
            ]);
        }
        handles.extend([
            self.gpu_air_vx.clone(),
            self.gpu_air_vy.clone(),
            self.gpu_air_pv.clone(),
            self.gpu_air_hv.clone(),
            self.gpu_wall_emap.clone(),
        ]);

        let Ok(readback) = self.client.read(handles) else {
            return false;
        };

        let mut it = readback.iter();
        if has_particles {
            read_back(it.next(), &mut self.part_ty);
            read_back(it.next(), &mut self.part_life);
            read_back(it.next(), &mut self.part_ctype);
            read_back(it.next(), &mut self.part_x);
            read_back(it.next(), &mut self.part_y);
            read_back(it.next(), &mut self.part_vx);
            read_back(it.next(), &mut self.part_vy);
            read_back(it.next(), &mut self.part_temp);
            read_back(it.next(), &mut self.part_flags);
            read_back(it.next(), &mut self.part_tmp);
            read_back(it.next(), &mut self.part_tmp2);
            read_back(it.next(), &mut self.part_tmp3);
            read_back(it.next(), &mut self.part_tmp4);
            read_back(it.next(), &mut self.part_dcolour);
            read_back(it.next(), &mut self.pmap);
            read_back(it.next(), &mut self.photons);
        } else {
            self.pmap.fill(0);
            self.photons.fill(0);
        }
        read_back(it.next(), &mut self.air_vx);
        read_back(it.next(), &mut self.air_vy);
        read_back(it.next(), &mut self.air_pv);
        read_back(it.next(), &mut self.air_hv);
        read_back(it.next(), &mut self.wall_emap);
        true
    }

    /// Read the particle SoA from the device (optionally trimmed to the used
    /// prefix). Returns `false` when the device read failed.
    fn read_particles_device(&mut self, trim_particles: bool) -> bool {
        let used_slots = self.used_slot_bound();
        let trim = if trim_particles {
            ((self.capacity - used_slots) * WORD) as u64
        } else {
            0
        };

        let handles = vec![
            self.gpu_part_ty.clone().offset_end(trim),
            self.gpu_part_life.clone().offset_end(trim),
            self.gpu_part_ctype.clone().offset_end(trim),
            self.gpu_part_x.clone().offset_end(trim),
            self.gpu_part_y.clone().offset_end(trim),
            self.gpu_part_vx.clone().offset_end(trim),
            self.gpu_part_vy.clone().offset_end(trim),
            self.gpu_part_temp.clone().offset_end(trim),
            self.gpu_part_flags.clone().offset_end(trim),
            self.gpu_part_tmp.clone().offset_end(trim),
            self.gpu_part_tmp2.clone().offset_end(trim),
            self.gpu_part_tmp3.clone().offset_end(trim),
            self.gpu_part_tmp4.clone().offset_end(trim),
            self.gpu_part_dcolour.clone().offset_end(trim),
        ];
        let Ok(readback) = self.client.read(handles) else {
            return false;
        };
        let mut it = readback.iter();
        read_back(it.next(), &mut self.part_ty);
        read_back(it.next(), &mut self.part_life);
        read_back(it.next(), &mut self.part_ctype);
        read_back(it.next(), &mut self.part_x);
        read_back(it.next(), &mut self.part_y);
        read_back(it.next(), &mut self.part_vx);
        read_back(it.next(), &mut self.part_vy);
        read_back(it.next(), &mut self.part_temp);
        read_back(it.next(), &mut self.part_flags);
        read_back(it.next(), &mut self.part_tmp);
        read_back(it.next(), &mut self.part_tmp2);
        read_back(it.next(), &mut self.part_tmp3);
        read_back(it.next(), &mut self.part_tmp4);
        read_back(it.next(), &mut self.part_dcolour);
        true
    }

    /// Ensures the host shadows match the device before host code inspects them.
    ///
    /// This is the only full read-back left in the frame pipeline: the renderer
    /// never needs it, and [`crate::SimulationEngine::step`] only calls the
    /// narrow variants when rigid bodies or water equalisation are active.
    pub fn ensure_host_current(&mut self) {
        if self.gpu_dirty {
            self.sync_gpu_to_host();
        }
    }

    /// Reads the `pmap`/`photons` rectangle `[x0..=x1] x [y0..=y1]` from the
    /// device.
    ///
    /// Used by the property/decoration and gravity-well tools, whose
    /// read-modify-write needs the pixels under the stroke. `place_particles`
    /// deliberately does not call it: brush placement only allocates new slots
    /// and updates the cached map, so it never blocks on a read-back.
    pub fn read_map_rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
        let xres = self.config.xres as i32;
        let yres = self.config.yres as i32;
        let xa = x0.min(x1).clamp(0, xres - 1);
        let xb = x0.max(x1).clamp(0, xres - 1);
        let ya = y0.min(y1).clamp(0, yres - 1);
        let yb = y0.max(y1).clamp(0, yres - 1);
        if xa > xb || ya > yb {
            return true;
        }
        let stride = xres as usize;
        let total = self.pmap.len() * WORD;
        let rows = (yb - ya + 1) as usize;
        let mut handles = Vec::with_capacity(rows * 2);
        for y in ya..=yb {
            let start = (y as usize * stride + xa as usize) * WORD;
            let end = (y as usize * stride + xb as usize + 1) * WORD;
            handles.push(
                self.gpu_pmap
                    .clone()
                    .offset_start(start as u64)
                    .offset_end((total - end) as u64),
            );
            handles.push(
                self.gpu_photons
                    .clone()
                    .offset_start(start as u64)
                    .offset_end((total - end) as u64),
            );
        }
        let Ok(readback) = self.client.read(handles) else {
            return false;
        };
        for row in 0..rows {
            let y = ya as usize + row;
            for layer in 0..2usize {
                let src: &[u8] = &readback[row * 2 + layer];
                let dst = if layer == 0 {
                    &mut self.pmap
                } else {
                    &mut self.photons
                };
                for col in 0..=(xb - xa) as usize {
                    let base = col * WORD;
                    if base + WORD <= src.len() {
                        let word = u32::from_le_bytes([
                            src[base],
                            src[base + 1],
                            src[base + 2],
                            src[base + 3],
                        ]);
                        dst[y * stride + xa as usize + col] = word;
                    }
                }
            }
        }
        true
    }

    /// Reads one pixel's map, wall and air state plus the occupying particle.
    ///
    /// Used by the HUD inspector and the middle-click element sampler, so the
    /// renderer and the simulation never need a per-frame full read-back.
    pub fn probe_pixel(&self, x: i32, y: i32) -> Option<PixelProbe> {
        let pixel = self.pixel_index(x, y)?;
        let cell = self.cell_index(x, y)?;
        let word_handle = |handle: Handle, index: usize, len: usize| {
            handle
                .offset_start((index * WORD) as u64)
                .offset_end(((len - index - 1) * WORD) as u64)
        };
        let handles = vec![
            word_handle(self.gpu_pmap.clone(), pixel, self.pmap.len()),
            word_handle(self.gpu_photons.clone(), pixel, self.photons.len()),
            word_handle(self.gpu_wall_bmap.clone(), cell, self.wall_bmap.len()),
            word_handle(self.gpu_wall_emap.clone(), cell, self.wall_emap.len()),
            word_handle(self.gpu_air_pv.clone(), cell, self.air_pv.len()),
        ];
        let readback = self.client.read(handles).ok()?;
        let word = |index: usize| -> u32 {
            let bytes: &[u8] = &readback[index];
            if bytes.len() < WORD {
                return 0;
            }
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        };
        let pmap_word = word(0);
        let photons_word = word(1);
        let wall = (word(2) & 0xFF) as u8;
        let emap = (word(3) & 0xFF) as u8;
        let pressure = f32::from_bits(word(4));

        let id = pmap_id(pmap_word).or_else(|| pmap_id(photons_word));
        let particle = id.and_then(|id| {
            let id = id as usize;
            if id >= self.capacity {
                return None;
            }
            let handles = vec![
                word_handle(self.gpu_part_ty.clone(), id, self.capacity),
                word_handle(self.gpu_part_life.clone(), id, self.capacity),
                word_handle(self.gpu_part_ctype.clone(), id, self.capacity),
                word_handle(self.gpu_part_x.clone(), id, self.capacity),
                word_handle(self.gpu_part_y.clone(), id, self.capacity),
                word_handle(self.gpu_part_vx.clone(), id, self.capacity),
                word_handle(self.gpu_part_vy.clone(), id, self.capacity),
                word_handle(self.gpu_part_temp.clone(), id, self.capacity),
                word_handle(self.gpu_part_flags.clone(), id, self.capacity),
                word_handle(self.gpu_part_tmp.clone(), id, self.capacity),
                word_handle(self.gpu_part_tmp2.clone(), id, self.capacity),
                word_handle(self.gpu_part_tmp3.clone(), id, self.capacity),
                word_handle(self.gpu_part_tmp4.clone(), id, self.capacity),
                word_handle(self.gpu_part_dcolour.clone(), id, self.capacity),
            ];
            let rb = self.client.read(handles).ok()?;
            let w = |index: usize| -> u32 {
                let bytes: &[u8] = &rb[index];
                if bytes.len() < WORD {
                    return 0;
                }
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            };
            let p = GpuParticle {
                ty: w(0),
                life: w(1) as i32,
                ctype: w(2),
                x: f32::from_bits(w(3)),
                y: f32::from_bits(w(4)),
                vx: f32::from_bits(w(5)),
                vy: f32::from_bits(w(6)),
                temp: f32::from_bits(w(7)),
                flags: w(8),
                tmp: w(9) as i32,
                tmp2: w(10) as i32,
                tmp3: w(11) as i32,
                tmp4: w(12) as i32,
                dcolour: w(13),
                _pad0: 0,
                _pad1: 0,
            };
            if p.ty == PT_NONE { None } else { Some(p) }
        });

        Some(PixelProbe {
            particle,
            wall,
            emap,
            pressure,
        })
    }

    /// Refreshes [`Self::active_count`] from the device.
    ///
    /// The reduction runs on the device ([`crate::live_count`]) and only the
    /// resulting word crosses back, so the HUD never copies `part_ty` over the
    /// bus and never counts on the CPU.
    pub fn refresh_active_count_device(&mut self) -> Option<usize> {
        let count = crate::live_count::refresh_live_count(self)?;
        self.active_count = count;
        Some(count)
    }

    /// Starts a non-blocking device probe for pixel `(x, y)`.
    ///
    /// Used by the HUD inspector so hovering or dragging across a large canvas
    /// never stalls the render loop waiting for the GPU queue to drain.
    pub fn probe_pixel_async(
        &self,
        x: i32,
        y: i32,
    ) -> Option<crate::live_count::PendingRead<Option<PixelProbe>>> {
        let pixel = self.pixel_index(x, y)?;
        let cell = self.cell_index(x, y)?;
        let word_handle = |handle: Handle, index: usize, len: usize| {
            handle
                .offset_start((index * WORD) as u64)
                .offset_end(((len - index - 1) * WORD) as u64)
        };
        let handles = vec![
            word_handle(self.gpu_pmap.clone(), pixel, self.pmap.len()),
            word_handle(self.gpu_photons.clone(), pixel, self.photons.len()),
            word_handle(self.gpu_wall_bmap.clone(), cell, self.wall_bmap.len()),
            word_handle(self.gpu_wall_emap.clone(), cell, self.wall_emap.len()),
            word_handle(self.gpu_air_pv.clone(), cell, self.air_pv.len()),
        ];
        let client = self.client.clone();
        let capacity = self.capacity;
        let part_handles = [
            self.gpu_part_ty.clone(),
            self.gpu_part_life.clone(),
            self.gpu_part_ctype.clone(),
            self.gpu_part_x.clone(),
            self.gpu_part_y.clone(),
            self.gpu_part_vx.clone(),
            self.gpu_part_vy.clone(),
            self.gpu_part_temp.clone(),
            self.gpu_part_flags.clone(),
            self.gpu_part_tmp.clone(),
            self.gpu_part_tmp2.clone(),
            self.gpu_part_tmp3.clone(),
            self.gpu_part_tmp4.clone(),
            self.gpu_part_dcolour.clone(),
        ];
        let fut = Box::pin(async move {
            let readback = client.read_async(handles).await.ok()?;
            let word = |index: usize| -> u32 {
                let bytes: &[u8] = &readback[index];
                if bytes.len() < WORD {
                    return 0;
                }
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            };
            let pmap_word = word(0);
            let photons_word = word(1);
            let wall = (word(2) & 0xFF) as u8;
            let emap = (word(3) & 0xFF) as u8;
            let pressure = f32::from_bits(word(4));

            let id = pmap_id(pmap_word).or_else(|| pmap_id(photons_word));
            let particle = match id {
                Some(id) if (id as usize) < capacity => {
                    let id = id as usize;
                    let handles = part_handles
                        .into_iter()
                        .map(|h| {
                            h.offset_start((id * WORD) as u64)
                                .offset_end(((capacity - id - 1) * WORD) as u64)
                        })
                        .collect();
                    let rb = client.read_async(handles).await.ok()?;
                    let w = |index: usize| -> u32 {
                        let bytes: &[u8] = &rb[index];
                        if bytes.len() < WORD {
                            return 0;
                        }
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                    };
                    let p = GpuParticle {
                        ty: w(0),
                        life: w(1) as i32,
                        ctype: w(2),
                        x: f32::from_bits(w(3)),
                        y: f32::from_bits(w(4)),
                        vx: f32::from_bits(w(5)),
                        vy: f32::from_bits(w(6)),
                        temp: f32::from_bits(w(7)),
                        flags: w(8),
                        tmp: w(9) as i32,
                        tmp2: w(10) as i32,
                        tmp3: w(11) as i32,
                        tmp4: w(12) as i32,
                        dcolour: w(13),
                        _pad0: 0,
                        _pad1: 0,
                    };
                    if p.ty == PT_NONE { None } else { Some(p) }
                }
                _ => None,
            };

            Some(PixelProbe {
                particle,
                wall,
                emap,
                pressure,
            })
        });
        Some(crate::live_count::PendingRead::spawn(fut))
    }
}

/// Upload a host slice into an existing device buffer.
pub(crate) fn upload<T>(client: &Client, handle: &Handle, data: &[T])
where
    T: bytemuck::Pod + Send + Sync,
{
    if data.is_empty() {
        return;
    }
    client.write(handle, Bytes::from_elems(data.to_vec()));
}

/// Copy a device read-back into a host slice, tolerating misalignment.
fn read_back<T: bytemuck::Pod>(bytes: Option<&Bytes>, out: &mut [T]) {
    let Some(bytes) = bytes else {
        return;
    };
    let raw: &[u8] = bytes;
    let elem = core::mem::size_of::<T>();
    let n = out.len().min(raw.len() / elem);
    if n == 0 {
        return;
    }
    let raw = &raw[..n * elem];
    if let Ok(src) = bytemuck::try_cast_slice::<u8, T>(raw) {
        out[..n].copy_from_slice(src);
    } else {
        for (dst, chunk) in out[..n].iter_mut().zip(raw.chunks_exact(elem)) {
            *dst = *bytemuck::from_bytes::<T>(chunk);
        }
    }
}
