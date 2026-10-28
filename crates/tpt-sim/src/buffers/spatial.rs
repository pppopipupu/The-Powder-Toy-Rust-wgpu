//! Spatial maps (`pmap` / `photons` / `pmap_count`), derived wall masks and the
//! read-only grid queries the HUD and the Lua bridge use.
//!
//! `pmap` holds matter particles and `photons` holds `TYPE_ENERGY` particles,
//! both indexed `y * xres + x` and packed with
//! [`tpt_core::constants::pmap_pack`]. `wall_bmap` / `wall_emap` are per
//! `CELL x CELL` block; `block_air` / `block_airh` are derived from them.
//!
//! The device rebuild ([`SimBuffers::recalc_free_particles_device`]) is the port
//! of `Simulation::RecalcFreeParticles` (`Simulation.cpp:3347-3415`): it decays
//! particle life, clears and refills the maps, rebuilds `pmap_count`, and (on
//! the reference's 1-in-10 cadence) runs `CheckStacking`
//! (`Simulation.cpp:3620-3681`).

use cubecl::prelude::*;
use tpt_core::constants::*;
use tpt_core::particle::GpuParticle;
use tpt_core::walls::{is_wall_blocking_air, is_wall_blocking_heat};

use crate::move_kernel::{clear_spatial_maps_kernel, scatter_spatial_maps_kernel};

use super::SimBuffers;

/// Work units per cube (the portable WebGPU minimum for a 1-D cube).
const CUBE_DIM: u32 = 256;

/// `PT_NONE` as a `u32` for kernel-side guards.
const PT_NONE_U32: u32 = PT_NONE;
/// `PT_NBHL` (stack collapse result).
const PT_NBHL_U32: u32 = PT_NBHL;
const _: () = assert!(PT_NBHL_U32 == 40);
/// Stack-count exclusions from `RecalcFreeParticles` (`Simulation.cpp:3383`).
const PT_THDR_U32: u32 = PT_THDR;
const PT_EMBR_U32: u32 = PT_EMBR;
const PT_FIGH_U32: u32 = PT_FIGH;
const PT_PLSM_U32: u32 = PT_PLSM;
const _: () = assert!(PT_THDR_U32 == 96);
const _: () = assert!(PT_EMBR_U32 == 153);
const _: () = assert!(PT_FIGH_U32 == 159);
const _: () = assert!(PT_PLSM_U32 == 28);
/// `WL_STASIS` / `WL_EHOLE` as `u32`.
const WL_STASIS_U32: u32 = WL_STASIS as u32;
const WL_EHOLE_U32: u32 = WL_EHOLE as u32;
const _: () = assert!(WL_STASIS_U32 == 18);
const _: () = assert!(WL_EHOLE_U32 == 12);
/// `TYPE_ENERGY` and the three life flags, mirrored for kernel bodies.
const TYPE_ENERGY_U32: u32 = TYPE_ENERGY;
const PROP_LIFE_DEC_U32: u32 = PROP_LIFE_DEC;
const PROP_LIFE_KILL_U32: u32 = PROP_LIFE_KILL;
const PROP_LIFE_KILL_DEC_U32: u32 = PROP_LIFE_KILL_DEC;
const _: () = assert!(TYPE_ENERGY_U32 == 0x10);
const _: () = assert!(PROP_LIFE_DEC_U32 == 0x0000_4000);
const _: () = assert!(PROP_LIFE_KILL_U32 == 0x0000_8000);
const _: () = assert!(PROP_LIFE_KILL_DEC_U32 == 0x0001_0000);

/// Integer avalanche used for the stacking roll (lowbias32-style).
#[cube]
fn spatial_hash(seed: u32) -> u32 {
    let mut h = seed ^ 0x9E37_79B9u32;
    h ^= h >> 16u32;
    h *= 0x7FEB_352Du32;
    h ^= h >> 15u32;
    h *= 0x846C_A68Bu32;
    h ^ (h >> 16u32)
}

/// `index * salt`, kept behind a call so the product is never constant-folded
/// in the unexpanded host build.
#[cube]
fn mix_index(index: u32, salt: u32) -> u32 {
    index * salt
}

/// Decreases particle `life` and kills expired particles, mirroring the
/// `do_life_dec` block of `RecalcFreeParticles` (`Simulation.cpp:3396-3410`).
///
/// A particle with `PROP_LIFE_DEC` and positive life loses one tick unless it
/// sits in a `WL_STASIS` cell with `emap < 8`; a life that reaches zero kills
/// the particle when it also carries `PROP_LIFE_KILL` or `PROP_LIFE_KILL_DEC`.
/// A particle whose life is already zero dies when it carries
/// `PROP_LIFE_KILL`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn life_decay_kernel(
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_x: &[f32],
    part_y: &[f32],
    elem_props: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    slot_count: u32,
    elem_rows: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    _ycells: u32,
) {
    let i = ABSOLUTE_POS;
    if i < slot_count as usize {
        let ty = part_ty[i];
        if ty != PT_NONE_U32 && ty < elem_rows {
            let x = i32::cast_from(part_x[i] + 0.5f32);
            let y = i32::cast_from(part_y[i] + 0.5f32);
            let mut in_bounds = 0u32;
            let mut stasis = 0u32;
            if x >= 0i32 && y >= 0i32 && x < xres as i32 && y < yres as i32 {
                in_bounds = 1u32;
                let cell = ((y as u32) / CELL) * xcells + ((x as u32) / CELL);
                if (wall_bmap[cell as usize] & 0xFFu32) == WL_STASIS_U32
                    && (wall_emap[cell as usize] & 0xFFu32) < 8u32
                {
                    stasis = 1u32;
                }
            }

            let props = elem_props[(ty as usize) * 32usize + 1usize];
            let life = part_life[i];
            if life > 0i32
                && (props & PROP_LIFE_DEC_U32) != 0u32
                && (in_bounds == 0u32 || stasis == 0u32)
            {
                let next = life - 1i32;
                part_life[i] = next;
                if next <= 0i32 && (props & (PROP_LIFE_KILL_DEC_U32 | PROP_LIFE_KILL_U32)) != 0u32 {
                    part_ty[i] = PT_NONE_U32;
                }
            } else if life <= 0i32
                && (props & PROP_LIFE_KILL_U32) != 0u32
                && (in_bounds == 0u32 || stasis == 0u32)
            {
                part_ty[i] = PT_NONE_U32;
            }
        }
    }
}

/// Clears `pmap_count` before [`pmap_count_kernel`] refills it
/// (`Simulation.cpp:3363`).
#[cube(launch_unchecked)]
pub fn pmap_count_clear_kernel(pmap_count: &mut [Atomic<u32>], num_pixels: u32) {
    let idx = ABSOLUTE_POS;
    if idx < num_pixels as usize {
        pmap_count[idx].store(0u32);
    }
}

/// Rebuilds `pmap_count` from the live particle list
/// (`Simulation.cpp:3381-3384`).
///
/// Every non-energy particle increments its pixel; `THDR`, `EMBR`, `FIGH` and
/// `PLSM` are excluded because they are allowed to stack without triggering the
/// black-hole collapse.
#[cube(launch_unchecked)]
pub fn pmap_count_kernel(
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    elem_props: &[u32],
    pmap_count: &mut [Atomic<u32>],
    slot_count: u32,
    elem_rows: u32,
    xres: u32,
    yres: u32,
) {
    let i = ABSOLUTE_POS;
    if i < slot_count as usize {
        let ty = part_ty[i];
        if ty != PT_NONE_U32 && ty < elem_rows {
            let x = i32::cast_from(part_x[i] + 0.5f32);
            let y = i32::cast_from(part_y[i] + 0.5f32);
            if x >= 0i32 && y >= 0i32 && x < xres as i32 && y < yres as i32 {
                let props = elem_props[(ty as usize) * 32usize + 1usize];
                if (props & TYPE_ENERGY_U32) == 0u32
                    && ty != PT_THDR_U32
                    && ty != PT_EMBR_U32
                    && ty != PT_FIGH_U32
                    && ty != PT_PLSM_U32
                {
                    let idx = ((y as u32) * xres + (x as u32)) as usize;
                    pmap_count[idx].fetch_add(1u32);
                }
            }
        }
    }
}

/// First half of `CheckStacking` (`Simulation.cpp:3620-3652`): marks pixels
/// whose stack count is excessive by adding `NPART` to the count.
///
/// `NPART` is `XRES * YRES` in the reference (`SimulationConfig.h:20`), so a
/// marked pixel always has `pmap_count >= NPART`. An `EHOLE` wall tolerates up
/// to 1500 stacked particles; everywhere else the threshold is 1500 or the
/// `rng.between(0, 1599) <= pmap_count + 100` roll.
#[cube(launch_unchecked)]
pub fn stacking_mark_kernel(
    pmap_count: &mut [Atomic<u32>],
    wall_bmap: &[u32],
    num_pixels: u32,
    xres: u32,
    xcells: u32,
    npart: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < num_pixels as usize {
        let count = pmap_count[idx].load();
        if count > 5u32 {
            let x = (idx as u32) % xres;
            let y = (idx as u32) / xres;
            let cell = (y / CELL) * xcells + (x / CELL);
            let wall = wall_bmap[cell as usize] & 0xFFu32;
            if wall == WL_EHOLE_U32 {
                if count > 1500u32 {
                    pmap_count[idx].store(count + npart);
                }
            } else {
                let roll =
                    spatial_hash(frame_seed ^ mix_index(idx as u32, 0x2545_F491u32)) % 1600u32;
                if count > 1500u32 || roll <= count + 100u32 {
                    pmap_count[idx].store(count + npart);
                }
            }
        }
    }
}

/// Second half of `CheckStacking` (`Simulation.cpp:3654-3681`): collapses a
/// marked stack.
///
/// The particle that owns the `pmap` entry at the pixel becomes `NBHL` with
/// `temp = MAX_TEMP` and `tmp = pmap_count - NPART` (capped at 51200); every
/// other particle at the pixel is killed, and the count is clamped back to
/// `NPART`. The reference walks particles in slot order and lets the first one
/// convert; the parallel pass uses the deterministic `pmap` owner instead.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn stacking_apply_kernel(
    part_ty: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    pmap: &mut [u32],
    pmap_count: &mut [Atomic<u32>],
    elem_props: &[u32],
    slot_count: u32,
    elem_rows: u32,
    xres: u32,
    yres: u32,
    npart: u32,
) {
    let i = ABSOLUTE_POS;
    if i < slot_count as usize {
        let ty = part_ty[i];
        if ty != PT_NONE_U32 && ty < elem_rows {
            let x = i32::cast_from(part_x[i] + 0.5f32);
            let y = i32::cast_from(part_y[i] + 0.5f32);
            if x >= 0i32 && y >= 0i32 && x < xres as i32 && y < yres as i32 {
                let props = elem_props[(ty as usize) * 32usize + 1usize];
                if (props & TYPE_ENERGY_U32) == 0u32 {
                    let idx = ((y as u32) * xres + (x as u32)) as usize;
                    let count = pmap_count[idx].load();
                    if count >= npart {
                        let mut owner = 0xFFFF_FFFFu32;
                        let packed = pmap[idx];
                        if packed != 0u32 {
                            owner = (packed >> 9u32) - 1u32;
                        }
                        if count > npart && owner == i as u32 {
                            // Stacking -> NBHL (`Simulation.cpp:3665-3671`).
                            part_ty[i] = PT_NBHL_U32;
                            part_temp[i] = MAX_TEMP;
                            let mut strength = count - npart;
                            if strength > 51200u32 {
                                strength = 51200u32;
                            }
                            part_tmp[i] = strength as i32;
                            pmap_count[idx].store(npart);
                            pmap[idx] = ((i as u32) + 1u32) << 9u32 | PT_NBHL_U32;
                        } else {
                            part_ty[i] = PT_NONE_U32;
                        }
                    }
                }
            }
        }
    }
}

impl SimBuffers {
    /// Recompute `pmap`, `photons` and the wall air/heat masks from the host shadows.
    pub fn rebuild_spatial_maps(&mut self) {
        // Host-side rebuild: callers that need the device state must pull it
        // back first (`sync_gpu_to_host`). Save restore, undo and Lua all write
        // the host shadows and then rebuild, so an implicit read-back here would
        // overwrite the freshly restored state with the stale device contents.
        self.pmap.fill(0);
        self.photons.fill(0);
        self.pmap_count.fill(0);

        let xres = self.config.xres as i32;
        let yres = self.config.yres as i32;
        let stride = self.config.xres as usize;

        for id in 0..self.used_slot_bound() {
            let ty = self.part_ty[id];
            if ty == PT_NONE {
                continue;
            }
            let x = self.part_x[id].round() as i32;
            let y = self.part_y[id].round() as i32;
            if x < 0 || y < 0 || x >= xres || y >= yres {
                continue;
            }
            let idx = y as usize * stride + x as usize;
            let packed = pmap_pack(id as u32, ty);
            if self.is_energy_type(ty) {
                self.photons[idx] = packed;
            } else {
                // `RecalcFreeParticles` (`Simulation.cpp:3375-3380`): a particle
                // inside INVIS/FILT must not overwrite the entry of the particle
                // it shares the pixel with, so those two only write an empty
                // pixel.
                if self.pmap[idx] == 0 || (ty != PT_INVIS && ty != PT_FILT) {
                    self.pmap[idx] = packed;
                }
                if ty != PT_THDR && ty != PT_EMBR && ty != PT_FIGH && ty != PT_PLSM {
                    self.pmap_count[idx] += 1;
                }
            }
        }

        self.refresh_wall_masks();
        self.mark_maps_dirty();
    }

    /// Recompute `block_air` / `block_airh` from `wall_bmap` + `wall_emap`.
    pub fn rebuild_wall_masks(&mut self) {
        self.refresh_wall_masks();
        self.mark_walls_dirty();
    }

    /// Recompute the derived wall masks without touching the dirty flags.
    ///
    /// `block_airh` stores TPT's `bmap_blockairh` value: `0x8` for a wall that
    /// blocks ambient heat, `0` otherwise. The thermal pass increments the low
    /// counter bits for particles that fail their conduction gate, which is why
    /// the kernels test bit 3 instead of the whole word.
    pub(super) fn refresh_wall_masks(&mut self) {
        let num_cells = self.config.num_cells();
        if self.block_air.len() != num_cells {
            self.block_air.resize(num_cells, 0);
            self.block_airh.resize(num_cells, 0);
        }
        for i in 0..num_cells {
            let wall = (self.wall_bmap[i] & 0xFF) as u8;
            let emap = (self.wall_emap[i] & 0xFF) as u8;
            self.block_air[i] = u32::from(is_wall_blocking_air(wall, emap));
            self.block_airh[i] = if is_wall_blocking_heat(wall, emap) {
                0x8
            } else {
                0
            };
        }
    }

    /// Device port of `Simulation::RecalcFreeParticles`
    /// (`Simulation.cpp:3347-3415`) plus `CheckStacking`
    /// (`Simulation.cpp:3620-3681`).
    ///
    /// Runs once per frame before the thermal and reaction passes:
    ///
    /// 1. [`life_decay_kernel`] decrements/kills `PROP_LIFE_DEC` particles,
    /// 2. [`clear_spatial_maps_kernel`] + [`scatter_spatial_maps_kernel`]
    ///    rebuild `pmap` / `photons` (with the INVIS/FILT preservation rule),
    /// 3. [`pmap_count_clear_kernel`] + [`pmap_count_kernel`] rebuild the stack
    ///    counts,
    /// 4. when `stacking_check` is set (the reference's 1-in-10 cadence),
    ///    [`stacking_mark_kernel`] + [`stacking_apply_kernel`] collapse extreme
    ///    stacks into `NBHL`.
    ///
    /// Nothing is read back; the device stays authoritative for the maps.
    pub fn recalc_free_particles_device(&mut self, frame_seed: u32, stacking_check: bool) {
        let capacity = self.capacity;
        let xres = self.config.xres;
        let yres = self.config.yres;
        let xcells = self.config.xcells();
        let ycells = self.config.ycells();
        if capacity == 0 || xres == 0 || yres == 0 || xcells == 0 || ycells == 0 {
            return;
        }
        let pixels = (xres as usize) * (yres as usize);
        let elem_rows = self.elem_props_host.len().min(PT_NUM);
        if elem_rows == 0 || self.pmap.len() < pixels || self.pmap_count.len() < pixels {
            return;
        }
        let elem_len = elem_rows * 32;

        let client = self.client.clone();
        let slots = self.used_slot_bound();
        let slot_cubes = (slots as u32).div_ceil(CUBE_DIM).max(1);
        let pixel_cubes = (pixels as u32).div_ceil(CUBE_DIM).max(1);

        unsafe {
            // 1. `do_life_dec`.
            life_decay_kernel::launch_unchecked(
                &client,
                CubeCount::Static(slot_cubes, 1, 1),
                CubeDim::new_1d(CUBE_DIM),
                BufferArg::from_raw_parts(self.gpu_part_ty.clone(), capacity),
                BufferArg::from_raw_parts(self.gpu_part_life.clone(), capacity),
                BufferArg::from_raw_parts(self.gpu_part_x.clone(), capacity),
                BufferArg::from_raw_parts(self.gpu_part_y.clone(), capacity),
                BufferArg::from_raw_parts(self.gpu_elem_props.clone(), elem_len),
                BufferArg::from_raw_parts(
                    self.gpu_wall_bmap.clone(),
                    xcells as usize * ycells as usize,
                ),
                BufferArg::from_raw_parts(
                    self.gpu_wall_emap.clone(),
                    xcells as usize * ycells as usize,
                ),
                slots as u32,
                elem_rows as u32,
                xres,
                yres,
                xcells,
                ycells,
            );

            // 2. `memset(pmap/photons)` + the particle scatter.
            clear_spatial_maps_kernel::launch_unchecked(
                &client,
                CubeCount::Static(pixel_cubes, 1, 1),
                CubeDim::new_1d(CUBE_DIM),
                BufferArg::from_raw_parts(self.gpu_pmap.clone(), pixels),
                BufferArg::from_raw_parts(self.gpu_photons.clone(), pixels),
                pixels as u32,
            );
            scatter_spatial_maps_kernel::launch_unchecked(
                &client,
                CubeCount::Static(slot_cubes, 1, 1),
                CubeDim::new_1d(CUBE_DIM),
                BufferArg::from_raw_parts(self.gpu_part_ty.clone(), capacity),
                BufferArg::from_raw_parts(self.gpu_part_x.clone(), capacity),
                BufferArg::from_raw_parts(self.gpu_part_y.clone(), capacity),
                BufferArg::from_raw_parts(self.gpu_elem_props.clone(), elem_len),
                BufferArg::from_raw_parts(self.gpu_pmap.clone(), pixels),
                BufferArg::from_raw_parts(self.gpu_photons.clone(), pixels),
                slots as u32,
                xres,
                yres,
            );

            // 3. `pmap_count`.
            pmap_count_clear_kernel::launch_unchecked(
                &client,
                CubeCount::Static(pixel_cubes, 1, 1),
                CubeDim::new_1d(CUBE_DIM),
                BufferArg::from_raw_parts(self.gpu_pmap_count.clone(), pixels),
                pixels as u32,
            );
            pmap_count_kernel::launch_unchecked(
                &client,
                CubeCount::Static(slot_cubes, 1, 1),
                CubeDim::new_1d(CUBE_DIM),
                BufferArg::from_raw_parts(self.gpu_part_ty.clone(), capacity),
                BufferArg::from_raw_parts(self.gpu_part_x.clone(), capacity),
                BufferArg::from_raw_parts(self.gpu_part_y.clone(), capacity),
                BufferArg::from_raw_parts(self.gpu_elem_props.clone(), elem_len),
                BufferArg::from_raw_parts(self.gpu_pmap_count.clone(), pixels),
                slots as u32,
                elem_rows as u32,
                xres,
                yres,
            );

            // 4. `CheckStacking` (1-in-10 frames in the reference).
            if stacking_check {
                let npart = xres.wrapping_mul(yres);
                stacking_mark_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(pixel_cubes, 1, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    BufferArg::from_raw_parts(self.gpu_pmap_count.clone(), pixels),
                    BufferArg::from_raw_parts(
                        self.gpu_wall_bmap.clone(),
                        xcells as usize * ycells as usize,
                    ),
                    pixels as u32,
                    xres,
                    xcells,
                    npart,
                    frame_seed,
                );
                stacking_apply_kernel::launch_unchecked(
                    &client,
                    CubeCount::Static(slot_cubes, 1, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    BufferArg::from_raw_parts(self.gpu_part_ty.clone(), capacity),
                    BufferArg::from_raw_parts(self.gpu_part_x.clone(), capacity),
                    BufferArg::from_raw_parts(self.gpu_part_y.clone(), capacity),
                    BufferArg::from_raw_parts(self.gpu_part_temp.clone(), capacity),
                    BufferArg::from_raw_parts(self.gpu_part_tmp.clone(), capacity),
                    BufferArg::from_raw_parts(self.gpu_pmap.clone(), pixels),
                    BufferArg::from_raw_parts(self.gpu_pmap_count.clone(), pixels),
                    BufferArg::from_raw_parts(self.gpu_elem_props.clone(), elem_len),
                    slots as u32,
                    elem_rows as u32,
                    xres,
                    yres,
                    npart,
                );
            }
        }

        self.gpu_particles_dirty = true;
        self.gpu_maps_dirty = true;
        self.gpu_dirty = true;
    }

    /// Particle occupying pixel `(x, y)` (matter first, then energy).
    pub fn particle_index_at(&self, x: i32, y: i32) -> Option<usize> {
        let pixel = self.pixel_index(x, y)?;
        if let Some(id) = pmap_id(self.pmap[pixel]) {
            return Some(id as usize);
        }
        pmap_id(self.photons[pixel]).map(|id| id as usize)
    }

    /// Copy of the particle occupying pixel `(x, y)`.
    pub fn particle_at(&self, x: i32, y: i32) -> Option<GpuParticle> {
        let id = self.particle_index_at(x, y)?;
        if id >= self.capacity {
            return None;
        }
        let p = self.export_particle(id);
        if p.ty == PT_NONE { None } else { Some(p) }
    }

    /// Wall id at pixel `(x, y)`, `None` for an empty cell.
    pub fn wall_at(&self, x: i32, y: i32) -> Option<u8> {
        let cell = self.cell_index(x, y)?;
        let wall = (self.wall_bmap[cell] & 0xFF) as u8;
        if wall == WL_ERASE { None } else { Some(wall) }
    }

    /// Wall electrical state at pixel `(x, y)`.
    pub fn emap_at(&self, x: i32, y: i32) -> u8 {
        match self.cell_index(x, y) {
            Some(cell) => (self.wall_emap[cell] & 0xFF) as u8,
            None => 0,
        }
    }

    /// Air pressure at pixel `(x, y)` (0 outside the canvas).
    pub fn pressure_at(&self, x: i32, y: i32) -> f32 {
        match self.cell_index(x, y) {
            Some(cell) => self.air_pv[cell],
            None => 0.0,
        }
    }

    /// Ambient heat at pixel `(x, y)` (ambient outside the canvas).
    pub fn ambient_heat_at(&self, x: i32, y: i32) -> f32 {
        match self.cell_index(x, y) {
            Some(cell) => self.air_hv[cell],
            None => self.config.ambient_air_temp,
        }
    }

    /// Air velocity at pixel `(x, y)`.
    pub fn velocity_at(&self, x: i32, y: i32) -> (f32, f32) {
        match self.cell_index(x, y) {
            Some(cell) => (self.air_vx[cell], self.air_vy[cell]),
            None => (0.0, 0.0),
        }
    }
}
