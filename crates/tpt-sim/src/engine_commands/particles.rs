//! Particle placement, flood-fill and single-particle spawning commands.

use tpt_core::constants::*;

use crate::SimulationEngine;
use crate::buffers::CreationSource;

use super::MAX_TOOL_VELOCITY;

impl SimulationEngine {
    /// Paint (or erase, with `elem_ty == PT_NONE`) the given pixels.
    pub(super) fn place_particles(
        &mut self,
        points: &[(i32, i32)],
        elem_ty: u32,
        replace_mode: bool,
        dcolour: u32,
    ) {
        if points.is_empty() {
            return;
        }
        // No device read here: the mouse-down undo snapshot refreshed the host
        // maps and SoA, and re-reading the brush rectangle every frame was the
        // expensive part of a stroke. The cached shadow is only used to decide
        // whether a pixel is already occupied; the device map is rebuilt by the
        // movement pass every frame.
        //
        // Reserve the worst-case slots for this segment now, so allocation inside
        // the loop does not grow (and thereby read back) the pool mid-stroke.
        self.bufs.reserve_particles(points.len());

        // `PT_NONE` is the eraser: kill whatever sits under the brush.
        if elem_ty == PT_NONE {
            for &(x, y) in points {
                self.bufs.kill_at(x, y);
            }
            self.bufs.sync_host_to_gpu();
            return;
        }
        // `Editing.cpp:750-760` handles a bolt before the brush loop and never
        // lets it reach it: one bolt per call, powered by the brush radius.
        if elem_ty == PT_LIGH {
            return;
        }
        if elem_ty as usize >= self.registry.elements.len() {
            return;
        }
        // `CreatePartFlags` (`Editing.cpp:356-404`): replace mode swaps the
        // occupant in place and skips the brush checks, everything else goes
        // through `create_part(-2, ...)`.
        let source = if replace_mode {
            CreationSource::Replace
        } else {
            CreationSource::Brush
        };
        // `CreateParts` (`Editing.cpp:740-774`) packs the brush radius into the
        // creation value: `radiusX + radiusY` for a bolt's power and
        // `radiusX*4 + radiusY*4 + 7` for a coil's `tmp`. The stroke footprint
        // is the brush stamp, so its bounding box is the radius.
        let mut v = 0i32;
        if matches!(elem_ty, PT_LIGH | PT_TESC) {
            let (mut x0, mut x1) = (i32::MAX, i32::MIN);
            let (mut y0, mut y1) = (i32::MAX, i32::MIN);
            for &(x, y) in points {
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
            let rx = (x1 - x0) / 2;
            let ry = (y1 - y0) / 2;
            v = if elem_ty == PT_LIGH {
                (rx + ry).min(55)
            } else {
                (rx * 4 + ry * 4 + 7).min(300)
            };
        }
        let mut emplaced = 0i32;
        for &(x, y) in points {
            if let Some(id) = self
                .bufs
                .create_part(x, y, elem_ty, source, v, &self.registry)
            {
                if dcolour != 0 {
                    self.bufs.part_dcolour[id] = dcolour;
                }
                if elem_ty == PT_EMP {
                    emplaced += 1;
                }
            }
        }
        if emplaced > 0 {
            // `SPRK.cpp:281-284`: every triggered `EMP` adds 3, capped at 40.
            self.bufs.emp_decor = (self.bufs.emp_decor + 3 * emplaced).min(40);
        }
        // Push only the slots (and the pixel map words) this segment changed.
        self.bufs.sync_host_to_gpu();
    }

    /// Fire one lightning bolt (`Editing.cpp:750-760`).
    ///
    /// The bolt power is `radiusX + radiusY` clamped to TPT's 55, and the
    /// simulation refuses the next bolt for `power / 4` ticks.
    pub(super) fn create_lightning(&mut self, x: i32, y: i32, radius_x: i32, radius_y: i32) {
        let power = (radius_x + radius_y).min(55);
        if self.bufs.sim_tick < self.lightning_recreate {
            return;
        }
        self.lightning_recreate = self.bufs.sim_tick + (power as u64 / 4).max(1);
        if self.bufs.gpu_dirty {
            self.bufs.read_map_rect(x, y, x, y);
        }
        // The create hook reads the Newtonian field through the host shadow,
        // which the per-tick path never refreshes.
        if self.bufs.config.newton_grav {
            self.bufs.sync_gravity_gpu_to_host();
        }
        self.bufs
            .create_part(x, y, PT_LIGH, CreationSource::Brush, power, &self.registry);
        self.bufs.sync_host_to_gpu();
    }

    /// Flood-fill the empty region containing `(x, y)` with `elem_ty`.
    ///
    /// `Simulation::FloodParts` walks the pixel map with a scanline flood and
    /// calls `create_part(-1, ...)` per pixel. The walk and the creations run
    /// on the device (see [`crate::flood_kernel`]); the host only validates the
    /// element and the seed pixel, so no part of the map or the pool is copied
    /// back.
    pub(super) fn flood_fill(&mut self, x: i32, y: i32, elem_ty: u32) {
        if elem_ty == PT_NONE || elem_ty as usize >= self.registry.elements.len() {
            return;
        }
        // The device flood materialises through the creation queue, not
        // `create_part`, so the mod switch has to refuse the fill here.
        if is_mod_element(elem_ty) && !self.bufs.config.mod_elements_enabled {
            return;
        }
        crate::flood_kernel::flood_fill_device(&mut self.bufs, x, y, elem_ty);
    }

    /// Create exactly one particle, optionally with an initial velocity.
    pub(super) fn spawn_particle(&mut self, x: i32, y: i32, elem_ty: u32, vx: f32, vy: f32) {
        if elem_ty == PT_NONE || elem_ty as usize >= self.registry.elements.len() {
            return;
        }
        // The spawn tool allocates directly instead of going through
        // `create_part`, so it repeats the mod switch's refusal.
        if is_mod_element(elem_ty) && !self.bufs.config.mod_elements_enabled {
            return;
        }
        if self.bufs.gpu_dirty {
            self.bufs.read_map_rect(x, y, x, y);
        }
        // `create_part` semantics: one particle, replacing the occupant.
        self.bufs.kill_at(x, y);
        let Some(id) = self.bufs.allocate_particle(x, y, elem_ty, &self.registry) else {
            return;
        };
        self.bufs.part_vx[id] = vx.clamp(-MAX_TOOL_VELOCITY, MAX_TOOL_VELOCITY);
        self.bufs.part_vy[id] = vy.clamp(-MAX_TOOL_VELOCITY, MAX_TOOL_VELOCITY);
    }
}
