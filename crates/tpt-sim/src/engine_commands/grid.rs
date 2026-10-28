//! Grid-level and Lua-driven clearing and reset commands.

use tpt_core::constants::*;
use tpt_core::particle::EdgeMode;

use crate::SimulationEngine;

impl SimulationEngine {
    /// Clear all particles and walls inside the pixel box (x, y, w, h).
    pub fn clear_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let xres = self.bufs.config.xres as i32;
        let yres = self.bufs.config.yres as i32;
        let x0 = x.clamp(0, xres);
        let y0 = y.clamp(0, yres);
        let x1 = (x + w).clamp(0, xres);
        let y1 = (y + h).clamp(0, yres);
        for py in y0..y1 {
            for px in x0..x1 {
                self.bufs.kill_at(px, py);
            }
        }
        let xcells = self.bufs.config.xcells() as i32;
        let ycells = self.bufs.config.ycells() as i32;
        let cx0 = (x0 / CELL as i32).clamp(0, xcells);
        let cy0 = (y0 / CELL as i32).clamp(0, ycells);
        let cx1 = ((x1 + CELL as i32 - 1) / CELL as i32).clamp(0, xcells);
        let cy1 = ((y1 + CELL as i32 - 1) / CELL as i32).clamp(0, ycells);
        for cy in cy0..cy1 {
            for cx in cx0..cx1 {
                let idx = cy as usize * xcells as usize + cx as usize;
                self.bufs.wall_bmap[idx] = 0;
                self.bufs.wall_emap[idx] = 0;
            }
        }
        self.bufs.rebuild_wall_masks();
        self.bufs.sync_host_to_gpu();
    }

    /// Change the canvas edge mode (`Simulation::SetEdgeMode`,
    /// `Simulation.cpp:765-799`).
    ///
    /// `EDGE_SOLID` writes `WL_WALL` onto the four coarse border
    /// rows/columns; `EDGE_VOID` and `EDGE_LOOP` leave them empty. The
    /// movement kernels read `config.edge_mode` directly, so the change takes
    /// effect on the next tick; the wall masks are rebuilt and re-uploaded
    /// here, exactly like the reference's `SetEdgeMode` call.
    ///
    /// [`tpt_core::particle::SimulationConfig::edge_mode`] is also reachable
    /// through [`SimulationEngine::config_mut`], but that path does not restamp
    /// the border ring — use this method for a live mode change.
    pub fn set_edge_mode(&mut self, mode: EdgeMode) {
        if self.bufs.config.edge_mode == mode {
            return;
        }
        self.bufs.config.edge_mode = mode;
        if self.bufs.apply_edge_mode() {
            self.bufs.sync_host_to_gpu();
        }
    }

    /// Reset particle temperatures to element defaults.
    pub fn reset_temp(&mut self, only_conductors: bool) {
        for slot in 0..self.bufs.capacity {
            let ty = self.bufs.part_ty[slot];
            if ty != PT_NONE {
                let props = self.registry.get(ty);
                if !only_conductors || (props.properties & PROP_CONDUCTS != 0) {
                    self.bufs.part_temp[slot] = props.default_temp;
                    self.bufs.mark_particle_slot_dirty(slot);
                }
            }
        }
        self.bufs.host_particles_dirty = true;
        self.bufs.host_dirty = true;
    }

    /// Reset air pressure in cell rectangle (x, y, w, h) to ambient/edge pressure.
    pub fn reset_pressure(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let xcells = self.bufs.config.xcells() as i32;
        let ycells = self.bufs.config.ycells() as i32;
        let x0 = x.clamp(0, xcells);
        let y0 = y.clamp(0, ycells);
        let x1 = (x + w).clamp(0, xcells);
        let y1 = (y + h).clamp(0, ycells);
        let edge_p = self.bufs.config.edge_pressure;
        for cy in y0..y1 {
            for cx in x0..x1 {
                let idx = cy as usize * xcells as usize + cx as usize;
                self.bufs.air_pv[idx] = edge_p;
            }
        }
        self.bufs.host_air_dirty = true;
        self.bufs.host_dirty = true;
    }

    /// Zero air velocity in cell rectangle (x, y, w, h).
    pub fn reset_velocity(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let xcells = self.bufs.config.xcells() as i32;
        let ycells = self.bufs.config.ycells() as i32;
        let x0 = x.clamp(0, xcells);
        let y0 = y.clamp(0, ycells);
        let x1 = (x + w).clamp(0, xcells);
        let y1 = (y + h).clamp(0, ycells);
        for cy in y0..y1 {
            for cx in x0..x1 {
                let idx = cy as usize * xcells as usize + cx as usize;
                self.bufs.air_vx[idx] = 0.0;
                self.bufs.air_vy[idx] = 0.0;
            }
        }
        self.bufs.host_air_dirty = true;
        self.bufs.host_dirty = true;
    }

    /// Clear electrical sparks and reset spark life across the simulation.
    pub fn reset_spark(&mut self) {
        self.bufs.wall_emap.fill(0);
        for slot in 0..self.bufs.capacity {
            if self.bufs.part_ty[slot] == PT_SPRK {
                let ctype = self.bufs.part_ctype[slot];
                let new_ty = if ctype != PT_NONE && (ctype as usize) < self.registry.elements.len()
                {
                    ctype
                } else {
                    PT_METL
                };
                self.bufs.part_ty[slot] = new_ty;
                self.bufs.part_life[slot] = 0;
                self.bufs.mark_particle_slot_dirty(slot);
            }
        }
        self.bufs.host_wall_state_dirty = true;
        self.bufs.host_particles_dirty = true;
        self.bufs.host_dirty = true;
    }
}
