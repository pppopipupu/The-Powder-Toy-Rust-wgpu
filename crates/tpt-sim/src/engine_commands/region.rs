//! Region stamping: the placement half of copy / cut / stamp.
//!
//! Ports `Simulation::Load` (`Simulation.cpp:49-340`). The destination
//! rectangle is emptied before anything is created (`Simulation.cpp:63-103`),
//! every saved particle is recreated and patched with its saved fields
//! (`Simulation.cpp:110-260`), the wall cells follow (`Simulation.cpp:292-300`)
//! and the gated air grids and signs close the load
//! (`Simulation.cpp:277-321`).

use tpt_core::commands::RegionAir;
use tpt_core::constants::{CELL, PT_NONE};
use tpt_core::particle::GpuParticle;
use tpt_core::sign::Sign;

use crate::SimulationEngine;
use crate::buffers::CreationSource;

impl SimulationEngine {
    /// Stamp one captured region with its top-left corner at `(x, y)`.
    ///
    /// `particles` and `walls` use region-relative coordinates: a particle at
    /// `(0.0, 0.0)` lands on the destination anchor and a wall entry `(rx, ry)`
    /// covers the pixel `(x + rx, y + ry)`. `air` is empty when the region was
    /// captured with `IncludePressure` off; an empty payload leaves the
    /// destination air untouched, matching `Simulation.cpp:301-321`.
    // The saved region arrives field by field from the capture; a struct would
    // only rename the same list.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn place_region(
        &mut self,
        x: i32,
        y: i32,
        w: u32,
        h: u32,
        particles: &[GpuParticle],
        walls: &[u32],
        air: &RegionAir,
        signs: &[Sign],
    ) {
        if w == 0 || h == 0 {
            return;
        }
        // `Simulation::Load` makes sure every pixel it is about to fill is free
        // before it creates the saved particles (`Simulation.cpp:63-103,155`).
        // Clearing the whole rectangle up front makes the stamp atomic: the
        // pixels it covers show the region and nothing of the old canvas.
        self.clear_rect(x, y, w as i32, h as i32);

        let xres = self.bufs.config.xres as i32;
        let yres = self.bufs.config.yres as i32;
        self.bufs.reserve_particles(particles.len());
        for particle in particles {
            if particle.ty == PT_NONE {
                continue;
            }
            // `Simulation.cpp:120-121`: `int(floor(tempPart.x + 0.5f))`.
            let px = (x as f32 + particle.x + 0.5).floor() as i32;
            let py = (y as f32 + particle.y + 0.5).floor() as i32;
            if px < 0 || py < 0 || px >= xres || py >= yres {
                continue;
            }
            let Some(id) = self.bufs.create_part(
                px,
                py,
                particle.ty,
                CreationSource::SkipChecks,
                0,
                &self.registry,
            ) else {
                continue;
            };
            // `Simulation.cpp:163`: `parts[i] = tempPart`, i.e. every field the
            // save carried replaces the freshly created default.
            let bufs = &mut self.bufs;
            bufs.part_life[id] = particle.life;
            bufs.part_ctype[id] = particle.ctype;
            bufs.part_x[id] = x as f32 + particle.x;
            bufs.part_y[id] = y as f32 + particle.y;
            bufs.part_vx[id] = particle.vx;
            bufs.part_vy[id] = particle.vy;
            bufs.part_temp[id] = particle.temp;
            bufs.part_flags[id] = particle.flags;
            bufs.part_tmp[id] = particle.tmp;
            bufs.part_tmp2[id] = particle.tmp2;
            bufs.part_tmp3[id] = particle.tmp3;
            bufs.part_tmp4[id] = particle.tmp4;
            bufs.part_dcolour[id] = particle.dcolour;
            bufs.mark_particle_slot_dirty(id);
        }

        // `Simulation.cpp:292-300`: the saved block map wins where it has a
        // wall. The region payload carries no fan velocities, so `PlaceWalls`
        // semantics apply: a non-fan wall clears the cell's fan velocity.
        let xcells = self.bufs.config.xcells() as i32;
        let ycells = self.bufs.config.ycells() as i32;
        let mut cells_by_wall: Vec<(u32, Vec<(i32, i32)>)> = Vec::new();
        let pixel_w = w as i32;
        for ry in 0..h as i32 {
            for rx in 0..pixel_w {
                let Some(&wall) = walls.get((ry * pixel_w + rx) as usize) else {
                    continue;
                };
                if wall == 0 {
                    continue;
                }
                let cell = (
                    (x + rx).div_euclid(CELL as i32),
                    (y + ry).div_euclid(CELL as i32),
                );
                if cell.0 < 0 || cell.1 < 0 || cell.0 >= xcells || cell.1 >= ycells {
                    continue;
                }
                match cells_by_wall.iter_mut().find(|(id, _)| *id == wall) {
                    Some((_, cells)) => {
                        if !cells.contains(&cell) {
                            cells.push(cell);
                        }
                    }
                    None => cells_by_wall.push((wall, vec![cell])),
                }
            }
        }
        for (wall_id, cells) in cells_by_wall {
            self.place_walls(&cells, wall_id as u8, None);
        }

        // `Simulation.cpp:301-321`: the air fields are written only when the
        // save carries them, and the `includePressure` gate has already
        // produced an empty payload otherwise. Pixel-level alignment inside the
        // first cell is not preserved: the grids start at the cell containing
        // the anchor, like `Simulation::Load`'s `blockP`.
        let num_cells = self.bufs.config.num_cells();
        if !air.is_empty() && air.is_complete() {
            let origin_x = x.div_euclid(CELL as i32);
            let origin_y = y.div_euclid(CELL as i32);
            let mut touched = false;
            for cy in 0..air.hcells as i32 {
                for cx in 0..air.wcells as i32 {
                    let tx = origin_x + cx;
                    let ty = origin_y + cy;
                    if tx < 0 || ty < 0 || tx >= xcells || ty >= ycells {
                        continue;
                    }
                    let src = (cy * air.wcells as i32 + cx) as usize;
                    let dst = ty as usize * xcells as usize + tx as usize;
                    if dst >= num_cells {
                        continue;
                    }
                    self.bufs.air_pv[dst] = air.pv[src];
                    self.bufs.air_vx[dst] = air.vx[src];
                    self.bufs.air_vy[dst] = air.vy[src];
                    self.bufs.air_hv[dst] = air.hv[src];
                    touched = true;
                }
            }
            if touched {
                self.bufs.host_air_dirty = true;
                self.bufs.host_dirty = true;
            }
        }

        // `Simulation.cpp:277-291`: save order is preserved and `MAXSIGNS` is
        // enforced by `add_sign`. The reference appends unconditionally; the
        // port skips a sign whose anchor is already taken so copying a region
        // and pasting it back cannot duplicate its own signs.
        for sign in signs {
            if sign.text.is_empty() {
                continue;
            }
            let sx = x + sign.x;
            let sy = y + sign.y;
            if sx < 0 || sy < 0 || sx >= xres || sy >= yres {
                continue;
            }
            if self
                .signs
                .iter()
                .any(|existing| existing.x == sx && existing.y == sy)
            {
                continue;
            }
            self.add_sign(sx, sy, &sign.text, sign.ju);
        }
    }
}
