//! Particle decoration commands.
//!
//! The brush path runs on the device ([`crate::edit_kernel`]); only the
//! flood-fill decoration still walks the host shadow, and it reads it through
//! `ensure_host_current` because a flood fill needs the whole map anyway.

use std::collections::VecDeque;

use tpt_core::commands::DecoMode;

use crate::SimulationEngine;

impl SimulationEngine {
    /// Blend the decoration colour of the particles under a brush stroke.
    ///
    /// Port of `Simulation::ApplyDecorationPoint` behind the reference
    /// `DecorationTool`. The read-modify-write of `dcolour` happens on the
    /// device, so a stroke never pulls the particle SoA back to the host.
    pub(super) fn decorate_at(&mut self, points: &[(i32, i32)], colour: u32, mode: DecoMode) {
        if points.is_empty() {
            return;
        }
        crate::edit_kernel::apply_decoration(&mut self.bufs, points, colour, mode);
    }

    /// Flood-fill particle decoration colour.
    pub fn flood_decorate(&mut self, x: i32, y: i32, colour: u32) -> usize {
        let Some(start_slot) = self.bufs.particle_index_at(x, y) else {
            return 0;
        };
        let start_ty = self.bufs.part_ty[start_slot];
        let mut queue = VecDeque::new();
        queue.push_back((x, y));
        let mut visited = std::collections::HashSet::new();
        visited.insert((x, y));
        let mut count = 0;
        while let Some((px, py)) = queue.pop_front() {
            if let Some(slot) = self.bufs.particle_index_at(px, py)
                && self.bufs.part_ty[slot] == start_ty
            {
                self.bufs.part_dcolour[slot] = colour;
                self.bufs.mark_particle_slot_dirty(slot);
                count += 1;
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (px + dx, py + dy);
                    if nx >= 0
                        && ny >= 0
                        && nx < self.bufs.config.xres as i32
                        && ny < self.bufs.config.yres as i32
                        && visited.insert((nx, ny))
                    {
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
        if count > 0 {
            self.bufs.host_particles_dirty = true;
            self.bufs.host_dirty = true;
        }
        count
    }
}
