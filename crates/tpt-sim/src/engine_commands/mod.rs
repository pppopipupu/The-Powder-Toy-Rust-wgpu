//! User-command application: brushes, flood fill, wall paint, simulation tools,
//! rigid-body spawning and per-particle property edits.
//!
//! Most entry points here mutate the host shadow of [`crate::SimBuffers`] and
//! rely on the dirty flags to get the change onto the device. The flood fill
//! and the property-tool writes resolve their pixels on the device instead
//! ([`crate::flood_kernel`]) and only set the device-dirty flags.

mod air;
mod deco;
mod fields;
mod grid;
mod particles;
mod region;
mod rigid;
mod tools;
mod walls;

use tpt_core::commands::SimCommand;

use crate::SimulationEngine;

/// Largest velocity a tool may write, matching the `MAX_VELOCITY` window the
/// movement kernel integrates within.
pub(super) const MAX_TOOL_VELOCITY: f32 = 16.0;

impl SimulationEngine {
    /// Apply a user command (brush stroke, wall paint, tool, rigid spawn, …).
    pub fn apply_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::ClearAll => self.clear(),
            SimCommand::ResizeCanvas { xres, yres } => self.resize_canvas(xres, yres),
            SimCommand::PlaceParticles {
                points,
                elem_ty,
                replace_mode,
                dcolour,
            } => self.place_particles(&points, elem_ty, replace_mode, dcolour),
            SimCommand::CreateLightning {
                x,
                y,
                radius_x,
                radius_y,
            } => self.create_lightning(x, y, radius_x, radius_y),
            SimCommand::FloodFillParticles { x, y, elem_ty } => self.flood_fill(x, y, elem_ty),
            SimCommand::PlaceWalls {
                cells,
                wall_id,
                fan_vel,
            } => self.place_walls(&cells, wall_id, fan_vel),
            SimCommand::ApplySimTool {
                points,
                tool,
                strength,
                dir,
                center,
                dragging,
            } => self.apply_sim_tool(&points, tool, strength, dir, center, dragging),
            SimCommand::SpawnRigidBody {
                x,
                y,
                half_w,
                half_h,
                shape,
                elem_ty,
            } => {
                self.spawn_rigid_body(x, y, half_w, half_h, shape, elem_ty);
            }
            SimCommand::SetParticleProperty {
                part_id,
                field,
                value,
            } => self.set_particle_property(part_id, &field, value),
            SimCommand::SetParticleFieldAt {
                points,
                field,
                value,
            } => self.set_particle_field_at(&points, field, value),
            SimCommand::DecorateAt {
                points,
                colour,
                mode,
            } => self.decorate_at(&points, colour, mode),
            SimCommand::SpawnParticle {
                x,
                y,
                elem_ty,
                vx,
                vy,
            } => self.spawn_particle(x, y, elem_ty, vx, vy),
            SimCommand::KillAt { x, y } => {
                self.bufs.kill_at(x, y);
            }
            SimCommand::FloodFillWalls {
                x,
                y,
                wall_id,
                replace_wall,
            } => {
                self.flood_walls(x, y, wall_id, replace_wall);
            }
            SimCommand::ClearRect { x, y, w, h } => self.clear_rect(x, y, w, h),
            SimCommand::PlaceRegion {
                x,
                y,
                w,
                h,
                particles,
                walls,
                air,
                signs,
            } => self.place_region(x, y, w, h, &particles, &walls, &air, &signs),
            SimCommand::FloodDecorate { x, y, colour } => {
                self.flood_decorate(x, y, colour);
            }
            SimCommand::ResetTemp { only_conductors } => self.reset_temp(only_conductors),
            SimCommand::ResetPressure { x, y, w, h } => self.reset_pressure(x, y, w, h),
            SimCommand::ResetVelocity { x, y, w, h } => self.reset_velocity(x, y, w, h),
            SimCommand::ResetSpark => self.reset_spark(),
            SimCommand::InvertAir => self.invert_air(),
            SimCommand::ResetAmbientHeat => self.reset_ambient_heat(),
            SimCommand::SparkAt { x, y } => self.spark_at(x, y),
            SimCommand::SetStickmanControl {
                player,
                comm,
                pcomm,
            } => {
                let p = usize::from(player.min(1)) * 2;
                self.bufs.stickman_control[p] = comm;
                self.bufs.stickman_control[p + 1] = pcomm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tools::{gravity_tool_cell, tool_gravity_mass};

    #[test]
    fn positive_and_negative_gravity_tools_write_signed_mass() {
        assert_eq!(tool_gravity_mass(1.0, true), 5.0);
        assert_eq!(tool_gravity_mass(1.0, false), -5.0);
        assert_eq!(tool_gravity_mass(0.5, true), 2.5);
        assert_eq!(tool_gravity_mass(0.5, false), -2.5);
    }

    #[test]
    fn gravity_tool_cell_matches_pixel_division_by_cell() {
        // 612x384 canvas -> 153x96 cells.
        assert_eq!(gravity_tool_cell(0, 0, 612, 384, 153, 96), Some(0));
        assert_eq!(gravity_tool_cell(3, 0, 612, 384, 153, 96), Some(0));
        assert_eq!(gravity_tool_cell(4, 0, 612, 384, 153, 96), Some(1));
        assert_eq!(gravity_tool_cell(7, 5, 612, 384, 153, 96), Some(153 + 1));
        assert_eq!(gravity_tool_cell(-1, 0, 612, 384, 153, 96), None);
        assert_eq!(gravity_tool_cell(0, -1, 612, 384, 153, 96), None);
        assert_eq!(gravity_tool_cell(612, 0, 612, 384, 153, 96), None);
        assert_eq!(gravity_tool_cell(0, 384, 612, 384, 153, 96), None);
    }
}
