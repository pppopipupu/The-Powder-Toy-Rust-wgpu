//! Rigid-body spawning commands.

use tpt_core::commands::RigidSpawnShape;
use tpt_core::constants::*;
use tpt_core::particle::RigidBody2D;

use crate::SimulationEngine;

impl SimulationEngine {
    /// Spawn a Noita-style pixel-cluster rigid body and return its id.
    ///
    /// The body's pixels are rasterised into particle slots by
    /// [`crate::rigid_body::step_rigid_bodies`] on the next step.
    pub fn spawn_rigid_body(
        &mut self,
        x: f32,
        y: f32,
        half_w: i32,
        half_h: i32,
        shape: RigidSpawnShape,
        elem_ty: u32,
    ) -> u32 {
        let ty = if (elem_ty as usize) < self.registry.elements.len() {
            elem_ty
        } else {
            PT_STNE
        };
        let def = self.registry.get(ty);
        let temp = if def.default_temp.abs() > f32::EPSILON {
            def.default_temp
        } else {
            DEFAULT_TEMP_K
        };
        let unit_weight = if def.weight > 0 {
            def.weight as f32
        } else {
            1.0
        };

        let id = self.bufs.next_rigid_id;
        self.bufs.next_rigid_id = self.bufs.next_rigid_id.wrapping_add(1);
        let hw = half_w.max(1);
        let hh = half_h.max(1);

        let body = match shape {
            RigidSpawnShape::Box => RigidBody2D::new_box(id, x, y, hw, hh, ty, temp, unit_weight),
            RigidSpawnShape::Circle => {
                RigidBody2D::new_circle(id, x, y, hw.max(hh), ty, temp, unit_weight)
            }
        };
        self.bufs.rigid_bodies.push(body);
        // The body is rasterised into particle slots by the next
        // `step_rigid_bodies` pass, which syncs the device first and marks the
        // slots it touches; nothing here rewrites the particle SoA, so no
        // whole-pool upload is queued.
        id
    }
}
