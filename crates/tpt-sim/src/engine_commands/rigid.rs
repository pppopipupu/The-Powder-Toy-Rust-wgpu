//! Rigid-body spawning commands.

use tpt_core::commands::RigidSpawnShape;
use tpt_core::constants::*;
use tpt_core::elements::ElementRegistry;
use tpt_core::particle::RigidBody2D;

use crate::SimulationEngine;

/// Builds a fresh rigid body for `elem_ty` seeded with the element's defaults.
#[allow(clippy::too_many_arguments)]
fn build_rigid_body_for_element(
    id: u32,
    x: f32,
    y: f32,
    half_w: i32,
    half_h: i32,
    shape: RigidSpawnShape,
    elem_ty: u32,
    mod_elements_enabled: bool,
    registry: &ElementRegistry,
) -> Option<RigidBody2D> {
    let ty = if (elem_ty as usize) < registry.elements.len() && registry.get(elem_ty).enabled {
        elem_ty
    } else {
        PT_STNE
    };
    if is_mod_element(ty) && !mod_elements_enabled {
        return None;
    }
    let def = registry.get(ty);
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
    let default_life = def.default_life;
    let default_ctype = def.default_ctype;
    let hw = half_w.max(1);
    let hh = half_h.max(1);

    let mut body = match shape {
        RigidSpawnShape::Box => RigidBody2D::new_box(id, x, y, hw, hh, ty, temp, unit_weight),
        RigidSpawnShape::Circle => {
            RigidBody2D::new_ellipse(id, x, y, hw, hh, ty, temp, unit_weight)
        }
        RigidSpawnShape::Triangle => {
            RigidBody2D::new_triangle(id, x, y, hw, hh, ty, temp, unit_weight)
        }
    };
    for pixel in &mut body.pixels {
        pixel.life = default_life;
        pixel.ctype = default_ctype;
    }
    Some(body)
}

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
        let id = self.bufs.next_rigid_id;
        let Some(body) = build_rigid_body_for_element(
            id,
            x,
            y,
            half_w,
            half_h,
            shape,
            elem_ty,
            self.bufs.config.mod_elements_enabled,
            &self.registry,
        ) else {
            return 0;
        };
        self.bufs.next_rigid_id = self.bufs.next_rigid_id.wrapping_add(1);
        self.bufs.rigid_bodies.push(body);
        // The body is rasterised into particle slots by the next
        // `step_rigid_bodies` pass, which syncs the device first and marks the
        // slots it touches; nothing here rewrites the particle SoA, so no
        // whole-pool upload is queued.
        id
    }
}

#[cfg(test)]
mod tests {
    use super::build_rigid_body_for_element;
    use tpt_core::commands::RigidSpawnShape;
    use tpt_core::constants::{PT_COAL, PT_GRPH, PT_ICEI, PT_WATR};
    use tpt_core::elements::ElementRegistry;

    #[test]
    fn rigid_pixels_inherit_element_default_life_and_ctype() {
        let registry = ElementRegistry::new_default();

        let coal = build_rigid_body_for_element(
            1,
            32.0,
            32.0,
            2,
            2,
            RigidSpawnShape::Box,
            PT_COAL,
            true,
            &registry,
        )
        .expect("coal body must spawn");
        assert!(coal.pixels.iter().all(|px| px.life == 110));

        let ice = build_rigid_body_for_element(
            2,
            32.0,
            32.0,
            2,
            2,
            RigidSpawnShape::Circle,
            PT_ICEI,
            true,
            &registry,
        )
        .expect("ice body must spawn");
        assert!(ice.pixels.iter().all(|px| px.ctype == PT_WATR));
    }

    #[test]
    fn disabled_mod_elements_are_refused_for_rigid_spawn() {
        let registry = ElementRegistry::new_default();
        assert!(
            build_rigid_body_for_element(
                1,
                32.0,
                32.0,
                2,
                2,
                RigidSpawnShape::Triangle,
                PT_GRPH,
                false,
                &registry,
            )
            .is_none()
        );
        assert!(
            build_rigid_body_for_element(
                1,
                32.0,
                32.0,
                2,
                2,
                RigidSpawnShape::Triangle,
                PT_GRPH,
                true,
                &registry,
            )
            .is_some()
        );
    }
}
