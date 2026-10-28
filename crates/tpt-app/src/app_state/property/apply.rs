//! Property-tool commands: the brush stroke and the `DrawFill` flood.
//!
//! Both actions are resolved on the device ([`tpt_sim::flood_kernel`]): a
//! stroke uploads its packed pixel list, the fill uploads only the seed pixel,
//! and the kernels resolve the particles and rewrite the field in place. This
//! module only computes the brush footprint and maps the pref pair onto the
//! kernel's field id and value.

use tpt_core::commands::BrushShape;
use tpt_sim::flood_kernel::{self, PropertyField, PropertyValue as KernelValue};

use crate::app_state::AppState;

use super::kind::{PropertyKind, PropertyValue};

/// Pixel list a property-tool stroke covers.
///
/// Mirrors `PropertyTool::Draw` / `DrawLine` / `DrawRect`: a rectangle fills the
/// span, a line (or a drag) stamps the brush along the path, and a click is a
/// single brush stamp. `PropertyTool::DrawFill` does not come through here: the
/// fill mode floods from the press position.
pub fn property_stroke_points(
    p0: (i32, i32),
    p1: (i32, i32),
    rx: i32,
    ry: i32,
    shape: BrushShape,
    is_line: bool,
    is_rect: bool,
) -> Vec<(i32, i32)> {
    if is_rect {
        crate::brushes::generate_rect_points(p0.0, p0.1, p1.0, p1.1)
    } else if is_line || p0 != p1 {
        crate::brushes::generate_line_points(p0.0, p0.1, p1.0, p1.1, rx, ry, shape)
    } else {
        crate::brushes::generate_brush_points(p1.0, p1.1, rx, ry, shape)
    }
}

/// Kernel slot a [`PropertyKind`] writes.
///
/// The kernel ids are the same integers [`PropertyKind::index`] stores in
/// `Prop.Type`, so the mapping only has to name the variants.
pub(crate) fn kernel_field(kind: PropertyKind) -> PropertyField {
    match kind {
        PropertyKind::Type => PropertyField::Type,
        PropertyKind::Life => PropertyField::Life,
        PropertyKind::Ctype => PropertyField::Ctype,
        PropertyKind::Vx => PropertyField::Vx,
        PropertyKind::Vy => PropertyField::Vy,
        PropertyKind::Temp => PropertyField::Temp,
        PropertyKind::Tmp => PropertyField::Tmp,
        PropertyKind::Tmp2 => PropertyField::Tmp2,
        PropertyKind::Tmp3 => PropertyField::Tmp3,
        PropertyKind::Tmp4 => PropertyField::Tmp4,
        PropertyKind::Dcolour => PropertyField::Dcolour,
    }
}

/// Kernel value for a parsed pref value, keeping integer/float apart.
pub(crate) fn kernel_value(value: PropertyValue) -> KernelValue {
    match value {
        PropertyValue::Int(value) => KernelValue::Int(value),
        PropertyValue::Float(value) => KernelValue::Float(value),
    }
}

impl AppState {
    /// Applies one property-tool stroke with the configured field and value.
    ///
    /// `p0` is the previous cursor sample, `p1` the current one; `is_line` and
    /// `is_rect` select the reference `DrawLine` / `DrawRect` shapes.
    ///
    /// With [`PropertyWindow::fill`](super::window::PropertyWindow::fill) on, the stroke is replaced by
    /// `PropertyTool::DrawFill`: the press sample floods the connected particles
    /// and the rest of the gesture is ignored.
    pub fn apply_property_stroke(
        &mut self,
        p0: (i32, i32),
        p1: (i32, i32),
        is_line: bool,
        is_rect: bool,
    ) {
        if self.property_window.fill {
            // `DrawFill` takes a single position, so only the press sample (a
            // zero-length, unarmed stroke) floods.
            if p0 == p1 && !is_line && !is_rect {
                self.apply_property_fill(p1);
            }
            return;
        }

        let points = property_stroke_points(
            p0,
            p1,
            self.brush_rx,
            self.brush_ry,
            self.brush_shape,
            is_line,
            is_rect,
        );
        if points.is_empty() {
            return;
        }

        let kind = PropertyKind::from_index(self.prefs.prop_field);
        let value = super::kind::value_from_prefs(kind, &self.prefs);

        // The kernel resolves the brushed slots on the device and writes the
        // field in place; the host only uploads the packed pixel list.
        flood_kernel::set_property_stroke(
            &mut self.engine.bufs,
            &points,
            kernel_field(kind),
            kernel_value(value),
        );
    }

    /// `PropertyTool::DrawFill` -> `Simulation::flood_prop`.
    ///
    /// Floods the 4-neighbour region of particles that share the clicked
    /// particle's element type and its current value of the configured
    /// property, then writes the configured value to every particle of that
    /// region on the device. Returns the number of particles rewritten (`0`
    /// when the pixel holds no particle, or when they already hold the target
    /// value).
    pub fn apply_property_fill(&mut self, at: (i32, i32)) -> usize {
        let kind = PropertyKind::from_index(self.prefs.prop_field);
        let value = super::kind::value_from_prefs(kind, &self.prefs);
        flood_kernel::flood_property(
            &mut self.engine.bufs,
            at.0,
            at.1,
            kernel_field(kind),
            kernel_value(value),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stroke_points_follow_the_reference_draw_shapes() {
        let click = property_stroke_points((4, 4), (4, 4), 0, 0, BrushShape::Circle, false, false);
        assert_eq!(click, vec![(4, 4)]);

        let rect = property_stroke_points((0, 0), (1, 1), 5, 5, BrushShape::Circle, false, true);
        assert_eq!(rect, vec![(0, 1), (1, 1), (0, 0), (1, 0)]);

        let line = property_stroke_points((0, 0), (2, 0), 0, 0, BrushShape::Circle, true, false);
        assert_eq!(line, vec![(0, 0), (1, 0), (2, 0)]);

        // A drag interpolates even when the caller did not arm a line.
        let drag = property_stroke_points((0, 0), (2, 0), 0, 0, BrushShape::Circle, false, false);
        assert_eq!(drag, vec![(0, 0), (1, 0), (2, 0)]);
    }

    #[test]
    fn every_property_kind_has_a_kernel_slot() {
        for kind in PropertyKind::ALL {
            // The kernel ids match the pref index the app stores, so the two
            // crates only share one integer contract.
            assert_eq!(kernel_field(kind).id(), kind.index(), "{kind:?}");
        }
    }

    #[test]
    fn kernel_values_keep_integers_and_floats_apart() {
        assert_eq!(kernel_value(PropertyValue::Int(-7)), KernelValue::Int(-7));
        assert_eq!(
            kernel_value(PropertyValue::Float(12.5)),
            KernelValue::Float(12.5)
        );
    }
}
