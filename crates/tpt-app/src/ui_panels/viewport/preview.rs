//! Preview overlays: brush footprint, line/rect gestures and rigid bodies.

use egui::{Color32, CornerRadius, Rect, Shape, Stroke, StrokeKind, vec2};
use tpt_core::commands::{BrushShape, RigidSpawnShape};

use super::ViewTransform;

/// Preview outline colour (line / rect / rigid body).
const PREVIEW: Color32 = Color32::from_rgb(120, 210, 255);
/// Dark halo drawn under preview outlines so they stay readable on any canvas.
const PREVIEW_HALO: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 150);

pub(super) fn draw_brush_outline(
    painter: &egui::Painter,
    view: ViewTransform,
    cx: i32,
    cy: i32,
    rx: i32,
    ry: i32,
    shape: BrushShape,
) {
    let centre = view.to_screen(cx as f32 + 0.5, cy as f32 + 0.5);
    let half_x = (rx.max(0) as f32 + 0.5) * view.scale;
    let half_y = (ry.max(0) as f32 + 0.5) * view.scale;

    match shape {
        BrushShape::Circle => {
            // A polygon keeps elliptical (non-square) brushes accurate.
            const SEGMENTS: usize = 48;
            let mut points = Vec::with_capacity(SEGMENTS);
            for index in 0..SEGMENTS {
                let angle = index as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
                points.push(centre + vec2(angle.cos() * half_x, angle.sin() * half_y));
            }
            painter.add(Shape::closed_line(
                points.clone(),
                Stroke::new(3.0, PREVIEW_HALO),
            ));
            painter.add(Shape::closed_line(points, Stroke::new(1.0, PREVIEW)));
        }
        BrushShape::Square => {
            let rect = Rect::from_center_size(centre, vec2(half_x * 2.0, half_y * 2.0));
            painter.rect_stroke(
                rect,
                CornerRadius::ZERO,
                Stroke::new(3.0, PREVIEW_HALO),
                StrokeKind::Middle,
            );
            painter.rect_stroke(
                rect,
                CornerRadius::ZERO,
                Stroke::new(1.0, PREVIEW),
                StrokeKind::Middle,
            );
        }
        BrushShape::Triangle => {
            let apex = centre + vec2(0.0, -half_y);
            let left = centre + vec2(-half_x, half_y);
            let right = centre + vec2(half_x, half_y);
            painter.add(Shape::closed_line(
                vec![apex, left, right],
                Stroke::new(3.0, PREVIEW_HALO),
            ));
            painter.add(Shape::closed_line(
                vec![apex, left, right],
                Stroke::new(1.0, PREVIEW),
            ));
        }
    }
    painter.rect_stroke(
        view.pixel_rect(cx, cy),
        CornerRadius::ZERO,
        Stroke::new(1.0, Color32::from_rgba_premultiplied(255, 210, 90, 220)),
        StrokeKind::Middle,
    );
}

/// Draws the Shift-line gesture preview.
pub(super) fn draw_line_preview(
    painter: &egui::Painter,
    view: ViewTransform,
    start: (i32, i32),
    end: (i32, i32),
    rx: i32,
    ry: i32,
    shape: BrushShape,
) {
    let from = view.to_screen(start.0 as f32 + 0.5, start.1 as f32 + 0.5);
    let to = view.to_screen(end.0 as f32 + 0.5, end.1 as f32 + 0.5);
    painter.line_segment([from, to], Stroke::new(3.0, PREVIEW_HALO));
    painter.line_segment([from, to], Stroke::new(1.0, PREVIEW));
    draw_brush_outline(painter, view, start.0, start.1, rx, ry, shape);
    draw_brush_outline(painter, view, end.0, end.1, rx, ry, shape);
}

/// Draws the Ctrl-rect gesture preview.
pub(super) fn draw_rect_preview(
    painter: &egui::Painter,
    view: ViewTransform,
    start: (i32, i32),
    end: (i32, i32),
) {
    let from = view.to_screen(start.0 as f32, start.1 as f32);
    let to = view.to_screen(end.0 as f32 + 1.0, end.1 as f32 + 1.0);
    let rect = Rect::from_two_pos(from, to);
    painter.rect_filled(
        rect,
        CornerRadius::ZERO,
        Color32::from_rgba_premultiplied(40, 90, 120, 40),
    );
    painter.rect_stroke(
        rect,
        CornerRadius::ZERO,
        Stroke::new(3.0, PREVIEW_HALO),
        StrokeKind::Middle,
    );
    painter.rect_stroke(
        rect,
        CornerRadius::ZERO,
        Stroke::new(1.0, PREVIEW),
        StrokeKind::Middle,
    );
}

/// Draws the footprint of a rigid-body spawn.
pub(super) fn draw_rigid_preview(
    painter: &egui::Painter,
    view: ViewTransform,
    cx: i32,
    cy: i32,
    rx: i32,
    ry: i32,
    shape: RigidSpawnShape,
) {
    let centre = view.to_screen(cx as f32 + 0.5, cy as f32 + 0.5);
    let half_x = (rx.max(0) as f32 + 0.5) * view.scale;
    let half_y = (ry.max(0) as f32 + 0.5) * view.scale;

    match shape {
        RigidSpawnShape::Box => {
            let rect = Rect::from_center_size(centre, vec2(half_x * 2.0, half_y * 2.0));
            painter.rect_filled(
                rect,
                CornerRadius::same(2),
                Color32::from_rgba_premultiplied(60, 120, 160, 36),
            );
            painter.rect_stroke(
                rect,
                CornerRadius::same(2),
                Stroke::new(2.0, Color32::from_rgb(120, 210, 255)),
                StrokeKind::Middle,
            );
        }
        RigidSpawnShape::Circle => {
            const SEGMENTS: usize = 48;
            let mut points = Vec::with_capacity(SEGMENTS);
            for index in 0..SEGMENTS {
                let angle = index as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
                points.push(centre + vec2(angle.cos() * half_x, angle.sin() * half_y));
            }
            painter.add(Shape::convex_polygon(
                points,
                Color32::from_rgba_premultiplied(60, 120, 160, 36),
                Stroke::new(2.0, Color32::from_rgb(120, 210, 255)),
            ));
        }
    }
}
