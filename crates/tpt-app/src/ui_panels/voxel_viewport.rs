//! 3D viewport: the rendered voxel image, orbit/pan/zoom camera and painting.
//!
//! The image is the `view3d_tex_id` texture produced by
//! [`crate::render3d::Renderer3d`] in `app_window::present_frame`; this module
//! never touches `wgpu` itself. Mouse rays go through the same projection the
//! renderer uses, so a click paints where the crosshair sits.

use std::time::{Duration, Instant};

use egui::{
    Align2, Color32, FontId, Key, MouseWheelUnit, PointerButton, Pos2, Rect, Sense, Stroke,
    TextureId, Ui, pos2, vec2,
};
use glam::{Vec3, Vec4};
use tpt_core::voxel3d::{Brush3d, BrushShape3d};

use crate::app_state::{AppState, VoxelDrawMode, VoxelTool};
use crate::i18n::{t, t_fmt};
use crate::render3d::Camera;

/// Smallest camera distance in voxels.
const MIN_DISTANCE: f32 = 8.0;
/// Largest camera distance in voxels.
///
/// The pick kernel walks at most 1,024 voxels (`PICK_STEPS` half-voxel steps),
/// so a much larger orbit distance would make clicks miss the world entirely.
/// 900 still frames the complete 256³ world.
const MAX_DISTANCE: f32 = 900.0;
/// Orbit radians per dragged pixel.
const ORBIT_SPEED: f32 = 0.008;
/// Pan voxels per dragged pixel, per voxel of camera distance.
const PAN_SPEED: f32 = 0.0022;
/// Keyboard pan speed as a fraction of the camera distance per second.
const KEY_PAN_SPEED: f32 = 0.8;
/// egui reports wheel notches as lines; one line is this many scroll points.
const WHEEL_LINE_POINTS: f32 = 40.0;
/// Zoom exponent per scroll point. One notch is about a 10% distance change.
const WHEEL_ZOOM_RATE: f32 = 0.0025;
/// Minimum interval between hover picks, which perform a tiny GPU read-back.
const HOVER_PICK_INTERVAL: Duration = Duration::from_millis(50);

/// Ray from the camera eye through a normalized device coordinate.
///
/// NDC follows the renderer's convention: `x = -1` is the left edge, `y = -1`
/// the bottom edge and depth `0..1`. The returned direction is normalized and
/// points from [`Camera::eye`] into the scene, which is exactly what
/// `Voxel3dEngine::pick` expects.
pub(crate) fn camera_ray(camera: &Camera, aspect: f32, ndc: [f32; 2]) -> ([f32; 3], [f32; 3]) {
    let inverse = camera.view_proj(aspect).inverse();
    let unproject = |depth: f32| {
        let clip = inverse * Vec4::new(ndc[0], ndc[1], depth, 1.0);
        if clip.w.abs() <= f32::EPSILON {
            None
        } else {
            Some(clip.truncate() / clip.w)
        }
    };
    let near = unproject(0.0);
    let far = unproject(1.0);
    let eye = camera.eye();
    let direction = match (near, far) {
        (Some(near), Some(far)) => (far - near).normalize_or(camera.forward()),
        _ => camera.forward(),
    };
    (eye.to_array(), direction.to_array())
}

/// Projects a world point into viewport pixels, or `None` when it is behind
/// the camera.
pub(crate) fn world_to_screen(
    camera: &Camera,
    aspect: f32,
    rect: Rect,
    point: Vec3,
) -> Option<Pos2> {
    let clip = camera.view_proj(aspect) * point.extend(1.0);
    if clip.w <= 1.0e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(pos2(
        rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(),
        rect.top() + (0.5 - ndc.y * 0.5) * rect.height(),
    ))
}

/// Screen position to NDC, clamped to the viewport edges.
fn ndc_of(rect: Rect, pos: Pos2) -> [f32; 2] {
    let x = ((pos.x - rect.left()) / rect.width().max(1.0)) * 2.0 - 1.0;
    let y = 1.0 - ((pos.y - rect.top()) / rect.height().max(1.0)) * 2.0;
    [x.clamp(-1.0, 1.0), y.clamp(-1.0, 1.0)]
}

/// Centre of a picked voxel.
fn hit_center(hit: tpt_sim::voxel3d::pick::PickHit) -> [f32; 3] {
    [
        hit.voxel[0] as f32 + 0.5,
        hit.voxel[1] as f32 + 0.5,
        hit.voxel[2] as f32 + 0.5,
    ]
}

/// First intersection of a ray with the 256^3 world box.
///
/// The default void scene has no voxels, so `Voxel3dEngine::pick` returns
/// `None`; painting still needs a valid point inside the box. Returns the
/// entry point and the outward face normal.
fn world_box_entry(origin: [f32; 3], direction: [f32; 3]) -> Option<([f32; 3], [i32; 3])> {
    let dir = Vec3::from_array(direction);
    let mut t_near = 0.0f32;
    let mut t_far = f32::INFINITY;
    let mut normal = [0i32; 3];
    for axis in 0..3 {
        let o = origin[axis];
        let d = dir[axis];
        if d.abs() < 1.0e-8 {
            if o < 0.0 || o > 256.0 {
                return None;
            }
            continue;
        }
        let inv = 1.0 / d;
        let mut t1 = (0.0 - o) * inv;
        let mut t2 = (256.0 - o) * inv;
        let mut near_normal = -1i32;
        if t1 > t2 {
            std::mem::swap(&mut t1, &mut t2);
            near_normal = 1i32;
        }
        if t1 > t_near {
            t_near = t1;
            normal = [0i32; 3];
            normal[axis] = near_normal;
        }
        if t2 < t_far {
            t_far = t2;
        }
        if t_near > t_far {
            return None;
        }
    }
    if t_far < 0.0 {
        return None;
    }
    let point = Vec3::from_array(origin) + dir * t_near;
    Some((point.to_array(), normal))
}

/// Point inside the world box for a brush sphere.
fn brush_entry_center(point: [f32; 3], direction: [f32; 3], radius: f32) -> [f32; 3] {
    let dir = Vec3::from_array(direction);
    let mut center = Vec3::from_array(point) + dir * (radius * 0.5 + 1.0);
    let low = radius.max(0.5);
    let high = 256.0 - low;
    center = center.clamp(Vec3::splat(low), Vec3::splat(high.max(low)));
    center.to_array()
}

/// Draws the translucent world box around the default void scene.
fn draw_world_box(painter: &egui::Painter, camera: &Camera, aspect: f32, rect: Rect) {
    let corners = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(256.0, 0.0, 0.0),
        Vec3::new(256.0, 0.0, 256.0),
        Vec3::new(0.0, 0.0, 256.0),
        Vec3::new(0.0, 256.0, 0.0),
        Vec3::new(256.0, 256.0, 0.0),
        Vec3::new(256.0, 256.0, 256.0),
        Vec3::new(0.0, 256.0, 256.0),
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    let projected: Vec<Option<Pos2>> = corners
        .iter()
        .map(|corner| world_to_screen(camera, aspect, rect, *corner))
        .collect();
    let stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(120, 190, 255, 110));
    for (a, b) in edges {
        if let (Some(from), Some(to)) = (projected[a], projected[b]) {
            painter.line_segment([from, to], stroke);
        }
    }
}

/// Wheel delta in egui scroll points, normalising line/page units.
///
/// `Event::MouseWheel` reports mouse notches as lines on Windows. Treating a
/// line as one point made zoom 40 times weaker than intended.
fn wheel_points(ctx: &egui::Context) -> f32 {
    ctx.input(|input| {
        input
            .events
            .iter()
            .filter_map(|event| match event {
                egui::Event::MouseWheel { unit, delta, .. } => Some(match unit {
                    MouseWheelUnit::Line => delta.y * WHEEL_LINE_POINTS,
                    MouseWheelUnit::Point => delta.y,
                    MouseWheelUnit::Page => delta.y * WHEEL_LINE_POINTS * 10.0,
                }),
                _ => None,
            })
            .sum()
    })
}

/// Draw mode for this press: Ctrl = rectangle, Shift = line, otherwise the
/// palette selection. This mirrors TPT `GameView::UpdateDrawMode`.
fn effective_draw_mode(mods: egui::Modifiers, selected: VoxelDrawMode) -> VoxelDrawMode {
    if mods.ctrl || mods.command {
        VoxelDrawMode::Rect
    } else if mods.shift {
        VoxelDrawMode::Line
    } else {
        selected
    }
}

/// Centre of the brush at the current pointer.
///
/// A pick is a tiny GPU read-back, so hover picks are throttled; the cached hit
/// is refreshed on pointer movement or every [`HOVER_PICK_INTERVAL`]. The empty
/// world falls back to the CPU ray/box intersection.
fn hover_brush_center(
    state: &mut AppState,
    origin: [f32; 3],
    direction: [f32; 3],
    pointer: Pos2,
    now: Instant,
    force: bool,
) -> Option<[f32; 3]> {
    let moved = state.voxel_hover_pointer.is_none_or(|previous| {
        (previous[0] - pointer.x).abs() > 0.5 || (previous[1] - pointer.y).abs() > 0.5
    });
    if force || moved || now.duration_since(state.voxel_hover_at) >= HOVER_PICK_INTERVAL {
        let hit = state.pick_voxel(origin, direction);
        state.voxel_hover_pick = hit;
        state.voxel_hover_pointer = Some([pointer.x, pointer.y]);
        state.voxel_hover_at = now;
        if force {
            state.voxel_last_pick = hit;
        }
    }

    if let Some(hit) = state.voxel_hover_pick {
        return Some(hit_center(hit));
    }
    world_box_entry(origin, direction)
        .map(|(point, _)| brush_entry_center(point, direction, state.voxel_brush_radius()))
}

/// Projects and draws one line segment, skipping segments behind the eye.
fn draw_world_segment(
    painter: &egui::Painter,
    camera: &Camera,
    aspect: f32,
    rect: Rect,
    from: Vec3,
    to: Vec3,
    stroke: Stroke,
) {
    if let (Some(from), Some(to)) = (
        world_to_screen(camera, aspect, rect, from),
        world_to_screen(camera, aspect, rect, to),
    ) {
        painter.line_segment([from, to], stroke);
    }
}

/// Draws the 3-axis brush cage at `center`.
#[allow(clippy::too_many_arguments)]
fn draw_brush_cage(
    painter: &egui::Painter,
    camera: &Camera,
    aspect: f32,
    rect: Rect,
    center: [f32; 3],
    radii: [f32; 3],
    shape: BrushShape3d,
    stroke: Stroke,
) {
    let center = Vec3::from_array(center);
    match shape {
        BrushShape3d::Ellipsoid => {
            let segments = 48usize;
            for plane in 0..3 {
                let mut previous = None;
                for step in 0..=segments {
                    let angle = std::f32::consts::TAU * step as f32 / segments as f32;
                    let (sin, cos) = angle.sin_cos();
                    let offset = match plane {
                        0 => Vec3::new(radii[0] * cos, radii[1] * sin, 0.0),
                        1 => Vec3::new(radii[0] * cos, 0.0, radii[2] * sin),
                        _ => Vec3::new(0.0, radii[1] * cos, radii[2] * sin),
                    };
                    let point = center + offset;
                    if let Some(previous) = previous {
                        draw_world_segment(painter, camera, aspect, rect, previous, point, stroke);
                    }
                    previous = Some(point);
                }
            }
        }
        BrushShape3d::Cuboid => draw_cuboid_cage(
            painter,
            camera,
            aspect,
            rect,
            center,
            Vec3::from_array(radii),
            stroke,
        ),
        BrushShape3d::Prism => {
            let triangles = [-radii[2], radii[2]].map(|z| {
                [
                    Vec3::new(-radii[0], -radii[1], z),
                    Vec3::new(radii[0], -radii[1], z),
                    Vec3::new(0.0, radii[1], z),
                ]
            });
            for triangle in triangles.iter() {
                for index in 0..3 {
                    draw_world_segment(
                        painter,
                        camera,
                        aspect,
                        rect,
                        center + triangle[index],
                        center + triangle[(index + 1) % 3],
                        stroke,
                    );
                }
            }
            for (a, b) in triangles[0].iter().zip(triangles[1].iter()) {
                draw_world_segment(
                    painter,
                    camera,
                    aspect,
                    rect,
                    center + *a,
                    center + *b,
                    stroke,
                );
            }
        }
    }
}

/// Draws the 12 edges of an axis-aligned cuboid.
fn draw_cuboid_cage(
    painter: &egui::Painter,
    camera: &Camera,
    aspect: f32,
    rect: Rect,
    center: Vec3,
    half: Vec3,
    stroke: Stroke,
) {
    let mut corners = [Vec3::ZERO; 8];
    for (index, corner) in corners.iter_mut().enumerate() {
        let sign = Vec3::new(
            if index & 1 == 0 { -1.0 } else { 1.0 },
            if index & 2 == 0 { -1.0 } else { 1.0 },
            if index & 4 == 0 { -1.0 } else { 1.0 },
        );
        *corner = center + sign * half;
    }
    for a in 0..8usize {
        for b in (a + 1)..8usize {
            if (a ^ b).count_ones() == 1 {
                draw_world_segment(
                    painter, camera, aspect, rect, corners[a], corners[b], stroke,
                );
            }
        }
    }
}

/// Brush used to bound a swept stroke for the undo snapshot.
fn segment_capture_brush(state: &AppState, from: [f32; 3], to: [f32; 3]) -> Brush3d {
    let mut half = [0.0f32; 3];
    let mut center = [0.0f32; 3];
    for axis in 0..3 {
        let low = from[axis].min(to[axis]);
        let high = from[axis].max(to[axis]);
        half[axis] = (high - low) * 0.5 + state.voxel_brush_radii[axis].max(1) as f32;
        center[axis] = (low + high) * 0.5;
    }
    Brush3d::with_shape(center, half, BrushShape3d::Cuboid)
}

/// Draws the 3D image and handles the camera and paint input.
pub(super) fn draw_voxel_viewport(ui: &mut Ui, state: &mut AppState, view3d_tex_id: TextureId) {
    let available = ui.available_size();
    if available.x < 4.0 || available.y < 4.0 {
        return;
    }
    let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
    let rect = response.rect;
    let aspect = rect.width() / rect.height().max(1.0);
    let pixels_per_point = ui.ctx().pixels_per_point();
    state.voxel_viewport_size = Some((
        (rect.width() * pixels_per_point).round().max(1.0) as u32,
        (rect.height() * pixels_per_point).round().max(1.0) as u32,
    ));

    painter.image(
        view3d_tex_id,
        rect,
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        Color32::WHITE,
    );
    draw_world_box(&painter, &state.voxel_camera, aspect, rect);

    // Orbit with the right button, pan with the middle button.
    if response.dragged_by(PointerButton::Secondary) {
        let delta = response.drag_delta();
        state.voxel_camera.yaw -= delta.x * ORBIT_SPEED;
        state.voxel_camera.pitch += delta.y * ORBIT_SPEED;
        state.voxel_camera.clamp_pitch();
    }
    if response.dragged_by(PointerButton::Middle) {
        let delta = response.drag_delta();
        let scale = state.voxel_camera.distance * PAN_SPEED;
        let right = state.voxel_camera.right();
        let up = state.voxel_camera.up();
        state.voxel_camera.target += right * (-delta.x * scale) + up * (delta.y * scale);
    }

    if response.hovered() {
        let wheel = wheel_points(ui.ctx());
        if wheel.abs() > f32::EPSILON {
            // Zoom about the orbit target, with one wheel notch (~40 points)
            // changing the distance by about 10%. The previous raw line delta
            // made a notch 40x too small, which read as "zoom does nothing".
            let distance = state.voxel_camera.distance * (-wheel * WHEEL_ZOOM_RATE).exp();
            state.voxel_camera.distance = distance.clamp(MIN_DISTANCE, MAX_DISTANCE);
        }
    }

    if response.clicked() {
        response.request_focus();
    }

    if (response.hovered() || response.has_focus()) && !ui.ctx().text_edit_focused() {
        let (dir, dt) = ui.ctx().input(|input| {
            let mut dir = Vec3::ZERO;
            if input.key_down(Key::W) {
                dir.z += 1.0;
            }
            if input.key_down(Key::S) {
                dir.z -= 1.0;
            }
            if input.key_down(Key::A) {
                dir.x -= 1.0;
            }
            if input.key_down(Key::D) {
                dir.x += 1.0;
            }
            (dir, input.stable_dt.min(0.1))
        });
        if dir != Vec3::ZERO {
            let forward = state.voxel_camera.forward();
            let right = state.voxel_camera.right();
            let planar_forward = Vec3::new(forward.x, 0.0, forward.z).normalize_or(Vec3::NEG_Z);
            let planar_right = Vec3::new(right.x, 0.0, right.z).normalize_or(Vec3::X);
            let motion = planar_forward * dir.z + planar_right * dir.x;
            state.voxel_camera.target +=
                motion * (state.voxel_camera.distance * KEY_PAN_SPEED * dt);
        }
    }

    let modifiers = ui.ctx().input(|input| input.modifiers);
    let pressed = ui.input(|input| input.pointer.button_pressed(PointerButton::Primary));
    let released = ui.input(|input| input.pointer.button_released(PointerButton::Primary));
    let primary_down = ui.input(|input| input.pointer.primary_down());
    let now = Instant::now();

    // Hover pick drives the ghost. It is throttled because `pick` performs a
    // small GPU read-back; pointer movement refreshes it immediately.
    let hover_pointer = response
        .hover_pos()
        .or_else(|| response.interact_pointer_pos());
    let mut hover_center = None;
    if let Some(pointer) = hover_pointer.filter(|_| response.hovered()) {
        let ndc = ndc_of(rect, pointer);
        let (origin, direction) = camera_ray(&state.voxel_camera, aspect, ndc);
        hover_center = hover_brush_center(state, origin, direction, pointer, now, false);
    }

    if pressed && let Some(pointer) = hover_pointer {
        let ndc = ndc_of(rect, pointer);
        let (origin, direction) = camera_ray(&state.voxel_camera, aspect, ndc);
        let center = hover_brush_center(state, origin, direction, pointer, now, true);
        let hit = state.voxel_hover_pick;
        state.voxel_last_pick = hit;
        let box_entry = world_box_entry(origin, direction);

        if modifiers.alt {
            if let Some(hit) = hit
                && hit.ty != 0
            {
                state.voxel_tool = VoxelTool::Element(hit.ty);
                let name = super::voxel_palette::voxel_element_name(hit.ty);
                let message = t_fmt("voxel3d.status_picked", &[&name]);
                state.set_status(message, false);
            }
        } else if state.voxel_tool == VoxelTool::StoneBox {
            // The synthetic hit uses an inward normal, so the box is placed
            // inside the world volume when the scene is still empty.
            let synthetic = hit.or_else(|| {
                box_entry.map(|(point, normal)| {
                    let inside = Vec3::from_array(point) + Vec3::from_array(direction) * 1.0;
                    let voxel = [
                        inside.x.floor().clamp(0.0, 255.0) as u32,
                        inside.y.floor().clamp(0.0, 255.0) as u32,
                        inside.z.floor().clamp(0.0, 255.0) as u32,
                    ];
                    tpt_sim::voxel3d::pick::PickHit {
                        voxel,
                        normal: [-normal[0], -normal[1], -normal[2]],
                        ty: 0,
                        temp_k: 293.15,
                    }
                })
            });
            let _ = state.spawn_voxel_box(synthetic);
        } else if let Some(center) = center {
            let mut mode = effective_draw_mode(modifiers, state.voxel_draw_mode);
            // The sim tools have no line/rectangle gesture in TPT; keep them
            // on the freehand path so heat/cool always covers the brush.
            if state.voxel_tool.is_temperature() {
                mode = VoxelDrawMode::Brush;
            }
            if mode == VoxelDrawMode::Fill {
                // The fill seed is the empty cell the ray enters: one step
                // along the hit normal, or the entry point when the world is
                // still empty.
                let seed = if let Some(hit) = hit.filter(|hit| hit.ty != 0) {
                    [
                        (hit.voxel[0] as i32 + hit.normal[0]).clamp(0, 255) as u32,
                        (hit.voxel[1] as i32 + hit.normal[1]).clamp(0, 255) as u32,
                        (hit.voxel[2] as i32 + hit.normal[2]).clamp(0, 255) as u32,
                    ]
                } else {
                    [
                        center[0].floor().clamp(0.0, 255.0) as u32,
                        center[1].floor().clamp(0.0, 255.0) as u32,
                        center[2].floor().clamp(0.0, 255.0) as u32,
                    ]
                };
                state.fill_voxel_at(seed);
            } else {
                state.voxel_stroke_start = Some(center);
                state.voxel_stroke_last = Some(center);
                state.voxel_stroke_mode = Some(mode);
                if mode == VoxelDrawMode::Brush {
                    let brush = state.voxel_brush_at(center);
                    state.begin_voxel_stroke(brush);
                    state.stamp_voxel_brush(center, false);
                }
            }
        } else {
            state.set_status(t("voxel3d.status_no_target"), false);
        }
    }

    // Freehand strokes stamp the segment between the previous and current
    // pointer sample. Line/rect strokes are applied once on release, matching
    // TPT's `DrawPoints` / `DrawLine` / `DrawRect` dispatch.
    if primary_down
        && !pressed
        && let (Some(_), Some(last), Some(mode)) = (
            state.voxel_stroke_start,
            state.voxel_stroke_last,
            state.voxel_stroke_mode,
        )
        && let Some(pointer) = hover_pointer.filter(|_| response.hovered())
    {
        let ndc = ndc_of(rect, pointer);
        let (origin, direction) = camera_ray(&state.voxel_camera, aspect, ndc);
        if let Some(center) = hover_brush_center(state, origin, direction, pointer, now, false)
            && center != last
        {
            if mode == VoxelDrawMode::Brush {
                state.capture_voxel_segment(last, center);
                state.stamp_voxel_segment(last, center, false);
            }
            state.voxel_stroke_last = Some(center);
        }
    }

    if released {
        if let (Some(start), Some(last), Some(mode)) = (
            state.voxel_stroke_start.take(),
            state.voxel_stroke_last.take(),
            state.voxel_stroke_mode.take(),
        ) {
            match mode {
                VoxelDrawMode::Line => {
                    let capture = segment_capture_brush(state, start, last);
                    state.begin_voxel_stroke(capture);
                    state.stamp_voxel_segment(start, last, false);
                }
                VoxelDrawMode::Rect => {
                    state.paint_voxel_box(start, last, false);
                }
                VoxelDrawMode::Brush => {}
                VoxelDrawMode::Fill => {}
            }
        }
        state.end_voxel_stroke();
    }

    // Ghost brush: always at the current hover, and at the start point while a
    // line/rect gesture is being dragged. The egui overlay intentionally
    // ignores depth, like the existing world-box cage.
    if let Some(center) = state.voxel_stroke_last.or(hover_center) {
        let radii = state
            .voxel_brush_radii
            .map(|radius| radius.clamp(1, 64) as f32);
        let stroke = Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 212, 120, 185));
        draw_brush_cage(
            &painter,
            &state.voxel_camera,
            aspect,
            rect,
            center,
            radii,
            state.voxel_brush_shape,
            stroke,
        );
        if primary_down
            && let (Some(start), Some(mode)) = (state.voxel_stroke_start, state.voxel_stroke_mode)
        {
            if mode == VoxelDrawMode::Line {
                draw_world_segment(
                    &painter,
                    &state.voxel_camera,
                    aspect,
                    rect,
                    Vec3::from_array(start),
                    Vec3::from_array(center),
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 212, 120, 100)),
                );
                draw_brush_cage(
                    &painter,
                    &state.voxel_camera,
                    aspect,
                    rect,
                    start,
                    radii,
                    state.voxel_brush_shape,
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 212, 120, 100)),
                );
            } else if mode == VoxelDrawMode::Rect {
                let mut half = Vec3::ZERO;
                let mut mid = Vec3::ZERO;
                for axis in 0..3 {
                    let low = start[axis].min(center[axis]);
                    let high = start[axis].max(center[axis]);
                    half[axis] = (high - low) * 0.5 + 0.5;
                    mid[axis] = (low + high) * 0.5;
                }
                draw_cuboid_cage(
                    &painter,
                    &state.voxel_camera,
                    aspect,
                    rect,
                    mid,
                    half,
                    stroke,
                );
            }
        }
    }

    draw_hud(&painter, rect, state, aspect);
}

/// Crosshair and element label at the last picked voxel.
fn draw_hud(painter: &egui::Painter, rect: Rect, state: &AppState, aspect: f32) {
    let Some(hit) = state.voxel_last_pick else {
        return;
    };
    let point = Vec3::new(
        hit.voxel[0] as f32 + 0.5,
        hit.voxel[1] as f32 + 0.5,
        hit.voxel[2] as f32 + 0.5,
    );
    let Some(screen) = world_to_screen(&state.voxel_camera, aspect, rect, point) else {
        return;
    };
    if !rect.contains(screen) {
        return;
    }

    let colour = Color32::from_rgb(255, 200, 90);
    let arm = 7.0;
    painter.line_segment(
        [screen + vec2(-arm, 0.0), screen + vec2(arm, 0.0)],
        Stroke::new(1.5, colour),
    );
    painter.line_segment(
        [screen + vec2(0.0, -arm), screen + vec2(0.0, arm)],
        Stroke::new(1.5, colour),
    );
    painter.circle_stroke(screen, 3.0, Stroke::new(1.0, colour));
    let name = super::voxel_palette::voxel_element_name(hit.ty);
    let label = t_fmt(
        "voxel3d.hud_point",
        &[
            &hit.voxel[0].to_string(),
            &hit.voxel[1].to_string(),
            &hit.voxel[2].to_string(),
            &name,
        ],
    );
    painter.text(
        screen + vec2(arm + 4.0, -2.0),
        Align2::LEFT_BOTTOM,
        label,
        FontId::monospace(12.0),
        colour,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::voxel::brush_bricks;
    use tpt_core::voxel3d::{Brush3d, brick_index};

    #[test]
    fn centre_ray_starts_at_the_eye_and_follows_the_view() {
        let camera = Camera::default();
        let (origin, direction) = camera_ray(&camera, 16.0 / 9.0, [0.0, 0.0]);
        let eye = camera.eye();
        assert!((Vec3::from_array(origin) - eye).length() < 1.0e-3);
        let direction = Vec3::from_array(direction);
        assert!((direction.length() - 1.0).abs() < 1.0e-4);
        assert!(
            direction.dot(camera.forward()) > 0.999,
            "the centre ray must look along the camera forward vector"
        );
    }

    #[test]
    fn ndc_offsets_rotate_the_ray_towards_right_and_up() {
        let camera = Camera::default();
        let (_, centre) = camera_ray(&camera, 1.5, [0.0, 0.0]);
        let (_, right) = camera_ray(&camera, 1.5, [0.5, 0.0]);
        let (_, up) = camera_ray(&camera, 1.5, [0.0, 0.5]);
        let centre = Vec3::from_array(centre);
        assert!(Vec3::from_array(right).dot(camera.right()) > 0.0);
        assert!(Vec3::from_array(up).dot(camera.up()) > 0.0);
        assert!(Vec3::from_array(right).dot(centre) < 0.999);
        assert!(Vec3::from_array(up).dot(centre) < 0.999);
    }

    #[test]
    fn target_projects_to_the_viewport_centre() {
        let camera = Camera::default();
        let rect = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 600.0));
        let screen = world_to_screen(&camera, 800.0 / 600.0, rect, camera.target)
            .expect("the orbit target is in front of the camera");
        assert!((screen.x - rect.center().x).abs() < 0.5);
        assert!((screen.y - rect.center().y).abs() < 0.5);
    }

    #[test]
    fn brush_bounds_cover_every_touched_brick() {
        let brush = Brush3d::new([128.0, 96.0, 128.0], 4.0);
        let bricks = brush_bricks(brush);
        assert_eq!(bricks.len(), 8, "a 9-voxel AABB spans two bricks per axis");
        for expected in [
            brick_index(120, 88, 120) as u32,
            brick_index(128, 96, 128) as u32,
            brick_index(120, 96, 128) as u32,
            brick_index(128, 88, 120) as u32,
        ] {
            assert!(bricks.contains(&expected), "missing brick {expected}");
        }
        assert!(!bricks.contains(&(brick_index(112, 96, 128) as u32)));
    }

    #[test]
    fn empty_world_ray_places_the_brush_inside_the_box() {
        let (entry, normal) =
            world_box_entry([128.0, 128.0, -20.0], [0.0, 0.0, 1.0]).expect("ray enters the box");
        assert_eq!(entry, [128.0, 128.0, 0.0]);
        assert_eq!(normal, [0, 0, -1], "the near face points at the camera");

        let center = brush_entry_center(entry, [0.0, 0.0, 1.0], 4.0);
        assert!(center.iter().all(|value| *value >= 4.0 && *value <= 252.0));
        assert!(
            center[2] > 0.0,
            "the sphere must not be clipped away by the face"
        );
    }

    #[test]
    fn brush_bounds_at_the_world_edge_stay_inside() {
        let brush = Brush3d::new([255.0, 255.0, 255.0], 10.0);
        let bricks = brush_bricks(brush);
        assert!(!bricks.is_empty());
        assert!(
            bricks
                .iter()
                .all(|brick| (*brick as usize) < tpt_core::voxel3d::BRICK_COUNT)
        );
        assert_eq!(brick_index(248, 248, 248) as u32, *bricks.last().unwrap());
    }
}
