//! Spinning three-dimensional `WIP` badge for the title screen.
//!
//! The 5x7 dot-matrix wordmark that the 3D viewport used to draw is rebuilt
//! here as a grid of lit cubes. It spins about the vertical axis with a slight
//! pitch wobble, projects with a mild perspective and shades every visible
//! face from a fixed key light, so the mark reads as solid blocks instead of a
//! flat label.
//!
//! This is a title-screen ornament: it draws through the egui painter (the
//! same CPU-generated mesh path egui text uses) and never touches simulation
//! state or the device-side render passes.

use std::time::Duration;

use egui::{Color32, Pos2, Sense, Shape, Stroke, Ui, Vec2, vec2};

/// 5x7 glyph rows, one bit per column (bit 4 = leftmost).
const GLYPHS: [[u8; 7]; 3] = [
    // W
    [
        0b10001, 0b10001, 0b10001, 0b10101, 0b11111, 0b10101, 0b10001,
    ],
    // I
    [
        0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
    ],
    // P
    [
        0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
    ],
];

/// Glyph columns per letter.
const GLYPH_COLS: f32 = 5.0;
/// Glyph rows per letter.
const GLYPH_ROWS: f32 = 7.0;
/// Empty columns between letters.
const GLYPH_GAP: f32 = 1.0;
/// Badge colour; matches the warm orange of the former watermark.
const BADGE_COLOR: [f32; 3] = [233.0, 127.0, 59.0];
/// Cube half extent, in cells.
const CUBE_HALF: f32 = 0.42;
/// Camera distance in cells; larger values flatten the perspective.
const CAMERA_DIST: f32 = 15.0;
/// Key light direction in view space, pointing from the surface to the light.
const LIGHT_DIR: [f32; 3] = [-0.45, -0.60, -0.66];

/// Cube faces: unit normal plus the four face corners in half-extent units.
const FACES: [([f32; 3], [[f32; 3]; 4]); 6] = [
    (
        [1.0, 0.0, 0.0],
        [
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
        ],
    ),
    (
        [-1.0, 0.0, 0.0],
        [
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ],
    ),
    (
        [0.0, 1.0, 0.0],
        [
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
    ),
    (
        [0.0, -1.0, 0.0],
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ],
    ),
    (
        [0.0, 0.0, 1.0],
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
    ),
    (
        [0.0, 0.0, -1.0],
        [
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ],
    ),
];

/// Draws the badge inside a `size` box and schedules the next animation frame.
///
/// The title screen is otherwise static, so the badge drives the repaint: it
/// asks for one more frame 16 ms later, capping the animation near 60 fps.
pub(super) fn draw_wip_badge(ui: &mut Ui, size: Vec2) {
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter_at(rect);
    let time = ui.input(|input| input.time) as f32;
    let yaw = time * 1.1;
    let pitch = 0.30 * (time * 0.9).sin();

    let letters = GLYPHS.len() as f32;
    let cols = letters * GLYPH_COLS + (letters - 1.0) * GLYPH_GAP;
    // Reserve half a cell on each side so the cubes never touch the card edge.
    let unit = (rect.width() / (cols + 2.0)).min(rect.height() / (GLYPH_ROWS + 2.0));
    let centre = rect.center();

    // Collect every visible face with its depth, then paint far to near.
    let mut faces: Vec<(f32, [Pos2; 4], Color32)> = Vec::new();
    for (letter, glyph) in GLYPHS.iter().enumerate() {
        for (row, bits) in glyph.iter().enumerate() {
            let y = row as f32 - GLYPH_ROWS * 0.5 + 0.5;
            for col in 0..5usize {
                if *bits & (1u8 << (4 - col)) == 0 {
                    continue;
                }
                let x = letter as f32 * (GLYPH_COLS + GLYPH_GAP) + col as f32 - cols * 0.5 + 0.5;
                push_cube(&mut faces, [x, y, 0.0], yaw, pitch, centre, unit);
            }
        }
    }
    faces.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, corners, fill) in faces {
        painter.add(Shape::convex_polygon(corners.to_vec(), fill, Stroke::NONE));
    }

    ui.ctx().request_repaint_after(Duration::from_millis(16));
}

/// Projects the visible faces of one cube into `out`.
fn push_cube(
    out: &mut Vec<(f32, [Pos2; 4], Color32)>,
    cell: [f32; 3],
    yaw: f32,
    pitch: f32,
    screen_centre: Pos2,
    unit: f32,
) {
    for (normal, corners) in FACES {
        let normal = rotate(normal, yaw, pitch);
        // The camera sits at -Z looking towards +Z: only faces with a normal
        // pointing back at it are visible, edge-on faces are skipped.
        if normal[2] > -0.02 {
            continue;
        }
        let mut projected = [Pos2::ZERO; 4];
        let mut depth = 0.0;
        for (slot, corner) in projected.iter_mut().zip(corners) {
            let local = [
                cell[0] + corner[0] * CUBE_HALF,
                cell[1] + corner[1] * CUBE_HALF,
                cell[2] + corner[2] * CUBE_HALF,
            ];
            let rotated = rotate(local, yaw, pitch);
            depth += rotated[2];
            *slot = project(rotated, screen_centre, unit);
        }
        out.push((depth * 0.25, projected, shade(normal)));
    }
}

/// Rotates a point or direction by yaw (about Y) then pitch (about X).
fn rotate(p: [f32; 3], yaw: f32, pitch: f32) -> [f32; 3] {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let x = p[0] * cos_yaw - p[2] * sin_yaw;
    let z = p[0] * sin_yaw + p[2] * cos_yaw;
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let y = p[1] * cos_pitch - z * sin_pitch;
    let z = p[1] * sin_pitch + z * cos_pitch;
    [x, y, z]
}

/// Perspective-projects a rotated point onto the badge rectangle.
fn project(p: [f32; 3], centre: Pos2, unit: f32) -> Pos2 {
    let scale = CAMERA_DIST / (CAMERA_DIST + p[2]);
    centre + vec2(p[0] * scale * unit, p[1] * scale * unit)
}

/// Shades a face from the fixed key light.
fn shade(normal: [f32; 3]) -> Color32 {
    let dot =
        (normal[0] * LIGHT_DIR[0] + normal[1] * LIGHT_DIR[1] + normal[2] * LIGHT_DIR[2]).max(0.0);
    let k = 0.42 + 0.58 * dot;
    Color32::from_rgb(
        (BADGE_COLOR[0] * k) as u8,
        (BADGE_COLOR[1] * k) as u8,
        (BADGE_COLOR[2] * k) as u8,
    )
}
