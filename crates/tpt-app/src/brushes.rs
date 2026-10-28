//! Brush rasterisation and tool-stroke dispatch.
//!
//! The UI layer tracks the cursor in simulation pixels; this module turns that
//! into the concrete pixel lists consumed by
//! [`SimulationEngine::apply_command`](tpt_sim::SimulationEngine::apply_command)
//! and selects the right [`SimCommand`] for the active tool.
//!
//! Point ordering matters. Brush points are emitted bottom-to-top (`y`
//! descending, because `y` grows downwards in simulation space), matching The
//! Powder Toy's `Brush::iterator`. Powders drawn in that order settle into the
//! rows that were already written instead of falling through the part of the
//! brush that has not been painted yet.

use std::collections::HashSet;

use tpt_core::commands::{BrushShape, SimCommand};
use tpt_core::constants::{CELL, PT_LIGH, PT_NONE, WL_ERASE, WL_FAN};

use crate::app_state::{AppState, SelectedTool, pack_deco_colour};

/// Largest brush radius (per axis) the UI can produce, matching the `0..=100`
/// clamp used by [`adjust_brush_size`].
pub const MAX_BRUSH_RADIUS: i32 = 100;

/// Speed, in pixels per step, the single-particle tool launches a dragged
/// particle at. The reference `PlopTool` creates particles at rest; the port
/// gives the drag a direction so a drag behaves like `sim.partVelocity` without
/// a second Lua-style call. The value matches the velocity the reference
/// gravity brush used (`2.0`), well inside the movement kernel's +/-16 window.
pub const PARTICLE_LAUNCH_SPEED: f32 = 2.0;

/// Rasterises one brush stamp centred on `(cx, cy)`.
///
/// Rows are emitted bottom-to-top (`dy` from `+ry` down to `-ry`) and columns
/// left-to-right (`dx` from `-rx` to `+rx`). Negative radii are treated as
/// their absolute value and every radius is clamped to [`MAX_BRUSH_RADIUS`].
///
/// Degenerate brushes behave sensibly: `rx == ry == 0` yields the single pixel
/// `(cx, cy)`, `rx == 0` yields a vertical line and `ry == 0` a horizontal one.
pub fn generate_brush_points(
    cx: i32,
    cy: i32,
    rx: i32,
    ry: i32,
    shape: BrushShape,
) -> Vec<(i32, i32)> {
    let rx = rx.saturating_abs().min(MAX_BRUSH_RADIUS);
    let ry = ry.saturating_abs().min(MAX_BRUSH_RADIUS);

    let mut points = Vec::with_capacity(((2 * rx + 1) * (2 * ry + 1)) as usize);

    let rx64 = rx as i64;
    let ry64 = ry as i64;
    // `rx^2 * ry^2`; both radii are <= 100 so the product stays far below i64
    // overflow even before the clamp.
    let ellipse_limit = rx64 * rx64 * ry64 * ry64;

    // Bottom-to-top: `y` grows downwards, so the bottom row is `+ry`.
    for dy in (-ry..=ry).rev() {
        for dx in -rx..=rx {
            let inside = match shape {
                // Ellipse inequality: (dx/rx)^2 + (dy/ry)^2 <= 1, scaled by
                // `rx^2 * ry^2` to stay in integer arithmetic.
                BrushShape::Circle => {
                    let (dx64, dy64) = (dx as i64, dy as i64);
                    dx64 * dx64 * ry64 * ry64 + dy64 * dy64 * rx64 * rx64 <= ellipse_limit
                }
                BrushShape::Square => true,
                // Upward-pointing triangle: apex on the top row, base on the
                // bottom row. The half width grows linearly from 0 at
                // `dy == -ry` to `rx` at `dy == ry`.
                BrushShape::Triangle => {
                    let half_width = if ry64 == 0 {
                        rx64
                    } else {
                        rx64 * (dy as i64 + ry64) / (2 * ry64)
                    };
                    (dx as i64).abs() <= half_width
                }
            };
            if inside {
                points.push((cx + dx, cy + dy));
            }
        }
    }

    points
}

/// Appends `p` to `points` unless it was already produced by an earlier stamp.
///
/// `seen` is `None` for single-pixel brushes, where a Bresenham line is already
/// duplicate-free and the hash bookkeeping would be pure overhead.
fn push_unique(
    points: &mut Vec<(i32, i32)>,
    seen: &mut Option<HashSet<(i32, i32)>>,
    p: (i32, i32),
) {
    match seen {
        Some(seen) => {
            if seen.insert(p) {
                points.push(p);
            }
        }
        None => points.push(p),
    }
}

/// Stamps the brush along a Bresenham line from `(x0, y0)` to `(x1, y1)`.
///
/// Consecutive stamps overlap heavily, so each pixel is emitted exactly once, in
/// the order the stroke first touches it. The endpoints are inclusive; a zero
/// length line degenerates to a single brush stamp.
pub fn generate_line_points(
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    rx: i32,
    ry: i32,
    shape: BrushShape,
) -> Vec<(i32, i32)> {
    let rx = rx.saturating_abs();
    let ry = ry.saturating_abs();

    let mut points = Vec::new();
    let mut seen = if rx == 0 && ry == 0 {
        None
    } else {
        Some(HashSet::new())
    };

    // 64-bit error terms: no overflow for any i32 endpoint pair.
    let mut x = x0 as i64;
    let mut y = y0 as i64;
    let target_x = x1 as i64;
    let target_y = y1 as i64;

    let dx = (target_x - x).abs();
    let dy = -(target_y - y).abs();
    let step_x = if x < target_x { 1 } else { -1 };
    let step_y = if y < target_y { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        for p in generate_brush_points(x as i32, y as i32, rx, ry, shape) {
            push_unique(&mut points, &mut seen, p);
        }

        if x == target_x && y == target_y {
            break;
        }

        // Classic integer Bresenham: the doubled error decides whether this
        // step advances on the x axis, the y axis, or both (diagonal).
        let err2 = 2 * err;
        if err2 >= dy {
            err += dy;
            x += step_x;
        }
        if err2 <= dx {
            err += dx;
            y += step_y;
        }
    }

    points
}

/// Fills the axis-aligned rectangle spanned by the two corners, inclusive.
///
/// Rows are emitted bottom-to-top so a filled rectangle of powder settles like
/// a poured block. The brush outline is not applied: this is the rectangle
/// tool, not a swept brush.
pub fn generate_rect_points(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let (x_min, x_max) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    let (y_min, y_max) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };

    let width = (x_max as i64 - x_min as i64 + 1) as u64;
    let height = (y_max as i64 - y_min as i64 + 1) as u64;
    // Bounded capacity hint: a nonsensical corner pair must not reserve
    // gigabytes up front. `Vec` still grows as needed for real strokes.
    let capacity = width.saturating_mul(height).min(4_000_000) as usize;
    let mut points = Vec::with_capacity(capacity);

    for y in (y_min..=y_max).rev() {
        for x in x_min..=x_max {
            points.push((x, y));
        }
    }

    points
}

/// Snaps the line end point `(x1, y1)` to the nearest multiple of 45° around
/// `(x0, y0)`, producing a perfectly horizontal, vertical or diagonal line.
///
/// The UI calls this while the Alt key is held. The horizontal/vertical
/// boundary sits at `atan(0.4) ≈ 21.8°`, i.e. the nearest of the eight compass
/// directions wins. Diagonal snaps keep both axes at the same length so the
/// line really is 45°.
pub fn snap_line_coords(x0: i32, y0: i32, x1: i32, y1: i32) -> (i32, i32) {
    let dx = x1 as i64 - x0 as i64;
    let dy = y1 as i64 - y0 as i64;
    let adx = dx.abs();
    let ady = dy.abs();

    if adx == 0 && ady == 0 {
        return (x0, y0);
    }

    if ady * 5 <= adx * 2 {
        return (x1, y0);
    }
    if adx * 5 <= ady * 2 {
        return (x0, y1);
    }

    let len = (adx + ady) / 2;
    let nx = x0 as i64 + len * dx.signum();
    let ny = y0 as i64 + len * dy.signum();
    (nx as i32, ny as i32)
}

/// Applies a `delta` to the brush radii, clamped to `0..=MAX_BRUSH_RADIUS`.
///
/// `lock_x` / `lock_y` freeze the respective axis, which is what the X and Y
/// lock checkboxes in the brush size row do: a locked axis keeps its current
/// value so the brush can be stretched in a single direction.
pub fn adjust_brush_size(rx: &mut i32, ry: &mut i32, delta: i32, lock_x: bool, lock_y: bool) {
    if !lock_x {
        *rx = rx.saturating_add(delta).clamp(0, MAX_BRUSH_RADIUS);
    }
    if !lock_y {
        *ry = ry.saturating_add(delta).clamp(0, MAX_BRUSH_RADIUS);
    }
}

/// Maps simulation pixels onto the coarse `CELL x CELL` wall block grid used by
/// [`SimCommand::PlaceWalls`].
///
/// A brush stamp touches many pixels per block, so duplicates are dropped: the
/// engine only needs each block once. Negative coordinates floor correctly
/// (`div_euclid`); out-of-range blocks are clipped by the engine.
fn wall_cells_from_points(points: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let cell = CELL as i32;
    let mut cells = Vec::with_capacity(points.len());
    let mut seen = HashSet::with_capacity(points.len());

    for &(x, y) in points {
        let block = (x.div_euclid(cell), y.div_euclid(cell));
        if seen.insert(block) {
            cells.push(block);
        }
    }

    cells
}

/// Blocks of `wall_id` connected to `start` through the four edge neighbours.
///
/// This is the block set `Simulation::FloodWalls(x, y, wall, bm)` visits
/// (`Editing.cpp:299-355`), which `WallTool::DrawLine` uses to collect the
/// connected `FAN` blocks before handing them the drag velocity
/// (`WallTool.cpp:17-24`). The flood walks the host wall map, so a seed outside
/// the canvas or on a different wall yields no blocks.
fn connected_wall_blocks(
    wall_bmap: &[u32],
    xcells: i32,
    ycells: i32,
    start: (i32, i32),
    wall_id: u8,
) -> Vec<(i32, i32)> {
    let inside = |x: i32, y: i32| x >= 0 && y >= 0 && x < xcells && y < ycells;
    if !inside(start.0, start.1) {
        return Vec::new();
    }
    let index = |x: i32, y: i32| y as usize * xcells.max(0) as usize + x as usize;
    if (wall_bmap[index(start.0, start.1)] & 0xFF) as u8 != wall_id {
        return Vec::new();
    }

    let mut visited = vec![false; (xcells as usize) * (ycells as usize)];
    let mut queue = std::collections::VecDeque::from([start]);
    visited[index(start.0, start.1)] = true;
    let mut cells = Vec::new();
    while let Some((x, y)) = queue.pop_front() {
        cells.push((x, y));
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (nx, ny) = (x + dx, y + dy);
            if !inside(nx, ny) {
                continue;
            }
            let nidx = index(nx, ny);
            if visited[nidx] || (wall_bmap[nidx] & 0xFF) as u8 != wall_id {
                continue;
            }
            visited[nidx] = true;
            queue.push_back((nx, ny));
        }
    }
    cells
}

/// Velocity [`WallTool::DrawLine`] gives the flooded fans.
///
/// `WallTool.cpp:12-15` scales the raw pixel drag by `0.005f` and by the tool
/// strength; `GameController::DrawLine` sets `Strength = 1.0`
/// (`GameController.cpp:383`), so a 20-pixel drag is a velocity of `0.1`.
fn fan_drag_velocity(p0: (i32, i32), p1: (i32, i32)) -> (f32, f32) {
    (
        (p1.0 as f32 - p0.0 as f32) * 0.005,
        (p1.1 as f32 - p0.1 as f32) * 0.005,
    )
}

/// Clears everything under `points`.
///
/// TPT's eraser is the `PT_NONE` brush, which removes all three layers:
/// particles are erased with a single `PlaceParticles` command
/// (`elem_ty == PT_NONE` kills every particle at each point), walls with a
/// single `WL_ERASE` wall command on the touched blocks, and text signs whose
/// anchor the brush covers. The walls menu still offers `WL_ERASE` on its own
/// for walls-only erasing.
fn erase_points(state: &mut AppState, points: Vec<(i32, i32)>) {
    if points.is_empty() {
        return;
    }

    state.erase_signs(&points);

    state.engine.apply_command(SimCommand::PlaceParticles {
        points: points.clone(),
        elem_ty: PT_NONE,
        replace_mode: true,
        dcolour: 0,
    });

    state.engine.apply_command(SimCommand::PlaceWalls {
        cells: wall_cells_from_points(&points),
        wall_id: WL_ERASE,
        fan_vel: None,
    });
}

/// Executes one stroke (or click) of the active tool.
///
/// `p0` is the press position (the previous cursor sample for free-hand drags)
/// and `p1` the current cursor position. `is_secondary` picks
/// [`AppState::secondary_tool`] (usually the right mouse button) instead of
/// [`AppState::primary_tool`].
///
/// * `is_rect` — filled rectangle between `p0` and `p1`, brush outline ignored.
/// * `is_line` — straight line from the press position to the cursor, brush
///   stamped along the way.
/// * neither — free-hand drag: interpolates from `p0` to `p1` so fast mouse
///   movement leaves no gaps; a click degenerates to a single brush stamp.
/// * `is_fill` — particle tools flood-fill from `p0` instead of drawing. The
///   property tool has its own fill mode (`PropertyWindow::fill`, TPT's
///   `DrawFill`), which never reaches this function.
pub fn apply_tool_stroke(
    state: &mut AppState,
    p0: (i32, i32),
    p1: (i32, i32),
    is_secondary: bool,
    is_line: bool,
    is_rect: bool,
    is_fill: bool,
) {
    // Snapshot the brush configuration before the tool match: the match arms
    // borrow `state.engine` mutably, and these are the only brush values the
    // geometry needs.
    let shape = state.brush_shape;
    let rx = state.brush_rx;
    let ry = state.brush_ry;
    let replace_mode = state.replace_mode;

    // Built lazily so tools that do not sweep a brush (rigid bodies, flood
    // fill) never pay for rasterising a large stamp.
    let build_points = || -> Vec<(i32, i32)> {
        if is_rect {
            generate_rect_points(p0.0, p0.1, p1.0, p1.1)
        } else if is_line || p0 != p1 {
            generate_line_points(p0.0, p0.1, p1.0, p1.1, rx, ry, shape)
        } else {
            generate_brush_points(p1.0, p1.1, rx, ry, shape)
        }
    };

    // Copied out of the state: the `Sample` arm rebinds a tool, which would
    // otherwise conflict with the borrow held by the match scrutinee.
    let tool = if is_secondary {
        state.secondary_tool
    } else {
        state.primary_tool
    };

    // Flood fill only makes sense for particle elements: it seeds from the
    // press position and ignores the brush outline.
    if is_fill
        && let SelectedTool::Element(elem_ty) = tool
        && elem_ty != PT_NONE
    {
        state.engine.apply_command(SimCommand::FloodFillParticles {
            x: p0.0,
            y: p0.1,
            elem_ty,
        });
        return;
    }

    match tool {
        SelectedTool::Element(elem_ty) => {
            // `Editing.cpp:750-760`: a bolt ignores the brush outline. One
            // bolt per stroke call, powered by the brush radius.
            if elem_ty == PT_LIGH {
                state.engine.apply_command(SimCommand::CreateLightning {
                    x: p1.0,
                    y: p1.1,
                    radius_x: rx,
                    radius_y: ry,
                });
                return;
            }
            let points = build_points();
            if points.is_empty() {
                return;
            }

            if elem_ty == PT_NONE {
                // Element NONE is TPT's eraser.
                erase_points(state, points);
            } else {
                state.engine.apply_command(SimCommand::PlaceParticles {
                    points,
                    elem_ty,
                    replace_mode,
                    dcolour: 0, // no decoration colour
                });
            }
        }
        SelectedTool::Gol { rule } => {
            // `GOLTool` paints `LIFE` cells of the configured ruleset. The
            // element tool version of this is
            // `ElementTool(PMAP(ctype, PT_LIFE))`, so the stroke uses the
            // normal placement path plus TPT's `LIFE` create fields.
            let points = build_points();
            if points.is_empty() {
                return;
            }
            tpt_sim::gol_kernel::paint_life(&mut state.engine, &points, rule, replace_mode);
        }
        SelectedTool::Eraser => {
            let points = build_points();
            erase_points(state, points);
        }
        SelectedTool::Wall(wall_id) => {
            // `WallTool::DrawLine` (`WallTool.cpp:9-26`): when the line gesture
            // starts on a `FAN` block, the whole connected fan area is flooded
            // and pushed along the drag instead of drawing a line of walls.
            // The reference gates this on `dragging == false`, which is the
            // line tool here: free-hand drags paint one stamp per sample and a
            // rectangle stays a `CreateWallBox`.
            if is_line && wall_id == WL_FAN && state.wall_at(p0.0, p0.1) == Some(WL_FAN) {
                let cells = {
                    let bufs = &state.engine.bufs;
                    let xcells = bufs.config.xcells() as i32;
                    let ycells = bufs.config.ycells() as i32;
                    let start = (p0.0.div_euclid(CELL as i32), p0.1.div_euclid(CELL as i32));
                    connected_wall_blocks(&bufs.wall_bmap, xcells, ycells, start, WL_FAN)
                };
                if !cells.is_empty() {
                    // `WallTool.cpp:12-24`: the flood sets `fvx`/`fvy` and
                    // turns every helper block back into `WL_FAN`. Here the
                    // same write is one `PlaceWalls` over the collected
                    // blocks, which also carries the velocity.
                    state.engine.apply_command(SimCommand::PlaceWalls {
                        cells,
                        wall_id: WL_FAN,
                        fan_vel: Some(fan_drag_velocity(p0, p1)),
                    });
                    return;
                }
            }

            // Walls live on the coarse block grid, not the pixel grid.
            let cells = wall_cells_from_points(&build_points());

            state.engine.apply_command(SimCommand::PlaceWalls {
                cells,
                wall_id,
                // Fans start with no push; the wall tool's velocity editor
                // updates them afterwards.
                fan_vel: if wall_id == WL_FAN {
                    Some((0.0, 0.0))
                } else {
                    None
                },
            });
        }
        SelectedTool::SimTool(kind) => {
            let points = build_points();
            if points.is_empty() {
                return;
            }

            // `WIND.cpp:20-29` scales the raw pixel step, so the stroke
            // magnitude must survive: `(position2 - position1)`, not a unit
            // vector. Every other tool ignores it.
            let dir = ((p1.0 - p0.0) as f32, (p1.1 - p0.1) as f32);

            // `SimTool.cpp:16` hands `Perform` the brush centre. A stamp or a
            // free-hand drag step is centred on the current cursor; the line
            // and rect gestures use the stroke midpoint, which is exactly the
            // centre `defaultPerformDrawRect` passes (`SimTool.cpp:88-89`).
            let center = if is_line || is_rect {
                ((p0.0 + p1.0) / 2, (p0.1 + p1.1) / 2)
            } else {
                p1
            };

            state.engine.apply_command(SimCommand::ApplySimTool {
                points,
                tool: kind,
                strength: 1.0,
                dir,
                center,
                // `WIND.cpp:20-21`: the held drag path scales by `0.01`, the
                // line/rect gesture (`PerformDrag`) by `0.002`.
                dragging: !(is_line || is_rect),
            });
        }
        SelectedTool::RigidBody {
            shape: body_shape,
            elem_ty,
        } => {
            // Rigid bodies are spawned once per press at the press position;
            // dragging does not sweep them. A degenerate brush still spawns a
            // body with at least one pixel of half-extent.
            state.engine.apply_command(SimCommand::SpawnRigidBody {
                x: p0.0 as f32,
                y: p0.1 as f32,
                half_w: rx.clamp(1, MAX_BRUSH_RADIUS),
                half_h: ry.clamp(1, MAX_BRUSH_RADIUS),
                shape: body_shape,
                elem_ty,
            });
        }
        // `Property` strokes never reach this function: the viewport routes
        // them to `AppState::apply_property_stroke`, which owns the
        // `PropertyTool` semantics (`app_state/property/apply.rs`), including the
        // `DrawFill` flood that replaces a stroke while the window's checkbox
        // is on.
        SelectedTool::Property => {}
        SelectedTool::Sign { mode } => {
            let points = build_points();
            if points.is_empty() {
                return;
            }
            // `DecorationTool::Draw` -> `ApplyDecorationPoint`: blend the
            // chosen colour into every particle under the brush stamp.
            let colour = pack_deco_colour(state.prefs.deco_colour());
            state.engine.apply_command(SimCommand::DecorateAt {
                points,
                colour,
                mode,
            });
        }
        // Text signs are UI state, not simulation commands: the viewport opens
        // the sign editor directly on a click and never routes the tool here.
        SelectedTool::SignTool => {}
        SelectedTool::Sample => {
            // `SampleTool::Draw`: adopt the particle (or wall) under the
            // cursor. It always targets the primary brush, like the reference
            // `SetActiveTool(0, ...)`, and never mutates the simulation.
            let (x, y) = p1;
            if let Some(particle) = state.particle_at(x, y) {
                state.primary_tool = SelectedTool::Element(particle.ty);
                state.prefs.spawn_element = particle.ty;
            } else if let Some(sample) = state.wall_sample_at(x, y) {
                state.primary_tool = SelectedTool::Wall(sample.id);
                // A wall has no particle fields for the inspector to show, so
                // the sample readout is the wall report (`SimulationSample`'s
                // `WallType` + block state). Only on the press sample: a drag
                // would rewrite the line for every pixel it crosses.
                if p0 == p1 {
                    let report = sample.summary(state.prefs.temperature_scale);
                    state.set_status(report, false);
                }
            }
        }
        SelectedTool::Particle => {
            // One particle per stroke sample at the cursor, launched along the
            // drag direction (`sim.partCreate` + `sim.partVelocity`).
            let (vx, vy) = particle_launch_velocity(p0, p1);
            state.engine.apply_command(SimCommand::SpawnParticle {
                x: p1.0,
                y: p1.1,
                elem_ty: state.prefs.spawn_element,
                vx,
                vy,
            });
        }
    }
}

/// Velocity the single-particle tool launches a particle at.
///
/// `p0` is the previous cursor sample and `p1` the current one: a click (no
/// movement) spawns a particle at rest like the reference `PlopTool`, while a
/// drag launches it along the stroke direction at [`PARTICLE_LAUNCH_SPEED`].
pub fn particle_launch_velocity(p0: (i32, i32), p1: (i32, i32)) -> (f32, f32) {
    let (dx, dy) = ((p1.0 - p0.0) as f32, (p1.1 - p0.1) as f32);
    let len = (dx * dx + dy * dy).sqrt();
    if len <= f32::EPSILON {
        return (0.0, 0.0);
    }
    (
        dx / len * PARTICLE_LAUNCH_SPEED,
        dy / len * PARTICLE_LAUNCH_SPEED,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_core::constants::{WL_FAN, WL_STREAM};

    #[test]
    fn circle_zero_radius_is_a_single_pixel() {
        assert_eq!(
            generate_brush_points(5, 7, 0, 0, BrushShape::Circle),
            vec![(5, 7)]
        );
    }

    #[test]
    fn circle_radius_one_is_a_plus() {
        let mut points = generate_brush_points(0, 0, 1, 1, BrushShape::Circle);
        points.sort_unstable();
        assert_eq!(points, vec![(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]);
    }

    #[test]
    fn degenerate_ellipse_is_a_line() {
        let vertical = generate_brush_points(0, 0, 0, 2, BrushShape::Circle);
        assert_eq!(vertical, vec![(0, 2), (0, 1), (0, 0), (0, -1), (0, -2)]);

        let horizontal = generate_brush_points(0, 0, 2, 0, BrushShape::Circle);
        assert_eq!(horizontal, vec![(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)]);
    }

    #[test]
    fn square_covers_the_whole_box() {
        let points = generate_brush_points(10, 10, 1, 1, BrushShape::Square);
        assert_eq!(points.len(), 9);
        assert!(points.contains(&(9, 9)));
        assert!(points.contains(&(11, 11)));
    }

    #[test]
    fn triangle_points_up_and_is_symmetric() {
        let points = generate_brush_points(0, 0, 2, 2, BrushShape::Triangle);
        // dy = 2 -> 5 px (base), dy = 1 -> 3 px, dy = 0 -> 3 px, dy = -1 -> 1 px,
        // dy = -2 -> 1 px (the apex).
        assert_eq!(points.len(), 5 + 3 + 3 + 1 + 1);
        assert_eq!(points.first().copied(), Some((-2, 2)));
        assert_eq!(points.last().copied(), Some((0, -2)));
    }

    #[test]
    fn brush_rows_are_emitted_bottom_to_top() {
        let points = generate_brush_points(0, 0, 2, 2, BrushShape::Square);
        let mut previous_y = i32::MAX;
        for &(_, y) in &points {
            assert!(y <= previous_y, "rows must descend in y");
            previous_y = y;
        }
    }

    #[test]
    fn radius_is_clamped() {
        let points = generate_brush_points(0, 0, 10_000, 10_000, BrushShape::Square);
        let side = (2 * MAX_BRUSH_RADIUS + 1) as usize;
        assert_eq!(points.len(), side * side);
    }

    #[test]
    fn zero_length_line_is_one_stamp() {
        let points = generate_line_points(3, 4, 3, 4, 1, 1, BrushShape::Circle);
        assert_eq!(points.len(), 5);
        assert!(points.contains(&(3, 4)));
    }

    #[test]
    fn thin_line_is_contiguous_bresenham() {
        let points = generate_line_points(0, 0, 5, 2, 0, 0, BrushShape::Circle);
        assert_eq!(points.len(), 6);
        assert_eq!(points.first().copied(), Some((0, 0)));
        assert_eq!(points.last().copied(), Some((5, 2)));
        for pair in points.windows(2) {
            let dx = (pair[0].0 - pair[1].0).abs();
            let dy = (pair[0].1 - pair[1].1).abs();
            assert!(dx <= 1 && dy <= 1, "line must not skip pixels");
        }
    }

    #[test]
    fn thick_line_has_no_duplicate_pixels() {
        let points = generate_line_points(0, 0, 2, 0, 1, 1, BrushShape::Square);
        let unique: HashSet<(i32, i32)> = points.iter().copied().collect();
        assert_eq!(unique.len(), points.len());
        // Five columns (-1..=3) times three rows (-1..=1).
        assert_eq!(points.len(), 15);
    }

    #[test]
    fn rect_is_filled_and_ordered_bottom_to_top() {
        let points = generate_rect_points(2, 1, 0, 0);
        assert_eq!(points.len(), 6);
        assert_eq!(points.first().copied(), Some((0, 1)));
        assert_eq!(points.last().copied(), Some((2, 0)));
    }

    #[test]
    fn snapping_picks_the_nearest_axis() {
        assert_eq!(snap_line_coords(0, 0, 10, 1), (10, 0));
        assert_eq!(snap_line_coords(0, 0, 1, 10), (0, 10));
        assert_eq!(snap_line_coords(0, 0, 10, 9), (9, 9));
        assert_eq!(snap_line_coords(10, 10, 1, 1), (1, 1));
        assert_eq!(snap_line_coords(4, 4, 4, 4), (4, 4));
    }

    #[test]
    fn snapping_keeps_the_quadrant() {
        // Down-left drag: the diagonal snap stays down-left of the origin.
        let (x, y) = snap_line_coords(10, 10, 2, 1);
        assert_eq!((x, y), (2, 2));
        assert!(x < 10 && y < 10);
    }

    #[test]
    fn brush_size_is_clamped_and_axis_locks_hold() {
        let (mut rx, mut ry) = (5, 5);
        adjust_brush_size(&mut rx, &mut ry, 3, false, false);
        assert_eq!((rx, ry), (8, 8));

        adjust_brush_size(&mut rx, &mut ry, 3, true, false);
        assert_eq!((rx, ry), (8, 11));

        adjust_brush_size(&mut rx, &mut ry, 500, false, false);
        assert_eq!((rx, ry), (MAX_BRUSH_RADIUS, MAX_BRUSH_RADIUS));

        adjust_brush_size(&mut rx, &mut ry, -500, false, false);
        assert_eq!((rx, ry), (0, 0));
    }

    #[test]
    fn particle_launch_is_a_click_at_rest_or_a_unit_direction() {
        // A click (no movement) spawns at rest, like the reference plop tool.
        assert_eq!(particle_launch_velocity((4, 4), (4, 4)), (0.0, 0.0));

        // A drag launches along the (normalised) stroke direction.
        let (vx, vy) = particle_launch_velocity((0, 0), (3, 4));
        assert!((vx - PARTICLE_LAUNCH_SPEED * 0.6).abs() < 1e-6);
        assert!((vy - PARTICLE_LAUNCH_SPEED * 0.8).abs() < 1e-6);
        assert!((vx * vx + vy * vy).sqrt() - PARTICLE_LAUNCH_SPEED < 1e-6);

        // Long drags do not get faster: only the direction matters.
        let far = particle_launch_velocity((0, 0), (300, 400));
        assert!((far.0 - vx).abs() < 1e-6 && (far.1 - vy).abs() < 1e-6);
    }

    #[test]
    fn fan_line_flood_sets_the_documented_velocity() {
        // `WallTool.cpp:12-15`: `(position2 - position1) * 0.005f`, multiplied
        // by `Strength`, which `GameController::DrawLine` fixes at `1.0`
        // (`GameController.cpp:383`). A 20-pixel drag is `0.1` pixels/tick.
        let (vx, vy) = fan_drag_velocity((10, 20), (30, 60));
        assert!((vx - 0.1).abs() < 1e-6, "vx was {vx}");
        assert!((vy - 0.2).abs() < 1e-6, "vy was {vy}");

        // No drag and a drag towards the origin.
        assert_eq!(fan_drag_velocity((32, 48), (32, 48)), (0.0, 0.0));
        let (vx, vy) = fan_drag_velocity((40, 30), (30, 10));
        assert!((vx + 0.05).abs() < 1e-6, "vx was {vx}");
        assert!((vy + 0.1).abs() < 1e-6, "vy was {vy}");
    }

    #[test]
    fn fan_flood_collects_the_connected_fan_blocks_only() {
        // 5x4 block grid: an L-shaped fan patch around (1, 1), one isolated
        // fan at (0, 3) and a fan touching the patch only at the corner.
        let xcells = 5i32;
        let ycells = 4i32;
        let mut bmap = vec![0u32; (xcells * ycells) as usize];
        for (x, y) in [(1, 1), (2, 1), (1, 2)] {
            bmap[(y * xcells + x) as usize] = WL_FAN as u32;
        }
        bmap[(3 * xcells) as usize] = WL_FAN as u32;
        bmap[(3 * xcells + 3) as usize] = WL_FAN as u32;
        // A wall id stored above the low byte must still count as a fan.
        bmap[(2 * xcells + 2) as usize] = WL_FAN as u32 | 0x100;

        let mut cells = connected_wall_blocks(&bmap, xcells, ycells, (1, 1), WL_FAN);
        cells.sort_unstable();
        assert_eq!(cells, vec![(1, 1), (1, 2), (2, 1), (2, 2)]);

        // A non-fan seed, an out-of-canvas seed and a wall id none of the
        // blocks carry produce no cells.
        assert!(connected_wall_blocks(&bmap, xcells, ycells, (4, 0), WL_FAN).is_empty());
        assert!(connected_wall_blocks(&bmap, xcells, ycells, (-1, 0), WL_FAN).is_empty());
        assert!(connected_wall_blocks(&bmap, xcells, ycells, (1, 1), WL_STREAM).is_empty());
    }
}
