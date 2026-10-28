//! Geometric rasterization helpers for brushes, lines, boxes, and wall grids.

use std::collections::HashSet;
use tpt_core::commands::BrushShape;
use tpt_core::constants::CELL;

pub(crate) fn generate_brush_points(
    cx: i32,
    cy: i32,
    rx: i32,
    ry: i32,
    shape: BrushShape,
) -> Vec<(i32, i32)> {
    let rx = rx.abs().min(100);
    let ry = ry.abs().min(100);
    let mut points = Vec::new();
    let rx64 = rx as i64;
    let ry64 = ry as i64;
    let ellipse_limit = rx64 * rx64 * ry64 * ry64;

    for dy in -ry..=ry {
        for dx in -rx..=rx {
            let inside = match shape {
                BrushShape::Circle => {
                    let (dx64, dy64) = (dx as i64, dy as i64);
                    dx64 * dx64 * ry64 * ry64 + dy64 * dy64 * rx64 * rx64 <= ellipse_limit
                }
                BrushShape::Square => true,
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

pub(crate) fn generate_line_points(
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    rx: i32,
    ry: i32,
    shape: BrushShape,
) -> Vec<(i32, i32)> {
    let mut points = Vec::new();
    let mut seen = HashSet::new();

    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x1;
    let mut y = y1;

    loop {
        if rx > 0 || ry > 0 {
            for p in generate_brush_points(x, y, rx, ry, shape) {
                if seen.insert(p) {
                    points.push(p);
                }
            }
        } else if seen.insert((x, y)) {
            points.push((x, y));
        }

        if x == x2 && y == y2 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }

    points
}

pub(crate) fn generate_box_points(x1: i32, y1: i32, x2: i32, y2: i32) -> Vec<(i32, i32)> {
    let mut points = Vec::new();
    let (min_x, max_x) = (x1.min(x2), x1.max(x2));
    let (min_y, max_y) = (y1.min(y2), y1.max(y2));
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            points.push((x, y));
        }
    }
    points
}

pub(crate) fn generate_wall_brush_cells(x: i32, y: i32, rx: i32, ry: i32) -> Vec<(i32, i32)> {
    let cx = x / CELL as i32;
    let cy = y / CELL as i32;
    let crx = (rx / CELL as i32).max(0);
    let cry = (ry / CELL as i32).max(0);
    let mut cells = Vec::new();
    for dy in -cry..=cry {
        for dx in -crx..=crx {
            cells.push((cx + dx, cy + dy));
        }
    }
    cells
}

pub(crate) fn generate_wall_line_cells(
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    rx: i32,
    ry: i32,
) -> Vec<(i32, i32)> {
    let cx1 = x1 / CELL as i32;
    let cy1 = y1 / CELL as i32;
    let cx2 = x2 / CELL as i32;
    let cy2 = y2 / CELL as i32;
    let crx = (rx / CELL as i32).max(0);
    let cry = (ry / CELL as i32).max(0);

    let mut cells = Vec::new();
    let mut seen = HashSet::new();

    let dx = (cx2 - cx1).abs();
    let dy = (cy2 - cy1).abs();
    let sx = if cx1 < cx2 { 1 } else { -1 };
    let sy = if cy1 < cy2 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = cx1;
    let mut y = cy1;

    loop {
        for dy in -cry..=cry {
            for dx in -crx..=crx {
                let cell = (x + dx, y + dy);
                if seen.insert(cell) {
                    cells.push(cell);
                }
            }
        }
        if x == cx2 && y == cy2 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
    cells
}

pub(crate) fn generate_wall_box_cells(x1: i32, y1: i32, x2: i32, y2: i32) -> Vec<(i32, i32)> {
    let cx1 = x1 / CELL as i32;
    let cy1 = y1 / CELL as i32;
    let cx2 = x2 / CELL as i32;
    let cy2 = y2 / CELL as i32;
    let (min_x, max_x) = (cx1.min(cx2), cx1.max(cx2));
    let (min_y, max_y) = (cy1.min(cy2), cy1.max(cy2));
    let mut cells = Vec::new();
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            cells.push((x, y));
        }
    }
    cells
}
