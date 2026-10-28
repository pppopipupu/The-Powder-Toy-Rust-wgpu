use super::*;

/// `true` (1) when the cell sits in the damped [`BORDER`]-cell band.
///
/// `Air::update_air` (`Air.cpp:260-286`) mixes rows / columns `0`, `1`,
/// `XCELLS-2` and `XCELLS-1` toward the edge pressure / velocity every frame,
/// before any of the interior loops run.
#[cube]
fn in_border(x: usize, y: usize, xcells: usize, ycells: usize) -> u32 {
    let mut out = 0u32;
    if x < BORDER || y < BORDER || x + BORDER >= xcells || y + BORDER >= ycells {
        out = 1u32;
    }
    out
}

/// `true` (1) for the cells the C++ interior loops cover: `1..XCELLS-2` x
/// `1..YCELLS-2` (`Air.cpp:288-345`).
#[cube]
fn interior(x: usize, y: usize, xcells: usize, ycells: usize) -> u32 {
    let mut out = 0u32;
    if x >= 1usize && y >= 1usize && x + 1usize < xcells && y + 1usize < ycells {
        out = 1u32;
    }
    out
}

/// `bmap_blockair` at `(x, y)`, `0` outside the grid.
#[cube]
fn blocked_at(block_air: &[u32], x: i32, y: i32, xcells: usize, ycells: usize) -> u32 {
    let mut out = 0u32;
    if x >= 0i32 && y >= 0i32 && x < xcells as i32 && y < ycells as i32 {
        out = block_air[(y as usize) * xcells + (x as usize)];
    }
    out
}

/// `true` (1) when the C++ pre-clear loop zeroes `vx[x][y]`
/// (`Air.cpp:307-319`): the cell, or its horizontal neighbour, is a blocked
/// interior cell. The source wall must itself be in `1..XCELLS-2` x
/// `1..YCELLS-2`, because that is the loop the reference runs.
#[cube]
fn wall_cleared_vx(block_air: &[u32], x: usize, y: usize, xcells: usize, ycells: usize) -> u32 {
    let mut out = 0u32;
    if y >= 1usize && y + 1usize < ycells {
        if x >= 1usize
            && x + 1usize < xcells
            && blocked_at(block_air, x as i32, y as i32, xcells, ycells) != 0u32
        {
            out = 1u32;
        }
        if x >= 2usize && blocked_at(block_air, x as i32 - 1i32, y as i32, xcells, ycells) != 0u32 {
            out = 1u32;
        }
        if x + 2usize < xcells
            && blocked_at(block_air, x as i32 + 1i32, y as i32, xcells, ycells) != 0u32
        {
            out = 1u32;
        }
    }
    out
}

/// Vertical counterpart of [`wall_cleared_vx`] (`Air.cpp:313-318`).
#[cube]
fn wall_cleared_vy(block_air: &[u32], x: usize, y: usize, xcells: usize, ycells: usize) -> u32 {
    let mut out = 0u32;
    if x >= 1usize && x + 1usize < xcells {
        if y >= 1usize
            && y + 1usize < ycells
            && blocked_at(block_air, x as i32, y as i32, xcells, ycells) != 0u32
        {
            out = 1u32;
        }
        if y >= 2usize && blocked_at(block_air, x as i32, y as i32 - 1i32, xcells, ycells) != 0u32 {
            out = 1u32;
        }
        if y + 2usize < ycells
            && blocked_at(block_air, x as i32, y as i32 + 1i32, xcells, ycells) != 0u32
        {
            out = 1u32;
        }
    }
    out
}

/// `true` (1) when the C++ *post-gradient* wall test zeroes `vx`
/// (`Air.cpp:337-341`): any of the cell and its horizontal neighbours is
/// blocked. Unlike [`wall_cleared_vx`] the reference applies this test without
/// the interior-source restriction.
#[cube]
fn adjacent_blocked_x(block_air: &[u32], x: usize, y: usize, xcells: usize, ycells: usize) -> u32 {
    let mut out = 0u32;
    if blocked_at(block_air, x as i32 - 1i32, y as i32, xcells, ycells) != 0u32
        || blocked_at(block_air, x as i32, y as i32, xcells, ycells) != 0u32
        || blocked_at(block_air, x as i32 + 1i32, y as i32, xcells, ycells) != 0u32
    {
        out = 1u32;
    }
    out
}

/// Vertical counterpart of [`adjacent_blocked_x`] (`Air.cpp:342-344`).
#[cube]
fn adjacent_blocked_y(block_air: &[u32], x: usize, y: usize, xcells: usize, ycells: usize) -> u32 {
    let mut out = 0u32;
    if blocked_at(block_air, x as i32, y as i32 - 1i32, xcells, ycells) != 0u32
        || blocked_at(block_air, x as i32, y as i32, xcells, ycells) != 0u32
        || blocked_at(block_air, x as i32, y as i32 + 1i32, xcells, ycells) != 0u32
    {
        out = 1u32;
    }
    out
}

/// Velocity as the pressure loop sees it: border-mixed, then zeroed by the
/// three-cell wall clear (`Air.cpp:260-324`).
#[cube]
fn effective_vx(
    vx: &[f32],
    block_air: &[u32],
    x: usize,
    y: usize,
    xcells: usize,
    ycells: usize,
    edge_velocity_x: f32,
) -> f32 {
    let mut value = vx[y * xcells + x];
    if in_border(x, y, xcells, ycells) != 0u32 {
        value = edge_velocity_x + (value - edge_velocity_x) * 0.9f32;
    }
    if wall_cleared_vx(block_air, x, y, xcells, ycells) != 0u32 {
        value = 0.0f32;
    }
    value
}

/// Vertical counterpart of [`effective_vx`].
#[cube]
fn effective_vy(
    vy: &[f32],
    block_air: &[u32],
    x: usize,
    y: usize,
    xcells: usize,
    ycells: usize,
    edge_velocity_y: f32,
) -> f32 {
    let mut value = vy[y * xcells + x];
    if in_border(x, y, xcells, ycells) != 0u32 {
        value = edge_velocity_y + (value - edge_velocity_y) * 0.9f32;
    }
    if wall_cleared_vy(block_air, x, y, xcells, ycells) != 0u32 {
        value = 0.0f32;
    }
    value
}

/// First half of `Air::update_air` (`Air.cpp:260-334`): the border damping, the
/// three-cell velocity clear around walls, and the pressure update from the
/// velocity divergence.
///
/// TPT runs two *sequential* interior loops: pressure first, then velocity from
/// the just-updated pressure. Splitting them into two launches keeps that
/// Gauss-Seidel data flow without a race: this kernel reads the old `vx` / `vy`
/// and writes the new pressure into `opv`; [`air_velocity_kernel`] then reads
/// `opv`.
///
/// The outer ring is **not** pinned to `edge_pressure`: `Air.cpp:260-300` only
/// mixes the border band and then runs `Mix(edgePressure, pv, AIR_PLOSS)` on the
/// interior cells (`Air.cpp:326-330`). Cells outside `1..XCELLS-2` keep the
/// mixed value, exactly like the reference.
#[cube(launch_unchecked)]
pub fn air_pressure_kernel(
    pv: &[f32],
    vx: &[f32],
    vy: &[f32],
    block_air: &[u32],
    opv: &mut [f32],
    xcells: usize,
    ycells: usize,
    edge_pressure: f32,
    edge_velocity_x: f32,
    edge_velocity_y: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx < xcells * ycells {
        let x = idx % xcells;
        let y = idx / xcells;

        let mut cur_pv = pv[idx];
        if in_border(x, y, xcells, ycells) != 0u32 {
            cur_pv = edge_pressure + (cur_pv - edge_pressure) * 0.8f32;
        }

        // TPT only updates interior cells; the border band keeps its mixed
        // value until the next frame.
        let mut next_pv = cur_pv;
        if interior(x, y, xcells, ycells) != 0u32 {
            let dp = effective_vx(
                vx,
                block_air,
                x - 1usize,
                y,
                xcells,
                ycells,
                edge_velocity_x,
            ) - effective_vx(
                vx,
                block_air,
                x + 1usize,
                y,
                xcells,
                ycells,
                edge_velocity_x,
            ) + effective_vy(
                vy,
                block_air,
                x,
                y - 1usize,
                xcells,
                ycells,
                edge_velocity_y,
            ) - effective_vy(
                vy,
                block_air,
                x,
                y + 1usize,
                xcells,
                ycells,
                edge_velocity_y,
            );
            next_pv = edge_pressure + (cur_pv - edge_pressure) * 0.9999f32 + dp * 0.15f32;
        }

        opv[idx] = next_pv;
    }
}

/// Second half of `Air::update_air` (`Air.cpp:336-345`): the velocity update
/// from the pressure gradient produced by [`air_pressure_kernel`].
///
/// The border mix and the three-cell wall clear are applied first (they mutate
/// the velocity in place in TPT), then the interior cells are advanced with the
/// *new* pressure and zeroed again when a wall is adjacent, matching the two
/// wall tests at `Air.cpp:307-319` and `Air.cpp:337-344`.
#[cube(launch_unchecked)]
pub fn air_velocity_kernel(
    opv: &[f32],
    vx: &[f32],
    vy: &[f32],
    block_air: &[u32],
    ovx: &mut [f32],
    ovy: &mut [f32],
    xcells: usize,
    ycells: usize,
    edge_velocity_x: f32,
    edge_velocity_y: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx < xcells * ycells {
        let x = idx % xcells;
        let y = idx / xcells;

        let mut cur_vx = vx[idx];
        let mut cur_vy = vy[idx];
        if in_border(x, y, xcells, ycells) != 0u32 {
            cur_vx = edge_velocity_x + (cur_vx - edge_velocity_x) * 0.9f32;
            cur_vy = edge_velocity_y + (cur_vy - edge_velocity_y) * 0.9f32;
        }
        if wall_cleared_vx(block_air, x, y, xcells, ycells) != 0u32 {
            cur_vx = 0.0f32;
        }
        if wall_cleared_vy(block_air, x, y, xcells, ycells) != 0u32 {
            cur_vy = 0.0f32;
        }

        let mut next_vx = cur_vx;
        let mut next_vy = cur_vy;
        if interior(x, y, xcells, ycells) != 0u32 {
            let row = y * xcells;
            let row_m = (y - 1usize) * xcells;
            let row_p = (y + 1usize) * xcells;
            let dx = opv[row + x - 1usize] - opv[row + x + 1usize];
            let dy = opv[row_m + x] - opv[row_p + x];
            next_vx = edge_velocity_x + (cur_vx - edge_velocity_x) * 0.999f32 + dx * 0.2f32;
            next_vy = edge_velocity_y + (cur_vy - edge_velocity_y) * 0.999f32 + dy * 0.2f32;
            if adjacent_blocked_x(block_air, x, y, xcells, ycells) != 0u32 {
                next_vx = 0.0f32;
            }
            if adjacent_blocked_y(block_air, x, y, xcells, ycells) != 0u32 {
                next_vy = 0.0f32;
            }
        }

        ovx[idx] = next_vx;
        ovy[idx] = next_vy;
    }
}
