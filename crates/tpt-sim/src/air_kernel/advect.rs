use super::*;

/// `vorticityBmap()` (TPT `Air.cpp:53-70`): `(dvy/dx - dvx/dy) / 2`, with each
/// central difference forced to zero when the cell or one of the two samples it
/// differences is flagged in `block_air`.
///
/// Cells outside the `x > 1 && x < xcells-2` (and same for `y`) window return
/// `0`, exactly like the reference implementation.
#[cube]
fn vorticity_bmap(
    vx: &[f32],
    vy: &[f32],
    block_air: &[u32],
    xcells: usize,
    ycells: usize,
    x: usize,
    y: usize,
) -> f32 {
    let mut out = 0.0f32;
    if x > 1usize && x + 2usize < xcells && y > 1usize && y + 2usize < ycells {
        let row = y * xcells;
        let row_m = (y - 1usize) * xcells;
        let row_p = (y + 1usize) * xcells;
        let blocked_here = block_air[row + x] != 0u32;

        let mut dvydx = vy[row + x + 1usize] - vy[row + x - 1usize];
        if blocked_here
            || block_air[row + x + 1usize] != 0u32
            || block_air[row + x - 1usize] != 0u32
        {
            dvydx = 0.0f32;
        }

        let mut dvxdy = vx[row_p + x] - vx[row_m + x];
        if blocked_here || block_air[row_p + x] != 0u32 || block_air[row_m + x] != 0u32 {
            dvxdy = 0.0f32;
        }

        out = (dvydx - dvxdy) * 0.5f32;
    }
    out
}

/// `bmap_blockair` / `bmap_blockairh & 0x8` test used by the far-advection walk.
#[cube]
fn advect_blocked(block: &[u32], idx: usize, heat: u32) -> u32 {
    let mut out = block[idx];
    if heat != 0u32 {
        out = block[idx] & 0x8u32;
    }
    out
}

/// C++ far-advection step walk (`Air.cpp:130-162` for ambient heat,
/// `Air.cpp:392-425` for velocity).
///
/// When the semi-Lagrangian backtrace `(x - dx*0.7, y - dy*0.7)` is more than
/// one cell long and its destination lies in the interior `2..XCELLS-2 x
/// 2..YCELLS-2`, TPT walks from the current cell toward the destination one
/// cell at a time along the dominant axis (`stepLimit = |d*0.7|` steps) and
/// stops *before* the first out-of-bounds / blocked cell. If the walk reaches
/// the destination without finding a wall, the full backtrace is used instead.
///
/// Returns the `(tx, ty)` source position in cell coordinates.
#[cube]
pub fn far_advect_target(
    x: usize,
    y: usize,
    dx: f32,
    dy: f32,
    block: &[u32],
    xcells: usize,
    ycells: usize,
    heat: u32,
) -> (f32, f32) {
    let mut tx = x as f32 - dx * ADVECT_DIST;
    let mut ty = y as f32 - dy * ADVECT_DIST;
    let mut adx = dx * ADVECT_DIST;
    if adx < 0.0f32 {
        adx = 0.0f32 - adx;
    }
    let mut ady = dy * ADVECT_DIST;
    if ady < 0.0f32 {
        ady = 0.0f32 - ady;
    }
    let limit_x = xcells as f32 - 2.0f32;
    let limit_y = ycells as f32 - 2.0f32;
    if (adx > 1.0f32 || ady > 1.0f32)
        && tx >= 2.0f32
        && tx < limit_x
        && ty >= 2.0f32
        && ty < limit_y
    {
        let mut abs_dx = dx;
        if abs_dx < 0.0f32 {
            abs_dx = 0.0f32 - abs_dx;
        }
        let mut abs_dy = dy;
        if abs_dy < 0.0f32 {
            abs_dy = 0.0f32 - abs_dy;
        }
        let step_x = if abs_dx > abs_dy {
            if dx < 0.0f32 { 1.0f32 } else { -1.0f32 }
        } else {
            0.0f32 - dx / abs_dy
        };
        let step_y = if abs_dx > abs_dy {
            0.0f32 - dy / abs_dx
        } else {
            if dy < 0.0f32 { 1.0f32 } else { -1.0f32 }
        };
        let step_limit = if abs_dx > abs_dy {
            adx as i32
        } else {
            ady as i32
        };

        tx = x as f32;
        ty = y as f32;
        let mut step = 0i32;
        let mut stopped = 0u32;
        while step < step_limit {
            tx += step_x;
            ty += step_y;
            let cx = (tx + 0.5f32) as i32;
            let cy = (ty + 0.5f32) as i32;
            let mut hit = 1u32;
            if cx >= 0i32 && cy >= 0i32 && cx < xcells as i32 && cy < ycells as i32 {
                let cidx = (cy as usize) * xcells + (cx as usize);
                hit = 0u32;
                if advect_blocked(block, cidx, heat) != 0u32 {
                    hit = 1u32;
                }
            }
            if hit != 0u32 {
                // Stop before the first wall / out-of-bounds cell.
                tx -= step_x;
                ty -= step_y;
                stopped = 1u32;
                step = step_limit;
            } else {
                step += 1i32;
            }
        }
        if stopped == 0u32 {
            // No wall found: take the full backtrace.
            tx = x as f32 - dx * ADVECT_DIST;
            ty = y as f32 - dy * ADVECT_DIST;
        }
    }
    (tx, ty)
}

/// Velocity smoothing, advection, vorticity, fans and the air-mode overrides:
/// the final loop of `Air::update_air` (`Air.cpp:347-500`).
///
/// For every cell the kernel
///
/// * applies the 3x3 normalized Gaussian kernel — `w(0,0) = 1`,
///   `w(±1,0) = e^-2`, `w(±1,±1) = e^-4` — to the velocity **and** pressure
///   fields, reflecting the centre value when the neighbour is outside the
///   `1..XCELLS-2 x 1..YCELLS-2` interior or flagged in `block_air`,
/// * walks the far-advection step of [`far_advect_target`] and blends the
///   bilinear sample of `ovx` / `ovy` with `AIR_VADV = 0.3`,
/// * adds the vorticity confinement force of TPT `update_air`
///   (`Air.cpp:440-451`) when `vorticity_coeff > 0`,
/// * adds the fan velocity of `WL_FAN` cells (`wall_bmap == 5`),
/// * clamps pressure / velocity to `[-256, 256]` and applies the `air_mode`
///   overrides (`Air.cpp:480-500`).
///
/// The smoothed pressure becomes the new `pv`, exactly as the reference
/// `memcpy(pv, opv)` at the end of `update_air`; the velocity and pressure
/// buffers are written directly (this pass only reads the `o*` scratch pair).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn air_smooth_advect_kernel(
    ovx: &[f32],
    ovy: &[f32],
    opv: &[f32],
    block_air: &[u32],
    wall_bmap: &[u32],
    fan_vx: &[f32],
    fan_vy: &[f32],
    vx: &mut [f32],
    vy: &mut [f32],
    pv: &mut [f32],
    xcells: usize,
    ycells: usize,
    vorticity_coeff: f32,
    air_mode: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < xcells * ycells {
        let x = idx % xcells;
        let y = idx / xcells;

        let center_vx = ovx[idx];
        let center_vy = ovy[idx];
        let center_pv = opv[idx];

        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_p = 0.0f32;
        let mut wsum = 0.0f32;
        let mut dy = 0usize;
        while dy < 3usize {
            let mut dx = 0usize;
            while dx < 3usize {
                // `dx` / `dy` are biased by one: 0 = -1, 1 = 0, 2 = +1.
                let mut nx = x + dx;
                nx = nx.saturating_sub(1usize);
                let mut ny = y + dy;
                ny = ny.saturating_sub(1usize);

                // w(0,0) = 1, w(±1,0) = e^-2, w(±1,±1) = e^-4
                let mut w = 0.01831564f32;
                if dx == 1usize {
                    w = 0.13533528f32;
                }
                if dy == 1usize {
                    w = 0.13533528f32;
                }
                if dx == 1usize && dy == 1usize {
                    w = 1.0f32;
                }

                // `Air.cpp:362-377`: only the interior and unblocked neighbours
                // are sampled; every other tap repeats the centre value.
                let mut interior_tap = 0u32;
                if nx >= 1usize && ny >= 1usize && nx + 1usize < xcells && ny + 1usize < ycells {
                    interior_tap = 1u32;
                }
                let ni = ny * xcells + nx;
                let mut sx = center_vx;
                let mut sy = center_vy;
                let mut sp = center_pv;
                if interior_tap != 0u32 && block_air[ni] == 0u32 {
                    sx = ovx[ni];
                    sy = ovy[ni];
                    sp = opv[ni];
                }

                sum_x += sx * w;
                sum_y += sy * w;
                sum_p += sp * w;
                wsum += w;
                dx += 1usize;
            }
            dy += 1usize;
        }
        let mut next_vx = sum_x / wsum;
        let mut next_vy = sum_y / wsum;
        let mut next_pv = sum_p / wsum;

        // Trying to take velocity from far away (`Air.cpp:380-425`).
        let (tx0, ty0) = far_advect_target(x, y, next_vx, next_vy, block_air, xcells, ycells, 0u32);
        let i0 = tx0 as i32;
        let j0 = ty0 as i32;
        let fx = tx0 - i0 as f32;
        let fy = ty0 - j0 as f32;
        // `Air.cpp:427`: velocity samples use the stricter `2..XCELLS-3`
        // window (unlike the heat pass, which uses `0..XCELLS-1`).
        if block_air[idx] == 0u32
            && i0 >= 2i32
            && i0 < xcells as i32 - 3i32
            && j0 >= 2i32
            && j0 < ycells as i32 - 3i32
            && fx >= 0.0f32
            && fy >= 0.0f32
        {
            let iu = i0 as usize;
            let ju = j0 as usize;
            let r0 = ju * xcells;
            let r1 = (ju + 1usize) * xcells;
            let w00 = (1.0f32 - fx) * (1.0f32 - fy);
            let w10 = fx * (1.0f32 - fy);
            let w01 = (1.0f32 - fx) * fy;
            let w11 = fx * fy;
            next_vx = next_vx * 0.7f32
                + 0.3f32
                    * (ovx[r0 + iu] * w00
                        + ovx[r0 + iu + 1usize] * w10
                        + ovx[r1 + iu] * w01
                        + ovx[r1 + iu + 1usize] * w11);
            next_vy = next_vy * 0.7f32
                + 0.3f32
                    * (ovy[r0 + iu] * w00
                        + ovy[r0 + iu + 1usize] * w10
                        + ovy[r1 + iu] * w01
                        + ovy[r1 + iu + 1usize] * w11);
        }

        // Vorticity confinement, TPT `update_air` (Air.cpp:440-451).
        // The force pushes along the gradient of |vorticity|, scaled by the
        // local vorticity, so the small-scale eddies the solver would otherwise
        // diffuse away survive.
        if vorticity_coeff > 0.0f32
            && x > 1usize
            && x + 2usize < xcells
            && y > 1usize
            && y + 2usize < ycells
        {
            let w = vorticity_bmap(ovx, ovy, block_air, xcells, ycells, x, y);
            let mut wx_p = vorticity_bmap(ovx, ovy, block_air, xcells, ycells, x + 1usize, y);
            let mut wx_m = vorticity_bmap(ovx, ovy, block_air, xcells, ycells, x - 1usize, y);
            let mut wy_p = vorticity_bmap(ovx, ovy, block_air, xcells, ycells, x, y + 1usize);
            let mut wy_m = vorticity_bmap(ovx, ovy, block_air, xcells, ycells, x, y - 1usize);
            if wx_p < 0.0f32 {
                wx_p = 0.0f32 - wx_p;
            }
            if wx_m < 0.0f32 {
                wx_m = 0.0f32 - wx_m;
            }
            if wy_p < 0.0f32 {
                wy_p = 0.0f32 - wy_p;
            }
            if wy_m < 0.0f32 {
                wy_m = 0.0f32 - wy_m;
            }

            let dwx = (wx_p - wx_m) * 0.5f32;
            let dwy = (wy_p - wy_m) * 0.5f32;
            let norm = (dwx * dwx + dwy * dwy).sqrt();
            let k = vorticity_coeff / 5.0f32;

            next_vx += k * dwy / (norm + 0.001f32) * w;
            next_vy += k * (0.0f32 - dwx) / (norm + 0.001f32) * w;
        }

        // Fans drive the air: `WL_FAN` is wall id 5.
        if wall_bmap[idx] == 5u32 {
            next_vx += fan_vx[idx];
            next_vy += fan_vy[idx];
        }

        // Pressure / velocity caps (`Air.cpp:469-477`).
        next_vx = clampf(next_vx, -256.0f32, 256.0f32);
        next_vy = clampf(next_vy, -256.0f32, 256.0f32);
        next_pv = clampf(next_pv, -256.0f32, 256.0f32);

        // Air mode overrides (0 = on, 1 = pressure off, 2 = velocity off,
        // 3 = off). Mode 4 (`NoUpdate`) never reaches this kernel.
        if air_mode == 1u32 {
            next_pv = 0.0f32;
        }
        if air_mode == 2u32 {
            next_vx = 0.0f32;
            next_vy = 0.0f32;
        }
        if air_mode == 3u32 {
            next_vx = 0.0f32;
            next_vy = 0.0f32;
            next_pv = 0.0f32;
        }

        vx[idx] = next_vx;
        vy[idx] = next_vy;
        pv[idx] = next_pv;
    }
}
