//! Liquid and powder destination search: TPT `PlanMove` + `MovementPhase`.
//!
//! Port of `Simulation.cpp:2134-2256` (velocity sampling) and `3138-3346` (the
//! `do_move` fallbacks, the diagonal moves and the lateral scans). Powders take
//! the same branch as liquids in the reference (`MovementPhase`'s final `else`
//! covers "liquids and powders"); the only difference is that the lateral scans
//! are gated on `Falldown > 1` (`Simulation.cpp:3194`, `3239`), so a powder
//! skips them and falls through to the `clear` last resort.
//!
//! Every fallback damps the velocity by the element's **signed** `Collision`
//! (`Simulation.cpp:3144`, `3150`, `3168`, `3175`, `3186`, `3335`, `3343`);
//! gases use a negative `Collision` there, liquids usually zero.
//!
//! `PlanMove` receives velocities already scaled to `MAX_VELOCITY` by
//! [`super::velocity::clamp_plan_velocity`], mirroring the clamp inside
//! `PlanMove` (`Simulation.cpp:2170-2175`).
//!
//! # Edge mode
//!
//! `plan_move` folds its sampling position and its final velocity target into
//! the `CELL .. RES-CELL` band with `remainder_p` under `EDGE_LOOP`
//! (`Simulation.cpp:2191-2216`), so the obstacle test probes the pixel on the
//! opposite edge. Every `do_move` in the chain — the three forward probes, both
//! diagonal moves, both lateral scans, the vertical continuation and the
//! `clear` last resort — goes through [`super::do_move_at`], which applies the
//! same fold and the reference's out-of-band kill.

use cubecl::prelude::*;

use super::velocity::base_gravity;
use super::{DIE_INTENT, DO_MOVE_DIE, DO_MOVE_OK, EDGE_LOOP, do_move_at, remainder_p, saw_wood_at};

/// `ISTP = CELL / 2` (`SimulationConfig.h:31`), the `PlanMove` sampling step.
const ISTP: f32 = 2.0f32;

/// Hard bound on the `PlanMove` walk. `MAX_VELOCITY` is `1e4`
/// (`SimulationConfig.h:34`), so the reference's `while (1)` can sample at most
/// `1e4 / ISTP = 5000` times before `mv <= 0`.
const MAX_PLAN_STEPS: u32 = 5001;

/// `PT_PHOT` mirrored for kernel bodies (the `PlanMove` obstacle test blocks a
/// photon on any occupant, `Simulation.cpp:2228`).
const PT_PHOT_U32: u32 = 29;
const _: () = assert!(PT_PHOT_U32 == 29);

/// `WL_DESTROYALL` / `WL_STREAM` / `WL_EHOLE` mirrors (`Simulation.cpp:2226`,
/// `3194-3237`).
const W_DESTROYALL: u32 = 7;
const W_STREAM: u32 = 4;
const W_EHOLE: u32 = 12;

/// `Falldown > 1` selects the liquid-only lateral scans
/// (`Simulation.cpp:3194`, `3239`).
const FALLDOWN_LIQUID: u32 = 1;

/// `(int)(v + 0.5f)` as used by every TPT coordinate conversion.
#[cube]
fn round_i32(v: f32) -> i32 {
    i32::cast_from(v + 0.5f32)
}

/// `true` when the `PlanMove` sample `(fin_x, fin_y)` is an obstacle.
///
/// `Simulation.cpp:2225-2235`:
///
/// ```text
/// int eval = sim.eval_move(t, fin_x, fin_y, nullptr);
/// if (!eval || (can_move[t][TYP(pmap[fin_y][fin_x])] == 3 && eval == 1)
///     || (t == PT_PHOT && pmap[fin_y][fin_x])
///     || bmap[fin_y/CELL][fin_x/CELL]==WL_DESTROYALL
///     || closedEholeStart!=(bmap[fin_y/CELL][fin_x/CELL] == WL_EHOLE && !emap[...]))
/// ```
///
/// The `can_move == 3 && eval == 1` case is exactly the void / photonic-void
/// eat result, which [`super::eval_move_at`] resolves to `1`; the shutter,
/// pump, invisible-wall and switch pass states resolve to `2` and are not
/// blocked here.
#[cube]
#[allow(clippy::too_many_arguments)]
fn plan_sample_blocked(
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    props: u32,
    ty: u32,
    nx: i32,
    ny: i32,
    xr: i32,
    yr: i32,
    xcells: u32,
    closed_ehole_start: bool,
) -> bool {
    let mut blocked = false;
    if nx < 0i32 || ny < 0i32 || nx >= xr || ny >= yr {
        blocked = true;
    } else {
        let t_idx = (ny * xr + nx) as u32;
        let packed = pmap[t_idx as usize];
        let d_ty = packed & 0x1FFu32;
        let raw_can = crate::move_kernel::can_move_at(move_lut, ty as usize, d_ty);
        let eval = crate::move_kernel::eval_move_at(
            pmap, wall_bmap, wall_emap, move_lut, props, ty, nx, ny, xr, yr, xcells,
        );
        let wcell = (((ny as u32) / 4u32) * xcells + ((nx as u32) / 4u32)) as usize;
        let wall = wall_bmap[wcell];
        let closed_now = wall == W_EHOLE && (wall_emap[wcell] & 0xFFu32) == 0u32;
        if eval == 0u32
            || (raw_can == 3u32 && eval == 1u32)
            || (ty == PT_PHOT_U32 && packed != 0u32)
            || wall == W_DESTROYALL
            || closed_now != closed_ehole_start
        {
            blocked = true;
        }
    }
    blocked
}

/// `PlanMove`: sample the velocity in `ISTP = CELL / 2 = 2` px steps to find the
/// requested end position (`fin`) and the last certainly free pixel (`clear`).
///
/// `vx` / `vy` must already be scaled to `MAX_VELOCITY`
/// ([`super::velocity::clamp_plan_velocity`]); `PlanMove` does that itself in
/// the reference, but the caller needs the clamped values for the bounce and
/// diagonal rules too, so the scaling happens once before this call.
///
/// Under `EDGE_LOOP` the running sample position and the final velocity target
/// are folded into the `CELL .. RES-CELL` band with `remainder_p`
/// (`Simulation.cpp:2191-2216`), so the obstacle test that follows probes the
/// pixel on the opposite edge instead of reading it as out of bounds.
///
/// Returns `(fin_x, fin_y, clear_x, clear_y, clear_xf, clear_yf)`; the float
/// `clear` position feeds the diagonal velocity moves.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_move(
    vx: f32,
    vy: f32,
    px: f32,
    py: f32,
    xi: i32,
    yi: i32,
    props: u32,
    ty: u32,
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    xr: i32,
    yr: i32,
    xcells: u32,
    edge_mode: u32,
) -> (i32, i32, i32, i32, f32, f32) {
    let mut mv = vx.abs();
    if vy.abs() > mv {
        mv = vy.abs();
    }
    let mut clear_xf = px;
    let mut clear_yf = py;
    let mut fin_xf = px + vx;
    let mut fin_yf = py + vy;
    let mut fin_x = round_i32(fin_xf);
    let mut fin_y = round_i32(fin_yf);
    let mut clear_x = xi;
    let mut clear_y = yi;

    // A NaN velocity fails this comparison as well, mirroring TPT's
    // `mv < ISTP || isnan(mv)` guard: `clear` then stays on the current pixel
    // and `fin` on the raw velocity target.
    if mv >= ISTP {
        // `Simulation.cpp:2182`: the E-hole state at the start pixel decides
        // whether a sample that crosses the hole boundary is an obstacle.
        let start_wall = wall_bmap[(((yi as u32) / 4u32) * xcells + ((xi as u32) / 4u32)) as usize];
        let start_closed = start_wall == W_EHOLE
            && (wall_emap[(((yi as u32) / 4u32) * xcells + ((xi as u32) / 4u32)) as usize]
                & 0xFFu32)
                == 0u32;

        let step_xf = vx * ISTP / mv;
        let step_yf = vy * ISTP / mv;
        let mut walk_xf = px;
        let mut walk_yf = py;
        let mut remaining = mv;
        let mut planned = false;
        let mut guard = 0u32;
        // The reference loops `while (1)` until `mv <= 0` or an obstacle; with
        // `MAX_VELOCITY = 1e4` that is at most 5000 samples. `guard` keeps the
        // generated shader a bounded loop without changing the result.
        while !planned && guard < MAX_PLAN_STEPS {
            guard += 1u32;
            remaining -= ISTP;
            walk_xf += step_xf;
            walk_yf += step_yf;
            // `Simulation.cpp:2191-2200`: the sample wraps before the obstacle
            // test.
            if edge_mode == EDGE_LOOP {
                let x_ok = walk_xf >= 4.0f32 - 0.5f32 && walk_xf < (xr as f32) - 4.0f32 - 0.5f32;
                let y_ok = walk_yf >= 4.0f32 - 0.5f32 && walk_yf < (yr as f32) - 4.0f32 - 0.5f32;
                if !x_ok {
                    walk_xf = remainder_p(walk_xf - 4.0f32 + 0.5f32, xr as f32 - 8.0f32) + 4.0f32
                        - 0.5f32;
                }
                if !y_ok {
                    walk_yf = remainder_p(walk_yf - 4.0f32 + 0.5f32, yr as f32 - 8.0f32) + 4.0f32
                        - 0.5f32;
                }
            }
            let nfx = round_i32(walk_xf);
            let nfy = round_i32(walk_yf);
            if remaining <= 0.0f32 {
                // Nothing was in the way: use the raw velocity target.
                fin_xf = px + vx;
                fin_yf = py + vy;
                // `Simulation.cpp:2207-2216`: the final target wraps too.
                if edge_mode == EDGE_LOOP {
                    let x_ok = fin_xf >= 4.0f32 - 0.5f32 && fin_xf < (xr as f32) - 4.0f32 - 0.5f32;
                    let y_ok = fin_yf >= 4.0f32 - 0.5f32 && fin_yf < (yr as f32) - 4.0f32 - 0.5f32;
                    if !x_ok {
                        fin_xf = remainder_p(fin_xf - 4.0f32 + 0.5f32, xr as f32 - 8.0f32) + 4.0f32
                            - 0.5f32;
                    }
                    if !y_ok {
                        fin_yf = remainder_p(fin_yf - 4.0f32 + 0.5f32, yr as f32 - 8.0f32) + 4.0f32
                            - 0.5f32;
                    }
                }
                fin_x = round_i32(fin_xf);
                fin_y = round_i32(fin_yf);
                clear_xf = fin_xf - step_xf;
                clear_yf = fin_yf - step_yf;
                clear_x = round_i32(clear_xf);
                clear_y = round_i32(clear_yf);
                planned = true;
            } else if plan_sample_blocked(
                pmap,
                wall_bmap,
                wall_emap,
                move_lut,
                props,
                ty,
                nfx,
                nfy,
                xr,
                yr,
                xcells,
                start_closed,
            ) {
                // Obstacle: `fin` is this sample, `clear` the previous one.
                fin_x = nfx;
                fin_y = nfy;
                clear_xf = walk_xf - step_xf;
                clear_yf = walk_yf - step_yf;
                clear_x = round_i32(clear_xf);
                clear_y = round_i32(clear_yf);
                planned = true;
            }
        }
    }
    (fin_x, fin_y, clear_x, clear_y, clear_xf, clear_yf)
}

/// `GetNeighbourhood` (`Simulation.cpp:2284`): counts of empty and foreign
/// neighbour pixels. Both widen the liquid search.
#[cube]
fn liquid_neighbourhood(pmap: &[u32], ty: u32, xi: i32, yi: i32, xr: i32, yr: i32) -> (u32, u32) {
    let mut nt = 0u32;
    let mut surround_space = 0u32;
    for oy in 0..3usize {
        for ox in 0..3usize {
            if ox != 1usize || oy != 1usize {
                let nx = xi + (ox as i32) - 1i32;
                let ny = yi + (oy as i32) - 1i32;
                let mut nty = 0u32;
                if nx >= 0i32 && nx < xr && ny >= 0i32 && ny < yr {
                    nty = pmap[(ny * xr + nx) as usize] & 0x1FFu32;
                }
                if nty == 0u32 {
                    surround_space += 1u32;
                }
                if nty != ty {
                    nt += 1u32;
                }
            }
        }
    }
    (nt, surround_space)
}

/// Diagonal velocity moves (`Simulation.cpp:3172-3192`).
///
/// Tries `(clear + d)` first, then the perpendicular variant
/// `(dx, dy) -> (dy*r, -dx*r)`. Both probes are `do_move` calls, so
/// `EDGE_LOOP` folds the sampled pixel into the movement band before the
/// obstacle test (`Simulation.cpp:1509-1521`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn diagonal_liquid_move(
    vx: f32,
    vy: f32,
    clear_xf: f32,
    clear_yf: f32,
    rdir: i32,
    props: u32,
    ty: u32,
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    xr: i32,
    yr: i32,
    xcells: u32,
    edge_mode: u32,
) -> u32 {
    let mut ddx = vx - vy * (rdir as f32);
    let mut ddy = vy + vx * (rdir as f32);
    let mut dmv = ddx.abs();
    if ddy.abs() > dmv {
        dmv = ddy.abs();
    }
    let mut target = 0xFFFF_FFFFu32;
    if dmv > 0.0001f32 {
        ddx /= dmv;
        ddy /= dmv;
        let dx1 = round_i32(clear_xf + ddx);
        let dy1 = round_i32(clear_yf + ddy);
        let d1 = do_move_at(
            pmap, wall_bmap, wall_emap, move_lut, props, ty, dx1, dy1, xr, yr, xcells, edge_mode,
        );
        if d1.0 == DO_MOVE_DIE {
            target = DIE_INTENT;
        } else if d1.0 == DO_MOVE_OK {
            target = (d1.2 * xr + d1.1) as u32;
        } else {
            // Perpendicular variant `(dx, dy) -> (dy*r, -dx*r)`.
            let px1 = round_i32(clear_xf + ddy * (rdir as f32));
            let py1 = round_i32(clear_yf - ddx * (rdir as f32));
            let d2 = do_move_at(
                pmap, wall_bmap, wall_emap, move_lut, props, ty, px1, py1, xr, yr, xcells,
                edge_mode,
            );
            if d2.0 == DO_MOVE_DIE {
                target = DIE_INTENT;
            } else if d2.0 == DO_MOVE_OK {
                target = (d2.2 * xr + d2.1) as u32;
            }
        }
    }
    target
}

/// `rt`: 30 normally, 10 when the particle was stagnant last frame and has no
/// foreign neighbour (`Simulation.cpp:3196-3201`).
#[cube]
fn rt_from_flags(flags: u32, nt: u32) -> i32 {
    let mut rt = 30i32;
    if (flags & 0x01u32) != 0u32 && nt == 0u32 {
        rt = 10i32;
    }
    rt
}

/// The shared `MovementPhase` branch for `Falldown >= 1` (liquids and powders).
///
/// Order (`Simulation.cpp:3138-3346`):
/// 1. `do_move(fin)`;
/// 2. `do_move(fin_x, clear_y)` when `fin_x != x`;
/// 3. `do_move(clear_x, fin_y)` when `fin_y != y`;
/// 4. the two diagonal velocity moves, gated on
///    `(clear moved || nt || surround_space) && (|vx| > 0.01 || |vy| > 0.01)`;
/// 5. for liquids only: the vertical scan (`!grav && vertical && vy > |vx|`,
///    with the `GEL` `rt` override) or the perpendicular scan (gated on the
///    gravity alignment) or the `clear` last resort; powders go straight to
///    the `clear` last resort;
/// 6. `clear` last resort, then `FLAG_STAGNANT` is left to the commit pass.
///
/// Every fallback multiplies both velocity components by `Collision`.
///
/// # Integer-position adaptation
///
/// TPT stores float positions, so a sub-pixel velocity advances the particle a
/// fraction of a pixel per frame. This port stores integer pixels, so when
/// `PlanMove` rounds `pos + v` back onto the current pixel the stochastic
/// one-pixel step (`sx`, `sy`, see [`super::velocity`]) is promoted into the
/// plan. Without it a liquid at `vy = 0.4` would never move.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn fall_down_destination(
    vx_in: f32,
    vy_in: f32,
    px: f32,
    py: f32,
    xi: i32,
    yi: i32,
    sx: i32,
    sy: i32,
    props: u32,
    ty: u32,
    falldown: u32,
    flags: u32,
    collision: f32,
    h2: u32,
    gravity_mode: u32,
    newton_grav: u32,
    gravity: f32,
    custom_gravity_x: f32,
    custom_gravity_y: f32,
    gel_tmp: i32,
    edge_mode: u32,
    xres: u32,
    yres: u32,
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    part_ty: &mut [u32],
    xr: i32,
    yr: i32,
    xcells: u32,
) -> (u32, f32, f32) {
    let mut vx = vx_in;
    let mut vy = vy_in;

    let plan = plan_move(
        vx, vy, px, py, xi, yi, props, ty, pmap, wall_bmap, wall_emap, move_lut, xr, yr, xcells,
        edge_mode,
    );
    let mut fin_x = plan.0;
    let mut fin_y = plan.1;
    let mut clear_x = plan.2;
    let mut clear_y = plan.3;
    let mut clear_xf = plan.4;
    let mut clear_yf = plan.5;

    // Integer-position adaptation (see the function docs).
    if fin_x == xi && fin_y == yi && (sx != 0i32 || sy != 0i32) {
        fin_x = xi + sx;
        fin_y = yi + sy;
        clear_x = xi;
        clear_y = yi;
        clear_xf = px;
        clear_yf = py;
    }
    let mut target = 0xFFFF_FFFFu32;
    // `UpdateParticles` skips `MovementPhase` when the velocity is zero
    // (`Simulation.cpp:2404-2406`); with integer positions this is the
    // "no pixel step" case, which leaves `vx`/`vy` undamped.
    if fin_x != xi || fin_y != yi {
        // `do_move(fin)`.
        let d0 = do_move_at(
            pmap, wall_bmap, wall_emap, move_lut, props, ty, fin_x, fin_y, xr, yr, xcells,
            edge_mode,
        );
        if d0.0 == DO_MOVE_DIE {
            target = DIE_INTENT;
        } else if d0.0 == DO_MOVE_OK {
            target = (d0.2 * xr + d0.1) as u32;
        } else {
            // `WOOD -> SAWD` on the rejected probe (`Simulation.cpp:
            // 1211-1218`), before the fallback probes.
            saw_wood_at(pmap, part_ty, d0.1, d0.2, vx, vy, xr, yr);
        }
        // `do_move(fin_x, clear_y)` (`Simulation.cpp:3142`).
        if target == 0xFFFF_FFFFu32 && fin_x != xi {
            let d1 = do_move_at(
                pmap, wall_bmap, wall_emap, move_lut, props, ty, fin_x, clear_y, xr, yr, xcells,
                edge_mode,
            );
            if d1.0 == DO_MOVE_DIE {
                target = DIE_INTENT;
            } else if d1.0 == DO_MOVE_OK {
                target = (d1.2 * xr + d1.1) as u32;
            } else {
                saw_wood_at(pmap, part_ty, d1.1, d1.2, vx, vy, xr, yr);
            }
        }
        // `do_move(clear_x, fin_y)` (`Simulation.cpp:3147`).
        if target == 0xFFFF_FFFFu32 && fin_y != yi {
            let d2 = do_move_at(
                pmap, wall_bmap, wall_emap, move_lut, props, ty, clear_x, fin_y, xr, yr, xcells,
                edge_mode,
            );
            if d2.0 == DO_MOVE_DIE {
                target = DIE_INTENT;
            } else if d2.0 == DO_MOVE_OK {
                target = (d2.2 * xr + d2.1) as u32;
            } else {
                saw_wood_at(pmap, part_ty, d2.1, d2.2, vx, vy, xr, yr);
            }
        }

        // The forward / clear-axis moves do not damp; everything below is a
        // fallback and multiplies both velocity components by `Collision`.
        if target == 0xFFFF_FFFFu32 {
            let counts = liquid_neighbourhood(pmap, ty, xi, yi, xr, yr);
            let nt = counts.0;
            let surround_space = counts.1;
            // `rng.between(0, 1) * 2 - 1`: pick the first search side.
            let rdir = if (h2 & 2u32) == 0u32 { 1i32 } else { -1i32 };
            let clear_moved = clear_x != xi || clear_y != yi;
            let mut moving = false;
            if vx.abs() > 0.01f32 || vy.abs() > 0.01f32 {
                moving = true;
            }
            // Diagonal velocity moves (`Simulation.cpp:3166-3192`).
            if (clear_moved || nt > 0u32 || surround_space > 0u32) && moving {
                target = diagonal_liquid_move(
                    vx, vy, clear_xf, clear_yf, rdir, props, ty, pmap, wall_bmap, wall_emap,
                    move_lut, xr, yr, xcells, edge_mode,
                );
            }

            if target == 0xFFFF_FFFFu32 {
                if falldown > FALLDOWN_LIQUID {
                    // Vertical gravity scan (`Simulation.cpp:3194-3237`).
                    if newton_grav == 0u32 && gravity_mode == 0u32 && vy > vx.abs() {
                        let mut rt = rt_from_flags(flags, nt);
                        // `GEL` narrows the scan to its softness
                        // (`Simulation.cpp:3188-3190`).
                        if ty == PT_GEL_U32 {
                            rt = i32::cast_from((gel_tmp as f32) * 0.20f32 + 5.0f32);
                        }
                        let scan = scan_liquid_vertical(
                            xi, yi, fin_y, clear_x, clear_y, rdir, rt, ty, props, pmap, wall_bmap,
                            wall_emap, move_lut, xr, yr, xcells, edge_mode,
                        );
                        let scan_target = scan.0;
                        if scan_target != 0xFFFF_FFFFu32 {
                            if scan.1 {
                                let scan_x = (scan_target % (xr as u32)) as i32;
                                let scan_y = (scan_target / (xr as u32)) as i32;
                                target = scroll_liquid_down(
                                    scan_x, scan_y, vy, rt, ty, props, pmap, wall_bmap, wall_emap,
                                    move_lut, xr, yr, xcells, edge_mode,
                                );
                            } else {
                                target = scan_target;
                            }
                        }
                    } else {
                        // Perpendicular scan, gated on the gravity alignment
                        // (`Simulation.cpp:3239`):
                        // `|pGravX*vx + pGravY*vy| > |pGravY*vx - pGravX*vy|`.
                        // `pGrav` is the neighbourhood field at the particle
                        // pixel.
                        let g = base_gravity(
                            gravity_mode,
                            gravity,
                            xres,
                            yres,
                            xi as f32,
                            yi as f32,
                            custom_gravity_x,
                            custom_gravity_y,
                        );
                        let mut gnorm = g.0.abs();
                        if g.1.abs() > gnorm {
                            gnorm = g.1.abs();
                        }
                        let mut aligned = false;
                        if gnorm > 0.0001f32 {
                            let gx = g.0 / gnorm;
                            let gy = g.1 / gnorm;
                            if (gx * vx + gy * vy).abs() > (gy * vx - gx * vy).abs() {
                                aligned = true;
                            }
                        }
                        if aligned {
                            let rt = rt_from_flags(flags, nt);
                            target = scan_liquid_perpendicular(
                                clear_xf,
                                clear_yf,
                                clear_x,
                                clear_y,
                                rdir,
                                rt,
                                gravity,
                                gravity_mode,
                                custom_gravity_x,
                                custom_gravity_y,
                                props,
                                ty,
                                xres,
                                yres,
                                pmap,
                                wall_bmap,
                                wall_emap,
                                move_lut,
                                xr,
                                yr,
                                xcells,
                                edge_mode,
                            );
                        }
                    }
                }
                // `clear` last resort (`Simulation.cpp:3178`, `3306`, `3336`).
                // Every failed scan reaches it in the reference.
                if target == 0xFFFF_FFFFu32 && clear_moved {
                    let d = do_move_at(
                        pmap, wall_bmap, wall_emap, move_lut, props, ty, clear_x, clear_y, xr, yr,
                        xcells, edge_mode,
                    );
                    if d.0 == DO_MOVE_DIE {
                        target = DIE_INTENT;
                    } else if d.0 == DO_MOVE_OK {
                        target = (d.2 * xr + d.1) as u32;
                    }
                }
            }

            vx *= collision;
            vy *= collision;
        }
    }
    (target, vx, vy)
}

/// Vertical-gravity lateral scan: step along `clear_x` and stop at the first
/// pixel of a different type or a wall (`Simulation.cpp:3194-3237`).
///
/// Returns `(target, fall)`: `target` is `0xFFFF_FFFF` when nothing was found,
/// [`DIE_INTENT`] when a `do_move` probe left the movement band (the reference
/// reads `s == -1` and skips the vertical continuation), and `fall` requests
/// the vertical continuation down the column. The reported pixel is the
/// *post-`do_move`* position, so an `EDGE_LOOP` probe reports the wrapped
/// pixel the particle actually reached.
#[cube]
#[allow(clippy::too_many_arguments)]
fn scan_liquid_vertical(
    xi: i32,
    yi: i32,
    fin_y: i32,
    clear_x: i32,
    clear_y: i32,
    rdir: i32,
    rt: i32,
    ty: u32,
    props: u32,
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    xr: i32,
    yr: i32,
    xcells: u32,
    edge_mode: u32,
) -> (u32, bool) {
    let mut scan_x = xi;
    let mut scan_y = yi;
    let mut scan_hit = false;
    let mut scan_fall = false;
    let mut scan_die = false;
    let mut j = clear_x + rdir;
    let mut stop = false;
    let mut steps = 0i32;
    while !stop && steps < rt {
        if j < 0i32 || j >= xr || j < clear_x - rt || j >= clear_x + rt {
            stop = true;
        } else {
            // Row `fin_y` first.
            if fin_y >= 0i32 && fin_y < yr {
                let d_val = pmap[(fin_y * xr + j) as usize];
                let d_ty = d_val & 0x1FFu32;
                let wcell = (((fin_y as u32) / 4u32) * xcells + ((j as u32) / 4u32)) as usize;
                if d_ty != ty || wall_bmap[wcell] != 0u32 {
                    let d = do_move_at(
                        pmap, wall_bmap, wall_emap, move_lut, props, ty, j, fin_y, xr, yr, xcells,
                        edge_mode,
                    );
                    if d.0 == DO_MOVE_DIE {
                        scan_die = true;
                        stop = true;
                    } else if d.0 == DO_MOVE_OK {
                        scan_hit = true;
                        scan_fall = true;
                        scan_x = d.1;
                        scan_y = d.2;
                        stop = true;
                    }
                }
            }
            // Then row `clear_y`.
            if !stop && fin_y != clear_y && clear_y >= 0i32 && clear_y < yr {
                let d_val = pmap[(clear_y * xr + j) as usize];
                let d_ty = d_val & 0x1FFu32;
                let wcell = (((clear_y as u32) / 4u32) * xcells + ((j as u32) / 4u32)) as usize;
                if d_ty != ty || wall_bmap[wcell] != 0u32 {
                    let d = do_move_at(
                        pmap, wall_bmap, wall_emap, move_lut, props, ty, j, clear_y, xr, yr,
                        xcells, edge_mode,
                    );
                    if d.0 == DO_MOVE_DIE {
                        scan_die = true;
                        stop = true;
                    } else if d.0 == DO_MOVE_OK {
                        scan_hit = true;
                        scan_fall = true;
                        scan_x = d.1;
                        scan_y = d.2;
                        stop = true;
                    }
                }
            }
            // Stop at the first pixel that is neither the same liquid nor a
            // `WL_STREAM` wall.
            if !stop && clear_y >= 0i32 && clear_y < yr {
                let d_val = pmap[(clear_y * xr + j) as usize];
                let d_ty = d_val & 0x1FFu32;
                let wcell = (((clear_y as u32) / 4u32) * xcells + ((j as u32) / 4u32)) as usize;
                let wall = wall_bmap[wcell];
                if d_ty != ty || (wall != 0u32 && wall != W_STREAM) {
                    stop = true;
                }
            }
            if !stop {
                j += rdir;
                steps += 1i32;
            }
        }
    }
    let mut target = 0xFFFF_FFFFu32;
    if scan_die {
        target = DIE_INTENT;
    } else if scan_hit {
        target = (scan_y * xr + scan_x) as u32;
    }
    (target, scan_fall)
}

/// `Simulation.cpp:3205-3227`: continue down the column until the first pixel
/// that is not the same liquid.
#[cube]
#[allow(clippy::too_many_arguments)]
fn scroll_liquid_down(
    scan_x: i32,
    scan_y: i32,
    vy: f32,
    rt: i32,
    ty: u32,
    props: u32,
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    xr: i32,
    yr: i32,
    xcells: u32,
    edge_mode: u32,
) -> u32 {
    let mut target = (scan_y * xr + scan_x) as u32;
    // Negative literals must not be mutable seeds (see `rdir`).
    let vr = if vy > 0.0f32 { 1i32 } else { -1i32 };
    let mut jj = scan_y + vr;
    let mut stop = false;
    let mut steps = 0i32;
    while !stop && steps < rt {
        if jj < 0i32 || jj >= yr || jj < scan_y - rt || jj >= scan_y + rt {
            stop = true;
        } else {
            let d_val = pmap[(jj * xr + scan_x) as usize];
            let d_ty = d_val & 0x1FFu32;
            let wcell = (((jj as u32) / 4u32) * xcells + ((scan_x as u32) / 4u32)) as usize;
            let wall = wall_bmap[wcell];
            if d_ty != ty || wall != 0u32 {
                let d = do_move_at(
                    pmap, wall_bmap, wall_emap, move_lut, props, ty, scan_x, jj, xr, yr, xcells,
                    edge_mode,
                );
                if d.0 == DO_MOVE_DIE {
                    target = DIE_INTENT;
                    stop = true;
                } else if d.0 == DO_MOVE_OK {
                    target = (d.2 * xr + d.1) as u32;
                    stop = true;
                } else if d_ty != ty || wall != W_STREAM {
                    stop = true;
                }
            }
            if !stop {
                jj += vr;
                steps += 1i32;
            }
        }
    }
    target
}

/// Non-vertical gravity: scan perpendicular to the gravity field
/// (`Simulation.cpp:3238-3330`).
///
/// The gravity field is recomputed at **every** sample
/// (`GetGravityField(nx, ny, ptGrav, 1.0f, ...)`), with the reference's
/// `prev_pGrav` smoothing on the second and later steps, and the vertical
/// continuation also follows the field. The Newtonian component is not
/// available inside this kernel (the movement pass cannot bind the force grids
/// inside its 16-buffer budget), so only the configured `gravityMode` term is
/// recomputed; that is the term that curves in radial gravity.
#[cube]
#[allow(clippy::too_many_arguments)]
fn scan_liquid_perpendicular(
    clear_xf: f32,
    clear_yf: f32,
    clear_x: i32,
    clear_y: i32,
    rdir: i32,
    rt: i32,
    gravity: f32,
    gravity_mode: u32,
    custom_gravity_x: f32,
    custom_gravity_y: f32,
    props: u32,
    ty: u32,
    xres: u32,
    yres: u32,
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    xr: i32,
    yr: i32,
    xcells: u32,
    edge_mode: u32,
) -> u32 {
    let mut target = 0xFFFF_FFFFu32;
    let mut nxf = clear_xf;
    let mut nyf = clear_yf;
    let mut nx = clear_x;
    let mut ny = clear_y;
    let mut prev_gx = 0.0f32;
    let mut prev_gy = 0.0f32;
    let mut moved = false;
    let mut j = 0i32;
    let mut stop = false;
    while !stop && j < rt {
        // `GetGravityField(nx, ny, ptGrav, 1.0f, ...)`.
        let g = base_gravity(
            gravity_mode,
            gravity,
            xres,
            yres,
            nx as f32,
            ny as f32,
            custom_gravity_x,
            custom_gravity_y,
        );
        let mut gx = g.0;
        let mut gy = g.1;
        let mut gmv = gx.abs();
        if gy.abs() > gmv {
            gmv = gy.abs();
        }
        if gmv < 0.0001f32 {
            stop = true;
        } else {
            gx /= gmv;
            gy /= gmv;
            if j > 0i32 {
                // Gravity direction + last change in gravity direction.
                nxf += (rdir as f32) * (gy * 2.0f32 - prev_gy);
                nyf += -(rdir as f32) * (gx * 2.0f32 - prev_gx);
            } else {
                nxf += (rdir as f32) * gy;
                nyf += -(rdir as f32) * gx;
            }
            prev_gx = gx;
            prev_gy = gy;
            nx = round_i32(nxf);
            ny = round_i32(nyf);
            if nx < 0i32 || ny < 0i32 || nx >= xr || ny >= yr {
                stop = true;
            } else {
                let d_val = pmap[(ny * xr + nx) as usize];
                let d_ty = d_val & 0x1FFu32;
                let wcell = (((ny as u32) / 4u32) * xcells + ((nx as u32) / 4u32)) as usize;
                let wall = wall_bmap[wcell];
                if d_ty != ty || wall != 0u32 {
                    let d = do_move_at(
                        pmap, wall_bmap, wall_emap, move_lut, props, ty, nx, ny, xr, yr, xcells,
                        edge_mode,
                    );
                    if d.0 == DO_MOVE_DIE {
                        target = DIE_INTENT;
                        stop = true;
                    } else if d.0 == DO_MOVE_OK {
                        target = (d.2 * xr + d.1) as u32;
                        moved = true;
                        stop = true;
                    } else if d_ty != ty || wall != W_STREAM {
                        stop = true;
                    }
                }
            }
        }
        if !stop {
            j += 1i32;
        }
    }

    if moved {
        // Vertical continuation parallel to gravity (`Simulation.cpp:3300-3330`).
        let mut j2 = 0i32;
        let mut stop2 = false;
        while !stop2 && j2 < rt {
            let g = base_gravity(
                gravity_mode,
                gravity,
                xres,
                yres,
                nx as f32,
                ny as f32,
                custom_gravity_x,
                custom_gravity_y,
            );
            let mut gx = g.0;
            let mut gy = g.1;
            let mut gmv = gx.abs();
            if gy.abs() > gmv {
                gmv = gy.abs();
            }
            if gmv < 0.0001f32 {
                stop2 = true;
            } else {
                gx /= gmv;
                gy /= gmv;
                nxf += gx;
                nyf += gy;
                let nnx = round_i32(nxf);
                let nny = round_i32(nyf);
                if nnx < 0i32 || nny < 0i32 || nnx >= xr || nny >= yr {
                    stop2 = true;
                } else {
                    let d_val = pmap[(nny * xr + nnx) as usize];
                    let d_ty = d_val & 0x1FFu32;
                    let wcell = (((nny as u32) / 4u32) * xcells + ((nnx as u32) / 4u32)) as usize;
                    let wall = wall_bmap[wcell];
                    if d_ty != ty || wall != 0u32 {
                        let d = do_move_at(
                            pmap, wall_bmap, wall_emap, move_lut, props, ty, nnx, nny, xr, yr,
                            xcells, edge_mode,
                        );
                        if d.0 == DO_MOVE_DIE {
                            target = DIE_INTENT;
                            stop2 = true;
                        } else if d.0 == DO_MOVE_OK {
                            target = (d.2 * xr + d.1) as u32;
                            stop2 = true;
                        } else if d_ty != ty || wall != W_STREAM {
                            stop2 = true;
                        }
                    }
                }
            }
            if !stop2 {
                j2 += 1i32;
            }
        }
    }

    target
}

/// `PT_GEL` mirrored for kernel bodies (`Simulation.cpp:3188`).
const PT_GEL_U32: u32 = 150;
const _: () = assert!(PT_GEL_U32 == 150);
