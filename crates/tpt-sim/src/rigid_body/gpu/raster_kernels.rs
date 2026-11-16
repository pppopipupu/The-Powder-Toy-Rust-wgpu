//! Device spatial rasterisation and detachment kernels.

#![allow(clippy::too_many_arguments, clippy::collapsible_if)]

use super::helpers::*;
use super::layout::*;
use cubecl::prelude::*;

/// One thread per rigid pixel: re-read the authoritative particle state and
/// mark the pixel detached in the device mask when it is no longer the same
/// element (melting, burning, corrosion, explosion) or lost its rigid flag.
/// A detachment also clears the pixel's bit in the contact pass's surface
/// bitmap and sets its eight neighbours' bits: the hole exposes a new surface.
#[cube(launch_unchecked)]
pub(super) fn reconcile_kernel(
    bodies: &mut [u32],
    pixels: &mut [u32],
    surface: &mut [Atomic<u32>],
    pmap: &mut [u32],
    part_ty: &[u32],
    part_temp: &[f32],
    part_life: &[i32],
    part_ctype: &[u32],
    part_dcolour: &[u32],
    part_flags: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    owner: &mut [Atomic<u32>],
    total: u32,
    xres: u32,
    yres: u32,
) {
    let i = ABSOLUTE_POS;
    if i < total as usize {
        let pb = i * P_STRIDE;
        let pid = pixels[pb + P_PID];
        if pid != NO_PID {
            let slot = pid as usize;
            if pixels[pb + P_DETACHED] != 0u32 {
                // Already detached: it no longer owns the slot.
                owner[slot].store(0u32);
            } else {
                let ty = part_ty[slot];
                let elem = pixels[pb + P_ELEM];
                let flags = part_flags[slot];
                if ty == 0u32 || ty != elem || (flags & 16u32) == 0u32 {
                    pixels[pb + P_DETACHED] = 1u32;
                    part_flags[slot] = flags & 4294967277u32;
                    owner[slot].store(0u32);
                    // The hole puts this pixel's eight neighbours back on the
                    // surface: each of them lost a neighbour and can now meet
                    // a loose particle there.
                    surface_clear(surface, i);
                    let bi = pixels[pb + P_BODY] as usize;
                    let bb = bi * B_STRIDE;
                    // The pixel set changed: rebuild cached mass / centroid /
                    // inertia / surface_list in `prepare_geometry_kernel`.
                    bodies[bb + B_GEOM_DIRTY] = 1u32;
                    let base = bodies[bb + B_PIXEL_START] as usize;
                    let w0 = pixels[pb + P_NEIGHBORS_0];
                    let w1 = pixels[pb + P_NEIGHBORS_1];
                    let w2 = pixels[pb + P_NEIGHBORS_2];
                    let w3 = pixels[pb + P_NEIGHBORS_3];
                    promote_neighbor_pair(surface, base, w0);
                    promote_neighbor_pair(surface, base, w1);
                    promote_neighbor_pair(surface, base, w2);
                    promote_neighbor_pair(surface, base, w3);
                    if ty != 0u32 {
                        let cmx = f32::reinterpret(bodies[bb + B_CM_X]);
                        let cmy = f32::reinterpret(bodies[bb + B_CM_Y]);
                        let bvx = f32::reinterpret(bodies[bb + B_VX]);
                        let bvy = f32::reinterpret(bodies[bb + B_VY]);
                        let omega = f32::reinterpret(bodies[bb + B_OMEGA]);
                        let rx = part_x[slot] - cmx;
                        let ry = part_y[slot] - cmy;
                        part_vx[slot] = bvx - omega * ry;
                        part_vy[slot] = bvy + omega * rx;
                        let gx = pixels[pb + P_GRID_X] as i32;
                        let gy = pixels[pb + P_GRID_Y] as i32;
                        if gx >= 0i32 && gy >= 0i32 && gx < xres as i32 && gy < yres as i32 {
                            let idx = (gy as u32 * xres + gx as u32) as usize;
                            if pmap[idx] == 0u32 {
                                pmap[idx] = pack_map(pid, ty);
                            }
                        }
                    }
                } else {
                    pixels[pb + P_TEMP] = u32::reinterpret(part_temp[slot]);
                    pixels[pb + P_LIFE] = part_life[slot] as u32;
                    pixels[pb + P_CTYPE] = part_ctype[slot];
                    pixels[pb + P_DCOLOUR] = part_dcolour[slot];
                    owner[slot].store(pixels[pb + P_BODY] + 1u32);
                }
            }
        }
    }
}

/// Clears the surface bit of device pixel `index`.
#[cube]
fn surface_clear(surface: &mut [Atomic<u32>], index: usize) {
    surface[index / 32usize].fetch_and(!(1u32 << (index % 32usize) as u32));
}

/// Puts one body-local neighbour of a detached pixel back on the surface.
#[cube]
fn promote_neighbor(surface: &mut [Atomic<u32>], base: usize, local: u32) {
    if local != 65535u32 {
        let index = base + local as usize;
        surface[index / 32usize].fetch_or(1u32 << (index % 32usize) as u32);
    }
}

/// [`promote_neighbor`] for both `u16`s packed into one neighbour word.
#[cube]
fn promote_neighbor_pair(surface: &mut [Atomic<u32>], base: usize, word: u32) {
    promote_neighbor(surface, base, word & 65535u32);
    promote_neighbor(surface, base, (word >> 16u32) & 65535u32);
}

#[cube(launch_unchecked)]
pub(super) fn clear_rigid_kernel(
    pixels: &[u32],
    pmap: &mut [u32],
    part_ty: &mut [u32],
    part_flags: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    owner: &mut [Atomic<u32>],
    total: u32,
    xres: u32,
    yres: u32,
) {
    let i = ABSOLUTE_POS;
    if i < total as usize {
        let pb = i * P_STRIDE;
        let pid = pixels[pb + P_PID];
        if pid != NO_PID && pixels[pb + P_DETACHED] == 0u32 {
            let slot = pid as usize;
            let ty = pixels[pb + P_ELEM];
            let gx = pixels[pb + P_GRID_X] as i32;
            let gy = pixels[pb + P_GRID_Y] as i32;
            if gx >= 0i32 && gy >= 0i32 && gx < xres as i32 && gy < yres as i32 {
                let idx = (gy as u32 * xres + gx as u32) as usize;
                if pmap[idx] == pack_map(pid, ty) {
                    pmap[idx] = 0u32;
                }
            }
            part_ty[slot] = 0u32;
            part_flags[slot] = 0u32;
            part_vx[slot] = 0.0f32;
            part_vy[slot] = 0.0f32;
            owner[slot].store(0u32);
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn clear_claims_kernel(claims: &mut [Atomic<u32>], total: u32) {
    let i = ABSOLUTE_POS;
    if i < total as usize {
        claims[i].store(CLAIM_FREE);
    }
}

/// A cell no rigid pixel or displaced grain has claimed in this frame.
pub(super) const CLAIM_FREE: u32 = u32::MAX;

/// Priority band added to the claim key of a pixel that has to change cell.
pub(super) const CLAIM_MOVE_BAND: u32 = 1u32 << 24;
const _: () = assert!(CLAIM_MOVE_BAND == 16_777_216);

/// Claims the desired cell of every rigid pixel for this frame.
#[cube(launch_unchecked)]
pub(super) fn claim_cells_kernel(
    bodies: &[u32],
    pixels: &[u32],
    claims: &mut [Atomic<u32>],
    total: u32,
    xres: u32,
    yres: u32,
) {
    let i = ABSOLUTE_POS;
    if i < total as usize {
        let pb = i * P_STRIDE;
        let pid = pixels[pb + P_PID];
        if pid != NO_PID && pixels[pb + P_DETACHED] == 0u32 {
            let bi = pixels[pb + P_BODY] as usize;
            let bb = bi * B_STRIDE;
            if (bodies[bb + B_FLAGS] & 1u32) != 0u32 {
                let cos_a = fword(bodies, bb + B_COS_A);
                let sin_a = fword(bodies, bb + B_SIN_A);
                let origin_x = fword(bodies, bb + B_ORIGIN_X);
                let origin_y = fword(bodies, bb + B_ORIGIN_Y);
                let lx = fword(pixels, pb + P_LOCAL_X) - origin_x;
                let ly = fword(pixels, pb + P_LOCAL_Y) - origin_y;
                let rx = lx * cos_a - ly * sin_a;
                let ry = lx * sin_a + ly * cos_a;
                let wx = fword(bodies, bb + B_CM_X) + rx;
                let wy = fword(bodies, bb + B_CM_Y) + ry;
                let gx = clamp_cell(wx.round() as i32, xres);
                let gy = clamp_cell(wy.round() as i32, yres);
                let mut key = i as u32 + 1u32;
                if gx != pixels[pb + P_GRID_X] as i32 || gy != pixels[pb + P_GRID_Y] as i32 {
                    key += CLAIM_MOVE_BAND;
                }
                claims[(gy as u32 * xres + gx as u32) as usize].fetch_min(key);
            }
        }
    }
}

/// Clamps a rounded world coordinate to the canvas.
#[cube]
fn clamp_cell(v: i32, limit: u32) -> i32 {
    let mut out = v;
    if out < 0i32 {
        out = 0i32;
    }
    if out > limit as i32 - 1i32 {
        out = limit as i32 - 1i32;
    }
    out
}

/// Publishes the live cell of every pair-active rigid pixel for the current sub-step.
///
/// Bodies without a broadphase pair candidate (`B_HAS_PAIR == 0`) are skipped
/// because no other body can reach their cells during this tick.
#[cube(launch_unchecked)]
pub(super) fn scatter_live_kernel(
    bodies: &[u32],
    pixels: &[u32],
    live: &mut [Atomic<u32>],
    total: u32,
    xres: u32,
    yres: u32,
    generation: u32,
) {
    let i = ABSOLUTE_POS;
    if i < total as usize {
        let pb = i * P_STRIDE;
        let pid = pixels[pb + P_PID];
        if pid != NO_PID && pixels[pb + P_DETACHED] == 0u32 {
            let bi = pixels[pb + P_BODY] as usize;
            let bb = bi * B_STRIDE;
            if (bodies[bb + B_FLAGS] & 1u32) != 0u32 && bodies[bb + B_HAS_PAIR] != 0u32 {
                let cos_a = fword(bodies, bb + B_COS_A);
                let sin_a = fword(bodies, bb + B_SIN_A);
                let origin_x = fword(bodies, bb + B_ORIGIN_X);
                let origin_y = fword(bodies, bb + B_ORIGIN_Y);
                let lx = fword(pixels, pb + P_LOCAL_X) - origin_x;
                let ly = fword(pixels, pb + P_LOCAL_Y) - origin_y;
                let rx = lx * cos_a - ly * sin_a;
                let ry = lx * sin_a + ly * cos_a;
                let wx = fword(bodies, bb + B_CM_X) + rx;
                let wy = fword(bodies, bb + B_CM_Y) + ry;
                let gx = clamp_cell(wx.round() as i32, xres);
                let gy = clamp_cell(wy.round() as i32, yres);
                let tag = (generation << 23u32) | ((pid + 1u32) & 8388607u32);
                live[(gy as u32 * xres + gx as u32) as usize].store(tag);
            }
        }
    }
}

/// The claim key of one rigid pixel: the global index inside its band.
#[cube]
fn claim_key(index: u32, moving: u32) -> u32 {
    let mut key = index + 1u32;
    if moving != 0u32 {
        key += 16777216u32;
    }
    key
}

/// Attempts to claim a single candidate cell `(nx, ny)` in `claim_ring`.
#[cube]
fn try_claim_cell(
    claims: &mut [Atomic<u32>],
    pmap: &mut [u32],
    key: u32,
    nx: i32,
    ny: i32,
    xres: u32,
    yres: u32,
    wall_bmap: &[u32],
    wall_emap: &[u32],
    xcells: u32,
    ycells: u32,
    empty_map: u32,
) -> bool {
    let mut ok = seed_u32(nx as u32, 0u32) != 0u32;
    if nx >= 0i32 && ny >= 0i32 && nx < xres as i32 && ny < yres as i32 {
        let ni = (ny as u32 * xres + nx as u32) as usize;
        if claims[ni].load() == CLAIM_FREE
            && (empty_map == 0u32 || pmap[ni] == 0u32)
            && !wall_blocks(
                wall_bmap,
                wall_emap,
                nx,
                ny,
                xcells as i32,
                ycells as i32,
                4u32,
            )
            && claims[ni].compare_exchange_weak(CLAIM_FREE, key) == CLAIM_FREE
        {
            ok = true;
        }
    }
    ok
}

/// Claims the first free cell on the perimeter of the ring `radius` around `(cx, cy)`.
///
/// Walks only the 4 ring edges (`8 * radius` cells) in top-left to bottom-right
/// order instead of scanning the `(2 * radius + 1)^2` interior, and breaks as
/// soon as a cell is claimed.
#[cube]
fn claim_ring(
    claims: &mut [Atomic<u32>],
    pmap: &mut [u32],
    key: u32,
    cx: i32,
    cy: i32,
    radius: i32,
    xres: u32,
    yres: u32,
    wall_bmap: &[u32],
    wall_emap: &[u32],
    xcells: u32,
    ycells: u32,
    empty_map: u32,
) -> (bool, i32, i32) {
    let mut found = seed_u32(radius as u32, 0u32) != 0u32;
    let mut bx = cx;
    let mut by = cy;
    let span = 2i32 * radius + 1i32;

    // Top edge (oy = -radius, ox = -radius ..= +radius).
    let top_y = cy - radius;
    for ox_i in 0i32..span {
        let nx = cx + (ox_i - radius);
        if try_claim_cell(
            claims, pmap, key, nx, top_y, xres, yres, wall_bmap, wall_emap, xcells, ycells,
            empty_map,
        ) {
            bx = nx;
            by = top_y;
            found = true;
            break;
        }
    }

    // Middle rows (oy = -radius + 1 ..= +radius - 1, left then right edge).
    if !found && radius > 0i32 {
        let mid_rows = 2i32 * radius - 1i32;
        for row_i in 0i32..mid_rows {
            let ny = cy - radius + 1i32 + row_i;
            let lx = cx - radius;
            if try_claim_cell(
                claims, pmap, key, lx, ny, xres, yres, wall_bmap, wall_emap, xcells, ycells,
                empty_map,
            ) {
                bx = lx;
                by = ny;
                found = true;
                break;
            }
            let rx = cx + radius;
            if try_claim_cell(
                claims, pmap, key, rx, ny, xres, yres, wall_bmap, wall_emap, xcells, ycells,
                empty_map,
            ) {
                bx = rx;
                by = ny;
                found = true;
                break;
            }
        }
    }

    // Bottom edge (oy = +radius, ox = -radius ..= +radius).
    if !found && radius > 0i32 {
        let bot_y = cy + radius;
        for ox_i in 0i32..span {
            let nx = cx + (ox_i - radius);
            if try_claim_cell(
                claims, pmap, key, nx, bot_y, xres, yres, wall_bmap, wall_emap, xcells, ycells,
                empty_map,
            ) {
                bx = nx;
                by = bot_y;
                found = true;
                break;
            }
        }
    }

    (found, bx, by)
}

/// Sweeps every canvas cell once from `start`, claiming the first free one.
#[cube]
fn claim_scan(
    claims: &mut [Atomic<u32>],
    pmap: &mut [u32],
    key: u32,
    start: u32,
    xres: u32,
    yres: u32,
    wall_bmap: &[u32],
    wall_emap: &[u32],
    xcells: u32,
    ycells: u32,
) -> (bool, i32, i32) {
    let cells = xres * yres;
    let mut found = seed_u32(start, 0u32) != 0u32;
    let mut bx = seed_u32(start, 0u32) as i32;
    let mut by = seed_u32(start, 0u32) as i32;
    if cells > 0u32 {
        let mut probe = start % cells;
        for _k in 0usize..cells as usize {
            let nx = (probe % xres) as i32;
            let ny = (probe / xres) as i32;
            if claims[probe as usize].load() == CLAIM_FREE
                && pmap[probe as usize] == 0u32
                && !wall_blocks(
                    wall_bmap,
                    wall_emap,
                    nx,
                    ny,
                    xcells as i32,
                    ycells as i32,
                    4u32,
                )
                && claims[probe as usize].compare_exchange_weak(CLAIM_FREE, key) == CLAIM_FREE
            {
                bx = nx;
                by = ny;
                found = true;
                break;
            }
            probe = (probe + 1u32) % cells;
        }
    }
    (found, bx, by)
}

#[cube(launch_unchecked)]
pub(super) fn rasterise_kernel(
    bodies: &[u32],
    pixels: &mut [u32],
    pmap: &mut [u32],
    part_ty: &mut [u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &mut [u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    claims: &mut [Atomic<u32>],
    owner: &mut [Atomic<u32>],
    total: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
) {
    let i = ABSOLUTE_POS;
    if i < total as usize {
        let pb = i * P_STRIDE;
        let pid = pixels[pb + P_PID];
        if pid != NO_PID && pixels[pb + P_DETACHED] == 0u32 {
            let bi = pixels[pb + P_BODY] as usize;
            let bb = bi * B_STRIDE;
            if (bodies[bb + B_FLAGS] & 1u32) != 0u32 {
                let elem = pixels[pb + P_ELEM];
                let cmx = fword(bodies, bb + B_CM_X);
                let cmy = fword(bodies, bb + B_CM_Y);
                let cos_a = fword(bodies, bb + B_COS_A);
                let sin_a = fword(bodies, bb + B_SIN_A);
                let vx = fword(bodies, bb + B_VX);
                let vy = fword(bodies, bb + B_VY);
                let omega = fword(bodies, bb + B_OMEGA);
                let origin_x = fword(bodies, bb + B_ORIGIN_X);
                let origin_y = fword(bodies, bb + B_ORIGIN_Y);
                let lx = fword_m(pixels, pb + P_LOCAL_X) - origin_x;
                let ly = fword_m(pixels, pb + P_LOCAL_Y) - origin_y;
                let rx = lx * cos_a - ly * sin_a;
                let ry = lx * sin_a + ly * cos_a;
                let wx = cmx + rx;
                let wy = cmy + ry;
                let vpx = vx - omega * ry;
                let vpy = vy + omega * rx;

                let mut gx = clamp_cell(wx.round() as i32, xres);
                let mut gy = clamp_cell(wy.round() as i32, yres);

                let old_gx = pixels[pb + P_GRID_X] as i32;
                let old_gy = pixels[pb + P_GRID_Y] as i32;
                let mut moving = seed_u32(elem, 0u32);
                if gx != old_gx || gy != old_gy {
                    moving = 1u32;
                }
                let key = claim_key(i as u32, moving);
                let wanted = (gy as u32 * xres + gx as u32) as usize;
                let mut placed = claims[wanted].load() == key;
                for r in 0i32..2i32 {
                    if placed {
                        break;
                    }
                    let ring = claim_ring(
                        claims,
                        pmap,
                        key,
                        gx,
                        gy,
                        r + 1i32,
                        xres,
                        yres,
                        wall_bmap,
                        wall_emap,
                        xcells,
                        ycells,
                        0u32,
                    );
                    if ring.0 {
                        gx = ring.1;
                        gy = ring.2;
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    let ox = old_gx;
                    let oy = old_gy;
                    if (ox != gx || oy != gy)
                        && ox >= 0i32
                        && oy >= 0i32
                        && ox < xres as i32
                        && oy < yres as i32
                        && !wall_blocks(
                            wall_bmap,
                            wall_emap,
                            ox,
                            oy,
                            xcells as i32,
                            ycells as i32,
                            4u32,
                        )
                    {
                        let oi = (oy as u32 * xres + ox as u32) as usize;
                        if claims[oi].load() == CLAIM_FREE
                            && claims[oi].compare_exchange_weak(CLAIM_FREE, key) == CLAIM_FREE
                        {
                            gx = ox;
                            gy = oy;
                            placed = true;
                        }
                    }
                }
                for r in 2i32..8i32 {
                    if placed {
                        break;
                    }
                    let ring = claim_ring(
                        claims,
                        pmap,
                        key,
                        gx,
                        gy,
                        r + 1i32,
                        xres,
                        yres,
                        wall_bmap,
                        wall_emap,
                        xcells,
                        ycells,
                        0u32,
                    );
                    if ring.0 {
                        gx = ring.1;
                        gy = ring.2;
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    let scan = claim_scan(
                        claims, pmap, key, i as u32, xres, yres, wall_bmap, wall_emap, xcells,
                        ycells,
                    );
                    if scan.0 {
                        gx = scan.1;
                        gy = scan.2;
                    }
                }

                pixels[pb + P_GRID_X] = gx as u32;
                pixels[pb + P_GRID_Y] = gy as u32;

                let ci = (gy as u32 * xres + gx as u32) as usize;

                // Reclaim the cell from a loose particle if one sits there.
                let pm = pmap[ci];
                if pm != 0u32 {
                    let other = unpack_slot(pm);
                    if other != 4294967295u32 && other != pid {
                        let oslot = other as usize;
                        let own = owner[oslot].load();
                        if own == 0u32 && part_ty[oslot] != 0u32 {
                            let other_ty = part_ty[oslot];
                            let mut dir_x = part_x[oslot] - cmx;
                            let mut dir_y = part_y[oslot] - cmy;
                            let len = (dir_x * dir_x + dir_y * dir_y).sqrt();
                            if len > 1e-4f32 {
                                dir_x /= len;
                                dir_y /= len;
                            } else {
                                dir_x = 0.0f32;
                                dir_y = 1.0f32;
                            }
                            let ogx = part_x[oslot].round() as i32;
                            let ogy = part_y[oslot].round() as i32;
                            let mut bx = ogx;
                            let mut by = ogy;
                            let mut found = seed_u32(other, 0u32) != 0u32;
                            for r in 0i32..8i32 {
                                let ring = claim_ring(
                                    claims,
                                    pmap,
                                    key,
                                    ogx,
                                    ogy,
                                    r + 1i32,
                                    xres,
                                    yres,
                                    wall_bmap,
                                    wall_emap,
                                    xcells,
                                    ycells,
                                    1u32,
                                );
                                if ring.0 {
                                    bx = ring.1;
                                    by = ring.2;
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                let pgx = old_gx;
                                let pgy = old_gy;
                                if (pgx != ogx || pgy != ogy)
                                    && pgx >= 0i32
                                    && pgy >= 0i32
                                    && pgx < xres as i32
                                    && pgy < yres as i32
                                {
                                    let pi = (pgy as u32 * xres + pgx as u32) as usize;
                                    if pmap[pi] == 0u32
                                        && claims[pi].load() == CLAIM_FREE
                                        && !wall_blocks(
                                            wall_bmap,
                                            wall_emap,
                                            pgx,
                                            pgy,
                                            xcells as i32,
                                            ycells as i32,
                                            4u32,
                                        )
                                        && claims[pi].compare_exchange_weak(CLAIM_FREE, key)
                                            == CLAIM_FREE
                                    {
                                        bx = pgx;
                                        by = pgy;
                                        found = true;
                                    }
                                }
                            }
                            if !found {
                                let scan = claim_scan(
                                    claims,
                                    pmap,
                                    key,
                                    ogy as u32 * xres + ogx as u32,
                                    xres,
                                    yres,
                                    wall_bmap,
                                    wall_emap,
                                    xcells,
                                    ycells,
                                );
                                if scan.0 {
                                    bx = scan.1;
                                    by = scan.2;
                                    found = true;
                                }
                            }
                            if found {
                                let old = (ogy as u32 * xres + ogx as u32) as usize;
                                if pmap[old] == pack_map(other, other_ty) {
                                    pmap[old] = 0u32;
                                }
                                part_x[oslot] = bx as f32;
                                part_y[oslot] = by as f32;
                                part_vx[oslot] = vpx + dir_x * 0.5f32;
                                part_vy[oslot] = vpy + dir_y * 0.5f32;
                                let ni = (by as u32 * xres + bx as u32) as usize;
                                pmap[ni] = pack_map(other, other_ty);
                            } else {
                                part_vx[oslot] += dir_x * 0.5f32;
                                part_vy[oslot] += dir_y * 0.5f32;
                            }
                        }
                    }
                }

                let slot = pid as usize;
                part_ty[slot] = elem;
                part_x[slot] = gx as f32;
                part_y[slot] = gy as f32;
                part_vx[slot] = vpx;
                part_vy[slot] = vpy;
                part_flags[slot] = 18u32;
                owner[slot].store(bi as u32 + 1u32);
                pmap[ci] = pack_map(pid, elem);
            }
        }
    }
}
