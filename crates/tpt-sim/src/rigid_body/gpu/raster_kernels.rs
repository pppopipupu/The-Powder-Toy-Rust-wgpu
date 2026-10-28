//! Device spatial rasterisation and detachment kernels.

#![allow(clippy::too_many_arguments)]

use super::helpers::*;
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
        let pb = i * 16usize;
        let pid = pixels[pb + 7usize];
        if pid != 4294967295u32 {
            let slot = pid as usize;
            if pixels[pb + 10usize] != 0u32 {
                // Already detached: it no longer owns the slot.
                owner[slot].store(0u32);
            } else {
                let ty = part_ty[slot];
                let elem = pixels[pb + 2usize];
                let flags = part_flags[slot];
                if ty == 0u32 || ty != elem || (flags & 16u32) == 0u32 {
                    pixels[pb + 10usize] = 1u32;
                    part_flags[slot] = flags & 4294967277u32;
                    owner[slot].store(0u32);
                    // The hole puts this pixel's eight neighbours back on the
                    // surface: each of them lost a neighbour and can now meet
                    // a loose particle there. The upload stored their
                    // body-local indices; the body record holds the base.
                    surface_clear(surface, i);
                    // The pixel set changed: the cached mass / centroid /
                    // inertia in `prepare_geometry_kernel` is stale.
                    bodies[pixels[pb + 11usize] as usize * 32usize + 26usize] = 1u32;
                    let base = bodies[pixels[pb + 11usize] as usize * 32usize + 2usize] as usize;
                    let w0 = pixels[pb + 12usize];
                    let w1 = pixels[pb + 13usize];
                    let w2 = pixels[pb + 14usize];
                    let w3 = pixels[pb + 15usize];
                    promote_neighbor_pair(surface, base, w0);
                    promote_neighbor_pair(surface, base, w1);
                    promote_neighbor_pair(surface, base, w2);
                    promote_neighbor_pair(surface, base, w3);
                    if ty != 0u32 {
                        let bi = pixels[pb + 11usize] as usize;
                        let bb = bi * 32usize;
                        // `bodies` is mutable here (the detach sets the
                        // geometry-dirty word), so read the words directly.
                        let cmx = f32::reinterpret(bodies[bb + 4usize]);
                        let cmy = f32::reinterpret(bodies[bb + 5usize]);
                        let bvx = f32::reinterpret(bodies[bb + 6usize]);
                        let bvy = f32::reinterpret(bodies[bb + 7usize]);
                        let omega = f32::reinterpret(bodies[bb + 9usize]);
                        let rx = part_x[slot] - cmx;
                        let ry = part_y[slot] - cmy;
                        part_vx[slot] = bvx - omega * ry;
                        part_vy[slot] = bvy + omega * rx;
                        let gx = pixels[pb + 8usize] as i32;
                        let gy = pixels[pb + 9usize] as i32;
                        if gx >= 0i32 && gy >= 0i32 && gx < xres as i32 && gy < yres as i32 {
                            let idx = (gy as u32 * xres + gx as u32) as usize;
                            if pmap[idx] == 0u32 {
                                pmap[idx] = pack_map(pid, ty);
                            }
                        }
                    }
                } else {
                    pixels[pb + 3usize] = u32::reinterpret(part_temp[slot]);
                    pixels[pb + 4usize] = part_life[slot] as u32;
                    pixels[pb + 5usize] = part_ctype[slot];
                    pixels[pb + 6usize] = part_dcolour[slot];
                    owner[slot].store(pixels[pb + 11usize] + 1u32);
                }
            }
        }
    }
}

/// Clears the surface bit of device pixel `index`.
///
/// The bitmap is shared by every body's thread, so a plain read-modify-write
/// would lose a neighbour promotion that lands in the same word; `fetch_and`
/// makes the clear atomic and therefore order-independent.
#[cube]
fn surface_clear(surface: &mut [Atomic<u32>], index: usize) {
    surface[index / 32usize].fetch_and(!(1u32 << (index % 32usize) as u32));
}

/// Puts one body-local neighbour of a detached pixel back on the surface.
///
/// `local == 0xFFFF` means the neighbour slot was empty at upload, so there is
/// no pixel to promote.
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
        let pb = i * 16usize;
        let pid = pixels[pb + 7usize];
        if pid != 4294967295u32 && pixels[pb + 10usize] == 0u32 {
            let slot = pid as usize;
            let ty = pixels[pb + 2usize];
            let gx = pixels[pb + 8usize] as i32;
            let gy = pixels[pb + 9usize] as i32;
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
///
/// The desired-cell arbitration is a `fetch_min`, so the free marker has to be
/// the largest value rather than zero.
pub(super) const CLAIM_FREE: u32 = u32::MAX;

/// Priority band added to the claim key of a pixel that has to change cell.
///
/// A key is `band * 2^24 + global pixel index`; the pool's id field is 23 bits,
/// so the index fits below the band and the keys stay unique and ordered. A
/// pixel that keeps the cell it already occupies stays in the lower band and
/// therefore wins every cell a mover wants: the mover is the one that yields
/// (its pose was supposed to be stopped at the face), instead of pushing a
/// resting body's pixel out of its own cell.
pub(super) const CLAIM_MOVE_BAND: u32 = 1u32 << 24;
const _: () = assert!(CLAIM_MOVE_BAND == 16_777_216);

/// Claims the desired cell of every rigid pixel for this frame.
///
/// This is a separate dispatch from [`rasterise_kernel`] on purpose: a
/// `fetch_min` only settles once every pixel has published its key, and a pixel
/// can only learn whether it won after that point. Resolving the contested
/// cells with a compare-exchange inside the rasterise pass would hand the cell
/// to whichever thread happened to run first; the lowest key (a stayer, then
/// the lowest global pixel index) winning is what keeps the placement
/// deterministic.
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
        let pb = i * 16usize;
        let pid = pixels[pb + 7usize];
        if pid != 4294967295u32 && pixels[pb + 10usize] == 0u32 {
            let bi = pixels[pb + 11usize] as usize;
            let bb = bi * 32usize;
            if (bodies[bb + 1usize] & 1u32) != 0u32 {
                let angle = fword(bodies, bb + 8usize);
                let origin_x = fword(bodies, bb + 14usize);
                let origin_y = fword(bodies, bb + 15usize);
                let lx = fword(pixels, pb) - origin_x;
                let ly = fword(pixels, pb + 1usize) - origin_y;
                let rx = lx * angle.cos() - ly * angle.sin();
                let ry = lx * angle.sin() + ly * angle.cos();
                let wx = fword(bodies, bb + 4usize) + rx;
                let wy = fword(bodies, bb + 5usize) + ry;
                let gx = clamp_cell(wx.round() as i32, xres);
                let gy = clamp_cell(wy.round() as i32, yres);
                // The previous placement is still in the pixel record: the
                // clear pass frees the cells but does not rewrite the record.
                let mut key = i as u32 + 1u32;
                if gx != pixels[pb + 8usize] as i32 || gy != pixels[pb + 9usize] as i32 {
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

/// Publishes the live cell of every rigid pixel for the current sub-step.
///
/// The pair contact phase samples the receiver through this grid instead of
/// the frame-start rasterisation in `pmap`: two bodies moving toward each other
/// close by their *combined* speed, so a sample against last frame's cells
/// misses the contact entirely and the bodies pass through each other. The
/// generation tag tells a live entry from one left by an earlier sub-step
/// without clearing the grid per sub-step; the frame clears it once before the
/// sub-step loop.
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
        let pb = i * 16usize;
        let pid = pixels[pb + 7usize];
        if pid != 4294967295u32 && pixels[pb + 10usize] == 0u32 {
            let bi = pixels[pb + 11usize] as usize;
            let bb = bi * 32usize;
            if (bodies[bb + 1usize] & 1u32) != 0u32 {
                let angle = fword(bodies, bb + 8usize);
                let origin_x = fword(bodies, bb + 14usize);
                let origin_y = fword(bodies, bb + 15usize);
                let lx = fword(pixels, pb) - origin_x;
                let ly = fword(pixels, pb + 1usize) - origin_y;
                let rx = lx * angle.cos() - ly * angle.sin();
                let ry = lx * angle.sin() + ly * angle.cos();
                let wx = fword(bodies, bb + 4usize) + rx;
                let wy = fword(bodies, bb + 5usize) + ry;
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

/// Claims the first free cell on the ring `radius` around `(cx, cy)`.
///
/// Returns `(claimed, x, y)`. The scan order is fixed (top-left to
/// bottom-right), so the choice does not depend on how the threads are
/// scheduled; the compare-exchange is what makes the claim itself race-free.
///
/// `empty_map` demands that the cell hold no loose particle: a displaced grain
/// may only be moved into an empty cell, while a body pixel is allowed to take
/// a grain's cell and displace the grain afterwards.
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
    let mut found = false;
    let mut bx = cx;
    let mut by = cy;
    for oy_i in 0i32..(2i32 * radius + 1i32) {
        for ox_i in 0i32..(2i32 * radius + 1i32) {
            let ox = ox_i - radius;
            let oy = oy_i - radius;
            if !found && (iabs(ox) == radius || iabs(oy) == radius) {
                let nx = cx + ox;
                let ny = cy + oy;
                if nx >= 0i32 && ny >= 0i32 && nx < xres as i32 && ny < yres as i32 {
                    let ni = (ny as u32 * xres + nx as u32) as usize;
                    if (empty_map == 0u32 || pmap[ni] == 0u32)
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
                        bx = nx;
                        by = ny;
                        found = true;
                    }
                }
            }
        }
    }
    (found, bx, by)
}

/// Sweeps every canvas cell once from `start`, claiming the first free one.
///
/// This is the last placement resort: a body wedged into a fully claimed
/// neighbourhood still has to own a cell of its own. Only a canvas whose every
/// cell carries a claim (a body larger than the canvas) can exhaust it, and
/// nothing can resolve that case.
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
    let mut found = false;
    let mut bx = 0i32;
    let mut by = 0i32;
    if cells > 0u32 {
        let mut probe = start % cells;
        for _k in 0usize..cells as usize {
            if !found {
                let nx = (probe % xres) as i32;
                let ny = (probe / xres) as i32;
                if pmap[probe as usize] == 0u32
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
                }
                probe = (probe + 1u32) % cells;
            }
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
        let pb = i * 16usize;
        let pid = pixels[pb + 7usize];
        if pid != 4294967295u32 && pixels[pb + 10usize] == 0u32 {
            let bi = pixels[pb + 11usize] as usize;
            let bb = bi * 32usize;
            if (bodies[bb + 1usize] & 1u32) != 0u32 {
                let elem = pixels[pb + 2usize];
                let cmx = fword(bodies, bb + 4usize);
                let cmy = fword(bodies, bb + 5usize);
                let angle = fword(bodies, bb + 8usize);
                let vx = fword(bodies, bb + 6usize);
                let vy = fword(bodies, bb + 7usize);
                let omega = fword(bodies, bb + 9usize);
                let origin_x = fword(bodies, bb + 14usize);
                let origin_y = fword(bodies, bb + 15usize);
                let lx = fword_m(pixels, pb) - origin_x;
                let ly = fword_m(pixels, pb + 1usize) - origin_y;
                let rx = lx * angle.cos() - ly * angle.sin();
                let ry = lx * angle.sin() + ly * angle.cos();
                let wx = cmx + rx;
                let wy = cmy + ry;
                let vpx = vx - omega * ry;
                let vpy = vy + omega * rx;

                let mut gx = wx.round() as i32;
                let mut gy = wy.round() as i32;
                if gx < 0i32 {
                    gx = 0i32;
                }
                if gy < 0i32 {
                    gy = 0i32;
                }
                if gx > xres as i32 - 1i32 {
                    gx = xres as i32 - 1i32;
                }
                if gy > yres as i32 - 1i32 {
                    gy = yres as i32 - 1i32;
                }

                // The previous placement is still in the pixel record: the
                // claim pass used it to choose the priority band, and the
                // fallback below needs it, so read it before the placement
                // overwrites the words.
                let old_gx = pixels[pb + 8usize] as i32;
                let old_gy = pixels[pb + 9usize] as i32;
                let mut moving = 0u32;
                if gx != old_gx || gy != old_gy {
                    moving = 1u32;
                }
                let key = claim_key(i as u32, moving);
                let wanted = (gy as u32 * xres + gx as u32) as usize;
                // `claim_cells_kernel` published every desired cell with
                // `fetch_min`, so the lowest key owns a contested cell: a
                // pixel that keeps its cell beats any mover, and the global
                // pixel index breaks the remaining ties. A pixel that did not
                // win has to find a cell nobody else claimed, which the
                // compare-exchange below guarantees.
                let mut placed = claims[wanted].load() == key;
                for r in 0i32..2i32 {
                    if !placed {
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
                        }
                    }
                }
                if !placed {
                    // Before widening the search, try the cell this pixel
                    // stands in. The clear pass released it, so it is free
                    // unless another pixel claimed it this frame; staying
                    // there is exactly "stop at the face" for a mover whose
                    // pose was not corrected out of the receiver's cells.
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
                        if claims[oi].compare_exchange_weak(CLAIM_FREE, key) == CLAIM_FREE {
                            gx = ox;
                            gy = oy;
                            placed = true;
                        }
                    }
                }
                for r in 2i32..8i32 {
                    if !placed {
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
                        }
                    }
                }
                if !placed {
                    // Every claim is a distinct cell, so a free cell exists
                    // unless the canvas itself is full; sweep it rather than
                    // dropping the pixel on top of another body's cell. The
                    // sweep is the only case that still overlaps a pixel when
                    // the body does not fit on the canvas at all.
                    let scan = claim_scan(
                        claims, pmap, key, i as u32, xres, yres, wall_bmap, wall_emap, xcells,
                        ycells,
                    );
                    if scan.0 {
                        gx = scan.1;
                        gy = scan.2;
                    }
                }

                pixels[pb + 8usize] = gx as u32;
                pixels[pb + 9usize] = gy as u32;

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
                            // The grain's destination is claimed exactly like
                            // a body pixel's cell. Without the claim the grain
                            // could be moved into a cell another body's pixel
                            // has already claimed for this frame, and the two
                            // `pmap` writes would race out of it.
                            let mut bx = ogx;
                            let mut by = ogy;
                            let mut found = false;
                            // Rings 1..=8, unlike the host solver's 3x3: a
                            // grain buried under the middle of a body has no
                            // free cell beside it (the body's own cells are
                            // claimed), so a two-cell search used to give up
                            // and leave the grain out of the map. A grain may
                            // only move into a cell that holds no particle, so
                            // the ring scan is asked for an empty map cell.
                            for r in 0i32..8i32 {
                                if !found {
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
                                    }
                                }
                            }
                            if !found {
                                // A fully packed neighbourhood: fall back to
                                // the cell this body pixel is vacating, which
                                // the clear pass freed earlier this frame.
                                // Another pixel may have claimed it since.
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
                                // Every claim is a distinct cell, so an empty
                                // one exists unless the whole canvas is
                                // occupied; sweep it instead of writing the
                                // grain over a cell that already has an owner.
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
                                // Not reachable unless the canvas is entirely
                                // claimed: the grain keeps its cell, which the
                                // body pixel is about to overwrite.
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
