//! Rigid-body rasterisation: map the rotated cluster back onto the pixel
//! grid, reclaim cells from loose particles and close diagonal gaps.

use std::collections::HashSet;

use super::contacts::displace_free_particle;
use super::*;

/// Is the cell free for a rigid pixel?  Cells occupied by rigid pixels are not
/// available; cells occupied by ordinary particles may be reclaimed (the free
/// particle is pushed away).
fn cell_available(bufs: &SimBuffers, gx: i32, gy: i32, xres: i32, yres: i32) -> bool {
    if !in_bounds(gx, gy, xres, yres) {
        return false;
    }
    let idx = grid_index(gx, gy, xres);
    if idx >= bufs.pmap.len() {
        return false;
    }
    let pm = bufs.pmap[idx];
    if pm == 0 {
        return true;
    }
    match pmap_id(pm) {
        Some(id) => {
            let id = id as usize;
            if !particle_alive(bufs, id) {
                return true;
            }
            bufs.part_flags[id] & FLAG_RIGID_PIXEL == 0
        }
        None => true,
    }
}

/// Find a placement cell for a rigid pixel, closing sub-pixel diagonal gaps and
/// avoiding cells already claimed by another rigid body.
fn find_place_cell(
    bufs: &SimBuffers,
    claimed: &HashSet<i32>,
    gx: i32,
    gy: i32,
    xres: i32,
    yres: i32,
) -> (i32, i32) {
    if !claimed.contains(&cell_key(gx, gy, xres)) && cell_available(bufs, gx, gy, xres, yres) {
        return (gx, gy);
    }

    let mut best: Option<(i32, i32, f32)> = None;
    for radius in 1..=2i32 {
        for oy in -radius..=radius {
            for ox in -radius..=radius {
                if ox.abs() != radius && oy.abs() != radius {
                    continue;
                }
                let nx = gx + ox;
                let ny = gy + oy;
                if !in_bounds(nx, ny, xres, yres) {
                    continue;
                }
                if claimed.contains(&cell_key(nx, ny, xres)) {
                    continue;
                }
                if !cell_available(bufs, nx, ny, xres, yres) {
                    continue;
                }
                let d = (ox * ox + oy * oy) as f32;
                if best.is_none_or(|(_, _, bd)| d < bd) {
                    best = Some((nx, ny, d));
                }
            }
        }
        if best.is_some() {
            break;
        }
    }

    match best {
        Some((nx, ny, _)) => (nx, ny),
        // Nowhere to go: overlap is preferable to losing a pixel.
        None => (gx, gy),
    }
}

pub(super) fn rasterize_body(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    body: &mut RigidBody2D,
    ids: &[Option<usize>],
) {
    let xres = xres_of(bufs);
    let yres = yres_of(bufs);
    if xres <= 0 || yres <= 0 {
        return;
    }
    let cos_a = body.angle.cos();
    let sin_a = body.angle.sin();

    let mut claimed: HashSet<i32> = HashSet::with_capacity(body.pixels.len() * 2);
    let mut prev_cell: Option<(i32, i32)> = None;
    let mut prev_local_y = f32::NAN;

    for i in 0..body.pixels.len() {
        let px = body.pixels[i];

        // New local row -> do not bridge across rows.
        if (px.local_y - prev_local_y).abs() > 0.01 {
            prev_cell = None;
        }
        prev_local_y = px.local_y;

        let (rx, ry) = rotate_local(px.local_x, px.local_y, cos_a, sin_a);
        let wx = body.cm_x + rx;
        let wy = body.cm_y + ry;
        let vpx = body.vx - body.omega * ry;
        let vpy = body.vy + body.omega * rx;

        if !wx.is_finite() || !wy.is_finite() {
            continue;
        }

        let mut gx = wx.round().clamp(0.0, (xres - 1) as f32) as i32;
        let mut gy = wy.round().clamp(0.0, (yres - 1) as f32) as i32;

        // Close the diagonal gap between this pixel and its left neighbour in
        // body space.  Rounding two 4-connected pixels can produce a diagonal
        // step; snapping one coordinate keeps the cluster watertight.
        if let Some((pgx, pgy)) = prev_cell
            && (gx - pgx).abs() == 1
            && (gy - pgy).abs() == 1
        {
            let cand_a = (pgx, gy);
            let cand_b = (gx, pgy);
            let da = (cand_a.0 as f32 - wx).powi(2) + (cand_a.1 as f32 - wy).powi(2);
            let db = (cand_b.0 as f32 - wx).powi(2) + (cand_b.1 as f32 - wy).powi(2);
            let a_ok = !claimed.contains(&cell_key(cand_a.0, cand_a.1, xres))
                && cell_available(bufs, cand_a.0, cand_a.1, xres, yres);
            let b_ok = !claimed.contains(&cell_key(cand_b.0, cand_b.1, xres))
                && cell_available(bufs, cand_b.0, cand_b.1, xres, yres);
            if a_ok && (!b_ok || da <= db) {
                gx = cand_a.0;
                gy = cand_a.1;
            } else if b_ok {
                gx = cand_b.0;
                gy = cand_b.1;
            }
        }

        let (gx, gy) = find_place_cell(bufs, &claimed, gx, gy, xres, yres);
        claimed.insert(cell_key(gx, gy, xres));
        prev_cell = Some((gx, gy));

        // Reclaim the cell from a free particle if needed.
        let idx = grid_index(gx, gy, xres);
        if idx < bufs.pmap.len()
            && bufs.pmap[idx] != 0
            && let Some(other) = pmap_id(bufs.pmap[idx])
        {
            let other = other as usize;
            if particle_alive(bufs, other) && bufs.part_flags[other] & FLAG_RIGID_PIXEL == 0 {
                // Push it radially away from the body's centre of mass.
                displace_free_particle(bufs, other, body.cm_x, body.cm_y, vpx, vpy);
            }
        }

        body.pixels[i].grid_x = gx;
        body.pixels[i].grid_y = gy;

        // Reuse the slot reserved in step 1; every kept pixel has one.  Never
        // allocate here: `allocate_particle` would kill an occupant (possibly
        // another body's pixel) in the target cell.
        let id = match ids.get(i).copied().flatten() {
            Some(id) if id < particle_len(bufs) => id,
            _ => continue,
        };

        bufs.part_ty[id] = px.elem_ty;
        bufs.part_life[id] = px.life;
        bufs.part_ctype[id] = px.ctype;
        // Keep the stored position exactly on the rasterised cell so that the
        // next step's grid-position matching is unambiguous.
        bufs.part_x[id] = gx as f32;
        bufs.part_y[id] = gy as f32;
        bufs.part_vx[id] = if vpx.is_finite() { vpx } else { 0.0 };
        bufs.part_vy[id] = if vpy.is_finite() { vpy } else { 0.0 };
        bufs.part_temp[id] = px.temp;
        bufs.part_flags[id] = FLAG_RIGID_PIXEL | FLAG_SKIPMOVE;
        bufs.part_dcolour[id] = px.dcolour;

        if idx < bufs.pmap.len() {
            bufs.pmap[idx] = pmap_pack(id as u32, px.elem_ty);
        }
    }

    fill_diagonal_gaps(bufs, registry, body, &claimed, xres, yres);
}

/// Fill the remaining single-cell holes of a rotated cluster with extra rigid
/// particles so that fluids cannot seep through a diagonally-rasterised body.
///
/// The fillers are ordinary `FLAG_RIGID_PIXEL` particles that no body pixel
/// claims; the rigid pass reaps them on the next step and recreates them at the
/// new hole positions, so they never accumulate.
fn fill_diagonal_gaps(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    body: &RigidBody2D,
    claimed: &HashSet<i32>,
    xres: i32,
    yres: i32,
) {
    if claimed.len() < 8 || body.pixels.is_empty() {
        return;
    }

    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    for &key in claimed.iter() {
        let x = key % xres;
        let y = key / xres;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    min_x = (min_x + 1).max(0);
    min_y = (min_y + 1).max(0);
    max_x = (max_x - 1).min(xres - 1);
    max_y = (max_y - 1).min(yres - 1);

    let first = body.pixels[0];
    let elem_ty = first.elem_ty;
    let temp = first.temp;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if claimed.contains(&cell_key(x, y, xres)) {
                continue;
            }
            let idx = grid_index(x, y, xres);
            if idx >= bufs.pmap.len() || bufs.pmap[idx] != 0 {
                continue;
            }
            let up = claimed.contains(&cell_key(x, y - 1, xres));
            let down = claimed.contains(&cell_key(x, y + 1, xres));
            let left = claimed.contains(&cell_key(x - 1, y, xres));
            let right = claimed.contains(&cell_key(x + 1, y, xres));
            if !(up && down && left && right) {
                continue;
            }
            let Some(id) = bufs.allocate_particle(x, y, elem_ty, registry) else {
                continue;
            };
            if id >= particle_len(bufs) {
                continue;
            }
            bufs.part_ty[id] = elem_ty;
            bufs.part_life[id] = 0;
            bufs.part_ctype[id] = first.ctype;
            bufs.part_x[id] = x as f32;
            bufs.part_y[id] = y as f32;
            bufs.part_vx[id] = 0.0;
            bufs.part_vy[id] = 0.0;
            bufs.part_temp[id] = temp;
            bufs.part_flags[id] = FLAG_RIGID_PIXEL | FLAG_SKIPMOVE;
            bufs.part_dcolour[id] = first.dcolour;
            bufs.pmap[idx] = pmap_pack(id as u32, elem_ty);
        }
    }
}
