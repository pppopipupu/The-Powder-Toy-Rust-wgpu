//! Rigid-body contacts: canvas bounds, walls, solids, other rigid pixels and
//! the fluid (buoyancy / drag / displacement) response.
//!
//! Everything here mirrors the Force- and impulse-based response of the
//! original pixel-cluster solver: contacts are resolved per pixel against the
//! cell it entered, using the *previous* sub-step separation to pick the
//! contact face.

use tpt_core::particle::GravityMode;

use super::*;

/// Resolve all contacts of one rigid pixel at world position `(wx, wy)`.
pub(super) fn resolve_pixel_contacts(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    body: &mut RigidBody2D,
    elem_ty: u32,
    wx: f32,
    wy: f32,
    prev_wx: f32,
    prev_wy: f32,
    rx: f32,
    ry: f32,
    body_unit_weight: f32,
    sdt: f32,
    corr_x: &mut f32,
    corr_y: &mut f32,
) {
    let xres = xres_of(bufs);
    let yres = yres_of(bufs);
    let props = element_properties(registry, elem_ty);

    resolve_boundary_contacts(body, wx, wy, rx, ry, xres, yres, corr_x, corr_y);

    let gx = wx.floor() as i32;
    let gy = wy.floor() as i32;
    if !in_bounds(gx, gy, xres, yres) {
        return;
    }

    resolve_wall_contact(
        bufs,
        body,
        props,
        gx,
        gy,
        (wx, wy),
        (prev_wx, prev_wy),
        (rx, ry),
        corr_x,
        corr_y,
    );
    resolve_particle_contact(
        bufs,
        registry,
        body,
        (wx, wy),
        (prev_wx, prev_wy),
        (rx, ry),
        body_unit_weight,
        sdt,
        corr_x,
        corr_y,
    );
}

/// Canvas boundary: every edge mode behaves as a solid wall for rigid bodies,
/// so they must never leave the simulation volume.
fn resolve_boundary_contacts(
    body: &mut RigidBody2D,
    wx: f32,
    wy: f32,
    rx: f32,
    ry: f32,
    xres: i32,
    yres: i32,
    corr_x: &mut f32,
    corr_y: &mut f32,
) {
    if wx < 0.5 {
        let pen = 0.5 - wx;
        resolve_impulse(body, rx, ry, (1.0, 0.0), pen, corr_x, corr_y);
    } else if wx > xres as f32 - 0.5 {
        let pen = wx - (xres as f32 - 0.5);
        resolve_impulse(body, rx, ry, (-1.0, 0.0), pen, corr_x, corr_y);
    }
    if wy < 0.5 {
        let pen = 0.5 - wy;
        resolve_impulse(body, rx, ry, (0.0, 1.0), pen, corr_x, corr_y);
    } else if wy > yres as f32 - 0.5 {
        let pen = wy - (yres as f32 - 0.5);
        resolve_impulse(body, rx, ry, (0.0, -1.0), pen, corr_x, corr_y);
    }
}

/// Wall collision against the coarse block covering the pixel.
fn resolve_wall_contact(
    bufs: &SimBuffers,
    body: &mut RigidBody2D,
    props: u32,
    gx: i32,
    gy: i32,
    world: (f32, f32),
    prev: (f32, f32),
    arm: (f32, f32),
    corr_x: &mut f32,
    corr_y: &mut f32,
) {
    if !wall_blocks_at(bufs, gx, gy, props) {
        return;
    }
    let cell_x = gx as f32 + 0.5;
    let cell_y = gy as f32 + 0.5;
    let (nx, ny, pen) = contact_axes_prev(
        prev.0 - cell_x,
        prev.1 - cell_y,
        world.0 - cell_x,
        world.1 - cell_y,
    );
    resolve_impulse(body, arm.0, arm.1, (nx, ny), pen, corr_x, corr_y);
}

/// Contact with the occupant of the pixel's cell: a hard impulse for solids and
/// other rigid pixels, buoyancy / drag / displacement for fluids.
fn resolve_particle_contact(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    body: &mut RigidBody2D,
    world: (f32, f32),
    prev: (f32, f32),
    arm: (f32, f32),
    body_unit_weight: f32,
    sdt: f32,
    corr_x: &mut f32,
    corr_y: &mut f32,
) {
    let xres = xres_of(bufs);
    let gx = world.0.floor() as i32;
    let gy = world.1.floor() as i32;
    let idx = grid_index(gx, gy, xres);
    if idx >= bufs.pmap.len() {
        return;
    }
    let pm = bufs.pmap[idx];
    if pm == 0 {
        return;
    }
    let Some(other) = pmap_id(pm) else { return };
    let other = other as usize;
    if !particle_alive(bufs, other) {
        return;
    }
    let other_ty = bufs.part_ty[other];
    let other_props = element_properties(registry, other_ty);

    // Rigid pixels are hard contacts even when their element is not
    // `TYPE_SOLID`: a `PT_STNE` box is `TYPE_PART` in the element table, but it
    // must stop other bodies instead of being displaced like loose powder.
    if (other_props & TYPE_SOLID != 0) || (bufs.part_flags[other] & FLAG_RIGID_PIXEL != 0) {
        resolve_solid_contact(bufs, body, other, world, prev, arm, corr_x, corr_y);
    } else if other_props & (TYPE_LIQUID | TYPE_PART | TYPE_GAS) != 0 {
        apply_fluid_contact(
            bufs,
            registry,
            body,
            other,
            other_ty,
            world.0,
            world.1,
            arm.0,
            arm.1,
            body_unit_weight,
            sdt,
        );
    }
}

/// Hard contact against a static solid or another rigid body's pixel.
fn resolve_solid_contact(
    bufs: &SimBuffers,
    body: &mut RigidBody2D,
    other: usize,
    world: (f32, f32),
    prev: (f32, f32),
    arm: (f32, f32),
    corr_x: &mut f32,
    corr_y: &mut f32,
) {
    let ox = bufs.part_x[other];
    let oy = bufs.part_y[other];
    if !ox.is_finite() || !oy.is_finite() {
        return;
    }
    let (nx, ny, pen) = contact_axes_prev(prev.0 - ox, prev.1 - oy, world.0 - ox, world.1 - oy);
    resolve_impulse(body, arm.0, arm.1, (nx, ny), pen, corr_x, corr_y);
}

/// Contact axis and penetration depth between two unit squares.
///
/// The axis is chosen from the *previous* separation so that a pixel which
/// entered a cell through its top face is pushed back up, even when the
/// centre-based minimum translation vector would point sideways.
fn contact_axes_prev(prev_dx: f32, prev_dy: f32, dx: f32, dy: f32) -> (f32, f32, f32) {
    let (px, py) = if prev_dx.is_finite() && prev_dy.is_finite() {
        (prev_dx, prev_dy)
    } else {
        (dx, dy)
    };
    if px.abs() >= py.abs() {
        let nx = axis_sign(dx, px);
        (nx, 0.0, (1.0 - dx.abs()).clamp(0.0, 1.0))
    } else {
        let ny = axis_sign(dy, py);
        (0.0, ny, (1.0 - dy.abs()).clamp(0.0, 1.0))
    }
}

#[inline]
fn axis_sign(current: f32, previous: f32) -> f32 {
    if current > 1e-6 {
        1.0
    } else if current < -1e-6 {
        -1.0
    } else if previous > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Apply a normal + friction impulse at lever arm `(rx, ry)` for contact normal
/// `(nx, ny)` and penetration `pen`.
fn resolve_impulse(
    body: &mut RigidBody2D,
    rx: f32,
    ry: f32,
    n: (f32, f32),
    pen: f32,
    corr_x: &mut f32,
    corr_y: &mut f32,
) {
    if !pen.is_finite() || pen <= 0.0 {
        return;
    }
    let (nx, ny) = n;
    if !nx.is_finite() || !ny.is_finite() {
        return;
    }

    let vpx = body.vx - body.omega * ry;
    let vpy = body.vy + body.omega * rx;
    let vn = vpx * nx + vpy * ny;

    if vn < 0.0 {
        let rn = rx * ny - ry * nx;
        let denom = body.inv_mass + body.inv_inertia * rn * rn;
        if denom > 1e-9 {
            let j = -(1.0 + RESTITUTION) * vn / denom;
            body.vx += j * nx * body.inv_mass;
            body.vy += j * ny * body.inv_mass;
            body.omega += body.inv_inertia * (rx * j * ny - ry * j * nx);

            // Coulomb friction along the contact tangent.
            let vtx = vpx - vn * nx;
            let vty = vpy - vn * ny;
            let vt_len = (vtx * vtx + vty * vty).sqrt();
            if vt_len > 1e-5 {
                let tx = vtx / vt_len;
                let ty = vty / vt_len;
                let rt = rx * ty - ry * tx;
                let denom_t = body.inv_mass + body.inv_inertia * rt * rt;
                if denom_t > 1e-9 {
                    let max_friction = FRICTION_MU * j;
                    let jt = (-vt_len / denom_t).clamp(-max_friction, max_friction);
                    body.vx += jt * tx * body.inv_mass;
                    body.vy += jt * ty * body.inv_mass;
                    body.omega += body.inv_inertia * (rx * jt * ty - ry * jt * tx);
                }
            }
            clamp_body_velocity(body);
        }
    }

    // Positional correction (linear only, rotation is handled by the impulses).
    *corr_x += nx * pen * POSITION_CORRECTION;
    *corr_y += ny * pen * POSITION_CORRECTION;
}

/// Buoyancy + drag + displacement for a liquid / powder / gas contact.
fn apply_fluid_contact(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    body: &mut RigidBody2D,
    other: usize,
    other_ty: u32,
    wx: f32,
    wy: f32,
    rx: f32,
    ry: f32,
    body_unit_weight: f32,
    sdt: f32,
) {
    let vpx = body.vx - body.omega * ry;
    let vpy = body.vy + body.omega * rx;
    // Each contact displaces one pixel of medium; the resulting acceleration is
    // the buoyant force divided by the whole body's mass.
    let inv_pixels = 1.0 / body.pixels.len().max(1) as f32;

    // Archimedes: the displaced medium pushes the body upwards.
    if bufs.config.gravity_mode != GravityMode::Off {
        let fluid_weight = element_weight(registry, other_ty);
        let ratio = (fluid_weight / body_unit_weight.max(1.0)).clamp(0.0, 2.0);
        body.vy -= GRAVITY_ACCEL * ratio * BUOYANCY_SCALE * sdt * inv_pixels;
    }

    // Viscous drag: couple the body to the fluid's velocity.
    let rel_x = bufs.part_vx[other] - vpx;
    let rel_y = bufs.part_vy[other] - vpy;
    let k = FLUID_DRAG * sdt * inv_pixels;
    body.vx += rel_x * k;
    body.vy += rel_y * k;
    body.omega += (rx * rel_y - ry * rel_x) * k * body.inv_inertia * 0.25;
    clamp_body_velocity(body);

    // Push the free particle out of the rigid body's way.
    displace_free_particle(bufs, other, wx, wy, vpx, vpy);
}

/// Move a free particle to an adjacent free cell, away from the rigid surface,
/// imparting the rigid body's local surface velocity.
pub(super) fn displace_free_particle(
    bufs: &mut SimBuffers,
    id: usize,
    from_x: f32,
    from_y: f32,
    vpx: f32,
    vpy: f32,
) {
    if !particle_alive(bufs, id) {
        return;
    }
    let xres = xres_of(bufs);
    let yres = yres_of(bufs);
    if xres <= 0 || yres <= 0 {
        return;
    }

    let gx = bufs.part_x[id].round() as i32;
    let gy = bufs.part_y[id].round() as i32;
    if !in_bounds(gx, gy, xres, yres) {
        return;
    }

    let mut dir_x = bufs.part_x[id] - from_x;
    let mut dir_y = bufs.part_y[id] - from_y;
    let len = (dir_x * dir_x + dir_y * dir_y).sqrt();
    if len > 1e-4 && len.is_finite() {
        dir_x /= len;
        dir_y /= len;
    } else {
        dir_x = 0.0;
        dir_y = 1.0;
    }

    let old_idx = grid_index(gx, gy, xres);
    let ty = bufs.part_ty[id];

    let mut best: Option<(i32, i32, f32)> = None;
    for oy in -1..=1i32 {
        for ox in -1..=1i32 {
            if ox == 0 && oy == 0 {
                continue;
            }
            let nx = gx + ox;
            let ny = gy + oy;
            if !in_bounds(nx, ny, xres, yres) {
                continue;
            }
            let ni = grid_index(nx, ny, xres);
            if ni >= bufs.pmap.len() || bufs.pmap[ni] != 0 {
                continue;
            }
            if wall_blocks_at(bufs, nx, ny, TYPE_SOLID) {
                continue;
            }
            let score = ox as f32 * dir_x + oy as f32 * dir_y;
            if best.is_none_or(|(_, _, s)| score > s) {
                best = Some((nx, ny, score));
            }
        }
    }

    if let Some((nx, ny, _)) = best {
        if old_idx < bufs.pmap.len() && pmap_id(bufs.pmap[old_idx]) == Some(id as u32) {
            bufs.pmap[old_idx] = 0;
        }
        bufs.part_x[id] = nx as f32;
        bufs.part_y[id] = ny as f32;
        bufs.part_vx[id] = vpx + dir_x * 0.5;
        bufs.part_vy[id] = vpy + dir_y * 0.5;
        let ni = grid_index(nx, ny, xres);
        bufs.pmap[ni] = pmap_pack(id as u32, ty);
    } else {
        // Completely boxed in: just transfer momentum instead of moving.
        bufs.part_vx[id] += dir_x * 0.5;
        bufs.part_vy[id] += dir_y * 0.5;
    }
}
