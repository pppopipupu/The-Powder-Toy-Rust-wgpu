//! Rigid-body integration: external acceleration, air coupling, sub-stepping and
//! the per-sub-step contact pass.
//!
//! The sub-stepper bounds how far a body may travel per resolution step so a
//! fast body cannot tunnel through a wall; `integrate_body` only accumulates the
//! forces and hands the sub-step to [`integrate_substep`].

use tpt_core::elements::ElementRegistry;
use tpt_core::particle::{AirMode, GravityMode, RigidBody2D};

use super::contacts::resolve_pixel_contacts;
use super::*;

/// Integrate gravity, air pressure/wind, buoyancy, contacts and rotation.
pub(super) fn integrate_body(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    body: &mut RigidBody2D,
    frames: f32,
) {
    let xres = xres_of(bufs);
    let yres = yres_of(bufs);
    if xres <= 1 || yres <= 1 || body.pixels.is_empty() {
        return;
    }

    let (cos_a, sin_a) = (body.angle.cos(), body.angle.sin());
    let n_pixels = body.pixels.len() as f32;
    let inv_n = 1.0 / n_pixels.max(1.0);
    let body_unit_weight = (body.mass * inv_n).max(1.0);

    let (grav_x, grav_y) = gravity_acceleration(bufs, body, xres, yres);
    let (force_x, force_y, torque) = air_forces(bufs, registry, body, cos_a, sin_a);

    let mut ax = grav_x + force_x * body.inv_mass;
    let mut ay = grav_y + force_y * body.inv_mass;
    let angular_acc = (torque * body.inv_inertia).clamp(-0.25, 0.25);

    // Clamp the external accelerations to keep the sub-stepper well behaved.
    ax = ax.clamp(-4.0, 4.0);
    ay = ay.clamp(-4.0, 4.0);

    let radius = motion_radius(body);
    let substeps = substep_count(body, frames, radius);
    let sdt = frames / substeps as f32;

    for _ in 0..substeps {
        integrate_substep(
            bufs,
            registry,
            body,
            sdt,
            ax,
            ay,
            angular_acc,
            body_unit_weight,
        );
    }

    clamp_body_velocity(body);
    sanitize_mass(body);
}

/// External acceleration from the configured gravity mode.
fn gravity_acceleration(bufs: &SimBuffers, body: &RigidBody2D, xres: i32, yres: i32) -> (f32, f32) {
    let mut ax = 0.0f32;
    let mut ay = 0.0f32;
    match bufs.config.gravity_mode {
        GravityMode::Vertical => ay += GRAVITY_ACCEL,
        GravityMode::Radial => {
            let cx = xres as f32 * 0.5;
            let cy = yres as f32 * 0.5;
            let dx = cx - body.cm_x;
            let dy = cy - body.cm_y;
            let d = (dx * dx + dy * dy).sqrt().max(1.0);
            ax += GRAVITY_ACCEL * dx / d;
            ay += GRAVITY_ACCEL * dy / d;
        }
        GravityMode::Off => {}
        GravityMode::Custom => {
            ax += GRAVITY_ACCEL * bufs.config.custom_gravity_x;
            ay += GRAVITY_ACCEL * bufs.config.custom_gravity_y;
        }
    }
    (ax, ay)
}

/// Air pressure gradient + wind drag summed over the body pixels.
///
/// Returns `(force_x, force_y, torque)` in frame units, still un-scaled by the
/// body's inverse mass / inertia.
fn air_forces(
    bufs: &SimBuffers,
    registry: &ElementRegistry,
    body: &RigidBody2D,
    cos_a: f32,
    sin_a: f32,
) -> (f32, f32, f32) {
    let xcells = xcells_of(bufs);
    let ycells = ycells_of(bufs);
    let air_cells = (xcells * ycells).max(0) as usize;
    let has_air = bufs.config.air_mode != AirMode::Off
        && air_cells > 0
        && bufs.air_pv.len() >= air_cells
        && bufs.air_vx.len() >= air_cells
        && bufs.air_vy.len() >= air_cells;

    let mut force_x = 0.0f32;
    let mut force_y = 0.0f32;
    let mut torque = 0.0f32;
    if !has_air {
        return (force_x, force_y, torque);
    }

    for px in body.pixels.iter() {
        let (rx, ry) = rotate_local(px.local_x, px.local_y, cos_a, sin_a);
        let wx = body.cm_x + rx;
        let wy = body.cm_y + ry;
        if !wx.is_finite() || !wy.is_finite() {
            continue;
        }
        let cx = (wx / CELL as f32).floor();
        let cy = (wy / CELL as f32).floor();
        if cx < 0.0 || cy < 0.0 || cx >= xcells as f32 || cy >= ycells as f32 {
            continue;
        }
        let cx = cx as i32;
        let cy = cy as i32;
        let cxm = (cx - 1).max(0);
        let cxp = (cx + 1).min(xcells - 1);
        let cym = (cy - 1).max(0);
        let cyp = (cy + 1).min(ycells - 1);
        let row = cy as usize * xcells as usize;

        let grad_x = (bufs.air_pv[row + cxm as usize] - bufs.air_pv[row + cxp as usize])
            .clamp(-AIR_PRESSURE_GRAD_CLAMP, AIR_PRESSURE_GRAD_CLAMP);
        let grad_y = (bufs.air_pv[cym as usize * xcells as usize + cx as usize]
            - bufs.air_pv[cyp as usize * xcells as usize + cx as usize])
            .clamp(-AIR_PRESSURE_GRAD_CLAMP, AIR_PRESSURE_GRAD_CLAMP);

        let drag_k = AIR_DRAG_FORCE * (1.0 + element_air_drag(registry, px.elem_ty));
        let cell = row + cx as usize;
        let avx = bufs.air_vx[cell];
        let avy = bufs.air_vy[cell];

        let vpx = body.vx - body.omega * ry;
        let vpy = body.vy + body.omega * rx;

        let fx = grad_x * AIR_PRESSURE_FORCE + (avx - vpx) * drag_k;
        let fy = grad_y * AIR_PRESSURE_FORCE + (avy - vpy) * drag_k;

        force_x += fx;
        force_y += fy;
        torque += rx * fy - ry * fx;
    }

    (force_x, force_y, torque)
}

/// Largest pixel distance from the centre of mass.
fn motion_radius(body: &RigidBody2D) -> f32 {
    let mut radius = 0.0f32;
    for px in &body.pixels {
        let r2 = px.local_x * px.local_x + px.local_y * px.local_y;
        if r2.is_finite() && r2 > radius {
            radius = r2;
        }
    }
    radius.sqrt()
}

/// Number of collision sub-steps needed to keep the travel per step bounded.
fn substep_count(body: &RigidBody2D, frames: f32, radius: f32) -> usize {
    let travel = (body.vx.abs().max(body.vy.abs()) + body.omega.abs() * radius) * frames;
    if travel.is_finite() {
        ((travel / SUBSTEP_DISTANCE).ceil() as usize).clamp(1, MAX_SUBSTEPS)
    } else {
        1
    }
}

/// Advance one collision sub-step: apply the accelerations, damp, then resolve
/// every pixel's contacts and move the centre of mass.
fn integrate_substep(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    body: &mut RigidBody2D,
    sdt: f32,
    ax: f32,
    ay: f32,
    angular_acc: f32,
    body_unit_weight: f32,
) {
    let xres = xres_of(bufs);
    let yres = yres_of(bufs);

    body.vx += ax * sdt;
    body.vy += ay * sdt;
    body.omega += angular_acc * sdt;

    let damp = LINEAR_DAMPING.powf(sdt);
    body.vx *= damp;
    body.vy *= damp;
    body.omega *= ANGULAR_DAMPING.powf(sdt);
    clamp_body_velocity(body);

    let pred_cm_x = body.cm_x + body.vx * sdt;
    let pred_cm_y = body.cm_y + body.vy * sdt;
    let pred_angle = body.angle + body.omega * sdt;
    let (pc, ps) = (pred_angle.cos(), pred_angle.sin());

    let mut corr_x = 0.0f32;
    let mut corr_y = 0.0f32;

    let count = body.pixels.len();
    let (prev_c, prev_s) = (body.angle.cos(), body.angle.sin());
    for i in 0..count {
        let px = body.pixels[i];
        let (rx, ry) = rotate_local(px.local_x, px.local_y, pc, ps);
        let wx = pred_cm_x + rx;
        let wy = pred_cm_y + ry;
        if !wx.is_finite() || !wy.is_finite() {
            continue;
        }
        // Position at the start of this sub-step: used to pick the contact
        // face a pixel entered through instead of an ambiguous centre-based
        // minimum translation vector.
        let (prx, pry) = rotate_local(px.local_x, px.local_y, prev_c, prev_s);
        let prev_wx = body.cm_x + prx;
        let prev_wy = body.cm_y + pry;
        resolve_pixel_contacts(
            bufs,
            registry,
            body,
            px.elem_ty,
            wx,
            wy,
            prev_wx,
            prev_wy,
            rx,
            ry,
            body_unit_weight,
            sdt,
            &mut corr_x,
            &mut corr_y,
        );
    }

    // Bound the total positional correction of this sub-step.
    let corr_len = (corr_x * corr_x + corr_y * corr_y).sqrt();
    if corr_len > MAX_POSITION_CORRECTION && corr_len.is_finite() {
        let scale = MAX_POSITION_CORRECTION / corr_len;
        corr_x *= scale;
        corr_y *= scale;
    }

    body.cm_x = (pred_cm_x + corr_x).clamp(0.5, xres as f32 - 0.5);
    body.cm_y = (pred_cm_y + corr_y).clamp(0.5, yres as f32 - 0.5);
    body.angle = pred_angle.rem_euclid(std::f32::consts::TAU);
}
