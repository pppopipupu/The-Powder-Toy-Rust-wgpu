//! Noita-style pixel-cluster rigid-body simulation.
//!
//! Rigid bodies are clusters of ordinary Powder Toy particles that carry the
//! [`FLAG_RIGID_PIXEL`] flag.
//!
//! # Two paths, one description
//!
//! [`RigidBody2D`] is the host description: it is what spawns, Lua scripts,
//! save files and the tests see. The steady-state simulation itself lives on
//! the device ([`gpu`]):
//!
//! * when the description changes (spawn, edit, save restore, destroy) the
//!   historical host solver runs once to materialise pixels, sanitise mass and
//!   reap orphaned rigid particles, then uploads the description;
//! * afterwards every frame runs the device passes only — reconciliation,
//!   mass/force preparation, sub-stepped integration, contacts and
//!   rasterisation — with no particle, map or air transfer in either direction.
//!
//! The host solver below is that initialiser:
//!
//! * synchronises the cluster description with the authoritative particle
//!   buffers (thermal, chemical and explosion state changes are read back and
//!   detached pixels become free-flowing particles again),
//! * clears the old spatial-map occupancy,
//! * integrates the 2D rigid-body state (gravity including the Newtonian
//!   field, air pressure, wind, buoyancy, boundary/wall/solid contacts and
//!   fluid displacement), and
//! * re-rasterises the rotated cluster into the particle pool and `pmap`.
//!
//! The integrator works in Powder Toy "frame" units: one unit of velocity is
//! one grid cell per simulation frame, and `dt` is the number of frames to
//! advance (the engine passes `1.0` once per step).  `dt` is clamped so that a
//! large time step cannot make the sub-stepper explode.

#![allow(clippy::too_many_arguments)]

mod contacts;
mod gpu;
mod integrate;
mod raster;

pub use gpu::RigidGpu;

use std::collections::HashMap;

use tpt_core::constants::*;
use tpt_core::elements::ElementRegistry;
use tpt_core::particle::{RigidBody2D, RigidPixel};
use tpt_core::walls::is_wall_blocking_particle;

use crate::buffers::SimBuffers;

use self::integrate::integrate_body;
use self::raster::rasterize_body;

/// Largest number of simulation frames a single call may advance.
const MAX_FRAMES_PER_STEP: f32 = 4.0;
/// Gravity acceleration in cells / frame^2.
const GRAVITY_ACCEL: f32 = 0.15;
/// Coefficient of restitution for rigid contacts.
const RESTITUTION: f32 = 0.25;
/// Coulomb friction coefficient for rigid contacts.
const FRICTION_MU: f32 = 0.4;
/// Maximum number of collision sub-steps per simulation step.
///
/// Mirrored as the literal `16u32` in `motion_kernels::prepare_finish_kernel` /
/// `integrate_substep_kernel` (see the assertion below): the device loop is
/// fixed because the host must not read the per-body count back.
const MAX_SUBSTEPS: usize = 16;
/// Maximum distance (in cells) a pixel may travel per collision sub-step.
///
/// One cell is the hard limit of the destination-cell contact sample: a pixel
/// that moves at most one cell cannot skip the rounded cell between its start
/// and its destination, so a one-pixel wall or a thin body can never be
/// stepped over. `prepare_finish_kernel` sizes the sub-step count from this and the
/// integrate pass clamps the surface velocity to the same budget.
const SUBSTEP_DISTANCE: f32 = 1.0;
/// Linear velocity clamp in cells / frame.
const MAX_LINEAR_SPEED: f32 = 16.0;
/// Angular velocity clamp in radians / frame.
const MAX_ANGULAR_SPEED: f32 = 0.75;
/// Upper bound for `inv_inertia` to keep tiny clusters from spinning wildly.
const MAX_INV_INERTIA: f32 = 0.5;
/// Fraction of the penetration depth corrected per contact.
const POSITION_CORRECTION: f32 = 0.6;
/// Maximum positional correction applied to the centre of mass per sub-step.
const MAX_POSITION_CORRECTION: f32 = 1.0;
/// Pressure gradient to acceleration conversion (per pixel).
const AIR_PRESSURE_FORCE: f32 = 0.05;
/// Air velocity coupling (per pixel).
const AIR_DRAG_FORCE: f32 = 0.02;
/// Clamp for the sampled pressure gradient.
const AIR_PRESSURE_GRAD_CLAMP: f32 = 24.0;
/// Velocity coupling with displaced liquids / powders / gases.
const FLUID_DRAG: f32 = 0.08;
/// Fraction of gravity cancelled by Archimedes buoyancy at density parity.
const BUOYANCY_SCALE: f32 = 0.5;
/// Per-frame linear damping.
const LINEAR_DAMPING: f32 = 0.999;
/// Per-frame angular damping.
const ANGULAR_DAMPING: f32 = 0.995;

/// Advance every active rigid body by `dt` simulation frames.
///
/// This is the device entry point: it keeps the device authoritative in the
/// steady state and only falls back to the host initialiser when the host
/// [`RigidBody2D`] description changed or when the caller synchronised the host
/// shadows after the previous frame.
pub fn step_rigid_bodies(
    bufs: &mut SimBuffers,
    rigid_gpu: &mut RigidGpu,
    registry: &ElementRegistry,
    dt: f32,
) {
    if bufs.rigid_bodies.is_empty() && rigid_gpu.is_idle() {
        return;
    }
    rigid_gpu.step(bufs, registry, dt);
}

/// Host-side rigid-body initialiser.
///
/// Runs only when the body description changed (spawn / edit / restore /
/// destroy). It reads the host particle and map shadows, materialises new
/// pixels, sanitises mass, reaps orphaned rigid particles and rasterises the
/// result; [`RigidGpu`] then uploads the device description.
pub(super) fn step_rigid_bodies_host(bufs: &mut SimBuffers, registry: &ElementRegistry, dt: f32) {
    if bufs.rigid_bodies.is_empty() {
        return;
    }
    // The thermal / reaction / movement kernels may have produced newer state
    // on the GPU than the host shadows we are about to read. Only the arrays the
    // solver touches are pulled back.
    if bufs.gpu_particles_dirty {
        bufs.sync_particles_gpu_to_host();
    }
    if bufs.gpu_maps_dirty {
        bufs.sync_maps_gpu_to_host();
    }
    if particle_len(bufs) == 0 || xres_of(bufs) <= 1 || yres_of(bufs) <= 1 {
        return;
    }

    let frames = if dt.is_finite() && dt > 0.0 {
        dt.clamp(0.0, MAX_FRAMES_PER_STEP)
    } else {
        1.0
    };
    if frames <= 0.0 {
        return;
    }

    // Take the bodies out of the buffer so that the helpers can freely borrow
    // `bufs` mutably while mutating a single body.
    let mut bodies = std::mem::take(&mut bufs.rigid_bodies);
    let mut occupancy = build_rigid_occupancy(bufs);
    let mut touched = false;

    for body in bodies.iter_mut() {
        if !body.active || body.pixels.is_empty() {
            body.active = false;
            touched = true;
            continue;
        }
        step_single_body(bufs, registry, body, frames, &mut occupancy);
        touched = true;
    }

    // Reap rigid particles that no active body claimed: gap fillers created on
    // the previous step and pixels of bodies that died during the sync pass.
    let xres = xres_of(bufs);
    let yres = yres_of(bufs);
    for ids in occupancy.values() {
        for &id in ids {
            if id >= particle_len(bufs) {
                continue;
            }
            if bufs.part_ty[id] == 0 {
                // Dead slot that still carried the rigid flag.
                bufs.part_flags[id] &= !(FLAG_RIGID_PIXEL | FLAG_SKIPMOVE);
                continue;
            }
            let gx = bufs.part_x[id].round() as i32;
            let gy = bufs.part_y[id].round() as i32;
            if in_bounds(gx, gy, xres, yres) {
                let idx = grid_index(gx, gy, xres);
                if idx < bufs.pmap.len() && pmap_id(bufs.pmap[idx]) == Some(id as u32) {
                    bufs.pmap[idx] = 0;
                }
            }
            bufs.kill_particle(id);
        }
    }

    drop(occupancy);
    bufs.rigid_bodies = bodies;
    let before = bufs.rigid_bodies.len();
    bufs.rigid_bodies
        .retain(|b| b.active && !b.pixels.is_empty());
    if touched || bufs.rigid_bodies.len() != before {
        // Particle slots, `pmap` and the body list changed on the host: make
        // sure the engine uploads them before the next GPU dispatch.
        bufs.mark_particles_dirty();
        bufs.mark_maps_dirty();
    }
}

#[inline]
fn xres_of(bufs: &SimBuffers) -> i32 {
    bufs.config.xres as i32
}

#[inline]
fn yres_of(bufs: &SimBuffers) -> i32 {
    bufs.config.yres as i32
}

#[inline]
fn xcells_of(bufs: &SimBuffers) -> i32 {
    bufs.config.xcells() as i32
}

#[inline]
fn ycells_of(bufs: &SimBuffers) -> i32 {
    bufs.config.ycells() as i32
}

#[inline]
fn in_bounds(x: i32, y: i32, xres: i32, yres: i32) -> bool {
    x >= 0 && y >= 0 && x < xres && y < yres
}

#[inline]
fn grid_index(x: i32, y: i32, xres: i32) -> usize {
    y as usize * xres as usize + x as usize
}

#[inline]
fn cell_key(x: i32, y: i32, xres: i32) -> i32 {
    y * xres + x
}

#[inline]
fn rotate_local(lx: f32, ly: f32, cos_a: f32, sin_a: f32) -> (f32, f32) {
    (lx * cos_a - ly * sin_a, lx * sin_a + ly * cos_a)
}

/// Number of usable particle slots (all SoA vectors have the same capacity).
fn particle_len(bufs: &SimBuffers) -> usize {
    bufs.part_ty
        .len()
        .min(bufs.part_life.len())
        .min(bufs.part_ctype.len())
        .min(bufs.part_x.len())
        .min(bufs.part_y.len())
        .min(bufs.part_vx.len())
        .min(bufs.part_vy.len())
        .min(bufs.part_temp.len())
        .min(bufs.part_flags.len())
        .min(bufs.part_dcolour.len())
}

#[inline]
fn particle_alive(bufs: &SimBuffers, id: usize) -> bool {
    id < particle_len(bufs) && bufs.part_ty[id] != 0
}

/// Element weight, floored at 10 like the rest of the rigid-body code.
fn element_weight(registry: &ElementRegistry, ty: u32) -> f32 {
    if (ty as usize) < PT_NUM {
        registry.get(ty).weight.max(10) as f32
    } else {
        10.0
    }
}

fn element_properties(registry: &ElementRegistry, ty: u32) -> u32 {
    if (ty as usize) < PT_NUM {
        registry.get(ty).properties
    } else {
        TYPE_SOLID
    }
}

fn element_air_drag(registry: &ElementRegistry, ty: u32) -> f32 {
    if (ty as usize) < PT_NUM {
        registry.get(ty).air_drag.clamp(0.0, 2.0)
    } else {
        0.0
    }
}

/// Return `true` when the coarse wall cell covering `(gx, gy)` blocks a
/// particle with the supplied element properties.
fn wall_blocks_at(bufs: &SimBuffers, gx: i32, gy: i32, props: u32) -> bool {
    let xcells = xcells_of(bufs);
    let ycells = ycells_of(bufs);
    if xcells <= 0 || ycells <= 0 {
        return false;
    }
    let cx = (gx / CELL as i32).clamp(0, xcells - 1);
    let cy = (gy / CELL as i32).clamp(0, ycells - 1);
    let ci = cy as usize * xcells as usize + cx as usize;
    if ci >= bufs.wall_bmap.len() {
        return false;
    }
    let wall_id = (bufs.wall_bmap[ci] & 0xFF) as u8;
    if wall_id == WL_ERASE {
        return false;
    }
    let emap_val = if ci < bufs.wall_emap.len() {
        (bufs.wall_emap[ci] & 0xFF) as u8
    } else {
        0
    };
    is_wall_blocking_particle(wall_id, emap_val, props)
}

/// Build a map of grid cell -> rigid-flagged particle ids.
///
/// The particle arrays are authoritative (thermal / reaction / explosion
/// kernels mutate them directly), so the map is rebuilt from the particles
/// instead of trusting `pmap`, which may hold stale entries.
fn build_rigid_occupancy(bufs: &SimBuffers) -> HashMap<(i32, i32), Vec<usize>> {
    let mut map: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    let n = particle_len(bufs);
    for id in 0..n {
        // Include dead slots that still carry the rigid flag: the sync pass
        // uses them to notice that a pixel was killed and must detach.
        if bufs.part_flags[id] & FLAG_RIGID_PIXEL == 0 {
            continue;
        }
        let gx = bufs.part_x[id].round();
        let gy = bufs.part_y[id].round();
        if !gx.is_finite() || !gy.is_finite() {
            continue;
        }
        let entry = map.entry((gx as i32, gy as i32)).or_default();
        if bufs.part_ty[id] != 0 {
            // Live particles are pushed last so that `pop()` hands them out
            // before dead slots sharing the same cell.
            entry.push(id);
        } else {
            entry.insert(0, id);
        }
    }
    map
}

fn take_rigid_particle(
    map: &mut HashMap<(i32, i32), Vec<usize>>,
    gx: i32,
    gy: i32,
) -> Option<usize> {
    let key = (gx, gy);
    let id = map.get_mut(&key).and_then(|v| v.pop());
    if id.is_some() && map.get(&key).is_some_and(|v| v.is_empty()) {
        map.remove(&key);
    }
    id
}

/// Synchronise `body.pixels` with the live particles and detach any pixel whose
/// element melted, burned, corroded, shattered or was otherwise destroyed.
///
/// Returns the particle id backing each remaining pixel (aligned with
/// `body.pixels`).
fn sync_body_pixels(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    body: &mut RigidBody2D,
    occupancy: &mut HashMap<(i32, i32), Vec<usize>>,
) -> Vec<Option<usize>> {
    let xres = xres_of(bufs);
    let yres = yres_of(bufs);
    if xres <= 0 || yres <= 0 {
        body.active = false;
        body.pixels.clear();
        return Vec::new();
    }

    // Is any pixel of this body still backed by a live rigid particle?  Bodies
    // enter `bufs.rigid_bodies` through `SimulationEngine::spawn_rigid_body`,
    // which does not allocate particles: the first pass over a body therefore
    // materialises every pixel.  Later passes reuse the existing slots.
    //
    // A body that was already materialised but has lost *all* its particles
    // (melted, dissolved, erased) must not be re-materialised, otherwise it
    // would resurrect from nothing.  `body.materialised` is the discriminator;
    // without it the two situations are indistinguishable.
    let found_any = body.pixels.iter().any(|px| {
        occupancy
            .get(&(px.grid_x, px.grid_y))
            .is_some_and(|v| v.iter().any(|&id| particle_alive(bufs, id)))
    });
    let materialise_missing = !found_any && !body.materialised;

    let old_pixels = std::mem::take(&mut body.pixels);
    let mut kept: Vec<RigidPixel> = Vec::with_capacity(old_pixels.len());
    let mut ids: Vec<Option<usize>> = Vec::with_capacity(old_pixels.len());
    let mut removed = false;

    for px in old_pixels {
        match sync_one_pixel(
            bufs,
            registry,
            body,
            px,
            materialise_missing,
            occupancy,
            (xres, yres),
        ) {
            Some((px, id)) => {
                kept.push(px);
                ids.push(Some(id));
            }
            None => removed = true,
        }
    }

    body.pixels = kept;
    // Remember that the pixels are now (or were already) backed by particles,
    // so a later pass cannot mistake an erased body for a brand-new one.
    body.materialised |= ids.iter().any(|id| id.is_some());

    if body.pixels.len() < 2 {
        release_pixels(bufs, body, &ids);
        body.pixels.clear();
        body.active = false;
        ids.clear();
        return ids;
    }

    if removed {
        body.recompute_mass_properties(|ty| element_weight(registry, ty));
        sanitize_mass(body);
        if !body.active {
            release_pixels(bufs, body, &ids);
            body.pixels.clear();
            ids.clear();
        }
    }

    ids
}

/// Match one `body.pixels` entry against the particle pool.
///
/// Returns the (clamped) pixel and the slot backing it, or `None` when the pixel
/// was lost, lost its rigid flag, changed element or could not be materialised.
fn sync_one_pixel(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    body: &RigidBody2D,
    mut px: RigidPixel,
    materialise_missing: bool,
    occupancy: &mut HashMap<(i32, i32), Vec<usize>>,
    size: (i32, i32),
) -> Option<(RigidPixel, usize)> {
    let (xres, yres) = size;
    let gx = px.grid_x.clamp(0, xres - 1);
    let gy = px.grid_y.clamp(0, yres - 1);
    px.grid_x = gx;
    px.grid_y = gy;

    let id = match take_rigid_particle(occupancy, gx, gy) {
        Some(id) if particle_alive(bufs, id) => id,
        taken => {
            // The pixel's backing particle was killed or lost its flag.
            if let Some(id) = taken
                && id < particle_len(bufs)
            {
                bufs.part_flags[id] &= !(FLAG_RIGID_PIXEL | FLAG_SKIPMOVE);
            }
            if !materialise_missing {
                return None;
            }
            materialise_pixel(bufs, registry, &px)?
        }
    };

    let ty = bufs.part_ty[id];
    if ty == 0 {
        return None;
    }

    // Read back the state the thermal / reaction kernels produced.
    px.temp = bufs.part_temp[id];
    px.life = bufs.part_life[id];
    px.ctype = bufs.part_ctype[id];
    px.dcolour = bufs.part_dcolour[id];

    // Any element change detaches the pixel: it melted, burned, corroded,
    // shattered or was transmuted and must flow freely from now on.
    if ty != px.elem_ty || bufs.part_flags[id] & FLAG_RIGID_PIXEL == 0 {
        detach_pixel(bufs, body, id, gx, gy);
        return None;
    }

    Some((px, id))
}

/// Materialise a brand-new body's pixel in a fresh particle slot.
fn materialise_pixel(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    px: &RigidPixel,
) -> Option<usize> {
    let id = bufs.allocate_particle(px.grid_x, px.grid_y, px.elem_ty, registry)?;
    if id >= particle_len(bufs) {
        return None;
    }
    bufs.part_flags[id] = FLAG_RIGID_PIXEL | FLAG_SKIPMOVE;
    bufs.part_temp[id] = px.temp;
    bufs.part_life[id] = px.life;
    bufs.part_ctype[id] = px.ctype;
    bufs.part_dcolour[id] = px.dcolour;
    Some(id)
}

/// Turn a rigid pixel back into an ordinary free-flowing particle.
fn detach_pixel(bufs: &mut SimBuffers, body: &RigidBody2D, id: usize, gx: i32, gy: i32) {
    if !particle_alive(bufs, id) {
        return;
    }
    bufs.part_flags[id] &= !(FLAG_RIGID_PIXEL | FLAG_SKIPMOVE);

    // Inherit the local surface velocity so the detached fragment keeps the
    // momentum it had while it was welded to the body.
    let rx = bufs.part_x[id] - body.cm_x;
    let ry = bufs.part_y[id] - body.cm_y;
    let vpx = body.vx - body.omega * ry;
    let vpy = body.vy + body.omega * rx;
    if vpx.is_finite() {
        bufs.part_vx[id] = vpx;
    }
    if vpy.is_finite() {
        bufs.part_vy[id] = vpy;
    }

    // The particle keeps its cell; make sure the spatial map still sees it.
    let xres = xres_of(bufs);
    let yres = yres_of(bufs);
    if in_bounds(gx, gy, xres, yres) {
        let idx = grid_index(gx, gy, xres);
        if idx < bufs.pmap.len() && bufs.pmap[idx] == 0 {
            bufs.pmap[idx] = pmap_pack(id as u32, bufs.part_ty[id]);
        }
    }
}

/// Release every still-attached pixel of a body that fell apart.
fn release_pixels(bufs: &mut SimBuffers, body: &RigidBody2D, ids: &[Option<usize>]) {
    let xres = xres_of(bufs);
    let yres = yres_of(bufs);
    for (px, slot) in body.pixels.iter().zip(ids.iter()) {
        let Some(id) = *slot else { continue };
        if !particle_alive(bufs, id) {
            continue;
        }
        bufs.part_flags[id] &= !(FLAG_RIGID_PIXEL | FLAG_SKIPMOVE);
        let rx = bufs.part_x[id] - body.cm_x;
        let ry = bufs.part_y[id] - body.cm_y;
        let vpx = body.vx - body.omega * ry;
        let vpy = body.vy + body.omega * rx;
        if vpx.is_finite() {
            bufs.part_vx[id] = vpx;
        }
        if vpy.is_finite() {
            bufs.part_vy[id] = vpy;
        }
        if in_bounds(px.grid_x, px.grid_y, xres, yres) {
            let idx = grid_index(px.grid_x, px.grid_y, xres);
            if idx < bufs.pmap.len() && bufs.pmap[idx] == 0 {
                bufs.pmap[idx] = pmap_pack(id as u32, bufs.part_ty[id]);
            }
        }
    }
}

/// Repair mass properties after a fracture so that the integrator never sees
/// zero / NaN mass or inertia.
fn sanitize_mass(body: &mut RigidBody2D) {
    if !body.mass.is_finite() || body.mass <= 0.0 {
        body.mass = 1.0;
    }
    body.inv_mass = 1.0 / body.mass;

    if !body.inertia.is_finite() || body.inertia <= 1e-4 {
        let mut max_r2 = 0.0f32;
        for px in &body.pixels {
            let r2 = px.local_x * px.local_x + px.local_y * px.local_y;
            if r2.is_finite() && r2 > max_r2 {
                max_r2 = r2;
            }
        }
        body.inertia = body.mass * max_r2.max(1.0);
    }
    body.inv_inertia = (1.0 / body.inertia).min(MAX_INV_INERTIA);

    if !body.cm_x.is_finite() {
        body.cm_x = 0.5;
    }
    if !body.cm_y.is_finite() {
        body.cm_y = 0.5;
    }
    if !body.vx.is_finite() {
        body.vx = 0.0;
    }
    if !body.vy.is_finite() {
        body.vy = 0.0;
    }
    if !body.omega.is_finite() {
        body.omega = 0.0;
    }
}

fn clear_old_occupancy(bufs: &mut SimBuffers, body: &RigidBody2D, ids: &[Option<usize>]) {
    let xres = xres_of(bufs);
    let yres = yres_of(bufs);
    for (px, slot) in body.pixels.iter().zip(ids.iter()) {
        let Some(id) = *slot else { continue };
        if id >= particle_len(bufs) {
            continue;
        }
        if in_bounds(px.grid_x, px.grid_y, xres, yres) {
            let idx = grid_index(px.grid_x, px.grid_y, xres);
            if idx < bufs.pmap.len() && pmap_id(bufs.pmap[idx]) == Some(id as u32) {
                bufs.pmap[idx] = 0;
            }
        }
        // Free the slot: the body is about to move and the pixel will be
        // re-rasterised (usually into this very slot) in step 4.
        bufs.part_ty[id] = 0;
        bufs.part_flags[id] = 0;
        bufs.part_vx[id] = 0.0;
        bufs.part_vy[id] = 0.0;
    }
}

fn step_single_body(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    body: &mut RigidBody2D,
    frames: f32,
    occupancy: &mut HashMap<(i32, i32), Vec<usize>>,
) {
    let ids = sync_body_pixels(bufs, registry, body, occupancy);
    if !body.active || body.pixels.len() < 2 {
        return;
    }

    clear_old_occupancy(bufs, body, &ids);
    integrate_body(bufs, registry, body, frames);
    rasterize_body(bufs, registry, body, &ids);
}

/// Clamp the body velocity to the solver limits after an impulse.
fn clamp_body_velocity(body: &mut RigidBody2D) {
    if !body.vx.is_finite() {
        body.vx = 0.0;
    }
    if !body.vy.is_finite() {
        body.vy = 0.0;
    }
    if !body.omega.is_finite() {
        body.omega = 0.0;
    }
    let speed = (body.vx * body.vx + body.vy * body.vy).sqrt();
    if speed > MAX_LINEAR_SPEED {
        let scale = MAX_LINEAR_SPEED / speed;
        body.vx *= scale;
        body.vy *= scale;
    }
    body.omega = body.omega.clamp(-MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED);
}
