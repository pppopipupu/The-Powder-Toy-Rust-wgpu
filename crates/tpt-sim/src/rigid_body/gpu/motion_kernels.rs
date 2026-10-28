//! Device physics and contact kernels.

#![allow(clippy::too_many_arguments)]

use super::helpers::*;
use super::layout::*;
use cubecl::prelude::*;

/// One thread per rigid pixel: the force and torque this pixel contributes,
/// from the air pressure gradient, the wind drag and the Newtonian field.
///
/// The body's own thread used to walk every pixel of the body for this, and a
/// single thread pays a full memory latency per iteration with no other warp
/// to hide it behind: a 200x200 body spent most of its frame here. The stage
/// keeps the per-pixel values so [`prepare_finish_kernel`] can still sum them in
/// pixel order, which is bit-identical to the former in-line loop.
///
/// The Newtonian term is `weight * newton_grav * force[cell]`: a rigid pixel is
/// a mass `weight_of` and `grav_fx`/`grav_fy` hold the field's acceleration
/// (`Simulation.cpp:2120-2124`, the same term the particle pass adds to a
/// velocity), so the staged force integrates to the mass-weighted acceleration.
/// Sampling per pixel keeps the field's gradient, which gives the body a torque.
/// The rigid solver already applies the mode's own gravity to the whole body,
/// so the Newtonian field follows the same rule instead of the particle path's
/// `TYPE_SOLID` exclusion.
#[cube(launch_unchecked)]
pub(super) fn prepare_forces_kernel(
    bodies: &[u32],
    pixels: &[u32],
    elem_props: &[u32],
    air_pv: &[f32],
    air_vx: &[f32],
    air_vy: &[f32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    stage: &mut [f32],
    total: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    air_mode: u32,
    newton_grav: f32,
) {
    let i = ABSOLUTE_POS;
    if i < total as usize {
        let pb = i * 16usize;
        let si = i * 3usize;
        stage[si] = 0.0f32;
        stage[si + 1usize] = 0.0f32;
        stage[si + 2usize] = 0.0f32;
        let bb = pixels[pb + 11usize] as usize * 32usize;
        let cell_count = xcells * ycells;
        let wants_air = air_mode != 3u32;
        let wants_grav = newton_grav != 0.0f32;
        // A detached pixel and a pixel of an inactive body contribute nothing,
        // exactly like the guards of the sequential loop this replaces.
        if (wants_air || wants_grav)
            && cell_count > 0u32
            && pixels[pb + 10usize] == 0u32
            && (bodies[bb + 1usize] & 1u32) != 0u32
        {
            let ty = pixels[pb + 2usize];
            let origin_x = fword(bodies, bb + 14usize);
            let origin_y = fword(bodies, bb + 15usize);
            let cmx = fword(bodies, bb + 4usize);
            let cmy = fword(bodies, bb + 5usize);
            let angle = fword(bodies, bb + 8usize);
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let bvx = fword(bodies, bb + 6usize);
            let bvy = fword(bodies, bb + 7usize);
            let omega = fword(bodies, bb + 9usize);
            let lx = fword(pixels, pb) - origin_x;
            let ly = fword(pixels, pb + 1usize) - origin_y;
            let rx = lx * cos_a - ly * sin_a;
            let ry = lx * sin_a + ly * cos_a;
            let wx = cmx + rx;
            let wy = cmy + ry;
            if wx >= 0.0f32 && wy >= 0.0f32 && wx < xres as f32 && wy < yres as f32 {
                let cx = (wx * 0.25f32) as u32;
                let cy = (wy * 0.25f32) as u32;
                if cx < xcells && cy < ycells {
                    let row = cy * xcells;
                    let cell = (row + cx) as usize;
                    let mut fx = 0.0f32;
                    let mut fy = 0.0f32;
                    if wants_air {
                        let cxm = if cx == 0u32 { 0u32 } else { cx - 1u32 };
                        let cxp = if cx + 1u32 < xcells { cx + 1u32 } else { cx };
                        let cym = if cy == 0u32 { 0u32 } else { cy - 1u32 };
                        let cyp = if cy + 1u32 < ycells { cy + 1u32 } else { cy };
                        let grad_x = clampf(
                            air_pv[(row + cxm) as usize] - air_pv[(row + cxp) as usize],
                            -24.0f32,
                            24.0f32,
                        );
                        let grad_y = clampf(
                            air_pv[(cym * xcells + cx) as usize]
                                - air_pv[(cyp * xcells + cx) as usize],
                            -24.0f32,
                            24.0f32,
                        );
                        let drag_k = 0.02f32 * (1.0f32 + air_drag_of(elem_props, ty));
                        let avx = air_vx[cell];
                        let avy = air_vy[cell];
                        let vpx = bvx - omega * ry;
                        let vpy = bvy + omega * rx;
                        fx += grad_x * 0.05f32 + (avx - vpx) * drag_k;
                        fy += grad_y * 0.05f32 + (avy - vpy) * drag_k;
                    }
                    if wants_grav {
                        let w = weight_of(elem_props, ty);
                        fx += w * newton_grav * grav_fx[cell];
                        fy += w * newton_grav * grav_fy[cell];
                    }
                    stage[si] = fx;
                    stage[si + 1usize] = fy;
                    stage[si + 2usize] = rx * fy - ry * fx;
                }
            }
        }
    }
}

/// One thread per body: rebuilds the cached geometry when the pixel set
/// changed (upload or a detachment).
///
/// It runs before [`prepare_forces_kernel`] because the air force of a
/// fractured body must use the centroid / origin the rebuild just produced,
/// exactly the order the former single pass used.
#[allow(clippy::manual_clamp)]
#[cube(launch_unchecked)]
pub(super) fn prepare_geometry_kernel(
    bodies: &mut [u32],
    pixels: &mut [u32],
    elem_props: &[u32],
    part_flags: &mut [u32],
    owner: &mut [Atomic<u32>],
    body_count: u32,
) {
    let bi = ABSOLUTE_POS;
    if bi < body_count as usize {
        let bb = bi * 32usize;
        let flags = bodies[bb + 1usize];
        if (flags & 1u32) != 0u32 {
            let start = bodies[bb + 2usize];
            let count = bodies[bb + 3usize];
            // Geometry cache: mass, the weighted centroid offset, the
            // origin shift, inertia, radius and the alive count change only
            // when the pixel set does - on upload, and when `reconcile_kernel`
            // detaches a pixel. Both set the `B_GEOM_DIRTY` word, and every
            // other frame reads the cached words back instead of walking the
            // pixel list a second and third time.
            if bodies[bb + 26usize] != 0u32 {
                // Alive mass, weighted centroid offset and pixel count.
                let mut mass = 0.0f32;
                let mut moment_x = 0.0f32;
                let mut moment_y = 0.0f32;
                let mut alive = 0u32;
                for k in 0..count as usize {
                    let pb = (start as usize + k) * 16usize;
                    if pixels[pb + 10usize] == 0u32 {
                        let ty = pixels[pb + 2usize];
                        let w = weight_of(elem_props, ty);
                        let lx = fword_m(pixels, pb);
                        let ly = fword_m(pixels, pb + 1usize);
                        mass += w;
                        moment_x += w * lx;
                        moment_y += w * ly;
                        alive += 1u32;
                    }
                }

                if alive < 2u32 {
                    // The cluster fell apart: release every surviving pixel and
                    // deactivate the body. `mass` is irrelevant from here on.
                    bodies[bb + 1usize] = flags & 4294967294u32;
                    bodies[bb + 21usize] = 0u32;
                    for k in 0..count as usize {
                        let pb = (start as usize + k) * 16usize;
                        let pid = pixels[pb + 7usize];
                        if pixels[pb + 10usize] == 0u32 && pid != 4294967295u32 {
                            let slot = pid as usize;
                            part_flags[slot] &= 4294967277u32;
                            owner[slot].store(0u32);
                        }
                        pixels[pb + 10usize] = 1u32;
                    }
                } else {
                    if mass > 0.0f32 {
                        // Finite positive mass: nothing to repair.
                    } else {
                        mass = 1.0f32;
                    }
                    let mut cmx = fword_m(bodies, bb + 4usize);
                    let mut cmy = fword_m(bodies, bb + 5usize);
                    let mut origin_x = fword_m(bodies, bb + 14usize);
                    let mut origin_y = fword_m(bodies, bb + 15usize);
                    // `moment / mass` is the centroid in raw local coordinates;
                    // subtracting the current origin gives the incremental shift
                    // since the last fracture, so a stable body never drifts.
                    let off_x = moment_x / mass - origin_x;
                    let off_y = moment_y / mass - origin_y;
                    let angle = fword_m(bodies, bb + 8usize);
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    cmx += cos_a * off_x - sin_a * off_y;
                    cmy += sin_a * off_x + cos_a * off_y;
                    origin_x += off_x;
                    origin_y += off_y;

                    // Inertia about the new origin (`1/6` = unit-square moment).
                    let mut inertia = 0.0f32;
                    let mut max_r2 = 0.0f32;
                    for k in 0..count as usize {
                        let pb = (start as usize + k) * 16usize;
                        if pixels[pb + 10usize] == 0u32 {
                            let ty = pixels[pb + 2usize];
                            let w = weight_of(elem_props, ty);
                            let lx = fword_m(pixels, pb) - origin_x;
                            let ly = fword_m(pixels, pb + 1usize) - origin_y;
                            let r2 = lx * lx + ly * ly;
                            inertia += w * (r2 + 0.166_666_67f32);
                            if r2 > max_r2 {
                                max_r2 = r2;
                            }
                        }
                    }
                    if inertia > 1e-4f32 {
                        // Degenerate clusters keep their exact inertia.
                    } else {
                        inertia = mass * maxf(max_r2, 1.0f32);
                    }
                    let inv_mass = 1.0f32 / mass;
                    let inv_inertia = minf(1.0f32 / inertia, 0.5f32);
                    bodies[bb + 4usize] = u32::reinterpret(cmx);
                    bodies[bb + 5usize] = u32::reinterpret(cmy);
                    bodies[bb + 10usize] = u32::reinterpret(mass);
                    bodies[bb + 11usize] = u32::reinterpret(inv_mass);
                    bodies[bb + 12usize] = u32::reinterpret(inertia);
                    bodies[bb + 13usize] = u32::reinterpret(inv_inertia);
                    bodies[bb + 14usize] = u32::reinterpret(origin_x);
                    bodies[bb + 15usize] = u32::reinterpret(origin_y);
                    bodies[bb + 20usize] = u32::reinterpret(maxf(mass / alive as f32, 1.0f32));
                    bodies[bb + 21usize] = alive;
                    bodies[bb + 22usize] = u32::reinterpret(max_r2.sqrt());
                }
                bodies[bb + 26usize] = 0u32;
            }
        }
    }
}

/// One thread per body: sums the staged air force in pixel order, applies
/// gravity, the surface-velocity clamp, the sub-step count and the pose.
#[allow(clippy::manual_clamp)]
#[cube(launch_unchecked)]
pub(super) fn prepare_finish_kernel(
    bodies: &mut [u32],
    stage: &[f32],
    body_count: u32,
    xres: u32,
    yres: u32,
    gravity_mode: u32,
    custom_gx: f32,
    custom_gy: f32,
    frames: f32,
) {
    let bi = ABSOLUTE_POS;
    if bi < body_count as usize {
        let bb = bi * 32usize;
        if (bodies[bb + 1usize] & 1u32) != 0u32 {
            let start = bodies[bb + 2usize];
            let count = bodies[bb + 3usize];
            let inv_mass = fword_m(bodies, bb + 11usize);
            let inv_inertia = fword_m(bodies, bb + 13usize);
            let cmx = fword_m(bodies, bb + 4usize);
            let cmy = fword_m(bodies, bb + 5usize);

            // Gravity acceleration.
            let mut grav_x = 0.0f32;
            let mut grav_y = 0.0f32;
            if gravity_mode == 0u32 {
                grav_y += 0.15f32;
            } else if gravity_mode == 2u32 {
                let dx = xres as f32 * 0.5f32 - cmx;
                let dy = yres as f32 * 0.5f32 - cmy;
                let d = maxf((dx * dx + dy * dy).sqrt(), 1.0f32);
                grav_x += 0.15f32 * dx / d;
                grav_y += 0.15f32 * dy / d;
            } else if gravity_mode == 3u32 {
                grav_x += 0.15f32 * custom_gx;
                grav_y += 0.15f32 * custom_gy;
            }

            // Air pressure gradient + wind drag: `prepare_forces_kernel`
            // staged one contribution per pixel, and this sum visits them
            // in pixel order - the exact order the former in-line loop
            // used - so the parallel gate keeps the identical f32 sum.
            let mut force_x = 0.0f32;
            let mut force_y = 0.0f32;
            let mut torque = 0.0f32;
            for k in 0..count as usize {
                let si = (start as usize + k) * 3usize;
                force_x += stage[si];
                force_y += stage[si + 1usize];
                torque += stage[si + 2usize];
            }

            // Accelerations, sub-step count and pose write-back.
            let ax = clampf(grav_x + force_x * inv_mass, -4.0f32, 4.0f32);
            let ay = clampf(grav_y + force_y * inv_mass, -4.0f32, 4.0f32);
            let angular_acc = clampf(torque * inv_inertia, -0.25f32, 0.25f32);
            let radius = fword_m(bodies, bb + 22usize);
            // A pixel must never move more than one cell between two
            // contact passes, otherwise the contact's destination-cell
            // sample can jump over a one-pixel wall or a thin body. The
            // sub-step count is sized for the per-frame *surface* travel
            // (translation plus the rim speed `|omega| * radius`), and the
            // velocity is clamped to the same budget so the invariant also
            // holds for the acceleration the frame adds.
            let budget = 16.0f32 / frames;
            let raw_vx = fword_m(bodies, bb + 6usize);
            let raw_vy = fword_m(bodies, bb + 7usize);
            let raw_omega = fword_m(bodies, bb + 9usize);
            let limited = clamp_surface_velocity(raw_vx, raw_vy, raw_omega, radius, budget);
            // Size the sub-step count for the *fastest* surface speed this
            // frame can reach, including the acceleration still to come.
            // Using the current velocity alone made a free-falling body
            // stall at `n * 0.999` px/frame: the integrate clamp derived
            // from the same count could never let it reach the next
            // sub-step.
            let surface_now =
                (limited.0 * limited.0 + limited.1 * limited.1).sqrt() + fabs(limited.2) * radius;
            let next_vx = limited.0 + ax * frames;
            let next_vy = limited.1 + ay * frames;
            let next_omega = limited.2 + angular_acc * frames;
            let surface_next =
                (next_vx * next_vx + next_vy * next_vy).sqrt() + fabs(next_omega) * radius;
            let travel = maxf(surface_now, surface_next) * frames;
            let mut substeps = travel as u32;
            if (substeps as f32) < travel {
                substeps += 1u32;
            }
            if substeps < 1u32 {
                substeps = 1u32;
            }
            if substeps > 16u32 {
                substeps = 16u32;
            } // keep the manual bound: `clamp` is unavailable in WGSL kernels

            bodies[bb + 6usize] = u32::reinterpret(limited.0);
            bodies[bb + 7usize] = u32::reinterpret(limited.1);
            bodies[bb + 9usize] = u32::reinterpret(limited.2);
            bodies[bb + 16usize] = u32::reinterpret(ax);
            bodies[bb + 17usize] = u32::reinterpret(ay);
            bodies[bb + 18usize] = u32::reinterpret(angular_acc);
            bodies[bb + 19usize] = substeps;
        }
    }
}

#[cube(launch_unchecked)]
pub(super) fn integrate_substep_kernel(
    bodies: &mut [u32],
    body_count: u32,
    substep: u32,
    frames: f32,
) {
    let bi = ABSOLUTE_POS;
    if bi < body_count as usize {
        let bb = bi * 32usize;
        let flags = bodies[bb + 1usize];
        if (flags & 1u32) != 0u32 && substep < bodies[bb + 19usize] {
            let sdt = frames / bodies[bb + 19usize] as f32;
            let mut vx = fword_m(bodies, bb + 6usize);
            let mut vy = fword_m(bodies, bb + 7usize);
            let mut omega = fword_m(bodies, bb + 9usize);
            let ax = fword_m(bodies, bb + 16usize);
            let ay = fword_m(bodies, bb + 17usize);
            let angular_acc = fword_m(bodies, bb + 18usize);

            // Remember the pose this sub-step started from: contacts pick their
            // face from the sub-step's swept segment.
            bodies[bb + 23usize] = bodies[bb + 4usize];
            bodies[bb + 24usize] = bodies[bb + 5usize];
            bodies[bb + 25usize] = bodies[bb + 8usize];

            vx += ax * sdt;
            vy += ay * sdt;
            omega += angular_acc * sdt;
            // `0.999^sdt` / `0.995^sdt`, linearised around the per-frame
            // value: exact when `sdt == 1` (the common single-sub-step case)
            // and within `1e-4` for the sub-stepped fractions.
            let linear_damp = 1.0f32 - 0.001f32 * sdt;
            let angular_damp = 1.0f32 - 0.005f32 * sdt;
            vx *= linear_damp;
            vy *= linear_damp;
            omega *= angular_damp;
            let clamped = clamp_velocity(vx, vy, omega);
            vx = clamped.0;
            vy = clamped.1;
            omega = clamped.2;
            // Keep every pixel within one cell of its sub-step start even
            // after the acceleration of this sub-step: the contact pass's
            // single destination-cell sample is only complete under that
            // invariant. `0.999` leaves a small guard against float rounding
            // landing exactly on a cell boundary.
            let radius = fword_m(bodies, bb + 22usize);
            let sub_budget = bodies[bb + 19usize] as f32 * 0.999f32 / frames;
            let surface = clamp_surface_velocity(vx, vy, omega, radius, sub_budget);
            vx = surface.0;
            vy = surface.1;
            omega = surface.2;

            let cmx = fword_m(bodies, bb + 4usize);
            let cmy = fword_m(bodies, bb + 5usize);
            let angle = fword_m(bodies, bb + 8usize);
            let pred_cm_x = cmx + vx * sdt;
            let pred_cm_y = cmy + vy * sdt;
            let pred_angle = wrap_angle(angle + omega * sdt);

            bodies[bb + 6usize] = u32::reinterpret(vx);
            bodies[bb + 7usize] = u32::reinterpret(vy);
            bodies[bb + 9usize] = u32::reinterpret(omega);
            bodies[bb + 4usize] = u32::reinterpret(pred_cm_x);
            bodies[bb + 5usize] = u32::reinterpret(pred_cm_y);
            bodies[bb + 8usize] = u32::reinterpret(pred_angle);
        }
    }
}

/// Adds one contact sample to the manifold slot keyed by
/// `(other body, signed axis)`.
///
/// Every contact thread only writes its own body's slot row, so no
/// synchronisation is needed; slots are searched in a fixed order, which
/// keeps the manifold layout (and therefore the resolution order) identical
/// no matter how the GPU schedules the threads.
#[cube]
fn manifold_add(
    manifolds: &mut [u32],
    mbase: usize,
    dir: u32,
    other: u32,
    rxa: f32,
    rya: f32,
    obx: f32,
    oby: f32,
) {
    let mut matched = 0u32;
    for s in 0..MANIFOLDS {
        let sb = mbase + s * M_STRIDE;
        if matched == 0u32 && manifolds[sb + M_DIR] == dir && manifolds[sb + M_BODY] == other {
            manifold_add_slot(manifolds, sb, rxa, rya, obx, oby);
            matched = 1u32;
        }
    }
    if matched == 0u32 {
        let mut created = 0u32;
        for s in 0..MANIFOLDS {
            let sb = mbase + s * M_STRIDE;
            if created == 0u32 && manifolds[sb + M_DIR] == M_FREE {
                manifolds[sb + M_DIR] = dir;
                manifolds[sb + M_BODY] = other;
                manifolds[sb + M_COUNT] = 0u32;
                manifolds[sb + M_ARM_AX] = u32::reinterpret(0.0f32);
                manifolds[sb + M_ARM_AY] = u32::reinterpret(0.0f32);
                manifolds[sb + M_OB_X] = u32::reinterpret(0.0f32);
                manifolds[sb + M_OB_Y] = u32::reinterpret(0.0f32);
                manifold_add_slot(manifolds, sb, rxa, rya, obx, oby);
                created = 1u32;
            }
        }
    }
}

/// Accumulates one contact sample into an initialised slot.
#[cube]
fn manifold_add_slot(manifolds: &mut [u32], sb: usize, rxa: f32, rya: f32, obx: f32, oby: f32) {
    manifolds[sb + M_COUNT] += 1u32;
    manifolds[sb + M_ARM_AX] = u32::reinterpret(f32::reinterpret(manifolds[sb + M_ARM_AX]) + rxa);
    manifolds[sb + M_ARM_AY] = u32::reinterpret(f32::reinterpret(manifolds[sb + M_ARM_AY]) + rya);
    manifolds[sb + M_OB_X] = u32::reinterpret(f32::reinterpret(manifolds[sb + M_OB_X]) + obx);
    manifolds[sb + M_OB_Y] = u32::reinterpret(f32::reinterpret(manifolds[sb + M_OB_Y]) + oby);
}

/// Manifold direction word of a swept contact face (`0=+x, 1=-x, 2=+y, 3=-y`).
#[cube]
fn face_dir(nx: f32, ny: f32) -> u32 {
    let mut dir = 0u32;
    if nx < 0.0f32 {
        dir = 1u32;
    } else if ny > 0.0f32 {
        dir = 2u32;
    } else if ny < 0.0f32 {
        dir = 3u32;
    }
    dir
}

/// Adds one rigid-pair contact sample to the manifold.
///
/// The receiver is identified by its cell centre `(ox, oy)`; the sender's
/// swept segment must cross that cell's box. Returns
/// `(dir, corr_pos_x, corr_neg_x, corr_pos_y, corr_neg_y)`: `dir == M_FREE`
/// means the segment did not cross, otherwise the four values are the
/// positional corrections the caller merges into its deepest-per-axis
/// accumulators. The frame-start `pmap` path uses both halves; the live-grid
/// fallback uses only the manifold sample (the receiver's predicted pose has
/// not gone through this sub-step's support correction yet), so the two
/// sources agree on the face while the pose correction stays with the
/// settled cell.
#[cube]
fn pair_contact_sample(
    manifolds: &mut [u32],
    mbase: usize,
    prev_wx: f32,
    prev_wy: f32,
    wx: f32,
    wy: f32,
    ox: f32,
    oy: f32,
    rx: f32,
    ry: f32,
    other: u32,
    slop: f32,
) -> (u32, f32, f32, f32, f32) {
    let mut dir = seed_u32(mbase as u32, 65535u32);
    let mut pos_x = seed_f32(slop, 0.0f32);
    let mut neg_x = seed_f32(slop, 0.0f32);
    let mut pos_y = seed_f32(slop, 0.0f32);
    let mut neg_y = seed_f32(slop, 0.0f32);
    let face = swept_face(
        prev_wx,
        prev_wy,
        wx,
        wy,
        ox - 0.5f32,
        oy - 0.5f32,
        ox + 0.5f32,
        oy + 0.5f32,
    );
    if face.0 != 0.0f32 || face.1 != 0.0f32 {
        dir = face_dir(face.0, face.1);
        manifold_add(manifolds, mbase, dir, other, rx, ry, ox, oy);
        if face.0 > 0.0f32 {
            pos_x = face.2 + slop;
        } else if face.0 < 0.0f32 {
            neg_x = face.2 + slop;
        } else if face.1 > 0.0f32 {
            pos_y = face.2 + slop;
        } else {
            neg_y = face.2 + slop;
        }
    }
    (dir, pos_x, neg_x, pos_y, neg_y)
}

#[cube(launch_unchecked)]
pub(super) fn contact_kernel(
    bodies: &mut [Atomic<u32>],
    pixels: &[u32],
    surface: &[u32],
    manifolds: &mut [u32],
    pmap: &mut [u32],
    part_ty: &[u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    elem_props: &[u32],
    owner: &mut [Atomic<u32>],
    live: &mut [Atomic<u32>],
    body_count: u32,
    substep: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    gravity_mode: u32,
    frames: f32,
    mode: u32,
    iter: u32,
    generation: u32,
) {
    let bi = ABSOLUTE_POS;
    if bi < body_count as usize {
        let bb = bi * 32usize;
        let flags = bodies[bb + 1usize].load();
        if (flags & 1u32) != 0u32 && substep < bodies[bb + 19usize].load() {
            let sdt = frames / bodies[bb + 19usize].load() as f32;
            let start = bodies[bb + 2usize].load();
            let count = bodies[bb + 3usize].load();
            let alive = maxf(bodies[bb + 21usize].load() as f32, 1.0f32);
            let inv_pixels = 1.0f32 / alive;
            let unit_weight = fword_a(bodies, bb + 20usize);
            let inv_mass = fword_a(bodies, bb + 11usize);
            let inv_inertia = fword_a(bodies, bb + 13usize);
            let origin_x = fword_a(bodies, bb + 14usize);
            let origin_y = fword_a(bodies, bb + 15usize);
            let radius = fword_a(bodies, bb + 22usize);
            let cmx = fword_a(bodies, bb + 4usize);
            let cmy = fword_a(bodies, bb + 5usize);
            let prev_cmx = fword_a(bodies, bb + 23usize);
            let prev_cmy = fword_a(bodies, bb + 24usize);
            let angle = fword_a(bodies, bb + 8usize);
            let prev_angle = fword_a(bodies, bb + 25usize);
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let prev_cos = prev_angle.cos();
            let prev_sin = prev_angle.sin();
            let mut vx = fword_a(bodies, bb + 6usize);
            let mut vy = fword_a(bodies, bb + 7usize);
            let mut omega = fword_a(bodies, bb + 9usize);

            // Loose powder supports a cluster only while the overburden
            // pressure stays below the powder's yield strength. `unit_weight *
            // radius` stands in for the height of the body: a brush-sized
            // boulder ploughs through a sand pile (the fluid branch below
            // displaces the grains), while a small cluster rests on the
            // surface instead of sinking at free-fall speed.
            let support_powder = unit_weight * radius < 1500.0f32;

            // Positional correction: the deepest penetration per signed axis.
            // Taking the maximum (instead of summing every pixel) keeps a flat
            // landing symmetric - one cell of correction is enough for the
            // whole row - and stops the body being launched out of contact.
            let mut corr_pos_x = 0.0f32;
            let mut corr_neg_x = 0.0f32;
            let mut corr_pos_y = 0.0f32;
            let mut corr_neg_y = 0.0f32;
            // Separation kept between a pixel centre and a blocked face so the
            // rasteriser's `round` cannot land back inside the cell.
            let slop = 0.02f32;
            // Every contact sample is grouped by `(other body, axis)` into this
            // body's manifold row and resolved once per group at the average
            // lever arm. Resolving pixel by pixel in list order tips a flat
            // landing, and for a rigid-rigid pair the partner's reaction is
            // applied only after the pass, so per-pixel resolution would also
            // re-apply the impulse for every contact pixel.
            let mbase = bi * (MANIFOLDS * M_STRIDE);
            for s in 0..MANIFOLDS {
                manifolds[mbase + s * M_STRIDE + M_DIR] = M_FREE;
            }
            if mode == 0u32 && iter == 0u32 {
                // The pair phase starts a fresh support chain: the previous
                // sub-step's published direction stays readable, the scratch
                // this phase fills is cleared here (own body, no race).
                bodies[bb + B_SUPPORT_NEXT].store(SUPPORT_NONE);
            }

            // The surface bitmap packs one flag per device pixel (see
            // `RigidGpu`). This thread pays a full memory latency per word it
            // reads, so a word whose remaining bits are all clear is skipped in
            // one step instead of one latency per pixel: a solid 200x200 body
            // has 800 surface pixels among 40 000 and costs ~1 250 word reads
            // rather than 40 000. Interior pixels - whose eight neighbours are
            // all body pixels - never keep their bit; a detachment sets the
            // neighbours' bits (see `reconcile_kernel`). The detached test
            // stays because a pixel can be promoted by a collision with
            // another detachment.
            let mut k = 0usize;
            while k < count as usize {
                let gi = start as usize + k;
                let pb = gi * 16usize;
                let word_bits = surface[gi / 32usize] >> (gi % 32usize) as u32;
                if word_bits == 0u32 {
                    // Nothing left in this word: resume at the next one.
                    k = (gi / 32usize + 1usize) * 32usize - start as usize;
                }
                if word_bits != 0u32 && (word_bits & 1u32) != 0u32 && pixels[pb + 10usize] == 0u32 {
                    let ty = pixels[pb + 2usize];
                    let props = props_of(elem_props, ty);
                    let lx = fword(pixels, pb) - origin_x;
                    let ly = fword(pixels, pb + 1usize) - origin_y;
                    let rx = lx * cos_a - ly * sin_a;
                    let ry = lx * sin_a + ly * cos_a;
                    let wx = cmx + rx;
                    let wy = cmy + ry;
                    let prx = lx * prev_cos - ly * prev_sin;
                    let pry = lx * prev_sin + ly * prev_cos;
                    let prev_wx = prev_cmx + prx;
                    let prev_wy = prev_cmy + pry;

                    // Canvas boundary: every edge behaves as a solid wall.
                    // Only the static phase owns world contacts; the pair
                    // phase must not touch them (see the launch loop).
                    if mode != 0u32 {
                        if wx < 0.5f32 {
                            manifold_add(manifolds, mbase, 0u32, 0u32, rx, ry, 0.0f32, 0.0f32);
                            corr_pos_x =
                                maxf(corr_pos_x, clampf(0.5f32 - wx, 0.0f32, 1.0f32) + slop);
                        } else if wx > xres as f32 - 0.5f32 {
                            manifold_add(manifolds, mbase, 1u32, 0u32, rx, ry, 0.0f32, 0.0f32);
                            corr_neg_x = maxf(
                                corr_neg_x,
                                clampf(wx - (xres as f32 - 0.5f32), 0.0f32, 1.0f32) + slop,
                            );
                        }
                        if wy < 0.5f32 {
                            manifold_add(manifolds, mbase, 2u32, 0u32, rx, ry, 0.0f32, 0.0f32);
                            corr_pos_y =
                                maxf(corr_pos_y, clampf(0.5f32 - wy, 0.0f32, 1.0f32) + slop);
                        } else if wy > yres as f32 - 0.5f32 {
                            manifold_add(manifolds, mbase, 3u32, 0u32, rx, ry, 0.0f32, 0.0f32);
                            corr_neg_y = maxf(
                                corr_neg_y,
                                clampf(wy - (yres as f32 - 0.5f32), 0.0f32, 1.0f32) + slop,
                            );
                        }
                    }

                    // `round` matches the rasteriser's cell: `wx as i32` used
                    // to truncate, so the contact sampled one cell behind the
                    // cell the pixel was actually placed in.
                    let gx = wx.round() as i32;
                    let gy = wy.round() as i32;
                    if gx >= 0i32 && gy >= 0i32 && gx < xres as i32 && gy < yres as i32 {
                        // Wall contact against the coarse block. The block's
                        // bounds are in pixel-centre coordinates, and the
                        // swept face is the one the pixel crossed.
                        if mode != 0u32
                            && wall_blocks(
                                wall_bmap,
                                wall_emap,
                                gx,
                                gy,
                                xcells as i32,
                                ycells as i32,
                                props,
                            )
                        {
                            let block_x = (gx / 4i32) * 4i32;
                            let block_y = (gy / 4i32) * 4i32;
                            let face = swept_face(
                                prev_wx,
                                prev_wy,
                                wx,
                                wy,
                                block_x as f32 - 0.5f32,
                                block_y as f32 - 0.5f32,
                                block_x as f32 + 3.5f32,
                                block_y as f32 + 3.5f32,
                            );
                            if face.0 != 0.0f32 || face.1 != 0.0f32 {
                                manifold_add(
                                    manifolds,
                                    mbase,
                                    face_dir(face.0, face.1),
                                    0u32,
                                    rx,
                                    ry,
                                    0.0f32,
                                    0.0f32,
                                );
                                if face.0 > 0.0f32 {
                                    corr_pos_x = maxf(corr_pos_x, face.2 + slop);
                                } else if face.0 < 0.0f32 {
                                    corr_neg_x = maxf(corr_neg_x, face.2 + slop);
                                } else if face.1 > 0.0f32 {
                                    corr_pos_y = maxf(corr_pos_y, face.2 + slop);
                                } else {
                                    corr_neg_y = maxf(corr_neg_y, face.2 + slop);
                                }
                            }
                        }

                        // Particle contact through the authoritative pmap.
                        let idx = (gy as u32 * xres + gx as u32) as usize;
                        // Pair phase: the frame-start `pmap` cell is sampled
                        // first, exactly as before, because its pose already
                        // carries the receiver's support correction from the
                        // previous sub-step. Only when no rigid pixel sits
                        // there is the live grid consulted: a partner that
                        // moved into this cell during the frame is not in
                        // `pmap` yet, and without the live sample two bodies
                        // approaching each other would cross entirely.
                        let mut rigid_paired = 0u32;
                        if mode == 0u32 {
                            let pm0 = pmap[idx];
                            if pm0 != 0u32 {
                                let other0 = unpack_slot(pm0);
                                if other0 != 4294967295u32 {
                                    let s0 = other0 as usize;
                                    let own0 = owner[s0].load();
                                    if part_ty[s0] != 0u32
                                        && own0 != bi as u32 + 1u32
                                        && (own0 != 0u32 || (part_flags[s0] & 16u32) != 0u32)
                                    {
                                        rigid_paired = 1u32;
                                    }
                                }
                            }
                        }
                        if mode == 0u32 && rigid_paired == 0u32 {
                            let word = live[idx].load();
                            let slot = word & 8388607u32;
                            if (word >> 23u32) == generation && slot != 0u32 {
                                let oslot = (slot - 1u32) as usize;
                                let recv = owner[oslot].load();
                                if recv != 0u32
                                    && recv != bi as u32 + 1u32
                                    && part_ty[oslot] != 0u32
                                {
                                    // Only the impulse comes from the live
                                    // sample. Its positional correction would
                                    // be measured against the receiver's
                                    // *pre-support* pose for this sub-step
                                    // (the static phase corrects the receiver
                                    // after the pair phase), so it would fight
                                    // the support that keeps a resting stack
                                    // apart; the frame-start `pmap` path owns
                                    // the pose correction.
                                    let _sample = pair_contact_sample(
                                        manifolds, mbase, prev_wx, prev_wy, wx, wy, gx as f32,
                                        gy as f32, rx, ry, recv, slop,
                                    );
                                }
                            }
                        }
                        let pm = pmap[idx];
                        if pm != 0u32 {
                            let other = unpack_slot(pm);
                            if other != 4294967295u32 {
                                let oslot = other as usize;
                                let other_ty = part_ty[oslot];
                                if other_ty != 0u32 {
                                    let own = owner[oslot].load();
                                    if own != bi as u32 + 1u32 {
                                        // Two phases, in launch order:
                                        // * pair phase (`mode == 0`): a rigid
                                        //   obstacle of another body. The
                                        //   impulse is resolved against the
                                        //   partner's inverse mass / inertia /
                                        //   centre and the equal-and-opposite
                                        //   share is queued on the partner.
                                        // * static phase (`mode != 0`): the
                                        //   world - walls, terrain, canvas
                                        //   edge and loose particles. It runs
                                        //   *after* every queued reaction has
                                        //   been folded in, so a body pushed
                                        //   by the body above it has its
                                        //   support absorb that motion in the
                                        //   same sub-step instead of ending
                                        //   the sub-step moving into it.
                                        // A flag-only rigid particle (no body
                                        // owner) has no live-grid entry and no
                                        // body record: it stays a static
                                        // obstacle keyed as body 0, through the
                                        // frame-start `pmap` cell.
                                        let other_rigid =
                                            own != 0u32 || (part_flags[oslot] & 16u32) != 0u32;
                                        if other_rigid && mode == 0u32 {
                                            let ox = part_x[oslot].round();
                                            let oy = part_y[oslot].round();
                                            let sample = pair_contact_sample(
                                                manifolds, mbase, prev_wx, prev_wy, wx, wy, ox, oy,
                                                rx, ry, own, slop,
                                            );
                                            if sample.0 != M_FREE {
                                                corr_pos_x = maxf(corr_pos_x, sample.1);
                                                corr_neg_x = maxf(corr_neg_x, sample.2);
                                                corr_pos_y = maxf(corr_pos_y, sample.3);
                                                corr_neg_y = maxf(corr_neg_y, sample.4);
                                            }
                                        } else if !other_rigid && mode != 0u32 {
                                            let other_props = props_of(elem_props, other_ty);
                                            // A powder that can carry this body is a
                                            // hard contact; otherwise (liquids, gases
                                            // and powder under a body too heavy for
                                            // its yield strength) it is displaced.
                                            let powder = (other_props & 1u32) != 0u32;
                                            let hard = (other_props & 4u32) != 0u32
                                                || (powder && support_powder);
                                            if hard {
                                                let ox = part_x[oslot].round();
                                                let oy = part_y[oslot].round();
                                                let face = swept_face(
                                                    prev_wx,
                                                    prev_wy,
                                                    wx,
                                                    wy,
                                                    ox - 0.5f32,
                                                    oy - 0.5f32,
                                                    ox + 0.5f32,
                                                    oy + 0.5f32,
                                                );
                                                if face.0 != 0.0f32 || face.1 != 0.0f32 {
                                                    // A loose particle's CA velocity
                                                    // is not its motion: a powder at
                                                    // rest keeps accumulating
                                                    // gravity in `part_vy` while the
                                                    // movement pass refuses the
                                                    // step, and feeding that back
                                                    // made a resting pile "carry"
                                                    // the body at a clamped 1 px per
                                                    // frame. A world contact carries
                                                    // no surface velocity.
                                                    manifold_add(
                                                        manifolds,
                                                        mbase,
                                                        face_dir(face.0, face.1),
                                                        0u32,
                                                        rx,
                                                        ry,
                                                        ox,
                                                        oy,
                                                    );
                                                    if face.0 > 0.0f32 {
                                                        corr_pos_x =
                                                            maxf(corr_pos_x, face.2 + slop);
                                                    } else if face.0 < 0.0f32 {
                                                        corr_neg_x =
                                                            maxf(corr_neg_x, face.2 + slop);
                                                    } else if face.1 > 0.0f32 {
                                                        corr_pos_y =
                                                            maxf(corr_pos_y, face.2 + slop);
                                                    } else {
                                                        corr_neg_y =
                                                            maxf(corr_neg_y, face.2 + slop);
                                                    }
                                                }
                                            } else if (other_props & (2u32 | 1u32 | 8u32)) != 0u32 {
                                                // Fluid: buoyancy + viscous drag.
                                                if gravity_mode != 1u32 {
                                                    let fluid_weight =
                                                        weight_of(elem_props, other_ty);
                                                    let ratio = clampf(
                                                        fluid_weight / maxf(unit_weight, 1.0f32),
                                                        0.0f32,
                                                        2.0f32,
                                                    );
                                                    vy -=
                                                        0.15f32 * ratio * 0.5f32 * sdt * inv_pixels;
                                                }
                                                let surf_vx = vx - omega * ry;
                                                let surf_vy = vy + omega * rx;
                                                let rel_x = part_vx[oslot] - surf_vx;
                                                let rel_y = part_vy[oslot] - surf_vy;
                                                let k2 = 0.08f32 * sdt * inv_pixels;
                                                vx += rel_x * k2;
                                                vy += rel_y * k2;
                                                omega += (rx * rel_y - ry * rel_x)
                                                    * k2
                                                    * inv_inertia
                                                    * 0.25f32;
                                                let clamped = clamp_velocity(vx, vy, omega);
                                                vx = clamped.0;
                                                vy = clamped.1;
                                                omega = clamped.2;

                                                // Push the free particle one cell away
                                                // from the body pixel, like
                                                // `displace_free_particle`.
                                                let mut dir_x = part_x[oslot] - wx;
                                                let mut dir_y = part_y[oslot] - wy;
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
                                                let mut best = -(xres as f32) - 1000.0f32;
                                                let mut bx = ogx;
                                                let mut by = ogy;
                                                let mut found = false;
                                                for oy_i in 0i32..3i32 {
                                                    for ox_i in 0i32..3i32 {
                                                        let oxi = ox_i - 1i32;
                                                        let oyi = oy_i - 1i32;
                                                        if oxi != 0i32 || oyi != 0i32 {
                                                            let nx = ogx + oxi;
                                                            let ny = ogy + oyi;
                                                            if nx >= 0i32
                                                                && ny >= 0i32
                                                                && nx < xres as i32
                                                                && ny < yres as i32
                                                            {
                                                                let ni = (ny as u32 * xres
                                                                    + nx as u32)
                                                                    as usize;
                                                                if pmap[ni] == 0u32
                                                                    && !wall_blocks(
                                                                        wall_bmap,
                                                                        wall_emap,
                                                                        nx,
                                                                        ny,
                                                                        xcells as i32,
                                                                        ycells as i32,
                                                                        4u32,
                                                                    )
                                                                {
                                                                    let score = oxi as f32 * dir_x
                                                                        + oyi as f32 * dir_y;
                                                                    if score > best {
                                                                        best = score;
                                                                        bx = nx;
                                                                        by = ny;
                                                                        found = true;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                if found {
                                                    let old =
                                                        (ogy as u32 * xres + ogx as u32) as usize;
                                                    if pmap[old] == pack_map(other, other_ty) {
                                                        pmap[old] = 0u32;
                                                    }
                                                    part_x[oslot] = bx as f32;
                                                    part_y[oslot] = by as f32;
                                                    part_vx[oslot] = surf_vx + dir_x * 0.5f32;
                                                    part_vy[oslot] = surf_vy + dir_y * 0.5f32;
                                                    let ni =
                                                        (by as u32 * xres + bx as u32) as usize;
                                                    pmap[ni] = pack_map(other, other_ty);
                                                } else {
                                                    part_vx[oslot] += dir_x * 0.5f32;
                                                    part_vy[oslot] += dir_y * 0.5f32;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if word_bits != 0u32 {
                    k += 1usize;
                }
            }

            // Resolve every manifold once, in slot order. A slot with a
            // receiver supplies that body's inverse mass / inertia / centre and
            // its *current* velocity (this phase runs as its own dispatch, so no
            // other thread can be writing a body record), which makes the pair
            // response symmetric and free of the rasteriser's one-frame lag.
            //
            // Neither share is applied here: both are queued as fixed-point
            // velocity deltas and folded in by `apply_body_impulses_kernel`.
            // That keeps this phase a pure function of the pre-phase state, so
            // running it twice is a deterministic constraint iteration instead
            // of an accumulating one.
            for s in 0..MANIFOLDS {
                let sb = mbase + s * M_STRIDE;
                let dir = manifolds[sb + M_DIR];
                if dir != M_FREE {
                    let samples = manifolds[sb + M_COUNT];
                    if samples > 0u32 {
                        let inv_samples = 1.0f32 / samples as f32;
                        let normal = dir_normal(dir);
                        let rxa = f32::reinterpret(manifolds[sb + M_ARM_AX]) * inv_samples;
                        let rya = f32::reinterpret(manifolds[sb + M_ARM_AY]) * inv_samples;
                        let obx = f32::reinterpret(manifolds[sb + M_OB_X]) * inv_samples;
                        let oby = f32::reinterpret(manifolds[sb + M_OB_Y]) * inv_samples;
                        let other = manifolds[sb + M_BODY];
                        // Runtime-seeded zeros: a literal seed cannot be
                        // overwritten by the conditional loads below.
                        let mut inv_mass_b = zero_of(inv_mass);
                        let mut inv_inertia_b = zero_of(inv_inertia);
                        let mut rbx = zero_of(rxa);
                        let mut rby = zero_of(rya);
                        let mut ovx = zero_of(rxa);
                        let mut ovy = zero_of(rya);
                        if other != 0u32 {
                            let obb = (other as usize - 1usize) * 32usize;
                            inv_mass_b = fword_a(bodies, obb + 11usize);
                            inv_inertia_b = fword_a(bodies, obb + 13usize);
                            rbx = obx - fword_a(bodies, obb + 4usize);
                            rby = oby - fword_a(bodies, obb + 5usize);
                            let omega_b = fword_a(bodies, obb + 9usize);
                            ovx = fword_a(bodies, obb + 6usize) - omega_b * rby;
                            ovy = fword_a(bodies, obb + 7usize) + omega_b * rbx;
                            // A receiver that is itself held by the world along
                            // this contact normal is an anchor: sharing the
                            // impulse with a mass that its own support already
                            // pins would leave the sender with half its normal
                            // velocity every contact, and a stack would creep
                            // into itself. The support direction is published by
                            // the receiver's static phase and propagated up a
                            // resting stack one link per pair phase.
                            let support = support_normal(bodies[obb + B_SUPPORT].load());
                            if support.0 * normal.0 + support.1 * normal.1 > 0.5f32 {
                                inv_mass_b = 0.0f32;
                                inv_inertia_b = 0.0f32;
                                // The receiver is held by the world, so the
                                // velocity it carries right now is the one its
                                // own static contact is about to cancel. Using
                                // it here would hand the sender that residual
                                // (a stack would sink one gravity step per
                                // frame); an immovable receiver contributes no
                                // velocity at all.
                                ovx = 0.0f32;
                                ovy = 0.0f32;
                                bodies[bb + B_SUPPORT_NEXT].store(support_code(normal.0, normal.1));
                            }
                        }
                        let res = resolve_pair_impulse(
                            vx,
                            vy,
                            omega,
                            inv_mass,
                            inv_inertia,
                            inv_mass_b,
                            inv_inertia_b,
                            rxa,
                            rya,
                            rbx,
                            rby,
                            normal.0,
                            normal.1,
                            ovx,
                            ovy,
                        );
                        if mode == 0u32 {
                            // Pair phase: queue the sender's share like the
                            // receiver's, so the phase never mutates a state it
                            // may already have read for another manifold. The
                            // accumulator words hold the resulting *velocity*
                            // delta (`J * inv_mass`, torque * inv_inertia`)
                            // rather than the raw impulse, whose magnitude for
                            // a brush-radius body (~4e7) is far outside the
                            // fixed-point range; the algebra is identical
                            // because the apply pass would multiply by the
                            // same inverse mass.
                            let dvx = res.3 * inv_mass;
                            let dvy = res.4 * inv_mass;
                            let dw = (rxa * res.4 - rya * res.3) * inv_inertia;
                            bodies[bb + B_REACT_VX]
                                .fetch_add(u32::reinterpret((dvx * REACTION_SCALE) as i32));
                            bodies[bb + B_REACT_VY]
                                .fetch_add(u32::reinterpret((dvy * REACTION_SCALE) as i32));
                            bodies[bb + B_REACT_VW]
                                .fetch_add(u32::reinterpret((dw * REACTION_SCALE) as i32));
                            if other != 0u32 {
                                let obb = (other as usize - 1usize) * 32usize;
                                let rvx = zero_of(res.3) - res.3 * inv_mass_b;
                                let rvy = zero_of(res.4) - res.4 * inv_mass_b;
                                let rw =
                                    zero_of(res.3) - (rbx * res.4 - rby * res.3) * inv_inertia_b;
                                bodies[obb + B_REACT_VX]
                                    .fetch_add(u32::reinterpret((rvx * REACTION_SCALE) as i32));
                                bodies[obb + B_REACT_VY]
                                    .fetch_add(u32::reinterpret((rvy * REACTION_SCALE) as i32));
                                bodies[obb + B_REACT_VW]
                                    .fetch_add(u32::reinterpret((rw * REACTION_SCALE) as i32));
                            }
                        } else {
                            // Static phase: the last phase of the sub-step, so
                            // its response is applied directly.
                            vx = res.0;
                            vy = res.1;
                            omega = res.2;
                        }
                    }
                }
            }

            if mode != 0u32 {
                // Publish the support direction for the next sub-step's pair
                // phase: the world contact with the deepest correction, else the
                // direction this sub-step's pair phase propagated up the stack.
                let mut best = zero_of(corr_pos_x);
                let mut code = 0u32;
                if corr_pos_x > best {
                    best = corr_pos_x;
                    code = 1u32;
                }
                if corr_neg_x > best {
                    best = corr_neg_x;
                    code = 2u32;
                }
                if corr_pos_y > best {
                    best = corr_pos_y;
                    code = 3u32;
                }
                if corr_neg_y > best {
                    // The last comparison needs no `best` update: nothing
                    // reads it after this branch.
                    code = 4u32;
                }
                if code == 0u32 {
                    code = bodies[bb + B_SUPPORT_NEXT].load();
                }
                if code == 0u32 {
                    // A resting contact only crosses the sampled face on some
                    // frames (the sub-step advance has to reach it), so the
                    // published support has to outlive the frame that found it.
                    // Keep it while the body is slow; a body that jumps or is
                    // launched is well above this threshold and drops it.
                    let vx = fword_a(bodies, bb + 6usize);
                    let vy = fword_a(bodies, bb + 7usize);
                    if vx * vx + vy * vy < 0.25f32 {
                        code = bodies[bb + B_SUPPORT].load();
                    }
                }
                bodies[bb + B_SUPPORT].store(code);
            }

            // Commit the deepest separation per axis; a body wedged between
            // two walls keeps the two corrections opposing each other.
            let final_x = clampf(cmx + corr_pos_x - corr_neg_x, 0.5f32, xres as f32 - 0.5f32);
            let final_y = clampf(cmy + corr_pos_y - corr_neg_y, 0.5f32, yres as f32 - 0.5f32);

            bodies[bb + 6usize].store(u32::reinterpret(vx));
            bodies[bb + 7usize].store(u32::reinterpret(vy));
            bodies[bb + 9usize].store(u32::reinterpret(omega));
            bodies[bb + 4usize].store(u32::reinterpret(final_x));
            bodies[bb + 5usize].store(u32::reinterpret(final_y));
        }
    }
}

/// Applies the reactions queued by every contact thread.
///
/// One thread per body, run after the whole contact pass, so a receiver's
/// velocity never depends on the order in which the senders ran: the atomic
/// adds are modular integers, and this pass converts them to floats once.
#[cube(launch_unchecked)]
pub(super) fn apply_body_impulses_kernel(bodies: &mut [Atomic<u32>], body_count: u32) {
    let bi = ABSOLUTE_POS;
    if bi < body_count as usize {
        let bb = bi * 32usize;
        let fx = i32::reinterpret(bodies[bb + B_REACT_VX].load());
        let fy = i32::reinterpret(bodies[bb + B_REACT_VY].load());
        let fw = i32::reinterpret(bodies[bb + B_REACT_VW].load());
        bodies[bb + B_REACT_VX].store(0u32);
        bodies[bb + B_REACT_VY].store(0u32);
        bodies[bb + B_REACT_VW].store(0u32);
        let flags = bodies[bb + 1usize].load();
        if (flags & 1u32) != 0u32 {
            let dvx = fx as f32 * REACTION_INV_SCALE;
            let dvy = fy as f32 * REACTION_INV_SCALE;
            let dw = fw as f32 * REACTION_INV_SCALE;
            let vx = fword_a(bodies, bb + 6usize) + dvx;
            let vy = fword_a(bodies, bb + 7usize) + dvy;
            let omega = fword_a(bodies, bb + 9usize) + dw;
            // The *pose* is not advanced by the delta here. The pair phase
            // already stops this body's predicted pose at the receiver's
            // rasterised face, and that correction is measured against the
            // sub-step's integrated position. Advancing the pose again by
            // `dv * sdt` would subtract this sub-step's relative motion a
            // second time: the body would be lifted off its support, fall
            // back under gravity and repeat, which read as a see-saw under a
            // resting stack. The velocity delta is folded in alone and the
            // next sub-step integrates from the corrected pose.
            let clamped = clamp_velocity(vx, vy, omega);
            bodies[bb + 6usize].store(u32::reinterpret(clamped.0));
            bodies[bb + 7usize].store(u32::reinterpret(clamped.1));
            bodies[bb + 9usize].store(u32::reinterpret(clamped.2));
        }
    }
}
