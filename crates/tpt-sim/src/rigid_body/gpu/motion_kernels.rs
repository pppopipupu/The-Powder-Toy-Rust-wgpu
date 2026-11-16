//! Device preparation, broadphase, and impulse-fold kernels.

#![allow(clippy::too_many_arguments, clippy::manual_clamp, clippy::collapsible_if)]

use super::helpers::*;
use super::layout::*;
use cubecl::prelude::*;

/// One thread per rigid pixel: stages the air pressure gradient, wind drag and
/// Newtonian gravity force/torque contribution for [`prepare_finish_kernel`].
///
/// Only launched when `air_mode != Off` or `newton_grav != 0`.
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
        let pb = i * P_STRIDE;
        let si = i * 3usize;
        stage[si] = 0.0f32;
        stage[si + 1usize] = 0.0f32;
        stage[si + 2usize] = 0.0f32;
        let bb = pixels[pb + P_BODY] as usize * B_STRIDE;
        let cell_count = xcells * ycells;
        let wants_air = air_mode != 3u32;
        let wants_grav = newton_grav != 0.0f32;
        if (wants_air || wants_grav)
            && cell_count > 0u32
            && pixels[pb + P_DETACHED] == 0u32
            && (bodies[bb + B_FLAGS] & 1u32) != 0u32
        {
            let ty = pixels[pb + P_ELEM];
            let origin_x = fword(bodies, bb + B_ORIGIN_X);
            let origin_y = fword(bodies, bb + B_ORIGIN_Y);
            let cmx = fword(bodies, bb + B_CM_X);
            let cmy = fword(bodies, bb + B_CM_Y);
            let cos_a = fword(bodies, bb + B_COS_A);
            let sin_a = fword(bodies, bb + B_SIN_A);
            let bvx = fword(bodies, bb + B_VX);
            let bvy = fword(bodies, bb + B_VY);
            let omega = fword(bodies, bb + B_OMEGA);
            let lx = fword(pixels, pb + P_LOCAL_X) - origin_x;
            let ly = fword(pixels, pb + P_LOCAL_Y) - origin_y;
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
                    let mut fx = zero_of(wx);
                    let mut fy = zero_of(wy);
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

/// One workgroup of `WG_SIZE` threads per body: rebuilds cached mass /
/// centroid / inertia / radius and compacts `surface_list` when `B_GEOM_DIRTY`
/// is set (upload or detachment).
#[cube(launch_unchecked)]
pub(super) fn prepare_geometry_kernel(
    bodies: &mut [u32],
    pixels: &mut [u32],
    surface: &[u32],
    surface_list: &mut [u32],
    elem_props: &[u32],
    part_flags: &mut [u32],
    owner: &mut [Atomic<u32>],
    body_count: u32,
) {
    let bi = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let bb = bi * B_STRIDE;

    let mut sm_run = Shared::<u32>::new();
    let mut sm_count = Shared::<u32>::new();
    let mut sm_alive = Shared::<u32>::new();
    if tid == 0usize {
        let mut run = seed_u32(body_count, 0u32);
        let mut cnt = seed_u32(body_count, 0u32);
        if bi < body_count as usize
            && (bodies[bb + B_FLAGS] & 1u32) != 0u32
            && bodies[bb + B_GEOM_DIRTY] != 0u32
        {
            run = 1u32;
            cnt = bodies[bb + B_PIXEL_COUNT];
        }
        *sm_run.inner_mut() = run;
        *sm_count.inner_mut() = cnt;
    }
    let run = workgroup_uniform_load(sm_run.inner_ref());
    let count = workgroup_uniform_load(sm_count.inner_ref());
    if run == 0u32 {
        terminate!();
    }

    let start = bodies[bb + B_PIXEL_START] as usize;
    let mut my_mass = zero_of(body_count as f32);
    let mut my_mx = zero_of(body_count as f32);
    let mut my_my = zero_of(body_count as f32);
    let mut my_alive = seed_u32(body_count, 0u32);

    let mut k = tid;
    while k < count as usize {
        let pb = (start + k) * P_STRIDE;
        if pixels[pb + P_DETACHED] == 0u32 {
            let ty = pixels[pb + P_ELEM];
            let w = weight_of(elem_props, ty);
            let lx = fword_m(pixels, pb + P_LOCAL_X);
            let ly = fword_m(pixels, pb + P_LOCAL_Y);
            my_mass += w;
            my_mx += w * lx;
            my_my += w * ly;
            my_alive += 1u32;
        }
        k += WG_SIZE as usize;
    }

    let mut sm_f = Shared::<[f32]>::new_slice(192usize);
    let mut sm_u = Shared::<[u32]>::new_slice(64usize);
    sm_f[tid] = my_mass;
    sm_f[64usize + tid] = my_mx;
    sm_f[128usize + tid] = my_my;
    sm_u[tid] = my_alive;
    sync_cube();

    let mut s = 32usize;
    while s > 0usize {
        if tid < s {
            sm_f[tid] += sm_f[tid + s];
            sm_f[64usize + tid] += sm_f[64usize + tid + s];
            sm_f[128usize + tid] += sm_f[128usize + tid + s];
            sm_u[tid] += sm_u[tid + s];
        }
        sync_cube();
        s /= 2usize;
    }

    if tid == 0usize {
        *sm_alive.inner_mut() = sm_u[0];
    }
    let alive = workgroup_uniform_load(sm_alive.inner_ref());
    if alive < 2u32 {
        if tid == 0usize {
            let flags = bodies[bb + B_FLAGS];
            bodies[bb + B_FLAGS] = flags & 4294967294u32;
            bodies[bb + B_ALIVE] = 0u32;
            bodies[bb + B_SURFACE_COUNT] = 0u32;
            bodies[bb + B_GEOM_DIRTY] = 0u32;
        }
        let mut rk = tid;
        while rk < count as usize {
            let pb = (start + rk) * P_STRIDE;
            let pid = pixels[pb + P_PID];
            if pixels[pb + P_DETACHED] == 0u32 && pid != NO_PID {
                let slot = pid as usize;
                part_flags[slot] &= 4294967277u32;
                owner[slot].store(0u32);
            }
            pixels[pb + P_DETACHED] = 1u32;
            rk += WG_SIZE as usize;
        }
        terminate!();
    }

    let mut mass = sm_f[0];
    if mass <= 0.0f32 {
        mass = 1.0f32;
    }
    if tid == 0usize {
        let mut cmx = fword_m(bodies, bb + B_CM_X);
        let mut cmy = fword_m(bodies, bb + B_CM_Y);
        let mut origin_x = fword_m(bodies, bb + B_ORIGIN_X);
        let mut origin_y = fword_m(bodies, bb + B_ORIGIN_Y);
        let off_x = sm_f[64usize] / mass - origin_x;
        let off_y = sm_f[128usize] / mass - origin_y;
        let cos_a = fword_m(bodies, bb + B_COS_A);
        let sin_a = fword_m(bodies, bb + B_SIN_A);
        cmx += cos_a * off_x - sin_a * off_y;
        cmy += sin_a * off_x + cos_a * off_y;
        origin_x += off_x;
        origin_y += off_y;
        bodies[bb + B_CM_X] = u32::reinterpret(cmx);
        bodies[bb + B_CM_Y] = u32::reinterpret(cmy);
        bodies[bb + B_ORIGIN_X] = u32::reinterpret(origin_x);
        bodies[bb + B_ORIGIN_Y] = u32::reinterpret(origin_y);
        sm_f[128usize] = origin_x;
        sm_f[129usize] = origin_y;
    }
    sync_cube();

    let origin_x = sm_f[128usize];
    let origin_y = sm_f[129usize];
    let mut my_inertia = zero_of(origin_x);
    let mut my_max_r2 = zero_of(origin_x);
    let mut ik = tid;
    while ik < count as usize {
        let pb = (start + ik) * P_STRIDE;
        if pixels[pb + P_DETACHED] == 0u32 {
            let ty = pixels[pb + P_ELEM];
            let w = weight_of(elem_props, ty);
            let lx = fword_m(pixels, pb + P_LOCAL_X) - origin_x;
            let ly = fword_m(pixels, pb + P_LOCAL_Y) - origin_y;
            let r2 = lx * lx + ly * ly;
            my_inertia += w * (r2 + 0.166_666_67f32);
            if r2 > my_max_r2 {
                my_max_r2 = r2;
            }
        }
        ik += WG_SIZE as usize;
    }
    sm_f[tid] = my_inertia;
    sm_f[64usize + tid] = my_max_r2;
    sync_cube();

    let mut s2 = 32usize;
    while s2 > 0usize {
        if tid < s2 {
            sm_f[tid] += sm_f[tid + s2];
            sm_f[64usize + tid] = maxf(sm_f[64usize + tid], sm_f[64usize + tid + s2]);
        }
        sync_cube();
        s2 /= 2usize;
    }

    if tid == 0usize {
        let mut inertia = sm_f[0];
        let max_r2 = sm_f[64usize];
        if inertia <= 1e-4f32 {
            inertia = mass * maxf(max_r2, 1.0f32);
        }
        let inv_mass = 1.0f32 / mass;
        let inv_inertia = minf(1.0f32 / inertia, 0.5f32);
        bodies[bb + B_MASS] = u32::reinterpret(mass);
        bodies[bb + B_INV_MASS] = u32::reinterpret(inv_mass);
        bodies[bb + B_INERTIA] = u32::reinterpret(inertia);
        bodies[bb + B_INV_INERTIA] = u32::reinterpret(inv_inertia);
        bodies[bb + B_UNIT_WEIGHT] = u32::reinterpret(maxf(mass / alive as f32, 1.0f32));
        bodies[bb + B_ALIVE] = alive;
        bodies[bb + B_RADIUS] = u32::reinterpret(max_r2.sqrt());
        bodies[bb + B_GEOM_DIRTY] = 0u32;
    }

    // Compact active surface pixels into `surface_list[start .. start + surf_n]`.
    let chunks = count.div_ceil(WG_SIZE);
    let mut surf_n = seed_u32(count, 0u32);
    for c in 0u32..chunks {
        let ck = c as usize * WG_SIZE as usize + tid;
        let mut keep = seed_u32(count, 0u32);
        if ck < count as usize {
            let gi = start + ck;
            let pb = gi * P_STRIDE;
            if pixels[pb + P_DETACHED] == 0u32
                && pixels[pb + P_PID] != NO_PID
                && ((surface[gi / 32usize] >> (gi % 32usize) as u32) & 1u32) != 0u32
            {
                keep = 1u32;
            }
        }
        sm_u[tid] = keep;
        sync_cube();
        if tid == 0usize {
            let base = c as usize * WG_SIZE as usize;
            for j in 0usize..WG_SIZE as usize {
                if sm_u[j] != 0u32 {
                    surface_list[start + surf_n as usize] = (start + base + j) as u32;
                    surf_n += 1u32;
                }
            }
        }
        sync_cube();
    }
    if tid == 0usize {
        bodies[bb + B_SURFACE_COUNT] = surf_n;
    }
}

/// One workgroup of `WG_SIZE` threads per body: sums staged forces in parallel,
/// computes gravity, accelerations, sub-step count, and integrates sub-step 0.
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
    needs_forces: u32,
) {
    let bi = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let bb = bi * B_STRIDE;

    let mut sm_run = Shared::<u32>::new();
    if tid == 0usize {
        let mut run = seed_u32(body_count, 0u32);
        if bi < body_count as usize && (bodies[bb + B_FLAGS] & 1u32) != 0u32 {
            run = 1u32;
        }
        *sm_run.inner_mut() = run;
    }
    let run = workgroup_uniform_load(sm_run.inner_ref());
    if run == 0u32 {
        terminate!();
    }

    let start = bodies[bb + B_PIXEL_START] as usize;
    let count = bodies[bb + B_PIXEL_COUNT] as usize;
    let mut sm_f = Shared::<[f32]>::new_slice(192usize);

    let mut my_fx = zero_of(frames);
    let mut my_fy = zero_of(frames);
    let mut my_trq = zero_of(frames);
    if needs_forces != 0u32 {
        let mut k = tid;
        while k < count {
            let si = (start + k) * 3usize;
            my_fx += stage[si];
            my_fy += stage[si + 1usize];
            my_trq += stage[si + 2usize];
            k += WG_SIZE as usize;
        }
    }
    sm_f[tid] = my_fx;
    sm_f[64usize + tid] = my_fy;
    sm_f[128usize + tid] = my_trq;
    sync_cube();

    if needs_forces != 0u32 {
        let mut s = 32usize;
        while s > 0usize {
            if tid < s {
                sm_f[tid] += sm_f[tid + s];
                sm_f[64usize + tid] += sm_f[64usize + tid + s];
                sm_f[128usize + tid] += sm_f[128usize + tid + s];
            }
            sync_cube();
            s /= 2usize;
        }
    }

    if tid == 0usize {
        let force_x = sm_f[0];
        let force_y = sm_f[64usize];
        let torque = sm_f[128usize];

        let inv_mass = fword_m(bodies, bb + B_INV_MASS);
        let inv_inertia = fword_m(bodies, bb + B_INV_INERTIA);
        let cmx = fword_m(bodies, bb + B_CM_X);
        let cmy = fword_m(bodies, bb + B_CM_Y);
        let angle = fword_m(bodies, bb + B_ANGLE);

        let mut grav_x = zero_of(frames);
        let mut grav_y = zero_of(frames);
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

        let ax = clampf(grav_x + force_x * inv_mass, -4.0f32, 4.0f32);
        let ay = clampf(grav_y + force_y * inv_mass, -4.0f32, 4.0f32);
        let angular_acc = clampf(torque * inv_inertia, -0.25f32, 0.25f32);
        let radius = fword_m(bodies, bb + B_RADIUS);

        let budget = 16.0f32 / frames;
        let raw_vx = fword_m(bodies, bb + B_VX);
        let raw_vy = fword_m(bodies, bb + B_VY);
        let raw_omega = fword_m(bodies, bb + B_OMEGA);
        let limited = clamp_surface_velocity(raw_vx, raw_vy, raw_omega, radius, budget);

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
        }

        bodies[bb + B_AX] = u32::reinterpret(ax);
        bodies[bb + B_AY] = u32::reinterpret(ay);
        bodies[bb + B_ANGULAR_ACC] = u32::reinterpret(angular_acc);
        bodies[bb + B_SUBSTEPS] = substeps;
        bodies[bb + B_REACT_VX] = 0u32;
        bodies[bb + B_REACT_VY] = 0u32;
        bodies[bb + B_REACT_VW] = 0u32;
        bodies[bb + B_SUPPORT_NEXT] = SUPPORT_NONE;

        // Save frame-start pose and integrate sub-step 0 directly.
        bodies[bb + B_PREV_CM_X] = bodies[bb + B_CM_X];
        bodies[bb + B_PREV_CM_Y] = bodies[bb + B_CM_Y];
        bodies[bb + B_PREV_ANGLE] = bodies[bb + B_ANGLE];
        bodies[bb + B_PREV_COS_A] = bodies[bb + B_COS_A];
        bodies[bb + B_PREV_SIN_A] = bodies[bb + B_SIN_A];

        let step0 = integrate_substep_step(
            cmx,
            cmy,
            angle,
            limited.0,
            limited.1,
            limited.2,
            ax,
            ay,
            angular_acc,
            radius,
            substeps,
            frames,
        );
        bodies[bb + B_CM_X] = u32::reinterpret(step0.0);
        bodies[bb + B_CM_Y] = u32::reinterpret(step0.1);
        bodies[bb + B_ANGLE] = u32::reinterpret(step0.2);
        bodies[bb + B_VX] = u32::reinterpret(step0.3);
        bodies[bb + B_VY] = u32::reinterpret(step0.4);
        bodies[bb + B_OMEGA] = u32::reinterpret(step0.5);
        bodies[bb + B_COS_A] = u32::reinterpret(step0.6);
        bodies[bb + B_SIN_A] = u32::reinterpret(step0.7);
    }
}

/// Single workgroup (`WG_SIZE` threads): computes per-body bounding-circle
/// broadphase (`B_HAS_PAIR`), `global_max_substeps`, `any_pair`, and writes the
/// indirect `dispatch_counts` slots for the sub-step loop.
#[cube(launch_unchecked)]
pub(super) fn broadphase_dispatch_kernel(
    bodies: &mut [u32],
    dispatch_counts: &mut [u32],
    body_count: u32,
    pixel_cubes: u32,
    body_cubes: u32,
    cell_cubes: u32,
) {
    let tid = UNIT_POS_X as usize;
    let mut my_max_sub = seed_u32(body_count, 0u32);
    let mut my_any_pair = seed_u32(body_count, 0u32);

    let mut bi = tid;
    while bi < body_count as usize {
        let bb = bi * B_STRIDE;
        if (bodies[bb + B_FLAGS] & 1u32) != 0u32 {
            let sub_i = bodies[bb + B_SUBSTEPS];
            if sub_i > my_max_sub {
                my_max_sub = sub_i;
            }
            let cmx_i = fword_m(bodies, bb + B_CM_X);
            let cmy_i = fword_m(bodies, bb + B_CM_Y);
            let rad_i = fword_m(bodies, bb + B_RADIUS);
            let mut has_pair = seed_u32(body_count, 0u32);
            for bj in 0usize..body_count as usize {
                if bj != bi {
                    let obb = bj * B_STRIDE;
                    if (bodies[obb + B_FLAGS] & 1u32) != 0u32 {
                        let sub_j = bodies[obb + B_SUBSTEPS];
                        let reach = rad_i
                            + fword_m(bodies, obb + B_RADIUS)
                            + (sub_i + sub_j) as f32
                            + 4.0f32;
                        let dx = cmx_i - fword_m(bodies, obb + B_CM_X);
                        let dy = cmy_i - fword_m(bodies, obb + B_CM_Y);
                        if dx * dx + dy * dy <= reach * reach {
                            has_pair = 1u32;
                            break;
                        }
                    }
                }
            }
            bodies[bb + B_HAS_PAIR] = has_pair;
            if has_pair != 0u32 {
                my_any_pair = 1u32;
            }
        } else {
            bodies[bb + B_HAS_PAIR] = 0u32;
        }
        bi += WG_SIZE as usize;
    }

    let mut sm_u = Shared::<[u32]>::new_slice(128usize);
    sm_u[tid] = my_max_sub;
    sm_u[64usize + tid] = my_any_pair;
    sync_cube();

    let mut s = 32usize;
    while s > 0usize {
        if tid < s {
            if sm_u[tid + s] > sm_u[tid] {
                sm_u[tid] = sm_u[tid + s];
            }
            if sm_u[64usize + tid + s] > sm_u[64usize + tid] {
                sm_u[64usize + tid] = sm_u[64usize + tid + s];
            }
        }
        sync_cube();
        s /= 2usize;
    }

    let global_max_sub = sm_u[0];
    let any_pair = sm_u[64usize];

    if tid < 16usize {
        let sub = tid as u32;
        let active_pair = sub < global_max_sub && any_pair != 0u32;

        let sc_base = tid * DISPATCH_SLOT_WORDS;
        let bd_base = (16usize + tid) * DISPATCH_SLOT_WORDS;
        let ap_base = (32usize + tid) * DISPATCH_SLOT_WORDS;

        let mut sc_x = seed_u32(body_count, 0u32);
        let mut bd_x = seed_u32(body_count, 0u32);
        let mut ap_x = seed_u32(body_count, 0u32);
        if active_pair {
            sc_x = pixel_cubes;
            bd_x = body_count;
            ap_x = body_cubes;
        }
        dispatch_counts[sc_base] = sc_x;
        dispatch_counts[sc_base + 1usize] = 1u32;
        dispatch_counts[sc_base + 2usize] = 1u32;
        dispatch_counts[sc_base + 3usize] = 0u32;

        dispatch_counts[bd_base] = bd_x;
        dispatch_counts[bd_base + 1usize] = 1u32;
        dispatch_counts[bd_base + 2usize] = 1u32;
        dispatch_counts[bd_base + 3usize] = 0u32;

        dispatch_counts[ap_base] = ap_x;
        dispatch_counts[ap_base + 1usize] = 1u32;
        dispatch_counts[ap_base + 2usize] = 1u32;
        dispatch_counts[ap_base + 3usize] = 0u32;
    } else if tid == 16usize {
        let cl_base = DISPATCH_CLEAR_LIVE_SLOT * DISPATCH_SLOT_WORDS;
        let mut cl_x = seed_u32(body_count, 0u32);
        if any_pair != 0u32 {
            cl_x = cell_cubes;
        }
        dispatch_counts[cl_base] = cl_x;
        dispatch_counts[cl_base + 1usize] = 1u32;
        dispatch_counts[cl_base + 2usize] = 1u32;
        dispatch_counts[cl_base + 3usize] = 0u32;
    } else if tid == 17usize {
        let sa_base = DISPATCH_STATIC_ALL_SLOT * DISPATCH_SLOT_WORDS;
        let mut sa_x = seed_u32(body_count, 0u32);
        if any_pair == 0u32 {
            sa_x = body_count;
        }
        dispatch_counts[sa_base] = sa_x;
        dispatch_counts[sa_base + 1usize] = 1u32;
        dispatch_counts[sa_base + 2usize] = 1u32;
        dispatch_counts[sa_base + 3usize] = 0u32;
    }
}

/// Applies the reactions queued by every contact thread between pair
/// iterations 0 and 1.
#[cube(launch_unchecked)]
pub(super) fn apply_body_impulses_kernel(bodies: &mut [Atomic<u32>], body_count: u32) {
    let bi = ABSOLUTE_POS;
    if bi < body_count as usize {
        let bb = bi * B_STRIDE;
        let fx = i32::reinterpret(bodies[bb + B_REACT_VX].load());
        let fy = i32::reinterpret(bodies[bb + B_REACT_VY].load());
        let fw = i32::reinterpret(bodies[bb + B_REACT_VW].load());
        bodies[bb + B_REACT_VX].store(0u32);
        bodies[bb + B_REACT_VY].store(0u32);
        bodies[bb + B_REACT_VW].store(0u32);
        let flags = bodies[bb + B_FLAGS].load();
        if (flags & 1u32) != 0u32 && (fx != 0i32 || fy != 0i32 || fw != 0i32) {
            let dvx = fx as f32 * REACTION_INV_SCALE;
            let dvy = fy as f32 * REACTION_INV_SCALE;
            let dw = fw as f32 * REACTION_INV_SCALE;
            let vx = fword_a(bodies, bb + B_VX) + dvx;
            let vy = fword_a(bodies, bb + B_VY) + dvy;
            let omega = fword_a(bodies, bb + B_OMEGA) + dw;
            let clamped = clamp_velocity(vx, vy, omega);
            bodies[bb + B_VX].store(u32::reinterpret(clamped.0));
            bodies[bb + B_VY].store(u32::reinterpret(clamped.1));
            bodies[bb + B_OMEGA].store(u32::reinterpret(clamped.2));
        }
    }
}
