//! Workgroup-parallel rigid-body contact and sub-step kernels.

#![allow(clippy::too_many_arguments, clippy::manual_clamp, clippy::collapsible_if)]

use super::helpers::*;
use super::layout::*;
use cubecl::prelude::*;

/// Adds one contact sample into the workgroup's shared manifold table.
///
/// Only called by thread 0 in fixed ascending pixel order so manifold slot
/// allocation and accumulation stay deterministic.
#[cube]
fn manifold_add_shared(
    manifolds: &mut [u32],
    dir: u32,
    other: u32,
    rxa: f32,
    rya: f32,
    obx: f32,
    oby: f32,
) {
    let mut matched = seed_u32(dir, 0u32);
    for s in 0usize..MANIFOLDS {
        let sb = s * M_STRIDE;
        if matched == 0u32 && manifolds[sb + M_DIR] == dir && manifolds[sb + M_BODY] == other {
            manifolds[sb + M_COUNT] += 1u32;
            manifolds[sb + M_ARM_AX] =
                u32::reinterpret(f32::reinterpret(manifolds[sb + M_ARM_AX]) + rxa);
            manifolds[sb + M_ARM_AY] =
                u32::reinterpret(f32::reinterpret(manifolds[sb + M_ARM_AY]) + rya);
            manifolds[sb + M_OB_X] =
                u32::reinterpret(f32::reinterpret(manifolds[sb + M_OB_X]) + obx);
            manifolds[sb + M_OB_Y] =
                u32::reinterpret(f32::reinterpret(manifolds[sb + M_OB_Y]) + oby);
            matched = 1u32;
        }
    }
    if matched == 0u32 {
        let mut created = seed_u32(dir, 0u32);
        for s in 0usize..MANIFOLDS {
            let sb = s * M_STRIDE;
            if created == 0u32 && manifolds[sb + M_DIR] == M_FREE {
                manifolds[sb + M_DIR] = dir;
                manifolds[sb + M_BODY] = other;
                manifolds[sb + M_COUNT] = 1u32;
                manifolds[sb + M_ARM_AX] = u32::reinterpret(rxa);
                manifolds[sb + M_ARM_AY] = u32::reinterpret(rya);
                manifolds[sb + M_OB_X] = u32::reinterpret(obx);
                manifolds[sb + M_OB_Y] = u32::reinterpret(oby);
                created = 1u32;
            }
        }
    }
}

/// Pair contact phase (`mode = 0`), one `WG_SIZE`-thread workgroup per body.
///
/// Surface pixels are evaluated in parallel over `surface_list`; any hits in
/// each 64-pixel chunk are merged in pixel order into workgroup-shared
/// manifolds, resolved once per manifold, and queued as fixed-point velocity
/// deltas on both bodies.
#[cube(launch_unchecked)]
pub(super) fn contact_pair_kernel(
    bodies: &mut [Atomic<u32>],
    pixels: &[u32],
    surface_list: &[u32],
    pmap: &[u32],
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    owner: &mut [Atomic<u32>],
    live: &mut [Atomic<u32>],
    body_count: u32,
    substep: u32,
    xres: u32,
    yres: u32,
    iter: u32,
    generation: u32,
) {
    let bi = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let bb = bi * B_STRIDE;

    let mut sm_run = Shared::<u32>::new();
    let mut sm_sc = Shared::<u32>::new();
    let mut sm_any = Shared::<u32>::new();
    let mut sm_body = Shared::<[u32]>::new_slice(16usize);
    if tid == 0usize {
        let mut run = seed_u32(body_count, 0u32);
        let mut sc = seed_u32(body_count, 0u32);
        if bi < body_count as usize {
            let flags = bodies[bb + B_FLAGS].load();
            let substeps = bodies[bb + B_SUBSTEPS].load();
            let has_pair = bodies[bb + B_HAS_PAIR].load();
            let surf_cnt = bodies[bb + B_SURFACE_COUNT].load();
            if (flags & 1u32) != 0u32 && substep < substeps && has_pair != 0u32 && surf_cnt > 0u32 {
                run = 1u32;
                sc = surf_cnt;
                sm_body[0] = bodies[bb + B_PIXEL_START].load();
                sm_body[1] = bodies[bb + B_INV_MASS].load();
                sm_body[2] = bodies[bb + B_INV_INERTIA].load();
                sm_body[3] = bodies[bb + B_ORIGIN_X].load();
                sm_body[4] = bodies[bb + B_ORIGIN_Y].load();
                sm_body[5] = bodies[bb + B_CM_X].load();
                sm_body[6] = bodies[bb + B_CM_Y].load();
                sm_body[7] = bodies[bb + B_PREV_CM_X].load();
                sm_body[8] = bodies[bb + B_PREV_CM_Y].load();
                sm_body[9] = bodies[bb + B_COS_A].load();
                sm_body[10] = bodies[bb + B_SIN_A].load();
                sm_body[11] = bodies[bb + B_PREV_COS_A].load();
                sm_body[12] = bodies[bb + B_PREV_SIN_A].load();
                sm_body[13] = bodies[bb + B_VX].load();
                sm_body[14] = bodies[bb + B_VY].load();
                sm_body[15] = bodies[bb + B_OMEGA].load();
                if iter == 0u32 {
                    bodies[bb + B_SUPPORT_NEXT].store(SUPPORT_NONE);
                }
            }
        }
        *sm_run.inner_mut() = run;
        *sm_sc.inner_mut() = sc;
        *sm_any.inner_mut() = 0u32;
    }
    let run = workgroup_uniform_load(sm_run.inner_ref());
    let surf_count = workgroup_uniform_load(sm_sc.inner_ref());
    if run == 0u32 {
        terminate!();
    }

    let mut sm_man = Shared::<[u32]>::new_slice(64usize);
    if tid % M_STRIDE == M_DIR {
        sm_man[tid] = M_FREE;
    } else {
        sm_man[tid] = 0u32;
    }
    sync_cube();

    let start = sm_body[0] as usize;
    let inv_mass = f32::reinterpret(sm_body[1]);
    let inv_inertia = f32::reinterpret(sm_body[2]);
    let origin_x = f32::reinterpret(sm_body[3]);
    let origin_y = f32::reinterpret(sm_body[4]);
    let cmx = f32::reinterpret(sm_body[5]);
    let cmy = f32::reinterpret(sm_body[6]);
    let prev_cmx = f32::reinterpret(sm_body[7]);
    let prev_cmy = f32::reinterpret(sm_body[8]);
    let cos_a = f32::reinterpret(sm_body[9]);
    let sin_a = f32::reinterpret(sm_body[10]);
    let prev_cos = f32::reinterpret(sm_body[11]);
    let prev_sin = f32::reinterpret(sm_body[12]);
    let vx = f32::reinterpret(sm_body[13]);
    let vy = f32::reinterpret(sm_body[14]);
    let omega = f32::reinterpret(sm_body[15]);

    let slop = 0.02f32;
    let mut corr_px = zero_of(cmx);
    let mut corr_nx = zero_of(cmx);
    let mut corr_py = zero_of(cmx);
    let mut corr_ny = zero_of(cmx);

    let mut sm_hit_u = Shared::<[u32]>::new_slice(128usize);
    let mut sm_hit_f = Shared::<[f32]>::new_slice(256usize);

    let chunks = surf_count.div_ceil(WG_SIZE);
    for chunk in 0u32..chunks {
        let mut h_dir = seed_u32(surf_count, M_FREE);
        let mut h_other = seed_u32(surf_count, 0u32);
        let mut h_rx = zero_of(cmx);
        let mut h_ry = zero_of(cmx);
        let mut h_ox = zero_of(cmx);
        let mut h_oy = zero_of(cmx);

        let k = chunk as usize * WG_SIZE as usize + tid;
        if k < surf_count as usize {
            let gi = surface_list[start + k] as usize;
            let pb = gi * P_STRIDE;
            let lx = fword(pixels, pb + P_LOCAL_X) - origin_x;
            let ly = fword(pixels, pb + P_LOCAL_Y) - origin_y;
            let rx = lx * cos_a - ly * sin_a;
            let ry = lx * sin_a + ly * cos_a;
            let wx = cmx + rx;
            let wy = cmy + ry;
            let prx = lx * prev_cos - ly * prev_sin;
            let pry = lx * prev_sin + ly * prev_cos;
            let prev_wx = prev_cmx + prx;
            let prev_wy = prev_cmy + pry;

            let gx = wx.round() as i32;
            let gy = wy.round() as i32;
            if gx >= 0i32 && gy >= 0i32 && gx < xres as i32 && gy < yres as i32 {
                let idx = (gy as u32 * xres + gx as u32) as usize;
                let mut paired = seed_u32(surf_count, 0u32);
                let pm = pmap[idx];
                if pm != 0u32 {
                    let other = unpack_slot(pm);
                    if other != 4294967295u32 {
                        let oslot = other as usize;
                        let own = owner[oslot].load();
                        if own != 0u32 && own != bi as u32 + 1u32 && part_ty[oslot] != 0u32 {
                            paired = 1u32;
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
                                h_dir = face_dir(face.0, face.1);
                                h_other = own;
                                h_rx = rx;
                                h_ry = ry;
                                h_ox = ox;
                                h_oy = oy;
                                if face.0 > 0.0f32 {
                                    corr_px = maxf(corr_px, face.2 + slop);
                                } else if face.0 < 0.0f32 {
                                    corr_nx = maxf(corr_nx, face.2 + slop);
                                } else if face.1 > 0.0f32 {
                                    corr_py = maxf(corr_py, face.2 + slop);
                                } else {
                                    corr_ny = maxf(corr_ny, face.2 + slop);
                                }
                            }
                        }
                    }
                }
                if paired == 0u32 {
                    let word = live[idx].load();
                    let slot = word & 8388607u32;
                    if (word >> 23u32) == generation && slot != 0u32 {
                        let oslot = (slot - 1u32) as usize;
                        let recv = owner[oslot].load();
                        if recv != 0u32 && recv != bi as u32 + 1u32 && part_ty[oslot] != 0u32 {
                            let ox = gx as f32;
                            let oy = gy as f32;
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
                                h_dir = face_dir(face.0, face.1);
                                h_other = recv;
                                h_rx = rx;
                                h_ry = ry;
                                h_ox = ox;
                                h_oy = oy;
                            }
                        }
                    }
                }
            }
        }

        sm_hit_u[tid] = h_dir;
        sm_hit_u[64usize + tid] = h_other;
        sm_hit_f[tid] = h_rx;
        sm_hit_f[64usize + tid] = h_ry;
        sm_hit_f[128usize + tid] = h_ox;
        sm_hit_f[192usize + tid] = h_oy;
        if h_dir != M_FREE {
            *sm_any.inner_mut() = 1u32;
        }
        sync_cube();

        if tid == 0usize && *sm_any.inner_ref() != 0u32 {
            for j in 0usize..64usize {
                if sm_hit_u[j] != M_FREE {
                    manifold_add_shared(
                        sm_man.as_mut_slice(),
                        sm_hit_u[j],
                        sm_hit_u[64usize + j],
                        sm_hit_f[j],
                        sm_hit_f[64usize + j],
                        sm_hit_f[128usize + j],
                        sm_hit_f[192usize + j],
                    );
                }
            }
            *sm_any.inner_mut() = 0u32;
        }
        sync_cube();
    }

    // Reduce deepest positional corrections across the workgroup.
    sm_hit_f[tid] = corr_px;
    sm_hit_f[64usize + tid] = corr_nx;
    sm_hit_f[128usize + tid] = corr_py;
    sm_hit_f[192usize + tid] = corr_ny;
    sync_cube();

    let mut rs = 32usize;
    while rs > 0usize {
        if tid < rs {
            sm_hit_f[tid] = maxf(sm_hit_f[tid], sm_hit_f[tid + rs]);
            sm_hit_f[64usize + tid] = maxf(sm_hit_f[64usize + tid], sm_hit_f[64usize + tid + rs]);
            sm_hit_f[128usize + tid] =
                maxf(sm_hit_f[128usize + tid], sm_hit_f[128usize + tid + rs]);
            sm_hit_f[192usize + tid] =
                maxf(sm_hit_f[192usize + tid], sm_hit_f[192usize + tid + rs]);
        }
        sync_cube();
        rs /= 2usize;
    }

    if tid == 0usize {
        for s in 0usize..MANIFOLDS {
            let sb = s * M_STRIDE;
            let dir = sm_man[sb + M_DIR];
            if dir != M_FREE {
                let samples = sm_man[sb + M_COUNT];
                if samples > 0u32 {
                    let inv_samples = 1.0f32 / samples as f32;
                    let normal = dir_normal(dir);
                    let rxa = f32::reinterpret(sm_man[sb + M_ARM_AX]) * inv_samples;
                    let rya = f32::reinterpret(sm_man[sb + M_ARM_AY]) * inv_samples;
                    let obx = f32::reinterpret(sm_man[sb + M_OB_X]) * inv_samples;
                    let oby = f32::reinterpret(sm_man[sb + M_OB_Y]) * inv_samples;
                    let other = sm_man[sb + M_BODY];
                    if other != 0u32 {
                        let obb = (other as usize - 1usize) * B_STRIDE;
                        let mut inv_mass_b = fword_a(bodies, obb + B_INV_MASS);
                        let mut inv_inertia_b = fword_a(bodies, obb + B_INV_INERTIA);
                        let rbx = obx - fword_a(bodies, obb + B_CM_X);
                        let rby = oby - fword_a(bodies, obb + B_CM_Y);
                        let omega_b = fword_a(bodies, obb + B_OMEGA);
                        let mut ovx = fword_a(bodies, obb + B_VX) - omega_b * rby;
                        let mut ovy = fword_a(bodies, obb + B_VY) + omega_b * rbx;
                        let support = support_normal(bodies[obb + B_SUPPORT].load());
                        if support.0 * normal.0 + support.1 * normal.1 > 0.5f32 {
                            inv_mass_b = 0.0f32;
                            inv_inertia_b = 0.0f32;
                            ovx = 0.0f32;
                            ovy = 0.0f32;
                            bodies[bb + B_SUPPORT_NEXT].store(support_code(normal.0, normal.1));
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
                        let dvx = res.3 * inv_mass;
                        let dvy = res.4 * inv_mass;
                        let dw = (rxa * res.4 - rya * res.3) * inv_inertia;
                        bodies[bb + B_REACT_VX]
                            .fetch_add(u32::reinterpret((dvx * REACTION_SCALE) as i32));
                        bodies[bb + B_REACT_VY]
                            .fetch_add(u32::reinterpret((dvy * REACTION_SCALE) as i32));
                        bodies[bb + B_REACT_VW]
                            .fetch_add(u32::reinterpret((dw * REACTION_SCALE) as i32));

                        let rvx = zero_of(res.3) - res.3 * inv_mass_b;
                        let rvy = zero_of(res.4) - res.4 * inv_mass_b;
                        let rw = zero_of(res.3) - (rbx * res.4 - rby * res.3) * inv_inertia_b;
                        bodies[obb + B_REACT_VX]
                            .fetch_add(u32::reinterpret((rvx * REACTION_SCALE) as i32));
                        bodies[obb + B_REACT_VY]
                            .fetch_add(u32::reinterpret((rvy * REACTION_SCALE) as i32));
                        bodies[obb + B_REACT_VW]
                            .fetch_add(u32::reinterpret((rw * REACTION_SCALE) as i32));
                    }
                }
            }
        }

        let final_x = clampf(cmx + sm_hit_f[0] - sm_hit_f[64usize], 0.5f32, xres as f32 - 0.5f32);
        let final_y = clampf(
            cmy + sm_hit_f[128usize] - sm_hit_f[192usize],
            0.5f32,
            yres as f32 - 0.5f32,
        );
        bodies[bb + B_CM_X].store(u32::reinterpret(final_x));
        bodies[bb + B_CM_Y].store(u32::reinterpret(final_y));
    }
}

/// Atomically moves a displaced loose particle from `old` to an adjacent empty
/// cell in `pmap` using a transient sentinel (`1u32`) so concurrent surface
/// threads never lose or duplicate a particle.
#[cube]
fn displace_fluid_atomic(
    pmap: &mut [Atomic<u32>],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    other: u32,
    other_ty: u32,
    oslot: usize,
    wx: f32,
    wy: f32,
    surf_vx: f32,
    surf_vy: f32,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
) {
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
    let mut found = seed_u32(other, 0u32) != 0u32;
    for oy_i in 0i32..3i32 {
        for ox_i in 0i32..3i32 {
            let oxi = ox_i - 1i32;
            let oyi = oy_i - 1i32;
            if oxi != 0i32 || oyi != 0i32 {
                let nx = ogx + oxi;
                let ny = ogy + oyi;
                if nx >= 0i32 && ny >= 0i32 && nx < xres as i32 && ny < yres as i32 {
                    let ni = (ny as u32 * xres + nx as u32) as usize;
                    if pmap[ni].load() == 0u32
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
                        let score = oxi as f32 * dir_x + oyi as f32 * dir_y;
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
    let mut moved = seed_u32(other, 0u32) != 0u32;
    if found && ogx >= 0i32 && ogy >= 0i32 && ogx < xres as i32 && ogy < yres as i32 {
        let old = (ogy as u32 * xres + ogx as u32) as usize;
        let ni = (by as u32 * xres + bx as u32) as usize;
        let packed = pack_map(other, other_ty);
        if pmap[ni].compare_exchange_weak(0u32, 1u32) == 0u32 {
            if pmap[old].compare_exchange_weak(packed, 0u32) == packed {
                part_x[oslot] = bx as f32;
                part_y[oslot] = by as f32;
                part_vx[oslot] = surf_vx + dir_x * 0.5f32;
                part_vy[oslot] = surf_vy + dir_y * 0.5f32;
                pmap[ni].store(packed);
                moved = true;
            } else {
                pmap[ni].store(0u32);
            }
        }
    }
    if !moved {
        part_vx[oslot] += dir_x * 0.5f32;
        part_vy[oslot] += dir_y * 0.5f32;
    }
}

/// Runs one static-world contact + fluid + optional next-substep integration
/// inside a single `wg_size`-thread workgroup owning body `bi`.
///
/// `sm_body` holds 25 words describing the body state; `sm_red` holds
/// `4 * wg_size` words reused across the 5 parallel tree reductions.
#[cube]
fn run_one_static_substep(
    bi: usize,
    tid: usize,
    substep: u32,
    advance_next: u32,
    bodies: &mut [u32],
    pixels: &[u32],
    surface_list: &[u32],
    pmap: &mut [Atomic<u32>],
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
    sm_body: &mut [u32],
    sm_red: &mut [u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    gravity_mode: u32,
    frames: f32,
    wg_size: u32,
) {
    let wg_usize = wg_size as usize;
    let start = sm_body[0] as usize;
    let surf_count = sm_body[1] as usize;
    let substeps = sm_body[2];
    let alive = maxf(sm_body[3] as f32, 1.0f32);
    let inv_pixels = 1.0f32 / alive;
    let unit_weight = f32::reinterpret(sm_body[4]);
    let inv_mass = f32::reinterpret(sm_body[5]);
    let inv_inertia = f32::reinterpret(sm_body[6]);
    let origin_x = f32::reinterpret(sm_body[7]);
    let origin_y = f32::reinterpret(sm_body[8]);
    let radius = f32::reinterpret(sm_body[9]);
    let cmx = f32::reinterpret(sm_body[10]);
    let cmy = f32::reinterpret(sm_body[11]);
    let prev_cmx = f32::reinterpret(sm_body[12]);
    let prev_cmy = f32::reinterpret(sm_body[13]);
    let angle = f32::reinterpret(sm_body[14]);
    let cos_a = f32::reinterpret(sm_body[15]);
    let sin_a = f32::reinterpret(sm_body[16]);
    let prev_cos = f32::reinterpret(sm_body[17]);
    let prev_sin = f32::reinterpret(sm_body[18]);
    let mut vx = f32::reinterpret(sm_body[19]);
    let mut vy = f32::reinterpret(sm_body[20]);
    let mut omega = f32::reinterpret(sm_body[21]);
    let ax = f32::reinterpret(sm_body[22]);
    let ay = f32::reinterpret(sm_body[23]);
    let angular_acc = f32::reinterpret(sm_body[24]);

    let sdt = frames / substeps as f32;
    let support_powder = unit_weight * radius < 1500.0f32;
    let slop = 0.02f32;

    let mut corr_px = zero_of(cmx);
    let mut corr_nx = zero_of(cmx);
    let mut corr_py = zero_of(cmx);
    let mut corr_ny = zero_of(cmx);

    let mut cnt0 = seed_u32(substeps, 0u32);
    let mut cnt1 = seed_u32(substeps, 0u32);
    let mut cnt2 = seed_u32(substeps, 0u32);
    let mut cnt3 = seed_u32(substeps, 0u32);
    let mut ax0 = zero_of(cmx);
    let mut ay0 = zero_of(cmx);
    let mut ax1 = zero_of(cmx);
    let mut ay1 = zero_of(cmx);
    let mut ax2 = zero_of(cmx);
    let mut ay2 = zero_of(cmx);
    let mut ax3 = zero_of(cmx);
    let mut ay3 = zero_of(cmx);

    let mut f_dvx = zero_of(cmx);
    let mut f_dvy = zero_of(cmx);
    let mut f_dw = zero_of(cmx);

    let mut k = tid;
    while k < surf_count {
        let gi = surface_list[start + k] as usize;
        let pb = gi * P_STRIDE;
        let ty = pixels[pb + P_ELEM];
        let props = props_of(elem_props, ty);
        let lx = fword(pixels, pb + P_LOCAL_X) - origin_x;
        let ly = fword(pixels, pb + P_LOCAL_Y) - origin_y;
        let rx = lx * cos_a - ly * sin_a;
        let ry = lx * sin_a + ly * cos_a;
        let wx = cmx + rx;
        let wy = cmy + ry;
        let prx = lx * prev_cos - ly * prev_sin;
        let pry = lx * prev_sin + ly * prev_cos;
        let prev_wx = prev_cmx + prx;
        let prev_wy = prev_cmy + pry;

        if wx < 0.5f32 {
            cnt0 += 1u32;
            ax0 += rx;
            ay0 += ry;
            corr_px = maxf(corr_px, clampf(0.5f32 - wx, 0.0f32, 1.0f32) + slop);
        } else if wx > xres as f32 - 0.5f32 {
            cnt1 += 1u32;
            ax1 += rx;
            ay1 += ry;
            corr_nx = maxf(
                corr_nx,
                clampf(wx - (xres as f32 - 0.5f32), 0.0f32, 1.0f32) + slop,
            );
        }
        if wy < 0.5f32 {
            cnt2 += 1u32;
            ax2 += rx;
            ay2 += ry;
            corr_py = maxf(corr_py, clampf(0.5f32 - wy, 0.0f32, 1.0f32) + slop);
        } else if wy > yres as f32 - 0.5f32 {
            cnt3 += 1u32;
            ax3 += rx;
            ay3 += ry;
            corr_ny = maxf(
                corr_ny,
                clampf(wy - (yres as f32 - 0.5f32), 0.0f32, 1.0f32) + slop,
            );
        }

        let gx = wx.round() as i32;
        let gy = wy.round() as i32;
        if gx >= 0i32 && gy >= 0i32 && gx < xres as i32 && gy < yres as i32 {
            if wall_blocks(
                wall_bmap,
                wall_emap,
                gx,
                gy,
                xcells as i32,
                ycells as i32,
                props,
            ) {
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
                if face.0 > 0.0f32 {
                    cnt0 += 1u32;
                    ax0 += rx;
                    ay0 += ry;
                    corr_px = maxf(corr_px, face.2 + slop);
                } else if face.0 < 0.0f32 {
                    cnt1 += 1u32;
                    ax1 += rx;
                    ay1 += ry;
                    corr_nx = maxf(corr_nx, face.2 + slop);
                } else if face.1 > 0.0f32 {
                    cnt2 += 1u32;
                    ax2 += rx;
                    ay2 += ry;
                    corr_py = maxf(corr_py, face.2 + slop);
                } else if face.1 < 0.0f32 {
                    cnt3 += 1u32;
                    ax3 += rx;
                    ay3 += ry;
                    corr_ny = maxf(corr_ny, face.2 + slop);
                }
            }

            let idx = (gy as u32 * xres + gx as u32) as usize;
            let pm = pmap[idx].load();
            if pm != 0u32 {
                let other = unpack_slot(pm);
                if other != 4294967295u32 {
                    let oslot = other as usize;
                    let own = owner[oslot].load();
                    if own == 0u32 {
                        let other_ty = part_ty[oslot];
                        if other_ty != 0u32 {
                            let flag_only_rigid = (part_flags[oslot] & 16u32) != 0u32;
                            let other_props = props_of(elem_props, other_ty);
                            let powder = (other_props & 1u32) != 0u32;
                            let hard = (other_props & 4u32) != 0u32
                                || (powder && support_powder)
                                || flag_only_rigid;
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
                                if face.0 > 0.0f32 {
                                    cnt0 += 1u32;
                                    ax0 += rx;
                                    ay0 += ry;
                                    corr_px = maxf(corr_px, face.2 + slop);
                                } else if face.0 < 0.0f32 {
                                    cnt1 += 1u32;
                                    ax1 += rx;
                                    ay1 += ry;
                                    corr_nx = maxf(corr_nx, face.2 + slop);
                                } else if face.1 > 0.0f32 {
                                    cnt2 += 1u32;
                                    ax2 += rx;
                                    ay2 += ry;
                                    corr_py = maxf(corr_py, face.2 + slop);
                                } else if face.1 < 0.0f32 {
                                    cnt3 += 1u32;
                                    ax3 += rx;
                                    ay3 += ry;
                                    corr_ny = maxf(corr_ny, face.2 + slop);
                                }
                            } else if (other_props & (2u32 | 1u32 | 8u32)) != 0u32 {
                                if gravity_mode != 1u32 {
                                    let fluid_weight = weight_of(elem_props, other_ty);
                                    let ratio = clampf(
                                        fluid_weight / maxf(unit_weight, 1.0f32),
                                        0.0f32,
                                        2.0f32,
                                    );
                                    f_dvy -= 0.15f32 * ratio * 0.5f32 * sdt * inv_pixels;
                                }
                                let surf_vx = vx - omega * ry;
                                let surf_vy = vy + omega * rx;
                                let rel_x = part_vx[oslot] - surf_vx;
                                let rel_y = part_vy[oslot] - surf_vy;
                                let k2 = 0.08f32 * sdt * inv_pixels;
                                f_dvx += rel_x * k2;
                                f_dvy += rel_y * k2;
                                f_dw += (rx * rel_y - ry * rel_x) * k2 * inv_inertia * 0.25f32;
                                displace_fluid_atomic(
                                    pmap,
                                    part_x,
                                    part_y,
                                    part_vx,
                                    part_vy,
                                    wall_bmap,
                                    wall_emap,
                                    other,
                                    other_ty,
                                    oslot,
                                    wx,
                                    wy,
                                    surf_vx,
                                    surf_vy,
                                    xres,
                                    yres,
                                    xcells,
                                    ycells,
                                );
                            }
                        }
                    }
                }
            }
        }
        k += wg_usize;
    }

    // Stage 1: max positional corrections.
    sm_red[tid] = u32::reinterpret(corr_px);
    sm_red[wg_usize + tid] = u32::reinterpret(corr_nx);
    sm_red[2usize * wg_usize + tid] = u32::reinterpret(corr_py);
    sm_red[3usize * wg_usize + tid] = u32::reinterpret(corr_ny);
    sync_cube();
    let mut s1 = wg_size / 2u32;
    while s1 > 0u32 {
        let step = s1 as usize;
        if tid < step {
            sm_red[tid] = u32::reinterpret(maxf(
                f32::reinterpret(sm_red[tid]),
                f32::reinterpret(sm_red[tid + step]),
            ));
            sm_red[wg_usize + tid] = u32::reinterpret(maxf(
                f32::reinterpret(sm_red[wg_usize + tid]),
                f32::reinterpret(sm_red[wg_usize + tid + step]),
            ));
            sm_red[2usize * wg_usize + tid] = u32::reinterpret(maxf(
                f32::reinterpret(sm_red[2usize * wg_usize + tid]),
                f32::reinterpret(sm_red[2usize * wg_usize + tid + step]),
            ));
            sm_red[3usize * wg_usize + tid] = u32::reinterpret(maxf(
                f32::reinterpret(sm_red[3usize * wg_usize + tid]),
                f32::reinterpret(sm_red[3usize * wg_usize + tid + step]),
            ));
        }
        sync_cube();
        s1 /= 2u32;
    }
    let tot_cpx = f32::reinterpret(sm_red[0]);
    let tot_cnx = f32::reinterpret(sm_red[wg_usize]);
    let tot_cpy = f32::reinterpret(sm_red[2usize * wg_usize]);
    let tot_cny = f32::reinterpret(sm_red[3usize * wg_usize]);
    sync_cube();

    // Stage 2: sample counts per axis direction.
    sm_red[tid] = cnt0;
    sm_red[wg_usize + tid] = cnt1;
    sm_red[2usize * wg_usize + tid] = cnt2;
    sm_red[3usize * wg_usize + tid] = cnt3;
    sync_cube();
    let mut s2 = wg_size / 2u32;
    while s2 > 0u32 {
        let step = s2 as usize;
        if tid < step {
            sm_red[tid] += sm_red[tid + step];
            sm_red[wg_usize + tid] += sm_red[wg_usize + tid + step];
            sm_red[2usize * wg_usize + tid] += sm_red[2usize * wg_usize + tid + step];
            sm_red[3usize * wg_usize + tid] += sm_red[3usize * wg_usize + tid + step];
        }
        sync_cube();
        s2 /= 2u32;
    }
    let tot_cnt0 = sm_red[0];
    let tot_cnt1 = sm_red[wg_usize];
    let tot_cnt2 = sm_red[2usize * wg_usize];
    let tot_cnt3 = sm_red[3usize * wg_usize];
    sync_cube();

    // Stage 3: lever arm sums for +x / -x.
    sm_red[tid] = u32::reinterpret(ax0);
    sm_red[wg_usize + tid] = u32::reinterpret(ay0);
    sm_red[2usize * wg_usize + tid] = u32::reinterpret(ax1);
    sm_red[3usize * wg_usize + tid] = u32::reinterpret(ay1);
    sync_cube();
    let mut s3 = wg_size / 2u32;
    while s3 > 0u32 {
        let step = s3 as usize;
        if tid < step {
            sm_red[tid] = u32::reinterpret(
                f32::reinterpret(sm_red[tid]) + f32::reinterpret(sm_red[tid + step]),
            );
            sm_red[wg_usize + tid] = u32::reinterpret(
                f32::reinterpret(sm_red[wg_usize + tid])
                    + f32::reinterpret(sm_red[wg_usize + tid + step]),
            );
            sm_red[2usize * wg_usize + tid] = u32::reinterpret(
                f32::reinterpret(sm_red[2usize * wg_usize + tid])
                    + f32::reinterpret(sm_red[2usize * wg_usize + tid + step]),
            );
            sm_red[3usize * wg_usize + tid] = u32::reinterpret(
                f32::reinterpret(sm_red[3usize * wg_usize + tid])
                    + f32::reinterpret(sm_red[3usize * wg_usize + tid + step]),
            );
        }
        sync_cube();
        s3 /= 2u32;
    }
    let tot_ax0 = f32::reinterpret(sm_red[0]);
    let tot_ay0 = f32::reinterpret(sm_red[wg_usize]);
    let tot_ax1 = f32::reinterpret(sm_red[2usize * wg_usize]);
    let tot_ay1 = f32::reinterpret(sm_red[3usize * wg_usize]);
    sync_cube();

    // Stage 4: lever arm sums for +y / -y.
    sm_red[tid] = u32::reinterpret(ax2);
    sm_red[wg_usize + tid] = u32::reinterpret(ay2);
    sm_red[2usize * wg_usize + tid] = u32::reinterpret(ax3);
    sm_red[3usize * wg_usize + tid] = u32::reinterpret(ay3);
    sync_cube();
    let mut s4 = wg_size / 2u32;
    while s4 > 0u32 {
        let step = s4 as usize;
        if tid < step {
            sm_red[tid] = u32::reinterpret(
                f32::reinterpret(sm_red[tid]) + f32::reinterpret(sm_red[tid + step]),
            );
            sm_red[wg_usize + tid] = u32::reinterpret(
                f32::reinterpret(sm_red[wg_usize + tid])
                    + f32::reinterpret(sm_red[wg_usize + tid + step]),
            );
            sm_red[2usize * wg_usize + tid] = u32::reinterpret(
                f32::reinterpret(sm_red[2usize * wg_usize + tid])
                    + f32::reinterpret(sm_red[2usize * wg_usize + tid + step]),
            );
            sm_red[3usize * wg_usize + tid] = u32::reinterpret(
                f32::reinterpret(sm_red[3usize * wg_usize + tid])
                    + f32::reinterpret(sm_red[3usize * wg_usize + tid + step]),
            );
        }
        sync_cube();
        s4 /= 2u32;
    }
    let tot_ax2 = f32::reinterpret(sm_red[0]);
    let tot_ay2 = f32::reinterpret(sm_red[wg_usize]);
    let tot_ax3 = f32::reinterpret(sm_red[2usize * wg_usize]);
    let tot_ay3 = f32::reinterpret(sm_red[3usize * wg_usize]);
    sync_cube();

    // Stage 5: fluid buoyancy & viscous drag deltas.
    sm_red[tid] = u32::reinterpret(f_dvx);
    sm_red[wg_usize + tid] = u32::reinterpret(f_dvy);
    sm_red[2usize * wg_usize + tid] = u32::reinterpret(f_dw);
    sync_cube();
    let mut s5 = wg_size / 2u32;
    while s5 > 0u32 {
        let step = s5 as usize;
        if tid < step {
            sm_red[tid] = u32::reinterpret(
                f32::reinterpret(sm_red[tid]) + f32::reinterpret(sm_red[tid + step]),
            );
            sm_red[wg_usize + tid] = u32::reinterpret(
                f32::reinterpret(sm_red[wg_usize + tid])
                    + f32::reinterpret(sm_red[wg_usize + tid + step]),
            );
            sm_red[2usize * wg_usize + tid] = u32::reinterpret(
                f32::reinterpret(sm_red[2usize * wg_usize + tid])
                    + f32::reinterpret(sm_red[2usize * wg_usize + tid + step]),
            );
        }
        sync_cube();
        s5 /= 2u32;
    }

    if tid == 0usize {
        let clamped_f = clamp_velocity(
            vx + f32::reinterpret(sm_red[0]),
            vy + f32::reinterpret(sm_red[wg_usize]),
            omega + f32::reinterpret(sm_red[2usize * wg_usize]),
        );
        vx = clamped_f.0;
        vy = clamped_f.1;
        omega = clamped_f.2;

        if tot_cnt0 > 0u32 {
            let inv_s = 1.0f32 / tot_cnt0 as f32;
            let res = resolve_pair_impulse(
                vx,
                vy,
                omega,
                inv_mass,
                inv_inertia,
                0.0f32,
                0.0f32,
                tot_ax0 * inv_s,
                tot_ay0 * inv_s,
                0.0f32,
                0.0f32,
                1.0f32,
                0.0f32,
                0.0f32,
                0.0f32,
            );
            vx = res.0;
            vy = res.1;
            omega = res.2;
        }
        if tot_cnt1 > 0u32 {
            let inv_s = 1.0f32 / tot_cnt1 as f32;
            let res = resolve_pair_impulse(
                vx,
                vy,
                omega,
                inv_mass,
                inv_inertia,
                0.0f32,
                0.0f32,
                tot_ax1 * inv_s,
                tot_ay1 * inv_s,
                0.0f32,
                0.0f32,
                -1.0f32,
                0.0f32,
                0.0f32,
                0.0f32,
            );
            vx = res.0;
            vy = res.1;
            omega = res.2;
        }
        if tot_cnt2 > 0u32 {
            let inv_s = 1.0f32 / tot_cnt2 as f32;
            let res = resolve_pair_impulse(
                vx,
                vy,
                omega,
                inv_mass,
                inv_inertia,
                0.0f32,
                0.0f32,
                tot_ax2 * inv_s,
                tot_ay2 * inv_s,
                0.0f32,
                0.0f32,
                0.0f32,
                1.0f32,
                0.0f32,
                0.0f32,
            );
            vx = res.0;
            vy = res.1;
            omega = res.2;
        }
        if tot_cnt3 > 0u32 {
            let inv_s = 1.0f32 / tot_cnt3 as f32;
            let res = resolve_pair_impulse(
                vx,
                vy,
                omega,
                inv_mass,
                inv_inertia,
                0.0f32,
                0.0f32,
                tot_ax3 * inv_s,
                tot_ay3 * inv_s,
                0.0f32,
                0.0f32,
                0.0f32,
                -1.0f32,
                0.0f32,
                0.0f32,
            );
            vx = res.0;
            vy = res.1;
            omega = res.2;
        }

        let bb = bi * B_STRIDE;
        let mut best = zero_of(tot_cpx);
        let mut code = seed_u32(substeps, 0u32);
        if tot_cpx > best {
            best = tot_cpx;
            code = 1u32;
        }
        if tot_cnx > best {
            best = tot_cnx;
            code = 2u32;
        }
        if tot_cpy > best {
            best = tot_cpy;
            code = 3u32;
        }
        if tot_cny > best {
            code = 4u32;
        }
        if code == 0u32 {
            code = bodies[bb + B_SUPPORT_NEXT];
        }
        if code == 0u32 && vx * vx + vy * vy < 0.25f32 {
            code = sm_body[25];
        }
        sm_body[25] = code;
        bodies[bb + B_SUPPORT] = code;

        let final_x = clampf(cmx + tot_cpx - tot_cnx, 0.5f32, xres as f32 - 0.5f32);
        let final_y = clampf(cmy + tot_cpy - tot_cny, 0.5f32, yres as f32 - 0.5f32);

        if advance_next != 0u32 && substep + 1u32 < substeps {
            sm_body[12] = u32::reinterpret(final_x);
            sm_body[13] = u32::reinterpret(final_y);
            sm_body[17] = sm_body[15];
            sm_body[18] = sm_body[16];
            bodies[bb + B_PREV_CM_X] = u32::reinterpret(final_x);
            bodies[bb + B_PREV_CM_Y] = u32::reinterpret(final_y);
            bodies[bb + B_PREV_ANGLE] = u32::reinterpret(angle);
            bodies[bb + B_PREV_COS_A] = sm_body[15];
            bodies[bb + B_PREV_SIN_A] = sm_body[16];

            let next = integrate_substep_step(
                final_x,
                final_y,
                angle,
                vx,
                vy,
                omega,
                ax,
                ay,
                angular_acc,
                radius,
                substeps,
                frames,
            );
            sm_body[10] = u32::reinterpret(next.0);
            sm_body[11] = u32::reinterpret(next.1);
            sm_body[14] = u32::reinterpret(next.2);
            sm_body[19] = u32::reinterpret(next.3);
            sm_body[20] = u32::reinterpret(next.4);
            sm_body[21] = u32::reinterpret(next.5);
            sm_body[15] = u32::reinterpret(next.6);
            sm_body[16] = u32::reinterpret(next.7);
        } else {
            sm_body[10] = u32::reinterpret(final_x);
            sm_body[11] = u32::reinterpret(final_y);
            sm_body[19] = u32::reinterpret(vx);
            sm_body[20] = u32::reinterpret(vy);
            sm_body[21] = u32::reinterpret(omega);
        }
    }
    sync_cube();
}

/// Loads `sm_body` from `bodies[bi]` and optionally folds `B_REACT_*` queued
/// by `contact_pair_kernel(iter = 1)`.
#[cube]
fn load_body_into_shared(
    bodies: &mut [u32],
    bb: usize,
    sm_body: &mut [u32],
    fold_react: u32,
) {
    let mut vx = fword_m(bodies, bb + B_VX);
    let mut vy = fword_m(bodies, bb + B_VY);
    let mut omega = fword_m(bodies, bb + B_OMEGA);
    if fold_react != 0u32 {
        let fx = i32::reinterpret(bodies[bb + B_REACT_VX]);
        let fy = i32::reinterpret(bodies[bb + B_REACT_VY]);
        let fw = i32::reinterpret(bodies[bb + B_REACT_VW]);
        bodies[bb + B_REACT_VX] = 0u32;
        bodies[bb + B_REACT_VY] = 0u32;
        bodies[bb + B_REACT_VW] = 0u32;
        if (bodies[bb + B_FLAGS] & 1u32) != 0u32 && (fx != 0i32 || fy != 0i32 || fw != 0i32) {
            let dvx = fx as f32 * REACTION_INV_SCALE;
            let dvy = fy as f32 * REACTION_INV_SCALE;
            let dw = fw as f32 * REACTION_INV_SCALE;
            let clamped = clamp_velocity(vx + dvx, vy + dvy, omega + dw);
            vx = clamped.0;
            vy = clamped.1;
            omega = clamped.2;
            bodies[bb + B_VX] = u32::reinterpret(vx);
            bodies[bb + B_VY] = u32::reinterpret(vy);
            bodies[bb + B_OMEGA] = u32::reinterpret(omega);
        }
    }
    sm_body[0] = bodies[bb + B_PIXEL_START];
    sm_body[1] = bodies[bb + B_SURFACE_COUNT];
    sm_body[2] = bodies[bb + B_SUBSTEPS];
    sm_body[3] = bodies[bb + B_ALIVE];
    sm_body[4] = bodies[bb + B_UNIT_WEIGHT];
    sm_body[5] = bodies[bb + B_INV_MASS];
    sm_body[6] = bodies[bb + B_INV_INERTIA];
    sm_body[7] = bodies[bb + B_ORIGIN_X];
    sm_body[8] = bodies[bb + B_ORIGIN_Y];
    sm_body[9] = bodies[bb + B_RADIUS];
    sm_body[10] = bodies[bb + B_CM_X];
    sm_body[11] = bodies[bb + B_CM_Y];
    sm_body[12] = bodies[bb + B_PREV_CM_X];
    sm_body[13] = bodies[bb + B_PREV_CM_Y];
    sm_body[14] = bodies[bb + B_ANGLE];
    sm_body[15] = bodies[bb + B_COS_A];
    sm_body[16] = bodies[bb + B_SIN_A];
    sm_body[17] = bodies[bb + B_PREV_COS_A];
    sm_body[18] = bodies[bb + B_PREV_SIN_A];
    sm_body[19] = u32::reinterpret(vx);
    sm_body[20] = u32::reinterpret(vy);
    sm_body[21] = u32::reinterpret(omega);
    sm_body[22] = bodies[bb + B_AX];
    sm_body[23] = bodies[bb + B_AY];
    sm_body[24] = bodies[bb + B_ANGULAR_ACC];
    sm_body[25] = bodies[bb + B_SUPPORT];
}

#[cube]
fn store_body_from_shared(bodies: &mut [u32], bb: usize, sm_body: &[u32]) {
    bodies[bb + B_CM_X] = sm_body[10];
    bodies[bb + B_CM_Y] = sm_body[11];
    bodies[bb + B_ANGLE] = sm_body[14];
    bodies[bb + B_COS_A] = sm_body[15];
    bodies[bb + B_SIN_A] = sm_body[16];
    bodies[bb + B_VX] = sm_body[19];
    bodies[bb + B_VY] = sm_body[20];
    bodies[bb + B_OMEGA] = sm_body[21];
}

/// Static contact pass (`mode = 1`) for one sub-step in the multi-body pair loop.
///
/// Folds queued reactions from pair iteration 1, resolves world / fluid
/// contacts across `WG_SIZE` threads, updates pose/velocity, and integrates
/// `substep + 1` when `substep + 1 < body.substeps`.
#[cube(launch_unchecked)]
pub(super) fn contact_static_kernel(
    bodies: &mut [u32],
    pixels: &[u32],
    surface_list: &[u32],
    pmap: &mut [Atomic<u32>],
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
    body_count: u32,
    substep: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    gravity_mode: u32,
    frames: f32,
) {
    let bi = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let bb = bi * B_STRIDE;

    let mut sm_run = Shared::<u32>::new();
    let mut sm_body = Shared::<[u32]>::new_slice(26usize);
    let mut sm_red = Shared::<[u32]>::new_slice(256usize);
    if tid == 0usize {
        let mut run = seed_u32(body_count, 0u32);
        if bi < body_count as usize {
            load_body_into_shared(bodies, bb, sm_body.as_mut_slice(), 1u32);
            if (bodies[bb + B_FLAGS] & 1u32) != 0u32 && substep < sm_body[2] {
                run = 1u32;
            }
        }
        *sm_run.inner_mut() = run;
    }
    let run = workgroup_uniform_load(sm_run.inner_ref());
    if run == 0u32 {
        terminate!();
    }
    sync_cube();

    run_one_static_substep(
        bi,
        tid,
        substep,
        1u32,
        bodies,
        pixels,
        surface_list,
        pmap,
        part_ty,
        part_x,
        part_y,
        part_vx,
        part_vy,
        part_flags,
        wall_bmap,
        wall_emap,
        elem_props,
        owner,
        sm_body.as_mut_slice(),
        sm_red.as_mut_slice(),
        xres,
        yres,
        xcells,
        ycells,
        gravity_mode,
        frames,
        64u32,
    );

    if tid == 0usize {
        store_body_from_shared(bodies, bb, sm_body.as_slice());
    }
}

/// Runs every sub-step `0..body.substeps` in a single GPU dispatch when no body
/// pairs overlap (or when `body_count == 1`).
#[cube]
fn run_static_all_substeps(
    bodies: &mut [u32],
    pixels: &[u32],
    surface_list: &[u32],
    pmap: &mut [Atomic<u32>],
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
    sm_red: &mut [u32],
    body_count: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    gravity_mode: u32,
    frames: f32,
    wg_size: u32,
) {
    let bi = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let bb = bi * B_STRIDE;

    let mut sm_substeps = Shared::<u32>::new();
    let mut sm_body = Shared::<[u32]>::new_slice(26usize);
    if tid == 0usize {
        let mut subs = seed_u32(body_count, 0u32);
        if bi < body_count as usize && (bodies[bb + B_FLAGS] & 1u32) != 0u32 {
            load_body_into_shared(bodies, bb, sm_body.as_mut_slice(), 0u32);
            subs = sm_body[2];
        }
        *sm_substeps.inner_mut() = subs;
    }
    let uniform_substeps = workgroup_uniform_load(sm_substeps.inner_ref());
    if uniform_substeps == 0u32 {
        terminate!();
    }
    sync_cube();

    for substep in 0u32..uniform_substeps {
        run_one_static_substep(
            bi,
            tid,
            substep,
            1u32,
            bodies,
            pixels,
            surface_list,
            pmap,
            part_ty,
            part_x,
            part_y,
            part_vx,
            part_vy,
            part_flags,
            wall_bmap,
            wall_emap,
            elem_props,
            owner,
            sm_body.as_mut_slice(),
            sm_red,
            xres,
            yres,
            xcells,
            ycells,
            gravity_mode,
            frames,
            wg_size,
        );
    }

    if tid == 0usize {
        store_body_from_shared(bodies, bb, sm_body.as_slice());
    }
}

#[cube(launch_unchecked)]
pub(super) fn static_all_substeps_64_kernel(
    bodies: &mut [u32],
    pixels: &[u32],
    surface_list: &[u32],
    pmap: &mut [Atomic<u32>],
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
    body_count: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    gravity_mode: u32,
    frames: f32,
) {
    let mut sm_red = Shared::<[u32]>::new_slice(256usize);
    run_static_all_substeps(
        bodies,
        pixels,
        surface_list,
        pmap,
        part_ty,
        part_x,
        part_y,
        part_vx,
        part_vy,
        part_flags,
        wall_bmap,
        wall_emap,
        elem_props,
        owner,
        sm_red.as_mut_slice(),
        body_count,
        xres,
        yres,
        xcells,
        ycells,
        gravity_mode,
        frames,
        64u32,
    );
}

#[cube(launch_unchecked)]
pub(super) fn static_all_substeps_256_kernel(
    bodies: &mut [u32],
    pixels: &[u32],
    surface_list: &[u32],
    pmap: &mut [Atomic<u32>],
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
    body_count: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    gravity_mode: u32,
    frames: f32,
) {
    let mut sm_red = Shared::<[u32]>::new_slice(1024usize);
    run_static_all_substeps(
        bodies,
        pixels,
        surface_list,
        pmap,
        part_ty,
        part_x,
        part_y,
        part_vx,
        part_vy,
        part_flags,
        wall_bmap,
        wall_emap,
        elem_props,
        owner,
        sm_red.as_mut_slice(),
        body_count,
        xres,
        yres,
        xcells,
        ycells,
        gravity_mode,
        frames,
        256u32,
    );
}
