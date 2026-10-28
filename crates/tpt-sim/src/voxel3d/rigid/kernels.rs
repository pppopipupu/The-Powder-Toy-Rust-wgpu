//! CubeCL kernels of the 3D rigid-body solver.
//!
//! The solver keeps the whole 64-body state on the device (plan §7.5). One
//! tick runs four substeps of
//!
//! 1. [`integrate_kernel`]: gravity, damping, pose integration and the render
//!    description (`model` / `inv_model` / world AABB).
//! 2. [`world_contacts_kernel`]: one workgroup per body, one thread per entry
//!    of the body's surface list. Each surface voxel samples the static world
//!    and accumulates a Q12 impulse into the body's shared `contact_accum`.
//! 3. [`pair_contacts_kernel`]: one thread per body pair. It broad-phases with
//!    the world AABBs and samples the other body's local occupancy.
//! 4. [`apply_impulses_kernel`]: converts the integer accumulators back to
//!    floats, applies them, updates the pose correction and handles sleeping.
//!
//! After the four substeps [`displace_loose_kernel`] pushes loose voxels out of
//! the body volume and [`body_block_claim_kernel`] publishes the body cells
//! into the world `claim` buffer so voxel movement cannot enter a body.
//!
//! Fixed-point impulses use 2^12 units. All integer atomics are additive and
//!
//! Range checks are spelled out instead of `Range::contains` and nested guards
//! stay separate because CubeCL's kernel expander does not support the iterator
//! and range forms; clippy's host-side suggestions would change the IR.
#![allow(clippy::manual_range_contains, clippy::collapsible_if)]
//! therefore independent of thread scheduling; the only per-thread division
//! ([`apply_impulses_kernel`]) uses the contact count collected in the same
//! accumulator, so repeated runs produce identical results on one device.
//!
//! The kernels deliberately use numeric literals instead of the host stride
//! constants (CubeCL cannot expand host constants inside a kernel body); the
//! assertions in `rigid::state` pin the two values together.

#![allow(clippy::too_many_arguments)]

use cubecl::prelude::*;

/// Element ids are the 2D Powder Toy ids, mirrored here because a kernel
/// cannot read the host element table (plan §6.1).
///
/// `PT_DUST=1, PT_WATR=2, PT_OIL=3, PT_FIRE=4, PT_STNE=5, PT_LAVA=6,
/// PT_GUNP=7, PT_METL=11, PT_ICEI=14, PT_WOOD=15, PT_SMKE=16, PT_SAND=17,
/// PT_WTRV=21`.
#[cube]
fn is_hard_ty(ty: u32) -> u32 {
    let mut hard = 0u32;
    if ty == 5u32 || ty == 15u32 || ty == 14u32 || ty == 11u32 {
        hard = 1u32;
    }
    hard
}

#[cube]
fn is_loose_ty(ty: u32) -> u32 {
    let mut loose = 0u32;
    if ty == 17u32
        || ty == 1u32
        || ty == 2u32
        || ty == 6u32
        || ty == 3u32
        || ty == 21u32
        || ty == 16u32
        || ty == 4u32
        || ty == 7u32
    {
        loose = 1u32;
    }
    loose
}

/// Runtime-seeded zero, see AGENTS.md "Runtime-Seeded Mutables".
#[cube]
#[allow(clippy::erasing_op)]
fn seed_f32(anchor: f32, value: f32) -> f32 {
    anchor * 0.0f32 + value
}

/// Runtime-seeded integer, see AGENTS.md "Runtime-Seeded Mutables".
#[cube]
#[allow(clippy::erasing_op)]
fn seed_u32(anchor: u32, value: u32) -> u32 {
    anchor * 0u32 + value
}

#[cube]
fn maxf(a: f32, b: f32) -> f32 {
    let mut out = a;
    if b > out {
        out = b;
    }
    out
}

#[cube]
fn minf(a: f32, b: f32) -> f32 {
    let mut out = a;
    if b < out {
        out = b;
    }
    out
}

/// Rotates `v` by the unit quaternion `q = (x, y, z, w)`.
#[cube]
fn quat_rotate(qx: f32, qy: f32, qz: f32, qw: f32, vx: f32, vy: f32, vz: f32) -> (f32, f32, f32) {
    let tx = 2.0f32 * (qy * vz - qz * vy);
    let ty = 2.0f32 * (qz * vx - qx * vz);
    let tz = 2.0f32 * (qx * vy - qy * vx);
    (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )
}

/// Applies the body's diagonal local inverse inertia in world space.
#[cube]
fn inv_inertia_vec(
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    ixx: f32,
    iyy: f32,
    izz: f32,
    vx: f32,
    vy: f32,
    vz: f32,
) -> (f32, f32, f32) {
    let (lx, ly, lz) = quat_rotate(-qx, -qy, -qz, qw, vx, vy, vz);
    quat_rotate(qx, qy, qz, qw, lx * ixx, ly * iyy, lz * izz)
}

/// Reads one voxel type from a body's local 32^3 grid.
#[cube]
fn body_voxel_at(voxels: &[u32], base: usize, x: i32, y: i32, z: i32) -> u32 {
    let mut out = seed_u32(x as u32, 0u32);
    if x >= 0i32 && x < 32i32 && y >= 0i32 && y < 32i32 && z >= 0i32 && z < 32i32 {
        let index = (y as u32 * 1024u32) + (z as u32 * 32u32) + x as u32;
        out = voxels[base + index as usize] & 255u32;
    }
    out
}

/// Reads one voxel type from the 256^3 world.
#[cube]
fn world_voxel_at(voxels: &[u32], x: i32, y: i32, z: i32) -> u32 {
    let mut out = seed_u32(x as u32, 0u32);
    if x >= 0i32 && x < 256i32 && y >= 0i32 && y < 256i32 && z >= 0i32 && z < 256i32 {
        let index = (y as u32 * 65536u32) + (z as u32 * 256u32) + x as u32;
        out = voxels[index as usize] & 255u32;
    }
    out
}

/// Writes the 40-word render description of one body (plan §7.2).
///
/// `model` maps local grid coordinates `[0, 32]` to world coordinates and is
/// stored column-major, matching the `array<vec4<f32>>` binding.
#[cube]
fn write_desc(
    desc: &mut [f32],
    base: usize,
    px: f32,
    py: f32,
    pz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    lminx: f32,
    lminy: f32,
    lminz: f32,
    lmaxx: f32,
    lmaxy: f32,
    lmaxz: f32,
) {
    let (r00, r10, r20) = quat_rotate(qx, qy, qz, qw, 1.0f32, 0.0f32, 0.0f32);
    let (r01, r11, r21) = quat_rotate(qx, qy, qz, qw, 0.0f32, 1.0f32, 0.0f32);
    let (r02, r12, r22) = quat_rotate(qx, qy, qz, qw, 0.0f32, 0.0f32, 1.0f32);
    let tx = px - 16.0f32 * (r00 + r01 + r02);
    let ty = py - 16.0f32 * (r10 + r11 + r12);
    let tz = pz - 16.0f32 * (r20 + r21 + r22);

    desc[base] = r00;
    desc[base + 1usize] = r10;
    desc[base + 2usize] = r20;
    desc[base + 3usize] = 0.0f32;
    desc[base + 4usize] = r01;
    desc[base + 5usize] = r11;
    desc[base + 6usize] = r21;
    desc[base + 7usize] = 0.0f32;
    desc[base + 8usize] = r02;
    desc[base + 9usize] = r12;
    desc[base + 10usize] = r22;
    desc[base + 11usize] = 0.0f32;
    desc[base + 12usize] = tx;
    desc[base + 13usize] = ty;
    desc[base + 14usize] = tz;
    desc[base + 15usize] = 1.0f32;

    let ux = 16.0f32 - (r00 * px + r01 * py + r02 * pz);
    let uy = 16.0f32 - (r10 * px + r11 * py + r12 * pz);
    let uz = 16.0f32 - (r20 * px + r21 * py + r22 * pz);
    desc[base + 16usize] = r00;
    desc[base + 17usize] = r01;
    desc[base + 18usize] = r02;
    desc[base + 19usize] = 0.0f32;
    desc[base + 20usize] = r10;
    desc[base + 21usize] = r11;
    desc[base + 22usize] = r12;
    desc[base + 23usize] = 0.0f32;
    desc[base + 24usize] = r20;
    desc[base + 25usize] = r21;
    desc[base + 26usize] = r22;
    desc[base + 27usize] = 0.0f32;
    desc[base + 28usize] = ux;
    desc[base + 29usize] = uy;
    desc[base + 30usize] = uz;
    desc[base + 31usize] = 1.0f32;

    let mut mnx = seed_f32(px, 0.0f32);
    let mut mny = seed_f32(px, 0.0f32);
    let mut mnz = seed_f32(px, 0.0f32);
    let mut mxx = seed_f32(px, 0.0f32);
    let mut mxy = seed_f32(px, 0.0f32);
    let mut mxz = seed_f32(px, 0.0f32);
    for corner in 0..8usize {
        let mut lx = seed_f32(px, lminx);
        let mut ly = seed_f32(px, lminy);
        let mut lz = seed_f32(px, lminz);
        if corner % 2usize == 1usize {
            lx = lmaxx;
        }
        if (corner / 2usize) % 2usize == 1usize {
            ly = lmaxy;
        }
        if (corner / 4usize) % 2usize == 1usize {
            lz = lmaxz;
        }
        let wx = r00 * lx + r01 * ly + r02 * lz + tx;
        let wy = r10 * lx + r11 * ly + r12 * lz + ty;
        let wz = r20 * lx + r21 * ly + r22 * lz + tz;
        if corner == 0usize {
            mnx = wx;
            mny = wy;
            mnz = wz;
            mxx = wx;
            mxy = wy;
            mxz = wz;
        } else {
            if wx < mnx {
                mnx = wx;
            }
            if wy < mny {
                mny = wy;
            }
            if wz < mnz {
                mnz = wz;
            }
            if wx > mxx {
                mxx = wx;
            }
            if wy > mxy {
                mxy = wy;
            }
            if wz > mxz {
                mxz = wz;
            }
        }
    }
    desc[base + 32usize] = mnx;
    desc[base + 33usize] = mny;
    desc[base + 34usize] = mnz;
    desc[base + 35usize] = 1.0f32;
    desc[base + 36usize] = mxx;
    desc[base + 37usize] = mxy;
    desc[base + 38usize] = mxz;
    desc[base + 39usize] = 1.0f32;
}

/// One impulse of a single body against a static point.
///
/// `r` is the contact point relative to the body centre, `n` the unit contact
/// normal pointing away from the obstacle, `v` / `w` the body velocities. The
/// helper returns the world-space impulse to add to the body's accumulator
/// before the shared `1 / 4096 / contact_count` scaling.
#[cube]
fn contact_impulse(
    rx: f32,
    ry: f32,
    rz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    wx: f32,
    wy: f32,
    wz: f32,
    inv_mass: f32,
    ixx: f32,
    iyy: f32,
    izz: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    mu: f32,
) -> (f32, f32, f32) {
    let rnx = ry * nz - rz * ny;
    let rny = rz * nx - rx * nz;
    let rnz = rx * ny - ry * nx;
    let (i0x, i0y, i0z) = inv_inertia_vec(qx, qy, qz, qw, ixx, iyy, izz, rnx, rny, rnz);
    let cx = i0y * rz - i0z * ry;
    let cy = i0z * rx - i0x * rz;
    let cz = i0x * ry - i0y * rx;
    let mut k = inv_mass + cx * nx + cy * ny + cz * nz;
    if k < 0.000001f32 {
        k = 0.000001f32;
    }

    let vpx = vx + wy * rz - wz * ry;
    let vpy = vy + wz * rx - wx * rz;
    let vpz = vz + wx * ry - wy * rx;
    let vn = vpx * nx + vpy * ny + vpz * nz;

    let mut jn = seed_f32(vn, 0.0f32);
    if vn < 0.0f32 {
        jn = -vn / k;
    }

    let vtx = vpx - nx * vn;
    let vty = vpy - ny * vn;
    let vtz = vpz - nz * vn;
    let vtl = (vtx * vtx + vty * vty + vtz * vtz).sqrt();
    let mut tx = seed_f32(vtl, 0.0f32);
    let mut ty = seed_f32(vtl, 0.0f32);
    let mut tz = seed_f32(vtl, 0.0f32);
    let mut friction = seed_f32(vtl, 0.0f32);
    if vtl > 0.0001f32 && jn > 0.0f32 {
        tx = vtx / vtl;
        ty = vty / vtl;
        tz = vtz / vtl;
        let rtx = ry * tz - rz * ty;
        let rty = rz * tx - rx * tz;
        let rtz = rx * ty - ry * tx;
        let (j0x, j0y, j0z) = inv_inertia_vec(qx, qy, qz, qw, ixx, iyy, izz, rtx, rty, rtz);
        let c2x = j0y * rz - j0z * ry;
        let c2y = j0z * rx - j0x * rz;
        let c2z = j0x * ry - j0y * rx;
        let kt = inv_mass + c2x * tx + c2y * ty + c2z * tz;
        if kt > 0.000001f32 {
            friction = minf(vtl / kt, mu * jn);
        }
    }

    (
        nx * jn - tx * friction,
        ny * jn - ty * friction,
        nz * jn - tz * friction,
    )
}

/// Integrates gravity, damping and the pose of every awake body.
#[cube(launch_unchecked)]
pub(super) fn integrate_kernel(
    body_state: &mut [f32],
    body_desc: &mut [f32],
    body_count: u32,
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
) {
    let i = ABSOLUTE_POS;
    if i < body_count as usize {
        let b = i * 32usize;
        let awake = body_state[b + 29usize];
        let mut npx = body_state[b];
        let mut npy = body_state[b + 1usize];
        let mut npz = body_state[b + 2usize];
        let mut nqx = body_state[b + 4usize];
        let mut nqy = body_state[b + 5usize];
        let mut nqz = body_state[b + 6usize];
        let mut nqw = body_state[b + 7usize];
        if awake > 0.5f32 {
            let mut nvx = body_state[b + 8usize] + gravity_x * dt;
            let mut nvy = body_state[b + 9usize] + gravity_y * dt;
            let mut nvz = body_state[b + 10usize] + gravity_z * dt;
            nvx *= 0.999f32;
            nvy *= 0.999f32;
            nvz *= 0.999f32;
            npx += nvx * dt;
            npy += nvy * dt;
            npz += nvz * dt;

            let wx = body_state[b + 12usize] * 0.995f32;
            let wy = body_state[b + 13usize] * 0.995f32;
            let wz = body_state[b + 14usize] * 0.995f32;
            let hdt = 0.5f32 * dt;
            let qx0 = body_state[b + 4usize];
            let qy0 = body_state[b + 5usize];
            let qz0 = body_state[b + 6usize];
            let qw0 = body_state[b + 7usize];
            nqx = qx0 + hdt * (wx * qw0 + wy * qz0 - wz * qy0);
            nqy = qy0 + hdt * (-wx * qz0 + wy * qw0 + wz * qx0);
            nqz = qz0 + hdt * (wx * qy0 - wy * qx0 + wz * qw0);
            nqw = qw0 + hdt * (-wx * qx0 - wy * qy0 - wz * qz0);
            let qlen = (nqx * nqx + nqy * nqy + nqz * nqz + nqw * nqw).sqrt();
            let qscale = 1.0f32 / maxf(qlen, 0.000001f32);
            nqx *= qscale;
            nqy *= qscale;
            nqz *= qscale;
            nqw *= qscale;

            body_state[b] = npx;
            body_state[b + 1usize] = npy;
            body_state[b + 2usize] = npz;
            body_state[b + 4usize] = nqx;
            body_state[b + 5usize] = nqy;
            body_state[b + 6usize] = nqz;
            body_state[b + 7usize] = nqw;
            body_state[b + 8usize] = nvx;
            body_state[b + 9usize] = nvy;
            body_state[b + 10usize] = nvz;
            body_state[b + 12usize] = wx;
            body_state[b + 13usize] = wy;
            body_state[b + 14usize] = wz;
        }

        let lminx = body_state[b + 20usize];
        let lminy = body_state[b + 21usize];
        let lminz = body_state[b + 22usize];
        let lmaxx = body_state[b + 24usize];
        let lmaxy = body_state[b + 25usize];
        let lmaxz = body_state[b + 26usize];
        write_desc(
            body_desc,
            i * 40usize,
            npx,
            npy,
            npz,
            nqx,
            nqy,
            nqz,
            nqw,
            lminx,
            lminy,
            lminz,
            lmaxx,
            lmaxy,
            lmaxz,
        );
    }
}

/// Resolves static-world contacts for every body surface voxel.
#[cube(launch_unchecked)]
pub(super) fn world_contacts_kernel(
    body_state: &[f32],
    surface: &[u32],
    surface_count: &[u32],
    voxel_a: &[u32],
    contact_accum: &mut [Atomic<i32>],
    body_count: u32,
) {
    let body = CUBE_POS_X as usize;
    if body < body_count as usize {
        let sb = body * 32usize;
        let vb = body * 32768usize;
        let ab = body * 16usize;
        let px = body_state[sb];
        let py = body_state[sb + 1usize];
        let pz = body_state[sb + 2usize];
        let qx = body_state[sb + 4usize];
        let qy = body_state[sb + 5usize];
        let qz = body_state[sb + 6usize];
        let qw = body_state[sb + 7usize];
        let vx = body_state[sb + 8usize];
        let vy = body_state[sb + 9usize];
        let vz = body_state[sb + 10usize];
        let wx = body_state[sb + 12usize];
        let wy = body_state[sb + 13usize];
        let wz = body_state[sb + 14usize];
        let inv_mass = body_state[sb + 16usize];
        let ixx = body_state[sb + 17usize];
        let iyy = body_state[sb + 18usize];
        let izz = body_state[sb + 19usize];
        let count = surface_count[body] as usize;
        let thread = UNIT_POS_X as usize;
        for slot in 0..128usize {
            let entry = thread + slot * 256usize;
            if entry < count {
                let index = surface[vb + entry] as usize;
                let idx = index as u32;
                let lx = (idx & 31u32) as i32;
                let ly = ((idx >> 10u32) & 31u32) as i32;
                let lz = ((idx >> 5u32) & 31u32) as i32;
                let (ox, oy, oz) = quat_rotate(
                    qx,
                    qy,
                    qz,
                    qw,
                    lx as f32 - 15.5f32,
                    ly as f32 - 15.5f32,
                    lz as f32 - 15.5f32,
                );
                let wpx = px + ox;
                let wpy = py + oy;
                let wpz = pz + oz;
                let cx = wpx.floor() as i32;
                let cy = wpy.floor() as i32;
                let cz = wpz.floor() as i32;
                if cx >= 0i32
                    && cx < 256i32
                    && cy >= 0i32
                    && cy < 256i32
                    && cz >= 0i32
                    && cz < 256i32
                {
                    // Negative gradient of the hard-solid occupancy in the
                    // 3x3x3 neighbourhood (plan §7.4). A flat floor gives +Y
                    // instead of the internal seam directions.
                    let mut gx = seed_f32(wpx, 0.0f32);
                    let mut gy = seed_f32(wpx, 0.0f32);
                    let mut gz = seed_f32(wpx, 0.0f32);
                    let mut solids = seed_u32(body_count, 0u32);
                    for sample in 0..27usize {
                        let oxi = (sample % 3usize) as i32 - 1i32;
                        let oyi = ((sample / 3usize) % 3usize) as i32 - 1i32;
                        let ozi = ((sample / 9usize) % 3usize) as i32 - 1i32;
                        let aty = world_voxel_at(voxel_a, cx + oxi, cy + oyi, cz + ozi);
                        if is_hard_ty(aty) != 0u32 {
                            gx += oxi as f32;
                            gy += oyi as f32;
                            gz += ozi as f32;
                            solids += 1u32;
                        }
                    }

                    if solids > 0u32 {
                        let glen2 = gx * gx + gy * gy + gz * gz;
                        let mut nx = seed_f32(glen2, 0.0f32);
                        let mut ny = seed_f32(glen2, 0.0f32);
                        let mut nz = seed_f32(glen2, 0.0f32);
                        let mut pen = seed_f32(glen2, -1.0f32);
                        if glen2 > 0.00000001f32 {
                            let inv = 1.0f32 / glen2.sqrt();
                            nx = -gx * inv;
                            ny = -gy * inv;
                            nz = -gz * inv;
                            for sample in 0..27usize {
                                let oxi = (sample % 3usize) as i32 - 1i32;
                                let oyi = ((sample / 3usize) % 3usize) as i32 - 1i32;
                                let ozi = ((sample / 9usize) % 3usize) as i32 - 1i32;
                                let aty = world_voxel_at(voxel_a, cx + oxi, cy + oyi, cz + ozi);
                                if is_hard_ty(aty) != 0u32 {
                                    let csx = (cx + oxi) as f32 + 0.5f32;
                                    let csy = (cy + oyi) as f32 + 0.5f32;
                                    let csz = (cz + ozi) as f32 + 0.5f32;
                                    let depth = 1.0f32
                                        - ((wpx - csx) * nx + (wpy - csy) * ny + (wpz - csz) * nz);
                                    if depth > pen {
                                        pen = depth;
                                    }
                                }
                            }
                        } else {
                            // Symmetric neighbourhood: fall back to the face
                            // normal of the deepest exposed axis.
                            for dir in 0..6usize {
                                let mut oxi = 0i32;
                                let mut oyi = 0i32;
                                let mut ozi = 0i32;
                                if dir == 0usize {
                                    oxi = 1i32;
                                } else if dir == 1usize {
                                    oxi = -1i32;
                                } else if dir == 2usize {
                                    oyi = 1i32;
                                } else if dir == 3usize {
                                    oyi = -1i32;
                                } else if dir == 4usize {
                                    ozi = 1i32;
                                } else {
                                    ozi = -1i32;
                                }
                                let aty = world_voxel_at(voxel_a, cx + oxi, cy + oyi, cz + ozi);
                                if is_hard_ty(aty) != 0u32 {
                                    let dnx = -(oxi as f32);
                                    let dny = -(oyi as f32);
                                    let dnz = -(ozi as f32);
                                    let csx = (cx + oxi) as f32 + 0.5f32;
                                    let csy = (cy + oyi) as f32 + 0.5f32;
                                    let csz = (cz + ozi) as f32 + 0.5f32;
                                    let depth = 1.0f32
                                        - ((wpx - csx) * dnx
                                            + (wpy - csy) * dny
                                            + (wpz - csz) * dnz);
                                    if depth > pen {
                                        pen = depth;
                                        nx = dnx;
                                        ny = dny;
                                        nz = dnz;
                                    }
                                }
                            }
                        }

                        // A small speculative margin lets the solver stop a
                        // falling body before its centre enters the floor cell.
                        if pen > 1.0f32 {
                            pen = 1.0f32;
                        }
                        if pen > -0.05f32 {
                            let rx = wpx - px;
                            let ry = wpy - py;
                            let rz = wpz - pz;
                            let (jx, jy, jz) = contact_impulse(
                                rx, ry, rz, nx, ny, nz, vx, vy, vz, wx, wy, wz, inv_mass, ixx, iyy,
                                izz, qx, qy, qz, qw, 0.5f32,
                            );
                            // Correct the full penetration: the speculative
                            // margin already keeps a falling body from
                            // entering the floor, and no slop means the resting
                            // bottom face cannot sit below the floor top.
                            let corr = maxf(pen, 0.0f32) * 0.8f32;
                            contact_accum[ab].fetch_add((jx * 4096.0f32) as i32);
                            contact_accum[ab + 1usize].fetch_add((jy * 4096.0f32) as i32);
                            contact_accum[ab + 2usize].fetch_add((jz * 4096.0f32) as i32);
                            contact_accum[ab + 4usize]
                                .fetch_add(((ry * jz - rz * jy) * 4096.0f32) as i32);
                            contact_accum[ab + 5usize]
                                .fetch_add(((rz * jx - rx * jz) * 4096.0f32) as i32);
                            contact_accum[ab + 6usize]
                                .fetch_add(((rx * jy - ry * jx) * 4096.0f32) as i32);
                            contact_accum[ab + 8usize].fetch_add((nx * corr * 4096.0f32) as i32);
                            contact_accum[ab + 9usize].fetch_add((ny * corr * 4096.0f32) as i32);
                            contact_accum[ab + 10usize].fetch_add((nz * corr * 4096.0f32) as i32);
                            contact_accum[ab + 12usize].fetch_add(1i32);
                        }
                    }
                }
            }
        }
    }
}

/// Gradient of the local occupancy of body `b` around a local grid cell.
#[cube]
fn local_gradient(voxels: &[u32], base: usize, ix: i32, iy: i32, iz: i32) -> (f32, f32, f32) {
    let mut gx = seed_f32(ix as f32, 0.0f32);
    let mut gy = seed_f32(ix as f32, 0.0f32);
    let mut gz = seed_f32(ix as f32, 0.0f32);
    for sample in 0..27usize {
        let oxi = (sample % 3usize) as i32 - 1i32;
        let oyi = ((sample / 3usize) % 3usize) as i32 - 1i32;
        let ozi = ((sample / 9usize) % 3usize) as i32 - 1i32;
        if body_voxel_at(voxels, base, ix + oxi, iy + oyi, iz + ozi) != 0u32 {
            gx += oxi as f32;
            gy += oyi as f32;
            gz += ozi as f32;
        }
    }
    (gx, gy, gz)
}

/// Resolves contacts between every body pair.
#[cube(launch_unchecked)]
pub(super) fn pair_contacts_kernel(
    body_state: &[f32],
    body_desc: &[f32],
    body_voxels: &[u32],
    surface: &[u32],
    surface_count: &[u32],
    contact_accum: &mut [Atomic<i32>],
    body_count: u32,
) {
    let a = CUBE_POS_X as usize;
    let b = UNIT_POS_X as usize;
    if a < body_count as usize && b < body_count as usize && a < b {
        let da = a * 40usize;
        let db = b * 40usize;
        let aminx = body_desc[da + 32usize];
        let aminy = body_desc[da + 33usize];
        let aminz = body_desc[da + 34usize];
        let amaxx = body_desc[da + 36usize];
        let amaxy = body_desc[da + 37usize];
        let amaxz = body_desc[da + 38usize];
        let bminx = body_desc[db + 32usize];
        let bminy = body_desc[db + 33usize];
        let bminz = body_desc[db + 34usize];
        let bmaxx = body_desc[db + 36usize];
        let bmaxy = body_desc[db + 37usize];
        let bmaxz = body_desc[db + 38usize];
        if aminx <= bmaxx
            && amaxx >= bminx
            && aminy <= bmaxy
            && amaxy >= bminy
            && aminz <= bmaxz
            && amaxz >= bminz
        {
            let sa = a * 32usize;
            let sb = b * 32usize;
            let va = a * 32768usize;
            let vb = b * 32768usize;
            let aa = a * 16usize;
            let ab = b * 16usize;
            let apx = body_state[sa];
            let apy = body_state[sa + 1usize];
            let apz = body_state[sa + 2usize];
            let aqx = body_state[sa + 4usize];
            let aqy = body_state[sa + 5usize];
            let aqz = body_state[sa + 6usize];
            let aqw = body_state[sa + 7usize];
            let avx = body_state[sa + 8usize];
            let avy = body_state[sa + 9usize];
            let avz = body_state[sa + 10usize];
            let awx = body_state[sa + 12usize];
            let awy = body_state[sa + 13usize];
            let awz = body_state[sa + 14usize];
            let aim = body_state[sa + 16usize];
            let aix = body_state[sa + 17usize];
            let aiy = body_state[sa + 18usize];
            let aiz = body_state[sa + 19usize];
            let bpx = body_state[sb];
            let bpy = body_state[sb + 1usize];
            let bpz = body_state[sb + 2usize];
            let bqx = body_state[sb + 4usize];
            let bqy = body_state[sb + 5usize];
            let bqz = body_state[sb + 6usize];
            let bqw = body_state[sb + 7usize];
            let bvx = body_state[sb + 8usize];
            let bvy = body_state[sb + 9usize];
            let bvz = body_state[sb + 10usize];
            let bwx = body_state[sb + 12usize];
            let bwy = body_state[sb + 13usize];
            let bwz = body_state[sb + 14usize];
            let bim = body_state[sb + 16usize];
            let bix = body_state[sb + 17usize];
            let biy = body_state[sb + 18usize];
            let biz = body_state[sb + 19usize];
            let count = surface_count[a] as usize;
            for entry in 0..count {
                let index = surface[va + entry] as usize;
                let idx = index as u32;
                let lx = (idx & 31u32) as i32;
                let ly = ((idx >> 10u32) & 31u32) as i32;
                let lz = ((idx >> 5u32) & 31u32) as i32;
                let (ox, oy, oz) = quat_rotate(
                    aqx,
                    aqy,
                    aqz,
                    aqw,
                    lx as f32 - 15.5f32,
                    ly as f32 - 15.5f32,
                    lz as f32 - 15.5f32,
                );
                let wpx = apx + ox;
                let wpy = apy + oy;
                let wpz = apz + oz;
                let (rlx, rly, rlz) =
                    quat_rotate(-bqx, -bqy, -bqz, bqw, wpx - bpx, wpy - bpy, wpz - bpz);
                let blx = rlx + 16.0f32;
                let bly = rly + 16.0f32;
                let blz = rlz + 16.0f32;
                let ix = blx.floor() as i32;
                let iy = bly.floor() as i32;
                let iz = blz.floor() as i32;
                if ix >= 0i32 && ix < 32i32 && iy >= 0i32 && iy < 32i32 && iz >= 0i32 && iz < 32i32
                {
                    let local = (iy as u32 * 1024u32) + (iz as u32 * 32u32) + ix as u32;
                    if (body_voxels[vb + local as usize] & 255u32) != 0u32 {
                        let (gx, gy, gz) = local_gradient(body_voxels, vb, ix, iy, iz);
                        let glen2 = gx * gx + gy * gy + gz * gz;
                        let mut nlx = seed_f32(glen2, 0.0f32);
                        let mut nly = seed_f32(glen2, 1.0f32);
                        let mut nlz = seed_f32(glen2, 0.0f32);
                        if glen2 > 0.00000001f32 {
                            let inv = 1.0f32 / glen2.sqrt();
                            nlx = -gx * inv;
                            nly = -gy * inv;
                            nlz = -gz * inv;
                        } else {
                            let cx = blx - 16.0f32;
                            let cy = bly - 16.0f32;
                            let cz = blz - 16.0f32;
                            let cl = (cx * cx + cy * cy + cz * cz).sqrt();
                            if cl > 0.000001f32 {
                                nlx = cx / cl;
                                nly = cy / cl;
                                nlz = cz / cl;
                            }
                        }
                        let (nx, ny, nz) = quat_rotate(bqx, bqy, bqz, bqw, nlx, nly, nlz);
                        let (csx, csy, csz) = quat_rotate(
                            bqx,
                            bqy,
                            bqz,
                            bqw,
                            ix as f32 + 0.5f32 - 16.0f32,
                            iy as f32 + 0.5f32 - 16.0f32,
                            iz as f32 + 0.5f32 - 16.0f32,
                        );
                        let wx = bpx + csx;
                        let wy = bpy + csy;
                        let wz = bpz + csz;
                        let pen = 1.0f32 - ((wpx - wx) * nx + (wpy - wy) * ny + (wpz - wz) * nz);
                        if pen > -0.05f32 {
                            let rax = wpx - apx;
                            let ray = wpy - apy;
                            let raz = wpz - apz;
                            let rbx = wpx - bpx;
                            let rby = wpy - bpy;
                            let rbz = wpz - bpz;
                            let avpx = avx + awy * raz - awz * ray;
                            let avpy = avy + awz * rax - awx * raz;
                            let avpz = avz + awx * ray - awy * rax;
                            let bvpx = bvx + bwy * rbz - bwz * rby;
                            let bvpy = bvy + bwz * rbx - bwx * rbz;
                            let bvpz = bvz + bwx * rby - bwy * rbx;
                            let relx = avpx - bvpx;
                            let rely = avpy - bvpy;
                            let relz = avpz - bvpz;
                            let vn = relx * nx + rely * ny + relz * nz;

                            let ranx = ray * nz - raz * ny;
                            let rany = raz * nx - rax * nz;
                            let ranz = rax * ny - ray * nx;
                            let (ia0x, ia0y, ia0z) = inv_inertia_vec(
                                aqx, aqy, aqz, aqw, aix, aiy, aiz, ranx, rany, ranz,
                            );
                            let ca = (ia0y * raz - ia0z * ray) * nx
                                + (ia0z * rax - ia0x * raz) * ny
                                + (ia0x * ray - ia0y * rax) * nz;
                            let rbnx = rby * nz - rbz * ny;
                            let rbny = rbz * nx - rbx * nz;
                            let rbnz = rbx * ny - rby * nx;
                            let (ib0x, ib0y, ib0z) = inv_inertia_vec(
                                bqx, bqy, bqz, bqw, bix, biy, biz, rbnx, rbny, rbnz,
                            );
                            let cb = (ib0y * rbz - ib0z * rby) * nx
                                + (ib0z * rbx - ib0x * rbz) * ny
                                + (ib0x * rby - ib0y * rbx) * nz;
                            let mut k = aim + bim + ca + cb;
                            if k < 0.000001f32 {
                                k = 0.000001f32;
                            }

                            let mut jn = seed_f32(vn, 0.0f32);
                            if vn < 0.0f32 {
                                jn = -vn / k;
                            }

                            let vtx = relx - nx * vn;
                            let vty = rely - ny * vn;
                            let vtz = relz - nz * vn;
                            let vtl = (vtx * vtx + vty * vty + vtz * vtz).sqrt();
                            let mut tx = seed_f32(vtl, 0.0f32);
                            let mut ty = seed_f32(vtl, 0.0f32);
                            let mut tz = seed_f32(vtl, 0.0f32);
                            let mut friction = seed_f32(vtl, 0.0f32);
                            if vtl > 0.0001f32 && jn > 0.0f32 {
                                tx = vtx / vtl;
                                ty = vty / vtl;
                                tz = vtz / vtl;
                                let ratx = ray * tz - raz * ty;
                                let raty = raz * tx - rax * tz;
                                let ratz = rax * ty - ray * tx;
                                let (ia1x, ia1y, ia1z) = inv_inertia_vec(
                                    aqx, aqy, aqz, aqw, aix, aiy, aiz, ratx, raty, ratz,
                                );
                                let cat = (ia1y * raz - ia1z * ray) * tx
                                    + (ia1z * rax - ia1x * raz) * ty
                                    + (ia1x * ray - ia1y * rax) * tz;
                                let rbtx = rby * tz - rbz * ty;
                                let rbty = rbz * tx - rbx * tz;
                                let rbtz = rbx * ty - rby * tx;
                                let (ib1x, ib1y, ib1z) = inv_inertia_vec(
                                    bqx, bqy, bqz, bqw, bix, biy, biz, rbtx, rbty, rbtz,
                                );
                                let cbt = (ib1y * rbz - ib1z * rby) * tx
                                    + (ib1z * rbx - ib1x * rbz) * ty
                                    + (ib1x * rby - ib1y * rbx) * tz;
                                let kt = aim + bim + cat + cbt;
                                if kt > 0.000001f32 {
                                    friction = minf(vtl / kt, 0.5f32 * jn);
                                }
                            }

                            let jx = nx * jn - tx * friction;
                            let jy = ny * jn - ty * friction;
                            let jz = nz * jn - tz * friction;
                            let corr = maxf(pen, 0.0f32) * 0.8f32;
                            contact_accum[aa].fetch_add((jx * 4096.0f32) as i32);
                            contact_accum[aa + 1usize].fetch_add((jy * 4096.0f32) as i32);
                            contact_accum[aa + 2usize].fetch_add((jz * 4096.0f32) as i32);
                            contact_accum[aa + 4usize]
                                .fetch_add(((ray * jz - raz * jy) * 4096.0f32) as i32);
                            contact_accum[aa + 5usize]
                                .fetch_add(((raz * jx - rax * jz) * 4096.0f32) as i32);
                            contact_accum[aa + 6usize]
                                .fetch_add(((rax * jy - ray * jx) * 4096.0f32) as i32);
                            contact_accum[aa + 8usize].fetch_add((nx * corr * 4096.0f32) as i32);
                            contact_accum[aa + 9usize].fetch_add((ny * corr * 4096.0f32) as i32);
                            contact_accum[aa + 10usize].fetch_add((nz * corr * 4096.0f32) as i32);
                            contact_accum[ab].fetch_add((-jx * 4096.0f32) as i32);
                            contact_accum[ab + 1usize].fetch_add((-jy * 4096.0f32) as i32);
                            contact_accum[ab + 2usize].fetch_add((-jz * 4096.0f32) as i32);
                            contact_accum[ab + 4usize]
                                .fetch_add((-(rby * jz - rbz * jy) * 4096.0f32) as i32);
                            contact_accum[ab + 5usize]
                                .fetch_add((-(rbz * jx - rbx * jz) * 4096.0f32) as i32);
                            contact_accum[ab + 6usize]
                                .fetch_add((-(rbx * jy - rby * jx) * 4096.0f32) as i32);
                            contact_accum[ab + 8usize].fetch_add((-nx * corr * 4096.0f32) as i32);
                            contact_accum[ab + 9usize].fetch_add((-ny * corr * 4096.0f32) as i32);
                            contact_accum[ab + 10usize].fetch_add((-nz * corr * 4096.0f32) as i32);
                            contact_accum[aa + 12usize].fetch_add(1i32);
                            contact_accum[ab + 12usize].fetch_add(1i32);
                        }
                    }
                }
            }
        }
    }
}

/// Applies the accumulated impulses, position corrections and sleep state.
#[cube(launch_unchecked)]
pub(super) fn apply_impulses_kernel(
    body_state: &mut [f32],
    body_desc: &mut [f32],
    contact_accum: &mut [Atomic<i32>],
    body_count: u32,
    quiet_step: f32,
) {
    let i = ABSOLUTE_POS;
    if i < body_count as usize {
        let b = i * 32usize;
        let a = i * 16usize;
        let l0 = contact_accum[a].load();
        let l1 = contact_accum[a + 1usize].load();
        let l2 = contact_accum[a + 2usize].load();
        let w0 = contact_accum[a + 4usize].load();
        let w1 = contact_accum[a + 5usize].load();
        let w2 = contact_accum[a + 6usize].load();
        let p0 = contact_accum[a + 8usize].load();
        let p1 = contact_accum[a + 9usize].load();
        let p2 = contact_accum[a + 10usize].load();
        let contacts = contact_accum[a + 12usize].load();
        for word in 0..13usize {
            contact_accum[a + word].store(0i32);
        }

        let mut inv_count = seed_f32(contacts as f32, 0.0f32);
        if contacts > 0i32 {
            inv_count = 1.0f32 / contacts as f32;
        }
        let scale = inv_count / 4096.0f32;
        let jx = (l0 as f32) * scale;
        let jy = (l1 as f32) * scale;
        let jz = (l2 as f32) * scale;
        let twx = (w0 as f32) * scale;
        let twy = (w1 as f32) * scale;
        let twz = (w2 as f32) * scale;
        let pcx = (p0 as f32) * scale;
        let pcy = (p1 as f32) * scale;
        let pcz = (p2 as f32) * scale;

        let qx = body_state[b + 4usize];
        let qy = body_state[b + 5usize];
        let qz = body_state[b + 6usize];
        let qw = body_state[b + 7usize];
        let inv_mass = body_state[b + 16usize];
        let ixx = body_state[b + 17usize];
        let iyy = body_state[b + 18usize];
        let izz = body_state[b + 19usize];
        let (dwx, dwy, dwz) = inv_inertia_vec(qx, qy, qz, qw, ixx, iyy, izz, twx, twy, twz);
        let mut nvx = body_state[b + 8usize] + jx * inv_mass;
        let mut nvy = body_state[b + 9usize] + jy * inv_mass;
        let mut nvz = body_state[b + 10usize] + jz * inv_mass;
        let mut nwx = body_state[b + 12usize] + dwx;
        let mut nwy = body_state[b + 13usize] + dwy;
        let mut nwz = body_state[b + 14usize] + dwz;
        let npx = body_state[b] + pcx;
        let npy = body_state[b + 1usize] + pcy;
        let npz = body_state[b + 2usize] + pcz;

        let awake = body_state[b + 29usize];
        let quiet = body_state[b + 28usize];
        let mut nquiet = quiet;
        let mut nawake = awake;
        let speed2 = nvx * nvx + nvy * nvy + nvz * nvz;
        let spin2 = nwx * nwx + nwy * nwy + nwz * nwz;
        let corr2 = pcx * pcx + pcy * pcy + pcz * pcz;
        let impulse2 = jx * jx + jy * jy + jz * jz;
        if awake > 0.5f32 {
            let mut calm = seed_u32(contacts as u32, 0u32);
            if speed2 < 0.0025f32 && spin2 < 0.0025f32 && corr2 < 0.000001f32 {
                calm = 1u32;
            }
            if calm != 0u32 {
                nquiet = quiet + quiet_step;
            } else {
                nquiet = 0.0f32;
            }
            if nquiet >= 30.0f32 {
                nawake = 0.0f32;
                nvx = 0.0f32;
                nvy = 0.0f32;
                nvz = 0.0f32;
                nwx = 0.0f32;
                nwy = 0.0f32;
                nwz = 0.0f32;
                nquiet = 30.0f32;
            }
        } else {
            if impulse2 > 0.0001f32 || corr2 > 0.000001f32 {
                nawake = 1.0f32;
                nquiet = 0.0f32;
            }
        }

        body_state[b] = npx;
        body_state[b + 1usize] = npy;
        body_state[b + 2usize] = npz;
        body_state[b + 8usize] = nvx;
        body_state[b + 9usize] = nvy;
        body_state[b + 10usize] = nvz;
        body_state[b + 12usize] = nwx;
        body_state[b + 13usize] = nwy;
        body_state[b + 14usize] = nwz;
        body_state[b + 28usize] = nquiet;
        body_state[b + 29usize] = nawake;

        let lminx = body_state[b + 20usize];
        let lminy = body_state[b + 21usize];
        let lminz = body_state[b + 22usize];
        let lmaxx = body_state[b + 24usize];
        let lmaxy = body_state[b + 25usize];
        let lmaxz = body_state[b + 26usize];
        write_desc(
            body_desc,
            i * 40usize,
            npx,
            npy,
            npz,
            qx,
            qy,
            qz,
            qw,
            lminx,
            lminy,
            lminz,
            lmaxx,
            lmaxy,
            lmaxz,
        );
    }
}

/// Marks one 8^3 brick bit in a bitset.
#[cube]
fn mark_brick(bits: &mut [Atomic<u32>], brick: u32) {
    bits[(brick >> 5u32) as usize].fetch_or(1u32 << (brick & 31u32));
}

/// Whether a world cell is inside any rigid body.
#[cube]
fn cell_inside_body(
    voxels: &[u32],
    body_state: &[f32],
    body_count: u32,
    wx: f32,
    wy: f32,
    wz: f32,
) -> u32 {
    let mut hit = seed_u32(body_count, 0u32);
    for body in 0..body_count as usize {
        let sb = body * 32usize;
        let (lx, ly, lz) = quat_rotate(
            -body_state[sb + 4usize],
            -body_state[sb + 5usize],
            -body_state[sb + 6usize],
            body_state[sb + 7usize],
            wx - body_state[sb],
            wy - body_state[sb + 1usize],
            wz - body_state[sb + 2usize],
        );
        let ix = (lx + 16.0f32).floor() as i32;
        let iy = (ly + 16.0f32).floor() as i32;
        let iz = (lz + 16.0f32).floor() as i32;
        if ix >= 0i32 && ix < 32i32 && iy >= 0i32 && iy < 32i32 && iz >= 0i32 && iz < 32i32 {
            let index = (iy as u32 * 1024u32) + (iz as u32 * 32u32) + ix as u32;
            if (voxels[body * 32768usize + index as usize] & 255u32) != 0u32 {
                hit = 1u32;
            }
        }
    }
    hit
}

/// Moves loose voxels out of the body volume, or deletes them.
#[cube(launch_unchecked)]
pub(super) fn displace_loose_kernel(
    voxel_a: &mut [Atomic<u32>],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    body_voxels: &[u32],
    body_state: &[f32],
    brick_occupied: &mut [Atomic<u32>],
    dirty_bricks: &mut [Atomic<u32>],
    body_count: u32,
) {
    let body = CUBE_POS_X as usize;
    if body < body_count as usize {
        let sb = body * 32usize;
        let vb = body * 32768usize;
        let px = body_state[sb];
        let py = body_state[sb + 1usize];
        let pz = body_state[sb + 2usize];
        let qx = body_state[sb + 4usize];
        let qy = body_state[sb + 5usize];
        let qz = body_state[sb + 6usize];
        let qw = body_state[sb + 7usize];
        let thread = UNIT_POS_X as usize;
        for slot in 0..128usize {
            let index = thread + slot * 256usize;
            if (body_voxels[vb + index] & 255u32) != 0u32 {
                let idx = index as u32;
                let lx = (idx & 31u32) as i32;
                let ly = ((idx >> 10u32) & 31u32) as i32;
                let lz = ((idx >> 5u32) & 31u32) as i32;
                let (ox, oy, oz) = quat_rotate(
                    qx,
                    qy,
                    qz,
                    qw,
                    lx as f32 - 15.5f32,
                    ly as f32 - 15.5f32,
                    lz as f32 - 15.5f32,
                );
                let wpx = px + ox;
                let wpy = py + oy;
                let wpz = pz + oz;
                let cx = wpx.floor() as i32;
                let cy = wpy.floor() as i32;
                let cz = wpz.floor() as i32;
                if cx >= 0i32
                    && cx < 256i32
                    && cy >= 0i32
                    && cy < 256i32
                    && cz >= 0i32
                    && cz < 256i32
                {
                    let src = ((cy as u32 * 65536u32) + (cz as u32 * 256u32) + cx as u32) as usize;
                    let state = voxel_a[src].load();
                    if is_loose_ty(state & 255u32) != 0u32 {
                        let mut moved = seed_u32(state, 0u32);
                        let mut dest = seed_u32(state, 0u32);
                        for dir in 0..6usize {
                            let mut oxi = 0i32;
                            let mut oyi = 0i32;
                            let mut ozi = 0i32;
                            if dir == 0usize {
                                oxi = 1i32;
                            } else if dir == 1usize {
                                oxi = -1i32;
                            } else if dir == 2usize {
                                ozi = 1i32;
                            } else if dir == 3usize {
                                ozi = -1i32;
                            } else if dir == 4usize {
                                oyi = 1i32;
                            } else {
                                oyi = -1i32;
                            }
                            let tx = cx + oxi;
                            let ty = cy + oyi;
                            let tz = cz + ozi;
                            if moved == 0u32
                                && tx >= 0i32
                                && tx < 256i32
                                && ty >= 0i32
                                && ty < 256i32
                                && tz >= 0i32
                                && tz < 256i32
                            {
                                let tc = ((ty as u32 * 65536u32) + (tz as u32 * 256u32) + tx as u32)
                                    as usize;
                                if voxel_a[tc].load() == 0u32
                                    && cell_inside_body(
                                        body_voxels,
                                        body_state,
                                        body_count,
                                        tx as f32 + 0.5f32,
                                        ty as f32 + 0.5f32,
                                        tz as f32 + 0.5f32,
                                    ) == 0u32
                                {
                                    if voxel_a[tc].compare_exchange_weak(0u32, state) == 0u32 {
                                        moved = 1u32;
                                        dest = tc as u32;
                                    }
                                }
                            }
                        }
                        voxel_a[src].exchange(0u32);
                        if moved != 0u32 {
                            let aux_dest = dest as usize;
                            aux_ctype[aux_dest] = aux_ctype[src];
                            aux_tmp[aux_dest] = aux_tmp[src];
                            aux_tmp2[aux_dest] = aux_tmp2[src];
                            aux_tmp3[aux_dest] = aux_tmp3[src];
                        }
                        aux_ctype[src] = 0u32;
                        aux_tmp[src] = 0i32;
                        aux_tmp2[src] = 0i32;
                        aux_tmp3[src] = 0i32;
                        if moved != 0u32 {
                            let dx = dest & 255u32;
                            let dz = (dest >> 8u32) & 255u32;
                            let dy = (dest >> 16u32) & 255u32;
                            let brick =
                                ((dy >> 3u32) << 10u32) | ((dz >> 3u32) << 5u32) | (dx >> 3u32);
                            mark_brick(brick_occupied, brick);
                            mark_brick(dirty_bricks, brick);
                        }
                        let sx = cx as u32;
                        let sz = cz as u32;
                        let sy = cy as u32;
                        let brick = ((sy >> 3u32) << 10u32) | ((sz >> 3u32) << 5u32) | (sx >> 3u32);
                        mark_brick(dirty_bricks, brick);
                    }
                }
            }
        }
    }
}

/// Publishes every body cell into the world claim buffer.
#[cube(launch_unchecked)]
pub(super) fn body_block_claim_kernel(
    body_voxels: &[u32],
    body_state: &[f32],
    claim: &mut [Atomic<u32>],
    body_count: u32,
    blocked_key: u32,
) {
    let body = CUBE_POS_X as usize;
    if body < body_count as usize {
        let sb = body * 32usize;
        let vb = body * 32768usize;
        let px = body_state[sb];
        let py = body_state[sb + 1usize];
        let pz = body_state[sb + 2usize];
        let qx = body_state[sb + 4usize];
        let qy = body_state[sb + 5usize];
        let qz = body_state[sb + 6usize];
        let qw = body_state[sb + 7usize];
        let thread = UNIT_POS_X as usize;
        for slot in 0..128usize {
            let index = thread + slot * 256usize;
            if (body_voxels[vb + index] & 255u32) != 0u32 {
                let idx = index as u32;
                let lx = (idx & 31u32) as i32;
                let ly = ((idx >> 10u32) & 31u32) as i32;
                let lz = ((idx >> 5u32) & 31u32) as i32;
                let (ox, oy, oz) = quat_rotate(
                    qx,
                    qy,
                    qz,
                    qw,
                    lx as f32 - 15.5f32,
                    ly as f32 - 15.5f32,
                    lz as f32 - 15.5f32,
                );
                let cx = (px + ox).floor() as i32;
                let cy = (py + oy).floor() as i32;
                let cz = (pz + oz).floor() as i32;
                if cx >= 0i32
                    && cx < 256i32
                    && cy >= 0i32
                    && cy < 256i32
                    && cz >= 0i32
                    && cz < 256i32
                {
                    let cell = ((cy as u32 * 65536u32) + (cz as u32 * 256u32) + cx as u32) as usize;
                    claim[cell].fetch_max(blocked_key);
                }
            }
        }
    }
}
