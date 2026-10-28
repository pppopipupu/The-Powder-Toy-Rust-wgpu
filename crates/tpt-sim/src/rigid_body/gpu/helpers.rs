//! Device helpers (kept local so the kernel file compiles on its own).

use cubecl::prelude::*;

/// Reads a `f32` from a packed `u32` word.
#[cube]
pub(super) fn fword(words: &[u32], index: usize) -> f32 {
    f32::reinterpret(words[index])
}

/// Mutable-slice counterpart of [`fword`].
#[cube]
pub(super) fn fword_m(words: &mut [u32], index: usize) -> f32 {
    f32::reinterpret(words[index])
}

/// Reads a `f32` out of an atomically-updated body record.
#[cube]
pub(super) fn fword_a(words: &mut [Atomic<u32>], index: usize) -> f32 {
    f32::reinterpret(words[index].load())
}

/// Runtime-seeded zero.
///
/// A mutable local seeded with a literal cannot later be assigned a runtime
/// expression (AGENTS "Runtime-Seeded Zeros"), so zeros that may be
/// overwritten start from a runtime value instead.
#[cube]
#[allow(clippy::erasing_op)]
pub(super) fn zero_of(x: f32) -> f32 {
    x * 0.0f32
}

/// Runtime-seeded `u32`: the integer counterpart of [`zero_of`].
#[cube]
#[allow(clippy::erasing_op)]
pub(super) fn seed_u32(anchor: u32, value: u32) -> u32 {
    anchor * 0u32 + value
}

/// Runtime-seeded `f32` with an arbitrary seed value.
#[cube]
#[allow(clippy::erasing_op)]
pub(super) fn seed_f32(anchor: f32, value: f32) -> f32 {
    anchor * 0.0f32 + value
}

/// Support-direction code of a unit normal (`0` none, `1=+x, 2=-x, 3=+y, 4=-y`).
#[cube]
pub(super) fn support_code(nx: f32, ny: f32) -> u32 {
    let mut code = 0u32;
    if nx > 0.5f32 {
        code = 1u32;
    } else if nx < -0.5f32 {
        code = 2u32;
    } else if ny > 0.5f32 {
        code = 3u32;
    } else if ny < -0.5f32 {
        code = 4u32;
    }
    code
}

/// Unit normal of a [`support_code`] (`(0, 0)` for none).
#[cube]
pub(super) fn support_normal(code: u32) -> (f32, f32) {
    let mut nx = 0.0f32;
    let mut ny = 0.0f32;
    if code == 1u32 {
        nx = 1.0f32;
    } else if code == 2u32 {
        nx = -1.0f32;
    } else if code == 3u32 {
        ny = 1.0f32;
    } else if code == 4u32 {
        ny = -1.0f32;
    }
    (nx, ny)
}

/// Unit normal of a manifold direction word (`0=+x, 1=-x, 2=+y, 3=-y`).
#[cube]
pub(super) fn dir_normal(dir: u32) -> (f32, f32) {
    let mut nx = 0.0f32;
    let mut ny = 0.0f32;
    if dir == 0u32 {
        nx = 1.0f32;
    } else if dir == 1u32 {
        nx = -1.0f32;
    } else if dir == 2u32 {
        ny = 1.0f32;
    } else if dir == 3u32 {
        ny = -1.0f32;
    }
    (nx, ny)
}

/// Packed map word of `id` / `ty` (`pmap_pack`).
#[cube]
pub(super) fn pack_map(id: u32, ty: u32) -> u32 {
    ((id + 1u32) << 9u32) | (ty & 0x1FFu32)
}

/// Particle slot stored in a packed map word (`u32::MAX` for an empty word).
#[cube]
pub(super) fn unpack_slot(packed: u32) -> u32 {
    let raw = packed >> 9u32;
    let mut slot = 4294967295u32;
    if raw > 0u32 {
        slot = raw - 1u32;
    }
    slot
}

#[cube]
pub(super) fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut out = v;
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

#[cube]
pub(super) fn maxf(a: f32, b: f32) -> f32 {
    let mut out = a;
    if b > out {
        out = b;
    }
    out
}

#[cube]
pub(super) fn minf(a: f32, b: f32) -> f32 {
    let mut out = a;
    if b < out {
        out = b;
    }
    out
}

#[cube]
pub(super) fn fabs(v: f32) -> f32 {
    let mut out = v;
    if out < 0.0f32 {
        out = -out;
    }
    out
}

#[cube]
pub(super) fn iabs(v: i32) -> i32 {
    let mut out = v;
    if out < 0i32 {
        out = -out;
    }
    out
}

/// Keeps an angle inside `[0, TAU)`.
#[cube]
pub(super) fn wrap_angle(a: f32) -> f32 {
    let tau = 6.283_185_5f32;
    let mut out = a;
    if out >= tau {
        out -= tau;
    }
    if out >= tau {
        out -= tau;
    }
    if out < 0.0f32 {
        out += tau;
    }
    if out < 0.0f32 {
        out += tau;
    }
    out
}

/// Element weight, floored at 10 like the host solver.
#[cube]
pub(super) fn weight_of(elem_props: &[u32], ty: u32) -> f32 {
    let mut w = i32::reinterpret(elem_props[ty as usize * 32usize + 2usize]) as f32;
    if w < 10.0f32 {
        w = 10.0f32;
    }
    w
}

#[cube]
pub(super) fn props_of(elem_props: &[u32], ty: u32) -> u32 {
    elem_props[ty as usize * 32usize + 1usize]
}

#[cube]
pub(super) fn air_drag_of(elem_props: &[u32], ty: u32) -> f32 {
    clampf(
        f32::reinterpret(elem_props[ty as usize * 32usize + 5usize]),
        0.0f32,
        2.0f32,
    )
}

/// `is_wall_blocking_particle` for a coarse wall cell (`CELL == 4`).
#[cube]
pub(super) fn wall_blocks(
    wall_bmap: &[u32],
    wall_emap: &[u32],
    gx: i32,
    gy: i32,
    xcells: i32,
    ycells: i32,
    props: u32,
) -> bool {
    let mut blocked = false;
    if xcells > 0i32 && ycells > 0i32 {
        let mut cx = gx / 4i32;
        let mut cy = gy / 4i32;
        if cx < 0i32 {
            cx = 0i32;
        }
        if cy < 0i32 {
            cy = 0i32;
        }
        if cx > xcells - 1i32 {
            cx = xcells - 1i32;
        }
        if cy > ycells - 1i32 {
            cy = ycells - 1i32;
        }
        let cell = (cy * xcells + cx) as usize;
        let wall = wall_bmap[cell] & 0xFFu32;
        let emap = wall_emap[cell] & 0xFFu32;
        if wall == 1u32 || wall == 8u32 || wall == 9u32 || wall == 7u32 {
            // WL_WALLELEC / WL_WALL / WL_ALLOWAIR / WL_DESTROYALL.
            blocked = true;
        } else if wall == 6u32 {
            // WL_ALLOWLIQUID
            blocked = (props & 2u32) == 0u32;
        } else if wall == 10u32 {
            // WL_ALLOWPOWDER
            blocked = (props & 1u32) == 0u32;
        } else if wall == 13u32 {
            // WL_ALLOWGAS
            blocked = (props & 8u32) == 0u32;
        } else if wall == 15u32 {
            // WL_ALLOWENERGY
            blocked = (props & 16u32) == 0u32;
        } else if wall == 2u32 {
            // WL_EWALL blocks while unpowered.
            blocked = emap == 0u32;
        }
    }
    blocked
}

/// Contact face and penetration of a swept pixel against an axis-aligned
/// blocked box.
///
/// `(prev_x, prev_y)` is the pixel centre at the start of the sub-step and
/// `(cur_x, cur_y)` its destination; `lo`/`hi` are the box bounds in
/// pixel-centre coordinates. Returns `(nx, ny, pen)` where the normal points
/// out of the box along the face the pixel crossed and `pen` is how far the
/// destination centre has gone past that face.
///
/// The entry face is the *last* face the segment crosses, not the axis of the
/// largest separation. That distinction is what stops a pixel that has already
/// passed an obstacle's centre from being pushed out of its far side (which
/// used to shove a fast body straight through a resting one).
#[cube]
pub(super) fn swept_face(
    prev_x: f32,
    prev_y: f32,
    cur_x: f32,
    cur_y: f32,
    lo_x: f32,
    lo_y: f32,
    hi_x: f32,
    hi_y: f32,
) -> (f32, f32, f32) {
    let dx = cur_x - prev_x;
    let dy = cur_y - prev_y;

    // Which faces does the segment cross? A pixel can start inside a blocked
    // box (a body spawned inside terrain), in which case no face is crossed.
    let cross_lo_x = prev_x <= lo_x && cur_x > lo_x && dx > 1e-6f32;
    let cross_hi_x = prev_x >= hi_x && cur_x < hi_x && dx < -1e-6f32;
    let cross_x = cross_lo_x || cross_hi_x;
    let cross_lo_y = prev_y <= lo_y && cur_y > lo_y && dy > 1e-6f32;
    let cross_hi_y = prev_y >= hi_y && cur_y < hi_y && dy < -1e-6f32;
    let cross_y = cross_lo_y || cross_hi_y;

    let mut out_nx = 0.0f32;
    let mut out_ny = 0.0f32;
    if cross_x && cross_y {
        // Both faces crossed: the *later* crossing is the face the pixel
        // actually enters the box through. The low-face expression is the seed
        // and the high-face one overwrites it; CubeCL rejects an assignment of
        // one mutable local to another, so the sign is re-derived inline.
        let mut tx = (lo_x - prev_x) / dx;
        if cross_hi_x {
            tx = (prev_x - hi_x) / (0.0f32 - dx);
        }
        let mut ty = (lo_y - prev_y) / dy;
        if cross_hi_y {
            ty = (prev_y - hi_y) / (0.0f32 - dy);
        }
        if tx >= ty {
            if cross_hi_x {
                out_nx = 1.0f32;
            } else {
                out_nx = -1.0f32;
            }
        } else if cross_hi_y {
            out_ny = 1.0f32;
        } else {
            out_ny = -1.0f32;
        }
    } else if cross_x {
        if cross_lo_x {
            out_nx = -1.0f32;
        } else {
            out_nx = 1.0f32;
        }
    } else if cross_y {
        if cross_lo_y {
            out_ny = -1.0f32;
        } else {
            out_ny = 1.0f32;
        }
    } else {
        // Already inside the box at the start of the sub-step (a capped
        // correction or a body spawned inside terrain): back out against the
        // motion, or toward the nearest face when the pixel is at rest.
        let adx = fabs(dx);
        let ady = fabs(dy);
        if adx > 1e-5f32 || ady > 1e-5f32 {
            if adx >= ady {
                if dx > 0.0f32 {
                    out_nx = -1.0f32;
                } else {
                    out_nx = 1.0f32;
                }
            } else if dy > 0.0f32 {
                out_ny = -1.0f32;
            } else {
                out_ny = 1.0f32;
            }
        } else {
            let d_lo_x = cur_x - lo_x;
            let d_hi_x = hi_x - cur_x;
            let d_lo_y = cur_y - lo_y;
            let d_hi_y = hi_y - cur_y;
            if d_lo_x <= d_hi_x && d_lo_x <= d_lo_y && d_lo_x <= d_hi_y {
                out_nx = -1.0f32;
            } else if d_hi_x <= d_lo_y && d_hi_x <= d_hi_y {
                out_nx = 1.0f32;
            } else if d_lo_y <= d_hi_y {
                out_ny = -1.0f32;
            } else {
                out_ny = 1.0f32;
            }
        }
    }

    let mut pen = cur_x - lo_x;
    if out_nx > 0.0f32 {
        pen = hi_x - cur_x;
    } else if out_nx == 0.0f32 {
        pen = cur_y - lo_y;
        if out_ny > 0.0f32 {
            pen = hi_y - cur_y;
        } else if out_ny == 0.0f32 {
            pen = 0.0f32;
        }
    }
    (out_nx, out_ny, clampf(pen, 0.0f32, 1.0f32))
}

/// Clamp the linear / angular velocity to the solver limits.
#[cube]
pub(super) fn clamp_velocity(mut vx: f32, mut vy: f32, mut omega: f32) -> (f32, f32, f32) {
    let speed = (vx * vx + vy * vy).sqrt();
    if speed > 16.0f32 {
        let scale = 16.0f32 / speed;
        vx *= scale;
        vy *= scale;
    }
    omega = clampf(omega, -0.75f32, 0.75f32);
    (vx, vy, omega)
}

/// Clamp the *surface* speed (translation plus rotation at `radius`) so a
/// pixel can never travel more than one cell between two contact passes.
///
/// `budget` is the per-frame surface speed the sub-step count was sized for;
/// scaling the angular velocity too keeps the invariant for the outermost
/// pixel of a large spinning cluster, whose rim moves `|omega| * radius`.
#[cube]
pub(super) fn clamp_surface_velocity(
    mut vx: f32,
    mut vy: f32,
    mut omega: f32,
    radius: f32,
    budget: f32,
) -> (f32, f32, f32) {
    let surface = (vx * vx + vy * vy).sqrt() + fabs(omega) * radius;
    if surface > budget && surface > 1e-6f32 {
        let scale = budget / surface;
        vx *= scale;
        vy *= scale;
        omega *= scale;
    }
    (vx, vy, omega)
}

/// Normal + friction impulse at a contact manifold, for one or two bodies.
///
/// `(rxa, rya)` is the sender's lever arm and `(rbx, rby)` the receiver's, both
/// about their own centres of mass; `(ovx, ovy)` is the receiver's surface
/// velocity at the contact (zero for the static world). Passing the receiver's
/// inverse mass / inertia makes the pair response symmetric: momentum flows
/// through `denom` in both directions instead of the sender responding to an
/// immovable surface.
///
/// The positional part of the contact is handled by the caller (it takes the
/// deepest penetration per axis over the whole body, which keeps a flat
/// landing symmetric instead of tipping the body toward its first pixel).
///
/// Returns the updated sender state plus the total impulse `(jx, jy)` applied
/// to it, so the caller can queue exactly `-J` to the other body.
#[cube]
pub(super) fn resolve_pair_impulse(
    mut vx: f32,
    mut vy: f32,
    mut omega: f32,
    inv_mass_a: f32,
    inv_inertia_a: f32,
    inv_mass_b: f32,
    inv_inertia_b: f32,
    rxa: f32,
    rya: f32,
    rbx: f32,
    rby: f32,
    nx: f32,
    ny: f32,
    ovx: f32,
    ovy: f32,
) -> (f32, f32, f32, f32, f32) {
    let vpx = vx - omega * rya;
    let vpy = vy + omega * rxa;
    let rvx = vpx - ovx;
    let rvy = vpy - ovy;
    let vn = rvx * nx + rvy * ny;
    let rna = rxa * ny - rya * nx;
    let rnb = rbx * ny - rby * nx;
    let denom = maxf(
        inv_mass_a + inv_mass_b + inv_inertia_a * rna * rna + inv_inertia_b * rnb * rnb,
        1e-9f32,
    );
    // Restitution only above the resting threshold: a body settling on a
    // surface must not micro-bounce for ever. Pressing contacts only
    // (`vn < 0`); a separating contact produces no impulse, which is
    // expressed as a clamped normal speed instead of a branch so the
    // friction below is still computed once.
    let mut restitution = 1.0f32;
    if vn < -0.5f32 {
        restitution = 1.25f32;
    }
    let press = minf(vn, 0.0f32);
    let j = -restitution * press / denom;

    // Coulomb friction along the contact tangent of the *relative* motion.
    let vtx = rvx - vn * nx;
    let vty = rvy - vn * ny;
    let vt_len = (vtx * vtx + vty * vty).sqrt();
    let tx = vtx / maxf(vt_len, 1e-5f32);
    let ty = vty / maxf(vt_len, 1e-5f32);
    let rta = rxa * ty - rya * tx;
    let rtb = rbx * ty - rby * tx;
    let denom_t = maxf(
        inv_mass_a + inv_mass_b + inv_inertia_a * rta * rta + inv_inertia_b * rtb * rtb,
        1e-9f32,
    );
    let max_friction = 0.4f32 * j;
    let jt = clampf(-vt_len / denom_t, -max_friction, max_friction);

    let jx = j * nx + jt * tx;
    let jy = j * ny + jt * ty;
    vx += jx * inv_mass_a;
    vy += jy * inv_mass_a;
    omega += (rxa * jy - rya * jx) * inv_inertia_a;
    let clamped = clamp_velocity(vx, vy, omega);
    (clamped.0, clamped.1, clamped.2, jx, jy)
}
