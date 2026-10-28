//! Velocity integration and per-`Falldown` destination selection.
//!
//! One thread per particle slot. The kernel body is a thin driver around the
//! stage helpers below, so the physics reads as the same ordered sequence as
//! `SimulationImpl::UpdateParticles` (`Simulation.cpp:2295-2412`):
//!
//! * [`pre_move_checks`] — off-screen / killing-wall / stasis / `WL_DETECT`
//!   gates (`Simulation.cpp:2312-2338`).
//! * [`base_gravity`] — the configured `gravityMode` field
//!   (`GetGravityField`, `Simulation.cpp:2088-2125`).
//! * [`integrate_velocity`] — `Loss` / `Advection` / gravity / `Diffusion`
//!   (`Simulation.cpp:2361-2377`).
//! * [`clamp_plan_velocity`] — the vector `MAX_VELOCITY` scale that `PlanMove`
//!   performs before sampling (`Simulation.cpp:2170-2175`).
//! * [`choose_destination`] — dispatch on `Falldown` to
//!   [`gas_destination`] (`Simulation.cpp:3138-3152`) or the shared
//!   liquid/powder branch in [`super::liquid::fall_down_destination`]
//!   (`Simulation.cpp:3138-3346`).
//! * Arbitration — publish the target and `fetch_max` the priority key.
//!
//! The 16 storage-buffer binding order of
//! `velocity_and_intent_kernel` is fixed: a movement kernel may never exceed
//! the 16-buffer budget, and the host (`step_movement_gpu`) passes exactly the
//! buffers declared here, in this order.
//!
//! # Deviations forced by the device-resident design
//!
//! * TPT's `UpdateParticles` is a sequential per-particle loop; this port
//!   reads the pre-move snapshot and arbitrates with `fetch_max`. Earlier
//!   particles' new positions/types are therefore not visible to later ones in
//!   the same tick.
//! * Newtonian gravity is applied by [`crate::gravity_kernel`] before this
//!   kernel, so the per-particle `pGrav` here is only the `gravityMode` term.
//!   The gravity kernel applies `Loss` for the same reason (`GetGravityField`'s
//!   Newtonian term is added after `vx *= Loss`, `Simulation.cpp:2361-2384`).
//! * `WL_DETECT`'s `set_emap` trigger (`Simulation.cpp:2330-2331`) runs in
//!   `air_deposit_kernel`'s companion `detect_charge_kernel`, which fills the
//!   horizontal span and marks its ends; `emap_closure_kernel` then finishes
//!   the reference's vertical up/down recursion.
//! * `set_emap`'s per-particle `HotAir` scatter is performed by
//!   `air_kernel::hot_air_pressure_kernel` before movement.

use cubecl::prelude::*;

use tpt_core::constants::{PT_NUM, PT_PHOT};

use super::liquid::fall_down_destination;
use super::{
    DIE_INTENT, DO_MOVE_BLOCKED, DO_MOVE_DIE, DO_MOVE_OK, INTENT_HALF_MIRROR, INTENT_PIXEL_MASK,
    do_move_at, mix_index, pcg_hash, saw_wood_at, wall_kills_particle,
};

/// Runtime-seeded `i32`, see `AGENTS.md` rule 4.
///
/// A mutable scalar seeded from a folded constant (a negative literal, a cast
/// of one) stays a host `i32` in the expanded body, so a later runtime
/// assignment would not type-check. Routing the seed through this call keeps
/// the local a `NativeExpand<i32>`.
#[cube]
#[allow(clippy::erasing_op)]
fn seed_i32(anchor: i32, value: i32) -> i32 {
    anchor * 0i32 + value
}

/// `PT_NUM` as a `u32` for kernel-side range checks.
///
/// Kernel bodies cannot reference the host constant directly, so it is mirrored
/// here (same convention as `thermal_kernel`); every element-id guard below
/// uses this instead of a `64` literal.
const PT_NUM_U32: u32 = PT_NUM as u32;

/// `PT_PHOT` as a `u32`: the reflection rule at `Simulation.cpp:3044` filters a
/// bouncing photon's wavelength by the mask of the pixel it hit.
const PT_PHOT_U32: u32 = PT_PHOT;
const _: () = assert!(PT_PHOT == 29);

/// `MAX_VELOCITY` (`SimulationConfig.h:34`). `PlanMove` scales the velocity
/// vector so its largest component is at most this value
/// (`Simulation.cpp:2170-2175`); the port used to clamp each axis to `16`.
const MAX_VELOCITY: f32 = 1.0e4f32;

/// `ISTP = CELL / 2` (`SimulationConfig.h:31`), used by the gas bounce clamp
/// (`Simulation.cpp:3110-3113`).
const ISTP: f32 = 2.0f32;

/// `-1`, `0` or `+1`: the per-axis pixel step of one frame.
///
/// TPT stores the float target in `parts[i].x` (`Simulation::move`,
/// Simulation.cpp:1541), so a velocity below half a pixel still moves the
/// particle on `|v|` of the frames: `WTRV`'s `0.75` diffusion kick and
/// `FIRE`'s `-0.1` gravity both drift well below the half-pixel threshold.
/// The port keeps integer positions, so the sub-pixel remainder is drawn from
/// the frame hash instead of carried in the coordinate. The average speed is
/// then the same `|v|` pixels per frame, and the phase is random instead of
/// deterministic - invisible at the tick rate, but the rate itself is what
/// decides how fast a gas cloud spreads.
#[cube]
fn step_axis(v: f32, h: u32) -> i32 {
    let a = if v < 0.0f32 { -v } else { v };
    let r = ((h & 0xFFFFu32) as f32) * (1.0f32 / 65535.0f32);
    let mut s = 0i32;
    if a >= 1.0f32 || r < a {
        if v > 0.0f32 {
            s = 1i32;
        } else if v < 0.0f32 {
            s = -1i32;
        }
    }
    s
}

/// Element gravity for the configured [`tpt_core::particle::GravityMode`].
///
/// `0` vertical (+y), `2` radial, `3` custom direction scaled by the element
/// gravity; mode `1` (off) returns `(0, 0)`.
///
/// The radial branch is `GetGravityField`'s `GRAV_RADIAL`
/// (`Simulation.cpp:2099-2114`) verbatim:
///
/// ```text
/// auto dx = float(x - XCNTR);
/// auto dy = float(y - YCNTR);
/// if (dx || dy) {
///     auto pGravD = 0.01f - hypotf(dx, dy);
///     pGravX = particleGrav * (dx / pGravD);
///     pGravY = particleGrav * (dy / pGravD);
/// }
/// ```
///
/// `XCNTR = XRES / 2` (`SimulationConfig.h:22`), integer division.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn base_gravity(
    gravity_mode: u32,
    gravity: f32,
    xres: u32,
    yres: u32,
    px: f32,
    py: f32,
    custom_gravity_x: f32,
    custom_gravity_y: f32,
) -> (f32, f32) {
    let mut grav_x = 0.0f32;
    let mut grav_y = 0.0f32;
    if gravity_mode == 0u32 {
        // `GRAV_VERTICAL`: pull towards +y (the bottom of the canvas).
        grav_y = gravity;
    } else if gravity_mode == 2u32 {
        // `GRAV_RADIAL` (`Simulation.cpp:2099-2114`).
        let dx = px - ((xres / 2u32) as f32);
        let dy = py - ((yres / 2u32) as f32);
        if dx != 0.0f32 || dy != 0.0f32 {
            let d = 0.01f32 - (dx * dx + dy * dy).sqrt();
            grav_x = gravity * (dx / d);
            grav_y = gravity * (dy / d);
        }
    } else if gravity_mode == 3u32 {
        // `GRAV_CUSTOM` (`Simulation.cpp:2115-2118`).
        grav_x = gravity * custom_gravity_x;
        grav_y = gravity * custom_gravity_y;
    }
    (grav_x, grav_y)
}

/// `vx = vx * loss + advection * air_vx + grav_x + diffusion * rand_dx`
/// (and the same for `vy`), with a deterministic per-frame hash for the
/// diffusion kick so a replay of the same frame produces the same motion.
///
/// `Simulation.cpp:2361-2377` performs no clamp here: the only velocity limit
/// is `PlanMove`'s vector `MAX_VELOCITY` scale
/// ([`clamp_plan_velocity`]).
#[cube]
#[allow(clippy::too_many_arguments)]
fn integrate_velocity(
    part_vx: f32,
    part_vy: f32,
    loss: f32,
    advection: f32,
    ax: f32,
    ay: f32,
    grav_x: f32,
    grav_y: f32,
    diffusion: f32,
    h1: u32,
    h2: u32,
) -> (f32, f32) {
    let rand_dx = ((h1 & 0xFFFFu32) as f32) * (2.0f32 / 65535.0f32) - 1.0f32;
    let rand_dy = ((h2 & 0xFFFFu32) as f32) * (2.0f32 / 65535.0f32) - 1.0f32;

    let vx = part_vx * loss + advection * ax + grav_x + diffusion * rand_dx;
    let vy = part_vy * loss + advection * ay + grav_y + diffusion * rand_dy;
    (vx, vy)
}

/// `PlanMove`'s velocity clamp (`Simulation.cpp:2170-2175`):
///
/// ```text
/// if (mv > MAX_VELOCITY) { vx *= MAX_VELOCITY/mv; vy *= MAX_VELOCITY/mv; }
/// ```
///
/// The vector is scaled, not each axis clamped: a diagonal at `(3e4, 4e4)`
/// becomes `(6e3, 8e3)` (magnitude `1e4`), while a per-axis clamp would keep
/// the direction wrong. `MAX_VELOCITY` is `1e4`, not the `16` the port used.
#[cube]
pub(super) fn clamp_plan_velocity(vx: f32, vy: f32) -> (f32, f32) {
    let mut out_x = vx;
    let mut out_y = vy;
    let mut mv = vx.abs();
    if vy.abs() > mv {
        mv = vy.abs();
    }
    if mv > MAX_VELOCITY {
        out_x = vx * (MAX_VELOCITY / mv);
        out_y = vy * (MAX_VELOCITY / mv);
    }
    (out_x, out_y)
}

/// `Simulation.cpp:2312-2338`: the gates the reference applies before touching
/// the particle.
///
/// Returns `0` to skip, `1` to kill and `2` to simulate. `x` / `y` are the
/// rounded pixel; `cell` the coarse air cell; `props` the element properties.
///
/// * off-screen kill (`x<CELL || y<CELL || x>=XRES-CELL || y>=YRES-CELL`),
/// * wall kill (explicit list, `STKM`/`STKM2`/`FIGH` exempt),
/// * unpowered `WL_STASIS` skip.
///
/// `WL_DETECT` charging (`Simulation.cpp:2330-2331`) is not part of this
/// predicate: it is a wall-map side effect and runs in `detect_charge_kernel`
/// (plus `emap_closure_kernel` for the vertical span) before the velocity
/// pass.
#[cube]
#[allow(clippy::too_many_arguments)]
fn pre_move_checks(
    ty: u32,
    props: u32,
    xi: i32,
    yi: i32,
    xres: u32,
    yres: u32,
    wall: u32,
    emap: u32,
) -> u32 {
    let mut action = 2u32;
    let mut in_margin = true;
    if xi < 4i32 || yi < 4i32 || xi >= xres as i32 - 4i32 || yi >= yres as i32 - 4i32 {
        in_margin = false;
    }
    // The stickman heads pass through everything and are exempt from the wall
    // kill (`Simulation.cpp:2314-2330`), but not from the off-screen kill.
    let stickman = ty == PT_STKM_U32 || ty == PT_STKM2_U32 || ty == PT_FIGH_U32;
    if !in_margin || (!stickman && wall_kills_particle(wall, emap, props)) {
        action = 1u32;
    } else if wall == 18u32 && (emap & 0xFFu32) < 8u32 {
        // `WL_STASIS` with `emap < 8` freezes the particle entirely.
        action = 0u32;
    }
    action
}

/// `true` when `(x, y)` holds `PT_GLAS` or `PT_BGLA`.
///
/// `Simulation::is_blocking` with `REFRACT` set (`Simulation.cpp:1626-1635`):
/// only glass counts as a surface for the refraction normal.
#[cube]
fn is_refract_blocking(pmap: &[u32], x: i32, y: i32, xr: i32, yr: i32) -> bool {
    let mut blocking = false;
    if x >= 0i32 && y >= 0i32 && x < xr && y < yr {
        let ty = pmap[(y * xr + x) as usize] & 0x1FFu32;
        if ty == PT_GLAS_U32 || ty == PT_BGLA_U32 {
            blocking = true;
        }
    }
    blocking
}

/// `Simulation::is_boundary` for a `REFRACT` probe (`Simulation.cpp:1637-1644`):
/// a glass pixel that is not fully surrounded by glass in the 4-neighbourhood.
#[cube]
fn is_refract_boundary(pmap: &[u32], x: i32, y: i32, xr: i32, yr: i32) -> bool {
    let mut boundary = false;
    if is_refract_blocking(pmap, x, y, xr, yr) {
        let surrounded = is_refract_blocking(pmap, x, y - 1, xr, yr)
            && is_refract_blocking(pmap, x, y + 1, xr, yr)
            && is_refract_blocking(pmap, x - 1, y, xr, yr)
            && is_refract_blocking(pmap, x + 1, y, xr, yr);
        if !surrounded {
            boundary = true;
        }
    }
    boundary
}

/// `dx[8]` / `dy[8]` of `Simulation::find_next_boundary`
/// (`Simulation.cpp:1646-1648`).
#[cube]
fn boundary_dir_x(i: u32) -> i32 {
    let mut v = 0i32;
    if i == 0u32 || i == 1u32 || i == 7u32 {
        v = 1i32;
    } else if i == 3u32 || i == 4u32 || i == 5u32 {
        v = -1i32;
    }
    v
}

/// Y component of the 8-direction table.
#[cube]
fn boundary_dir_y(i: u32) -> i32 {
    let mut v = 0i32;
    if i == 1u32 || i == 2u32 || i == 3u32 {
        v = 1i32;
    } else if i == 5u32 || i == 6u32 || i == 7u32 {
        v = -1i32;
    }
    v
}

/// `de[8]` of `Simulation::find_next_boundary` (`Simulation.cpp:1649`).
#[cube]
fn boundary_dir_mask(i: i32) -> u32 {
    let mut v = 0u32;
    if i == 0i32 {
        v = 0x83u32;
    } else if i == 1i32 {
        v = 0x07u32;
    } else if i == 2i32 {
        v = 0x0Eu32;
    } else if i == 3i32 {
        v = 0x1Cu32;
    } else if i == 4i32 {
        v = 0x38u32;
    } else if i == 5i32 {
        v = 0x70u32;
    } else if i == 6i32 {
        v = 0xE0u32;
    } else if i == 7i32 {
        v = 0xC1u32;
    }
    v
}

/// `Simulation::find_next_boundary` (`Simulation.cpp:1646-1672`).
///
/// Returns `(found, x, y, em)`; the caller rebinds its walkers from the tuple.
#[cube]
#[allow(clippy::too_many_arguments)]
fn find_next_boundary(
    x_in: i32,
    y_in: i32,
    dm_in: u32,
    em_in: i32,
    reverse: bool,
    pmap: &[u32],
    xr: i32,
    yr: i32,
) -> (u32, i32, i32, i32) {
    let mut found = 0u32;
    let mut out_x = x_in;
    let mut out_y = y_in;
    let mut out_em = em_in;
    if x_in > 0i32 && x_in < xr - 1i32 && y_in > 0i32 && y_in < yr - 1i32 {
        let mut dm = dm_in;
        if em_in != -1i32 {
            dm &= boundary_dir_mask(em_in);
        }
        let mut mask = 0u32;
        for i in 0..8u32 {
            if (dm & (1u32 << i)) != 0u32
                && is_refract_blocking(
                    pmap,
                    x_in + boundary_dir_x(i),
                    y_in + boundary_dir_y(i),
                    xr,
                    yr,
                )
            {
                mask |= 1u32 << i;
            }
        }
        let mut done = false;
        for i in 0..8u32 {
            if !done {
                // `(i + (reverse ? 1 : -1)) & 7`.
                let step = if reverse { 1u32 } else { 7u32 };
                let n = (i + step) & 7u32;
                if (mask & (1u32 << i)) != 0u32 && (mask & (1u32 << n)) == 0u32 {
                    out_x = x_in + boundary_dir_x(i);
                    out_y = y_in + boundary_dir_y(i);
                    out_em = i as i32;
                    found = 1u32;
                    done = true;
                }
            }
        }
    }
    (found, out_x, out_y, out_em)
}

/// `Simulation::get_normal` for a `REFRACT` probe (`Simulation.cpp:1674-1718`).
///
/// Returns `(success, nx, ny)`. The reference also reports the two boundary
/// endpoints, but only `nx`/`ny` reach the caller.
#[cube]
fn get_normal(
    x: i32,
    y: i32,
    dx: f32,
    dy: f32,
    pmap: &[u32],
    xr: i32,
    yr: i32,
) -> (bool, f32, f32) {
    let mut ok = false;
    let mut out_nx = 0.0f32;
    let mut out_ny = 0.0f32;
    if (dx != 0.0f32 || dy != 0.0f32) && is_refract_boundary(pmap, x, y, xr, yr) {
        // `pt & REFRACT` forces `ldm = rdm = 0xFF` (no direction mask).
        let ldm = 0xFFu32;
        let rdm = 0xFFu32;
        let mut lx = x;
        let mut ly = y;
        let mut rx = x;
        let mut ry = y;
        let mut lv = 1u32;
        let mut rv = 1u32;
        let mut lm = seed_i32(x, -1i32);
        let mut rm = seed_i32(x, -1i32);
        let mut j = 0i32;
        let mut stop = false;
        for _i in 0..10usize {
            if !stop {
                if lv != 0u32 {
                    let r = find_next_boundary(lx, ly, ldm, lm, true, pmap, xr, yr);
                    lv = r.0;
                    lx = r.1;
                    ly = r.2;
                    lm = r.3;
                }
                if rv != 0u32 {
                    let r = find_next_boundary(rx, ry, rdm, rm, false, pmap, xr, yr);
                    rv = r.0;
                    rx = r.1;
                    ry = r.2;
                    rm = r.3;
                }
                j += lv as i32 + rv as i32;
                if lv == 0u32 && rv == 0u32 {
                    stop = true;
                }
            }
        }
        // `NORMAL_MIN_EST = 3` (`SimulationConfig.h:55`).
        if j >= 3i32 && (lx != rx || ly != ry) {
            let ex = (rx - lx) as f32;
            let ey = (ry - ly) as f32;
            let inv = 1.0f32 / (ex * ex + ey * ey).sqrt();
            out_nx = ey * inv;
            out_ny = -ex * inv;
            ok = true;
        }
    }
    (ok, out_nx, out_ny)
}

/// `Simulation::get_normal_interp` (`Simulation.cpp:1720-1750`).
///
/// Marches from the particle position along the velocity in
/// `NORMAL_FRAC = 16` sub-steps until the first glass boundary, then asks
/// [`get_normal`] for the surface normal.
#[cube]
#[allow(clippy::too_many_arguments)]
fn get_normal_interp(
    x0_in: f32,
    y0_in: f32,
    dx_in: f32,
    dy_in: f32,
    pmap: &[u32],
    xr: i32,
    yr: i32,
) -> (bool, f32, f32) {
    // `NORMAL_FRAC = 16` (`SimulationConfig.h:57`).
    let dx = dx_in / 16.0f32;
    let dy = dy_in / 16.0f32;
    let mut x0 = x0_in;
    let mut y0 = y0_in;
    let mut found = false;
    let mut oob = false;
    let mut hx = 0i32;
    let mut hy = 0i32;
    // `NORMAL_INTERP = 20` (`SimulationConfig.h:56`).
    for _i in 0..20usize {
        if !found && !oob {
            let x = i32::cast_from(x0 + 0.5f32);
            let y = i32::cast_from(y0 + 0.5f32);
            if x < 0i32 || y < 0i32 || x >= xr || y >= yr {
                oob = true;
            } else if is_refract_boundary(pmap, x, y, xr, yr) {
                found = true;
                hx = x;
                hy = y;
            }
            x0 += dx;
            y0 += dy;
        }
    }
    let mut ok = false;
    let mut nx = 0.0f32;
    let mut ny = 0.0f32;
    if found {
        let gn = get_normal(hx, hy, dx, dy, pmap, xr, yr);
        ok = gn.0;
        nx = gn.1;
        ny = gn.2;
    }
    (ok, nx, ny)
}

/// `Simulation::get_wavelength_bin` (`Simulation.cpp:886-922`).
///
/// Narrows `wm` to the sampled wavelength bin and returns
/// `(bin, narrowed wm)`; `bin == -1` means the photon carries no wavelength
/// and must die. The reference draws the bin from the global RNG; this port
/// uses the deterministic per-slot hash.
#[cube]
fn get_wavelength_bin(wm: u32, seed: u32) -> (i32, u32) {
    let masked = wm & 0x3FFF_FFFFu32;
    // `bin == -1` means the photon carries no wavelength and must die; the
    // runtime seed keeps this a `NativeExpand<i32>` (see `seed_i32`).
    let mut bin = seed_i32(masked as i32, -1i32);
    let mut out = wm;
    if masked != 0u32 {
        // `w0 = ctz(wm | 0xC0000000)`, `wM = 31 - clz(wm & 0x3FFFFFFF)`.
        let mut w0 = 30u32;
        let mut wm_high = 0u32;
        for i in 0..30u32 {
            if (masked & (1u32 << i)) != 0u32 {
                if i < w0 {
                    w0 = i;
                }
                if i > wm_high {
                    wm_high = i;
                }
            }
        }
        if wm_high - w0 < 5u32 {
            bin = (wm_high + w0) as i32;
        } else {
            let r = pcg_hash(seed ^ mix_index(w0, 2654435769u32));
            let i = ((r >> 1u32) % (wm_high - w0 - 4u32)) + w0;
            if (r & 1u32) != 0u32 {
                out = wm & (0x1Fu32 << i);
                bin = ((i + 2u32) * 2u32) as i32;
            } else {
                out = wm & (0xFu32 << i);
                bin = ((i + 2u32) * 2u32 - 1u32) as i32;
            }
        }
    }
    (bin, out)
}

/// Photon refraction / total internal reflection at a glass surface.
///
/// `Simulation.cpp:2966-3005`. Returns `(die, tir, vx, vy)`:
///
/// * `die` — `get_normal_interp` or `get_wavelength_bin` failed; the photon is
///   killed.
/// * `tir` — total internal reflection; the velocity is mirrored around the
///   normal and the particle does not move this tick
///   (`fin_xf = parts[i].x`).
///
/// `GLASS_IOR = 1.9`, `GLASS_DISP = 0.07` (`SimulationConfig.h:63-64`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn refract_photon(
    px: f32,
    py: f32,
    xi: i32,
    yi: i32,
    vx_in: f32,
    vy_in: f32,
    fin_x: i32,
    fin_y: i32,
    pmap: &[u32],
    part_ctype: &mut [u32],
    slot: usize,
    xr: i32,
    yr: i32,
    seed: u32,
) -> (bool, bool, f32, f32) {
    let mut vx = vx_in;
    let mut vy = vy_in;
    let mut die = false;
    let mut tir = false;
    let rt = pmap[(fin_y * xr + fin_x) as usize] & 0x1FFu32;
    let lt = pmap[(yi * xr + xi) as usize] & 0x1FFu32;
    let rt_glas = rt == PT_GLAS_U32 || rt == PT_BGLA_U32;
    let lt_glas = lt == PT_GLAS_U32 || lt == PT_BGLA_U32;
    if (rt_glas && !lt_glas) || (lt_glas && !rt_glas) {
        let gn = get_normal_interp(px, py, vx, vy, pmap, xr, yr);
        if !gn.0 {
            die = true;
        } else {
            let mut nrx = gn.1;
            let mut nry = gn.2;
            let wb = get_wavelength_bin(part_ctype[slot], seed);
            if wb.0 == -1i32 {
                die = true;
            } else {
                part_ctype[slot] = wb.1;
                let mut nn = 1.9f32 - 0.07f32 * ((wb.0 as f32) - 30.0f32) / 30.0f32;
                nn *= nn;
                let enter = rt_glas && !lt_glas;
                if enter {
                    nrx = -nrx;
                    nry = -nry;
                }
                if enter {
                    nn = 1.0f32 / nn;
                }
                let ct1 = vx * nrx + vy * nry;
                let ct2 = 1.0f32 - (nn * nn) * (1.0f32 - (ct1 * ct1));
                if ct2 < 0.0f32 {
                    // Total internal reflection.
                    vx -= 2.0f32 * ct1 * nrx;
                    vy -= 2.0f32 * ct1 * nry;
                    tir = true;
                } else {
                    let ct2s = ct2.sqrt() - nn * ct1;
                    vx = nn * vx + ct2s * nrx;
                    vy = nn * vy + ct2s * nry;
                }
            }
        }
    }
    (die, tir, vx, vy)
}

/// Per-`Falldown` destination search.
///
/// Returns `(target, vx, vy)`: `target` is `0xFFFF_FFFF` when the particle
/// stays, [`DIE_INTENT`] when it must die, a `0x8000_0000`-tagged pixel when a
/// blocked photon's wavelength mask must be applied, and a plain pixel index
/// otherwise.
#[cube]
#[allow(clippy::too_many_arguments)]
fn choose_destination(
    falldown: u32,
    vx: f32,
    vy: f32,
    px: f32,
    py: f32,
    xi: i32,
    yi: i32,
    sx: i32,
    sy: i32,
    props: u32,
    ty: u32,
    slot: usize,
    flags: u32,
    collision: f32,
    h2: u32,
    gravity_mode: u32,
    newton_grav: u32,
    gravity: f32,
    custom_gravity_x: f32,
    custom_gravity_y: f32,
    gel_tmp: i32,
    edge_mode: u32,
    frame_seed: u32,
    xres: u32,
    yres: u32,
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    part_ctype: &mut [u32],
    part_ty: &mut [u32],
    xr: i32,
    yr: i32,
    xcells: u32,
) -> (u32, f32, f32) {
    if falldown == 0u32 {
        gas_destination(
            vx, vy, px, py, xi, yi, sx, sy, props, ty, slot, collision, edge_mode, frame_seed,
            pmap, wall_bmap, wall_emap, move_lut, part_ctype, part_ty, xr, yr, xcells,
        )
    } else {
        fall_down_destination(
            vx,
            vy,
            px,
            py,
            xi,
            yi,
            sx,
            sy,
            props,
            ty,
            falldown,
            flags,
            collision,
            h2,
            gravity_mode,
            newton_grav,
            gravity,
            custom_gravity_x,
            custom_gravity_y,
            gel_tmp,
            edge_mode,
            xres,
            yres,
            pmap,
            wall_bmap,
            wall_emap,
            move_lut,
            part_ty,
            xr,
            yr,
            xcells,
        )
    }
}

/// Gases, moving solids and energy: `PlanMove` then the two-axis bounce.
///
/// `Simulation.cpp:3138-3152` (`Falldown == 0`):
///
/// ```text
/// if (!do_move(i, x, y, fin_xf, fin_yf)) {
///     clamp fin to +-ISTP around (x, y);
///     if (do_move(2x-fin_x, fin_y))        vx *= Collision;
///     else if (do_move(fin_x, 2y-fin_y))   vy *= Collision;
///     else { vx *= Collision; vy *= Collision; }
/// }
/// ```
///
/// The bounce tries **X-only, then Y-only, then both**, and multiplies by the
/// element's signed `Collision` (gases store a negative value there).
///
/// Photons keep the port's reflection-mask sentinel: the commit pass ANDs the
/// wavelength mask of the blocking pixel into `ctype`
/// (`Simulation.cpp:3044-3056`). A photon that cannot enter the pixel in front
/// of it reverses on the blocked axes and claims no cell. Its velocity is
/// reflected by the same signed `Collision` instead of the reference's surface
/// normal (`get_normal_interp`), which the movement pass cannot compute for
/// non-glass surfaces without another table.
///
/// A blocked photon that crosses a glass surface is refracted here first
/// ([`refract_photon`]).
#[cube]
#[allow(clippy::too_many_arguments)]
fn gas_destination(
    vx_in: f32,
    vy_in: f32,
    px: f32,
    py: f32,
    xi: i32,
    yi: i32,
    sub_x: i32,
    sub_y: i32,
    props: u32,
    ty: u32,
    slot: usize,
    collision: f32,
    edge_mode: u32,
    frame_seed: u32,
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    part_ctype: &mut [u32],
    part_ty: &mut [u32],
    xr: i32,
    yr: i32,
    xcells: u32,
) -> (u32, f32, f32) {
    let mut vx = vx_in;
    let mut vy = vy_in;

    let plan = crate::move_kernel::liquid::plan_move(
        vx, vy, px, py, xi, yi, props, ty, pmap, wall_bmap, wall_emap, move_lut, xr, yr, xcells,
        edge_mode,
    );
    let mut fin_x = plan.0;
    let mut fin_y = plan.1;

    // Integer-position adaptation: a sub-pixel velocity that rounds back onto
    // the current pixel still advances on `|v|` of the frames (see
    // `step_axis`). TPT's float position does this implicitly.
    if fin_x == xi && fin_y == yi && (sub_x != 0i32 || sub_y != 0i32) {
        fin_x = xi + sub_x;
        fin_y = yi + sub_y;
    }

    let mut target = 0xFFFF_FFFFu32;
    let mut blocked = false;
    let mut died = false;
    let mut tir = false;
    if fin_x != xi || fin_y != yi {
        let d = do_move_at(
            pmap, wall_bmap, wall_emap, move_lut, props, ty, fin_x, fin_y, xr, yr, xcells,
            edge_mode,
        );
        if d.0 == DO_MOVE_OK {
            if ty == PT_PHOT_U32 {
                // `eval_move(PT_PHOT, fin)` succeeded, so the move may cross a
                // glass surface: refract before claiming the pixel.
                let r = refract_photon(
                    px, py, xi, yi, vx, vy, d.1, d.2, pmap, part_ctype, slot, xr, yr, frame_seed,
                );
                if r.0 {
                    died = true;
                } else {
                    vx = r.2;
                    vy = r.3;
                    // Total internal reflection: stay put this tick.
                    tir = r.1;
                }
            }
            if !died && !tir {
                target = (d.2 * xr + d.1) as u32;
            }
        } else if d.0 == DO_MOVE_DIE {
            // `move`'s out-of-band kill: `EDGE_VOID` erases a particle that
            // leaves the `CELL .. RES-CELL` band, and the reference's
            // `do_move` returns `-1`, which stops the bounce chain.
            died = true;
        } else {
            blocked = true;
            if ty == PT_PHOT_U32 {
                // Half-silvered mirror (`Simulation.cpp:1205-1206`): a photon
                // whose move `eval_move` rejects still enters the pixel when the
                // destination is `BMTL` and the 1-in-2 roll passes, or when the
                // photon is itself standing in `BMTL`. Both cases force `e = 2`,
                // so the photon shares the pixel instead of reflecting and the
                // commit pass must not re-apply `can_move` (the bit survives in
                // `intent_target`). Otherwise the blocking pixel rides the
                // wavelength-mask marker.
                let dest_px = (d.2 * xr + d.1) as u32;
                let dest_ty = pmap[dest_px as usize] & 0x1FFu32;
                let src_ty = pmap[(yi * xr + xi) as usize] & 0x1FFu32;
                let roll = pcg_hash(
                    frame_seed
                        ^ mix_index(slot as u32, 0x9E37_79B9u32)
                        ^ mix_index(dest_px, 0x85EB_CA6Bu32),
                );
                if (dest_ty == PT_BMTL_U32 && (roll & 1u32) == 0u32) || src_ty == PT_BMTL_U32 {
                    blocked = false;
                    target = INTENT_HALF_MIRROR | dest_px;
                } else {
                    target = 0x8000_0000u32 | dest_px;
                    saw_wood_at(pmap, part_ty, d.1, d.2, vx, vy, xr, yr);
                }
            } else {
                // `WOOD -> SAWD` on the rejected probe (`Simulation.cpp:
                // 1211-1218`), before any fallback destination is tried.
                saw_wood_at(pmap, part_ty, d.1, d.2, vx, vy, xr, yr);
            }
        }
    }

    if died {
        target = DIE_INTENT;
    } else if !tir && blocked {
        if ty == PT_PHOT_U32 {
            // Photon reflection: reverse on the blocked axes; the reference
            // uses the surface normal, which needs the non-glass normal
            // machinery.
            if fin_x != xi {
                vx *= collision;
            }
            if fin_y != yi {
                vy *= collision;
            }
        } else {
            // Gas / moving solid bounce (`Simulation.cpp:3140-3152`).
            let mut bx = fin_x;
            if bx > xi + ISTP as i32 {
                bx = xi + ISTP as i32;
            }
            if bx < xi - ISTP as i32 {
                bx = xi - ISTP as i32;
            }
            let mut by = fin_y;
            if by > yi + ISTP as i32 {
                by = yi + ISTP as i32;
            }
            if by < yi - ISTP as i32 {
                by = yi - ISTP as i32;
            }
            let rx = 2i32 * xi - bx;
            let ry = 2i32 * yi - by;
            let d1 = do_move_at(
                pmap, wall_bmap, wall_emap, move_lut, props, ty, rx, by, xr, yr, xcells, edge_mode,
            );
            if d1.0 != DO_MOVE_BLOCKED {
                if d1.0 == DO_MOVE_DIE {
                    target = DIE_INTENT;
                } else if d1.1 != xi || d1.2 != yi {
                    target = (d1.2 * xr + d1.1) as u32;
                }
                vx *= collision;
            } else {
                saw_wood_at(pmap, part_ty, d1.1, d1.2, vx, vy, xr, yr);
                let d2 = do_move_at(
                    pmap, wall_bmap, wall_emap, move_lut, props, ty, bx, ry, xr, yr, xcells,
                    edge_mode,
                );
                if d2.0 != DO_MOVE_BLOCKED {
                    if d2.0 == DO_MOVE_DIE {
                        target = DIE_INTENT;
                    } else if d2.1 != xi || d2.2 != yi {
                        target = (d2.2 * xr + d2.1) as u32;
                    }
                    vy *= collision;
                } else {
                    saw_wood_at(pmap, part_ty, d2.1, d2.2, vx, vy, xr, yr);
                    vx *= collision;
                    vy *= collision;
                }
            }
        }
    }
    (target, vx, vy)
}

/// Integrates every particle's velocity and claims its desired destination pixel.
///
/// # Arguments
///
/// All buffers are SoA and indexed by particle slot (`part_*`, `intent_target`),
/// by pixel (`pmap`, `cell_winner`) or by air cell (`air_*`, `wall_*`).
///
/// `move_lut` packs the per-element record `[properties, Falldown, weight_rank,
/// HeatConduct, advection, air_drag, air_loss, loss, collision, gravity,
/// diffusion, hot_air]` followed by the flat `PT_NUM * PT_NUM` can-move matrix
/// (`0` block, `1` swap, `2` pass, `3` special), addressed through
/// [`super::can_move_at`].
///
/// The 16 storage buffers and their order are fixed by the host launch; do not
/// add, remove or reorder parameters. `part_ctype` / `part_tmp` took the slots
/// of the former `air_pv` binding and the standalone physics table: TPT has no
/// pressure-gradient force on particles (the only air coupling is
/// `Advection * vx[cell]`, `Simulation.cpp:2383-2384`), and the physics words
/// now live in `move_lut`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn velocity_and_intent_kernel(
    part_ty: &mut [u32],
    part_x: &[f32],
    part_y: &[f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &[u32],
    intent_target: &mut [u32],
    cell_winner: &mut [Atomic<u32>],
    pmap: &[u32],
    air_vx: &[f32],
    air_vy: &[f32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    part_ctype: &mut [u32],
    part_tmp: &[i32],
    num_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    _ycells: u32,
    gravity_mode: u32,
    custom_gravity_x: f32,
    custom_gravity_y: f32,
    newton_grav: u32,
    edge_mode: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i >= num_slots as usize {
        terminate!();
    }

    // Stage 1: every slot starts with "no move"; a candidate overwrites this.
    intent_target[i] = 0xFFFF_FFFFu32;

    let ty = part_ty[i];
    if ty == 0u32 {
        terminate!();
    }
    // Element ids must fit the packed tables (`PT_NUM_U32`, mirrored PT_NUM).
    if ty >= PT_NUM_U32 {
        terminate!();
    }

    let flags = part_flags[i];
    // Rigid pixels belong to the pixel-cluster rigid-body solver, and
    // `FLAG_SKIPMOVE` particles (sparks, walls-in-progress) never move.
    if (flags & 0x10u32) != 0u32 {
        terminate!();
    }
    if (flags & 0x02u32) != 0u32 {
        terminate!();
    }

    let px = part_x[i];
    let py = part_y[i];
    let xi = i32::cast_from(px + 0.5f32);
    let yi = i32::cast_from(py + 0.5f32);
    let xr = xres as i32;
    let yr = yres as i32;
    if xi < 0i32 || xi >= xr || yi < 0i32 || yi >= yr {
        // `Simulation.cpp:2312-2317` kills off-screen particles.
        part_ty[i] = 0u32;
        terminate!();
    }

    let xu = xi as u32;
    let yu = yi as u32;
    let cx = xu / 4u32;
    let cy = yu / 4u32;
    let cell = (cy * xcells + cx) as usize;

    let ty_idx = ty as usize;
    let meta = ty_idx * 12usize;
    let props = move_lut[meta];
    let falldown = move_lut[meta + 1usize];
    let weight_rank = move_lut[meta + 2usize];
    // Physics words: `[advection, air_drag, air_loss, loss, collision, gravity,
    // diffusion, hot_air]` starting at `meta + 4`. `air_drag` / `air_loss` are
    // consumed by the particle -> air deposit pass, which reads the same record.
    let phys = meta + 4usize;
    let advection = f32::reinterpret(move_lut[phys]);
    let loss = f32::reinterpret(move_lut[phys + 3usize]);
    let collision = f32::reinterpret(move_lut[phys + 4usize]);
    let gravity = f32::reinterpret(move_lut[phys + 5usize]);
    let diffusion = f32::reinterpret(move_lut[phys + 6usize]);

    // Pre-move gates (`Simulation.cpp:2312-2338`).
    let action = pre_move_checks(
        ty,
        props,
        xi,
        yi,
        xres,
        yres,
        wall_bmap[cell],
        wall_emap[cell],
    );
    if action == 1u32 {
        part_ty[i] = 0u32;
        terminate!();
    }
    if action == 0u32 {
        terminate!();
    }

    let base = base_gravity(
        gravity_mode,
        gravity,
        xres,
        yres,
        xi as f32,
        yi as f32,
        custom_gravity_x,
        custom_gravity_y,
    );

    let ax = air_vx[cell];
    let ay = air_vy[cell];

    // Deterministic per-frame hash for the diffusion kick, so a replay of the
    // same frame produces the same motion.
    let h1 = pcg_hash(frame_seed ^ mix_index(i as u32, 2654435769u32));
    let h2 = pcg_hash(h1);

    // `Loss` is applied by the gravity kernel when Newtonian gravity ran
    // (`Simulation.cpp:2361-2384` puts `vx *= Loss` before the `pGrav` add);
    // otherwise this kernel applies it here.
    let mut loss_here = loss;
    if newton_grav != 0u32 {
        loss_here = 1.0f32;
    }
    let integrated = integrate_velocity(
        part_vx[i], part_vy[i], loss_here, advection, ax, ay, base.0, base.1, diffusion, h1, h2,
    );
    let clamped = clamp_plan_velocity(integrated.0, integrated.1);
    let vx = clamped.0;
    let vy = clamped.1;

    let sx = step_axis(vx, h1);
    let sy = step_axis(vy, h2);

    let chosen = choose_destination(
        falldown,
        vx,
        vy,
        px,
        py,
        xi,
        yi,
        sx,
        sy,
        props,
        ty,
        i,
        flags,
        collision,
        h2,
        gravity_mode,
        newton_grav,
        gravity,
        custom_gravity_x,
        custom_gravity_y,
        part_tmp[i],
        edge_mode,
        frame_seed,
        xres,
        yres,
        pmap,
        wall_bmap,
        wall_emap,
        move_lut,
        part_ctype,
        part_ty,
        xr,
        yr,
        xcells,
    );
    let target = chosen.0;

    if target == DIE_INTENT {
        part_ty[i] = 0u32;
        terminate!();
    }

    if target != 0xFFFF_FFFFu32 {
        intent_target[i] = target;
        // A blocked photon carries the blocking pixel in the low 31 bits and
        // never claims a cell (it stays put; the commit pass only filters its
        // wavelength). The half-silvered-mirror flag rides bit 30 above the
        // pixel field and the photon does claim that pixel.
        if (target & 0x8000_0000u32) == 0u32 {
            let pixel = (target & INTENT_PIXEL_MASK) as usize;
            // `(weight_rank << 23) | (id + 1)`: 9 bits of rank above the 23-bit
            // slot id field shared with `pmap_pack`. `id + 1` never reaches
            // `2^23` because the pool saturates at
            // `SimBuffers::MAX_PARTICLE_CAPACITY`.
            let key = ((weight_rank & 0x1FFu32) << 23u32) | (((i as u32) + 1u32) & 0x7FFFFFu32);
            cell_winner[pixel].fetch_max(key);
        }
    }

    part_vx[i] = chosen.1;
    part_vy[i] = chosen.2;
}

/// `PT_STKM` / `PT_STKM2` / `PT_FIGH` / `PT_GLAS` / `PT_BGLA` / `PT_BMTL`
/// mirrors.
const PT_STKM_U32: u32 = 59;
const PT_STKM2_U32: u32 = 137;
const PT_FIGH_U32: u32 = 159;
const PT_GLAS_U32: u32 = 24;
const PT_BGLA_U32: u32 = 95;
/// `PT_BMTL`, the half-silvered mirror (`Simulation.cpp:1205-1206`).
const PT_BMTL_U32: u32 = tpt_core::constants::PT_BMTL;
const _: () = assert!(PT_STKM_U32 == 59);
const _: () = assert!(PT_STKM2_U32 == 137);
const _: () = assert!(PT_FIGH_U32 == 159);
const _: () = assert!(PT_GLAS_U32 == 24);
const _: () = assert!(PT_BGLA_U32 == 95);
const _: () = assert!(PT_BMTL_U32 == 46);
