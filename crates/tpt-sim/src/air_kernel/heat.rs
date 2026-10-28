use super::*;

/// Ambient heat (`air_hv`, Kelvin) diffusion, advection and convection for one
/// air step: the port of `Air::update_airh` (`Air.cpp:78-250`).
///
/// The reference smooths **three** fields with the same 3x3 Gaussian kernel —
/// ambient heat, `vx` and `vy` — then advects `hv` from the smoothed velocity
/// field. This kernel reproduces that:
///
/// * 3x3 normalized Gaussian smoothing — `w(0,0) = 1`, `w(±1,0) = e^-2`,
///   `w(±1,±1) = e^-4` — sampling only the interior `1..XCELLS-2 x
///   1..YCELLS-2` cells that are not flagged in `block_airh & 0x8`; every
///   other tap repeats the centre value (`Air.cpp:100-120`).
/// * The far-advection step walk of [`far_advect_target`] followed by a
///   bilinear sample of `hv`, blended with the diffused value by
///   `AIR_VADV = 0.3`; a sample whose cell blocks ambient heat falls back to
///   the diffused value `odh` (`Air.cpp:124-190`).
/// * A cell flagged in `block_airh & 0x8` keeps its old `hv`
///   (`Air.cpp:170-176`), then the result is clamped to `[0, 9999]` K and the
///   border band is pinned to `ambient_air_temp`.
/// * Air convection (`Air.cpp:196-246`), applied to the raw `vx` / `vy` of the
///   cell: `AIRC_LEGACY` uses the finite difference of the **old** `hv` along
///   the gravity field, `AIRC_BOUSSINESQ` uses
///   `weight = min(0.01, (hv - ambient) / 10000)` from the **old** `hv`.
///
/// The velocity fields are written in place (only the cell's own value is
/// read), while the heat result goes to `ohv` and is settled into `hv` by
/// [`air_copy_back_kernel`].
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn air_heat_convect_kernel(
    hv: &[f32],
    vx: &mut [f32],
    vy: &mut [f32],
    block_airh: &[u32],
    ohv: &mut [f32],
    xcells: usize,
    ycells: usize,
    ambient_air_temp: f32,
    aheat_enable: u32,
    convection_mode: u32,
    gravity_mode: u32,
    custom_gravity_x: f32,
    custom_gravity_y: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx < xcells * ycells {
        let x = idx % xcells;
        let y = idx / xcells;

        let center = hv[idx];
        let center_vx = vx[idx];
        let center_vy = vy[idx];

        let mut sum_h = 0.0f32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut wsum = 0.0f32;
        let mut dy = 0usize;
        while dy < 3usize {
            let mut dx = 0usize;
            while dx < 3usize {
                // `dx` / `dy` are biased by one: 0 = -1, 1 = 0, 2 = +1.
                let mut nx = x + dx;
                nx = nx.saturating_sub(1usize);
                let mut ny = y + dy;
                ny = ny.saturating_sub(1usize);

                // w(0,0) = 1, w(±1,0) = e^-2, w(±1,±1) = e^-4
                let mut w = 0.01831564f32;
                if dx == 1usize {
                    w = 0.13533528f32;
                }
                if dy == 1usize {
                    w = 0.13533528f32;
                }
                if dx == 1usize && dy == 1usize {
                    w = 1.0f32;
                }

                // `Air.cpp:100-120`: the neighbour is used only when it is in
                // the interior and does not block ambient heat.
                let mut interior_tap = 0u32;
                if nx >= 1usize && ny >= 1usize && nx + 1usize < xcells && ny + 1usize < ycells {
                    interior_tap = 1u32;
                }
                let ni = ny * xcells + nx;
                let mut sh = center;
                let mut sx = center_vx;
                let mut sy = center_vy;
                if interior_tap != 0u32 && (block_airh[ni] & 0x8u32) == 0u32 {
                    sh = hv[ni];
                    sx = vx[ni];
                    sy = vy[ni];
                }

                sum_h += sh * w;
                sum_x += sx * w;
                sum_y += sy * w;
                wsum += w;
                dx += 1usize;
            }
            dy += 1usize;
        }
        let mut dh = sum_h / wsum;
        let smooth_vx = sum_x / wsum;
        let smooth_vy = sum_y / wsum;

        // Trying to take air temp from far away (`Air.cpp:124-162`), using the
        // smoothed velocity exactly like the reference.
        let (tx, ty) =
            far_advect_target(x, y, smooth_vx, smooth_vy, block_airh, xcells, ycells, 1u32);
        let i0 = tx as i32;
        let j0 = ty as i32;
        let fx = tx - i0 as f32;
        let fy = ty - j0 as f32;
        if (block_airh[idx] & 0x8u32) == 0u32
            && i0 >= 0i32
            && i0 < xcells as i32 - 1i32
            && j0 >= 0i32
            && j0 < ycells as i32 - 1i32
            && fx >= 0.0f32
            && fy >= 0.0f32
        {
            let iu = i0 as usize;
            let ju = j0 as usize;
            let r0 = ju * xcells;
            let r1 = (ju + 1usize) * xcells;
            // Blocked samples fall back to the diffused value (`odh`).
            let s00 = if (block_airh[r0 + iu] & 0x8u32) != 0u32 {
                dh
            } else {
                hv[r0 + iu]
            };
            let s10 = if (block_airh[r0 + iu + 1usize] & 0x8u32) != 0u32 {
                dh
            } else {
                hv[r0 + iu + 1usize]
            };
            let s01 = if (block_airh[r1 + iu] & 0x8u32) != 0u32 {
                dh
            } else {
                hv[r1 + iu]
            };
            let s11 = if (block_airh[r1 + iu + 1usize] & 0x8u32) != 0u32 {
                dh
            } else {
                hv[r1 + iu + 1usize]
            };
            let w00 = (1.0f32 - fx) * (1.0f32 - fy);
            let w10 = fx * (1.0f32 - fy);
            let w01 = (1.0f32 - fx) * fy;
            let w11 = fx * fy;
            dh = dh * 0.7f32 + 0.3f32 * (s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11);
        }

        // Don't update a cell that blocks ambient heat (`Air.cpp:170-176`).
        if (block_airh[idx] & 0x8u32) != 0u32 {
            dh = center;
        }

        // Ambient heat stays inside [0, 9999] K.
        dh = clampf(dh, 0.0f32, 9999.0f32);

        // The border band is set to ambient before the loop in TPT; pinning the
        // result is equivalent because the input border already holds it.
        if x < BORDER || y < BORDER || x + BORDER >= xcells || y + BORDER >= ycells {
            dh = ambient_air_temp;
        }
        if aheat_enable == 0u32 {
            dh = ambient_air_temp;
        }

        ohv[idx] = dh;

        // Air convection (`Air.cpp:196-246`). TPT queries the gravity field
        // with a particle gravity of `-1` (`GetGravityField(x*CELL, y*CELL,
        // -1.0f, -1.0f, ...)`), so the convection acceleration is the negated
        // config gravity direction: upward for `Vertical`, away from the canvas
        // centre for `Radial`, `-(custom_gravity_x, custom_gravity_y)` for
        // `Custom`, and zero when gravity is off.
        if x >= 2usize && x + 2usize < xcells && y >= 2usize && y + 2usize < ycells {
            let mut grav_x = 0.0f32;
            let mut grav_y = 0.0f32;
            if gravity_mode == 0u32 {
                // `GravityMode::Vertical`: gravity points down the canvas.
                grav_y = 1.0f32;
            } else if gravity_mode == 2u32 {
                // `GravityMode::Radial`: from the cell centre toward the canvas
                // centre. TPT's field magnitude is `r / (r - 0.01 px) ~ 1`, so
                // the unit direction is used here.
                let rx = xcells as f32 * 0.5f32 - (x as f32 + 0.5f32);
                let ry = ycells as f32 * 0.5f32 - (y as f32 + 0.5f32);
                let rdist = (rx * rx + ry * ry).sqrt();
                if rdist > 0.0f32 {
                    grav_x = rx / rdist;
                    grav_y = ry / rdist;
                }
            } else if gravity_mode == 3u32 {
                // `GravityMode::Custom`.
                grav_x = custom_gravity_x;
                grav_y = custom_gravity_y;
            }

            let mut conv_x = 0.0f32 - grav_x;
            let mut conv_y = 0.0f32 - grav_y;

            let mut next_vx = center_vx;
            let mut next_vy = center_vy;

            if convection_mode == 1u32 {
                // `AIRC_LEGACY`, air convection pre 99.0 (Air.cpp:207-218):
                // finite difference of the ambient heat along the gravity field,
                // only applied where it pushes air and the cell above is not
                // heat blocked. The difference uses the old `hv`.
                let row_m = (y - 1usize) * xcells;
                let weight = ((center - hv[idx - 1usize]) * conv_x
                    + (center - hv[row_m + x]) * conv_y)
                    / 5000.0f32;
                if weight > 0.0f32 && (block_airh[row_m + x] & 0x8u32) == 0u32 {
                    next_vx += weight * conv_x;
                    next_vy += weight * conv_y;
                }
            } else if convection_mode == 2u32 {
                // `AIRC_BOUSSINESQ` (Air.cpp:220-238): density is assumed to
                // vary linearly with `hv - ambientAirTemp`; the approximation is
                // capped from above at `0.01`. The weight reads the **old**
                // `hv` (`Air.cpp:236`), not the diffused/advected `dh`.
                //
                // The gravity cap is part of this branch only
                // (`Air.cpp:222-227`); the legacy branch uses the raw field.
                let grav_magn = (conv_x * conv_x + conv_y * conv_y).sqrt();
                if grav_magn > 10.0f32 {
                    conv_x /= 0.1f32 * grav_magn;
                    conv_y /= 0.1f32 * grav_magn;
                }
                let mut weight = (center - ambient_air_temp) / 10000.0f32;
                if weight > 0.01f32 {
                    weight = 0.01f32;
                }
                next_vx += weight * conv_x;
                next_vy += weight * conv_y;
            }

            // Velocity caps (Air.cpp:241-246).
            next_vx = clampf(next_vx, -256.0f32, 256.0f32);
            next_vy = clampf(next_vy, -256.0f32, 256.0f32);

            vx[idx] = next_vx;
            vy[idx] = next_vy;
        }
    }
}
