//! Full-canvas overlays drawn on top of the particle passes.
//!
//! Ports the three `Renderer` helpers that are not tied to a particle or a
//! wall: `draw_grav` (gravity field vectors), `draw_other` (`EMP` flash) and
//! `draw_grav_zones` (gravity-tool cell markers).

use super::framebuffer::{add_px, blend_px, clampf, clampi, pack_rgb};
use super::*;

/// TPT `Renderer::draw_grav`: a four-step white trail along each cell's force
/// vector, additively blended with `alpha = |fx| + |fy| * 20`.
///
/// TPT passes `int(dist * 20.0f)` to `RGBA::WithAlpha(uint8_t)`, so a force
/// large enough to push the product past 255 wraps instead of saturating; the
/// `& 0xFF` below keeps that quirk.
#[cube(launch_unchecked)]
pub fn grav_field_kernel(
    framebuf: &mut [u32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xcells * ycells {
        let gx = grav_fx[idx as usize];
        let gy = grav_fy[idx as usize];
        let agx = if gx < 0.0f32 { -gx } else { gx };
        let agy = if gy < 0.0f32 { -gy } else { gy };
        if agx > 0.001f32 || agy > 0.001f32 {
            let dist = agx + agy;
            let alpha = ((dist * 20.0f32) as i32 & 0xFF) as f32;
            let cx = idx % xcells;
            let cy = idx / xcells;
            let mut px = (cx as f32) * 4.0f32;
            let mut py = (cy as f32) * 4.0f32;
            let white = pack_rgb(255.0f32, 255.0f32, 255.0f32);
            for _ in 0u32..4u32 {
                px -= gx * 0.5f32;
                py -= gy * 0.5f32;
                let ix = (px + 0.5f32) as i32;
                let iy = (py + 0.5f32) as i32;
                if ix >= 0i32 && iy >= 0i32 && (ix as u32) < xres && (iy as u32) < yres {
                    let d = ((iy as u32) * xres + ix as u32) as usize;
                    framebuf[d] = add_px(framebuf[d], white, alpha);
                }
            }
        }
    }
}

/// TPT `Renderer::draw_other`: the screen-wide `EMP` flash.
///
/// `emp_decor` is clamped to `0..=40`; the tint is
/// `RGB(emp_decor * 2.5, 100 + emp_decor * 1.5, 255)` blended at
/// `alpha = emp_decor / 110 * 255`.
#[cube(launch_unchecked)]
pub fn emp_tint_kernel(framebuf: &mut [u32], xres: u32, yres: u32, emp_decor: i32) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xres * yres {
        let d = clampi(emp_decor, 0i32, 40i32) as f32;
        if d > 0.0f32 {
            let r = clampf(d * 2.5f32, 0.0f32, 255.0f32);
            let g = clampf(100.0f32 + d * 1.5f32, 0.0f32, 255.0f32);
            let b = 255.0f32;
            let a = clampf((d / 110.0f32) * 255.0f32, 0.0f32, 255.0f32);
            framebuf[idx as usize] = blend_px(framebuf[idx as usize], pack_rgb(r, g, b), a);
        }
    }
}

/// TPT `Renderer::draw_grav_zones`: a 4x4 checker over every air cell whose
/// gravity mask is set (`0xFFC800` on the diagonal, `0x202020` elsewhere,
/// alpha 120).
#[cube(launch_unchecked)]
pub fn grav_zones_kernel(
    framebuf: &mut [u32],
    grav_mask: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xcells * ycells && grav_mask[idx as usize] != 0u32 {
        let cx = idx % xcells;
        let cy = idx / xcells;
        let diag = pack_rgb(255.0f32, 200.0f32, 0.0f32);
        let plain = pack_rgb(32.0f32, 32.0f32, 32.0f32);
        for oy in 0u32..4u32 {
            for ox in 0u32..4u32 {
                let x = cx * 4u32 + ox;
                let y = cy * 4u32 + oy;
                if x < xres && y < yres {
                    let rgb = if ox == oy { diag } else { plain };
                    let d = (y * xres + x) as usize;
                    framebuf[d] = blend_px(framebuf[d], rgb, 120.0f32);
                }
            }
        }
    }
}
