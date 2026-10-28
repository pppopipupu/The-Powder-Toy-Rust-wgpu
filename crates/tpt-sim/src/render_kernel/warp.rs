use super::framebuffer::{fb_blue, fb_green, fb_red, pack_rgb};
use super::*;

/// Copies a `u32` buffer (used to snapshot the framebuffer before lensing).
#[cube(launch_unchecked)]
pub fn copy_u32_kernel(src: &[u32], dst: &mut [u32], len: u32) {
    let idx = ABSOLUTE_POS as u32;
    if idx < len {
        dst[idx as usize] = src[idx as usize];
    }
}

/// TPT `Renderer::render_gravlensing`: bends the image by the gravity force grid.
///
/// Each pixel takes its red channel from a sample pulled `0.75 * force` towards
/// the source, green from `0.875 * force` and blue from `force`, which fringes
/// strong fields into colour (`Renderer.cpp:60-80`). `RenderSimulation` zeroes
/// the frame before this pass (`Renderer.cpp:45`) and only writes a pixel when
/// all three samples are inside the canvas, so a pixel whose samples leave the
/// canvas stays black; the port stores that black as opaque so the viewport
/// does not blend it with the window background (`0xFF00_0000` instead of
/// TPT's raw `0`).
#[cube(launch_unchecked)]
pub fn render_warp_kernel(
    framebuf: &mut [u32],
    source: &[u32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xres * yres {
        let x = idx % xres;
        let y = idx / xres;
        let cell = (y >> 2u32) * xcells + (x >> 2u32);
        let fx = grav_fx[cell as usize];
        let fy = grav_fy[cell as usize];

        let rx = (x as f32 - fx * 0.75f32 + 0.5f32) as i32;
        let ry = (y as f32 - fy * 0.75f32 + 0.5f32) as i32;
        let gx = (x as f32 - fx * 0.875f32 + 0.5f32) as i32;
        let gy = (y as f32 - fy * 0.875f32 + 0.5f32) as i32;
        let bx = (x as f32 - fx + 0.5f32) as i32;
        let by = (y as f32 - fy + 0.5f32) as i32;

        if rx >= 0i32
            && ry >= 0i32
            && (rx as u32) < xres
            && (ry as u32) < yres
            && gx >= 0i32
            && gy >= 0i32
            && (gx as u32) < xres
            && (gy as u32) < yres
            && bx >= 0i32
            && by >= 0i32
            && (bx as u32) < xres
            && (by as u32) < yres
        {
            let rp = source[((ry as u32) * xres + (rx as u32)) as usize];
            let gp = source[((gy as u32) * xres + (gx as u32)) as usize];
            let bp = source[((by as u32) * xres + (bx as u32)) as usize];
            framebuf[idx as usize] = pack_rgb(fb_red(rp), fb_green(gp), fb_blue(bp));
        } else {
            // TPT filled the whole frame with zero before the pass, so every
            // pixel whose samples leave `RES.OriginRect()` stays black.
            framebuf[idx as usize] = 0xFF00_0000u32;
        }
    }
}
