//! `SPARK` / `FLARE` / `LFLARE` ray passes (TPT `render_parts` ray effects).
//!
//! TPT walks a ray outwards from each particle, dividing the alpha by 1.5
//! (`SPARK`), 1.2 (`FLARE`) or 1.01 (`LFLARE`) per pixel. This pass walks the
//! four axis directions *towards* the pixel and sums every source it meets, which
//! is the same set of contributions in a different order.

use cubecl::prelude::*;

use tpt_core::constants::*;

use super::framebuffer::{add_px, blend_px, fb_blue, fb_green, fb_red, hash01, minf, pack_rgb};
use super::particles::{effective_pixel_mode, packed_at, particle_rgb};
use super::{
    ELEM_COLOUR, ELEM_HIGH_TEMP, ELEM_PIXEL_MODE, ELEM_PROPS, ELEM_WORDS, PMAP_ID_SHIFT,
    PMAP_TYPE_MASK, PMODE_FLARE, PMODE_LFLARE, PMODE_SPARK, RAY_STEPS,
};

/// Per-frame flicker term TPT adds to every ray alpha (`0..20`).
#[cube]
fn flicker_at(npid: u32, frame_seed: u32) -> f32 {
    let fi = (hash01(npid ^ frame_seed) * 20.0f32) as u32;
    fi as f32
}

/// `FLARE` / `LFLARE` 3x3 blend around the pixel.
///
/// TPT: `gradv = flicker + |vx|*17 + |vy|*17`, then BlendPixel with `gradv*4` at
/// the centre, `gradv*2` orthogonally and `gradv` on the diagonals.
#[cube]
#[allow(clippy::too_many_arguments)]
fn flare_pixels(
    dst_in: u32,
    photons: &[u32],
    pmap: &[u32],
    part_flags: &[u32],
    part_life: &[i32],
    part_ctype: &[u32],
    part_temp: &[f32],
    part_dcolour: &[u32],
    part_tmp: &[i32],
    part_tmp2: &[i32],
    part_tmp3: &[i32],
    part_tmp4: &[i32],
    part_vx: &[f32],
    part_vy: &[f32],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
    render_mode: u32,
    colour_mode: u32,
    decoration_level: u32,
    pretty_powder: u32,
    elem_count: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    frame_seed: u32,
    tick: u32,
) -> u32 {
    let mut dst = dst_in;
    for n in 0u32..9u32 {
        let dx = (n % 3u32) as i32 - 1i32;
        let dy = (n / 3u32) as i32 - 1i32;
        let np = packed_at(pmap, photons, xres, yres, x + dx, y + dy);
        if np != 0u32 {
            let nty = np & PMAP_TYPE_MASK;
            let npid = np >> PMAP_ID_SHIFT;
            let nslot = (npid - 1u32) as usize;
            if nslot < part_life.len() {
                let base = (nty % elem_count) * ELEM_WORDS;
                let npix = elem_props[(base + ELEM_PIXEL_MODE) as usize];
                let nmode = effective_pixel_mode(
                    npix,
                    nty,
                    part_life[nslot],
                    part_ctype[nslot],
                    part_temp[nslot],
                    part_tmp[nslot],
                    part_tmp2[nslot],
                    part_tmp3[nslot],
                    part_tmp4[nslot],
                    part_flags[nslot],
                    npid,
                    tick,
                    render_mode,
                    colour_mode,
                    part_vx[nslot],
                    part_vy[nslot],
                );
                if (nmode & (PMODE_FLARE | PMODE_LFLARE)) != 0u32 {
                    let flicker = flicker_at(npid, frame_seed);
                    let mut gradv =
                        flicker + part_vx[nslot].abs() * 17.0f32 + part_vy[nslot].abs() * 17.0f32;
                    if gradv > 255.0f32 {
                        gradv = 255.0f32;
                    }
                    let mut alpha = gradv;
                    if dx == 0i32 && dy == 0i32 {
                        alpha = minf(gradv * 4.0f32, 255.0f32);
                    } else if dx == 0i32 || dy == 0i32 {
                        alpha = minf(gradv * 2.0f32, 255.0f32);
                    }
                    if alpha > 0.0f32 {
                        let rgb = particle_rgb(
                            nty,
                            elem_props[(base + ELEM_COLOUR) as usize],
                            elem_props[(base + ELEM_PROPS) as usize],
                            f32::reinterpret(elem_props[(base + ELEM_HIGH_TEMP) as usize]),
                            part_temp[nslot],
                            part_life[nslot],
                            part_ctype[nslot],
                            npid,
                            part_dcolour[nslot],
                            colour_mode,
                            pretty_powder,
                            nmode,
                            part_tmp[nslot],
                            part_tmp2[nslot],
                            part_tmp3[nslot],
                            part_tmp4[nslot],
                            part_vx[nslot],
                            part_vy[nslot],
                            tick,
                            hdisp_min,
                            hdisp_max,
                            decoration_level,
                            elem_props,
                            elem_count,
                        );
                        dst = blend_px(dst, rgb, alpha);
                    }
                }
            }
        }
    }
    dst
}

/// Apply `SPARK` / `FLARE` / `LFLARE` from one source at ray distance `d`.
///
/// `decay15` is `1.5^d` and `decay12`/`decay101` are `1.2^(d-1)` and
/// `1.01^(d-1)`: TPT's `PMODE_FLARE`/`PMODE_LFLARE` loops start at `x = 1`
/// with the full `gradv` and divide only after drawing
/// (`Renderer.cpp:761-767`, `Renderer.cpp:784-790`), while `PMODE_SPARK`
/// starts at `x = 0` (`Renderer.cpp:733-745`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn ray_source(
    dst_in: u32,
    nmode: u32,
    rgb: u32,
    flicker: f32,
    life: i32,
    vx: f32,
    vy: f32,
    d: u32,
    decay15: f32,
    decay12: f32,
    decay101: f32,
) -> u32 {
    let mut dst = dst_in;

    // PMODE_SPARK: alpha `4*life + flicker`, /1.5 per pixel.
    if (nmode & PMODE_SPARK) != 0u32 {
        let g = (4i32 * life) as f32 + flicker;
        let a = g / decay15;
        if a > 0.5f32 {
            let ai = a as u32;
            let scaled = pack_rgb(
                fb_red(rgb) * (ai as f32) / 255.0f32,
                fb_green(rgb) * (ai as f32) / 255.0f32,
                fb_blue(rgb) * (ai as f32) / 255.0f32,
            );
            dst = add_px(dst, scaled, 255.0f32);
        }
    }
    // PMODE_FLARE: alpha `gradv`, /1.2 per pixel.
    if (nmode & PMODE_FLARE) != 0u32 && d > 0u32 {
        let g = minf(flicker + vx.abs() * 17.0f32 + vy.abs() * 17.0f32, 255.0f32);
        let a = g / decay12;
        if a > 0.5f32 {
            dst = add_px(dst, rgb, a);
        }
    }
    // PMODE_LFLARE: alpha `gradv`, /1.01 per pixel.
    if (nmode & PMODE_LFLARE) != 0u32 && d > 0u32 {
        let g = minf(flicker + vx.abs() * 17.0f32 + vy.abs() * 17.0f32, 255.0f32);
        let a = g / decay101;
        if a > 0.5f32 {
            dst = add_px(dst, rgb, a);
        }
    }
    dst
}

/// Axis rays: walk all four directions towards the pixel and sum every source.
///
/// TPT walks `x` outwards from the source while `gradv > 0.5`; walking inwards we
/// accumulate every source whose ray still reaches us. The `d == 0` step is the
/// pixel itself, which TPT's `SPARK` loop visits once per direction
/// (`AddPixel(nx + 0, ny)` and `AddPixel(nx - 0, ny)`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn axis_rays(
    dst_in: u32,
    photons: &[u32],
    pmap: &[u32],
    part_flags: &[u32],
    part_life: &[i32],
    part_ctype: &[u32],
    part_temp: &[f32],
    part_dcolour: &[u32],
    part_tmp: &[i32],
    part_tmp2: &[i32],
    part_tmp3: &[i32],
    part_tmp4: &[i32],
    part_vx: &[f32],
    part_vy: &[f32],
    elem_props: &[u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
    render_mode: u32,
    colour_mode: u32,
    decoration_level: u32,
    pretty_powder: u32,
    elem_count: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    frame_seed: u32,
    tick: u32,
) -> u32 {
    let mut dst = dst_in;
    for dir in 0u32..4u32 {
        let mut stepx = 0i32;
        let mut stepy = 0i32;
        if dir == 0u32 {
            stepx = 1i32;
        } else if dir == 1u32 {
            stepx = -1i32;
        } else if dir == 2u32 {
            stepy = 1i32;
        } else {
            stepy = -1i32;
        }

        let mut decay15 = 1.0f32;
        let mut decay12 = 1.0f32;
        let mut decay101 = 1.0f32;
        for d in 0u32..RAY_STEPS {
            let px = x + stepx * (d as i32);
            let py = y + stepy * (d as i32);
            let np = packed_at(pmap, photons, xres, yres, px, py);
            if np != 0u32 {
                let nty = np & PMAP_TYPE_MASK;
                let npid = np >> PMAP_ID_SHIFT;
                let nslot = (npid - 1u32) as usize;
                if nslot < part_life.len() {
                    let base = (nty % elem_count) * ELEM_WORDS;
                    let npix = elem_props[(base + ELEM_PIXEL_MODE) as usize];
                    let nmode = effective_pixel_mode(
                        npix,
                        nty,
                        part_life[nslot],
                        part_ctype[nslot],
                        part_temp[nslot],
                        part_tmp[nslot],
                        part_tmp2[nslot],
                        part_tmp3[nslot],
                        part_tmp4[nslot],
                        part_flags[nslot],
                        npid,
                        tick,
                        render_mode,
                        colour_mode,
                        part_vx[nslot],
                        part_vy[nslot],
                    );
                    if (nmode & (PMODE_SPARK | PMODE_FLARE | PMODE_LFLARE)) != 0u32 {
                        let rgb = particle_rgb(
                            nty,
                            elem_props[(base + ELEM_COLOUR) as usize],
                            elem_props[(base + ELEM_PROPS) as usize],
                            f32::reinterpret(elem_props[(base + ELEM_HIGH_TEMP) as usize]),
                            part_temp[nslot],
                            part_life[nslot],
                            part_ctype[nslot],
                            npid,
                            part_dcolour[nslot],
                            colour_mode,
                            pretty_powder,
                            nmode,
                            part_tmp[nslot],
                            part_tmp2[nslot],
                            part_tmp3[nslot],
                            part_tmp4[nslot],
                            part_vx[nslot],
                            part_vy[nslot],
                            tick,
                            hdisp_min,
                            hdisp_max,
                            decoration_level,
                            elem_props,
                            elem_count,
                        );
                        dst = ray_source(
                            dst,
                            nmode,
                            rgb,
                            flicker_at(npid, frame_seed),
                            part_life[nslot],
                            part_vx[nslot],
                            part_vy[nslot],
                            d,
                            decay15,
                            decay12,
                            decay101,
                        );
                    }
                }
            }
            decay15 *= 1.5f32;
            // The `FLARE`/`LFLARE` loops draw at `x = 1` with the undivided
            // `gradv` and divide afterwards, so the first axis step keeps
            // divisor 1 and distance `d` uses `1.2^(d-1)` / `1.01^(d-1)`.
            if d > 0u32 {
                decay12 *= 1.2f32;
                decay101 *= 1.01f32;
            }
        }
    }
    dst
}

/// Applies `PMODE_SPARK`, `PMODE_FLARE` and `PMODE_LFLARE`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn render_particles_rays_kernel(
    photons: &[u32],
    pmap: &[u32],
    part_flags: &[u32],
    part_life: &[i32],
    part_ctype: &[u32],
    part_temp: &[f32],
    part_dcolour: &[u32],
    part_tmp: &[i32],
    part_tmp2: &[i32],
    part_tmp3: &[i32],
    part_tmp4: &[i32],
    part_vx: &[f32],
    part_vy: &[f32],
    elem_props: &[u32],
    framebuf: &mut [u32],
    fx_axis: &[u32],
    xres: u32,
    yres: u32,
    render_mode: u32,
    colour_mode: u32,
    decoration_level: u32,
    pretty_powder: u32,
    elem_count: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    frame_seed: u32,
    tick: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xres * yres {
        let x = idx % xres;
        let y = idx / xres;
        // Rays walk up to `RAY_STEPS` pixels along the four axes, so a pixel
        // only receives one from its own row or column.
        if fx_axis[y as usize] == 0u32 && fx_axis[(yres + x) as usize] == 0u32 {
            terminate!();
        }
        let mut dst = framebuf[idx as usize];
        dst = flare_pixels(
            dst,
            photons,
            pmap,
            part_flags,
            part_life,
            part_ctype,
            part_temp,
            part_dcolour,
            part_tmp,
            part_tmp2,
            part_tmp3,
            part_tmp4,
            part_vx,
            part_vy,
            elem_props,
            xres,
            yres,
            x as i32,
            y as i32,
            render_mode,
            colour_mode,
            decoration_level,
            pretty_powder,
            elem_count,
            hdisp_min,
            hdisp_max,
            frame_seed,
            tick,
        );
        if (render_mode & (RENDER_SPRK | RENDER_EFFE)) != 0u32 {
            dst = axis_rays(
                dst,
                photons,
                pmap,
                part_flags,
                part_life,
                part_ctype,
                part_temp,
                part_dcolour,
                part_tmp,
                part_tmp2,
                part_tmp3,
                part_tmp4,
                part_vx,
                part_vy,
                elem_props,
                xres,
                yres,
                x as i32,
                y as i32,
                render_mode,
                colour_mode,
                decoration_level,
                pretty_powder,
                elem_count,
                hdisp_min,
                hdisp_max,
                frame_seed,
                tick,
            );
        }
        framebuf[idx as usize] = dst;
    }
}
