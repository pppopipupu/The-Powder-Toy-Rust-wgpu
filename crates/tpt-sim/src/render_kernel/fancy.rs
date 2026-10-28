//! Fancy pixel effects: `BLOB`/`GLOW`/`BLUR` and the fire bloom (TPT
//! `render_parts` pixel effects + `Renderer::render_fire`).
//!
//! TPT scatters these effects from each particle; here every pixel gathers the
//! contributions of the particles in its neighbourhood, with the same alpha
//! weights.
//!
//! There is no rigid-body bevel pass: TPT's `render_parts` only composites
//! each particle's `PMODE_*` pixels (`Renderer.cpp:660-720`), so the port-only
//! rim highlight this module used to add has been removed.

use cubecl::prelude::*;

use tpt_core::constants::*;

use super::framebuffer::{add_px, blend_px};
use super::particles::{effective_pixel_mode, glow_alpha, packed_at, particle_cola, particle_rgb};
use super::{
    ELEM_COLA, ELEM_COLOUR, ELEM_HIGH_TEMP, ELEM_PIXEL_MODE, ELEM_PROPS, ELEM_WORDS, FX_GLOW,
    FX_PMBLUR, FX_PMGLOW, PMAP_ID_SHIFT, PMAP_TYPE_MASK, PMODE_BLOB, PMODE_BLUR, PMODE_FLAT,
    PMODE_GLOW, cell_fx_any,
};

/// `PMODE_BLOB`: centre opaque, orthogonal neighbours alpha 223, diagonals 112.
///
/// The centre is applied last so it wins over the neighbours, matching TPT's
/// "the last particle drawn owns the pixel".
#[cube]
#[allow(clippy::too_many_arguments)]
fn blob_pixels(
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
    tick: u32,
) -> u32 {
    let mut dst = dst_in;
    let mut centre_rgb = 0u32;
    let mut has_centre = 0u32;
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
                if (nmode & PMODE_BLOB) != 0u32 {
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
                    if dx == 0i32 && dy == 0i32 {
                        centre_rgb = rgb;
                        has_centre = 1u32;
                    } else if dx == 0i32 || dy == 0i32 {
                        dst = blend_px(dst, rgb, 223.0f32);
                    } else {
                        dst = blend_px(dst, rgb, 112.0f32);
                    }
                }
            }
        }
    }
    if has_centre != 0u32 {
        dst = centre_rgb;
    }
    dst
}

/// `PMODE_GLOW`: centre 192, orthogonal 96 (+5 from the ray loop), axis rays up
/// to 5 px and diagonals with `|dx| + |dy| <= 7` at `cola1 = 5*cola/255`.
///
/// TPT draws the flat pixel with `video[pos] = colour` (`Renderer.cpp:683-686`)
/// and only then adds the particle's own glow, so a particle's flat pixel wipes
/// every glow an earlier particle already added to that pixel. A gather cannot
/// replay that ordering, so the contributions of particles with a lower slot id
/// than the pixel's owner are dropped whenever the owner paints a flat pixel.
/// Without it a dense radioactive mass (PLUT, URAN) accumulates the glow of
/// every neighbour and saturates into a flat neon slab instead of keeping the
/// dark body and the glowing rim TPT shows.
#[cube]
#[allow(clippy::too_many_arguments)]
fn glow_pixels(
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
    fx_cell: &[u32],
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
    tick: u32,
) -> u32 {
    let mut dst = dst_in;
    // Owner of this pixel: its flat draw wipes the glow of every earlier
    // particle (`Renderer.cpp:683-686`), see the doc comment above.
    let mut owner_id = 0u32;
    let mut owner_flat = 0u32;
    let own = packed_at(pmap, photons, xres, yres, x, y);
    if own != 0u32 {
        owner_id = own >> PMAP_ID_SHIFT;
        let oty = own & PMAP_TYPE_MASK;
        let oslot = (owner_id - 1u32) as usize;
        if oslot < part_life.len() {
            let obase = (oty % elem_count) * ELEM_WORDS;
            let omode = effective_pixel_mode(
                elem_props[(obase + ELEM_PIXEL_MODE) as usize],
                oty,
                part_life[oslot],
                part_ctype[oslot],
                part_temp[oslot],
                part_tmp[oslot],
                part_tmp2[oslot],
                part_tmp3[oslot],
                part_tmp4[oslot],
                part_flags[oslot],
                owner_id,
                tick,
                render_mode,
                colour_mode,
                part_vx[oslot],
                part_vy[oslot],
            );
            if (omode & PMODE_FLAT) != 0u32 {
                owner_flat = 1u32;
            }
        }
    }
    for n in 0u32..121u32 {
        let dx = (n % 11u32) as i32 - 5i32;
        let dy = (n / 11u32) as i32 - 5i32;
        let np = packed_at(pmap, photons, xres, yres, x + dx, y + dy);
        // Reject a tap with two cheap tests before the expensive effective
        // pixel mode runs: the glow diamond is `|dx| + |dy| <= 7`, and the
        // neighbour's cell must hold at least one `PMODE_GLOW` pixel (the base
        // pass records the same `effective_pixel_mode` result in `fx_cell`, so
        // this never drops a contribution).
        let mut glow_tap = false;
        if np != 0u32 && (dx.abs() + dy.abs()) <= 7i32 {
            let ncell = (((y + dy) as u32) >> 2u32) * (xres >> 2u32) + (((x + dx) as u32) >> 2u32);
            if (fx_cell[ncell as usize] & FX_PMGLOW) != 0u32 {
                glow_tap = true;
            }
        }
        if glow_tap {
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
                let wiped = owner_flat != 0u32 && npid < owner_id;
                if (nmode & PMODE_GLOW) != 0u32 && !wiped {
                    // `ad` is `|dx| + |dy|`; the 11x11 window already caps
                    // both axes at 5, so only the diamond test is left.
                    let ad = dx.abs() + dy.abs();
                    let mut alpha = 0.0f32;
                    if ad <= 7i32 {
                        let cola = particle_cola(
                            nty,
                            elem_props[(base + ELEM_COLA) as usize],
                            part_life[nslot],
                            part_ctype[nslot],
                            part_tmp[nslot],
                            part_tmp2[nslot],
                            part_tmp3[nslot],
                            part_tmp4[nslot],
                            colour_mode,
                            elem_props,
                            elem_count,
                        );
                        let (centre, ortho, ray) = glow_alpha(cola);
                        if ad == 0i32 {
                            alpha = centre;
                        } else if ad == 1i32 {
                            alpha = ortho;
                        } else {
                            alpha = ray;
                        }
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
                        dst = add_px(dst, rgb, alpha);
                    }
                }
            }
        }
    }
    dst
}

/// `PMODE_BLUR`: a 7x7 box with three overlapping blend passes (alpha 30 for
/// Manhattan distance 0..1, 20 for 1..3, 10 for exactly 2).
#[cube]
#[allow(clippy::too_many_arguments)]
fn blur_pixels(
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
    fx_cell: &[u32],
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
    tick: u32,
) -> u32 {
    let mut dst = dst_in;
    for n in 0u32..49u32 {
        let dx = (n % 7u32) as i32 - 3i32;
        let dy = (n / 7u32) as i32 - 3i32;
        let np = packed_at(pmap, photons, xres, yres, x + dx, y + dy);
        // Same two-step rejection as `glow_pixels`: only taps inside the blur
        // diamond (`|dx| + |dy| <= 3`) whose cell holds a `PMODE_BLUR` pixel.
        let mut blur_tap = false;
        if np != 0u32 && (dx.abs() + dy.abs()) <= 3i32 {
            let ncell = (((y + dy) as u32) >> 2u32) * (xres >> 2u32) + (((x + dx) as u32) >> 2u32);
            if (fx_cell[ncell as usize] & FX_PMBLUR) != 0u32 {
                blur_tap = true;
            }
        }
        if blur_tap {
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
                if (nmode & PMODE_BLUR) != 0u32 {
                    let ad = dx.abs() + dy.abs();
                    let mut a30 = 0.0f32;
                    let mut a20 = 0.0f32;
                    let mut a10 = 0.0f32;
                    if ad < 2i32 {
                        a30 = 30.0f32;
                    }
                    if ad <= 3i32 && ad != 0i32 {
                        a20 = 20.0f32;
                    }
                    if ad == 2i32 {
                        a10 = 10.0f32;
                    }
                    if a30 + a20 + a10 > 0.0f32 {
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
                        if a30 > 0.0f32 {
                            dst = blend_px(dst, rgb, a30);
                        }
                        if a20 > 0.0f32 {
                            dst = blend_px(dst, rgb, a20);
                        }
                        if a10 > 0.0f32 {
                            dst = blend_px(dst, rgb, a10);
                        }
                    }
                }
            }
        }
    }
    dst
}

/// Applies `PMODE_BLOB`, `PMODE_GLOW` and `PMODE_BLUR` around every particle.
///
/// TPT scatters these effects from each particle; here every pixel gathers the
/// contributions of the particles in its neighbourhood, with the same alpha
/// weights.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn render_particles_glow_kernel(
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
    fx_cell: &[u32],
    xres: u32,
    yres: u32,
    render_mode: u32,
    colour_mode: u32,
    decoration_level: u32,
    pretty_powder: u32,
    elem_count: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    tick: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xres * yres {
        let x = idx % xres;
        let y = idx / xres;
        // Nine taps per pixel: a neighbourhood without a blob/glow/blur pixel
        // cannot contribute, and most of a sand or steam canvas is exactly
        // that. The glow reach is five pixels, so the window is 5x5 cells.
        let xcells = xres >> 2u32;
        let ycells = yres >> 2u32;
        if !cell_fx_any(fx_cell, x >> 2u32, y >> 2u32, xcells, ycells, FX_GLOW, 2u32) {
            terminate!();
        }
        let mut dst = framebuf[idx as usize];
        if (render_mode & RENDER_BLOB) != 0u32 {
            dst = blob_pixels(
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
                tick,
            );
        }
        if (render_mode & RENDER_GLOW) != 0u32 {
            dst = glow_pixels(
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
                fx_cell,
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
                tick,
            );
        }
        if (render_mode & RENDER_BLUR) != 0u32 {
            dst = blur_pixels(
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
                fx_cell,
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
                tick,
            );
        }
        framebuf[idx as usize] = dst;
    }
}
