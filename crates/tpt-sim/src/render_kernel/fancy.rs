//! Fancy pixel effects: `BLOB`/`GLOW`/`BLUR` (TPT `render_parts` pixel effects).
//!
//! Each `16x16` output tile (`26x26` source tile with a 5-pixel halo) stages
//! packed particle metadata and pre-shaded `(rgb, cola)` into workgroup shared
//! memory once per source pixel, then evaluates the `BLOB` (3x3), `GLOW` (11x11
//! diamond), and `BLUR` (7x7 diamond) gathers strictly out of shared memory.

use cubecl::prelude::*;

use tpt_core::constants::*;

use super::framebuffer::{add_px, blend_px, seed_u32};
use super::particles::{
    effective_pixel_mode, fancy_view, glow_alpha, packed_at, particle_cola, particle_rgb,
    pipe_pixel_mode,
};
use super::{
    ELEM_COLA, ELEM_COLOUR, ELEM_HIGH_TEMP, ELEM_PIXEL_MODE, ELEM_PROPS, ELEM_WORDS, FX_GLOW,
    FX_PMBLOB, FX_PMBLUR, FX_PMGLOW, PMAP_ID_SHIFT, PMAP_TYPE_MASK, PMODE_BLOB, PMODE_BLUR,
    PMODE_FLAT, PMODE_GLOW,
};

const SH_TILE_SIDE: u32 = 26;
const SH_TILE_PIXELS: usize = (SH_TILE_SIDE * SH_TILE_SIDE) as usize;
const SH_PID_MASK: u32 = 0x007F_FFFF;
const SH_FLAT: u32 = 0x0100_0000;
const SH_BLOB: u32 = 0x0200_0000;
const SH_GLOW: u32 = 0x0400_0000;
const SH_BLUR: u32 = 0x0800_0000;

const _: () = assert!(crate::SimBuffers::MAX_PARTICLE_CAPACITY <= SH_PID_MASK as usize);

/// Loads and shades a single `(sx, sy)` source pixel for the tile cache.
///
/// Returns `(meta, rgb_cola)`: `meta == 0` when the pixel has none of
/// `PMODE_BLOB`/`PMODE_GLOW`/`PMODE_BLUR`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn eval_tile_pixel(
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
    xcells: u32,
    sx: i32,
    sy: i32,
    render_mode: u32,
    colour_mode: u32,
    decoration_level: u32,
    pretty_powder: u32,
    elem_count: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    tick: u32,
) -> (u32, u32) {
    let mut meta = seed_u32(xres, 0u32);
    let mut rgb_cola = seed_u32(xres, 0u32);
    if sx >= 0i32 && sy >= 0i32 && (sx as u32) < xres && (sy as u32) < yres {
        let scell = ((sy as u32) >> 2u32) * xcells + ((sx as u32) >> 2u32);
        if (fx_cell[scell as usize] & FX_GLOW) != 0u32 {
            let np = packed_at(pmap, photons, xres, yres, sx, sy);
            if np != 0u32 {
                let npid = np >> PMAP_ID_SHIFT;
                let nslot = (npid - 1u32) as usize;
                if npid != 0u32 && nslot < part_life.len() {
                    let nty = np & PMAP_TYPE_MASK;
                    let base = (nty % elem_count) * ELEM_WORDS;
                    let life = part_life[nslot];
                    let ctype = part_ctype[nslot];
                    let temp = part_temp[nslot];
                    let tmp = part_tmp[nslot];
                    let tmp2 = part_tmp2[nslot];
                    let tmp3 = part_tmp3[nslot];
                    let tmp4 = part_tmp4[nslot];
                    let vx = part_vx[nslot];
                    let vy = part_vy[nslot];
                    let npix = pipe_pixel_mode(
                        elem_props[(base + ELEM_PIXEL_MODE) as usize],
                        nty,
                        ctype,
                        elem_props,
                        elem_count,
                    );
                    let nmode = effective_pixel_mode(
                        npix,
                        nty,
                        life,
                        ctype,
                        temp,
                        tmp,
                        tmp2,
                        tmp3,
                        tmp4,
                        part_flags[nslot],
                        npid,
                        tick,
                        render_mode,
                        colour_mode,
                        vx,
                        vy,
                    );
                    if (nmode & (PMODE_BLOB | PMODE_GLOW | PMODE_BLUR)) != 0u32 {
                        meta = npid & SH_PID_MASK;
                        if (nmode & PMODE_FLAT) != 0u32 {
                            meta |= SH_FLAT;
                        }
                        if (nmode & PMODE_BLOB) != 0u32 {
                            meta |= SH_BLOB;
                        }
                        if (nmode & PMODE_GLOW) != 0u32 {
                            meta |= SH_GLOW;
                        }
                        if (nmode & PMODE_BLUR) != 0u32 {
                            meta |= SH_BLUR;
                        }
                        let rgb = particle_rgb(
                            nty,
                            elem_props[(base + ELEM_COLOUR) as usize],
                            elem_props[(base + ELEM_PROPS) as usize],
                            f32::reinterpret(elem_props[(base + ELEM_HIGH_TEMP) as usize]),
                            temp,
                            life,
                            ctype,
                            npid,
                            sx,
                            sy,
                            part_dcolour[nslot],
                            colour_mode,
                            pretty_powder,
                            nmode,
                            fancy_view(render_mode),
                            tmp,
                            tmp2,
                            tmp3,
                            tmp4,
                            vx,
                            vy,
                            tick,
                            hdisp_min,
                            hdisp_max,
                            decoration_level,
                            elem_props,
                            elem_count,
                        );
                        let mut cola = 255u32;
                        if (nmode & PMODE_GLOW) != 0u32 {
                            cola = particle_cola(
                                nty,
                                elem_props[(base + ELEM_COLA) as usize],
                                life,
                                ctype,
                                tmp,
                                tmp2,
                                tmp3,
                                tmp4,
                                colour_mode,
                                elem_props,
                                elem_count,
                            );
                        }
                        rgb_cola = ((cola & 0xFFu32) << 24u32) | (rgb & 0x00FF_FFFFu32);
                    }
                }
            }
        }
    }
    (meta, rgb_cola)
}

/// Resolves `(owner_id, owner_flat)` for the pixel at `(x, y)` so `PMODE_FLAT`
/// owners suppress glows from lower-slot particles (`Renderer.cpp:683-686`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn resolve_glow_owner(
    self_meta: u32,
    photons: &[u32],
    pmap: &[u32],
    part_flags: &[u32],
    part_life: &[i32],
    part_ctype: &[u32],
    part_temp: &[f32],
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
    elem_count: u32,
    tick: u32,
) -> (u32, u32) {
    let mut owner_id = seed_u32(xres, 0u32);
    let mut owner_flat = seed_u32(xres, 0u32);
    if (self_meta & SH_PID_MASK) != 0u32 {
        owner_id = self_meta & SH_PID_MASK;
        if (self_meta & SH_FLAT) != 0u32 {
            owner_flat = 1u32;
        }
    } else {
        let own = packed_at(pmap, photons, xres, yres, x, y);
        if own != 0u32 {
            owner_id = own >> PMAP_ID_SHIFT;
            let oslot = (owner_id - 1u32) as usize;
            if owner_id != 0u32 && oslot < part_life.len() {
                let oty = own & PMAP_TYPE_MASK;
                let obase = (oty % elem_count) * ELEM_WORDS;
                let opix = pipe_pixel_mode(
                    elem_props[(obase + ELEM_PIXEL_MODE) as usize],
                    oty,
                    part_ctype[oslot],
                    elem_props,
                    elem_count,
                );
                let omode = effective_pixel_mode(
                    opix,
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
    }
    (owner_id, owner_flat)
}

/// `PMODE_BLOB`: centre opaque, orthogonal neighbours alpha 223, diagonals 112.
#[cube]
fn blob_pixels_sh(
    dst_in: u32,
    sm_meta: &[u32],
    sm_rgb: &[u32],
    lx0: i32,
    ly0: i32,
) -> u32 {
    let mut dst = dst_in;
    let mut centre_rgb = 0u32;
    let mut has_centre = 0u32;
    for n in 0u32..9u32 {
        let dx = (n % 3u32) as i32 - 1i32;
        let dy = (n / 3u32) as i32 - 1i32;
        let sidx = ((ly0 + dy) * SH_TILE_SIDE as i32 + (lx0 + dx)) as usize;
        let meta = sm_meta[sidx];
        if (meta & SH_BLOB) != 0u32 {
            let rgb = (sm_rgb[sidx] & 0x00FF_FFFFu32) | 0xFF00_0000u32;
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
    if has_centre != 0u32 {
        dst = centre_rgb;
    }
    dst
}

/// `PMODE_GLOW`: 11x11 diamond (`|dx| + |dy| <= 7`) over shared tile cache.
#[cube]
fn glow_pixels_sh(
    dst_in: u32,
    sm_meta: &[u32],
    sm_rgb: &[u32],
    lx0: i32,
    ly0: i32,
    owner_id: u32,
    owner_flat: u32,
) -> u32 {
    let mut dst = dst_in;
    for n in 0u32..121u32 {
        let dx = (n % 11u32) as i32 - 5i32;
        let dy = (n / 11u32) as i32 - 5i32;
        let ad = dx.abs() + dy.abs();
        if ad <= 7i32 {
            let sidx = ((ly0 + dy) * SH_TILE_SIDE as i32 + (lx0 + dx)) as usize;
            let meta = sm_meta[sidx];
            if (meta & SH_GLOW) != 0u32 {
                let npid = meta & SH_PID_MASK;
                let wiped = owner_flat != 0u32 && npid < owner_id;
                if !wiped {
                    let word = sm_rgb[sidx];
                    let cola = (word >> 24u32) & 0xFFu32;
                    let (centre, ortho, ray) = glow_alpha(cola);
                    let mut alpha = ray;
                    if ad == 0i32 {
                        alpha = centre;
                    } else if ad == 1i32 {
                        alpha = ortho;
                    }
                    if alpha > 0.0f32 {
                        let rgb = (word & 0x00FF_FFFFu32) | 0xFF00_0000u32;
                        dst = add_px(dst, rgb, alpha);
                    }
                }
            }
        }
    }
    dst
}

/// `PMODE_BLUR`: 7x7 diamond (`|dx| + |dy| <= 3`) with alpha passes 30/20/10.
#[cube]
fn blur_pixels_sh(
    dst_in: u32,
    sm_meta: &[u32],
    sm_rgb: &[u32],
    lx0: i32,
    ly0: i32,
) -> u32 {
    let mut dst = dst_in;
    for n in 0u32..49u32 {
        let dx = (n % 7u32) as i32 - 3i32;
        let dy = (n / 7u32) as i32 - 3i32;
        let ad = dx.abs() + dy.abs();
        if ad <= 3i32 {
            let sidx = ((ly0 + dy) * SH_TILE_SIDE as i32 + (lx0 + dx)) as usize;
            let meta = sm_meta[sidx];
            if (meta & SH_BLUR) != 0u32 {
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
                let rgb = (sm_rgb[sidx] & 0x00FF_FFFFu32) | 0xFF00_0000u32;
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
    dst
}

/// Applies `PMODE_BLOB`, `PMODE_GLOW` and `PMODE_BLUR` using a 16x16 workgroup
/// over a 26x26 shared-memory tile.
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
    let tid = UNIT_POS_Y * 16u32 + UNIT_POS_X;
    let tile_x0 = (CUBE_POS_X * 16u32) as i32;
    let tile_y0 = (CUBE_POS_Y * 16u32) as i32;
    let xcells = xres >> 2u32;
    let ycells = yres >> 2u32;

    let mut sm_meta = Shared::<[u32]>::new_slice(SH_TILE_PIXELS);
    let mut sm_rgb = Shared::<[u32]>::new_slice(SH_TILE_PIXELS);
    let mut sm_tile_fx = Shared::<u32>::new();

    // First check the 8x8 coarse-cell box covering the 26x26 source tile.
    if tid < 64u32 {
        let scx = (tile_x0 >> 2i32) - 2i32 + (tid % 8u32) as i32;
        let scy = (tile_y0 >> 2i32) - 2i32 + (tid / 8u32) as i32;
        let mut cell_bits = seed_u32(xres, 0u32);
        if scx >= 0i32 && scy >= 0i32 && (scx as u32) < xcells && (scy as u32) < ycells {
            cell_bits = fx_cell[((scy as u32) * xcells + (scx as u32)) as usize];
        }
        sm_meta[tid as usize] = cell_bits;
    }
    sync_cube();
    if tid < 32u32 {
        sm_meta[tid as usize] |= sm_meta[(tid + 32u32) as usize];
    }
    sync_cube();
    if tid < 16u32 {
        sm_meta[tid as usize] |= sm_meta[(tid + 16u32) as usize];
    }
    sync_cube();
    if tid < 8u32 {
        sm_meta[tid as usize] |= sm_meta[(tid + 8u32) as usize];
    }
    sync_cube();
    if tid < 4u32 {
        sm_meta[tid as usize] |= sm_meta[(tid + 4u32) as usize];
    }
    sync_cube();
    if tid < 2u32 {
        sm_meta[tid as usize] |= sm_meta[(tid + 2u32) as usize];
    }
    sync_cube();
    if tid == 0u32 {
        *sm_tile_fx.inner_mut() = sm_meta[0usize] | sm_meta[1usize];
    }
    let tile_fx = workgroup_uniform_load(sm_tile_fx.inner_ref());
    if (tile_fx & FX_GLOW) == 0u32 {
        terminate!();
    }

    let need_blob = (render_mode & RENDER_BLOB) != 0u32 && (tile_fx & FX_PMBLOB) != 0u32;
    let need_glow = (render_mode & RENDER_GLOW) != 0u32 && (tile_fx & FX_PMGLOW) != 0u32;
    let need_blur = (render_mode & RENDER_BLUR) != 0u32 && (tile_fx & FX_PMBLUR) != 0u32;
    if !need_blob && !need_glow && !need_blur {
        terminate!();
    }

    // Cooperatively load and shade each source pixel in the 26x26 tile once.
    for pass in 0u32..3u32 {
        let k = pass * 256u32 + tid;
        if k < SH_TILE_SIDE * SH_TILE_SIDE {
            let lx = (k % SH_TILE_SIDE) as i32;
            let ly = (k / SH_TILE_SIDE) as i32;
            let (meta, rgb_cola) = eval_tile_pixel(
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
                xcells,
                tile_x0 - 5i32 + lx,
                tile_y0 - 5i32 + ly,
                render_mode,
                colour_mode,
                decoration_level,
                pretty_powder,
                elem_count,
                hdisp_min,
                hdisp_max,
                tick,
            );
            sm_meta[k as usize] = meta;
            sm_rgb[k as usize] = rgb_cola;
        }
    }
    sync_cube();

    let x = tile_x0 as u32 + UNIT_POS_X;
    let y = tile_y0 as u32 + UNIT_POS_Y;
    if x < xres && y < yres {
        let lx0 = (UNIT_POS_X + 5u32) as i32;
        let ly0 = (UNIT_POS_Y + 5u32) as i32;
        let idx = (y * xres + x) as usize;
        let dst_in = framebuf[idx];
        let mut dst = dst_in;
        if need_blob {
            dst = blob_pixels_sh(dst, sm_meta.as_slice(), sm_rgb.as_slice(), lx0, ly0);
        }
        if need_glow {
            let self_meta = sm_meta[(ly0 * SH_TILE_SIDE as i32 + lx0) as usize];
            let (owner_id, owner_flat) = resolve_glow_owner(
                self_meta,
                photons,
                pmap,
                part_flags,
                part_life,
                part_ctype,
                part_temp,
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
                elem_count,
                tick,
            );
            dst = glow_pixels_sh(
                dst,
                sm_meta.as_slice(),
                sm_rgb.as_slice(),
                lx0,
                ly0,
                owner_id,
                owner_flat,
            );
        }
        if need_blur {
            dst = blur_pixels_sh(dst, sm_meta.as_slice(), sm_rgb.as_slice(), lx0, ly0);
        }
        if dst != dst_in {
            framebuf[idx] = dst;
        }
    }
}
