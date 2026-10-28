//! Render passes for base particles, stickmen, and orbitals.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::framebuffer::{
    add_px, blend_px, fb_blue, fb_green, fb_red, pack_rgb, rgb_from_u32,
};
use super::super::{
    ELEM_COLA, ELEM_COLOUR, ELEM_HIGH_TEMP, ELEM_PIXEL_MODE, ELEM_PROPS, ELEM_WORDS, FX_GLOW,
    FX_PMBLUR, FX_PMGLOW, FX_SPARK, PMAP_ID_SHIFT, PMAP_TYPE_MASK, PMODE_ADD, PMODE_BLEND,
    PMODE_BLOB, PMODE_BLUR, PMODE_FLARE, PMODE_FLAT, PMODE_GLOW, PMODE_LFLARE, PMODE_SPARK,
};
use super::colour::{particle_cola, particle_rgb};
use super::elements_special::pipe_pixel_mode;
use super::pixel_mode::effective_pixel_mode;
use super::{EFFECT_LINES, packed_at};

/// TPT `Renderer.cpp:527-538` `EFFECT_LINES`: a `SOAP` particle attached at both
/// ends (`ctype & 3 == 3`, `SOAP.cpp:110-180`) draws a `BlendLine` from its own
/// cell to the partner stored in `tmp`, in the particle's own colour and `cola`.
///
/// TPT draws the line with `BlendLine` from `(nx, ny)` to `(parts[tmp].x, parts[tmp].y)`.
/// In our pixel-parallel pass, the bubble at `(px, py)` locates its bonded partner in
/// `pmap` within a bounded search window (radius 8, 17x17) and stamps the whole
/// segment using bounded Bresenham interpolation (capped at 16 steps so stretched
/// bonds cannot run away).
///
/// Along the segment, colour and alpha blend from this bubble towards the partner:
/// [`bond_colour_ramp`] and [`bond_alpha_ramp`].
#[cube]
pub(crate) fn bond_colour_ramp(c0: u32, c1: u32, step: u32, total_steps: u32) -> u32 {
    if total_steps == 0u32 {
        c0
    } else {
        let t = (step as f32) / (total_steps as f32);
        let inv = 1.0f32 - t;
        let r = inv * fb_red(c0) + t * fb_red(c1);
        let g = inv * fb_green(c0) + t * fb_green(c1);
        let b = inv * fb_blue(c0) + t * fb_blue(c1);
        pack_rgb(r, g, b)
    }
}

#[cube]
pub(crate) fn bond_alpha_ramp(a0: f32, a1: f32, step: u32, total_steps: u32) -> f32 {
    if total_steps == 0u32 {
        a0
    } else {
        let t = (step as f32) / (total_steps as f32);
        (1.0f32 - t) * a0 + t * a1
    }
}
#[cube]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(crate) fn soap_outline(
    dst_in: u32,
    own_rgb: u32,
    own_cola: u32,
    mode: u32,
    ty: u32,
    ctype: u32,
    tmp: i32,
    slot: u32,
    idx: u32,
    pmap: &[u32],
    photons: &[u32],
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
    xres: u32,
    yres: u32,
    colour_mode: u32,
    pretty_powder: u32,
    elem_count: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    decoration_level: u32,
    tick: u32,
) -> u32 {
    let mut dst = dst_in;
    if ty == PT_SOAP
        && (mode & EFFECT_LINES) != 0u32
        && (ctype & 3u32) == 3u32
        && tmp >= 0i32
        && (tmp as usize) < part_life.len()
    {
        let px = (idx % xres) as i32;
        let py = (idx / xres) as i32;
        let partner_slot = tmp as u32;

        let mut found = 0u32;
        let mut target_x = px;
        let mut target_y = py;

        // Search neighbourhood for partner bubble (radius 8, 17x17 window = 289 iterations).
        for n in 0u32..289u32 {
            let dx = (n % 17u32) as i32 - 8i32;
            let dy = (n / 17u32) as i32 - 8i32;
            if dx != 0i32 || dy != 0i32 {
                let np = packed_at(pmap, photons, xres, yres, px + dx, py + dy);
                if np != 0u32 && (np & PMAP_TYPE_MASK) == PT_SOAP {
                    let npid = np >> PMAP_ID_SHIFT;
                    if npid != 0u32 && (npid - 1u32) == partner_slot {
                        found = 1u32;
                        target_x = px + dx;
                        target_y = py + dy;
                    }
                }
            }
        }

        if found != 0u32 {
            let pslot = partner_slot as usize;
            let nbase = (PT_SOAP % elem_count) * ELEM_WORDS;
            let p_colour = elem_props[(nbase + ELEM_COLOUR) as usize];
            let p_props = elem_props[(nbase + ELEM_PROPS) as usize];
            let p_high_temp = f32::reinterpret(elem_props[(nbase + ELEM_HIGH_TEMP) as usize]);
            let p_table_cola = elem_props[(nbase + ELEM_COLA) as usize];

            let p_rgb = particle_rgb(
                PT_SOAP,
                p_colour,
                p_props,
                p_high_temp,
                part_temp[pslot],
                part_life[pslot],
                part_ctype[pslot],
                partner_slot + 1u32,
                part_dcolour[pslot],
                colour_mode,
                pretty_powder,
                mode,
                part_tmp[pslot],
                part_tmp2[pslot],
                part_tmp3[pslot],
                part_tmp4[pslot],
                part_vx[pslot],
                part_vy[pslot],
                tick,
                hdisp_min,
                hdisp_max,
                decoration_level,
                elem_props,
                elem_count,
            );
            let p_cola = particle_cola(
                PT_SOAP,
                p_table_cola,
                part_life[pslot],
                part_ctype[pslot],
                part_tmp[pslot],
                part_tmp2[pslot],
                part_tmp3[pslot],
                part_tmp4[pslot],
                colour_mode,
                elem_props,
                elem_count,
            ) as f32;

            let dx = target_x - px;
            let dy = target_y - py;
            let adx = if dx >= 0i32 { dx } else { -dx };
            let ady = if dy >= 0i32 { dy } else { -dy };
            let mut steps = if adx > ady { adx } else { ady };
            // Loop is strictly guarded so a stretched bond cannot run away (cap at 16 steps).
            if steps > 16i32 {
                steps = 16i32;
            }

            // Step 0 blends the near endpoint into the current pixel's dst.
            dst = blend_px(dst, own_rgb, own_cola as f32);

            // Steps 1..=steps stamp the interior and the far endpoint.
            if steps > 0i32 {
                let x_bias = if target_x >= px { steps } else { -steps };
                let y_bias = if target_y >= py { steps } else { -steps };
                for s in 1..17 {
                    let si = s;
                    if si <= steps {
                        let px_s = px + ((target_x - px) * 2i32 * si + x_bias) / (2i32 * steps);
                        let py_s = py + ((target_y - py) * 2i32 * si + y_bias) / (2i32 * steps);
                        if px_s >= 0i32 && px_s < xres as i32 && py_s >= 0i32 && py_s < yres as i32
                        {
                            let pidx = (py_s as u32 * xres + px_s as u32) as usize;
                            let s_rgb = bond_colour_ramp(own_rgb, p_rgb, s as u32, steps as u32);
                            let s_alpha =
                                bond_alpha_ramp(own_cola as f32, p_cola, s as u32, steps as u32);
                            framebuf[pidx] = blend_px(framebuf[pidx], s_rgb, s_alpha);
                        }
                    }
                }
            }
        } else {
            dst = blend_px(dst, own_rgb, own_cola as f32);
        }
    }
    dst
}

/// Colours every particle pixel: `PMODE_FLAT`, `PMODE_BLEND` and `PMODE_ADD`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn render_particles_base_kernel(
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
    fx_cell: &mut [Atomic<u32>],
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
        // The energy layer is drawn over the matter layer.
        let mut packed = photons[idx as usize];
        if packed == 0u32 {
            packed = pmap[idx as usize];
        }
        if packed != 0u32 {
            let pid = packed >> PMAP_ID_SHIFT;
            if pid != 0u32 {
                let slot = (pid - 1u32) as usize;
                if slot < part_life.len() {
                    let ty = packed & PMAP_TYPE_MASK;
                    // Element ids are `< PT_NUM`, but a stale map must never index
                    // outside the uploaded property table.
                    let lut_ty = ty % elem_count;
                    let base = lut_ty * ELEM_WORDS;
                    let colour = elem_props[(base + ELEM_COLOUR) as usize];
                    let props = elem_props[(base + ELEM_PROPS) as usize];
                    let high_temp = f32::reinterpret(elem_props[(base + ELEM_HIGH_TEMP) as usize]);

                    let life = part_life[slot];
                    let temp = part_temp[slot];
                    let ctype = part_ctype[slot];
                    let dcolour = part_dcolour[slot];
                    let flags = part_flags[slot];
                    let tmp = part_tmp[slot];
                    let tmp2 = part_tmp2[slot];
                    let tmp3 = part_tmp3[slot];
                    let tmp4 = part_tmp4[slot];
                    let vx = part_vx[slot];
                    let vy = part_vy[slot];

                    // `PIPE.cpp:349-352`: a loaded pipe draws with the cargo's
                    // mode, so the cargo's row is resolved before the dynamic
                    // rules run.
                    let pixel_mode = pipe_pixel_mode(
                        elem_props[(base + ELEM_PIXEL_MODE) as usize],
                        ty,
                        ctype,
                        elem_props,
                        elem_count,
                    );

                    let mode = effective_pixel_mode(
                        pixel_mode,
                        ty,
                        life,
                        ctype,
                        temp,
                        tmp,
                        tmp2,
                        tmp3,
                        tmp4,
                        flags,
                        pid,
                        tick,
                        render_mode,
                        colour_mode,
                        vx,
                        vy,
                    );
                    // Mark the cell for the nine-tap glow and spark gathers.
                    // `fetch_or` because the sixteen pixels of a cell report
                    // concurrently, and one particle may need both bits.
                    let cell = ((idx / xres) >> 2u32) * (xres >> 2u32) + ((idx % xres) >> 2u32);
                    let mut fx = 0u32;
                    if (mode & (PMODE_BLOB | PMODE_GLOW | PMODE_BLUR)) != 0u32 {
                        fx |= FX_GLOW;
                    }
                    if (mode & PMODE_GLOW) != 0u32 {
                        fx |= FX_PMGLOW;
                    }
                    if (mode & PMODE_BLUR) != 0u32 {
                        fx |= FX_PMBLUR;
                    }
                    if (mode & (PMODE_SPARK | PMODE_FLARE | PMODE_LFLARE)) != 0u32 {
                        fx |= FX_SPARK;
                    }
                    if fx != 0u32 {
                        fx_cell[cell as usize].fetch_or(fx);
                    }
                    let rgb = particle_rgb(
                        ty,
                        colour,
                        props,
                        high_temp,
                        temp,
                        life,
                        ctype,
                        pid,
                        dcolour,
                        colour_mode,
                        pretty_powder,
                        mode,
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
                    // TPT composites `PMODE_BLEND`/`PMODE_ADD` with `cola`;
                    // `PMODE_FLAT` ignores it.
                    let cola = particle_cola(
                        ty,
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
                    ) as f32;
                    let mut dst = framebuf[idx as usize];
                    // `Renderer.cpp:527-538`: the `EFFECT_LINES` bond is drawn
                    // before the particle's own pixel mode.
                    dst = soap_outline(
                        dst,
                        rgb,
                        cola as u32,
                        mode,
                        ty,
                        ctype,
                        tmp,
                        slot as u32,
                        idx,
                        pmap,
                        photons,
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
                        framebuf,
                        xres,
                        yres,
                        colour_mode,
                        pretty_powder,
                        elem_count,
                        hdisp_min,
                        hdisp_max,
                        decoration_level,
                        tick,
                    );
                    // TPT order: FLAT overwrites, then BLEND, then ADD.
                    if (mode & PMODE_FLAT) != 0u32 {
                        dst = rgb;
                    }
                    if (mode & PMODE_BLEND) != 0u32 {
                        dst = blend_px(dst, rgb, cola);
                    }
                    if (mode & PMODE_ADD) != 0u32 {
                        dst = add_px(dst, rgb, cola);
                    }
                    framebuf[idx as usize] = dst;
                }
            }
        }
    }
}

const R_PT_FIGH: u32 = 159;
const R_PT_STKM: u32 = 59;
const R_PT_STKM2: u32 = 137;
const R_PT_PRTI: u32 = 118;
const R_PT_PRTO: u32 = 119;

const _: () = assert!(R_PT_FIGH == PT_FIGH);
const _: () = assert!(R_PT_STKM == PT_STKM);
const _: () = assert!(R_PT_STKM2 == PT_STKM2);
const _: () = assert!(R_PT_PRTI == PT_PRTI);
const _: () = assert!(R_PT_PRTO == PT_PRTO);

#[cube]
pub(crate) fn plot_line(
    framebuf: &mut [u32],
    xres: u32,
    yres: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: u32,
) {
    let adx = if x1 >= x0 { x1 - x0 } else { x0 - x1 };
    let ady = if y1 >= y0 { y1 - y0 } else { y0 - y1 };
    let mut steps = if adx > ady { adx } else { ady };
    if steps > 31i32 {
        steps = 31i32;
    }
    if steps == 0i32 {
        if x0 >= 0i32 && x0 < xres as i32 && y0 >= 0i32 && y0 < yres as i32 {
            framebuf[(y0 as u32 * xres + x0 as u32) as usize] = color;
        }
    } else {
        let x_bias = if x1 >= x0 { steps } else { -steps };
        let y_bias = if y1 >= y0 { steps } else { -steps };
        for s in 0..32 {
            let si = s;
            if si <= steps {
                let px = x0 + ((x1 - x0) * 2i32 * si + x_bias) / (2i32 * steps);
                let py = y0 + ((y1 - y0) * 2i32 * si + y_bias) / (2i32 * steps);
                if px >= 0i32 && px < xres as i32 && py >= 0i32 && py < yres as i32 {
                    framebuf[(py as u32 * xres + px as u32) as usize] = color;
                }
            }
        }
    }
}

/// Slot-parallel rendering of stickman rigs (`FIGH` / `STKM` / `STKM2`) and portal orbital dots (`PRTI` / `PRTO`).
///
/// TPT `Renderer.cpp:793-828`: the dots are an `EFFECT_GRAVIN`/`EFFECT_GRAVOUT`
/// effect, so the render mask keeps them only under `RENDER_EFFE`
/// (`ElementGraphics.h:32`); the target cell is read from `pmap` like TPT's
/// `TYP(sim->pmap[y + nyo][x + nxo])`.
#[cube(launch_unchecked)]
#[allow(
    clippy::identity_op,
    clippy::manual_clamp,
    clippy::too_many_arguments,
    clippy::approx_constant
)]
pub fn render_stickmen_and_orbitals_kernel(
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_life: &[i32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    part_temp: &[f32],
    stkm_legs: &[f32],
    stkm_meta: &[u32],
    elem_props: &[u32],
    framebuf: &mut [u32],
    xres: u32,
    yres: u32,
    capacity: u32,
    elem_count: u32,
    render_mode: u32,
    pmap: &[u32],
    hover_x: i32,
    hover_y: i32,
    debug_lines: u32,
    persistent: u32,
) {
    let id = ABSOLUTE_POS as u32;
    if id >= capacity {
        terminate!();
    }
    let ty = part_ty[id as usize];
    if ty != R_PT_FIGH && ty != R_PT_STKM && ty != R_PT_STKM2 && ty != R_PT_PRTI && ty != R_PT_PRTO
    {
        terminate!();
    }

    if ty == R_PT_FIGH || ty == R_PT_STKM || ty == R_PT_STKM2 {
        let sid = if ty == R_PT_STKM {
            100usize
        } else if ty == R_PT_STKM2 {
            101usize
        } else {
            let tmp = part_tmp[id as usize];
            let ctmp = if tmp < 0i32 {
                0i32
            } else if tmp > 99i32 {
                99i32
            } else {
                tmp
            };
            ctmp as usize
        };

        let elem = stkm_meta[sid * 8usize + 3usize];
        let head_rgb = if elem > 0u32 && elem < elem_count {
            let c = elem_props[(elem * ELEM_WORDS + ELEM_COLOUR) as usize];
            let (cr, cg, cb) = rgb_from_u32(c);
            pack_rgb(cr, cg, cb)
        } else if ty == R_PT_STKM2 {
            pack_rgb(100.0f32, 100.0f32, 255.0f32)
        } else if ty == R_PT_FIGH {
            pack_rgb(255.0f32, 224.0f32, 160.0f32)
        } else {
            pack_rgb(128.0f32, 128.0f32, 255.0f32)
        };

        let leg_rgb = if ty == R_PT_STKM2 {
            pack_rgb(100.0f32, 100.0f32, 255.0f32)
        } else {
            pack_rgb(255.0f32, 255.0f32, 255.0f32)
        };

        let x = (part_x[id as usize] + 0.5f32) as i32;
        let y = (part_y[id as usize] + 0.5f32) as i32;

        // Head
        if ty == R_PT_FIGH {
            plot_line(framebuf, xres, yres, x, y + 2i32, x + 2i32, y, head_rgb);
            plot_line(framebuf, xres, yres, x + 2i32, y, x, y - 2i32, head_rgb);
            plot_line(framebuf, xres, yres, x, y - 2i32, x - 2i32, y, head_rgb);
            plot_line(framebuf, xres, yres, x - 2i32, y, x, y + 2i32, head_rgb);
        } else {
            plot_line(
                framebuf,
                xres,
                yres,
                x - 2i32,
                y + 2i32,
                x + 2i32,
                y + 2i32,
                head_rgb,
            );
            plot_line(
                framebuf,
                xres,
                yres,
                x - 2i32,
                y - 2i32,
                x + 2i32,
                y - 2i32,
                head_rgb,
            );
            plot_line(
                framebuf,
                xres,
                yres,
                x - 2i32,
                y - 2i32,
                x - 2i32,
                y + 2i32,
                head_rgb,
            );
            plot_line(
                framebuf,
                xres,
                yres,
                x + 2i32,
                y - 2i32,
                x + 2i32,
                y + 2i32,
                head_rgb,
            );
        }

        // Legs: 4 line segments
        let lb = sid * 16usize;
        let l0 = (stkm_legs[lb + 0usize] + 0.5f32) as i32;
        let l1 = (stkm_legs[lb + 1usize] + 0.5f32) as i32;
        let l4 = (stkm_legs[lb + 4usize] + 0.5f32) as i32;
        let l5 = (stkm_legs[lb + 5usize] + 0.5f32) as i32;
        let l8 = (stkm_legs[lb + 8usize] + 0.5f32) as i32;
        let l9 = (stkm_legs[lb + 9usize] + 0.5f32) as i32;
        let l12 = (stkm_legs[lb + 12usize] + 0.5f32) as i32;
        let l13 = (stkm_legs[lb + 13usize] + 0.5f32) as i32;

        // `Renderer.cpp:628-631`: the legs hang off `(nx, ny+3)`.
        plot_line(framebuf, xres, yres, x, y + 3i32, l0, l1, leg_rgb);
        plot_line(framebuf, xres, yres, l0, l1, l4, l5, leg_rgb);
        plot_line(framebuf, xres, yres, x, y + 3i32, l8, l9, leg_rgb);
        plot_line(framebuf, xres, yres, l8, l9, l12, l13, leg_rgb);
    } else {
        let x = (part_x[id as usize] + 0.5f32) as i32;
        let y = (part_y[id as usize] + 0.5f32) as i32;
        // TPT `Renderer.cpp:793-828` `EFFECT_GRAVIN` / `EFFECT_GRAVOUT`: the
        // dots are an `EFFECT` bit, so `render_parts` keeps them only when the
        // render mode contains `RENDER_EFFE` (`ElementGraphics.h:32`).
        if (render_mode & RENDER_EFFE) != 0u32 {
            let life_u = part_life[id as usize] as u32;
            let ctype_u = part_ctype[id as usize];

            let c = elem_props[(ty * ELEM_WORDS + ELEM_COLOUR) as usize];
            let (colr, colg, colb) = rgb_from_u32(c);
            let orb_rgb = pack_rgb(colr, colg, colb);

            for r in 0u32..4u32 {
                let shift = r * 8u32;
                let orbd = (life_u >> shift) & 0xFFu32;
                let orbl = (ctype_u >> shift) & 0xFFu32;

                if orbd > 0u32 {
                    let rad = (orbl as f32) * 0.024_683_545f32;
                    let dist = (orbd as f32) / 16.0f32;

                    let nxo = (rad.cos() * dist) as i32;
                    let nyo = (rad.sin() * dist) as i32;

                    let px = x + nxo;
                    let py = y + nyo;

                    // `Renderer.cpp:807` / `Renderer.cpp:825`: the bounds test
                    // is strict (`> 0`) and a cell already holding a portal of
                    // the same type (`PT_PRTI` / `PT_PRTO`) is skipped.
                    if px > 0i32 && px < xres as i32 && py > 0i32 && py < yres as i32 {
                        let pidx = (py as u32 * xres + px as u32) as usize;
                        if (pmap[pidx] & PMAP_TYPE_MASK) != ty {
                            let alpha = (255u32 - orbd) as f32;
                            framebuf[pidx] = add_px(framebuf[pidx], orb_rgb, alpha);
                        }
                    }
                }
            }
        }

        // `Renderer.cpp:829-847` `EFFECT_DBGLINES`: while the debug HUD is on
        // and the pointer rests on a portal, draw a line to every portal of the
        // matching channel. Only the particle that owns the hovered pixel
        // draws, so an overlapping pair does not double the line.
        if debug_lines != 0u32
            && persistent == 0u32
            && x == hover_x
            && y == hover_y
            && pmap[(y as u32 * xres + x as u32) as usize]
                == (((id + 1u32) << 9u32) | (ty & 0x1FFu32))
        {
            let other = if ty == R_PT_PRTI {
                R_PT_PRTO
            } else {
                R_PT_PRTI
            };
            let channel = portal_channel(part_temp[id as usize]);
            for z in 0u32..capacity {
                if part_ty[z as usize] == other && portal_channel(part_temp[z as usize]) == channel
                {
                    xor_line(
                        framebuf,
                        xres,
                        yres,
                        x,
                        y,
                        (part_x[z as usize] + 0.5f32) as i32,
                        (part_y[z as usize] + 0.5f32) as i32,
                    );
                }
            }
        }
    }
}

/// `Renderer.cpp:833`: a portal's channel is `(temp - 73.15) / 100 + 1`.
#[cube]
pub(crate) fn portal_channel(temp: f32) -> i32 {
    ((temp - 73.15f32) / 100.0f32 + 1.0f32) as i32
}

/// `RasterDrawMethods::XorLine` (`RasterDrawMethodsImpl.h:100`).
#[cube]
pub(crate) fn xor_line(
    framebuf: &mut [u32],
    xres: u32,
    yres: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
) {
    let dx = if x1 >= x0 { x1 - x0 } else { x0 - x1 };
    let dy = if y1 >= y0 { y1 - y0 } else { y0 - y1 };
    let steps = if dx > dy { dx } else { dy };
    if steps == 0i32 {
        xor_pixel(framebuf, xres, yres, x0, y0);
    } else {
        let x_bias = if x1 >= x0 { steps } else { -steps };
        let y_bias = if y1 >= y0 { steps } else { -steps };
        for s in 0..4096u32 {
            let si = s as i32;
            if si <= steps {
                let px = x0 + ((x1 - x0) * 2i32 * si + x_bias) / (2i32 * steps);
                let py = y0 + ((y1 - y0) * 2i32 * si + y_bias) / (2i32 * steps);
                xor_pixel(framebuf, xres, yres, px, py);
            }
        }
    }
}

/// `xorPixelUnchecked` (`RasterDrawMethodsImpl.h:24`): a dark pixel turns
/// light grey, a bright one dark grey, so the line reads on both backgrounds.
#[cube]
fn xor_pixel(framebuf: &mut [u32], xres: u32, yres: u32, x: i32, y: i32) {
    if x >= 0i32 && x < xres as i32 && y >= 0i32 && y < yres as i32 {
        let idx = (y as u32 * xres + x as u32) as usize;
        let px = framebuf[idx];
        let r = px & 0xFFu32;
        let g = (px >> 8u32) & 0xFFu32;
        let b = (px >> 16u32) & 0xFFu32;
        if 2u32 * r + 3u32 * g + b < 512u32 {
            framebuf[idx] = 0xFFC0_C0C0u32;
        } else {
            framebuf[idx] = 0xFF40_4040u32;
        }
    }
}

#[cfg(test)]
pub fn host_bond_alpha_ramp(a0: f32, a1: f32, step: u32, total_steps: u32) -> f32 {
    if total_steps == 0 {
        a0
    } else {
        let t = (step as f32) / (total_steps as f32);
        (1.0 - t) * a0 + t * a1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soap_bond_alpha_ramp() {
        assert_eq!(host_bond_alpha_ramp(100.0, 200.0, 0, 10), 100.0);
        assert_eq!(host_bond_alpha_ramp(100.0, 200.0, 5, 10), 150.0);
        assert_eq!(host_bond_alpha_ramp(100.0, 200.0, 10, 10), 200.0);
        assert_eq!(host_bond_alpha_ramp(255.0, 255.0, 3, 10), 255.0);
        assert_eq!(host_bond_alpha_ramp(50.0, 150.0, 0, 0), 50.0);
    }
}
