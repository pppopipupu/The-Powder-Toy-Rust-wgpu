//! Background, air overlays and wall drawing (TPT `Renderer::Clear`, `draw_air`
//! and `DrawWalls`).
//!
//! The overlays are the `DISPLAY_AIR*` debug views. TPT's `draw_air` is an
//! `if / else if` chain in `AIRP`, `AIRV`, `AIRH`, `AIRC`, `AIRW` order
//! (`Renderer.cpp:981-1037`), so when several bits are set only the first one
//! that is enabled is painted. `DISPLAY_AIRH` with ambient heat disabled
//! returns from `draw_air` before *any* overlay is drawn
//! (`Renderer.cpp:969-970`).

use cubecl::prelude::*;

use tpt_core::constants::*;

use super::framebuffer::{
    absf, clampf, decay_px, fb_blue, fb_green, fb_red, heat_temp_colour, pack_rgb, solid_rgb,
};
use super::{
    STYLE_DOTS, STYLE_HOLE, STYLE_MESH, STYLE_SOLID, STYLE_STRIPES, W_ALLOWAIR, W_ALLOWALLELEC,
    W_ALLOWENERGY, W_ALLOWGAS, W_ALLOWLIQUID, W_ALLOWPOWDER, W_DETECT, W_EHOLE, W_EWALL, W_FAN,
    W_STASIS, W_WALL, W_WALLELEC, WALL_COLOUR, WALL_COUNT, WALL_EGLOW, WALL_GLOW_SHIFT, WALL_STYLE,
    WALL_WORDS,
};

/// `DISPLAY_AIRP`: TPT `PressureToColour` (`Renderer.cpp:925-932`).
///
/// Positive pressure is red, negative pressure is blue, and both are clamped to
/// `0..8` before TPT stores them in the `uint8_t` channels (`RGB`'s
/// constructor), so the channels truncate.
#[cube]
fn air_pressure(apv: f32) -> (f32, f32, f32) {
    let mut r = 0.0f32;
    let mut b = 0.0f32;
    if apv > 0.0f32 {
        r = (clampf(apv, 0.0f32, 8.0f32) as u32) as f32;
    } else {
        b = (clampf(-apv, 0.0f32, 8.0f32) as u32) as f32;
    }
    (r, 0.0f32, b)
}

/// `DISPLAY_AIRV`: raw air velocity and pressure (`Renderer.cpp:934-940`).
///
/// `vx` adds red, pressure adds green, `vy` adds blue; every channel is
/// `clamp(value, 0, 8)` and therefore truncates to `0..8`.
#[cube]
fn air_velocity(avx: f32, avy: f32, apv: f32) -> (f32, f32, f32) {
    let r = (clampf(absf(avx), 0.0f32, 8.0f32) as u32) as f32;
    let g = (clampf(apv, 0.0f32, 8.0f32) as u32) as f32;
    let b = (clampf(absf(avy), 0.0f32, 8.0f32) as u32) as f32;
    (r, g, b)
}

/// `DISPLAY_AIRH`: `HeatToColour` of the cell's heat (`Renderer.cpp:992-994`,
/// `1346-1351`).
///
/// `HeatToColour` indexes the heat table with `hdispLimitMin/Max` and then
/// multiplies every channel by `0.7f` with another `uint8_t` truncation.
#[cube]
fn air_heat(ahv: f32, hdisp_min: f32, hdisp_max: f32) -> (f32, f32, f32) {
    let col = heat_temp_colour(ahv, hdisp_min, hdisp_max);
    (
        (fb_red(col) * 0.7f32) as u32 as f32,
        (fb_green(col) * 0.7f32) as u32 as f32,
        (fb_blue(col) * 0.7f32) as u32 as f32,
    )
}

/// `DISPLAY_AIRC`: velocity magnitude in grey plus a pressure tint
/// (`Renderer.cpp:1008-1026`).
///
/// Red and blue sum the `24`/`20`-capped `|vx|` and `|vy|`; green swaps the
/// caps. Positive pressure adds up to `16` to red, negative pressure adds up to
/// `16` to blue, and the channels saturate at `255`. TPT stores the sums in
/// `int`s, so every intermediate truncates.
#[cube]
fn air_compress(avx: f32, avy: f32, apv: f32) -> (f32, f32, f32) {
    let ax = absf(avx);
    let ay = absf(avy);
    let base_rb = (clampf(ax, 0.0f32, 24.0f32) + clampf(ay, 0.0f32, 20.0f32)) as u32;
    let base_g = (clampf(ax, 0.0f32, 20.0f32) + clampf(ay, 0.0f32, 24.0f32)) as u32;
    let mut r = base_rb;
    let mut g = base_g;
    let mut b = base_rb;
    if apv > 0.0f32 {
        r = base_rb + clampf(apv, 0.0f32, 16.0f32) as u32;
    } else {
        b = base_rb + clampf(-apv, 0.0f32, 16.0f32) as u32;
    }
    if r > 255u32 {
        r = 255u32;
    }
    if g > 255u32 {
        g = 255u32;
    }
    if b > 255u32 {
        b = 255u32;
    }
    (r as f32, g as f32, b as f32)
}

/// `DISPLAY_AIRW`: vorticity of the coarse velocity field.
///
/// TPT `draw_air`: `w = 4 * Air::vorticity`, i.e. `2 * (dvy/dx - dvx/dy)` of
/// the coarse velocity field. Positive vorticity is red, negative is blue, and
/// the channels saturate at 8 like TPT's `clamp_flt(w, 0, 8)`.
#[cube]
fn air_vorticity(
    air_vx: &[f32],
    air_vy: &[f32],
    cx: u32,
    cy: u32,
    xcells: u32,
    ycells: u32,
) -> (f32, f32, f32) {
    let mut w = 0.0f32;
    if cx > 1u32 && cx + 2u32 < xcells && cy > 1u32 && cy + 2u32 < ycells {
        let dvydx =
            air_vy[(cy * xcells + cx + 1u32) as usize] - air_vy[(cy * xcells + cx - 1u32) as usize];
        let dvxdy = air_vx[((cy + 1u32) * xcells + cx) as usize]
            - air_vx[((cy - 1u32) * xcells + cx) as usize];
        w = (dvydx - dvxdy) * 0.5f32;
    }
    let wv = 4.0f32 * w;
    let mut r = 0.0f32;
    let g = 0.0f32;
    let mut b = 0.0f32;
    if wv > 0.0f32 {
        r = clampf(wv, 0.0f32, 8.0f32);
    } else {
        b = clampf(-wv, 0.0f32, 8.0f32);
    }
    (r, g, b)
}

/// Clears the framebuffer (or decays it for `DISPLAY_PERS`) and paints the air overlays.
#[cube(launch_unchecked)]
pub fn render_bg_kernel(
    framebuf: &mut [u32],
    air_vx: &[f32],
    air_vy: &[f32],
    air_pv: &[f32],
    air_hv: &[f32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    aheat_enable: u32,
    display_mode: u32,
    hdisp_min: f32,
    hdisp_max: f32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xres * yres {
        let x = idx % xres;
        let y = idx / xres;
        let cx = x >> 2u32;
        let cy = y >> 2u32;
        let cell = cy * xcells + cx;

        let mut rgb = solid_rgb(0.0f32);
        // `Renderer.cpp:30-33`: `RGB::Unpack(p).Decay()` — one off every
        // nonzero channel, not a multiply.
        if (display_mode & DISPLAY_PERS) != 0u32 {
            let prev = decay_px(framebuf[idx as usize]);
            rgb = (fb_red(prev), fb_green(prev), fb_blue(prev));
        }

        let avx = air_vx[cell as usize];
        let avy = air_vy[cell as usize];
        let apv = air_pv[cell as usize];
        let ahv = air_hv[cell as usize];

        // `Renderer.cpp:969-970`: `DISPLAY_AIRH` without ambient heat returns
        // from `draw_air` before any of the overlays run.
        if aheat_enable != 0u32 || (display_mode & DISPLAY_AIRH) == 0u32 {
            // `Renderer.cpp:981-1037` is an else-if chain, so the first
            // enabled view in this order wins.
            if (display_mode & DISPLAY_AIRP) != 0u32 {
                rgb = air_pressure(apv);
            } else if (display_mode & DISPLAY_AIRV) != 0u32 {
                rgb = air_velocity(avx, avy, apv);
            } else if (display_mode & DISPLAY_AIRH) != 0u32 {
                rgb = air_heat(ahv, hdisp_min, hdisp_max);
            } else if (display_mode & DISPLAY_AIRC) != 0u32 {
                rgb = air_compress(avx, avy, apv);
            } else if (display_mode & DISPLAY_AIRW) != 0u32 {
                rgb = air_vorticity(air_vx, air_vy, cx, cy, xcells, ycells);
            }
        }

        let (r, g, b) = rgb;
        framebuf[idx as usize] = pack_rgb(r, g, b);
    }
}

/// Resolve the draw pattern of a wall and return `1` when pixel `(x, y)` is painted.
///
/// The per-wall contract overrides the table's `drawstyle` for the walls whose
/// look is part of the Powder Toy identity (solid `WALL`, striped `FAN`/`DETECT`,
/// mesh `EWALL`/`STASIS`/`WALLELEC`/`ALLOW*`, dark `EHOLE`); everything else
/// falls back to the data-driven `drawstyle` from the wall table.
#[cube]
fn wall_pattern(wall_id: u32, drawstyle: u32, x: u32, y: u32) -> u32 {
    let mut style = drawstyle;
    if wall_id == W_WALL {
        style = STYLE_SOLID;
    } else if wall_id == W_FAN || wall_id == W_DETECT {
        // Both are drawn as diagonal stripes.
        style = STYLE_STRIPES;
    } else if wall_id == W_EHOLE {
        style = STYLE_HOLE;
    } else if wall_id == W_EWALL
        || wall_id == W_WALLELEC
        || wall_id == W_STASIS
        || wall_id == W_ALLOWLIQUID
        || wall_id == W_ALLOWAIR
        || wall_id == W_ALLOWPOWDER
        || wall_id == W_ALLOWALLELEC
        || wall_id == W_ALLOWGAS
        || wall_id == W_ALLOWENERGY
    {
        // Conductors, the stasis field and the "allow" filters share the mesh.
        style = STYLE_MESH;
    }

    let mut draw = 1u32;
    if style == STYLE_DOTS {
        draw = 0u32;
        if (x & 1u32) == 0u32 && (y & 1u32) == 0u32 {
            draw = 1u32;
        }
    } else if style == STYLE_MESH {
        draw = 0u32;
        if ((x + y) & 1u32) == 0u32 {
            draw = 1u32;
        }
    } else if style == STYLE_STRIPES {
        draw = 0u32;
        if ((x + y) & 3u32) < 2u32 {
            draw = 1u32;
        }
    } else if style == STYLE_HOLE {
        draw = 0u32;
        let lx = x & 3u32;
        let ly = y & 3u32;
        if lx == 0u32 || lx == 3u32 || ly == 0u32 || ly == 3u32 {
            draw = 1u32;
        }
    }
    draw
}

/// Draws the wall grid and feeds electrified walls into the coarse glow buffers.
#[cube(launch_unchecked)]
pub fn render_walls_kernel(
    framebuf: &mut [u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    wall_lut: &[u32],
    fire_r: &mut [u32],
    fire_g: &mut [u32],
    fire_b: &mut [u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    render_mode: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xres * yres {
        let x = idx % xres;
        let y = idx / xres;
        let cell = (y >> 2u32) * xcells + (x >> 2u32);

        let wid = wall_bmap[cell as usize] & 0xFFu32;
        if wid != 0u32 && wid < WALL_COUNT {
            let wbase = wid * WALL_WORDS;
            let colour = wall_lut[(wbase + WALL_COLOUR) as usize];
            let eglow = wall_lut[(wbase + WALL_EGLOW) as usize];
            let style = wall_lut[(wbase + WALL_STYLE) as usize];
            let emap = wall_emap[cell as usize] & 0xFFu32;
            let draw = wall_pattern(wid, style, x, y);

            let mut wr = ((colour >> 16u32) & 0xFFu32) as f32;
            let mut wg = ((colour >> 8u32) & 0xFFu32) as f32;
            let mut wb = (colour & 0xFFu32) as f32;
            if emap != 0u32 && eglow != 0u32 {
                // TPT `DrawWalls`: an electrified wall is drawn in its glow
                // colour and stores that colour in the fire buffers, so the
                // bloom pass lights it up.
                wr = ((eglow >> 16u32) & 0xFFu32) as f32;
                wg = ((eglow >> 8u32) & 0xFFu32) as f32;
                wb = (eglow & 0xFFu32) as f32;
                if (render_mode & (RENDER_FIRE | RENDER_SPRK)) != 0u32 {
                    let cr = (255u32 * (eglow >> 16u32)) >> WALL_GLOW_SHIFT;
                    let cg = (255u32 * ((eglow >> 8u32) & 0xFFu32)) >> WALL_GLOW_SHIFT;
                    let cb = (255u32 * (eglow & 0xFFu32)) >> WALL_GLOW_SHIFT;
                    fire_r[cell as usize] = cr;
                    fire_g[cell as usize] = cg;
                    fire_b[cell as usize] = cb;
                }
            }

            if draw != 0u32 {
                framebuf[idx as usize] = pack_rgb(wr, wg, wb);
            } else if wid == W_EHOLE {
                // hole interior: near black so the rim reads as a hole
                framebuf[idx as usize] = pack_rgb(6.0f32, 6.0f32, 8.0f32);
            } else if emap != 0u32 {
                // keep the pattern readable while powered
                framebuf[idx as usize] = pack_rgb(wr * 0.2f32, wg * 0.2f32, wb * 0.2f32);
            }
        }
    }
}
