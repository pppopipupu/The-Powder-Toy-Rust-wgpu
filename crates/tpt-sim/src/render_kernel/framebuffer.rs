//! Framebuffer packing, blending and colour-gradient helpers.
//!
//! Every item here is a `#[cube]` function shared by the render passes in
//! [`super`]; none of them touches device memory. The framebuffer packs
//! `0xAABBGGRR`, i.e. the little-endian byte order `[R, G, B, A]` that
//! `wgpu::TextureFormat::Rgba8Unorm` expects.
//!
//! A colour in flight between the particle stages is a plain `(r, g, b)` tuple
//! of `f32` channels in `0..255` that has *not* been clamped or rounded yet;
//! only the final [`pack_rgb`] quantises it, exactly like TPT's `RGB` struct.

// Clippy's rewrite for these lints uses methods the device IR does not have
// (`is_multiple_of`, `clamp`, `Range::contains`, `saturating_sub`), and
// collapsing a nested `if` changes what the CubeCL macro expands, so the plain
// operators are deliberate.
#![allow(clippy::implicit_saturating_sub)]

use cubecl::prelude::*;

/// Clamp `v` into `[lo, hi]`.
#[cube]
pub(crate) fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut out = v;
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

/// Clamp `v` into `[lo, hi]`.
///
/// Integer twin of [`clampf`]; `i32::clamp` is not available in kernels.
#[cube]
pub(crate) fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    let mut out = v;
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

/// Linear interpolation between `a` and `b`.
#[cube]
pub(crate) fn mixf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smaller of two floats.
#[cube]
pub(crate) fn minf(a: f32, b: f32) -> f32 {
    let mut out = a;
    if b < a {
        out = b;
    }
    out
}

/// Absolute value (`f32::abs` is not available in kernels).
#[cube]
pub(crate) fn absf(v: f32) -> f32 {
    if v < 0.0f32 { -v } else { v }
}

/// Pack three `0..255` channels into `0xAABBGGRR` with an opaque alpha.
#[cube]
pub(crate) fn pack_rgb(r: f32, g: f32, b: f32) -> u32 {
    let ri = clampf(r, 0.0f32, 255.0f32) as u32;
    let gi = clampf(g, 0.0f32, 255.0f32) as u32;
    let bi = clampf(b, 0.0f32, 255.0f32) as u32;
    (255u32 << 24u32) | (bi << 16u32) | (gi << 8u32) | ri
}

/// Pack `0xAARRGGBB` (the `fire_colour` table layout).
#[cube]
pub(crate) fn pack_argb(a: u32, r: u32, g: u32, b: u32) -> u32 {
    (a << 24u32) | (r << 16u32) | (g << 8u32) | b
}

/// Red channel of a packed framebuffer pixel.
#[cube]
pub(crate) fn fb_red(px: u32) -> f32 {
    (px & 0xFFu32) as f32
}

/// Green channel of a packed framebuffer pixel.
#[cube]
pub(crate) fn fb_green(px: u32) -> f32 {
    ((px >> 8u32) & 0xFFu32) as f32
}

/// Blue channel of a packed framebuffer pixel.
#[cube]
pub(crate) fn fb_blue(px: u32) -> f32 {
    ((px >> 16u32) & 0xFFu32) as f32
}

/// TPT `RGB::Decay` (`Pixel.h:52-60`): decrement every nonzero channel.
///
/// `Renderer.cpp:30-33` decays the whole framebuffer by one per channel for
/// `DISPLAY_PERS`; TPT moves the alpha byte out of the packed colour first, so
/// it is preserved here too.
#[cube]
pub(crate) fn decay_px(px: u32) -> u32 {
    let mut r = px & 0xFFu32;
    let mut g = (px >> 8u32) & 0xFFu32;
    let mut b = (px >> 16u32) & 0xFFu32;
    if r > 0u32 {
        r -= 1u32;
    }
    if g > 0u32 {
        g -= 1u32;
    }
    if b > 0u32 {
        b -= 1u32;
    }
    (px & 0xFF00_0000u32) | (b << 16u32) | (g << 8u32) | r
}

/// A single flat colour, i.e. the same value in all three channels.
#[cube]
pub(crate) fn solid_rgb(v: f32) -> (f32, f32, f32) {
    (v, v, v)
}

/// Unpack a `0x00RRGGBB` colour into unclamped channels.
#[cube]
pub(crate) fn rgb_from_u32(c: u32) -> (f32, f32, f32) {
    (
        ((c >> 16u32) & 0xFFu32) as f32,
        ((c >> 8u32) & 0xFFu32) as f32,
        (c & 0xFFu32) as f32,
    )
}

/// TPT `RGB::Blend`: `(a * src + (255 - a) * dst) / 255`.
#[cube]
pub(crate) fn blend_px(dst: u32, rgb: u32, a: f32) -> u32 {
    let inv = 255.0f32 - a;
    let r = (a * fb_red(rgb) + inv * fb_red(dst)) / 255.0f32;
    let g = (a * fb_green(rgb) + inv * fb_green(dst)) / 255.0f32;
    let b = (a * fb_blue(rgb) + inv * fb_blue(dst)) / 255.0f32;
    pack_rgb(r, g, b)
}

/// TPT `RGB::Add`: `min(255, dst + a * src / 255)`.
#[cube]
pub(crate) fn add_px(dst: u32, rgb: u32, a: f32) -> u32 {
    let r = fb_red(dst) + a * fb_red(rgb) / 255.0f32;
    let g = fb_green(dst) + a * fb_green(rgb) / 255.0f32;
    let b = fb_blue(dst) + a * fb_blue(rgb) / 255.0f32;
    pack_rgb(r, g, b)
}

/// TPT `RGB::AddFire`: `min(255, dst + fireAlpha * src / 255)`.
#[cube]
pub(crate) fn add_fire_px(dst: u32, r: f32, g: f32, b: f32, a: f32) -> u32 {
    let nr = fb_red(dst) + a * r / 255.0f32;
    let ng = fb_green(dst) + a * g / 255.0f32;
    let nb = fb_blue(dst) + a * b / 255.0f32;
    pack_rgb(nr, ng, nb)
}

/// Blend two `0x00RRGGBB` colours; `t = 0` returns `a`, `t = 1` returns `b`.
#[cube]
pub(crate) fn lerp_rgb(a: u32, b: u32, t: f32) -> u32 {
    let r = mixf(
        ((a >> 16u32) & 0xFFu32) as f32,
        ((b >> 16u32) & 0xFFu32) as f32,
        t,
    );
    let g = mixf(
        ((a >> 8u32) & 0xFFu32) as f32,
        ((b >> 8u32) & 0xFFu32) as f32,
        t,
    );
    let bl = mixf((a & 0xFFu32) as f32, (b & 0xFFu32) as f32, t);
    ((r as u32) << 16u32) | ((g as u32) << 8u32) | (bl as u32)
}

/// Deterministic `0.0..1.0` hash of a particle id.
#[cube]
pub(crate) fn hash01(seed: u32) -> f32 {
    let mut x = seed * 747796405u32 + 2891336453u32;
    x = (x ^ (x >> 16u32)) * 2246822519u32;
    x = (x ^ (x >> 13u32)) * 3266489917u32;
    x = x ^ (x >> 16u32);
    let m = (x & 0x0000FFFFu32) as f32;
    m / 65535.0f32
}

/// TPT `Renderer::flameTableAt`: flame gradient, 200 entries indexed by life.
///
/// Stops: `0x000000 @ 0`, `0x60300F @ 0.5`, `0xDFBF6F @ 0.9`, `0xAF9F0F @ 1`.
#[cube]
pub(crate) fn flame_colour(life: i32) -> u32 {
    let l = clampi(life, 0i32, 199i32);
    let t = (l as f32) / 200.0f32;
    if t < 0.5f32 {
        lerp_rgb(0x000000u32, 0x60300Fu32, t / 0.5f32)
    } else if t < 0.9f32 {
        lerp_rgb(0x60300Fu32, 0xDFBF6Fu32, (t - 0.5f32) / 0.4f32)
    } else {
        lerp_rgb(0xDFBF6Fu32, 0xAF9F0Fu32, (t - 0.9f32) / 0.1f32)
    }
}

/// TPT `Renderer::plasmaTableAt`: plasma gradient, 200 entries indexed by life.
///
/// Stops: `0x000000 @ 0`, `0x301040 @ 0.25`, `0x301060 @ 0.5`,
/// `0xAFFFFF @ 0.9`, `0xAFFFFF @ 1`.
#[cube]
pub(crate) fn plasma_colour(life: i32) -> u32 {
    let l = clampi(life, 0i32, 199i32);
    let t = (l as f32) / 200.0f32;
    if t < 0.25f32 {
        lerp_rgb(0x000000u32, 0x301040u32, t / 0.25f32)
    } else if t < 0.5f32 {
        lerp_rgb(0x301040u32, 0x301060u32, (t - 0.25f32) / 0.25f32)
    } else if t < 0.9f32 {
        lerp_rgb(0x301060u32, 0xAFFFFFu32, (t - 0.5f32) / 0.4f32)
    } else {
        0xAFFFFFu32
    }
}

/// TPT heat table (`Renderer::PopulateTables`, `Renderer.cpp:1585-1598`).
///
/// `Gradient` (`Gradient.cpp:7-31`) interpolates ten stops into 1024 entries:
/// `0x2B00FF @ 0.00`, `0x003CFF @ 0.01`, `0x00C0FF @ 0.05`,
/// `0x00FFEB @ 0.08`, `0x00FF14 @ 0.19`, `0x4BFF00 @ 0.25`,
/// `0xC8FF00 @ 0.37`, `0xFFDC00 @ 0.45`, `0xFF0000 @ 0.71`,
/// `0xFF00DC @ 1.00`. The blend factor is the truncated `uint8_t(f * 255)` and
/// the channels blend through the integer `RGB::Blend` (`Pixel.h:139-154`), so
/// this is integer channel math, not a float ramp.
///
/// `Renderer::heatTableAt` clamps the index into `0..1023`
/// (`Renderer.h:100-110`). Returns `0x00RRGGBB`.
#[cube]
fn heat_table_at(index: i32) -> u32 {
    let i = clampi(index, 0i32, 1023i32);
    // `Gradient` samples at `i / (float)resolution`, not `(i + 0.5)`.
    let point = (i as f32) / 1024.0f32;

    // Last stop whose `point <= index` selects the left end of the segment
    // (`Gradient.cpp:17-20`), hence `>=` here. Each `if` overwrites the
    // previous segment, so the final matching range wins.
    let mut left = 0x2B00FFu32;
    let mut right = 0x003CFFu32;
    let mut lo = 0.0f32;
    let mut hi = 0.01f32;
    if point >= 0.01f32 {
        left = 0x003CFFu32;
        right = 0x00C0FFu32;
        lo = 0.01f32;
        hi = 0.05f32;
    }
    if point >= 0.05f32 {
        left = 0x00C0FFu32;
        right = 0x00FFEBu32;
        lo = 0.05f32;
        hi = 0.08f32;
    }
    if point >= 0.08f32 {
        left = 0x00FFEBu32;
        right = 0x00FF14u32;
        lo = 0.08f32;
        hi = 0.19f32;
    }
    if point >= 0.19f32 {
        left = 0x00FF14u32;
        right = 0x4BFF00u32;
        lo = 0.19f32;
        hi = 0.25f32;
    }
    if point >= 0.25f32 {
        left = 0x4BFF00u32;
        right = 0xC8FF00u32;
        lo = 0.25f32;
        hi = 0.37f32;
    }
    if point >= 0.37f32 {
        left = 0xC8FF00u32;
        right = 0xFFDC00u32;
        lo = 0.37f32;
        hi = 0.45f32;
    }
    if point >= 0.45f32 {
        left = 0xFFDC00u32;
        right = 0xFF0000u32;
        lo = 0.45f32;
        hi = 0.71f32;
    }
    if point >= 0.71f32 {
        left = 0xFF0000u32;
        right = 0xFF00DCu32;
        lo = 0.71f32;
        hi = 1.0f32;
    }

    let f = (point - lo) / (hi - lo);
    let alpha = (f * 255.0f32) as u32;
    let inv = 255u32 - alpha;
    let lr = (left >> 16u32) & 0xFFu32;
    let lg = (left >> 8u32) & 0xFFu32;
    let lb = left & 0xFFu32;
    let rr = (right >> 16u32) & 0xFFu32;
    let rg = (right >> 8u32) & 0xFFu32;
    let rb = right & 0xFFu32;
    let r = (alpha * rr + inv * lr) / 255u32;
    let g = (alpha * rg + inv * lg) / 255u32;
    let b = (alpha * rb + inv * lb) / 255u32;
    (r << 16u32) | (g << 8u32) | b
}

/// [`heat_table_at`] repacked for the framebuffer's `0xAABBGGRR`.
#[cube]
fn heat_table_packed(index: i32) -> u32 {
    let c = heat_table_at(index);
    (255u32 << 24u32) | ((c & 0xFFu32) << 16u32) | (c & 0x0000_FF00u32) | ((c >> 16u32) & 0xFFu32)
}

/// TPT's temperature-to-colour mapping (`Renderer.cpp:378`,
/// `Renderer::HeatToColour` at `Renderer.cpp:1346-1351`).
///
/// The index is exactly `int((temp - min) / (max - min) * 1024)`, then
/// [`heat_table_at`] clamps it. `MIN_TEMP = 0` and `MAX_TEMP = 9999` are the
/// reference's auto limits (`ElementDefs.h:5-6`); callers pass the resolved
/// `SimulationConfig::heat_display_limit_*` values instead when they have them.
#[cube]
pub(crate) fn heat_temp_colour(temp: f32, min_temp: f32, max_temp: f32) -> u32 {
    let index = ((temp - min_temp) / (max_temp - min_temp) * 1024.0f32) as i32;
    heat_table_packed(index)
}

/// TPT heat gradient at a normalised `0.0..1.0` position, as `0xAABBGGRR`.
///
/// Wrapper over [`heat_table_at`] for the fire-buffer path
/// (`Renderer.cpp:378-384`), which already scales the temperature.
#[cube]
pub(crate) fn heat_colour(t: f32) -> u32 {
    heat_table_packed((clampf(t, 0.0f32, 1.0f32) * 1024.0f32) as i32)
}
