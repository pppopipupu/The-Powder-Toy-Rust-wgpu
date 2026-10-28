//! Basic element graphics functions: matter, plants, devices, light.

use cubecl::prelude::*;

use super::super::framebuffer::{clampf, clampi, hash01, rgb_from_u32, solid_rgb};

/// `colour_mode` bit the render driver sets while `decorations` is switched off
/// (`render_kernel/mod.rs`, from `bufs.config.decorations`): TPT's
/// `RendererSettings::decorationLevel == decorationDisabled`.
const DECO_DISABLED: u32 = 0x8000_0000;

/// TPT `LAVA::graphics`: `life*2+0xE0`, `life+0x50`, `life/2+0x10`.
#[cube]
pub(crate) fn lava_graphics(life: i32) -> (f32, f32, f32) {
    let r = clampf((life * 2i32 + 0xE0i32) as f32, 0.0f32, 255.0f32);
    let g = clampf((life + 0x50i32) as f32, 0.0f32, 192.0f32);
    let b = clampf((life / 2i32 + 0x10i32) as f32, 0.0f32, 128.0f32);
    (r, g, b)
}

/// TPT `PHOT::graphics`: three 12-bit wavelength bit fields.
#[cube]
pub(crate) fn photon_graphics(ctype: u32) -> (f32, f32, f32) {
    let mut cr = 0i32;
    let mut cg = 0i32;
    let mut cb = 0i32;
    for i in 0u32..12u32 {
        cr += ((ctype >> (i + 18u32)) & 1u32) as i32;
        cb += ((ctype >> i) & 1u32) as i32;
        cg += ((ctype >> (i + 9u32)) & 1u32) as i32;
    }
    let s = 624i32 / (cr + cg + cb + 1i32);
    let r = clampf((cr * s) as f32, 0.0f32, 255.0f32);
    let g = clampf((cg * s) as f32, 0.0f32, 255.0f32);
    let b = clampf((cb * s) as f32, 0.0f32, 255.0f32);
    (r, g, b)
}

/// TPT `ACID::graphics`: fresh acid is nearly white, spent acid purple.
#[cube]
pub(crate) fn acid_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    let (mut r, mut g, mut b) = base;
    // TPT clamps `life` to `49..=75` before scaling it.
    let mut s = clampi(life, 49i32, 75i32);
    s = (s - 49i32) * 3i32;
    if s == 0i32 {
        s = 1i32;
    }
    r += (s * 4i32) as f32;
    g += s as f32;
    b += (s * 2i32) as f32;
    (r, g, b)
}

/// TPT `WOOD::graphics`: chars above 400 K, frosts below 273 K.
#[cube]
pub(crate) fn wood_graphics(base: (f32, f32, f32), temp: f32, tmp: i32) -> (f32, f32, f32) {
    let (mut r, mut g, mut b) = base;
    let maxtemp = if (tmp as f32) > temp {
        tmp as f32
    } else {
        temp
    };
    if maxtemp > 400.0f32 {
        r -= clampf((maxtemp - 400.0f32) / 3.0f32, 0.0f32, 172.0f32);
        g -= clampf((maxtemp - 400.0f32) / 4.0f32, 0.0f32, 140.0f32);
        b -= clampf((maxtemp - 400.0f32) / 20.0f32, 0.0f32, 44.0f32);
    }
    if maxtemp < 273.0f32 {
        r -= clampf((273.0f32 - maxtemp) / 5.0f32, 0.0f32, 40.0f32);
        g += clampf((273.0f32 - maxtemp) / 4.0f32, 0.0f32, 40.0f32);
        b += clampf((273.0f32 - maxtemp) / 1.5f32, 0.0f32, 150.0f32);
    }
    (r, g, b)
}

/// TPT `GPMP::graphics`: greener/bluer while it charges.
#[cube]
pub(crate) fn gpmp_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    let (r, mut g, mut b) = base;
    let mut l = life;
    if l > 10i32 {
        l = 10i32;
    }
    let lifemod = (l * 19i32) as f32;
    g += lifemod;
    b += lifemod;
    (r, g, b)
}

/// TPT `SPNG::graphics`: darker as it fills up with water (`life`).
#[cube]
pub(crate) fn spng_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    let d = (clampi(life, 0i32, 1000i32) as f32) * 15.0f32;
    (
        clampf(base.0 - d, 50.0f32, 255.0f32),
        clampf(base.1 - d, 50.0f32, 255.0f32),
        clampf(base.2 - d, 20.0f32, 255.0f32),
    )
}

/// TPT `PLNT::graphics` (`PLNT.cpp:360-392`): temperature tint for wild
/// plants, the `leafColor` table for plants grown from a seed (colour index in
/// `ctype` bits 6..11).
///
/// TPT tints by `max(tmp2, temp)`, the higher of the growth and current
/// temperatures.
#[cube]
pub(crate) fn plnt_graphics(
    base: (f32, f32, f32),
    ctype: u32,
    temp: f32,
    tmp2: i32,
) -> (f32, f32, f32) {
    let mut r = base.0;
    let mut g = base.1;
    let mut b = base.2;
    if ctype == 0u32 {
        let grown = tmp2 as f32;
        let maxtemp = if grown > temp { grown } else { temp };
        if maxtemp > 300.0f32 {
            r += clampf((maxtemp - 300.0f32) / 5.0f32, 0.0f32, 58.0f32);
            g -= clampf((maxtemp - 300.0f32) / 2.0f32, 0.0f32, 102.0f32);
            b += clampf((maxtemp - 300.0f32) / 5.0f32, 0.0f32, 70.0f32);
        }
        if maxtemp < 273.0f32 {
            g += clampf((273.0f32 - maxtemp) / 4.0f32, 0.0f32, 255.0f32);
            b += clampf((273.0f32 - maxtemp) / 1.5f32, 0.0f32, 255.0f32);
        }
    } else {
        let colour = (ctype >> 6u32) & 0x3Fu32;
        let cyan = if (colour & 0x30u32) != 0u32 {
            4u32
        } else {
            0u32
        };
        let magenta = if (colour & 0x0Cu32) != 0u32 {
            2u32
        } else {
            0u32
        };
        let yellow = if (colour & 0x03u32) != 0u32 {
            1u32
        } else {
            0u32
        };
        let ind = cyan + magenta + yellow;
        // `leafColor` from `PLNT.cpp`, in index order.
        if ind == 0u32 {
            r = 243.0f32;
            g = 246.0f32;
            b = 244.0f32;
        } else if ind == 1u32 {
            r = 255.0f32;
            g = 223.0f32;
            b = 50.0f32;
        } else if ind == 2u32 {
            r = 255.0f32;
            g = 183.0f32;
            b = 197.0f32;
        } else if ind == 3u32 {
            r = 250.0f32;
            g = 0.0f32;
            b = 25.0f32;
        } else if ind == 4u32 {
            r = 128.0f32;
            g = 206.0f32;
            b = 196.0f32;
        } else if ind == 5u32 {
            r = 127.0f32;
            g = 255.0f32;
            b = 0.0f32;
        } else if ind == 6u32 {
            r = 0.0f32;
            g = 74.0f32;
            b = 178.0f32;
        } else {
            r = 12.0f32;
            g = 172.0f32;
            b = 0.0f32;
        }
    }
    (r, g, b)
}

/// TPT `SWCH::graphics`: a switched-on switch is bright green.
#[cube]
pub(crate) fn swch_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    if life >= 10i32 {
        (17.0f32, 217.0f32, 24.0f32)
    } else {
        base
    }
}

/// TPT `HSWC::graphics`: red glow proportional to the remaining life.
#[cube]
pub(crate) fn hswc_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    let lifemod = (clampi(life, 0i32, 10i32) * 19i32) as f32;
    (base.0 + lifemod, base.1, base.2)
}

/// TPT `PCLN::graphics`: yellow tint proportional to the remaining life.
#[cube]
pub(crate) fn pcln_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    let lifemod = (clampi(life, 0i32, 10i32) * 10i32) as f32;
    (base.0 + lifemod, base.1 + lifemod, base.2)
}

/// TPT `PUMP::graphics`: blue tint proportional to the remaining life.
#[cube]
pub(crate) fn pump_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    let lifemod = (clampi(life, 0i32, 10i32) * 19i32) as f32;
    (base.0, base.1, base.2 + lifemod)
}

/// TPT `DLAY::graphics`: brightness follows the charged percentage.
#[cube]
pub(crate) fn dlay_graphics(base: (f32, f32, f32), life: i32, temp: f32) -> (f32, f32, f32) {
    let mut stage = 0.0f32;
    if temp > 273.16f32 {
        stage = clampf(
            (life as f32) / (temp - 273.15f32) * 100.0f32,
            0.0f32,
            255.0f32,
        );
    }
    (base.0 + stage, base.1 + stage, base.2 + stage)
}

/// TPT `GLOW::graphics`: the three channels come from temperature, pressure
/// (`ctype`) and `tmp`.
///
/// TPT dims all three channels by 54 whenever their sum exceeds
/// `256 + rng.between(1, 32)`; that per-frame flicker is derived from the
/// particle id and the frame tick here, because the render pass is a pure
/// function of the particle.
#[cube]
pub(crate) fn glow_graphics(
    ctype: u32,
    temp: f32,
    tmp: i32,
    pid: u32,
    tick: u32,
) -> (f32, f32, f32) {
    let r = clampf(64.0f32 + temp - 307.15f32, 0.0f32, 255.0f32);
    let g = clampf(64.0f32 + ctype as f32, 0.0f32, 255.0f32);
    let b = clampf(64.0f32 + tmp as f32, 0.0f32, 255.0f32);
    if glow_bright(ctype, temp, tmp, pid, tick) {
        (r - 54.0f32, g - 54.0f32, b - 54.0f32)
    } else {
        (r, g, b)
    }
}

/// TPT `GLOW::graphics` brightness test (`GLOW.cpp:105-106`): the channel sum
/// against `256 + rng.between(1, 32)`, hashed per frame.
#[cube]
pub(crate) fn glow_bright(ctype: u32, temp: f32, tmp: i32, pid: u32, tick: u32) -> bool {
    let r = clampf(64.0f32 + temp - 307.15f32, 0.0f32, 255.0f32);
    let g = clampf(64.0f32 + ctype as f32, 0.0f32, 255.0f32);
    let b = clampf(64.0f32 + tmp as f32, 0.0f32, 255.0f32);
    // `RNG::between` is inclusive (`tpt-rand.cpp:35-39`), so the offset is an
    // integer in `1..=32`, not the `[1, 33)` the float form produced.
    let mut rng = (hash01(pid ^ tick) * 32.0f32) as u32;
    if rng > 31u32 {
        rng = 31u32;
    }
    r + g + b > 256.0f32 + (rng + 1u32) as f32
}

/// TPT `STOR::graphics`: the storage tint, brighter while a particle is held
/// (`tmp != 0`).
#[cube]
pub(crate) fn stor_graphics(tmp: i32) -> (f32, f32, f32) {
    if tmp != 0i32 {
        (0x50u32 as f32, 0xDFu32 as f32, 0xDFu32 as f32)
    } else {
        (0x20u32 as f32, 0xAFu32 as f32, 0xAFu32 as f32)
    }
}

/// TPT `PSTN::graphics`: piston darkens while it moves.
#[cube]
pub(crate) fn pstn_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    if life != 0i32 {
        (base.0 - 60.0f32, base.1 - 60.0f32, base.2)
    } else {
        base
    }
}

/// TPT `PBCN::graphics`: pink tint proportional to the remaining life.
#[cube]
pub(crate) fn pbcn_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    let lifemod = (clampi(life, 0i32, 10i32) * 10i32) as f32;
    (base.0 + lifemod, base.1 + lifemod / 2.0f32, base.2)
}

/// TPT `WIRE::graphics`: the three tool colours stored in `ctype`.
#[cube]
pub(crate) fn wire_graphics(base: (f32, f32, f32), ctype: u32) -> (f32, f32, f32) {
    if ctype == 0u32 {
        (255.0f32, 204.0f32, 0.0f32)
    } else if ctype == 1u32 {
        (50.0f32, 100.0f32, 255.0f32)
    } else if ctype == 2u32 {
        (255.0f32, 100.0f32, 50.0f32)
    } else {
        base
    }
}

/// TPT `SEED::graphics`: darker once the seed has drunk water (`ctype`).
#[cube]
pub(crate) fn seed_graphics(base: (f32, f32, f32), ctype: u32) -> (f32, f32, f32) {
    let water = (ctype >> 12u32) & 0xFFu32;
    if water > 3u32 {
        (base.0 - 50.0f32, base.1 - 50.0f32, base.2 - 50.0f32)
    } else if water > 0u32 {
        (base.0 - 25.0f32, base.1 - 25.0f32, base.2 - 25.0f32)
    } else {
        base
    }
}

/// TPT `BASE::graphics`: four lifecycle tints plus a blur.
#[cube]
pub(crate) fn base_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    if life <= 25i32 {
        (0x33u32 as f32, 0x4Cu32 as f32, 0xD8u32 as f32)
    } else if life <= 50i32 {
        (0x58u32 as f32, 0x83u32 as f32, 0xE8u32 as f32)
    } else if life <= 75i32 {
        (0x7Du32 as f32, 0xBAu32 as f32, 0xF7u32 as f32)
    } else {
        base
    }
}

/// TPT `LITH::graphics` (`LITH.cpp:254-276`): exploding lithium is orange, a
/// charged one drops red and gains blue by a hashed random amount.
#[cube]
pub(crate) fn lith_graphics(
    base: (f32, f32, f32),
    life: i32,
    ctype: u32,
    pid: u32,
    tick: u32,
) -> (f32, f32, f32) {
    if life >= 1000i32 {
        (0xFFu32 as f32, 0xA0u32 as f32, 0x40u32 as f32)
    } else if ctype > 0u32 {
        // `rng.between(ctype / 3, ctype) / 15`, capped at 6, hashed per frame.
        let lo = (ctype / 3u32) as i32;
        let hi = ctype as i32;
        let span = (hi - lo + 1i32) as f32;
        let drawn = lo + (hash01(pid ^ tick) * span) as i32;
        let mult = clampi(clampi(drawn, lo, hi) / 15i32, 0i32, 6i32) as f32;
        (base.0 - 30.0f32 * mult, base.1, base.2 + 20.0f32 * mult)
    } else {
        base
    }
}

/// TPT `VINE::graphics` (`VINE.cpp:79-93`): heat and frost tinting.
///
/// TPT tints by `max(tmp2, temp)`, the higher of the growth and current
/// temperatures.
#[cube]
pub(crate) fn vine_graphics(base: (f32, f32, f32), temp: f32, tmp2: i32) -> (f32, f32, f32) {
    let mut r = base.0;
    let mut g = base.1;
    let mut b = base.2;
    let grown = tmp2 as f32;
    let maxtemp = if grown > temp { grown } else { temp };
    if maxtemp > 300.0f32 {
        r += clampf((maxtemp - 300.0f32) / 5.0f32, 0.0f32, 58.0f32);
        g -= clampf((maxtemp - 300.0f32) / 2.0f32, 0.0f32, 102.0f32);
        b += clampf((maxtemp - 300.0f32) / 5.0f32, 0.0f32, 70.0f32);
    }
    if maxtemp < 273.0f32 {
        g += clampf((273.0f32 - maxtemp) / 4.0f32, 0.0f32, 255.0f32);
        b += clampf((273.0f32 - maxtemp) / 1.5f32, 0.0f32, 255.0f32);
    }
    (r, g, b)
}

/// TPT `WIFI::graphics`: a sine of the temperature channel (`FREQUENCY`).
#[cube]
pub(crate) fn wifi_graphics(temp: f32) -> (f32, f32, f32) {
    let q = ((temp - 73.15f32) / 100.0f32 + 1.0f32) as i32 as f32;
    (
        (0.0628f32 * q).sin() * 127.0f32 + 128.0f32,
        (0.0628f32 * q + 2.0f32).sin() * 127.0f32 + 128.0f32,
        (0.0628f32 * q + 4.0f32).sin() * 127.0f32 + 128.0f32,
    )
}

/// TPT `DEUT::graphics`: superheated deuterium turns white, warm deuterium
/// gains a red/green/blue gradient.
#[cube]
pub(crate) fn deut_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    if life >= 240i32 {
        (base.0 + 255.0f32, base.1 + 255.0f32, base.2 + 255.0f32)
    } else if life > 0i32 {
        let l = life as f32;
        (base.0 + l, base.1 + l * 2.0f32, base.2 + l * 3.0f32)
    } else {
        base
    }
}

/// TPT `LIGH::graphics`: lightning is a flat bluish-white.
#[cube]
pub(crate) fn ligh_graphics() -> (f32, f32, f32) {
    (235.0f32, 245.0f32, 255.0f32)
}

/// Mix two packed colours by `t`.
#[cube]
pub(crate) fn mix_rgb(a: u32, b: u32, t: f32) -> (f32, f32, f32) {
    let ca = rgb_from_u32(a);
    let cb = rgb_from_u32(b);
    (
        ca.0 + (cb.0 - ca.0) * t,
        ca.1 + (cb.1 - ca.1) * t,
        ca.2 + (cb.2 - ca.2) * t,
    )
}

/// TPT `Renderer::clfmTableAt(life / 2)`: the cold-flame gradient.
///
/// The table is built with resolution 200 (`Renderer.cpp:1597-1606`) and entry
/// `i` sits at `i / 200` (`Gradient.cpp:12`), so the continuous stand-in for
/// index `life / 2` divides by 200, not 199.
#[cube]
pub(crate) fn clfm_colour(life: i32) -> (f32, f32, f32) {
    let t = clampf((life as f32) / 2.0f32 / 200.0f32, 0.0f32, 1.0f32);
    if t < 0.1f32 {
        mix_rgb(0x000000u32, 0x0A0917u32, t / 0.1f32)
    } else if t < 0.2f32 {
        mix_rgb(0x0A0917u32, 0x19163Cu32, (t - 0.1f32) / 0.1f32)
    } else if t < 0.3f32 {
        mix_rgb(0x19163Cu32, 0x28285Eu32, (t - 0.2f32) / 0.1f32)
    } else if t < 0.4f32 {
        mix_rgb(0x28285Eu32, 0x343E77u32, (t - 0.3f32) / 0.1f32)
    } else if t < 0.6f32 {
        mix_rgb(0x343E77u32, 0x49769Au32, (t - 0.4f32) / 0.2f32)
    } else if t < 0.8f32 {
        mix_rgb(0x49769Au32, 0x57A0B4u32, (t - 0.6f32) / 0.2f32)
    } else {
        mix_rgb(0x57A0B4u32, 0x5EC4C6u32, (t - 0.8f32) / 0.2f32)
    }
}

/// TPT `FILT`/`BRAY`/`C5`/`BIZR` wavelength bit field (`12+12+12` bits) to colour.
///
/// Shares the exact same integer scaling as `PHOT::graphics` (`PHOT.cpp:153-162`).
#[cube]
pub(crate) fn wavelength_rgb(mask: u32) -> (f32, f32, f32) {
    photon_graphics(mask)
}

/// TPT `SMKE::graphics` (`SMKE.cpp:49-64`): flat grey (55, 55, 55).
#[cube]
pub(crate) fn smke_graphics() -> (f32, f32, f32) {
    solid_rgb(55.0f32)
}

/// TPT `CFLM::graphics` (`CFLM.cpp:51-67`): cold flame gradient from `life / 2`.
#[cube]
pub(crate) fn cflm_graphics(life: i32) -> (f32, f32, f32) {
    clfm_colour(life)
}

/// TPT `C5::graphics` (`C5.cpp:99-125`): wavelength bitfield if set, else base colour.
#[cube]
pub(crate) fn c5_graphics(base: (f32, f32, f32), ctype: u32) -> (f32, f32, f32) {
    if ctype != 0u32 {
        wavelength_rgb(ctype)
    } else {
        base
    }
}

/// TPT `FILT::graphics` (`FILT.cpp:51-74`): wavelength spectrum from ctype or temperature.
#[cube]
pub(crate) fn filt_graphics(ctype: u32, temp: f32) -> (f32, f32, f32) {
    wavelength_rgb(filt_wavelengths(ctype, temp))
}

/// TPT `Element_BIZR_graphics` (`BIZR.cpp:91-111`), shared by `BIZR`, `BIZRG`
/// and `BIZRS`: the wavelength mask in `ctype`, else the table colour.
///
/// The speed-driven fire channels of the same function are `fire_rgba`'s.
#[cube]
pub(crate) fn bizr_graphics(base: (f32, f32, f32), ctype: u32) -> (f32, f32, f32) {
    if (ctype & 0x3FFF_FFFFu32) != 0u32 {
        wavelength_rgb(ctype)
    } else {
        base
    }
}

/// TPT `EMBR::graphics` (`EMBR.cpp:97-112`): an embedded colour in `ctype`
/// (lightened if dark), white while `tmp` is set, the table colour otherwise.
#[cube]
pub(crate) fn embr_graphics(base: (f32, f32, f32), ctype: u32, tmp: i32) -> (f32, f32, f32) {
    if (ctype & 0x00FF_FFFFu32) != 0u32 {
        let mut r = ((ctype >> 16u32) & 0xFFu32) as f32;
        let mut g = ((ctype >> 8u32) & 0xFFu32) as f32;
        let mut b = (ctype & 0xFFu32) as f32;
        let mut max_component = r;
        if g > max_component {
            max_component = g;
        }
        if b > max_component {
            max_component = b;
        }
        if max_component < 60.0f32 {
            let mul = 60.0f32 / max_component;
            r *= mul;
            g *= mul;
            b *= mul;
        }
        (r, g, b)
    } else if tmp != 0i32 {
        solid_rgb(255.0f32)
    } else {
        base
    }
}

/// TPT `LCRY::graphics` (`LCRY.cpp:107-137`): a decorated crystal is painted in
/// its `dcolour` (divided by `10 - tmp2`), otherwise the grey base colour
/// brightens with the `tmp2` charge (`0x50 + min(tmp2, 10) * 10`).
///
/// TPT ORs `NO_DECO` into the mode in *both* branches (`LCRY.cpp:135`), so the
/// crystal's own blend above is the only decoration it gets: the shared
/// `decorate_rgb` stage stays off, which is why the table row keeps the bit.
#[cube]
pub(crate) fn lcry_graphics(tmp2: i32, dcolour: u32, colour_mode: u32) -> (f32, f32, f32) {
    // `deco = decorationLevel != decorationDisabled && dcolour &&
    // (dcolour & 0xFF000000)` (`LCRY.cpp:109-116`). The driver folds the
    // decoration switch into the top `colour_mode` bit from
    // `bufs.config.decorations` (`render_kernel/mod.rs`, `colour_mode = ...`).
    let deca = (dcolour >> 24u32) & 0xFFu32;
    if dcolour != 0u32 && deca != 0u32 && (colour_mode & DECO_DISABLED) == 0u32 {
        let mut r = ((dcolour >> 16u32) & 0xFFu32) as f32;
        let mut g = ((dcolour >> 8u32) & 0xFFu32) as f32;
        let mut b = (dcolour & 0xFFu32) as f32;
        // `LCRY.cpp:124-128`: only `tmp2 < 10` divides, by `10 - tmp2`.
        if tmp2 < 10i32 {
            let div = (10i32 - tmp2) as f32;
            r /= div;
            g /= div;
            b /= div;
        }
        (r, g, b)
    } else {
        let charge = clampi(tmp2, 0i32, 10i32) as f32;
        solid_rgb(0x50u32 as f32 + charge * 10.0f32)
    }
}

/// TPT `Element_FILT_getWavelengths`: `ctype` when set, otherwise a temperature
/// bin.
#[cube]
pub(crate) fn filt_wavelengths(ctype: u32, temp: f32) -> u32 {
    if (ctype & 0x3FFF_FFFFu32) != 0u32 {
        ctype
    } else {
        let bin = clampi(((temp - 273.0f32) * 0.025f32) as i32, 0i32, 25i32) as u32;
        0x1Fu32 << bin
    }
}

/// TPT `TUNG::graphics`: tungsten glows with a sinusoid as it approaches its
/// 3695 K melting point.
#[cube]
// TPT `TUNG::graphics` uses `std::numbers::pi`; the literals are kept to stay
// bit-identical with the reference gradient.
#[allow(clippy::approx_constant)]
pub(crate) fn tung_graphics(base: (f32, f32, f32), temp: f32) -> (f32, f32, f32) {
    let start_temp = 3695.0f32 - 1500.0f32;
    let mut temp_over = ((temp - start_temp) / 1500.0f32) * 3.1415927f32 - 1.5707963f32;
    if temp_over > -1.5707963f32 {
        if temp_over > 1.5707963f32 {
            temp_over = 1.5707963f32;
        }
        let gradv = temp_over.sin() + 1.0f32;
        (
            base.0 + gradv * 258.0f32,
            base.1 + gradv * 156.0f32,
            base.2 + gradv * 112.0f32,
        )
    } else {
        base
    }
}

/// TPT `TUNG::graphics` fire channels (`TUNG.cpp:131-142`): the same sinusoid
/// [`tung_graphics`] adds to the pixel colour, at the fixed alpha `30`.
///
/// Returns `(alpha, r, g, b)`; the alpha is `0` below `HighTemperature - 1500`
/// K, where TPT leaves `firea` alone (and `pixel_mode` without `FIRE_ADD`).
#[cube]
// Same `std::numbers::pi` literals as [`tung_graphics`], kept bit-identical.
#[allow(clippy::approx_constant)]
pub(crate) fn tung_fire(temp: f32) -> (u32, u32, u32, u32) {
    let start_temp = 3695.0f32 - 1500.0f32;
    let mut temp_over = ((temp - start_temp) / 1500.0f32) * 3.1415927f32 - 1.5707963f32;
    if temp_over > -1.5707963f32 {
        if temp_over > 1.5707963f32 {
            temp_over = 1.5707963f32;
        }
        let gradv = temp_over.sin() + 1.0f32;
        (
            30u32,
            clampf(gradv * 258.0f32, 0.0f32, 255.0f32) as u32,
            clampf(gradv * 156.0f32, 0.0f32, 255.0f32) as u32,
            clampf(gradv * 112.0f32, 0.0f32, 255.0f32) as u32,
        )
    } else {
        (0u32, 0u32, 0u32, 0u32)
    }
}

/// TPT `GOLD::graphics`: a +/-5 per channel jitter.
///
/// TPT draws one `rng.gen()` per particle per frame; the render pass is a pure
/// function of the particle, so the jitter is derived from the particle id. The
/// amplitude and channel offsets are identical.
#[cube]
pub(crate) fn gold_graphics(base: (f32, f32, f32), pid: u32) -> (f32, f32, f32) {
    let mut h = pid * 2654435761u32;
    let dr = (h % 10u32) as f32 - 5.0f32;
    h >>= 4u32;
    let dg = (h % 10u32) as f32 - 5.0f32;
    h >>= 4u32;
    let db = (h % 10u32) as f32 - 5.0f32;
    (base.0 + dr, base.1 + dg, base.2 + db)
}
