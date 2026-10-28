//! Special element graphics functions: force, exotic, materials, life, logic.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::framebuffer::{clampf, hash01, rgb_from_u32};
use super::super::{
    DECORATION_ANTI_CLICKBAIT, DECORATION_DISABLED, ELEM_COLOUR, ELEM_PIXEL_MODE, ELEM_WORDS,
    PMAP_TYPE_MASK,
};
use super::PT_NUM_U32;
use super::elements_basic::wavelength_rgb;

/// `Renderer.cpp:300-314`: the `dcolour` channels after the anti-clickbait
/// collapse.
///
/// TPT rewrites `deca/decr/decg/decb` before any decoration is drawn: under
/// `decorationAntiClickbait` a colour that is not a 250+ alpha near-black
/// becomes fully transparent, and a near-black one becomes opaque black. Every
/// consumer (`decorate_rgb`, the `EMBR` fire tint) works on the collapsed
/// values.
#[cube]
pub(crate) fn collapse_deco(dcolour: u32, decoration_level: u32) -> (u32, u32, u32, u32) {
    let mut deca = (dcolour >> 24u32) & 0xFFu32;
    let mut decr = (dcolour >> 16u32) & 0xFFu32;
    let mut decg = (dcolour >> 8u32) & 0xFFu32;
    let mut decb = dcolour & 0xFFu32;
    if decoration_level == DECORATION_ANTI_CLICKBAIT {
        if deca < 250u32 || decr > 5u32 || decg > 5u32 || decb > 5u32 {
            deca = 0u32;
        } else {
            deca = 255u32;
            decr = 0u32;
            decg = 0u32;
            decb = 0u32;
        }
    }
    (deca, decr, decg, decb)
}

/// `EMBR.cpp:104-122` / `LCRY.cpp:107-116`: whether the element paints its
/// `dcolour` at all.
///
/// TPT enables the decoration when the level is not `decorationDisabled`,
/// `dcolour` carries a non-zero alpha byte, and the level is
/// `decorationEnabled` or the colour is a near-black `dcolour` (which is
/// exactly what survives [`collapse_deco`] under anti-clickbait).
#[cube]
pub(crate) fn deco_visible(dcolour: u32, decoration_level: u32) -> bool {
    if decoration_level == DECORATION_DISABLED {
        false
    } else {
        collapse_deco(dcolour, decoration_level).0 != 0u32
    }
}

/// TPT `VIBR::graphics` (`VIBR.cpp:219-240`): the exploding branch
/// (`tmp/10 >= 100 || life != 0`)
/// and the arming gradient.
///
/// The emitter strength (`firea/firer/fireg/fireb`) of both branches is
/// `fire_rgba`'s; this function only produces the pixel colour they reuse.
#[cube]
pub(crate) fn vibr_graphics(
    base: (f32, f32, f32),
    life: i32,
    tmp: i32,
    tmp2: i32,
) -> (f32, f32, f32) {
    let gradient = tmp / 10i32;
    if gradient >= 100i32 || life != 0i32 {
        let e = ((750.0f32 - life as f32) / 170.0f32).exp();
        let mut s = e.sin();
        if s < 0.0f32 {
            s = -s;
        }
        let c = s * 200.0f32;
        // TPT swaps the green/blue channels while `tmp2` is set.
        if tmp2 != 0i32 {
            (c, c, 255.0f32)
        } else {
            (c, 255.0f32, c)
        }
    } else {
        let g = clampf(gradient as f32 * 2.0f32, 0.0f32, 255.0f32) as i32 as f32;
        let g2 = clampf(gradient as f32 * 2.0f32, 0.0f32, 175.0f32) as i32 as f32;
        (base.0 + g, base.1 + g2, base.2 + g)
    }
}

/// TPT `GRAV::graphics` (`GRAV.cpp:54-117`): a velocity ramp on the `20/20/20`
/// base, modulated by three phase-shifted `currentTick` ramps.
///
/// TPT keeps every channel in `int` and truncates each contribution; the same
/// truncation is applied here so the ramp stays bit-identical.
#[cube]
pub(crate) fn grav_graphics(vx: f32, vy: f32, tick: u32) -> (f32, f32, f32) {
    let grav_r = ((tick % 120u32) as i32 - 60i32).abs();
    let grav_g = (((tick + 60u32) % 120u32) as i32 - 60i32).abs();
    let grav_b = (((tick + 120u32) % 120u32) as i32 - 60i32).abs();
    let grav_r2 = ((tick % 60u32) as i32 - 30i32).abs();
    let grav_g2 = (((tick + 30u32) % 60u32) as i32 - 30i32).abs();
    let grav_b2 = (((tick + 60u32) % 60u32) as i32 - 30i32).abs();
    let mut r = 20i32;
    let mut g = 20i32;
    let mut b = 20i32;
    if vx > 0.0f32 {
        r += (vx * grav_r as f32) as i32;
        g += (vx * grav_g as f32) as i32;
        b += (vx * grav_b as f32) as i32;
    }
    if vy > 0.0f32 {
        r += (vy * grav_g as f32) as i32;
        g += (vy * grav_b as f32) as i32;
        b += (vy * grav_r as f32) as i32;
    }
    if vx < 0.0f32 {
        r -= (vx * grav_b as f32) as i32;
        g -= (vx * grav_r as f32) as i32;
        b -= (vx * grav_g as f32) as i32;
    }
    if vy < 0.0f32 {
        r -= (vy * grav_r2 as f32) as i32;
        g -= (vy * grav_g2 as f32) as i32;
        b -= (vy * grav_b2 as f32) as i32;
    }
    (r as f32, g as f32, b as f32)
}

/// Cargo element of a `PIPE`/`PPIP`, or `0` when TPT skips the sub-call.
///
/// `PIPE.cpp:340-346`: the cargo field holds a type, and the stickmen store a
/// player index there, so TPT returns early for them.
#[cube]
pub(crate) fn pipe_cargo(ctype: u32) -> u32 {
    let cargo = ctype & PMAP_TYPE_MASK;
    let skippable = cargo == PT_STKM || cargo == PT_STKM2 || cargo == PT_FIGH;
    if cargo == 0u32 || cargo >= PT_NUM_U32 || skippable {
        0u32
    } else {
        cargo
    }
}

/// TPT `PIPE.cpp:349-352`: a loaded pipe draws with the cargo's cached
/// `Graphics` `pixel_mode`, not the pipe's own.
///
/// TPT reads `graphicscache[TYP(cpart->ctype)].pixel_mode`, i.e. the cargo
/// element's static mode; this looks the same row up in the element table. An
/// empty pipe, or a cargo TPT skips, keeps the pipe's row ([`pipe_cargo`]).
///
/// The substitution belongs to `PIPE::graphics` alone, so only `PIPE`/`PPIP`
/// run it. Every other element stores an unrelated id in `ctype` (`LAVA`
/// keeps the element it melted from, `FIRE` the fuel it burned); looking that
/// row up replaces the element's pixel mode with the recorded one, which
/// dropped `FIRE_ADD` from `LAVA(PLUT)` (`ctype == PT_PLUT`) and left the
/// fission product without its bloom.
#[cube]
pub(crate) fn pipe_pixel_mode(
    pixel_mode: u32,
    ty: u32,
    ctype: u32,
    elem_props: &[u32],
    elem_count: u32,
) -> u32 {
    let mut mode = pixel_mode;
    if ty == PT_PIPE || ty == PT_PPIP {
        let cargo = pipe_cargo(ctype);
        if cargo != 0u32 {
            mode = elem_props[(((cargo % elem_count) * ELEM_WORDS) + ELEM_PIXEL_MODE) as usize];
        }
    }
    mode
}

/// TPT `PIPE.cpp:337-418` / `PPIP`: the stored cargo's colour, else the pipe's
/// colour flag (`tmp & PFLAG_COLORS`, `PIPE.h:21-25`).
///
/// TPT calls the cargo element's own `Graphics` function on a particle rebuilt
/// from the pipe fields (`PIPE.cpp:374-392`); a device kernel has no function
/// pointer, so the cargo's table colour is used instead. The `PFLAG_*` colours
/// are the red/green/blue pipe tags.
#[cube]
pub(crate) fn pipe_graphics(
    base: (f32, f32, f32),
    ctype: u32,
    tmp: i32,
    elem_props: &[u32],
    elem_count: u32,
) -> (f32, f32, f32) {
    let cargo = pipe_cargo(ctype);
    if cargo != 0u32 {
        let slot = (cargo % elem_count) * ELEM_WORDS;
        rgb_from_u32(elem_props[(slot + ELEM_COLOUR) as usize])
    } else {
        // `tmp & PFLAG_COLORS` (`PIPE.h:22-24`): an empty pipe keeps its own
        // colour unless one of the three colour flags is set.
        let flags = tmp & 0x000C_0000i32;
        if flags == 0x0004_0000i32 {
            (50.0f32, 1.0f32, 1.0f32)
        } else if flags == 0x0008_0000i32 {
            (1.0f32, 50.0f32, 1.0f32)
        } else if flags == 0x000C_0000i32 {
            (1.0f32, 1.0f32, 50.0f32)
        } else {
            base
        }
    }
}

/// TPT `POLO::graphics` (`POLO.cpp:111-123`): a spent polo ball (`tmp >= LIMIT`)
/// is flat grey, a live one keeps its colour (and glows, see
/// [`effective_pixel_mode`]).
#[cube]
pub(crate) fn polo_graphics(base: (f32, f32, f32), tmp: i32) -> (f32, f32, f32) {
    // `POLO.cpp:53` — `constexpr int LIMIT = 5`.
    if tmp >= 5i32 {
        (0x70u32 as f32, 0x70u32 as f32, 0x70u32 as f32)
    } else {
        base
    }
}

/// One entry of TPT's `SLCN_COLOUR[16]` sparkle palette (`SLCN.cpp:52-57`).
#[cube]
pub(crate) fn slcn_colour(index: u32) -> u32 {
    if index == 0u32 {
        0x5A_6679u32
    } else if index == 1u32 {
        0x68_78A1u32
    } else if index == 2u32 {
        0xAB_BFDDu32
    } else if index == 3u32 {
        0x83_8490u32
    } else if index == 4u32 {
        0xBC_CDDFu32
    } else if index == 5u32 {
        0x82_A0D2u32
    } else if index == 6u32 {
        0x5B_6680u32
    } else if index == 7u32 {
        0x23_2C3Bu32
    } else if index == 8u32 {
        0x48_5067u32
    } else if index == 9u32 {
        0x8B_9AB6u32
    } else if index == 10u32 {
        0xAD_B1C1u32
    } else if index == 11u32 {
        0xC3_C6D1u32
    } else if index == 12u32 {
        0x85_94ADu32
    } else if index == 13u32 {
        0x26_2F47u32
    } else if index == 14u32 {
        0xA9_AEBCu32
    } else {
        0xC2_E1F7u32
    }
}

/// TPT `SLCN::graphics` (`SLCN.cpp:109-131`): the sparkle phase selects a
/// palette entry, and a phase past the halfway mark averages it with the next.
#[cube]
pub(crate) fn slcn_graphics(tmp: i32) -> (f32, f32, f32) {
    let t = tmp as u32;
    let mut c = slcn_colour((t >> 12u32) & 15u32);
    if (t & 0x800u32) != 0u32 {
        let next = slcn_colour((t >> 16u32) & 15u32);
        // TPT averages the channels as integers before unpacking them.
        let r = (((c >> 16u32) & 0xFFu32) + ((next >> 16u32) & 0xFFu32)) / 2u32;
        let g = (((c >> 8u32) & 0xFFu32) + ((next >> 8u32) & 0xFFu32)) / 2u32;
        let b = ((c & 0xFFu32) + (next & 0xFFu32)) / 2u32;
        c = (r << 16u32) | (g << 8u32) | b;
    }
    rgb_from_u32(c)
}

/// TPT `QRTZ::graphics` (`QRTZ.cpp:161-171`), shared by `PQRT`: the `tmp2`
/// growth counter offsets all three channels, giving the crystal its speckle.
#[cube]
pub(crate) fn qrtz_graphics(base: (f32, f32, f32), tmp2: i32) -> (f32, f32, f32) {
    let z = ((tmp2 - 5i32) * 16i32) as f32;
    (base.0 + z, base.1 + z, base.2 + z)
}

/// TPT `FRME::graphics` (`FRME.cpp:48-57`): a frame carrying a particle is
/// brightened by 30 per channel.
#[cube]
pub(crate) fn frme_graphics(base: (f32, f32, f32), tmp: i32) -> (f32, f32, f32) {
    if tmp != 0i32 {
        (base.0 + 30.0f32, base.1 + 30.0f32, base.2 + 30.0f32)
    } else {
        base
    }
}

/// TPT `BRAY::graphics` (`BRAY.cpp:50-105`): the beam colour depends on the
/// `tmp` beam mode. Mode 2 paints a fixed amber; modes 0 and 1 decode the
/// `ctype` wavelength mask and fall back to the base colour without one.
///
/// The beam transparency (`cola`, `BRAY.cpp:53-101`) is [`graphics_cola`]'s.
#[cube]
pub(crate) fn bray_graphics(base: (f32, f32, f32), ctype: u32, tmp: i32) -> (f32, f32, f32) {
    if tmp == 2i32 {
        (255.0f32, 150.0f32, 50.0f32)
    } else if (ctype & 0x3FFF_FFFFu32) != 0u32 {
        wavelength_rgb(ctype)
    } else {
        base
    }
}

/// TPT `COAL::graphics` (`COAL.cpp:80-102`), shared by `BCOL`: the `tmp2`
/// heat counter sets a grey level, and above 395.15 K a sine gradient adds the
/// incandescent tint.
#[cube]
pub(crate) fn coal_graphics(base: (f32, f32, f32), tmp2: i32, temp: f32) -> (f32, f32, f32) {
    // `COAL.cpp:78` — `FREQUENCY = pi / (2*300 - (300-200))`.
    let frequency = 0.006_283_185f32;
    let mut r = base.0 + ((tmp2 as f32 - 295.15f32) / 3.0f32) as i32 as f32;
    if r > 170.0f32 {
        r = 170.0f32;
    }
    if r < base.1 {
        r = base.1;
    }
    let mut g = r;
    let mut b = r;
    if temp > 395.15f32 {
        let q = if temp > 595.15f32 {
            200.0f32
        } else {
            temp - 395.15f32
        };
        r += ((frequency * q).sin() * 226.0f32) as i32 as f32;
        g += (-(frequency * q * 4.55f32).sin() * 34.0f32) as i32 as f32;
        b += (-(frequency * q * 2.22f32).sin() * 64.0f32) as i32 as f32;
    }
    (r, g, b)
}

/// TPT `LIFE::graphics` (`LIFE.cpp:51-91`): the GoL ruleset colour.
///
/// `dcolour` holds the live colour and `tmp` the dying colour (the GoL pass
/// stamps both from the builtin palette, `gol_kernel::driver`), `tmp2` is the
/// cell's current state and `ctype` the ruleset.
///
/// `decoration_level` is threaded in for signature parity with the other
/// `Graphics` rules, but `LIFE`'s `dcolour` is a simulation colour (the GoL
/// palette), not a user decoration, so `Renderer.cpp:300-314`'s anti-clickbait
/// collapse does not apply here and the output never depends on it.
#[cube]
pub(crate) fn life_graphics(
    ctype: u32,
    dcolour: u32,
    tmp: i32,
    tmp2: i32,
    _decoration_level: u32,
) -> (f32, f32, f32) {
    let mut r1 = 255.0f32;
    let mut g1 = 255.0f32;
    let mut b1 = 255.0f32;
    if dcolour != 0u32 {
        let live = rgb_from_u32(dcolour);
        r1 = live.0;
        g1 = live.1;
        b1 = live.2;
    }
    let dying = rgb_from_u32(tmp as u32);
    let states = gol_states(ctype);
    if states <= 2u32 {
        (r1, g1, b1)
    } else {
        let mul = (tmp2 as f32 - 1.0f32) / (states as f32 - 2.0f32);
        (
            (r1 * mul + dying.0 * (1.0f32 - mul)) as i32 as f32,
            (g1 * mul + dying.1 * (1.0f32 - mul)) as i32 as f32,
            (b1 * mul + dying.2 * (1.0f32 - mul)) as i32 as f32,
        )
    }
}

/// Number of states of a GoL ruleset, i.e. TPT's `((ruleset >> 17) & 0xF) + 2`.
///
/// A `ctype` below `NGOL` is a builtin index; only five builtins are not the
/// two-state default, so the index table is spelled out instead of uploaded.
#[cube]
pub(crate) fn gol_states(ctype: u32) -> u32 {
    if ctype == 19u32 {
        4u32
    } else if ctype == 20u32 || ctype == 22u32 || ctype == 23u32 {
        3u32
    } else if ctype == 21u32 {
        6u32
    } else if ctype < 24u32 {
        2u32
    } else {
        ((ctype >> 17u32) & 0xFu32) + 2u32
    }
}

/// TPT `EXOT::graphics` (`EXOT.cpp:197-251`): a colour sine on the temperature,
/// phase-shifted per channel, with a random "exotic" branch.
///
/// The branch is drawn from `rng.chance(tmp2 - 1, 1000)`, which the render pass
/// derives from the particle id and the frame tick.
#[cube]
pub(crate) fn exot_graphics(
    temp: f32,
    tmp: i32,
    tmp2: i32,
    life: i32,
    pid: u32,
    tick: u32,
) -> (f32, f32, f32) {
    let b = tmp as f32;
    if life < 1001i32 {
        if exot_flicker(tmp2, pid, tick) {
            let f = 0.04045f32;
            let c = tmp2 as f32;
            (
                ((f * c + 4.0f32).sin() * 127.0f32 + 150.0f32) as i32 as f32,
                ((f * c + 6.0f32).sin() * 127.0f32 + 150.0f32) as i32 as f32,
                ((f * c + 8.0f32).sin() * 127.0f32 + 150.0f32) as i32 as f32,
            )
        } else {
            let f = 0.00045f32;
            (
                ((f * temp + 4.0f32).sin() * 127.0f32 + b / 1.7f32) as i32 as f32,
                ((f * temp + 6.0f32).sin() * 127.0f32 + b / 1.7f32) as i32 as f32,
                ((f * temp + 8.0f32).sin() * 127.0f32 + b / 1.7f32) as i32 as f32,
            )
        }
    } else {
        let f = 0.013f32;
        let v = ((f * temp + 6.0f32).sin() * 127.0f32 + (b / 2.9f32 + 80.0f32)) as i32 as f32;
        (v, v, v)
    }
}

/// `rng.chance(tmp2 - 1, 1000)` from TPT's `EXOT::graphics`, hashed per frame.
#[cube]
pub(crate) fn exot_flicker(tmp2: i32, pid: u32, tick: u32) -> bool {
    hash01(pid ^ tick) * 1000.0f32 < (tmp2 - 1i32) as f32
}

/// TPT `CLST::graphics` (`CLST.cpp:98-105`): the `tmp` counter offsets all
/// three channels, giving clay dust its speckle.
#[cube]
pub(crate) fn clst_graphics(base: (f32, f32, f32), tmp: i32) -> (f32, f32, f32) {
    let z = ((tmp - 5i32) * 16i32) as f32;
    (base.0 + z, base.1 + z, base.2 + z)
}

/// TPT `ROCK::graphics` (`ROCK.cpp:51-68`): a `tmp2` speckle. TPT also feeds
/// the hot-rock glow (`firea = (temp - 810.15) / 45`) from this colour, which
/// `fire_rgba` reproduces.
#[cube]
pub(crate) fn rock_graphics(base: (f32, f32, f32), tmp2: i32) -> (f32, f32, f32) {
    let z = ((tmp2 - 7i32) * 6i32) as f32;
    (base.0 + z, base.1 + z, base.2 + z)
}

/// TPT `CBNW::graphics` (`CBNW.cpp:148-156`, displayed as `BUBW`): the `tmp2`
/// carbonation counter offsets the channels with a blue bias.
#[cube]
pub(crate) fn cbnw_graphics(base: (f32, f32, f32), tmp2: i32) -> (f32, f32, f32) {
    let z = (tmp2 - 20i32) as f32;
    (base.0 + z, base.1 + z * 2.0f32, base.2 + z * 8.0f32)
}

/// TPT `INVIS::graphics` (`INVIS.cpp:65-77`): under pressure (`tmp2`) the wall
/// shows a violet tint through `cola = 100`.
///
/// The tint is drawn with `PMODE_BLEND`, so [`graphics_cola`] supplies the
/// `100` alpha.
#[cube]
pub(crate) fn invis_graphics(base: (f32, f32, f32), tmp2: i32) -> (f32, f32, f32) {
    if tmp2 != 0i32 {
        (15.0f32, 0.0f32, 150.0f32)
    } else {
        base
    }
}

/// TPT `WARP::graphics` (`WARP.cpp:89-94`): a fully transparent particle
/// (`colr = colg = colb = cola = 0`).
///
/// The black tint is what the `PMODE_FLAT` stage writes; `cola = 0` (from
/// [`graphics_cola`]) would only matter for a blend, which WARP never uses.
#[cube]
pub(crate) fn warp_graphics() -> (f32, f32, f32) {
    (0.0f32, 0.0f32, 0.0f32)
}

/// TPT `GEL::graphics` (`GEL.cpp:44-53`): colour derived from `tmp`.
#[cube]
pub(crate) fn gel_graphics(tmp: i32) -> (f32, f32, f32) {
    let q = tmp;
    let r = q * (32i32 - 255i32) / 120i32 + 255i32;
    let g = q * (48i32 - 186i32) / 120i32 + 186i32;
    let b = q * 208i32 / 120i32;
    (r as f32, g as f32, b as f32)
}

/// One entry of TPT's `tron_colours[32]` palette (`TRON.cpp:33-47`).
#[cube]
pub(crate) fn tron_colour(index: u32) -> u32 {
    if index == 0u32 {
        0xFF_0000u32
    } else if index == 1u32 {
        0xFF_4300u32
    } else if index == 2u32 {
        0xFF_8800u32
    } else if index == 3u32 {
        0xFF_CC00u32
    } else if index == 4u32 {
        0xEE_FF00u32
    } else if index == 5u32 {
        0xAA_FF00u32
    } else if index == 6u32 {
        0x65_FF00u32
    } else if index == 7u32 {
        0x21_FF00u32
    } else if index == 8u32 {
        0x00_FF21u32
    } else if index == 9u32 {
        0x00_FF65u32
    } else if index == 10u32 {
        0x00_FFA9u32
    } else if index == 11u32 {
        0x00_FFEDu32
    } else if index == 12u32 {
        0x00_CBFFu32
    } else if index == 13u32 {
        0x00_87FFu32
    } else if index == 14u32 {
        0x00_43FFu32
    } else if index == 15u32 {
        0x00_00FFu32
    } else if index == 16u32 {
        0x43_00FFu32
    } else if index == 17u32 {
        0x87_00FFu32
    } else if index == 18u32 {
        0xCB_00FFu32
    } else if index == 19u32 {
        0xFF_00EEu32
    } else if index == 20u32 {
        0xFF_00AAu32
    } else if index == 21u32 {
        0xFF_0066u32
    } else {
        0xFF_0022u32
    }
}

/// TPT `TRON::graphics` (`TRON.cpp:103-122`): palette entry from `(tmp & 0xF800) >> 11`.
#[cube]
pub(crate) fn tron_graphics(tmp: i32) -> (f32, f32, f32) {
    let index = ((tmp as u32 & 0xF800u32) >> 11u32) & 0x1Fu32;
    let col = tron_colour(index);
    (
        ((col >> 16u32) & 0xFFu32) as f32,
        ((col >> 8u32) & 0xFFu32) as f32,
        (col & 0xFFu32) as f32,
    )
}

/// TPT `BRCK::graphics` (`BRCK.cpp:53-65`): glowing blue tint when `tmp == 1`.
#[cube]
pub(crate) fn brck_graphics(base: (f32, f32, f32), tmp: i32) -> (f32, f32, f32) {
    if tmp == 1i32 {
        (base.0, base.1, base.2 + 100.0f32)
    } else {
        base
    }
}

/// TPT `CRMC::graphics` (`CRMC.cpp:66-73`): speckle offset from `tmp2`.
#[cube]
pub(crate) fn crmc_graphics(base: (f32, f32, f32), tmp2: i32) -> (f32, f32, f32) {
    let z = ((tmp2 - 2i32) * 8i32) as f32;
    (base.0 + z, base.1 + z, base.2 + z)
}

/// TPT `EMP::graphics` (`EMP.cpp:51-60`): life-dependent colour when active.
#[cube]
pub(crate) fn emp_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    if life != 0i32 {
        (
            (life as f32 * 1.5f32) as i32 as f32,
            (life as f32 * 1.5f32) as i32 as f32,
            (200i32 - life) as f32,
        )
    } else {
        base
    }
}

/// TPT `PVOD::graphics` (`PVOD.cpp:66-72`): red channel brightens with life.
#[cube]
pub(crate) fn pvod_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    let l = if life > 10i32 { 10i32 } else { life };
    let lifemod = (l * 16i32) as f32;
    (base.0 + lifemod, base.1, base.2)
}
