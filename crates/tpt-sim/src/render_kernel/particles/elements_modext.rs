//! `Graphics` overrides of the mod's own elements (`PT_CHLR ..= PT_MGNT`).
//!
//! Transcribed from the `graphics` functions in the mod's
//! `src/simulation/elements/*.cpp`. Those ids sit past the built-in range, so
//! [`element_graphics_rgb`](super::colour::element_graphics_rgb) ends in
//! [`modext_graphics_rgb`], [`element_pixel_mode`](super::pixel_mode::element_pixel_mode)
//! ends in [`modext_pixel_mode`], and `fire_rgba` keeps one arm per element that
//! writes `firer/fireg/fireb/firea`.
//!
//! TPT's `Graphics` call hands every rule a mutable `pixel_mode` (starting at
//! `PMODE_FLAT`, `Renderer.cpp:289`), `cola = 255`, the element table colour
//! and four zeroed fire channels (`Renderer.cpp:298-302`). The port keeps the
//! table colour as `base`, the element table row as the starting `pixel_mode`
//! and the fire channels at the table's `fire_colour` word (zero for every
//! element in this batch), then applies the same mutations.
//!
//! # Randomness
//!
//! `CLUD`, `DFOM` and `ALMP` draw from `gfctx.rng` (`Renderer.cpp:251`), which
//! the reference reseeds every frame and advances once per particle and per
//! draw. The render pass here is a pure function of a particle, so each roll is
//! derived from the slot id and the tick with [`hash01`], like `GLOW`, `LITH`
//! and `EXOT`. `between(lo, hi)` is inclusive (`tpt-rand.cpp:35-39`) and
//! `chance(1, 6)` is a uniform one-in-six roll (`tpt-rand.cpp:41-46`).
//!
//! # Not ported here
//!
//! * `MGNT::graphics` (`MGNT.cpp:185-194`) consumes one `rng.chance(1, 3)` and
//!   draws nothing: its only drawing call is commented out in the reference.
//!   The element therefore keeps the default `Graphics` colour and mode, and
//!   the roll is not reproduced (it has no observable result).
//! * No function in this batch writes `*cola`, so there is no `modext_cola`;
//!   [`element_graphics_cola`](super::colour::element_graphics_cola) stays
//!   untouched.
//! * `LED` reads `dcolour`, but only for its pixel colour and fire channels;
//!   its `pixel_mode` rule is the unconditional `NO_DECO`, which the element
//!   table row already carries. [`modext_pixel_mode`] therefore needs no
//!   `dcolour` argument.
//! * `DFOM`'s `tmp4`, `GRPH`'s `tmp3` and `PHOS`'s `tmp3` are per-particle
//!   fields, so they are threaded through the dispatchers instead of the table.

// Same rationale as `pixel_mode.rs`: collapsing a nested `if` changes what the
// CubeCL macro expands, and `Range::contains` does not exist on the device, so
// the plain nested form and the two-sided comparisons are deliberate.
#![allow(
    clippy::collapsible_if,
    clippy::manual_range_contains,
    clippy::assign_op_pattern
)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::framebuffer::{clampi, hash01};
use super::super::{FIRE_ADD, FIRE_BLEND, NO_DECO, PMODE_FLARE, PMODE_LFLARE, PMODE_SPARK};

// ---------------------------------------------------------------------------
// Colour rules
// ---------------------------------------------------------------------------

/// TPT `ALMP::graphics` colour half (`ALMP.cpp:127-141`).
///
/// `z = tmp2 * 10 - 18` is the sparkle offset; the oxygen tint halves it with
/// C++ integer division and rebases it by 20. The spark/flare rolls that follow
/// in the reference only touch `pixel_mode` and live in [`modext_pixel_mode`].
#[cube]
pub(crate) fn almp_graphics(base: (f32, f32, f32), ctype: u32, tmp2: i32) -> (f32, f32, f32) {
    let mut z = tmp2 * 10i32 - 18i32;
    if ctype == PT_O2 {
        z = z / 2i32;
        z += 20i32;
    }
    let zf = z as f32;
    (base.0 + zf, base.1 + zf, base.2 + zf)
}

/// TPT `ALUM::graphics` (`ALUM.cpp:271-288`): oxidation brightens, alloying
/// darkens and shifts the balance towards `METL`'s grey.
#[cube]
pub(crate) fn alum_graphics(
    base: (f32, f32, f32),
    ctype: u32,
    tmp: i32,
    tmp3: i32,
) -> (f32, f32, f32) {
    let (mut r, mut g, mut b) = base;
    if tmp > 0i32 || ctype == PT_O2 {
        r += 40.0f32;
        g += 40.0f32;
        b += 40.0f32;
    }
    if tmp3 > 0i32 {
        r -= 50.0f32;
        g -= 40.0f32;
        b -= 20.0f32;
    }
    (r, g, b)
}

/// TPT `AMBE::graphics` (`AMBE.cpp:89-97`): a sensor at full life flashes
/// violet.
#[cube]
pub(crate) fn ambe_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    if life == 10i32 {
        (143.0f32, 0.0f32, 255.0f32)
    } else {
        base
    }
}

/// TPT `BFLM::graphics` colour half (`BFLM.cpp:82-92`).
///
/// `colg`/`colb` are `10` in both branches; only red tracks the flame state.
/// The fire channels are [`bflm_fire`]'s.
#[cube]
pub(crate) fn bflm_graphics(tmp: i32) -> (f32, f32, f32) {
    let r = if tmp > 0i32 { 120.0f32 } else { 10.0f32 };
    (r, 10.0f32, 10.0f32)
}

/// TPT `CLNT::graphics` (`CLNT.cpp:66-79`): hotter than the set point reads
/// red, colder blue, and the set point keeps the coolant green.
#[cube]
pub(crate) fn clnt_graphics(base: (f32, f32, f32), temp: f32, tmp: i32) -> (f32, f32, f32) {
    let target = tmp as f32;
    if temp - 294.15f32 > target {
        (200.0f32, 0.0f32, 50.0f32)
    } else if temp - 273.15f32 < target {
        (50.0f32, 0.0f32, 200.0f32)
    } else {
        base
    }
}

/// TPT `CLRC::graphics` (`CLRC.cpp:81-89`): a received UV ray darkens the
/// curable compound.
#[cube]
pub(crate) fn clrc_graphics(base: (f32, f32, f32), tmp: i32) -> (f32, f32, f32) {
    if tmp == 1i32 {
        (20.0f32, 20.0f32, 20.0f32)
    } else {
        base
    }
}

/// TPT `CMNT::graphics` (`CMNT.cpp:84-92`): curing darkens every channel by
/// `tmp / 3` (C++ integer division).
#[cube]
pub(crate) fn cmnt_graphics(base: (f32, f32, f32), tmp: i32) -> (f32, f32, f32) {
    if tmp != 0i32 {
        let d = (tmp / 3i32) as f32;
        (base.0 - d, base.1 - d, base.2 - d)
    } else {
        base
    }
}

/// TPT `COPR::graphics` colour half (`COPR.cpp:117-123`): charging copper
/// swaps red/blue for green.
#[cube]
pub(crate) fn copr_graphics(base: (f32, f32, f32), tmp: i32) -> (f32, f32, f32) {
    if tmp > 0i32 {
        let t = tmp as f32;
        (base.0 - t, base.1 + t, base.2 - t)
    } else {
        base
    }
}

/// TPT `DFOM::graphics` colour half (`DFOM.cpp:149-152`): frozen foam is red.
#[cube]
pub(crate) fn dfom_graphics(base: (f32, f32, f32), temp: f32) -> (f32, f32, f32) {
    if temp < 273.15f32 {
        (244.0f32, 84.0f32, 84.0f32)
    } else {
        base
    }
}

/// TPT `ECLR::graphics` colour half (`ECLR.cpp:75-81`): a live clone emitter
/// is pure red.
#[cube]
pub(crate) fn eclr_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    if life > 0i32 {
        (255.0f32, 0.0f32, 0.0f32)
    } else {
        base
    }
}

/// TPT `GRPH::graphics` colour half (`GRPH.cpp:150-166`).
///
/// Above `tmp2 = 4` the graphite reports its own conductivity in red; below
/// that the `tmp2` counter shifts every channel by `(tmp2 - 2) * 8`.
#[cube]
pub(crate) fn grph_graphics(base: (f32, f32, f32), tmp2: i32, tmp3: i32) -> (f32, f32, f32) {
    if tmp2 > 4i32 {
        (tmp3 as f32, 50.0f32, 50.0f32)
    } else {
        let z = ((tmp2 - 2i32) * 8i32) as f32;
        (base.0 + z, base.1 + z, base.2 + z)
    }
}

/// TPT `LED::graphics` colour half (`LED.cpp:79-84`): a fifth of the `dcolour`
/// deco colour. The reference divides the byte as an `int`, so the truncating
/// `u32` division is kept.
#[cube]
pub(crate) fn led_graphics(dcolour: u32) -> (f32, f32, f32) {
    (
        (((dcolour >> 16u32) & 0xFFu32) / 5u32) as f32,
        (((dcolour >> 8u32) & 0xFFu32) / 5u32) as f32,
        ((dcolour & 0xFFu32) / 5u32) as f32,
    )
}

/// TPT `PCON::graphics` (`PCON.cpp:127-135`): a fully powered conductor turns
/// red.
#[cube]
pub(crate) fn pcon_graphics(base: (f32, f32, f32), tmp2: i32) -> (f32, f32, f32) {
    if tmp2 == 10i32 {
        (191.0f32, 10.0f32, 10.0f32)
    } else {
        base
    }
}

/// TPT `PHOS::graphics` colour half (`PHOS.cpp:123-137`).
///
/// Red is capped at 155 before the `tmp` charge is added; green and blue lose
/// half of it (C++ integer division). The fire channels reuse this result, see
/// [`phos_fire`].
#[cube]
pub(crate) fn phos_graphics(base: (f32, f32, f32), tmp: i32) -> (f32, f32, f32) {
    let mut r = base.0;
    if r > 155.0f32 {
        r = 155.0f32;
    }
    let half = (tmp / 2i32) as f32;
    (r + tmp as f32, base.1 - half, base.2 - half)
}

/// TPT `PINV::graphics` (`PINV.cpp:77-85`): a full-life inverter is blue.
#[cube]
pub(crate) fn pinv_graphics(base: (f32, f32, f32), life: i32) -> (f32, f32, f32) {
    if life == 10i32 {
        (10.0f32, 10.0f32, 50.0f32)
    } else {
        base
    }
}

/// TPT `PPTI::graphics` colour half (`PPTI.cpp:194-196`): the pixel is painted
/// with the fire colour [`ppti_fire`] just computed.
#[cube]
pub(crate) fn ppti_graphics(tmp2: i32) -> (f32, f32, f32) {
    let f = ppti_fire(tmp2);
    (f.1 as f32, f.2 as f32, f.3 as f32)
}

/// TPT `PPTO::graphics` colour half (`PPTO.cpp:223-225`): the pixel is painted
/// with the fire colour [`ppto_fire`] just computed.
#[cube]
pub(crate) fn ppto_graphics(tmp2: i32) -> (f32, f32, f32) {
    let f = ppto_fire(tmp2);
    (f.1 as f32, f.2 as f32, f.3 as f32)
}

/// TPT `PRMT::graphics` (`PRMT.cpp:98-105`): the charge counter shifts every
/// channel by `(tmp2 - 2) * 8`.
#[cube]
pub(crate) fn prmt_graphics(base: (f32, f32, f32), tmp2: i32) -> (f32, f32, f32) {
    let z = ((tmp2 - 2i32) * 8i32) as f32;
    (base.0 + z, base.1 + z, base.2 + z)
}

/// TPT `QGPP::graphics` colour half (`QGPP.cpp:71-99`).
///
/// The reference runs three independent rules in order: the superheated
/// `tmp <= 150` branch paints a three-phase sine wave, the cold branch
/// (`temp < 373.15`) overrides it with violet, and the hot charged branch
/// (`temp >= 374.15 && tmp > 50`) with a red/`tmp2` ramp. A temperature in
/// `[373.15, 374.15)` keeps the sine colour.
#[cube]
pub(crate) fn qgpp_graphics(
    base: (f32, f32, f32),
    temp: f32,
    tmp: i32,
    tmp2: i32,
) -> (f32, f32, f32) {
    let (mut r, mut g, mut b) = base;
    if tmp <= 150i32 && temp >= 374.15f32 {
        // `(int)(sin(frequency * tmp + phase) * 127 + 150)`.
        let frequency = 0.04045f32;
        let t = tmp as f32;
        r = ((frequency * t + 4.0f32).sin() * 127.0f32 + 150.0f32) as i32 as f32;
        g = ((frequency * t + 8.0f32).sin() * 127.0f32 + 150.0f32) as i32 as f32;
        b = ((frequency * t + 5.0f32).sin() * 127.0f32 + 150.0f32) as i32 as f32;
    }
    if temp < 373.15f32 {
        (139.0f32, 0.0f32, 139.0f32)
    } else if temp >= 374.15f32 && tmp > 50i32 {
        let fifth = (tmp2 / 5i32) as f32;
        (240.0f32, fifth, fifth)
    } else {
        (r, g, b)
    }
}

/// TPT `RUBR::graphics` (`RUBR.cpp:117-130`): a torn rubber particle is grey,
/// an intact one keeps the pink table colour.
#[cube]
pub(crate) fn rubr_graphics(tmp: i32) -> (f32, f32, f32) {
    if tmp == 1i32 {
        (105.0f32, 105.0f32, 105.0f32)
    } else {
        (255.0f32, 217.0f32, 209.0f32)
    }
}

/// TPT `UVRD::graphics` colour half (`UVRD.cpp:110-120`): UV is a purple that
/// fades with `life`, and black once expired.
#[cube]
pub(crate) fn uvrd_graphics(life: i32) -> (f32, f32, f32) {
    if life > 0i32 {
        let c = (life + 50i32) as f32;
        (c, 0.0f32, c)
    } else {
        (0.0f32, 0.0f32, 0.0f32)
    }
}

// ---------------------------------------------------------------------------
// Fire channels (`firea/firer/fireg/fireb`)
// ---------------------------------------------------------------------------

/// TPT `ACTY::graphics` fire channels (`ACTY.cpp:111-127`).
///
/// Burning acetylene (`tmp > 0`) emits violet-blue at `tmp2` alpha; unburnt
/// acetylene emits a dim pale cyan at 8. Negative alphas clamp to zero like the
/// reference's post-`Graphics` clamp (`Renderer.cpp:473-474`).
#[cube]
pub(crate) fn acty_fire(tmp: i32, tmp2: i32) -> (u32, u32, u32, u32) {
    if tmp > 0i32 {
        (clampi(tmp2, 0i32, 255i32) as u32, 75u32, 75u32, 195u32)
    } else {
        (8u32, 173u32, 216u32, 220u32)
    }
}

/// TPT `BFLM::graphics` fire channels (`BFLM.cpp:84-99`): alpha 200 with the
/// same red/blue-white split as the pixel colour.
#[cube]
pub(crate) fn bflm_fire(tmp: i32) -> (u32, u32, u32, u32) {
    let r = if tmp > 0i32 { 140u32 } else { 10u32 };
    (200u32, r, 10u32, 10u32)
}

/// TPT `CHLR::graphics` fire channels (`CHLR.cpp:163-181`).
///
/// The reference writes `fireg = 295`, above the 8-bit range; TPT's clamp
/// (`Renderer.cpp:469-470`) and `fire_rgba`'s final pack both saturate it, so
/// the raw value is kept here. A charged chlorine can (`tmp != 0`) scales its
/// alpha by three.
#[cube]
pub(crate) fn chlr_fire(tmp: i32) -> (u32, u32, u32, u32) {
    if tmp == 0i32 {
        (15u32, 139u32, 295u32, 74u32)
    } else {
        (
            clampi(tmp * 3i32, 0i32, 255i32) as u32,
            205u32,
            205u32,
            255u32,
        )
    }
}

/// TPT `CLUD::graphics` (`CLUD.cpp:94-113`).
///
/// A cloud at `tmp` 995-998 is about to rain: rare lightning flare with a hard
/// fire colour. Every other cloud blends a white fire that fades as `tmp`
/// grows, at a random alpha of 0..=15 (`gfctx.rng.between(0, 15)`).
#[cube]
pub(crate) fn clud_fire(tmp: i32, pid: u32, tick: u32) -> (u32, u32, u32, u32) {
    if tmp > 994i32 && tmp < 999i32 {
        (20u32, 20u32, 20u32, 15u32)
    } else {
        let roll = clampi((hash01(pid ^ tick) * 16.0f32) as i32, 0i32, 15i32) as u32;
        let c = clampi(255i32 - tmp / 8i32, 0i32, 255i32) as u32;
        (roll, c, c, c)
    }
}

/// TPT `COPR::graphics` fire channels (`COPR.cpp:127-140`): a copper particle
/// that has burnt (`life > 1`) emits at alpha 30, blue while frozen and yellow
/// above 173.15 K.
///
/// The reference uses two independent `if`s. At exactly 173.15 K neither runs,
/// which leaves the fire colour black while the alpha is still 30, so the
/// initial value is kept there too.
#[cube]
pub(crate) fn copr_fire(temp: f32, life: i32) -> (u32, u32, u32, u32) {
    if life > 1i32 {
        let fire_a = 30u32;
        let mut fire_r = 0u32;
        let mut fire_g = 0u32;
        let mut fire_b = 0u32;
        if temp <= 173.15f32 {
            fire_b = 250u32;
        }
        if temp > 173.15f32 {
            fire_r = 250u32;
            fire_g = 250u32;
        }
        (fire_a, fire_r, fire_g, fire_b)
    } else {
        (0u32, 0u32, 0u32, 0u32)
    }
}

/// TPT `DFOM::graphics` fire channels (`DFOM.cpp:132-147`).
///
/// A deshaping foam (`tmp4` in 1..199) emits a random hue at alpha 250 - three
/// independent `rng.between(50, 255)` draws in the reference; every other state
/// emits the fixed grey at alpha 30.
#[cube]
pub(crate) fn dfom_fire(tmp4: i32, pid: u32, tick: u32) -> (u32, u32, u32, u32) {
    if tmp4 < 200i32 && tmp4 > 0i32 {
        let seed = pid ^ tick;
        let r = clampi(50i32 + (hash01(seed) * 206.0f32) as i32, 50i32, 255i32) as u32;
        let g = clampi(
            50i32 + (hash01(seed ^ 0x9E37_79B9u32) * 206.0f32) as i32,
            50i32,
            255i32,
        ) as u32;
        let b = clampi(
            50i32 + (hash01(seed ^ 0x85EB_CA6Bu32) * 206.0f32) as i32,
            50i32,
            255i32,
        ) as u32;
        (250u32, r, g, b)
    } else {
        (30u32, 112u32, 128u32, 144u32)
    }
}

/// TPT `LED::graphics` fire channels (`LED.cpp:86-90`): a lit diode emits its
/// `dcolour` at an alpha of its own temperature above freezing.
#[cube]
pub(crate) fn led_fire(dcolour: u32, temp: f32, life: i32) -> (u32, u32, u32, u32) {
    if life > 0i32 {
        (
            clampi((temp - 273.15f32) as i32, 0i32, 255i32) as u32,
            (dcolour >> 16u32) & 0xFFu32,
            (dcolour >> 8u32) & 0xFFu32,
            dcolour & 0xFFu32,
        )
    } else {
        (0u32, 0u32, 0u32, 0u32)
    }
}

/// TPT `PHOS::graphics` fire channels (`PHOS.cpp:135-143`): a phosphorus
/// particle carrying a `tmp3` charge emits its own pixel colour
/// ([`phos_graphics`]) at alpha 55.
#[cube]
pub(crate) fn phos_fire(base: (f32, f32, f32), tmp: i32, tmp3: i32) -> (u32, u32, u32, u32) {
    if tmp3 > 0i32 {
        let c = phos_graphics(base, tmp);
        (
            55u32,
            clampi(c.0 as i32, 0i32, 255i32) as u32,
            clampi(c.1 as i32, 0i32, 255i32) as u32,
            clampi(c.2 as i32, 0i32, 255i32) as u32,
        )
    } else {
        (0u32, 0u32, 0u32, 0u32)
    }
}

/// TPT `PPTI::graphics` fire channels (`PPTI.cpp:178-192`): red at alpha 20
/// while the portal inlet is empty, a dark red that emits nothing once a
/// particle is stored (`tmp2 > 0`). A negative `tmp2` leaves every channel at
/// the initial zero, which is why the colour goes black too.
///
/// The reference uses two independent `if`s, not an `else if`; the second one
/// overwrites the first, so the order is kept.
#[cube]
pub(crate) fn ppti_fire(tmp2: i32) -> (u32, u32, u32, u32) {
    let mut fire_a = 0u32;
    let mut fire_r = 0u32;
    let fire_g = 0u32;
    let fire_b = 0u32;
    if tmp2 == 0i32 {
        fire_a = 20u32;
        fire_r = 250u32;
    }
    if tmp2 > 0i32 {
        fire_r = 90u32;
    }
    (fire_a, fire_r, fire_g, fire_b)
}

/// TPT `PPTO::graphics` fire channels (`PPTO.cpp:207-221`): blue at alpha 20
/// while the outlet is empty, a dark blue that emits nothing once a particle is
/// stored (`tmp2 > 0`).
///
/// Two independent `if`s in the reference, like [`ppti_fire`].
#[cube]
pub(crate) fn ppto_fire(tmp2: i32) -> (u32, u32, u32, u32) {
    let mut fire_a = 0u32;
    let fire_r = 0u32;
    let fire_g = 0u32;
    let mut fire_b = 0u32;
    if tmp2 == 0i32 {
        fire_a = 20u32;
        fire_b = 250u32;
    }
    if tmp2 > 0i32 {
        fire_b = 90u32;
    }
    (fire_a, fire_r, fire_g, fire_b)
}

/// TPT `RADN::graphics` fire channels (`RADN.cpp:129-144`): an excited radon
/// atom (`tmp2 > 0`) emits a bright warm glow, a settled one a dim blue.
#[cube]
pub(crate) fn radn_fire(tmp2: i32) -> (u32, u32, u32, u32) {
    if tmp2 > 0i32 {
        (80u32, 200u32, 100u32, 100u32)
    } else {
        (30u32, 12u32, 24u32, 117u32)
    }
}

/// TPT `SUN::graphics` (`SUN.cpp:95-111`): below `tmp = 50` the sun emits an
/// orange glow, above it a blue one, both at alpha 60.
#[cube]
pub(crate) fn sun_fire(tmp: i32) -> (u32, u32, u32, u32) {
    if tmp <= 50i32 {
        (60u32, 190u32, 90u32, 0u32)
    } else {
        (60u32, 0u32, 0u32, 200u32)
    }
}

/// TPT `TIMC::graphics` fire channels (`TIMC.cpp:96-102`): a slow clock
/// (`tmp2 > 1`) emits a green glow at alpha 30.
#[cube]
pub(crate) fn timc_fire(tmp2: i32) -> (u32, u32, u32, u32) {
    if tmp2 > 1i32 {
        (30u32, 95u32, 235u32, 95u32)
    } else {
        (0u32, 0u32, 0u32, 0u32)
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Colour half of every mod-element `Graphics` function that writes one.
///
/// The colour rules read only `base`, `life`, `ctype`, `temp`, `dcolour` and
/// the `tmp*` counters; none of them draws a random colour, so `pid`, `vx`,
/// `vy`, `tick` and `tmp4` are part of the shared `GRAPHICS_FUNC_ARGS` surface
/// but unused here (the rolls of `ALMP`, `CLUD` and `DFOM` decide their
/// `pixel_mode`/fire channels only).
///
/// `ACTY`, `CHLR`, `CLUD`, `MGNT`, `PROJ`, `RADN`, `SUN` and `TIMC` never write
/// `colr/colg/colb`, so they fall through to the table colour exactly like
/// `Element::defaultGraphics` (`Element.cpp:230-249`).
#[cube]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(crate) fn modext_graphics_rgb(
    base: (f32, f32, f32),
    ty: u32,
    life: i32,
    ctype: u32,
    temp: f32,
    pid: u32,
    dcolour: u32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
    vx: f32,
    vy: f32,
    tick: u32,
) -> (f32, f32, f32) {
    if ty == PT_ALMP {
        almp_graphics(base, ctype, tmp2)
    } else if ty == PT_ALUM {
        alum_graphics(base, ctype, tmp, tmp3)
    } else if ty == PT_AMBE {
        ambe_graphics(base, life)
    } else if ty == PT_BFLM {
        bflm_graphics(tmp)
    } else if ty == PT_CLNT {
        clnt_graphics(base, temp, tmp)
    } else if ty == PT_CLRC {
        clrc_graphics(base, tmp)
    } else if ty == PT_CMNT {
        cmnt_graphics(base, tmp)
    } else if ty == PT_COPR {
        copr_graphics(base, tmp)
    } else if ty == PT_DFOM {
        dfom_graphics(base, temp)
    } else if ty == PT_ECLR {
        eclr_graphics(base, life)
    } else if ty == PT_GRPH {
        grph_graphics(base, tmp2, tmp3)
    } else if ty == PT_LED {
        led_graphics(dcolour)
    } else if ty == PT_PCON {
        pcon_graphics(base, tmp2)
    } else if ty == PT_PHOS {
        phos_graphics(base, tmp)
    } else if ty == PT_PINV {
        pinv_graphics(base, life)
    } else if ty == PT_PPTI {
        ppti_graphics(tmp2)
    } else if ty == PT_PPTO {
        ppto_graphics(tmp2)
    } else if ty == PT_PRMT {
        prmt_graphics(base, tmp2)
    } else if ty == PT_QGPP {
        qgpp_graphics(base, temp, tmp, tmp2)
    } else if ty == PT_RUBR {
        rubr_graphics(tmp)
    } else if ty == PT_UVRD {
        uvrd_graphics(life)
    } else {
        base
    }
}

/// `pixel_mode` half of every mod-element `Graphics` function.
///
/// Same starting value as TPT (`PMODE_FLAT`, `Renderer.cpp:289`, carried by the
/// element table row here), same operations: an unqualified assignment
/// (`*pixel_mode = ...`) replaces the mode and drops the flat pixel, an OR
/// (`|=`) keeps it. `ALUM`, `AMBE`, `CLRC`, `CMNT`, `MGNT`, `PCON`, `PINV`,
/// `PRMT` and `RUBR` never touch the mode.
///
/// `tmp4` belongs to the shared `GRAPHICS_FUNC_ARGS` surface; no mode rule in
/// this batch reads it (only `DFOM`'s fire channels do).
#[cube]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(crate) fn modext_pixel_mode(
    pixel_mode: u32,
    ty: u32,
    life: i32,
    ctype: u32,
    temp: f32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
    vx: f32,
    vy: f32,
    pid: u32,
    tick: u32,
) -> u32 {
    let mut mode = pixel_mode;

    // ACTY.cpp:127: every acetylene particle emits through the fire buffer only.
    if ty == PT_ACTY {
        mode = FIRE_BLEND;
    }

    // ALMP.cpp:146-154: sparks one time in six, and flares while the particle
    // moves faster than its own sparkle offset. The oxygen tint returns early
    // in the reference, so it gets neither roll.
    if ty == PT_ALMP {
        if ctype != PT_O2 {
            if hash01(pid ^ tick) * 6.0f32 < 1.0f32 {
                mode |= PMODE_SPARK;
            }
            let speed = (vx * vx + vy * vy).sqrt();
            if ((tmp2 + 1i32) as f32) < speed {
                mode |= PMODE_FLARE;
            }
        }
    }

    // BFLM.cpp:98-99: both fire stages run on the same particle.
    if ty == PT_BFLM {
        mode |= FIRE_BLEND;
        mode |= FIRE_ADD;
    }

    // CHLR.cpp:179.
    if ty == PT_CHLR {
        mode = FIRE_BLEND;
    }

    // CLNT.cpp:81: the coolant always flares over its flat pixel.
    if ty == PT_CLNT {
        mode |= PMODE_FLARE;
    }

    // CLUD.cpp:102,110: an about-to-rain cloud is a long flare, every other
    // cloud only feeds the fire buffer. Both branches replace the mode.
    if ty == PT_CLUD {
        if tmp > 994i32 && tmp < 999i32 {
            mode = PMODE_LFLARE;
        } else {
            mode = FIRE_BLEND;
        }
    }

    // COPR.cpp:139-140: only a particle that has burnt emits.
    if ty == PT_COPR && life > 1i32 {
        mode |= FIRE_ADD;
    }

    // DFOM.cpp:151,155: frozen foam flares over its flat pixel, a deshaping one
    // switches to the fire blend.
    if ty == PT_DFOM {
        if temp < 273.15f32 {
            mode |= PMODE_FLARE;
        } else {
            mode = FIRE_BLEND;
        }
    }

    // ECLR.cpp:82: a live emitter is a long flare.
    if ty == PT_ECLR && life > 0i32 {
        mode |= PMODE_LFLARE;
    }

    // GRPH.cpp:157: conducting graphite flares over its flat pixel.
    if ty == PT_GRPH && tmp2 > 4i32 {
        mode |= PMODE_FLARE;
    }

    // LED.cpp:90-92: a lit diode feeds the fire buffer; the `NO_DECO` that
    // follows is unconditional (and already part of the table row).
    if ty == PT_LED {
        if life > 0i32 {
            mode |= FIRE_ADD;
        }
        mode |= NO_DECO;
    }

    // PHOS.cpp:131,143: a charge below 100 burns as a flare, and any `tmp3`
    // charge also feeds the fire buffer.
    if ty == PT_PHOS {
        if tmp <= 100i32 {
            mode |= PMODE_FLARE;
        }
        if tmp3 > 0i32 {
            mode |= FIRE_ADD;
        }
    }

    // PPTI.cpp:197 / PPTO.cpp:226: both portals feed the fire buffer over their
    // flat pixel, whether or not a particle is stored.
    if ty == PT_PPTI || ty == PT_PPTO {
        mode |= FIRE_ADD;
    }

    // PROJ.cpp:104: a fired projectile is a long flare.
    if ty == PT_PROJ && life == 10i32 {
        mode |= PMODE_LFLARE;
    }

    // QGPP.cpp:73-98: the superheated plasma is a long flare; the cold branch
    // and the hot charged branch both add a plain flare. They are mutually
    // exclusive in the reference (`temp < 373.15` / `temp >= 374.15`), so the
    // identical body is one condition here.
    if ty == PT_QGPP {
        if tmp <= 400i32 && temp >= 374.15f32 {
            mode |= PMODE_LFLARE;
        }
        if temp < 373.15f32 || (temp >= 374.15f32 && tmp > 50i32) {
            mode |= PMODE_FLARE;
        }
    }

    // RADN.cpp:145: radon always emits through the fire buffer only.
    if ty == PT_RADN {
        mode = FIRE_BLEND;
    }

    // SUN.cpp:110: the sun feeds the fire buffer over its flat pixel.
    if ty == PT_SUN {
        mode |= FIRE_ADD;
    }

    // TIMC.cpp:102-104: a slow clock emits, and every clock flares.
    if ty == PT_TIMC {
        if tmp2 > 1i32 {
            mode |= FIRE_ADD;
        }
        mode |= PMODE_FLARE;
    }

    // UVRD.cpp:117,124: live UV is a bare long flare, an expired ray tints the
    // pixel with its deco colour only.
    if ty == PT_UVRD {
        if life > 0i32 {
            mode = PMODE_FLARE;
        } else {
            mode |= NO_DECO;
        }
    }

    mode
}
