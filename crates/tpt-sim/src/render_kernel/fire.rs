//! Fire buffer: TPT `render_parts`' fire accumulation plus `render_fire`.
//!
//! [`fire_rgba`] resolves the `firea/firer/fireg/fireb` quadruple of one
//! particle, [`accum_fire`] writes it into the coarse `fire_r/g/b` cells, and
//! the composite and decay passes turn those cells into the bloom that TPT
//! draws with `Renderer::render_fire`.
//!
//! The quadruple comes from the element table's `fire_colour` word
//! (`GpuElementProps::fire_colour`) for every element whose `Graphics` computes
//! it statically, and from the particle fields for the dynamic emitters; see
//! [`fire_rgba`] for the per-element references.
//!
//! # Deliberate deviations
//!
//! * TPT accumulates the fire cells while walking the particles in index
//!   order; the pixel-parallel passes here update the same cell from many
//!   pixels, so a concurrent update can be lost. The buffer only drives a soft
//!   additive bloom, so the difference is a sub-percent brightness offset.

use super::framebuffer::{
    add_fire_px, clampf, clampi, fb_blue, fb_green, fb_red, heat_temp_colour, pack_argb, pack_rgb,
};
use super::particles::{
    acty_fire, base_rgb, bflm_fire, bizr_graphics, chlr_fire, clud_fire, collapse_deco, copr_fire,
    deco_visible, dfom_fire, embr_graphics, exot_flicker, exot_graphics, grav_graphics, led_fire,
    particle_rgb, phos_fire, pipe_cargo, pipe_pixel_mode, ppti_fire, ppto_fire, radn_fire,
    rock_graphics, sun_fire, timc_fire, tron_graphics, tung_fire, vibr_graphics, wavelength_rgb,
};
use super::*;

/// TPT `TRON_DEATH` (`TRON.cpp:22`): crashed / dying tron flag.
const TRON_DEATH: i32 = 0x0000_0010;

/// Emissive colour of one particle as `0xAARRGGBB` (`a == 0` = no emission).
///
/// `fire_word` holds TPT's `firea/firer/fireg/fireb` from the element table. An
/// entry whose RGB is zero but whose alpha is not means "use the element's
/// `Graphics` colour", which is how the static emitters (FIRE, PLSM, LAVA,
/// PHOT) are encoded.
///
/// The elements whose `Graphics` computes its fire channels from the particle
/// fields override the table entry below; they are evaluated on the raw
/// `Graphics` colour, i.e. before `PROP_HOT_GLOW` and the display overrides,
/// exactly like TPT. Those are `GRAV` (`GRAV.cpp:106-111`), `VIBR`/`BVBR`
/// (`VIBR.cpp:228-247`), `ROCK` (`ROCK.cpp:60-65`), `GLOW` (`GLOW.cpp:96-99`),
/// `EXOT` (`EXOT.cpp:227-245`), `EMBR` (`EMBR.cpp:110-146`), `TUNG`
/// (`TUNG.cpp:131-142`), `VRSG` (`VRSG.cpp:60-63`), `BIZR`/`BIZRG`/`BIZRS`
/// (`BIZR.cpp:110-120`) and `C5` (`C5.cpp:117-135`); `DEUT` and `PIPE` override
/// only the alpha or the whole word from the table.
///
/// The mod's own emitters (`elements_modext.rs`) follow the same rule: `ACTY`,
/// `BFLM`, `CHLR`, `CLUD`, `COPR`, `DFOM`, `LED`, `PHOS`, `PPTI`, `PPTO`,
/// `RADN`, `SUN` and `TIMC`. Their table `fire_colour` is zero, so an arm
/// decides the whole quadruple; `MGNT`, `GRPH` and the other mod elements never
/// write a fire channel and keep the zeroed table entry.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fire_rgba(
    ty: u32,
    fire_word: u32,
    colour: u32,
    props: u32,
    high_temp: f32,
    temp: f32,
    life: i32,
    ctype: u32,
    pid: u32,
    dcolour: u32,
    colour_mode: u32,
    pretty_powder: u32,
    mode: u32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
    vx: f32,
    vy: f32,
    tick: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    decoration_level: u32,
    elem_props: &[u32],
    elem_count: u32,
) -> u32 {
    let mut a = (fire_word >> 24u32) & 0xFFu32;
    let mut r = (fire_word >> 16u32) & 0xFFu32;
    let mut g = (fire_word >> 8u32) & 0xFFu32;
    let mut b = fire_word & 0xFFu32;

    let (mut eval_ty, eval_temp, mut eval_life, mut eval_ctype) = (ty, temp, life, ctype);
    let (mut eval_tmp, mut eval_tmp2, mut eval_vx, mut eval_vy) = (tmp, tmp2, vx, vy);
    let (mut eval_tmp3, mut eval_tmp4) = (tmp3, tmp4);
    let (mut eval_colour, mut eval_props) = (colour, props);

    // PIPE.cpp:340-392, 421-448: a loaded pipe emits the cargo's fire channels
    // from the cargo's table row and re-evaluates the cargo's dynamic Graphics fire.
    if ty == PT_PIPE || ty == PT_PPIP {
        let cargo = pipe_cargo(ctype);
        if cargo != 0u32 {
            let slot = (cargo % elem_count) * ELEM_WORDS;
            let word = elem_props[(slot + ELEM_FIRE_COLOUR) as usize];
            a = (word >> 24u32) & 0xFFu32;
            r = (word >> 16u32) & 0xFFu32;
            g = (word >> 8u32) & 0xFFu32;
            b = word & 0xFFu32;

            eval_ty = cargo;
            eval_life = tmp2;
            eval_tmp = tmp3;
            eval_ctype = tmp4 as u32;
            eval_tmp2 = 0i32;
            eval_tmp3 = 0i32;
            eval_tmp4 = 0i32;
            eval_colour = elem_props[(slot + ELEM_COLOUR) as usize];
            eval_props = elem_props[(slot + ELEM_PROPS) as usize];
            if (eval_props & 0x0000_0008u32) == 0u32 {
                // TYPE_ENERGY
                eval_vx = 0.0f32;
                eval_vy = 0.0f32;
            }
        } else {
            a = 0u32;
            r = 0u32;
            g = 0u32;
            b = 0u32;
        }
    }

    // Dynamic `Graphics` fire outputs.
    if eval_ty == PT_DEUT {
        // `DEUT.cpp:148-157`: above life 240 the deuterium emits its own pixel
        // colour (the base brightened by 255) at alpha 60; below that it emits
        // no fire at all.
        if eval_life >= 240i32 {
            let base = base_rgb(eval_colour, eval_props, pid, pretty_powder);
            let c = deut_graphics(base, eval_life);
            a = 60u32;
            r = clampf(c.0, 0.0f32, 255.0f32) as u32;
            g = clampf(c.1, 0.0f32, 255.0f32) as u32;
            b = clampf(c.2, 0.0f32, 255.0f32) as u32;
        } else {
            a = 0u32;
        }
    } else if eval_ty == PT_GRAV && eval_life != 0i32 {
        // GRAV.cpp:106-111: three times the velocity-ramped pixel colour.
        let c = grav_graphics(eval_vx, eval_vy, tick);
        a = clampi((eval_life + 15i32) * 4i32, 0i32, 255i32) as u32;
        r = clampf(c.0 * 3.0f32, 0.0f32, 255.0f32) as u32;
        g = clampf(c.1 * 3.0f32, 0.0f32, 255.0f32) as u32;
        b = clampf(c.2 * 3.0f32, 0.0f32, 255.0f32) as u32;
    } else if eval_ty == PT_VIBR || eval_ty == PT_BVBR {
        // VIBR.cpp:228-232 (armed) / 244-247 (arming): the fire colour is the
        // pixel colour, the strength follows the gradient.
        let base = base_rgb(eval_colour, eval_props, pid, pretty_powder);
        let c = vibr_graphics(base, eval_life, eval_tmp, eval_tmp2);
        let gradient = eval_tmp / 10i32;
        if gradient >= 100i32 || eval_life != 0i32 {
            a = 90u32;
            r = clampf(c.0, 0.0f32, 255.0f32) as u32;
            g = clampf(c.1, 0.0f32, 255.0f32) as u32;
            b = clampf(c.2, 0.0f32, 255.0f32) as u32;
        } else {
            // `(int)restrict_flt(gradient * 0.6f, 0, 60)` and `*colr / 2`.
            a = clampf(gradient as f32 * 0.6f32, 0.0f32, 60.0f32) as u32;
            r = (clampf(c.0, 0.0f32, 255.0f32) as i32 / 2i32) as u32;
            g = (clampf(c.1, 0.0f32, 255.0f32) as i32 / 2i32) as u32;
            b = (clampf(c.2, 0.0f32, 255.0f32) as i32 / 2i32) as u32;
        }
    } else if eval_ty == PT_ROCK && eval_temp >= 810.15f32 {
        // ROCK.cpp:60-65: the hot-rock glow ramps up with temperature.
        let base = base_rgb(eval_colour, eval_props, pid, pretty_powder);
        let c = rock_graphics(base, eval_tmp2);
        a = clampf((eval_temp - 810.15f32) / 45.0f32, 0.0f32, 255.0f32) as u32;
        r = clampf(c.0, 0.0f32, 255.0f32) as u32;
        g = clampf(c.1, 0.0f32, 255.0f32) as u32;
        b = clampf(c.2, 0.0f32, 255.0f32) as u32;
    } else if eval_ty == PT_GLOW {
        // GLOW.cpp:96-99: fire from temperature, `ctype` and `tmp` at alpha 64.
        a = 64u32;
        r = 16u32 + (clampf(eval_temp - 307.15f32, 0.0f32, 128.0f32) / 2.0f32) as u32;
        g = 16u32 + (clampf(eval_ctype as f32, 0.0f32, 128.0f32) / 2.0f32) as u32;
        b = 16u32 + (clampf(eval_tmp as f32, 0.0f32, 128.0f32) / 2.0f32) as u32;
    } else if eval_ty == PT_EXOT {
        // EXOT.cpp:205-249: the flicker branch sets `firea = 100` with a black
        // fire colour (and no `FIRE_*` mode bit), the regular and superheated
        // branches set `firea = cola = tmp / 6` and the fire colour to the
        // pixel colour.
        if eval_life < 1001i32 && exot_flicker(eval_tmp2, pid, tick) {
            a = 100u32;
            r = 0u32;
            g = 0u32;
            b = 0u32;
        } else {
            let c = exot_graphics(eval_temp, eval_tmp, eval_tmp2, eval_life, pid, tick);
            a = clampi(eval_tmp / 6i32, 0i32, 255i32) as u32;
            r = clampf(c.0, 0.0f32, 255.0f32) as u32;
            g = clampf(c.1, 0.0f32, 255.0f32) as u32;
            b = clampf(c.2, 0.0f32, 255.0f32) as u32;
        }
    } else if eval_ty == PT_EMBR {
        // EMBR.cpp:97-146: the fire channels are the pixel colour (the
        // embedded `ctype` colour, white while `tmp` is set, else the table
        // colour), tinted by the deco colour inside the same `graphics` call.
        // `tmp == 1` fades in with `life`, `tmp == 2` is a flat glow, and the
        // plain spark emits no fire at all.
        let base = base_rgb(eval_colour, eval_props, pid, pretty_powder);
        let mut c = embr_graphics(base, eval_ctype, eval_tmp);
        // `EMBR.cpp:104-122`: the tint is gated by the decoration level alone
        // (unlike the general deco blend, no colour mode suppresses it).
        if deco_visible(dcolour, decoration_level) {
            // The blend uses the raw alpha (`a`), not `deca + 1`, and divides
            // by 256 (`>> 8`) with `255 - a`.
            let deca = ((dcolour >> 24u32) & 0xFFu32) as f32;
            let decr = ((dcolour >> 16u32) & 0xFFu32) as f32;
            let decg = ((dcolour >> 8u32) & 0xFFu32) as f32;
            let decb = (dcolour & 0xFFu32) as f32;
            c = (
                (deca * decr + (255.0f32 - deca) * c.0) / 256.0f32,
                (deca * decg + (255.0f32 - deca) * c.1) / 256.0f32,
                (deca * decb + (255.0f32 - deca) * c.2) / 256.0f32,
            );
        }
        if eval_tmp == 1i32 {
            a = clampi((eval_life - 15i32) * 4i32, 0i32, 255i32) as u32;
        } else if eval_tmp == 2i32 {
            a = 255u32;
        } else {
            a = 0u32;
        }
        r = clampf(c.0, 0.0f32, 255.0f32) as u32;
        g = clampf(c.1, 0.0f32, 255.0f32) as u32;
        b = clampf(c.2, 0.0f32, 255.0f32) as u32;
    } else if eval_ty == PT_TUNG {
        // TUNG.cpp:131-142: a fixed alpha 30 under the glow sinusoid.
        let (fa, fr, fg, fb) = tung_fire(eval_temp);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_VRSG {
        // VRSG.cpp:60-63: half the pixel colour at alpha 125. `VRSG` has no
        // other colour rule, so this is the table colour halved.
        let base = base_rgb(eval_colour, eval_props, pid, pretty_powder);
        a = 125u32;
        r = clampf(base.0, 0.0f32, 255.0f32) as u32 / 2u32;
        g = clampf(base.1, 0.0f32, 255.0f32) as u32 / 2u32;
        b = clampf(base.2, 0.0f32, 255.0f32) as u32 / 2u32;
    } else if eval_ty == PT_BIZR || eval_ty == PT_BIZRG || eval_ty == PT_BIZRS {
        // BIZR.cpp:110-120: the pixel colour scaled by `(|vx| + |vy|) / 5`,
        // emitted only while the particle is moving.
        let base = base_rgb(eval_colour, eval_props, pid, pretty_powder);
        let c = bizr_graphics(base, eval_ctype);
        let ax = if eval_vx < 0.0f32 { -eval_vx } else { eval_vx };
        let ay = if eval_vy < 0.0f32 { -eval_vy } else { eval_vy };
        let speed = ax + ay;
        if speed > 0.0f32 {
            let brightness = speed / 5.0f32;
            a = 255u32;
            r = clampf(c.0 * brightness, 0.0f32, 255.0f32) as u32;
            g = clampf(c.1 * brightness, 0.0f32, 255.0f32) as u32;
            b = clampf(c.2 * brightness, 0.0f32, 255.0f32) as u32;
        } else {
            a = 0u32;
        }
    } else if eval_ty == PT_C5 {
        // C5.cpp:117-135: no stored wavelength means the early return, which
        // leaves `firea` at zero; otherwise the wavelength colour at alpha 100.
        if eval_ctype == 0u32 {
            a = 0u32;
        } else {
            let c = wavelength_rgb(eval_ctype);
            a = 100u32;
            r = clampf(c.0, 0.0f32, 255.0f32) as u32;
            g = clampf(c.1, 0.0f32, 255.0f32) as u32;
            b = clampf(c.2, 0.0f32, 255.0f32) as u32;
        }
    } else if eval_ty == PT_TRON && (eval_tmp & TRON_DEATH) != 0i32 {
        // TRON.cpp:108-114: dying TRON emits fire at full alpha in its colour.
        let c = tron_graphics(eval_tmp);
        a = 255u32;
        r = clampf(c.0, 0.0f32, 255.0f32) as u32;
        g = clampf(c.1, 0.0f32, 255.0f32) as u32;
        b = clampf(c.2, 0.0f32, 255.0f32) as u32;
    } else if eval_ty == PT_BRCK && eval_tmp == 1i32 {
        // BRCK.cpp:58-63: fire from brick at alpha 40 with boosted blue.
        let base = base_rgb(eval_colour, eval_props, pid, pretty_powder);
        a = 40u32;
        r = clampf(base.0, 0.0f32, 255.0f32) as u32;
        g = clampf(base.1, 0.0f32, 255.0f32) as u32;
        b = clampf(base.2 + 100.0f32, 0.0f32, 255.0f32) as u32;
    } else if eval_ty == PT_PRTI {
        // PRTI.cpp:131-134: red fire emission at alpha 8.
        a = 8u32;
        r = 255u32;
        g = 0u32;
        b = 0u32;
    } else if eval_ty == PT_PRTO {
        // PRTO.cpp:92-95: blue fire emission at alpha 8.
        a = 8u32;
        r = 0u32;
        g = 0u32;
        b = 255u32;
    } else if eval_ty == PT_ACTY {
        // ACTY.cpp:111-127: burning acetylene emits the violet-blue fire whose
        // alpha is the stored `tmp2`; unburnt acetylene a dim pale cyan.
        let (fa, fr, fg, fb) = acty_fire(eval_tmp, eval_tmp2);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_BFLM {
        // BFLM.cpp:82-99: alpha 200, blue-white fire while burning and the red
        // one from the `tmp > 0` branch.
        let (fa, fr, fg, fb) = bflm_fire(eval_tmp);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_CHLR {
        // CHLR.cpp:163-181: chlorine's green fire, its alpha scaled by `tmp`
        // when the can is charged.
        let (fa, fr, fg, fb) = chlr_fire(eval_tmp);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_CLUD {
        // CLUD.cpp:94-113: an about-to-rain cloud flashes a fixed fire, every
        // other cloud emits a white fire that fades with `tmp` at a rolled
        // alpha (`gfctx.rng.between(0, 15)`).
        let (fa, fr, fg, fb) = clud_fire(eval_tmp, pid, tick);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_COPR {
        // COPR.cpp:127-140: a copper particle that has burnt emits at alpha 30,
        // blue below 173.15 K and yellow above it.
        let (fa, fr, fg, fb) = copr_fire(eval_temp, eval_life);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_DFOM {
        // DFOM.cpp:130-147: a deshaping foam emits a rolled hue at alpha 250,
        // every other state the fixed grey at alpha 30.
        let (fa, fr, fg, fb) = dfom_fire(eval_tmp4, pid, tick);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_LED {
        // LED.cpp:86-90: a lit diode emits its `dcolour` at an alpha of its own
        // temperature above freezing.
        let (fa, fr, fg, fb) = led_fire(dcolour, eval_temp, eval_life);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_PHOS {
        // PHOS.cpp:135-143: a charged phosphorus particle emits its own pixel
        // colour (the `PHOS::graphics` result) at alpha 55.
        let base = base_rgb(eval_colour, eval_props, pid, pretty_powder);
        let (fa, fr, fg, fb) = phos_fire(base, eval_tmp, eval_tmp3);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_PPTI {
        // PPTI.cpp:178-192: red at alpha 20 while empty, it stops emitting once
        // a particle is stored.
        let (fa, fr, fg, fb) = ppti_fire(eval_tmp2);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_PPTO {
        // PPTO.cpp:207-221: blue at alpha 20 while empty, it stops emitting once
        // a particle is stored.
        let (fa, fr, fg, fb) = ppto_fire(eval_tmp2);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_RADN {
        // RADN.cpp:129-144: an excited atom emits a warm glow, a settled one a
        // dim blue.
        let (fa, fr, fg, fb) = radn_fire(eval_tmp2);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_SUN {
        // SUN.cpp:95-111: an orange glow below `tmp = 50`, a blue one above it,
        // both at alpha 60.
        let (fa, fr, fg, fb) = sun_fire(eval_tmp);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    } else if eval_ty == PT_TIMC {
        // TIMC.cpp:96-102: a slow clock emits a green glow at alpha 30.
        let (fa, fr, fg, fb) = timc_fire(eval_tmp2);
        a = fa;
        r = fr;
        g = fg;
        b = fb;
    }

    if (colour_mode & COLOUR_HEAT) != 0u32 {
        // `Renderer.cpp:375-384`: the heat display paints both the pixel and
        // the fire channels with `heatTableAt((temp - hdispMin) /
        // (hdispMax - hdispMin) * 1024)`, i.e. the same resolved limits the air
        // overlay uses.
        let col = heat_temp_colour(temp, hdisp_min, hdisp_max);
        a = 255u32;
        r = fb_red(col) as u32;
        g = fb_green(col) as u32;
        b = fb_blue(col) as u32;
    } else if a != 0u32 && r == 0u32 && g == 0u32 && b == 0u32 {
        let col = particle_rgb(
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
        r = fb_red(col) as u32;
        g = fb_green(col) as u32;
        b = fb_blue(col) as u32;
    }

    // TPT `DECO_FIRE`: `LIGH.cpp:304-308` sets the bit, `Renderer.cpp:428-443`
    // tints the fire channels with the `dcolour` decoration, `deca + 1` of 256,
    // after the anti-clickbait collapse. The gate is the collapsed alpha byte,
    // the decoration level and any colour mode other than `COLOUR_GRAD`.
    if (mode & DECO_FIRE) != 0u32
        && decoration_level != DECORATION_DISABLED
        && (colour_mode & NOT_GRAD_ONLY) == 0u32
    {
        let (deca0, decr, decg, decb) = collapse_deco(dcolour, decoration_level);
        let deca = deca0 + 1u32;
        if deca > 1u32 {
            r = (deca * decr + (256u32 - deca) * r) >> 8u32;
            g = (deca * decg + (256u32 - deca) * g) >> 8u32;
            b = (deca * decb + (256u32 - deca) * b) >> 8u32;
        }
    }

    // TPT clamps all four channels before drawing (`Renderer.cpp:463-474`).
    pack_argb(
        clampi(a as i32, 0i32, 255i32) as u32,
        clampi(r as i32, 0i32, 255i32) as u32,
        clampi(g as i32, 0i32, 255i32) as u32,
        clampi(b as i32, 0i32, 255i32) as u32,
    )
}

/// Accumulate one emissive particle into its coarse `fire_r/g/b` cell.
///
/// TPT `render_parts`:
/// `FIRE_BLEND` blends `firea/2`, `FIRE_ADD` adds `firea/8` (clamped), and
/// `FIRE_SPARK` blends `firea/4` into the particle's coarse cell.
///
/// The `firea/firer/fireg/fireb` quadruple comes from [`fire_rgba`], which also
/// applies the `dcolour` `DECO_FIRE` tint; `part_dcolour` is bound for that.
///
/// TPT applies the three rules particle by particle, in slot order, on the
/// shared `fire_r/g/b` cells. A pixel-parallel kernel cannot: the 16 pixels of
/// one cell would each do a read-modify-write of the same word, and every write
/// but the last is lost. The driver therefore runs one thread per cell and
/// threads the accumulated value through the pixels of that cell, which
/// reproduces TPT's sum exactly - a cell holding 16 `LAVA` pixels adds its 16
/// contributions instead of one.
#[cube]
#[allow(clippy::too_many_arguments)]
fn accum_fire(
    packed: u32,
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
    cell_in_r: u32,
    cell_in_g: u32,
    cell_in_b: u32,
    render_mode: u32,
    colour_mode: u32,
    decoration_level: u32,
    pretty_powder: u32,
    elem_count: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    tick: u32,
) -> (u32, u32, u32) {
    let mut cell_r = cell_in_r;
    let mut cell_g = cell_in_g;
    let mut cell_b = cell_in_b;
    if packed != 0u32 {
        let pid = packed >> PMAP_ID_SHIFT;
        if pid != 0u32 {
            let slot = (pid - 1u32) as usize;
            if slot < part_life.len() {
                let ty = packed & PMAP_TYPE_MASK;
                let base = (ty % elem_count) * ELEM_WORDS;
                let life = part_life[slot];
                let temp = part_temp[slot];
                let ctype = part_ctype[slot];
                let tmp = part_tmp[slot];
                let tmp2 = part_tmp2[slot];
                let tmp3 = part_tmp3[slot];
                let tmp4 = part_tmp4[slot];
                // `PIPE.cpp:349-352`: a loaded pipe feeds the fire buffer with
                // the cargo's `FIRE_*` bits, not the pipe's.
                let mode = effective_pixel_mode(
                    pipe_pixel_mode(
                        elem_props[(base + ELEM_PIXEL_MODE) as usize],
                        ty,
                        ctype,
                        elem_props,
                        elem_count,
                    ),
                    ty,
                    life,
                    ctype,
                    temp,
                    tmp,
                    tmp2,
                    tmp3,
                    tmp4,
                    0u32,
                    pid,
                    tick,
                    render_mode,
                    colour_mode,
                    part_vx[slot],
                    part_vy[slot],
                );
                if (mode & FIREMODE) != 0u32 {
                    let fw = fire_rgba(
                        ty,
                        elem_props[(base + ELEM_FIRE_COLOUR) as usize],
                        elem_props[(base + ELEM_COLOUR) as usize],
                        elem_props[(base + ELEM_PROPS) as usize],
                        f32::reinterpret(elem_props[(base + ELEM_HIGH_TEMP) as usize]),
                        temp,
                        life,
                        ctype,
                        pid,
                        part_dcolour[slot],
                        colour_mode,
                        pretty_powder,
                        mode,
                        tmp,
                        tmp2,
                        tmp3,
                        tmp4,
                        part_vx[slot],
                        part_vy[slot],
                        tick,
                        hdisp_min,
                        hdisp_max,
                        decoration_level,
                        elem_props,
                        elem_count,
                    );
                    let mut fa = (fw >> 24u32) & 0xFFu32;
                    let fr = (fw >> 16u32) & 0xFFu32;
                    let fg = (fw >> 8u32) & 0xFFu32;
                    let fb = fw & 0xFFu32;
                    if fa != 0u32 && (mode & FIRE_BLEND) != 0u32 {
                        fa /= 2u32;
                        cell_r = (fa * fr + (255u32 - fa) * cell_r) >> 8u32;
                        cell_g = (fa * fg + (255u32 - fa) * cell_g) >> 8u32;
                        cell_b = (fa * fb + (255u32 - fa) * cell_b) >> 8u32;
                    }
                    if fa != 0u32 && (mode & FIRE_ADD) != 0u32 {
                        fa /= 8u32;
                        let mut nr = ((fa * fr) >> 8u32) + cell_r;
                        let mut ng = ((fa * fg) >> 8u32) + cell_g;
                        let mut nb = ((fa * fb) >> 8u32) + cell_b;
                        if nr > 255u32 {
                            nr = 255u32;
                        }
                        if ng > 255u32 {
                            ng = 255u32;
                        }
                        if nb > 255u32 {
                            nb = 255u32;
                        }
                        cell_r = nr;
                        cell_g = ng;
                        cell_b = nb;
                    }
                    if fa != 0u32 && (mode & FIRE_SPARK) != 0u32 {
                        fa /= 4u32;
                        cell_r = (fa * fr + (255u32 - fa) * cell_r) >> 8u32;
                        cell_g = (fa * fg + (255u32 - fa) * cell_g) >> 8u32;
                        cell_b = (fa * fb + (255u32 - fa) * cell_b) >> 8u32;
                    }
                }
            }
        }
    }
    (cell_r, cell_g, cell_b)
}

/// Accumulates the emissive matter particles into `fire_r/g/b`, one thread per
/// coarse cell so the 16 pixels of a cell accumulate without a race.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn render_fire_accum_matter_kernel(
    pmap: &[u32],
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
    fire_r: &mut [u32],
    fire_g: &mut [u32],
    fire_b: &mut [u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    render_mode: u32,
    colour_mode: u32,
    decoration_level: u32,
    pretty_powder: u32,
    elem_count: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    tick: u32,
) {
    let cell = ABSOLUTE_POS as u32;
    if cell < xcells * ycells {
        let cx = cell % xcells;
        let cy = cell / xcells;
        let mut cr = fire_r[cell as usize];
        let mut cg = fire_g[cell as usize];
        let mut cb = fire_b[cell as usize];
        for j in 0u32..4u32 {
            for i in 0u32..4u32 {
                let x = cx * 4u32 + i;
                let y = cy * 4u32 + j;
                if x < xres && y < yres {
                    let (nr, ng, nb) = accum_fire(
                        pmap[(y * xres + x) as usize],
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
                        cr,
                        cg,
                        cb,
                        render_mode,
                        colour_mode,
                        decoration_level,
                        pretty_powder,
                        elem_count,
                        hdisp_min,
                        hdisp_max,
                        tick,
                    );
                    cr = nr;
                    cg = ng;
                    cb = nb;
                }
            }
        }
        fire_r[cell as usize] = cr;
        fire_g[cell as usize] = cg;
        fire_b[cell as usize] = cb;
    }
}

/// Accumulates the emissive energy particles (PHOT, NEUT, ELEC, GRVT) into
/// `fire_r/g/b`, one thread per coarse cell for the same reason as the matter
/// kernel: the 16 pixels of a cell share one word.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn render_fire_accum_energy_kernel(
    photons: &[u32],
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
    fire_r: &mut [u32],
    fire_g: &mut [u32],
    fire_b: &mut [u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    render_mode: u32,
    colour_mode: u32,
    decoration_level: u32,
    pretty_powder: u32,
    elem_count: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    tick: u32,
) {
    let cell = ABSOLUTE_POS as u32;
    if cell < xcells * ycells {
        let cx = cell % xcells;
        let cy = cell / xcells;
        let mut cr = fire_r[cell as usize];
        let mut cg = fire_g[cell as usize];
        let mut cb = fire_b[cell as usize];
        for j in 0u32..4u32 {
            for i in 0u32..4u32 {
                let x = cx * 4u32 + i;
                let y = cy * 4u32 + j;
                if x < xres && y < yres {
                    let (nr, ng, nb) = accum_fire(
                        photons[(y * xres + x) as usize],
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
                        cr,
                        cg,
                        cb,
                        render_mode,
                        colour_mode,
                        decoration_level,
                        pretty_powder,
                        elem_count,
                        hdisp_min,
                        hdisp_max,
                        tick,
                    );
                    cr = nr;
                    cg = ng;
                    cb = nb;
                }
            }
        }
        fire_r[cell as usize] = cr;
        fire_g[cell as usize] = cg;
        fire_b[cell as usize] = cb;
    }
}

/// Spreads `fire_r/g/b` into the per-pixel glow layer with TPT's `fire_alpha`
/// kernel.
///
/// TPT precomputes a 12x12 alpha table (`Renderer::prepare_alpha`, an 8x8
/// Gaussian of `exp(-0.1 * (dx^2 + dy^2))` folded with the 4x4 cell, scaled by
/// `255 * fireIntensity`) and adds each coarse cell over the 12x12 pixel block
/// around it. The table is uploaded once; the intensity scales it here.
///
/// The nine-tap gather only depends on the fire cells, which change at most
/// once per simulation tick, so it is resolved into `layer` on that tick and
/// [`render_fire_add_kernel`] re-applies the result every presented frame. The
/// layer holds the same packed `0xAABBGGRR` words as the framebuffer, and
/// saturating there and again on the final add is equivalent to TPT's clamp
/// after every tap because clamping is monotone.
#[cube(launch_unchecked)]
pub fn render_fire_spread_kernel(
    fire_r: &[u32],
    fire_g: &[u32],
    fire_b: &[u32],
    fire_alpha: &[u32],
    layer: &mut [u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
    fire_intensity: f32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xres * yres {
        let x = idx % xres;
        let y = idx / xres;
        let cx = x >> 2u32;
        let cy = y >> 2u32;

        // Accumulate in the layer's own channel order (`pack_rgb` is BGRA).
        let mut sr = 0.0f32;
        let mut sg = 0.0f32;
        let mut sb = 0.0f32;

        for cj in 0u32..3u32 {
            for ci in 0u32..3u32 {
                let sx = cx as i32 + ci as i32 - 1i32;
                let sy = cy as i32 + cj as i32 - 1i32;
                if sx >= 0i32 && sy >= 0i32 && (sx as u32) < xcells && (sy as u32) < ycells {
                    let cell = (sy as u32) * xcells + (sx as u32);
                    let cr = fire_r[cell as usize];
                    let cg = fire_g[cell as usize];
                    let cb = fire_b[cell as usize];
                    if cr != 0u32 || cg != 0u32 || cb != 0u32 {
                        let dx = x as i32 - sx * 4i32 + 4i32;
                        let dy = y as i32 - sy * 4i32 + 4i32;
                        let a = (fire_alpha[(dy * FIRE_ALPHA_SPAN + dx) as usize] as f32)
                            * fire_intensity
                            / 255.0f32;
                        sr += a * cr as f32;
                        sg += a * cg as f32;
                        sb += a * cb as f32;
                    }
                }
            }
        }

        layer[idx as usize] = pack_rgb(sr, sg, sb);
    }
}

/// Adds the cached fire glow layer onto the framebuffer.
///
/// One tap per pixel instead of nine: the layer already carries the clamped
/// sum, and [`add_fire_px`] with a full-weight alpha reproduces TPT's additive
/// blend.
#[cube(launch_unchecked)]
pub fn render_fire_add_kernel(framebuf: &mut [u32], layer: &[u32], xres: u32, yres: u32) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xres * yres {
        let glow = layer[idx as usize];
        let r = fb_red(glow);
        let g = fb_green(glow);
        let b = fb_blue(glow);
        if r != 0.0f32 || g != 0.0f32 || b != 0.0f32 {
            framebuf[idx as usize] = add_fire_px(framebuf[idx as usize], r, g, b, 255.0f32);
        }
    }
}

/// Diffuses and decays `fire_r/g/b` for the next frame (TPT `render_fire` tail).
///
/// `r = (8 * r_self + sum of the 8 neighbours) / 16`, then `-4` when above 4.
/// TPT runs this in place in scan order; here every cell is independent, so a
/// neighbour read may see the old or the new value. The decay is a slow blur, so
/// the difference is a sub-percent per-frame offset.
// `saturating_sub` has no CubeCL expansion, so the clamped decrement is written out with `if`.
#[allow(clippy::implicit_saturating_sub)]
#[cube(launch_unchecked)]
pub fn render_fire_decay_kernel(
    fire_r: &mut [u32],
    fire_g: &mut [u32],
    fire_b: &mut [u32],
    xcells: u32,
    ycells: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < xcells * ycells {
        let cx = idx % xcells;
        let cy = idx / xcells;
        let mut sr = fire_r[idx as usize] * 8u32;
        let mut sg = fire_g[idx as usize] * 8u32;
        let mut sb = fire_b[idx as usize] * 8u32;
        for j in 0u32..3u32 {
            for i in 0u32..3u32 {
                if i != 1u32 || j != 1u32 {
                    let nx = cx as i32 + i as i32 - 1i32;
                    let ny = cy as i32 + j as i32 - 1i32;
                    if nx >= 0i32 && ny >= 0i32 && (nx as u32) < xcells && (ny as u32) < ycells {
                        let c = ((ny as u32) * xcells + (nx as u32)) as usize;
                        sr += fire_r[c];
                        sg += fire_g[c];
                        sb += fire_b[c];
                    }
                }
            }
        }
        let r = sr / 16u32;
        let g = sg / 16u32;
        let b = sb / 16u32;
        fire_r[idx as usize] = if r > 4u32 { r - 4u32 } else { 0u32 };
        fire_g[idx as usize] = if g > 4u32 { g - 4u32 } else { 0u32 };
        fire_b[idx as usize] = if b > 4u32 { b - 4u32 } else { 0u32 };
    }
}
