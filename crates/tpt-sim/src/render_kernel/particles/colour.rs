//! Particle colouring stages and element graphics dispatch.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::framebuffer::{
    clampi, fb_blue, fb_green, fb_red, flame_colour, hash01, heat_temp_colour, pack_rgb,
    plasma_colour, rgb_from_u32,
};
use super::super::{
    DECORATION_DISABLED, ELEM_COLA, ELEM_COLOUR, ELEM_FLAMMABLE, ELEM_HARDNESS,
    ELEM_HIGH_PRESSURE_TRANSITION, ELEM_HIGH_TEMP_TRANSITION, ELEM_LOW_PRESSURE_TRANSITION,
    ELEM_LOW_TEMP_TRANSITION, ELEM_PROPS, ELEM_WORDS, NO_DECO, NOT_GRAD_ONLY,
};
use super::TRON_HEAD;
use super::elements_basic::*;
use super::elements_special::*;

/// Element `Graphics` colour overrides, in TPT's element order.
///
/// Every rule whose colour reads `tmp`, `tmp2`, `vx`, `vy` or the tick is
/// ported. The non-colour outputs of the same functions live next to it:
/// [`graphics_cola`] carries `cola`, `fire_rgba` the `firea/firer/fireg/
/// fireb` quadruple and [`element_pixel_mode`] the dynamic mode bits, so
/// `FILT`, `BRAY`, `INVIS`, `WARP`, `EXOT`, `GRAV`, `VIBR`, `ROCK`, `GLOW`,
/// `LIGH`, `EMBR`, `TUNG`, `VRSG`, `BIZR` and `C5` are complete. The remaining
/// gap is `FIGH.cpp` / `Element_STKM_graphics`, which reads the fighter rig and
/// velocity.
///
/// The final arm is `Element::defaultGraphics`' colour behaviour
/// (`Element.cpp:230-249`): it never writes `colr/colg/colb`, so a particle
/// whose element has no colour rule keeps the table colour. It must stay last,
/// so no element can inherit a stale or zero colour from a neighbour's rule.
///
/// Elements in this batch with no `Graphics` override (`FRAY`, `RPEL`,
/// `CRAY`, `RFRG`, `RFGL`, `RSST`, `RSSS`, `BREC`, `MERC`, `HEAC`, `ARAY`,
/// `PSTS`, `EQVE`, `SPWN`, `SPWN2`, `LOLZ`, `SAWD`) keep their base colour.
#[cube]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(crate) fn element_graphics_rgb(
    base: (f32, f32, f32),
    ty: u32,
    life: i32,
    ctype: u32,
    temp: f32,
    pid: u32,
    dcolour: u32,
    colour_mode: u32,
    decoration_level: u32,
    tmp: i32,
    tmp2: i32,
    vx: f32,
    vy: f32,
    tick: u32,
    elem_props: &[u32],
    elem_count: u32,
) -> (f32, f32, f32) {
    if ty == PT_FIRE {
        rgb_from_u32(flame_colour(life))
    } else if ty == PT_PLSM {
        rgb_from_u32(plasma_colour(life))
    } else if ty == PT_LAVA {
        lava_graphics(life)
    } else if ty == PT_SMKE {
        smke_graphics()
    } else if ty == PT_PHOT {
        photon_graphics(ctype)
    } else if ty == PT_ACID {
        acid_graphics(base, life)
    } else if ty == PT_WOOD {
        wood_graphics(base, temp, tmp)
    } else if ty == PT_GPMP {
        gpmp_graphics(base, life)
    } else if ty == PT_PLNT {
        plnt_graphics(base, ctype, temp, tmp2)
    } else if ty == PT_SPNG {
        spng_graphics(base, life)
    } else if ty == PT_SWCH {
        swch_graphics(base, life)
    } else if ty == PT_HSWC {
        hswc_graphics(base, life)
    } else if ty == PT_PCLN {
        pcln_graphics(base, life)
    } else if ty == PT_PUMP {
        pump_graphics(base, life)
    } else if ty == PT_DLAY {
        dlay_graphics(base, life, temp)
    } else if ty == PT_GLOW {
        glow_graphics(ctype, temp, tmp, pid, tick)
    } else if ty == PT_STOR {
        stor_graphics(tmp)
    } else if ty == PT_PSTN {
        pstn_graphics(base, life)
    } else if ty == PT_PBCN {
        pbcn_graphics(base, life)
    } else if ty == PT_WIRE {
        wire_graphics(base, ctype)
    } else if ty == PT_SEED {
        seed_graphics(base, ctype)
    } else if ty == PT_BASE {
        base_graphics(base, life)
    } else if ty == PT_LITH {
        lith_graphics(base, life, ctype, pid, tick)
    } else if ty == PT_VINE {
        vine_graphics(base, temp, tmp2)
    } else if ty == PT_WIFI {
        wifi_graphics(temp)
    } else if ty == PT_DEUT {
        deut_graphics(base, life)
    } else if ty == PT_LIGH {
        ligh_graphics()
    } else if ty == PT_CFLM || ty == PT_HFLM {
        cflm_graphics(life)
    } else if ty == PT_BIZR || ty == PT_BIZRG || ty == PT_BIZRS {
        // TPT `Element_BIZR_graphics`: the deco colour is stored as a photon
        // wavelength mask in `ctype`.
        bizr_graphics(base, ctype)
    } else if ty == PT_FILT {
        filt_graphics(ctype, temp)
    } else if ty == PT_BRAY {
        bray_graphics(base, ctype, tmp)
    } else if ty == PT_C5 {
        c5_graphics(base, ctype)
    } else if ty == PT_TUNG {
        tung_graphics(base, temp)
    } else if ty == PT_EMBR {
        embr_graphics(base, ctype, tmp)
    } else if ty == PT_LCRY {
        lcry_graphics(tmp2, dcolour, decoration_level)
    } else if ty == PT_GOLD {
        gold_graphics(base, pid)
    } else if ty == PT_VIBR || ty == PT_BVBR {
        vibr_graphics(base, life, tmp, tmp2)
    } else if ty == PT_GRAV {
        grav_graphics(vx, vy, tick)
    } else if ty == PT_POLO {
        polo_graphics(base, tmp)
    } else if ty == PT_SLCN {
        slcn_graphics(tmp)
    } else if ty == PT_QRTZ || ty == PT_PQRT {
        qrtz_graphics(base, tmp2)
    } else if ty == PT_FRME {
        frme_graphics(base, tmp)
    } else if ty == PT_COAL || ty == PT_BCOL {
        coal_graphics(base, tmp2, temp)
    } else if ty == PT_LIFE {
        life_graphics(ctype, dcolour, tmp, tmp2, decoration_level)
    } else if ty == PT_EXOT {
        exot_graphics(temp, tmp, tmp2, life, pid, tick)
    } else if ty == PT_CLST {
        clst_graphics(base, tmp)
    } else if ty == PT_ROCK {
        rock_graphics(base, tmp2)
    } else if ty == PT_CBNW {
        cbnw_graphics(base, tmp2)
    } else if ty == PT_INVIS {
        invis_graphics(base, tmp2)
    } else if ty == PT_WARP {
        warp_graphics()
    } else if ty == PT_GEL {
        gel_graphics(tmp)
    } else if ty == PT_TRON {
        tron_graphics(tmp)
    } else if ty == PT_BRCK {
        brck_graphics(base, tmp)
    } else if ty == PT_CRMC {
        crmc_graphics(base, tmp2)
    } else if ty == PT_EMP {
        emp_graphics(base, life)
    } else if ty == PT_PVOD {
        pvod_graphics(base, life)
    } else if ty == PT_SPRK
        || ty == PT_NEUT
        || ty == PT_ELEC
        || ty == PT_PROT
        || ty == PT_PTNM
        || ty == PT_VIRS
        || ty == PT_VRSS
        || ty == PT_VRSG
        || ty == PT_THDR
        || ty == PT_ACEL
        || ty == PT_DCEL
        || ty == PT_GBMB
        || ty == PT_GRVT
        || ty == PT_BOMB
        || ty == PT_DEST
        || ty == PT_DMG
        || ty == PT_FIRW
        || ty == PT_PRTI
        || ty == PT_PRTO
    {
        // TPT custom graphics() functions for these elements modify pixel_mode,
        // fire channels, or orbital dots, leaving the base pixel colour unmodified:
        // - SPRK: SPRK.cpp:413-421
        // - NEUT: NEUT.cpp:251-260
        // - ELEC: ELEC.cpp:148-157
        // - PROT: PROT.cpp:232-241
        // - PTNM: PTNM.cpp:267-272
        // - VIRS: VIRS.cpp:151-156
        // - VRSS / VRSG: VRSS.cpp:54-58, VRSG.cpp:54-64
        // - THDR: THDR.cpp:95-103
        // - ACEL / DCEL: ACEL.cpp:103-108, DCEL.cpp:83-88
        // - GBMB: GBMB.cpp:83-93
        // - GRVT: GRVT.cpp:76-85
        // - BOMB: BOMB.cpp:131-135
        // - DEST: DEST.cpp:100-111
        // - DMG: DMG.cpp:119-123
        // - FIRW: FIRW.cpp:121-128
        // - PRTI / PRTO: PRTI.cpp:152-163, PRTO.cpp:181-192
        base
    } else if ty == PT_STKM || ty == PT_STKM2 || ty == PT_FIGH {
        (0.0f32, 0.0f32, 0.0f32)
    } else {
        base
    }
}
/// Element `Graphics` colour override dispatch including loaded `PIPE` cargo.
///
/// `PIPE.cpp:340-392`: a loaded pipe runs the cargo's `Graphics` rule with
/// fields mapped via `props_pipe_to_part` (`PIPE.cpp:421-448`).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn graphics_rgb(
    base: (f32, f32, f32),
    ty: u32,
    life: i32,
    ctype: u32,
    temp: f32,
    pid: u32,
    dcolour: u32,
    colour_mode: u32,
    decoration_level: u32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
    vx: f32,
    vy: f32,
    tick: u32,
    elem_props: &[u32],
    elem_count: u32,
) -> (f32, f32, f32) {
    if ty == PT_PIPE || ty == PT_PPIP {
        let cargo = pipe_cargo(ctype);
        if cargo != 0u32 {
            let slot = (cargo % elem_count) * ELEM_WORDS;
            let cargo_base = rgb_from_u32(elem_props[(slot + ELEM_COLOUR) as usize]);
            let cargo_props = elem_props[(slot + ELEM_PROPS) as usize];
            let mut cargo_vx = 0.0f32;
            let mut cargo_vy = 0.0f32;
            if (cargo_props & 0x0000_0008u32) != 0u32 {
                // TYPE_ENERGY
                cargo_vx = vx;
                cargo_vy = vy;
            }
            element_graphics_rgb(
                cargo_base,
                cargo,
                tmp2,
                tmp4 as u32,
                temp,
                pid,
                0u32,
                colour_mode,
                decoration_level,
                tmp3,
                0i32,
                cargo_vx,
                cargo_vy,
                tick,
                elem_props,
                elem_count,
            )
        } else {
            // TPT `PIPE.cpp:397-416`: empty pipe colour flags.
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
    } else {
        element_graphics_rgb(
            base,
            ty,
            life,
            ctype,
            temp,
            pid,
            dcolour,
            colour_mode,
            decoration_level,
            tmp,
            tmp2,
            vx,
            vy,
            tick,
            elem_props,
            elem_count,
        )
    }
}

/// Per-particle `cola`: the alpha TPT's `PMODE_BLEND` and `PMODE_ADD` stages
/// composite with (`GRAPHICS_FUNC_ARGS`' `cola`, `255` by default).
///
/// Starts from the element table's `GpuElementProps::cola` and applies the
/// rules of every element whose `Graphics` writes it. `WARP` is the only
/// built-in with an unconditional value, so it is folded in here rather than
/// stored in the table.
#[cube]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(crate) fn element_graphics_cola(
    table_cola: u32,
    ty: u32,
    life: i32,
    ctype: u32,
    tmp: i32,
    tmp2: i32,
    elem_props: &[u32],
    elem_count: u32,
) -> u32 {
    let mut cola = table_cola as i32;
    if ty == PT_FILT {
        // FILT.cpp:64-67: a freshly-created filter is more opaque.
        if life > 0i32 && life <= 4i32 {
            cola = 127i32 + life * 30i32;
        } else {
            cola = 127i32;
        }
    } else if ty == PT_BRAY {
        // BRAY.cpp:51-102: the beam's transparency follows the mode and life.
        let mut trans = 255i32;
        if tmp == 0i32 {
            trans = life * 7i32;
        } else if tmp == 1i32 {
            trans = life / 4i32;
        } else if tmp == 2i32 {
            trans = life * 100i32;
        }
        cola = trans;
    } else if ty == PT_INVIS {
        // INVIS.cpp:68-75: under pressure the wall is a violet blend.
        if tmp2 != 0i32 {
            cola = 100i32;
        }
    } else if ty == PT_WARP {
        // WARP.cpp:89-94: fully transparent.
        cola = 0i32;
    } else if ty == PT_EXOT {
        // EXOT.cpp:225,235: `cola = tmp / 6`. The flicker branch keeps the
        // default, but its mode (`FLAT | FLARE`) ignores `cola` either way.
        cola = tmp / 6i32;
    } else if ty == PT_GRAV {
        // GRAV.cpp:110-111: a live graviton shares its fire alpha.
        if life != 0i32 {
            cola = (life + 15i32) * 4i32;
        }
    } else if ty == PT_EMBR {
        // EMBR.cpp:121-146: the burning spark fades in with `life`, the flat
        // glow keeps the default, and the plain spark is dim while fresh.
        if tmp == 1i32 {
            cola = (life + 15i32) * 4i32;
        } else if tmp != 2i32 && life < 64i32 {
            cola = 4i32 * life;
        }
    } else if ty == PT_TRON {
        // TRON.cpp:115-120: fading tail when life < tmp2 and not head.
        if life < tmp2 && (tmp & TRON_HEAD) == 0i32 && tmp2 > 0i32 {
            cola = ((life as f32 / tmp2 as f32) * 255.0f32) as i32;
        }
    } else if ty == PT_STKM || ty == PT_STKM2 || ty == PT_FIGH {
        cola = 0i32;
    }
    clampi(cola, 0i32, 255i32) as u32
}
/// Element `Graphics` alpha dispatch including loaded `PIPE` cargo.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn graphics_cola(
    table_cola: u32,
    ty: u32,
    life: i32,
    ctype: u32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
    elem_props: &[u32],
    elem_count: u32,
) -> u32 {
    if ty == PT_PIPE || ty == PT_PPIP {
        let cargo = pipe_cargo(ctype);
        if cargo != 0u32 {
            let slot = (cargo % elem_count) * ELEM_WORDS;
            let cargo_table_cola = elem_props[(slot + ELEM_COLA) as usize];
            element_graphics_cola(
                cargo_table_cola,
                cargo,
                tmp2,
                tmp4 as u32,
                tmp3,
                0i32,
                elem_props,
                elem_count,
            )
        } else {
            table_cola
        }
    } else {
        element_graphics_cola(
            table_cola, ty, life, ctype, tmp, tmp2, elem_props, elem_count,
        )
    }
}

/// Effective `cola` of one particle after the `COLOUR_*` overrides.
///
/// TPT never calls `Graphics` under `COLOUR_BASC` and forces `cola = 255`
/// under `COLOUR_HEAT`/`COLOUR_LIFE` (`Renderer.cpp:382`, `398`, `421`).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn particle_cola(
    ty: u32,
    table_cola: u32,
    life: i32,
    ctype: u32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
    colour_mode: u32,
    elem_props: &[u32],
    elem_count: u32,
) -> u32 {
    if (colour_mode & (COLOUR_HEAT | COLOUR_LIFE | COLOUR_BASC)) != 0u32 {
        255u32
    } else {
        graphics_cola(
            table_cola, ty, life, ctype, tmp, tmp2, tmp3, tmp4, elem_props, elem_count,
        )
    }
}

/// `PMODE_GLOW` alphas of one particle (`Renderer.cpp:690-716`).
///
/// TPT adds the glow with `AddPixel`, and every alpha it uses is scaled by the
/// particle's `cola`: the centre pixel at `(192*cola)/255`, the four
/// orthogonal neighbours at `(96*cola)/255` plus the `(5*cola)/255` that the
/// `x = 1` step of the ray loop also lands on them, and the rest of the
/// `|dx| + |dy| <= 7` diamond at `(5*cola)/255`.
///
/// Returns `(centre, orthogonal, ray)`; a `cola` of `0` blanks the glow, which
/// is what `WARP` relies on.
#[cube]
pub(crate) fn glow_alpha(cola: u32) -> (f32, f32, f32) {
    let c = cola as f32;
    let centre = (192.0f32 * c) / 255.0f32;
    let ortho = ((96.0f32 * c) / 255.0f32) + ((5.0f32 * c) / 255.0f32);
    let ray = (5.0f32 * c) / 255.0f32;
    (centre, ortho, ray)
}

/// Base colour of a particle plus the per-particle powder jitter.
///
/// Port extension, **not** TPT fidelity: TPT's `render_parts` never varies a
/// particle's brightness, so this ±8% hash jitter only exists while
/// `SimulationConfig::pretty_powder` is switched on (off by default). It is
/// kept because it is a documented port option, not a port of `Graphics.cpp`.
#[cube]
pub(crate) fn base_rgb(colour: u32, props: u32, pid: u32, pretty_powder: u32) -> (f32, f32, f32) {
    let mut r = ((colour >> 16u32) & 0xFFu32) as f32;
    let mut g = ((colour >> 8u32) & 0xFFu32) as f32;
    let mut b = (colour & 0xFFu32) as f32;

    // Port extension (see the doc comment): per-particle brightness jitter for
    // powders, `0.92..1.08`. TPT has no equivalent in `Renderer::render_parts`.
    if pretty_powder != 0u32 && (props & TYPE_PART) != 0u32 {
        let jitter = 0.92f32 + hash01(pid * 2654435761u32) * 0.16f32;
        r *= jitter;
        g *= jitter;
        b *= jitter;
    }
    (r, g, b)
}

/// `PROP_HOT_GLOW` incandescence (TPT `render_parts`).
#[cube]
// The `3.1415927` literal is TPT's own `M_PI` approximation in the
// `PROP_HOT_GLOW` gradient (`Graphics.cpp`), kept bit-identical on purpose.
#[allow(clippy::approx_constant)]
pub(crate) fn hot_glow_rgb(
    base: (f32, f32, f32),
    props: u32,
    high_temp: f32,
    temp: f32,
) -> (f32, f32, f32) {
    let (mut r, mut g, mut b) = base;
    if (props & PROP_HOT_GLOW) != 0u32 && temp > (high_temp - 800.0f32) {
        let gradv = 3.1415927f32 / (2.0f32 * high_temp - (high_temp - 800.0f32));
        let mut caddr = temp - (high_temp - 800.0f32);
        if temp > high_temp {
            caddr = 800.0f32;
        }
        let ci = caddr as u32;
        let c = ci as f32;
        r += (gradv * c).sin() * 226.0f32;
        g -= (gradv * c * 4.55f32).sin() * 34.0f32;
        b -= (gradv * c * 2.22f32).sin() * 64.0f32;
    }
    (r, g, b)
}

// `MIN_TEMP` / `MAX_TEMP` are mirrored as literals below so no host constant
// enters a kernel body (`ElementDefs.h:5-6`).
const _: () = assert!(MIN_TEMP.to_bits() == 0.0f32.to_bits());
const _: () = assert!(MAX_TEMP.to_bits() == 9999.0f32.to_bits());

/// `COLOUR_HEAT` / `COLOUR_LIFE` / `COLOUR_BASC` / `COLOUR_DEST` display
/// overrides (`Renderer.cpp:375-426`).
///
/// `COLOUR_HEAT` indexes TPT's 1024-entry heat table with
/// `int((temp - hdispLimitMin) / (hdispLimitMax - hdispLimitMin) * 1024)` and
/// leaves `cola` at 255 (`Renderer.cpp:378-382`). `hdisp_min`/`hdisp_max` are
/// the resolved `stats.hdispLimitMin/Max` from `Renderer::AdjustHdispLimit`
/// (`Renderer.cpp:1440-1512`), so a custom or auto heat display range maps the
/// particles exactly like the air overlay does.
///
/// `COLOUR_LIFE` ignores the element colour entirely: `q` is `sqrt(life)`
/// truncated for `life >= 5` and `life` itself below that, and every channel
/// becomes `int(sin(0.4 * q) * 100 + 128)` (`Renderer.cpp:392-402`).
///
/// `COLOUR_BASC` restores the raw element table colour (`colour`), which erases
/// the `Graphics` result *and* the `PROP_HOT_GLOW` incandescence applied above
/// it (`Renderer.cpp:406-412`). `COLOUR_DEST` paints the destructibility score
/// as grey: `score * 1.3` below 100, white otherwise
/// (`Renderer.cpp:413-426`, `SimulationData::DestructibilityScore`).
#[cube]
pub(crate) fn colour_override_rgb(
    base: (f32, f32, f32),
    table: (f32, f32, f32),
    life: i32,
    temp: f32,
    colour_mode: u32,
    hdisp_min: f32,
    hdisp_max: f32,
    dest_score: f32,
) -> (f32, f32, f32) {
    let (mut r, mut g, mut b) = base;
    if (colour_mode & COLOUR_HEAT) != 0u32 {
        let col = heat_temp_colour(temp, hdisp_min, hdisp_max);
        r = fb_red(col);
        g = fb_green(col);
        b = fb_blue(col);
    } else if (colour_mode & COLOUR_LIFE) != 0u32 {
        let q = if life >= 5i32 {
            (life as f32).sqrt() as i32
        } else {
            life
        };
        let value = ((0.4f32 * q as f32).sin() * 100.0f32 + 128.0f32) as i32;
        let v = value as f32;
        r = v;
        g = v;
        b = v;
    } else if (colour_mode & COLOUR_BASC) != 0u32 {
        r = table.0;
        g = table.1;
        b = table.2;
    } else if (colour_mode & COLOUR_DEST) != 0u32 {
        let v = if dest_score < 100.0f32 {
            (dest_score * 1.3f32) as i32 as f32
        } else {
            255.0f32
        };
        r = v;
        g = v;
        b = v;
    }
    (r, g, b)
}

/// `dcolour` decoration and the `COLOUR_GRAD` shimmer.
///
/// TPT only applies the decoration when the colour mode has no bit other than
/// `COLOUR_GRAD` and the decoration level is not `decorationDisabled`
/// (`Renderer.cpp:428-431`). Under `decorationAntiClickbait` the decoration
/// collapses first (`Renderer.cpp:300-314`): a `dcolour` that is not
/// 250+ alpha near-black becomes fully transparent (no decoration), a
/// near-black one becomes opaque black.
#[cube]
pub(crate) fn decorate_rgb(
    base: (f32, f32, f32),
    dcolour: u32,
    colour_mode: u32,
    decoration_level: u32,
    mode: u32,
    temp: f32,
) -> (f32, f32, f32) {
    let (mut r, mut g, mut b) = base;
    // `Renderer.cpp:300-314`: the anti-clickbait collapse of `dcolour`.
    let (deca, decr, decg, decb) = collapse_deco(dcolour, decoration_level);
    // `dcolour` decoration: `deca + 1` of 256, skipped by `NO_DECO` and by any
    // colour mode other than `COLOUR_GRAD`.
    if deca != 0u32
        && decoration_level != DECORATION_DISABLED
        && (colour_mode & NOT_GRAD_ONLY) == 0u32
        && (mode & NO_DECO) == 0u32
    {
        let da = (deca + 1u32) as f32;
        let dr = decr as f32;
        let dg = decg as f32;
        let db = decb as f32;
        r = (da * dr + (256.0f32 - da) * r) / 256.0f32;
        g = (da * dg + (256.0f32 - da) * g) / 256.0f32;
        b = (da * db + (256.0f32 - da) * b) / 256.0f32;
    }
    // `COLOUR_GRAD` shimmer (TPT: `sin(0.05 * int(temp - 40)) * 16` on every channel).
    if (colour_mode & COLOUR_GRAD) != 0u32 {
        let q = ((temp - 40.0f32) as i32) as f32;
        let s = (0.05f32 * q).sin() * 16.0f32;
        r += s;
        g += s;
        b += s;
    }
    (r, g, b)
}

/// TPT `SimulationData::DestructibilityScore` (`SimulationData.cpp:352-383`).
///
/// `0` is "nothing", `100` an indestructible element. `CLNE`/`PCLN` are
/// hard-coded to 100 despite a non-zero hardness; the property classes then
/// decide in `TYPE_ENERGY`/`TYPE_GAS`/`TYPE_LIQUID`/`TYPE_PART` order, followed
/// by flammability, pressure transitions, temperature transitions, hardness and
/// the three element ids that are destructible despite no rule matching.
#[cube]
pub(crate) fn destructibility_score(ty: u32, elem_props: &[u32], elem_count: u32) -> f32 {
    // CubeCL has no `return`, so the two early exits of the reference become a
    // nested branch around the property-class ladder.
    let mut score = 0.0f32;
    if ty != 0u32 {
        if ty == PT_CLNE || ty == PT_PCLN {
            score = 100.0f32;
        } else {
            let base = (ty % elem_count) * ELEM_WORDS;
            let props = elem_props[(base + ELEM_PROPS) as usize];
            if (props & TYPE_ENERGY) != 0u32 {
                score = 10.0f32;
            } else if (props & TYPE_GAS) != 0u32 {
                score = 20.0f32;
            } else if (props & TYPE_LIQUID) != 0u32 {
                score = 30.0f32;
            } else if (props & TYPE_PART) != 0u32 {
                score = 40.0f32;
            } else if elem_props[(base + ELEM_FLAMMABLE) as usize] != 0u32 {
                score = 50.0f32;
            } else if elem_props[(base + ELEM_HIGH_PRESSURE_TRANSITION) as usize] != (NT as u32)
                || elem_props[(base + ELEM_LOW_PRESSURE_TRANSITION) as usize] != (NT as u32)
            {
                score = 60.0f32;
            } else if elem_props[(base + ELEM_HIGH_TEMP_TRANSITION) as usize] != (NT as u32)
                || elem_props[(base + ELEM_LOW_TEMP_TRANSITION) as usize] != (NT as u32)
            {
                score = 70.0f32;
            } else if elem_props[(base + ELEM_HARDNESS) as usize] != 0u32 {
                score = 80.0f32;
            } else if ty == PT_VIBR || ty == PT_RSSS || ty == PT_HEAC {
                score = 90.0f32;
            } else {
                score = 100.0f32;
            }
        }
    }
    score
}

/// Final `0xAABBGGRR` colour of one particle (TPT `render_parts` + element `Graphics`).
///
/// Order matches TPT exactly: base colour, `Graphics` overrides, `PROP_HOT_GLOW`
/// incandescence, `COLOUR_*` override, `dcolour` decoration, `COLOUR_GRAD`
/// shimmer. Only the final pack quantises the channels.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn particle_rgb(
    ty: u32,
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
    let table = rgb_from_u32(colour);
    let mut c = base_rgb(colour, props, pid, pretty_powder);
    // Element `Graphics` colour overrides. `COLOUR_BASC` bypasses them.
    if (colour_mode & COLOUR_BASC) == 0u32 {
        c = graphics_rgb(
            c,
            ty,
            life,
            ctype,
            temp,
            pid,
            dcolour,
            colour_mode,
            decoration_level,
            tmp,
            tmp2,
            tmp3,
            tmp4,
            vx,
            vy,
            tick,
            elem_props,
            elem_count,
        );
    }
    c = hot_glow_rgb(c, props, high_temp, temp);
    // The score is only read by `COLOUR_DEST`; skip the table reads otherwise.
    let mut dest_score = 0.0f32;
    if (colour_mode & COLOUR_DEST) != 0u32 {
        dest_score = destructibility_score(ty, elem_props, elem_count);
    }
    c = colour_override_rgb(
        c,
        table,
        life,
        temp,
        colour_mode,
        hdisp_min,
        hdisp_max,
        dest_score,
    );
    c = decorate_rgb(c, dcolour, colour_mode, decoration_level, mode, temp);
    let (r, g, b) = c;
    pack_rgb(r, g, b)
}
