//! Particle render flags and dynamic pixel modes.

// Clippy's rewrite for these lints uses methods the device IR does not have
// (`is_multiple_of`, `clamp`, `Range::contains`, `saturating_sub`), and
// collapsing a nested `if` changes what the CubeCL macro expands, so the plain
// operators are deliberate.
#![allow(clippy::collapsible_if)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::{
    CLEAR_BLEND_ADD, CLEAR_FIRE_GLOW, DECO_FIRE, FIRE_ADD, FIRE_BLEND, FIRE_SPARK, FIREMODE,
    FLAG_PHOTDECO, NO_DECO, PMODE_ADD, PMODE_BLEND, PMODE_BLOB, PMODE_BLUR, PMODE_FLARE,
    PMODE_FLAT, PMODE_GLOW, PMODE_LFLARE, PMODE_OPTIONS, PMODE_SPARK,
};
use super::elements_basic::glow_bright;
use super::elements_special::{exot_flicker, pipe_cargo};
use super::{EFFECT_LINES, TRON_DEATH, TRON_HEAD};

const _: () = assert!(0xFFFF_EFFFu32 == !NO_DECO);
const _: () = assert!(0x0000_0008u32 == FLAG_PHOTDECO);

/// `RENDER_*` flags expanded into the `PMODE_*`/`FIRE_*` mask TPT calls
/// `renderMode`.
///
/// TPT stores whole flag *sets* in `renderMode` (`RENDER_FIRE` contains
/// `PMODE_ADD | PMODE_BLEND | FIRE_ADD | FIRE_BLEND`); this port stores one bit
/// per switch, so the set is rebuilt here.
#[cube]
pub(crate) fn render_mask(render_mode: u32) -> u32 {
    let mut mask = 0u32;
    if (render_mode & RENDER_BASC) != 0u32 {
        // `RENDER_BASC` is `OPTIONS | PSPEC_STICKMAN | PMODE_FLAT | PMODE_ADD |
        // PMODE_BLEND | EFFECT_LINES` (`ElementGraphics.h:38`).
        mask |= PMODE_FLAT | PMODE_ADD | PMODE_BLEND | EFFECT_LINES;
    }
    if (render_mode & RENDER_FIRE) != 0u32 {
        // `RENDER_FIRE` is `OPTIONS | PSPEC_STICKMAN | PMODE_ADD | PMODE_BLEND |
        // FIRE_ADD | FIRE_BLEND` (`ElementGraphics.h:33`); `OPTIONS` is kept
        // separately below and `PSPEC_STICKMAN` has no port bit.
        mask |= PMODE_ADD | PMODE_BLEND | FIRE_ADD | FIRE_BLEND;
    }
    if (render_mode & RENDER_SPRK) != 0u32 {
        // `RENDER_SPRK` is `OPTIONS | PSPEC_STICKMAN | PMODE_ADD | PMODE_BLEND |
        // FIRE_SPARK` (`ElementGraphics.h:34`).
        mask |= PMODE_ADD | PMODE_BLEND | FIRE_SPARK;
    }
    if (render_mode & RENDER_GLOW) != 0u32 {
        mask |= PMODE_GLOW | PMODE_ADD | PMODE_BLEND;
    }
    if (render_mode & RENDER_BLUR) != 0u32 {
        mask |= PMODE_BLUR | PMODE_ADD | PMODE_BLEND;
    }
    if (render_mode & RENDER_BLOB) != 0u32 {
        mask |= PMODE_BLOB | PMODE_ADD | PMODE_BLEND;
    }
    if (render_mode & RENDER_EFFE) != 0u32 {
        // `RENDER_EFFE` is `OPTIONS | PSPEC_STICKMAN | EFFECT | PMODE_SPARK |
        // PMODE_FLARE | PMODE_LFLARE` (`ElementGraphics.h:32`); `EFFECT` is
        // `0xFF000000`, so `EFFECT_LINES` survives this mode too.
        mask |= PMODE_SPARK | PMODE_FLARE | PMODE_LFLARE | EFFECT_LINES;
    }
    if (render_mode & RENDER_NONE) != 0u32 {
        mask |= PMODE_FLAT;
    }
    mask
}

/// Element `Graphics` `pixel_mode` overrides that read the particle fields.
///
/// TPT evaluates these inside the element's own `Graphics` function, which
/// receives the particle; the static half of the mode is the element table row
/// (`GpuElementProps::pixel_mode`). The fallback chain for disabled render modes
/// and the `COLOUR_*` rewrites stay in [`effective_pixel_mode`], because TPT
/// applies them in `render_parts` after the `Graphics` call.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn element_pixel_mode(
    pixel_mode: u32,
    ty: u32,
    life: i32,
    ctype: u32,
    temp: f32,
    tmp: i32,
    tmp2: i32,
    vx: f32,
    vy: f32,
    pid: u32,
    tick: u32,
    flags: u32,
) -> u32 {
    // Custom (Lua) elements have no `Graphics` function; TPT's default is FLAT.
    let mut mode = pixel_mode;
    if mode == 0u32 {
        mode = PMODE_FLAT;
    }

    // TPT `GBMB::graphics` (`GBMB.cpp:83-93`): a spent gravity bomb flares, a live one sparks.
    if ty == PT_GBMB {
        if life <= 0i32 {
            mode |= PMODE_FLARE;
        } else {
            mode |= PMODE_SPARK;
        }
    }

    // TPT `DEST::graphics` (`DEST.cpp:100-111`): a live device flares, a spent one sparks.
    if ty == PT_DEST {
        if life != 0i32 {
            mode |= PMODE_LFLARE;
        } else {
            mode |= PMODE_SPARK;
        }
    }

    // TPT `DEUT::graphics`: superheated deuterium glows, everything else is
    // blurred.
    if ty == PT_DEUT {
        if life >= 240i32 {
            // Drop the table's base BLUR flag before adding the glow, exactly
            // like TPT's `*pixel_mode |= PMODE_GLOW | FIRE_ADD` on a blank mode.
            mode &= 0xFFFF_FFFBu32;
            mode |= PMODE_GLOW | FIRE_ADD;
        } else {
            mode |= PMODE_BLUR;
        }
    }

    // TPT `STOR::graphics` (`STOR.cpp:110-123`): a charged storage glows.
    if ty == PT_STOR && tmp != 0i32 {
        mode |= PMODE_GLOW;
    }

    // TPT `LITH::graphics` (`LITH.cpp:254-276`): exploding lithium and, in the
    // `else if` branch, a charged one (`ctype > 0`) both flare and glow.
    if ty == PT_LITH {
        if life >= 1000i32 || ctype > 0u32 {
            mode |= PMODE_FLARE | PMODE_GLOW;
        }
    }

    // TPT `LIGH::graphics` (`LIGH.cpp:304-308`): the bolt glows, feeds the
    // fire buffer and lets `dcolour` tint that fire. The table carries the
    // first two bits; `DECO_FIRE` is only ever set here in TPT.
    if ty == PT_LIGH {
        mode |= DECO_FIRE;
    }

    // TPT `GRAV::graphics` (`GRAV.cpp:104-113`): a graviton moving fast enough
    // to have been given a `life` glows, flares and feeds the fire buffer.
    if ty == PT_GRAV && life != 0i32 {
        mode = FIRE_ADD | PMODE_ADD | PMODE_GLOW | PMODE_FLARE;
    }

    // TPT `POLO::graphics` (`POLO.cpp:111-123`): a live polo ball glows.
    if ty == PT_POLO && tmp < 5i32 {
        mode |= PMODE_GLOW;
    }

    // TPT `SLCN::graphics` (`SLCN.cpp:125-132`): the sparkle phase decides
    // between a flare and a spark. TPT multiplies the two 16-bit halves of
    // `tmp` as `int`; the product is taken modulo 2^32 here so an extreme
    // `tmp` cannot trap on the CPU backend.
    if ty == PT_SLCN {
        let t = tmp as u32;
        let rnd = (t & 0xFFFFu32) * ((t >> 16u32) & 0xFFFFu32);
        let flare_phase = rnd % 887u32;
        let spark_phase = rnd % 593u32;
        if flare_phase == 0u32 {
            mode |= PMODE_FLARE | PMODE_GLOW;
        }
        if spark_phase == 0u32 {
            mode |= PMODE_SPARK;
        }
    }

    // TPT `ACEL::graphics` / `DCEL::graphics` (`ACEL.cpp:103-108`,
    // `DCEL.cpp:83-88`): an accelerator holding a particle glows.
    if (ty == PT_ACEL || ty == PT_DCEL) && tmp != 0i32 {
        mode |= PMODE_GLOW;
    }

    // TPT `VIBR::graphics` (`VIBR.cpp:219-254`): the armed state replaces the
    // mode with `PMODE_NONE | FIRE_BLEND`; the arming state keeps the default
    // `PMODE_FLAT` and adds `FIRE_BLEND`.
    if ty == PT_VIBR || ty == PT_BVBR {
        if tmp / 10i32 >= 100i32 || life != 0i32 {
            mode = FIRE_BLEND;
        } else {
            mode = PMODE_FLAT | FIRE_BLEND;
        }
    }

    // TPT `INVIS::graphics` (`INVIS.cpp:65-77`): under pressure the wall is a
    // blend of a violet tint.
    if ty == PT_INVIS && tmp2 != 0i32 {
        mode = PMODE_BLEND;
    }

    // TPT `GLOW::graphics` (`GLOW.cpp:105-115`): a bright glow adds fire and
    // drops the flat pixel, a dim one blurs.
    if ty == PT_GLOW {
        if glow_bright(ctype, temp, tmp, pid, tick) {
            // `GLOW.cpp:110-112` clears the flat pixel for the bright branch.
            mode = FIRE_ADD | PMODE_GLOW | PMODE_ADD;
        } else {
            // `GLOW.cpp:114` only adds the blur, so the flat pixel stays.
            mode = PMODE_FLAT | PMODE_BLUR;
        }
    }

    // TPT `EXOT::graphics` (`EXOT.cpp:197-251`): the flicker branch adds
    // `PMODE_FLARE` to the default flat pixel, the regular branch adds
    // `FIRE_ADD` and `PMODE_BLUR`. TPT starts from `PMODE_FLAT`
    // (`Renderer.cpp:286`) and ORs into it, so the port has to spell the flat
    // pixel out: the element table row is the non-flicker result and would
    // otherwise leak `FIRE_ADD`/`PMODE_BLUR` into the flicker branch.
    if ty == PT_EXOT {
        if life < 1001i32 && exot_flicker(tmp2, pid, tick) {
            mode = PMODE_FLAT | PMODE_FLARE;
        } else {
            mode = PMODE_FLAT | FIRE_ADD | PMODE_BLUR;
        }
    }

    // TPT `ROCK::graphics` (`ROCK.cpp:58-66`): hot rock feeds the fire buffer.
    if ty == PT_ROCK && temp >= 810.15f32 {
        mode |= FIRE_ADD;
    }

    // TPT `SOAP::graphics` (`SOAP.cpp:323-326`): the bubble outline. The bit
    // survives the render mask only under `RENDER_BASC` (`ElementGraphics.h:38`).
    if ty == PT_SOAP {
        mode |= EFFECT_LINES;
    }

    // TPT `EMBR::graphics` (`EMBR.cpp:120-146`): the mode is replaced, not
    // extended. `tmp == 1` is the burning spark, `tmp == 2` a flat glow, and
    // everything else a spark with an additive pixel.
    if ty == PT_EMBR {
        if tmp == 1i32 {
            mode = FIRE_ADD | PMODE_BLEND | PMODE_GLOW;
        } else if tmp == 2i32 {
            mode = PMODE_FLAT | FIRE_ADD;
        } else {
            mode = PMODE_SPARK | PMODE_ADD;
        }
    }

    // TPT `TUNG::graphics` (`TUNG.cpp:130-152`): only tungsten above
    // `HighTemperature - 1500` K feeds the fire buffer; the table row carries
    // `FIRE_ADD` unconditionally, so a cold one has to drop it.
    // `0xFFFE_FFFF` is `!FIRE_ADD`.
    if ty == PT_TUNG {
        if temp > (3695.0f32 - 1500.0f32) {
            mode |= FIRE_ADD;
        } else {
            mode &= 0xFFFE_FFFFu32;
        }
    }

    // TPT `BIZR::graphics` (`BIZR.cpp:91-125`): `brightness = |vx| + |vy|`;
    // the fire buffer and `FIRE_ADD` are only touched while `brightness > 0`,
    // and `PMODE_BLUR` is always added. TPT ORs into the default `PMODE_FLAT`,
    // so the flat pixel stays; the element table row only carries the `BLUR`
    // the rule adds.
    if ty == PT_BIZR || ty == PT_BIZRG || ty == PT_BIZRS {
        let ax = if vx < 0.0f32 { -vx } else { vx };
        let ay = if vy < 0.0f32 { -vy } else { vy };
        mode = PMODE_FLAT | PMODE_BLUR;
        if (ax + ay) > 0.0f32 {
            mode |= FIRE_ADD;
        }
    }

    // TPT `C5::graphics` (`C5.cpp:99-125`): with a stored wavelength, adds fire
    // and additive pixel mode with NO_DECO; without it, stays flat.
    if ty == PT_C5 {
        if ctype != 0u32 {
            mode &= 0xFFFF_FFFEu32;
            mode |= FIRE_ADD | PMODE_ADD | NO_DECO;
        } else {
            mode = PMODE_FLAT;
        }
    }
    // TPT `GEL::graphics` (`GEL.cpp:51`): gel is blurred.
    if ty == PT_GEL {
        mode |= PMODE_BLUR;
    }

    // TPT `PTNM::graphics` (`PTNM.cpp:267-272`): sparked platinum flares.
    if ty == PT_PTNM && tmp != 0i32 {
        mode |= PMODE_FLARE;
    }

    // TPT `TRON::graphics` (`TRON.cpp:106-118`): head glows, dying tron adds fire
    // and flares, fading tail blends.
    if ty == PT_TRON {
        if (tmp & TRON_HEAD) != 0i32 {
            mode |= PMODE_GLOW;
        }
        if (tmp & TRON_DEATH) != 0i32 {
            mode |= FIRE_ADD | PMODE_FLARE;
        }
        if life < tmp2 && (tmp & TRON_HEAD) == 0i32 {
            mode |= PMODE_BLEND;
            mode &= 0xFFFF_FFFEu32;
        }
    }

    // TPT `BRCK::graphics` (`BRCK.cpp:55`): glowing brick adds fire.
    if ty == PT_BRCK && tmp == 1i32 {
        mode |= FIRE_ADD;
    }

    // TPT `FIRW::graphics` (`FIRW.cpp:121-128`): lit firework glows.
    if ty == PT_FIRW && tmp > 0i32 {
        mode |= PMODE_GLOW;
    }

    // TPT `SWCH::graphics` (`SWCH.cpp:54`): switched-on switch glows.
    if ty == PT_SWCH && life >= 10i32 {
        mode |= PMODE_GLOW;
    }

    // TPT `BASE::graphics` (`BASE.cpp:77`): charged base sparks.
    if ty == PT_BASE && tmp == 1i32 {
        mode |= PMODE_SPARK;
    }

    // TPT `PRTI::graphics` / `PRTO::graphics` (`PRTI.cpp:152-163`, `PRTO.cpp:181-192`): portal particles add.
    if ty == PT_PRTI || ty == PT_PRTO {
        mode &= 0xFFFF_F000u32;
        mode |= PMODE_ADD;
    }

    // TPT `SPRK::graphics` (`SPRK.cpp:413-421`): spark emits spark fire.
    if ty == PT_SPRK {
        mode |= FIRE_SPARK;
    }

    // TPT `NEUT::graphics` (`NEUT.cpp:251-260`): neutron emits additive fire.
    if ty == PT_NEUT {
        mode |= FIRE_ADD;
    }

    // TPT `ELEC::graphics` (`ELEC.cpp:148-157`): electricity emits additive fire.
    if ty == PT_ELEC {
        mode |= FIRE_ADD;
    }

    // TPT `PROT::graphics` (`PROT.cpp:232-241`): proton emits blended fire.
    if ty == PT_PROT {
        mode |= FIRE_BLEND;
    }

    // TPT `VIRS::graphics` (`VIRS.cpp:151-156`): virus blurs and disables deco.
    if ty == PT_VIRS {
        mode |= PMODE_BLUR | NO_DECO;
    }

    // TPT `VRSS::graphics` (`VRSS.cpp:54-58`): solid virus disables deco.
    if ty == PT_VRSS {
        mode |= NO_DECO;
    }

    // TPT `VRSG::graphics` (`VRSG.cpp:54-64`): gas virus clears pixel mode, blends fire, and disables deco.
    if ty == PT_VRSG {
        mode &= 0xFFFF_F000u32;
        mode |= FIRE_BLEND | NO_DECO;
    }

    // TPT `THDR::graphics` (`THDR.cpp:95-103`): thunder emits additive fire.
    if ty == PT_THDR {
        mode |= FIRE_ADD;
    }

    // TPT `SMKE::graphics` (`SMKE.cpp:49-64`): smoke clears pixel mode and blends fire.
    if ty == PT_SMKE {
        mode = (mode & 0xFFFF_F000u32) | FIRE_BLEND;
    }

    // TPT `GRVT::graphics` (`GRVT.cpp:76-85`): graviton emits blended fire.
    if ty == PT_GRVT {
        mode |= FIRE_BLEND;
    }

    // TPT `BOMB::graphics` (`BOMB.cpp:131-135`): bomb flares.
    if ty == PT_BOMB {
        mode |= PMODE_FLARE;
    }

    // TPT `CFLM::graphics` (`CFLM.cpp:51-67`): cold flame clears pixel mode and adds fire.
    if ty == PT_CFLM || ty == PT_HFLM {
        mode = (mode & 0xFFFF_F000u32) | FIRE_ADD;
    }

    // TPT `DMG::graphics` (`DMG.cpp:119-123`): generator flares.
    if ty == PT_DMG {
        mode |= PMODE_FLARE;
    }

    // TPT `PHOT::graphics` (`PHOT.cpp:149-176`): photon drops flat, adds additive pixel mode, fire, and NO_DECO.
    // Clears NO_DECO when FLAG_PHOTDECO is set (ElementDefs.h:39).
    if ty == PT_PHOT {
        mode &= 0xFFFF_FFFEu32;
        mode |= FIRE_ADD | PMODE_ADD | NO_DECO;
        if (flags & 0x0000_0008u32) != 0u32 {
            mode &= 0xFFFF_EFFFu32;
        }
    }

    // TPT `FILT::graphics` (`FILT.cpp:51-74`): filter particles blend.
    if ty == PT_FILT {
        mode &= 0xFFFF_F000u32;
        mode |= PMODE_BLEND;
    }

    // TPT `Element_STKM_graphics` (`STKM.cpp:78`): stickman rigs draw in pass 3b; suppress base pixel.
    if ty == PT_STKM || ty == PT_STKM2 || ty == PT_FIGH {
        mode = 0u32;
    }

    mode
}

/// Effective `PMODE_*`/`FIRE_*` flags of one particle.
///
/// Mirrors TPT `render_parts`: the element's `Graphics` flags (with the `PIPE`
/// cargo substitution), the fallback chain for disabled render modes, the
/// render mask, then the `COLOUR_*` rewrites (heat/life/basic/destructibility/
/// gradient).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn effective_pixel_mode(
    pixel_mode: u32,
    ty: u32,
    life: i32,
    ctype: u32,
    temp: f32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
    flags: u32,
    pid: u32,
    tick: u32,
    render_mode: u32,
    colour_mode: u32,
    vx: f32,
    vy: f32,
) -> u32 {
    // `PIPE.cpp:340-355`: a loaded pipe takes the cargo's `Graphics` results,
    // so the dynamic rules below run for the cargo type. `props_pipe_to_part`
    // (`PIPE.cpp:421-448`) maps the pipe fields onto TPT's temporary particle as
    // `life = pipe->tmp2`, `tmp = pipe->tmp3`, `ctype = pipe->tmp4` and
    // `tmp2 = 0`, `flags = 0`.
    let mut gfx_ty = ty;
    let mut gfx_life = life;
    let mut gfx_ctype = ctype;
    let mut gfx_tmp = tmp;
    let mut gfx_tmp2 = tmp2;
    let mut gfx_flags = flags;
    if ty == PT_PIPE || ty == PT_PPIP {
        let cargo = pipe_cargo(ctype);
        if cargo != 0u32 {
            gfx_ty = cargo;
            gfx_life = tmp2;
            gfx_tmp = tmp3;
            gfx_ctype = tmp4 as u32;
            gfx_tmp2 = 0i32;
            gfx_flags = 0u32;
        }
    }
    let mut mode = element_pixel_mode(
        pixel_mode, gfx_ty, gfx_life, gfx_ctype, temp, gfx_tmp, gfx_tmp2, vx, vy, pid, tick,
        gfx_flags,
    );

    // Fallbacks: a disabled effect degrades to the next cheaper one.
    if (mode & FIRE_ADD) != 0u32 && (render_mode & RENDER_FIRE) == 0u32 {
        mode |= PMODE_GLOW;
    }
    if (mode & FIRE_BLEND) != 0u32 && (render_mode & RENDER_FIRE) == 0u32 {
        mode |= PMODE_BLUR;
    }
    if (mode & PMODE_BLUR) != 0u32 && (render_mode & RENDER_BLUR) == 0u32 {
        mode |= PMODE_FLAT;
    }
    if (mode & PMODE_GLOW) != 0u32 && (render_mode & RENDER_GLOW) == 0u32 {
        mode |= PMODE_BLEND;
    }
    if (render_mode & RENDER_BLOB) != 0u32 {
        mode |= PMODE_BLOB;
    }

    mode &= render_mask(render_mode) | PMODE_OPTIONS;

    // `COLOUR_*` overrides (TPT: heat/life replace glow with blur, basic and
    // destructibility force flat, gradient softens glow into blur).
    if (colour_mode & (COLOUR_HEAT | COLOUR_LIFE)) != 0u32 {
        if (mode & (FIREMODE | PMODE_GLOW)) != 0u32 {
            mode = (mode & CLEAR_FIRE_GLOW) | PMODE_BLUR;
        } else if (mode & (PMODE_BLEND | PMODE_ADD)) == (PMODE_BLEND | PMODE_ADD) {
            mode = (mode & CLEAR_BLEND_ADD) | PMODE_FLAT;
        } else if mode == 0u32 {
            mode |= PMODE_FLAT;
        }
    } else if (colour_mode & (COLOUR_BASC | COLOUR_DEST)) != 0u32 {
        // `Renderer.cpp:406-426`: both modes replace the mode with a flat
        // pixel. The destructibility grey itself is painted by
        // `colour_override_rgb`.
        mode = PMODE_FLAT;
    }

    if (colour_mode & COLOUR_GRAD) != 0u32 && (mode & (FIREMODE | PMODE_GLOW)) != 0u32 {
        mode = (mode & CLEAR_FIRE_GLOW) | PMODE_BLUR;
    }
    mode
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn host_element_pixel_mode(
    pixel_mode: u32,
    ty: u32,
    _life: i32,
    _ctype: u32,
    _temp: f32,
    _tmp: i32,
    _tmp2: i32,
    _pid: u32,
    _tick: u32,
    flags: u32,
) -> u32 {
    let mut mode = pixel_mode;
    if mode == 0 {
        mode = PMODE_FLAT;
    }
    if ty == PT_PHOT {
        mode &= 0xFFFF_FFFE;
        mode |= FIRE_ADD | PMODE_ADD | NO_DECO;
        if (flags & FLAG_PHOTDECO) != 0 {
            mode &= !NO_DECO;
        }
    }
    mode
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phot_mode_decision() {
        let base = PMODE_FLAT;
        let mode_without = host_element_pixel_mode(base, PT_PHOT, 0, 0, 295.15, 0, 0, 1, 0, 0);
        assert_eq!(mode_without & PMODE_FLAT, 0);
        assert_ne!(mode_without & (FIRE_ADD | PMODE_ADD), 0);
        assert_ne!(mode_without & NO_DECO, 0);

        let mode_with =
            host_element_pixel_mode(base, PT_PHOT, 0, 0, 295.15, 0, 0, 1, 0, FLAG_PHOTDECO);
        assert_eq!(mode_with & PMODE_FLAT, 0);
        assert_ne!(mode_with & (FIRE_ADD | PMODE_ADD), 0);
        assert_eq!(mode_with & NO_DECO, 0);
    }
}
