//! Particle colouring: TPT `render_parts` plus every element `Graphics` override.
//!
//! [`particle_rgb`] runs the same stages, in the same order, as TPT:
//! base colour, `Graphics` overrides, `PROP_HOT_GLOW` incandescence, the
//! `COLOUR_*` display override, `dcolour` decoration and the `COLOUR_GRAD`
//! shimmer. Each stage is a `#[cube]` helper that takes and returns an
//! unquantised `(r, g, b)` triple, so splitting the pipeline does not change a
//! single rounding step.
//!
//! The `Graphics` stage reads the same particle fields TPT's `GRAPHICS_FUNC_ARGS`
//! hands an element: `type`, `life`, `ctype`, `temp`, `dcolour`, `tmp`, `tmp2`,
//! `vx`, `vy` and `sim->currentTick` (passed as `tick`). Every render pass that
//! colours a particle binds all of them, so an element's `Graphics` rule gives
//! the same answer in the base, glow, ray and fire passes.
//!
//! The three outputs of a `Graphics` function that are not the pixel colour
//! live beside it: [`element_pixel_mode`] carries the dynamic `pixel_mode` bits,
//! [`graphics_cola`] carries `cola` and `fire_rgba` the
//! `firea/firer/fireg/fireb` quadruple.
//!
//! # Deliberate deviations
//!
//! * `Renderer.cpp:829-848` `EFFECT_DBGLINES` (the PRTI/PRTO channel lines) is
//!   drawn by [`render_stickmen_and_orbitals_kernel`] once the pointer rests on
//!   a portal, the debug HUD is on and `DISPLAY_PERS` is off; the kernel scans
//!   the particle pool for the matching channel and XORs one line per partner.
//!   TPT's `debugLines` toggle is the port's `Renderer.DebugMode` debug HUD
//!   flag (`config.debug_hud`), which the same pass receives.
//! * `Renderer.cpp:793-828` `EFFECT_GRAVIN`/`EFFECT_GRAVOUT` (the portal's four
//!   orbital dots) and `Renderer.cpp:540-660` (`FIGH` / `STKM` / `STKM2` stickman
//!   rigs) are rendered by [`render_stickmen_and_orbitals_kernel`].
//! * `Element::defaultGraphics` (`Element.cpp:230-249`) never writes a colour,
//!   so a particle without a colour rule keeps the element table colour; its
//!   property-based *mode* rules are already part of each table row. The final
//!   arm of [`graphics_rgb`] is that fallback.

use cubecl::prelude::*;
use tpt_core::constants::*;

pub(crate) mod colour;
pub(crate) mod elements_basic;
pub(crate) mod elements_special;
pub(crate) mod kernels;
pub(crate) mod pixel_mode;

pub(crate) use colour::*;
pub(crate) use elements_basic::*;
pub(crate) use elements_special::*;
pub use kernels::*;
pub(crate) use pixel_mode::*;

/// `PT_NUM` as the `u32` the kernels compare a stored cargo type against.
pub(crate) const PT_NUM_U32: u32 = PT_NUM as u32;

/// TPT `EFFECT_LINES` (`ElementGraphics.h:29`): the particle draws a line to a
/// partner instead of only a pixel. `SOAP.cpp:325` is the only built-in that
/// sets it.
pub(crate) const EFFECT_LINES: u32 = 0x0400_0000;

/// TPT `TRON_HEAD` (`TRON.cpp:18`): tron head particle flag.
pub(crate) const TRON_HEAD: i32 = 0x0000_0001;

/// TPT `TRON_DEATH` (`TRON.cpp:22`): crashed / dying tron flag.
pub(crate) const TRON_DEATH: i32 = 0x0000_0010;

/// Resolve the packed `pmap`/`photons` word at `(x, y)`, or `0` when empty.
#[cube]
pub(crate) fn packed_at(
    pmap: &[u32],
    photons: &[u32],
    xres: u32,
    yres: u32,
    x: i32,
    y: i32,
) -> u32 {
    let mut out = 0u32;
    if x >= 0i32 && y >= 0i32 && (x as u32) < xres && (y as u32) < yres {
        let idx = ((y as u32) * xres + (x as u32)) as usize;
        out = photons[idx];
        if out == 0u32 {
            out = pmap[idx];
        }
    }
    out
}
