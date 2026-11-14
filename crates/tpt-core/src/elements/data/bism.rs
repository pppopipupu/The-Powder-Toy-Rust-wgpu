//! The port's own additions above the reference / mod tables.
//!
//! These rows have no C++ counterpart: the behaviour is requested by the port
//! and grounded in the real element, with the code shape taken from the
//! reference `Update` handlers cited in `tpt-sim/src/reaction_kernel/`.
//!
//! `BISM` (bismuth) is a solid like `PTNM`: it does not fall, it conducts
//! electricity, and its `ctype` is the thickness of the oxide film on its
//! surface (`0` = bare metal, `100` = fully oxidised `Bi2O3`).
//!
//! * `ctype` — oxide film, `0..=100`; `>= 80` passivates the surface.
//! * `tmp`   — crystal grain `0..=7` (seeded per [`BISM_GRAIN_SHIFT`] cell and
//!   inherited by a growing crystal), used by the iridescent render phase.
//! * `tmp2`  — thermocouple charge `0..=100` (Seebeck / Peltier pair).
//! * `life`  — `> 1000` burning timer, otherwise the spark cooldown, exactly
//!   like `LITH.cpp:44-47`'s dual-purpose `life`.

use super::super::{ElementDef, MenuCategory, PMODE_FLAT};
use crate::constants::*;

/// Appends the port additions, in id order.
pub(crate) fn push(v: &mut Vec<ElementDef>) {
    // --- BISM (port id 239) --------------------------------------------------------
    //
    // Real constants: density 9.78 g/cm^3, melting point 271.5 C (544.7 K),
    // thermal conductivity 8 W/(m*K) (one of the worst metals), and a
    // passivating `Bi2O3` film that shows thin-film interference colours.
    v.push(elem!(
        id: PT_BISM, ident: "DEFAULT_PT_BISM", name: "BISM", cat: MenuCategory::Solids,
        colour: 0xD9DEE6, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC,
        weight: 100,
        desc: "Bismuth. Tarnishes into an iridescent oxide film; melts at 271 C and recrystallises as it cools.",
        // A solid block: no fall speed, no advection, only air loss.
        air_loss: 0.94,
        loss: 0.0,
        meltable: 1,
        // Brittle, with no structural hardness to speak of.
        hardness: 1,
        // Poor metal conductor (8 W/mK); low enough that a thick bismuth
        // block insulates its own melt, which is what makes the crystals.
        heat_conduct: 25,
        pixel_mode: PMODE_FLAT,
        high_temp: 544.7,
        high_temp_transition: PT_LAVA as i32,
    ));
}
