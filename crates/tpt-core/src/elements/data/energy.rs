//! Electronics and radioactive (energy-class) elements.
//!
//! Electronics stay grouped with the radioactive elements because both are the
//! electrically / energetically active part of the table; the two sections are
//! emitted separately to keep the original table order (see
//! [`super::super::builtin_elements`]).

use super::super::{
    ElementDef, FIRE_ADD, FIRE_BLEND, FIRE_SPARK, MenuCategory, NO_DECO, PMODE_ADD, PMODE_FLAT,
};
use crate::constants::*;

/// Appends the electronics definitions in their original table order.
pub(crate) fn push_electronics(v: &mut Vec<ElementDef>) {
    v.push(elem!(
        id: PT_METL, ident: "DEFAULT_PT_METL", name: "METL", cat: MenuCategory::Electronics,
        colour: 0x404060,
        props: TYPE_SOLID | PROP_CONDUCTS | PROP_HOT_GLOW | PROP_LIFE_DEC, weight: 100,
        desc: "Metal, the basic conductor. Meltable.",
        air_loss: 0.9, loss: 0.0,
        meltable: 1, hardness: 1, heat_conduct: 251,
        high_temp: 1273.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_SPRK, ident: "DEFAULT_PT_SPRK", name: "SPRK", cat: MenuCategory::Electronics,
        photon_reflect: 0x00000000,
        colour: 0xFFFF80, props: TYPE_SOLID | PROP_LIFE_DEC, weight: 100,
        desc: "Electricity. The basis of all electronics in TPT, travels along conductive elements.",
        air_loss: 0.9, loss: 0.0, hot_air: 0.001,
        hardness: 1, heat_conduct: 251,
        pixel_mode: PMODE_FLAT | FIRE_SPARK, fire_colour: 0x3C7F_7F40,
    ));
}

/// Appends the radioactive definitions in their original table order.
pub(crate) fn push_radioactive(v: &mut Vec<ElementDef>) {
    v.push(elem!(
        id: PT_PHOT, ident: "DEFAULT_PT_PHOT", name: "PHOT", cat: MenuCategory::Radioactive,
        colour: 0xFFFFFF,
        props: TYPE_ENERGY | PROP_PHOTPASS | PROP_LIFE_DEC | PROP_LIFE_KILL_DEC, weight: -1,
        desc: "Photons. Refract through glass, and different elements change its color. Can ignite flammable materials.",
        collision: -0.99,
        hardness: 0, heat_conduct: 251,
        default_temp: 1195.15, default_life: 680, default_ctype: 0x3FFF_FFFF,
        pixel_mode: FIRE_ADD | PMODE_ADD | NO_DECO, fire_colour: 0x6400_0000,
    ));
    v.push(elem!(
        id: PT_NEUT, ident: "DEFAULT_PT_NEUT", name: "NEUT", cat: MenuCategory::Radioactive,
        colour: 0x20E0FF, props: TYPE_ENERGY | PROP_LIFE_DEC | PROP_LIFE_KILL_DEC, weight: -1,
        desc: "Neutrons. Interact with matter in odd ways.",
        collision: -0.99, diffusion: 0.01, hot_air: 0.002,
        hardness: 0, heat_conduct: 60,
        default_temp: 299.15,
        pixel_mode: PMODE_FLAT | FIRE_ADD, fire_colour: 0x780A_5078,
    ));
    v.push(elem!(
        id: PT_ELEC, ident: "DEFAULT_PT_ELEC", name: "ELEC", cat: MenuCategory::Radioactive,
        colour: 0xDFEFFF, props: TYPE_ENERGY | PROP_LIFE_DEC | PROP_LIFE_KILL_DEC, weight: -1,
        desc: "Electrons. Sparks electronics, reacts with NEUT and WATR.",
        collision: -0.99,
        hardness: 0, heat_conduct: 251,
        default_temp: 495.15,
        pixel_mode: PMODE_FLAT | FIRE_ADD, fire_colour: 0x46DF_EFFF,
    ));
    // Newtonian gravity sources.
    v.push(elem!(
        id: PT_GRVT, ident: "DEFAULT_PT_GRVT", name: "GRVT", cat: MenuCategory::Radioactive,
        colour: 0x00EE76, props: TYPE_ENERGY | PROP_LIFE_DEC | PROP_LIFE_KILL_DEC, weight: -1,
        desc: "Gravitons. Create Newtonian Gravity.",
        air_loss: 1.0, loss: 1.0, collision: -0.99,
        hardness: 1, heat_conduct: 61,
        default_tmp: 7,
        pixel_mode: PMODE_FLAT | FIRE_BLEND, fire_colour: 0x0500_FAAA,
    ));
}
