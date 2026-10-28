//! Powder elements (`MenuCategory::Powders`).

use super::super::{ElementDef, MenuCategory, PMODE_FLAT};
use crate::constants::*;

/// Appends the powder definitions in their original table order.
pub(crate) fn push(v: &mut Vec<ElementDef>) {
    v.push(elem!(
        id: PT_DUST, ident: "DEFAULT_PT_DUST", name: "DUST", cat: MenuCategory::Powders,
        photon_reflect: 0x3FFFFFC0,
        colour: 0xFFE0A0, props: TYPE_PART, weight: 85,
        desc: "Very light dust. Flammable.",
        advection: 0.7, air_drag: 0.02, air_loss: 0.96, loss: 0.8, gravity: 0.1, falldown: 1,
        flammable: 10, heat_conduct: 70,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_STNE, ident: "DEFAULT_PT_STNE", name: "STNE", cat: MenuCategory::Powders,
        colour: 0xA0A0A0, props: TYPE_PART, weight: 90,
        desc: "Stone. Heavy particles, meltable.",
        advection: 0.4, air_drag: 0.04, air_loss: 0.94, loss: 0.95, collision: -0.1, gravity: 0.3,
        falldown: 1,
        meltable: 5, hardness: 1, heat_conduct: 150,
        high_temp: 983.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_SAND, ident: "DEFAULT_PT_SAND", name: "SAND", cat: MenuCategory::Powders,
        colour: 0xFFD090, props: TYPE_PART, weight: 90,
        desc: "Sand, Heavy particles. Melts into glass.",
        advection: 0.4, air_drag: 0.04, air_loss: 0.94, loss: 0.95, collision: -0.1, gravity: 0.3,
        falldown: 1,
        meltable: 5, hardness: 1, heat_conduct: 150,
        high_temp: 1973.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_SALT, ident: "DEFAULT_PT_SALT", name: "SALT", cat: MenuCategory::Powders,
        colour: 0xFFFFFF, props: TYPE_PART, weight: 75,
        desc: "Salt, dissolves in water.",
        advection: 0.4, air_drag: 0.04, air_loss: 0.94, loss: 0.95, collision: -0.1, gravity: 0.3,
        falldown: 1,
        meltable: 5, hardness: 1, heat_conduct: 110,
        high_temp: 1173.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_SNOW, ident: "DEFAULT_PT_SNOW", name: "SNOW", cat: MenuCategory::Powders,
        photon_reflect: 0x03FFFFFF,
        colour: 0xC0E0FF, props: TYPE_PART | PROP_NEUTPASS, weight: 50,
        desc: "Light particles. Created when ICE breaks under pressure.",
        advection: 0.7, air_drag: 0.01, air_loss: 0.96, loss: 0.9, collision: -0.1, gravity: 0.05,
        diffusion: 0.01, hot_air: -0.00005, falldown: 1,
        hardness: 20, heat_conduct: 46,
        default_temp: 265.15,
        high_temp: 252.05, high_temp_transition: ST,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_CNCT, ident: "DEFAULT_PT_CNCT", name: "CNCT", cat: MenuCategory::Powders,
        colour: 0xC0C0C0, props: TYPE_PART | PROP_HOT_GLOW, weight: 55,
        desc: "Concrete. Can stack on itself or ROCK, collapses with pressure.",
        advection: 0.4, air_drag: 0.04, air_loss: 0.94, loss: 0.95, collision: -0.1, gravity: 0.3,
        falldown: 1,
        meltable: 2, hardness: 2, heat_conduct: 100,
        high_temp: 1123.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_BCOL, ident: "DEFAULT_PT_BCOL", name: "BCOL", cat: MenuCategory::Powders,
        photon_reflect: 0x00000000,
        colour: 0x333333, props: TYPE_PART, weight: 90,
        desc: "Broken Coal. Heavy particles, burns slowly.",
        advection: 0.4, air_drag: 0.04, air_loss: 0.94, loss: 0.95, collision: -0.1, gravity: 0.3,
        falldown: 1,
        hardness: 2, heat_conduct: 150,
        default_life: 110,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
}
