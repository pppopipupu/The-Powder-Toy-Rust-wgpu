//! Gas elements (`MenuCategory::Gases`).

use super::super::{
    ElementDef, FIRE_ADD, FIRE_BLEND, MenuCategory, PMODE_ADD, PMODE_FLAT, PMODE_GLOW,
};
use crate::constants::*;

/// Appends the gas definitions in their original table order.
pub(crate) fn push(v: &mut Vec<ElementDef>) {
    v.push(elem!(
        id: PT_GAS, ident: "DEFAULT_PT_GAS", name: "GAS", cat: MenuCategory::Gases,
        colour: 0xE0FF20, props: TYPE_GAS | PROP_NEUTPASS, weight: 1,
        desc: "Diffuses quickly and is flammable. Liquefies into OIL under pressure.",
        advection: 1.0, air_drag: 0.01, air_loss: 0.99, loss: 0.3, collision: -0.1,
        diffusion: 0.75, hot_air: 0.001,
        flammable: 600, hardness: 1, heat_conduct: 42,
        default_temp: 297.15,
        high_pressure: 6.0, high_pressure_transition: PT_OIL as i32,
        high_temp: 573.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_SMKE, ident: "DEFAULT_PT_SMKE", name: "SMKE", cat: MenuCategory::Gases,
        colour: 0x222222, props: TYPE_GAS | PROP_LIFE_DEC | PROP_LIFE_KILL_DEC, weight: 1,
        desc: "Smoke, created by fire.",
        advection: 0.9, air_drag: 0.04, air_loss: 0.97, loss: 0.2, gravity: -0.1,
        hot_air: 0.001, falldown: 1,
        hardness: 1, heat_conduct: 88,
        default_temp: 615.15,
        high_temp: 625.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: FIRE_BLEND, fire_colour: 0x4B37_3737,
    ));
    v.push(elem!(
        id: PT_WTRV, ident: "DEFAULT_PT_WTRV", name: "WTRV", cat: MenuCategory::Gases,
        colour: 0xA0A0FF, props: TYPE_GAS, weight: 1,
        desc: "Steam. Produced from hot water.",
        advection: 1.0, air_drag: 0.01, air_loss: 0.99, loss: 0.3, collision: -0.1,
        gravity: -0.1, diffusion: 0.75, hot_air: 0.0003,
        hardness: 4, heat_conduct: 48,
        default_temp: 395.15,
        low_temp: 371.0, low_temp_transition: ST,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_PLSM, ident: "DEFAULT_PT_PLSM", name: "PLSM", cat: MenuCategory::Gases,
        colour: 0xBB99FF, props: TYPE_GAS | PROP_LIFE_DEC | PROP_LIFE_KILL, weight: 1,
        desc: "Plasma, extremely hot.",
        advection: 0.9, air_drag: 0.04, air_loss: 0.97, loss: 0.2, gravity: -0.1,
        diffusion: 0.3, hot_air: 0.001,
        hardness: 0, heat_conduct: 5,
        default_temp: 9999.0,
        pixel_mode: PMODE_GLOW | PMODE_ADD | FIRE_ADD, fire_colour: 0xFF00_0000,
    ));
    v.push(elem!(
        id: PT_H2, ident: "DEFAULT_PT_H2", name: "H2", cat: MenuCategory::Gases,
        colour: 0x5070FF, props: TYPE_GAS | PROP_PHOTPASS, weight: 1,
        desc: "Hydrogen. Combusts with OXYG to make WATR. Undergoes fusion at high temperature and pressure.",
        advection: 2.0, air_loss: 0.99, loss: 0.3, collision: -0.1, diffusion: 3.0,
        hardness: 0, heat_conduct: 251,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_O2, ident: "DEFAULT_PT_O2", name: "O2", cat: MenuCategory::Gases,
        colour: 0x80A0FF, props: TYPE_GAS, weight: 1,
        desc: "Oxygen gas. Ignites easily.",
        advection: 2.0, air_loss: 0.99, loss: 0.3, collision: -0.1, diffusion: 3.0,
        hardness: 0, heat_conduct: 70,
        low_temp: 90.0, low_temp_transition: PT_LO2 as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_CAUS, ident: "DEFAULT_PT_CAUS", name: "CAUS", cat: MenuCategory::Gases,
        colour: 0x80FFA0, props: TYPE_GAS | PROP_DEADLY, weight: 1,
        desc: "Caustic Gas, acts like ACID.",
        advection: 2.0, air_loss: 0.99, loss: 0.3, collision: -0.1, diffusion: 1.5,
        hardness: 0, heat_conduct: 70,
        default_life: 75,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
}
