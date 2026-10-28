//! Liquid elements (`MenuCategory::Liquids`).

use super::super::{ElementDef, FIRE_ADD, MenuCategory, PMODE_BLUR, PMODE_FLAT};
use crate::constants::*;

/// Appends the liquid definitions in their original table order.
pub(crate) fn push(v: &mut Vec<ElementDef>) {
    v.push(elem!(
        id: PT_WATR, ident: "DEFAULT_PT_WATR", name: "WATR", cat: MenuCategory::Liquids,
        colour: 0x2030D0,
        props: TYPE_LIQUID | PROP_CONDUCTS | PROP_PHOTPASS | PROP_NEUTPASS | PROP_LIFE_DEC,
        weight: 30,
        desc: "Water. Conducts electricity, freezes, and extinguishes fires.",
        advection: 0.6, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1, falldown: 2,
        hardness: 20, heat_conduct: 29,
        default_temp: 293.15,
        low_temp: 273.15, low_temp_transition: PT_ICEI as i32,
        high_temp: 373.0, high_temp_transition: PT_WTRV as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_OIL, ident: "DEFAULT_PT_OIL", name: "OIL", cat: MenuCategory::Liquids,
        colour: 0x404010, props: TYPE_LIQUID | PROP_NEUTPASS, weight: 20,
        desc: "Flammable, turns into GAS at low pressure or high temperature.",
        advection: 0.6, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1, falldown: 2,
        flammable: 20, hardness: 5, heat_conduct: 42,
        high_temp: 333.0, high_temp_transition: PT_GAS as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_LAVA, ident: "DEFAULT_PT_LAVA", name: "LAVA", cat: MenuCategory::Liquids,
        photon_reflect: 0x3FF00000,
        colour: 0xE05010, props: TYPE_LIQUID | PROP_LIFE_DEC, weight: 45,
        desc: "Molten lava. Ignites flammable materials. Generated when metals and other materials melt, solidifies when cold.",
        advection: 0.3, air_drag: 0.02, air_loss: 0.95, loss: 0.8, gravity: 0.15,
        hot_air: 0.0003, falldown: 2,
        hardness: 2, heat_conduct: 60,
        default_temp: 1795.15,
        low_temp: 9999.0, low_temp_transition: ST,
        pixel_mode: PMODE_FLAT | FIRE_ADD | PMODE_BLUR, fire_colour: 0x2800_0000,
    ));
    v.push(elem!(
        id: PT_SLTW, ident: "DEFAULT_PT_SLTW", name: "SLTW", cat: MenuCategory::Liquids,
        colour: 0x4050F0,
        props: TYPE_LIQUID | PROP_CONDUCTS | PROP_PHOTPASS | PROP_NEUTPENETRATE
            | PROP_LIFE_DEC,
        weight: 35,
        desc: "Saltwater, conducts electricity, difficult to freeze.",
        advection: 0.6, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1, falldown: 2,
        hardness: 20, heat_conduct: 75,
        low_temp: 252.05, low_temp_transition: PT_ICEI as i32,
        high_temp: 383.0, high_temp_transition: ST,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_ACID, ident: "DEFAULT_PT_ACID", name: "ACID", cat: MenuCategory::Liquids,
        photon_reflect: 0x1FE001FE,
        colour: 0xED55FF, props: TYPE_LIQUID | PROP_DEADLY, weight: 10,
        desc: "Dissolves almost everything.",
        advection: 0.6, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1, falldown: 2,
        flammable: 40, hardness: 0, heat_conduct: 34,
        default_life: 75,
        pixel_mode: PMODE_FLAT | PMODE_BLUR, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_DSTW, ident: "DEFAULT_PT_DSTW", name: "DSTW", cat: MenuCategory::Liquids,
        colour: 0x1020C0, props: TYPE_LIQUID | PROP_PHOTPASS | PROP_NEUTPASS, weight: 30,
        desc: "Distilled water, does not conduct electricity.",
        advection: 0.6, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1, falldown: 2,
        hardness: 20, heat_conduct: 23,
        default_temp: 293.15,
        low_temp: 273.15, low_temp_transition: PT_ICEI as i32,
        high_temp: 373.0, high_temp_transition: PT_WTRV as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // TPT calls this element LNTG internally; `PT_LN2` keeps the classic Rust
    // slot constant while the identifier matches `LNTG.cpp`. It freezes into
    // nitrogen ice (`PT_NICE`) below 63 K and disappears above 77 K.
    v.push(elem!(
        id: PT_LN2, ident: "DEFAULT_PT_LNTG", name: "LN2", cat: MenuCategory::Liquids,
        colour: 0x80A0DF, props: TYPE_LIQUID, weight: 30,
        desc: "Liquid Nitrogen. Very cold, disappears whenever it touches anything warmer.",
        advection: 0.6, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1, falldown: 2,
        hardness: 0, heat_conduct: 70,
        default_temp: 70.15,
        low_temp: 63.0, low_temp_transition: PT_NICE as i32,
        high_temp: 77.0, high_temp_transition: PT_NONE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
}
