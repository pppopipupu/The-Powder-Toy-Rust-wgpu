//! Solid elements (`MenuCategory::Solids`).

use super::super::{ElementDef, MenuCategory, PMODE_FLAT};
use crate::constants::*;

/// Appends the solid definitions in their original table order.
pub(crate) fn push(v: &mut Vec<ElementDef>) {
    v.push(elem!(
        id: PT_WOOD, ident: "DEFAULT_PT_WOOD", name: "WOOD", cat: MenuCategory::Solids,
        colour: 0xC0A040, props: TYPE_SOLID | PROP_NEUTPENETRATE, weight: 100,
        desc: "Wood, flammable.",
        air_loss: 0.9, loss: 0.0,
        flammable: 20, hardness: 16, heat_conduct: 164,
        high_temp: 873.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_ICEI, ident: "DEFAULT_PT_ICEI", name: "ICEI", cat: MenuCategory::Solids,
        colour: 0xA0C0FF, props: TYPE_SOLID | PROP_NEUTPASS | PROP_LIFE_DEC, weight: 100,
        desc: "Crushes under pressure. Cools down air.",
        air_loss: 0.9, loss: 0.0, hot_air: -0.0003,
        hardness: 20, heat_conduct: 46,
        default_temp: 245.15, default_ctype: PT_WATR,
        high_pressure: 0.8, high_pressure_transition: PT_SNOW as i32,
        high_temp: 252.05, high_temp_transition: ST,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_GLAS, ident: "DEFAULT_PT_GLAS", name: "GLAS", cat: MenuCategory::Solids,
        colour: 0x404040,
        props: TYPE_SOLID | PROP_PHOTPASS | PROP_NEUTPASS | PROP_HOT_GLOW | PROP_SPARKSETTLE,
        weight: 100,
        desc: "Glass. Meltable. Shatters under pressure, and refracts photons.",
        air_loss: 0.9, loss: 0.0,
        hardness: 0, heat_conduct: 150,
        high_temp: 1973.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_DMND, ident: "DEFAULT_PT_DMND", name: "DMND", cat: MenuCategory::Special,
        colour: 0xCCFFFF, props: TYPE_SOLID, weight: 100,
        desc: "Diamond. Indestructible.",
        air_loss: 0.9, loss: 0.0,
        hardness: 0, heat_conduct: 186,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_TTAN, ident: "DEFAULT_PT_TTAN", name: "TTAN", cat: MenuCategory::Solids,
        colour: 0x909090,
        props: TYPE_SOLID | PROP_CONDUCTS | PROP_HOT_GLOW | PROP_LIFE_DEC, weight: 100,
        desc: "Titanium. Higher melting temperature than most other metals, blocks all air pressure.",
        air_loss: 0.9, loss: 0.0,
        meltable: 1, hardness: 48, heat_conduct: 251,
        high_temp: 1941.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_BRCK, ident: "DEFAULT_PT_BRCK", name: "BRCK", cat: MenuCategory::Solids,
        colour: 0x808080, props: TYPE_SOLID | PROP_HOT_GLOW, weight: 100,
        desc: "Brick, breakable building material.",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        high_pressure: 8.8, high_pressure_transition: PT_STNE as i32,
        high_temp: 1223.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
}
