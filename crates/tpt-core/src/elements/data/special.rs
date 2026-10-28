//! Explosive and special-purpose elements (`MenuCategory::Explosives` and
//! `MenuCategory::Special`).
//!
//! The special-category slots are spread across the built-in table in TPT's
//! order (`PT_NONE` first, `CLNE`/`VOID` after the electronics, the Newtonian
//! gravity family last), so this module exposes one `push_*` per section and
//! [`super::super::builtin_elements`] calls them at the matching positions.

use super::super::{CtypeDraw, ElementDef, FIRE_ADD, MenuCategory, PMODE_FLAT};
use crate::constants::*;

/// Appends the `PT_NONE` empty-cell placeholder.
pub(crate) fn push_empty(v: &mut Vec<ElementDef>) {
    // 0: empty-cell placeholder. Never simulated and not drawable, but
    // present so that every `pmap` type is a valid registry index and
    // `find_by_name("NONE")` works.
    v.push(ElementDef {
        id: PT_NONE,
        identifier: String::from("DEFAULT_PT_NONE"),
        name: String::from("NONE"),
        description: String::from("Empty space. No particle."),
        category: MenuCategory::Special,
        colour: 0x000000,
        properties: 0,
        weight: 0,
        heat_conduct: 0,
        default_temp: 295.15,
        ..ElementDef::default()
    });
}

/// Appends the explosive definitions in their original table order.
pub(crate) fn push_explosives(v: &mut Vec<ElementDef>) {
    v.push(elem!(
        id: PT_FIRE, ident: "DEFAULT_PT_FIRE", name: "FIRE", cat: MenuCategory::Explosives,
        colour: 0xFF1000, props: TYPE_GAS | PROP_LIFE_DEC | PROP_LIFE_KILL, weight: 2,
        desc: "Ignites flammable materials. Heats air.",
        advection: 0.9, air_drag: 0.04, air_loss: 0.97, loss: 0.2, gravity: -0.1,
        hot_air: 0.001, falldown: 1,
        hardness: 1, heat_conduct: 88,
        default_temp: 695.15,
        high_temp: 2773.0, high_temp_transition: PT_PLSM as i32,
        pixel_mode: FIRE_ADD, fire_colour: 0xFF00_0000,
    ));
    v.push(elem!(
        id: PT_GUNP, ident: "DEFAULT_PT_GUNP", name: "GUNP", cat: MenuCategory::Explosives,
        colour: 0xC0C0D0, props: TYPE_PART, weight: 85,
        desc: "Gunpowder. Light dust, explodes on contact with fire or spark.",
        advection: 0.7, air_drag: 0.02, air_loss: 0.94, loss: 0.8, collision: -0.1,
        gravity: 0.1, falldown: 1,
        flammable: 600, explosive: 1, hardness: 10, heat_conduct: 97,
        high_temp: 673.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_NITR, ident: "DEFAULT_PT_NITR", name: "NITR", cat: MenuCategory::Explosives,
        photon_reflect: 0x0007C000,
        colour: 0x20E010, props: TYPE_LIQUID, weight: 23,
        desc: "Nitroglycerin. Pressure sensitive explosive. Mix with CLST to make TNT.",
        advection: 0.5, air_drag: 0.02, air_loss: 0.92, loss: 0.97, gravity: 0.2, falldown: 2,
        flammable: 1000, explosive: 2, hardness: 3, heat_conduct: 50,
        high_temp: 673.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // TPT renamed C-4 to PLEX internally; `PT_C4` keeps the classic id while
    // the definition (and the Lua name `DEFAULT_PT_PLEX`) match `PLEX.cpp`.
    v.push(elem!(
        id: PT_C4, ident: "DEFAULT_PT_PLEX", name: "C-4", cat: MenuCategory::Explosives,
        photon_reflect: 0x1F00003E,
        colour: 0xD080E0, props: TYPE_SOLID | PROP_NEUTPENETRATE, weight: 100,
        desc: "Solid pressure sensitive explosive.",
        air_loss: 0.9, loss: 0.0,
        flammable: 1000, explosive: 2, meltable: 50, hardness: 1, heat_conduct: 88,
        high_temp: 673.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
}

/// Appends `CLNE` and `VOID`, the first special-category section.
pub(crate) fn push_specials(v: &mut Vec<ElementDef>) {
    v.push(elem!(
        id: PT_CLNE, ident: "DEFAULT_PT_CLNE", name: "CLNE", cat: MenuCategory::Special,
        colour: 0xFFD010, props: TYPE_SOLID | PROP_PHOTPASS | PROP_NOCTYPEDRAW, weight: 100,
        desc: "Clone. Duplicates any particles it touches.",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ctype_draw: CtypeDraw::ValueInTmp,
    ));
    v.push(elem!(
        id: PT_VOID, ident: "DEFAULT_PT_VOID", name: "VOID", cat: MenuCategory::Special,
        colour: 0x790B0B, props: TYPE_SOLID, weight: 100,
        desc: "Hole, will drain away any particles.",
        loss: 0.0, hot_air: -0.0003,
        hardness: 0, heat_conduct: 251,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ctype_draw: CtypeDraw::Basic,
    ));
}

/// Appends the Newtonian gravity specials, the last section of the table.
pub(crate) fn push_gravity(v: &mut Vec<ElementDef>) {
    v.push(elem!(
        id: PT_NBHL, ident: "DEFAULT_PT_NBHL", name: "BHOL", cat: MenuCategory::Special,
        colour: 0x202020, props: TYPE_SOLID, weight: 100,
        desc: "Black hole, sucks in particles using gravity. (Requires Newtonian gravity)",
        air_loss: 0.9, loss: 0.0,
        hardness: 0, heat_conduct: 186,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_NWHL, ident: "DEFAULT_PT_NWHL", name: "WHOL", cat: MenuCategory::Special,
        colour: 0xFFFFFF, props: TYPE_SOLID, weight: 100,
        desc: "White hole, pushes away other particles with gravity. (Requires Newtonian gravity)",
        air_loss: 0.9, loss: 0.0,
        hardness: 0, heat_conduct: 186,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // TPT `GBMB::graphics` adds `PMODE_SPARK` while the bomb still has life
    // and `PMODE_FLARE` once it has spent it; the renderer adds those two
    // dynamically because they depend on `life`, so only the base stays here.
    v.push(elem!(
        id: PT_GBMB, ident: "DEFAULT_PT_GBMB", name: "GBMB", cat: MenuCategory::Force,
        colour: 0x1144BB, props: TYPE_PART | PROP_LIFE_DEC | PROP_LIFE_KILL_DEC, weight: 30,
        desc: "Gravity bomb. Sticks to the first object it touches then produces a strong gravity push.",
        advection: 0.6, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1, falldown: 1,
        hardness: 20, heat_conduct: 29,
        default_temp: 293.15,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    v.push(elem!(
        id: PT_GPMP, ident: "DEFAULT_PT_GPMP", name: "GPMP", cat: MenuCategory::Powered,
        colour: 0x0A3B3B, props: TYPE_SOLID, weight: 100,
        desc: "Gravity pump. Changes gravity to its temp when activated. (use HEAT/COOL)",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 0,
        default_life: 10,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
}
