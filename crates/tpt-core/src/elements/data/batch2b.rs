//! Batch 2b of the remaining Powder Toy elements (`PT_ANAR ..= PT_WIRE`).

use super::super::{
    CtypeDraw, ElementDef, FIRE_ADD, MenuCategory, NO_DECO, PMODE_ADD, PMODE_BLEND, PMODE_BLUR,
    PMODE_FLARE, PMODE_FLAT, PMODE_GLOW,
};
use crate::constants::*;

/// Appends the batch 2b definitions in The Powder Toy's element order.
pub(crate) fn push(v: &mut Vec<ElementDef>) {
    // --- ANAR.cpp (TPT 113) --------------------------------------------------------
    v.push(elem!(
        id: PT_ANAR, ident: "DEFAULT_PT_ANAR", name: "ANAR", cat: MenuCategory::Powders,
        colour: 0xFFFFEE, props: TYPE_PART,
        weight: 85,
        desc: "Anti-air. Very light gravity-defying dust. Burns cold instead of hot.",
        advection: -0.7,
        air_drag: -0.02,
        air_loss: 0.96,
        loss: 0.8,
        collision: 0.1,
        gravity: -0.1,
        falldown: 1,
        hardness: 29,
        heat_conduct: 70,
        pixel_mode: PMODE_FLAT,
    ));
    // --- VINE.cpp (TPT 114) --------------------------------------------------------
    v.push(elem!(
        id: PT_VINE, ident: "DEFAULT_PT_VINE", name: "VINE", cat: MenuCategory::Solids,
        colour: 0x079A00, props: TYPE_SOLID,
        weight: 100,
        desc: "Vine, can grow along WOOD.",
        air_loss: 0.95,
        loss: 0.0,
        flammable: 20,
        hardness: 10,
        heat_conduct: 65,
        default_tmp: 1,
        pixel_mode: PMODE_FLAT,
        high_temp: 573.0,
        high_temp_transition: PT_FIRE as i32,
    ));
    // --- INVIS.cpp (TPT 115) -------------------------------------------------------
    v.push(elem!(
        id: PT_INVIS, ident: "DEFAULT_PT_INVIS", name: "INVS", cat: MenuCategory::Sensors,
        colour: 0x00CCCC, props: TYPE_SOLID | PROP_NEUTPASS | PROP_PHOTPASS,
        weight: 100,
        desc: "Invisible under pressure, allowing particles through.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 15,
        heat_conduct: 164,
        pixel_mode: PMODE_FLAT,
    ));
    // --- E116.cpp (TPT 116) --------------------------------------------------------
    v.push(elem!(
        id: PT_116, ident: "DEFAULT_PT_116", name: "EQVE", cat: MenuCategory::Special,
        colour: 0xFFE0A0, props: TYPE_PART,
        weight: 85,
        desc: "A failed shared velocity test.",
        advection: 0.7,
        air_drag: 0.02,
        air_loss: 0.96,
        loss: 0.8,
        gravity: 0.1,
        falldown: 1,
        heat_conduct: 70,
        pixel_mode: PMODE_FLAT,
    ));
    // --- SPAWN2.cpp (TPT 117) ------------------------------------------------------
    v.push(elem!(
        id: PT_SPAWN2, ident: "DEFAULT_PT_SPAWN2", name: "SPWN2", cat: MenuCategory::Solids,
        colour: 0xAAAAAA, props: TYPE_SOLID,
        weight: 100,
        desc: "STK2 spawn point.",
        air_loss: 1.0,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
    ));
    // --- SPAWN.cpp (TPT 118) -------------------------------------------------------
    v.push(elem!(
        id: PT_SPAWN, ident: "DEFAULT_PT_SPAWN", name: "SPWN", cat: MenuCategory::Solids,
        colour: 0xAAAAAA, props: TYPE_SOLID,
        weight: 100,
        desc: "STKM spawn point.",
        air_loss: 1.0,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
    ));
    // --- SHLD1.cpp (TPT 119) -------------------------------------------------------
    v.push(elem!(
        id: PT_SHLD1, ident: "DEFAULT_PT_SHLD1", name: "SHLD", cat: MenuCategory::Solids,
        colour: 0xAAAAAA, props: TYPE_SOLID | PROP_LIFE_DEC,
        weight: 100,
        desc: "Shield. Grows around spark, broken by pressure.",
        air_loss: 1.0,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
        high_pressure: 7.0,
        high_pressure_transition: PT_NONE as i32,
    ));
    // --- SHLD2.cpp (TPT 120) -------------------------------------------------------
    v.push(elem!(
        id: PT_SHLD2, ident: "DEFAULT_PT_SHLD2", name: "SHD2", cat: MenuCategory::Solids,
        colour: 0x777777, props: TYPE_SOLID | PROP_LIFE_DEC,
        weight: 100,
        desc: "Shield lvl 2.",
        air_loss: 1.0,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
        high_pressure: 15.0,
        high_pressure_transition: PT_NONE as i32,
    ));
    // --- SHLD3.cpp (TPT 121) -------------------------------------------------------
    v.push(elem!(
        id: PT_SHLD3, ident: "DEFAULT_PT_SHLD3", name: "SHD3", cat: MenuCategory::Solids,
        colour: 0x444444, props: TYPE_SOLID | PROP_LIFE_DEC,
        weight: 100,
        desc: "Shield lvl 3.",
        air_loss: 1.0,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
        high_pressure: 25.0,
        high_pressure_transition: PT_NONE as i32,
    ));
    // --- SHLD4.cpp (TPT 122) -------------------------------------------------------
    v.push(elem!(
        id: PT_SHLD4, ident: "DEFAULT_PT_SHLD4", name: "SHD4", cat: MenuCategory::Solids,
        colour: 0x212121, props: TYPE_SOLID | PROP_LIFE_DEC,
        weight: 100,
        desc: "Shield lvl 4.",
        air_loss: 1.0,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
        high_pressure: 40.0,
        high_pressure_transition: PT_NONE as i32,
    ));
    // --- LOLZ.cpp (TPT 123) --------------------------------------------------------
    v.push(elem!(
        id: PT_LOLZ, ident: "DEFAULT_PT_LOLZ", name: "LOLZ", cat: MenuCategory::Special,
        colour: 0x569212, props: TYPE_SOLID,
        weight: 100,
        desc: "Lolz",
        loss: 0.0,
        air_loss: 0.0,
        hardness: 0,
        heat_conduct: 40,
        default_temp: 373.0,
        pixel_mode: PMODE_FLAT,
    ));
    // --- WIFI.cpp (TPT 124) --------------------------------------------------------
    v.push(elem!(
        id: PT_WIFI, ident: "DEFAULT_PT_WIFI", name: "WIFI", cat: MenuCategory::Electronics,
        colour: 0x40A060, props: TYPE_SOLID,
        weight: 100,
        desc: "Wireless transmitter, transfers spark to any other wifi on the same temperature channel.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 2,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
        high_pressure: 15.0,
        high_pressure_transition: PT_BRMT as i32,
    ));
    // --- FILT.cpp (TPT 125) --------------------------------------------------------
    v.push(elem!(
        id: PT_FILT, ident: "DEFAULT_PT_FILT", name: "FILT", cat: MenuCategory::Solids,
        colour: 0x000056, props: TYPE_SOLID | PROP_PHOTPASS | PROP_NOAMBHEAT | PROP_LIFE_DEC,
        weight: 100,
        desc: "Filter. Changes color of PHOT and BIZR. Color depends on temperature.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 251,
        pixel_mode: PMODE_BLEND,
    ));
    // --- ARAY.cpp (TPT 126) --------------------------------------------------------
    v.push(elem!(
        id: PT_ARAY, ident: "DEFAULT_PT_ARAY", name: "ARAY", cat: MenuCategory::Electronics,
        colour: 0xFFBB00, props: TYPE_SOLID,
        weight: 100,
        desc: "Ray Emitter. Rays create points when they collide.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
    ));
    // --- BRAY.cpp (TPT 127) --------------------------------------------------------
    v.push(elem!(
        id: PT_BRAY, ident: "DEFAULT_PT_BRAY", name: "BRAY", cat: MenuCategory::Electronics,
        colour: 0xFFFFFF, props: TYPE_SOLID | PROP_LIFE_DEC | PROP_LIFE_KILL,
        weight: 100,
        desc: "Ray Point. Rays create points when they collide.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 251,
        default_life: 30,
        pixel_mode: PMODE_BLEND | PMODE_GLOW,
    ));
    // --- STKM2.cpp (TPT 128) -------------------------------------------------------
    v.push(elem!(
        id: PT_STKM2, ident: "DEFAULT_PT_STKM2", name: "STK2", cat: MenuCategory::Special,
        colour: 0x6464FF, props: PROP_NOCTYPEDRAW,
        weight: 50,
        desc: "Second stickman. Don't kill him! Control with wasd.",
        advection: 0.5,
        air_loss: 0.2,
        loss: 1.0,
        hardness: 0,
        heat_conduct: 0,
        default_temp: 309.75,
        default_life: 100,
        pixel_mode: PMODE_FLAT,
        high_temp: 620.0,
        high_temp_transition: PT_FIRE as i32,
    ));
    // --- BOMB.cpp (TPT 129) --------------------------------------------------------
    v.push(elem!(
        id: PT_BOMB, ident: "DEFAULT_PT_BOMB", name: "BOMB", cat: MenuCategory::Explosives,
        colour: 0xFFF288, props: TYPE_PART | PROP_SPARKSETTLE,
        weight: 30,
        desc: "Bomb. Explodes and destroys all surrounding particles when it touches something.",
        advection: 0.6,
        air_drag: 0.01,
        air_loss: 0.98,
        loss: 0.95,
        gravity: 0.1,
        falldown: 1,
        hardness: 20,
        heat_conduct: 29,
        default_temp: 293.15,
        pixel_mode: PMODE_FLAT | PMODE_FLARE,
    ));
    // --- C5.cpp (TPT 130) ----------------------------------------------------------
    v.push(elem!(
        id: PT_C5, ident: "DEFAULT_PT_C5", name: "C-5", cat: MenuCategory::Explosives,
        colour: 0x2050E0, props: TYPE_SOLID | PROP_NEUTPENETRATE | PROP_PHOTPASS | PROP_LIFE_DEC,
        weight: 100,
        desc: "Cold explosive, set off by anything cold.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 88,
        pixel_mode: FIRE_ADD | PMODE_ADD | NO_DECO,
        fire_colour: 0x64000000,
    ));
    // --- SING.cpp (TPT 131) --------------------------------------------------------
    v.push(elem!(
        id: PT_SING, ident: "DEFAULT_PT_SING", name: "SING", cat: MenuCategory::Radioactive,
        colour: 0x242424, props: TYPE_PART | PROP_LIFE_DEC,
        weight: 86,
        desc: "Singularity. Creates huge amounts of negative pressure and destroys everything.",
        advection: 0.7,
        air_drag: 0.36,
        air_loss: 0.96,
        loss: 0.8,
        collision: 0.1,
        gravity: 0.12,
        hot_air: -0.001,
        falldown: 1,
        hardness: 0,
        heat_conduct: 70,
        pixel_mode: PMODE_FLAT,
    ));
    // --- QRTZ.cpp (TPT 132) --------------------------------------------------------
    v.push(elem!(
        id: PT_QRTZ, ident: "DEFAULT_PT_QRTZ", name: "QRTZ", cat: MenuCategory::Solids,
        colour: 0xAADDDD, props: TYPE_SOLID | PROP_PHOTPASS | PROP_HOT_GLOW | PROP_LIFE_DEC,
        weight: 100,
        desc: "Quartz, breakable mineral. Conducts but becomes brittle when cold. Scatters photons.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 3,
        pixel_mode: PMODE_FLAT,
        high_temp: 2573.15,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- PQRT.cpp (TPT 133) --------------------------------------------------------
    v.push(elem!(
        id: PT_PQRT, ident: "DEFAULT_PT_PQRT", name: "PQRT", cat: MenuCategory::Powders,
        colour: 0x88BBBB, props: TYPE_PART | PROP_PHOTPASS | PROP_HOT_GLOW,
        weight: 90,
        desc: "Powdered quartz, broken form of QRTZ.",
        advection: 0.4,
        air_drag: 0.04,
        air_loss: 0.94,
        loss: 0.95,
        collision: -0.1,
        gravity: 0.27,
        falldown: 1,
        hardness: 0,
        heat_conduct: 3,
        pixel_mode: PMODE_FLAT,
        high_temp: 2573.15,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- EMP.cpp (TPT 134) ---------------------------------------------------------
    v.push(elem!(
        id: PT_EMP, ident: "DEFAULT_PT_EMP", name: "EMP", cat: MenuCategory::Electronics,
        colour: 0x66AAFF, props: TYPE_SOLID | PROP_LIFE_DEC,
        weight: 100,
        desc: "Electromagnetic pulse. Breaks activated electronics.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 3,
        heat_conduct: 121,
        pixel_mode: PMODE_FLAT,
    ));
    // --- BREC.cpp (TPT 135) --------------------------------------------------------
    v.push(elem!(
        id: PT_BREC, ident: "DEFAULT_PT_BREC", name: "BREL", cat: MenuCategory::Powders,
        colour: 0x707060, props: TYPE_PART | PROP_CONDUCTS | PROP_LIFE_DEC | PROP_HOT_GLOW,
        weight: 90,
        desc: "Broken electronics. Formed from EMP blasts, and when constantly sparked while under pressure, turns to EXOT.",
        advection: 0.4,
        air_drag: 0.04,
        air_loss: 0.94,
        loss: 0.95,
        collision: -0.1,
        gravity: 0.18,
        falldown: 1,
        meltable: 2,
        hardness: 2,
        heat_conduct: 211,
        pixel_mode: PMODE_FLAT,
    ));
    // --- ACEL.cpp (TPT 137) --------------------------------------------------------
    v.push(elem!(
        id: PT_ACEL, ident: "DEFAULT_PT_ACEL", name: "ACEL", cat: MenuCategory::Force,
        colour: 0x0099CC, props: TYPE_SOLID,
        weight: 100,
        desc: "Accelerator, speeds up nearby elements.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 251,
        pixel_mode: PMODE_FLAT,
    ));
    // --- DCEL.cpp (TPT 138) --------------------------------------------------------
    v.push(elem!(
        id: PT_DCEL, ident: "DEFAULT_PT_DCEL", name: "DCEL", cat: MenuCategory::Force,
        colour: 0x99CC00, props: TYPE_SOLID,
        weight: 100,
        desc: "Decelerator, slows down nearby elements.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 251,
        pixel_mode: PMODE_FLAT,
    ));
    // --- BANG.cpp (TPT 139) --------------------------------------------------------
    v.push(elem!(
        id: PT_BANG, ident: "DEFAULT_PT_BANG", name: "TNT", cat: MenuCategory::Explosives,
        colour: 0xC05050, props: TYPE_SOLID | PROP_NEUTPENETRATE,
        weight: 100,
        desc: "TNT, explodes all at once.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 88,
        pixel_mode: PMODE_FLAT,
    ));
    // --- IGNT.cpp (TPT 140) --------------------------------------------------------
    v.push(elem!(
        id: PT_IGNT, ident: "DEFAULT_PT_IGNT", name: "IGNC", cat: MenuCategory::Explosives,
        colour: 0xC0B050, props: TYPE_SOLID | PROP_NEUTPENETRATE | PROP_SPARKSETTLE | PROP_LIFE_KILL,
        weight: 100,
        desc: "Ignition cord. Burns slowly with fire and sparks.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 88,
        default_life: 3,
        pixel_mode: PMODE_FLAT,
        high_temp: 673.0,
        high_temp_transition: PT_FIRE as i32,
    ));
    // --- BOYL.cpp (TPT 141) --------------------------------------------------------
    v.push(elem!(
        id: PT_BOYL, ident: "DEFAULT_PT_BOYL", name: "BOYL", cat: MenuCategory::Gases,
        colour: 0x0A3200, props: TYPE_GAS,
        weight: 1,
        desc: "Boyle, variable pressure gas. Expands when heated.",
        advection: 1.0,
        air_drag: 0.01,
        air_loss: 0.99,
        loss: 0.3,
        collision: -0.1,
        diffusion: 0.18,
        hardness: 1,
        heat_conduct: 42,
        default_temp: 297.15,
        pixel_mode: PMODE_FLAT,
    ));
    // --- GEL.cpp (TPT 142) ---------------------------------------------------------
    v.push(elem!(
        id: PT_GEL, ident: "DEFAULT_PT_GEL", name: "GEL", cat: MenuCategory::Liquids,
        colour: 0xFF9900, props: TYPE_LIQUID | PROP_LIFE_DEC | PROP_NEUTPENETRATE,
        weight: 35,
        desc: "A liquid with variable viscosity and heat conductivity. Absorbs water.",
        advection: 0.6,
        air_drag: 0.01,
        air_loss: 0.98,
        loss: 0.95,
        gravity: 0.1,
        falldown: 2,
        hardness: 19,
        heat_conduct: 29,
        default_temp: 293.15,
        pixel_mode: PMODE_FLAT | PMODE_BLUR,
    ));
    // --- TRON.cpp (TPT 143) --------------------------------------------------------
    v.push(elem!(
        id: PT_TRON, ident: "DEFAULT_PT_TRON", name: "TRON", cat: MenuCategory::Special,
        colour: 0xA9FF00, props: TYPE_SOLID | PROP_LIFE_DEC | PROP_LIFE_KILL,
        weight: 100,
        desc: "Smart particles, Travels in straight lines and avoids obstacles. Grows with time.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 40,
        default_temp: 0.0,
        pixel_mode: PMODE_FLAT,
    ));
    // --- EXOT.cpp (TPT 145) --------------------------------------------------------
    v.push(elem!(
        id: PT_EXOT, ident: "DEFAULT_PT_EXOT", name: "EXOT", cat: MenuCategory::Radioactive,
        colour: 0x247BFE, props: TYPE_LIQUID | PROP_NEUTPASS,
        weight: 46,
        desc: "Exotic matter. Explodes with excess exposure to electrons. Has many other odd reactions.",
        advection: 0.3,
        air_drag: 0.02,
        air_loss: 0.95,
        loss: 0.8,
        gravity: 0.15,
        hot_air: 0.0003,
        falldown: 2,
        hardness: 2,
        heat_conduct: 250,
        default_temp: 293.15,
        default_life: 1000,
        default_tmp: 244,
        pixel_mode: FIRE_ADD | PMODE_BLUR,
    ));
    // --- EMBR.cpp (TPT 147) --------------------------------------------------------
    v.push(elem!(
        id: PT_EMBR, ident: "DEFAULT_PT_EMBR", name: "EMBR", cat: MenuCategory::Explosives,
        colour: 0xFFF288, props: TYPE_PART | PROP_LIFE_DEC | PROP_LIFE_KILL | PROP_SPARKSETTLE,
        weight: 30,
        desc: "Sparks. Formed by explosions.",
        advection: 0.4,
        air_drag: 0.001,
        air_loss: 0.99,
        loss: 0.9,
        gravity: 0.07,
        falldown: 1,
        hardness: 21,
        heat_conduct: 29,
        default_temp: 773.15,
        default_life: 50,
        pixel_mode: FIRE_ADD | PMODE_BLEND | PMODE_GLOW,
    ));
    // --- SOAP.cpp (TPT 149) --------------------------------------------------------
    v.push(elem!(
        id: PT_SOAP, ident: "DEFAULT_PT_SOAP", name: "SOAP", cat: MenuCategory::Liquids,
        colour: 0xF5F5DC, props: TYPE_LIQUID | PROP_NEUTPENETRATE | PROP_LIFE_DEC,
        weight: 35,
        desc: "Soap. Creates bubbles, washes off deco color, and cures virus.",
        advection: 0.6,
        air_drag: 0.01,
        air_loss: 0.98,
        loss: 0.95,
        gravity: 0.1,
        falldown: 2,
        hardness: 19,
        heat_conduct: 29,
        default_temp: 293.15,
        default_tmp: -1,
        default_tmp2: -1,
        pixel_mode: PMODE_FLAT | PMODE_BLUR,
    ));
    // --- MERC.cpp (TPT 152) --------------------------------------------------------
    v.push(elem!(
        id: PT_MERC, ident: "DEFAULT_PT_MERC", name: "MERC", cat: MenuCategory::Liquids,
        colour: 0x736B6D, props: TYPE_LIQUID | PROP_CONDUCTS | PROP_NEUTABSORB | PROP_LIFE_DEC,
        weight: 91,
        desc: "Mercury. Volume changes with temperature, Conductive.",
        advection: 0.4,
        air_drag: 0.04,
        air_loss: 0.94,
        loss: 0.8,
        gravity: 0.3,
        falldown: 2,
        hardness: 18,
        heat_conduct: 251,
        default_tmp: 10,
        pixel_mode: PMODE_FLAT,
    ));
    // --- PBCN.cpp (TPT 153) --------------------------------------------------------
    v.push(elem!(
        id: PT_PBCN, ident: "DEFAULT_PT_PBCN", name: "PBCN", cat: MenuCategory::Powered,
        colour: 0x3B1D0A, props: TYPE_SOLID | PROP_PHOTPASS | PROP_NOCTYPEDRAW,
        weight: 100,
        desc: "Powered breakable clone.",
        air_loss: 0.97,
        loss: 0.5,
        hardness: 12,
        heat_conduct: 251,
        pixel_mode: PMODE_FLAT,
    ctype_draw: CtypeDraw::Pcln,
    ));
    // --- CLST.cpp (TPT 155) --------------------------------------------------------
    v.push(elem!(
        id: PT_CLST, ident: "DEFAULT_PT_CLST", name: "CLST", cat: MenuCategory::Powders,
        colour: 0xE4A4A4, props: TYPE_PART,
        weight: 55,
        desc: "Clay dust. Produces paste when mixed with water.",
        advection: 0.7,
        air_drag: 0.02,
        air_loss: 0.94,
        loss: 0.95,
        gravity: 0.2,
        falldown: 1,
        meltable: 2,
        hardness: 2,
        heat_conduct: 70,
        pixel_mode: PMODE_FLAT,
        high_temp: 1256.0,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- WIRE.cpp (TPT 156) --------------------------------------------------------
    v.push(elem!(
        id: PT_WIRE, ident: "DEFAULT_PT_WIRE", name: "WWLD", cat: MenuCategory::Electronics,
        colour: 0xFFCC00, props: TYPE_SOLID,
        weight: 100,
        desc: "WireWorld wires, conducts based on a set of GOL-like rules.",
        loss: 0.0,
        air_loss: 0.0,
        hardness: 0,
        heat_conduct: 250,
        pixel_mode: PMODE_FLAT,
    ));
}
