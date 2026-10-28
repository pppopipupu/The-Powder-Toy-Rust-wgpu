//! Batch 1 of the remaining Powder Toy elements (`PT_GOO ..= PT_TSNS`).
//!
//! These slots were appended above the classic ids (which saves, Lua scripts
//! and the displacement tests pin), so unlike the category modules above they
//! cannot be grouped by menu section: the emitted order is The Powder Toy's own
//! element order, and every definition carries its own [`MenuCategory`].
//!
//! Values are transcribed from
//! `../The-Powder-Toy/src/simulation/elements/<IDENT>.cpp` with `CFDS == 1`
//! (one particle per pixel), so `AirDrag`/`HotAir` keep the raw table values.
//! Each entry names the reference `PT_*` id in [`crate::constants`] so the
//! mapping to the C++ `ElementNumbers.h` slot stays checkable.

use super::super::{CtypeDraw, ElementDef, MenuCategory, NO_DECO, PMODE_BLUR, PMODE_FLAT};
use crate::constants::*;

/// Appends the batch 1 definitions in The Powder Toy's element order.
pub(crate) fn push(v: &mut Vec<ElementDef>) {
    // --- GOO.cpp (TPT 12) --------------------------------------------------
    v.push(elem!(
        id: PT_GOO, ident: "DEFAULT_PT_GOO", name: "GOO", cat: MenuCategory::Solids,
        photon_reflect: 0x3FFAAA00,
        colour: 0x804000,
        props: TYPE_SOLID | PROP_NEUTPENETRATE | PROP_LIFE_DEC | PROP_LIFE_KILL_DEC,
        weight: 100,
        desc: "Deforms and disappears under pressure.",
        air_loss: 0.97, loss: 0.5,
        hardness: 12, heat_conduct: 75,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- PLNT.cpp (TPT 20) -------------------------------------------------
    v.push(elem!(
        id: PT_PLNT, ident: "DEFAULT_PT_PLNT", name: "PLNT", cat: MenuCategory::Solids,
        photon_reflect: 0x0007C000,
        colour: 0x0CAC00, props: TYPE_SOLID | PROP_NEUTPENETRATE | PROP_LIFE_DEC,
        weight: 100,
        desc: "Plant, drinks water and grows.",
        air_loss: 0.95, loss: 0.0,
        flammable: 20, hardness: 10, heat_conduct: 65,
        high_temp: 573.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- BMTL.cpp (TPT 29) -------------------------------------------------
    v.push(elem!(
        id: PT_BMTL, ident: "DEFAULT_PT_BMTL", name: "BMTL", cat: MenuCategory::Solids,
        colour: 0x505070,
        props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC | PROP_HOT_GLOW,
        weight: 100,
        desc: "Breakable metal. Common conductive building material, can melt and break under pressure.",
        air_loss: 0.9, loss: 0.0,
        meltable: 1, hardness: 1, heat_conduct: 251,
        // `ST` because TPT's own update owns the breakage; like the reference,
        // the value is data only and never fires in the generic pressure path.
        high_pressure: 1.0, high_pressure_transition: ST,
        high_temp: 1273.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- BRMT.cpp (TPT 30) -------------------------------------------------
    v.push(elem!(
        id: PT_BRMT, ident: "DEFAULT_PT_BRMT", name: "BRMT", cat: MenuCategory::Powders,
        colour: 0x705060,
        props: TYPE_PART | PROP_CONDUCTS | PROP_LIFE_DEC | PROP_HOT_GLOW,
        weight: 90,
        desc: "Broken metal. Created when iron rusts or when metals break from pressure.",
        advection: 0.4, air_drag: 0.04, air_loss: 0.94, loss: 0.95, collision: -0.1, gravity: 0.3,
        falldown: 1,
        meltable: 2, hardness: 2, heat_conduct: 211,
        // `ST`: the generic phase path melts it at 1273 K, see
        // `thermal_kernel::phase_transition_kernel`.
        high_temp: 1273.0, high_temp_transition: ST,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- WAX.cpp (TPT 33) --------------------------------------------------
    v.push(elem!(
        id: PT_WAX, ident: "DEFAULT_PT_WAX", name: "WAX", cat: MenuCategory::Solids,
        colour: 0xF0F0BB, props: TYPE_SOLID, weight: 100,
        desc: "Flammable, melts at moderately high temperatures.",
        air_loss: 0.9, loss: 0.0,
        hardness: 10, heat_conduct: 44,
        high_temp: 319.0, high_temp_transition: PT_MWAX as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- MWAX.cpp (TPT 34) -------------------------------------------------
    v.push(elem!(
        id: PT_MWAX, ident: "DEFAULT_PT_MWAX", name: "MWAX", cat: MenuCategory::Liquids,
        colour: 0xE0E0AA, props: TYPE_LIQUID, weight: 25,
        desc: "Liquid Wax. Hardens into WAX at 45 degrees.",
        advection: 0.3, air_drag: 0.02, air_loss: 0.95, loss: 0.8, gravity: 0.15,
        hot_air: 0.000001, falldown: 2,
        flammable: 5, hardness: 2, heat_conduct: 44,
        default_temp: 323.15,
        low_temp: 318.0, low_temp_transition: PT_WAX as i32,
        high_temp: 673.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- PSCN.cpp (TPT 35) -------------------------------------------------
    v.push(elem!(
        id: PT_PSCN, ident: "DEFAULT_PT_PSCN", name: "PSCN", cat: MenuCategory::Electronics,
        photon_reflect: 0x00000000,
        colour: 0x805050, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC, weight: 100,
        desc: "P-Type Silicon, Will transfer current to any conductor. Enables powered materials.",
        air_loss: 0.9, loss: 0.0,
        meltable: 1, hardness: 1, heat_conduct: 251,
        high_temp: 1687.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- NSCN.cpp (TPT 36) -------------------------------------------------
    v.push(elem!(
        id: PT_NSCN, ident: "DEFAULT_PT_NSCN", name: "NSCN", cat: MenuCategory::Electronics,
        photon_reflect: 0x00000000,
        colour: 0x505080, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC, weight: 100,
        desc: "N-Type Silicon, Will not transfer current to P-Type Silicon. Disables powered materials.",
        air_loss: 0.9, loss: 0.0,
        meltable: 1, hardness: 1, heat_conduct: 251,
        high_temp: 1687.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- INSL.cpp (TPT 38) -------------------------------------------------
    v.push(elem!(
        id: PT_INSL, ident: "DEFAULT_PT_INSL", name: "INSL", cat: MenuCategory::Electronics,
        colour: 0x9EA3B6, props: TYPE_SOLID, weight: 100,
        desc: "Insulator. Blocks heat, electricity, and radiation.",
        air_loss: 0.95, loss: 0.0,
        flammable: 7, hardness: 10,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- NTCT.cpp (TPT 43) -------------------------------------------------
    v.push(elem!(
        id: PT_NTCT, ident: "DEFAULT_PT_NTCT", name: "NTCT", cat: MenuCategory::Electronics,
        colour: 0x505040, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC, weight: 100,
        desc: "NTC Thermistor. Conducts with PSCN and NSCN, but only when heated above 100C.",
        air_loss: 0.9, loss: 0.0,
        meltable: 1, hardness: 1, heat_conduct: 251,
        high_temp: 1687.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- PTCT.cpp (TPT 46) -------------------------------------------------
    v.push(elem!(
        id: PT_PTCT, ident: "DEFAULT_PT_PTCT", name: "PTCT", cat: MenuCategory::Electronics,
        colour: 0x405050, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC, weight: 100,
        desc: "PTC Thermistor. Conducts with PSCN and NSCN, but only when cooled below 100C.",
        air_loss: 0.9, loss: 0.0,
        meltable: 1, hardness: 1, heat_conduct: 251,
        high_temp: 1687.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- ETRD.cpp (TPT 50) -------------------------------------------------
    v.push(elem!(
        id: PT_ETRD, ident: "DEFAULT_PT_ETRD", name: "ETRD", cat: MenuCategory::Electronics,
        colour: 0x404040, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC, weight: 100,
        desc: "Electrode. Creates plasma arcs with electricity. (Use sparingly)",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- NICE.cpp (TPT 51) -------------------------------------------------
    v.push(elem!(
        id: PT_NICE, ident: "DEFAULT_PT_NICE", name: "NICE", cat: MenuCategory::Solids,
        colour: 0xC0E0FF, props: TYPE_SOLID, weight: 100,
        desc: "Nitrogen Ice. Very cold, will melt into LN2 when heated only slightly.",
        air_loss: 0.9, loss: 0.0,
        hot_air: -0.0005,
        hardness: 20, heat_conduct: 46,
        default_temp: 35.0,
        high_temp: 63.1, high_temp_transition: PT_LN2 as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- BTRY.cpp (TPT 53) -------------------------------------------------
    v.push(elem!(
        id: PT_BTRY, ident: "DEFAULT_PT_BTRY", name: "BTRY", cat: MenuCategory::Electronics,
        colour: 0x858505, props: TYPE_SOLID, weight: 100,
        desc: "Battery. Generates infinite electricity.",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        high_temp: 2273.0, high_temp_transition: PT_PLSM as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- LCRY.cpp (TPT 54) -------------------------------------------------
    v.push(elem!(
        id: PT_LCRY, ident: "DEFAULT_PT_LCRY", name: "LCRY", cat: MenuCategory::Powered,
        colour: 0x505050, props: TYPE_SOLID, weight: 100,
        desc: "Liquid Crystal. Changes colour when charged. (PSCN Charges, NSCN Discharges)",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        high_temp: 1273.0, high_temp_transition: PT_BGLA as i32,
        pixel_mode: PMODE_FLAT | NO_DECO, fire_colour: 0,
    ));
    // --- STKM.cpp (TPT 55) -------------------------------------------------
    v.push(elem!(
        id: PT_STKM, ident: "DEFAULT_PT_STKM", name: "STKM", cat: MenuCategory::Special,
        colour: 0xFFE0A0, props: PROP_NOCTYPEDRAW, weight: 50,
        desc: "Stickman. Don't kill him! Control with the arrow keys.",
        advection: 0.5, air_loss: 0.2, loss: 1.0,
        hardness: 0, heat_conduct: 0,
        default_temp: 309.75, default_life: 100,
        high_temp: 620.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- SWCH.cpp (TPT 56) -------------------------------------------------
    v.push(elem!(
        id: PT_SWCH, ident: "DEFAULT_PT_SWCH", name: "SWCH", cat: MenuCategory::Electronics,
        colour: 0x103B11, props: TYPE_SOLID, weight: 100,
        desc: "Switch. Only conducts when switched on. (PSCN switches on, NSCN switches off)",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- DESL.cpp (TPT 58) -------------------------------------------------
    v.push(elem!(
        id: PT_DESL, ident: "DEFAULT_PT_DESL", name: "DESL", cat: MenuCategory::Liquids,
        colour: 0x440000, props: TYPE_LIQUID, weight: 15,
        desc: "Liquid diesel. Explodes under high pressure and temperatures.",
        advection: 1.0, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1,
        falldown: 2,
        flammable: 2, hardness: 5, heat_conduct: 42,
        high_pressure: 5.0, high_pressure_transition: PT_FIRE as i32,
        high_temp: 335.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- COAL.cpp (TPT 59) -------------------------------------------------
    v.push(elem!(
        id: PT_COAL, ident: "DEFAULT_PT_COAL", name: "COAL", cat: MenuCategory::Solids,
        photon_reflect: 0x00000000,
        colour: 0x222222, props: TYPE_SOLID, weight: 100,
        desc: "Coal, Burns very slowly. Gets red when hot.",
        air_loss: 0.9, loss: 0.0,
        hardness: 18, heat_conduct: 200,
        default_life: 110, default_tmp: 50,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- INWR.cpp (TPT 62) -------------------------------------------------
    v.push(elem!(
        id: PT_INWR, ident: "DEFAULT_PT_INWR", name: "INWR", cat: MenuCategory::Electronics,
        colour: 0x544141, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC, weight: 100,
        desc: "Insulated wire. Only conducts to PSCN, NSCN, WIFI, and SWCH.",
        air_loss: 0.9, loss: 0.0,
        meltable: 1, hardness: 1, heat_conduct: 251,
        high_temp: 1687.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- YEST.cpp (TPT 63) -------------------------------------------------
    v.push(elem!(
        id: PT_YEST, ident: "DEFAULT_PT_YEST", name: "YEST", cat: MenuCategory::Powders,
        colour: 0xEEE0C0, props: TYPE_PART, weight: 80,
        desc: "Yeast, grows when warm (~37C).",
        advection: 0.7, air_drag: 0.02, air_loss: 0.96, loss: 0.8, gravity: 0.1,
        falldown: 1,
        flammable: 15, hardness: 31, heat_conduct: 70,
        high_temp: 373.0, high_temp_transition: PT_DYST as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- DYST.cpp (TPT 64) -------------------------------------------------
    v.push(elem!(
        id: PT_DYST, ident: "DEFAULT_PT_DYST", name: "DYST", cat: MenuCategory::Powders,
        colour: 0xBBB0A0, props: TYPE_PART, weight: 80,
        desc: "Dead Yeast.",
        advection: 0.7, air_drag: 0.02, air_loss: 0.96, loss: 0.8, gravity: 0.1,
        falldown: 1,
        flammable: 20, hardness: 30, heat_conduct: 70,
        high_temp: 473.0, high_temp_transition: PT_DUST as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- GLOW.cpp (TPT 66) -------------------------------------------------
    // TPT emits its glow colour from `firea = 64` with an RGB computed per
    // particle, which the table encodes as "alpha only" (see `fire_colour`).
    v.push(elem!(
        id: PT_GLOW, ident: "DEFAULT_PT_GLOW", name: "GLOW", cat: MenuCategory::Liquids,
        colour: 0x445464, props: TYPE_LIQUID | PROP_PHOTPASS | PROP_LIFE_DEC, weight: 40,
        desc: "Glow, Glows under pressure.",
        advection: 0.3, air_drag: 0.02, air_loss: 0.98, loss: 0.8, gravity: 0.15,
        falldown: 2,
        hardness: 2, heat_conduct: 44,
        default_temp: 315.15,
        pixel_mode: PMODE_BLUR, fire_colour: 0x4000_0000,
    ));
    // --- PCLN.cpp (TPT 74) -------------------------------------------------
    v.push(elem!(
        id: PT_PCLN, ident: "DEFAULT_PT_PCLN", name: "PCLN", cat: MenuCategory::Powered,
        colour: 0x3B3B0A, props: TYPE_SOLID | PROP_PHOTPASS | PROP_NOCTYPEDRAW, weight: 100,
        desc: "Powered clone. When activated, duplicates any particles it touches.",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ctype_draw: CtypeDraw::Pcln,
    ));
    // --- HSWC.cpp (TPT 75) -------------------------------------------------
    v.push(elem!(
        id: PT_HSWC, ident: "DEFAULT_PT_HSWC", name: "HSWC", cat: MenuCategory::Powered,
        colour: 0x3B0A0A, props: TYPE_SOLID, weight: 100,
        desc: "Heat switch. Conducts heat only when activated.",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- IRON.cpp (TPT 76) -------------------------------------------------
    v.push(elem!(
        id: PT_IRON, ident: "DEFAULT_PT_IRON", name: "IRON", cat: MenuCategory::Solids,
        colour: 0x707070,
        props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC | PROP_HOT_GLOW, weight: 100,
        desc: "Rusts with salt, can be used for electrolysis of WATR.",
        air_loss: 0.9, loss: 0.0,
        meltable: 1, hardness: 49, heat_conduct: 251,
        high_temp: 1687.0, high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- MORT.cpp (TPT 77) -------------------------------------------------
    v.push(elem!(
        id: PT_MORT, ident: "DEFAULT_PT_MORT", name: "MORT", cat: MenuCategory::Special,
        colour: 0xE0E0E0, props: TYPE_GAS, weight: -1,
        desc: "Steam Train.",
        air_loss: 1.0, loss: 1.0, collision: -0.99, diffusion: 0.01, hot_air: 0.002,
        hardness: 0, heat_conduct: 60,
        default_temp: 299.15,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- DLAY.cpp (TPT 79) -------------------------------------------------
    v.push(elem!(
        id: PT_DLAY, ident: "DEFAULT_PT_DLAY", name: "DLAY", cat: MenuCategory::Powered,
        colour: 0x753590, props: TYPE_SOLID, weight: 100,
        desc: "Conducts with temperature-dependent delay. (use HEAT/COOL).",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 0,
        default_temp: 277.15,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- DRIC.cpp (TPT 81) -------------------------------------------------
    v.push(elem!(
        id: PT_DRIC, ident: "DEFAULT_PT_DRIC", name: "DRIC", cat: MenuCategory::Solids,
        colour: 0xE0E0E0, props: TYPE_SOLID, weight: 100,
        desc: "Dry Ice, formed when CO2 is cooled.",
        air_loss: 0.9, loss: 0.0,
        hot_air: -0.0005,
        hardness: 20, heat_conduct: 2,
        default_temp: 172.65,
        high_temp: 195.65, high_temp_transition: PT_CO2 as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- CBNW.cpp (TPT 82, displayed as BUBW) ------------------------------
    v.push(elem!(
        id: PT_CBNW, ident: "DEFAULT_PT_CBNW", name: "BUBW", cat: MenuCategory::Liquids,
        colour: 0x2030D0,
        props: TYPE_LIQUID | PROP_CONDUCTS | PROP_LIFE_DEC | PROP_NEUTPENETRATE, weight: 30,
        desc: "Carbonated water. Slowly releases CO2.",
        advection: 0.6, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1,
        falldown: 2,
        hardness: 20, heat_conduct: 29,
        default_temp: 293.15,
        low_temp: 273.15, low_temp_transition: PT_ICEI as i32,
        high_temp: 373.0, high_temp_transition: PT_WTRV as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- STOR.cpp (TPT 83) -------------------------------------------------
    v.push(elem!(
        id: PT_STOR, ident: "DEFAULT_PT_STOR", name: "STOR", cat: MenuCategory::Powered,
        colour: 0x50DFDF, props: TYPE_SOLID | PROP_NOCTYPEDRAW, weight: 100,
        desc: "Storage. Captures and stores a single particle. Releases when charged with PSCN, also passes to PIPE.",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 0,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ctype_draw: CtypeDraw::Stor,
    ));
    // --- PVOD.cpp (TPT 84) -------------------------------------------------
    v.push(elem!(
        id: PT_PVOD, ident: "DEFAULT_PT_PVOD", name: "PVOD", cat: MenuCategory::Powered,
        colour: 0x792020, props: TYPE_SOLID, weight: 100,
        desc: "Powered VOID. When activated, destroys entering particles.",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ctype_draw: CtypeDraw::Basic,
    ));
    // --- CONV.cpp (TPT 85) -------------------------------------------------
    v.push(elem!(
        id: PT_CONV, ident: "DEFAULT_PT_CONV", name: "CONV", cat: MenuCategory::Special,
        colour: 0x0AAB0A, props: TYPE_SOLID | PROP_NOCTYPEDRAW, weight: 100,
        desc: "Converter. Converts everything into whatever it first touches.",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ctype_draw: CtypeDraw::ValueInCtype,
    ));
    // --- TESC.cpp (TPT 88) -------------------------------------------------
    v.push(elem!(
        id: PT_TESC, ident: "DEFAULT_PT_TESC", name: "TESC", cat: MenuCategory::Electronics,
        colour: 0x707040,
        props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC | PROP_HOT_GLOW, weight: 100,
        desc: "Tesla coil! Creates lightning when sparked.",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- SPNG.cpp (TPT 90) -------------------------------------------------
    v.push(elem!(
        id: PT_SPNG, ident: "DEFAULT_PT_SPNG", name: "SPNG", cat: MenuCategory::Solids,
        colour: 0xFFBE30, props: TYPE_SOLID, weight: 100,
        desc: "Sponge, absorbs water. Is not a moving solid.",
        air_drag: 0.0, air_loss: 0.0, loss: 0.0,
        flammable: 20, hardness: 31, heat_conduct: 251,
        high_temp: 2730.0, high_temp_transition: PT_FIRE as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- RIME.cpp (TPT 91) -------------------------------------------------
    v.push(elem!(
        id: PT_RIME, ident: "DEFAULT_PT_RIME", name: "RIME", cat: MenuCategory::Solids,
        colour: 0xCCCCCC, props: TYPE_SOLID, weight: 100,
        desc: "Solid, created when steam cools rapidly and goes through deposition, skipping the liquid phase.",
        air_drag: 0.0, air_loss: 0.0, loss: 0.0,
        hardness: 32, heat_conduct: 100,
        default_temp: 243.15,
        // `ST`: melts back into water (or the liquid it froze from) at 273.15 K.
        high_temp: 273.15, high_temp_transition: ST,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- FOG.cpp (TPT 92) --------------------------------------------------
    v.push(elem!(
        id: PT_FOG, ident: "DEFAULT_PT_FOG", name: "FOG", cat: MenuCategory::Gases,
        colour: 0xAAAAAA, props: TYPE_GAS | PROP_LIFE_DEC, weight: 1,
        desc: "Fog, created when an electric current is passed through RIME.",
        advection: 0.8, air_loss: 0.4, loss: 0.7, collision: -0.1, diffusion: 0.99,
        falldown: 0,
        hardness: 30, heat_conduct: 100,
        default_temp: 243.15,
        high_temp: 373.15, high_temp_transition: PT_WTRV as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- PUMP.cpp (TPT 97) -------------------------------------------------
    v.push(elem!(
        id: PT_PUMP, ident: "DEFAULT_PT_PUMP", name: "PUMP", cat: MenuCategory::Powered,
        colour: 0x0A0A3B, props: TYPE_SOLID, weight: 100,
        desc: "Pressure pump. Changes pressure to its temp when activated. (use HEAT/COOL).",
        air_loss: 0.95, loss: 0.0,
        hardness: 9, heat_conduct: 0,
        default_temp: 273.15, default_life: 10,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- PIPE.cpp (TPT 99) -------------------------------------------------
    v.push(elem!(
        id: PT_PIPE, ident: "DEFAULT_PT_PIPE", name: "PIPE", cat: MenuCategory::Force,
        colour: 0x444444, props: TYPE_SOLID | PROP_LIFE_DEC, weight: 100,
        desc: "PIPE, moves particles around. Once the BRCK generates, erase some for the exit. Then the PIPE generates and is usable.",
        air_loss: 0.95, loss: 0.0,
        hardness: 0, heat_conduct: 251,
        default_temp: 295.15, default_life: 60,
        high_pressure: 10.0, high_pressure_transition: PT_BRMT as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- FRZZ.cpp (TPT 100) ------------------------------------------------
    v.push(elem!(
        id: PT_FRZZ, ident: "DEFAULT_PT_FRZZ", name: "FRZZ", cat: MenuCategory::Powders,
        colour: 0xC0E0FF, props: TYPE_PART, weight: 50,
        desc: "Freeze powder. When melted, forms ice that always cools. Spreads with regular water.",
        advection: 0.7, air_drag: 0.01, air_loss: 0.96, loss: 0.9, collision: -0.1,
        gravity: 0.05, diffusion: 0.01, hot_air: -0.00005, falldown: 1,
        hardness: 20, heat_conduct: 46,
        default_temp: 253.15,
        high_pressure: 1.8, high_pressure_transition: PT_SNOW as i32,
        low_temp: 50.0, low_temp_transition: PT_ICEI as i32,
        high_temp: 273.15, high_temp_transition: PT_FRZW as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- FRZW.cpp (TPT 101) ------------------------------------------------
    v.push(elem!(
        id: PT_FRZW, ident: "DEFAULT_PT_FRZW", name: "FRZW", cat: MenuCategory::Liquids,
        colour: 0x1020C0, props: TYPE_LIQUID | PROP_LIFE_DEC, weight: 30,
        desc: "Freeze water. Hybrid liquid formed when Freeze powder melts.",
        advection: 0.6, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1,
        falldown: 2,
        hardness: 20, heat_conduct: 29,
        default_temp: 120.0, default_life: 100,
        low_temp: 53.0, low_temp_transition: PT_ICEI as i32,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
    // --- BIZR.cpp (TPT 103) ------------------------------------------------
    v.push(elem!(
        id: PT_BIZR, ident: "DEFAULT_PT_BIZR", name: "BIZR", cat: MenuCategory::Liquids,
        colour: 0x00FF77, props: TYPE_LIQUID, weight: 30,
        desc: "Bizarre... contradicts the normal state changes. Paints other elements with its deco color.",
        advection: 0.6, air_drag: 0.01, air_loss: 0.98, loss: 0.95, gravity: 0.1,
        falldown: 2,
        hardness: 20, heat_conduct: 29,
        default_ctype: 0x47FFFF,
        low_temp: 100.0, low_temp_transition: PT_BIZRG as i32,
        high_temp: 400.0, high_temp_transition: PT_BIZRS as i32,
        pixel_mode: PMODE_BLUR, fire_colour: 0,
    ));
    // --- BIZRG.cpp (TPT 104, displayed as BIZG) ----------------------------
    v.push(elem!(
        id: PT_BIZRG, ident: "DEFAULT_PT_BIZRG", name: "BIZG", cat: MenuCategory::Gases,
        colour: 0x00FFBB, props: TYPE_GAS, weight: 1,
        desc: "Bizarre gas.",
        advection: 1.0, air_drag: 0.01, air_loss: 0.99, loss: 0.3, collision: -0.1,
        diffusion: 2.75, falldown: 0,
        hardness: 1, heat_conduct: 42,
        default_temp: 95.15, default_ctype: 0x47FFFF,
        high_temp: 100.0, high_temp_transition: PT_BIZR as i32,
        pixel_mode: PMODE_BLUR, fire_colour: 0,
    ));
    // --- BIZRS.cpp (TPT 105, displayed as BIZS) ----------------------------
    v.push(elem!(
        id: PT_BIZRS, ident: "DEFAULT_PT_BIZRS", name: "BIZS", cat: MenuCategory::Solids,
        colour: 0x00E455, props: TYPE_SOLID, weight: 100,
        desc: "Bizarre solid.",
        air_loss: 0.9, loss: 0.0,
        hardness: 1, heat_conduct: 251,
        default_temp: 595.15, default_ctype: 0x47FFFF,
        low_temp: 400.0, low_temp_transition: PT_BIZR as i32,
        pixel_mode: PMODE_BLUR, fire_colour: 0,
    ));
    // --- TSNS.cpp (TPT 164, requested as TCSN) -----------------------------
    v.push(elem!(
        id: PT_TSNS, ident: "DEFAULT_PT_TSNS", name: "TSNS", cat: MenuCategory::Sensors,
        colour: 0xFD00D5, props: TYPE_SOLID, weight: 100,
        desc: "Temperature sensor, creates a spark when there's a nearby particle with a greater temperature.",
        air_loss: 0.96, loss: 0.0,
        hardness: 1, heat_conduct: 0,
        default_tmp2: 2,
        pixel_mode: PMODE_FLAT, fire_colour: 0,
    ));
}
