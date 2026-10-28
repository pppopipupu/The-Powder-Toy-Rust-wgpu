//! Batch 2c of the remaining Powder Toy elements (`PT_FIGH ..= PT_SEED`).

use super::super::{
    CtypeDraw, ElementDef, FIRE_ADD, FIRE_BLEND, MenuCategory, NO_DECO, PMODE_BLUR, PMODE_FLARE,
    PMODE_FLAT,
};
use crate::constants::*;

/// Appends the batch 2c definitions in The Powder Toy's element order.
pub(crate) fn push(v: &mut Vec<ElementDef>) {
    // --- FIGH.cpp (TPT 158) --------------------------------------------------------
    v.push(elem!(
        id: PT_FIGH, ident: "DEFAULT_PT_FIGH", name: "FIGH", cat: MenuCategory::Special,
        colour: 0xFFE0A0, props: PROP_NOCTYPEDRAW,
        weight: 50,
        desc: "Fighter. Tries to kill stickmen. You must first give it an element to kill him with.",
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
    // --- FRAY.cpp (TPT 159) --------------------------------------------------------
    v.push(elem!(
        id: PT_FRAY, ident: "DEFAULT_PT_FRAY", name: "FRAY", cat: MenuCategory::Force,
        colour: 0x00BBFF, props: TYPE_SOLID,
        weight: 100,
        desc: "Force Emitter. Pushes or pulls objects based on its temperature. Use like ARAY.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_temp: 293.15,
        pixel_mode: PMODE_FLAT,
    ));
    // --- RPEL.cpp (TPT 160) --------------------------------------------------------
    v.push(elem!(
        id: PT_RPEL, ident: "DEFAULT_PT_RPEL", name: "RPEL", cat: MenuCategory::Force,
        colour: 0x99CC00, props: TYPE_SOLID,
        weight: 100,
        desc: "Repels or attracts particles based on its temperature.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_temp: 293.15,
        pixel_mode: PMODE_FLAT,
    ctype_draw: CtypeDraw::Basic,
    ));
    // --- PPIP.cpp (TPT 161) --------------------------------------------------------
    v.push(elem!(
        id: PT_PPIP, ident: "DEFAULT_PT_PPIP", name: "PPIP", cat: MenuCategory::Powered,
        colour: 0x444466, props: TYPE_SOLID | PROP_LIFE_DEC,
        weight: 100,
        desc: "Powered version of PIPE, use PSCN/NSCN to Activate/Deactivate.",
        air_loss: 0.95,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 251,
        default_life: 60,
        pixel_mode: PMODE_FLAT,
    ));
    // --- DTEC.cpp (TPT 162) --------------------------------------------------------
    v.push(elem!(
        id: PT_DTEC, ident: "DEFAULT_PT_DTEC", name: "DTEC", cat: MenuCategory::Sensors,
        colour: 0xFD9D18, props: TYPE_SOLID,
        weight: 100,
        desc: "Detector, creates a spark when something with its ctype is nearby.",
        air_loss: 0.96,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_tmp2: 2,
        pixel_mode: PMODE_FLAT,
    ctype_draw: CtypeDraw::ValueInTmp,
    ));
    // --- DMG.cpp (TPT 163) ---------------------------------------------------------
    v.push(elem!(
        id: PT_DMG, ident: "DEFAULT_PT_DMG", name: "DMG", cat: MenuCategory::Force,
        colour: 0x88FF88, props: TYPE_PART | PROP_SPARKSETTLE,
        weight: 30,
        desc: "Generates damaging pressure and breaks any elements it hits.",
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
    // --- VIBR.cpp (TPT 165) --------------------------------------------------------
    v.push(elem!(
        id: PT_VIBR, ident: "DEFAULT_PT_VIBR", name: "VIBR", cat: MenuCategory::Radioactive,
        colour: 0x005000, props: TYPE_SOLID | PROP_LIFE_DEC,
        weight: 100,
        desc: "Vibranium. Stores energy and releases it in violent explosions.",
        air_loss: 0.85,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 251,
        default_temp: 273.15,
        pixel_mode: FIRE_BLEND,
    ));
    // --- BVBR.cpp (TPT 166) --------------------------------------------------------
    v.push(elem!(
        id: PT_BVBR, ident: "DEFAULT_PT_BVBR", name: "BVBR", cat: MenuCategory::Radioactive,
        colour: 0x005000, props: TYPE_PART | PROP_LIFE_DEC,
        weight: 67,
        desc: "Broken vibranium.",
        advection: 0.3,
        air_drag: 0.02,
        air_loss: 0.95,
        loss: 0.8,
        gravity: 0.15,
        falldown: 1,
        hardness: 0,
        heat_conduct: 164,
        default_temp: 273.15,
        pixel_mode: FIRE_BLEND,
    ));
    // --- CRAY.cpp (TPT 167) --------------------------------------------------------
    v.push(elem!(
        id: PT_CRAY, ident: "DEFAULT_PT_CRAY", name: "CRAY", cat: MenuCategory::Electronics,
        colour: 0xBBFF00, props: TYPE_SOLID,
        weight: 100,
        desc: "Particle Ray Emitter. Creates a beam of particles set by its ctype, with a range set by tmp.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
    ctype_draw: CtypeDraw::Cray,
    ));
    // --- PSTN.cpp (TPT 168) --------------------------------------------------------
    v.push(elem!(
        id: PT_PSTN, ident: "DEFAULT_PT_PSTN", name: "PSTN", cat: MenuCategory::Force,
        colour: 0xAA9999, props: TYPE_SOLID,
        weight: 100,
        desc: "Piston, pushes particles. PSCN extends, NSCN retracts",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 0,
        default_temp: 283.15,
        pixel_mode: PMODE_FLAT,
    ctype_draw: CtypeDraw::Pstn,
    ));
    // --- FRME.cpp (TPT 169) --------------------------------------------------------
    v.push(elem!(
        id: PT_FRME, ident: "DEFAULT_PT_FRME", name: "FRME", cat: MenuCategory::Force,
        colour: 0x999988, props: TYPE_SOLID,
        weight: 100,
        desc: "Frame, can be used with pistons to push many particles.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
    ));
    // --- GOLD.cpp (TPT 170) --------------------------------------------------------
    v.push(elem!(
        id: PT_GOLD, ident: "DEFAULT_PT_GOLD", name: "GOLD", cat: MenuCategory::Solids,
        photon_reflect: 0x3C038100,
        colour: 0xDCAD2C, props: TYPE_SOLID | PROP_CONDUCTS | PROP_HOT_GLOW | PROP_LIFE_DEC | PROP_NEUTPASS,
        weight: 100,
        desc: "Corrosion resistant metal, will reverse corrosion of iron. Excellent conductor.",
        air_loss: 0.9,
        loss: 0.0,
        meltable: 1,
        hardness: 0,
        heat_conduct: 251,
        pixel_mode: PMODE_FLAT,
        high_temp: 1337.0,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- TUNG.cpp (TPT 171) --------------------------------------------------------
    v.push(elem!(
        id: PT_TUNG, ident: "DEFAULT_PT_TUNG", name: "TUNG", cat: MenuCategory::Electronics,
        colour: 0x505050, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC,
        weight: 100,
        desc: "Tungsten. Brittle metal with a very high melting point.",
        air_loss: 0.9,
        loss: 0.0,
        meltable: 1,
        hardness: 1,
        heat_conduct: 251,
        pixel_mode: PMODE_FLAT | FIRE_ADD,
        fire_colour: 0x1E000000,
        high_temp: 3695.0,
    ));
    // --- PSNS.cpp (TPT 172) --------------------------------------------------------
    v.push(elem!(
        id: PT_PSNS, ident: "DEFAULT_PT_PSNS", name: "PSNS", cat: MenuCategory::Sensors,
        colour: 0xDB2020, props: TYPE_SOLID,
        weight: 100,
        desc: "Pressure sensor, creates a spark when the pressure is greater than its temperature.",
        air_loss: 0.96,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_temp: 277.15,
        pixel_mode: PMODE_FLAT,
    ));
    // --- PROT.cpp (TPT 173) --------------------------------------------------------
    v.push(elem!(
        id: PT_PROT, ident: "DEFAULT_PT_PROT", name: "PROT", cat: MenuCategory::Radioactive,
        colour: 0x990000, props: TYPE_ENERGY,
        weight: -1,
        desc: "Protons. Transfer heat to materials, and remove sparks.",
        air_loss: 1.0,
        loss: 1.0,
        collision: -0.99,
        hardness: 0,
        heat_conduct: 61,
        default_life: 75,
        pixel_mode: PMODE_FLAT | FIRE_BLEND,
        fire_colour: 0x07FAAAFF,
    ));
    // --- VIRS.cpp (TPT 174) --------------------------------------------------------
    v.push(elem!(
        id: PT_VIRS, ident: "DEFAULT_PT_VIRS", name: "VIRS", cat: MenuCategory::Liquids,
        colour: 0xFE11F6, props: TYPE_LIQUID | PROP_DEADLY,
        weight: 31,
        desc: "Virus. Turns everything it touches into virus.",
        advection: 0.6,
        air_drag: 0.01,
        air_loss: 0.98,
        loss: 0.95,
        gravity: 0.1,
        falldown: 2,
        hardness: 20,
        heat_conduct: 251,
        default_temp: 345.15,
        default_tmp4: 250,
        pixel_mode: PMODE_FLAT | PMODE_BLUR | NO_DECO,
        low_temp: 305.0,
        high_temp: 673.0,
        low_temp_transition: PT_VRSS as i32,
        high_temp_transition: PT_VRSG as i32,
    ));
    // --- VRSS.cpp (TPT 175) --------------------------------------------------------
    v.push(elem!(
        id: PT_VRSS, ident: "DEFAULT_PT_VRSS", name: "VRSS", cat: MenuCategory::Solids,
        colour: 0xD408CD, props: TYPE_SOLID | PROP_DEADLY,
        weight: 100,
        desc: "Solid Virus. Turns everything it touches into virus.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 251,
        default_tmp4: 250,
        pixel_mode: PMODE_FLAT | NO_DECO,
        high_temp: 305.0,
        high_temp_transition: PT_VIRS as i32,
    ));
    // --- VRSG.cpp (TPT 176) --------------------------------------------------------
    v.push(elem!(
        id: PT_VRSG, ident: "DEFAULT_PT_VRSG", name: "VRSG", cat: MenuCategory::Gases,
        colour: 0xFE68FE, props: TYPE_GAS | PROP_DEADLY,
        weight: 1,
        desc: "Gas Virus. Turns everything it touches into virus.",
        advection: 1.0,
        air_drag: 0.01,
        air_loss: 0.99,
        loss: 0.3,
        collision: -0.1,
        diffusion: 0.75,
        flammable: 500,
        hardness: 0,
        heat_conduct: 251,
        default_temp: 795.15,
        default_tmp4: 250,
        pixel_mode: FIRE_BLEND | NO_DECO,
        fire_colour: 0x7D000000,
        low_temp: 673.0,
        low_temp_transition: PT_VIRS as i32,
    ));
    // --- DRAY.cpp (TPT 178) --------------------------------------------------------
    v.push(elem!(
        id: PT_DRAY, ident: "DEFAULT_PT_DRAY", name: "DRAY", cat: MenuCategory::Electronics,
        colour: 0xFFAA22, props: TYPE_SOLID,
        weight: 100,
        desc: "Duplicator ray. Replicates a line of particles in front of it.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
    ctype_draw: CtypeDraw::ValueInCtype,
    ));
    // --- CRMC.cpp (TPT 179) --------------------------------------------------------
    v.push(elem!(
        id: PT_CRMC, ident: "DEFAULT_PT_CRMC", name: "CRMC", cat: MenuCategory::Solids,
        colour: 0xD6D1D4, props: TYPE_SOLID | PROP_NEUTPASS,
        weight: 100,
        desc: "Ceramic. Gets stronger under pressure.",
        loss: 0.0,
        air_loss: 0.0,
        hardness: 0,
        heat_conduct: 35,
        pixel_mode: PMODE_FLAT,
        high_temp: 2887.15,
        high_temp_transition: ST,
    ));
    // --- HEAC.cpp (TPT 180) --------------------------------------------------------
    v.push(elem!(
        id: PT_HEAC, ident: "DEFAULT_PT_HEAC", name: "HEAC", cat: MenuCategory::Solids,
        colour: 0xCB6351, props: TYPE_SOLID,
        weight: 100,
        desc: "Rapid heat conductor.",
        air_loss: 0.9,
        loss: 0.0,
        meltable: 1,
        hardness: 0,
        heat_conduct: 251,
        pixel_mode: PMODE_FLAT,
        high_temp: 1887.15,
    ));
    // --- SAWD.cpp (TPT 181) --------------------------------------------------------
    v.push(elem!(
        id: PT_SAWD, ident: "DEFAULT_PT_SAWD", name: "SAWD", cat: MenuCategory::Powders,
        colour: 0xF0F0A0, props: TYPE_PART | PROP_NEUTPASS,
        weight: 18,
        desc: "Sawdust. Floats on water.",
        advection: 0.7,
        air_drag: 0.02,
        air_loss: 0.96,
        loss: 0.8,
        gravity: 0.1,
        falldown: 1,
        flammable: 10,
        hardness: 29,
        heat_conduct: 70,
        pixel_mode: PMODE_FLAT,
    ));
    // --- POLO.cpp (TPT 182) --------------------------------------------------------
    v.push(elem!(
        id: PT_POLO, ident: "DEFAULT_PT_POLO", name: "POLO", cat: MenuCategory::Radioactive,
        photon_reflect: 0x000FF200,
        colour: 0x506030, props: TYPE_PART | PROP_NEUTPASS | PROP_RADIOACTIVE | PROP_LIFE_DEC | PROP_DEADLY,
        weight: 90,
        desc: "Polonium, highly radioactive. Decays into NEUT and heats up.",
        advection: 0.4,
        air_drag: 0.01,
        air_loss: 0.99,
        loss: 0.95,
        gravity: 0.4,
        falldown: 1,
        meltable: 1,
        hardness: 0,
        heat_conduct: 251,
        default_temp: 388.15,
        pixel_mode: PMODE_FLAT,
        high_temp: 526.95,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- RFRG.cpp (TPT 183) --------------------------------------------------------
    v.push(elem!(
        id: PT_RFRG, ident: "DEFAULT_PT_RFRG", name: "RFRG", cat: MenuCategory::Gases,
        colour: 0x72D2D4, props: TYPE_GAS | PROP_DEADLY,
        weight: 1,
        desc: "Refrigerant. Heats up and liquefies under pressure.",
        advection: 1.2,
        air_loss: 0.99,
        loss: 0.3,
        collision: -0.1,
        diffusion: 1.3,
        hot_air: 0.0001,
        hardness: 21,
        heat_conduct: 3,
        pixel_mode: PMODE_FLAT,
        high_pressure: 2.0,
        high_pressure_transition: PT_RFGL as i32,
    ));
    // --- RFGL.cpp (TPT 184) --------------------------------------------------------
    v.push(elem!(
        id: PT_RFGL, ident: "DEFAULT_PT_RFGL", name: "RFGL", cat: MenuCategory::Liquids,
        colour: 0x84C2CF, props: TYPE_LIQUID | PROP_DEADLY,
        weight: 10,
        desc: "Liquid refrigerant.",
        advection: 0.6,
        air_drag: 0.01,
        air_loss: 0.98,
        loss: 0.95,
        gravity: 0.1,
        falldown: 2,
        hardness: 21,
        heat_conduct: 3,
        pixel_mode: PMODE_FLAT,
        low_pressure: 2.0,
        low_pressure_transition: PT_RFRG as i32,
    ));
    // --- LSNS.cpp (TPT 185) --------------------------------------------------------
    v.push(elem!(
        id: PT_LSNS, ident: "DEFAULT_PT_LSNS", name: "LSNS", cat: MenuCategory::Sensors,
        colour: 0x336699, props: TYPE_SOLID,
        weight: 100,
        desc: "Life sensor, creates a spark when there's a nearby particle with a life higher than its temperature.",
        air_loss: 0.96,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_temp: 277.15,
        default_tmp2: 2,
        pixel_mode: PMODE_FLAT,
    ));
    // --- LDTC.cpp (TPT 186) --------------------------------------------------------
    v.push(elem!(
        id: PT_LDTC, ident: "DEFAULT_PT_LDTC", name: "LDTC", cat: MenuCategory::Sensors,
        colour: 0x66FF66, props: TYPE_SOLID | PROP_NOCTYPEDRAW,
        weight: 100,
        desc: "Linear detector. Scans in 8 directions for particles with its ctype and creates a spark on the opposite side.",
        air_loss: 0.96,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT,
    ctype_draw: CtypeDraw::ValueInCtype,
    ));
    // --- SLCN.cpp (TPT 187) --------------------------------------------------------
    v.push(elem!(
        id: PT_SLCN, ident: "DEFAULT_PT_SLCN", name: "SLCN", cat: MenuCategory::Powders,
        colour: 0xBCCDDF, props: TYPE_PART | PROP_CONDUCTS | PROP_HOT_GLOW | PROP_LIFE_DEC,
        weight: 90,
        desc: "Powdered Silicon. A key ingredient in producing multiple materials.",
        advection: 0.4,
        air_drag: 0.04,
        air_loss: 0.94,
        loss: 0.95,
        collision: -0.1,
        gravity: 0.27,
        falldown: 1,
        hardness: 0,
        heat_conduct: 100,
        pixel_mode: PMODE_FLAT,
        high_temp: 3538.15,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- PTNM.cpp (TPT 188) --------------------------------------------------------
    v.push(elem!(
        id: PT_PTNM, ident: "DEFAULT_PT_PTNM", name: "PTNM", cat: MenuCategory::Solids,
        colour: 0xD5E0EB, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC | PROP_HOT_GLOW | PROP_SPARKSETTLE,
        weight: 100,
        desc: "Platinum. Catalyzes certain reactions.",
        air_loss: 0.9,
        loss: 0.0,
        meltable: 1,
        hardness: 0,
        heat_conduct: 251,
        pixel_mode: PMODE_FLAT,
        high_temp: 2041.15,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- VSNS.cpp (TPT 189) --------------------------------------------------------
    v.push(elem!(
        id: PT_VSNS, ident: "DEFAULT_PT_VSNS", name: "VSNS", cat: MenuCategory::Sensors,
        colour: 0x7C9C00, props: TYPE_SOLID,
        weight: 100,
        desc: "Velocity sensor, creates a spark when there's a nearby particle with velocity higher than its temperature.",
        air_loss: 0.96,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_temp: 277.15,
        default_tmp2: 2,
        pixel_mode: PMODE_FLAT,
    ));
    // --- ROCK.cpp (TPT 190) --------------------------------------------------------
    v.push(elem!(
        id: PT_ROCK, ident: "DEFAULT_PT_ROCK", name: "ROCK", cat: MenuCategory::Solids,
        colour: 0x727272, props: TYPE_SOLID | PROP_HOT_GLOW,
        weight: 100,
        desc: "Solid, melts into various elements.",
        air_loss: 0.94,
        loss: 0.0,
        meltable: 5,
        hardness: 1,
        heat_conduct: 200,
        pixel_mode: PMODE_FLAT,
        high_pressure: 120.0,
        high_temp: 1943.15,
        high_pressure_transition: PT_STNE as i32,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- LITH.cpp (TPT 191) --------------------------------------------------------
    v.push(elem!(
        id: PT_LITH, ident: "DEFAULT_PT_LITH", name: "LITH", cat: MenuCategory::Explosives,
        colour: 0xB6AABF, props: TYPE_PART | PROP_LIFE_DEC,
        weight: 17,
        desc: "Lithium. Reactive element that explodes on contact with water.",
        advection: 0.2,
        air_drag: 0.01,
        air_loss: 0.96,
        loss: 0.95,
        collision: -0.1,
        gravity: 0.2,
        falldown: 1,
        hardness: 14,
        heat_conduct: 70,
        pixel_mode: PMODE_FLAT,
        high_temp: 453.65,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- RSST.cpp (TPT 192) --------------------------------------------------------
    v.push(elem!(
        id: PT_RSST, ident: "DEFAULT_PT_RSST", name: "RSST", cat: MenuCategory::Liquids,
        colour: 0xF95B49, props: TYPE_LIQUID | PROP_CONDUCTS | PROP_LIFE_DEC | PROP_NEUTPASS | PROP_PHOTPASS,
        weight: 33,
        desc: "Resist. Solidifies on contact with photons, is destroyed by electrons and spark.",
        advection: 0.3,
        air_drag: 0.02,
        air_loss: 0.98,
        loss: 0.8,
        gravity: 0.15,
        falldown: 2,
        hardness: 50,
        heat_conduct: 55,
        default_temp: 315.15,
        pixel_mode: PMODE_FLAT,
    ));
    // --- RSSS.cpp (TPT 193) --------------------------------------------------------
    v.push(elem!(
        id: PT_RSSS, ident: "DEFAULT_PT_RSSS", name: "RSSS", cat: MenuCategory::Solids,
        colour: 0xC43626, props: TYPE_SOLID | PROP_NEUTPASS,
        weight: 100,
        desc: "Solidified resist. Blocks pressure and insulates electricity. Liquefies on contact with neutrons.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 130,
        pixel_mode: PMODE_FLAT,
    ));
    // --- BASE.cpp (TPT 194) --------------------------------------------------------
    v.push(elem!(
        id: PT_BASE, ident: "DEFAULT_PT_BASE", name: "BASE", cat: MenuCategory::Liquids,
        colour: 0x90D5FF, props: TYPE_LIQUID | PROP_DEADLY,
        weight: 16,
        desc: "Corrosive liquid. Rusts conductive solids, neutralizes acid.",
        advection: 0.5,
        air_drag: 0.01,
        air_loss: 0.97,
        loss: 0.96,
        gravity: 0.08,
        falldown: 2,
        hardness: 0,
        heat_conduct: 31,
        default_life: 76,
        pixel_mode: PMODE_FLAT | PMODE_BLUR,
    ));
    // --- SEED.cpp (TPT 195) --------------------------------------------------------
    v.push(elem!(
        id: PT_SEED, ident: "DEFAULT_PT_SEED", name: "SEED", cat: MenuCategory::Powders,
        photon_reflect: 0x00000E00,
        colour: 0x88E788, props: TYPE_PART | PROP_NEUTPASS,
        weight: 36,
        desc: "Seeds. Put on sand and add water to grow a tree.",
        advection: 0.95,
        air_drag: 0.005,
        air_loss: 0.98,
        loss: 0.8,
        collision: -0.01,
        gravity: 0.05,
        falldown: 1,
        hardness: 19,
        heat_conduct: 32,
        pixel_mode: PMODE_FLAT,
        high_temp: 673.15,
        high_temp_transition: PT_FIRE as i32,
    ));
}
