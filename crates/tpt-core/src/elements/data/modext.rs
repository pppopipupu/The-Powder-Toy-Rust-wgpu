//! The mod's own elements (`PT_FNTC ..= PT_MGNT`).
//!
//! Transcribed from the element constructors in the mod's
//! `src/simulation/elements/*.cpp` (the files that are not part of upstream
//! The Powder Toy). The rows keep the mod's element order, so slot `i` always
//! holds element `i` (`PT_FNTC == 195`).
//!
//! `AirDrag` / `HotAir` are copied with `CFDS == 1`, like the rest of the
//! table. Elements whose C++ class assigns `Graphics` keep their static
//! `pixel_mode` bits here; the per-particle half lives in
//! `render_kernel/particles/pixel_mode.rs`.

use super::super::{
    CtypeDraw, ElementDef, FIRE_ADD, FIRE_BLEND, MenuCategory, NO_DECO, PMODE_FLARE, PMODE_FLAT,
};
use crate::constants::*;

/// Appends the mod's definitions in the order its `meson.build` lists them.
pub(crate) fn push(v: &mut Vec<ElementDef>) {
    // --- FNTC.cpp (mod id 195) -----------------------------------------------------
    v.push(elem!(
        id: PT_FNTC, ident: "DEFAULT_PT_FNTC", name: "FNTC", cat: MenuCategory::Electronics,
        colour: 0x505040, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC,
        weight: 100,
        desc: "Faster NTCT.",
        air_loss: 0.9,
        loss: 0.0,
        meltable: 1,
        hardness: 1,
        heat_conduct: 251,
        high_temp: 1687.0,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- CLNT.cpp (mod id 196) -----------------------------------------------------
    v.push(elem!(
        id: PT_CLNT, ident: "DEFAULT_PT_CLNT", name: "CLNT", cat: MenuCategory::Liquids,
        colour: 0x00FF77, props: TYPE_LIQUID | PROP_HOT_GLOW,
        weight: 100,
        desc: "Coolant for reactors and Engines. Use .tmp to set cooling.",
        advection: 0.3,
        air_drag: 0.02,
        air_loss: 0.98,
        loss: 0.8,
        gravity: 0.15,
        falldown: 2,
        hardness: 1,
        heat_conduct: 251,
        high_temp: 8273.0,
        high_temp_transition: PT_WATR as i32,
        pixel_mode: PMODE_FLAT | PMODE_FLARE,
    ));
    // --- LED.cpp (mod id 197) ------------------------------------------------------
    v.push(elem!(
        id: PT_LED, ident: "DEFAULT_PT_LED", name: "LED", cat: MenuCategory::Powered,
        colour: 0x404040, props: TYPE_SOLID,
        weight: 100,
        desc: "Light emitting diode, use decorations to set the glow colour. Activates with PSCN.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_temp: 308.15,
        pixel_mode: PMODE_FLAT | NO_DECO,
    ));
    // --- TIMC.cpp (mod id 198) -----------------------------------------------------
    v.push(elem!(
        id: PT_TIMC, ident: "DEFAULT_PT_TIMC", name: "TIMC", cat: MenuCategory::Powered,
        colour: 0x336699, props: TYPE_PART,
        weight: 100,
        desc: "Time crystal, converts into its ctype when sparked with PSCN.",
        advection: 0.4,
        air_drag: 0.04,
        air_loss: 0.94,
        loss: 0.95,
        collision: -0.1,
        gravity: 0.3,
        falldown: 1,
        hardness: 0,
        heat_conduct: 0,
        default_ctype: PT_CHLR,
        default_tmp: 100,
        pixel_mode: PMODE_FLAT | PMODE_FLARE,
    ));
    // --- FUEL.cpp (mod id 199) -----------------------------------------------------
    v.push(elem!(
        id: PT_FUEL, ident: "DEFAULT_PT_FUEL", name: "FUEL", cat: MenuCategory::Liquids,
        colour: 0xDB2020, props: TYPE_LIQUID,
        weight: 20,
        desc: "Extremely powerful fuel.",
        air_drag: 0.01,
        air_loss: 0.98,
        loss: 0.95,
        gravity: 0.1,
        hot_air: 0.0001,
        falldown: 2,
        flammable: 100,
        explosive: 1,
        hardness: 5,
        heat_conduct: 255,
        high_pressure: 14.0,
        high_pressure_transition: PT_PLSM as i32,
    ));
    // --- FPTC.cpp (mod id 200) -----------------------------------------------------
    v.push(elem!(
        id: PT_FPTC, ident: "DEFAULT_PT_FPTC", name: "FPTC", cat: MenuCategory::Electronics,
        colour: 0x405050, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC,
        weight: 100,
        desc: "Faster PTCT.",
        air_loss: 0.9,
        loss: 0.0,
        meltable: 1,
        hardness: 1,
        heat_conduct: 251,
        high_temp: 1687.0,
        high_temp_transition: PT_LAVA as i32,
    ));
    // --- DMRN.cpp (mod id 201) -----------------------------------------------------
    v.push(elem!(
        id: PT_DMRN, ident: "DEFAULT_PT_DMRN", name: "DMRN", cat: MenuCategory::Solids,
        colour: 0x1EA2B3, props: TYPE_SOLID,
        weight: 100,
        desc: "Demron, radiation-shielding fabric and indestructible insulator.",
        air_loss: 0.95,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 0,
    ));
    // --- PINV.cpp (mod id 202) -----------------------------------------------------
    v.push(elem!(
        id: PT_PINV, ident: "DEFAULT_PT_PINV", name: "PINV", cat: MenuCategory::Powered,
        colour: 0x00CCCC, props: TYPE_SOLID | PROP_NEUTPASS,
        weight: 100,
        desc: "Powered invisible, allows particles to pass when activated.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 15,
        heat_conduct: 164,
        pixel_mode: PMODE_FLAT,
    ));
    // --- WALL.cpp (mod id 203) -----------------------------------------------------
    v.push(elem!(
        id: PT_WALL, ident: "DEFAULT_PT_WALL", name: "WALL", cat: MenuCategory::Special,
        colour: 0x808080, props: TYPE_SOLID,
        weight: 102,
        desc: "Elemental wall (1x1), indestructible, immune to VIRS. Blocks pressure like TTAN.",
        air_loss: 0.0,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 0,
        default_temp: 293.15,
    ));
    // --- CHLR.cpp (mod id 204) -----------------------------------------------------
    v.push(elem!(
        id: PT_CHLR, ident: "DEFAULT_PT_CHLR", name: "Cl", cat: MenuCategory::Gases,
        colour: 0x8BC34A, props: TYPE_GAS | PROP_NEUTPASS,
        weight: 27,
        desc: "Chlorine, photochemical rxn with H2. WATR-> DSTW below 50C (ACID > 50C), rusts IRON/BMTL. Decays organic matter.",
        advection: 1.0,
        air_drag: 0.01,
        air_loss: 0.99,
        loss: 0.3,
        collision: -0.1,
        gravity: 0.2,
        diffusion: 0.4,
        hot_air: 0.001,
        hardness: 0,
        heat_conduct: 42,
        pixel_mode: FIRE_BLEND,
    ));
    // --- ACTY.cpp (mod id 205) -----------------------------------------------------
    v.push(elem!(
        id: PT_ACTY, ident: "DEFAULT_PT_ACTY", name: "ACTY", cat: MenuCategory::Gases,
        colour: 0xADD8E6, props: TYPE_GAS,
        weight: 30,
        desc: "Acetylene gas, clean gas that reaches temp. of 1100C when ignited and around 3500C with O2. Makes LRBD with Chlorine.",
        advection: 1.0,
        air_drag: 0.01,
        air_loss: 0.99,
        loss: 0.3,
        collision: -0.1,
        gravity: -0.5,
        diffusion: 0.4,
        hot_air: 0.001,
        hardness: 0,
        heat_conduct: 255,
        default_tmp2: 40,
        high_temp: 4073.15,
        high_temp_transition: PT_PLSM as i32,
        pixel_mode: FIRE_BLEND,
    ));
    // --- COND.cpp (mod id 206) -----------------------------------------------------
    v.push(elem!(
        id: PT_COND, ident: "DEFAULT_PT_COND", name: "THMO", cat: MenuCategory::Force,
        colour: 0x005000, props: TYPE_SOLID,
        weight: 100,
        desc: "Thermostat, sets nearby temp as per it's temp.",
        air_loss: 0.95,
        loss: 0.0,
        hardness: 10,
        heat_conduct: 0,
    ));
    // --- QGPP.cpp (mod id 207) -----------------------------------------------------
    v.push(elem!(
        id: PT_QGPP, ident: "DEFAULT_PT_QGPP", name: "QGP", cat: MenuCategory::Gases,
        colour: 0x9400D3, props: TYPE_GAS | PROP_NEUTPASS,
        weight: 1,
        desc: "Quark Gluon Plasma, a strange element under investigation.",
        advection: 1.0,
        air_drag: 0.01,
        air_loss: 0.99,
        loss: 0.3,
        collision: -0.1,
        diffusion: 0.4,
        hot_air: 0.001,
        hardness: 1,
        heat_conduct: 42,
        default_temp: 897.15,
        default_tmp: 900,
        pixel_mode: PMODE_FLAT,
    ));
    // --- TMPS.cpp (mod id 208) -----------------------------------------------------
    v.push(elem!(
        id: PT_TMPS, ident: "DEFAULT_PT_TMPS", name: "TMPS", cat: MenuCategory::Sensors,
        colour: 0x247BFE, props: TYPE_SOLID,
        weight: 100,
        desc: "Tmp sensor. Use .tmp3 for detecting different tmp values. Read wiki for more information.",
        air_loss: 0.96,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_temp: 277.15,
        default_tmp2: 2,
    ));
    // --- PHOS.cpp (mod id 209) -----------------------------------------------------
    v.push(elem!(
        id: PT_PHOS, ident: "DEFAULT_PT_PHOS", name: "PHOS", cat: MenuCategory::Powders,
        colour: 0xFFFFFF, props: TYPE_PART,
        weight: 75,
        desc: "Phosphorus, slowly turns red, melts at 45C and reacts violently with O2. Glows under UV. Acts as a fertiliser for PLNT.",
        advection: 0.4,
        air_drag: 0.04,
        air_loss: 0.94,
        loss: 0.95,
        collision: -0.1,
        gravity: 0.3,
        falldown: 1,
        explosive: 1,
        meltable: 5,
        hardness: 1,
        heat_conduct: 110,
        high_temp: 317.0,
        high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT,
    ));
    // --- CMNT.cpp (mod id 210) -----------------------------------------------------
    v.push(elem!(
        id: PT_CMNT, ident: "DEFAULT_PT_CMNT", name: "CMNT", cat: MenuCategory::Powders,
        colour: 0x808080, props: TYPE_PART,
        weight: 100,
        desc: "Cement, starts to solidify when mixed with water. Useful in making buildings.",
        advection: 0.7,
        air_drag: 0.02,
        air_loss: 0.96,
        loss: 0.8,
        gravity: 0.1,
        falldown: 1,
        meltable: 2,
        hardness: 1,
        heat_conduct: 100,
        high_temp: 2887.15,
        high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT,
    ));
    // --- NTRG.cpp (mod id 211) -----------------------------------------------------
    v.push(elem!(
        id: PT_NTRG, ident: "DEFAULT_PT_NTRG", name: "NTRG", cat: MenuCategory::Gases,
        colour: 0x101010, props: TYPE_GAS | PROP_NEUTPASS,
        weight: 1,
        desc: "Nitrogen, colourless gas. Makes NITR with H2 Liquefies into LN2 when cooled below 190C or when under pressure.",
        advection: 1.0,
        air_drag: 0.01,
        air_loss: 0.99,
        loss: 0.3,
        collision: -0.1,
        diffusion: 0.75,
        hot_air: 0.001,
        hardness: 1,
        heat_conduct: 42,
        default_temp: 297.15,
        high_pressure: 12.0,
        high_pressure_transition: PT_LN2 as i32,
        low_temp: 174.15,
        low_temp_transition: PT_LN2 as i32,
    ));
    // --- PRMT.cpp (mod id 212) -----------------------------------------------------
    v.push(elem!(
        id: PT_PRMT, ident: "DEFAULT_PT_PRMT", name: "PRMT", cat: MenuCategory::Radioactive,
        photon_reflect: 0x008000,
        colour: 0x008000, props: TYPE_PART | PROP_NEUTPASS | PROP_RADIOACTIVE,
        weight: 100,
        desc: "Promethium, catches fire at high velocity, emits NEUT at high temp and with PLUT, explosive at low temp.",
        advection: 0.4,
        air_drag: 0.01,
        air_loss: 0.99,
        loss: 0.95,
        gravity: 0.4,
        falldown: 1,
        flammable: 100,
        explosive: 1,
        hardness: 5,
        heat_conduct: 35,
        pixel_mode: PMODE_FLAT,
    ));
    // --- CSNS.cpp (mod id 213) -----------------------------------------------------
    v.push(elem!(
        id: PT_CSNS, ident: "DEFAULT_PT_CSNS", name: "CSNS", cat: MenuCategory::Sensors,
        colour: 0xFFFF00, props: TYPE_SOLID,
        weight: 100,
        desc: "Ctype sensor, creates a spark when there's a nearby particle with same ctype.",
        air_loss: 0.96,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_temp: 277.15,
        default_tmp2: 2,
        ctype_draw: CtypeDraw::ValueInCtypeLigh,
    ));
    // --- CWIR.cpp (mod id 214) -----------------------------------------------------
    v.push(elem!(
        id: PT_CWIR, ident: "DEFAULT_PT_CWIR", name: "CWIR", cat: MenuCategory::Electronics,
        photon_reflect: 0xCC0000,
        colour: 0xCC0000, props: TYPE_SOLID | PROP_CONDUCTS | PROP_HOT_GLOW | PROP_LIFE_DEC,
        weight: 100,
        desc: "Customisable wire, conduction speed set with .tmp, melting point with .tmp2",
        air_loss: 0.9,
        loss: 0.0,
        meltable: 1,
        hardness: 0,
        heat_conduct: 250,
    ));
    // --- CLRC.cpp (mod id 215) -----------------------------------------------------
    v.push(elem!(
        id: PT_CLRC, ident: "DEFAULT_PT_CLRC", name: "CLRC", cat: MenuCategory::Liquids,
        colour: 0xD1D1D1, props: TYPE_LIQUID | PROP_LIFE_DEC,
        weight: 99,
        desc: "Clear protective coat, liquid that sticks to solids and becomes invisible under UV.",
        advection: 0.3,
        air_drag: 0.02,
        air_loss: 0.98,
        loss: 0.8,
        gravity: 0.15,
        falldown: 2,
        hardness: 0,
        heat_conduct: 0,
        default_temp: 295.15,
        pixel_mode: PMODE_FLAT,
    ));
    // --- ELEX.cpp (mod id 216) -----------------------------------------------------
    v.push(elem!(
        id: PT_ELEX, ident: "DEFAULT_PT_ELEX", name: "ELEX", cat: MenuCategory::Special,
        colour: 0x303030, props: TYPE_GAS,
        weight: 100,
        desc: "Element X, turns into random element when above 0C.",
        advection: 0.5,
        air_loss: 0.6,
        loss: 0.6,
        collision: -0.1,
        diffusion: 2.0,
        hardness: 0,
        heat_conduct: 51,
    ));
    // --- RADN.cpp (mod id 217) -----------------------------------------------------
    v.push(elem!(
        id: PT_RADN, ident: "DEFAULT_PT_RADN", name: "RADN", cat: MenuCategory::Radioactive,
        colour: 0x0C1875, props: TYPE_GAS | PROP_NEUTPASS | PROP_CONDUCTS | PROP_LIFE_DEC,
        weight: 38,
        desc: "Radon, a heavy radioactive conductive gas with short half-life. Gets ionised in presence of UV.",
        advection: 1.0,
        air_drag: 0.01,
        air_loss: 0.99,
        loss: 0.3,
        collision: -0.1,
        gravity: 0.2,
        diffusion: 0.4,
        hot_air: 0.001,
        hardness: 0,
        heat_conduct: 52,
        pixel_mode: FIRE_BLEND,
    ));
    // --- GRPH.cpp (mod id 218) -----------------------------------------------------
    // `DefaultProperties.tmp2 = interfaceRng.between(0, 4)` cannot be evaluated
    // for a static table; the row records the lower bound and `create` seeds the
    // per-particle value.
    v.push(elem!(
        id: PT_GRPH, ident: "DEFAULT_PT_GRPH", name: "GRPH", cat: MenuCategory::Solids,
        colour: 0x3B3B3B,
        props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC | PROP_HOT_GLOW | PROP_NEUTPASS,
        weight: 100,
        desc: "Graphite, efficient heat and electricity conductor. Ignites when above 450C. Absorbs NEUT. GRPH + O2 -> CO2.",
        air_loss: 0.9,
        loss: 0.0,
        meltable: 5,
        hardness: 1,
        heat_conduct: 255,
        default_tmp2: 0,
        high_temp: 3950.15,
        high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT,
    ));
    // --- RUBR.cpp (mod id 219) -----------------------------------------------------
    v.push(elem!(
        id: PT_RUBR, ident: "DEFAULT_PT_RUBR", name: "RUBR", cat: MenuCategory::Liquids,
        colour: 0x5A5A5A, props: TYPE_LIQUID,
        weight: 100,
        desc: "Rubber (latex form). Bounces off particles, can be set into shape when above 230C. Blocks heat and pressure. Read wiki.",
        advection: 0.3,
        air_drag: 0.02,
        air_loss: 0.98,
        loss: 0.8,
        gravity: 0.15,
        falldown: 2,
        hardness: 2,
        heat_conduct: 10,
        pixel_mode: PMODE_FLAT,
    ));
    // --- DFOM.cpp (mod id 220) -----------------------------------------------------
    v.push(elem!(
        id: PT_DFOM, ident: "DEFAULT_PT_DFOM", name: "DFOM", cat: MenuCategory::Gases,
        colour: 0x708090, props: TYPE_PART | PROP_HOT_GLOW,
        weight: 100,
        desc: "Defence foam. absorbs shock of an explosion and then returns to set shape. Read wiki.",
        air_loss: 0.9,
        loss: 0.0,
        meltable: 1,
        hardness: 1,
        heat_conduct: 251,
        high_temp: 6273.0,
        high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT,
    ));
    // --- ECLR.cpp (mod id 221) -----------------------------------------------------
    v.push(elem!(
        id: PT_ECLR, ident: "DEFAULT_PT_ECLR", name: "ECLR", cat: MenuCategory::Force,
        colour: 0xD6D1D4, props: TYPE_SOLID,
        weight: 100,
        desc: "Electronic eraser, clears surrounding when sparked. Use with PSCN & NSCN. Use .tmp to set radius.",
        air_loss: 0.0,
        loss: 0.0,
        hardness: 5,
        heat_conduct: 0,
        default_tmp: 10,
        pixel_mode: PMODE_FLAT,
    ));
    // --- CEXP.cpp (mod id 222) -----------------------------------------------------
    // The class sets no `DefaultProperties`; `create` supplies the spawn
    // temperature, pressure, gravity and element at runtime.
    v.push(elem!(
        id: PT_CEXP, ident: "DEFAULT_PT_CEXP", name: "CEXP", cat: MenuCategory::Explosives,
        photon_reflect: 0xFF6347,
        colour: 0xFFA500, props: TYPE_SOLID,
        weight: 100,
        desc: "Custom explosive. Temp. = Temp upon explosion, Life = Pressure it creates, tmp = gravity. ctype = exploding element.",
        air_loss: 0.95,
        loss: 0.0,
        hardness: 10,
        heat_conduct: 0,
    ));
    // --- CLUD.cpp (mod id 223) -----------------------------------------------------
    v.push(elem!(
        id: PT_CLUD, ident: "DEFAULT_PT_CLUD", name: "CLUD", cat: MenuCategory::Gases,
        colour: 0xAAAAAA, props: TYPE_GAS,
        weight: 35,
        desc: "Cloud, rains/ snows after sometime and creates LIGH.",
        air_loss: 0.0,
        loss: 0.0,
        collision: 0.1,
        hardness: 0,
        heat_conduct: 0,
        default_temp: 353.15,
        pixel_mode: PMODE_FLAT,
    ));
    // --- COPR.cpp (mod id 224) -----------------------------------------------------
    v.push(elem!(
        id: PT_COPR, ident: "DEFAULT_PT_COPR", name: "COPR", cat: MenuCategory::Electronics,
        colour: 0xB87333, props: TYPE_PART | PROP_CONDUCTS | PROP_LIFE_DEC | PROP_HOT_GLOW,
        weight: 100,
        desc: "Excellent conductor. Turns into oxide when exposed to O2, losing conductivity. Shows superconductivity at low temps.",
        advection: 0.7,
        air_drag: 0.02,
        air_loss: 0.96,
        loss: 0.8,
        gravity: 0.1,
        falldown: 1,
        meltable: 2,
        hardness: 1,
        heat_conduct: 255,
        high_temp: 1359.0,
        high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT,
    ));
    // --- UVRD.cpp (mod id 225) -----------------------------------------------------
    v.push(elem!(
        id: PT_UVRD, ident: "DEFAULT_PT_UVRD", name: "UV", cat: MenuCategory::Radioactive,
        colour: 0x9400D3, props: TYPE_ENERGY | PROP_LIFE_DEC,
        weight: -1,
        desc: "UV rays emitted by SUN, reacts differently with different elements. Visible when passing through FILT ",
        air_loss: 1.0,
        loss: 1.0,
        collision: -0.99,
        hardness: 0,
        heat_conduct: 251,
        default_temp: 295.3,
        pixel_mode: PMODE_FLAT,
    ));
    // --- SUN.cpp (mod id 226) ------------------------------------------------------
    v.push(elem!(
        id: PT_SUN, ident: "DEFAULT_PT_SUN", name: "SUN", cat: MenuCategory::Special,
        colour: 0xFF4500, props: TYPE_SOLID,
        weight: 100,
        desc: "SUN, extremely hot, emits harmful UV radiation. Makes PLNT grow in direction of sunlight.",
        air_loss: 0.95,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_temp: 6300.15,
        pixel_mode: PMODE_FLAT | FIRE_ADD,
    ));
    // --- PROJ.cpp (mod id 227) -----------------------------------------------------
    // `Colour = 0xFFE21D24_rgb`: `RGB::Unpack` keeps the low 24 bits.
    v.push(elem!(
        id: PT_PROJ, ident: "DEFAULT_PT_PROJ", name: "PROJ", cat: MenuCategory::Force,
        colour: 0xE21D24, props: TYPE_PART,
        weight: 100,
        desc: "Projectile, converts into ctype upon collision. SPRK with PSCN to launch. Use .tmp for range and temp. for power.",
        air_loss: 0.0,
        loss: 0.0,
        hardness: 5,
        heat_conduct: 0,
        default_temp: 52.0,
        default_ctype: PT_BOMB,
        default_tmp: 10,
        ctype_draw: CtypeDraw::ValueInCtypeLigh,
        pixel_mode: PMODE_FLAT,
    ));
    // --- GSNS.cpp (mod id 228) -----------------------------------------------------
    v.push(elem!(
        id: PT_GSNS, ident: "DEFAULT_PT_GSNS", name: "GSNS", cat: MenuCategory::Sensors,
        colour: 0x840777, props: TYPE_SOLID,
        weight: 100,
        desc: "Gravity sensor, creates a spark when the nearby gravity is higher than its temperature.",
        air_loss: 0.96,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 0,
        default_temp: 277.15,
    ));
    // --- BFLM.cpp (mod id 229) -----------------------------------------------------
    v.push(elem!(
        id: PT_BFLM, ident: "DEFAULT_PT_BFLM", name: "BFLM", cat: MenuCategory::Explosives,
        colour: 0x202020, props: TYPE_GAS | PROP_LIFE_DEC | PROP_LIFE_KILL,
        weight: 2,
        desc: "Black Flames, unstoppable flames that eat everything in its way (except DMRN & WALL).",
        advection: 0.9,
        air_drag: 0.04,
        air_loss: 0.97,
        loss: 0.2,
        gravity: -0.1,
        hot_air: 0.001,
        falldown: 1,
        hardness: 1,
        heat_conduct: 88,
        default_temp: 695.15,
        pixel_mode: PMODE_FLAT | FIRE_BLEND | FIRE_ADD,
    ));
    // --- AMBE.cpp (mod id 230) -----------------------------------------------------
    v.push(elem!(
        id: PT_AMBE, ident: "DEFAULT_PT_AMBE", name: "AMBE", cat: MenuCategory::Powered,
        colour: 0x420075, props: TYPE_SOLID,
        weight: 100,
        desc: "Element that sets the nearby ambient air temperature to its own temperature. tmp = effective area. Use with PSCN & NSCN.",
        air_loss: 0.0,
        loss: 0.0,
        hardness: 5,
        heat_conduct: 0,
        default_tmp: 4,
        pixel_mode: PMODE_FLAT,
    ));
    // --- NAPM.cpp (mod id 231) -----------------------------------------------------
    v.push(elem!(
        id: PT_NAPM, ident: "DEFAULT_PT_NAPM", name: "NAPM", cat: MenuCategory::Explosives,
        colour: 0xFF7518, props: TYPE_LIQUID | PROP_LIFE_DEC,
        weight: 20,
        desc: "Napalm, liquid that sticks to solids and can reach temp. around 1200C while burning. Cannot be extinguished once ignited.",
        advection: 0.2,
        air_drag: 0.02,
        air_loss: 0.77,
        loss: 0.6,
        gravity: 0.3,
        falldown: 2,
        hardness: 5,
        heat_conduct: 255,
    ));
    // --- ALUM.cpp (mod id 232) -----------------------------------------------------
    // `HighTemperature = ITH` with `HighTemperatureTransition = PT_LAVA`: the
    // element melts through its `ChangeType` hook, not the temperature rule.
    v.push(elem!(
        id: PT_ALUM, ident: "DEFAULT_PT_ALUM", name: "ALUM", cat: MenuCategory::Solids,
        colour: 0xB0B0C4, props: TYPE_SOLID | PROP_CONDUCTS | PROP_LIFE_DEC,
        weight: 100,
        desc: "Aluminium. Breakable metal that contains pressure.",
        air_loss: 0.9,
        loss: 0.7,
        meltable: 1,
        hardness: 20,
        heat_conduct: 251,
        high_temp: ITH,
        high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT,
    ));
    // --- ALMP.cpp (mod id 233) -----------------------------------------------------
    v.push(elem!(
        id: PT_ALMP, ident: "DEFAULT_PT_ALMP", name: "ALMP", cat: MenuCategory::Explosives,
        photon_reflect: 0x3FFF_FFC0,
        colour: 0x87A4AF, props: TYPE_PART | PROP_SPARKSETTLE,
        weight: 90,
        desc: "Aluminium powder. Flammable, burns with brilliant sparks.",
        advection: 0.4,
        air_drag: 0.04,
        air_loss: 0.94,
        loss: 0.92,
        collision: -0.1,
        gravity: 0.1,
        falldown: 1,
        meltable: 1,
        hardness: 10,
        heat_conduct: 70,
        default_tmp: 40,
        pixel_mode: PMODE_FLAT,
    ));
    // --- PPTI.cpp (mod id 234) -----------------------------------------------------
    v.push(elem!(
        id: PT_PPTI, ident: "DEFAULT_PT_PPTI", name: "PPTI", cat: MenuCategory::Powered,
        colour: 0xEB5917, props: TYPE_SOLID,
        weight: 100,
        desc: "Powered PRTI.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT | FIRE_ADD,
    ));
    // --- PPTO.cpp (mod id 235) -----------------------------------------------------
    v.push(elem!(
        id: PT_PPTO, ident: "DEFAULT_PT_PPTO", name: "PPTO", cat: MenuCategory::Powered,
        colour: 0x0020EB, props: TYPE_SOLID,
        weight: 100,
        desc: "Powered PRTO.",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 0,
        heat_conduct: 0,
        pixel_mode: PMODE_FLAT | FIRE_ADD,
    ));
    // --- PCON.cpp (mod id 236) -----------------------------------------------------
    v.push(elem!(
        id: PT_PCON, ident: "DEFAULT_PT_PCON", name: "PCON", cat: MenuCategory::Powered,
        colour: 0x32CD32, props: TYPE_SOLID | PROP_NOCTYPEDRAW,
        weight: 100,
        desc: "Powered Converter. Converts everything into whatever it first touches when powered. Set its Ctype carefully!",
        air_loss: 0.9,
        loss: 0.0,
        hardness: 1,
        heat_conduct: 251,
        ctype_draw: CtypeDraw::ValueInCtype,
        pixel_mode: PMODE_FLAT,
    ));
    // --- MISL.cpp (mod id 237) -----------------------------------------------------
    v.push(elem!(
        id: PT_MISL, ident: "DEFAULT_PT_MISL", name: "MISL", cat: MenuCategory::Force,
        colour: 0xFFA500, props: TYPE_PART,
        weight: 100,
        desc: "Missile, flies to target coordinates (tmp = X, tmp2 = Y) and then goes booom. Activates with PSCN. Use MIST tool for ease.",
        air_loss: 0.0,
        loss: 0.0,
        hardness: 5,
        heat_conduct: 0,
    ));
    // --- MGNT.cpp (mod id 238) -----------------------------------------------------
    v.push(elem!(
        id: PT_MGNT, ident: "DEFAULT_PT_MGNT", name: "EMGT", cat: MenuCategory::Force,
        colour: 0x708090, props: TYPE_SOLID | PROP_HOT_GLOW,
        weight: 100,
        desc: "Strong electro magnet, creates EM fields when powered. Can attract/ repel certain metals or Energy particles. Read WIKI.",
        air_loss: 0.9,
        loss: 0.0,
        meltable: 1,
        hardness: 1,
        heat_conduct: 251,
        high_temp: 1573.0,
        high_temp_transition: PT_LAVA as i32,
        pixel_mode: PMODE_FLAT,
    ));
}
