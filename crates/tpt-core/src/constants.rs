//! Global simulation constants shared by every crate in the workspace.
//!
//! The simulation grid stores one element per pixel (`xres * yres`), while the
//! Eulerian air and wall grids operate on `CELL x CELL` blocks (`xcells * ycells`).
//! Values mirror the reference Powder Toy implementation so that element
//! behaviour, save compatibility and rendering stay consistent.

/// Number of pixels per air/wall cell along one axis.
pub const CELL: u32 = 4;

/// Default canvas width in pixels.
pub const DEFAULT_XRES: u32 = 612;
/// Default canvas height in pixels.
pub const DEFAULT_YRES: u32 = 384;

/// Room temperature in degrees Celsius.
pub const R_TEMP: f32 = 22.0;
/// Default ambient temperature in Kelvin.
pub const DEFAULT_TEMP_K: f32 = 295.15;
/// Lowest representable particle temperature (Kelvin).
pub const MIN_TEMP: f32 = 0.0;
/// Highest representable particle temperature (Kelvin).
pub const MAX_TEMP: f32 = 9999.0;
/// Lowest representable air pressure.
pub const MIN_PRESSURE: f32 = -256.0;
/// Highest representable air pressure.
pub const MAX_PRESSURE: f32 = 256.0;

/// Lowest representable air pressure for the pressure field (sentinel bound).
pub const IPL: f32 = -257.0;
/// Highest representable air pressure for the pressure field (sentinel bound).
pub const IPH: f32 = 257.0;
/// Lowest representable temperature for the heat field.
pub const ITL: f32 = -1.0;
/// Highest representable temperature for the heat field.
pub const ITH: f32 = 10000.0;
/// Sentinel meaning "no temperature/pressure transition".
pub const NT: i32 = -1;
/// Sentinel marking a transition that needs special-case handling in the kernels.
pub const ST: i32 = 512;

/// Air velocity time step for pressure integration.
pub const AIR_TSTEPP: f32 = 0.3;
/// Air velocity time step for advection.
pub const AIR_TSTEPV: f32 = 0.4;
/// Air advection velocity scale.
pub const AIR_VADV: f32 = 0.3;
/// Per-step air velocity damping.
pub const AIR_VLOSS: f32 = 0.999;
/// Per-step air pressure damping.
pub const AIR_PLOSS: f32 = 0.9999;

/// Total number of element slots (built-in plus custom elements).
///
/// Derived from [`BUILTIN_ELEMENT_COUNT`], with room for sixteen Lua-defined
/// elements, rounded up to a whole number of eight-slot groups, and kept at or
/// below 256 because every element row costs 128 bytes on the GPU. Every id
/// must also fit the 9-bit type field of a packed `pmap` word (`pmap_pack`),
/// i.e. `PT_NUM <= 512`.
pub const PT_NUM: usize = (BUILTIN_ELEMENT_COUNT + 16 + 7) & !7;

/// Empty pixel.
pub const PT_NONE: u32 = 0;
/// Dust powder.
pub const PT_DUST: u32 = 1;
/// Water.
pub const PT_WATR: u32 = 2;
/// Oil.
pub const PT_OIL: u32 = 3;
/// Fire.
pub const PT_FIRE: u32 = 4;
/// Stone.
pub const PT_STNE: u32 = 5;
/// Lava.
pub const PT_LAVA: u32 = 6;
/// Gunpowder.
pub const PT_GUNP: u32 = 7;
/// Nitroglycerin.
pub const PT_NITR: u32 = 8;
/// C-4 explosive.
pub const PT_C4: u32 = 9;
/// Gas (flammable gas).
pub const PT_GAS: u32 = 10;
/// Metal.
pub const PT_METL: u32 = 11;
/// Spark (electrical energy).
pub const PT_SPRK: u32 = 12;
/// Snow.
pub const PT_SNOW: u32 = 13;
/// Ice.
pub const PT_ICEI: u32 = 14;
/// Wood.
pub const PT_WOOD: u32 = 15;
/// Smoke.
pub const PT_SMKE: u32 = 16;
/// Sand.
pub const PT_SAND: u32 = 17;
/// Salt.
pub const PT_SALT: u32 = 18;
/// Salt water.
pub const PT_SLTW: u32 = 19;
/// Acid.
pub const PT_ACID: u32 = 20;
/// Water vapour / steam.
pub const PT_WTRV: u32 = 21;
/// Clone.
pub const PT_CLNE: u32 = 22;
/// Void (destroys particles).
pub const PT_VOID: u32 = 23;
/// Glass.
pub const PT_GLAS: u32 = 24;
/// Diamond.
pub const PT_DMND: u32 = 25;
/// Construction (concrete).
pub const PT_CNCT: u32 = 26;
/// Black coal.
pub const PT_BCOL: u32 = 27;
/// Plasma.
pub const PT_PLSM: u32 = 28;
/// Photon.
pub const PT_PHOT: u32 = 29;
/// Neutron.
pub const PT_NEUT: u32 = 30;
/// Electron / electricity.
pub const PT_ELEC: u32 = 31;
/// Hydrogen.
pub const PT_H2: u32 = 32;
/// Oxygen.
pub const PT_O2: u32 = 33;
/// Caustic (strong acid).
pub const PT_CAUS: u32 = 34;
/// Distilled water.
pub const PT_DSTW: u32 = 35;
/// Titanium.
pub const PT_TTAN: u32 = 36;
/// Brick.
pub const PT_BRCK: u32 = 37;
/// Liquid nitrogen.
pub const PT_LN2: u32 = 38;
/// Gravitons (Newtonian gravity source).
pub const PT_GRVT: u32 = 39;
/// Black hole (negative-gravity vacuum, Newtonian gravity sink).
pub const PT_NBHL: u32 = 40;
/// White hole (positive-gravity vacuum, Newtonian gravity source).
pub const PT_NWHL: u32 = 41;
/// Gravity bomb.
pub const PT_GBMB: u32 = 42;
/// Gravity pump.
pub const PT_GPMP: u32 = 43;

// Batch 1: the remaining TPT "plain matter" elements
//
// Appended above the classic ids (which saves and tests pin) in The Powder
// Toy's own element order. The doc comment of each entry gives the reference
// `PT_*` id from `src/simulation/elements/meson.build` and the `.cpp` the
// definition was transcribed from.

/// Goo (TPT 12, `GOO.cpp`). Deforms and disappears under pressure.
pub const PT_GOO: u32 = 44;
/// Plant (TPT 20, `PLNT.cpp`). Drinks water and grows.
pub const PT_PLNT: u32 = 45;
/// Breakable metal (TPT 29, `BMTL.cpp`).
pub const PT_BMTL: u32 = 46;
/// Broken metal powder (TPT 30, `BRMT.cpp`).
pub const PT_BRMT: u32 = 47;
/// Wax (TPT 33, `WAX.cpp`).
pub const PT_WAX: u32 = 48;
/// Liquid wax (TPT 34, `MWAX.cpp`).
pub const PT_MWAX: u32 = 49;
/// P-type silicon (TPT 35, `PSCN.cpp`).
pub const PT_PSCN: u32 = 50;
/// N-type silicon (TPT 36, `NSCN.cpp`).
pub const PT_NSCN: u32 = 51;
/// Insulator (TPT 38, `INSL.cpp`).
pub const PT_INSL: u32 = 52;
/// NTC thermistor (TPT 43, `NTCT.cpp`).
pub const PT_NTCT: u32 = 53;
/// PTC thermistor (TPT 46, `PTCT.cpp`).
pub const PT_PTCT: u32 = 54;
/// Electrode (TPT 50, `ETRD.cpp`).
pub const PT_ETRD: u32 = 55;
/// Nitrogen ice (TPT 51, `NICE.cpp`).
pub const PT_NICE: u32 = 56;
/// Battery (TPT 53, `BTRY.cpp`).
pub const PT_BTRY: u32 = 57;
/// Liquid crystal (TPT 54, `LCRY.cpp`).
pub const PT_LCRY: u32 = 58;
/// Stickman (TPT 55, `STKM.cpp`).
pub const PT_STKM: u32 = 59;
/// Switch (TPT 56, `SWCH.cpp`).
pub const PT_SWCH: u32 = 60;
/// Diesel (TPT 58, `DESL.cpp`).
pub const PT_DESL: u32 = 61;
/// Coal (TPT 59, `COAL.cpp`).
pub const PT_COAL: u32 = 62;
/// Insulated wire (TPT 62, `INWR.cpp`).
pub const PT_INWR: u32 = 63;
/// Yeast (TPT 63, `YEST.cpp`).
pub const PT_YEST: u32 = 64;
/// Dead yeast (TPT 64, `DYST.cpp`).
pub const PT_DYST: u32 = 65;
/// Glow (TPT 66, `GLOW.cpp`).
pub const PT_GLOW: u32 = 66;
/// Powered clone (TPT 74, `PCLN.cpp`).
pub const PT_PCLN: u32 = 67;
/// Heat switch (TPT 75, `HSWC.cpp`).
pub const PT_HSWC: u32 = 68;
/// Iron (TPT 76, `IRON.cpp`).
pub const PT_IRON: u32 = 69;
/// Steam train (TPT 77, `MORT.cpp`).
pub const PT_MORT: u32 = 70;
/// Delay conductor (TPT 79, `DLAY.cpp`).
pub const PT_DLAY: u32 = 71;
/// Dry ice (TPT 81, `DRIC.cpp`).
pub const PT_DRIC: u32 = 72;
/// Carbonated water / `BUBW` (TPT 82, `CBNW.cpp`).
pub const PT_CBNW: u32 = 73;
/// Storage (TPT 83, `STOR.cpp`).
pub const PT_STOR: u32 = 74;
/// Powered void (TPT 84, `PVOD.cpp`).
pub const PT_PVOD: u32 = 75;
/// Converter (TPT 85, `CONV.cpp`).
pub const PT_CONV: u32 = 76;
/// Tesla coil (TPT 88, `TESC.cpp`).
pub const PT_TESC: u32 = 77;
/// Sponge (TPT 90, `SPNG.cpp`).
pub const PT_SPNG: u32 = 78;
/// Rime / hoar frost (TPT 91, `RIME.cpp`).
pub const PT_RIME: u32 = 79;
/// Fog (TPT 92, `FOG.cpp`).
pub const PT_FOG: u32 = 80;
/// Pressure pump (TPT 97, `PUMP.cpp`).
pub const PT_PUMP: u32 = 81;
/// Pipe (TPT 99, `PIPE.cpp`).
pub const PT_PIPE: u32 = 82;
/// Freeze powder (TPT 100, `FRZZ.cpp`).
pub const PT_FRZZ: u32 = 83;
/// Freeze water (TPT 101, `FRZW.cpp`).
pub const PT_FRZW: u32 = 84;
/// Bizarre liquid (TPT 103, `BIZR.cpp`).
pub const PT_BIZR: u32 = 85;
/// Bizarre gas (TPT 104, `BIZRG.cpp`).
pub const PT_BIZRG: u32 = 86;
/// Bizarre solid (TPT 105, `BIZRS.cpp`).
pub const PT_BIZRS: u32 = 87;
/// Temperature sensor (TPT `TSNS`, listed here as `TCSN`; `TSNS.cpp`).
pub const PT_TSNS: u32 = 88;

// Batch 2: every remaining TPT element
//
// Appended after batch 1 in The Powder Toy's own element order (`meson.build`).
// The doc comment gives the reference TPT id from `ElementNumbers.h`.

/// PLUT (TPT 19, `PLUT.cpp`).
pub const PT_PLUT: u32 = 89;
/// URAN (TPT 32, `URAN.cpp`).
pub const PT_URAN: u32 = 90;
/// VACU (TPT 39, `BHOL.cpp`).
pub const PT_BHOL: u32 = 91;
/// VENT (TPT 40, `WHOL.cpp`).
pub const PT_WHOL: u32 = 92;
/// RBDM (TPT 41, `RBDM.cpp`).
pub const PT_RBDM: u32 = 93;
/// LRBD (TPT 42, `LRBD.cpp`).
pub const PT_LRBD: u32 = 94;
/// BGLA (TPT 47, `BGLA.cpp`).
pub const PT_BGLA: u32 = 95;
/// THDR (TPT 48, `THDR.cpp`).
pub const PT_THDR: u32 = 96;
/// NBLE (TPT 52, `NBLE.cpp`).
pub const PT_NBLE: u32 = 97;
/// LOXY (TPT 60, `LO2.cpp`).
pub const PT_LO2: u32 = 98;
/// THRM (TPT 65, `THRM.cpp`).
pub const PT_THRM: u32 = 99;
/// CFLM (TPT 68, `CFLM.cpp`).
pub const PT_HFLM: u32 = 100;
pub const PT_CFLM: u32 = PT_HFLM;
/// FIRW (TPT 69, `FIRW.cpp`).
pub const PT_FIRW: u32 = 101;
/// FUSE (TPT 70, `FUSE.cpp`).
pub const PT_FUSE: u32 = 102;
/// FSEP (TPT 71, `FSEP.cpp`).
pub const PT_FSEP: u32 = 103;
/// AMTR (TPT 72, `AMTR.cpp`).
pub const PT_AMTR: u32 = 104;
/// LIFE (TPT 78, `LIFE.cpp`).
pub const PT_LIFE: u32 = 105;
/// CO2 (TPT 80, `CO2.cpp`).
pub const PT_CO2: u32 = 106;
/// LIGH (TPT 87, `LIGH.cpp`).
pub const PT_LIGH: u32 = 107;
/// DEST (TPT 89, `DEST.cpp`).
pub const PT_DEST: u32 = 108;
/// BCLN (TPT 93, `BCLN.cpp`).
pub const PT_BCLN: u32 = 109;
/// LOVE (TPT 94, `LOVE.cpp`).
pub const PT_LOVE: u32 = 110;
/// DEUT (TPT 95, `DEUT.cpp`).
pub const PT_DEUT: u32 = 111;
/// WARP (TPT 96, `WARP.cpp`).
pub const PT_WARP: u32 = 112;
/// FWRK (TPT 98, `FWRK.cpp`).
pub const PT_FWRK: u32 = 113;
/// GRAV (TPT 102, `GRAV.cpp`).
pub const PT_GRAV: u32 = 114;
/// INST (TPT 106, `INST.cpp`).
pub const PT_INST: u32 = 115;
/// ISOZ (TPT 107, `ISOZ.cpp`).
pub const PT_ISOZ: u32 = 116;
/// ISZS (TPT 108, `ISZS.cpp`).
pub const PT_ISZS: u32 = 117;
/// PRTI (TPT 109, `PRTI.cpp`).
pub const PT_PRTI: u32 = 118;
/// PRTO (TPT 110, `PRTO.cpp`).
pub const PT_PRTO: u32 = 119;
/// PSTE (TPT 111, `PSTE.cpp`).
pub const PT_PSTE: u32 = 120;
/// PSTS (TPT 112, `PSTS.cpp`).
pub const PT_PSTS: u32 = 121;
/// ANAR (TPT 113, `ANAR.cpp`).
pub const PT_ANAR: u32 = 122;
/// VINE (TPT 114, `VINE.cpp`).
pub const PT_VINE: u32 = 123;
/// INVS (TPT 115, `INVIS.cpp`).
pub const PT_INVIS: u32 = 124;
/// EQVE (TPT 116, `E116.cpp`).
pub const PT_116: u32 = 125;
/// SPWN2 (TPT 117, `SPAWN2.cpp`).
pub const PT_SPAWN2: u32 = 126;
/// SPWN (TPT 118, `SPAWN.cpp`).
pub const PT_SPAWN: u32 = 127;
/// SHLD (TPT 119, `SHLD1.cpp`).
pub const PT_SHLD1: u32 = 128;
/// SHD2 (TPT 120, `SHLD2.cpp`).
pub const PT_SHLD2: u32 = 129;
/// SHD3 (TPT 121, `SHLD3.cpp`).
pub const PT_SHLD3: u32 = 130;
/// SHD4 (TPT 122, `SHLD4.cpp`).
pub const PT_SHLD4: u32 = 131;
/// LOLZ (TPT 123, `LOLZ.cpp`).
pub const PT_LOLZ: u32 = 132;
/// WIFI (TPT 124, `WIFI.cpp`).
pub const PT_WIFI: u32 = 133;
/// FILT (TPT 125, `FILT.cpp`).
pub const PT_FILT: u32 = 134;
/// ARAY (TPT 126, `ARAY.cpp`).
pub const PT_ARAY: u32 = 135;
/// BRAY (TPT 127, `BRAY.cpp`).
pub const PT_BRAY: u32 = 136;
/// STK2 (TPT 128, `STKM2.cpp`).
pub const PT_STKM2: u32 = 137;
/// BOMB (TPT 129, `BOMB.cpp`).
pub const PT_BOMB: u32 = 138;
/// C-5 (TPT 130, `C5.cpp`).
pub const PT_C5: u32 = 139;
/// SING (TPT 131, `SING.cpp`).
pub const PT_SING: u32 = 140;
/// QRTZ (TPT 132, `QRTZ.cpp`).
pub const PT_QRTZ: u32 = 141;
/// PQRT (TPT 133, `PQRT.cpp`).
pub const PT_PQRT: u32 = 142;
/// EMP (TPT 134, `EMP.cpp`).
pub const PT_EMP: u32 = 143;
/// BREL (TPT 135, `BREC.cpp`).
pub const PT_BREC: u32 = 144;
/// ACEL (TPT 137, `ACEL.cpp`).
pub const PT_ACEL: u32 = 145;
/// DCEL (TPT 138, `DCEL.cpp`).
pub const PT_DCEL: u32 = 146;
/// TNT (TPT 139, `BANG.cpp`).
pub const PT_BANG: u32 = 147;
/// IGNC (TPT 140, `IGNT.cpp`).
pub const PT_IGNT: u32 = 148;
/// BOYL (TPT 141, `BOYL.cpp`).
pub const PT_BOYL: u32 = 149;
/// GEL (TPT 142, `GEL.cpp`).
pub const PT_GEL: u32 = 150;
/// TRON (TPT 143, `TRON.cpp`).
pub const PT_TRON: u32 = 151;
/// EXOT (TPT 145, `EXOT.cpp`).
pub const PT_EXOT: u32 = 152;
/// EMBR (TPT 147, `EMBR.cpp`).
pub const PT_EMBR: u32 = 153;
/// SOAP (TPT 149, `SOAP.cpp`).
pub const PT_SOAP: u32 = 154;
/// MERC (TPT 152, `MERC.cpp`).
pub const PT_MERC: u32 = 155;
/// PBCN (TPT 153, `PBCN.cpp`).
pub const PT_PBCN: u32 = 156;
/// CLST (TPT 155, `CLST.cpp`).
pub const PT_CLST: u32 = 157;
/// WWLD (TPT 156, `WIRE.cpp`).
pub const PT_WIRE: u32 = 158;
/// FIGH (TPT 158, `FIGH.cpp`).
pub const PT_FIGH: u32 = 159;
/// FRAY (TPT 159, `FRAY.cpp`).
pub const PT_FRAY: u32 = 160;
/// RPEL (TPT 160, `RPEL.cpp`).
pub const PT_RPEL: u32 = 161;
/// PPIP (TPT 161, `PPIP.cpp`).
pub const PT_PPIP: u32 = 162;
/// DTEC (TPT 162, `DTEC.cpp`).
pub const PT_DTEC: u32 = 163;
/// DMG (TPT 163, `DMG.cpp`).
pub const PT_DMG: u32 = 164;
/// VIBR (TPT 165, `VIBR.cpp`).
pub const PT_VIBR: u32 = 165;
/// BVBR (TPT 166, `BVBR.cpp`).
pub const PT_BVBR: u32 = 166;
/// CRAY (TPT 167, `CRAY.cpp`).
pub const PT_CRAY: u32 = 167;
/// PSTN (TPT 168, `PSTN.cpp`).
pub const PT_PSTN: u32 = 168;
/// FRME (TPT 169, `FRME.cpp`).
pub const PT_FRME: u32 = 169;
/// GOLD (TPT 170, `GOLD.cpp`).
pub const PT_GOLD: u32 = 170;
/// TUNG (TPT 171, `TUNG.cpp`).
pub const PT_TUNG: u32 = 171;
/// PSNS (TPT 172, `PSNS.cpp`).
pub const PT_PSNS: u32 = 172;
/// PROT (TPT 173, `PROT.cpp`).
pub const PT_PROT: u32 = 173;
/// VIRS (TPT 174, `VIRS.cpp`).
pub const PT_VIRS: u32 = 174;
/// VRSS (TPT 175, `VRSS.cpp`).
pub const PT_VRSS: u32 = 175;
/// VRSG (TPT 176, `VRSG.cpp`).
pub const PT_VRSG: u32 = 176;
/// DRAY (TPT 178, `DRAY.cpp`).
pub const PT_DRAY: u32 = 177;
/// CRMC (TPT 179, `CRMC.cpp`).
pub const PT_CRMC: u32 = 178;
/// HEAC (TPT 180, `HEAC.cpp`).
pub const PT_HEAC: u32 = 179;
/// SAWD (TPT 181, `SAWD.cpp`).
pub const PT_SAWD: u32 = 180;
/// POLO (TPT 182, `POLO.cpp`).
pub const PT_POLO: u32 = 181;
/// RFRG (TPT 183, `RFRG.cpp`).
pub const PT_RFRG: u32 = 182;
/// RFGL (TPT 184, `RFGL.cpp`).
pub const PT_RFGL: u32 = 183;
/// LSNS (TPT 185, `LSNS.cpp`).
pub const PT_LSNS: u32 = 184;
/// LDTC (TPT 186, `LDTC.cpp`).
pub const PT_LDTC: u32 = 185;
/// SLCN (TPT 187, `SLCN.cpp`).
pub const PT_SLCN: u32 = 186;
/// PTNM (TPT 188, `PTNM.cpp`).
pub const PT_PTNM: u32 = 187;
/// VSNS (TPT 189, `VSNS.cpp`).
pub const PT_VSNS: u32 = 188;
/// ROCK (TPT 190, `ROCK.cpp`).
pub const PT_ROCK: u32 = 189;
/// LITH (TPT 191, `LITH.cpp`).
pub const PT_LITH: u32 = 190;
/// RSST (TPT 192, `RSST.cpp`).
pub const PT_RSST: u32 = 191;
/// RSSS (TPT 193, `RSSS.cpp`).
pub const PT_RSSS: u32 = 192;
/// BASE (TPT 194, `BASE.cpp`).
pub const PT_BASE: u32 = 193;
/// SEED (TPT 195, `SEED.cpp`).
pub const PT_SEED: u32 = 194;

// Mod elements
//
// The Powder Toy mod appends these 44 elements after `SEED`, in the order its
// `meson.build` lists them. They are appended (never interleaved) so every
// existing id stays pinned for saves and tests.

/// FNTC (mod id 195, `FNTC.cpp`).
pub const PT_FNTC: u32 = 195;
/// CLNT (mod id 196, `CLNT.cpp`).
pub const PT_CLNT: u32 = 196;
/// LED (mod id 197, `LED.cpp`).
pub const PT_LED: u32 = 197;
/// TIMC (mod id 198, `TIMC.cpp`).
pub const PT_TIMC: u32 = 198;
/// FUEL (mod id 199, `FUEL.cpp`).
pub const PT_FUEL: u32 = 199;
/// FPTC (mod id 200, `FPTC.cpp`).
pub const PT_FPTC: u32 = 200;
/// DMRN (mod id 201, `DMRN.cpp`).
pub const PT_DMRN: u32 = 201;
/// PINV (mod id 202, `PINV.cpp`).
pub const PT_PINV: u32 = 202;
/// WALL (mod id 203, `WALL.cpp`).
pub const PT_WALL: u32 = 203;
/// CHLR (mod id 204, `CHLR.cpp`).
pub const PT_CHLR: u32 = 204;
/// ACTY (mod id 205, `ACTY.cpp`).
pub const PT_ACTY: u32 = 205;
/// COND (mod id 206, `COND.cpp`).
pub const PT_COND: u32 = 206;
/// QGPP (mod id 207, `QGPP.cpp`).
pub const PT_QGPP: u32 = 207;
/// TMPS (mod id 208, `TMPS.cpp`).
pub const PT_TMPS: u32 = 208;
/// PHOS (mod id 209, `PHOS.cpp`).
pub const PT_PHOS: u32 = 209;
/// CMNT (mod id 210, `CMNT.cpp`).
pub const PT_CMNT: u32 = 210;
/// NTRG (mod id 211, `NTRG.cpp`).
pub const PT_NTRG: u32 = 211;
/// PRMT (mod id 212, `PRMT.cpp`).
pub const PT_PRMT: u32 = 212;
/// CSNS (mod id 213, `CSNS.cpp`).
pub const PT_CSNS: u32 = 213;
/// CWIR (mod id 214, `CWIR.cpp`).
pub const PT_CWIR: u32 = 214;
/// CLRC (mod id 215, `CLRC.cpp`).
pub const PT_CLRC: u32 = 215;
/// ELEX (mod id 216, `ELEX.cpp`).
pub const PT_ELEX: u32 = 216;
/// RADN (mod id 217, `RADN.cpp`).
pub const PT_RADN: u32 = 217;
/// GRPH (mod id 218, `GRPH.cpp`).
pub const PT_GRPH: u32 = 218;
/// RUBR (mod id 219, `RUBR.cpp`).
pub const PT_RUBR: u32 = 219;
/// DFOM (mod id 220, `DFOM.cpp`).
pub const PT_DFOM: u32 = 220;
/// ECLR (mod id 221, `ECLR.cpp`).
pub const PT_ECLR: u32 = 221;
/// CEXP (mod id 222, `CEXP.cpp`).
pub const PT_CEXP: u32 = 222;
/// CLUD (mod id 223, `CLUD.cpp`).
pub const PT_CLUD: u32 = 223;
/// COPR (mod id 224, `COPR.cpp`).
pub const PT_COPR: u32 = 224;
/// UVRD (mod id 225, `UVRD.cpp`).
pub const PT_UVRD: u32 = 225;
/// SUN (mod id 226, `SUN.cpp`).
pub const PT_SUN: u32 = 226;
/// PROJ (mod id 227, `PROJ.cpp`).
pub const PT_PROJ: u32 = 227;
/// GSNS (mod id 228, `GSNS.cpp`).
pub const PT_GSNS: u32 = 228;
/// BFLM (mod id 229, `BFLM.cpp`).
pub const PT_BFLM: u32 = 229;
/// AMBE (mod id 230, `AMBE.cpp`).
pub const PT_AMBE: u32 = 230;
/// NAPM (mod id 231, `NAPM.cpp`).
pub const PT_NAPM: u32 = 231;
/// ALUM (mod id 232, `ALUM.cpp`).
pub const PT_ALUM: u32 = 232;
/// ALMP (mod id 233, `ALMP.cpp`).
pub const PT_ALMP: u32 = 233;
/// PPTI (mod id 234, `PPTI.cpp`).
pub const PT_PPTI: u32 = 234;
/// PPTO (mod id 235, `PPTO.cpp`).
pub const PT_PPTO: u32 = 235;
/// PCON (mod id 236, `PCON.cpp`).
pub const PT_PCON: u32 = 236;
/// MISL (mod id 237, `MISL.cpp`).
pub const PT_MISL: u32 = 237;
/// MGNT (mod id 238, `MGNT.cpp`).
pub const PT_MGNT: u32 = 238;

/// Whether `ty` is one of the 44 elements the Cracker1000 mod appends to the
/// reference element table (`PT_FNTC ..= PT_MGNT`).
///
/// The port defines them unconditionally so saves made with the mod keep
/// loading; `SimulationConfig::mod_elements_enabled` only decides whether the
/// player may create new ones.
#[inline]
pub fn is_mod_element(ty: u32) -> bool {
    (PT_FNTC..=PT_MGNT).contains(&ty)
}

/// Number of built-in element definitions (`PT_NONE ..= PT_MGNT`).
///
/// Derived from the last element id so appending a definition only needs its
/// `PT_*` constant (and a table row).
pub const BUILTIN_ELEMENT_COUNT: usize = PT_MGNT as usize + 1;

// Built-in element IDs must fit inside the 9-bit type field of a packed pmap.
const _: () = assert!(BUILTIN_ELEMENT_COUNT <= PT_NUM);
// The GPU upload keeps one 128-byte row per slot; cap the table at 256
// (`PT_NUM * 128 == 32 768` bytes).
const _: () = assert!(PT_NUM <= 256);
// `pmap_pack` stores the element type in 9 bits, so no slot may exceed 511.
const _: () = assert!(PT_NUM <= 512);

/// Matter state: particle (powder-like faller).
pub const TYPE_PART: u32 = 0x0000_0001;
/// Matter state: liquid.
pub const TYPE_LIQUID: u32 = 0x0000_0002;
/// Matter state: solid.
pub const TYPE_SOLID: u32 = 0x0000_0004;
/// Matter state: gas.
pub const TYPE_GAS: u32 = 0x0000_0008;
/// Matter state: energy (photons, neutrons, electrons, sparks).
pub const TYPE_ENERGY: u32 = 0x0000_0010;
/// Mask covering every matter-state bit.
pub const STATE_FLAGS: u32 = 0x0000_001F;

/// Conducts electricity.
pub const PROP_CONDUCTS: u32 = 0x0000_0020;
/// Photons pass through.
pub const PROP_PHOTPASS: u32 = 0x0000_0040;
/// Neutrons penetrate.
pub const PROP_NEUTPENETRATE: u32 = 0x0000_0080;
/// Absorbs neutrons.
pub const PROP_NEUTABSORB: u32 = 0x0000_0100;
/// Neutrons pass through.
pub const PROP_NEUTPASS: u32 = 0x0000_0200;
/// Deadly to the player / other particles on contact.
pub const PROP_DEADLY: u32 = 0x0000_0400;
/// Glows when hot.
pub const PROP_HOT_GLOW: u32 = 0x0000_0800;
/// Element is a Game of Life ruleset holder (`LIFE`).
pub const PROP_LIFE: u32 = 0x0000_1000;
/// Emits radiation (decays / irradiates neighbours).
pub const PROP_RADIOACTIVE: u32 = 0x0000_2000;
/// Life decreases every step.
pub const PROP_LIFE_DEC: u32 = 0x0000_4000;
/// Particle is killed when its life reaches zero.
pub const PROP_LIFE_KILL: u32 = 0x0000_8000;
/// Kills the particle when life reaches zero and decrements life each step.
pub const PROP_LIFE_KILL_DEC: u32 = 0x0001_0000;
/// Spark settles into the element's `ctype` after discharging.
pub const PROP_SPARKSETTLE: u32 = 0x0002_0000;
/// Hidden from the "replace" drawing mode.
pub const PROP_NOCTYPEDRAW: u32 = 0x0010_0000;
/// Excluded from ambient heat exchange.
pub const PROP_NOAMBHEAT: u32 = 0x0020_0000;

/// Particle has already moved this frame.
pub const FLAG_STAGNANT: u32 = 0x01;
/// Skip the movement pass for this particle.
pub const FLAG_SKIPMOVE: u32 = 0x02;
/// Particle slot is currently held inside a portal queue (`PRTI` -> `PRTO`).
pub const FLAG_PORTAL_HELD: u32 = 0x04;
/// Particle is allowed to move.
pub const FLAG_MOVABLE: u32 = 0x08;
/// Particle is a rigid-body pixel owned by a `RigidBody2D`.
pub const FLAG_RIGID_PIXEL: u32 = 0x10;
/// `Simulation.cpp:2335-2338`: the particle sits in an unpowered `WL_STASIS`
/// cell and is skipped entirely (no reaction, no life countdown, no movement).
pub const FLAG_STASIS: u32 = 0x20;

/// Life a freshly placed `PT_SPRK` starts with.
///
/// A spark discharges after four ticks; the reaction kernel decrements `life`
/// and restores `ctype` (the conductor it travelled through) once it hits zero.
pub const SPARK_LIFE: i32 = 4;

/// Temperature channels for `PRTI` / `PRTO` / `WIFI` (`int(MAX_TEMP - 73) / 100 + 2`).
pub const PORTAL_CHANNELS: usize = 103;
/// Neighbour lanes per portal channel (`0..8`).
pub const PORTAL_DIRS: usize = 8;
/// Particle storage capacity per `(channel, dir)` lane (`portalp[ch][dir][80]`).
pub const PORTAL_SLOTS: usize = 80;
/// Total portal particle slots across all channels and directions (`65 920`).
pub const PORTAL_CAPACITY: usize = PORTAL_CHANNELS * PORTAL_DIRS * PORTAL_SLOTS;
/// `u32` payload words per portal slot (`life`, `ctype`, `temp`, `tmp`, `tmp2`, `vx`, `vy`, `dcolour`).
pub const PORTAL_WORDS: usize = 8;

/// Maximum concurrent `FIGH` fighters (`Stickman.h` `MAX_FIGHTERS`).
pub const STKM_MAX_FIGHTERS: usize = 100;
/// Stickman table index for `STKM` (`sim->player`).
pub const STKM_PLAYER1: usize = 100;
/// Stickman table index for `STKM2` (`sim->player2`).
pub const STKM_PLAYER2: usize = 101;
/// Total entries in the stickman SoA tables (`fighters[0..100]` + `player` + `player2`).
pub const STKM_SLOTS: usize = 102;
/// `f32` words per stickman in `gpu_stkm_legs` (`legs[16]`).
pub const STKM_LEGS_STRIDE: usize = 16;
/// `f32` words per stickman in `gpu_stkm_accs` (`accs[8]`).
pub const STKM_ACCS_STRIDE: usize = 8;
/// `u32` words per stickman in `gpu_stkm_meta` (`spwn`, `comm`, `pcomm`, `elem`, `frames`, `rocket_boots`, `fan`, `spawn_id`).
pub const STKM_META_STRIDE: usize = 8;

/// Erases particles that touch the wall.
pub const WL_ERASE: u8 = 0;
/// Conducts electricity.
pub const WL_WALLELEC: u8 = 1;
/// Basic wall that blocks everything.
pub const WL_EWALL: u8 = 2;
/// Detects particles and sparks when hit.
pub const WL_DETECT: u8 = 3;
/// Streams particles in the direction of its fan velocity.
pub const WL_STREAM: u8 = 4;
/// Fan: pushes air and particles.
pub const WL_FAN: u8 = 5;
/// Blocks only liquids.
pub const WL_ALLOWLIQUID: u8 = 6;
/// Destroys every particle that touches the wall.
pub const WL_DESTROYALL: u8 = 7;
/// Basic wall (no air flow).
pub const WL_WALL: u8 = 8;
/// Blocks only air.
pub const WL_ALLOWAIR: u8 = 9;
/// Blocks only powders.
pub const WL_ALLOWPOWDER: u8 = 10;
/// Conducts electricity in every direction.
pub const WL_ALLOWALLELEC: u8 = 11;
/// Absorbs energy particles.
pub const WL_EHOLE: u8 = 12;
/// Blocks only gases.
pub const WL_ALLOWGAS: u8 = 13;
/// Applies gravity to particles.
pub const WL_GRAV: u8 = 14;
/// Blocks only energy particles.
pub const WL_ALLOWENERGY: u8 = 15;
/// Blocks air flow.
pub const WL_BLOCKAIR: u8 = 16;
/// Erases everything (particles and walls).
pub const WL_ERASEALL: u8 = 17;
/// Freezes particles in place.
pub const WL_STASIS: u8 = 18;
/// Number of built-in wall types.
pub const WL_NUM: usize = 19;

/// Overlay air velocity direction (air colouring).
pub const DISPLAY_AIRC: u32 = 0x01;
/// Overlay air pressure.
pub const DISPLAY_AIRP: u32 = 0x02;
/// Overlay air velocity magnitude.
pub const DISPLAY_AIRV: u32 = 0x04;
/// Overlay ambient heat.
pub const DISPLAY_AIRH: u32 = 0x08;
/// Gravity lensing (Newtonian gravity bends light).
pub const DISPLAY_WARP: u32 = 0x10;
/// Persistence / glow trails.
pub const DISPLAY_PERS: u32 = 0x20;
/// `DISPLAY_EFFE` (`ElementGraphics.h:56`): "Enables moving solids, stickmen
/// guns, and premium(tm) graphics" (`RenderView.cpp:97`). The reference
/// renderer never reads the bit — the effect passes are selected by the
/// `RENDER_*` preset — so this only records the user's choice.
pub const DISPLAY_EFFE: u32 = 0x40;
/// Vorticity overlay.
pub const DISPLAY_AIRW: u32 = 0x80;

/// Element base colours.
pub const COLOUR_DEFAULT: u32 = 0x00;
/// Colour by temperature.
pub const COLOUR_HEAT: u32 = 0x01;
/// Colour by remaining life.
pub const COLOUR_LIFE: u32 = 0x02;
/// Velocity gradient colouring.
pub const COLOUR_GRAD: u32 = 0x04;
/// Basic / simple colouring.
pub const COLOUR_BASC: u32 = 0x08;
/// Colour by destructibility score.
pub const COLOUR_DEST: u32 = 0x10;

/// Basic particle rendering.
pub const RENDER_BASC: u32 = 0x0000_0001;
/// Additive fire/glow effects for gases.
pub const RENDER_FIRE: u32 = 0x0000_0002;
/// Spark/lightning effects.
pub const RENDER_SPRK: u32 = 0x0000_0004;
/// Glow effect on emissive elements.
pub const RENDER_GLOW: u32 = 0x0000_0008;
/// Blur effect for liquids.
pub const RENDER_BLUR: u32 = 0x0000_0010;
/// Blob rendering for powders/liquids.
pub const RENDER_BLOB: u32 = 0x0000_0020;
/// Special flare effects.
pub const RENDER_EFFE: u32 = 0x0000_0040;
/// Render nothing but flat colours.
pub const RENDER_NONE: u32 = 0x0000_0080;
/// Default render mode: basic + fire + spark + effects.
pub const RENDER_DEFAULT: u32 = RENDER_BASC | RENDER_FIRE | RENDER_SPRK | RENDER_EFFE;
/// Fancy display preset.
pub const RENDER_FANCY: u32 =
    RENDER_BASC | RENDER_FIRE | RENDER_SPRK | RENDER_GLOW | RENDER_BLUR | RENDER_EFFE;

/// Kelvin temperature scale.
pub const TEMPSCALE_KELVIN: u32 = 0;
/// Celsius temperature scale.
pub const TEMPSCALE_CELSIUS: u32 = 1;
/// Fahrenheit temperature scale.
pub const TEMPSCALE_FAHRENHEIT: u32 = 2;

/// No air heat convection.
pub const AIRC_NONE: u32 = 0;
/// Legacy air heat convection.
pub const AIRC_LEGACY: u32 = 1;
/// Boussinesq air heat convection (default).
pub const AIRC_BOUSSINESQ: u32 = 2;

/// Packs a particle index and element type into one spatial-map word.
///
/// The low 9 bits store the element type, the upper 23 bits store `part_id + 1`
/// so that `0` unambiguously means "no particle". The id field therefore holds
/// at most `0x7FFFFF`: `part_id <= 0x7FFFFE` (`8 388 606`) is the largest slot
/// the map can address, which is why the particle pool saturates at
/// `8 388 607` slots. A larger id would silently alias into the type field.
#[inline]
pub const fn pmap_pack(part_id: u32, elem_type: u32) -> u32 {
    ((part_id + 1) << 9) | (elem_type & 0x1FF)
}

/// Extracts the element type from a packed spatial-map word.
#[inline]
pub const fn pmap_type(pmap_val: u32) -> u32 {
    pmap_val & 0x1FF
}

/// Extracts the particle index from a packed spatial-map word.
#[inline]
pub const fn pmap_id(pmap_val: u32) -> Option<u32> {
    let raw = pmap_val >> 9;
    if raw == 0 { None } else { Some(raw - 1) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mod_element_range_matches_the_cracker_appendix() {
        // The reference table ends at `SEED`; the mod appends 44 ids after it.
        assert!(!is_mod_element(PT_SEED));
        assert!(is_mod_element(PT_FNTC));
        assert!(is_mod_element(PT_MGNT));
        assert!(!is_mod_element(PT_MGNT + 1));
        assert_eq!(PT_MGNT - PT_FNTC + 1, 44);
        assert_eq!(PT_MGNT as usize + 1, BUILTIN_ELEMENT_COUNT);
    }
}
