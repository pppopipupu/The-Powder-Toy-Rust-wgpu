//! Data-driven element table for the 3D voxel mode.
//!
//! The simulation kernels and the renderer share one fixed table layout: every
//! element id owns one row of [`GPU_ELEMENT_WORDS`] `u32` words, and the row
//! index is the element id. Rows without a [`VoxelElement`] are all zero, so a
//! kernel can index the table without bounds checks.
//!
//! ```text
//! word  field
//! 0     id
//! 1     phase
//! 2     albedo r
//! 3     albedo g
//! 4     albedo b
//! 5     albedo a
//! 6     sigma
//! 7     emission r
//! 8     emission g
//! 9     emission b
//! 10    density
//! 11    heat_conduct
//! ...   see the `ELEM_*` word offsets below
//! ```
//!
//! Float fields are stored as [`f32::to_bits`]; product ids and the flag word
//! stay plain integers. Keeping the floats bit-exact means the CPU reference
//! code and the GPU kernels observe the same constants.
//!
//! The table holds fifty-nine representative elements, each assigned to the
//! [`VoxelCategory`] of its TPT `MenuSection`. Every transition slot is either a real
//! threshold/product pair or explicitly disabled with a `0` threshold and
//! [`PT_NONE`]. The values are copied from the C++ element definitions
//! (`Colour`, `HeatConduct`, `HighTemperature`/`LowTemperature`, ...); the
//! simulation kernel consumes the threshold/product slots to drive the local
//! reactions (water/ice/steam, lava solidifying, and the element-specific
//! ignition/melt branches).
//!
//! `sigma`, `emission` and `density` have no C++ counterpart (C++ has no
//! volumetric absorption and its glow comes from per-element `graphics`
//! functions); those fields are 3D-renderer-only constructs and are not C++
//! ports — see `PLUT.cpp` (`PMODE_FLAT`, no glow) for the contrast.
//!
//! [`VoxelElement::category`] is host-side menu grouping only: it never reaches
//! [`pack_row`], so the GPU row layout is unchanged.

use crate::constants::{
    PT_ACID, PT_AMTR, PT_BANG, PT_BCOL, PT_BMTL, PT_BOMB, PT_BREC, PT_BRMT, PT_C4, PT_CLNE, PT_CO2,
    PT_COAL, PT_DEST, PT_DUST, PT_EMBR, PT_EXOT, PT_FIRE, PT_FSEP, PT_FUSE, PT_GAS, PT_GBMB,
    PT_GLAS, PT_GPMP, PT_GRVT, PT_GUNP, PT_H2, PT_ICEI, PT_IGNT, PT_ISOZ, PT_ISZS, PT_LAVA,
    PT_MERC, PT_METL, PT_NBHL, PT_NEUT, PT_NITR, PT_NONE, PT_NSCN, PT_NWHL, PT_O2, PT_OIL, PT_PHOT,
    PT_PLSM, PT_PLUT, PT_PSCN, PT_SALT, PT_SAND, PT_SING, PT_SLTW, PT_SMKE, PT_SNOW, PT_SPRK,
    PT_STNE, PT_THRM, PT_URAN, PT_VOID, PT_WARP, PT_WATR, PT_WOOD, PT_WTRV,
};

/// Number of `u32` words in one GPU element-table row.
pub const GPU_ELEMENT_WORDS: usize = 26;
/// Number of rows in the GPU element table (one per possible 8-bit type).
pub const GPU_ELEMENT_ROWS: usize = 256;
/// Number of `f32` components in the RGBA palette (one `vec4` per type).
pub const GPU_PALETTE_FLOATS: usize = GPU_ELEMENT_ROWS * 4;

/// Word 0: element id.
pub const ELEM_ID: usize = 0;
/// Word 1: [`VoxelPhase`] discriminant.
pub const ELEM_PHASE: usize = 1;
/// Word 2: albedo red.
pub const ELEM_ALBEDO_R: usize = 2;
/// Word 3: albedo green.
pub const ELEM_ALBEDO_G: usize = 3;
/// Word 4: albedo blue.
pub const ELEM_ALBEDO_B: usize = 4;
/// Word 5: albedo alpha. All v1 entries store `1.0`; medium opacity comes from
/// [`ELEM_SIGMA`] and [`ELEM_FLAGS`].
pub const ELEM_ALBEDO_A: usize = 5;
/// Word 6: volumetric absorption coefficient.
pub const ELEM_SIGMA: usize = 6;
/// Word 7: emission red.
pub const ELEM_EMISSION_R: usize = 7;
/// Word 8: emission green.
pub const ELEM_EMISSION_G: usize = 8;
/// Word 9: emission blue.
pub const ELEM_EMISSION_B: usize = 9;
/// Word 10: density in kg/m^3.
pub const ELEM_DENSITY: usize = 10;
/// Word 11: TPT `HeatConduct` (0-255 int, not W/(m*K)).
pub const ELEM_HEAT_CONDUCT: usize = 11;
/// Word 12: ignition temperature in K, `0` when the element cannot burn.
pub const ELEM_IGNITION: usize = 12;
/// Word 13: product left when a burning element finishes, `PT_NONE` if unused.
pub const ELEM_BURN_PRODUCT: usize = 13;
/// Word 14: melting temperature in K, `0` when unused.
pub const ELEM_MELT: usize = 14;
/// Word 15: product of the melting transition, `PT_NONE` when unused.
pub const ELEM_MELT_PRODUCT: usize = 15;
/// Word 16: boiling temperature in K, `0` when unused.
pub const ELEM_BOIL: usize = 16;
/// Word 17: product of the boiling transition, `PT_NONE` when unused.
pub const ELEM_BOIL_PRODUCT: usize = 17;
/// Word 18: freezing temperature in K, `0` when unused.
pub const ELEM_FREEZE: usize = 18;
/// Word 19: product of the freezing transition, `PT_NONE` when unused.
pub const ELEM_FREEZE_PRODUCT: usize = 19;
/// Word 20: relative hardness (TPT `0..=255` scale).
pub const ELEM_HARDNESS: usize = 20;
/// Word 21: explosion radius in voxels, `0` when the element is inert.
pub const ELEM_EXPLOSION: usize = 21;
/// Word 22: bitwise OR of the `ELEM_FLAG_*` bits.
pub const ELEM_FLAGS: usize = 22;
/// Word 23: TPT `Flammable` chance per fire exposure, `0..=1000`.
pub const ELEM_FLAMMABLE: usize = 23;
/// Word 24: TPT `Explosive` bitmask (`1` = fire ignition, `2` = pressure).
pub const ELEM_EXPLOSIVE: usize = 24;
/// Word 25: TPT property bits (`ELEM_TPT_*`), used by the life pass and the
/// element branches. See `ELEM_TPT_LIFE_DEC` and friends.
pub const ELEM_TPT_PROPS: usize = 25;

/// TPT `PROP_LIFE_DEC`: life decreases by one every tick while positive.
pub const ELEM_TPT_LIFE_DEC: u32 = 1 << 0;
/// TPT `PROP_LIFE_KILL`: the particle is killed when life is `<= 0`.
pub const ELEM_TPT_LIFE_KILL: u32 = 1 << 1;
/// TPT `PROP_LIFE_KILL_DEC`: killed when the decrement reaches zero.
pub const ELEM_TPT_LIFE_KILL_DEC: u32 = 1 << 2;
/// TPT `PROP_SPARKSETTLE`: embers may settle next to this element and do not
/// kill themselves on contact with it (`EMBR.cpp:68-75`).
pub const ELEM_TPT_SPARKSETTLE: u32 = 1 << 3;
/// TPT `PROP_CONDUCTS`: the element can carry a spark.
pub const ELEM_TPT_CONDUCTS: u32 = 1 << 4;
/// TPT `PROP_HOT_GLOW`: the element glows when hot.
pub const ELEM_TPT_HOT_GLOW: u32 = 1 << 5;
/// TPT `PROP_NEUTPENETRATE`: neutrons pass through.
pub const ELEM_TPT_NEUTPENETRATE: u32 = 1 << 6;
/// TPT `PROP_NEUTPASS`: neutrons pass through without absorption.
pub const ELEM_TPT_NEUTPASS: u32 = 1 << 7;
/// TPT `PROP_PHOTPASS`: photons pass through.
pub const ELEM_TPT_PHOTPASS: u32 = 1 << 8;
/// TPT `CarriesTypeIn = 1U << FIELD_CTYPE`: a type change keeps `ctype`.
pub const ELEM_TPT_CARRIES_CTYPE: u32 = 1 << 9;

/// TPT `MAX_TEMP`; the packed temperature field saturates below it (see
/// [`crate::voxel3d::packing::TEMP_Q_SCALE`]).
pub const TPT_MAX_TEMP_K: f32 = 9999.0;
/// TPT `R_TEMP` ambient offset: `R_TEMP + 273.15` is the default particle
/// temperature (`Element.cpp:54`).
pub const TPT_DEFAULT_TEMP_K: f32 = 295.15;

/// Flag bit: the element is part of the volumetric accumulation set.
pub const ELEM_FLAG_TRANSLUCENT: u32 = 1 << 0;
/// Flag bit: [`VoxelPhase::Static`].
pub const ELEM_FLAG_STATIC: u32 = 1 << 1;
/// Flag bit: [`VoxelPhase::Powder`].
pub const ELEM_FLAG_POWDER: u32 = 1 << 2;
/// Flag bit: [`VoxelPhase::Liquid`].
pub const ELEM_FLAG_LIQUID: u32 = 1 << 3;
/// Flag bit: [`VoxelPhase::Gas`].
pub const ELEM_FLAG_GAS: u32 = 1 << 4;
/// Flag bit: [`VoxelPhase::Fire`].
pub const ELEM_FLAG_FIRE: u32 = 1 << 5;
/// Flag bit: the element takes part in the loose-voxel movement passes.
pub const ELEM_FLAG_LOOSE: u32 = 1 << 6;
/// Flag bit: the element has a burn product.
pub const ELEM_FLAG_COMBUSTIBLE: u32 = 1 << 7;
/// Flag bit: the element has a non-zero explosion radius.
pub const ELEM_FLAG_EXPLOSIVE: u32 = 1 << 8;

const _: () = assert!(GPU_ELEMENT_WORDS == ELEM_TPT_PROPS + 1);
const _: () = assert!(GPU_PALETTE_FLOATS == GPU_ELEMENT_ROWS * 4);

/// Simulation class of an element.
///
/// The discriminants are part of the GPU table contract and must not be
/// reordered; `Empty` doubles as the "unknown id" fallback.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VoxelPhase {
    /// No voxel.
    #[default]
    Empty = 0,
    /// Permanently fixed solid (wood, ice, metal, glass, ...).
    Static = 1,
    /// Falling powder (sand, dust, gunpowder).
    Powder = 2,
    /// Flowing liquid (water, lava, oil).
    Liquid = 3,
    /// Rising gas (steam, smoke).
    Gas = 4,
    /// Burning energy (fire).
    Fire = 5,
    /// Moving energy particle (photon, neutron, graviton); C++ `TYPE_ENERGY`.
    ///
    /// Energy has no counterpart in the solid/powder/liquid/gas movement
    /// classes: the element update owns its transport, so this phase derives
    /// no `ELEM_FLAG_*` movement bit.
    Energy = 6,
}

/// Menu grouping of an element.
///
/// This is host-side metadata for the 3D palette; it is deliberately absent
/// from [`VoxelElement`]'s GPU row so the shader layout stays fixed.
///
/// The variants are the TPT menu sections (`SimulationData.cpp:79-95`
/// `msections`, following each element's `MenuSection` in
/// `src/simulation/elements/*.cpp`) that hold elements the 3D palette can
/// spawn, plus `Tools` (which holds the palette's actions, matching the 2D
/// bottom bar's `Tools` menu section). `Sensors` and `Life` are kept for the
/// palette even though no 3D element uses them yet; every other section is
/// populated from the C++ `MenuSection` line. The discriminants are stable and
/// [`VOXEL_CATEGORIES`] uses this order.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VoxelCategory {
    /// TPT `SC_ELEC` (`METL.cpp`).
    Electronics = 0,
    /// TPT `SC_POWERED` ("Powered Materials").
    Powered = 1,
    /// TPT `SC_SENSOR`.
    Sensors = 2,
    /// TPT `SC_FORCE`.
    Force = 3,
    /// TPT `SC_EXPLOSIVE` (`FIRE.cpp`, `GUNP.cpp`, ...).
    Explosives = 4,
    /// TPT `SC_GAS`.
    Gases = 5,
    /// TPT `SC_LIQUID`.
    Liquids = 6,
    /// TPT `SC_POWDERS`.
    #[default]
    Powders = 7,
    /// TPT `SC_SOLIDS`.
    Solids = 8,
    /// TPT `SC_NUCLEAR` (`PLUT.cpp`).
    Radioactive = 9,
    /// TPT `SC_SPECIAL`.
    Special = 10,
    /// TPT `SC_LIFE` ("Game Of Life").
    Life = 11,
    /// TPT `SC_TOOL`: palette actions, not elements.
    Tools = 12,
}

/// Every [`VoxelCategory`], in palette order.
pub const VOXEL_CATEGORIES: [VoxelCategory; 13] = [
    VoxelCategory::Electronics,
    VoxelCategory::Powered,
    VoxelCategory::Sensors,
    VoxelCategory::Force,
    VoxelCategory::Explosives,
    VoxelCategory::Gases,
    VoxelCategory::Liquids,
    VoxelCategory::Powders,
    VoxelCategory::Solids,
    VoxelCategory::Radioactive,
    VoxelCategory::Special,
    VoxelCategory::Life,
    VoxelCategory::Tools,
];

/// One row of the element table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelElement {
    /// Powder Toy element id (`PT_*`).
    pub id: u32,
    /// Short display name.
    pub name: &'static str,
    /// Simulation class.
    pub phase: VoxelPhase,
    /// Host-side menu grouping; never packed into the GPU row.
    pub category: VoxelCategory,
    /// Diffuse albedo, linear RGB in `0..=1`.
    pub albedo: [f32; 3],
    /// Volumetric absorption coefficient in inverse voxels; `0` is opaque
    /// enough to end the ray, non-zero participates in volume accumulation.
    /// 3D-renderer-only; TPT has no volumetric absorption.
    pub sigma: f32,
    /// Emissive radiance added by the volume renderer. 3D-renderer-only;
    /// TPT glow comes from per-element `graphics` functions.
    pub emission: [f32; 3],
    /// Density in kg/m^3. 3D-renderer-only; C++ uses `Weight` (1-100) instead.
    pub density: f32,
    /// TPT `HeatConduct` (0-255 int). NOT W/(m*K); copied from each element's
    /// `HeatConduct = N;` line (e.g. `SAND.cpp:36`).
    pub heat_conduct: f32,
    /// TPT `HighTemperature` in K, `0` when the element has no such change.
    pub ignition_k: f32,
    /// Product of the high-temperature transition, or [`PT_NONE`].
    pub burn_product: u32,
    /// Melting temperature in K; `0` when unused.
    pub melt_k: f32,
    /// Product of the melting transition, or [`PT_NONE`].
    pub melt_product: u32,
    /// Boiling temperature in K; `0` when unused.
    pub boil_k: f32,
    /// Product of the boiling transition, or [`PT_NONE`].
    pub boil_product: u32,
    /// Freezing/quench temperature in K; `0` when unused.
    pub freeze_k: f32,
    /// Product of the freezing/quench transition, or [`PT_NONE`].
    pub freeze_product: u32,
    /// Relative hardness on TPT's `0..=255` scale.
    pub hardness: f32,
    /// Explosion radius in voxels; `0` when the element is inert.
    pub explosion_radius: f32,
}

/// The fifty-nine elements, each tagged with its TPT [`VoxelCategory`].
///
/// The order is public-facing (the palette uses it); GPU rows are selected by
/// [`VoxelElement::id`], not by position.
pub const VOXEL_ELEMENTS: [VoxelElement; 59] = [
    VoxelElement {
        id: PT_SAND,
        name: "SAND",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Powders,
        albedo: [1.0, 0.8157, 0.5647], // `SAND.cpp:7` `Colour = 0xFFD090`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 1600.0,
        heat_conduct: 150.0, // `SAND.cpp:29` `HeatConduct = 150`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // `SAND.cpp:40-41` `HighTemperature = 1973` -> `PT_LAVA` (ctype GLAS).
        melt_k: 1973.0,
        melt_product: PT_LAVA,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_DUST, // `DUST.cpp:7,30,25`: Colour 0xFFE0A0, HeatConduct 70, Hardness 30.
        name: "DUST",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Powders,
        albedo: [1.0, 0.8784, 0.6275],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 600.0,
        heat_conduct: 70.0,
        // DUST has no HighTemperature transition in TPT; it is only an
        // ignition target with Flammable=10.
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 30.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_SNOW, // `SNOW.cpp:9,33,32,36`: Colour 0xC0E0FF, HeatConduct 46, temp 265.15, NEUTPASS.
        name: "SNOW",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Powders,
        albedo: [0.7529, 0.8784, 1.0], // `SNOW.cpp:6` `Colour = 0xC0E0FF`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 400.0,
        heat_conduct: 46.0, // `SNOW.cpp:34` `HeatConduct = 46`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // Snow melts just above freezing; the reverse transition lives in the
        // water row's freeze slot.
        melt_k: 252.05, // `SNOW.cpp:45` `HighTemperature = 252.05` -> ST
        melt_product: PT_WATR,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 20.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_SALT, // `SALT.cpp:7,29,40`: Colour 0xFFFFFF, HeatConduct 110, melt 1173.
        name: "SALT",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Powders,
        albedo: [1.0, 1.0, 1.0], // `SALT.cpp:6` `Colour = 0xFFFFFF`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 2170.0,
        heat_conduct: 110.0, // `SALT.cpp` `HeatConduct = 110`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // TPT salt liquefies at its 1173 K high-temperature transition.
        melt_k: 1173.0,
        melt_product: PT_LAVA,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_COAL, // `COAL.cpp:8,31,26,45-46`: Colour 0x222222, HeatConduct 200, Hardness 18, life 110.
        name: "COAL",
        phase: VoxelPhase::Static,
        // `COAL.cpp:9`: `MenuSection = SC_SOLIDS` (a solid despite coal-burning).
        category: VoxelCategory::Solids,
        albedo: [0.1333, 0.1333, 0.1333],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 1300.0,
        heat_conduct: 200.0,
        // COAL has no HighTemperature transition and Flammable=0. It burns
        // through its 110-tick life in the reaction pass and drops FIRE.
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 18.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_WATR, // `WATR.cpp:9,32,35,41,43`: Colour 0x2030D0, HeatConduct 29, freeze 273.15, boil 373.
        name: "WATR",
        phase: VoxelPhase::Liquid,
        category: VoxelCategory::Liquids,
        albedo: [0.1255, 0.1882, 0.8157], // `WATR.cpp:9` `Colour = 0x2030D0`
        sigma: 0.35,
        emission: [0.0, 0.0, 0.0],
        density: 1000.0,
        heat_conduct: 29.0, // `WATR.cpp:29` `HeatConduct = 29`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // Water <-> ice <-> steam. The kernel adds the small hysteresis band;
        // the table only stores the nominal transition point.
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 373.0,
        boil_product: PT_WTRV,
        freeze_k: 273.15, // `WATR.cpp:42` `LowTemperature = 273.15` -> ICEI
        freeze_product: PT_ICEI,
        hardness: 20.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_LAVA, // `LAVA.cpp:11,35,29,34,38,80-83`: Colour 0xE05010, HeatConduct 60, life 240..359.
        name: "LAVA",
        phase: VoxelPhase::Liquid,
        category: VoxelCategory::Liquids,
        albedo: [0.8784, 0.3137, 0.0627],
        sigma: 0.0,
        emission: [4.0, 0.75, 0.04],
        density: 2600.0,
        heat_conduct: 60.0,
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        // Cooling below 973 K solidifies lava. The water-adjacency branch
        // reuses the same product.
        freeze_k: 973.0,
        freeze_product: PT_STNE,
        hardness: 2.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_OIL, // `OIL.cpp:7,29,25,32,40`: Colour 0x404010, HeatConduct 42, Flammable 20, melt 333.
        name: "OIL",
        phase: VoxelPhase::Liquid,
        category: VoxelCategory::Liquids,
        albedo: [0.2510, 0.2510, 0.0627],
        sigma: 0.60,
        emission: [0.0, 0.0, 0.0],
        density: 900.0,
        heat_conduct: 42.0,
        // TPT OIL has Flammable=20 and vaporises to GAS at 333 K.
        ignition_k: 333.0,
        burn_product: PT_GAS,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 5.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_SLTW, // `SLTW.cpp:9,31,34,40,42`: Colour 0x4050F0, HeatConduct 75, freeze 252.05, boil 383.
        name: "SLTW",
        phase: VoxelPhase::Liquid,
        category: VoxelCategory::Liquids,
        albedo: [0.2510, 0.3137, 0.9412], // `SLTW.cpp:9` `Colour = 0x4050F0`
        sigma: 0.35,
        emission: [0.0, 0.0, 0.0],
        density: 1025.0,
        heat_conduct: 75.0, // `SLTW.cpp:31` `HeatConduct = 75`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        // `SLTW.cpp:42` `HighTemperature = 383` -> `ST`; `Simulation.cpp:2559-2562`
        // resolves `ST` to SALT 1/4 and WTRV 3/4, so the table stores WTRV as
        // the majority product. `SLTW.cpp:40` freezes at the 252.05 K eutectic.
        boil_k: 383.0,
        boil_product: PT_WTRV,
        freeze_k: 252.05,
        freeze_product: PT_ICEI,
        hardness: 20.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_ACID, // `ACID.cpp:10,33,25,28,47`: Colour 0xED55FF, HeatConduct 34, Flammable 40, life 75.
        name: "ACID",
        phase: VoxelPhase::Liquid,
        category: VoxelCategory::Liquids,
        albedo: [0.9294, 0.3333, 1.0], // `ACID.cpp:10` `Colour = 0xED55FF`
        sigma: 0.40,
        emission: [0.0, 0.0, 0.0],
        density: 1050.0,
        heat_conduct: 34.0, // `ACID.cpp:33` `HeatConduct = 34`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_MERC, // `MERC.cpp:9,31,27,34,45`: Colour 0x736B6D, HeatConduct 251, Hardness 18, tmp 10.
        name: "MERC",
        phase: VoxelPhase::Liquid,
        category: VoxelCategory::Liquids,
        albedo: [0.4510, 0.4196, 0.4275], // `MERC.cpp:6` `Colour = 0x736B6D`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 13593.0,
        heat_conduct: 251.0, // `MERC.cpp:31` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 18.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_WTRV, // `WTRV.cpp:9,32,31,42`: Colour 0xA0A0FF, HeatConduct 48, temp 395.15, freeze 371.
        name: "WTRV",
        phase: VoxelPhase::Gas,
        category: VoxelCategory::Gases,
        albedo: [0.6275, 0.6275, 1.0], // `WTRV.cpp:6` `Colour = 0xA0A0FF`
        sigma: 0.15,
        emission: [0.0, 0.0, 0.0],
        density: 0.6,
        heat_conduct: 48.0, // `WTRV.cpp:33` `HeatConduct = 48`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        // `WTRV.cpp:41-42` `LowTemperature = 371` -> `ST`, which
        // `Simulation.cpp:2617-2621` resolves to RIME below 273 K and DSTW
        // above; neither is ported, so the 3D table condenses to WATR.
        freeze_k: 371.0,
        freeze_product: PT_WATR,
        hardness: 4.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_SMKE, // `SMKE.cpp:9,32,31,35,43`: Colour 0x222222, HeatConduct 88, temp 615.15, life 250..269.
        name: "SMKE",
        phase: VoxelPhase::Gas,
        category: VoxelCategory::Gases,
        albedo: [0.1333, 0.1333, 0.1333], // `SMKE.cpp:6` `Colour = 0x222222`
        sigma: 0.50,
        emission: [0.0, 0.0, 0.0],
        density: 0.5,
        heat_conduct: 88.0, // `SMKE.cpp:32` `HeatConduct = 88`
        // `SMKE.cpp:43-44` `HighTemperature = 625` -> `PT_FIRE`.
        ignition_k: 625.0,
        burn_product: PT_FIRE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_H2, // `H2.cpp:9,31,34`: Colour 0x5070FF, HeatConduct 251, PHOTPASS.
        name: "H2",
        phase: VoxelPhase::Gas,
        category: VoxelCategory::Gases,
        albedo: [0.3137, 0.4392, 1.0], // `H2.cpp:9` `Colour = 0x5070FF`
        sigma: 0.08,
        emission: [0.0, 0.0, 0.0],
        density: 0.09,
        heat_conduct: 251.0, // `H2.cpp:34` `HeatConduct = 251`
        // H2 has Flammable=0 and no HighTemperature transition; its burn is
        // the dedicated H2 + FIRE/PLSM/LAVA branch.
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_O2, // `O2.cpp:9,31`: Colour 0x80A0FF, HeatConduct 70.
        name: "O2",
        phase: VoxelPhase::Gas,
        category: VoxelCategory::Gases,
        albedo: [0.5020, 0.6275, 1.0], // `O2.cpp:6` `Colour = 0x80A0FF`
        sigma: 0.08,
        emission: [0.0, 0.0, 0.0],
        density: 1.43,
        heat_conduct: 70.0, // `O2.cpp` `HeatConduct = 70`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_CO2, // `CO2.cpp:9,31,40`: Colour 0x666666, HeatConduct 88, freeze 194.65 (DRIC, not ported).
        name: "CO2",
        phase: VoxelPhase::Gas,
        category: VoxelCategory::Gases,
        albedo: [0.4, 0.4, 0.4], // `CO2.cpp:9` `Colour = 0x666666`
        sigma: 0.12,
        emission: [0.0, 0.0, 0.0],
        density: 1.98,
        heat_conduct: 88.0, // `CO2.cpp` `HeatConduct = 88`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_GAS, // `GAS.cpp:7,30,29,33,41`: Colour 0xE0FF20, HeatConduct 42, temp 297.15, melt 573.
        name: "GAS",
        phase: VoxelPhase::Gas,
        category: VoxelCategory::Gases,
        albedo: [0.8784, 1.0, 0.1255],
        sigma: 0.10,
        emission: [0.0, 0.0, 0.0],
        density: 0.72,
        heat_conduct: 42.0,
        // TPT GAS has Flammable=600 and a 573 K HighTemperature -> FIRE.
        ignition_k: 573.0,
        burn_product: PT_FIRE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_STNE, // `STNE.cpp:7,29,25,32,40`: Colour 0xA0A0A0, HeatConduct 150, TYPE_PART.
        name: "STNE",
        phase: VoxelPhase::Powder, // `STNE.cpp:35` `TYPE_PART`
        // `STNE.cpp`: `MenuSection = SC_POWDERS`.
        category: VoxelCategory::Powders,
        albedo: [0.6275, 0.6275, 0.6275], // `STNE.cpp:7` `Colour = 0xA0A0A0`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 2600.0,
        heat_conduct: 150.0, // `STNE.cpp:33` `HeatConduct = 150`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // `STNE.cpp:40-41` `HighTemperature = 983` -> `PT_LAVA`.
        melt_k: 983.0,
        melt_product: PT_LAVA,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_WOOD, // `WOOD.cpp:11,33,26,36,44`: Colour 0xC0A040, HeatConduct 164, Flammable 20, melt 873.
        name: "WOOD",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Solids,
        albedo: [0.7529, 0.6275, 0.2510],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 700.0,
        heat_conduct: 164.0,
        // TPT WOOD has Flammable=20 and a 873 K HighTemperature -> FIRE.
        ignition_k: 873.0,
        burn_product: PT_FIRE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 16.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_ICEI, // `ICEI.cpp:9,32,31,35,44,47`: Colour 0xA0C0FF, HeatConduct 46, temp 245.15, ctype WATR.
        name: "ICEI",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Solids,
        albedo: [0.6275, 0.7529, 1.0], // `ICEI.cpp:9` `Colour = 0xA0C0FF`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 917.0,
        heat_conduct: 46.0, // `ICEI.cpp:31` `HeatConduct = 46`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // `ICEI.cpp:44-45` `HighTemperature = 252.05` -> `ST`; the default
        // ctype is WATR (`ICEI.cpp:47`), so the table stores WATR.
        melt_k: 252.05,
        melt_product: PT_WATR,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 20.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_METL, // `METL.cpp:7,29,25,32,40`: Colour 0x404060, HeatConduct 251, SC_ELEC.
        name: "METL",
        phase: VoxelPhase::Static,
        // `METL.cpp`: `MenuSection = SC_ELEC` (conductive metal).
        category: VoxelCategory::Electronics,
        albedo: [0.2510, 0.2510, 0.3765], // `METL.cpp:6` `Colour = 0x404060`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 7800.0,
        heat_conduct: 251.0, // `METL.cpp:33` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 1273.0, // `METL.cpp:44` `HighTemperature = 1273` -> LAVA
        melt_product: PT_LAVA,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_GLAS, // `GLAS.cpp:10,32,28,35,43`: Colour 0x404040, HeatConduct 150, melt 1973.
        name: "GLAS",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Solids,
        albedo: [0.2510, 0.2510, 0.2510], // `GLAS.cpp:6` `Colour = 0x404040`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 2500.0,
        heat_conduct: 150.0, // `GLAS.cpp:31` `HeatConduct = 150`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // Glass melts into lava at TPT's 1973 K threshold.
        melt_k: 1973.0,
        melt_product: PT_LAVA,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_BMTL, // `BMTL.cpp:9,31,27,34,42`: Colour 0x505070, HeatConduct 251, melt 1273.
        name: "BMTL",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Solids,
        albedo: [0.3137, 0.3137, 0.4392],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 7850.0,
        heat_conduct: 251.0,
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // TPT breakable metal liquefies at 1273 K, below stone and glass.
        melt_k: 1273.0,
        melt_product: PT_LAVA,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_PLUT, // `PLUT.cpp:9,33,27,36`: Colour 0x407020, HeatConduct 251, Hardness 0, TYPE_PART.
        name: "PLUT",
        phase: VoxelPhase::Powder, // `PLUT.cpp:36` `TYPE_PART`
        // `PLUT.cpp:11`: `MenuSection = SC_NUCLEAR`.
        category: VoxelCategory::Radioactive,
        albedo: [0.2510, 0.4392, 0.1255], // `PLUT.cpp:9` `Colour = 0x407020`
        sigma: 0.0,
        // `PLUT.cpp` defines no `Graphics`, so TPT renders it with PMODE_FLAT;
        // there is no glow to reproduce and the emission stays zero.
        emission: [0.0, 0.0, 0.0],
        density: 19800.0,
        heat_conduct: 251.0, // `PLUT.cpp:33` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0, // `PLUT.cpp:27` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_FIRE, // `FIRE.cpp:13,36,31,35,39,48`: Colour 0xFF1000, HeatConduct 88, temp 695.15, melt 2773.
        name: "FIRE",
        phase: VoxelPhase::Fire,
        // `FIRE.cpp`: `MenuSection = SC_EXPLOSIVE`.
        category: VoxelCategory::Explosives,
        albedo: [1.0, 0.0627, 0.0],
        sigma: 0.05,
        emission: [3.0, 1.1, 0.12],
        density: 0.3,
        heat_conduct: 88.0,
        // TPT FIRE becomes PLSM at 2773 K (`FIRE.cpp:51`).
        ignition_k: 2773.0,
        burn_product: PT_PLSM,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        // TPT FIRE has no LowTemperature transition; the smoke product is the
        // `life <= 1 && temp < 625` burnout in `FIRE.cpp:83-94`.
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_GUNP, // `GUNP.cpp:7,29,22-23,25,40`: Colour 0xC0C0D0, HeatConduct 97, Flammable 600, Explosive 1.
        name: "GUNP",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Explosives,
        albedo: [0.7529, 0.7529, 0.8157],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 1200.0,
        heat_conduct: 97.0,
        // TPT GUNP has Flammable=600, Explosive=1 and a 673 K transition to
        // FIRE. The 3D pass does not invent a blast radius.
        ignition_k: 673.0,
        burn_product: PT_FIRE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 10.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_BANG, // `BANG.cpp:9,31,27,34`: Colour 0xC05050, HeatConduct 88, NEUTPENETRATE.
        name: "TNT",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Explosives,
        albedo: [0.7529, 0.3137, 0.3137],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 1650.0,
        heat_conduct: 88.0,
        // TPT TNT is a static solid with no high-temperature transition; its
        // four-stage state machine lives in the reaction pass (`BANG.cpp:49-127`).
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_BCOL, // `BCOL.cpp:8,31,26,45`: Colour 0x333333, HeatConduct 150, life 110.
        name: "BCOL",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Powders,
        albedo: [0.2, 0.2, 0.2],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 150.0,
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 2.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_EMBR, // `EMBR.cpp:10,33,28,32,36,47`: Colour 0xFFF288, HeatConduct 29, temp 773.15, life 50.
        name: "EMBR",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Explosives,
        albedo: [1.0, 0.9490, 0.5333],
        sigma: 0.0,
        emission: [1.0, 0.85, 0.35],
        density: 0.0,
        heat_conduct: 29.0,
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 21.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_BOMB, // `BOMB.cpp:10,33,28,32,36`: Colour 0xFFF288, HeatConduct 29, temp 293.15.
        name: "BOMB",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Explosives,
        albedo: [1.0, 0.9490, 0.5333], // `BOMB.cpp:10` `Colour = 0xFFF288`
        sigma: 0.0,
        // `BOMB.cpp:131-135` `graphics` sets only `PMODE_FLARE`; the element
        // has no FIRE_ADD/GLOW term, so the invented glow is removed.
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 29.0,
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 20.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_IGNT, // `IGNT.cpp:9,31,27,34,42,45`: Colour 0xC0B050, HeatConduct 88, melt 673, life 3.
        name: "IGNC",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Explosives,
        albedo: [0.7529, 0.6902, 0.3137],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 88.0,
        // `IGNT.cpp:40-41`: HighTemperature = 673 -> FIRE.
        ignition_k: 673.0,
        burn_product: PT_FIRE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_FUSE, // `FUSE.cpp:9,31,27,45-46`: Colour 0x0A5706, HeatConduct 200, life 50, tmp 50.
        name: "FUSE",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Explosives,
        albedo: [0.0392, 0.3412, 0.0235],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 200.0,
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 19.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_FSEP, // `FSEP.cpp:9,31,27,45`: Colour 0x63AD5F, HeatConduct 70, life 50.
        name: "FSEP",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Explosives,
        albedo: [0.3882, 0.6784, 0.3725],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 70.0,
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 27.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_NITR, // `NITR.cpp:7,30,22-23,25,41`: Colour 0x20E010, HeatConduct 50, Flammable 1000, Explosive 2.
        name: "NITR",
        phase: VoxelPhase::Liquid,
        category: VoxelCategory::Explosives,
        albedo: [0.1255, 0.8784, 0.0627],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 50.0,
        // `NITR.cpp:34-36`: HighTemperature = 673 -> FIRE.
        ignition_k: 673.0,
        burn_product: PT_FIRE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 3.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_BRMT, // `BRMT.cpp:9,31,27,34,43`: Colour 0x705060, HeatConduct 211, melt 1273 -> LAVA.
        name: "BRMT",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Powders,
        albedo: [0.4392, 0.3137, 0.3765],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 211.0,
        // `BRMT.cpp:43-44`: `HighTemperature = 1273` -> `ST`, resolved to
        // LAVA (ctype TUNG keeps its own threshold) by `Simulation.cpp:2564-2578`.
        ignition_k: 1273.0,
        burn_product: PT_LAVA,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 2.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_BREC, // `BREC.cpp:9,31,27,34`: Colour 0x707060, HeatConduct 211, NEUTPENETRATE.
        name: "BREL",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Powders,
        albedo: [0.4392, 0.4392, 0.3765],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 211.0,
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 2.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_THRM, // `THRM.cpp:7,29,25,32`: Colour 0xA08090, HeatConduct 211, TYPE_PART.
        name: "THRM",
        phase: VoxelPhase::Powder,
        category: VoxelCategory::Explosives,
        albedo: [0.6275, 0.5020, 0.5647],
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 211.0,
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 2.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_PLSM, // `PLSM.cpp:11,34,29,33,37`: Colour 0xBB99FF, HeatConduct 5, temp MAX_TEMP, life 50..199.
        name: "PLSM",
        phase: VoxelPhase::Gas,
        category: VoxelCategory::Gases,
        albedo: [0.7333, 0.6, 1.0],
        sigma: 0.10,
        emission: [1.5, 1.0, 2.0],
        density: 0.0,
        heat_conduct: 5.0,
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0,
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_C4, // `PLEX.cpp:7,30,25,22-23,33,41`: Colour 0xD080E0, HeatConduct 88, Flammable 1000, Explosive 2.
        name: "C4",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Explosives,
        albedo: [0.8157, 0.5020, 0.8784], // `PLEX.cpp:7` `Colour = 0xD080E0`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 1600.0,
        heat_conduct: 88.0, // `PLEX.cpp:30` `HeatConduct = 88`
        // `PLEX.cpp:41-42` `HighTemperature = 673` -> `PT_FIRE`.
        ignition_k: 673.0,
        burn_product: PT_FIRE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0,
        // 3D-only blast radius for the blast pass. C++ has no radius field:
        // `PLEX.cpp:23` `Explosive = 2` only marks the pressure trigger.
        explosion_radius: 12.0,
    },
    // ---- 3D electronics / nuclear / special / force wave ----
    //
    // Each row copies the C++ `Colour`, `HeatConduct`, `Hardness`,
    // `Properties`, `DefaultProperties` and temperature-transition values from
    // `src/simulation/elements/<X>.cpp`; the file:line comment on the `id`
    // line is the source. `sigma`, `emission` and `density` are 3D-renderer-only
    // (the module doc explains why) and are never C++ ports; the emission stays
    // zero because TPT's per-element `graphics` glow is not a volumetric term.
    VoxelElement {
        id: PT_SPRK, // `SPRK.cpp:14,37,33,40`: Colour 0xFFFF80, HeatConduct 251, TYPE_SOLID|LIFE_DEC.
        name: "SPRK",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Electronics, // `SPRK.cpp:16` `SC_ELEC`
        albedo: [1.0, 1.0, 0.5020],           // `SPRK.cpp:14` `Colour = 0xFFFF80`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 251.0, // `SPRK.cpp:37` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0, // `SPRK.cpp:33` `Hardness = 1`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_PSCN, // `PSCN.cpp:7,30,33,41`: Colour 0x805050, HeatConduct 251, melt 1687.
        name: "PSCN",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Electronics, // `PSCN.cpp:9` `SC_ELEC`
        albedo: [0.5020, 0.3137, 0.3137],     // `PSCN.cpp:7` `Colour = 0x805050`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 251.0, // `PSCN.cpp:30` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // `PSCN.cpp:41-42` `HighTemperature = 1687` -> `PT_LAVA`.
        melt_k: 1687.0,
        melt_product: PT_LAVA,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0, // `PSCN.cpp:26` `Hardness = 1`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_NSCN, // `NSCN.cpp:7,30,33,41`: Colour 0x505080, HeatConduct 251, melt 1687.
        name: "NSCN",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Electronics, // `NSCN.cpp:9` `SC_ELEC`
        albedo: [0.3137, 0.3137, 0.5020],     // `NSCN.cpp:7` `Colour = 0x505080`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 251.0, // `NSCN.cpp:30` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // `NSCN.cpp:41-42` `HighTemperature = 1687` -> `PT_LAVA`.
        melt_k: 1687.0,
        melt_product: PT_LAVA,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0, // `NSCN.cpp:26` `Hardness = 1`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_URAN, // `URAN.cpp:9,33,32,36`: Colour 0x707020, HeatConduct 251, temp 325.15, TYPE_PART.
        name: "URAN",
        phase: VoxelPhase::Powder,            // `URAN.cpp:36` `TYPE_PART`
        category: VoxelCategory::Radioactive, // `URAN.cpp:11` `SC_NUCLEAR`
        albedo: [0.4392, 0.4392, 0.1255],     // `URAN.cpp:9` `Colour = 0x707020`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 251.0, // `URAN.cpp:33` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0, // `URAN.cpp:28` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_ISOZ, // `ISOZ.cpp:9,32,31,35,41`: Colour 0xAA30D0, HeatConduct 29, freeze 160 -> ISZS.
        name: "ISOZ",
        phase: VoxelPhase::Liquid,            // `ISOZ.cpp:35` `TYPE_LIQUID`
        category: VoxelCategory::Radioactive, // `ISOZ.cpp:11` `SC_NUCLEAR`
        albedo: [0.6667, 0.1882, 0.8157],     // `ISOZ.cpp:9` `Colour = 0xAA30D0`
        sigma: 0.30,                          // 3D-only volumetric absorption.
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 29.0, // `ISOZ.cpp:32` `HeatConduct = 29`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        // `ISOZ.cpp:41-42` `LowTemperature = 160` -> `PT_ISZS`.
        freeze_k: 160.0,
        freeze_product: PT_ISZS,
        hardness: 0.0, // `ISOZ.cpp:28` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_ISZS, // `ISZS.cpp:9,32,31,35,42`: Colour 0x662089, HeatConduct 251, melt 300 -> ISOZ.
        name: "ISZS",
        phase: VoxelPhase::Static,            // `ISZS.cpp:35` `TYPE_SOLID`
        category: VoxelCategory::Radioactive, // `ISZS.cpp:11` `SC_NUCLEAR`
        albedo: [0.4, 0.1255, 0.5373],        // `ISZS.cpp:9` `Colour = 0x662089`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 251.0, // `ISZS.cpp:32` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        // `ISZS.cpp:42-43` `HighTemperature = 300` -> `PT_ISOZ`.
        melt_k: 300.0,
        melt_product: PT_ISOZ,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0, // `ISZS.cpp:28` `Hardness = 1`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_NEUT, // `NEUT.cpp:14,37,36,40,266`: Colour 0x20E0FF, HeatConduct 60, temp 299.15, life 480..959.
        name: "NEUT",
        phase: VoxelPhase::Energy,            // `NEUT.cpp:40` `TYPE_ENERGY`
        category: VoxelCategory::Radioactive, // `NEUT.cpp:16` `SC_NUCLEAR`
        albedo: [0.1255, 0.8784, 1.0],        // `NEUT.cpp:14` `Colour = 0x20E0FF`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 60.0, // `NEUT.cpp:37` `HeatConduct = 60`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0, // `NEUT.cpp:32` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_PHOT, // `PHOT.cpp:13,36,35,39,50-51`: Colour 0xFFFFFF, HeatConduct 251, temp 1195.15, life 680.
        name: "PHOT",
        phase: VoxelPhase::Energy,            // `PHOT.cpp:39` `TYPE_ENERGY`
        category: VoxelCategory::Radioactive, // `PHOT.cpp:15` `SC_NUCLEAR`
        albedo: [1.0, 1.0, 1.0],              // `PHOT.cpp:13` `Colour = 0xFFFFFF`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 251.0, // `PHOT.cpp:36` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0, // `PHOT.cpp:32` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_VOID, // `VOID.cpp:7,29,25,32`: Colour 0x790B0B, HeatConduct 251, TYPE_SOLID.
        name: "VOID",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Special, // `VOID.cpp:9` `SC_SPECIAL`
        albedo: [0.4745, 0.0431, 0.0431], // `VOID.cpp:7` `Colour = 0x790B0B`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 251.0, // `VOID.cpp:29` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0, // `VOID.cpp:25` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_CLNE, // `CLNE.cpp:9,31,28,34`: Colour 0xFFD010, HeatConduct 251, TYPE_SOLID|PHOTPASS.
        name: "CLNE",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Special, // `CLNE.cpp:12` `SC_SPECIAL`
        albedo: [1.0, 0.8157, 0.0627],    // `CLNE.cpp:9` `Colour = 0xFFD010`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 251.0, // `CLNE.cpp:31` `HeatConduct = 251`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0, // `CLNE.cpp:28` `Hardness = 1`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_DEST, // `DEST.cpp:10,32,28,35`: Colour 0xFF3311, HeatConduct 150, TYPE_PART|LIFE_DEC.
        name: "DEST",
        phase: VoxelPhase::Powder,           // `DEST.cpp:35` `TYPE_PART`
        category: VoxelCategory::Explosives, // `DEST.cpp:12` `SC_EXPLOSIVE`
        albedo: [1.0, 0.2, 0.0667],          // `DEST.cpp:10` `Colour = 0xFF3311`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 150.0, // `DEST.cpp:32` `HeatConduct = 150`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0, // `DEST.cpp:28` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_AMTR, // `AMTR.cpp:10,32,28,35`: Colour 0x808080, HeatConduct 70, TYPE_GAS.
        name: "AMTR",
        phase: VoxelPhase::Gas,
        category: VoxelCategory::Radioactive, // `AMTR.cpp:12` `SC_NUCLEAR`
        albedo: [0.5020, 0.5020, 0.5020],     // `AMTR.cpp:10` `Colour = 0x808080`
        sigma: 0.10,                          // 3D-only volumetric absorption.
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 70.0, // `AMTR.cpp:32` `HeatConduct = 70`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0, // `AMTR.cpp:28` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_EXOT, // `EXOT.cpp:10,33,32,36,48-49`: Colour 0x247BFE, HeatConduct 250, temp 293.15, life 1000, tmp 244.
        name: "EXOT",
        phase: VoxelPhase::Liquid,            // `EXOT.cpp:36` `TYPE_LIQUID`
        category: VoxelCategory::Radioactive, // `EXOT.cpp:12` `SC_NUCLEAR`
        albedo: [0.1412, 0.4824, 0.9961],     // `EXOT.cpp:10` `Colour = 0x247BFE`
        sigma: 0.30,                          // 3D-only volumetric absorption.
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 250.0, // `EXOT.cpp:33` `HeatConduct = 250`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 2.0, // `EXOT.cpp:28` `Hardness = 2`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_SING, // `SING.cpp:10,32,28,35,create`: Colour 0x242424, HeatConduct 70, TYPE_PART|LIFE_DEC.
        name: "SING",
        phase: VoxelPhase::Powder,            // `SING.cpp:35` `TYPE_PART`
        category: VoxelCategory::Radioactive, // `SING.cpp:12` `SC_NUCLEAR`
        albedo: [0.1412, 0.1412, 0.1412],     // `SING.cpp:10` `Colour = 0x242424`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 70.0, // `SING.cpp:32` `HeatConduct = 70`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0, // `SING.cpp:28` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_WARP, // `WARP.cpp:11,33,30,36,create`: Colour 0x101010, HeatConduct 100, TYPE_GAS|LIFE_DEC|LIFE_KILL.
        name: "WARP",
        phase: VoxelPhase::Gas,               // `WARP.cpp:36` `TYPE_GAS`
        category: VoxelCategory::Radioactive, // `WARP.cpp:13` `SC_NUCLEAR`
        albedo: [0.0627, 0.0627, 0.0627],     // `WARP.cpp:11` `Colour = 0x101010`
        sigma: 0.10,                          // 3D-only volumetric absorption.
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 100.0, // `WARP.cpp:33` `HeatConduct = 100`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 30.0, // `WARP.cpp:30` `Hardness = 30`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_GRVT, // `GRVT.cpp:11,33,29,36,47`: Colour 0x00EE76, HeatConduct 61, TYPE_ENERGY, tmp 7.
        name: "GRVT",
        phase: VoxelPhase::Energy,            // `GRVT.cpp:36` `TYPE_ENERGY`
        category: VoxelCategory::Radioactive, // `GRVT.cpp:13` `SC_NUCLEAR`
        albedo: [0.0, 0.9333, 0.4627],        // `GRVT.cpp:11` `Colour = 0x00EE76`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 61.0, // `GRVT.cpp:33` `HeatConduct = 61`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0, // `GRVT.cpp:29` `Hardness = 1`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_NBHL,  // `NBHL.cpp:9,31,27,34`: Colour 0x202020, HeatConduct 186, TYPE_SOLID.
        name: "BHOL", // `NBHL.cpp:8` `Name = "BHOL"`
        phase: VoxelPhase::Static,
        category: VoxelCategory::Special, // `NBHL.cpp:11` `SC_SPECIAL`
        albedo: [0.1255, 0.1255, 0.1255], // `NBHL.cpp:9` `Colour = 0x202020`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 186.0, // `NBHL.cpp:31` `HeatConduct = 186`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0, // `NBHL.cpp:27` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_NWHL,  // `NWHL.cpp:9,31,27,34`: Colour 0xFFFFFF, HeatConduct 186, TYPE_SOLID.
        name: "WHOL", // `NWHL.cpp:8` `Name = "WHOL"`
        phase: VoxelPhase::Static,
        category: VoxelCategory::Special, // `NWHL.cpp:11` `SC_SPECIAL`
        albedo: [1.0, 1.0, 1.0],          // `NWHL.cpp:9` `Colour = 0xFFFFFF`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 186.0, // `NWHL.cpp:31` `HeatConduct = 186`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 0.0, // `NWHL.cpp:27` `Hardness = 0`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_GBMB, // `GBMB.cpp:10,33,32,28,36`: Colour 0x1144BB, HeatConduct 29, temp 293.15, TYPE_PART.
        name: "GBMB",
        phase: VoxelPhase::Powder,        // `GBMB.cpp:36` `TYPE_PART`
        category: VoxelCategory::Force,   // `GBMB.cpp:12` `SC_FORCE`
        albedo: [0.0667, 0.2667, 0.7333], // `GBMB.cpp:10` `Colour = 0x1144BB`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 29.0, // `GBMB.cpp:33` `HeatConduct = 29`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 20.0, // `GBMB.cpp:28` `Hardness = 20`
        explosion_radius: 0.0,
    },
    VoxelElement {
        id: PT_GPMP, // `GPMP.cpp:10,32,28,35,46`: Colour 0x0A3B3B, HeatConduct 0, TYPE_SOLID, life 10.
        name: "GPMP",
        phase: VoxelPhase::Static,
        category: VoxelCategory::Powered, // `GPMP.cpp:12` `SC_POWERED`
        albedo: [0.0392, 0.2314, 0.2314], // `GPMP.cpp:10` `Colour = 0x0A3B3B`
        sigma: 0.0,
        emission: [0.0, 0.0, 0.0],
        density: 0.0,
        heat_conduct: 0.0, // `GPMP.cpp:32` `HeatConduct = 0`
        ignition_k: 0.0,
        burn_product: PT_NONE,
        melt_k: 0.0,
        melt_product: PT_NONE,
        boil_k: 0.0,
        boil_product: PT_NONE,
        freeze_k: 0.0,
        freeze_product: PT_NONE,
        hardness: 1.0, // `GPMP.cpp:28` `Hardness = 1`
        explosion_radius: 0.0,
    },
];

/// Looks up a table entry by Powder Toy element id.
///
/// Returns `None` for [`PT_NONE`] and for every id outside the fifty-nine
/// table elements.
#[inline]
pub fn element(id: u32) -> Option<&'static VoxelElement> {
    VOXEL_ELEMENTS.iter().find(|elem| elem.id == id)
}

/// Phase of `id`; unknown ids and [`PT_NONE`] map to [`VoxelPhase::Empty`].
#[inline]
pub fn phase_of(id: u32) -> VoxelPhase {
    match element(id) {
        Some(elem) => elem.phase,
        None => VoxelPhase::Empty,
    }
}

/// Whether the volume renderer must accumulate `id` instead of terminating the
/// ray.
///
/// A non-zero [`VoxelElement::sigma`] is the single source of truth: those
/// elements participate in volumetric accumulation, zero-sigma elements stay
/// opaque. Unknown ids and [`PT_NONE`] are never translucent.
#[inline]
pub fn is_translucent(id: u32) -> bool {
    element(id).is_some_and(|elem| elem.sigma > 0.0)
}

/// Whether `id` is a static solid (`TYPE_SOLID`: wood, ice, metal, glass,
/// coal, ...). Stone and plutonium are `TYPE_PART` in C++, so they are powders.
#[inline]
pub fn is_static(id: u32) -> bool {
    phase_of(id) == VoxelPhase::Static
}

/// Whether `id` participates in loose-voxel movement (powders, liquids, gases
/// and fire). Static solids, energy particles and empty cells are not loose.
#[inline]
pub fn is_loose(id: u32) -> bool {
    matches!(
        phase_of(id),
        VoxelPhase::Powder | VoxelPhase::Liquid | VoxelPhase::Gas | VoxelPhase::Fire
    )
}

/// Whether `id` is fire.
#[inline]
pub fn is_fire(id: u32) -> bool {
    phase_of(id) == VoxelPhase::Fire
}

/// TPT `Flammable` chance per fire exposure (`FIRE.cpp:304-305`).
///
/// Kept out of [`VoxelElement`] so the host table stays comparable with the
/// original element definitions without repeating a constant on every row.
#[inline]
pub fn flammable_of(id: u32) -> u32 {
    match id {
        PT_DUST => 10,
        PT_WOOD | PT_OIL => 20,
        // `ACID.cpp:25` `Flammable = 40`.
        PT_ACID => 40,
        PT_GAS | PT_GUNP => 600,
        // `NITR.cpp:22`, `PLEX.cpp:22` (C-4): `Flammable = 1000`.
        PT_NITR | PT_C4 => 1000,
        _ => 0,
    }
}

/// TPT `Explosive` bitmask (`1` = fire-triggered, `2` = pressure-triggered).
///
/// This is distinct from [`VoxelElement::explosion_radius`], which is the 3D
/// blast pass's own radius for [`PT_BOMB`].
#[inline]
pub fn explosive_of(id: u32) -> u32 {
    match id {
        PT_GUNP => 1,
        // `NITR.cpp:23`, `PLEX.cpp:23` (C-4): `Explosive = 2` (pressure).
        PT_NITR | PT_C4 => 2,
        _ => 0,
    }
}

/// TPT property bits (`ELEM_TPT_*`) for `id`.
///
/// Only the bits the 3D simulation branches on are listed; every other
/// element has no life or spark-settling property. The values are copied from
/// each element definition (`Properties = ...` line) and the C++ element file
/// is named in the comment where a bit changes behaviour.
#[inline]
pub fn tpt_props_of(id: u32) -> u32 {
    match id {
        // `EMBR.cpp:36`: TYPE_PART|PROP_LIFE_DEC|PROP_LIFE_KILL|PROP_SPARKSETTLE.
        PT_EMBR => ELEM_TPT_LIFE_DEC | ELEM_TPT_LIFE_KILL | ELEM_TPT_SPARKSETTLE,
        // `BOMB.cpp:36`: TYPE_PART|PROP_SPARKSETTLE.
        PT_BOMB => ELEM_TPT_SPARKSETTLE,
        // `IGNT.cpp:34`: TYPE_SOLID|PROP_NEUTPENETRATE|PROP_SPARKSETTLE|PROP_LIFE_KILL.
        PT_IGNT => ELEM_TPT_NEUTPENETRATE | ELEM_TPT_SPARKSETTLE | ELEM_TPT_LIFE_KILL,
        // `BRMT.cpp:34`, `BREC.cpp:34`: TYPE_PART|PROP_CONDUCTS|PROP_LIFE_DEC|PROP_HOT_GLOW.
        PT_BRMT | PT_BREC => ELEM_TPT_CONDUCTS | ELEM_TPT_LIFE_DEC | ELEM_TPT_HOT_GLOW,
        // `PLSM.cpp:37`: TYPE_GAS|PROP_LIFE_DEC|PROP_LIFE_KILL, CarriesTypeIn=ctype.
        PT_PLSM => ELEM_TPT_LIFE_DEC | ELEM_TPT_LIFE_KILL | ELEM_TPT_CARRIES_CTYPE,
        // `FIRE.cpp:39`: TYPE_GAS|PROP_LIFE_DEC|PROP_LIFE_KILL, CarriesTypeIn=ctype.
        PT_FIRE => ELEM_TPT_LIFE_DEC | ELEM_TPT_LIFE_KILL | ELEM_TPT_CARRIES_CTYPE,
        // `LAVA.cpp:38`: TYPE_LIQUID|PROP_LIFE_DEC, CarriesTypeIn=ctype.
        PT_LAVA => ELEM_TPT_LIFE_DEC | ELEM_TPT_CARRIES_CTYPE,
        // `BMTL.cpp:34`: TYPE_SOLID|PROP_CONDUCTS|PROP_LIFE_DEC|PROP_HOT_GLOW.
        PT_BMTL => ELEM_TPT_CONDUCTS | ELEM_TPT_LIFE_DEC | ELEM_TPT_HOT_GLOW,
        // `WOOD.cpp:36`, `BANG.cpp:34`: neutron penetration only.
        PT_WOOD | PT_BANG => ELEM_TPT_NEUTPENETRATE,
        // `H2.cpp:34`: TYPE_GAS|PROP_PHOTPASS.
        PT_H2 => ELEM_TPT_PHOTPASS,
        // `SMKE.cpp:35`: TYPE_GAS|PROP_LIFE_DEC|PROP_LIFE_KILL_DEC.
        PT_SMKE => ELEM_TPT_LIFE_DEC | ELEM_TPT_LIFE_KILL_DEC,
        // `WATR.cpp:35`: TYPE_LIQUID|PROP_CONDUCTS|PROP_LIFE_DEC|PROP_NEUTPASS|PROP_PHOTPASS.
        PT_WATR => ELEM_TPT_CONDUCTS | ELEM_TPT_LIFE_DEC | ELEM_TPT_NEUTPASS | ELEM_TPT_PHOTPASS,
        // `SLTW.cpp:34`: TYPE_LIQUID|PROP_CONDUCTS|PROP_LIFE_DEC|PROP_NEUTPENETRATE|PROP_PHOTPASS.
        PT_SLTW => {
            ELEM_TPT_CONDUCTS | ELEM_TPT_LIFE_DEC | ELEM_TPT_NEUTPENETRATE | ELEM_TPT_PHOTPASS
        }
        // `ICEI.cpp:35`: TYPE_SOLID|PROP_LIFE_DEC|PROP_NEUTPASS.
        PT_ICEI => ELEM_TPT_LIFE_DEC | ELEM_TPT_NEUTPASS,
        // `SNOW.cpp:36`: TYPE_PART|PROP_NEUTPASS.
        PT_SNOW => ELEM_TPT_NEUTPASS,
        // `METL.cpp:32`: TYPE_SOLID|PROP_CONDUCTS|PROP_LIFE_DEC|PROP_HOT_GLOW.
        PT_METL => ELEM_TPT_CONDUCTS | ELEM_TPT_LIFE_DEC | ELEM_TPT_HOT_GLOW,
        // `MERC.cpp:34`: TYPE_LIQUID|PROP_CONDUCTS|PROP_NEUTABSORB|PROP_LIFE_DEC.
        // `PROP_NEUTABSORB` has no 3D bit; the other two are kept.
        PT_MERC => ELEM_TPT_CONDUCTS | ELEM_TPT_LIFE_DEC,
        // `GLAS.cpp:35`: TYPE_SOLID|PROP_NEUTPASS|PROP_PHOTPASS|PROP_HOT_GLOW|PROP_SPARKSETTLE.
        PT_GLAS => ELEM_TPT_NEUTPASS | ELEM_TPT_PHOTPASS | ELEM_TPT_HOT_GLOW | ELEM_TPT_SPARKSETTLE,
        // `OIL.cpp:32`: TYPE_LIQUID|PROP_NEUTPASS.
        PT_OIL => ELEM_TPT_NEUTPASS,
        // `GAS.cpp:33`: TYPE_GAS|PROP_NEUTPASS.
        PT_GAS => ELEM_TPT_NEUTPASS,
        // `PLUT.cpp:36`: TYPE_PART|PROP_NEUTPASS|PROP_RADIOACTIVE.
        PT_PLUT => ELEM_TPT_NEUTPASS,
        // `PLEX.cpp:33` (C-4): TYPE_SOLID|PROP_NEUTPENETRATE.
        PT_C4 => ELEM_TPT_NEUTPENETRATE,
        // `SPRK.cpp:40`: TYPE_SOLID|PROP_LIFE_DEC, CarriesTypeIn=ctype.
        PT_SPRK => ELEM_TPT_LIFE_DEC | ELEM_TPT_CARRIES_CTYPE,
        // `PSCN.cpp:33`, `NSCN.cpp:33`: TYPE_SOLID|PROP_CONDUCTS|PROP_LIFE_DEC.
        PT_PSCN | PT_NSCN => ELEM_TPT_CONDUCTS | ELEM_TPT_LIFE_DEC,
        // `ISOZ.cpp:35`: TYPE_LIQUID|PROP_NEUTPENETRATE|PROP_PHOTPASS.
        PT_ISOZ => ELEM_TPT_NEUTPENETRATE | ELEM_TPT_PHOTPASS,
        // `ISZS.cpp:35`: TYPE_SOLID|PROP_PHOTPASS.
        PT_ISZS => ELEM_TPT_PHOTPASS,
        // `NEUT.cpp:40`: TYPE_ENERGY|PROP_LIFE_DEC|PROP_LIFE_KILL_DEC.
        PT_NEUT => ELEM_TPT_LIFE_DEC | ELEM_TPT_LIFE_KILL_DEC,
        // `PHOT.cpp:39`: TYPE_ENERGY|PROP_PHOTPASS|PROP_LIFE_DEC|PROP_LIFE_KILL_DEC.
        PT_PHOT => ELEM_TPT_PHOTPASS | ELEM_TPT_LIFE_DEC | ELEM_TPT_LIFE_KILL_DEC,
        // `CLNE.cpp:34`: TYPE_SOLID|PROP_PHOTPASS|PROP_NOCTYPEDRAW, CarriesTypeIn=ctype.
        PT_CLNE => ELEM_TPT_PHOTPASS | ELEM_TPT_CARRIES_CTYPE,
        // `DEST.cpp:35`: TYPE_PART|PROP_LIFE_DEC|PROP_LIFE_KILL_DEC.
        PT_DEST => ELEM_TPT_LIFE_DEC | ELEM_TPT_LIFE_KILL_DEC,
        // `EXOT.cpp:36`: TYPE_LIQUID|PROP_NEUTPASS, CarriesTypeIn=ctype.
        PT_EXOT => ELEM_TPT_NEUTPASS | ELEM_TPT_CARRIES_CTYPE,
        // `SING.cpp:35`: TYPE_PART|PROP_LIFE_DEC.
        PT_SING => ELEM_TPT_LIFE_DEC,
        // `WARP.cpp:36`: TYPE_GAS|PROP_LIFE_DEC|PROP_LIFE_KILL.
        PT_WARP => ELEM_TPT_LIFE_DEC | ELEM_TPT_LIFE_KILL,
        // `GRVT.cpp:36`: TYPE_ENERGY|PROP_LIFE_DEC|PROP_LIFE_KILL_DEC.
        PT_GRVT => ELEM_TPT_LIFE_DEC | ELEM_TPT_LIFE_KILL_DEC,
        // `GBMB.cpp:36`: TYPE_PART|PROP_LIFE_DEC|PROP_LIFE_KILL_DEC.
        PT_GBMB => ELEM_TPT_LIFE_DEC | ELEM_TPT_LIFE_KILL_DEC,
        _ => 0,
    }
}

/// Create-time values copied from TPT's `DefaultProperties` / `Create` hook.
///
/// The packed word carries `ty`, temperature and life only, so `ctype`, `tmp`,
/// `tmp2` and `tmp3` live in the SoA auxiliary arrays. `random_life` selects a
/// per-voxel `Create` hook: `1` is `FIRE::create` (`life = rng 120..169`,
/// `FIRE.cpp:421-424`), `2` is `PLSM::create` (`life = rng 50..199`,
/// `PLSM.cpp:69-72`) and `3` is `LAVA::create` (`life = rng 240..359`,
/// `LAVA.cpp:80-83`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelSpawn {
    /// Packed-word temperature in Kelvin.
    pub temp_k: f32,
    /// Fixed life for elements whose `Create` hook does not randomise it.
    ///
    /// The packed voxel word stores 8 bits, so a C++ default above 255 (PHOT
    /// 680, NEUT 480..959, EXOT 1000) is stored as 255, the field maximum;
    /// `spawn_defaults` pre-saturates those rows to match the kernel's
    /// `pack_word` clamp.
    pub life: u32,
    /// `FIELD_CTYPE` default.
    pub ctype: u32,
    /// `FIELD_TMP` default.
    pub tmp: i32,
    /// `FIELD_TMP2` default.
    pub tmp2: i32,
    /// `FIELD_TMP3` default.
    pub tmp3: i32,
    /// Random-`life` create hook, or `0`.
    pub random_life: u32,
}

impl VoxelSpawn {
    const fn fixed(id: u32, life: u32, tmp: i32) -> Self {
        let _ = id;
        Self {
            temp_k: TPT_DEFAULT_TEMP_K,
            life,
            ctype: 0,
            tmp,
            tmp2: 0,
            tmp3: 0,
            random_life: 0,
        }
    }

    const fn with_temp(temp_k: f32, life: u32, tmp: i32) -> Self {
        Self {
            temp_k,
            life,
            ctype: 0,
            tmp,
            tmp2: 0,
            tmp3: 0,
            random_life: 0,
        }
    }
}

/// Exact create defaults for the elements ported in this wave.
///
/// Every id in [`VOXEL_ELEMENTS`] has a row here: TPT copies
/// `DefaultProperties` in `create_part`, and the C++ default is 295.15 K with
/// zero life and fields. Returns `None` only for ids outside the ported table.
pub fn spawn_defaults(id: u32) -> Option<VoxelSpawn> {
    Some(match id {
        // `COAL.cpp:45-46`: DefaultProperties.life = 110, tmp = 50.
        PT_COAL => VoxelSpawn::fixed(id, 110, 50),
        // `BCOL.cpp:45`: DefaultProperties.life = 110.
        PT_BCOL => VoxelSpawn::fixed(id, 110, 0),
        // `EMBR.cpp:32,47`: temp = 500 + 273.15, life = 50.
        PT_EMBR => VoxelSpawn::with_temp(773.15, 50, 0),
        // `BOMB.cpp:32`: temp = R_TEMP - 2 + 273.15.
        PT_BOMB => VoxelSpawn::with_temp(293.15, 0, 0),
        // `IGNT.cpp:45`: DefaultProperties.life = 3.
        PT_IGNT => VoxelSpawn::fixed(id, 3, 0),
        // `FUSE.cpp:45-46`: life = 50, tmp = 50.
        PT_FUSE => VoxelSpawn::fixed(id, 50, 50),
        // `FSEP.cpp:45`: life = 50.
        PT_FSEP => VoxelSpawn::fixed(id, 50, 0),
        // `NITR.cpp`: no DefaultProperties overrides -> 295.15 K.
        PT_NITR => VoxelSpawn::fixed(id, 0, 0),
        // `ACID.cpp:47`: `DefaultProperties.life = 75`.
        PT_ACID => VoxelSpawn::fixed(id, 75, 0),
        // `MERC.cpp:45`: `DefaultProperties.tmp = 10`.
        PT_MERC => VoxelSpawn::fixed(id, 0, 10),
        // `BRMT.cpp`, `BREC.cpp`, `THRM.cpp`: no DefaultProperties overrides.
        PT_BRMT | PT_BREC | PT_THRM => VoxelSpawn::fixed(id, 0, 0),
        // `PLSM.cpp:33,69-72`: temp = MAX_TEMP, life = rng 50..199.
        PT_PLSM => VoxelSpawn {
            temp_k: TPT_MAX_TEMP_K,
            life: 50,
            ctype: 0,
            tmp: 0,
            tmp2: 0,
            tmp3: 0,
            random_life: 2,
        },
        // `FIRE.cpp:35,421-424`: temp = R_TEMP + 400 + 273.15, life 120..169.
        PT_FIRE => VoxelSpawn {
            temp_k: 695.15,
            life: 120,
            ctype: 0,
            tmp: 0,
            tmp2: 0,
            tmp3: 0,
            random_life: 1,
        },
        // `LAVA.cpp:32,80-83`: temp = R_TEMP + 1500 + 273.15, life 240..359.
        PT_LAVA => VoxelSpawn {
            temp_k: 1795.15,
            life: 240,
            ctype: 0,
            tmp: 0,
            tmp2: 0,
            tmp3: 0,
            random_life: 3,
        },
        // `GAS.cpp:29`: temp = R_TEMP + 2 + 273.15.
        PT_GAS => VoxelSpawn::with_temp(297.15, 0, 0),
        // `PLUT.cpp:32`: temp = R_TEMP + 4 + 273.15.
        PT_PLUT => VoxelSpawn::with_temp(299.15, 0, 0),
        // `WATR.cpp:31`: temp = R_TEMP - 2 + 273.15.
        PT_WATR => VoxelSpawn::with_temp(293.15, 0, 0),
        // `WOOD.cpp`, `GUNP.cpp`, `BANG.cpp`, `DUST.cpp`, `OIL.cpp`: 295.15 K.
        PT_WOOD | PT_GUNP | PT_BANG | PT_DUST | PT_OIL => VoxelSpawn::fixed(id, 0, 0),
        // `ICEI.cpp:31,47`: temp = R_TEMP - 50 + 273.15, ctype = PT_WATR.
        PT_ICEI => VoxelSpawn {
            temp_k: 245.15,
            life: 0,
            ctype: PT_WATR,
            tmp: 0,
            tmp2: 0,
            tmp3: 0,
            random_life: 0,
        },
        // `SNOW.cpp:32`: temp = R_TEMP - 30 + 273.15.
        PT_SNOW => VoxelSpawn::with_temp(265.15, 0, 0),
        // `SMKE.cpp:31`: temp = R_TEMP + 320 + 273.15.
        PT_SMKE => VoxelSpawn::with_temp(615.15, 0, 0),
        // `WTRV.cpp:31`: temp = R_TEMP + 100 + 273.15.
        PT_WTRV => VoxelSpawn::with_temp(395.15, 0, 0),
        // `SPRK.cpp`, `PSCN.cpp`, `NSCN.cpp`: no DefaultProperties overrides.
        PT_SPRK | PT_PSCN | PT_NSCN => VoxelSpawn::fixed(id, 0, 0),
        // `URAN.cpp:32`: temp = R_TEMP + 30 + 273.15.
        PT_URAN => VoxelSpawn::with_temp(325.15, 0, 0),
        // `ISOZ.cpp:31`: temp = R_TEMP - 2 + 273.15.
        PT_ISOZ => VoxelSpawn::with_temp(293.15, 0, 0),
        // `ISZS.cpp:30`: temp = 140 K.
        PT_ISZS => VoxelSpawn::with_temp(140.0, 0, 0),
        // `NEUT.cpp:36,266`: temp = R_TEMP + 4 + 273.15; `Create` randomises
        // life to 480..959, which exceeds the packed 8-bit field, so the table
        // stores 255 (the field maximum, matching the kernel's `pack_word`).
        PT_NEUT => VoxelSpawn::with_temp(299.15, 255, 0),
        // `PHOT.cpp:35,50-51`: temp = R_TEMP + 900 + 273.15, life = 680,
        // ctype = 0x3FFFFFFF. The packed voxel life is 8 bits, so 680 is stored
        // as 255 (the field maximum, matching the kernel's `pack_word` clamp).
        PT_PHOT => VoxelSpawn {
            temp_k: 1195.15,
            life: 255,
            ctype: 0x3FFF_FFFF,
            tmp: 0,
            tmp2: 0,
            tmp3: 0,
            random_life: 0,
        },
        // `VOID.cpp`, `CLNE.cpp`, `DEST.cpp`, `AMTR.cpp`: no DefaultProperties
        // overrides.
        PT_VOID | PT_CLNE | PT_DEST | PT_AMTR => VoxelSpawn::fixed(id, 0, 0),
        // `EXOT.cpp:32,48-49`: temp = R_TEMP - 2 + 273.15, life = 1000,
        // tmp = 244. The packed life field is 8 bits, so 1000 is stored as 255
        // (the field maximum, matching the kernel's `pack_word` clamp).
        PT_EXOT => VoxelSpawn {
            temp_k: 293.15,
            life: 255,
            ctype: 0,
            tmp: 244,
            tmp2: 0,
            tmp3: 0,
            random_life: 0,
        },
        // `SING.cpp:create`, `WARP.cpp:create`, `GRVT.cpp:create`: the create
        // hook randomises life (SING 60..109, WARP 70..164, GRVT 250..448);
        // the 3D kernel owns that roll, so the table stores 0.
        PT_SING | PT_WARP => VoxelSpawn::fixed(id, 0, 0),
        // `GRVT.cpp:47`: `DefaultProperties.tmp = 7`.
        PT_GRVT => VoxelSpawn {
            temp_k: TPT_DEFAULT_TEMP_K,
            life: 0,
            ctype: 0,
            tmp: 7,
            tmp2: 0,
            tmp3: 0,
            random_life: 0,
        },
        // `NBHL.cpp`, `NWHL.cpp`: no DefaultProperties overrides.
        PT_NBHL | PT_NWHL => VoxelSpawn::fixed(id, 0, 0),
        // `GBMB.cpp:32`: temp = R_TEMP - 2 + 273.15.
        PT_GBMB => VoxelSpawn::with_temp(293.15, 0, 0),
        // `GPMP.cpp:46`: `DefaultProperties.life = 10`.
        PT_GPMP => VoxelSpawn::fixed(id, 10, 0),
        // `SAND.cpp`, `SALT.cpp`, `SLTW.cpp`, `H2.cpp`, `O2.cpp`, `CO2.cpp`,
        // `STNE.cpp`, `METL.cpp`, `GLAS.cpp`, `BMTL.cpp`, `PLEX.cpp`: no
        // `DefaultProperties` overrides, so TPT's default (295.15 K, zero
        // life/fields) applies.
        PT_SAND | PT_SALT | PT_SLTW | PT_H2 | PT_O2 | PT_CO2 | PT_STNE | PT_METL | PT_GLAS
        | PT_BMTL | PT_C4 => VoxelSpawn::fixed(id, 0, 0),
        _ => return None,
    })
}

/// Builds the flag word stored at [`ELEM_FLAGS`].
#[inline]
fn flags_of(elem: &VoxelElement) -> u32 {
    let mut flags = match elem.phase {
        VoxelPhase::Empty => 0,
        VoxelPhase::Static => ELEM_FLAG_STATIC,
        VoxelPhase::Powder => ELEM_FLAG_POWDER,
        VoxelPhase::Liquid => ELEM_FLAG_LIQUID,
        VoxelPhase::Gas => ELEM_FLAG_GAS,
        VoxelPhase::Fire => ELEM_FLAG_FIRE,
        // Energy transport is element-specific; no movement flag is derived.
        VoxelPhase::Energy => 0,
    };
    if is_translucent(elem.id) {
        flags |= ELEM_FLAG_TRANSLUCENT;
    }
    if is_loose(elem.id) {
        flags |= ELEM_FLAG_LOOSE;
    }
    if flammable_of(elem.id) > 0 {
        flags |= ELEM_FLAG_COMBUSTIBLE;
    }
    if elem.explosion_radius > 0.0 {
        flags |= ELEM_FLAG_EXPLOSIVE;
    }
    flags
}

/// Packs one element into its [`GPU_ELEMENT_WORDS`]-word row.
#[inline]
fn pack_row(elem: &VoxelElement, row: &mut [u32]) {
    debug_assert_eq!(row.len(), GPU_ELEMENT_WORDS);
    row[ELEM_ID] = elem.id;
    row[ELEM_PHASE] = elem.phase as u32;
    row[ELEM_ALBEDO_R] = elem.albedo[0].to_bits();
    row[ELEM_ALBEDO_G] = elem.albedo[1].to_bits();
    row[ELEM_ALBEDO_B] = elem.albedo[2].to_bits();
    row[ELEM_ALBEDO_A] = 1.0f32.to_bits();
    row[ELEM_SIGMA] = elem.sigma.to_bits();
    row[ELEM_EMISSION_R] = elem.emission[0].to_bits();
    row[ELEM_EMISSION_G] = elem.emission[1].to_bits();
    row[ELEM_EMISSION_B] = elem.emission[2].to_bits();
    row[ELEM_DENSITY] = elem.density.to_bits();
    row[ELEM_HEAT_CONDUCT] = elem.heat_conduct.to_bits();
    row[ELEM_IGNITION] = elem.ignition_k.to_bits();
    row[ELEM_BURN_PRODUCT] = elem.burn_product;
    row[ELEM_MELT] = elem.melt_k.to_bits();
    row[ELEM_MELT_PRODUCT] = elem.melt_product;
    row[ELEM_BOIL] = elem.boil_k.to_bits();
    row[ELEM_BOIL_PRODUCT] = elem.boil_product;
    row[ELEM_FREEZE] = elem.freeze_k.to_bits();
    row[ELEM_FREEZE_PRODUCT] = elem.freeze_product;
    row[ELEM_HARDNESS] = elem.hardness.to_bits();
    row[ELEM_EXPLOSION] = elem.explosion_radius.to_bits();
    row[ELEM_FLAGS] = flags_of(elem);
    row[ELEM_FLAMMABLE] = flammable_of(elem.id);
    row[ELEM_EXPLOSIVE] = explosive_of(elem.id);
    row[ELEM_TPT_PROPS] = tpt_props_of(elem.id);
}

/// Builds the GPU element table: [`GPU_ELEMENT_ROWS`] rows of
/// [`GPU_ELEMENT_WORDS`] `u32` words, indexed by element id.
///
/// Ids outside the element table are all-zero rows, including [`PT_NONE`].
pub fn gpu_table() -> Vec<u32> {
    let mut table = vec![0u32; GPU_ELEMENT_ROWS * GPU_ELEMENT_WORDS];
    for elem in &VOXEL_ELEMENTS {
        let base = elem.id as usize * GPU_ELEMENT_WORDS;
        debug_assert!(base + GPU_ELEMENT_WORDS <= table.len());
        pack_row(elem, &mut table[base..base + GPU_ELEMENT_WORDS]);
    }
    table
}

/// Builds the RGBA renderer palette: [`GPU_ELEMENT_ROWS`] `vec4<f32>` entries
/// (`r, g, b, a`) indexed by element id.
///
/// Rows for ids outside the element table are transparent black, which lets the
/// renderer use the palette as a complete 256-entry lookup table.
pub fn gpu_palette() -> Vec<f32> {
    let mut palette = vec![0.0f32; GPU_PALETTE_FLOATS];
    for elem in &VOXEL_ELEMENTS {
        let base = elem.id as usize * 4;
        palette[base] = elem.albedo[0];
        palette[base + 1] = elem.albedo[1];
        palette[base + 2] = elem.albedo[2];
        palette[base + 3] = 1.0;
    }
    palette
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn table_row(table: &[u32], id: u32) -> &[u32] {
        let base = id as usize * GPU_ELEMENT_WORDS;
        &table[base..base + GPU_ELEMENT_WORDS]
    }

    fn f32_word(row: &[u32], index: usize) -> f32 {
        f32::from_bits(row[index])
    }

    #[test]
    fn fifty_nine_unique_elements_cover_the_plan_set() {
        let expected = [
            PT_SAND, PT_DUST, PT_SNOW, PT_SALT, PT_COAL, PT_WATR, PT_LAVA, PT_OIL, PT_SLTW,
            PT_ACID, PT_MERC, PT_WTRV, PT_SMKE, PT_H2, PT_O2, PT_CO2, PT_GAS, PT_STNE, PT_WOOD,
            PT_ICEI, PT_METL, PT_GLAS, PT_BMTL, PT_PLUT, PT_FIRE, PT_GUNP, PT_BANG, PT_BCOL,
            PT_EMBR, PT_BOMB, PT_IGNT, PT_FUSE, PT_FSEP, PT_NITR, PT_BRMT, PT_BREC, PT_THRM,
            PT_PLSM, PT_C4, // 3D electronics / nuclear / special / force wave.
            PT_SPRK, PT_PSCN, PT_NSCN, PT_URAN, PT_ISOZ, PT_ISZS, PT_NEUT, PT_PHOT, PT_VOID,
            PT_CLNE, PT_DEST, PT_AMTR, PT_EXOT, PT_SING, PT_WARP, PT_GRVT, PT_NBHL, PT_NWHL,
            PT_GBMB, PT_GPMP,
        ];
        let ids: HashSet<u32> = VOXEL_ELEMENTS.iter().map(|elem| elem.id).collect();
        assert_eq!(VOXEL_ELEMENTS.len(), 59);
        assert_eq!(
            VOXEL_ELEMENTS
                .iter()
                .map(|elem| elem.id)
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(ids.len(), VOXEL_ELEMENTS.len());
        assert_eq!(ids.len(), expected.len());
        for id in expected {
            assert!(ids.contains(&id), "missing element id {id}");
        }
        for elem in &VOXEL_ELEMENTS {
            assert!(
                elem.id < (1 << 8),
                "element id {} does not fit 8 bits",
                elem.id
            );
            assert_eq!(element(elem.id), Some(elem));
            assert_eq!(phase_of(elem.id), elem.phase);
            assert!(!elem.name.is_empty());
            assert!(VOXEL_CATEGORIES.contains(&elem.category));
            assert!(
                elem.albedo
                    .iter()
                    .all(|channel| (0.0..=1.0).contains(channel))
            );
            assert!(elem.sigma >= 0.0);
        }
        assert!(element(PT_NONE).is_none());
        assert_eq!(phase_of(PT_NONE), VoxelPhase::Empty);
    }

    #[test]
    fn categories_cover_every_element() {
        // Membership follows each element's TPT `MenuSection` (`SimulationData.cpp`
        // `msections` order): `COAL.cpp` is `SC_SOLIDS`, `STNE.cpp` `SC_POWDERS`,
        // `METL.cpp`/`SPRK.cpp`/`PSCN.cpp`/`NSCN.cpp` are `SC_ELEC`,
        // `GPMP.cpp` is `SC_POWERED`, `GBMB.cpp` `SC_FORCE`, `PLUT.cpp` and the
        // nuclear wave are `SC_NUCLEAR`, `VOID.cpp`/`CLNE.cpp`/`NBHL.cpp`/
        // `NWHL.cpp` are `SC_SPECIAL`, `DEST.cpp` is `SC_EXPLOSIVE`.
        const ELECTRONICS: &[u32] = &[PT_METL, PT_SPRK, PT_PSCN, PT_NSCN];
        const EXPLOSIVES: &[u32] = &[
            PT_FIRE, PT_GUNP, PT_BANG, PT_C4, PT_EMBR, PT_BOMB, PT_IGNT, PT_FUSE, PT_FSEP, PT_NITR,
            PT_THRM, PT_DEST,
        ];
        const GASES: &[u32] = &[PT_WTRV, PT_SMKE, PT_H2, PT_O2, PT_CO2, PT_GAS, PT_PLSM];
        const LIQUIDS: &[u32] = &[PT_WATR, PT_LAVA, PT_OIL, PT_SLTW, PT_ACID, PT_MERC];
        const POWDERS: &[u32] = &[
            PT_SAND, PT_DUST, PT_SNOW, PT_SALT, PT_STNE, PT_BCOL, PT_BRMT, PT_BREC,
        ];
        const SOLIDS: &[u32] = &[PT_WOOD, PT_ICEI, PT_GLAS, PT_BMTL, PT_COAL];
        const RADIOACTIVE: &[u32] = &[
            PT_PLUT, PT_URAN, PT_ISOZ, PT_ISZS, PT_NEUT, PT_PHOT, PT_AMTR, PT_EXOT, PT_SING,
            PT_WARP, PT_GRVT,
        ];
        const SPECIAL: &[u32] = &[PT_VOID, PT_CLNE, PT_NBHL, PT_NWHL];
        // `Sensors` and `Life` have no 3D element yet; `Tools` holds palette
        // actions, not elements.
        const POWERED: &[u32] = &[PT_GPMP];
        const SENSORS: &[u32] = &[];
        const FORCE: &[u32] = &[PT_GBMB];
        const LIFE: &[u32] = &[];
        const TOOLS: &[u32] = &[];
        let groups = [
            (VoxelCategory::Electronics, ELECTRONICS),
            (VoxelCategory::Powered, POWERED),
            (VoxelCategory::Sensors, SENSORS),
            (VoxelCategory::Force, FORCE),
            (VoxelCategory::Explosives, EXPLOSIVES),
            (VoxelCategory::Gases, GASES),
            (VoxelCategory::Liquids, LIQUIDS),
            (VoxelCategory::Powders, POWDERS),
            (VoxelCategory::Solids, SOLIDS),
            (VoxelCategory::Radioactive, RADIOACTIVE),
            (VoxelCategory::Special, SPECIAL),
            (VoxelCategory::Life, LIFE),
            (VoxelCategory::Tools, TOOLS),
        ];
        assert_eq!(VOXEL_CATEGORIES.len(), groups.len());
        let mut grouped = 0;
        let mut grouped_ids = HashSet::new();
        for (index, (category, members)) in groups.iter().enumerate() {
            assert_eq!(VOXEL_CATEGORIES[index], *category);
            for id in *members {
                let elem = element(*id).unwrap();
                assert_eq!(elem.category, *category, "wrong category for {}", elem.name);
                assert!(
                    grouped_ids.insert(*id),
                    "{} is listed in more than one category",
                    elem.name
                );
                grouped += 1;
            }
        }
        assert_eq!(grouped, VOXEL_ELEMENTS.len());
        assert_eq!(grouped_ids.len(), VOXEL_ELEMENTS.len());
    }

    #[test]
    fn table_and_palette_have_fixed_sizes() {
        assert_eq!(GPU_ELEMENT_WORDS, 26);
        assert_eq!(ELEM_TPT_PROPS, GPU_ELEMENT_WORDS - 1);
        let table = gpu_table();
        let palette = gpu_palette();
        assert_eq!(table.len(), GPU_ELEMENT_ROWS * GPU_ELEMENT_WORDS);
        assert_eq!(palette.len(), GPU_ELEMENT_ROWS * 4);

        for elem in &VOXEL_ELEMENTS {
            let row = table_row(&table, elem.id);
            assert_eq!(row[ELEM_ID], elem.id);
            assert_eq!(row[ELEM_PHASE], elem.phase as u32);
            assert_eq!(f32_word(row, ELEM_ALBEDO_R), elem.albedo[0]);
            assert_eq!(f32_word(row, ELEM_ALBEDO_G), elem.albedo[1]);
            assert_eq!(f32_word(row, ELEM_ALBEDO_B), elem.albedo[2]);
            assert_eq!(f32_word(row, ELEM_ALBEDO_A), 1.0);
            assert_eq!(f32_word(row, ELEM_SIGMA), elem.sigma);
            assert_eq!(f32_word(row, ELEM_EMISSION_R), elem.emission[0]);
            assert_eq!(f32_word(row, ELEM_EMISSION_G), elem.emission[1]);
            assert_eq!(f32_word(row, ELEM_EMISSION_B), elem.emission[2]);
            assert_eq!(f32_word(row, ELEM_DENSITY), elem.density);
            assert_eq!(f32_word(row, ELEM_HEAT_CONDUCT), elem.heat_conduct);
            assert_eq!(f32_word(row, ELEM_IGNITION), elem.ignition_k);
            assert_eq!(row[ELEM_BURN_PRODUCT], elem.burn_product);
            assert_eq!(f32_word(row, ELEM_MELT), elem.melt_k);
            assert_eq!(row[ELEM_MELT_PRODUCT], elem.melt_product);
            assert_eq!(f32_word(row, ELEM_BOIL), elem.boil_k);
            assert_eq!(row[ELEM_BOIL_PRODUCT], elem.boil_product);
            assert_eq!(f32_word(row, ELEM_FREEZE), elem.freeze_k);
            assert_eq!(row[ELEM_FREEZE_PRODUCT], elem.freeze_product);
            assert_eq!(f32_word(row, ELEM_HARDNESS), elem.hardness);
            assert_eq!(f32_word(row, ELEM_EXPLOSION), elem.explosion_radius);
            assert_eq!(row[ELEM_TPT_PROPS], tpt_props_of(elem.id));
            assert_eq!(
                row[ELEM_FLAGS] & ELEM_FLAG_TRANSLUCENT != 0,
                is_translucent(elem.id)
            );
            assert_eq!(row[ELEM_FLAGS] & ELEM_FLAG_STATIC != 0, is_static(elem.id));
            assert_eq!(row[ELEM_FLAGS] & ELEM_FLAG_LOOSE != 0, is_loose(elem.id));
            assert_eq!(
                row[ELEM_FLAGS] & ELEM_FLAG_COMBUSTIBLE != 0,
                flammable_of(elem.id) > 0
            );
            assert_eq!(
                row[ELEM_FLAGS] & ELEM_FLAG_EXPLOSIVE != 0,
                elem.explosion_radius > 0.0
            );

            let pbase = elem.id as usize * 4;
            assert_eq!(
                &palette[pbase..pbase + 4],
                &[elem.albedo[0], elem.albedo[1], elem.albedo[2], 1.0]
            );
        }

        // Unused ids are transparent black and their table rows are all zero.
        for id in [PT_NONE, 254, 255] {
            let base = id as usize * GPU_ELEMENT_WORDS;
            assert!(
                table[base..base + GPU_ELEMENT_WORDS]
                    .iter()
                    .all(|word| *word == 0)
            );
            let pbase = id as usize * 4;
            assert_eq!(&palette[pbase..pbase + 4], &[0.0, 0.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn phase_predicates_match_the_plan() {
        assert_eq!(phase_of(PT_SAND), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_DUST), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_SNOW), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_SALT), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_GUNP), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_BANG), VoxelPhase::Static);
        assert_eq!(phase_of(PT_WATR), VoxelPhase::Liquid);
        assert_eq!(phase_of(PT_LAVA), VoxelPhase::Liquid);
        assert_eq!(phase_of(PT_OIL), VoxelPhase::Liquid);
        assert_eq!(phase_of(PT_SLTW), VoxelPhase::Liquid);
        assert_eq!(phase_of(PT_ACID), VoxelPhase::Liquid);
        assert_eq!(phase_of(PT_MERC), VoxelPhase::Liquid);
        assert_eq!(phase_of(PT_WTRV), VoxelPhase::Gas);
        assert_eq!(phase_of(PT_SMKE), VoxelPhase::Gas);
        assert_eq!(phase_of(PT_H2), VoxelPhase::Gas);
        assert_eq!(phase_of(PT_O2), VoxelPhase::Gas);
        assert_eq!(phase_of(PT_CO2), VoxelPhase::Gas);
        assert_eq!(phase_of(PT_GAS), VoxelPhase::Gas);
        // `STNE.cpp:32` and `PLUT.cpp:36` are `TYPE_PART`, not `TYPE_SOLID`.
        assert_eq!(phase_of(PT_STNE), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_PLUT), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_WOOD), VoxelPhase::Static);
        assert_eq!(phase_of(PT_ICEI), VoxelPhase::Static);
        assert_eq!(phase_of(PT_METL), VoxelPhase::Static);
        assert_eq!(phase_of(PT_COAL), VoxelPhase::Static);
        assert_eq!(phase_of(PT_GLAS), VoxelPhase::Static);
        assert_eq!(phase_of(PT_BMTL), VoxelPhase::Static);
        assert_eq!(phase_of(PT_C4), VoxelPhase::Static);
        assert_eq!(phase_of(PT_FIRE), VoxelPhase::Fire);
        // Electronics / special wave.
        assert_eq!(phase_of(PT_SPRK), VoxelPhase::Static);
        assert_eq!(phase_of(PT_PSCN), VoxelPhase::Static);
        assert_eq!(phase_of(PT_NSCN), VoxelPhase::Static);
        assert_eq!(phase_of(PT_ISZS), VoxelPhase::Static);
        assert_eq!(phase_of(PT_VOID), VoxelPhase::Static);
        assert_eq!(phase_of(PT_CLNE), VoxelPhase::Static);
        assert_eq!(phase_of(PT_NBHL), VoxelPhase::Static);
        assert_eq!(phase_of(PT_NWHL), VoxelPhase::Static);
        assert_eq!(phase_of(PT_GPMP), VoxelPhase::Static);
        assert_eq!(phase_of(PT_URAN), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_DEST), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_SING), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_GBMB), VoxelPhase::Powder);
        assert_eq!(phase_of(PT_ISOZ), VoxelPhase::Liquid);
        assert_eq!(phase_of(PT_EXOT), VoxelPhase::Liquid);
        assert_eq!(phase_of(PT_AMTR), VoxelPhase::Gas);
        assert_eq!(phase_of(PT_WARP), VoxelPhase::Gas);
        // `TYPE_ENERGY` has no movement phase.
        assert_eq!(phase_of(PT_NEUT), VoxelPhase::Energy);
        assert_eq!(phase_of(PT_PHOT), VoxelPhase::Energy);
        assert_eq!(phase_of(PT_GRVT), VoxelPhase::Energy);

        assert!(is_static(PT_METL));
        assert!(is_static(PT_COAL));
        assert!(is_static(PT_GLAS));
        assert!(is_static(PT_BMTL));
        assert!(is_static(PT_C4));
        assert!(is_static(PT_SPRK));
        assert!(is_static(PT_PSCN));
        assert!(is_static(PT_NSCN));
        assert!(is_static(PT_ISZS));
        assert!(is_static(PT_VOID));
        assert!(is_static(PT_CLNE));
        assert!(is_static(PT_NBHL));
        assert!(is_static(PT_NWHL));
        assert!(is_static(PT_GPMP));
        assert!(!is_static(PT_STNE));
        assert!(!is_static(PT_PLUT));
        assert!(!is_static(PT_SAND));
        assert!(!is_static(PT_NEUT));
        assert!(!is_static(PT_NONE));

        assert!(is_loose(PT_SAND));
        assert!(is_loose(PT_SNOW));
        assert!(is_loose(PT_SALT));
        assert!(is_loose(PT_WATR));
        assert!(is_loose(PT_SLTW));
        assert!(is_loose(PT_WTRV));
        assert!(is_loose(PT_GAS));
        assert!(is_loose(PT_FIRE));
        assert!(is_loose(PT_STNE));
        assert!(is_loose(PT_PLUT));
        assert!(is_loose(PT_URAN));
        assert!(is_loose(PT_ISOZ));
        assert!(is_loose(PT_EXOT));
        assert!(is_loose(PT_AMTR));
        assert!(is_loose(PT_WARP));
        assert!(is_loose(PT_DEST));
        assert!(is_loose(PT_SING));
        assert!(is_loose(PT_GBMB));
        // Energy transport is element-specific, so no loose flag is derived.
        assert!(!is_loose(PT_NEUT));
        assert!(!is_loose(PT_PHOT));
        assert!(!is_loose(PT_GRVT));
        assert!(!is_loose(PT_NONE));

        assert!(is_fire(PT_FIRE));
        assert!(!is_fire(PT_LAVA));
        assert!(!is_fire(PT_NONE));

        for id in [
            PT_WATR, PT_OIL, PT_SLTW, PT_ACID, PT_WTRV, PT_SMKE, PT_H2, PT_O2, PT_CO2, PT_GAS,
            PT_FIRE, PT_ISOZ, PT_EXOT, PT_AMTR, PT_WARP,
        ] {
            assert!(
                is_translucent(id),
                "{} should be translucent",
                element(id).unwrap().name
            );
        }
        for id in [
            PT_SAND, PT_DUST, PT_SNOW, PT_SALT, PT_COAL, PT_LAVA, PT_MERC, PT_STNE, PT_WOOD,
            PT_ICEI, PT_METL, PT_GLAS, PT_BMTL, PT_PLUT, PT_GUNP, PT_BANG, PT_C4, PT_SPRK, PT_PSCN,
            PT_NSCN, PT_URAN, PT_ISZS, PT_NEUT, PT_PHOT, PT_VOID, PT_CLNE, PT_DEST, PT_SING,
            PT_GRVT, PT_NBHL, PT_NWHL, PT_GBMB, PT_GPMP,
        ] {
            assert!(
                !is_translucent(id),
                "{} should be opaque",
                element(id).unwrap().name
            );
        }
        assert!(!is_translucent(PT_NONE));
    }

    #[test]
    fn table_represents_the_local_reactions() {
        let water = element(PT_WATR).unwrap();
        assert_eq!(water.freeze_k, 273.15); // `WATR.cpp:41`
        assert_eq!(water.freeze_product, PT_ICEI);
        assert_eq!(water.boil_k, 373.0); // `WATR.cpp:43`
        assert_eq!(water.boil_product, PT_WTRV);

        let ice = element(PT_ICEI).unwrap();
        assert_eq!(ice.melt_k, 252.05); // `ICEI.cpp:44`
        assert_eq!(ice.melt_product, PT_WATR);
        let steam = element(PT_WTRV).unwrap();
        assert_eq!(steam.freeze_k, 371.0); // `WTRV.cpp:42`
        assert_eq!(steam.freeze_product, PT_WATR);

        let lava = element(PT_LAVA).unwrap();
        assert_eq!(lava.freeze_k, 973.0); // `Simulation.cpp:2622-2660`
        assert_eq!(lava.freeze_product, PT_STNE);

        let fire = element(PT_FIRE).unwrap();
        assert_eq!(fire.freeze_k, 0.0);
        assert_eq!(fire.freeze_product, PT_NONE);

        let oil = element(PT_OIL).unwrap();
        assert_eq!(oil.ignition_k, 333.0);
        assert_eq!(oil.burn_product, PT_GAS);
        let wood = element(PT_WOOD).unwrap();
        assert_eq!(wood.ignition_k, 873.0);
        assert_eq!(wood.burn_product, PT_FIRE);
        for id in [PT_DUST, PT_COAL] {
            let fuel = element(id).unwrap();
            assert_eq!(fuel.ignition_k, 0.0);
            assert_eq!(fuel.burn_product, PT_NONE);
        }

        let gunp = element(PT_GUNP).unwrap();
        assert_eq!(gunp.ignition_k, 673.0);
        assert_eq!(gunp.burn_product, PT_FIRE);
        assert_eq!(gunp.explosion_radius, 0.0);
        let nitr = element(PT_NITR).unwrap();
        assert_eq!(nitr.ignition_k, 673.0);
        assert_eq!(nitr.burn_product, PT_FIRE);

        // Snow melts into water just above freezing; the water row owns the
        // reverse freeze transition.
        let snow = element(PT_SNOW).unwrap();
        assert_eq!(snow.melt_k, 252.05); // `SNOW.cpp:45`
        assert_eq!(snow.melt_product, PT_WATR);

        let salt_water = element(PT_SLTW).unwrap();
        assert_eq!(salt_water.boil_k, 383.0); // `SLTW.cpp:42`
        assert_eq!(salt_water.boil_product, PT_WTRV);
        assert_eq!(salt_water.freeze_product, PT_ICEI);

        // Glass, stone, sand, metal and silicon liquefy at their TPT thresholds.
        assert_eq!(element(PT_GLAS).unwrap().melt_k, 1973.0);
        assert_eq!(element(PT_GLAS).unwrap().melt_product, PT_LAVA);
        assert_eq!(element(PT_BMTL).unwrap().melt_k, 1273.0);
        assert_eq!(element(PT_BMTL).unwrap().melt_product, PT_LAVA);
        assert_eq!(element(PT_METL).unwrap().melt_k, 1273.0); // `METL.cpp:40`
        assert_eq!(element(PT_STNE).unwrap().melt_k, 983.0); // `STNE.cpp:40`
        assert_eq!(element(PT_SAND).unwrap().melt_k, 1973.0); // `SAND.cpp:40`
        assert_eq!(element(PT_PSCN).unwrap().melt_k, 1687.0); // `PSCN.cpp:41`
        assert_eq!(element(PT_NSCN).unwrap().melt_k, 1687.0); // `NSCN.cpp:41`
        // `SMKE.cpp:43` and `BRMT.cpp:43` transition to FIRE and LAVA.
        assert_eq!(element(PT_SMKE).unwrap().ignition_k, 625.0);
        assert_eq!(element(PT_SMKE).unwrap().burn_product, PT_FIRE);
        assert_eq!(element(PT_BRMT).unwrap().ignition_k, 1273.0);
        assert_eq!(element(PT_BRMT).unwrap().burn_product, PT_LAVA);
        // `ISOZ.cpp:41` freezes to ISZS; `ISZS.cpp:42` melts back.
        assert_eq!(element(PT_ISOZ).unwrap().freeze_k, 160.0);
        assert_eq!(element(PT_ISOZ).unwrap().freeze_product, PT_ISZS);
        assert_eq!(element(PT_ISZS).unwrap().melt_k, 300.0);
        assert_eq!(element(PT_ISZS).unwrap().melt_product, PT_ISOZ);

        // TNT is a static block whose four-stage state machine lives in the
        // reaction pass; C4 is a static explosive with a real blast radius.
        let tnt = element(PT_BANG).unwrap();
        assert_eq!(tnt.phase, VoxelPhase::Static);
        assert_eq!(tnt.burn_product, PT_NONE);
        assert_eq!(tnt.explosion_radius, 0.0);

        let c4 = element(PT_C4).unwrap();
        assert_eq!(c4.phase, VoxelPhase::Static);
        assert_eq!(c4.category, VoxelCategory::Explosives);
        assert_eq!(c4.burn_product, PT_FIRE);
        assert_eq!(flammable_of(PT_C4), 1000); // `PLEX.cpp:22`
        assert_eq!(explosive_of(PT_C4), 2); // `PLEX.cpp:23`
        assert!(c4.explosion_radius > 0.0);

        let gas = element(PT_GAS).unwrap();
        assert_eq!(gas.ignition_k, 573.0);
        assert_eq!(gas.burn_product, PT_FIRE);
        assert_eq!(gas.explosion_radius, 0.0);
        // H2 reacts in the fire update instead of carrying a table transition.
        let hydrogen = element(PT_H2).unwrap();
        assert_eq!(hydrogen.ignition_k, 0.0);
        assert_eq!(hydrogen.burn_product, PT_NONE);

        // `ACID.cpp:25` `Flammable = 40`.
        assert_eq!(flammable_of(PT_ACID), 40);

        // Inert data entries carry no transition at all.
        for id in [
            PT_ACID, PT_MERC, PT_O2, PT_CO2, PT_PLUT, PT_URAN, PT_NEUT, PT_PHOT, PT_VOID, PT_CLNE,
            PT_DEST, PT_AMTR, PT_EXOT, PT_SING, PT_WARP, PT_GRVT, PT_NBHL, PT_NWHL, PT_GBMB,
            PT_GPMP,
        ] {
            let inert = element(id).unwrap();
            assert_eq!(inert.ignition_k, 0.0);
            assert_eq!(inert.burn_product, PT_NONE);
            assert_eq!(inert.melt_k, 0.0);
            assert_eq!(inert.boil_k, 0.0);
            assert_eq!(inert.freeze_k, 0.0);
            assert_eq!(inert.explosion_radius, 0.0);
        }
    }

    #[test]
    fn spawn_defaults_cover_the_painted_elements() {
        // `(id, temp_k, life, ctype, tmp, random_life)` copied from each C++
        // `DefaultProperties` block and `Create` hook. Elements without an
        // override are listed separately below.
        let cases = [
            (PT_COAL, 295.15, 110, 0, 50, 0),
            (PT_BCOL, 295.15, 110, 0, 0, 0),
            (PT_EMBR, 773.15, 50, 0, 0, 0),
            (PT_BOMB, 293.15, 0, 0, 0, 0),
            (PT_IGNT, 295.15, 3, 0, 0, 0),
            (PT_FUSE, 295.15, 50, 0, 50, 0),
            (PT_FSEP, 295.15, 50, 0, 0, 0),
            (PT_NITR, 295.15, 0, 0, 0, 0),
            (PT_ACID, 295.15, 75, 0, 0, 0),
            (PT_MERC, 295.15, 0, 0, 10, 0),
            (PT_BRMT, 295.15, 0, 0, 0, 0),
            (PT_BREC, 295.15, 0, 0, 0, 0),
            (PT_THRM, 295.15, 0, 0, 0, 0),
            (PT_PLSM, TPT_MAX_TEMP_K, 50, 0, 0, 2),
            (PT_FIRE, 695.15, 120, 0, 0, 1),
            (PT_LAVA, 1795.15, 240, 0, 0, 3),
            (PT_GAS, 297.15, 0, 0, 0, 0),
            (PT_PLUT, 299.15, 0, 0, 0, 0),
            (PT_WATR, 293.15, 0, 0, 0, 0),
            (PT_SAND, 295.15, 0, 0, 0, 0),
            (PT_SALT, 295.15, 0, 0, 0, 0),
            (PT_SLTW, 295.15, 0, 0, 0, 0),
            (PT_H2, 295.15, 0, 0, 0, 0),
            (PT_O2, 295.15, 0, 0, 0, 0),
            (PT_CO2, 295.15, 0, 0, 0, 0),
            (PT_STNE, 295.15, 0, 0, 0, 0),
            (PT_METL, 295.15, 0, 0, 0, 0),
            (PT_GLAS, 295.15, 0, 0, 0, 0),
            (PT_BMTL, 295.15, 0, 0, 0, 0),
            (PT_C4, 295.15, 0, 0, 0, 0),
            (PT_WOOD, 295.15, 0, 0, 0, 0),
            (PT_GUNP, 295.15, 0, 0, 0, 0),
            (PT_BANG, 295.15, 0, 0, 0, 0),
            (PT_DUST, 295.15, 0, 0, 0, 0),
            (PT_OIL, 295.15, 0, 0, 0, 0),
            (PT_ICEI, 245.15, 0, PT_WATR, 0, 0),
            (PT_SNOW, 265.15, 0, 0, 0, 0),
            (PT_SMKE, 615.15, 0, 0, 0, 0),
            (PT_WTRV, 395.15, 0, 0, 0, 0),
            (PT_SPRK, 295.15, 0, 0, 0, 0),
            (PT_PSCN, 295.15, 0, 0, 0, 0),
            (PT_NSCN, 295.15, 0, 0, 0, 0),
            (PT_URAN, 325.15, 0, 0, 0, 0),
            (PT_ISOZ, 293.15, 0, 0, 0, 0),
            (PT_ISZS, 140.0, 0, 0, 0, 0),
            (PT_NEUT, 299.15, 255, 0, 0, 0),
            (PT_PHOT, 1195.15, 255, 0x3FFF_FFFF, 0, 0),
            (PT_VOID, 295.15, 0, 0, 0, 0),
            (PT_CLNE, 295.15, 0, 0, 0, 0),
            (PT_DEST, 295.15, 0, 0, 0, 0),
            (PT_AMTR, 295.15, 0, 0, 0, 0),
            (PT_EXOT, 293.15, 255, 0, 244, 0),
            (PT_SING, 295.15, 0, 0, 0, 0),
            (PT_WARP, 295.15, 0, 0, 0, 0),
            (PT_GRVT, 295.15, 0, 0, 7, 0),
            (PT_NBHL, 295.15, 0, 0, 0, 0),
            (PT_NWHL, 295.15, 0, 0, 0, 0),
            (PT_GBMB, 293.15, 0, 0, 0, 0),
            (PT_GPMP, 295.15, 10, 0, 0, 0),
        ];
        for (id, temp, life, ctype, tmp, random_life) in cases {
            let name = element(id).unwrap().name;
            let spawn = spawn_defaults(id).unwrap_or_else(|| panic!("{name} has no defaults"));
            assert_eq!(spawn.temp_k, temp, "spawn temp for {name}");
            assert_eq!(spawn.life, life, "spawn life for {name}");
            assert_eq!(spawn.ctype, ctype, "spawn ctype for {name}");
            assert_eq!(spawn.tmp, tmp, "spawn tmp for {name}");
            assert_eq!(spawn.random_life, random_life, "create hook for {name}");
        }
        // Every painted element has a row: TPT's `create_part` always copies
        // `DefaultProperties`, whose default is 295.15 K / zero life.
        for elem in &VOXEL_ELEMENTS {
            assert!(
                spawn_defaults(elem.id).is_some(),
                "{} has no spawn defaults",
                elem.name
            );
        }
        assert_eq!(TPT_DEFAULT_TEMP_K, 295.15);
        assert_eq!(spawn_defaults(PT_NONE), None);
    }

    #[test]
    fn every_element_spawns_on_the_safe_side_of_its_transitions() {
        for elem in &VOXEL_ELEMENTS {
            let temp = spawn_defaults(elem.id)
                .map(|spawn| spawn.temp_k)
                .unwrap_or(TPT_DEFAULT_TEMP_K);
            if elem.melt_k > 0.0 {
                if elem.id == PT_SNOW {
                    // `SNOW.cpp:32,47`: the one C++ element created above its
                    // own melt point (265.15 K vs 252.05 K); it melts on the
                    // first heat pass by design.
                    assert!(temp > elem.melt_k, "SNOW should spawn above its melt point");
                } else {
                    assert!(
                        temp + 2.0 < elem.melt_k,
                        "{} spawns at {} but melts at {}",
                        elem.name,
                        temp,
                        elem.melt_k
                    );
                }
            }
            if elem.boil_k > 0.0 {
                assert!(
                    temp + 2.0 < elem.boil_k,
                    "{} spawns at {} but boils at {}",
                    elem.name,
                    temp,
                    elem.boil_k
                );
            }
            if elem.ignition_k > 0.0 {
                assert!(
                    temp + 2.0 < elem.ignition_k,
                    "{} spawns at {} but ignites at {}",
                    elem.name,
                    temp,
                    elem.ignition_k
                );
            }
            if elem.freeze_k > 0.0 {
                assert!(
                    temp > elem.freeze_k + 2.0,
                    "{} spawns at {} but freezes at {}",
                    elem.name,
                    temp,
                    elem.freeze_k
                );
            }
        }
    }
}
