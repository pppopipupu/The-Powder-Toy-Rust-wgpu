//! Voxel tick passes: thermal/reactions, movement, masks, blast.
//!
//! Every pass is a CubeCL kernel launched against the persistent buffers of
//! [`super::Voxel3dBuffers`]; the host only enqueues work, so one
//! [`super::Voxel3dEngine::step`] never blocks on the device (plan §0.5).
//!
//! # Pass order (plan §4.1)
//!
//! | module | pass |
//! |---|---|
//! | [`active`] | active-brick snapshot, compaction, previous snapshot |
//! | [`reaction`] | thermal conduction and the v1 reactions (A -> B) |
//! | [`movement`] | claim clearing, claim, commit, destination masks |
//! | [`blast`] | the gunpowder explosion queue (direct writes to A) |
//! | [`mask`] | leaf and coarse mask rebuild over the tick's bricks |
//!
//! The shared helpers below mirror host-side kernels conventions: a kernel
//! body may not read a host constant, so element ids, table offsets and flag
//! bits are repeated as literals and pinned to `tpt-core` with `const _: () =
//! assert!(...)` (see `AGENTS.md`, "No Host Constants in Kernel Bodies").
//!
//! Mutable locals are seeded from a runtime value (`seed_u32`/`seed_i32`/
//! `seed_f32`) because CubeCL otherwise expands a literal-initialised `let mut`
//! into a plain constant and rejects the later device assignment.

pub mod active;
pub mod blast;
pub mod electric;
pub mod life;
pub mod mask;
pub mod movement;
pub mod radioactive;
pub mod reaction;
pub mod special;

use cubecl::prelude::*;

/// World edge length in voxels.
pub const C_WORLD_DIM: u32 = 256;
/// Voxel index bits per axis.
pub const C_AXIS_BITS: u32 = 8;
/// `z` stride in the voxel index.
pub const C_STRIDE_Z: u32 = 256;
/// `y` stride in the voxel index.
pub const C_STRIDE_Y: u32 = 65_536;
/// Words in one brick-occupancy / brick-list bitset.
pub const C_BRICK_WORDS: u32 = 1_024;
/// Total number of bricks.
pub const C_BRICK_COUNT: u32 = 32_768;
/// Sentinel for "no destination".
pub const C_NONE: u32 = 0xFFFF_FFFF;

/// `u32` words per element-table row.
pub const C_ELEM_WORDS: usize = 26;
/// Table word: [`tpt_core::voxel3d::VoxelPhase`] discriminant.
pub const C_ELEM_PHASE: usize = 1;
/// Table word: thermal conductivity.
pub const C_ELEM_HEAT_CONDUCT: usize = 11;
/// Table word: ignition temperature.
pub const C_ELEM_IGNITION: usize = 12;
/// Table word: burn product.
pub const C_ELEM_BURN_PRODUCT: usize = 13;
/// Table word: melting temperature.
pub const C_ELEM_MELT: usize = 14;
/// Table word: melting product.
pub const C_ELEM_MELT_PRODUCT: usize = 15;
/// Table word: boiling temperature.
pub const C_ELEM_BOIL: usize = 16;
/// Table word: boiling product.
pub const C_ELEM_BOIL_PRODUCT: usize = 17;
/// Table word: freezing/quench temperature.
pub const C_ELEM_FREEZE: usize = 18;
/// Table word: freezing/quench product.
pub const C_ELEM_FREEZE_PRODUCT: usize = 19;
/// Table word: explosion radius.
pub const C_ELEM_EXPLOSION: usize = 21;
/// Table word: flag bits.
pub const C_ELEM_FLAGS: usize = 22;
/// Table word: TPT `Flammable`.
pub const C_ELEM_FLAMMABLE: usize = 23;
/// Table word: TPT `Explosive` bitmask.
pub const C_ELEM_EXPLOSIVE: usize = 24;
/// Table word: TPT property bits.
pub const C_ELEM_TPT_PROPS: usize = 25;

/// TPT `PROP_LIFE_DEC`.
pub const C_TPT_LIFE_DEC: u32 = 1 << 0;
/// TPT `PROP_LIFE_KILL`.
pub const C_TPT_LIFE_KILL: u32 = 1 << 1;
/// TPT `PROP_LIFE_KILL_DEC`.
pub const C_TPT_LIFE_KILL_DEC: u32 = 1 << 2;
/// TPT `PROP_SPARKSETTLE`.
pub const C_TPT_SPARKSETTLE: u32 = 1 << 3;
/// TPT `PROP_CONDUCTS`.
pub const C_TPT_CONDUCTS: u32 = 1 << 4;
/// TPT `PROP_HOT_GLOW`.
pub const C_TPT_HOT_GLOW: u32 = 1 << 5;
/// TPT `PROP_NEUTPENETRATE`.
pub const C_TPT_NEUTPENETRATE: u32 = 1 << 6;
/// TPT `PROP_NEUTPASS`.
pub const C_TPT_NEUTPASS: u32 = 1 << 7;
/// TPT `PROP_PHOTPASS`.
pub const C_TPT_PHOTPASS: u32 = 1 << 8;
/// TPT `CarriesTypeIn` for `ctype`.
pub const C_TPT_CARRIES_CTYPE: u32 = 1 << 9;

/// Flag: permanently static solid.
pub const C_FLAG_STATIC: u32 = 1 << 1;
/// Flag: falling powder.
pub const C_FLAG_POWDER: u32 = 1 << 2;
/// Flag: flowing liquid.
pub const C_FLAG_LIQUID: u32 = 1 << 3;
/// Flag: rising gas.
pub const C_FLAG_GAS: u32 = 1 << 4;
/// Flag: fire.
pub const C_FLAG_FIRE: u32 = 1 << 5;
/// Flag: participates in loose-voxel movement.
pub const C_FLAG_LOOSE: u32 = 1 << 6;
/// Flag: has a burn product.
pub const C_FLAG_COMBUSTIBLE: u32 = 1 << 7;
/// Flag: has a non-zero explosion radius.
pub const C_FLAG_EXPLOSIVE: u32 = 1 << 8;

/// Voxel word bits: element id.
pub const C_TY_MASK: u32 = 0xFF;
/// Voxel word shift of the fixed-point temperature.
pub const C_TEMP_SHIFT: u32 = 8;
/// Voxel word mask of the fixed-point temperature.
pub const C_TEMP_MASK: u32 = 0xFFFF;
/// Voxel word shift of the life counter.
pub const C_LIFE_SHIFT: u32 = 24;
/// Voxel word mask of the life counter.
pub const C_LIFE_MASK: u32 = 0xFF;

/// Powder Toy element ids, repeated for the kernels.
pub const C_PT_DUST: u32 = 1;
/// Water.
pub const C_PT_WATR: u32 = 2;
/// Oil.
pub const C_PT_OIL: u32 = 3;
/// Fire.
pub const C_PT_FIRE: u32 = 4;
/// Stone.
pub const C_PT_STNE: u32 = 5;
/// Lava.
pub const C_PT_LAVA: u32 = 6;
/// Gunpowder.
pub const C_PT_GUNP: u32 = 7;
/// Metal.
pub const C_PT_METL: u32 = 11;
/// Gas.
pub const C_PT_GAS: u32 = 10;
/// Hydrogen.
pub const C_PT_H2: u32 = 32;
/// Oxygen.
pub const C_PT_O2: u32 = 33;
/// Coal.
pub const C_PT_COAL: u32 = 62;
/// Nitroglycerin.
pub const C_PT_NITR: u32 = 8;
/// Ice.
pub const C_PT_ICEI: u32 = 14;
/// Snow.
pub const C_PT_SNOW: u32 = 13;
/// Salt water.
pub const C_PT_SLTW: u32 = 19;
/// Distilled water.
pub const C_PT_DSTW: u32 = 35;
/// Wood.
pub const C_PT_WOOD: u32 = 15;
/// Smoke.
pub const C_PT_SMKE: u32 = 16;
/// Sand.
pub const C_PT_SAND: u32 = 17;
/// Acid (`ACID.cpp`); corrosion target of neutrons and photons.
pub const C_PT_ACID: u32 = 20;
/// Steam.
pub const C_PT_WTRV: u32 = 21;
/// Broken coal.
pub const C_PT_BCOL: u32 = 27;
/// Plasma.
pub const C_PT_PLSM: u32 = 28;
/// Broken metal.
pub const C_PT_BRMT: u32 = 47;
/// Thermite.
pub const C_PT_THRM: u32 = 99;
/// Fuse.
pub const C_PT_FUSE: u32 = 102;
/// Fuse powder.
pub const C_PT_FSEP: u32 = 103;
/// Spark (used only as a neighbour check).
pub const C_PT_SPRK: u32 = 12;
/// Lightning (used only as a neighbour check).
pub const C_PT_LIGH: u32 = 107;
/// Bomb.
pub const C_PT_BOMB: u32 = 138;
/// Broken electronics.
pub const C_PT_BREC: u32 = 144;
/// TNT.
pub const C_PT_BANG: u32 = 147;
/// Ignition cord.
pub const C_PT_IGNT: u32 = 148;
/// Ember.
pub const C_PT_EMBR: u32 = 153;
/// Neutronium (only reachable through H2 fusion, which is not ported).
pub const C_PT_NBLE: u32 = 97;
/// Exotic matter (BREC's life>0 pressure branch; unreachable from Create).
pub const C_PT_EXOT: u32 = 152;
/// Breakable metal.
pub const C_PT_BMTL: u32 = 46;
/// Salt.
pub const C_PT_SALT: u32 = 18;
/// Glass.
pub const C_PT_GLAS: u32 = 24;
/// P-type silicon (`PSCN.cpp`).
pub const C_PT_PSCN: u32 = 50;
/// N-type silicon (`NSCN.cpp`).
pub const C_PT_NSCN: u32 = 51;
/// Uranium (`URAN.cpp`).
pub const C_PT_URAN: u32 = 90;
/// Isotope-Z liquid (`ISOZ.cpp`).
pub const C_PT_ISOZ: u32 = 116;
/// Isotope-Z solid (`ISZS.cpp`).
pub const C_PT_ISZS: u32 = 117;
/// Neutron (`NEUT.cpp`).
pub const C_PT_NEUT: u32 = 30;
/// Photon (`PHOT.cpp`).
pub const C_PT_PHOT: u32 = 29;
/// Void (`VOID.cpp`).
pub const C_PT_VOID: u32 = 23;
/// Clone (`CLNE.cpp`).
pub const C_PT_CLNE: u32 = 22;
/// Destructive bomb (`DEST.cpp`).
pub const C_PT_DEST: u32 = 108;
/// Anti-matter (`AMTR.cpp`).
pub const C_PT_AMTR: u32 = 104;
/// Singularity (`SING.cpp`).
pub const C_PT_SING: u32 = 140;
/// Warp (`WARP.cpp`).
pub const C_PT_WARP: u32 = 112;
/// Graviton (`GRVT.cpp`).
pub const C_PT_GRVT: u32 = 39;
/// Black hole (`NBHL.cpp`).
pub const C_PT_NBHL: u32 = 40;
/// White hole (`NWHL.cpp`).
pub const C_PT_NWHL: u32 = 41;
/// Gravity bomb (`GBMB.cpp`).
pub const C_PT_GBMB: u32 = 42;
/// Gravity pump (`GPMP.cpp`).
pub const C_PT_GPMP: u32 = 43;
/// Plutonium (DEST's fission target, `DEST.cpp:74-82`).
pub const C_PT_PLUT: u32 = 89;

// Host-side pins: the literals above are part of the device contract.
const _: () = assert!(C_WORLD_DIM == tpt_core::voxel3d::VOXEL_DIM);
const _: () = assert!(C_STRIDE_Z * 256 == C_STRIDE_Y);
const _: () = assert!(C_STRIDE_Y * 256 == tpt_core::voxel3d::VOXEL_COUNT as u32);
const _: () = assert!(C_BRICK_WORDS * 32 == C_BRICK_COUNT);
const _: () = assert!(C_ELEM_WORDS == tpt_core::voxel3d::GPU_ELEMENT_WORDS);
const _: () = assert!(C_ELEM_PHASE == tpt_core::voxel3d::ELEM_PHASE);
const _: () = assert!(C_ELEM_HEAT_CONDUCT == tpt_core::voxel3d::ELEM_HEAT_CONDUCT);
const _: () = assert!(C_ELEM_IGNITION == tpt_core::voxel3d::ELEM_IGNITION);
const _: () = assert!(C_ELEM_BURN_PRODUCT == tpt_core::voxel3d::ELEM_BURN_PRODUCT);
const _: () = assert!(C_ELEM_MELT == tpt_core::voxel3d::ELEM_MELT);
const _: () = assert!(C_ELEM_MELT_PRODUCT == tpt_core::voxel3d::ELEM_MELT_PRODUCT);
const _: () = assert!(C_ELEM_BOIL == tpt_core::voxel3d::ELEM_BOIL);
const _: () = assert!(C_ELEM_BOIL_PRODUCT == tpt_core::voxel3d::ELEM_BOIL_PRODUCT);
const _: () = assert!(C_ELEM_FREEZE == tpt_core::voxel3d::ELEM_FREEZE);
const _: () = assert!(C_ELEM_FREEZE_PRODUCT == tpt_core::voxel3d::ELEM_FREEZE_PRODUCT);
const _: () = assert!(C_ELEM_EXPLOSION == tpt_core::voxel3d::ELEM_EXPLOSION);
const _: () = assert!(C_ELEM_FLAGS == tpt_core::voxel3d::ELEM_FLAGS);
const _: () = assert!(C_ELEM_FLAMMABLE == tpt_core::voxel3d::ELEM_FLAMMABLE);
const _: () = assert!(C_ELEM_EXPLOSIVE == tpt_core::voxel3d::ELEM_EXPLOSIVE);
const _: () = assert!(C_ELEM_TPT_PROPS == tpt_core::voxel3d::ELEM_TPT_PROPS);
const _: () = assert!(C_TPT_LIFE_DEC == tpt_core::voxel3d::ELEM_TPT_LIFE_DEC);
const _: () = assert!(C_TPT_LIFE_KILL == tpt_core::voxel3d::ELEM_TPT_LIFE_KILL);
const _: () = assert!(C_TPT_LIFE_KILL_DEC == tpt_core::voxel3d::ELEM_TPT_LIFE_KILL_DEC);
const _: () = assert!(C_TPT_SPARKSETTLE == tpt_core::voxel3d::ELEM_TPT_SPARKSETTLE);
const _: () = assert!(C_TPT_CONDUCTS == tpt_core::voxel3d::ELEM_TPT_CONDUCTS);
const _: () = assert!(C_TPT_HOT_GLOW == tpt_core::voxel3d::ELEM_TPT_HOT_GLOW);
const _: () = assert!(C_TPT_NEUTPENETRATE == tpt_core::voxel3d::ELEM_TPT_NEUTPENETRATE);
const _: () = assert!(C_TPT_NEUTPASS == tpt_core::voxel3d::ELEM_TPT_NEUTPASS);
const _: () = assert!(C_TPT_PHOTPASS == tpt_core::voxel3d::ELEM_TPT_PHOTPASS);
const _: () = assert!(C_TPT_CARRIES_CTYPE == tpt_core::voxel3d::ELEM_TPT_CARRIES_CTYPE);
const _: () = assert!(C_FLAG_STATIC == tpt_core::voxel3d::ELEM_FLAG_STATIC);
const _: () = assert!(C_FLAG_POWDER == tpt_core::voxel3d::ELEM_FLAG_POWDER);
const _: () = assert!(C_FLAG_LIQUID == tpt_core::voxel3d::ELEM_FLAG_LIQUID);
const _: () = assert!(C_FLAG_GAS == tpt_core::voxel3d::ELEM_FLAG_GAS);
const _: () = assert!(C_FLAG_FIRE == tpt_core::voxel3d::ELEM_FLAG_FIRE);
const _: () = assert!(C_FLAG_LOOSE == tpt_core::voxel3d::ELEM_FLAG_LOOSE);
const _: () = assert!(C_FLAG_COMBUSTIBLE == tpt_core::voxel3d::ELEM_FLAG_COMBUSTIBLE);
const _: () = assert!(C_FLAG_EXPLOSIVE == tpt_core::voxel3d::ELEM_FLAG_EXPLOSIVE);
const _: () = assert!(C_PT_DUST == tpt_core::constants::PT_DUST);
const _: () = assert!(C_PT_WATR == tpt_core::constants::PT_WATR);
const _: () = assert!(C_PT_OIL == tpt_core::constants::PT_OIL);
const _: () = assert!(C_PT_FIRE == tpt_core::constants::PT_FIRE);
const _: () = assert!(C_PT_STNE == tpt_core::constants::PT_STNE);
const _: () = assert!(C_PT_LAVA == tpt_core::constants::PT_LAVA);
const _: () = assert!(C_PT_GUNP == tpt_core::constants::PT_GUNP);
const _: () = assert!(C_PT_METL == tpt_core::constants::PT_METL);
const _: () = assert!(C_PT_GAS == tpt_core::constants::PT_GAS);
const _: () = assert!(C_PT_H2 == tpt_core::constants::PT_H2);
const _: () = assert!(C_PT_O2 == tpt_core::constants::PT_O2);
const _: () = assert!(C_PT_COAL == tpt_core::constants::PT_COAL);
const _: () = assert!(C_PT_ICEI == tpt_core::constants::PT_ICEI);
const _: () = assert!(C_PT_SNOW == tpt_core::constants::PT_SNOW);
const _: () = assert!(C_PT_SLTW == tpt_core::constants::PT_SLTW);
const _: () = assert!(C_PT_DSTW == tpt_core::constants::PT_DSTW);
const _: () = assert!(C_PT_WOOD == tpt_core::constants::PT_WOOD);
const _: () = assert!(C_PT_SMKE == tpt_core::constants::PT_SMKE);
const _: () = assert!(C_PT_SAND == tpt_core::constants::PT_SAND);
const _: () = assert!(C_PT_ACID == tpt_core::constants::PT_ACID);
const _: () = assert!(C_PT_WTRV == tpt_core::constants::PT_WTRV);
const _: () = assert!(C_PT_NITR == tpt_core::constants::PT_NITR);
const _: () = assert!(C_PT_BCOL == tpt_core::constants::PT_BCOL);
const _: () = assert!(C_PT_PLSM == tpt_core::constants::PT_PLSM);
const _: () = assert!(C_PT_BRMT == tpt_core::constants::PT_BRMT);
const _: () = assert!(C_PT_THRM == tpt_core::constants::PT_THRM);
const _: () = assert!(C_PT_FUSE == tpt_core::constants::PT_FUSE);
const _: () = assert!(C_PT_FSEP == tpt_core::constants::PT_FSEP);
const _: () = assert!(C_PT_SPRK == tpt_core::constants::PT_SPRK);
const _: () = assert!(C_PT_LIGH == tpt_core::constants::PT_LIGH);
const _: () = assert!(C_PT_BOMB == tpt_core::constants::PT_BOMB);
const _: () = assert!(C_PT_BREC == tpt_core::constants::PT_BREC);
const _: () = assert!(C_PT_BANG == tpt_core::constants::PT_BANG);
const _: () = assert!(C_PT_IGNT == tpt_core::constants::PT_IGNT);
const _: () = assert!(C_PT_EMBR == tpt_core::constants::PT_EMBR);
const _: () = assert!(C_PT_NBLE == tpt_core::constants::PT_NBLE);
const _: () = assert!(C_PT_EXOT == tpt_core::constants::PT_EXOT);
const _: () = assert!(C_PT_BMTL == tpt_core::constants::PT_BMTL);
const _: () = assert!(C_PT_SALT == tpt_core::constants::PT_SALT);
const _: () = assert!(C_PT_GLAS == tpt_core::constants::PT_GLAS);
const _: () = assert!(C_PT_PSCN == tpt_core::constants::PT_PSCN);
const _: () = assert!(C_PT_NSCN == tpt_core::constants::PT_NSCN);
const _: () = assert!(C_PT_URAN == tpt_core::constants::PT_URAN);
const _: () = assert!(C_PT_ISOZ == tpt_core::constants::PT_ISOZ);
const _: () = assert!(C_PT_ISZS == tpt_core::constants::PT_ISZS);
const _: () = assert!(C_PT_NEUT == tpt_core::constants::PT_NEUT);
const _: () = assert!(C_PT_PHOT == tpt_core::constants::PT_PHOT);
const _: () = assert!(C_PT_VOID == tpt_core::constants::PT_VOID);
const _: () = assert!(C_PT_CLNE == tpt_core::constants::PT_CLNE);
const _: () = assert!(C_PT_DEST == tpt_core::constants::PT_DEST);
const _: () = assert!(C_PT_AMTR == tpt_core::constants::PT_AMTR);
const _: () = assert!(C_PT_SING == tpt_core::constants::PT_SING);
const _: () = assert!(C_PT_WARP == tpt_core::constants::PT_WARP);
const _: () = assert!(C_PT_GRVT == tpt_core::constants::PT_GRVT);
const _: () = assert!(C_PT_NBHL == tpt_core::constants::PT_NBHL);
const _: () = assert!(C_PT_NWHL == tpt_core::constants::PT_NWHL);
const _: () = assert!(C_PT_GBMB == tpt_core::constants::PT_GBMB);
const _: () = assert!(C_PT_GPMP == tpt_core::constants::PT_GPMP);
const _: () = assert!(C_PT_PLUT == tpt_core::constants::PT_PLUT);
const _: () = assert!(C_LIFE_SHIFT + 8 == 32);

/// Runtime-seeded `u32`, see `AGENTS.md` rule 4.
#[cube]
#[allow(clippy::erasing_op)]
pub fn seed_u32(anchor: u32, value: u32) -> u32 {
    anchor * 0u32 + value
}

/// Runtime-seeded `i32`, see `AGENTS.md` rule 4.
#[cube]
#[allow(clippy::erasing_op)]
pub fn seed_i32(anchor: i32, value: i32) -> i32 {
    anchor * 0i32 + value
}

/// Runtime-seeded `f32`, see `AGENTS.md` rule 4.
#[cube]
#[allow(clippy::erasing_op)]
pub fn seed_f32(anchor: f32, value: f32) -> f32 {
    anchor * 0.0f32 + value
}

/// `index * salt`, kept behind a call so the product is never constant-folded.
#[cube]
pub fn mix_index(index: u32, salt: u32) -> u32 {
    index * salt
}

/// Deterministic integer avalanche (the finaliser of `xxh3`).
#[cube]
pub fn hash_u32(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16u32;
    h *= 2_246_822_507u32;
    h ^= h >> 13u32;
    h *= 3_266_489_909u32;
    h ^= h >> 16u32;
    h
}

/// `min(a, b)` without the unsupported `f32::min` method.
#[cube]
pub fn minf(a: f32, b: f32) -> f32 {
    let mut out = a;
    if b < out {
        out = b;
    }
    out
}

/// `max(a, b)` without the unsupported `f32::max` method.
#[cube]
pub fn maxf(a: f32, b: f32) -> f32 {
    let mut out = a;
    if b > out {
        out = b;
    }
    out
}

/// `|v|` without the unsupported `f32::abs` method.
#[cube]
pub fn fabsf(v: f32) -> f32 {
    let mut out = v;
    if v < 0.0f32 {
        out = -v;
    }
    out
}

/// Clamps `v` to `[lo, hi]` without the unsupported `f32::clamp` method.
#[cube]
pub fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut out = v;
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

/// `|v|` for a signed integer.
#[cube]
pub fn iabsf(v: i32) -> i32 {
    let mut out = v;
    if v < 0i32 {
        out = -v;
    }
    out
}

/// `(word, bit)` of the leaf-mask bit that covers voxel `index`.
///
/// The layout matches `tpt_core::voxel3d::leaf_word_index` / `leaf_bit_mask`:
/// one bit per 2³ block, two words per brick.
#[cube]
pub fn leaf_cell(index: u32) -> (u32, u32) {
    let x = index & 255u32;
    let z = (index >> 8u32) & 255u32;
    let y = (index >> 16u32) & 255u32;
    let block =
        (((y & 7u32) >> 1u32) << 4u32) | (((z & 7u32) >> 1u32) << 2u32) | ((x & 7u32) >> 1u32);
    let brick = ((y >> 3u32) << 10u32) | ((z >> 3u32) << 5u32) | (x >> 3u32);
    (brick * 2u32 + (block >> 5u32), 1u32 << (block & 31u32))
}

/// `(word, bit)` of the `brick_occupied` bit for `brick`.
#[cube]
pub fn brick_bit(brick: u32) -> (u32, u32) {
    (brick >> 5u32, 1u32 << (brick & 31u32))
}

/// Brick index of a linear voxel index.
#[cube]
pub fn brick_of_index(index: u32) -> u32 {
    let x = index & 255u32;
    let z = (index >> 8u32) & 255u32;
    let y = (index >> 16u32) & 255u32;
    ((y >> 3u32) << 10u32) | ((z >> 3u32) << 5u32) | (x >> 3u32)
}

/// Sets the leaf bit and all six coarse bits covering `index`.
///
/// Setting is idempotent and never under-reports; the exact clears live in the
/// mask rebuild pass.
#[cube]
pub fn mark_leaf_coarse(
    leaf_masks: &mut [Atomic<u32>],
    coarse_masks: &mut [Atomic<u32>],
    index: u32,
) {
    let (leaf_word, leaf_bit) = leaf_cell(index);
    leaf_masks[leaf_word as usize].fetch_or(leaf_bit);
    let x = index & 255u32;
    let z = (index >> 8u32) & 255u32;
    let y = (index >> 16u32) & 255u32;
    for level in 0..6u32 {
        let (word, mask) = coarse_cell(level, x, y, z);
        coarse_masks[word as usize].fetch_or(mask);
    }
}

/// Sets `brick_occupied` and `dirty_bricks` for `brick`.
#[cube]
pub fn mark_brick_bits(
    brick_occupied: &mut [Atomic<u32>],
    dirty_bricks: &mut [Atomic<u32>],
    brick: u32,
) {
    let (word, mask) = brick_bit(brick);
    brick_occupied[word as usize].fetch_or(mask);
    dirty_bricks[word as usize].fetch_or(mask);
}

/// `(word, bit)` of the coarse occupancy bit at `level` covering `(x, y, z)`.
///
/// `level` is `0` for the 4³ grid up to `5` for the single 128³ bit. The word
/// bases and shifts mirror `COARSE_LEVEL_BASES` / `COARSE_LEVELS` in
/// `tpt-core`; the module-level asserts pin them.
#[cube]
pub fn coarse_cell(level: u32, x: u32, y: u32, z: u32) -> (u32, u32) {
    let mut shift = seed_u32(x, 2u32);
    let mut bits = seed_u32(x, 64u32);
    let mut base = seed_u32(x, 0u32);
    if level == 1u32 {
        shift = 3u32;
        bits = 32u32;
        base = 8_192u32;
    } else if level == 2u32 {
        shift = 4u32;
        bits = 16u32;
        base = 9_216u32;
    } else if level == 3u32 {
        shift = 5u32;
        bits = 8u32;
        base = 9_344u32;
    } else if level == 4u32 {
        shift = 6u32;
        bits = 4u32;
        base = 9_360u32;
    } else if level == 5u32 {
        shift = 7u32;
        bits = 2u32;
        base = 9_362u32;
    }
    let cell = ((y >> shift) * bits + (z >> shift)) * bits + (x >> shift);
    (base + (cell >> 5u32), 1u32 << (cell & 31u32))
}

const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BASES[1] == 8_192);
const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BASES[2] == 9_216);
const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BASES[3] == 9_344);
const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BASES[4] == 9_360);
const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BASES[5] == 9_362);
const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BITS[0] == 64);
const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BITS[1] == 32);
const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BITS[2] == 16);
const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BITS[3] == 8);
const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BITS[4] == 4);
const _: () = assert!(tpt_core::voxel3d::COARSE_LEVEL_BITS[5] == 2);
