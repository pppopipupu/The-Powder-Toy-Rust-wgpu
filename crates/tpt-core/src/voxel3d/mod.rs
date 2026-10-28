//! Data model for the 3D voxel mode.
//!
//! The world is a fixed [`VOXEL_DIM`]^3 grid of packed `u32` voxels layered on
//! 8^3 bricks. Coordinates are `(x, y, z)` with `y` pointing up and `x` as the
//! fastest-varying axis. Everything in this module is host-side reference code;
//! the GPU kernels mirror the same formulas.

pub mod brush;
pub mod claim;
pub mod elements;
pub mod mask;
pub mod packing;

pub use brush::{Brush3d, BrushShape3d};
pub use claim::{
    blocked_claim_key, claim_is_blocked, claim_matches_blocked, claim_matches_move, claim_owner,
    claim_tag, move_claim_key, move_tag,
};
pub use elements::{
    ELEM_ALBEDO_A, ELEM_ALBEDO_B, ELEM_ALBEDO_G, ELEM_ALBEDO_R, ELEM_BOIL, ELEM_BOIL_PRODUCT,
    ELEM_BURN_PRODUCT, ELEM_DENSITY, ELEM_EMISSION_B, ELEM_EMISSION_G, ELEM_EMISSION_R,
    ELEM_EXPLOSION, ELEM_EXPLOSIVE, ELEM_FLAG_COMBUSTIBLE, ELEM_FLAG_EXPLOSIVE, ELEM_FLAG_FIRE,
    ELEM_FLAG_GAS, ELEM_FLAG_LIQUID, ELEM_FLAG_LOOSE, ELEM_FLAG_POWDER, ELEM_FLAG_STATIC,
    ELEM_FLAG_TRANSLUCENT, ELEM_FLAGS, ELEM_FLAMMABLE, ELEM_FREEZE, ELEM_FREEZE_PRODUCT,
    ELEM_HARDNESS, ELEM_HEAT_CONDUCT, ELEM_ID, ELEM_IGNITION, ELEM_MELT, ELEM_MELT_PRODUCT,
    ELEM_PHASE, ELEM_SIGMA, ELEM_TPT_CARRIES_CTYPE, ELEM_TPT_CONDUCTS, ELEM_TPT_HOT_GLOW,
    ELEM_TPT_LIFE_DEC, ELEM_TPT_LIFE_KILL, ELEM_TPT_LIFE_KILL_DEC, ELEM_TPT_NEUTPASS,
    ELEM_TPT_NEUTPENETRATE, ELEM_TPT_PHOTPASS, ELEM_TPT_PROPS, ELEM_TPT_SPARKSETTLE,
    GPU_ELEMENT_ROWS, GPU_ELEMENT_WORDS, GPU_PALETTE_FLOATS, TPT_DEFAULT_TEMP_K, TPT_MAX_TEMP_K,
    VOXEL_CATEGORIES, VOXEL_ELEMENTS, VoxelCategory, VoxelElement, VoxelPhase, VoxelSpawn, element,
    explosive_of, flammable_of, gpu_palette, gpu_table, is_fire, is_loose, is_static,
    is_translucent, phase_of, spawn_defaults, tpt_props_of,
};
pub use mask::{
    COARSE_LEVEL_BASES, COARSE_LEVEL_BITS, COARSE_LEVELS, COARSE_WORD_COUNT, brick_bit_mask,
    brick_word_index, coarse_bit_mask, coarse_word_index, leaf_bit_mask, leaf_word_index,
};
pub use packing::{LIFE_MAX, TEMP_Q_SCALE, Voxel, q_to_temp, temp_to_q};

use serde::{Deserialize, Serialize};

use crate::particle::EdgeMode;

/// World edge length in voxels.
pub const VOXEL_DIM: u32 = 256;
/// Total number of voxels in the world.
pub const VOXEL_COUNT: usize = 16_777_216;

/// Edge length of one brick in voxels.
pub const BRICK_DIM: u32 = 8;
/// Number of bricks along each world axis.
pub const BRICKS_PER_AXIS: u32 = 32;
/// Total number of bricks in the world.
pub const BRICK_COUNT: usize = 32_768;

/// Edge length of one leaf block in voxels.
pub const LEAF_BLOCK: u32 = 2;
/// Number of leaf blocks along each world axis.
pub const LEAF_BLOCKS_PER_AXIS: u32 = 128;
/// Total number of leaf blocks in the world.
pub const LEAF_BLOCK_COUNT: usize = 2_097_152;

/// Number of `u32` leaf-mask words owned by one brick.
pub const LEAF_WORDS_PER_BRICK: usize = 2;
/// Total number of `u32` leaf-mask words in the world.
pub const LEAF_WORD_COUNT: usize = 65_536;

/// Edge length of one coarse air cell in voxels.
pub const AIR_CELL: u32 = 4;
/// Number of coarse air cells along each world axis.
pub const AIR_DIM: u32 = 64;
/// Total number of coarse air cells in the world.
pub const AIR_COUNT: usize = 262_144;

/// Maximum number of simultaneous rigid bodies.
pub const MAX_BODIES: usize = 64;
/// Edge length of a rigid body's local voxel grid.
pub const BODY_GRID: u32 = 32;
/// Number of voxels in one rigid body's local grid.
pub const BODY_VOXELS: usize = 32_768;
/// Number of voxels in the complete rigid-body atlas.
pub const BODY_ATLAS_VOXELS: usize = 2_097_152;

/// `BRICK_COUNT * BRICK_DIM^3 == VOXEL_COUNT`.
const _: () = assert!(BRICK_COUNT * 512 == VOXEL_COUNT);
/// Every brick owns exactly its two leaf-mask words.
const _: () = assert!(BRICK_COUNT * LEAF_WORDS_PER_BRICK == LEAF_WORD_COUNT);
/// The leaf-block grid is `LEAF_BLOCKS_PER_AXIS^3`.
const _: () = assert!(LEAF_BLOCK_COUNT == 128 * 128 * 128);

/// Global 3D simulation and rendering settings.
///
/// The 2D [`crate::SimulationConfig`] is deliberately kept separate: the two
/// modes share no resolution, gravity or air-field representation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Voxel3dConfig {
    /// World edge length in voxels; v1 fixes this to [`VOXEL_DIM`].
    pub dim: u32,
    /// Whether the simulation is paused.
    pub paused: bool,
    /// Ambient air temperature in Kelvin.
    pub ambient_temp_k: f32,
    /// Gravity in voxels per second squared.
    pub gravity: [f32; 3],
    /// Boundary behaviour of loose voxels, ported from TPT `EdgeMode`.
    #[serde(default)]
    pub edge_mode: EdgeMode,
    /// Whether the coarse air/pressure field is simulated.
    pub air_enabled: bool,
    /// Jacobi iterations for the air pressure solve.
    pub air_iterations: u32,
    /// Normalised direction towards the sun.
    pub sun_dir: [f32; 3],
    /// Tone-mapping exposure multiplier.
    pub exposure: f32,
    /// Render resolution scale relative to the viewport (0.5, 0.75 or 1.0).
    pub render_scale: f32,
    /// Sparse ambient-occlusion rays per sample (0, 2 or 4).
    pub ao_rays: u32,
    /// Whether sun shadows are enabled.
    pub shadows: bool,
}

impl Default for Voxel3dConfig {
    fn default() -> Self {
        Self {
            dim: VOXEL_DIM,
            paused: false,
            ambient_temp_k: 293.15,
            gravity: [0.0, -9.81, 0.0],
            edge_mode: EdgeMode::Void,
            air_enabled: true,
            air_iterations: 12,
            sun_dir: [0.5, 0.707_106_77, 0.5],
            exposure: 1.0,
            render_scale: 1.0,
            ao_rays: 2,
            shadows: true,
        }
    }
}

/// Linear voxel index, `x` fastest, then `z`, then `y`.
#[inline]
pub fn voxel_index(x: u32, y: u32, z: u32) -> usize {
    ((y << 16) | (z << 8) | x) as usize
}

/// Linear brick index, `x` fastest, then `z`, then `y`.
#[inline]
pub fn brick_index(x: u32, y: u32, z: u32) -> usize {
    (((y >> 3) << 10) | ((z >> 3) << 5) | (x >> 3)) as usize
}

/// Leaf-block index within a brick, `x` fastest, then `z`, then `y`.
#[inline]
pub fn local_block(x: u32, y: u32, z: u32) -> usize {
    ((((y & 7) >> 1) << 4) | (((z & 7) >> 1) << 2) | ((x & 7) >> 1)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinates_match_plan_layout() {
        assert_eq!(voxel_index(0, 0, 0), 0);
        assert_eq!(voxel_index(1, 0, 0), 1);
        assert_eq!(voxel_index(0, 0, 1), 256);
        assert_eq!(voxel_index(0, 1, 0), 65_536);
        assert_eq!(voxel_index(255, 255, 255), VOXEL_COUNT - 1);

        assert_eq!(brick_index(0, 0, 0), 0);
        assert_eq!(brick_index(8, 0, 0), 1);
        assert_eq!(brick_index(0, 0, 8), 32);
        assert_eq!(brick_index(0, 8, 0), 1024);
        assert_eq!(brick_index(255, 255, 255), BRICK_COUNT - 1);

        assert_eq!(local_block(0, 0, 0), 0);
        assert_eq!(local_block(2, 0, 0), 1);
        assert_eq!(local_block(0, 0, 2), 4);
        assert_eq!(local_block(0, 2, 0), 16);
        assert_eq!(local_block(7, 7, 7), 63);
    }

    #[test]
    fn default_config_matches_plan() {
        let config = Voxel3dConfig::default();
        assert_eq!(config.dim, VOXEL_DIM);
        assert!(!config.paused);
        assert_eq!(config.ambient_temp_k, 293.15);
        assert_eq!(config.gravity, [0.0, -9.81, 0.0]);
        assert!(config.air_enabled);
        assert_eq!(config.air_iterations, 12);
        assert!((config.render_scale - 1.0).abs() < f32::EPSILON);
        assert!(matches!(config.ao_rays, 0 | 2 | 4));
        assert!(config.shadows);

        let [sx, sy, sz] = config.sun_dir;
        let length = (sx * sx + sy * sy + sz * sz).sqrt();
        assert!((length - 1.0).abs() < 1e-5);
    }
}
