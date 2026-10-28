//! `tpt-core` — the shared vocabulary of the Powder Toy rewrite.
//!
//! This crate contains no GPU, windowing or I/O code. It defines the constants,
//! particle records, element registry, wall definitions and high-level
//! simulation commands that `tpt-sim`, `tpt-save`, `tpt-lua` and `tpt-app` all
//! agree on.

pub mod commands;
pub mod constants;
pub mod elements;
pub mod particle;
pub mod sign;
pub mod voxel3d;
pub mod walls;

pub use commands::{BrushShape, RigidSpawnShape, SimCommand, SimToolKind};
pub use constants::*;
pub use elements::{ElementDef, ElementRegistry, GpuElementProps, MenuCategory};
pub use particle::{
    AirMode, EdgeMode, GpuParticle, GravityMode, MIN_CANVAS_DIM, RigidBody2D, RigidPixel,
    SimulationConfig,
};
pub use sign::{MAX_SIGN_TEXT, MAX_SIGNS, Sign, sanitise_text};
pub use voxel3d::{Voxel, Voxel3dConfig};
pub use walls::{
    WallDef, default_walls, is_wall_blocking_air, is_wall_blocking_heat, is_wall_blocking_particle,
};
