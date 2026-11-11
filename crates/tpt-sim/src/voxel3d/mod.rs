//! 3D voxel world: persistent buffers, air, simulation passes, rigid bodies
//! and device-side brush edits.
//!
//! The data model lives in [`tpt_core::voxel3d`]: one `u32` per voxel, the
//! leaf masks and the claim encoding are defined there and shared with the
//! renderer and the save format.
//!
//! * [`Voxel3dBuffers`] allocates the fixed buffers of a 256³ world: A/B state,
//!   claims, occupancy masks and the compacted active-brick list.
//! * Active-brick passes launch with `CubeCount::Dynamic` driven by
//!   `sim_count`, so only the bricks that are actually active are dispatched.
//!
//! The world keeps `voxel_a` authoritative at all times; `voxel_b` is scratch
//! within one tick and is never swapped (plan §3.6).

pub mod air;
pub mod air_kernel;
pub mod buffers;
pub mod clear;
pub mod edit;
pub mod elements;
pub mod engine;
pub mod flood;
pub mod gravity;
#[cfg(feature = "diagnostics")]
pub mod indirect;
pub mod mutation;
pub mod pick;
pub mod rigid;
pub mod stats;
pub mod step;

pub use buffers::{COARSE_MASK_WORDS, SIM_COUNT_WORDS, Voxel3dBuffers};
pub use engine::{Voxel3dEngine, VoxelBrickSnapshot};
#[cfg(feature = "diagnostics")]
pub use indirect::{PROBE_WORDS, run_indirect_probe};
pub use stats::Voxel3dStats;
