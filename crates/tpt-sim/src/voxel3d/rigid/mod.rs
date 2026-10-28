//! 3D rigid bodies: host description, device state and the GPU solver.
//!
//! The solver handles up to 64 axis-aligned voxel boxes. Each body owns a
//! local 32^3 voxel grid, a precomputed mass/inertia tensor and a surface
//! voxel list. The whole steady state lives on the device (plan §7.5): the
//! host description is only used by [`Rigid3dState::spawn_box`].
//!
//! # Tick shape (plan §7.3)
//!
//! Four substeps, each running
//! `integrate -> world_contacts -> pair_contacts -> apply_impulses`, then once
//! per tick `displace_loose` followed by `body_block_claim`.
//!
//! * World contacts sample the hard static solids (`STONE/WOOD/ICE/METAL`) in
//!   a 3x3x3 neighbourhood and use the negative occupancy gradient as the
//!   contact normal; loose voxels are ignored because `displace_loose` removes
//!   them from the body volume.
//! * Pair contacts broad-phase with the world AABBs from `body_desc`, then
//!   sample the other body's local occupancy. The normal is the negative
//!   occupancy gradient in that body's local frame, rotated to world space.
//! * Both contact passes accumulate Q12 integer impulses into
//!   `contact_accum`; `apply_impulses` is the only reader and clears the
//!   accumulator for the next substep.
//! * A body sleeps after 30 quiet ticks. A significant impulse or position
//!   correction wakes it again.
//!
//! # Device buffers
//!
//! [`Rigid3dState`] exposes `body_count`, `body_state`, `body_voxels`,
//! `body_masks`, `contact_accum` and `body_desc`; see `rigid/state.rs` for the
//! exact word layouts. `body_desc_handle` returns the render description to
//! bind in the 3D renderer.

mod host;
mod kernels;
mod state;

pub use host::{RigidBody3d, VOXEL_DENSITY};
pub use state::{
    ACCUM_STRIDE, BODY_MASK_WORDS, BRICK_BIT_WORDS, DESC_STRIDE, Rigid3dState, STATE_STRIDE,
    SUBSTEPS, TPS,
};
