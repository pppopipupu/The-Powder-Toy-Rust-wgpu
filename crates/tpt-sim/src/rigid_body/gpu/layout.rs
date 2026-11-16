//! Device memory layout constants.

use tpt_core::constants::{FLAG_RIGID_PIXEL, FLAG_SKIPMOVE};

/// Words per device body record.
pub(super) const B_STRIDE: usize = 40;
pub(super) const B_ID: usize = 0;
pub(super) const B_FLAGS: usize = 1;
pub(super) const B_PIXEL_START: usize = 2;
pub(super) const B_PIXEL_COUNT: usize = 3;
pub(super) const B_CM_X: usize = 4;
pub(super) const B_CM_Y: usize = 5;
pub(super) const B_VX: usize = 6;
pub(super) const B_VY: usize = 7;
pub(super) const B_ANGLE: usize = 8;
pub(super) const B_OMEGA: usize = 9;
pub(super) const B_MASS: usize = 10;
pub(super) const B_INV_MASS: usize = 11;
pub(super) const B_INERTIA: usize = 12;
pub(super) const B_INV_INERTIA: usize = 13;
pub(super) const B_ORIGIN_X: usize = 14;
pub(super) const B_ORIGIN_Y: usize = 15;
pub(super) const B_AX: usize = 16;
pub(super) const B_AY: usize = 17;
pub(super) const B_ANGULAR_ACC: usize = 18;
pub(super) const B_SUBSTEPS: usize = 19;
pub(super) const B_UNIT_WEIGHT: usize = 20;
pub(super) const B_ALIVE: usize = 21;
pub(super) const B_RADIUS: usize = 22;
pub(super) const B_PREV_CM_X: usize = 23;
pub(super) const B_PREV_CM_Y: usize = 24;
pub(super) const B_PREV_ANGLE: usize = 25;
/// Set when the cached geometry (mass / centroid / inertia / radius / alive /
/// surface list) must be rebuilt by `prepare_geometry_kernel`.
///
/// Set by the upload and by a detachment in `reconcile_kernel`, cleared by
/// `prepare_geometry_kernel` after it rewrites the cached words.
pub(super) const B_GEOM_DIRTY: usize = 26;
/// Pending reaction to a rigid-rigid contact, as a *velocity* delta in
/// two's-complement 2^14 fixed point (`REACTION_SCALE`).
pub(super) const B_REACT_VX: usize = 27;
pub(super) const B_REACT_VY: usize = 28;
pub(super) const B_REACT_VW: usize = 29;
/// Support direction of the body, published once per sub-step by the static
/// phase: `0` = none, `1` = `+x`, `2` = `-x`, `3` = `+y`, `4` = `-y`.
pub(super) const B_SUPPORT: usize = 30;
/// Pair-phase scratch for [`B_SUPPORT`] propagation.
pub(super) const B_SUPPORT_NEXT: usize = 31;
/// Number of active surface pixels for this body in `surface_list[start .. start + count]`.
pub(super) const B_SURFACE_COUNT: usize = 32;
/// Cached `angle.cos()` / `angle.sin()` of the current pose.
pub(super) const B_COS_A: usize = 33;
pub(super) const B_SIN_A: usize = 34;
/// Cached `prev_angle.cos()` / `prev_angle.sin()` at the start of the sub-step.
pub(super) const B_PREV_COS_A: usize = 35;
pub(super) const B_PREV_SIN_A: usize = 36;
/// Non-zero when this body's conservative bounding circle overlaps at least one
/// other active body during the current tick.
pub(super) const B_HAS_PAIR: usize = 37;

pub(super) const SUPPORT_NONE: u32 = 0;

pub(super) const REACTION_SCALE: f32 = 16384.0;
pub(super) const REACTION_INV_SCALE: f32 = 0.000061035156; // 1 / 16384

/// Words per contact manifold slot in workgroup shared memory.
pub(super) const M_STRIDE: usize = 8;
/// Other body index + 1; `0` = static world.
pub(super) const M_BODY: usize = 0;
/// 0 = `+x`, 1 = `-x`, 2 = `+y`, 3 = `-y`, [`M_FREE`] = slot unused.
pub(super) const M_DIR: usize = 1;
/// Number of contact pixels accumulated into this slot.
pub(super) const M_COUNT: usize = 2;
/// Sum of the sender's pixel lever arms about its centre of mass.
pub(super) const M_ARM_AX: usize = 3;
pub(super) const M_ARM_AY: usize = 4;
/// Sum of the obstacle pixels' world cells (dynamic pairs only).
pub(super) const M_OB_X: usize = 5;
pub(super) const M_OB_Y: usize = 6;

/// Manifold slots per body.
pub(super) const MANIFOLDS: usize = 8;
pub(super) const M_FREE: u32 = 0xFFFF;

/// Body flag bits.
pub(super) const BF_ACTIVE: u32 = 0x1;

/// Words per device pixel record.
pub(super) const P_STRIDE: usize = 16;
pub(super) const P_LOCAL_X: usize = 0;
pub(super) const P_LOCAL_Y: usize = 1;
pub(super) const P_ELEM: usize = 2;
pub(super) const P_TEMP: usize = 3;
pub(super) const P_LIFE: usize = 4;
pub(super) const P_CTYPE: usize = 5;
pub(super) const P_DCOLOUR: usize = 6;
pub(super) const P_PID: usize = 7;
pub(super) const P_GRID_X: usize = 8;
pub(super) const P_GRID_Y: usize = 9;
pub(super) const P_DETACHED: usize = 10;
pub(super) const P_BODY: usize = 11;
/// Neighbour indices `(-1,-1)`, `(0,-1)` (low / high `u16`).
pub(super) const P_NEIGHBORS_0: usize = 12;
/// Neighbour indices `(1,-1)`, `(-1,0)`.
pub(super) const P_NEIGHBORS_1: usize = 13;
/// Neighbour indices `(1,0)`, `(-1,1)`.
pub(super) const P_NEIGHBORS_2: usize = 14;
/// Neighbour indices `(0,1)`, `(1,1)`.
pub(super) const P_NEIGHBORS_3: usize = 15;

/// `u32::MAX` stands for "this pixel currently owns no particle slot".
pub(super) const NO_PID: u32 = u32::MAX;
/// Body-local pixel index standing for "no pixel there" in the neighbour words.
pub(super) const NO_NEIGHBOR: u16 = u16::MAX;

/// Threads per 1-D cube for the pixel passes.
pub(super) const PIXEL_CUBE: u32 = 256;

/// Workgroup size for multi-body per-body passes.
pub(super) const WG_SIZE: u32 = 64;

/// Workgroup size for the single-body fast path.
pub(super) const SINGLE_WG_SIZE: u32 = 256;

/// Indirect dispatch slots in `RigidGpu::dispatch_counts`.
///
/// * `0..16`: `scatter_live` count per substep (`pixel_cubes` or `0`)
/// * `16..32`: `contact_pair` / `contact_static` count per substep (`body_count` or `0`)
/// * `32..48`: `apply_body_impulses` count per substep (`body_cubes` or `0`)
/// * `48`: `clear_claims` (live grid clear, `cell_cubes` or `0`)
/// * `49`: `static_all_substeps` (`body_count` when `any_pair == 0`, else `0`)
pub(super) const DISPATCH_SLOTS: usize = 50;
/// Words per indirect dispatch slot (`[x, y, z, pad]` = 16 bytes, 4-byte aligned).
pub(super) const DISPATCH_SLOT_WORDS: usize = 4;
pub(super) const DISPATCH_CLEAR_LIVE_SLOT: usize = 48;
pub(super) const DISPATCH_STATIC_ALL_SLOT: usize = 49;

const _: () = assert!(B_STRIDE == 40);
const _: () = assert!(B_HAS_PAIR < B_STRIDE);
const _: () = assert!(P_STRIDE == 16);
const _: () = assert!(M_STRIDE == 8);
const _: () = assert!(FLAG_SKIPMOVE == 0x02);
const _: () = assert!(FLAG_RIGID_PIXEL == 0x10);
const _: () = assert!(WG_SIZE == 64);
const _: () = assert!(SINGLE_WG_SIZE == 256);
