//! Device memory layout constants.

use tpt_core::constants::{FLAG_RIGID_PIXEL, FLAG_SKIPMOVE};

/// Words per device body record.
pub(super) const B_STRIDE: usize = 32;
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
/// Set when the cached geometry (mass / centroid / inertia / radius / alive)
/// must be rebuilt by `prepare_geometry_kernel`.
///
/// Set by the upload and by a detachment in `reconcile_kernel`, cleared by
/// `prepare_geometry_kernel` after it rewrites the cached words. Kernel bodies mirror
/// this word index as the literal `26`.
pub(super) const B_GEOM_DIRTY: usize = 26;
/// Pending reaction to a rigid-rigid contact, as a *velocity* delta in
/// two's-complement 2^14 fixed point (`REACTION_SCALE`).
///
/// One word each for the receiver's `vx`, `vy` and `omega`; queued with
/// `fetch_add` by the sender's contact thread and applied once by
/// `apply_body_impulses_kernel` after every contact thread finished, so the
/// result does not depend on thread order. The accumulator holds
/// `J * inv_mass` (and `(r x J) * inv_inertia`) rather than the raw impulse:
/// a brush-radius body's pair impulse is ~4e7, which no i32 fixed point of
/// usable precision can hold, while the resulting velocity delta is bounded
/// by the speed clamp and fits with ample headroom.
pub(super) const B_REACT_VX: usize = 27;
pub(super) const B_REACT_VY: usize = 28;
pub(super) const B_REACT_VW: usize = 29;

/// Fixed-point scale of [`B_REACT_VX`] / [`B_REACT_VY`] / [`B_REACT_VW`].
///
/// 2^14 resolves 6.1e-5 px/frame, far below the smallest positional correction,
/// and a contact queues at most one impulse per manifold, so the accumulated
/// i32 sum cannot approach its range for any realistic pile.
/// Support direction of the body, published once per sub-step by the static
/// phase: `0` = none, `1` = `+x`, `2` = `-x`, `3` = `+y`, `4` = `-y`.
///
/// A body with a published direction is held by the world along it (or by a
/// body that is), so a pair contact pressing along that direction may treat it
/// as an anchor instead of sharing the impulse with its (already constrained)
/// mass. Without this, a stack cancels only half of the normal velocity per
/// contact, and the residual creeps the upper bodies into the lower ones.
pub(super) const B_SUPPORT: usize = 30;
/// Pair-phase scratch for [`B_SUPPORT`] propagation (owned by the same body's
/// thread, merged into [`B_SUPPORT`] by the static phase).
pub(super) const B_SUPPORT_NEXT: usize = 31;

pub(super) const SUPPORT_NONE: u32 = 0;

pub(super) const REACTION_SCALE: f32 = 16384.0;
pub(super) const REACTION_INV_SCALE: f32 = 0.000061035156; // 1 / 16384

/// Words per contact manifold slot in `RigidGpu::manifolds`.
///
/// A body's contact pixels are grouped by `(other body, signed axis)`, so a
/// face contact resolves as one impulse at the average contact lever arm
/// instead of one impulse per pixel (which would inject spurious torque and,
/// for a dynamic pair, over-count because the partner's reaction is applied
/// only after the pass).
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
///
/// Eight cover the four world faces plus simultaneous contacts with up to
/// four other bodies (two per face in the worst realistic pile).
pub(super) const MANIFOLDS: usize = 8;
pub(super) const M_FREE: u32 = 0xFFFF;

/// Body flag bits.
pub(super) const BF_ACTIVE: u32 = 0x1;

/// Words per device pixel record.
///
/// Words 12..=15 name the pixel's body-local neighbours (two `u16` indices per
/// word, `0xFFFF` for "no such neighbour") so a detachment can put them back on
/// the contact pass's active list. Whether a pixel runs the contact pass lives
/// in a separate contiguous array (`RigidGpu::active`), because the contact
/// pass scans it once per sub-step: reading a word per pixel out of this
/// 64-byte record would make every one of those reads its own cache line.
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
///
/// The live-cell grid (`scatter_live_kernel` / the pair phase) packs the same
/// slot + 1 in its low 23 bits and the sub-step generation above them, so it
/// shares the pool's 23-bit id limit.
pub(super) const NO_PID: u32 = u32::MAX;
/// Body-local pixel index standing for "no pixel there" in the neighbour words.
pub(super) const NO_NEIGHBOR: u16 = u16::MAX;

const _: () = assert!(B_STRIDE == 32);
const _: () = assert!(B_REACT_VW < B_STRIDE);
const _: () = assert!(B_SUPPORT_NEXT < B_STRIDE);
const _: () = assert!(P_STRIDE == 16);
const _: () = assert!(M_STRIDE == 8);
const _: () = assert!(FLAG_SKIPMOVE == 0x02);
const _: () = assert!(FLAG_RIGID_PIXEL == 0x10);

/// Threads per 1-D cube for the pixel passes.
pub(super) const PIXEL_CUBE: u32 = 256;
