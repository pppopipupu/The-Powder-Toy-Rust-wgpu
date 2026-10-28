//! Device-resident rigid-body pipeline.
//!
//! The host [`RigidBody2D`] list is only a *description*: it is uploaded when a
//! body is spawned, edited, restored from a save or destroyed, and read back
//! only when the caller asks [`crate::SimBuffers::ensure_host_current`] for a
//! sync. Between those events the whole steady state lives on the device:
//!
//! | pass | work |
//! |---|---|
//! | [`reconcile_kernel`] | one thread per rigid pixel; reads the authoritative `part_ty` / thermal fields and marks a pixel detached in the device mask when it melted, burned or was killed |
//! | [`prepare_geometry_kernel`] | one thread per body; rebuilds mass / centroid / inertia only when the geometry-dirty word says the pixel set changed (upload or a detachment) |
//! | [`prepare_forces_kernel`] | one thread per rigid pixel; stages the air pressure / wind / Newtonian force and torque contribution of that pixel |
//! | [`prepare_finish_kernel`] | one thread per body; sums the staged forces in pixel order, then applies the mode's own gravity, torque and the sub-step count |
//! | [`integrate_substep_kernel`] | one thread per body; applies accelerations, damping, the surface-speed clamp and predicts the pose for this sub-step |
//! | [`scatter_live_kernel`] | one thread per rigid pixel; publishes the predicted pose's cell into the live rigid grid (the claim grid reused with a sub-step generation tag), so the pair phase can still find a partner that moved into its destination cell during the frame |
//! | [`contact_kernel`] (pair phase, `mode = 0`) | one thread per body; walks the active surface pixels and groups every contact with another body's rigid pixel into a per-body manifold keyed by `(receiver, axis)`; resolves each group once against the pair's combined inverse mass / inertia, applies the sender's share inline and queues the reaction on the receiver's record |
//! | [`apply_body_impulses_kernel`] | one thread per body; converts the fixed-point reactions the pair phase queued on each receiver's record into velocity / angular-velocity deltas and clears them |
//! | [`contact_kernel`] (static phase, `mode = 1`) | one thread per body; the same walk for the world - canvas edges, walls, terrain, loose particles and fluids - resolved *after* the reactions landed so a body pushed by the stack above it has its support absorb the motion in the same sub-step |
//! | [`clear_rigid_kernel`] | frees the particle slots and stale `pmap` words of the previous placement |
//! | [`claim_cells_kernel`] | one thread per rigid pixel; publishes its desired cell with an atomic `fetch_min`, so a pixel that keeps its previous cell outranks every mover and the lowest global pixel index settles the rest |
//! | [`rasterise_kernel`] | one thread per rigid pixel; reads the claim back, takes a free cell with a compare-exchange when it lost, writes `part_x/part_y`, `pmap`, `FLAG_RIGID_PIXEL \\| FLAG_SKIPMOVE` and moves free occupants into a claimed cell |
//!
//! Nothing in this module copies particles, maps or air back to the host on the
//! steady-state path. The only transfers are the tiny per-body / per-pixel
//! descriptions at spawn / edit time and the optional read-back of that
//! description when the host explicitly asks for a sync.
//!
//! # Collision invariants
//!
//! The contact pass samples one destination cell per pixel per sub-step, so
//! the sub-step length *is* the collision guarantee:
//!
//! * `prepare_finish_kernel` sizes the per-body sub-step count for the fastest
//!   surface speed the frame can reach (`sqrt(vx^2 + vy^2) + |omega| * radius`,
//!   including the acceleration still to come) and caps it at
//!   [`MAX_SUBSTEPS`] (`1` cell each).
//! * `integrate_substep_kernel` clamps the surface speed to
//!   `substeps / frames` after the acceleration, so no pixel can move more than
//!   one cell between two contact passes. A destination cell is therefore
//!   always the start cell or an adjacent one - a one-pixel wall or a thin body
//!   cannot be stepped over at any speed.
//! * The contact cell is `round(world)` (the rasteriser's cell, not a
//!   truncation) and the contact normal is the face the swept segment crossed
//!   (`swept_face`), with the penetration measured from that face. The old
//!   "previous separation" axis could point into an obstacle once the pixel had
//!   passed its centre and launched the body straight through it.
//! * The pair phase samples the receiver's frame-start cell in `pmap` first,
//!   because that pose already carries the receiver's support correction from
//!   the previous sub-step. When no rigid pixel sits in the destination cell it
//!   also looks the cell up in the *live* rigid grid, which
//!   `scatter_live_kernel` rebuilds from the predicted pose after every
//!   integration sub-step. `pmap` alone only carries the frame-start
//!   rasterisation, so two bodies closing faster than a frame's gap would each
//!   miss the other and cross; the live sample supplies the impulse (not the
//!   pose correction, whose own phase ordering keeps the resting contacts
//!   consistent) and the two bodies bounce apart.
//! * Positional corrections take the deepest penetration per signed axis (not
//!   the sum), and normal impulses are aggregated per normal direction and
//!   applied once at the average lever arm of that direction. Resolving them
//!   pixel by pixel tipped a flat landing and turned head-on impacts into
//!   sideways drift. The pair phase applies the same correction against the
//!   receiver's rasterised cell, so a body whose support is itself compressed
//!   still stops at the face instead of being rounded into the body below it.
//! * A loose powder supports a cluster only while the overburden pressure
//!   (`unit_weight * radius`) stays under the powder's yield strength; a larger
//!   body keeps the fluid buoyancy / drag response and ploughs, and the
//!   rasteriser moves the grains it replaces into claimed empty cells instead
//!   of overwriting them out of `pmap`.
//! * Rigid-body contacts are symmetric. A contact pixel that lands in another
//!   body's cell keys a manifold slot by the receiver's body index, and the
//!   manifold impulse uses the receiver's inverse mass, inverse inertia and
//!   centre, so the sender responds to a movable partner instead of an
//!   immovable surface. The equal-and-opposite impulse is queued on the
//!   receiver's record as a two's-complement 2^14 fixed-point *velocity* delta
//!   (`J * inv_mass`, torque * inv_inertia`) with `fetch_add`; integers add in
//!   any order, so the result does not depend on thread scheduling. A
//!   follow-up pass ([`apply_body_impulses_kernel`]) folds the accumulated
//!   deltas into the receiver after every sender finished. Cells are never
//!   shared: the pair phase stops the sender's pose at the receiver's face and
//!   the claim grid arbitrates the placement.
//! * Liquids and gases are not a rigid support; that is the fluid model. A
//!   body in a fluid feels buoyancy (an upward acceleration scaled by the
//!   fluid's weight) and viscous drag toward the fluid's own velocity, and the
//!   rasteriser displaces the free particles it replaces. A body therefore
//!   settles slowly through a liquid and pushes it aside, which is the
//!   intended behaviour.
//!
//! # Determinism
//!
//! The integrate and contact passes use one thread per body, so every body
//! walks its pixels in the same sequential order as the historical host solver.
//! The contact pass skips interior pixels (all eight neighbours belong to the
//! same body), whose sampled cell is always claimed by the body itself; a
//! detachment promotes the neighbours back to the surface. The skip reads a
//! packed bitmap, so a word with no surface pixel costs one latency instead of
//! 32; one thread per body cannot hide a memory latency behind other warps. That skip is what
//! keeps a brush-radius cluster (`MAX_BRUSH_RADIUS` gives 201x201 pixels) from
//! running tens of thousands of latency-bound iterations in a single thread per
//! sub-step. The rasteriser is pixel-parallel: [`claim_cells_kernel`] resolves
//! every contested desired cell with an atomic `fetch_min` (a pixel that keeps
//! its previous cell outranks a mover, then the lower global pixel index wins)
//! and [`rasterise_kernel`] only takes free cells with a compare-exchange, so
//! the outcome does not depend on scheduling.

mod helpers;
mod launch;
mod layout;
mod motion_kernels;
mod raster_kernels;
mod state;

pub use state::RigidGpu;

// The sub-step loop lengths are literals in the kernels (a host constant cannot
// cross into a `#[cube]` body), so pin the host value to them.
const _: () = assert!(MAX_SUBSTEPS == 16);

// Re-exports so intra-doc links in module docs resolve cleanly.
#[allow(unused_imports)]
use motion_kernels::{
    apply_body_impulses_kernel, contact_kernel, integrate_substep_kernel, prepare_finish_kernel,
    prepare_forces_kernel, prepare_geometry_kernel,
};
#[allow(unused_imports)]
use raster_kernels::{
    claim_cells_kernel, clear_rigid_kernel, rasterise_kernel, reconcile_kernel, scatter_live_kernel,
};

// Re-exports so child modules can reference `super::` helpers without body modifications.
pub(super) use super::{
    MAX_FRAMES_PER_STEP, MAX_SUBSTEPS, build_rigid_occupancy, particle_alive,
    step_rigid_bodies_host, take_rigid_particle,
};
