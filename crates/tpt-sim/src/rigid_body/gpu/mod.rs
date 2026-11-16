//! Device-resident rigid-body pipeline.
//!
//! The host [`RigidBody2D`] list is only a *description*: it is uploaded when a
//! body is spawned, edited, restored from a save or destroyed, and read back
//! only when the caller asks [`crate::SimBuffers::ensure_host_current`] for a
//! sync. Between those events the whole steady state lives on the device:
//!
//! | pass | work |
//! |---|---|
//! | [`reconcile_kernel`] | one thread per rigid pixel; marks detached pixels in the device mask and sets neighbour surface bits |
//! | [`prepare_geometry_kernel`] | one workgroup per body; rebuilds mass / centroid / inertia and compacts `surface_list` when `B_GEOM_DIRTY` is set |
//! | [`prepare_forces_kernel`] | one thread per rigid pixel (skipped when air & Newtonian gravity are off); stages air pressure / wind / gravity forces |
//! | [`prepare_finish_kernel`] | one workgroup per body; reduces staged forces, sets accelerations & sub-step count, and integrates sub-step 0 |
//! | [`broadphase_dispatch_kernel`] | single workgroup; computes per-body bounding-circle overlap (`B_HAS_PAIR`) and writes indirect `dispatch_counts` |
//! | [`static_all_substeps_256_kernel`] / [`static_all_substeps_64_kernel`] | one workgroup per body; runs every sub-step in a single dispatch when `body_count == 1` or no body pairs overlap |
//! | [`scatter_live_kernel`] | one thread per pixel of pair-active bodies; publishes live rigid cells for the pair phase |
//! | [`contact_pair_kernel`] | one workgroup per body; evaluates `surface_list` in parallel, merges hits in pixel order into workgroup-shared manifolds, and queues fixed-point pair reactions |
//! | [`apply_body_impulses_kernel`] | folds queued pair reactions between pair iterations 0 and 1 |
//! | [`contact_static_kernel`] | one workgroup per body; folds reactions from pair iter 1, evaluates static world / fluid contacts in parallel, and integrates the next sub-step |
//! | [`clear_rigid_kernel`] | frees the particle slots and stale `pmap` words of the previous placement |
//! | [`claim_cells_kernel`] | one thread per rigid pixel; publishes desired cells via `fetch_min` |
//! | [`rasterise_kernel`] | one thread per rigid pixel; resolves contested cells, displaces loose occupants and updates `pmap` / particles |

mod contact_kernels;
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
use contact_kernels::{
    contact_pair_kernel, contact_static_kernel, static_all_substeps_64_kernel,
    static_all_substeps_256_kernel,
};
#[allow(unused_imports)]
use motion_kernels::{
    apply_body_impulses_kernel, broadphase_dispatch_kernel, prepare_finish_kernel,
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
