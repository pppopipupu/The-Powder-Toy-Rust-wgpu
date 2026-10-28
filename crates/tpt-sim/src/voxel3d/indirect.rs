//! Indirect kernel dispatch for the 3D voxel passes.
//!
//! The brick passes (thermal/reaction, claim, commit, mask rebuild) must only
//! run for the bricks that are actually active; the plan (§0.4) forbids
//! dispatching a fixed 32,768 workgroups of empty bricks. The active count is
//! produced on the device by the compaction pass into
//! [`Voxel3dBuffers::sim_count`](super::Voxel3dBuffers::sim_count), so the host
//! cannot choose `CubeCount::Static` and instead hands the count buffer to
//! [`CubeCount::Dynamic`].
//!
//! [`run_indirect_probe`] is the smallest kernel that exercises that path: one
//! thread per workgroup copying one word. It exists so the integration test can
//! prove, without the simulation kernels, that the wgpu backend really honours
//! a buffer-supplied grid size.

use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Words copied by [`run_indirect_probe`], i.e. the probe's grid width.
pub const PROBE_WORDS: usize = 3;

/// Copies `list[i]` to `out[i]` for the first [`PROBE_WORDS`] workgroups.
///
/// The body does not read `PROBE_WORDS` itself: kernels cannot observe host
/// constants, so the bound is a literal guarded by the host-side length.
#[cube(launch_unchecked)]
fn indirect_probe_kernel(list: &[u32], out: &mut [u32]) {
    let cube = CUBE_POS;
    if cube < 3usize {
        out[cube] = list[cube];
    }
}

/// Runs the indirect-dispatch probe.
///
/// `count` must hold `[x, y, z]` workgroup counts; only `x` matters because
/// the kernel is one-dimensional. `list` and `out` must hold at least
/// [`PROBE_WORDS`] words. The launch is asynchronous like every other CubeCL
/// pass; the caller observes the result by reading `out` back.
pub fn run_indirect_probe(client: &Client, count: &Handle, list: &Handle, out: &Handle) {
    unsafe {
        indirect_probe_kernel::launch_unchecked(
            client,
            CubeCount::Dynamic(count.clone().binding()),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(list.clone(), PROBE_WORDS),
            BufferArg::from_raw_parts(out.clone(), PROBE_WORDS),
        );
    }
}
