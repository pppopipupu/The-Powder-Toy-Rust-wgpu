//! Device-side live-particle count.
//!
//! The kernels create and kill particles without telling the host, so the
//! incrementally maintained [`SimBuffers::active_count`] drifts. The HUD needs
//! the real number, and reading `part_ty` back to count on the CPU moved up to
//! 6 MB per call on a 1024x768 canvas plus a 1.5 M iteration loop on the main
//! thread, every half second. The reduction runs on the device instead and the
//! host reads one word.

// Clippy's rewrite for these lints uses methods the device IR does not have
// (`is_multiple_of`, `clamp`, `Range::contains`, `saturating_sub`), and
// collapsing a nested `if` changes what the CubeCL macro expands, so the plain
// operators are deliberate.
#![allow(clippy::collapsible_if)]

use cubecl::prelude::*;

use crate::buffers::SimBuffers;

/// `PT_NONE` is the free-slot marker; mirrored as a literal because a `#[cube]`
/// kernel cannot read host constants.
const C_PT_NONE: u32 = 0;
const _: () = assert!(tpt_core::constants::PT_NONE == 0);

const CUBE_DIM: u32 = 256;

#[cube(launch_unchecked)]
fn reset_live_kernel(live: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        live[0].store(0u32);
    }
}

#[cube(launch_unchecked)]
fn count_live_kernel(part_ty: &[u32], live: &mut [Atomic<u32>], bound: u32) {
    let idx = ABSOLUTE_POS as u32;
    if idx < bound {
        if part_ty[idx as usize] != C_PT_NONE {
            live[0].fetch_add(1u32);
        }
    }
}

/// Count the live slots in `0..used_slot_bound()` on the device and return the
/// result, or `None` when the read-back fails.
///
/// The count is exact for every slot the host already knows about plus the
/// device spawn window, which is the same range the old trimmed read-back
/// covered.
pub fn refresh_live_count(bufs: &SimBuffers) -> Option<usize> {
    let bound = bufs.used_slot_bound().min(bufs.capacity);
    if bound == 0 {
        return Some(0);
    }
    let cubes = (bound as u32).div_ceil(CUBE_DIM).max(1);
    unsafe {
        reset_live_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_live_count.clone(), 1),
        );
        count_live_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(CUBE_DIM),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_live_count.clone(), 1),
            bound as u32,
        );
    }
    let readback = bufs.client.read(vec![bufs.gpu_live_count.clone()]).ok()?;
    let raw: &[u8] = readback.first()?;
    let word = u32::from_le_bytes(raw.get(..4)?.try_into().ok()?);
    Some(word as usize)
}
