//! One-pass 4-connected component flood for the per-tick element reactions.
//!
//! `Simulation::flood_prop` (`Simulation.cpp:520-597`) writes one field of
//! every pixel of the 4-connected run of the seed's element, in a single call,
//! with no distance limit. The live element passes (`LED.cpp:78`, `MGNT.cpp:96`
//! / `:106`, `CEXP.cpp:73`, `BANG.cpp:74`) all rely on that: a charge, a lit
//! diode run or a magnet block reaches its final state in one frame.
//!
//! A parallel wavefront kernel can only advance the mark one hop per dispatch,
//! so reproducing the reference with rounds needs an unbounded loop. The host
//! may not poll a progress word on a per-frame path (the device stays
//! resident), and a multi-round chunk per frame is exactly the truncation this
//! module replaces. The flood is therefore drained by **one device thread**
//! that keeps popping the work list until it is empty:
//!
//! * the element pass that detects the trigger is untouched; a slot-parallel
//!   [`flood_seed_scan_kernel`] pushes every pixel that already carries the
//!   trigger value (`life == 8`, `tmp == -2`, `tmp2 == -2`, `tmp == 1`);
//! * [`flood_drain_kernel`] expands each queued pixel to its four neighbours
//!   of the target type, writes the field once per discovered pixel and pushes
//!   it; it runs alone, so the depth is not a frame budget;
//! * the queue is a fixed `pixels`-word ring-free list: a pixel is marked
//!   before it is pushed, so each pixel enters at most once and the tail can
//!   never pass the buffer end. The drain clears the marks of the queued pixels
//!   when it is done (a pixel stays marked for the whole walk, otherwise two
//!   expanded neighbours could queue each other forever), which leaves the
//!   buffer zero for the next flood, and resets the two counters.
//!
//! The serial walk is the price of an exact one-call flood without a host
//! round trip; it only runs while a trigger is present, and the node order is
//! irrelevant because every reachable pixel receives the same field value.
//!
//! # Field rules
//!
//! [`FloodMode::Life`] writes `mark_value` to `life` (the `LED` latch),
//! [`FloodMode::Tmp`] / [`FloodMode::Tmp2`] write the caller's sentinel, and
//! [`FloodMode::Bang`] mirrors the two `BANG` sentinels: an unarmed pixel
//! (`tmp == 0`) takes `mark_value`, a countdown pixel (`tmp == 2 || tmp == 3`)
//! takes `rearm_value`, and anything else is only traversed. The sentinels keep
//! the reference's one-frame ordering, which the consumers resolve in the next
//! pass (`explosives.rs`, `cexp_blast_kernel`, `mgnt_attract_kernel`).

use cubecl::prelude::*;

use crate::buffers::SimBuffers;
use crate::reaction_kernel::PassDims;
use crate::reaction_kernel::shared::{launch_grid, pmap_at, pmap_slot};

/// `[head, tail]` counters at the start of [`SimBuffers::gpu_flood_counters`].
const C_HEAD: usize = 0;
/// See [`C_HEAD`].
const C_TAIL: usize = 1;

/// Slot field a seed scan matches.
pub(crate) const FLOOD_FIELD_LIFE: u32 = 0;
/// See [`FLOOD_FIELD_LIFE`].
pub(crate) const FLOOD_FIELD_TMP: u32 = 1;
/// See [`FLOOD_FIELD_LIFE`].
pub(crate) const FLOOD_FIELD_TMP2: u32 = 2;

/// Field the drain writes (`life`).
pub(crate) const FLOOD_MODE_LIFE: u32 = 0;
/// Field the drain writes (`tmp`).
pub(crate) const FLOOD_MODE_TMP: u32 = 1;
/// Field the drain writes (`tmp2`).
pub(crate) const FLOOD_MODE_TMP2: u32 = 2;
/// `BANG`'s two-sentinel rule.
pub(crate) const FLOOD_MODE_BANG: u32 = 3;

/// Pushes the pixel of every seed slot into the queue.
///
/// One thread per pixel, like the reaction passes. A pixel enters the queue at
/// most once because the scan visits each pixel once, so the tail stays below
/// `queue_len`; the bound check is a guard, not a policy.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn flood_seed_scan_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_life: &[i32],
    part_tmp: &[i32],
    part_tmp2: &[i32],
    queue: &mut [u32],
    counters: &mut [Atomic<u32>],
    marks: &mut [u32],
    pixels: u32,
    field: u32,
    value_a: i32,
    value_b: i32,
    target_ty: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < pixels as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == target_ty {
                let mut cur = part_life[id];
                if field == FLOOD_FIELD_TMP {
                    cur = part_tmp[id];
                } else if field == FLOOD_FIELD_TMP2 {
                    cur = part_tmp2[id];
                }
                if cur == value_a || cur == value_b {
                    marks[idx] = 1u32;
                    let t = counters[C_TAIL].fetch_add(1u32);
                    if (t as usize) < queue.len() {
                        queue[t as usize] = idx as u32;
                    }
                }
            }
        }
    }
}

/// Writes the flood's field on one discovered pixel.
#[cube]
fn flood_apply(
    mode: u32,
    nid: usize,
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    mark_value: i32,
    rearm_value: i32,
) {
    if mode == FLOOD_MODE_LIFE {
        part_life[nid] = mark_value;
    } else if mode == FLOOD_MODE_TMP {
        part_tmp[nid] = mark_value;
    } else if mode == FLOOD_MODE_TMP2 {
        part_tmp2[nid] = mark_value;
    } else {
        let cur = part_tmp[nid];
        if cur == 0i32 {
            part_tmp[nid] = mark_value;
        } else if cur == 2i32 || cur == 3i32 {
            part_tmp[nid] = rearm_value;
        }
    }
}

/// Drains the queue, one launch per flood.
///
/// Runs as a single thread: the counters are only advanced here and by the
/// preceding seed scan, which has finished, so `tail` is stable apart from this
/// thread's own pushes. Every popped pixel clears its mark, so the buffer is
/// all-zero again when the loop ends.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn flood_drain_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    queue: &mut [u32],
    counters: &mut [Atomic<u32>],
    marks: &mut [u32],
    xres: u32,
    yres: u32,
    target_ty: u32,
    mode: u32,
    mark_value: i32,
    rearm_value: i32,
) {
    if ABSOLUTE_POS != 0usize {
        terminate!();
    }
    let mut head = 0u32;
    loop {
        let tail = counters[C_TAIL].fetch_add(0u32);
        if head >= tail {
            break;
        }
        let p = queue[head as usize];
        head += 1u32;

        let px = (p % xres) as i32;
        let py = (p / xres) as i32;
        for n in 0u32..4u32 {
            let mut nx = px;
            let mut ny = py;
            if n == 0u32 {
                nx = px - 1i32;
            } else if n == 1u32 {
                nx = px + 1i32;
            } else if n == 2u32 {
                ny = py - 1i32;
            } else {
                ny = py + 1i32;
            }
            if nx >= 0i32 && ny >= 0i32 && (nx as u32) < xres && (ny as u32) < yres {
                let np = (ny as u32 * xres + nx as u32) as usize;
                if marks[np] == 0u32 {
                    let nv = pmap_at(pmap, xres, yres, nx, ny);
                    if nv != 0u32 {
                        let nid = pmap_slot(nv);
                        if part_ty[nid] == target_ty {
                            marks[np] = 1u32;
                            flood_apply(
                                mode,
                                nid,
                                part_life,
                                part_tmp,
                                part_tmp2,
                                mark_value,
                                rearm_value,
                            );
                            let t = counters[C_TAIL].fetch_add(1u32);
                            if (t as usize) < queue.len() {
                                queue[t as usize] = np as u32;
                            }
                        }
                    }
                }
            }
        }
    }
    // A pixel keeps its mark while the walk runs: unmarking it at pop time
    // would let two already-expanded neighbours queue each other forever. Every
    // visited pixel sits in `queue[0..head]` exactly once, so clearing those
    // leaves the buffer zero for the next flood.
    let mut i = 0u32;
    while i < head {
        marks[queue[i as usize] as usize] = 0u32;
        i += 1u32;
    }
    counters[C_HEAD].store(0u32);
    counters[C_TAIL].store(0u32);
}

/// Queue every pixel of `target_ty` whose `field` holds `value_a` or
/// `value_b`.
pub(crate) fn launch_seed_scan(
    bufs: &SimBuffers,
    dims: &PassDims,
    field: u32,
    value_a: i32,
    value_b: i32,
    target_ty: u32,
) {
    let (cubes, dim) = launch_grid(dims.pixels);
    let pixels = dims.pixels as u32;
    unsafe {
        flood_seed_scan_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_flood_queue.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_flood_counters.clone(), 2),
            BufferArg::from_raw_parts(bufs.gpu_flood_marks.clone(), dims.pixels),
            pixels,
            field,
            value_a,
            value_b,
            target_ty,
        );
    }
}

/// Drain the queue into the 4-connected run of `target_ty`.
pub(crate) fn launch_drain(
    bufs: &SimBuffers,
    dims: &PassDims,
    mode: u32,
    target_ty: u32,
    mark_value: i32,
    rearm_value: i32,
) {
    unsafe {
        flood_drain_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_flood_queue.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_flood_counters.clone(), 2),
            BufferArg::from_raw_parts(bufs.gpu_flood_marks.clone(), dims.pixels),
            dims.xres,
            dims.yres,
            target_ty,
            mode,
            mark_value,
            rearm_value,
        );
    }
}
