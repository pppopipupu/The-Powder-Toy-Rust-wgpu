//! `ETRD` plasma bolt pass.
//!
//! When an electrode fires into an idle electrode, `arcs.rs` converts the target
//! to a spark and latches [`FLAG_ETRD_BOLT`] on the source along with the two
//! endpoints of the arc (`SPRK.cpp:100-113`, the `case PT_ETRD` branch, which
//! calls `Simulation::CreateLine(..., PT_PLSM)`).
//!
//! A kernel cannot allocate pool slots, so this pass runs the Bresenham walk
//! (`Simulation.cpp:803-853`) and pushes one creation record per pixel through
//! the device-side creation queue (`shared::request_create`). The queue is
//! drained right after the reaction passes, so the plasma is live for the
//! movement and render passes of the same frame and each pixel keeps `PLSM`'s
//! own `DefaultProperties` (temperature `MAX_TEMP`, and the `Create` hook's
//! random `50..=199` life) rather than inheriting the electrode's state.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::PassDims;
use crate::buffers::SimBuffers;
use crate::reaction_kernel::shared::{CREATE_QUEUE_WORDS, iabs, launch_grid, request_create_plain};

/// Flag bit on `part_flags` marking an electrode that has an arcing plasma bolt
/// pending for this frame.
pub(in crate::reaction_kernel) const FLAG_ETRD_BOLT: u32 = 0x0001_0000;
const FLAG_ETRD_BOLT_U32: u32 = 0x0001_0000;

const PT_PLSM_U32: u32 = 28;
const _: () = assert!(PT_PLSM == 28);

/// `ETRD_SEARCH_RADIUS` (`ETRD.cpp:74`): the arc never spans more than twelve
/// pixels, so the walk below is bounded by 64 steps with room to spare.
const ETRD_MAX_STEPS: u32 = 64;

#[cube]
fn seed_i32(val: i32) -> i32 {
    val
}

#[cube]
fn seed_f32(val: f32) -> f32 {
    val
}

#[cube]
fn seed_bool(val: bool) -> bool {
    val
}

/// Pack 2D integer coordinates `(x, y)` into an `i32` word for device kernels.
#[cube]
pub(super) fn pack_coords_cube(x: i32, y: i32) -> i32 {
    let ux = (x & 0xFFFFi32) as u32;
    let uy = (y & 0xFFFFi32) as u32;
    (ux | (uy << 16u32)) as i32
}

/// Unpack 2D integer coordinates `(x, y)` from an `i32` word for device kernels.
#[cube]
pub(super) fn unpack_coords_cube(packed: i32) -> (i32, i32) {
    let u = packed as u32;
    let x = (u & 0xFFFFu32) as i32;
    let y = ((u >> 16u32) & 0xFFFFu32) as i32;
    (x, y)
}

/// Host mirror of [`pack_coords_cube`].
#[cfg(test)]
pub(super) fn pack_coords_host(x: i32, y: i32) -> i32 {
    let ux = (x & 0xFFFF) as u32;
    let uy = (y & 0xFFFF) as u32;
    (ux | (uy << 16)) as i32
}

/// Host mirror of [`unpack_coords_cube`].
#[cfg(test)]
pub(super) fn unpack_coords_host(packed: i32) -> (i32, i32) {
    let u = packed as u32;
    let x = (u & 0xFFFF) as i32;
    let y = ((u >> 16) & 0xFFFF) as i32;
    (x, y)
}

/// Slot-parallel pass scanning for pending `ETRD` bolts and queueing the
/// `PLSM` pixels of `Simulation::CreateLine`.
#[cube(launch_unchecked)]
// The axis/endpoint swaps below mirror `Simulation::CreateLine` statement for
// statement; `std::mem::swap` has no CubeCL expansion.
#[allow(clippy::manual_swap)]
pub(crate) fn etrd_bolt_kernel(
    create_queue: &mut [Atomic<u32>],
    part_flags: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    xres: u32,
    yres: u32,
    capacity: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < capacity as usize {
        let flags = part_flags[idx];
        if (flags & FLAG_ETRD_BOLT_U32) != 0u32 {
            part_flags[idx] = flags & !FLAG_ETRD_BOLT_U32;

            let (p_x1, p_y1) = unpack_coords_cube(part_tmp[idx]);
            let (p_x2, p_y2) = unpack_coords_cube(part_tmp2[idx]);
            part_tmp[idx] = 0i32;
            part_tmp2[idx] = 0i32;

            let mut x1 = seed_i32(p_x1);
            let mut y1 = seed_i32(p_y1);
            let mut x2 = seed_i32(p_x2);
            let mut y2 = seed_i32(p_y2);

            // `Simulation.cpp:805-827`: swap the axes when the line is steeper
            // than 45 degrees, then always walk left to right.
            let dx_orig = iabs(x2 - x1);
            let dy_orig = iabs(y2 - y1);
            let reverse_xy = seed_bool(dy_orig > dx_orig);
            if reverse_xy {
                let t1 = x1;
                x1 = y1;
                y1 = t1;
                let t2 = x2;
                x2 = y2;
                y2 = t2;
            }
            if x1 > x2 {
                let tx = x1;
                x1 = x2;
                x2 = tx;
                let ty = y1;
                y1 = y2;
                y2 = ty;
            }

            let dx = x2 - x1;
            let dy = iabs(y2 - y1);
            let mut e = seed_f32(0.0f32);
            let de = if dx > 0i32 {
                dy as f32 / dx as f32
            } else {
                0.0f32
            };
            let mut cur_y = seed_i32(y1);
            let sy = if y1 < y2 { 1i32 } else { -1i32 };
            let total_steps = (dx + 1i32) as u32;

            for step in 0u32..ETRD_MAX_STEPS {
                if step < total_steps {
                    let px = if reverse_xy { cur_y } else { x1 + step as i32 };
                    let py = if reverse_xy { x1 + step as i32 } else { cur_y };
                    if px >= 0i32 && py >= 0i32 && (px as u32) < xres && (py as u32) < yres {
                        request_create_plain(create_queue, PT_PLSM_U32, px, py);
                    }

                    e += de;
                    if e >= 0.5f32 {
                        cur_y += sy;
                        let within = if y1 < y2 { cur_y <= y2 } else { cur_y >= y2 };
                        if within {
                            let px2 = if reverse_xy { cur_y } else { x1 + step as i32 };
                            let py2 = if reverse_xy { x1 + step as i32 } else { cur_y };
                            if px2 >= 0i32
                                && py2 >= 0i32
                                && (px2 as u32) < xres
                                && (py2 as u32) < yres
                            {
                                request_create_plain(create_queue, PT_PLSM_U32, px2, py2);
                            }
                        }
                        e -= 1.0f32;
                    }
                }
            }
        }
    }
}

/// Launch the `ETRD` plasma bolt pass. Records land in the creation queue, which
/// `step_reactions_gpu` drains after every reaction pass.
pub(in crate::reaction_kernel) fn launch_etrd_bolt_pass(bufs: &SimBuffers, dims: &PassDims) {
    if dims.capacity == 0 || dims.pixels == 0 {
        return;
    }
    let (cubes, dim) = launch_grid(dims.used_slots);
    unsafe {
        etrd_bolt_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.used_slots as u32,
        );
    }
}

/// Host mirror of the Bresenham line walk (`Simulation::CreateLine` in
/// `Simulation.cpp:803-853`) for unit testing.
#[cfg(test)]
pub(super) fn bresenham_line_points(
    mut x1: i32,
    mut y1: i32,
    mut x2: i32,
    mut y2: i32,
) -> Vec<(i32, i32)> {
    let mut points = Vec::new();
    let dx_orig = (x2 - x1).abs();
    let dy_orig = (y2 - y1).abs();
    let reverse_xy = dy_orig > dx_orig;
    if reverse_xy {
        std::mem::swap(&mut x1, &mut y1);
        std::mem::swap(&mut x2, &mut y2);
    }
    if x1 > x2 {
        std::mem::swap(&mut x1, &mut x2);
        std::mem::swap(&mut y1, &mut y2);
    }
    let dx = x2 - x1;
    let dy = (y2 - y1).abs();
    let mut e = 0.0f32;
    let de = if dx > 0 {
        dy as f32 / dx as f32
    } else {
        0.0f32
    };
    let mut cur_y = y1;
    let sy = if y1 < y2 { 1 } else { -1 };

    for cur_x in x1..=x2 {
        let p = if reverse_xy {
            (cur_y, cur_x)
        } else {
            (cur_x, cur_y)
        };
        points.push(p);
        e += de;
        if e >= 0.5f32 {
            cur_y += sy;
            let within = if y1 < y2 { cur_y <= y2 } else { cur_y >= y2 };
            if within {
                let p2 = if reverse_xy {
                    (cur_y, cur_x)
                } else {
                    (cur_x, cur_y)
                };
                points.push(p2);
            }
            e -= 1.0f32;
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_packing_roundtrip() {
        for (x, y) in [(0, 0), (10, 20), (2895, 2895), (100, 500), (0, 65535)] {
            let packed = pack_coords_host(x, y);
            let (ux, uy) = unpack_coords_host(packed);
            assert_eq!((ux, uy), (x, y));
        }
    }

    #[test]
    fn bresenham_horizontal_and_vertical() {
        let h = bresenham_line_points(2, 5, 6, 5);
        assert_eq!(h, vec![(2, 5), (3, 5), (4, 5), (5, 5), (6, 5)]);

        let v = bresenham_line_points(3, 2, 3, 5);
        assert_eq!(v, vec![(3, 2), (3, 3), (3, 4), (3, 5)]);
    }

    #[test]
    fn bresenham_diagonal_has_interpolated_steps() {
        let d = bresenham_line_points(0, 0, 3, 3);
        // Step de = 1.0 >= 0.5, so each x step emits (x, y) and (x, y + 1)
        assert!(d.contains(&(0, 0)));
        assert!(d.contains(&(3, 3)));
        assert_eq!(d.first().copied(), Some((0, 0)));
        assert_eq!(d.last().copied(), Some((3, 3)));
    }
}
