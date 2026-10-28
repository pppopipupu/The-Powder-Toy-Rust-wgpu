#![allow(clippy::manual_range_contains)]

//! CubeCL kernels for the 64³ air/pressure field (plan §5).
//!
//! The coarse grid is `AIR_AXIS³` cells, four voxels along each axis. Cells are
//! stored `y` major, then `z`, then `x`, so the linear index is
//! `(y << 12) | (z << 6) | x`; one `x` step is `+1`, one `z` step `+64` and
//! one `y` step `+4096`.
//!
//! # Passes
//!
//! | kernel | buffers | work |
//! |---|---|---|
//! | [`air_clear_kernel`] | 7 write | neutral field: zero velocity/pressure/solid, ambient temperature |
//! | [`air_gather_kernel`] | 2 read, 2 write | one thread per 4³ cell of an active brick |
//! | [`air_velocity_snapshot_kernel`] | 3 read, 1 write | copy the velocity field before the update |
//! | [`air_pressure_jacobi_kernel`] | 6 read, 1 write | one pressure Jacobi step |
//! | [`air_advect_kernel`] | 4 read, 3 write | damping, buoyancy, pressure gradient, upwind averaging |
//!
//! The snapshot exists because the advection step reads the six velocity
//! neighbours of every cell. Reading them from the buffer the same launch is
//! writing would make the result depend on thread scheduling; copying the
//! field first keeps the pass race free and deterministic while still using a
//! single `wx + wy + wz` buffer.
//!
//! Every kernel here is launched either from the compacted brick list
//! (`air_gather_kernel`, dynamic count) or over the fixed 64³ grid
//! (`1024` workgroups of `256` threads cover `262_144` cells). No pass reads
//! anything back to the host.

use cubecl::prelude::*;
use tpt_core::voxel3d::AIR_COUNT;

/// Coarse air cells per world axis.
pub const AIR_AXIS: usize = 64;

/// Total number of coarse air cells, `AIR_AXIS³`.
pub const AIR_CELL_COUNT: usize = AIR_AXIS * AIR_AXIS * AIR_AXIS;

/// Threads per workgroup for the fixed 64³ passes.
pub const AIR_THREADS: usize = 256;

/// Workgroups of the fixed 64³ dispatch.
pub const AIR_WORKGROUPS: usize = AIR_CELL_COUNT / AIR_THREADS;

/// Default number of Jacobi pressure iterations (plan §5.3).
pub const AIR_JACOBI_ITERATIONS: u32 = 12;

/// Index stride between two cells along `z`.
const STRIDE_Z: usize = AIR_AXIS;

/// Index stride between two cells along `y`.
const STRIDE_Y: usize = AIR_AXIS * AIR_AXIS;

/// Ambient air temperature in Kelvin, matching `Voxel3dConfig::ambient_temp_k`.
const AMBIENT_K: f32 = 293.15;

/// Velocity gained per Kelvin above ambient, in cells per tick.
const BUOYANCY: f32 = 0.02;

/// Velocity damping applied once per solve.
const DAMPING: f32 = 0.99;

/// Weight of the six-neighbour velocity average in the update.
const NEIGHBOUR_WEIGHT: f32 = 0.1;

/// Weight of the upwind donor sample in the update.
const UPWIND_WEIGHT: f32 = 0.25;

/// Maximum absolute air velocity, in cells per tick.
const MAX_SPEED: f32 = 8.0;

const _: () = assert!(AIR_CELL_COUNT == AIR_COUNT);
const _: () = assert!(AIR_WORKGROUPS * AIR_THREADS == AIR_CELL_COUNT);

/// Clamps an air velocity component to [`MAX_SPEED`].
///
/// `f32::clamp` is unavailable inside a cube body, so the two comparisons are
/// spelled out.
#[allow(clippy::manual_clamp)]
#[cube]
fn clamp_speed(value: f32) -> f32 {
    let mut out = value;
    if out > MAX_SPEED {
        out = MAX_SPEED;
    }
    if out < -MAX_SPEED {
        out = -MAX_SPEED;
    }
    out
}

/// Average of one snapshot velocity component over the cell and its six
/// in-world neighbours.
///
/// Out-of-world neighbours are omitted instead of being clamped to the centre,
/// which avoids a systematic bias against the world faces where the hot source
/// is most likely to sit.
#[cube]
fn neighbour_average(
    snapshot: &[f32],
    base: usize,
    index: usize,
    x: usize,
    y: usize,
    z: usize,
) -> f32 {
    let mut sum = snapshot[base + index];
    let mut count = 1.0f32;
    if x > 0usize {
        sum += snapshot[base + index - 1usize];
        count += 1.0f32;
    }
    if x + 1usize < AIR_AXIS {
        sum += snapshot[base + index + 1usize];
        count += 1.0f32;
    }
    if z > 0usize {
        sum += snapshot[base + index - STRIDE_Z];
        count += 1.0f32;
    }
    if z + 1usize < AIR_AXIS {
        sum += snapshot[base + index + STRIDE_Z];
        count += 1.0f32;
    }
    if y > 0usize {
        sum += snapshot[base + index - STRIDE_Y];
        count += 1.0f32;
    }
    if y + 1usize < AIR_AXIS {
        sum += snapshot[base + index + STRIDE_Y];
        count += 1.0f32;
    }
    sum / count
}

/// Donor-cell sample of one velocity component along its own axis.
///
/// A positive velocity means the flow arrives from the lower coordinate, a
/// negative velocity from the upper one; at a world face the cell keeps its
/// own value. `axis` is `0` for x, `1` for y and `2` for z.
#[cube]
fn upwind_sample(
    snapshot: &[f32],
    base: usize,
    index: usize,
    current: f32,
    axis: usize,
    x: usize,
    y: usize,
    z: usize,
) -> f32 {
    let mut sample = current;
    if current > 0.0f32 {
        if axis == 0usize && x > 0usize {
            sample = snapshot[base + index - 1usize];
        }
        if axis == 1usize && y > 0usize {
            sample = snapshot[base + index - STRIDE_Y];
        }
        if axis == 2usize && z > 0usize {
            sample = snapshot[base + index - STRIDE_Z];
        }
    } else if current < 0.0f32 {
        if axis == 0usize && x + 1usize < AIR_AXIS {
            sample = snapshot[base + index + 1usize];
        }
        if axis == 1usize && y + 1usize < AIR_AXIS {
            sample = snapshot[base + index + STRIDE_Y];
        }
        if axis == 2usize && z + 1usize < AIR_AXIS {
            sample = snapshot[base + index + STRIDE_Z];
        }
    }
    sample
}

/// Fills the air field with its neutral state.
///
/// Velocities, pressure and the solid fraction start at zero; the temperature
/// starts at [`AMBIENT_K`] so untouched regions do not act as a 0 K heat sink.
#[cube(launch_unchecked)]
pub(crate) fn air_clear_kernel(
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
    p: &mut [f32],
    p_tmp: &mut [f32],
    temp: &mut [f32],
    solid: &mut [f32],
    pv: &mut [Atomic<i32>],
) {
    let index = ABSOLUTE_POS;
    if index < AIR_CELL_COUNT {
        vx[index] = 0.0f32;
        vy[index] = 0.0f32;
        vz[index] = 0.0f32;
        p[index] = 0.0f32;
        p_tmp[index] = 0.0f32;
        temp[index] = AMBIENT_K;
        solid[index] = 0.0f32;
        pv[index].store(0i32);
    }
}

/// Reduces the voxels of one simulated brick into its eight air cells.
///
/// One workgroup handles one brick from `sim_list`; the eight threads of the
/// group each own one 4³ cell (thread bit 0 selects x, bit 1 z, bit 2 y, so
/// the 2×2×2 block of cells is covered exactly once). Every cell overwrites
/// its `solid` fraction and temperature directly: bricks and air cells are
/// aligned, so no atomics are needed and inactive bricks keep their previous
/// values.
///
/// `solid` is the fraction of non-empty voxels in the cell. The temperature is
/// the mean of the non-empty voxels; a cell without voxels stays at ambient
/// temperature.
#[cube(launch_unchecked)]
pub(crate) fn air_gather_kernel(
    voxel_a: &[u32],
    sim_list: &[u32],
    solid: &mut [f32],
    temp: &mut [f32],
) {
    let brick = sim_list[CUBE_POS];

    // Brick coordinates from the packed brick index.
    let brick_x = (brick & 31u32) as usize;
    let brick_z = ((brick >> 5u32) & 31u32) as usize;
    let brick_y = ((brick >> 10u32) & 31u32) as usize;

    // 2x2x2 air cells per brick, one thread each.
    let slot = UNIT_POS_X as usize;
    let cell_x = brick_x * 2usize + (slot & 1usize);
    let cell_z = brick_z * 2usize + ((slot >> 1usize) & 1usize);
    let cell_y = brick_y * 2usize + ((slot >> 2usize) & 1usize);

    let air = (cell_y << 12usize) | (cell_z << 6usize) | cell_x;

    let base_x = cell_x * 4usize;
    let base_y = cell_y * 4usize;
    let base_z = cell_z * 4usize;

    let mut occupied = 0usize;
    let mut temp_sum = 0.0f32;

    let mut local_y = 0usize;
    while local_y < 4usize {
        let mut local_z = 0usize;
        while local_z < 4usize {
            let mut local_x = 0usize;
            while local_x < 4usize {
                let world_x = base_x + local_x;
                let world_y = base_y + local_y;
                let world_z = base_z + local_z;
                let voxel = voxel_a[(world_y << 16usize) | (world_z << 8usize) | world_x];

                let ty = voxel & 0xffu32;
                if ty != 0u32 {
                    occupied += 1usize;
                    let temp_q = (voxel >> 8u32) & 0xffffu32;
                    temp_sum += temp_q as f32 * 0.125f32;
                }

                local_x += 1usize;
            }
            local_z += 1usize;
        }
        local_y += 1usize;
    }

    solid[air] = occupied as f32 * (1.0f32 / 64.0f32);
    if occupied > 0usize {
        temp[air] = temp_sum / occupied as f32;
    } else {
        temp[air] = AMBIENT_K;
    }
}

/// Advances the TPT `pv` pressure field from the air velocity divergence.
///
/// This is `Air::update_air`'s "pressure adjustments from velocity"
/// (`Air.cpp:315-327`) lifted to three dimensions:
///
/// ```text
/// pv = Mix(edgePressure, pv, AIR_PLOSS);
/// pv += dp * AIR_TSTEPP * 0.5;
/// ```
///
/// with `dp` the six-neighbour velocity divergence and `AIR_PLOSS = 0.9999`,
/// `AIR_TSTEPP = 0.3` (`SimulationConfig.h:37-41`). `pv` is stored as `i32`
/// fixed point with [`PV_SCALE`] units per TPT pressure unit; the near-edge
/// cells are damped to zero like the two-cell rim in `Air.cpp:271-299`.
#[cube(launch_unchecked)]
pub(crate) fn air_pressure_volume_kernel(
    vx: &[f32],
    vy: &[f32],
    vz: &[f32],
    pv: &mut [Atomic<i32>],
) {
    let index = ABSOLUTE_POS;
    if index < AIR_CELL_COUNT {
        let x = index & 63usize;
        let z = (index >> 6usize) & 63usize;
        let y = index >> 12usize;
        let mut value = pv[index].load();
        if x < 2usize || x >= 62usize || y < 2usize || y >= 62usize || z < 2usize || z >= 62usize {
            // Mix(0, pv, 0.8) == pv * 0.2 at the world rim.
            value = (value as f32 * 0.2f32) as i32;
        } else {
            let dp = (vx[index - 1usize] - vx[index + 1usize])
                + (vy[index - STRIDE_Y] - vy[index + STRIDE_Y])
                + (vz[index - STRIDE_Z] - vz[index + STRIDE_Z]);
            let mixed = value as f32 * 0.9999f32;
            let next = mixed + dp * 0.3f32 * 0.5f32 * (PV_SCALE as f32);
            if next > MAX_PV_Q {
                value = MAX_PV_Q as i32;
            } else if next < -MAX_PV_Q {
                value = -MAX_PV_Q as i32;
            } else {
                value = next as i32;
            }
        }
        pv[index].store(value);
    }
}

/// Fixed-point units per TPT pressure unit.
pub const PV_SCALE: i32 = 1024;
/// `MAX_PRESSURE` (`ElementDefs.h:7`) in fixed-point units.
pub const MAX_PV_Q: f32 = 256.0 * PV_SCALE as f32;

/// Copies `vx`, `vy`, `vz` into the interleaved snapshot buffer.
///
/// Snapshot layout is component major: the x component occupies the first
/// [`AIR_CELL_COUNT`] words, y the second and z the third. The advection pass
/// reads the snapshot, so it never observes a half-updated velocity field.
#[cube(launch_unchecked)]
pub(crate) fn air_velocity_snapshot_kernel(
    vx: &[f32],
    vy: &[f32],
    vz: &[f32],
    snapshot: &mut [f32],
) {
    let index = ABSOLUTE_POS;
    if index < AIR_CELL_COUNT {
        snapshot[index] = vx[index];
        snapshot[AIR_CELL_COUNT + index] = vy[index];
        snapshot[2usize * AIR_CELL_COUNT + index] = vz[index];
    }
}

/// One Jacobi iteration for the air pressure.
///
/// Interior cells solve `∇²p = div v` as
/// `p = (Σ p_neighbour - div v * h²) / 6` with `h = 1` cell. The divergence
/// uses the central difference of the velocity field, which is constant
/// during the pressure solve. World faces and cells that are more than half
/// solid are Dirichlet boundaries with `p = 0`; the top face is therefore open
/// like the plan requires.
#[cube(launch_unchecked)]
pub(crate) fn air_pressure_jacobi_kernel(
    pressure: &[f32],
    pressure_out: &mut [f32],
    vx: &[f32],
    vy: &[f32],
    vz: &[f32],
    solid: &[f32],
) {
    let index = ABSOLUTE_POS;
    if index < AIR_CELL_COUNT {
        let x = index & 63usize;
        let z = (index >> 6usize) & 63usize;
        let y = index >> 12usize;

        let boundary = x == 0usize
            || y == 0usize
            || z == 0usize
            || x == 63usize
            || y == 63usize
            || z == 63usize;
        if boundary || solid[index] > 0.5f32 {
            pressure_out[index] = 0.0f32;
        } else {
            let divergence = 0.5f32
                * ((vx[index + 1usize] - vx[index - 1usize])
                    + (vy[index + STRIDE_Y] - vy[index - STRIDE_Y])
                    + (vz[index + STRIDE_Z] - vz[index - STRIDE_Z]));
            let neighbours = pressure[index - 1usize]
                + pressure[index + 1usize]
                + pressure[index - STRIDE_Z]
                + pressure[index + STRIDE_Z]
                + pressure[index - STRIDE_Y]
                + pressure[index + STRIDE_Y];
            pressure_out[index] = (neighbours - divergence) / 6.0f32;
        }
    }
}

/// One advection/forcing step of the air field.
///
/// For every non-solid cell the kernel
///
/// * diffuses the local temperature over the six in-world neighbours and adds
///   the buoyancy `vy += (T - 293.15) * 0.02`,
/// * damps the velocity,
/// * subtracts `0.5 * grad p` using the converged pressure field,
/// * blends in the six-neighbour average and an upwind donor sample taken from
///   the snapshot, and clamps the result to `±8` cells per tick.
///
/// Solid cells (`solid > 0.5`) end with zero velocity. All neighbour reads come
/// from the snapshot, the temperature, the pressure and the solid fraction, so
/// the launch is race free even though it writes the velocity buffers in place.
#[cube(launch_unchecked)]
pub(crate) fn air_advect_kernel(
    snapshot: &[f32],
    pressure: &[f32],
    temp: &[f32],
    solid: &[f32],
    vx: &mut [f32],
    vy: &mut [f32],
    vz: &mut [f32],
) {
    let index = ABSOLUTE_POS;
    if index < AIR_CELL_COUNT {
        if solid[index] > 0.5f32 {
            vx[index] = 0.0f32;
            vy[index] = 0.0f32;
            vz[index] = 0.0f32;
        } else {
            let x = index & 63usize;
            let z = (index >> 6usize) & 63usize;
            let y = index >> 12usize;

            // Neighbour-averaged temperature: the gathered values change only
            // in the walls of a brick, but smoothing them here keeps the
            // buoyancy field continuous across brick boundaries.
            let mut temp_sum = temp[index];
            let mut temp_count = 1.0f32;
            if x > 0usize {
                temp_sum += temp[index - 1usize];
                temp_count += 1.0f32;
            }
            if x + 1usize < AIR_AXIS {
                temp_sum += temp[index + 1usize];
                temp_count += 1.0f32;
            }
            if z > 0usize {
                temp_sum += temp[index - STRIDE_Z];
                temp_count += 1.0f32;
            }
            if z + 1usize < AIR_AXIS {
                temp_sum += temp[index + STRIDE_Z];
                temp_count += 1.0f32;
            }
            if y > 0usize {
                temp_sum += temp[index - STRIDE_Y];
                temp_count += 1.0f32;
            }
            if y + 1usize < AIR_AXIS {
                temp_sum += temp[index + STRIDE_Y];
                temp_count += 1.0f32;
            }
            let local_temp = temp_sum / temp_count;

            // One-sided difference at the world faces, central inside.
            let mut lower_x = index;
            if x > 0usize {
                lower_x = index - 1usize;
            }
            let mut upper_x = index;
            if x + 1usize < AIR_AXIS {
                upper_x = index + 1usize;
            }
            let mut lower_z = index;
            if z > 0usize {
                lower_z = index - STRIDE_Z;
            }
            let mut upper_z = index;
            if z + 1usize < AIR_AXIS {
                upper_z = index + STRIDE_Z;
            }
            let mut lower_y = index;
            if y > 0usize {
                lower_y = index - STRIDE_Y;
            }
            let mut upper_y = index;
            if y + 1usize < AIR_AXIS {
                upper_y = index + STRIDE_Y;
            }

            let grad_x = (pressure[upper_x] - pressure[lower_x]) * 0.5f32;
            let grad_y = (pressure[upper_y] - pressure[lower_y]) * 0.5f32;
            let grad_z = (pressure[upper_z] - pressure[lower_z]) * 0.5f32;

            let old_x = snapshot[index];
            let old_y = snapshot[AIR_CELL_COUNT + index];
            let old_z = snapshot[2usize * AIR_CELL_COUNT + index];

            let buoyancy = (local_temp - AMBIENT_K) * BUOYANCY;

            let mut next_x = (old_x - grad_x) * DAMPING;
            let mut next_y = (old_y + buoyancy - grad_y) * DAMPING;
            let mut next_z = (old_z - grad_z) * DAMPING;

            let average_x = neighbour_average(snapshot, 0usize, index, x, y, z);
            let average_y = neighbour_average(snapshot, AIR_CELL_COUNT, index, x, y, z);
            let average_z = neighbour_average(snapshot, 2usize * AIR_CELL_COUNT, index, x, y, z);

            next_x = next_x * (1.0f32 - NEIGHBOUR_WEIGHT) + average_x * NEIGHBOUR_WEIGHT;
            next_y = next_y * (1.0f32 - NEIGHBOUR_WEIGHT) + average_y * NEIGHBOUR_WEIGHT;
            next_z = next_z * (1.0f32 - NEIGHBOUR_WEIGHT) + average_z * NEIGHBOUR_WEIGHT;

            let upwind_x = upwind_sample(snapshot, 0usize, index, old_x, 0usize, x, y, z);
            let upwind_y = upwind_sample(snapshot, AIR_CELL_COUNT, index, old_y, 1usize, x, y, z);
            let upwind_z = upwind_sample(
                snapshot,
                2usize * AIR_CELL_COUNT,
                index,
                old_z,
                2usize,
                x,
                y,
                z,
            );

            next_x = next_x * (1.0f32 - UPWIND_WEIGHT) + upwind_x * UPWIND_WEIGHT;
            next_y = next_y * (1.0f32 - UPWIND_WEIGHT) + upwind_y * UPWIND_WEIGHT;
            next_z = next_z * (1.0f32 - UPWIND_WEIGHT) + upwind_z * UPWIND_WEIGHT;

            vx[index] = clamp_speed(next_x);
            vy[index] = clamp_speed(next_y);
            vz[index] = clamp_speed(next_z);
        }
    }
}
