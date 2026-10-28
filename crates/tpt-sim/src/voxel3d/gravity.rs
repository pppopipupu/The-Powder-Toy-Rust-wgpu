//! Device-resident 3D Newtonian gravity.
//!
//! TPT solves gravity as a convolution of the mass grid with the point-mass
//! kernel `-G * d / |d|^3` (`gravity/Fft.cpp:22,162-180`) and adds the result
//! to every particle's own gravity in `Simulation::GetGravityField`
//! (`Simulation.cpp:2088-2120`). The 3D port keeps the same physical model on
//! the 64³ coarse grid that already carries pressure and temperature:
//!
//! 1. [`gather`] walks the tick's active bricks, cell by cell, and accumulates
//!    the mass each gravity element contributes (`GRVT.cpp:53-80`,
//!    `NBHL.cpp:51-61`, `NWHL.cpp:51-61`, `GBMB.cpp:53-68`, `GPMP.cpp:53-84`)
//!    into [`Voxel3dBuffers::grav_mass`] as `i32` fixed point.
//! 2. [`solve`] relaxes the discrete Poisson equation
//!    `laplacian(u) = 4 * pi * G * rho` with Jacobi sweeps. The force used by
//!    the movement passes is `-grad(u)`, which is the same field the C++ FFT
//!    convolution evaluates with its point-mass kernel.
//! 3. [`clear_mass`] resets the accumulator after the solve, exactly where
//!    `Simulation::BeforeSim` clears `gravIn.mass` after dispatching
//!    (`Simulation.cpp:3739-3743`). Mass therefore has the same one-frame lag
//!    as the asynchronous C++ FFT.
//!
//! Everything is launched over the existing coarse grid (262144 cells) or the
//! compacted active-brick list; no pass scans the 16.7M voxels and no value is
//! copied back to the host.
//!
//! Clippy's `manual_range_contains` rewrite produces a construct CubeCL does
//! not expand, so it is disabled for the module.
#![allow(clippy::manual_range_contains)]

use cubecl::prelude::*;

use super::Voxel3dBuffers;
use super::engine::Voxel3dEngine;
use super::step::{
    C_PT_GBMB, C_PT_GPMP, C_PT_GRVT, C_PT_NBHL, C_PT_NWHL, C_TEMP_MASK, C_TEMP_SHIFT, C_TY_MASK,
    clampf, seed_f32, seed_i32,
};
use tpt_core::voxel3d::{AIR_COUNT, VOXEL_COUNT};

/// Number of cells along one edge of the coarse gravity grid.
pub const GRAV_DIM: u32 = 64;
/// Cells in the coarse gravity grid.
pub const GRAV_CELLS: u32 = GRAV_DIM * GRAV_DIM * GRAV_DIM;
/// Cells along one `z` stride.
const GRAV_STRIDE_Z: u32 = GRAV_DIM;
/// Cells along one `y` stride.
const GRAV_STRIDE_Y: u32 = GRAV_DIM * GRAV_DIM;
/// Jacobi sweeps per tick.
pub const GRAVITY_ITERATIONS: u32 = 24;
/// Threads per coarse-grid launch.
const GRAV_THREADS: u32 = 256;
/// Workgroups per coarse-grid launch.
const GRAV_GROUPS: u32 = GRAV_CELLS / GRAV_THREADS;
/// `i32` fixed-point units per TPT gravity mass unit.
pub const MASS_SCALE: f32 = 1024.0;
/// TPT `M_GRAV` (`SimulationConfig.h:8`).
pub const M_GRAV: f32 = 6.673_00e-1;
/// `4 * pi * G`, the discrete Poisson source factor.
pub const FOUR_PI_G: f32 = 8.385_539;
/// Conversion from one TPT gravity unit to the 3D acceleration units.
///
/// C++ `GetGravityField` adds `newtonGrav * force` to the particle's own
/// `Gravity`, a value of order `0.1..1` (`URAN.cpp:19`). The 3D configuration
/// stores gravity in voxels per second squared (`9.81` by default), so one TPT
/// gravity unit is one standard gravity.
pub const NEWTON_SCALE: f32 = 9.81;

const _: () = assert!(GRAV_CELLS == AIR_COUNT as u32);
const _: () = assert!(GRAV_CELLS == 262_144);
const _: () = assert!(GRAV_GROUPS * GRAV_THREADS == GRAV_CELLS);
const _: () = assert!(GRAV_STRIDE_Y == 4_096);

/// Fixed-point mass of one voxel, or `0` for a non-gravity element.
#[cube]
fn voxel_mass(ty: u32, life: u32, tmp: i32, temp_k: f32) -> i32 {
    let mut out = seed_i32(tmp, 0i32);
    if ty == C_PT_GRVT {
        // `GRVT.cpp:53-56,80`: clamp tmp to +-100, then `0.2 * tmp`.
        let t = clampf(tmp as f32, -100.0f32, 100.0f32) as i32;
        out = (0.2f32 * t as f32 * MASS_SCALE) as i32;
    } else if ty == C_PT_NBHL {
        // `NBHL.cpp:51-61`: 0.001 * tmp clamped to 0.1..51.2, or 0.1.
        let mut add = seed_f32(temp_k, 0.1f32);
        if tmp != 0i32 {
            add = clampf(0.001f32 * tmp as f32, 0.1f32, 51.2f32);
        }
        out = (add * MASS_SCALE) as i32;
    } else if ty == C_PT_NWHL {
        // `NWHL.cpp:51-61`: the negative of the black-hole contribution.
        let mut add = seed_f32(temp_k, 0.1f32);
        if tmp != 0i32 {
            add = clampf(0.001f32 * tmp as f32, 0.1f32, 51.2f32);
        }
        out = (-add * MASS_SCALE) as i32;
    } else if ty == C_PT_GBMB {
        // `GBMB.cpp:66-75`: 20 units while the fuse burns, -80 while it pushes.
        if life > 20u32 {
            out = (20.0f32 * MASS_SCALE) as i32;
        } else if life >= 1u32 {
            out = (-80.0f32 * MASS_SCALE) as i32;
        }
    } else if ty == C_PT_GPMP {
        // `GPMP.cpp:66-70`: `0.2 * (temp - 273.15)` while life == 10.
        if life == 10u32 {
            out = (0.2f32 * (temp_k - 273.15f32) * MASS_SCALE) as i32;
        }
    }
    out
}

/// Zeros the mass accumulator.
#[cube(launch_unchecked)]
fn gravity_clear_kernel(mass: &mut [Atomic<i32>]) {
    let i = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if i < GRAV_CELLS {
        mass[i as usize].store(0i32);
    }
}

/// Accumulates the gravity mass of the tick's active bricks.
///
/// One workgroup of eight threads handles one 8³ brick; each thread scans the
/// 4³ cell it owns and performs a single atomic add, mirroring the air gather
/// (`Air::gather`) and obeying the active-brick dispatch rule.
#[cube(launch_unchecked)]
fn gravity_gather_kernel(
    voxel_a: &[u32],
    aux_tmp: &[i32],
    sim_list: &[u32],
    mass: &mut [Atomic<i32>],
) {
    let brick = sim_list[CUBE_POS_X as usize];
    let brick_x = brick & 31u32;
    let brick_z = (brick >> 5u32) & 31u32;
    let brick_y = (brick >> 10u32) & 31u32;

    let slot = UNIT_POS_X;
    let cell_x = brick_x * 2u32 + (slot & 1u32);
    let cell_z = brick_z * 2u32 + ((slot >> 1u32) & 1u32);
    let cell_y = brick_y * 2u32 + ((slot >> 2u32) & 1u32);
    let cell = (cell_y << 12u32) | (cell_z << 6u32) | cell_x;

    let base_x = cell_x * 4u32;
    let base_y = cell_y * 4u32;
    let base_z = cell_z * 4u32;

    let mut acc = seed_i32(cell as i32, 0i32);
    for local_y in 0..4u32 {
        for local_z in 0..4u32 {
            for local_x in 0..4u32 {
                let x = base_x + local_x;
                let y = base_y + local_y;
                let z = base_z + local_z;
                let index = (y << 16u32) | (z << 8u32) | x;
                let word = voxel_a[index as usize];
                let ty = word & C_TY_MASK;
                if ty != 0u32 {
                    let life = (word >> 24u32) & 0xFFu32;
                    let temp_k = ((word >> C_TEMP_SHIFT) & C_TEMP_MASK) as f32 * 0.125f32;
                    acc += voxel_mass(ty, life, aux_tmp[index as usize], temp_k);
                }
            }
        }
    }
    mass[cell as usize].fetch_add(acc);
}

/// One Jacobi sweep of `laplacian(u) = 4 * pi * G * rho` with `u = 0` on the
/// boundary of the coarse grid (Dirichlet, like the zero-padded C++ FFT).
#[cube(launch_unchecked)]
fn gravity_jacobi_kernel(u_in: &[f32], u_out: &mut [f32], mass: &mut [Atomic<i32>]) {
    let i = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if i < GRAV_CELLS {
        let x = i & (GRAV_DIM - 1u32);
        let z = (i >> 6u32) & (GRAV_DIM - 1u32);
        let y = (i >> 12u32) & (GRAV_DIM - 1u32);
        let boundary = x == 0u32
            || x == GRAV_DIM - 1u32
            || y == 0u32
            || y == GRAV_DIM - 1u32
            || z == 0u32
            || z == GRAV_DIM - 1u32;
        let mut out = seed_f32(u_in[i as usize], 0.0f32);
        if !boundary {
            let sum = u_in[(i - 1u32) as usize]
                + u_in[(i + 1u32) as usize]
                + u_in[(i - GRAV_STRIDE_Z) as usize]
                + u_in[(i + GRAV_STRIDE_Z) as usize]
                + u_in[(i - GRAV_STRIDE_Y) as usize]
                + u_in[(i + GRAV_STRIDE_Y) as usize];
            let rho = mass[i as usize].load() as f32 * (1.0f32 / MASS_SCALE);
            out = (sum - FOUR_PI_G * rho) * (1.0f32 / 6.0f32);
        }
        u_out[i as usize] = out;
    }
}

/// Central-difference force `-grad(u)` of the solved potential.
#[cube(launch_unchecked)]
fn gravity_force_kernel(u: &[f32], force_x: &mut [f32], force_y: &mut [f32], force_z: &mut [f32]) {
    let i = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if i < GRAV_CELLS {
        let x = i & (GRAV_DIM - 1u32);
        let z = (i >> 6u32) & (GRAV_DIM - 1u32);
        let y = (i >> 12u32) & (GRAV_DIM - 1u32);
        let boundary = x == 0u32
            || x == GRAV_DIM - 1u32
            || y == 0u32
            || y == GRAV_DIM - 1u32
            || z == 0u32
            || z == GRAV_DIM - 1u32;
        let mut fx = seed_f32(u[i as usize], 0.0f32);
        let mut fy = seed_f32(u[i as usize], 0.0f32);
        let mut fz = seed_f32(u[i as usize], 0.0f32);
        if !boundary {
            fx = (u[(i - 1u32) as usize] - u[(i + 1u32) as usize]) * 0.5f32;
            fz = (u[(i - GRAV_STRIDE_Z) as usize] - u[(i + GRAV_STRIDE_Z) as usize]) * 0.5f32;
            fy = (u[(i - GRAV_STRIDE_Y) as usize] - u[(i + GRAV_STRIDE_Y) as usize]) * 0.5f32;
        }
        force_x[i as usize] = fx;
        force_y[i as usize] = fy;
        force_z[i as usize] = fz;
    }
}

/// Relaxes the potential from the mass accumulated on the previous tick and
/// publishes the force field consumed by the movement passes.
pub(crate) fn solve(engine: &mut Voxel3dEngine) {
    let client = engine.client.clone();
    for _ in 0..GRAVITY_ITERATIONS {
        unsafe {
            gravity_jacobi_kernel::launch_unchecked(
                &client,
                CubeCount::Static(GRAV_GROUPS, 1, 1),
                CubeDim::new_1d(GRAV_THREADS),
                BufferArg::from_raw_parts(engine.bufs.grav_potential.clone(), GRAV_CELLS as usize),
                BufferArg::from_raw_parts(
                    engine.bufs.grav_potential_tmp.clone(),
                    GRAV_CELLS as usize,
                ),
                BufferArg::from_raw_parts(engine.bufs.grav_mass.clone(), GRAV_CELLS as usize),
            );
        }
        core::mem::swap(
            &mut engine.bufs.grav_potential,
            &mut engine.bufs.grav_potential_tmp,
        );
    }
    unsafe {
        gravity_force_kernel::launch_unchecked(
            &client,
            CubeCount::Static(GRAV_GROUPS, 1, 1),
            CubeDim::new_1d(GRAV_THREADS),
            BufferArg::from_raw_parts(engine.bufs.grav_potential.clone(), GRAV_CELLS as usize),
            BufferArg::from_raw_parts(engine.bufs.grav_force_x.clone(), GRAV_CELLS as usize),
            BufferArg::from_raw_parts(engine.bufs.grav_force_y.clone(), GRAV_CELLS as usize),
            BufferArg::from_raw_parts(engine.bufs.grav_force_z.clone(), GRAV_CELLS as usize),
        );
    }
}

/// Clears the accumulator after the solve, matching `Simulation::BeforeSim`
/// (`Simulation.cpp:3739-3743`).
pub(crate) fn clear_mass(engine: &Voxel3dEngine) {
    unsafe {
        gravity_clear_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(GRAV_GROUPS, 1, 1),
            CubeDim::new_1d(GRAV_THREADS),
            BufferArg::from_raw_parts(engine.bufs.grav_mass.clone(), GRAV_CELLS as usize),
        );
    }
}

/// Accumulates this tick's mass for the next solve.
pub(crate) fn gather(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        gravity_gather_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Dynamic(bufs.sim_count.clone().binding()),
            CubeDim::new_1d(8),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), super::step::C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(bufs.grav_mass.clone(), GRAV_CELLS as usize),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_constants_match_the_cpp_source() {
        // `SimulationConfig.h:8` `M_GRAV = 6.67300e-1`.
        assert_eq!(M_GRAV, 0.667_3);
        // `gravity/Fft.cpp:22`: the convolution kernel is `-G * d / |d|^3`,
        // whose continuum Poisson source is `4 * pi * G`.
        let expected = 4.0 * core::f32::consts::PI * M_GRAV;
        assert!((FOUR_PI_G - expected).abs() < 1e-5);
    }

    #[test]
    fn gravity_grid_matches_the_air_grid() {
        assert_eq!(GRAV_CELLS as usize, AIR_COUNT);
        assert_eq!(GRAV_GROUPS * GRAV_THREADS, GRAV_CELLS);
    }
}
