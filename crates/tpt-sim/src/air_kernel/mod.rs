//! Eulerian air solver — CubeCL GPU kernels for the coarse `(xcells, ycells)` grid.
//!
//! One air cell covers `CELL` x `CELL` (`4 x 4`) pixels, so the grid is
//! `(xres / CELL, yres / CELL)` cells and every buffer here is indexed
//! `cy * xcells + cx`, exactly like The Powder Toy's `Air.cpp`.
//!
//! # Buffer roles
//!
//! | buffer | role |
//! |---|---|
//! | `air_pv` (`gpu_air_pv`) | pressure field — authoritative |
//! | `air_vx`, `air_vy` | velocity field — authoritative |
//! | `air_hv` | ambient heat field, in Kelvin — authoritative |
//! | `air_opv`, `air_ovx`, `air_ovy` | scratch: pressure / velocity produced by the pressure and velocity passes |
//! | `air_ohv` | scratch: ambient heat produced by the heat pass, mirrored back by the settle pass |
//!
//! Every pass only *reads* one set of buffers and *writes* another, and the
//! passes are launched back to back on the same stream. No thread ever reads a
//! value that another thread is writing in the same launch, so the solver is
//! deterministic instead of depending on GPU thread scheduling.
//!
//! # Pass order (one [`step_air_gpu`] call)
//!
//! `Air::update_air` (`Air.cpp:260-500`), skipped entirely for
//! `AirMode::NoUpdate` (TPT's `AIR_NOUPDATE`):
//!
//! * [`air_pressure_kernel`] — border damping, the three-cell wall velocity
//!   clear, and the pressure update from the velocity divergence. The border
//!   ring is not pinned; only the interior gets the `AIR_PLOSS` mix, matching
//!   `Air.cpp:288-334`.
//! * [`air_velocity_kernel`] — the velocity update from the **new** pressure
//!   gradient (Gauss-Seidel `pv -> vx`, `Air.cpp:336-345`).
//! * [`air_smooth_advect_kernel`] — the final loop (`Air.cpp:347-500`): 3x3
//!   Gaussian smoothing of velocity and pressure, far-advection step walk,
//!   vorticity confinement, fans, caps and the air-mode overrides.
//!
//! `Air::update_airh` (`Air.cpp:78-250`), run whenever `aheat_enable`:
//!
//! * [`air_heat_convect_kernel`] — 3x3 smoothing of `hv`/`vx`/`vy`,
//!   far-advection of `hv`, the blocked-cell freeze, the border pin and the
//!   `AIRC_*` convection term written into `vx`/`vy`.
//! * [`air_copy_back_kernel`] — `ohv -> hv`.
//!
//! Finally [`air_block_masks_kernel`] rebuilds `bmap_blockair` /
//! `bmap_blockairh` from the wall maps, exactly like the per-frame wall loop at
//! the end of `Simulation::BeforeSim` (`Simulation.cpp:3765-3775`). It runs
//! *after* the heat pass so the previous frame's `blockairh` counter increments
//! (written by the thermal pass) are visible to `update_airh`, and resets them
//! for the next frame.
//!
//! [`hot_air_pressure_kernel`] closes the step: TPT applies `HotAir` inside the
//! per-particle loop after the ambient-air passes and before `TransitionPhase`
//! and the element `Update` (`Simulation.cpp:2347-2358`), and the loop is
//! bounded by `parts.active` captured at entry (`Simulation.cpp:2300`), so
//! same-frame creations are never visited. The port therefore reads the map the
//! previous movement pass left behind and runs the pass once per frame, before
//! the reactions and the movement that consume the raised pressure.
//!
//! # Physics constants
//!
//! The numeric literals below mirror [`tpt_core::constants`]:
//! `AIR_PLOSS = 0.9999`, `AIR_TSTEPP * 0.5 = 0.15`, `AIR_TSTEPV * 0.5 = 0.2`,
//! `AIR_VLOSS = 0.999`, `AIR_VADV = 0.3`, and the `[-256, 256]` pressure /
//! velocity clamp. They are spelled out as `f32` literals inside the kernel
//! bodies so each pass reads as a self-contained piece of physics; the only
//! host constants a kernel references are [`BORDER`] and [`ADVECT_DIST`].

use cubecl::prelude::*;

use crate::buffers::SimBuffers;

/// Width of the damped band along the grid edge, in air cells.
const BORDER: usize = 2;

/// Advection distance per step: the semi-Lagrangian backtrace is
/// `(x - vx * 0.7, y - vy * 0.7)`.
const ADVECT_DIST: f32 = 0.7;

mod advect;
mod heat;
mod pressure;
mod settle;

pub use advect::*;
pub use heat::*;
pub use pressure::*;
pub use settle::*;

/// Clamp `v` into `[lo, hi]` (`f32::clamp` is unavailable inside a cube body).
#[cube]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut out = v;
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

/// Runs one Eulerian air step on the device.
///
/// The kernels are launched with a 64-wide 1D cube and
/// `(num_cells as u32).div_ceil(64)` cubes; `num_cells` is
/// [`tpt_core::particle::SimulationConfig::num_cells`], i.e. the coarse
/// `xcells * ycells` air grid, not the pixel grid.
///
/// The `update_air` passes are skipped for `AirMode::NoUpdate` (TPT's
/// `AIR_NOUPDATE`, `Air.cpp:260`); the ambient-heat pass still runs whenever
/// `aheat_enable` is set, exactly like `Simulation::BeforeSim`
/// (`Simulation.cpp:3734-3736`).
///
/// Nothing is read back here: the step only marks `gpu_air_dirty` and leaves
/// the air grids device-resident. [`crate::buffers::SimBuffers::sync_air_gpu_to_host`]
/// refreshes the host shadows on demand for brush edits, Lua queries, the HUD
/// and save/load.
pub fn step_air_gpu(bufs: &mut SimBuffers) {
    let xcells = bufs.config.xcells() as usize;
    let ycells = bufs.config.ycells() as usize;
    let num_cells = xcells * ycells;
    if xcells == 0 || ycells == 0 || num_cells == 0 {
        return;
    }

    // `AirMode` is a fieldless enum with the discriminants 0..=4.
    let air_mode = bufs.config.air_mode as u32;
    let edge_pressure = bufs.config.edge_pressure;
    let edge_velocity_x = bufs.config.edge_velocity_x;
    let edge_velocity_y = bufs.config.edge_velocity_y;
    let ambient_air_temp = bufs.config.ambient_air_temp;
    let aheat_enable = u32::from(bufs.config.aheat_enable);
    // `AIRC_*` (0 = none, 1 = legacy, 2 = Boussinesq).
    let convection_mode = bufs.config.convection_mode;
    // `GravityMode` is a fieldless enum with the discriminants 0..=3.
    let gravity_mode = bufs.config.gravity_mode as u32;
    let custom_gravity_x = bufs.config.custom_gravity_x;
    let custom_gravity_y = bufs.config.custom_gravity_y;
    let vorticity_coeff = bufs.config.vorticity_coeff;

    let client = bufs.client.clone();
    let cube_dim = CubeDim::new_1d(64);
    let cubes = (num_cells as u32).div_ceil(64);

    // `Air::update_air`: pressure, then velocity from the new pressure, then
    // the smoothing / advection / vorticity / fan loop. `AIR_NOUPDATE` skips
    // every one of these writes (`Air.cpp:260`).
    if air_mode != 4u32 {
        unsafe {
            air_pressure_kernel::launch_unchecked(
                &client,
                CubeCount::new_1d(cubes),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_block_air.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_opv.clone(), num_cells),
                xcells,
                ycells,
                edge_pressure,
                edge_velocity_x,
                edge_velocity_y,
            );
        }

        unsafe {
            air_velocity_kernel::launch_unchecked(
                &client,
                CubeCount::new_1d(cubes),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_air_opv.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_block_air.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_ovx.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_ovy.clone(), num_cells),
                xcells,
                ycells,
                edge_velocity_x,
                edge_velocity_y,
            );
        }

        unsafe {
            air_smooth_advect_kernel::launch_unchecked(
                &client,
                CubeCount::new_1d(cubes),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_air_ovx.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_ovy.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_opv.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_block_air.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_fan_vx.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_fan_vy.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), num_cells),
                xcells,
                ycells,
                vorticity_coeff,
                air_mode,
            );
        }
    }

    // `Air::update_airh`: `Simulation::BeforeSim` calls it whenever ambient
    // heat is enabled, even with `AIR_OFF` / `AIR_NOUPDATE`
    // (`Simulation.cpp:3734-3736`, `Air.cpp:78-250`).
    if aheat_enable != 0u32 {
        unsafe {
            air_heat_convect_kernel::launch_unchecked(
                &client,
                CubeCount::new_1d(cubes),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_air_hv.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_block_airh.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_ohv.clone(), num_cells),
                xcells,
                ycells,
                ambient_air_temp,
                aheat_enable,
                convection_mode,
                gravity_mode,
                custom_gravity_x,
                custom_gravity_y,
            );
        }

        unsafe {
            air_copy_back_kernel::launch_unchecked(
                &client,
                CubeCount::new_1d(cubes),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_air_ohv.clone(), num_cells),
                BufferArg::from_raw_parts(bufs.gpu_air_hv.clone(), num_cells),
                num_cells as u32,
            );
        }
    }

    // Per-frame wall mask refresh (`Simulation::BeforeSim`, `Simulation.cpp:3765-3775`).
    // This resets the `blockairh` counter the thermal pass increments, so it
    // has to run after the heat pass read the previous frame's counters.
    unsafe {
        air_block_masks_kernel::launch_unchecked(
            &client,
            CubeCount::new_1d(cubes),
            cube_dim,
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), num_cells),
            BufferArg::from_raw_parts(bufs.gpu_block_air.clone(), num_cells),
            BufferArg::from_raw_parts(bufs.gpu_block_airh.clone(), num_cells),
            num_cells as u32,
        );
    }

    // Particle `HotAir` -> pressure. TPT applies it inside the per-particle
    // loop before `TransitionPhase` and the element `Update`
    // (`Simulation.cpp:2347-2358`), over the particles alive when the loop
    // starts (`parts.active`, `Simulation.cpp:2300`), so same-frame creations
    // must not be visited. Running it here also means the map is the one the
    // previous movement pass rebuilt, not a map the reaction drain just
    // touched.
    let xres = bufs.config.xres as usize;
    let yres = bufs.config.yres as usize;
    if xres != 0 && yres != 0 {
        unsafe {
            hot_air_pressure_kernel::launch_unchecked(
                &client,
                CubeCount::new_1d(cubes),
                cube_dim,
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), xres * yres),
                BufferArg::from_raw_parts(
                    bufs.gpu_elem_props.clone(),
                    tpt_core::constants::PT_NUM * 32,
                ),
                BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), num_cells),
                xres,
                yres,
                xcells,
                ycells,
            );
        }
    }

    // The air grids stay device-resident: nothing is read back here. Host
    // shadows are refreshed on demand (`sync_air_gpu_to_host`) by the Lua
    // bridge, the HUD inspector and save capture.
    bufs.gpu_air_dirty = true;
    bufs.gpu_dirty = true;
}
