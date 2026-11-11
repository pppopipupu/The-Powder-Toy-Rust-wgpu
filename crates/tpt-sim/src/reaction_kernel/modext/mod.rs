//! Reactions for the second element set (ids `195..=238`).
//!
//! The 44 elements added after `SEED` (`PT_FNTC`..`PT_MGNT`, see
//! `crates/tpt-core/src/constants.rs`) each bring their `Update` /
//! `Create` / `Graphics` functions from
//! `src/simulation/elements/*.cpp`. They are grouped by the subsystem their
//! update touches so each launcher can bind the buffer set it needs instead of
//! one 44-element pass running out of the sixteen-binding storage ceiling:
//!
//! | module | elements |
//! |--------|----------|
//! | [`thermo`] | `CLNT`, `COND` (`THMO`), `AMBE`, `QGPP`, `FUEL` |
//! | [`conductors`] | `WALL`, `CWIR`, `COPR`, `FNTC`, `FPTC`, `ELEX`, `DMRN` |
//! | [`powered`] | `LED`, `TIMC`, `PINV`, `PPTI`, `PPTO`, `PCON` |
//! | [`sensors`] | `CSNS`, `TMPS`, `GSNS`, `ECLR` |
//! | [`chem_gas`] | `CHLR`, `ACTY`, `NTRG`, `RADN`, `DFOM`, `CLUD` |
//! | [`chem_solid`] | `PHOS`, `CMNT`, `RUBR`, `GRPH`, `ALUM`, `ALMP`, `CLRC`, `NAPM` |
//! | [`weather`] | `UVRD`, `SUN` |
//! | [`force`] | `MGNT`, `MISL`, `PROJ`, `PRMT`, `CEXP`, `BFLM` |
//!
//! The split is a partitioning of the ids, not a dependency order: the group
//! launchers are called in declaration order, which keeps the sequence stable.
//!
//! The reaction driver calls [`launch`] right after the `special` pass, so the
//! creations the group kernels queue drain in the same tick.

mod chem_gas;
mod chem_solid;
mod conductors;
mod force;
mod powered;
mod sensors;
mod thermo;
mod weather;

use cubecl::prelude::*;

use super::PassDims;
use super::shared::launch_grid;
use crate::buffers::SimBuffers;

/// Clear last tick's element-driven air / heat block stamps and one-shot
/// gravity mass overrides.
///
/// `WALL` / `RUBR` / `ALUM` write `bmap_blockair*` through
/// [`SimBuffers::gpu_elem_block_air`] / [`SimBuffers::gpu_elem_block_airh`]
/// (`WALL.cpp`), and the air mask rebuild folds one tick's worth of stamps into
/// the wall mask. Zeroing both grids first makes the stamps exactly one frame
/// long even though nothing uploads a host shadow over them.
///
/// `CEXP` / `GSNS` record their `gravIn.mass[cell]` assignment in
/// [`SimBuffers::gpu_grav_mass_override`] plus the write-flag grid
/// [`SimBuffers::gpu_grav_mass_override_set`] during this tick's passes; the
/// value grid is cleared with the flag so a forced `0` never survives into the
/// next tick. The previous tick's writes were already consumed by the gravity
/// pass that ran before this one, mirroring the `gravIn.mass` clear after each
/// FFT dispatch (`Simulation.cpp:3739-3743`).
#[cube(launch_unchecked)]
fn clear_elem_block_kernel(
    elem_block_air: &mut [u32],
    elem_block_airh: &mut [u32],
    override_mass: &mut [i32],
    override_set: &mut [u32],
    cells: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx < cells {
        elem_block_air[idx as usize] = 0u32;
        elem_block_airh[idx as usize] = 0u32;
        override_mass[idx as usize] = 0i32;
        override_set[idx as usize] = 0u32;
    }
}

/// Launch every second-element-set pass for one frame.
///
/// Runs after [`super::special::launch`] and before
/// [`super::force::launch`]: the creation queue is still armed, so records the
/// group passes queue are allocated by the same tick's
/// [`super::shared::drain_gpu_creates`].
pub(crate) fn launch(bufs: &SimBuffers, dims: &PassDims, seed: u32) {
    // The groups below stamp the element-driven block masks and queue the
    // one-shot gravity mass overrides during this tick; start from zero so a
    // stamp or an injection lives exactly one frame.
    let (cubes, dim) = launch_grid(dims.cells);
    unsafe {
        clear_elem_block_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_elem_block_air.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_elem_block_airh.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass_override.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass_override_set.clone(), dims.cells),
            dims.cells as u32,
        );
    }

    thermo::launch(bufs, dims, seed);
    conductors::launch(bufs, dims, seed);
    powered::launch(bufs, dims, seed);
    sensors::launch(bufs, dims, seed);
    chem_gas::launch(bufs, dims, seed);
    chem_solid::launch(bufs, dims, seed);
    weather::launch(bufs, dims, seed);
    force::launch(bufs, dims, seed);
}
