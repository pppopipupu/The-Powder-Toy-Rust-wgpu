//! 64³ Eulerian air/pressure field (plan §5).
//!
//! [`AirState`] owns the device-resident fields of the coarse air grid and the
//! two drivers the tick pipeline calls:
//!
//! * [`AirState::gather`] reduces the voxels of every simulated 8³ brick into
//!   its eight aligned 4³ air cells. Cells of inactive bricks are left alone,
//!   which keeps the `solid` fraction of a static wall blocking the air after
//!   the brick has fallen asleep.
//! * [`AirState::solve`] advances the field: the velocity is snapshotted, the
//!   pressure is solved with twelve Jacobi iterations, and one advection kernel
//!   applies damping, temperature buoyancy, the pressure gradient and upwind
//!   neighbour averaging.
//!
//! Both drivers only enqueue compute work on [`Voxel3dBuffers::client`]; no
//! value is read back to the host. The only read-back is the test, which
//! materialises handles explicitly.
//!
//! The field layout matches the kernels in [`super::air_kernel`]: a `u32`
//! sized allocation per field, used as `f32`, indexed `(y << 12) | (z << 6) | x`.
//! All fields start in their neutral state (zero velocity/pressure/solid,
//! ambient temperature) because [`cubecl::client::Client::empty`] returns
//! recycled device memory.

use core::mem::size_of;

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::Voxel3dBuffers;
use super::air_kernel::{
    AIR_CELL_COUNT, AIR_JACOBI_ITERATIONS, AIR_THREADS, AIR_WORKGROUPS, air_advect_kernel,
    air_clear_kernel, air_gather_kernel, air_pressure_jacobi_kernel, air_pressure_volume_kernel,
    air_velocity_snapshot_kernel,
};
use tpt_core::voxel3d::{AIR_COUNT, BRICK_COUNT, VOXEL_COUNT};

/// Words per air field.
const FIELD_WORDS: usize = AIR_COUNT;

/// Words in the component-major velocity snapshot (`vx | vy | vz`).
const SNAPSHOT_WORDS: usize = 3 * AIR_COUNT;

const _: () = assert!(FIELD_WORDS * size_of::<u32>() == AIR_COUNT * 4);
const _: () = assert!(AIR_CELL_COUNT == AIR_COUNT);
const _: () = assert!(AIR_WORKGROUPS * AIR_THREADS == AIR_COUNT);

/// Device-resident state of the coarse air/pressure field.
///
/// The seven public handles are the fields shared with the rest of the voxel
/// pipeline: velocities, the pressure pair used by the Jacobi solver, the air
/// temperature and the solid fraction. `iterations` tunes the pressure solve;
/// it defaults to [`AIR_JACOBI_ITERATIONS`] and may be lowered for preview
/// quality settings.
pub struct AirState {
    /// Air velocity along x, in cells per tick.
    pub vx: Handle,
    /// Air velocity along y, in cells per tick.
    pub vy: Handle,
    /// Air velocity along z, in cells per tick.
    pub vz: Handle,
    /// Authoritative pressure field.
    pub p: Handle,
    /// Scratch partner of `p`; the Jacobi solver swaps the two every iteration.
    pub p_tmp: Handle,
    /// Air temperature in Kelvin.
    pub t: Handle,
    /// Fraction of non-empty voxels in each 4³ cell, `0..=1`.
    pub solid: Handle,
    /// TPT `pv` pressure in 1/1024 units, clamped to `±256`
    /// (`Air.cpp:315-327`).
    pub pv: Handle,
    /// Number of Jacobi pressure iterations per [`AirState::solve`].
    pub iterations: u32,
    /// Component-major copy of `vx | vy | vz` taken before the advection
    /// kernel, so the neighbour and upwind reads never race with the writes.
    snapshot: Handle,
}

impl AirState {
    /// Allocates the air field on `client` and clears it on the device.
    ///
    /// The allocation is about 10 MiB (seven fields of 1 MiB plus the 3 MiB
    /// velocity snapshot). Clearing uses a compute kernel instead of a host
    /// upload: velocities, pressure and the solid fraction become zero and the
    /// temperature becomes ambient, which is the neutral state every later
    /// pass expects for bricks that have not been gathered yet.
    pub fn new(client: Client) -> Self {
        let field_bytes = FIELD_WORDS * size_of::<u32>();
        let state = Self {
            vx: client.empty(field_bytes),
            vy: client.empty(field_bytes),
            vz: client.empty(field_bytes),
            p: client.empty(field_bytes),
            p_tmp: client.empty(field_bytes),
            t: client.empty(field_bytes),
            solid: client.empty(field_bytes),
            pv: client.empty(field_bytes),
            iterations: AIR_JACOBI_ITERATIONS,
            snapshot: client.empty(SNAPSHOT_WORDS * size_of::<u32>()),
        };
        state.clear(&client);
        state
    }

    /// Enqueues the neutral-state fill for every field.
    fn clear(&self, client: &Client) {
        unsafe {
            air_clear_kernel::launch_unchecked(
                client,
                CubeCount::Static(AIR_WORKGROUPS as u32, 1, 1),
                CubeDim::new_1d(AIR_THREADS as u32),
                BufferArg::from_raw_parts(self.vx.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.vy.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.vz.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.p.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.p_tmp.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.t.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.solid.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.pv.clone(), FIELD_WORDS),
            );
        }
    }

    /// Reduces the simulated bricks into the air field.
    ///
    /// `sim_count` holds the indirect workgroup count `[x, y, z]` produced by
    /// the brick compaction pass and `sim_list` the matching brick indices.
    /// One workgroup of eight threads handles one brick: each thread scans the
    /// 64 voxels of one 4³ cell, writes the non-empty fraction to `solid` and
    /// the mean non-empty temperature to `t` (ambient when the cell is empty).
    ///
    /// Bricks outside `sim_list` are not touched, so a sleeping stone wall
    /// keeps blocking the air.
    pub fn gather(&self, bufs: &Voxel3dBuffers, sim_list: &Handle, sim_count: &Handle) {
        unsafe {
            air_gather_kernel::launch_unchecked(
                &bufs.client,
                CubeCount::Dynamic(sim_count.clone().binding()),
                CubeDim::new_1d(8),
                BufferArg::from_raw_parts(bufs.voxel_a.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(sim_list.clone(), BRICK_COUNT),
                BufferArg::from_raw_parts(self.solid.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.t.clone(), FIELD_WORDS),
            );
        }
    }

    /// Advances the air field by one tick.
    ///
    /// The sequence is:
    ///
    /// 1. snapshot `vx | vy | vz` into the private scratch buffer,
    /// 2. run `iterations` Jacobi pressure steps, alternating `p` and `p_tmp`
    ///    so the last result always ends up in the handle the loop just wrote,
    /// 3. run one advection kernel over the fixed 64³ grid.
    ///
    /// The solve is world-wide and does not depend on the active brick list;
    /// `bufs` only supplies the client that owns the device. `sim_list` and
    /// `sim_count` are accepted to keep the pass signature uniform with
    /// [`AirState::gather`] and are deliberately not consumed.
    pub fn solve(&self, bufs: &Voxel3dBuffers, _sim_list: &Handle, _sim_count: &Handle) {
        let client = &bufs.client;
        let cubes = CubeCount::Static(AIR_WORKGROUPS as u32, 1, 1);
        let threads = CubeDim::new_1d(AIR_THREADS as u32);

        // 0. TPT pressure from the previous tick's velocity field
        //    (`Air.cpp:315-327`). Element impulses are added atomically by the
        //    reaction/blast passes and survive into this update.
        unsafe {
            air_pressure_volume_kernel::launch_unchecked(
                client,
                cubes.clone(),
                threads,
                BufferArg::from_raw_parts(self.vx.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.vy.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.vz.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.pv.clone(), FIELD_WORDS),
            );
        }

        // 1. Freeze the velocities the advection step reads.
        unsafe {
            air_velocity_snapshot_kernel::launch_unchecked(
                client,
                cubes.clone(),
                threads,
                BufferArg::from_raw_parts(self.vx.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.vy.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.vz.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.snapshot.clone(), SNAPSHOT_WORDS),
            );
        }

        // 2. Jacobi pressure solve. Swapping after every launch leaves the
        //    newest field in `pressure`, regardless of the iteration parity.
        let mut pressure = self.p.clone();
        let mut pressure_out = self.p_tmp.clone();
        for _ in 0..self.iterations {
            unsafe {
                air_pressure_jacobi_kernel::launch_unchecked(
                    client,
                    cubes.clone(),
                    threads,
                    BufferArg::from_raw_parts(pressure.clone(), FIELD_WORDS),
                    BufferArg::from_raw_parts(pressure_out.clone(), FIELD_WORDS),
                    BufferArg::from_raw_parts(self.vx.clone(), FIELD_WORDS),
                    BufferArg::from_raw_parts(self.vy.clone(), FIELD_WORDS),
                    BufferArg::from_raw_parts(self.vz.clone(), FIELD_WORDS),
                    BufferArg::from_raw_parts(self.solid.clone(), FIELD_WORDS),
                );
            }
            core::mem::swap(&mut pressure, &mut pressure_out);
        }

        // 3. Buoyancy, damping, pressure gradient and upwind averaging.
        unsafe {
            air_advect_kernel::launch_unchecked(
                client,
                cubes.clone(),
                threads,
                BufferArg::from_raw_parts(self.snapshot.clone(), SNAPSHOT_WORDS),
                BufferArg::from_raw_parts(pressure.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.t.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.solid.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.vx.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.vy.clone(), FIELD_WORDS),
                BufferArg::from_raw_parts(self.vz.clone(), FIELD_WORDS),
            );
        }
    }

    /// Copies the TPT `pv` field to the host (debug/save helper only).
    pub fn read_pv(&self, client: &Client) -> Option<Vec<i32>> {
        let readback = client.read(vec![self.pv.clone()]).ok()?;
        let raw = readback.first()?;
        if raw.len() < FIELD_WORDS * 4 {
            return None;
        }
        let (chunks, _) = raw.as_chunks::<4>();
        Some(
            chunks
                .iter()
                .map(|word| i32::from_le_bytes(*word))
                .collect(),
        )
    }

    /// Uploads a complete `pv` field (debug/save helper only).
    pub fn write_pv(&self, client: &Client, values: &[i32]) {
        debug_assert_eq!(values.len(), FIELD_WORDS);
        client.write(&self.pv, Bytes::from_elems(values.to_vec()));
    }
}
