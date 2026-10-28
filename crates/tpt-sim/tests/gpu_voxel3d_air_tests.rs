//! GPU integration test for the M3 64³ air field (plan §5).
//!
//! The test uses the same bring-up as `gpu_voxel3d_tests.rs`: one raw
//! `wgpu::Device`/`Queue` created with `adapter.limits()`, handed to CubeCL
//! through [`WgpuSetup`], and an `ExclusivePages` memory configuration so
//! `sim_count` can be bound as an indirect dispatch argument without aliasing
//! a storage buffer.
//!
//! The scenario is the smallest one that exercises both drivers:
//!
//! 1. one `FIRE` voxel at world `(0, 0, 0)`, written by a test kernel;
//! 2. `sim_list = [0]`, `sim_count = [1, 1, 1]`, so only brick 0 is gathered;
//! 3. after [`AirState::gather`] the containing 4³ air cell is hot and its
//!    solid fraction is `1/64`, while an untouched cell stays at ambient;
//! 4. after [`AirState::solve`] some cell above the hot cell has a positive
//!    `vy`, proving that buoyancy reaches the neighbouring cells.
//!
//! A dynamic-dispatch failure is reported as-is; there is no fallback to a
//! static launch.

use std::sync::LazyLock;

use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::wgpu;
use cubecl::wgpu::{
    AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuRuntime, WgpuSetup, init_device,
};
use cubecl_runtime::runtime::Runtime;

use tpt_core::constants::PT_FIRE;
use tpt_core::voxel3d::{AIR_COUNT, AIR_DIM, BRICK_COUNT, VOXEL_COUNT, Voxel};
use tpt_sim::voxel3d::Voxel3dBuffers;
use tpt_sim::voxel3d::air::AirState;

/// Air temperature used by the kernels when no voxel is present.
const AMBIENT_K: f32 = 293.15;

/// Temperature of the test `FIRE` voxel, comfortably above ambient.
const FIRE_K: f32 = 973.0;

/// One CubeCL client on the process-wide test device.
struct TestGpu {
    client: Client,
    backend: String,
    adapter: String,
}

static GPU: LazyLock<TestGpu> = LazyLock::new(|| {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("no GPU adapter available for the 3D air tests");

    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("tpt-sim-voxel3d-air-test-device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .expect("failed to create the 3D air test wgpu device/queue");

    let setup = WgpuSetup {
        instance: instance.clone(),
        adapter: adapter.clone(),
        device,
        queue,
        backend: info.backend,
    };
    // CubeCL's default `Adaptive` allocator sub-allocates small handles from
    // one wgpu buffer. wgpu forbids using one buffer as both `INDIRECT` and
    // `STORAGE_READ_WRITE` in the same dispatch, so `sim_count` would alias
    // another handle and the indirect gather launch would fail validation.
    // `ExclusivePages` keeps one wgpu buffer per handle, which is the
    // configuration the 3D indirect passes require.
    let options = RuntimeOptions {
        memory_config: MemoryConfiguration::ExclusivePages,
        ..RuntimeOptions::default()
    };
    let cubecl_device = init_device(setup, options);

    TestGpu {
        client: WgpuRuntime::<AutoCompiler>::client(&cubecl_device),
        backend: format!("{:?}", info.backend),
        adapter: info.name,
    }
});

/// Writes one voxel word at a linear world index.
#[cube(launch_unchecked)]
fn write_voxel_kernel(voxels: &mut [u32], index: u32, word: u32) {
    if ABSOLUTE_POS == 0usize {
        voxels[index as usize] = word;
    }
}

/// Reads a whole `f32` field back to the host.
///
/// `read_one_unchecked` blocks until every previously queued launch on this
/// client has completed, so the returned values include the gather/solve work.
fn read_f32(client: &Client, handle: Handle, words: usize) -> Vec<f32> {
    let bytes = client.read_one_unchecked(handle);
    let values: &[f32] = bytemuck::cast_slice(&bytes);
    assert_eq!(
        values.len(),
        words,
        "read back {} words instead of {words}",
        values.len()
    );
    values.to_vec()
}

/// Gathers one hot voxel and checks that the air above it rises.
#[test]
fn voxel3d_air_gather_and_buoyancy() {
    let gpu = &*GPU;
    println!(
        "3D air test adapter: {} (backend {})",
        gpu.adapter, gpu.backend
    );

    let client = gpu.client.clone();
    let bufs = Voxel3dBuffers::new(client.clone());

    // One FIRE voxel at (0, 0, 0), which is inside brick 0 and air cell 0.
    let fire = Voxel::new(PT_FIRE as u8, FIRE_K, 0).to_word();
    unsafe {
        write_voxel_kernel::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), VOXEL_COUNT),
            0u32,
            fire,
        );
    }

    // Only brick 0 participates in the gather.
    let mut list = vec![0u32; BRICK_COUNT];
    list[0] = 0;
    let sim_list = client.create_from_slice(bytemuck::cast_slice(&list));
    let sim_count = client.create_from_slice(bytemuck::cast_slice(&[1u32, 1u32, 1u32]));

    let air = AirState::new(client.clone());
    air.gather(&bufs, &sim_list, &sim_count);

    let temperature = read_f32(&client, air.t.clone(), AIR_COUNT);
    let solid = read_f32(&client, air.solid.clone(), AIR_COUNT);

    let hot_temperature = temperature[0];
    let hot_solid = solid[0];
    let expected_solid = 1.0f32 / 64.0f32;
    println!(
        "hot air cell: t = {hot_temperature:.3} K, solid = {hot_solid:.5} (expected {expected_solid:.5})"
    );
    assert!(
        hot_temperature > AMBIENT_K,
        "gather left the air cell containing the FIRE voxel at {hot_temperature} K"
    );
    assert!(
        (hot_solid - expected_solid).abs() < 1.0e-6f32,
        "gather wrote a solid fraction of {hot_solid} instead of {expected_solid}"
    );

    // A cell far away from brick 0 was never gathered and keeps the neutral
    // ambient temperature written by AirState::new.
    let untouched = temperature[AIR_COUNT - 1];
    assert!(
        (untouched - AMBIENT_K).abs() < 1.0e-3f32,
        "inactive air cell changed to {untouched} K during gather"
    );

    air.solve(&bufs, &sim_list, &sim_count);

    let vy = read_f32(&client, air.vy.clone(), AIR_COUNT);

    // Scan everything above the y = 0 layer. The hot cell itself is on the
    // boundary; buoyancy must reach at least the cell directly above it.
    let first_above = (AIR_DIM as usize) * (AIR_DIM as usize);
    let mut max_vy = f32::MIN;
    let mut max_index = first_above;
    for (index, value) in vy.iter().enumerate().skip(first_above) {
        if *value > max_vy {
            max_vy = *value;
            max_index = index;
        }
    }
    let max_x = max_index & 63usize;
    let max_z = (max_index >> 6usize) & 63usize;
    let max_y = max_index >> 12usize;
    let above_hot = vy[1usize << 12usize];
    println!(
        "max vy above the hot layer: {max_vy:.4} cells/tick at air cell ({max_x}, {max_y}, {max_z}); vy directly above hot = {above_hot:.4}"
    );
    assert!(
        max_vy > 0.0f32,
        "solve produced no upward air velocity above the hot cell (max vy = {max_vy})"
    );
}
