//! GPU integration test for the 3D voxel core foundation (M0).
//!
//! The test drives the real CubeCL/wgpu stack: a raw `wgpu::Instance`,
//! adapter and device are created with `adapter.limits()` and handed to CubeCL
//! through [`WgpuSetup`], exactly like `crates/tpt-sim/tests/gpu_sim_tests.rs`.
//! One shared client serves the whole binary.
//!
//! It covers the two M0 acceptance points:
//!
//! 1. [`Voxel3dBuffers::new`] really allocates the 256³ world: the `voxel_a`
//!    wgpu resource must span `VOXEL_COUNT * 4` bytes.
//! 2. Indirect dispatch works on the backend: a three-workgroup probe driven by
//!    `sim_count = [3, 1, 1]` copies the three words of `sim_list` into `out`.
//!
//! A failure here is reported as-is; there is no static-dispatch fallback,
//! because every 3D simulation pass depends on the indirect path.

use std::sync::LazyLock;

use cubecl::client::Client;
use cubecl::wgpu::wgpu;
use cubecl::wgpu::{
    AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuRuntime, WgpuServer, WgpuSetup,
    init_device,
};
use cubecl_runtime::runtime::Runtime;

use tpt_core::voxel3d::VOXEL_COUNT;
use tpt_sim::voxel3d::{SIM_COUNT_WORDS, Voxel3dBuffers, run_indirect_probe};

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
    .expect("no GPU adapter available for the 3D voxel tests");

    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("tpt-sim-voxel3d-test-device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .expect("failed to create the 3D voxel test wgpu device/queue");

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
    // `sim_list`/`out` and the indirect launch would fail validation.
    // `ExclusivePages` keeps one wgpu buffer per handle and is the
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

/// The 256³ allocation and the indirect dispatch are the two things every
/// later 3D pass assumes, so they share one test and one device bring-up.
#[test]
fn voxel3d_world_allocation_and_indirect_dispatch() {
    let gpu = &*GPU;
    println!(
        "3D voxel test adapter: {} (backend {})",
        gpu.adapter, gpu.backend
    );

    let mut bufs = Voxel3dBuffers::new(gpu.client.clone());

    // 1. All three authoritative arrays must hold one `u32` per voxel on the
    // device. `Client::empty` is lazy, so materialise each handle to prove the
    // 256^3 allocation really succeeded.
    let expected = (VOXEL_COUNT * core::mem::size_of::<u32>()) as u64;
    for (name, handle) in [
        ("voxel_a", &bufs.voxel_a),
        ("voxel_b", &bufs.voxel_b),
        ("claim", &bufs.claim),
    ] {
        let managed = gpu
            .client
            .get_resource::<WgpuServer<AutoCompiler>>(handle.clone())
            .unwrap_or_else(|err| panic!("{name} must resolve to a live wgpu buffer: {err}"));
        let allocated = managed.resource().size;
        assert!(
            allocated >= expected,
            "{name} holds {allocated} bytes, expected at least {expected} (256^3 u32)"
        );
    }

    // 2. Point the indirect arguments at three workgroups and three words.
    let count = [3u32, 1, 1];
    assert_eq!(count.len(), SIM_COUNT_WORDS);
    bufs.sim_count = gpu.client.create_from_slice(bytemuck::cast_slice(&count));
    bufs.sim_list = gpu
        .client
        .create_from_slice(bytemuck::cast_slice(&[11u32, 22, 33]));
    let out = gpu.client.empty(3 * core::mem::size_of::<u32>());

    run_indirect_probe(&gpu.client, &bufs.sim_count, &bufs.sim_list, &out);

    let raw = gpu.client.read_one_unchecked(out);
    let words: &[u32] = bytemuck::cast_slice(&raw);
    assert_eq!(
        words,
        [11, 22, 33],
        "indirect dispatch copied {words:?} instead of the three list words"
    );
}
