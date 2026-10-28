//! GPU integration test for the 3D rigid-body solver.
//!
//! One shared CubeCL client on a real `wgpu` device (same bring-up as
//! `gpu_voxel3d_tests.rs`) owns a 256^3 [`Voxel3dBuffers`] world. A test kernel
//! fills a 4-voxel-thick STONE floor and plants four SAND voxels in the landing
//! volume; a second kernel spawns a 4^3 STONE box above the floor. The test
//! then runs 120 ticks and checks:
//!
//! * the box settles with a near-zero vertical velocity,
//! * its render description reports a bottom face at or above the floor,
//! * no loose voxel is left inside the final body volume,
//! * the four planted SAND cells were displaced (moved or deleted).
//!
//! The assertions use explicit bounds; the exact measured values are printed
//! so a run can be compared against the recorded GPU numbers.

use std::sync::LazyLock;

use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::wgpu::wgpu;
use cubecl::wgpu::{
    AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuRuntime, WgpuSetup, init_device,
};
use cubecl_runtime::runtime::Runtime;

use tpt_core::constants::{PT_SAND, PT_STNE};
use tpt_core::voxel3d::{BODY_VOXELS, VOXEL_COUNT};
use tpt_sim::voxel3d::Voxel3dBuffers;
use tpt_sim::voxel3d::rigid::{DESC_STRIDE, Rigid3dState, STATE_STRIDE};

/// Floor extents: x and z in `48..80`, y in `0..4` (four voxels thick).
const FLOOR_X0: u32 = 48;
const FLOOR_X1: u32 = 80;
const FLOOR_Y0: u32 = 0;
const FLOOR_Y1: u32 = 4;
const FLOOR_Z0: u32 = 48;
const FLOOR_Z1: u32 = 80;

/// Box spawn: a 4^3 STONE box whose bottom starts four voxels above the floor.
const BOX_CENTER: [f32; 3] = [64.0, 10.0, 64.0];
const BOX_HALF_EXTENTS: [u32; 3] = [2, 2, 2];
const TICKS: usize = 120;

/// SAND voxels planted directly in the landing footprint.
const SAND_CELLS: [[u32; 3]; 4] = [[63, 4, 63], [64, 4, 63], [63, 4, 64], [64, 4, 64]];

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
    .expect("no GPU adapter available for the 3D rigid-body test");

    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("tpt-sim-voxel3d-rigid-test-device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .expect("failed to create the 3D rigid-body test wgpu device/queue");

    let setup = WgpuSetup {
        instance: instance.clone(),
        adapter: adapter.clone(),
        device,
        queue,
        backend: info.backend,
    };
    // The world buffers are shared with the indirect-dispatch passes, so the
    // test device uses the production `ExclusivePages` configuration.
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

/// Fills an axis-aligned box of world cells with one element.
#[cube(launch_unchecked)]
fn fill_box_kernel(
    voxels: &mut [u32],
    x0: u32,
    y0: u32,
    z0: u32,
    x1: u32,
    y1: u32,
    z1: u32,
    ty: u32,
) {
    let i = ABSOLUTE_POS as u32;
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dz = z1 - z0;
    let total = dx * dy * dz;
    if i < total {
        let x = x0 + i % dx;
        let rest = i / dx;
        let z = z0 + rest % dz;
        let y = y0 + rest / dz;
        let index = (y << 16u32) | (z << 8u32) | x;
        voxels[index as usize] = ty;
    }
}

/// Writes one world cell, used to seed the SAND voxels.
#[cube(launch_unchecked)]
fn set_voxel_kernel(voxels: &mut [u32], index: u32, ty: u32) {
    voxels[index as usize] = ty;
}

/// Rotates `v` by the unit quaternion `q = (x, y, z, w)` on the host.
fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let (qx, qy, qz, qw) = (q[0], q[1], q[2], q[3]);
    let tx = 2.0 * (qy * v[2] - qz * v[1]);
    let ty = 2.0 * (qz * v[0] - qx * v[2]);
    let tz = 2.0 * (qx * v[1] - qy * v[0]);
    [
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    ]
}

/// Element ids treated as loose by the displacement pass.
fn is_loose(ty: u32) -> bool {
    matches!(ty, 17 | 1 | 2 | 6 | 3 | 21 | 16 | 4 | 7)
}

fn voxel_index(x: u32, y: u32, z: u32) -> usize {
    ((y << 16) | (z << 8) | x) as usize
}

#[test]
fn voxel3d_rigid_box_settles_and_displaces_sand() {
    let gpu = &*GPU;
    println!(
        "3D rigid test adapter: {} (backend {})",
        gpu.adapter, gpu.backend
    );

    let bufs = Voxel3dBuffers::new(gpu.client.clone());

    // A 4-voxel-thick STONE floor across a 32 x 32 area.
    let floor_cells = (FLOOR_X1 - FLOOR_X0) * (FLOOR_Y1 - FLOOR_Y0) * (FLOOR_Z1 - FLOOR_Z0);
    unsafe {
        fill_box_kernel::launch_unchecked(
            &gpu.client,
            CubeCount::Static(floor_cells.div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), VOXEL_COUNT),
            FLOOR_X0,
            FLOOR_Y0,
            FLOOR_Z0,
            FLOOR_X1,
            FLOOR_Y1,
            FLOOR_Z1,
            PT_STNE,
        );
    }
    for [x, y, z] in SAND_CELLS {
        unsafe {
            set_voxel_kernel::launch_unchecked(
                &gpu.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(bufs.voxel_a.clone(), VOXEL_COUNT),
                voxel_index(x, y, z) as u32,
                PT_SAND,
            );
        }
    }

    let mut rigid = Rigid3dState::new(&gpu.client);
    let slot = rigid.spawn_box(&gpu.client, BOX_CENTER, BOX_HALF_EXTENTS, PT_STNE);
    assert_eq!(slot, 0);
    assert_eq!(rigid.body_count(), 1);

    for tick in 0..TICKS {
        rigid.step(&gpu.client, &bufs, tick as u64);
    }

    // Body state: 32 f32 per body.
    let raw = gpu.client.read_one_unchecked(rigid.body_state.clone());
    let state: &[f32] = bytemuck::cast_slice(&raw);
    let body = &state[..STATE_STRIDE];
    let pos = [body[0], body[1], body[2]];
    let quat = [body[4], body[5], body[6], body[7]];
    let lin_vel = [body[8], body[9], body[10]];
    let ang_vel = [body[12], body[13], body[14]];
    let awake = body[29];

    // Render description: 40 f32 per body, AABB at [32..40].
    let raw_desc = gpu.client.read_one_unchecked(rigid.body_desc_handle());
    let desc: &[f32] = bytemuck::cast_slice(&raw_desc);
    let desc = &desc[..DESC_STRIDE];
    let aabb_min = [desc[32], desc[33], desc[34]];
    let aabb_max = [desc[36], desc[37], desc[38]];

    // World snapshot for the loose-voxel checks.
    let raw_world = gpu.client.read_one_unchecked(bufs.voxel_a.clone());
    let world: &[u32] = bytemuck::cast_slice(&raw_world);

    // Every solid body voxel, transformed with the final pose.
    let mut loose_inside = Vec::new();
    let mut body_cells = 0usize;
    for z in 14..18u32 {
        for y in 14..18u32 {
            for x in 14..18u32 {
                let local = ((y << 10) | (z << 5) | x) as usize;
                if local >= BODY_VOXELS {
                    continue;
                }
                let offset = [x as f32 - 15.5, y as f32 - 15.5, z as f32 - 15.5];
                let rotated = rotate(quat, offset);
                let wx = (pos[0] + rotated[0]).floor() as i32;
                let wy = (pos[1] + rotated[1]).floor() as i32;
                let wz = (pos[2] + rotated[2]).floor() as i32;
                if wx < 0 || wy < 0 || wz < 0 || wx >= 256 || wy >= 256 || wz >= 256 {
                    continue;
                }
                body_cells += 1;
                let index = voxel_index(wx as u32, wy as u32, wz as u32);
                let ty = world[index] & 0xFF;
                if is_loose(ty) {
                    loose_inside.push((wx, wy, wz, ty));
                }
            }
        }
    }

    let mut planted = Vec::new();
    for [x, y, z] in SAND_CELLS {
        let ty = world[voxel_index(x, y, z)] & 0xFF;
        planted.push((x, y, z, ty));
    }
    let mut remaining_sand = 0usize;
    for z in 56..72 {
        for y in 0..12 {
            for x in 56..72 {
                if world[voxel_index(x, y, z)] & 0xFF == PT_SAND {
                    remaining_sand += 1;
                }
            }
        }
    }

    println!(
        "body_count={} awake={:.1} pos=({:.6}, {:.6}, {:.6}) quat=({:.6}, {:.6}, {:.6}, {:.6})",
        rigid.body_count(),
        awake,
        pos[0],
        pos[1],
        pos[2],
        quat[0],
        quat[1],
        quat[2],
        quat[3]
    );
    println!(
        "lin_vel=({:.6}, {:.6}, {:.6}) ang_vel=({:.6}, {:.6}, {:.6})",
        lin_vel[0], lin_vel[1], lin_vel[2], ang_vel[0], ang_vel[1], ang_vel[2]
    );
    println!(
        "world_aabb_min=({:.6}, {:.6}, {:.6}) world_aabb_max=({:.6}, {:.6}, {:.6})",
        aabb_min[0], aabb_min[1], aabb_min[2], aabb_max[0], aabb_max[1], aabb_max[2]
    );
    println!(
        "body_cells={} loose_inside={:?} planted_after={:?} remaining_sand={}",
        body_cells, loose_inside, planted, remaining_sand
    );

    assert_eq!(rigid.body_count(), 1);
    assert!(body_cells > 0, "no body cell could be sampled");
    // The box fell four voxels and came to rest on the floor.
    assert!(
        lin_vel[1].abs() < 0.05,
        "vertical velocity {} is not near zero",
        lin_vel[1]
    );
    assert!(
        lin_vel[0].abs() < 0.05 && lin_vel[2].abs() < 0.05,
        "horizontal drift: {:?}",
        lin_vel
    );
    assert!(
        ang_vel[0].abs() < 0.05 && ang_vel[1].abs() < 0.05 && ang_vel[2].abs() < 0.05,
        "angular drift: {:?}",
        ang_vel
    );
    assert!(
        (5.99..=6.05).contains(&pos[1]),
        "body centre y {} did not settle on the floor",
        pos[1]
    );
    assert!(
        (pos[0] - 64.0).abs() < 0.01 && (pos[2] - 64.0).abs() < 0.01,
        "lateral drift: pos={:?}",
        pos
    );
    assert_eq!(awake, 0.0, "body did not fall asleep after settling");
    // The floor top is y = 4. The bottom face must not sink into it.
    assert!(
        aabb_min[1] >= 3.999,
        "body bottom {} sank below the floor top 4.0",
        aabb_min[1]
    );
    assert!(
        aabb_min[1] <= 4.05,
        "body bottom {} floats above the floor",
        aabb_min[1]
    );
    assert!(
        (aabb_max[1] - aabb_min[1] - 4.0).abs() < 0.01,
        "body height changed: {:?}..{:?}",
        aabb_min,
        aabb_max
    );
    assert!(
        loose_inside.is_empty(),
        "loose voxels remained inside the body: {loose_inside:?}"
    );
    for (x, y, z, ty) in planted {
        assert_ne!(
            ty, PT_SAND,
            "planted SAND at ({x}, {y}, {z}) was not displaced"
        );
    }
}
