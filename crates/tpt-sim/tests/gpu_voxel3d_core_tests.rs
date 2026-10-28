//! GPU integration test for the 3D simulation core (plan §3–§6).
//!
//! One test binary, one serialized `#[test]`: the adapter bring-up and the
//! kernel compilation dominate the runtime, so every scenario reuses one
//! engine and calls [`Voxel3dEngine::reset`] between cases. The client uses
//! `adapter.limits()` (the 8x8x8 brick kernels need 512 threads per workgroup)
//! and `ExclusivePages`, because the indirect dispatch argument is bound only
//! as an indirect buffer and must not alias a storage allocation.
//!
//! Covered here:
//!
//! * paint/read/erase at the world centre and across a brick seam, including
//!   the leaf/coarse bits and the two-tick activity fall-off,
//! * a powder voxel stopped by a solid floor and a free powder voxel falling,
//! * a water blob spreading along the world floor,
//! * fire/water and lava/water quenching, plus heated water boiling to steam,
//! * a bomb blast replacing loose voxels and leaving fire,
//! * pick from above, and pick again after erasing the hit voxel,
//! * a rigid stone box landing on a stone floor with a finite pose.
//! * `SPRK`'s `create_part` replace path: no voxel on empty sky, no
//!   replacement of non-conductors, ctype restoration after the spark cycle,

use std::sync::LazyLock;

use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::wgpu;
use cubecl::wgpu::{
    AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuRuntime, WgpuSetup, init_device,
};
use cubecl_runtime::runtime::Runtime;

use tpt_core::constants::{
    PT_BANG, PT_BCOL, PT_BOMB, PT_BRMT, PT_COAL, PT_EMBR, PT_FIRE, PT_GLAS, PT_LAVA, PT_METL,
    PT_NONE, PT_SAND, PT_SPRK, PT_STNE, PT_WATR, PT_WOOD, PT_WTRV,
};
use tpt_core::particle::EdgeMode;
use tpt_core::voxel3d::{
    AIR_COUNT, Brush3d, BrushShape3d, VOXEL_COUNT, Voxel3dConfig, brick_index, coarse_bit_mask,
    coarse_word_index, leaf_bit_mask, leaf_word_index, spawn_defaults,
};
use tpt_sim::voxel3d::Voxel3dEngine;

/// Sentinel type id that matches no voxel (`ty` is always a `u8`).
const NO_SECOND_TYPE: u32 = u32::MAX;

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
    .expect("no GPU adapter available for the 3D core tests");

    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("tpt-sim-voxel3d-core-test-device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .expect("failed to create the 3D core test wgpu device/queue");

    let setup = WgpuSetup {
        instance: instance.clone(),
        adapter: adapter.clone(),
        device,
        queue,
        backend: info.backend,
    };
    // One wgpu buffer per handle: the indirect dispatch buffers must not alias
    // a storage allocation in the same dispatch (see the air test for details).
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

/// Copies one device word into a one-word output buffer.
#[cube(launch_unchecked)]
fn read_word_kernel(words: &[u32], index: u32, out: &mut [u32]) {
    if ABSOLUTE_POS == 0usize {
        out[0] = words[index as usize];
    }
}

/// Scans an inclusive box and reduces the positions of matching voxels.
///
/// `out = [count, min_x, max_x, min_y, max_y, min_z, max_z]`. Two type ids are
/// accepted so a case can count "steam or smoke" in one dispatch.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn scan_box_kernel(
    voxels: &[u32],
    min_x: u32,
    min_y: u32,
    min_z: u32,
    max_x: u32,
    max_y: u32,
    max_z: u32,
    ty_a: u32,
    ty_b: u32,
    out: &mut [Atomic<u32>],
) {
    let x = min_x + CUBE_POS_X * 8u32 + UNIT_POS_X;
    let y = min_y + CUBE_POS_Y * 8u32 + UNIT_POS_Y;
    let z = min_z + CUBE_POS_Z * 8u32 + UNIT_POS_Z;
    if x <= max_x && y <= max_y && z <= max_z {
        let index = (y << 16u32) | (z << 8u32) | x;
        let ty = voxels[index as usize] & 255u32;
        if ty == ty_a || ty == ty_b {
            out[0].fetch_add(1u32);
            out[1].fetch_min(x);
            out[2].fetch_max(x);
            out[3].fetch_min(y);
            out[4].fetch_max(y);
            out[5].fetch_min(z);
            out[6].fetch_max(z);
        }
    }
}

/// Result of [`scan_box`].
#[derive(Clone, Copy, Debug)]
struct Scan {
    count: u32,
    min: [u32; 3],
    max: [u32; 3],
}

impl Scan {
    fn extent(&self, axis: usize) -> u32 {
        if self.count == 0 {
            0
        } else {
            self.max[axis] - self.min[axis]
        }
    }
}

fn scan_box(
    client: &Client,
    voxels: &Handle,
    min: [u32; 3],
    max: [u32; 3],
    ty_a: u32,
    ty_b: u32,
) -> Scan {
    let init = [0u32, u32::MAX, 0, u32::MAX, 0, u32::MAX, 0];
    let out = client.create_from_slice(bytemuck::cast_slice(&init));
    let groups = |lo: u32, hi: u32| (hi - lo + 1).div_ceil(8);
    unsafe {
        scan_box_kernel::launch_unchecked(
            client,
            CubeCount::Static(
                groups(min[0], max[0]),
                groups(min[1], max[1]),
                groups(min[2], max[2]),
            ),
            CubeDim::new_3d(8, 8, 8),
            BufferArg::from_raw_parts(voxels.clone(), VOXEL_COUNT),
            min[0],
            min[1],
            min[2],
            max[0],
            max[1],
            max[2],
            ty_a,
            ty_b,
            BufferArg::from_raw_parts(out.clone(), 7),
        );
    }
    let words = read_words(client, out, 7);
    Scan {
        count: words[0],
        min: [words[1], words[3], words[5]],
        max: [words[2], words[4], words[6]],
    }
}

fn read_words(client: &Client, handle: Handle, words: usize) -> Vec<u32> {
    let bytes = client.read_one_unchecked(handle);
    let values: &[u32] = bytemuck::cast_slice(&bytes);
    assert_eq!(values.len(), words, "read back the wrong word count");
    values.to_vec()
}

fn read_word(client: &Client, source: &Handle, index: u32, length: usize) -> u32 {
    let out = client.create_from_slice(bytemuck::cast_slice(&[0u32]));
    unsafe {
        read_word_kernel::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(source.clone(), length),
            index,
            BufferArg::from_raw_parts(out.clone(), 1),
        );
    }
    read_words(client, out, 1)[0]
}

fn leaf_bit(client: &Client, engine: &Voxel3dEngine, x: u32, y: u32, z: u32) -> bool {
    let word = leaf_word_index(x, y, z);
    let bit = leaf_bit_mask(x, y, z);
    read_word(client, &engine.bufs.leaf_masks, word as u32, 65_536) & bit != 0
}

fn coarse_bit(
    client: &Client,
    engine: &Voxel3dEngine,
    level: usize,
    x: u32,
    y: u32,
    z: u32,
) -> bool {
    let word = coarse_word_index(level, x, y, z);
    let bit = coarse_bit_mask(level, x, y, z);
    read_word(client, &engine.bufs.coarse_masks, word as u32, 9_363) & bit != 0
}

fn brick_occupied(_client: &Client, engine: &Voxel3dEngine, x: u32, y: u32, z: u32) -> bool {
    engine.read_brick_occupied(brick_index(x, y, z))
}

fn engine_config() -> Voxel3dConfig {
    Voxel3dConfig {
        ambient_temp_k: 293.15,
        ..Voxel3dConfig::default()
    }
}

/// Spawn temperature a freshly painted `PT_STNE` voxel must carry.
///
/// TPT's `Element.cpp:54` sets `DefaultProperties.temp = R_TEMP + 273.15f`
/// (`R_TEMP = 22`), i.e. 295.15 K, and `STNE.cpp` does not override it.
/// Deriving the value from the ported spawn table keeps this assertion tied to
/// the declared contract instead of a magic number; it is intentionally *not*
/// the configurable air ambient.
fn painted_stne_temp_k() -> f32 {
    spawn_defaults(PT_STNE)
        .expect("STNE must have a ported spawn default")
        .temp_k
}

/// Paint/read/erase round trip and the two-tick activity fall-off.
fn case_paint_erase(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.reset();

    let painted = engine.place_sphere(PT_STNE, [128.0, 128.0, 128.0], 1.0);
    assert_eq!(
        painted, 7,
        "radius-1 sphere must cover the centre plus six neighbours"
    );
    let centre = engine.read_voxel(128, 128, 128).expect("centre in world");
    assert_eq!(centre.ty as u32, PT_STNE);
    assert!(
        (centre.temp_k() - painted_stne_temp_k()).abs() < 0.2,
        "painted temperature"
    );
    assert_eq!(
        engine.read_voxel(129, 128, 128).map(|v| v.ty as u32),
        Some(PT_STNE)
    );
    assert_eq!(
        engine.read_voxel(128, 130, 128).map(|v| v.ty as u32),
        Some(PT_NONE)
    );
    assert!(brick_occupied(client, engine, 128, 128, 128));
    assert!(!brick_occupied(client, engine, 100, 128, 128));
    assert!(
        leaf_bit(client, engine, 128, 128, 128),
        "leaf bit after paint"
    );
    assert!(
        coarse_bit(client, engine, 0, 128, 128, 128),
        "4^3 bit after paint"
    );
    assert!(
        coarse_bit(client, engine, 1, 128, 128, 128),
        "8^3 bit after paint"
    );

    // A brush across the x = 8 brick seam sets both bricks.
    engine.place_sphere(PT_STNE, [8.0, 8.0, 8.0], 1.0);
    assert_eq!(
        engine.read_voxel(7, 8, 8).map(|v| v.ty as u32),
        Some(PT_STNE)
    );
    assert_eq!(
        engine.read_voxel(8, 8, 8).map(|v| v.ty as u32),
        Some(PT_STNE)
    );
    assert!(brick_occupied(client, engine, 7, 8, 8));
    assert!(brick_occupied(client, engine, 8, 8, 8));

    let erased = engine.erase_sphere([128.0, 128.0, 128.0], 1.0);
    assert_eq!(erased, painted);
    assert_eq!(
        engine.read_voxel(128, 128, 128).map(|v| v.ty as u32),
        Some(PT_NONE)
    );
    engine.erase_sphere([8.0, 8.0, 8.0], 1.0);

    engine.step();
    engine.step();
    assert!(
        !brick_occupied(client, engine, 128, 128, 128),
        "emptied brick still occupied after two ticks"
    );
    assert!(
        !leaf_bit(client, engine, 128, 128, 128),
        "leaf bit after erase"
    );
    assert!(
        !coarse_bit(client, engine, 0, 128, 128, 128),
        "4^3 bit after erase"
    );
    assert!(
        !coarse_bit(client, engine, 1, 128, 128, 128),
        "8^3 bit after erase"
    );
    println!("case paint/erase: ok");
}

/// `create_part(PT_SPRK)` replaces a resting conductor in place instead of
/// creating a spark voxel (`Simulation.cpp:1851-1874`). Drawing on empty sky
/// or on a non-conductor leaves the world untouched, and the spark restores
/// its remembered conductor instead of decaying into a stale `METL`.
fn case_spark_replace(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.reset();
    let centre = [128.0f32, 128.0, 128.0];
    let index = (128u32 << 16) | (128u32 << 8) | 128u32;

    // `create_part` returns -1 when pmap is empty: no particle is created.
    engine.place_sphere(PT_SPRK, centre, 0.25);
    assert_eq!(
        engine.read_voxel(128, 128, 128).map(|v| v.ty as u32),
        Some(PT_NONE),
        "a spark drawn on empty sky must not create a voxel"
    );

    // A non-conductor is not a valid spark target either.
    engine.place_sphere(PT_STNE, centre, 0.25);
    engine.place_sphere(PT_SPRK, centre, 0.25);
    assert_eq!(
        engine.read_voxel(128, 128, 128).map(|v| v.ty as u32),
        Some(PT_STNE),
        "a spark cannot replace a non-conductor"
    );
    engine.erase_sphere(centre, 0.25);

    // A resting conductor becomes a spark with life 4, ctype = old type and
    // the `Simulation.cpp:1871` +10 K heating.
    engine.place_sphere(PT_METL, centre, 0.25);
    let before = engine.read_voxel(128, 128, 128).expect("painted METL");
    engine.place_sphere(PT_SPRK, centre, 0.25);
    let spark = engine.read_voxel(128, 128, 128).expect("sparked METL");
    assert_eq!(spark.ty as u32, PT_SPRK);
    assert_eq!(spark.life, 4, "a newly placed spark starts with life 4");
    assert_eq!(
        read_word(client, &engine.bufs.aux_ctype, index, VOXEL_COUNT),
        PT_METL,
        "the conductor must be remembered in ctype"
    );
    assert!(
        (spark.temp_k() - (before.temp_k() + 10.0)).abs() < 0.2,
        "create_part heats METL by 10 K"
    );

    // After the spark cycle the conductor comes back; a missing ctype used to
    // make the spark decay into a brand-new METL voxel on empty sky.
    for _ in 0..8 {
        engine.step();
    }
    assert_eq!(
        engine.read_voxel(128, 128, 128).map(|v| v.ty as u32),
        Some(PT_METL),
        "the spark must restore its ctype conductor"
    );
    println!("case spark replace: ok");
}

/// Per-axis ellipsoid/cuboid/prism brushes, the swept segment and the
/// temperature tool all execute the device formula the host mirrors.
fn case_brush_shapes(engine: &mut Voxel3dEngine) {
    let centre = [64.0, 64.0, 64.0];
    for shape in [
        BrushShape3d::Ellipsoid,
        BrushShape3d::Cuboid,
        BrushShape3d::Prism,
    ] {
        engine.reset();
        let brush = Brush3d::with_shape(centre, [4.0, 2.0, 3.0], shape);
        let mut expected = 0u32;
        brush.for_each_voxel(|_| expected += 1);
        assert!(expected > 0, "{shape:?} brush must cover voxels");

        let painted = engine.place_brush(PT_STNE, brush);
        assert_eq!(painted, expected, "{shape:?} host/device count");
        assert_eq!(
            engine.read_voxel(64, 64, 64).map(|v| v.ty as u32),
            Some(PT_STNE),
            "{shape:?} centre"
        );
        assert_eq!(
            engine.read_voxel(59, 64, 64).map(|v| v.ty as u32),
            Some(PT_NONE),
            "{shape:?} outside +x"
        );

        let erased = engine.erase_brush(brush);
        assert_eq!(erased, expected, "{shape:?} erase count");
        assert_eq!(
            engine.read_voxel(64, 64, 64).map(|v| v.ty as u32),
            Some(PT_NONE),
            "{shape:?} erased centre"
        );
    }

    // The temperature brush preserves the type word and clamps the packed
    // fixed-point field.
    engine.reset();
    let brush = Brush3d::new(centre, 2.0);
    engine.place_brush(PT_STNE, brush);
    let before = engine.read_voxel(64, 64, 64).expect("centre").temp_k();
    let heated = engine.adjust_brush_temp(brush, 100.0);
    let after = engine.read_voxel(64, 64, 64).expect("centre");
    assert_eq!(after.ty as u32, PT_STNE, "temperature keeps the type");
    assert!(heated > 0);
    assert!(
        (after.temp_k() - (before + 100.0)).abs() < 0.2,
        "temperature delta {} -> {}",
        before,
        after.temp_k()
    );

    // A swept ellipsoid rail paints a contiguous line and erases it again.
    engine.reset();
    let from = [40.0, 64.0, 64.0];
    let to = [80.0, 64.0, 64.0];
    let radii = [1.5, 1.5, 1.5];
    let painted = engine.place_brush_segment(PT_SAND, from, to, radii, BrushShape3d::Ellipsoid);
    assert!(painted > 0, "the segment must paint voxels");
    for x in [40, 60, 80] {
        assert_eq!(
            engine.read_voxel(x, 64, 64).map(|v| v.ty as u32),
            Some(PT_SAND),
            "segment endpoint {x}"
        );
    }
    assert_eq!(
        engine.read_voxel(60, 68, 64).map(|v| v.ty as u32),
        Some(PT_NONE),
        "segment radius"
    );
    let erased = engine.erase_brush_segment(from, to, radii, BrushShape3d::Ellipsoid);
    assert_eq!(erased, painted);
    assert_eq!(
        engine.read_voxel(60, 64, 64).map(|v| v.ty as u32),
        Some(PT_NONE)
    );

    // Cuboid segments use the expanded-AABB footprint.
    engine.reset();
    let painted = engine.place_brush_segment(
        PT_STNE,
        [40.0, 64.0, 64.0],
        [50.0, 70.0, 64.0],
        [1.0, 1.0, 1.0],
        BrushShape3d::Cuboid,
    );
    assert!(painted > 0);
    assert_eq!(
        engine.read_voxel(45, 67, 64).map(|v| v.ty as u32),
        Some(PT_STNE),
        "cuboid sweep footprint"
    );

    // The flood-fill path paints an explicit index list (the connected set is
    // gathered on the host); every stamped voxel must be live and mark its
    // brick.
    engine.reset();
    let mut indices = Vec::new();
    for y in 100..104 {
        for z in 100..104 {
            for x in 100..104 {
                indices.push((y << 16) | (z << 8) | x);
            }
        }
    }
    assert_eq!(
        engine.paint_indices(PT_WATR, &indices),
        indices.len() as u32
    );
    assert_eq!(
        engine.read_voxel(102, 102, 102).map(|v| v.ty as u32),
        Some(PT_WATR)
    );
    assert_eq!(
        engine.read_voxel(104, 102, 102).map(|v| v.ty as u32),
        Some(PT_NONE)
    );

    println!("case brush shapes: ok");
}

/// A powder voxel falls and a solid floor stops it.
///
/// `STNE.cpp:35` makes stone a `TYPE_PART` powder, so a floating stone bar
/// would fall with the sand; the floor uses glass, which `GLAS.cpp` keeps a
/// `TYPE_SOLID`.
fn case_powder(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.reset();

    // Wide glass bar built from five overlapping radius-4 spheres.
    for x in [120.0f32, 124.0, 128.0, 132.0, 136.0] {
        engine.place_sphere(PT_GLAS, [x, 100.0, 128.0], 4.0);
    }
    let floor = scan_box(
        client,
        &engine.bufs.voxel_a,
        [116, 96, 124],
        [140, 106, 132],
        PT_GLAS,
        NO_SECOND_TYPE,
    );
    assert!(floor.count > 0, "glass bar missing");

    engine.place_sphere(PT_SAND, [128.0, 107.0, 128.0], 0.5);
    assert_eq!(
        engine.read_voxel(128, 107, 128).map(|v| v.ty as u32),
        Some(PT_SAND)
    );

    for _ in 0..5 {
        engine.step();
    }
    let sand = scan_box(
        client,
        &engine.bufs.voxel_a,
        [112, 90, 120],
        [144, 112, 136],
        PT_SAND,
        NO_SECOND_TYPE,
    );
    assert_eq!(sand.count, 1, "exactly one sand voxel must remain");
    assert!(
        sand.max[1] <= 105,
        "sand did not fall: still at y = {}",
        sand.max[1]
    );
    assert!(
        sand.min[1] >= 102,
        "sand sank into the solid floor: reached y = {}",
        sand.min[1]
    );
    let floor_after = scan_box(
        client,
        &engine.bufs.voxel_a,
        [116, 96, 124],
        [140, 106, 132],
        PT_GLAS,
        NO_SECOND_TYPE,
    );
    assert_eq!(floor_after.count, floor.count, "solid floor was modified");
    println!("case powder: sand y range {}..{}", sand.min[1], sand.max[1]);
}

/// `Void` kills a loose voxel leaving the bottom face, `Solid` holds it, and
/// `Loop` wraps it to the opposite face.
fn case_edges(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    engine.reset();
    engine.config.edge_mode = EdgeMode::Void;
    // A stone ring removes the sideways escape routes, so the only candidate
    // direction is the out-of-world downward one.
    for (x, z) in [(9.0f32, 10.0f32), (11.0, 10.0), (10.0, 9.0), (10.0, 11.0)] {
        engine.place_sphere(PT_STNE, [x, 0.0, z], 0.5);
    }
    engine.place_sphere(PT_SAND, [10.0, 0.0, 10.0], 0.5);
    assert_eq!(
        engine.read_voxel(10, 0, 10).map(|v| v.ty as u32),
        Some(PT_SAND)
    );
    engine.step();
    assert_eq!(
        engine.read_voxel(10, 0, 10).map(|v| v.ty as u32),
        Some(PT_NONE),
        "void mode must delete a voxel that steps off the bottom face"
    );
    let gone = scan_box(
        &gpu.client,
        &engine.bufs.voxel_a,
        [0, 0, 0],
        [255, 255, 255],
        PT_SAND,
        NO_SECOND_TYPE,
    );
    assert_eq!(gone.count, 0, "void mode must not teleport the voxel");

    // A bottom-face voxel with free sideways neighbours must leave the world as
    // well. Regression guard: the sideways fallbacks ran before the departure
    // check, so a powder sheet resting on the void slid along the face forever
    // instead of falling out, which looked like the sheet twitching in place.
    engine.reset();
    for x in 8..=12u32 {
        engine.place_sphere(PT_SAND, [x as f32, 0.0, 10.0], 0.5);
    }
    assert_eq!(
        engine.read_voxel(10, 0, 10).map(|v| v.ty as u32),
        Some(PT_SAND)
    );
    for _ in 0..8 {
        engine.step();
    }
    let sheet = scan_box(
        &gpu.client,
        &engine.bufs.voxel_a,
        [0, 0, 0],
        [255, 255, 255],
        PT_SAND,
        NO_SECOND_TYPE,
    );
    assert_eq!(
        sheet.count, 0,
        "void mode must empty a bottom-face sheet instead of sliding it sideways"
    );

    engine.config.edge_mode = EdgeMode::Solid;
    engine.reset();
    engine.place_sphere(PT_SAND, [10.0, 0.0, 10.0], 0.5);
    for _ in 0..4 {
        engine.step();
    }
    let held = scan_box(
        &gpu.client,
        &engine.bufs.voxel_a,
        [0, 0, 0],
        [255, 255, 255],
        PT_SAND,
        NO_SECOND_TYPE,
    );
    assert_eq!(held.count, 1, "solid mode must keep the voxel alive");
    assert_eq!(
        (held.min[1], held.max[1]),
        (0, 0),
        "solid mode must hold the voxel on the bottom face"
    );

    engine.config.edge_mode = EdgeMode::Loop;
    engine.reset();
    engine.place_sphere(PT_SAND, [10.0, 0.0, 10.0], 0.5);
    engine.step();
    assert_eq!(
        engine.read_voxel(10, 255, 10).map(|v| v.ty as u32),
        Some(PT_SAND),
        "loop mode must wrap the voxel onto the top face"
    );
    assert_eq!(
        engine.read_voxel(10, 0, 10).map(|v| v.ty as u32),
        Some(PT_NONE)
    );

    engine.config.edge_mode = EdgeMode::Void;
    println!("case edges: void/solid/loop boundary behaviour ok");
}

/// A water blob spreads along the world floor.
///
/// The floor only exists in `Solid` mode. A void world has no bottom face to
/// pile up on: water that reaches it leaves the world, because TPT moves a
/// particle to its velocity target and kills it when that target is out of
/// bounds (`Simulation::move`, Simulation.cpp:1560-1565) - the second half of
/// this case pins that down.
fn case_water_spread(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.config.edge_mode = EdgeMode::Solid;
    engine.reset();

    engine.place_sphere(PT_WATR, [128.0, 3.0, 128.0], 1.0);
    let before = scan_box(
        client,
        &engine.bufs.voxel_a,
        [121, 0, 121],
        [135, 6, 135],
        PT_WATR,
        NO_SECOND_TYPE,
    );
    assert_eq!(before.count, 7, "painted water voxel count");

    for _ in 0..20 {
        engine.step();
    }
    let after = scan_box(
        client,
        &engine.bufs.voxel_a,
        [100, 0, 100],
        [156, 8, 156],
        PT_WATR,
        NO_SECOND_TYPE,
    );
    assert_eq!(after.count, 7, "water must not disappear");
    assert!(
        after.extent(0) > before.extent(0) || after.extent(2) > before.extent(2),
        "water spread horizontally: before {}x{}, after {}x{}",
        before.extent(0),
        before.extent(2),
        after.extent(0),
        after.extent(2)
    );
    println!(
        "case water: x extent {} -> {}, z extent {} -> {}",
        before.extent(0),
        after.extent(0),
        before.extent(2),
        after.extent(2)
    );

    engine.config.edge_mode = EdgeMode::Void;
    engine.reset();
    engine.place_sphere(PT_WATR, [128.0, 3.0, 128.0], 1.0);
    for _ in 0..40 {
        engine.step();
    }
    let drained = scan_box(
        client,
        &engine.bufs.voxel_a,
        [0, 0, 0],
        [255, 255, 255],
        PT_WATR,
        NO_SECOND_TYPE,
    );
    assert_eq!(
        drained.count, 0,
        "void mode must drain a water blob out of the bottom face"
    );
}

/// Fire next to water is quenched and consumes the water.
///
/// `FIRE.cpp:384-398`: the flame's 5x5x5 loop kills the neighbouring water and
/// then kills the flame, with no steam product. Steam comes from the
/// temperature transition (`WATR.cpp:44-45`), covered separately below.
fn case_fire_water(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.reset();

    engine.place_sphere(PT_WATR, [128.0, 128.0, 128.0], 1.0);
    engine.place_sphere(PT_FIRE, [130.0, 128.0, 128.0], 0.5);
    let water_before = scan_box(
        client,
        &engine.bufs.voxel_a,
        [120, 120, 120],
        [138, 136, 136],
        PT_WATR,
        NO_SECOND_TYPE,
    );

    for _ in 0..3 {
        engine.step();
    }
    let water_after = scan_box(
        client,
        &engine.bufs.voxel_a,
        [120, 120, 120],
        [138, 136, 136],
        PT_WATR,
        NO_SECOND_TYPE,
    );
    let fire_after = scan_box(
        client,
        &engine.bufs.voxel_a,
        [120, 120, 120],
        [138, 136, 136],
        PT_FIRE,
        NO_SECOND_TYPE,
    );
    assert!(
        water_after.count < water_before.count,
        "fire did not consume any of the {} water voxels",
        water_before.count
    );
    assert_eq!(
        fire_after.count, 0,
        "the flame must be quenched by the water"
    );
    println!(
        "case fire/water: water {} -> {}, fire {}",
        water_before.count, water_after.count, fire_after.count
    );
}

/// Water heated past its boiling point turns into steam.
///
/// `WATR.cpp:44-45` is the real steam path; a direct flame contact kills the
/// water instead, so this case heats the same blob with the temperature tool.
fn case_heated_water_steams(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.reset();

    let brush = Brush3d::new([128.0, 128.0, 128.0], 1.0);
    engine.place_brush(PT_WATR, brush);
    engine.adjust_brush_temp(brush, 150.0);
    engine.step();
    let steam = scan_box(
        client,
        &engine.bufs.voxel_a,
        [120, 120, 120],
        [138, 136, 136],
        PT_WTRV,
        NO_SECOND_TYPE,
    );
    assert!(steam.count > 0, "heated water did not boil into steam");
    println!("case heated water: steam = {}", steam.count);
}

/// Lava next to water quenches to stone and consumes the water.
///
/// `FIRE.cpp:384-398` applies to the lava's flame loop as well: the water is
/// killed, the lava freezes to stone, and no steam is created on contact.
fn case_lava_water(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.reset();

    engine.place_sphere(PT_WATR, [128.0, 128.0, 128.0], 1.0);
    engine.place_sphere(PT_LAVA, [130.0, 128.0, 128.0], 0.5);
    let water_before = scan_box(
        client,
        &engine.bufs.voxel_a,
        [120, 120, 120],
        [138, 136, 136],
        PT_WATR,
        NO_SECOND_TYPE,
    );
    assert_eq!(
        engine.read_voxel(130, 128, 128).map(|v| v.ty as u32),
        Some(PT_LAVA)
    );

    for _ in 0..2 {
        engine.step();
    }
    let stone = scan_box(
        client,
        &engine.bufs.voxel_a,
        [120, 120, 120],
        [138, 136, 136],
        PT_STNE,
        NO_SECOND_TYPE,
    );
    let water_after = scan_box(
        client,
        &engine.bufs.voxel_a,
        [120, 120, 120],
        [138, 136, 136],
        PT_WATR,
        NO_SECOND_TYPE,
    );
    assert!(stone.count > 0, "lava did not quench to stone");
    assert!(
        water_after.count < water_before.count,
        "lava did not consume any of the {} water voxels",
        water_before.count
    );
    println!(
        "case lava/water: stone = {}, water {} -> {}",
        stone.count, water_before.count, water_after.count
    );
}

/// A bomb erases loose voxels, spares stone and leaves fire.
fn case_explosion(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.reset();

    engine.place_sphere(PT_SAND, [128.0, 120.0, 128.0], 2.0);
    engine.place_sphere(PT_STNE, [134.0, 124.0, 128.0], 0.5);
    engine.place_sphere(PT_BOMB, [128.0, 123.0, 128.0], 0.5);
    let sand_before = scan_box(
        client,
        &engine.bufs.voxel_a,
        [124, 116, 124],
        [132, 124, 132],
        PT_SAND,
        NO_SECOND_TYPE,
    );
    assert!(sand_before.count > 0, "sand target missing");

    engine.step();

    let sand_after = scan_box(
        client,
        &engine.bufs.voxel_a,
        [118, 112, 118],
        [138, 136, 138],
        PT_SAND,
        NO_SECOND_TYPE,
    );
    let ember_after = scan_box(
        client,
        &engine.bufs.voxel_a,
        [118, 112, 118],
        [138, 136, 138],
        PT_EMBR,
        NO_SECOND_TYPE,
    );
    let stone_after = scan_box(
        client,
        &engine.bufs.voxel_a,
        [118, 112, 118],
        [138, 136, 138],
        PT_STNE,
        NO_SECOND_TYPE,
    );
    assert_eq!(sand_after.count, 0, "explosion left sand behind");
    assert!(ember_after.count > 0, "explosion left no embers");
    assert_eq!(
        stone_after.count, 0,
        "the blast must replace stone with embers"
    );
    println!(
        "case explosion: embers = {}, stone = {}",
        ember_after.count, stone_after.count
    );
}

/// A connected TNT run arms and detonates as one mass.
///
/// `BANG.cpp:49-127`: the fresh seed floods `tmp = 2` through the whole
/// 4-connected run, then every voxel counts `2 -> 3 -> 0` and explodes on the
/// same frame.
fn case_tnt_mass(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.reset();

    for x in 120..=126 {
        engine.place_sphere(PT_BANG, [x as f32, 128.0, 128.0], 0.5);
    }
    engine.place_sphere(PT_FIRE, [119.0, 128.0, 128.0], 0.5);
    for _ in 0..5 {
        engine.step();
    }
    let tnt = scan_box(
        client,
        &engine.bufs.voxel_a,
        [116, 124, 124],
        [132, 132, 132],
        PT_BANG,
        NO_SECOND_TYPE,
    );
    let embers = scan_box(
        client,
        &engine.bufs.voxel_a,
        [116, 124, 124],
        [132, 132, 132],
        PT_EMBR,
        NO_SECOND_TYPE,
    );
    assert_eq!(tnt.count, 0, "the TNT run did not fully detonate");
    assert!(embers.count > 0, "the TNT run left no embers");
    println!("case TNT: embers = {}", embers.count);
}

/// Wood pyrolyses into charcoal at its high-temperature transition.
fn case_wood_pyrolysis(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.reset();

    let brush = Brush3d::new([128.0, 128.0, 128.0], 1.0);
    engine.place_brush(PT_WOOD, brush);
    // `WOOD.cpp:30-38`: above 873 K the generic high-temperature transition
    // turns wood into FIRE; the `Update` pyrolysis needs the window above
    // 773 K but below that threshold.
    engine.adjust_brush_temp(brush, 530.0);
    // `WOOD.cpp:56`: pyrolysis also needs a vacuum (`pv <= -10`). Reload the
    // just painted world with a negative pressure in the centre coarse cell.
    let voxels = engine.read_world_words().expect("world read-back");
    let mut air = vec![0.0f32; 6 * AIR_COUNT];
    let cell = ((128u32 >> 2) << 12) | ((128u32 >> 2) << 6) | (128u32 >> 2);
    air[3 * AIR_COUNT + cell as usize] = -20.0;
    engine.load_world(&voxels, &air).expect("world reload");
    engine.step();
    let charcoal = scan_box(
        client,
        &engine.bufs.voxel_a,
        [120, 120, 120],
        [136, 136, 136],
        PT_BCOL,
        NO_SECOND_TYPE,
    );
    assert!(charcoal.count > 0, "wood did not pyrolyse into charcoal");
    println!("case wood: charcoal = {}", charcoal.count);
}

/// Breakable metal melts into lava above its transition temperature.
fn case_brmt_melts(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let client = &gpu.client;
    engine.reset();

    let brush = Brush3d::new([128.0, 128.0, 128.0], 1.0);
    engine.place_brush(PT_BRMT, brush);
    engine.adjust_brush_temp(brush, 1000.0);
    engine.step();
    let lava = scan_box(
        client,
        &engine.bufs.voxel_a,
        [120, 120, 120],
        [136, 136, 136],
        PT_LAVA,
        NO_SECOND_TYPE,
    );
    assert!(lava.count > 0, "breakable metal did not melt into lava");
    println!("case BRMT: lava = {}", lava.count);
}

/// Picking hits the top voxel and, after erasing it, the one below.
fn case_pick(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    let _ = gpu;
    engine.reset();

    engine.place_sphere(PT_STNE, [128.0, 100.0, 128.0], 0.5);
    engine.place_sphere(PT_STNE, [128.0, 101.0, 128.0], 0.5);

    let origin = [128.5, 110.5, 128.5];
    let dir = [0.0, -1.0, 0.0];
    let top = engine
        .pick(origin, dir)
        .expect("pick must hit the top voxel");
    assert_eq!(top.voxel, [128, 101, 128]);
    assert_eq!(top.ty, PT_STNE);
    assert_eq!(top.normal, [0, 1, 0], "top face normal");
    assert!(
        (top.temp_k - painted_stne_temp_k()).abs() < 0.2,
        "picked temperature"
    );

    engine.erase_sphere([128.0, 101.0, 128.0], 0.5);
    let lower = engine
        .pick(origin, dir)
        .expect("pick must fall through to the lower voxel");
    assert_eq!(lower.voxel, [128, 100, 128]);
    assert_eq!(lower.ty, PT_STNE);

    // The ray misses the world entirely when it points the other way.
    assert!(engine.pick(origin, [0.0, 1.0, 0.0]).is_none());
    println!("case pick: top {:?}, lower {:?}", top.voxel, lower.voxel);
}

/// A rigid stone box falls onto a stone floor and keeps a finite pose.
fn case_rigid(gpu: &TestGpu, engine: &mut Voxel3dEngine) {
    engine.reset();
    engine.place_sphere(PT_STNE, [128.0, 60.0, 128.0], 20.0);
    let floor_top = 80.0f32;

    let slot = engine.spawn_box([128.0, 84.0, 128.0], [2, 2, 2], PT_STNE);
    assert_eq!(slot, 0);

    for _ in 0..60 {
        engine.step();
    }

    let bytes = gpu
        .client
        .read_one_unchecked(engine.rigid.body_state.clone());
    let state: &[f32] = bytemuck::cast_slice(&bytes);
    assert!(state.len() >= 32, "body_state too short");
    let pose = [state[0], state[1], state[2]];
    assert!(
        pose.iter().all(|value| value.is_finite()),
        "box pose is not finite: {pose:?}"
    );
    assert!(
        pose[1] > floor_top - 2.0,
        "box fell through the stone floor: y = {}",
        pose[1]
    );
    assert!(pose[1] < 90.0, "box did not fall: y = {}", pose[1]);
    println!(
        "case rigid: slot {slot}, pose ({:.3}, {:.3}, {:.3})",
        pose[0], pose[1], pose[2]
    );
}

/// Save/load and sparse-brick undo helpers round-trip the authoritative grid.
fn case_save_undo(engine: &mut Voxel3dEngine) {
    engine.reset();
    engine.place_sphere(PT_COAL, [64.0, 80.0, 64.0], 3.0);

    let voxels = engine.read_world_words().expect("world read-back");
    let air = engine.read_air_fields().expect("air read-back");
    assert_eq!(voxels.len(), VOXEL_COUNT);
    assert_eq!(air.len(), 6 * tpt_core::voxel3d::AIR_COUNT);
    let brick = brick_index(64, 80, 64);
    let saved_brick = engine
        .read_brick_snapshot(&[brick as u32])
        .expect("brick snapshot read-back");
    assert_eq!(saved_brick.voxels.len(), 1);
    assert_eq!(saved_brick.voxels[0].1.len(), 512);
    assert_eq!(saved_brick.ctype.len(), 1);
    assert_eq!(saved_brick.tmp.len(), 1);
    assert_eq!(saved_brick.tmp2.len(), 1);
    assert_eq!(saved_brick.tmp3.len(), 1);
    assert!(
        saved_brick.tmp[0].1.contains(&50),
        "the COAL snapshot lost its default `tmp = 50`"
    );

    engine.erase_sphere([64.0, 80.0, 64.0], 6.0);
    assert!(engine.read_voxel(64, 80, 64).is_some_and(|v| v.ty == 0));

    engine.restore_brick_snapshot(&saved_brick);
    assert_eq!(
        engine.read_voxel(64, 80, 64).map(|v| v.ty),
        Some(PT_COAL as u8)
    );

    engine.reset();
    engine.load_world(&voxels, &air).expect("world load");
    assert_eq!(
        engine.read_voxel(64, 80, 64).map(|v| v.ty),
        Some(PT_COAL as u8)
    );
    println!("case save/undo: world and brick round-trip ok");
}

#[test]
fn voxel3d_core_engine() {
    let gpu = &*GPU;
    println!(
        "3D core test adapter: {} (backend {})",
        gpu.adapter, gpu.backend
    );

    let mut engine = Voxel3dEngine::new(gpu.client.clone(), engine_config());

    case_paint_erase(gpu, &mut engine);
    case_spark_replace(gpu, &mut engine);
    case_brush_shapes(&mut engine);
    case_powder(gpu, &mut engine);
    case_edges(gpu, &mut engine);
    case_water_spread(gpu, &mut engine);
    case_fire_water(gpu, &mut engine);
    case_heated_water_steams(gpu, &mut engine);
    case_lava_water(gpu, &mut engine);
    case_explosion(gpu, &mut engine);
    case_tnt_mass(gpu, &mut engine);
    case_wood_pyrolysis(gpu, &mut engine);
    case_brmt_melts(gpu, &mut engine);
    case_pick(gpu, &mut engine);
    case_save_undo(&mut engine);
    case_rigid(gpu, &mut engine);

    println!("3D core test: all cases passed");
}
