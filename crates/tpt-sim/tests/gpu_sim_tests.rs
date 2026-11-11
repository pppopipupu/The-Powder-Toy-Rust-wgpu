//! GPU integration tests for the `tpt-sim` simulation engine.
//!
//! Every test drives the real CubeCL/wgpu pipeline: [`create_test_engine`]
//! performs the same device bring-up as `tpt-app` (one `wgpu::Device`/`Queue`
//! shared with CubeCL through [`WgpuSetup`]) so the tests exercise exactly the
//! kernels the application dispatches.
//!
//! The tests are serialised through [`GPU_LOCK`]: each test registers its own
//! CubeCL device, and CI machines only expose a single GPU, so running the
//! suites concurrently would contend for the same physical adapter.

use std::sync::{Mutex, MutexGuard};

use cubecl::client::Client;
use cubecl::wgpu::wgpu;
use cubecl::wgpu::{AutoCompiler, RuntimeOptions, WgpuRuntime, WgpuSetup, init_device};
use cubecl_runtime::runtime::Runtime;

use tpt_core::commands::{DecoMode, ParticleField, RigidSpawnShape, SimCommand, SimToolKind};
use tpt_core::constants::*;
use tpt_core::particle::{AirMode, GpuParticle, GravityMode, RigidBody2D, SimulationConfig};
use tpt_sim::SimulationEngine;

/// Serialises the GPU tests; see the module docs.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Acquires the GPU test lock, ignoring poisoning from a previously failed test.
fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// One CubeCL client for the whole test binary.
///
/// Every test used to register its own `wgpu` device, and CubeCL keeps those
/// devices alive for the life of the process, so a full run leaked one device
/// per test and the last ones failed with `Device(OutOfMemory)`. The tests are
/// serialised by [`GPU_LOCK`], so they can share one device safely.
///
/// The `wgpu` device is requested with `required_limits: adapter.limits()`
/// because CubeCL binds many storage buffers per kernel (see
/// `max_storage_buffers_per_shader_stage`), and the resulting handles are handed
/// to CubeCL through [`WgpuSetup`] so simulation and presentation share one
/// physical device.
static CLIENT: std::sync::LazyLock<Client> = std::sync::LazyLock::new(|| {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("no GPU adapter available for the tpt-sim integration tests");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("tpt-sim-test-device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .expect("failed to create the test wgpu device/queue");

    let setup = WgpuSetup {
        instance: instance.clone(),
        adapter: adapter.clone(),
        device: device.clone(),
        queue: queue.clone(),
        backend: adapter.get_info().backend,
    };
    let cubecl_device = init_device(setup, RuntimeOptions::default());
    WgpuRuntime::<AutoCompiler>::client(&cubecl_device)
});

/// Builds a simulation engine on the shared device.
fn create_test_engine(xres: u32, yres: u32) -> SimulationEngine {
    SimulationEngine::new(CLIENT.clone(), SimulationConfig::new(xres, yres))
}

/// Paints a filled pixel rectangle with `ty` (overwriting anything in the way).
fn fill_rect(engine: &mut SimulationEngine, x0: i32, y0: i32, x1: i32, y1: i32, ty: u32) {
    let width = (x1 - x0 + 1).max(0) as usize;
    let height = (y1 - y0 + 1).max(0) as usize;
    let mut points = Vec::with_capacity(width * height);
    for y in y0..=y1 {
        for x in x0..=x1 {
            points.push((x, y));
        }
    }
    engine.apply_command(SimCommand::PlaceParticles {
        points,
        elem_ty: ty,
        replace_mode: true,
        dcolour: 0,
    });
}

/// Turn the pixel into a spark the way TPT can: a spark never allocates, it
/// takes over the conductor at the pixel (`Simulation.cpp:1852-1878`).
fn spark_at(engine: &mut SimulationEngine, x: i32, y: i32) {
    if engine.bufs.particle_index_at(x, y).is_none() {
        fill_rect(engine, x, y, x, y, PT_METL);
    }
    engine.apply_command(SimCommand::PlaceParticles {
        points: vec![(x, y)],
        elem_ty: PT_SPRK,
        replace_mode: false,
        dcolour: 0,
    });
}

/// Number of live particles of element `ty` in the host shadow.
fn count_type(engine: &SimulationEngine, ty: u32) -> usize {
    engine.bufs.part_ty.iter().filter(|&&t| t == ty).count()
}

/// Number of live particles still welded to a rigid body, whatever element
/// they have become: a burning pixel turns into `FIRE` while it stays part of
/// the body until the fracture removes it.
fn count_rigid_any(engine: &SimulationEngine) -> usize {
    (0..engine.bufs.capacity)
        .filter(|&id| engine.bufs.part_ty[id] != PT_NONE)
        .filter(|&id| engine.bufs.part_flags[id] & FLAG_RIGID_PIXEL != 0)
        .count()
}

/// Number of live particles of element `ty` still welded to a rigid body.
fn count_rigid_type(engine: &SimulationEngine, ty: u32) -> usize {
    let mut count = 0usize;
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == ty && engine.bufs.part_flags[id] & FLAG_RIGID_PIXEL != 0 {
            count += 1;
        }
    }
    count
}

/// Mean `y` coordinate of every particle of element `ty` (`NAN` when absent).
fn mean_y(engine: &SimulationEngine, ty: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == ty {
            sum += engine.bufs.part_y[id];
            count += 1;
        }
    }
    if count == 0 {
        f32::NAN
    } else {
        sum / count as f32
    }
}

/// Hottest particle of element `ty` (`NEG_INFINITY` when absent).
fn max_temp_of_type(engine: &SimulationEngine, ty: u32) -> f32 {
    let mut hottest = f32::NEG_INFINITY;
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == ty {
            hottest = hottest.max(engine.bufs.part_temp[id]);
        }
    }
    hottest
}

/// Highest positive air pressure anywhere on the canvas.
fn max_air_pressure(engine: &SimulationEngine) -> f32 {
    engine.bufs.air_pv.iter().copied().fold(0.0f32, f32::max)
}

/// Overwrites the temperature of every particle of element `ty` (host shadow).
fn set_temp_of_type(engine: &mut SimulationEngine, ty: u32, temp: f32) {
    for id in 0..engine.bufs.capacity {
        if engine.bufs.part_ty[id] == ty {
            engine.bufs.part_temp[id] = temp;
        }
    }
    engine.bufs.mark_particles_dirty();
}

/// Renders one frame and returns `(byte length, number of non-black pixels)`.
fn render_stats(engine: &mut SimulationEngine) -> (usize, usize) {
    let rgba = engine.render_rgba();
    let len = rgba.len();
    let non_black = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|px| px[0] != 0 || px[1] != 0 || px[2] != 0)
        .count();
    (len, non_black)
}

/// Axis-aligned pixel bounds of a rigid body: `(min_x, min_y, max_x, max_y)`.
fn body_bounds(body: &RigidBody2D) -> (i32, i32, i32, i32) {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    for px in &body.pixels {
        min_x = min_x.min(px.grid_x);
        min_y = min_y.min(px.grid_y);
        max_x = max_x.max(px.grid_x);
        max_y = max_y.max(px.grid_y);
    }
    (min_x, min_y, max_x, max_y)
}

/// Advances the simulation and pulls the host shadows back.
///
/// The engine keeps its particle/air state device-resident while it runs; the
/// assertions below inspect the host SoA directly, so each step is followed by
/// the same explicit read-back a save or the Lua bridge performs.
fn step(engine: &mut SimulationEngine) {
    engine.step();
    engine.ensure_host_current();
}

#[path = "gpu_sim/edits.rs"]
mod edits;
#[path = "gpu_sim/elec_partial.rs"]
mod elec_partial;
#[path = "gpu_sim/electronics.rs"]
mod electronics;
#[path = "gpu_sim/flood_component.rs"]
mod flood_component;
#[path = "gpu_sim/gravity_fft.rs"]
mod gravity_fft;
#[path = "gpu_sim/movement.rs"]
mod movement;
#[path = "gpu_sim/pair_resolution.rs"]
mod pair_resolution;
#[path = "gpu_sim/pipe.rs"]
mod pipe;
#[path = "gpu_sim/reactions.rs"]
mod reactions;
#[path = "gpu_sim/render.rs"]
mod render;
#[path = "gpu_sim/rigid.rs"]
mod rigid;
#[path = "gpu_sim/stickman.rs"]
mod stickman;
#[path = "gpu_sim/thermal.rs"]
mod thermal;
#[path = "gpu_sim/tools.rs"]
mod tools;
#[path = "gpu_sim/water_equal.rs"]
mod water_equal;
