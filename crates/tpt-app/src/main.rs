//! The Powder Toy — Rust + wgpu + CubeCL front-end.
//!
//! This binary owns the `winit` event loop and ties the pieces together:
//!
//! * [`gpu_ctx::GpuContext`] creates the single `wgpu` device/queue triple that
//!   is shared by `egui-wgpu` and CubeCL, plus the simulation framebuffer
//!   texture that egui samples.
//! * [`app_state::AppState`] owns the GPU simulation, undo history, Lua runtime
//!   and all UI-facing state.
//! * [`ui_panels::draw_app_ui`] builds the main menu / HUD and returns a new
//!   canvas size when the simulation texture must be reallocated.
//!
//! Besides the interactive 1280x800 window, the binary supports a
//! `--headless-smoke` mode that runs a fixed number of GPU simulation + render
//! ticks on an offscreen device and prints telemetry. That mode exists so CI
//! can verify the whole CubeCL/wgpu stack without a display server.

pub mod app_state;
mod app_window;
pub mod brushes;
pub mod gpu_ctx;
pub mod i18n;
pub mod prefs;
pub mod render3d;
pub mod ui_panels;

use std::time::{Duration, Instant};

use tpt_core::commands::{RigidSpawnShape, SimCommand};
use tpt_core::constants::{PT_DUST, PT_NONE, PT_PLSM, PT_STNE, PT_WATR, RENDER_DEFAULT, RENDER_FANCY};
use tpt_sim::SimBuffers;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::app_state::AppState;
use crate::app_window::PowderToyApp;
use crate::i18n::{t, t_fmt};

/// Window title shown in the OS title bar.
///
/// A unit type instead of a `&'static str`: the title is localized, and
/// `i18n::t` is not a `const fn`, so the value has to be resolved at the point
/// the window is created.
#[derive(Clone, Copy, Debug)]
pub struct WindowTitle;

impl From<WindowTitle> for String {
    fn from(_: WindowTitle) -> Self {
        t("app.window_title").to_string()
    }
}

impl std::fmt::Display for WindowTitle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", t("app.window_title"))
    }
}

pub const WINDOW_TITLE: WindowTitle = WindowTitle;
/// Number of simulation/render ticks performed by `--headless-smoke`.
const HEADLESS_SMOKE_TICKS: u32 = 10;
/// Default number of presented frames for `--windowed-smoke`.
const WINDOWED_SMOKE_FRAMES: u32 = 90;
/// Canvas size used by `--headless-smoke` (small, so CI runs quickly).
const HEADLESS_CANVAS: (u32, u32) = (256, 256);
/// Canvas size used by `--bench` (TPT's default resolution).
const BENCH_CANVAS: (u32, u32) = (612, 384);
/// Default number of timed ticks for `--bench`.
const BENCH_DEFAULT_TICKS: u32 = 300;
/// Untimed warm-up ticks so shader compilation and pool growth do not skew the
/// benchmark.
const BENCH_WARMUP_TICKS: u32 = 20;
/// Default canvas side used by `--save-smoke` (the UI's maximum canvas).
const SAVE_SMOKE_SIDE: u32 = 2048;
/// Ticks simulated by `--save-smoke` before the scene is written out.
const SAVE_SMOKE_TICKS: u32 = 5;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{WINDOW_TITLE}");
        println!();
        println!(
            "Usage: tpt-app [--headless-smoke] [--windowed-smoke [frames]] [--voxel-windowed-smoke [frames]] [--bench [ticks]]"
        );
        println!();
        println!(
            "  --headless-smoke  Run {HEADLESS_SMOKE_TICKS} offscreen GPU simulation + render"
        );
        println!("                    ticks, print telemetry, then exit with status 0.");
        println!(
            "  --windowed-smoke  Open the real window, run a seeded scene for {WINDOWED_SMOKE_FRAMES}"
        );
        println!("                    presented frames, then exit with status 0.");
        println!("  --voxel-windowed-smoke [frames]  Open the window in the 3D sandbox, paint a");
        println!("                    floor and present {WINDOWED_SMOKE_FRAMES} frames.");
        println!(
            "  --bench [ticks]   Run {BENCH_DEFAULT_TICKS} timed GPU simulation + render ticks"
        );
        println!("                    on a 612x384 scene and print per-frame timings.");
        println!(
            "  --save-smoke [side]  Create a {SAVE_SMOKE_SIDE}x{SAVE_SMOKE_SIDE} canvas, save it,"
        );
        println!("                    load it back and compare the particle count.");
        return;
    }

    if args.iter().any(|arg| arg == "--headless-smoke") {
        std::process::exit(run_headless_smoke());
    }

    if let Some(pos) = args.iter().position(|arg| arg == "--save-smoke") {
        let side = args
            .get(pos + 1)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(SAVE_SMOKE_SIDE);
        std::process::exit(run_save_smoke(side));
    }

    if let Some(pos) = args.iter().position(|arg| arg == "--bench") {
        let ticks = args
            .get(pos + 1)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(BENCH_DEFAULT_TICKS);
        std::process::exit(run_bench(ticks));
    }

    let windowed_smoke = args
        .iter()
        .position(|arg| arg == "--windowed-smoke")
        .map(|pos| {
            args.get(pos + 1)
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(WINDOWED_SMOKE_FRAMES)
        });
    let voxel_smoke = args
        .iter()
        .position(|arg| arg == "--voxel-windowed-smoke")
        .map(|pos| {
            args.get(pos + 1)
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(WINDOWED_SMOKE_FRAMES)
        });
    let windowed_smoke = voxel_smoke.or(windowed_smoke);

    if let Err(err) = run_interactive(windowed_smoke, voxel_smoke.is_some()) {
        eprintln!("{}", t_fmt("app.fatal_error", &[&err]));
        std::process::exit(1);
    }
}

/// Creates the event loop and runs the interactive application until the user
/// closes the window.
fn run_interactive(smoke_frames: Option<u32>, voxel_smoke: bool) -> Result<(), String> {
    let prefs = crate::prefs::Prefs::load();
    crate::i18n::set_language(prefs.language);
    let event_loop = EventLoop::new()
        .map_err(|err| t_fmt("app.event_loop_create_failed", &[&err.to_string()]))?;
    // `about_to_wait` waits until the next draw deadline and requests the
    // redraw then, so the loop sleeps between frames instead of spinning.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = PowderToyApp::new(smoke_frames, voxel_smoke);
    event_loop
        .run_app(&mut app)
        .map_err(|err| t_fmt("app.event_loop_run_failed", &[&err.to_string()]))
}

/// Runs the `--headless-smoke` verification path.
///
/// Creates a display-less `wgpu` device, shares it with CubeCL, seeds a small
/// deterministic scene and performs [`HEADLESS_SMOKE_TICKS`] simulation +
/// render + texture-upload ticks. Returns the process exit code.
fn run_headless_smoke() -> i32 {
    match headless_smoke_inner() {
        Ok(()) => {
            println!("[headless-smoke] PASS");
            0
        }
        Err(err) => {
            eprintln!("[headless-smoke] FAIL: {err}");
            1
        }
    }
}

/// Implementation of [`run_headless_smoke`], returning a descriptive error.
fn headless_smoke_inner() -> Result<(), String> {
    let gpu = create_headless_gpu("tpt-headless-device")?;
    let (device, queue) = (gpu.device.clone(), gpu.queue.clone());
    println!(
        "[headless-smoke] adapter: {} ({:?}, backend {:?})",
        gpu.adapter_info.name, gpu.adapter_info.device_type, gpu.adapter_info.backend
    );

    let mut state = AppState::new(gpu.client.clone());
    let (canvas_w, canvas_h) = state.start_new_canvas(HEADLESS_CANVAS.0, HEADLESS_CANVAS.1);
    println!("[headless-smoke] canvas: {canvas_w}x{canvas_h}");

    seed_smoke_scene(&mut state, canvas_w, canvas_h);

    let initial_particles = state.engine.bufs.count_active_particles();
    if initial_particles == 0 {
        return Err(String::from(
            "smoke scene seeding produced no particles (PlaceParticles failed)",
        ));
    }

    // Offscreen presentation target with the same format/usage as the windowed
    // simulation texture, so the upload path is identical.
    let (sim_texture, sim_view) = gpu_ctx::create_sim_texture(&device, (canvas_w, canvas_h));
    let canvas_size = (canvas_w, canvas_h);
    let blit = gpu_ctx::FramebufBlit::new(&device);
    let mut blit_verified = false;

    for tick in 1..=HEADLESS_SMOKE_TICKS {
        // A fixed number of frames, independent of the wall-clock TPS pacing
        // `tick_frame` applies.
        state.step_simulation_once();

        let rgba = state.engine.render_rgba();
        if tick == HEADLESS_SMOKE_TICKS {
            // The seeded scene has a stone floor, a water pool and open air
            // above it; assert the pixels instead of trusting the blit parity
            // check, which also passes for a uniformly red or black frame.
            let w = canvas_size.0 as usize;
            let pixel = |x: usize, y: usize| {
                let i = (y * w + x) * 4;
                (rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3])
            };
            let air = pixel(w / 2, 4);
            if air != (0, 0, 0, 255) {
                return Err(format!("air must render black, got {air:?}"));
            }
            let water = pixel(w / 2, canvas_size.1 as usize - 30);
            if water.2 <= water.0 || water.2 < 100 {
                return Err(format!("water must render blue, got {water:?}"));
            }
        }
        let expected_bytes = (canvas_size.0 as usize) * (canvas_size.1 as usize) * 4;
        let uploaded = gpu_ctx::upload_rgba_to_texture(&queue, &sim_texture, canvas_size, rgba);
        if !uploaded {
            return Err(format!(
                "render_rgba returned {} bytes, expected {expected_bytes}",
                rgba.len()
            ));
        }

        // Verify the GPU blit against the host read-back on the first tick: it
        // is the path the interactive window uses instead of `write_texture`.
        // The host slice borrows the engine, so the comparison uses a copy.
        if !blit_verified {
            let rgba_copy = rgba.to_vec();
            let check = BlitCheck {
                blit: &blit,
                texture: &sim_texture,
                view: &sim_view,
                bufs: &state.engine.bufs,
                canvas: canvas_size,
                rgba: &rgba_copy,
            };
            blit_verified = verify_sim_blit(&device, &queue, &check)?;
        }

        let (particles, rigid_bodies, pressure, temperature) = {
            state.engine.ensure_host_current();
            sample_telemetry(&state.engine.bufs)
        };

        println!(
            "[headless-smoke] tick {tick:02}/{HEADLESS_SMOKE_TICKS}: particles={particles} rigid_bodies={rigid_bodies} air_pressure={pressure:+.4} temperature={temperature:.2} K"
        );
    }

    let (particles, rigid_bodies, pressure, temperature) = {
        state.engine.ensure_host_current();
        sample_telemetry(&state.engine.bufs)
    };
    println!(
        "[headless-smoke] final: particles={particles} rigid_bodies={rigid_bodies} air_pressure={pressure:+.4} temperature={temperature:.2} K"
    );

    Ok(())
}

/// A headless wgpu device plus the CubeCL client that shares it.
struct HeadlessGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    client: cubecl::client::Client,
    adapter_info: wgpu::AdapterInfo,
}

/// Creates the shared `wgpu` + CubeCL stack used by the offscreen modes.
fn create_headless_gpu(label: &str) -> Result<HeadlessGpu, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = gpu_ctx::request_high_performance_adapter(&instance, None)
        .map_err(|err| format!("no GPU adapter available: {err}"))?;

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .map_err(|err| format!("failed to create headless device: {err}"))?;

    let adapter_info = adapter.get_info();
    let setup = cubecl::wgpu::WgpuSetup {
        instance,
        adapter,
        device: device.clone(),
        queue: queue.clone(),
        backend: adapter_info.backend,
    };
    // The 3D mode's `CubeCount::Dynamic` launches need one wgpu buffer per
    // handle: the default `Adaptive` allocator aliases small buffers, and wgpu
    // rejects a dispatch using one buffer as both `INDIRECT` and
    // `STORAGE_READ_WRITE`.
    let cubecl_device = cubecl::wgpu::init_device(
        setup,
        cubecl::wgpu::RuntimeOptions {
            memory_config: cubecl::wgpu::MemoryConfiguration::ExclusivePages,
            ..Default::default()
        },
    );
    let client = {
        use cubecl_runtime::runtime::Runtime;
        cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(&cubecl_device)
    };

    Ok(HeadlessGpu {
        device,
        queue,
        client,
        adapter_info,
    })
}

/// Runs the timed benchmark and returns the process exit code.
fn run_bench(ticks: u32) -> i32 {
    match bench_inner(ticks) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("[bench] FAIL: {err}");
            1
        }
    }
}

/// Measures `step + render_gpu` on a full 612x384 scene with a GPU sync every
/// frame, which is the cost the interactive loop pays when it is GPU-bound.
fn bench_inner(ticks: u32) -> Result<(), String> {
    let gpu = create_headless_gpu("tpt-bench-device")?;
    println!(
        "[bench] adapter: {} ({:?}, backend {:?})",
        gpu.adapter_info.name, gpu.adapter_info.device_type, gpu.adapter_info.backend
    );

    let mut state = AppState::new(gpu.client.clone());
    let (width, height) = state.start_new_canvas(BENCH_CANVAS.0, BENCH_CANVAS.1);

    // Measure the fluid-only scene first (the configuration the original TPT
    // can run), then the same scene with one rigid body, whose CPU solver has
    // to read the particle pool back every frame.
    // The third scene keeps a dense plasma cloud alive (plasma dies within
    // ~200 ticks, so it is re-seeded periodically) to expose the cost of the
    // glow/fire render passes, which scale with the number of glowing pixels.
    for (label, with_rigid, with_plasma, render_mode) in [
        ("fluids", false, false, RENDER_DEFAULT),
        ("fluids+rigid", true, false, RENDER_DEFAULT),
        ("plasma", false, true, RENDER_DEFAULT),
        ("plasma-fancy", false, true, RENDER_FANCY),
    ] {
        state.engine.clear();
        state.engine.bufs.config.render_mode = render_mode;
        seed_bench_scene(&mut state, width, height, with_rigid);
        if with_plasma {
            seed_bench_plasma(&mut state, width, height);
        }
        // Upload the seeded scene before the clock starts.
        state.engine.bufs.sync_host_to_gpu();
        state.engine.bufs.refresh_active_count_device();

        for _ in 0..BENCH_WARMUP_TICKS {
            state.engine.step();
            state.engine.render_gpu();
        }
        let _ = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let mut step_time = Duration::ZERO;
        let mut render_time = Duration::ZERO;
        let started = Instant::now();
        for tick in 0..ticks {
            if with_plasma && tick % 40 == 0 {
                seed_bench_plasma(&mut state, width, height);
            }
            let mark = Instant::now();
            state.engine.step();
            // Draining the queue between the two halves attributes the GPU
            // time to the pass that issued it; without it `step` only measures
            // command recording and the whole cost lands in the final poll.
            let _ = gpu.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            step_time += mark.elapsed();

            let mark = Instant::now();
            state.engine.render_gpu();
            // The kernels are asynchronous; waiting every frame turns the wall
            // clock into the real frame cost.
            let _ = gpu.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            render_time += mark.elapsed();
        }
        let total = started.elapsed();

        let particles = state.engine.bufs.particle_count();
        let bodies = state
            .engine
            .bufs
            .rigid_bodies
            .iter()
            .filter(|body| body.active)
            .count();
        let step_ms = step_time.as_secs_f64() * 1000.0 / ticks.max(1) as f64;
        let render_ms = render_time.as_secs_f64() * 1000.0 / ticks.max(1) as f64;
        let frame_ms = total.as_secs_f64() * 1000.0 / ticks.max(1) as f64;
        println!(
            "[bench] {label}: canvas {}x{} ticks={} particles={} rigid_bodies={}",
            BENCH_CANVAS.0, BENCH_CANVAS.1, ticks, particles, bodies
        );
        println!(
            "[bench] {label}: step={step_ms:.3} ms render={render_ms:.3} ms frame={frame_ms:.3} ms -> {:.1} fps",
            1000.0 / frame_ms.max(f64::EPSILON)
        );
    }
    Ok(())
}

/// Seeds the benchmark scene: stone walls around a deep water pool with a sand
/// column and one rigid body, so every kernel pass has work to do.
fn seed_bench_scene(state: &mut AppState, width: u32, height: u32, with_rigid: bool) {
    let width = width as i32;
    let height = height as i32;
    if width <= 0 || height <= 0 {
        return;
    }

    // Stone floor.
    let floor_top = (height - 10).max(0);
    let mut floor = Vec::new();
    for y in floor_top..height {
        for x in 0..width {
            floor.push((x, y));
        }
    }
    state.engine.apply_command(SimCommand::PlaceParticles {
        points: floor,
        elem_ty: PT_STNE,
        replace_mode: true,
        dcolour: 0,
    });

    // Water pool filling most of the canvas.
    let pool_top = (height / 3).max(0);
    let mut water = Vec::new();
    for y in pool_top..floor_top {
        for x in 0..width {
            water.push((x, y));
        }
    }
    state.engine.apply_command(SimCommand::PlaceParticles {
        points: water,
        elem_ty: PT_WATR,
        replace_mode: true,
        dcolour: 0,
    });

    // Sand column above the pool.
    let mut sand = Vec::new();
    for y in 4..pool_top.min(80) {
        for x in (width / 3 - 10)..(width / 3 + 10) {
            sand.push((x, y));
        }
    }
    state.engine.apply_command(SimCommand::PlaceParticles {
        points: sand,
        elem_ty: PT_DUST,
        replace_mode: false,
        dcolour: 0,
    });

    // One rigid body dropped through the pool.
    if with_rigid {
        state.engine.apply_command(SimCommand::SpawnRigidBody {
            x: width as f32 * 0.5,
            y: 40.0,
            half_w: 12,
            half_h: 8,
            shape: RigidSpawnShape::Box,
            elem_ty: PT_STNE,
        });
    }
}

/// Fills the upper-middle of the canvas with plasma for the `plasma` bench scene.
fn seed_bench_plasma(state: &mut AppState, width: u32, height: u32) {
    let width = width as i32;
    let height = height as i32;
    let mut points = Vec::new();
    for y in (height / 8)..(height / 8 + height / 4) {
        for x in (width / 4)..(width * 3 / 4) {
            points.push((x, y));
        }
    }
    state.engine.apply_command(SimCommand::PlaceParticles {
        points,
        elem_ty: PT_PLSM,
        replace_mode: true,
        dcolour: 0,
    });
}

/// GPU resources and host reference frame compared by [`verify_sim_blit`].
struct BlitCheck<'a> {
    /// Framebuffer blit pipeline binding the CubeCL storage buffer.
    blit: &'a gpu_ctx::FramebufBlit,
    /// Destination texture written by the blit.
    texture: &'a wgpu::Texture,
    /// View of `texture` used as the render target.
    view: &'a wgpu::TextureView,
    /// Simulation buffers whose framebuffer is blitted.
    bufs: &'a SimBuffers,
    /// Canvas size in simulation pixels.
    canvas: (u32, u32),
    /// Host read-back the blit must match byte for byte.
    rgba: &'a [u8],
}

/// Runs the GPU framebuffer blit and compares the resulting texture with the
/// host read-back, byte for byte.
///
/// Returns `true` on a bit-identical match. A mismatch is reported as an error
/// so the smoke test fails loudly if the CubeCL buffer binding ever breaks.
fn verify_sim_blit(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    check: &BlitCheck<'_>,
) -> Result<bool, String> {
    let (width, height) = check.canvas;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tpt-smoke-blit"),
    });
    if !check.blit.encode(
        device,
        &mut encoder,
        check.bufs,
        check.view,
        (width, height),
    ) {
        return Err(String::from(
            "framebuf blit could not bind the CubeCL storage buffer",
        ));
    }

    let unpadded = (width as usize) * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tpt-smoke-blit-readback"),
        size: (padded * height as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: check.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    receiver
        .recv()
        .map_err(|err| format!("blit readback channel closed: {err}"))?
        .map_err(|err| format!("blit readback map failed: {err}"))?;

    let mapped = slice
        .get_mapped_range()
        .map_err(|err| format!("blit readback view failed: {err}"))?;
    let mut mismatch = None;
    for row in 0..height as usize {
        let want = &check.rgba[row * unpadded..(row + 1) * unpadded];
        let got = &mapped[row * padded..row * padded + unpadded];
        if want != got {
            let column = want.iter().zip(got).position(|(a, b)| a != b).unwrap_or(0) / 4;
            mismatch = Some((column as u32, row as u32));
            break;
        }
    }
    drop(mapped);
    readback.unmap();

    match mismatch {
        None => {
            println!("[headless-smoke] GPU blit matches the host framebuffer");
            Ok(true)
        }
        Some((x, y)) => Err(format!("GPU blit mismatch at pixel ({x}, {y})")),
    }
}

/// Seeds the headless smoke canvas with a stone floor, a water pool, a falling
/// dust column and one rigid body, so particles, air, heat and rigid-body
/// subsystems all produce non-trivial telemetry.
fn seed_smoke_scene(state: &mut AppState, width: u32, height: u32) {
    let width = width as i32;
    let height = height as i32;
    if width <= 0 || height <= 0 {
        return;
    }

    // Stone floor across the bottom of the canvas.
    let floor_top = (height - 12).max(0);
    let mut floor = Vec::new();
    for y in floor_top..height {
        for x in 0..width {
            floor.push((x, y));
        }
    }
    state.engine.apply_command(SimCommand::PlaceParticles {
        points: floor,
        elem_ty: PT_STNE,
        replace_mode: true,
        dcolour: 0,
    });

    // Water pool resting on the floor.
    let pool_top = (height - 40).max(0);
    let mut water = Vec::new();
    for y in pool_top..floor_top {
        for x in 8..(width - 8).max(9) {
            water.push((x, y));
        }
    }
    state.engine.apply_command(SimCommand::PlaceParticles {
        points: water,
        elem_ty: PT_WATR,
        replace_mode: true,
        dcolour: 0,
    });

    // Dust column that falls into the pool.
    let mut dust = Vec::new();
    for y in 8..(40.min(height / 2)) {
        for x in (width / 2 - 8)..(width / 2 + 8) {
            dust.push((x, y));
        }
    }
    state.engine.apply_command(SimCommand::PlaceParticles {
        points: dust,
        elem_ty: PT_DUST,
        replace_mode: false,
        dcolour: 0,
    });

    // One rigid box dropped from the top so the rigid-body engine is exercised.
    state.engine.apply_command(SimCommand::SpawnRigidBody {
        x: width as f32 * 0.5,
        y: 60.0,
        half_w: 10,
        half_h: 6,
        shape: RigidSpawnShape::Box,
        elem_ty: PT_STNE,
    });
}

/// Runs the `--save-smoke` verification path.
fn run_save_smoke(side: u32) -> i32 {
    match save_smoke_inner(side) {
        Ok(()) => {
            println!("[save-smoke] PASS");
            0
        }
        Err(err) => {
            eprintln!("[save-smoke] FAIL: {err}");
            1
        }
    }
}

/// Saves and reloads a `side`x`side` canvas through the application save path.
///
/// The application path is what the GUI uses (`AppState::save_current` ->
/// `SaveData::capture_from` -> `save_to_path`), so a canvas-size dependent
/// fault in the capture read-back, the serializer or the restore shows up here
/// instead of in the window.
fn save_smoke_inner(side: u32) -> Result<(), String> {
    let gpu = create_headless_gpu("tpt-save-smoke-device")?;
    println!(
        "[save-smoke] adapter: {} ({:?}, backend {:?})",
        gpu.adapter_info.name, gpu.adapter_info.device_type, gpu.adapter_info.backend
    );

    let mut state = AppState::new(gpu.client.clone());
    let (width, height) = state.start_new_canvas(side, side);
    println!("[save-smoke] canvas: {width}x{height}");
    seed_smoke_scene(&mut state, width, height);
    // A dense scene so the pool grows towards its cap and the read-back covers
    // a full device snapshot, which is the shape the GUI produces after a
    // screen-filling brush stroke.
    let mut dense = Vec::new();
    for y in 0..(height as i32 / 2) {
        for x in 0..width as i32 {
            dense.push((x, y));
        }
    }
    state.engine.apply_command(SimCommand::PlaceParticles {
        points: dense,
        elem_ty: tpt_core::constants::PT_SAND,
        replace_mode: true,
        dcolour: 0,
    });
    state.push_undo_snapshot();
    for _ in 0..SAVE_SMOKE_TICKS {
        state.step_simulation_once();
    }
    // The GPU render chain dispatches over the saturated pool, so it must run
    // here too: the interactive viewport is the only thing that used to do it.
    state.engine.render_gpu();

    let saved_count = state.engine.particle_count();
    println!("[save-smoke] particles before save: {saved_count}");
    if saved_count == 0 {
        return Err(String::from("save smoke scene seeded no particles"));
    }

    let message = state.save_current("save-smoke")?;
    println!("[save-smoke] {message}");
    if !state.undo.can_undo() {
        return Err(String::from(
            "the undo snapshot taken before the save is gone",
        ));
    }

    let path = std::path::Path::new(app_state::SAVES_DIR)
        .join(format!("save-smoke.{}", app_state::SAVE_EXTENSION));
    let (loaded_w, loaded_h) = state.load_from_file(&path)?;
    let loaded_count = state.engine.particle_count();
    println!("[save-smoke] reloaded: {loaded_w}x{loaded_h}, {loaded_count} particles");

    let _ = std::fs::remove_file(&path);
    if (loaded_w, loaded_h) != (width, height) {
        return Err(format!(
            "reloaded canvas is {loaded_w}x{loaded_h}, expected {width}x{height}"
        ));
    }
    if loaded_count == 0 {
        return Err(String::from("reloaded scene has no particles"));
    }
    Ok(())
}

/// Samples the aggregate telemetry printed by `--headless-smoke`.
///
/// Returns `(active_particles, active_rigid_bodies, mean_air_pressure, mean_particle_temperature)`.
fn sample_telemetry(bufs: &SimBuffers) -> (usize, usize, f32, f32) {
    (
        bufs.count_active_particles(),
        bufs.rigid_bodies.iter().filter(|body| body.active).count(),
        mean_finite(&bufs.air_pv),
        mean_particle_temperature(bufs),
    )
}

/// Mean temperature of the live particle slots in `bufs`.
///
/// Live slots are not a contiguous prefix (the pool has a free list), so the
/// whole `0..capacity` range is scanned and empty slots (`PT_NONE`) skipped.
fn mean_particle_temperature(bufs: &SimBuffers) -> f32 {
    let slots = bufs
        .capacity
        .min(bufs.part_ty.len())
        .min(bufs.part_temp.len());
    let mut sum = 0.0_f32;
    let mut count = 0_usize;
    for slot in 0..slots {
        let temp = bufs.part_temp[slot];
        if bufs.part_ty[slot] != PT_NONE && temp.is_finite() {
            sum += temp;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

/// Arithmetic mean of the finite values in `values` (`0.0` for empty input).
fn mean_finite(values: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    let mut count = 0_usize;
    for value in values {
        if value.is_finite() {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}
