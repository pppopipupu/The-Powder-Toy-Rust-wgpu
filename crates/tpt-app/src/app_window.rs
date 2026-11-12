//! Window creation, frame presentation and the `winit` application handler.
//!
//! [`PowderToyApp`] owns the window, the shared GPU context and the simulation
//! state. [`PowderToyApp::redraw`] advances the simulation, builds the egui UI
//! and presents the result. The per-frame stages live in [`frame`] and the
//! monitor geometry in [`geometry`], so the application handler only owns the
//! window, event and frame-sequence bookkeeping.

use std::sync::Arc;
use std::time::Instant;

use tpt_core::constants::{DEFAULT_XRES, DEFAULT_YRES, PT_SAND, PT_WATR};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::window::{Window, WindowId};

use crate::app_state::{AppScreen, AppState};
use crate::gpu_ctx::GpuContext;
use crate::seed_smoke_scene;
use crate::ui_panels;
use crate::{HEADLESS_CANVAS, WINDOW_TITLE};

mod frame;
mod geometry;

use self::frame::{
    PresentFrame, present_frame, render_simulation_frame, resize_sim_texture, resize_voxel_view,
    upload_egui_textures,
};
use self::geometry::{clamp_window_geometry, clamp_window_size, monitor_rects};

/// Initial logical window width in points.
const INITIAL_WINDOW_WIDTH: f64 = 1280.0;
/// Initial logical window height in points.
const INITIAL_WINDOW_HEIGHT: f64 = 800.0;
/// Smallest logical window size that still fits the HUD.
const MIN_WINDOW_WIDTH: f64 = 800.0;
/// Smallest logical window height that still fits the HUD.
const MIN_WINDOW_HEIGHT: f64 = 560.0;

/// The `winit` application handler that owns the window, GPU context and
/// simulation state.
pub(crate) struct PowderToyApp {
    /// The single application window; `None` until `resumed`.
    window: Option<Arc<Window>>,
    /// wgpu/CubeCL/egui renderer bundle; `None` until `resumed`.
    gpu: Option<GpuContext>,
    /// Simulation, undo history, Lua runtime and HUD state.
    state: Option<AppState>,
    /// Shared egui context (survives window recreation).
    egui_ctx: egui::Context,
    /// Bridges winit events/raw input with the egui context.
    egui_winit: Option<egui_winit::State>,
    /// Last time a frame was presented; drives the draw-rate cap.
    last_present_instant: Instant,
    /// `--windowed-smoke` frame budget; `None` in normal interactive use.
    smoke_ticks: Option<u32>,
    /// Frames presented so far by `--windowed-smoke`.
    smoke_frames: u32,
    /// Set once the smoke budget is exhausted; `about_to_wait` exits the loop.
    smoke_done: bool,
    /// Whether the smoke result line has already been printed.
    smoke_reported: bool,
    /// `--voxel-windowed-smoke`: seed the 3D sandbox instead of the 2D scene.
    voxel_smoke: bool,
}

impl PowderToyApp {
    /// Prints a row of pixels from the presented 3D texture.
    ///
    /// `--voxel-windowed-smoke` otherwise proves only that the frame path runs;
    /// the sampled row catches a renderer that presents a clear/sky texture.
    fn report_voxel_smoke_pixels(&self) {
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        let (width, height) = gpu.view3d_size;
        if width < 4 || height < 4 {
            println!("[voxel-windowed-smoke] texture {width}x{height}");
            return;
        }
        let rows = 8u32;
        let y0 = (height * 3 / 4).saturating_sub(rows / 2);
        let bytes_per_row = (width * 4).div_ceil(256) * 256;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tpt-voxel-smoke-readback"),
            size: u64::from(bytes_per_row) * u64::from(rows),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tpt-voxel-smoke-readback-encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &gpu.view3d_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: y0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows),
                },
            },
            wgpu::Extent3d {
                width,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        let _ = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let Ok(Ok(())) = rx.recv() else {
            println!("[voxel-windowed-smoke] readback failed");
            return;
        };
        let data = buffer.slice(..).get_mapped_range().expect("mapped row");
        let centre = ((width / 2) * 4) as usize;
        for row in 0..rows {
            let offset = row as usize * bytes_per_row as usize + centre;
            let pixel = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            println!("[voxel-windowed-smoke] row {} pixel {pixel:?}", y0 + row);
        }
        drop(data);
        buffer.unmap();
    }

    pub(crate) fn new(smoke_ticks: Option<u32>, voxel_smoke: bool) -> Self {
        let egui_ctx = egui::Context::default();
        Self::setup_custom_fonts(&egui_ctx);
        egui_ctx.set_theme(egui::ThemePreference::Dark);
        egui_ctx.set_visuals(egui::Visuals::dark());

        Self {
            window: None,
            gpu: None,
            state: None,
            egui_ctx,
            egui_winit: None,
            last_present_instant: Instant::now(),
            smoke_ticks,
            smoke_frames: 0,
            smoke_done: false,
            smoke_reported: false,
            voxel_smoke,
        }
    }

    /// Configures Maple Mono Medium as the primary proportional and monospace font.
    fn setup_custom_fonts(ctx: &egui::Context) {
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "maple_mono_medium".to_owned(),
            Arc::new(egui::FontData::from_static(include_bytes!(
                "../assets/fonts/MapleMono-NF-CN-Medium.ttf"
            ))),
        );
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, "maple_mono_medium".to_owned());
        fonts
            .families
            .entry(egui::FontFamily::Monospace)
            .or_default()
            .insert(0, "maple_mono_medium".to_owned());
        ctx.set_fonts(fonts);
    }

    /// Creates the window, GPU context, simulation state and egui integration.
    ///
    /// Called from [`ApplicationHandler::resumed`]; it is idempotent so a
    /// second `resumed` (some platforms emit several) does not rebuild
    /// anything.
    fn initialize(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // `OptionsView.cpp:366-371`: the Resizable and Fullscreen preferences
        // describe the window itself, so they are read here; the live values
        // reach the UI later through `AppState::prefs`.
        let window_prefs = crate::prefs::Prefs::load();

        // `PowderToy.cpp:44-45`/`PowderToySDL.cpp:212-216`: restore the saved
        // window rectangle, clamped to a monitor that is actually attached now.
        let monitors = monitor_rects(event_loop);
        let saved_position = match (window_prefs.window_x, window_prefs.window_y) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        };
        let saved_size = match (window_prefs.window_width, window_prefs.window_height) {
            (Some(width), Some(height)) => Some((width, height)),
            _ => None,
        };
        let (restored_position, restored_size) = if window_prefs.fullscreen {
            // Fullscreen owns the window rectangle; the windowed geometry is
            // kept in prefs for when fullscreen is turned off again.
            (None, None)
        } else if let Some(position) = saved_position {
            let size =
                saved_size.unwrap_or((INITIAL_WINDOW_WIDTH as u32, INITIAL_WINDOW_HEIGHT as u32));
            match clamp_window_geometry(position, size, &monitors) {
                Some(geometry) => (Some(geometry.position), Some(geometry.size)),
                None => (Some(position), Some(size)),
            }
        } else {
            (
                None,
                saved_size.map(|size| clamp_window_size(size, &monitors)),
            )
        };

        let mut attributes = Window::default_attributes()
            .with_title(WINDOW_TITLE)
            .with_resizable(window_prefs.resizable)
            .with_fullscreen(if window_prefs.fullscreen {
                Some(winit::window::Fullscreen::Borderless(None))
            } else {
                None
            })
            .with_inner_size(LogicalSize::new(
                INITIAL_WINDOW_WIDTH,
                INITIAL_WINDOW_HEIGHT,
            ))
            .with_min_inner_size(LogicalSize::new(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT));
        if let Some((width, height)) = restored_size {
            attributes = attributes.with_inner_size(PhysicalSize::new(width, height));
        }
        if let Some((x, y)) = restored_position {
            attributes = attributes.with_position(PhysicalPosition::new(x, y));
        }

        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                eprintln!("[tpt-app] failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        let gpu = GpuContext::new(window.clone(), DEFAULT_XRES, DEFAULT_YRES);
        println!("[tpt-app] GPU: {}", gpu.adapter_summary());

        let mut state = AppState::new(gpu.cubecl_client.clone());

        if let Some(frames) = self.smoke_ticks {
            if self.voxel_smoke {
                // Exercise the dedicated 3D texture, the voxel renderer and
                // the 3D UI branch in the real window path. A floor plus a
                // water and sand blob gives the readback a non-sky sample.
                state.enter_voxel_mode();
                state.paint_voxel_floor();
                let engine = state.ensure_voxel_engine();
                engine.place_sphere(PT_WATR, [128.0, 24.0, 128.0], 12.0);
                engine.place_sphere(PT_SAND, [128.0, 52.0, 128.0], 9.0);
                println!("[voxel-windowed-smoke] 256^3 world, {frames} frames");
            } else {
                // Enter the game screen with the same deterministic scene the
                // headless smoke uses, so the windowed path exercises the sim
                // texture registration, the blit and the egui composition.
                let (width, height) = state.start_new_canvas(HEADLESS_CANVAS.0, HEADLESS_CANVAS.1);
                state.screen = AppScreen::InGame;
                seed_smoke_scene(&mut state, width, height);
                println!("[windowed-smoke] canvas: {width}x{height}, {frames} frames");
            }
        }

        let max_texture_side = gpu.device.limits().max_texture_dimension_2d as usize;

        let egui_winit = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            Some(max_texture_side),
        );

        self.egui_winit = Some(egui_winit);
        self.gpu = Some(gpu);
        self.state = Some(state);
        self.window = Some(window.clone());

        window.request_redraw();
    }

    /// Renders one complete frame: advances the simulation, blits the
    /// framebuffer on the device, runs the egui UI, and presents the result.
    fn redraw(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let (Some(gpu), Some(state)) = (self.gpu.as_mut(), self.state.as_mut()) else {
            return;
        };
        let Some(egui_winit) = self.egui_winit.as_mut() else {
            return;
        };

        // Count every redraw toward the `--windowed-smoke` budget, including
        // frames the swap chain skips while the window is occluded.
        if let Some(limit) = self.smoke_ticks {
            self.smoke_frames = self.smoke_frames.saturating_add(1);
            if self.smoke_frames >= limit {
                self.smoke_done = true;
            }
        }

        if state.screen == AppScreen::InGame {
            state.tick_frame();
        }
        let sim_tex_id = gpu.sim_tex_id;
        let view3d_tex_id = gpu.view3d_tex_id;
        let raw_input = egui_winit.take_egui_input(&window);
        let egui_ctx = self.egui_ctx.clone();
        let mut requested_sim_size: Option<(u32, u32)> = None;
        let mut full_output = egui_ctx.run_pass(raw_input, |ctx| {
            if let Some(size) = ui_panels::draw_app_ui(ctx, &mut *state, sim_tex_id, view3d_tex_id)
            {
                requested_sim_size = Some(size);
            }
        });
        resize_sim_texture(gpu, state, requested_sim_size);
        resize_voxel_view(gpu, state);
        render_simulation_frame(state);

        egui_winit.handle_platform_output(&window, full_output.platform_output);

        let pixels_per_point = full_output.pixels_per_point;
        let clipped_primitives = egui_ctx.tessellate(full_output.shapes, pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.surface_config.width, gpu.surface_config.height],
            pixels_per_point,
        };
        upload_egui_textures(gpu, &full_output.textures_delta);

        // egui wants the `free` half applied *after* painting, but the delta
        // must not be dropped unhandled: `TexturesDelta::drop` asserts on that
        // in debug builds, and the swap-chain acquisition below can return
        // early. Take the ids out now and hand the delta back as handled.
        let freed_textures: Vec<egui::TextureId> =
            full_output.textures_delta.free.iter().copied().collect();
        full_output.textures_delta.clear();

        present_frame(
            gpu,
            state,
            &window,
            PresentFrame {
                clipped_primitives: &clipped_primitives,
                screen_descriptor,
                freed_textures: &freed_textures,
            },
        );
        self.last_present_instant = Instant::now();
        state.sample_presented_frame();
    }
}

impl ApplicationHandler for PowderToyApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.initialize(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.clone() else {
            return;
        };
        if window.id() != window_id {
            return;
        }

        // egui gets first look at every event (keyboard, pointer, IME, ...).
        if let Some(egui_winit) = self.egui_winit.as_mut() {
            let _ = egui_winit.on_window_event(&window, &event);
        }

        match event {
            WindowEvent::CloseRequested => {
                // The port closes immediately: TPT's `Engine::ConfirmExit`
                // prompt (`Engine.cpp:57-68`) is not carried over.
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize_surface(size.width, size.height);
                }
                window.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                // The new size is only available after the event has been
                // handled, so query the window instead of the event payload.
                let size = window.inner_size();
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize_surface(size.width, size.height);
                }
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // `Engine::Exit()`: the confirmation prompt sets the flag during the
        // egui pass, and the loop always reaches this hook afterwards.
        if self
            .state
            .as_ref()
            .is_some_and(|state| state.exit_requested)
        {
            event_loop.exit();
            return;
        }
        if self.smoke_ticks.is_some() {
            // Some automation environments never deliver `RedrawRequested`
            // (the window is created on a non-interactive window station), so
            // the smoke mode drives frames directly instead of waiting for
            // WM_PAINT. Interactive use keeps the normal redraw path.
            if self.smoke_done {
                if !self.smoke_reported {
                    self.smoke_reported = true;
                    if self.voxel_smoke {
                        self.report_voxel_smoke_pixels();
                        println!("[voxel-windowed-smoke] PASS ({} frames)", self.smoke_frames);
                    } else {
                        println!("[windowed-smoke] PASS ({} frames)", self.smoke_frames);
                    }
                }
                event_loop.exit();
                return;
            }
            // The smoke drives its own frames, so it keeps the loop spinning
            // even though interactive use waits for the next draw deadline.
            event_loop.set_control_flow(ControlFlow::Poll);
            self.redraw();
            return;
        }
        let Some(window) = &self.window else {
            return;
        };
        let Some(state) = self.state.as_mut() else {
            window.request_redraw();
            return;
        };
        // A console script on the main menu (where `tick_frame` does not run)
        // or a key/mouse handler may have changed the caps; fold the change
        // into `prefs.json` before pacing the loop.
        state.persist_script_caps();
        // Sleep until the next frame is due instead of spinning: the draw cap
        // is at most 240 fps, and a `ControlFlow::Poll` loop burned a whole
        // core re-running `redraw` for the ~40 iterations between frames. Input
        // events still wake the loop and are handled as they arrive.
        // Wake for whichever comes first: the next presented frame or the next
        // simulation step. A draw cap below the tick rate must not slow the
        // simulation down, and the step itself is paced inside `tick_frame`.
        let draw_interval = std::time::Duration::from_secs_f32(
            crate::prefs::draw_interval_secs(state.draw_fps()),
        );
        let draw_due = self.last_present_instant + draw_interval;
        let sim_due = state.next_sim_tick_due();
        let now = Instant::now();
        if now >= draw_due {
            window.request_redraw();
            let wake = sim_due.map_or(now + draw_interval, |t| t.min(now + draw_interval));
            event_loop.set_control_flow(ControlFlow::WaitUntil(wake));
        } else if let Some(tick_at) = sim_due {
            if now >= tick_at {
                // Simulation target exceeds the render cap: advance ticks here
                // without spinning `WM_PAINT` before the next draw deadline.
                state.tick_frame();
                let next_tick = state.next_sim_tick_due().unwrap_or(draw_due);
                event_loop.set_control_flow(ControlFlow::WaitUntil(draw_due.min(next_tick)));
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(draw_due.min(tick_at)));
            }
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(draw_due));
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // `SaveWindowPosition` (`PowderToy.cpp:72-83`): remember the window
        // rectangle before the window is destroyed. The inner size is the
        // resizable-window equivalent of `PowderToySDL.cpp:212-216`.
        self.save_window_geometry();
        // A cap changed by a script that never ran a tick (for example from the
        // console on the main menu) still has to reach `prefs.json`.
        if let Some(state) = self.state.as_mut() {
            state.persist_script_caps();
            // A mode change made by loading a canvas or a save still has to
            // reach `prefs.json`; explicit toggles already saved it.
            if state.prefs.game_mode != state.mode {
                state.prefs.game_mode = state.mode;
                state.prefs.save();
            }
        }
        // Drop the simulation before the GPU context: the engine holds CubeCL
        // buffers allocated on that device.
        self.state = None;
        self.egui_winit = None;
        self.gpu = None;
        self.window = None;
    }
}

impl PowderToyApp {
    /// Saves the live window rectangle into `prefs.json`.
    ///
    /// Skipped for smoke runs (no user geometry to remember) and while
    /// fullscreen (the fullscreen screen rect would replace the windowed
    /// rectangle the user actually wants restored).
    fn save_window_geometry(&mut self) {
        if self.smoke_ticks.is_some() {
            return;
        }
        let Some(window) = self.window.clone() else {
            return;
        };
        if window.fullscreen().is_some() {
            return;
        }
        let Some(state) = self.state.as_mut() else {
            return;
        };
        if let Ok(position) = window.outer_position() {
            state.prefs.window_x = Some(position.x);
            state.prefs.window_y = Some(position.y);
        }
        let size = window.inner_size();
        // `PowderToySDL.cpp:287-291`: a maximised window ignores
        // `SDL_SetWindowSize`; saving the maximised inner size would restore a
        // huge normal window next time, so keep the previous windowed size.
        if !window.is_maximized() && size.width != 0 && size.height != 0 {
            state.prefs.window_width = Some(size.width);
            state.prefs.window_height = Some(size.height);
        }
        state.prefs.save();
    }
}
