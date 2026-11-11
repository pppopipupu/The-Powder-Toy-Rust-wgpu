//! Per-frame simulation, render and present stages.
//!
//! `PowderToyApp::redraw` owns the frame sequence; the device-side stages live
//! here so the application handler stays about window/event bookkeeping.

use std::time::Instant;

use winit::window::Window;

use crate::app_state::{AppScreen, AppState, GameMode};
use crate::gpu_ctx::GpuContext;
use crate::render3d::RigidRenderBuffers;

/// Advances and renders the simulation, honouring the render framerate cap.
///
/// The cap is hard-locked at 240 fps (`prefs::MAX_SANE_DRAW_FPS`) by
/// `prefs::draw_interval_secs`: rendering and presenting are skipped until the
/// interval has elapsed, so the present rate never exceeds 240 regardless of
/// the preference. The simulation itself keeps stepping on its own cap.
///
/// Returns `false` when the frame was skipped; the caller must then return
/// without building or presenting a frame (a redraw is already requested).
pub(super) fn render_simulation(
    window: &Window,
    state: &mut AppState,
    last_present: &mut Instant,
) -> bool {
    if state.screen != AppScreen::InGame {
        // The title screen is drawn at the same cap as the game, so the pacer
        // in `about_to_wait` has a deadline to wait for.
        *last_present = Instant::now();
        return true;
    }

    state.tick_frame();

    let draw_due = last_present.elapsed().as_secs_f32()
        >= crate::prefs::draw_interval_secs(state.draw_fps()) * 0.98;
    if !draw_due {
        window.request_redraw();
        return false;
    }
    *last_present = Instant::now();
    state.sample_presented_frame();
    // The 3D path renders through `Renderer3d` in `present_frame`; the 2D
    // software/GPU powder renderer must never run while the sandbox is active.
    if state.mode == GameMode::Powder2D {
        state.engine.render_gpu();
    }
    true
}

/// Reallocates the simulation texture when the UI changed the canvas size.
///
/// The fresh texture is blank, so the current simulation is rendered and
/// blitted into it straight away. If the GPU blit cannot be encoded, the
/// update is skipped (the fresh texture stays blank for this frame) and a
/// redraw is requested so the next `present_frame` retries the blit; the
/// framebuffer is never read back to the host on this path.
pub(super) fn resize_sim_texture(
    window: &Window,
    gpu: &mut GpuContext,
    state: &mut AppState,
    requested: Option<(u32, u32)>,
) {
    // The 3D viewport never reports a simulation resize, and resizing the
    // powder texture while the sandbox is active would only discard its last
    // frame.
    if state.mode != GameMode::Powder2D {
        return;
    }
    let Some((width, height)) = requested else {
        return;
    };
    if gpu.sim_tex_size == (width.max(1), height.max(1)) {
        return;
    }

    gpu.resize_sim_texture(width, height);
    if state.screen != AppScreen::InGame {
        return;
    }

    state.engine.render_gpu();
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tpt-sim-resize-blit"),
        });
    let blitted = gpu.encode_sim_blit(&mut encoder, &state.engine.bufs);
    gpu.queue.submit(Some(encoder.finish()));
    if !blitted {
        // Never read the framebuffer back to the host here: that is a
        // full-canvas GPU->CPU->GPU round-trip on the interactive path. The
        // freshly allocated texture keeps its blank contents for this frame
        // and the next `present_frame` retries the blit.
        window.request_redraw();
    }
}

/// Creates the 3D renderer on first use and matches the texture to its target.
///
/// `Renderer3d::resize` applies the persisted render scale, so the render
/// target can be smaller than the viewport; the texture always follows
/// `target_size`, never the raw viewport.
pub(super) fn resize_voxel_view(gpu: &mut GpuContext, state: &mut AppState) {
    if state.mode != GameMode::Sandbox3D || state.screen != AppScreen::InGame {
        return;
    }
    state.ensure_voxel_renderer(&gpu.device);
    let render_scale = state.voxel_render_scale();
    let Some(renderer) = state.voxel_renderer.as_mut() else {
        return;
    };
    if let Some((width, height)) = state.voxel_viewport_size {
        renderer.resize(&gpu.device, width, height, render_scale);
    }
    let (width, height) = renderer.target_size();
    gpu.resize_view3d_texture(width, height);
}

/// Uploads this frame's egui font/icon texture deltas.
pub(super) fn upload_egui_textures(gpu: &mut GpuContext, deltas: &egui::TexturesDelta) {
    for (id, images) in &deltas.set {
        for image_delta in images {
            gpu.egui_renderer
                .update_texture(&gpu.device, &gpu.queue, *id, image_delta);
        }
    }
}

/// Inputs of [`present_frame`] that belong to the egui frame rather than the
/// GPU context.
pub(super) struct PresentFrame<'a> {
    /// Tessellated egui draw data for this frame.
    pub(super) clipped_primitives: &'a [egui::ClippedPrimitive],
    /// Surface size and pixels-per-point for the render pass.
    pub(super) screen_descriptor: egui_wgpu::ScreenDescriptor,
    /// Textures to release after painting.
    pub(super) freed_textures: &'a [egui::TextureId],
}

/// Publishes the simulation framebuffer, paints the egui pass and presents.
///
/// The framebuffer is blitted into [`GpuContext::sim_texture`] on the device.
/// If that blit cannot be encoded the texture keeps its previous contents for
/// this frame and a redraw is requested; the framebuffer is never read back to
/// the host on this path.
pub(super) fn present_frame(
    gpu: &mut GpuContext,
    state: &mut AppState,
    window: &Window,
    frame: PresentFrame<'_>,
) {
    // `OptionsView.cpp:298` (`BlurryScaling`): switch the simulation sampler
    // between nearest-neighbour pixels and a linear upscale. The setter is a
    // no-op unless the preference changed, so checking it every frame is free.
    gpu.set_sim_filter(if state.prefs.blurry_scaling {
        wgpu::FilterMode::Linear
    } else {
        wgpu::FilterMode::Nearest
    });

    // Keep the simulation texture in sync with the engine canvas. The UI only
    // reports a resize when one of its actions changed the canvas, so a canvas
    // created outside the UI (save restore, smoke seeding) would otherwise
    // leave a stale texture: the blit would be rejected for its mismatched
    // byte range and the viewport would present pure black.
    if state.mode == GameMode::Powder2D && state.screen == AppScreen::InGame {
        let (width, height) = (state.engine.config().xres, state.engine.config().yres);
        if gpu.sim_tex_size != (width.max(1), height.max(1)) {
            gpu.resize_sim_texture(width, height);
        }
    }

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tpt-frame-encoder"),
        });
    let staging_buffers = gpu.egui_renderer.update_buffers(
        &gpu.device,
        &gpu.queue,
        &mut encoder,
        frame.clipped_primitives,
        &frame.screen_descriptor,
    );
    // Encode the 3D world before the egui pass; egui then samples the texture
    // with `view3d_tex_id`. The renderer binds the solver's own rigid-body
    // handles, so no per-frame buffer copy is needed.
    if state.mode == GameMode::Sandbox3D && state.screen == AppScreen::InGame {
        let camera = state.voxel_camera;
        if let (Some(renderer), Some(voxel)) =
            (state.voxel_renderer.as_mut(), state.voxel3d.as_ref())
        {
            renderer.set_rigid_buffers(Some(RigidRenderBuffers {
                body_desc: voxel.rigid.body_desc.clone(),
                body_voxels: voxel.rigid.body_voxels.clone(),
                body_masks: voxel.rigid.body_masks.clone(),
                body_count: voxel.rigid.body_count,
            }));
            renderer.encode(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &gpu.view3d_texture_view,
                &voxel.bufs,
                &camera,
            );
        }
    }
    if state.mode == GameMode::Powder2D
        && state.screen == AppScreen::InGame
        && !gpu.encode_sim_blit(&mut encoder, &state.engine.bufs)
    {
        // `encode_sim_blit` returns `false` when the framebuffer cannot be
        // bound directly. Skip this frame's sim-texture update — the texture
        // keeps its previous contents — instead of reading the whole
        // framebuffer back to the host, which would stall the device on the
        // interactive hot path. Ask for another redraw so the blit is retried.
        window.request_redraw();
    }
    let Some((surface_texture, reconfigure_after_present)) =
        acquire_surface_texture(gpu, window, frame.freed_textures)
    else {
        return;
    };

    let view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    {
        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tpt-egui-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.043,
                        g: 0.051,
                        b: 0.063,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        gpu.egui_renderer.render(
            &mut render_pass.forget_lifetime(),
            frame.clipped_primitives,
            &frame.screen_descriptor,
        );
    }

    gpu.queue.submit(
        staging_buffers
            .into_iter()
            .chain(std::iter::once(encoder.finish())),
    );
    gpu.queue.present(surface_texture);

    if reconfigure_after_present {
        // Same reasoning as `GpuContext::resize_surface`: drain pending
        // GPU work before wgpu's one-second idle check inside configure.
        let _ = gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        gpu.surface.configure(&gpu.device, &gpu.surface_config);
    }

    free_egui_textures(&mut gpu.egui_renderer, frame.freed_textures);
}

/// Acquires the swap-chain texture, reconfiguring on stale surfaces.
///
/// Returns the texture plus whether the surface must be reconfigured after
/// presenting. `None` means the frame was skipped: the caller's textures have
/// already been freed and a redraw was requested.
fn acquire_surface_texture(
    gpu: &mut GpuContext,
    window: &Window,
    freed_textures: &[egui::TextureId],
) -> Option<(wgpu::SurfaceTexture, bool)> {
    match gpu.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture) => Some((texture, false)),
        wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Some((texture, true)),
        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
            gpu.surface.configure(&gpu.device, &gpu.surface_config);
            free_egui_textures(&mut gpu.egui_renderer, freed_textures);
            window.request_redraw();
            None
        }
        wgpu::CurrentSurfaceTexture::Timeout
        | wgpu::CurrentSurfaceTexture::Occluded
        | wgpu::CurrentSurfaceTexture::Validation => {
            free_egui_textures(&mut gpu.egui_renderer, freed_textures);
            window.request_redraw();
            None
        }
    }
}

/// Releases the egui textures a frame asked to free.
///
/// Shared by the normal post-present path and the swap-chain early returns so a
/// skipped frame cannot leak a texture.
fn free_egui_textures(renderer: &mut egui_wgpu::Renderer, ids: &[egui::TextureId]) {
    for id in ids {
        renderer.free_texture(id);
    }
}
