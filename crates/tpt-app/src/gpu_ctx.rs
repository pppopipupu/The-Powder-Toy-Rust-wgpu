//! Shared GPU context for the whole application.
//!
//! [`GpuContext`] owns the single `wgpu` instance/adapter/device/queue triple
//! used by *both* the simulation (through CubeCL) and the presentation layer
//! (`egui-wgpu`). Sharing the device is what lets the simulation render
//! straight into a texture that egui can sample without any CPU round-trip or
//! cross-device copy.
//!
//! The simulation output texture is registered with the egui renderer once and
//! keeps a stable [`egui::TextureId`] for the lifetime of the context, even
//! when the canvas is resized (see [`GpuContext::resize_sim_texture`]).
//!
//! The 3D sandbox uses a second registered texture
//! ([`GpuContext::view3d_texture`]) that `Renderer3d` renders into; it follows
//! the renderer's (possibly scaled) target size through
//! [`GpuContext::resize_view3d_texture`].

use std::sync::Arc;

use winit::window::Window;

// Screenshots and APNG recordings live in `src/screenshots.rs`. The module is
// declared here, next to the GPU context that owns the only read-back path,
// because `main.rs` deliberately keeps its top-level module list stable.
#[path = "screenshots.rs"]
pub(crate) mod screenshots;

/// Simulation output texture format.
///
/// `Rgba8Unorm` (not the `*Srgb` variant) is deliberate: the simulation
/// render kernel writes display-ready sRGB bytes, so the texture must not apply
/// a second sRGB decode on sampling.
pub const SIM_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Bytes per pixel of [`SIM_TEXTURE_FORMAT`].
pub const SIM_BYTES_PER_PIXEL: u32 = 4;

/// Preferred swap-chain formats, in order of preference.
///
/// Non-sRGB `Unorm` formats are preferred so that `egui-wgpu` performs its own
/// (correct, perceptual) blending in the shader instead of relying on hardware
/// sRGB blending.
const PREFERRED_SURFACE_FORMATS: [wgpu::TextureFormat; 2] = [
    wgpu::TextureFormat::Bgra8Unorm,
    wgpu::TextureFormat::Rgba8Unorm,
];

/// Compute shader that expands the packed simulation framebuffer into RGBA8.
///
/// The CubeCL render kernels write `0xAABBGGRR` words; on a little-endian host
/// the byte order is already `[R, G, B, A]`, which is exactly what
/// `Rgba8Unorm` stores. The shader unpacks the word and stores the four bytes
/// as normalised floats, so the texture bytes stay bit-identical to the packed
/// host framebuffer that `--headless-smoke` compares against.
const FRAMEBUF_BLIT_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let word = src[gid.y * dims.x + gid.x];
    let r = f32(word & 0xFFu) / 255.0;
    let g = f32((word >> 8u) & 0xFFu) / 255.0;
    let b = f32((word >> 16u) & 0xFFu) / 255.0;
    let a = f32((word >> 24u) & 0xFFu) / 255.0;
    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(r, g, b, a));
}
"#;

/// GPU-side copy of the packed simulation framebuffer into the egui texture.
///
/// The CubeCL render kernels write `0xAABBGGRR` words into a storage buffer;
/// this pass unpacks them into an `Rgba8Unorm` storage texture so the pixel
/// data never crosses the PCIe bus. Exposed separately from [`GpuContext`] so
/// the headless verification path can exercise the exact same pipeline.
pub struct FramebufBlit {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl FramebufBlit {
    /// Builds the blit pipeline for the current device.
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tpt-framebuf-blit-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: SIM_TEXTURE_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tpt-framebuf-blit"),
            source: wgpu::ShaderSource::Wgsl(FRAMEBUF_BLIT_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tpt-framebuf-blit"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("tpt-framebuf-blit-pipeline-layout"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                }),
            ),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// Encodes the copy of `bufs.gpu_framebuf` into `view`.
    ///
    /// Returns `false` when the CubeCL allocation cannot be bound directly
    /// (unexpected offset or size), having encoded nothing into `encoder`. The
    /// caller must treat that as a skip-frame signal: keep the previous texture
    /// contents and retry later, never read the framebuffer back to the host.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bufs: &tpt_sim::SimBuffers,
        view: &wgpu::TextureView,
        size: (u32, u32),
    ) -> bool {
        use cubecl_wgpu::{AutoCompiler, WgpuServer};

        let pixels = size.0 as usize * size.1 as usize;
        let bytes = (pixels * SIM_BYTES_PER_PIXEL as usize) as u64;
        let Ok(managed) = bufs
            .client
            .get_resource::<WgpuServer<AutoCompiler>>(bufs.gpu_framebuf.clone())
        else {
            return false;
        };
        let resource = managed.resource();
        if resource.size < bytes || resource.offset % 256 != 0 {
            return false;
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tpt-framebuf-blit-bind-group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &resource.buffer,
                        offset: resource.offset,
                        size: std::num::NonZeroU64::new(bytes),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tpt-framebuf-blit"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        true
    }
}

/// Every GPU object required to run and display a simulation.
///
/// There is exactly one of these per process. It is created in
/// `ApplicationHandler::resumed` once a window exists and is dropped in
/// `ApplicationHandler::exiting`.
pub struct GpuContext {
    /// wgpu entry point (surface creation, adapter enumeration).
    pub instance: wgpu::Instance,
    /// The physical GPU selected for rendering and compute.
    pub adapter: wgpu::Adapter,
    /// Logical device shared by `egui-wgpu`, the simulation renderer and CubeCL.
    pub device: wgpu::Device,
    /// Submission queue for [`Self::device`].
    pub queue: wgpu::Queue,
    /// Presentation surface backed by the application window.
    pub surface: wgpu::Surface<'static>,
    /// Current swap-chain configuration (size, format, present mode).
    pub surface_config: wgpu::SurfaceConfiguration,
    /// egui painter: converts tessellated UI primitives into draw calls.
    pub egui_renderer: egui_wgpu::Renderer,
    /// CubeCL compute client bound to [`Self::device`] / [`Self::queue`].
    pub cubecl_client: cubecl::client::Client,
    /// Compute pass that copies the packed CubeCL framebuffer into
    /// [`Self::sim_texture`] without a host round-trip.
    framebuf_blit: FramebufBlit,
    /// Simulation framebuffer, copied into from the CubeCL framebuffer by
    /// [`Self::encode_sim_blit`] entirely on the device.
    pub sim_texture: wgpu::Texture,
    /// View of [`Self::sim_texture`] handed to `egui-wgpu`.
    pub sim_texture_view: wgpu::TextureView,
    /// Stable egui texture id pointing at [`Self::sim_texture_view`].
    pub sim_tex_id: egui::TextureId,
    /// Size of [`Self::sim_texture`] in simulation pixels.
    pub sim_tex_size: (u32, u32),
    /// Sampler filter the simulation texture is registered with in egui.
    ///
    /// `OptionsView.cpp:298` (`BlurryScaling`) re-registers the texture with
    /// [`wgpu::FilterMode::Linear`]; the default is pixel-accurate `Nearest`.
    pub sim_filter: wgpu::FilterMode,
    /// Render target of the 3D sandbox, sampled by egui as `view3d_tex_id`.
    pub view3d_texture: wgpu::Texture,
    /// View of [`Self::view3d_texture`] handed to `egui-wgpu` and to
    /// [`crate::render3d::Renderer3d::encode`].
    pub view3d_texture_view: wgpu::TextureView,
    /// Stable egui texture id pointing at [`Self::view3d_texture_view`].
    pub view3d_tex_id: egui::TextureId,
    /// Size of [`Self::view3d_texture`] in physical pixels.
    pub view3d_size: (u32, u32),
    /// Screenshot / recording read-back state driven once per rendered frame
    /// by [`Self::encode_sim_blit`].
    capture_pump: screenshots::CapturePump,
}

impl GpuContext {
    /// Creates the GPU context for `window`, sizing the simulation texture to
    /// `init_sim_w` x `init_sim_h` pixels.
    ///
    /// The `wgpu` device and queue are shared with CubeCL through
    /// [`cubecl::wgpu::WgpuSetup`], so simulation kernels and presentation use
    /// the same physical device and no cross-device synchronisation is needed.
    pub fn new(window: Arc<Window>, init_sim_w: u32, init_sim_h: u32) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create a wgpu surface for the application window");

        let adapter = request_high_performance_adapter(&instance, Some(&surface))
            .expect("no suitable GPU adapter found (HighPerformance with a presentable surface)");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tpt-device"),
            // Simulation kernels only need core WGSL (storage buffers + atomics),
            // so no optional/experimental feature is requested. Limits are taken
            // from the adapter because CubeCL binds many storage buffers per
            // kernel (see `max_storage_buffers_per_shader_stage`).
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .expect("failed to create a wgpu device/queue");

        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let capabilities = surface.get_capabilities(&adapter);
        let surface_format = PREFERRED_SURFACE_FORMATS
            .iter()
            .copied()
            .find(|format| capabilities.formats.contains(format))
            .or_else(|| capabilities.formats.first().copied())
            .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
        let alpha_mode = capabilities
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: Vec::new(),
        };
        surface.configure(&device, &surface_config);

        let mut egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_config.format,
            egui_wgpu::RendererOptions::default(),
        );

        let framebuf_blit = FramebufBlit::new(&device);

        // Share the exact same wgpu objects with CubeCL. This is the whole
        // point of owning the context ourselves: the simulation runs on the
        // same device that presents to the swap chain.
        let setup = cubecl::wgpu::WgpuSetup {
            instance: instance.clone(),
            adapter: adapter.clone(),
            device: device.clone(),
            queue: queue.clone(),
            backend: adapter.get_info().backend,
        };
        // `ExclusivePages` is required by the 3D mode's indirect dispatches.
        // The default `Adaptive` allocator sub-allocates small handles from one
        // shared wgpu buffer; wgpu rejects a dispatch that uses the same buffer
        // as both `INDIRECT` and `STORAGE_READ_WRITE`, which aliases
        // `sim_count`, `sim_list` and the per-brick bitsets.
        let cubecl_dev = cubecl::wgpu::init_device(
            setup,
            cubecl::wgpu::RuntimeOptions {
                memory_config: cubecl::wgpu::MemoryConfiguration::ExclusivePages,
                ..Default::default()
            },
        );
        let cubecl_client = {
            use cubecl_runtime::runtime::Runtime;
            cubecl::wgpu::WgpuRuntime::<cubecl::wgpu::AutoCompiler>::client(&cubecl_dev)
        };

        let sim_tex_size = (init_sim_w.max(1), init_sim_h.max(1));
        let (sim_texture, sim_texture_view) = create_sim_texture(&device, sim_tex_size);
        // `BlurryScaling` is applied by `set_sim_filter` once the first frame
        // reads the preference; nearest keeps the pixels crisp in the default.
        let sim_filter = wgpu::FilterMode::Nearest;
        let sim_tex_id =
            egui_renderer.register_native_texture(&device, &sim_texture_view, sim_filter);

        // The 3D target is registered once and re-pointed on resize, exactly
        // like the simulation texture, so the viewport keeps its texture id.
        let view3d_size = (1u32, 1u32);
        let (view3d_texture, view3d_texture_view) = create_view3d_texture(&device, view3d_size);
        let view3d_tex_id = egui_renderer.register_native_texture(
            &device,
            &view3d_texture_view,
            wgpu::FilterMode::Linear,
        );

        Self {
            instance,
            adapter,
            device,
            queue,
            surface,
            surface_config,
            egui_renderer,
            cubecl_client,
            framebuf_blit,
            sim_texture,
            sim_texture_view,
            sim_tex_id,
            sim_tex_size,
            sim_filter,
            view3d_texture,
            view3d_texture_view,
            view3d_tex_id,
            view3d_size,
            capture_pump: screenshots::CapturePump::default(),
        }
    }

    /// Reconfigures the swap chain after a window resize.
    ///
    /// Zero-sized resizes (window minimised or hidden) are ignored because
    /// wgpu rejects a zero-sized surface configuration.
    pub fn resize_surface(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if self.surface_config.width == width && self.surface_config.height == height {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.configure_surface();
    }

    /// Reconfigures the swap chain, after draining every submission.
    ///
    /// `Surface::configure` only waits one second for the device to become
    /// idle; a first-frame kernel compilation easily exceeds that and wgpu
    /// then reports `GpuWaitTimeout` through the device error callback (which
    /// CubeCL turns into a panic). Waiting without a timeout first makes the
    /// internal check trivially pass, so the reconfigure cannot race a long
    /// shader-compile submission.
    fn configure_surface(&self) {
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        self.surface.configure(&self.device, &self.surface_config);
    }

    /// Reallocates the simulation texture to `sim_w` x `sim_h` pixels and
    /// re-points the existing egui texture id at the new view.
    ///
    /// Called when the user starts a new canvas or loads a save with a
    /// different resolution.
    pub fn resize_sim_texture(&mut self, sim_w: u32, sim_h: u32) {
        let sim_w = sim_w.max(1);
        let sim_h = sim_h.max(1);
        if self.sim_tex_size == (sim_w, sim_h) {
            return;
        }

        let (texture, view) = create_sim_texture(&self.device, (sim_w, sim_h));
        self.sim_texture = texture;
        self.sim_texture_view = view;
        self.sim_tex_size = (sim_w, sim_h);
        // The new texture is blank until the next blit. Drop any token left
        // by a frame that never reached `encode_sim_blit` so the resize
        // helper's own blit cannot capture it.
        self.capture_pump.cancel_frame();

        // Keep the same `TextureId` so every egui widget that already captured
        // it (the central canvas) keeps working across canvas resizes.
        self.egui_renderer.update_egui_texture_from_wgpu_texture(
            &self.device,
            &self.sim_texture_view,
            self.sim_filter,
            self.sim_tex_id,
        );
    }

    /// Applies the `BlurryScaling` preference to the simulation texture.
    ///
    /// The egui sampler lives in the registered bind group, so switching
    /// between crisp pixels and a smoothed upscale means re-registering the
    /// same [`egui::TextureId`] with the new [`wgpu::FilterMode`]
    /// (`OptionsView.cpp:298` / `Engine::SetBlurryScaling`). A no-op when the
    /// mode did not change, so the render path can call it every frame.
    ///
    /// This is also the once-per-presented-frame hook for
    /// [`screenshots::CapturePump`]: `present_frame` calls it before encoding
    /// any command buffer, so the next `encode_sim_blit` may capture exactly
    /// one frame. The canvas-resize helper also calls `encode_sim_blit`, but
    /// it runs before this token is opened and therefore cannot capture the
    /// freshly allocated (blank) texture.
    pub fn set_sim_filter(&mut self, filter: wgpu::FilterMode) {
        self.capture_pump.begin_frame();
        if self.sim_filter == filter {
            return;
        }
        self.sim_filter = filter;
        self.egui_renderer.update_egui_texture_from_wgpu_texture(
            &self.device,
            &self.sim_texture_view,
            filter,
            self.sim_tex_id,
        );
    }

    /// Reallocates the 3D render target and re-points the existing egui texture
    /// id at the fresh view.
    ///
    /// `Renderer3d::encode` renders into this texture; egui samples it with
    /// [`Self::view3d_tex_id`] in the 3D viewport.
    pub fn resize_view3d_texture(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if self.view3d_size == (width, height) {
            return;
        }

        let (texture, view) = create_view3d_texture(&self.device, (width, height));
        self.view3d_texture = texture;
        self.view3d_texture_view = view;
        self.view3d_size = (width, height);

        self.egui_renderer.update_egui_texture_from_wgpu_texture(
            &self.device,
            &self.view3d_texture_view,
            wgpu::FilterMode::Linear,
            self.view3d_tex_id,
        );
    }

    /// Copies the simulation framebuffer straight from its CubeCL storage
    /// buffer into [`Self::sim_texture`], entirely on the GPU.
    ///
    /// Returns `false` when the buffer cannot be bound directly (device hiccup,
    /// unexpected sub-allocation offset, or a size mismatch). That is a
    /// skip-frame signal, not an invitation to read the framebuffer back: the
    /// caller leaves the previous texture contents in place and retries on a
    /// later frame. A host round-trip here would stall the device on the
    /// interactive path.
    ///
    /// The blit is encoded into `encoder`, so it shares the frame's command
    /// buffer with the egui pass and needs no extra submit.
    ///
    /// The same call drives screenshot and recording read-backs
    /// ([`screenshots::CapturePump`]): the `P`/`F2` shortcut and
    /// `tpt.record` have no wgpu access of their own, so the one place that
    /// already presents `sim_texture` every frame serves them here. Only the
    /// first call after [`Self::set_sim_filter`] captures, so the resize
    /// helper's extra call is ignored. A screenshot blocks for its single
    /// read-back (the reference writes the PNG synchronously too,
    /// `GameView.cpp:1001-1004`); recording only encodes an async copy and
    /// never waits for the device.
    pub fn encode_sim_blit(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        bufs: &tpt_sim::SimBuffers,
    ) -> bool {
        let blitted = self.framebuf_blit.encode(
            &self.device,
            encoder,
            bufs,
            &self.sim_texture_view,
            self.sim_tex_size,
        );
        // Read the texture, not `bufs.gpu_framebuf`: the freshly encoded blit
        // is submitted after this method returns, so the captured frame is the
        // one currently presented (one frame behind the simulation).
        self.capture_pump.after_frame(
            &self.device,
            &self.queue,
            &self.sim_texture,
            self.sim_tex_size,
        );
        blitted
    }

    /// Human-readable description of the selected adapter, used by the HUD and
    /// the headless smoke test.
    pub fn adapter_summary(&self) -> String {
        let info = self.adapter.get_info();
        format!(
            "{} ({:?}, backend {:?})",
            info.name, info.device_type, info.backend
        )
    }
}

/// Requests a high-performance adapter, falling back to the platform default.
///
/// A hybrid-graphics machine can expose a discrete GPU that cannot present to
/// the surface owned by the integrated GPU, in which case the high-performance
/// request fails even though a usable adapter exists. The second request keeps
/// the application launchable in that case, and the final request accepts a
/// software rasteriser (WARP / lavapipe) so headless CI machines still work.
pub fn request_high_performance_adapter(
    instance: &wgpu::Instance,
    compatible_surface: Option<&wgpu::Surface<'_>>,
) -> Result<wgpu::Adapter, wgpu::RequestAdapterError> {
    let high_performance =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface,
            force_fallback_adapter: false,
            ..Default::default()
        }));
    if let Ok(adapter) = high_performance {
        return Ok(adapter);
    }

    if let Ok(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface,
            force_fallback_adapter: false,
            ..Default::default()
        }))
    {
        return Ok(adapter);
    }

    // Last resort: allow a software rasteriser so headless CI machines without
    // a hardware GPU can still run the smoke test.
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface,
        force_fallback_adapter: true,
        ..Default::default()
    }))
}

/// Creates the simulation framebuffer texture and a full-texture view of it.
///
/// The texture is both a sample source for egui (`TEXTURE_BINDING`) and a copy
/// destination for [`encode_sim_blit`](GpuContext::encode_sim_blit) and the
/// headless-smoke host upload (`COPY_DST`). `STORAGE_BINDING` keeps it usable
/// as a compute storage image for GPU-side post effects.
pub fn create_sim_texture(
    device: &wgpu::Device,
    (width, height): (u32, u32),
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tpt-sim-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SIM_TEXTURE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Creates the 3D render target and a full-texture view of it.
///
/// The format is the one [`crate::render3d::Renderer3d`] is created with
/// (`Rgba8Unorm`): the 3D lighting pass writes display-ready linear values and
/// egui samples them as an unorm texture. Usage is render target plus texture
/// binding, which is exactly what the two passes need.
pub fn create_view3d_texture(
    device: &wgpu::Device,
    (width, height): (u32, u32),
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tpt-view3d-texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Uploads a tightly packed RGBA8 image into `texture`.
///
/// Used by the `--headless-smoke` verification path, which is the only place a
/// rendered frame is allowed to exist on the host: the interactive viewport
/// blits the framebuffer on the device instead. Returns `false` (and performs
/// no GPU work) when `rgba_bytes` does not match `size`, which keeps a single
/// malformed frame from triggering a wgpu validation error.
pub fn upload_rgba_to_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    (width, height): (u32, u32),
    rgba_bytes: &[u8],
) -> bool {
    let expected = (width as usize) * (height as usize) * SIM_BYTES_PER_PIXEL as usize;
    if rgba_bytes.len() != expected {
        return false;
    }

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * SIM_BYTES_PER_PIXEL),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    true
}
