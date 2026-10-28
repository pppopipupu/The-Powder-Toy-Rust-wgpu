//! 3D voxel renderer (plan sections 8 and 9).
//!
//! [`Renderer3d`] owns the two GPU passes of the 3D mode:
//!
//! 1. **World OBB**: one draw of the 256^3 world cube with front-face culling.
//!    The fragment stage reconstructs the primary ray from the camera basis,
//!    traverses the world A/B state (authoritative `voxel_a`) through the
//!    brick-aligned leaf masks and the coarse pyramid, intersects the rigid
//!    bodies in the same trace, accumulates volume inscatter for
//!    water/oil/steam/smoke/fire and writes the G-buffer plus `frag_depth`.
//! 2. **Lighting**: a fullscreen triangle that rebuilds world positions from
//!    depth, shoots one sun shadow ray and two ambient-occlusion rays through
//!    the same hierarchy, composites the volume inscatter and tone maps the
//!    result into the presentation texture.
//!
//! World state is never read back: the seven CubeCL buffers are bound directly
//! with [`bindings::create_scene_bind_group`], and the only per-frame upload is
//! the camera uniform (the render palette is uploaded on resize). The renderer
//! reuses the application's `wgpu::Device`/`wgpu::Queue` and the existing
//! `sim_texture`; it never creates a second device.

pub mod bindings;
pub mod camera;
pub mod gbuffer;
pub mod pipelines;

use std::mem::size_of;

use tpt_sim::voxel3d::Voxel3dBuffers;

pub use bindings::{MAX_BODIES, RigidRenderBuffers, STORAGE_BUFFER_BINDINGS};
pub use camera::{Camera, RenderSettings, SceneUniforms};
pub use gbuffer::GBuffer;

use pipelines::{FULLSCREEN_VERTICES, RenderPipelines, WORLD_OBB_VERTICES};

/// Gamma-space sky colour.
///
/// The lighting shader uses the same constant for rays that leave the world
/// without an opaque hit, and the presentation target is cleared to it, so an
/// empty pixel is bit-identical to the clear colour.
pub const SKY_COLOR: [f32; 4] = [0.53, 0.72, 0.92, 1.0];

/// Clear value matching [`SKY_COLOR`].
const SKY_CLEAR: wgpu::Color = wgpu::Color {
    r: SKY_COLOR[0] as f64,
    g: SKY_COLOR[1] as f64,
    b: SKY_COLOR[2] as f64,
    a: SKY_COLOR[3] as f64,
};

/// Clear value of RT2 `(inscatter.rgb, transmittance)`.
const INSCATTER_CLEAR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};

/// Smallest render scale accepted by [`Renderer3d::resize`].
const MIN_RENDER_SCALE: f32 = 0.1;
/// Largest render scale accepted by [`Renderer3d::resize`].
const MAX_RENDER_SCALE: f32 = 1.0;

/// Deferred 3D renderer.
///
/// Lifetime: create once with the application's `wgpu::Device` and the format
/// of the presentation texture, call [`Self::resize`] whenever the viewport or
/// the render scale changes, then call [`Self::encode`] on every frame before
/// the encoder is submitted. The `target` passed to [`Self::encode`] must have
/// the size reported by [`Self::target_size`].
pub struct Renderer3d {
    /// Scene bind-group layout (`@group(0)`: seven storages plus the uniform).
    scene_layout: wgpu::BindGroupLayout,
    /// Lighting texture layout (`@group(1)`).
    lighting_layout: wgpu::BindGroupLayout,
    /// Both render pipelines.
    pipelines: RenderPipelines,
    /// RT0/RT1/RT2/depth targets.
    gbuffer: GBuffer,
    /// Bind group of the G-buffer views for the lighting pass.
    lighting_bind_group: wgpu::BindGroup,
    /// Per-frame camera and lighting uniform.
    uniform: wgpu::Buffer,
    /// Sun/ambient/exposure settings.
    settings: RenderSettings,
    /// Render palette allocated on the world's CubeCL client.
    palette: Option<cubecl::server::Handle>,
    /// Rigid-body buffers supplied by the rigid solver, if any.
    rigid: Option<RigidRenderBuffers>,
    /// Zeroed rigid-body placeholders used while no solver state is bound.
    placeholder_rigid: Option<RigidRenderBuffers>,
    /// Number of storage buffers bound by the last frame (0 before the first
    /// successful frame).
    storage_buffers: usize,
    /// Suppresses repeated binding-failure messages.
    reported_binding_error: bool,
}

impl Renderer3d {
    /// Number of storage buffers the scene bind group binds.
    pub const STORAGE_BUFFER_BINDINGS: usize = bindings::STORAGE_BUFFER_BINDINGS;

    /// Main pass count: world OBB and lighting.
    pub const PASS_COUNT: usize = 2;

    /// Creates the renderer and its 1x1 targets.
    ///
    /// [`Self::resize`] must be called with the viewport size before the first
    /// frame; the initial targets only exist so the pipelines and bind groups
    /// are always complete.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let scene_layout = bindings::scene_bind_group_layout(device);
        let lighting_layout = bindings::lighting_texture_layout(device);
        let pipelines =
            RenderPipelines::new(device, target_format, &scene_layout, &lighting_layout);
        let gbuffer = GBuffer::new(device, 1, 1);
        let lighting_bind_group =
            bindings::lighting_texture_bind_group(device, &lighting_layout, &gbuffer);

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tpt-render3d-scene-uniform"),
            size: size_of::<SceneUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            scene_layout,
            lighting_layout,
            pipelines,
            gbuffer,
            lighting_bind_group,
            uniform,
            settings: RenderSettings::default(),
            palette: None,
            rigid: None,
            placeholder_rigid: None,
            storage_buffers: 0,
            reported_binding_error: false,
        }
    }

    /// Reallocates the G-buffer for a viewport pixel size and render scale.
    ///
    /// `render_scale` is clamped to `0.1..=1.0`; the physical size is rounded
    /// and clamped to the device's maximum texture dimension. The presentation
    /// texture must be resized by the caller to match [`Self::target_size`].
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, render_scale: f32) {
        let scale = if render_scale.is_finite() {
            render_scale.clamp(MIN_RENDER_SCALE, MAX_RENDER_SCALE)
        } else {
            MAX_RENDER_SCALE
        };
        let max_dimension = device.limits().max_texture_dimension_2d;
        let width = ((width as f64 * f64::from(scale)).round() as u32).clamp(1, max_dimension);
        let height = ((height as f64 * f64::from(scale)).round() as u32).clamp(1, max_dimension);

        if self.gbuffer.resize(device, width, height) {
            self.lighting_bind_group =
                bindings::lighting_texture_bind_group(device, &self.lighting_layout, &self.gbuffer);
        }
    }

    /// Encodes the world OBB pass and the lighting pass into `encoder`.
    ///
    /// `target` must be the presentation texture view (created from the
    /// `target_format` passed to [`Self::new`]) at [`Self::target_size`].
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        world: &Voxel3dBuffers,
        camera: &Camera,
    ) {
        let size = self.gbuffer.size();
        let aspect = self.gbuffer.aspect();
        let body_count = self
            .rigid
            .as_ref()
            .map_or(0, |rigid| rigid.body_count.min(MAX_BODIES as u32));
        let uniforms = SceneUniforms::new(camera, aspect, size, &self.settings, body_count);
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniforms));

        // Palettes and placeholders belong to one CubeCL client; a new service
        // (for example after a device reset) invalidates them.
        let service = world.client.service_id();
        if !self
            .palette
            .as_ref()
            .is_some_and(|handle| handle.service == service)
        {
            self.palette = Some(bindings::palette_buffer(&world.client));
        }
        if !self
            .placeholder_rigid
            .as_ref()
            .is_some_and(|rigid| rigid.body_desc.service == service)
        {
            self.placeholder_rigid = Some(RigidRenderBuffers::placeholder(&world.client));
        }

        let palette = self
            .palette
            .as_ref()
            .expect("the render palette was created above");
        let rigid = match &self.rigid {
            Some(rigid) => rigid,
            None => self
                .placeholder_rigid
                .as_ref()
                .expect("the placeholder rigid buffers were created above"),
        };

        let binding = bindings::create_scene_bind_group(
            device,
            &self.scene_layout,
            &self.uniform,
            world,
            palette,
            rigid,
        );

        let binding = match binding {
            Ok(binding) => {
                self.storage_buffers = binding.storage_buffers;
                self.reported_binding_error = false;
                binding
            }
            Err(error) => {
                self.storage_buffers = 0;
                if !self.reported_binding_error {
                    eprintln!("[render3d] skipping frame: {error}");
                    self.reported_binding_error = true;
                }
                return;
            }
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tpt-render3d-world-obb-pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuffer.rt0_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuffer.rt1_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.gbuffer.rt2_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(INSCATTER_CLEAR),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.gbuffer.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines.world_obb);
            pass.set_bind_group(0, &binding.bind_group, &[]);
            pass.draw(0..WORLD_OBB_VERTICES, 0..1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tpt-render3d-lighting-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(SKY_CLEAR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines.lighting);
            pass.set_bind_group(0, &binding.bind_group, &[]);
            pass.set_bind_group(1, &self.lighting_bind_group, &[]);
            pass.draw(0..FULLSCREEN_VERTICES, 0..1);
        }
    }

    /// Render resolution of the G-buffer and of the presentation target.
    pub fn target_size(&self) -> (u32, u32) {
        self.gbuffer.size()
    }

    /// Number of storage buffers bound by the last [`Self::encode`] call.
    pub fn storage_buffers(&self) -> usize {
        self.storage_buffers
    }

    /// The G-buffer targets.
    pub fn gbuffer(&self) -> &GBuffer {
        &self.gbuffer
    }

    /// Current lighting settings.
    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }

    /// Mutable lighting settings, applied to the next frame.
    pub fn settings_mut(&mut self) -> &mut RenderSettings {
        &mut self.settings
    }

    /// Replaces the rigid-body render buffers (or clears the override).
    ///
    /// Pass `None` to fall back to the zeroed `body_count == 0` placeholders.
    pub fn set_rigid_buffers(&mut self, rigid: Option<RigidRenderBuffers>) {
        self.rigid = rigid;
    }

    /// Rigid-body buffers currently bound, if an override is set.
    pub fn rigid_buffers(&self) -> Option<&RigidRenderBuffers> {
        self.rigid.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    use cubecl::bytes::Bytes;
    use cubecl::client::Client;
    use cubecl::prelude::*;
    use cubecl::wgpu::{
        AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuRuntime, WgpuSetup, init_device,
    };
    use cubecl_runtime::runtime::Runtime;
    use glam::Vec3;
    use tpt_core::constants::PT_STNE;
    use tpt_core::voxel3d::{
        COARSE_LEVELS, LEAF_WORD_COUNT, VOXEL_COUNT, coarse_bit_mask, coarse_word_index,
    };
    use tpt_sim::voxel3d::rigid::Rigid3dState;
    use tpt_sim::voxel3d::{COARSE_MASK_WORDS, Voxel3dBuffers};

    /// Side length of the offscreen render used by the test.
    const SIZE: u32 = 256;
    /// Centre of the world and of the test sphere.
    const CENTER: Vec3 = Vec3::new(128.0, 128.0, 128.0);
    /// Origin of the 64^3 region written by the test kernel.
    const REGION_ORIGIN: u32 = 96;
    /// Bytes per pixel of both `Rgba8Unorm` and `Depth32Float`.
    const BYTES_PER_PIXEL: u32 = 4;

    /// Writes a stone sphere into a 64^3 region and rebuilds its leaf masks.
    ///
    /// One workgroup per 8^3 brick and one thread per workgroup: the thread
    /// fills all 512 voxels of its brick and derives that brick's two
    /// leaf-mask words, so the test never relies on racy read-modify-writes.
    #[cube(launch_unchecked)]
    fn stone_sphere_kernel(voxel_a: &mut [u32], leaf_masks: &mut [u32], stone_word: u32) {
        let origin_x = 96u32 + CUBE_POS_X * 8u32;
        let origin_y = 96u32 + CUBE_POS_Y * 8u32;
        let origin_z = 96u32 + CUBE_POS_Z * 8u32;

        let mut bits_low = 0u32;
        let mut bits_high = 0u32;

        for voxel in 0..512u32 {
            let lx = voxel & 7u32;
            let ly = (voxel >> 3u32) & 7u32;
            let lz = voxel >> 6u32;
            let x = origin_x + lx;
            let y = origin_y + ly;
            let z = origin_z + lz;

            let dx = x as f32 - 128.0f32;
            let dy = y as f32 - 128.0f32;
            let dz = z as f32 - 128.0f32;
            if dx * dx + dy * dy + dz * dz <= 576.0f32 {
                let index = (y << 16u32) | (z << 8u32) | x;
                voxel_a[index as usize] = stone_word;

                let block = ((ly >> 1u32) << 4u32) | ((lz >> 1u32) << 2u32) | (lx >> 1u32);
                if block < 32u32 {
                    bits_low |= 1u32 << block;
                } else {
                    bits_high |= 1u32 << (block - 32u32);
                }
            }
        }

        let brick =
            ((origin_y >> 3u32) << 10u32) | ((origin_z >> 3u32) << 5u32) | (origin_x >> 3u32);
        leaf_masks[(brick * 2u32) as usize] = bits_low;
        leaf_masks[(brick * 2u32 + 1u32) as usize] = bits_high;
    }

    /// A `wgpu` device plus the CubeCL client registered on the same device.
    struct TestGpu {
        device: wgpu::Device,
        queue: wgpu::Queue,
        client: Client,
        /// Keeps the instance alive for the lifetime of the adapter.
        _instance: wgpu::Instance,
    }

    /// Creates the raw device and a CubeCL client, mirroring `GpuContext::new`.
    fn create_gpu() -> TestGpu {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .expect("no GPU adapter available for the 3D renderer test");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tpt-render3d-test-device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .expect("failed to create the 3D renderer test device");

        let setup = WgpuSetup {
            instance: instance.clone(),
            adapter: adapter.clone(),
            device: device.clone(),
            queue: queue.clone(),
            backend: adapter.get_info().backend,
        };
        let cubecl_device = init_device(
            setup,
            RuntimeOptions {
                memory_config: MemoryConfiguration::ExclusivePages,
                ..Default::default()
            },
        );
        let client = WgpuRuntime::<AutoCompiler>::client(&cubecl_device);

        TestGpu {
            device,
            queue,
            client,
            _instance: instance,
        }
    }

    /// Conservative coarse-mask words covering the 64^3 sphere region.
    ///
    /// Every pyramid level marks each cell that overlaps `[96, 160)^3` using
    /// the frozen `tpt_core::voxel3d` index helpers, so the test builds the
    /// exact layout the shader reads. The masks are allowed to over-report
    /// occupancy (plan section 3.4), which keeps the sphere covered while the
    /// empty world outside the region stays skippable.
    fn coarse_words_for_region() -> Vec<u32> {
        let mut words = vec![0u32; COARSE_MASK_WORDS];
        let low = REGION_ORIGIN;
        let high = REGION_ORIGIN + 64;
        for (level, cell) in COARSE_LEVELS.iter().copied().enumerate() {
            let cells = 256 / cell;
            for cy in 0..cells {
                for cz in 0..cells {
                    for cx in 0..cells {
                        let axis_intersects = |start: u32| start < high && start + cell > low;
                        if axis_intersects(cx * cell)
                            && axis_intersects(cy * cell)
                            && axis_intersects(cz * cell)
                        {
                            let x = cx * cell;
                            let y = cy * cell;
                            let z = cz * cell;
                            words[coarse_word_index(level, x, y, z)] |=
                                coarse_bit_mask(level, x, y, z);
                        }
                    }
                }
            }
        }
        words
    }

    /// Creates a `COPY_DST | MAP_READ` readback buffer.
    fn create_readback(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// Maps `buffer`, waits for the submitted copies and returns its bytes.
    fn read_mapped(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
        let (tx, rx) = mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("failed to wait for the GPU readback copy");
        rx.recv()
            .expect("map_async callback was dropped without a result")
            .expect("failed to map the readback buffer");
        let data = buffer
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range")
            .to_vec();
        buffer.unmap();
        data
    }

    /// Expected 8-bit encoding of [`SKY_COLOR`].
    fn sky_bytes() -> [u8; 3] {
        [
            (SKY_COLOR[0] * 255.0).round() as u8,
            (SKY_COLOR[1] * 255.0).round() as u8,
            (SKY_COLOR[2] * 255.0).round() as u8,
        ]
    }

    #[test]
    fn offscreen_world_trace_writes_gbuffer_and_depth() {
        let gpu = create_gpu();
        let mut renderer = Renderer3d::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm);
        renderer.resize(&gpu.device, SIZE, SIZE, 1.0);
        assert_eq!(renderer.target_size(), (SIZE, SIZE));

        let world = Voxel3dBuffers::new(gpu.client.clone());
        gpu.client.write(
            &world.coarse_masks,
            Bytes::from_elems(coarse_words_for_region()),
        );

        // 293.15 K in the packed fixed-point (`temp_q * 8`) encoding, `y`-up
        // stone sphere.
        let stone_word = (2345u32 << 8) | PT_STNE;
        unsafe {
            stone_sphere_kernel::launch_unchecked(
                &gpu.client,
                CubeCount::Static(8, 8, 8),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(world.voxel_a.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(world.leaf_masks.clone(), LEAF_WORD_COUNT),
                stone_word,
            );
        }

        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tpt-render3d-test-target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let camera = Camera::looking_at(CENTER);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tpt-render3d-test-encoder"),
            });
        renderer.encode(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &target_view,
            &world,
            &camera,
        );

        let pixels = u64::from(SIZE * SIZE * BYTES_PER_PIXEL);
        let color_readback = create_readback(&gpu.device, "tpt-render3d-test-color", pixels);
        let depth_readback = create_readback(&gpu.device, "tpt-render3d-test-depth", pixels);
        let extent = wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        };
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIZE * BYTES_PER_PIXEL),
            rows_per_image: Some(SIZE),
        };
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &color_readback,
                layout,
            },
            extent,
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &renderer.gbuffer().depth,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &depth_readback,
                layout,
            },
            extent,
        );
        gpu.queue.submit([encoder.finish()]);

        let color = read_mapped(&gpu.device, &color_readback);
        let depth = read_mapped(&gpu.device, &depth_readback);

        let center_offset = ((SIZE / 2 * SIZE + SIZE / 2) * BYTES_PER_PIXEL) as usize;
        let sky_offset = ((2 * SIZE + 2) * BYTES_PER_PIXEL) as usize;
        let center: [u8; 4] = color[center_offset..center_offset + 4]
            .try_into()
            .expect("centre pixel is inside the readback buffer");
        let sky: [u8; 4] = color[sky_offset..sky_offset + 4]
            .try_into()
            .expect("sky pixel is inside the readback buffer");
        let center_depth = f32::from_ne_bytes(
            depth[center_offset..center_offset + 4]
                .try_into()
                .expect("centre depth is inside the readback buffer"),
        );
        let sky_depth = f32::from_ne_bytes(
            depth[sky_offset..sky_offset + 4]
                .try_into()
                .expect("sky depth is inside the readback buffer"),
        );

        println!(
            "render3d offscreen: centre={center:?} depth={center_depth} sky={sky:?} depth={sky_depth} storage_buffers={}",
            renderer.storage_buffers(),
        );

        let sky_expected = sky_bytes();
        assert_ne!(
            [center[0], center[1], center[2]],
            sky_expected,
            "the sphere centre must not render the clear/sky colour",
        );
        assert!(
            center[0] > 0 || center[1] > 0 || center[2] > 0,
            "the lit sphere is black: {center:?}",
        );
        assert_eq!(
            [sky[0], sky[1], sky[2]],
            sky_expected,
            "a pixel without a hit must equal the clear/sky colour",
        );
        assert_eq!(sky[3], 255, "sky alpha is {}", sky[3]);
        assert!(
            center_depth > 0.0 && center_depth < 1.0,
            "the centre depth must be written through frag_depth, got {center_depth}",
        );
        assert_eq!(
            sky_depth, 1.0,
            "the sky pixel must keep the cleared depth, got {sky_depth}",
        );
        assert_eq!(
            renderer.storage_buffers(),
            Renderer3d::STORAGE_BUFFER_BINDINGS,
        );
        assert_eq!(bindings::STORAGE_BUFFER_BINDINGS, 7);
    }

    #[test]
    fn offscreen_rigid_body_traces_with_the_solver_layout() {
        let gpu = create_gpu();
        let size = 128u32;
        let mut renderer = Renderer3d::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm);
        renderer.resize(&gpu.device, size, size, 1.0);

        let world = Voxel3dBuffers::new(gpu.client.clone());
        let mut rigid = Rigid3dState::new(&gpu.client);
        assert_eq!(
            rigid.spawn_box(&gpu.client, [128.0, 128.0, 128.0], [8, 8, 8], PT_STNE),
            0
        );
        rigid.step(&gpu.client, &world, 1);
        assert_eq!(rigid.body_count(), 1);
        renderer.set_rigid_buffers(Some(RigidRenderBuffers {
            body_desc: rigid.body_desc.clone(),
            body_voxels: rigid.body_voxels.clone(),
            body_masks: rigid.body_masks.clone(),
            body_count: rigid.body_count(),
        }));

        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tpt-render3d-body-test-target"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let camera = Camera::looking_at(Vec3::new(128.0, 128.0, 128.0));
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tpt-render3d-body-test-encoder"),
            });
        renderer.encode(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &target_view,
            &world,
            &camera,
        );

        let pixels = u64::from(size * size * 4);
        let readback = create_readback(&gpu.device, "tpt-render3d-body-test-color", pixels);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);

        let color = read_mapped(&gpu.device, &readback);
        let offset = (((size / 2) * size + size / 2) * 4) as usize;
        let center: [u8; 4] = color[offset..offset + 4]
            .try_into()
            .expect("centre pixel is inside the body readback buffer");
        assert_ne!(
            [center[0], center[1], center[2]],
            sky_bytes(),
            "the rigid body must occlude the centre pixel, got {center:?}"
        );
        println!("render3d body offscreen: centre={center:?}");
    }

    /// The scene bind group binds exactly the seven plan buffers.
    #[test]
    fn scene_layout_binds_seven_storage_buffers() {
        assert_eq!(STORAGE_BUFFER_BINDINGS, 7);
        assert_eq!(bindings::UNIFORM_BINDING, 7);
    }
}
