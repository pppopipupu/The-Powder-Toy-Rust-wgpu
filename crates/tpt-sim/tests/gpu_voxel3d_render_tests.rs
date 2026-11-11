//! Opt-in offscreen diagnostics for the M0 3D world-OBB render pass.
//!
//! Gated by `required-features = ["diagnostics"]`: a white-box check of the
//! CubeCL/wgpu storage-binding path, not element behaviour, so it stays out
//! of the default regression suite.
//!
//! The test drives the real `wgpu` device and a CubeCL client registered on
//! that same device (the same bring-up used by `gpu_sim_tests.rs`). Seven
//! CubeCL allocations are bound directly as read-only storage buffers, exactly
//! the way `Renderer3d` binds the `voxel3d` state:
//!
//! - binding 0 `voxel_a`, 1 `leaf_masks`, 2 `coarse_masks` are `array<u32>`;
//! - binding 3 `palette` and 4 `body_desc` are `array<vec4<f32>>`;
//! - binding 5 `body_voxels` and 6 `body_masks` are `array<u32>`.
//!
//! The fragment shader XORs the probe word with the other word buffers and
//! converts the low 24 bits to RGB, so the rendered centre pixel proves which
//! CubeCL allocation the bind group actually points at.
//!
//! The gate checks:
//!
//! - the adapter exposes at least seven storage buffers per shader stage;
//! - the world OBB is rendered as one draw of 36 vertices with
//!   `front_face: Ccw`, `cull_mode: Some(Front)` (back faces visible), three
//!   colour targets (`Rgba8Unorm`, `Rgba16Float`, `Rgba16Float`) and a
//!   `Depth32Float` target written through `@builtin(frag_depth)`;
//! - the centre pixel of RT0 is red and the centre depth is strictly inside
//!   `(0, 1)`, so neither the clear colour nor the cleared depth is visible;
//! - rebuilding the bind group around a fresh green `voxel_a` CubeCL buffer
//!   turns the centre pixel green, proving the bind group tracks the live
//!   allocation rather than a stale copy.

use std::num::NonZeroU64;
use std::sync::mpsc;

use cubecl::client::Client;
use cubecl::server::Handle;
use cubecl::wgpu::{
    AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuResource, WgpuRuntime, WgpuServer,
    WgpuSetup, init_device,
};
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::storage::ManagedResource;
use glam::Vec3;
use glam::camera::rh::{proj, view};

/// Render target edge length; the centre pixel is at `(128, 128)`.
const SIZE: u32 = 256;
/// Centre pixel used by the assertions.
const CENTER: (u32, u32) = (SIZE / 2, SIZE / 2);
/// Low byte red: byte 0 is R, byte 1 is G, byte 2 is B in the probe encoding.
const RED_PROBE: u32 = 0x0000_00FF;
/// Low byte zero, byte 1 green.
const GREEN_PROBE: u32 = 0x0000_FF00;
/// Number of world OBB vertices: 12 triangles.
const CUBE_VERTICES: u32 = 36;
/// Number of CubeCL storage buffers bound by the pass.
const STORAGE_BUFFERS: usize = 7;
/// Bytes per pixel of both `Rgba8Unorm` and `Depth32Float`.
const BYTES_PER_PIXEL: u32 = 4;

/// One offscreen frame read back to the host.
struct Sample {
    /// Centre pixel of RT0 as `[R, G, B, A]`.
    color: [u8; 4],
    /// Centre `Depth32Float` value written by `@builtin(frag_depth)`.
    depth: f32,
}

/// A bind group plus the CubeCL bindings that keep its allocations in place.
struct StorageBinding {
    /// Every allocation bound by [`Self::bind_group`]. `ManagedResource`
    /// prevents CubeCL from freeing or relocating the memory while the bind
    /// group still references it.
    _resources: Vec<ManagedResource<WgpuResource>>,
    bind_group: wgpu::BindGroup,
}

/// The `wgpu` objects shared with the CubeCL client.
struct Gpu {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    client: Client,
    /// Keeps the instance alive for the lifetime of the adapter.
    _instance: wgpu::Instance,
}

/// G-buffer textures and their views.
struct Targets {
    rt0: wgpu::Texture,
    /// RT1 is only validated by pipeline creation, never read back; the field
    /// keeps the texture alive next to the view it backs.
    #[expect(dead_code, reason = "render target kept alive for its view")]
    rt1: wgpu::Texture,
    /// RT2 is only validated by pipeline creation, never read back.
    #[expect(dead_code, reason = "render target kept alive for its view")]
    rt2: wgpu::Texture,
    depth: wgpu::Texture,
    rt0_view: wgpu::TextureView,
    rt1_view: wgpu::TextureView,
    rt2_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

/// Creates the raw device and a CubeCL client registered on the same device.
///
/// This mirrors `gpu_sim_tests.rs`: one adapter, one `wgpu::Device`/`Queue`,
/// and a [`WgpuSetup`] handed to CubeCL so storage buffers allocated through
/// the client can be bound directly by a `wgpu` pass.
fn create_gpu() -> Gpu {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .expect("no GPU adapter available for the M0 3D render gate");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("tpt-voxel3d-render-test-device"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::Performance,
        ..Default::default()
    }))
    .expect("failed to create the M0 3D render test wgpu device/queue");

    let setup = WgpuSetup {
        instance: instance.clone(),
        adapter: adapter.clone(),
        device: device.clone(),
        queue: queue.clone(),
        backend: adapter.get_info().backend,
    };
    // The app/3D configuration: one page per allocation rather than an
    // adaptive pool, so every handle is a distinct wgpu allocation.
    let cubecl_device = init_device(
        setup,
        RuntimeOptions {
            memory_config: MemoryConfiguration::ExclusivePages,
            ..Default::default()
        },
    );
    let client = WgpuRuntime::<AutoCompiler>::client(&cubecl_device);

    Gpu {
        adapter,
        device,
        queue,
        client,
        _instance: instance,
    }
}

/// Formats a `f32` as a WGSL literal (`Debug` is round-trip precise).
fn wgsl_f32(value: f32) -> String {
    format!("{value:?}")
}

/// Formats a `Vec3` as a WGSL constructor.
fn wgsl_vec3(value: Vec3) -> String {
    format!(
        "vec3<f32>({}, {}, {})",
        wgsl_f32(value.x),
        wgsl_f32(value.y),
        wgsl_f32(value.z)
    )
}

/// Builds the shader source with the frozen camera baked in.
///
/// The vertex stage projects the 36-vertex world cube and the fragment stage
/// reconstructs the primary ray from the pixel coordinate, intersects the
/// world AABB, and writes the hit depth through `@builtin(frag_depth)`.
fn shader_source() -> String {
    let eye = Vec3::new(420.0, 320.0, 420.0);
    let target = Vec3::new(128.0, 128.0, 128.0);
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward).normalize();

    let fov_y = 45.0f32.to_radians();
    let aspect = 1.0f32;
    let near = 1.0f32;
    let far = 4096.0f32;
    let tan_half_fov = (fov_y * 0.5).tan();

    let view_matrix = view::look_at_mat4(eye, target, Vec3::Y);
    // `Rgba8Unorm`/`Rgba16Float` targets use the WebGPU NDC convention.
    let projection = proj::directx::perspective(fov_y, aspect, near, far);
    let view_projection = projection * view_matrix;
    let columns = view_projection.to_cols_array();
    let matrix = format!(
        "mat4x4<f32>(vec4<f32>({}, {}, {}, {}), vec4<f32>({}, {}, {}, {}), \
         vec4<f32>({}, {}, {}, {}), vec4<f32>({}, {}, {}, {}))",
        wgsl_f32(columns[0]),
        wgsl_f32(columns[1]),
        wgsl_f32(columns[2]),
        wgsl_f32(columns[3]),
        wgsl_f32(columns[4]),
        wgsl_f32(columns[5]),
        wgsl_f32(columns[6]),
        wgsl_f32(columns[7]),
        wgsl_f32(columns[8]),
        wgsl_f32(columns[9]),
        wgsl_f32(columns[10]),
        wgsl_f32(columns[11]),
        wgsl_f32(columns[12]),
        wgsl_f32(columns[13]),
        wgsl_f32(columns[14]),
        wgsl_f32(columns[15]),
    );

    let template = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

struct FragmentOutput {
    @location(0) rt0: vec4<f32>,
    @location(1) rt1: vec4<f32>,
    @location(2) rt2: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

const VIEW_PROJ: mat4x4<f32> = __VIEW_PROJ__;
const EYE: vec3<f32> = __EYE__;
const FORWARD: vec3<f32> = __FORWARD__;
const RIGHT: vec3<f32> = __RIGHT__;
const UP: vec3<f32> = __UP__;
const TAN_HALF_FOV: f32 = __TAN_HALF_FOV__;
const ASPECT: f32 = __ASPECT__;
const NEAR: f32 = __NEAR__;
const FAR: f32 = __FAR__;
const TARGET_SIZE: vec2<f32> = vec2<f32>(256.0, 256.0);
const BOX_MIN: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const BOX_MAX: vec3<f32> = vec3<f32>(256.0, 256.0, 256.0);

const CUBE_CORNERS: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>(0.0, 0.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 1.0, 1.0),
);

// Outward-facing counter-clockwise triangles: two per cube face.
const CUBE_INDICES: array<u32, 36> = array<u32, 36>(
    5u, 1u, 2u, 5u, 2u, 6u,
    0u, 4u, 7u, 0u, 7u, 3u,
    3u, 7u, 6u, 3u, 6u, 2u,
    0u, 1u, 5u, 0u, 5u, 4u,
    4u, 5u, 6u, 4u, 6u, 7u,
    0u, 3u, 2u, 0u, 2u, 1u,
);

@group(0) @binding(0) var<storage, read> voxel_a: array<u32>;
@group(0) @binding(1) var<storage, read> leaf_masks: array<u32>;
@group(0) @binding(2) var<storage, read> coarse_masks: array<u32>;
@group(0) @binding(3) var<storage, read> palette: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> body_desc: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> body_voxels: array<u32>;
@group(0) @binding(6) var<storage, read> body_masks: array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let corner = CUBE_CORNERS[CUBE_INDICES[vertex_index]];
    var out: VertexOutput;
    out.clip_position = VIEW_PROJ * vec4<f32>(corner * 256.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> FragmentOutput {
    let uv = frag_coord.xy / TARGET_SIZE;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_dir = normalize(
        FORWARD + RIGHT * (ndc.x * TAN_HALF_FOV * ASPECT) + UP * (ndc.y * TAN_HALF_FOV),
    );

    let inv_dir = 1.0 / ray_dir;
    let t0 = (BOX_MIN - EYE) * inv_dir;
    let t1 = (BOX_MAX - EYE) * inv_dir;
    let t_small = min(t0, t1);
    let t_big = max(t0, t1);
    let t_enter = max(max(t_small.x, t_small.y), t_small.z);
    let t_exit = min(min(t_big.x, t_big.y), t_big.z);
    if (t_exit < max(t_enter, 0.0)) {
        discard;
    }

    let t_hit = max(t_enter, 0.0);
    let view_depth = max(t_hit * dot(ray_dir, FORWARD), NEAR);
    let depth = (FAR / (FAR - NEAR)) * (1.0 - NEAR / view_depth);

    var probe = voxel_a[0] ^ leaf_masks[0] ^ coarse_masks[0] ^ body_voxels[0] ^ body_masks[0];
    probe = probe ^ bitcast<u32>(palette[0].x) ^ bitcast<u32>(body_desc[0].x);
    let rgb_bits = probe & 0x00FFFFFFu;
    let rgb = vec3<f32>(
        f32(rgb_bits & 0xFFu),
        f32((rgb_bits >> 8u) & 0xFFu),
        f32((rgb_bits >> 16u) & 0xFFu),
    ) / 255.0;

    var out: FragmentOutput;
    out.rt0 = vec4<f32>(rgb, 1.0);
    out.rt1 = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    out.rt2 = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.depth = depth;
    return out;
}
"#;

    template
        .replace("__VIEW_PROJ__", &matrix)
        .replace("__EYE__", &wgsl_vec3(eye))
        .replace("__FORWARD__", &wgsl_vec3(forward))
        .replace("__RIGHT__", &wgsl_vec3(right))
        .replace("__UP__", &wgsl_vec3(up))
        .replace("__TAN_HALF_FOV__", &wgsl_f32(tan_half_fov))
        .replace("__ASPECT__", &wgsl_f32(aspect))
        .replace("__NEAR__", &wgsl_f32(near))
        .replace("__FAR__", &wgsl_f32(far))
}

/// Creates the bind group layout for the seven storage buffers.
fn storage_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tpt-voxel3d-storage-layout"),
        entries: &[
            storage_entry(0),
            storage_entry(1),
            storage_entry(2),
            storage_entry(3),
            storage_entry(4),
            storage_entry(5),
            storage_entry(6),
        ],
    })
}

/// One read-only fragment-stage storage buffer entry.
fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Creates the G-buffer render targets.
fn create_targets(device: &wgpu::Device) -> Targets {
    let rt0 = create_color_target(device, "tpt-voxel3d-rt0", wgpu::TextureFormat::Rgba8Unorm);
    let rt1 = create_color_target(device, "tpt-voxel3d-rt1", wgpu::TextureFormat::Rgba16Float);
    let rt2 = create_color_target(device, "tpt-voxel3d-rt2", wgpu::TextureFormat::Rgba16Float);
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tpt-voxel3d-depth"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let rt0_view = rt0.create_view(&wgpu::TextureViewDescriptor::default());
    let rt1_view = rt1.create_view(&wgpu::TextureViewDescriptor::default());
    let rt2_view = rt2.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    Targets {
        rt0,
        rt1,
        rt2,
        depth,
        rt0_view,
        rt1_view,
        rt2_view,
        depth_view,
    }
}

/// Creates a `RENDER_ATTACHMENT | COPY_SRC` colour texture.
fn create_color_target(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Builds the world-OBB pipeline used by the gate.
fn create_pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let source = shader_source();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tpt-voxel3d-world-obb"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tpt-voxel3d-world-obb-layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("tpt-voxel3d-world-obb"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            // Only the back faces of the world OBB are rasterized; the
            // fragment shader traces the world itself.
            cull_mode: Some(wgpu::Face::Front),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Allocates the seven probe buffers in CubeCL.
///
/// The caller keeps the handles alive; [`bind_cubecl_storage`] then resolves
/// them to the underlying `wgpu::Buffer` plus offset.
fn create_probe_handles(client: &Client, voxel_word: u32) -> Vec<Handle> {
    vec![
        // 0: voxel_a, word 0 carries the probe colour in its low 24 bits.
        client.create_from_slice(bytemuck::cast_slice(&[voxel_word])),
        // 1: leaf_masks
        client.create_from_slice(bytemuck::cast_slice(&[0u32])),
        // 2: coarse_masks
        client.create_from_slice(bytemuck::cast_slice(&[0u32])),
        // 3: palette (vec4<f32>)
        client.create_from_slice(bytemuck::cast_slice(&[0.0f32; 4])),
        // 4: body_desc (vec4<f32>)
        client.create_from_slice(bytemuck::cast_slice(&[0.0f32; 4])),
        // 5: body_voxels
        client.create_from_slice(bytemuck::cast_slice(&[0u32])),
        // 6: body_masks
        client.create_from_slice(bytemuck::cast_slice(&[0u32])),
    ]
}

/// Resolves the CubeCL allocations and binds all seven as storage buffers.
fn bind_cubecl_storage(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    client: &Client,
    handles: &[Handle],
) -> StorageBinding {
    assert_eq!(
        handles.len(),
        STORAGE_BUFFERS,
        "the world-OBB pass expects exactly {STORAGE_BUFFERS} CubeCL buffers",
    );

    let mut resources = Vec::with_capacity(STORAGE_BUFFERS);
    for (binding, handle) in handles.iter().enumerate() {
        let managed = client
            .get_resource::<WgpuServer<AutoCompiler>>(handle.clone())
            .unwrap_or_else(|error| {
                panic!("failed to acquire CubeCL storage buffer {binding}: {error:?}")
            });
        let resource = managed.resource();
        let offset_alignment =
            u64::from(device.limits().min_storage_buffer_offset_alignment).max(1);
        assert_eq!(
            resource.offset % offset_alignment,
            0,
            "CubeCL buffer {binding} offset {} is not aligned to the device's {offset_alignment}-byte storage-buffer offset alignment",
            resource.offset,
        );
        assert!(
            resource.size >= 4,
            "CubeCL buffer {binding} only exposes {} bytes",
            resource.size,
        );
        resources.push(managed);
    }

    let bind_group = {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = resources
            .iter()
            .enumerate()
            .map(|(binding, managed)| {
                let resource = managed.resource();
                wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &resource.buffer,
                        offset: resource.offset,
                        size: NonZeroU64::new(resource.size),
                    }),
                }
            })
            .collect();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tpt-voxel3d-live-cubecl-storage"),
            layout,
            entries: &entries,
        })
    };

    StorageBinding {
        _resources: resources,
        bind_group,
    }
}

/// Renders one offscreen frame and reads back the centre RT0 pixel and depth.
fn render_frame(
    gpu: &Gpu,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    targets: &Targets,
) -> Sample {
    let color_bytes = u64::from(SIZE * SIZE * BYTES_PER_PIXEL);
    let color_readback = create_readback(&gpu.device, "tpt-voxel3d-rt0-readback", color_bytes);
    let depth_readback = create_readback(&gpu.device, "tpt-voxel3d-depth-readback", color_bytes);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tpt-voxel3d-world-obb-encoder"),
        });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tpt-voxel3d-world-obb-pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.rt0_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.rt1_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.rt2_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth_view,
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

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..CUBE_VERTICES, 0..1);
    }

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
            texture: &targets.rt0,
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
            texture: &targets.depth,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &depth_readback,
            layout,
        },
        extent,
    );

    gpu.queue.submit([encoder.finish()]);

    let color_data = read_mapped(&gpu.device, &color_readback);
    let depth_data = read_mapped(&gpu.device, &depth_readback);

    let offset = center_byte_offset();
    let color: [u8; 4] = color_data[offset..offset + 4]
        .try_into()
        .expect("centre RT0 pixel is inside the readback buffer");
    let depth = f32::from_ne_bytes(
        depth_data[offset..offset + 4]
            .try_into()
            .expect("centre depth is inside the readback buffer"),
    );

    Sample { color, depth }
}

/// Byte offset of the centre pixel in a tightly packed `SIZE * SIZE` image.
fn center_byte_offset() -> usize {
    ((CENTER.1 * SIZE + CENTER.0) * BYTES_PER_PIXEL) as usize
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

/// Maps `buffer`, waits for the copy to finish, and returns the bytes.
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
        .expect("failed to map the GPU readback buffer");

    let data = buffer
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback range")
        .to_vec();
    buffer.unmap();
    data
}

/// Asserts that `color` is a saturated channel value with the other channels
/// dark; `channel` is 0 for red and 1 for green.
fn assert_probe(color: [u8; 4], channel: usize) {
    assert!(
        color[channel] >= 250,
        "probe channel {channel} is {color:?}, expected a saturated value",
    );
    for (index, value) in color.iter().enumerate() {
        if index != channel && index != 3 {
            assert!(
                *value <= 5,
                "probe channel {index} is {color:?}, expected it to stay dark",
            );
        }
    }
    assert_eq!(color[3], 255, "probe alpha is {color:?}, expected opaque");
}

#[test]
fn world_obb_render_binds_live_cubecl_storage() {
    let gpu = create_gpu();
    let limits = gpu.adapter.limits();
    let info = gpu.adapter.get_info();
    println!(
        "M0 3D render gate: backend={:?} adapter={} max_storage_buffers_per_shader_stage={} min_storage_buffer_offset_alignment={}",
        info.backend,
        info.name,
        limits.max_storage_buffers_per_shader_stage,
        limits.min_storage_buffer_offset_alignment,
    );
    assert!(
        limits.max_storage_buffers_per_shader_stage >= STORAGE_BUFFERS as u32,
        "adapter only exposes {} storage buffers per shader stage, the world-OBB pass needs {STORAGE_BUFFERS}",
        limits.max_storage_buffers_per_shader_stage,
    );

    let layout = storage_layout(&gpu.device);
    let pipeline = create_pipeline(&gpu.device, &layout);
    let targets = create_targets(&gpu.device);

    let mut handles = create_probe_handles(&gpu.client, RED_PROBE);
    let first_binding = bind_cubecl_storage(&gpu.device, &layout, &gpu.client, &handles);
    let first = render_frame(&gpu, &pipeline, &first_binding.bind_group, &targets);

    println!(
        "first pass (voxel_a=0x{RED_PROBE:08X}): centre RT0={:?} depth={}",
        first.color, first.depth,
    );
    assert_probe(first.color, 0);
    assert!(
        first.depth > 0.0 && first.depth < 1.0,
        "centre depth is {}, expected a hit strictly between 0 and 1",
        first.depth,
    );

    // A fresh allocation proves the new bind group follows the new CubeCL
    // buffer instead of a CPU copy or the previous allocation.
    let green_handle = gpu
        .client
        .create_from_slice(bytemuck::cast_slice(&[GREEN_PROBE]));
    handles[0] = green_handle;
    let second_binding = bind_cubecl_storage(&gpu.device, &layout, &gpu.client, &handles);
    drop(first_binding);

    let second = render_frame(&gpu, &pipeline, &second_binding.bind_group, &targets);
    println!(
        "second pass (voxel_a=0x{GREEN_PROBE:08X}): centre RT0={:?} depth={}",
        second.color, second.depth,
    );
    assert_probe(second.color, 1);
    assert!(
        second.depth > 0.0 && second.depth < 1.0,
        "centre depth is {}, expected a hit strictly between 0 and 1",
        second.depth,
    );
}
