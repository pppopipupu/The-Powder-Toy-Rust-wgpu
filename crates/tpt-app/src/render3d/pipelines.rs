//! Render pipelines of the 3D renderer (plan section 8.1).
//!
//! Two pipelines share one scene bind group:
//!
//! * the world OBB pipeline rasterises the 256^3 cube back faces and runs the
//!   hierarchical scene traversal in the fragment stage, writing three colour
//!   targets and `frag_depth`;
//! * the lighting pipeline is a fullscreen triangle that reconstructs world
//!   positions from the depth target, traces the shadow/AO rays and writes the
//!   tone-mapped frame into the presentation texture.
//!
//! The WGSL files are concatenated at pipeline creation because WGSL has no
//! include mechanism: `shaders/common.wgsl` provides the bindings, palette and
//! traversal helpers, and each entry file adds its stages.

use super::gbuffer::{DEPTH_FORMAT, RT0_FORMAT, RT1_FORMAT, RT2_FORMAT};

/// Shared shader source prepended to both modules.
pub const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
/// World OBB pass source.
pub const WORLD_OBB_SHADER: &str = include_str!("shaders/obb.wgsl");
/// Lighting pass source.
pub const LIGHTING_SHADER: &str = include_str!("shaders/lighting.wgsl");

/// Vertex entry point of the world OBB pass.
pub const WORLD_OBB_VERTEX: &str = "vs_world_obb";
/// Fragment entry point of the world OBB pass.
pub const WORLD_OBB_FRAGMENT: &str = "fs_world_obb";
/// Vertex entry point of the lighting pass.
pub const LIGHTING_VERTEX: &str = "vs_lighting";
/// Fragment entry point of the lighting pass.
pub const LIGHTING_FRAGMENT: &str = "fs_lighting";

/// Vertices of the world OBB (12 triangles).
pub const WORLD_OBB_VERTICES: u32 = 36;
/// Vertices of the fullscreen triangle.
pub const FULLSCREEN_VERTICES: u32 = 3;

/// Render pipelines of the 3D renderer.
pub struct RenderPipelines {
    /// World OBB + unified scene traversal.
    pub world_obb: wgpu::RenderPipeline,
    /// Deferred lighting and tone mapping into the presentation target.
    pub lighting: wgpu::RenderPipeline,
}

impl RenderPipelines {
    /// Builds all pipelines for `target_format` (the presentation texture).
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        scene_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let world_obb = create_world_obb_pipeline(device, scene_layout);
        let lighting =
            create_lighting_pipeline(device, target_format, scene_layout, lighting_layout);
        Self {
            world_obb,
            lighting,
        }
    }
}

/// Compiles one shader module from the shared prefix plus `stage`.
fn create_shader(device: &wgpu::Device, label: &str, stage: &str) -> wgpu::ShaderModule {
    let source = format!("{COMMON_SHADER}\n{stage}");
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// Builds the world OBB pipeline.
fn create_world_obb_pipeline(
    device: &wgpu::Device,
    scene_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = create_shader(device, "tpt-render3d-world-obb", WORLD_OBB_SHADER);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tpt-render3d-world-obb-layout"),
        bind_group_layouts: &[Some(scene_layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("tpt-render3d-world-obb"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(WORLD_OBB_VERTEX),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            // Exactly one world OBB is rasterised, and only its back faces:
            // the fragment stage traces the world and the rigid bodies.
            cull_mode: Some(wgpu::Face::Front),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            // `LessEqual` keeps the colour writes of rays that leave the world
            // with no opaque hit at the cleared depth; those pixels keep their
            // volume inscatter and the lighting pass selects the sky term.
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(WORLD_OBB_FRAGMENT),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: RT0_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: RT1_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: RT2_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Builds the fullscreen lighting pipeline.
fn create_lighting_pipeline(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    scene_layout: &wgpu::BindGroupLayout,
    lighting_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = create_shader(device, "tpt-render3d-lighting", LIGHTING_SHADER);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("tpt-render3d-lighting-layout"),
        bind_group_layouts: &[Some(scene_layout), Some(lighting_layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("tpt-render3d-lighting"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(LIGHTING_VERTEX),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(LIGHTING_FRAGMENT),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}
