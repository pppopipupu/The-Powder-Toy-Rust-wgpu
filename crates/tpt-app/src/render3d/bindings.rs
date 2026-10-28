//! Bind-group construction and the CubeCL -> wgpu bridge for the 3D renderer.
//!
//! The world, palette and rigid-body buffers are owned by CubeCL. Instead of
//! copying them, the renderer resolves each [`Handle`] with
//! `Client::get_resource::<WgpuServer<AutoCompiler>>` and binds the underlying
//! `wgpu::Buffer` slice directly, exactly like `gpu_ctx::FramebufBlit` does
//! (plan section 9.2). The bind group holds the wgpu buffer alive for the
//! lifetime of the bind group.
//!
//! Binding budget (plan section 8.5):
//!
//! | binding | resource |
//! |---|---|
//! | 0 | `voxel_a: array<u32>` |
//! | 1 | `leaf_masks: array<u32>` |
//! | 2 | `coarse_masks: array<u32>` |
//! | 3 | `palette: array<vec4<f32>>` |
//! | 4 | `body_desc: array<vec4<f32>>` |
//! | 5 | `body_voxels: array<u32>` |
//! | 6 | `body_masks: array<u32>` |
//! | 7 | `SceneUniforms` |

use std::mem::size_of;
use std::num::NonZeroU64;

use cubecl::client::Client;
use cubecl::server::Handle;
use cubecl_wgpu::{AutoCompiler, WgpuServer};
use tpt_core::constants::{PT_LAVA, PT_PLSM};
use tpt_core::voxel3d::{VOXEL_ELEMENTS, VoxelPhase, is_translucent};
use tpt_sim::voxel3d::Voxel3dBuffers;

use super::camera::SceneUniforms;
use super::gbuffer::GBuffer;

/// Number of storage buffers bound by the scene bind group.
pub const STORAGE_BUFFER_BINDINGS: usize = 7;
/// Binding index of the scene uniform.
pub const UNIFORM_BINDING: u32 = 7;

/// Number of `vec4<f32>` words per palette entry.
pub const PALETTE_VEC4_PER_ENTRY: usize = 4;
/// Number of palette entries (one per PT element id).
pub const PALETTE_ENTRIES: usize = 256;

/// Palette flag: the element is a participating medium and is volume traced.
pub const PALETTE_FLAG_VOLUME: u32 = 1;
/// Palette flag: emission scales with the voxel life counter.
pub const PALETTE_FLAG_LIFE_EMISSION: u32 = 2;

/// Number of `vec4<f32>` words per rigid-body render description.
///
/// Layout per body: `model` (4), `inv_model` (4), `world_aabb_min` (1),
/// `world_aabb_max` (1), `grid` (1: offset, dims xyz as bitcasts), `meta`
/// (1: element base, flags).
pub const BODY_DESC_VEC4_PER_BODY: usize = 12;
/// Number of `u32` mask words per rigid body (16^3 leaf bits).
pub const BODY_MASK_WORDS_PER_BODY: usize = 128;

/// Maximum number of rigid bodies the renderer traverses.
pub const MAX_BODIES: usize = 64;

/// Whether the shader's `palette_emission` must scale with the voxel life.
///
/// C++ drives FIRE, PLSM and LAVA colour from `life` (`FIRE.cpp` graphics,
/// `PLSM.cpp:graphics` which samples `plasmaTableAt(cpart->life)`, and
/// `LAVA.cpp:44-56` which scales all three channels by `cpart->life`), so all
/// three keep the life-modulated emission; every other element uses its
/// constant emission term.
fn palette_life_emission(id: u32, phase: VoxelPhase) -> bool {
    phase == VoxelPhase::Fire || id == PT_PLSM || id == PT_LAVA
}

/// Builds the palette upload words.
///
/// Every entry is four `vec4<f32>`: `(albedo.rgb, flags)`, `(sigma,
/// emission.rgb)`, and two reserved rows for roughness/metalness in a later
/// milestone. The albedo, sigma, emission and volume flags come directly from
/// [`VOXEL_ELEMENTS`], so the renderer cannot drift from the simulation table.
/// Unused PT ids default to a neutral, non-emissive grey.
pub fn palette_words() -> Vec<f32> {
    let stride = PALETTE_VEC4_PER_ENTRY * 4;
    let mut words = vec![0.0f32; PALETTE_ENTRIES * stride];
    for entry in 0..PALETTE_ENTRIES {
        let base = entry * stride;
        words[base] = 0.5;
        words[base + 1] = 0.5;
        words[base + 2] = 0.5;
        words[base + 3] = 0.0;
    }
    for element in &VOXEL_ELEMENTS {
        let base = element.id as usize * stride;
        words[base] = element.albedo[0];
        words[base + 1] = element.albedo[1];
        words[base + 2] = element.albedo[2];
        let mut flags = 0u32;
        if is_translucent(element.id) {
            flags |= PALETTE_FLAG_VOLUME;
        }
        if palette_life_emission(element.id, element.phase) {
            flags |= PALETTE_FLAG_LIFE_EMISSION;
        }
        words[base + 3] = flags as f32;
        words[base + 4] = element.sigma;
        words[base + 5] = element.emission[0];
        words[base + 6] = element.emission[1];
        words[base + 7] = element.emission[2];
    }
    words
}

/// Allocates and uploads the render palette as a CubeCL buffer.
pub fn palette_buffer(client: &Client) -> Handle {
    client.create_from_slice(bytemuck::cast_slice(&palette_words()))
}

/// Rigid-body render buffers read by the unified scene traversal.
///
/// The rigid-body solver writes these on the device; the renderer only binds
/// them. A caller without rigid state leaves the renderer without an override,
/// and the renderer binds small zeroed placeholders with `body_count == 0`.
#[derive(Clone, Debug)]
pub struct RigidRenderBuffers {
    /// `array<vec4<f32>>` render descriptions, [`BODY_DESC_VEC4_PER_BODY`] per body.
    pub body_desc: Handle,
    /// Voxel atlas, 32^3 words per body.
    pub body_voxels: Handle,
    /// Leaf masks, [`BODY_MASK_WORDS_PER_BODY`] words per body.
    pub body_masks: Handle,
    /// Number of live bodies, clamped to [`MAX_BODIES`].
    pub body_count: u32,
}

impl RigidRenderBuffers {
    /// Allocates the zeroed placeholder buffers bound while no rigid state is
    /// available.
    ///
    /// The placeholders are deliberately tiny: the shader never reads them
    /// while `body_count` is zero, and wgpu only requires a non-empty binding.
    pub fn placeholder(client: &Client) -> Self {
        let desc = vec![0.0f32; MAX_BODIES * BODY_DESC_VEC4_PER_BODY * 4];
        Self {
            body_desc: client.create_from_slice(bytemuck::cast_slice(&desc)),
            body_voxels: client.create_from_slice(bytemuck::cast_slice(&[0u32; 4])),
            body_masks: client.create_from_slice(bytemuck::cast_slice(&[0u32; 4])),
            body_count: 0,
        }
    }
}

/// Scene bind group plus the number of storage buffers actually bound.
pub struct SceneBinding {
    /// Bind group for `@group(0)` of both pipelines.
    pub bind_group: wgpu::BindGroup,
    /// Storage buffers bound by this group, always [`STORAGE_BUFFER_BINDINGS`].
    pub storage_buffers: usize,
}

/// Creates the scene bind-group layout (seven storages plus the uniform).
pub fn scene_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(STORAGE_BUFFER_BINDINGS + 1);
    for binding in 0..STORAGE_BUFFER_BINDINGS as u32 {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: UNIFORM_BINDING,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size_of::<SceneUniforms>() as u64),
        },
        count: None,
    });

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tpt-render3d-scene-layout"),
        entries: &entries,
    })
}

/// Creates the lighting pass's texture bind-group layout (`@group(1)`).
pub fn lighting_texture_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        texture_entry(1),
        texture_entry(2),
        texture_entry(3),
    ];
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tpt-render3d-lighting-textures"),
        entries: &entries,
    })
}

/// One `texture_2d<f32>` binding entry.
fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Binds the G-buffer views for the lighting pass.
pub fn lighting_texture_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    gbuffer: &GBuffer,
) -> wgpu::BindGroup {
    let entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&gbuffer.depth_view),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(&gbuffer.rt0_view),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(&gbuffer.rt1_view),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(&gbuffer.rt2_view),
        },
    ];
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tpt-render3d-lighting-textures"),
        layout,
        entries: &entries,
    })
}

/// Resolves the seven CubeCL buffers and binds them with the scene uniform.
///
/// Returns an error when CubeCL exposes an allocation whose offset does not
/// respect the device's storage-buffer alignment. That cannot happen with
/// `MemoryConfiguration::ExclusivePages` (the configuration the app and the
/// headless paths use), and is reported instead of silently sampling a shifted
/// world.
pub fn create_scene_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    world: &Voxel3dBuffers,
    palette: &Handle,
    rigid: &RigidRenderBuffers,
) -> Result<SceneBinding, String> {
    let handles = [
        &world.voxel_a,
        &world.leaf_masks,
        &world.coarse_masks,
        palette,
        &rigid.body_desc,
        &rigid.body_voxels,
        &rigid.body_masks,
    ];
    debug_assert_eq!(handles.len(), STORAGE_BUFFER_BINDINGS);

    let alignment = u64::from(device.limits().min_storage_buffer_offset_alignment).max(1);
    let mut resources = Vec::with_capacity(STORAGE_BUFFER_BINDINGS);

    for (binding, handle) in handles.iter().enumerate() {
        let managed = world
            .client
            .get_resource::<WgpuServer<AutoCompiler>>((*handle).clone())
            .map_err(|error| {
                format!("failed to acquire CubeCL buffer {binding} for the 3D renderer: {error:?}")
            })?;
        {
            let resource = managed.resource();
            if resource.offset % alignment != 0 {
                return Err(format!(
                    "CubeCL buffer {binding} has offset {} which is not aligned to the device's {alignment}-byte storage-buffer offset alignment",
                    resource.offset,
                ));
            }
            if resource.size < 4 {
                return Err(format!(
                    "CubeCL buffer {binding} exposes only {} bytes",
                    resource.size,
                ));
            }
        }
        resources.push(managed);
    }

    let mut entries: Vec<wgpu::BindGroupEntry<'_>> = resources
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

    entries.push(wgpu::BindGroupEntry {
        binding: UNIFORM_BINDING,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: uniform,
            offset: 0,
            size: NonZeroU64::new(size_of::<SceneUniforms>() as u64),
        }),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tpt-render3d-scene"),
        layout,
        entries: &entries,
    });

    Ok(SceneBinding {
        bind_group,
        storage_buffers: STORAGE_BUFFER_BINDINGS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_covers_every_core_element() {
        let words = palette_words();
        let stride = PALETTE_VEC4_PER_ENTRY * 4;
        assert_eq!(words.len(), PALETTE_ENTRIES * stride);
        assert_eq!(VOXEL_ELEMENTS.len(), 59);
        for element in &VOXEL_ELEMENTS {
            let base = element.id as usize * stride;
            assert_eq!(words[base], element.albedo[0], "{} red", element.name);
            assert_eq!(words[base + 1], element.albedo[1], "{} green", element.name);
            assert_eq!(words[base + 2], element.albedo[2], "{} blue", element.name);
            assert_eq!(words[base + 4], element.sigma, "{} sigma", element.name);
            assert_eq!(
                words[base + 5],
                element.emission[0],
                "{} emission r",
                element.name
            );
            assert_eq!(
                words[base + 6],
                element.emission[1],
                "{} emission g",
                element.name
            );
            assert_eq!(
                words[base + 7],
                element.emission[2],
                "{} emission b",
                element.name
            );
            // The palette must have overwritten the neutral grey default.
            assert!(
                words[base] != 0.5 || words[base + 1] != 0.5 || words[base + 2] != 0.5,
                "{} kept the default palette albedo",
                element.name
            );
        }
    }

    #[test]
    fn volume_flags_follow_the_core_translucent_set() {
        let words = palette_words();
        let stride = PALETTE_VEC4_PER_ENTRY * 4;
        let flags = |ty: u32| words[ty as usize * stride + 3] as u32;
        for element in &VOXEL_ELEMENTS {
            let element_flags = flags(element.id);
            assert_eq!(
                element_flags & PALETTE_FLAG_VOLUME != 0,
                is_translucent(element.id),
                "{} volume flag",
                element.name
            );
            assert_eq!(
                element_flags & PALETTE_FLAG_LIFE_EMISSION != 0,
                palette_life_emission(element.id, element.phase),
                "{} life-emission flag",
                element.name
            );
        }
        // FIRE, PLSM and LAVA modulate emission by life (`FIRE.cpp` graphics,
        // `PLSM.cpp:graphics`, `LAVA.cpp:44-56`); no other 3D row does.
        assert_ne!(
            flags(tpt_core::constants::PT_FIRE) & PALETTE_FLAG_LIFE_EMISSION,
            0
        );
        assert_ne!(flags(PT_PLSM) & PALETTE_FLAG_LIFE_EMISSION, 0);
        assert_ne!(flags(PT_LAVA) & PALETTE_FLAG_LIFE_EMISSION, 0);
        // Spot checks for the frozen renderer contract.
        for id in [
            tpt_core::constants::PT_WATR,
            tpt_core::constants::PT_OIL,
            tpt_core::constants::PT_WTRV,
            tpt_core::constants::PT_SMKE,
            tpt_core::constants::PT_FIRE,
        ] {
            assert_eq!(flags(id) & PALETTE_FLAG_VOLUME, PALETTE_FLAG_VOLUME);
        }
        for id in [
            tpt_core::constants::PT_SAND,
            tpt_core::constants::PT_LAVA,
            tpt_core::constants::PT_STNE,
            tpt_core::constants::PT_ICEI,
            tpt_core::constants::PT_METL,
            tpt_core::constants::PT_GUNP,
        ] {
            assert_eq!(flags(id) & PALETTE_FLAG_VOLUME, 0);
        }
    }
}
