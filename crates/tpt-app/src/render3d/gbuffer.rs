//! G-buffer targets of the 3D renderer (plan section 8.4).
//!
//! Four attachments are allocated at the render resolution:
//! RT0 `Rgba8Unorm` (albedo + element type), RT1/RT2 `Rgba16Float` (world
//! normal + voxel life, volume inscatter + transmittance) and `Depth32Float`.
//! The lighting pass decodes the type from `RT0.a` and the life from `RT1.w`
//! so life-modulated emission matches the volume path. They
//! are reallocated by [`GBuffer::resize`] when the viewport or the render scale
//! changes; RT0 and depth additionally carry `COPY_SRC` so the offscreen test
//! can read the rendered frame back without a second target.

/// Albedo/flag target format.
pub const RT0_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Normal/life target format.
pub const RT1_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Inscatter/transmittance target format.
pub const RT2_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Depth attachment format.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The G-buffer textures and their views.
pub struct GBuffer {
    /// Render resolution in pixels.
    size: (u32, u32),
    /// Albedo and material flags.
    pub rt0: wgpu::Texture,
    /// World normal and element type.
    pub rt1: wgpu::Texture,
    /// Volume inscatter (rgb) and transmittance (a).
    pub rt2: wgpu::Texture,
    /// Depth written through `@builtin(frag_depth)`.
    pub depth: wgpu::Texture,
    /// View of [`Self::rt0`].
    pub rt0_view: wgpu::TextureView,
    /// View of [`Self::rt1`].
    pub rt1_view: wgpu::TextureView,
    /// View of [`Self::rt2`].
    pub rt2_view: wgpu::TextureView,
    /// View of [`Self::depth`].
    pub depth_view: wgpu::TextureView,
}

impl GBuffer {
    /// Allocates a G-buffer of `width` x `height` pixels.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);

        let rt0 = create_texture(
            device,
            "tpt-render3d-rt0",
            width,
            height,
            RT0_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let rt1 = create_texture(
            device,
            "tpt-render3d-rt1",
            width,
            height,
            RT1_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let rt2 = create_texture(
            device,
            "tpt-render3d-rt2",
            width,
            height,
            RT2_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let depth = create_texture(
            device,
            "tpt-render3d-depth",
            width,
            height,
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );

        let rt0_view = rt0.create_view(&wgpu::TextureViewDescriptor::default());
        let rt1_view = rt1.create_view(&wgpu::TextureViewDescriptor::default());
        let rt2_view = rt2.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            size: (width, height),
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

    /// Reallocates the targets when the size changes.
    ///
    /// Returns `true` when the textures were recreated; the lighting bind group
    /// must then be rebuilt because it references the old views.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let width = width.max(1);
        let height = height.max(1);
        if (width, height) == self.size {
            return false;
        }
        *self = Self::new(device, width, height);
        true
    }

    /// Render resolution in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Width of the render targets in pixels.
    pub fn width(&self) -> u32 {
        self.size.0
    }

    /// Height of the render targets in pixels.
    pub fn height(&self) -> u32 {
        self.size.1
    }

    /// Aspect ratio used by the camera and the lighting pass.
    pub fn aspect(&self) -> f32 {
        self.size.0 as f32 / self.size.1 as f32
    }
}

/// Creates one render target texture.
fn create_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}
