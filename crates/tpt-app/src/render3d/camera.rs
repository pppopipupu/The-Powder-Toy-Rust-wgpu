//! Orbital camera and the scene uniform shared by the 3D shaders.
//!
//! The camera follows the plan's orbit model (section 8.8): `target`, `yaw`,
//! `pitch`, `distance` and a vertical field of view. The shader receives a
//! precomputed [`SceneUniforms`] block; nothing here reads world state, so the
//! uniform is rebuilt and uploaded every frame without a read-back.

use glam::camera::rh::{proj, view};
use glam::{Mat4, Vec3};

/// Planar near distance of the 3D camera.
pub const NEAR_PLANE: f32 = 1.0;
/// Planar far distance of the 3D camera.
pub const FAR_PLANE: f32 = 4096.0;

/// Default orbit target (the middle of the 256^3 world).
pub const DEFAULT_TARGET: Vec3 = Vec3::new(128.0, 96.0, 128.0);
/// Default yaw in radians (45 degrees).
pub const DEFAULT_YAW: f32 = std::f32::consts::FRAC_PI_4;
/// Default pitch in radians (25 degrees).
pub const DEFAULT_PITCH: f32 = 25.0 * std::f32::consts::PI / 180.0;
/// Default orbit distance in voxels.
pub const DEFAULT_DISTANCE: f32 = 420.0;
/// Default vertical field of view in radians (60 degrees).
pub const DEFAULT_FOV: f32 = 60.0 * std::f32::consts::PI / 180.0;
/// Pitch limit so the camera never looks exactly along the world up axis.
const PITCH_LIMIT: f32 = 89.0 * std::f32::consts::PI / 180.0;

/// Orbital camera used by the 3D viewport.
///
/// `yaw` rotates around the world up axis and `pitch` is the elevation of the
/// eye above the target, so a positive pitch looks down at the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// Point the camera orbits around, in world voxel units.
    pub target: Vec3,
    /// Rotation around the world up axis, in radians.
    pub yaw: f32,
    /// Elevation above the target, in radians, clamped to +/- 89 degrees.
    pub pitch: f32,
    /// Distance between the eye and the target, in voxels.
    pub distance: f32,
    /// Vertical field of view, in radians.
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: DEFAULT_TARGET,
            yaw: DEFAULT_YAW,
            pitch: DEFAULT_PITCH,
            distance: DEFAULT_DISTANCE,
            fov: DEFAULT_FOV,
        }
    }
}

impl Camera {
    /// Creates a camera from the plan's defaults with a custom target.
    pub fn looking_at(target: Vec3) -> Self {
        Self {
            target,
            ..Self::default()
        }
    }

    /// Clamps the pitch into the supported range.
    pub fn clamp_pitch(&mut self) {
        self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Unit vector from the target towards the eye.
    pub fn offset_direction(&self) -> Vec3 {
        let pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
        Vec3::new(
            pitch.cos() * self.yaw.sin(),
            pitch.sin(),
            pitch.cos() * self.yaw.cos(),
        )
    }

    /// World-space eye position.
    pub fn eye(&self) -> Vec3 {
        let distance = self.distance.max(0.001);
        self.target + self.offset_direction() * distance
    }

    /// Normalized view direction (from the eye towards the target).
    pub fn forward(&self) -> Vec3 {
        -self.offset_direction()
    }

    /// Camera right vector, matching the shader's basis.
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize_or_zero()
    }

    /// Camera up vector, matching the shader's basis.
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    /// Hemisphere view matrix (right-handed, `y` up).
    pub fn view(&self) -> Mat4 {
        view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// Right-handed perspective projection with a 0..1 depth range.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        proj::directx::perspective(
            self.fov.clamp(1.0e-3, std::f32::consts::PI - 1.0e-3),
            aspect.max(1.0e-6),
            NEAR_PLANE,
            FAR_PLANE,
        )
    }

    /// Combined view-projection matrix for `aspect = width / height`.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }
}

/// Lighting settings uploaded next to the camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderSettings {
    /// Direction towards the sun (normalized on upload).
    pub sun_dir: Vec3,
    /// Linear sun colour.
    pub sun_color: Vec3,
    /// Sun intensity multiplier.
    pub sun_intensity: f32,
    /// Linear ambient sky colour.
    pub ambient: Vec3,
    /// Ambient intensity multiplier.
    pub ambient_intensity: f32,
    /// Tone-mapping exposure.
    pub exposure: f32,
    /// Enables the single sun shadow ray.
    pub shadows: bool,
    /// Enables the two ambient-occlusion rays.
    pub ambient_occlusion: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            sun_dir: Vec3::new(0.45, 0.78, 0.35),
            sun_color: Vec3::new(1.0, 0.96, 0.88),
            sun_intensity: 3.2,
            ambient: Vec3::new(0.42, 0.55, 0.76),
            ambient_intensity: 0.55,
            exposure: 1.15,
            shadows: true,
            ambient_occlusion: true,
        }
    }
}

/// Uniform block shared by the world OBB and lighting pipelines.
///
/// The layout is 16-byte aligned and mirrored by `SceneUniforms` in
/// `shaders/common.wgsl`; the scalar tail of each `vec4` carries the values
/// that would otherwise need padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniforms {
    /// Camera view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Inverse of [`Self::view_proj`], used to rebuild world positions.
    pub inv_view_proj: [[f32; 4]; 4],
    /// `xyz` eye position, `w` = tan(fov / 2).
    pub eye: [f32; 4],
    /// `xyz` view direction, `w` = aspect ratio.
    pub forward: [f32; 4],
    /// `xyz` camera right, `w` = near plane.
    pub right: [f32; 4],
    /// `xyz` camera up, `w` = far plane.
    pub up: [f32; 4],
    /// `xyz` sun direction, `w` = exposure.
    pub sun_dir: [f32; 4],
    /// `rgb` ambient colour, `w` = intensity.
    pub ambient: [f32; 4],
    /// `rgb` sun colour, `w` = intensity.
    pub sun_color: [f32; 4],
    /// `xy` target size, `zw` reciprocal size.
    pub target_size: [f32; 4],
    /// `x` body count, `y` feature flags, `z`/`w` reserved.
    pub body: [f32; 4],
}

impl SceneUniforms {
    /// Builds the uniform for one frame.
    pub fn new(
        camera: &Camera,
        aspect: f32,
        size: (u32, u32),
        settings: &RenderSettings,
        body_count: u32,
    ) -> Self {
        let view_proj = camera.view_proj(aspect);
        let inv_view_proj = view_proj.inverse();
        let eye = camera.eye();
        let forward = camera.forward();
        let right = camera.right();
        let up = camera.up();
        let tan_half_fov = (camera.fov * 0.5).tan();
        let sun_dir = settings.sun_dir.normalize_or(Vec3::Y);

        let mut flags = 0u32;
        if settings.shadows {
            flags |= 1;
        }
        if settings.ambient_occlusion {
            flags |= 2;
        }

        let width = size.0.max(1) as f32;
        let height = size.1.max(1) as f32;

        Self {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv_view_proj.to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, tan_half_fov],
            forward: [forward.x, forward.y, forward.z, aspect.max(1.0e-6)],
            right: [right.x, right.y, right.z, NEAR_PLANE],
            up: [up.x, up.y, up.z, FAR_PLANE],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, settings.exposure.max(0.0)],
            ambient: [
                settings.ambient.x,
                settings.ambient.y,
                settings.ambient.z,
                settings.ambient_intensity.max(0.0),
            ],
            sun_color: [
                settings.sun_color.x,
                settings.sun_color.y,
                settings.sun_color.z,
                settings.sun_intensity.max(0.0),
            ],
            target_size: [width, height, 1.0 / width, 1.0 / height],
            body: [body_count as f32, flags as f32, 0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_and_forward_are_consistent() {
        let camera = Camera::default();
        let eye = camera.eye();
        let forward = camera.forward();
        assert!((camera.target - eye).normalize().dot(forward) > 0.999);
        assert!((eye - camera.target).length() - camera.distance < 1.0e-3);
        assert!(camera.pitch > 0.0);
        assert!(eye.y > camera.target.y);
    }

    #[test]
    fn view_proj_is_invertible() {
        let camera = Camera::default();
        let matrix = camera.view_proj(16.0 / 9.0);
        let inverse = matrix.inverse();
        let identity = matrix * inverse;
        assert!(identity.abs_diff_eq(Mat4::IDENTITY, 1.0e-4));
    }

    #[test]
    fn uniform_layout_is_16_byte_aligned() {
        assert_eq!(std::mem::size_of::<SceneUniforms>() % 16, 0);
        assert_eq!(std::mem::size_of::<SceneUniforms>(), 272);
    }
}
