//! Host description of one 3D rigid body.
//!
//! A [`RigidBody3d`] is only the *spawn-time* description of a body: it owns
//! the 32^3 local voxel grid, the initial pose and the precomputed mass
//! properties. Once [`super::Rigid3dState::spawn_box`] has uploaded it, the
//! device copies in `body_state` / `body_voxels` / `body_surface` are
//! authoritative and the host description is never read again by the
//! steady-state solver (plan §7.5).
//!
//! Local grid coordinates run over `0..32` per axis. The rigid-body frame
//! origin is the grid centre, so voxel `i` spans `[i, i + 1)` in local space
//! and the local-to-world transform written into `body_desc` is
//! `world = pos + R * (local - 16)`. That is also the convention the renderer
//! uses when it runs its local DDA over `[0, 32]^3`.

use tpt_core::voxel3d::{BODY_GRID, BODY_VOXELS};

/// Mass of one solid body voxel.
///
/// The v1 rigid element set is `STONE/WOOD/ICE/METAL`; all four use the same
/// density so the value is a constant rather than an element-table lookup.
pub const VOXEL_DENSITY: f32 = 1.0;

/// Local grid centre in voxel units.
pub const GRID_CENTRE: f32 = 16.0;

/// Number of 2^3 leaf blocks along one local body axis.
const LOCAL_BLOCKS_PER_AXIS: usize = 16;

/// Bit mask of the element id in a packed voxel word (plan §3.2).
const VOXEL_TYPE_MASK: u32 = 0xFF;

/// One rigid body as it exists on the host before upload.
#[derive(Clone, Debug)]
pub struct RigidBody3d {
    /// World-space centre of mass.
    pub pos: [f32; 3],
    /// Orientation quaternion `(x, y, z, w)`.
    pub quat: [f32; 4],
    /// Linear velocity in world voxels per second.
    pub lin_vel: [f32; 3],
    /// Angular velocity in radians per second (world space).
    pub ang_vel: [f32; 3],
    /// Half extents of the materialised box in voxels.
    pub half_extents: [u32; 3],
    /// Packed local 32^3 voxel grid, `(y << 10) | (z << 5) | x`.
    pub local: Vec<u32>,
    /// Element id of every solid voxel.
    pub ty: u32,
    /// Reciprocal of the total mass.
    pub inv_mass: f32,
    /// Reciprocal of the diagonal local inertia tensor.
    pub inv_inertia: [f32; 3],
    /// Tight local AABB minimum, in grid coordinates.
    pub local_min: [f32; 3],
    /// Tight local AABB maximum, in grid coordinates.
    pub local_max: [f32; 3],
}

impl RigidBody3d {
    /// Materialises an axis-aligned box of solid voxels.
    ///
    /// `half_extents` is the number of voxels from the box centre to each face,
    /// so `2 * half_extents` voxels are written per axis. Values are clamped to
    /// `1..=16`: the local grid only has 16 voxels on either side of its
    /// centre. `center` is clamped so the complete box stays inside the 256^3
    /// world.
    pub fn new_box(center: [f32; 3], half_extents: [u32; 3], ty: u32) -> Self {
        let hx = half_extents[0].clamp(1, 16);
        let hy = half_extents[1].clamp(1, 16);
        let hz = half_extents[2].clamp(1, 16);
        let half = [hx, hy, hz];

        let mut local = vec![0u32; BODY_VOXELS];
        let x0 = 16 - hx;
        let y0 = 16 - hy;
        let z0 = 16 - hz;
        for z in z0..(z0 + 2 * hz) {
            for y in y0..(y0 + 2 * hy) {
                for x in x0..(x0 + 2 * hx) {
                    let index = ((y << 10) | (z << 5) | x) as usize;
                    local[index] = ty & VOXEL_TYPE_MASK;
                }
            }
        }

        let count = (2 * hx) * (2 * hy) * (2 * hz);
        let mass = count as f32 * VOXEL_DENSITY;
        // Solid box inertia about the centre: I = m / 3 * (h_j^2 + h_k^2)
        // when `h` is a half extent (the full side is `2h`).
        let fx = hx as f32;
        let fy = hy as f32;
        let fz = hz as f32;
        let inertia = [
            mass / 3.0 * (fy * fy + fz * fz),
            mass / 3.0 * (fx * fx + fz * fz),
            mass / 3.0 * (fx * fx + fy * fy),
        ];
        let mut inv_inertia = [0.0f32; 3];
        for axis in 0..3 {
            if inertia[axis] > 0.0 {
                inv_inertia[axis] = 1.0 / inertia[axis];
            }
        }

        let mut pos = center;
        for axis in 0..3 {
            let extent = half[axis] as f32;
            let limit = 256.0 - extent;
            if pos[axis] < extent {
                pos[axis] = extent;
            }
            if pos[axis] > limit {
                pos[axis] = limit;
            }
        }

        Self {
            pos,
            quat: [0.0, 0.0, 0.0, 1.0],
            lin_vel: [0.0; 3],
            ang_vel: [0.0; 3],
            half_extents: half,
            local,
            ty,
            inv_mass: 1.0 / mass,
            inv_inertia,
            local_min: [16.0 - fx, 16.0 - fy, 16.0 - fz],
            local_max: [16.0 + fx, 16.0 + fy, 16.0 + fz],
        }
    }

    /// Reads one local voxel word, or zero outside the grid.
    pub fn local_voxel(&self, x: u32, y: u32, z: u32) -> u32 {
        if x >= BODY_GRID || y >= BODY_GRID || z >= BODY_GRID {
            return 0;
        }
        self.local[((y << 10) | (z << 5) | x) as usize]
    }

    /// Local indices of every solid voxel that has an empty face.
    ///
    /// The contact passes only visit this list, so a large body costs its
    /// surface rather than its volume for every world contact.
    pub fn surface_indices(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for z in 0..BODY_GRID {
            for y in 0..BODY_GRID {
                for x in 0..BODY_GRID {
                    if self.local_voxel(x, y, z) == 0 {
                        continue;
                    }
                    let interior = self.local_voxel(x.wrapping_sub(1), y, z) != 0
                        && self.local_voxel(x + 1, y, z) != 0
                        && self.local_voxel(x, y.wrapping_sub(1), z) != 0
                        && self.local_voxel(x, y + 1, z) != 0
                        && self.local_voxel(x, y, z.wrapping_sub(1)) != 0
                        && self.local_voxel(x, y, z + 1) != 0;
                    if !interior {
                        out.push((y << 10) | (z << 5) | x);
                    }
                }
            }
        }
        out
    }

    /// Occupancy bits for the local 16^3 leaf-block grid.
    ///
    /// Layout matches the world masks: two `u32` per 8^3 brick in the world,
    /// but the body grid is only one 32^3 volume, so the 4096 bits are stored
    /// as 128 consecutive words with
    /// `block = ((y >> 1) << 8) | ((z >> 1) << 4) | (x >> 1)`.
    pub fn mask_words(&self) -> Vec<u32> {
        let mut words = vec![0u32; (LOCAL_BLOCKS_PER_AXIS.pow(3)) / 32];
        for z in 0..BODY_GRID {
            for y in 0..BODY_GRID {
                for x in 0..BODY_GRID {
                    if self.local_voxel(x, y, z) == 0 {
                        continue;
                    }
                    let block = (((y >> 1) as usize) << 8)
                        | (((z >> 1) as usize) << 4)
                        | ((x >> 1) as usize);
                    words[block >> 5] |= 1u32 << (block & 31);
                }
            }
        }
        words
    }

    /// Writes the 32-word `body_state` record at `base`.
    pub fn write_state(&self, words: &mut [f32], base: usize) {
        words[base] = self.pos[0];
        words[base + 1] = self.pos[1];
        words[base + 2] = self.pos[2];
        words[base + 3] = 1.0;
        words[base + 4] = self.quat[0];
        words[base + 5] = self.quat[1];
        words[base + 6] = self.quat[2];
        words[base + 7] = self.quat[3];
        words[base + 8] = self.lin_vel[0];
        words[base + 9] = self.lin_vel[1];
        words[base + 10] = self.lin_vel[2];
        words[base + 11] = 0.0;
        words[base + 12] = self.ang_vel[0];
        words[base + 13] = self.ang_vel[1];
        words[base + 14] = self.ang_vel[2];
        words[base + 15] = 0.0;
        words[base + 16] = self.inv_mass;
        words[base + 17] = self.inv_inertia[0];
        words[base + 18] = self.inv_inertia[1];
        words[base + 19] = self.inv_inertia[2];
        words[base + 20] = self.local_min[0];
        words[base + 21] = self.local_min[1];
        words[base + 22] = self.local_min[2];
        words[base + 23] = 0.0;
        words[base + 24] = self.local_max[0];
        words[base + 25] = self.local_max[1];
        words[base + 26] = self.local_max[2];
        words[base + 27] = 0.0;
        words[base + 28] = 0.0; // quiet tick accumulator
        words[base + 29] = 1.0; // awake
        let hx = self.half_extents[0] as f32;
        let hy = self.half_extents[1] as f32;
        let hz = self.half_extents[2] as f32;
        words[base + 30] = (hx * hx + hy * hy + hz * hz).sqrt() + 0.5;
        words[base + 31] = 0.0;
    }

    /// Writes the 40-float render description at `base`.
    ///
    /// The first 32 words are the column-major `model` and `inv_model`
    /// matrices, matching the `array<vec4<f32>>` binding in plan §8.5. The
    /// last 8 words are `world_aabb_min` and `world_aabb_max` (xyz plus a
    /// padding component).
    pub fn write_desc(&self, words: &mut [f32], base: usize) {
        let (r00, r10, r20) = rotate(self.quat, [1.0, 0.0, 0.0]);
        let (r01, r11, r21) = rotate(self.quat, [0.0, 1.0, 0.0]);
        let (r02, r12, r22) = rotate(self.quat, [0.0, 0.0, 1.0]);

        let tx = self.pos[0] - GRID_CENTRE * (r00 + r01 + r02);
        let ty = self.pos[1] - GRID_CENTRE * (r10 + r11 + r12);
        let tz = self.pos[2] - GRID_CENTRE * (r20 + r21 + r22);

        // model, column-major.
        words[base] = r00;
        words[base + 1] = r10;
        words[base + 2] = r20;
        words[base + 3] = 0.0;
        words[base + 4] = r01;
        words[base + 5] = r11;
        words[base + 6] = r21;
        words[base + 7] = 0.0;
        words[base + 8] = r02;
        words[base + 9] = r12;
        words[base + 10] = r22;
        words[base + 11] = 0.0;
        words[base + 12] = tx;
        words[base + 13] = ty;
        words[base + 14] = tz;
        words[base + 15] = 1.0;

        // inv_model = R^T, translation 16 - R^T * pos.
        let ux = 16.0 - (r00 * self.pos[0] + r01 * self.pos[1] + r02 * self.pos[2]);
        let uy = 16.0 - (r10 * self.pos[0] + r11 * self.pos[1] + r12 * self.pos[2]);
        let uz = 16.0 - (r20 * self.pos[0] + r21 * self.pos[1] + r22 * self.pos[2]);
        words[base + 16] = r00;
        words[base + 17] = r01;
        words[base + 18] = r02;
        words[base + 19] = 0.0;
        words[base + 20] = r10;
        words[base + 21] = r11;
        words[base + 22] = r12;
        words[base + 23] = 0.0;
        words[base + 24] = r20;
        words[base + 25] = r21;
        words[base + 26] = r22;
        words[base + 27] = 0.0;
        words[base + 28] = ux;
        words[base + 29] = uy;
        words[base + 30] = uz;
        words[base + 31] = 1.0;

        let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
        let (mut max_x, mut max_y, mut max_z) = (f32::MIN, f32::MIN, f32::MIN);
        for corner in 0..8usize {
            let lx = if corner & 1 == 0 {
                self.local_min[0]
            } else {
                self.local_max[0]
            };
            let ly = if corner & 2 == 0 {
                self.local_min[1]
            } else {
                self.local_max[1]
            };
            let lz = if corner & 4 == 0 {
                self.local_min[2]
            } else {
                self.local_max[2]
            };
            let wx = r00 * lx + r01 * ly + r02 * lz + tx;
            let wy = r10 * lx + r11 * ly + r12 * lz + ty;
            let wz = r20 * lx + r21 * ly + r22 * lz + tz;
            min_x = min_x.min(wx);
            min_y = min_y.min(wy);
            min_z = min_z.min(wz);
            max_x = max_x.max(wx);
            max_y = max_y.max(wy);
            max_z = max_z.max(wz);
        }
        words[base + 32] = min_x;
        words[base + 33] = min_y;
        words[base + 34] = min_z;
        words[base + 35] = 1.0;
        words[base + 36] = max_x;
        words[base + 37] = max_y;
        words[base + 38] = max_z;
        words[base + 39] = 1.0;
    }
}

/// Rotates `v` by the unit quaternion `q = (x, y, z, w)`.
pub fn rotate(q: [f32; 4], v: [f32; 3]) -> (f32, f32, f32) {
    let (qx, qy, qz, qw) = (q[0], q[1], q[2], q[3]);
    let tx = 2.0 * (qy * v[2] - qz * v[1]);
    let ty = 2.0 * (qz * v[0] - qx * v[2]);
    let tz = 2.0 * (qx * v[1] - qy * v[0]);
    (
        v[0] + qw * tx + (qy * tz - qz * ty),
        v[1] + qw * ty + (qz * tx - qx * tz),
        v[2] + qw * tz + (qx * ty - qy * tx),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_grid_and_mass_are_centred() {
        let body = RigidBody3d::new_box([64.0, 10.0, 64.0], [2, 2, 2], 5);
        assert_eq!(body.local.iter().filter(|word| **word != 0).count(), 64);
        assert_eq!(body.local_min, [14.0, 14.0, 14.0]);
        assert_eq!(body.local_max, [18.0, 18.0, 18.0]);
        assert!((body.inv_mass - 1.0 / 64.0).abs() < 1.0e-6);
    }

    #[test]
    fn identity_desc_maps_local_grid_to_world() {
        let body = RigidBody3d::new_box([64.0, 10.0, 64.0], [2, 2, 2], 5);
        let mut desc = vec![0.0f32; 40];
        body.write_desc(&mut desc, 0);
        assert_eq!(&desc[32..36], &[62.0, 8.0, 62.0, 1.0]);
        assert_eq!(&desc[36..40], &[66.0, 12.0, 66.0, 1.0]);
        // Local (16, 16, 16) is the body centre.
        let wx = desc[0] * 16.0 + desc[4] * 16.0 + desc[8] * 16.0 + desc[12];
        let wy = desc[1] * 16.0 + desc[5] * 16.0 + desc[9] * 16.0 + desc[13];
        let wz = desc[2] * 16.0 + desc[6] * 16.0 + desc[10] * 16.0 + desc[14];
        assert!((wx - 64.0).abs() < 1.0e-5);
        assert!((wy - 10.0).abs() < 1.0e-5);
        assert!((wz - 64.0).abs() < 1.0e-5);
    }
}
