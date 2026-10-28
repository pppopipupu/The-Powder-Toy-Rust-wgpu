//! Host-side 3D brush bounds and membership. The GPU edit pass mirrors these
//! tests exactly.
//!
//! Coordinates are integer voxel coordinates in `0..VOXEL_DIM`; the brush
//! centre is a floating-point world position and each radius is measured in
//! voxels along one axis. Radii may be zero on an axis: the brush then has no
//! thickness there, so an ellipsoid becomes a disc, a cuboid a slab and a
//! prism a line inside its X-Y cross-section. A brush whose three radii are
//! all zero covers no voxels, matching the historical sphere behaviour of
//! `radius == 0`; negative or non-finite radii are empty as well.

use super::VOXEL_DIM;

/// Inclusive AABB returned by [`Brush3d::bounds`] for a brush that covers no
/// voxels. `min > max` on every axis, so an inclusive loop executes zero
/// iterations while both corners still fit the world bounds.
const EMPTY_BOUNDS: ([u32; 3], [u32; 3]) = ([1, 1, 1], [0, 0, 0]);

/// Largest valid world coordinate on one axis.
const MAX_COORD: f32 = (VOXEL_DIM - 1) as f32;

/// Cross-section of a [`Brush3d`].
///
/// The discriminant is part of the device contract: the edit kernels compare
/// against these codes and pin them with compile-time assertions.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[repr(u32)]
pub enum BrushShape3d {
    /// Three-axis ellipsoid with `radii` as semi-axes.
    #[default]
    Ellipsoid = 0,
    /// Axis-aligned box with `radii` as half-extents.
    Cuboid = 1,
    /// Triangular X-Y cross-section extruded over Z, with the apex at `+Y`
    /// and the base at `-Y`. The X-Y triangle is clipped to `radii[0]` by
    /// `radii[1]`, and `radii[2]` is the extrusion half-length.
    Prism = 2,
}

impl BrushShape3d {
    /// Device-side discriminant used by the edit kernels.
    #[inline]
    pub const fn code(self) -> u32 {
        self as u32
    }
}

/// A 3-axis brush in voxel space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Brush3d {
    /// Centre in voxel coordinates.
    pub center: [f32; 3],
    /// Per-axis radius in voxels.
    pub radii: [f32; 3],
    /// Cross-section of the brush.
    pub shape: BrushShape3d,
}

impl Brush3d {
    /// Creates an isotropic ellipsoid centred on `center` with the given voxel
    /// radius.
    #[inline]
    pub fn new(center: [f32; 3], radius: f32) -> Self {
        Self {
            center,
            radii: [radius; 3],
            shape: BrushShape3d::Ellipsoid,
        }
    }

    /// Creates a brush with explicit per-axis radii and a cross-section.
    #[inline]
    pub fn with_shape(center: [f32; 3], radii: [f32; 3], shape: BrushShape3d) -> Self {
        Self {
            center,
            radii,
            shape,
        }
    }

    /// Whether the brush can touch any voxel.
    ///
    /// Every radius must be finite and non-negative and at least one radius
    /// must be positive.
    #[inline]
    fn is_active(&self) -> bool {
        self.center.iter().all(|coord| coord.is_finite())
            && self
                .radii
                .iter()
                .all(|radius| radius.is_finite() && *radius >= 0.0)
            && self.radii.iter().any(|radius| *radius > 0.0)
    }

    /// Inclusive integer AABB of the brush, clamped to the world.
    ///
    /// Every shape is bounded by the box `center +/- radii`, so the lower
    /// corner is `ceil(center - radius)` and the upper corner is
    /// `floor(center + radius)` on each axis. A non-finite centre, a negative
    /// or non-finite radius, three zero radii, or a brush that misses the world
    /// returns [`EMPTY_BOUNDS`] (`min > max`).
    #[inline]
    pub fn bounds(&self) -> ([u32; 3], [u32; 3]) {
        if !self.is_active() {
            return EMPTY_BOUNDS;
        }

        let mut min = [0u32; 3];
        let mut max = [0u32; 3];
        for axis in 0..3 {
            let center = self.center[axis];
            let radius = self.radii[axis];
            let low = center - radius;
            let high = center + radius;
            if high < 0.0 || low > MAX_COORD {
                return EMPTY_BOUNDS;
            }
            // `ceil`/`floor` of finite values stay finite, and the clamps keep
            // the result inside the world even for huge radii.
            min[axis] = low.ceil().clamp(0.0, MAX_COORD) as u32;
            max[axis] = high.floor().clamp(0.0, MAX_COORD) as u32;
        }

        if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
            return EMPTY_BOUNDS;
        }
        (min, max)
    }

    /// Whether the integer voxel `(x, y, z)` lies inside the brush.
    ///
    /// Coordinates outside the world, and every query for an empty brush,
    /// return `false`.
    ///
    /// An axis with radius zero must be matched exactly, so the shape loses
    /// its thickness on that axis instead of rejecting the whole brush.
    #[inline]
    pub fn contains(&self, x: u32, y: u32, z: u32) -> bool {
        if !self.is_active() || x >= VOXEL_DIM || y >= VOXEL_DIM || z >= VOXEL_DIM {
            return false;
        }
        let dx = x as f32 - self.center[0];
        let dy = y as f32 - self.center[1];
        let dz = z as f32 - self.center[2];
        let [rx, ry, rz] = self.radii;
        match self.shape {
            BrushShape3d::Ellipsoid => {
                let mut sum = 0.0f32;
                let mut inside = true;
                if rx > 0.0 {
                    let nx = dx / rx;
                    sum += nx * nx;
                } else if dx != 0.0 {
                    inside = false;
                }
                if ry > 0.0 {
                    let ny = dy / ry;
                    sum += ny * ny;
                } else if dy != 0.0 {
                    inside = false;
                }
                if rz > 0.0 {
                    let nz = dz / rz;
                    sum += nz * nz;
                } else if dz != 0.0 {
                    inside = false;
                }
                inside && sum <= 1.0
            }
            BrushShape3d::Cuboid => dx.abs() <= rx && dy.abs() <= ry && dz.abs() <= rz,
            BrushShape3d::Prism => {
                let ax = dx.abs();
                ax <= rx && dy.abs() <= ry && dz.abs() <= rz && 2.0 * ax * ry + dy * rx <= rx * ry
            }
        }
    }

    /// Calls `visit` once for every voxel inside the brush, in voxel-index
    /// order (`x` fastest, then `z`, then `y`).
    ///
    /// An empty brush invokes the callback zero times.
    #[inline]
    pub fn for_each_voxel(&self, mut visit: impl FnMut([u32; 3])) {
        let (min, max) = self.bounds();
        if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
            return;
        }
        for y in min[1]..=max[1] {
            for z in min[2]..=max[2] {
                for x in min[0]..=max[0] {
                    if self.contains(x, y, z) {
                        visit([x, y, z]);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const CENTER: [f32; 3] = [10.0, 20.0, 30.0];

    #[test]
    fn shape_codes_match_the_device_contract() {
        assert_eq!(BrushShape3d::Ellipsoid.code(), 0);
        assert_eq!(BrushShape3d::Cuboid.code(), 1);
        assert_eq!(BrushShape3d::Prism.code(), 2);
    }

    #[test]
    fn bounds_are_inclusive_and_clamped() {
        let brush = Brush3d::new(CENTER, 2.0);
        assert_eq!(brush.bounds(), ([8, 18, 28], [12, 22, 32]));

        let at_origin = Brush3d::new([0.0, 0.0, 0.0], 10.0);
        assert_eq!(at_origin.bounds(), ([0, 0, 0], [10, 10, 10]));

        let at_far_corner = Brush3d::new([255.0, 255.0, 255.0], 10.0);
        assert_eq!(at_far_corner.bounds(), ([245, 245, 245], [255, 255, 255]));

        let fractional = Brush3d::new([10.5, 10.5, 10.5], 1.0);
        assert_eq!(fractional.bounds(), ([10, 10, 10], [11, 11, 11]));

        let huge = Brush3d::new([128.0, 128.0, 128.0], 1000.0);
        assert_eq!(huge.bounds(), ([0, 0, 0], [255, 255, 255]));
    }

    #[test]
    fn containment_matches_the_sphere() {
        let brush = Brush3d::new(CENTER, 2.5);
        assert!(brush.contains(10, 20, 30));
        assert!(brush.contains(12, 20, 30));
        assert!(brush.contains(10, 20, 32));
        assert!(!brush.contains(13, 20, 30));
        assert!(!brush.contains(12, 22, 30));
        assert!(!brush.contains(10, 20, 33));

        // The world is closed at 255; coordinates outside it never match.
        assert!(!Brush3d::new([255.0, 255.0, 255.0], 10.0).contains(300, 255, 255));
        assert!(!brush.contains(256, 20, 30));

        let zero = Brush3d::new(CENTER, 0.0);
        assert!(!zero.contains(10, 20, 30));
        let negative = Brush3d::new(CENTER, -2.0);
        assert!(!negative.contains(10, 20, 30));
    }

    #[test]
    fn per_axis_radii_degrade_cleanly() {
        // Zero Z radius: the ellipsoid becomes an ellipse lying in z = 30.
        let disc = Brush3d::with_shape(CENTER, [2.5, 3.5, 0.0], BrushShape3d::Ellipsoid);
        assert_eq!(disc.bounds(), ([8, 17, 30], [12, 23, 30]));
        assert!(disc.contains(10, 20, 30));
        assert!(disc.contains(12, 20, 30));
        assert!(disc.contains(10, 17, 30));
        assert!(!disc.contains(10, 20, 31), "zero radius must not thicken");
        assert!(!disc.contains(13, 20, 30));
        assert!(!disc.contains(10, 16, 30));

        // Two zero radii: a line along X.
        let line = Brush3d::with_shape(CENTER, [2.5, 0.0, 0.0], BrushShape3d::Ellipsoid);
        assert_eq!(line.bounds(), ([8, 20, 30], [12, 20, 30]));
        assert!(line.contains(12, 20, 30));
        assert!(!line.contains(12, 21, 30));
        assert!(!line.contains(12, 20, 31));

        // All three radii zero is the historical empty brush.
        let empty = Brush3d::with_shape(CENTER, [0.0, 0.0, 0.0], BrushShape3d::Ellipsoid);
        assert!(empty.bounds().0[0] > empty.bounds().1[0]);
        assert!(!empty.contains(10, 20, 30));

        // A zero-radius axis on a fractional centre cannot hit an integer
        // voxel, so the whole brush is empty.
        let off = Brush3d::with_shape([10.0, 20.0, 30.5], [2.0, 2.0, 0.0], BrushShape3d::Cuboid);
        assert!(off.bounds().0[2] > off.bounds().1[2]);
        let mut count = 0;
        off.for_each_voxel(|_| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn cuboid_is_a_slab_with_zero_radius() {
        let slab = Brush3d::with_shape(CENTER, [2.0, 3.0, 0.0], BrushShape3d::Cuboid);
        assert_eq!(slab.bounds(), ([8, 17, 30], [12, 23, 30]));
        assert!(slab.contains(8, 17, 30));
        assert!(slab.contains(12, 23, 30));
        assert!(!slab.contains(12, 24, 30));
        assert!(!slab.contains(13, 20, 30));
        assert!(!slab.contains(10, 20, 31));
    }

    #[test]
    fn prism_has_apex_at_positive_y() {
        let prism = Brush3d::with_shape(CENTER, [2.0, 2.0, 2.0], BrushShape3d::Prism);
        assert_eq!(prism.bounds(), ([8, 18, 28], [12, 22, 32]));
        assert!(prism.contains(10, 22, 30), "apex");
        assert!(prism.contains(12, 18, 30), "base corner");
        assert!(prism.contains(10, 18, 32), "extruded base centre");
        assert!(prism.contains(10, 20, 28), "mid-height front face");
        assert!(!prism.contains(12, 20, 30), "sides taper toward the apex");
        assert!(!prism.contains(11, 22, 30), "the apex is one voxel wide");
        assert!(!prism.contains(13, 18, 30));
        assert!(!prism.contains(10, 20, 33));

        // Zero extrusion: the triangle is a flat plate.
        let plate = Brush3d::with_shape(CENTER, [2.0, 2.0, 0.0], BrushShape3d::Prism);
        assert!(plate.contains(12, 18, 30));
        assert!(!plate.contains(12, 18, 31));

        // Zero X radius collapses the triangle onto its centre line.
        let ridge = Brush3d::with_shape(CENTER, [0.0, 2.0, 1.0], BrushShape3d::Prism);
        assert!(ridge.contains(10, 18, 30));
        assert!(ridge.contains(10, 22, 30));
        assert!(!ridge.contains(11, 18, 30));
    }

    #[test]
    fn for_each_voxel_visits_exactly_the_contained_voxels() {
        let brush = Brush3d::new(CENTER, 2.5);
        let mut visited = Vec::new();
        brush.for_each_voxel(|voxel| visited.push(voxel));

        let mut expected = Vec::new();
        for y in 0..=40 {
            for z in 0..=40 {
                for x in 0..=15 {
                    if brush.contains(x, y, z) {
                        expected.push([x, y, z]);
                    }
                }
            }
        }
        assert_eq!(visited, expected);

        let unique: HashSet<[u32; 3]> = visited.iter().copied().collect();
        assert_eq!(unique.len(), visited.len());
        assert!(!visited.is_empty());
        let (min, max) = brush.bounds();
        assert!(visited.iter().all(|voxel| {
            (min[0]..=max[0]).contains(&voxel[0])
                && (min[1]..=max[1]).contains(&voxel[1])
                && (min[2]..=max[2]).contains(&voxel[2])
        }));
    }

    #[test]
    fn every_shape_visits_exactly_the_contained_voxels() {
        let shapes = [
            BrushShape3d::Ellipsoid,
            BrushShape3d::Cuboid,
            BrushShape3d::Prism,
        ];
        let radii = [
            [2.0, 3.0, 1.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 1.5],
            [1.5, 0.5, 0.5],
        ];
        for shape in shapes {
            for radius in radii {
                let brush = Brush3d::with_shape(CENTER, radius, shape);
                let (min, max) = brush.bounds();
                let mut visited = Vec::new();
                brush.for_each_voxel(|voxel| visited.push(voxel));

                let mut expected = Vec::new();
                for y in 14..=26 {
                    for z in 26..=34 {
                        for x in 4..=16 {
                            if brush.contains(x, y, z) {
                                expected.push([x, y, z]);
                            }
                        }
                    }
                }
                assert_eq!(visited, expected, "{shape:?} radii {radius:?}");

                if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
                    assert!(visited.is_empty(), "{shape:?} radii {radius:?}");
                } else {
                    assert!(visited.iter().all(|voxel| {
                        (min[0]..=max[0]).contains(&voxel[0])
                            && (min[1]..=max[1]).contains(&voxel[1])
                            && (min[2]..=max[2]).contains(&voxel[2])
                    }));
                }
            }
        }
    }

    #[test]
    fn empty_brushes_yield_nothing() {
        for radius in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let brush = Brush3d::new(CENTER, radius);
            let (min, max) = brush.bounds();
            assert!(
                min[0] > max[0] && min[1] > max[1] && min[2] > max[2],
                "radius {radius} must produce an empty range"
            );
            let mut count = 0;
            brush.for_each_voxel(|_| count += 1);
            assert_eq!(count, 0);
            assert!(!brush.contains(10, 20, 30));
        }

        // A single negative radius empties the brush even if the others are
        // positive: a negative semi-axis has no interior.
        let mixed = Brush3d::with_shape(CENTER, [2.0, -1.0, 2.0], BrushShape3d::Prism);
        assert!(mixed.bounds().0[0] > mixed.bounds().1[0]);
        assert!(!mixed.contains(10, 20, 30));

        // A sphere entirely outside the world is empty as well.
        let outside = Brush3d::new([-50.0, 10.0, 10.0], 10.0);
        let mut count = 0;
        outside.for_each_voxel(|_| count += 1);
        assert_eq!(count, 0);
        assert!(outside.bounds().0[0] > outside.bounds().1[0]);
    }
}
