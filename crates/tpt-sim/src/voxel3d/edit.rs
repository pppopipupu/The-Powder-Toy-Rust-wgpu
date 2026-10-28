//! Device-side voxel brush edits (plan §3.4/§6.4).
//!
//! [`launch_edit_brush`] paints or erases a [`Brush3d`] over the AABB from
//! [`Brush3d::bounds`], mirroring the host's `contains` test exactly so the
//! returned voxel count is the number of written voxels. Painting sets the
//! leaf bit, all six coarse bits, `brick_occupied` and `dirty_bricks` for every
//! voxel; erasing only writes the zero word and marks the brick dirty.
//! [`launch_adjust_temp`] rewrites only the fixed-point temperature field of
//! the non-empty voxels inside the brush.
//!
//! [`launch_edit_segment`] sweeps the same brush along a line. Ellipsoid and
//! prism sweeps test the voxel against the closest point on the segment in a
//! space normalised by the per-axis radii; a zero radius is clamped to
//! [`MIN_SWEEP_RADIUS`] so a flat brush keeps its zero thickness. A cuboid
//! sweep is the segment's expanded AABB in voxel space (an approximation: a
//! swept box is not a box, but a box gesture is the only consumer and it wants
//! the rectangular footprint).
//!
//! Erasing deliberately leaves the leaf/coarse bits set. Clearing a leaf bit
//! needs to know whether the other seven voxels of the 2³ block are still
//! occupied, and several brush threads can erase different voxels of the same
//! block concurrently, so the only race-free choice is to let the next mask
//! rebuild clear the bits from the authoritative state. The renderer treats a
//! stale "occupied" bit as "descend and find nothing", which is safe.
//!
//! Painting `SPRK` does not create a voxel: it follows `create_part`'s
//! replace path (`Simulation.cpp:1851-1874`), converting a resting conductor
//! in place and recording its type in `ctype`. Empty voxels are left alone,
//! so a spark drawn on the sky no longer decays into `METL`.

use cubecl::prelude::*;
use tpt_core::voxel3d::{Brush3d, BrushShape3d, TEMP_Q_SCALE, VOXEL_COUNT, VOXEL_DIM};

use super::Voxel3dEngine;
use super::step::{
    C_ELEM_TPT_PROPS, C_ELEM_WORDS, C_LIFE_MASK, C_LIFE_SHIFT, C_PT_BMTL, C_PT_BRMT, C_PT_METL,
    C_PT_NSCN, C_PT_PSCN, C_PT_SPRK, C_TEMP_MASK, C_TEMP_SHIFT, C_TPT_CONDUCTS, C_TY_MASK,
    brick_bit, brick_of_index, clampf, fabsf, hash_u32, mark_brick_bits, mark_leaf_coarse,
    mix_index, seed_f32, seed_u32,
};

/// Salt for the per-voxel `Create`-hook life roll.
const HOOK_SALT: u32 = 0x9E37_79B1u32;
/// `Create` hook: `FIRE::create`, life `rng 120..169` (`FIRE.cpp:421-424`).
pub(crate) const HOOK_FIRE: u32 = 1;
/// `Create` hook: `PLSM::create`, life `rng 50..199` (`PLSM.cpp:69-72`).
pub(crate) const HOOK_PLSM: u32 = 2;
/// `Create` hook: `LAVA::create`, life `rng 240..359` (`LAVA.cpp:80-83`).
pub(crate) const HOOK_LAVA: u32 = 3;

/// `Simulation.cpp:1871`: a spark placed on a conductor adds 10 K while the
/// resulting temperature stays below 673 K. Packed words store 1/8 K per unit,
/// so the kernel bodies use the literals 80 and 5384; these guards pin them.
const _: () = assert!((10.0f32 * TEMP_Q_SCALE) as u32 == 80u32);
const _: () = assert!((673.0f32 * TEMP_Q_SCALE) as u32 == 5384u32);

/// Threads per edit workgroup along each axis.
const CUBE: u32 = 8;
/// World edge length in voxels.
const DIM: u32 = 256;
/// Largest valid world coordinate on one axis.
const MAX_COORD: f32 = (VOXEL_DIM - 1) as f32;
/// Minimum radius used to normalise a zero-thickness sweep axis.
const MIN_SWEEP_RADIUS: f32 = 1.0e-3;
/// Fixed-point temperature range of one voxel word.
const TEMP_Q_MAX: i32 = 0xFFFF;

/// Device-side [`BrushShape3d::Cuboid`] discriminant, pinned below.
const SHAPE_CUBOID: u32 = 1;
/// Device-side [`BrushShape3d::Prism`] discriminant, pinned below.
const SHAPE_PRISM: u32 = 2;

const _: () = assert!(DIM == tpt_core::voxel3d::VOXEL_DIM);
const _: () = assert!(SHAPE_CUBOID == BrushShape3d::Cuboid as u32);
const _: () = assert!(SHAPE_PRISM == BrushShape3d::Prism as u32);

/// Packed word and auxiliary fields stamped by a paint edit.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PaintFields {
    /// Packed `ty | temp | life` word; `0` means erase.
    pub word: u32,
    /// TPT `ctype`.
    pub ctype: u32,
    /// TPT `tmp`.
    pub tmp: i32,
    /// TPT `tmp2`.
    pub tmp2: i32,
    /// TPT `tmp3`.
    pub tmp3: i32,
    /// Per-voxel `Create` life hook (`HOOK_*`).
    pub hook: u32,
}

/// Conductors heated by `create_part(PT_SPRK)` (`Simulation.cpp:1871`).
///
/// The 3D subset matches `step::electric::heats_with_spark`; `ETRD`, `NBLE`
/// and `IRON` are not part of the 59-element 3D table.
#[cube]
fn spark_heats(ty: u32) -> bool {
    let mut hot = seed_u32(ty, 0u32);
    if ty == C_PT_METL || ty == C_PT_BMTL || ty == C_PT_BRMT || ty == C_PT_PSCN || ty == C_PT_NSCN {
        hot = 1u32;
    }
    hot != 0u32
}

/// Paints or erases one voxel and refreshes the tile bits.
///
/// Painting sets the leaf bit, all six coarse bits, `brick_occupied` and
/// `dirty_bricks`; erasing only clears the word and marks the brick dirty.
#[cube]
#[allow(clippy::too_many_arguments)]
fn write_brush_voxel(
    voxel_a: &mut [u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    leaf_masks: &mut [Atomic<u32>],
    coarse_masks: &mut [Atomic<u32>],
    brick_occupied: &mut [Atomic<u32>],
    dirty_bricks: &mut [Atomic<u32>],
    x: u32,
    y: u32,
    z: u32,
    new_word: u32,
    new_ctype: u32,
    new_tmp: i32,
    new_tmp2: i32,
    new_tmp3: i32,
    hook: u32,
    table: &[u32],
) {
    let index = (y << 16u32) | (z << 8u32) | x;
    if new_word != 0u32 {
        if (new_word & C_TY_MASK) == C_PT_SPRK {
            // `Simulation.cpp:1851-1874`: drawing a spark never creates a
            // particle. It replaces an existing, resting conductor in place
            // and records that conductor in ctype; an empty voxel, a
            // non-conductor or a conductor with a live timer is left alone.
            let old = voxel_a[index as usize];
            let ty = old & C_TY_MASK;
            let life = (old >> C_LIFE_SHIFT) & C_LIFE_MASK;
            if ty != 0u32 && life == 0u32 {
                let props = table[(ty as usize) * C_ELEM_WORDS + C_ELEM_TPT_PROPS];
                if (props & C_TPT_CONDUCTS) != 0u32 {
                    let temp_q = (old >> C_TEMP_SHIFT) & C_TEMP_MASK;
                    let mut next_q = seed_u32(temp_q, temp_q);
                    if spark_heats(ty) && temp_q + 80u32 < 5384u32 {
                        next_q = temp_q + 80u32;
                    }
                    voxel_a[index as usize] = C_PT_SPRK
                        | ((next_q & C_TEMP_MASK) << C_TEMP_SHIFT)
                        | (4u32 << C_LIFE_SHIFT);
                    aux_ctype[index as usize] = ty;
                    mark_leaf_coarse(leaf_masks, coarse_masks, index);
                    mark_brick_bits(brick_occupied, dirty_bricks, brick_of_index(index));
                }
            }
        } else {
            let mut word = seed_u32(new_word, new_word);
            // TPT calls the element's `Create` hook for every new particle, so a
            // brush stroke of FIRE/PLSM/LAVA rolls a per-voxel life instead of
            // stamping one value (`create_part` -> `elements[t].Create`).
            if hook == 1u32 {
                let roll = hash_u32(mix_index(index, HOOK_SALT) ^ 1u32) % 50u32;
                word = (word & 0x00FF_FFFFu32) | ((120u32 + roll) << 24u32);
            } else if hook == 2u32 {
                let roll = hash_u32(mix_index(index, HOOK_SALT) ^ 2u32) % 150u32;
                word = (word & 0x00FF_FFFFu32) | ((50u32 + roll) << 24u32);
            } else if hook == 3u32 {
                let roll = hash_u32(mix_index(index, HOOK_SALT) ^ 3u32) % 120u32;
                word = (word & 0x00FF_FFFFu32) | ((240u32 + roll) << 24u32);
            }
            voxel_a[index as usize] = word;
            aux_ctype[index as usize] = new_ctype;
            aux_tmp[index as usize] = new_tmp;
            aux_tmp2[index as usize] = new_tmp2;
            aux_tmp3[index as usize] = new_tmp3;
            mark_leaf_coarse(leaf_masks, coarse_masks, index);
            mark_brick_bits(brick_occupied, dirty_bricks, brick_of_index(index));
        }
    } else {
        voxel_a[index as usize] = 0u32;
        aux_ctype[index as usize] = 0u32;
        aux_tmp[index as usize] = 0i32;
        aux_tmp2[index as usize] = 0i32;
        aux_tmp3[index as usize] = 0i32;
        let (word, mask) = brick_bit(brick_of_index(index));
        dirty_bricks[word as usize].fetch_or(mask);
    }
}

/// Squared normalised offset of one axis, zero for a zero radius.
#[cube]
fn axis_norm(d: f32, r: f32) -> f32 {
    let mut out = seed_f32(d, 0.0f32);
    if r > 0.0f32 {
        let n = d / r;
        out = n * n;
    }
    out
}

/// Whether one axis keeps a voxel inside the brush: a zero radius must be
/// matched exactly.
#[cube]
fn axis_inside(d: f32, r: f32) -> bool {
    r > 0.0f32 || d == 0.0f32
}

/// Ellipsoid/cuboid/prism membership of one voxel offset.
///
/// Mirrors [`Brush3d::contains`] operation for operation; the unit tests
/// compare this formula with the host brush over a grid. The three predicates
/// are evaluated without a branch and selected by the shape code.
#[cube]
#[allow(clippy::too_many_arguments)]
fn brush_hit(dx: f32, dy: f32, dz: f32, rx: f32, ry: f32, rz: f32, shape: u32) -> bool {
    let ax = fabsf(dx);
    let ay = fabsf(dy);
    let az = fabsf(dz);
    let ellipsoid = axis_inside(dx, rx)
        && axis_inside(dy, ry)
        && axis_inside(dz, rz)
        && axis_norm(dx, rx) + axis_norm(dy, ry) + axis_norm(dz, rz) <= 1.0f32;
    let cuboid = ax <= rx && ay <= ry && az <= rz;
    let prism = cuboid && 2.0f32 * ax * ry + dy * rx <= rx * ry;
    (shape == 0u32 && ellipsoid) || (shape == 1u32 && cuboid) || (shape == 2u32 && prism)
}

/// Inclusive brush test over one AABB, mirroring [`Brush3d::contains`].
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn edit_brush_kernel(
    voxel_a: &mut [u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    leaf_masks: &mut [Atomic<u32>],
    coarse_masks: &mut [Atomic<u32>],
    brick_occupied: &mut [Atomic<u32>],
    dirty_bricks: &mut [Atomic<u32>],
    min_x: u32,
    min_y: u32,
    min_z: u32,
    max_x: u32,
    max_y: u32,
    max_z: u32,
    cx: f32,
    cy: f32,
    cz: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    shape: u32,
    new_word: u32,
    new_ctype: u32,
    new_tmp: i32,
    new_tmp2: i32,
    new_tmp3: i32,
    hook: u32,
    table: &[u32],
) {
    let x = min_x + CUBE_POS_X * 8u32 + UNIT_POS_X;
    let y = min_y + CUBE_POS_Y * 8u32 + UNIT_POS_Y;
    let z = min_z + CUBE_POS_Z * 8u32 + UNIT_POS_Z;
    if x <= max_x && y <= max_y && z <= max_z {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let dz = z as f32 - cz;
        if brush_hit(dx, dy, dz, rx, ry, rz, shape) {
            write_brush_voxel(
                voxel_a,
                aux_ctype,
                aux_tmp,
                aux_tmp2,
                aux_tmp3,
                leaf_masks,
                coarse_masks,
                brick_occupied,
                dirty_bricks,
                x,
                y,
                z,
                new_word,
                new_ctype,
                new_tmp,
                new_tmp2,
                new_tmp3,
                hook,
                table,
            );
        }
    }
}

/// Adds `delta_q` to the packed temperature of every non-empty voxel in the
/// brush, clamping to `0..=0xFFFF` and preserving every other word field.
// CubeCL does not expand `i32::clamp` reliably, so the branch pair is the
// clamp; the lint is scoped to this kernel.
#[allow(clippy::manual_clamp)]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn adjust_temp_kernel(
    voxel_a: &mut [u32],
    dirty_bricks: &mut [Atomic<u32>],
    min_x: u32,
    min_y: u32,
    min_z: u32,
    max_x: u32,
    max_y: u32,
    max_z: u32,
    cx: f32,
    cy: f32,
    cz: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    shape: u32,
    delta_q: i32,
) {
    let x = min_x + CUBE_POS_X * 8u32 + UNIT_POS_X;
    let y = min_y + CUBE_POS_Y * 8u32 + UNIT_POS_Y;
    let z = min_z + CUBE_POS_Z * 8u32 + UNIT_POS_Z;
    if x <= max_x && y <= max_y && z <= max_z {
        let dx = x as f32 - cx;
        let dy = y as f32 - cy;
        let dz = z as f32 - cz;
        if brush_hit(dx, dy, dz, rx, ry, rz, shape) {
            let index = (y << 16u32) | (z << 8u32) | x;
            let word = voxel_a[index as usize];
            if word & 0xFFu32 != 0u32 {
                let temp = ((word >> 8u32) & 0xFFFFu32) as i32;
                let mut next = temp + delta_q;
                if next < 0i32 {
                    next = 0i32;
                }
                if next > 65535i32 {
                    next = 65535i32;
                }
                voxel_a[index as usize] = (word & 0xFF00_00FFu32) | ((next as u32) << 8u32);
                let (brick_word, mask) = brick_bit(brick_of_index(index));
                dirty_bricks[brick_word as usize].fetch_or(mask);
            }
        }
    }
}

/// Offset of a voxel from the closest point on the normalised segment.
///
/// The endpoints are already scaled by the per-axis sweep radii; `inv` scales
/// the voxel coordinate into the same space before the closest-point solve.
#[cube]
#[allow(clippy::too_many_arguments)]
fn segment_offset(
    vx: f32,
    vy: f32,
    vz: f32,
    from_x: f32,
    from_y: f32,
    from_z: f32,
    to_x: f32,
    to_y: f32,
    to_z: f32,
    inv_x: f32,
    inv_y: f32,
    inv_z: f32,
) -> (f32, f32, f32) {
    let nx = vx * inv_x;
    let ny = vy * inv_y;
    let nz = vz * inv_z;
    let abx = to_x - from_x;
    let aby = to_y - from_y;
    let abz = to_z - from_z;
    let apx = nx - from_x;
    let apy = ny - from_y;
    let apz = nz - from_z;
    let denom = abx * abx + aby * aby + abz * abz;
    let mut t = seed_f32(denom, 0.0f32);
    if denom > 0.0f32 {
        let dot = apx * abx + apy * aby + apz * abz;
        t = clampf(dot / denom, 0.0f32, 1.0f32);
    }
    (
        nx - (from_x + abx * t),
        ny - (from_y + aby * t),
        nz - (from_z + abz * t),
    )
}

/// Swept brush test over one AABB for the ellipsoid and prism shapes.
///
/// `from`/`to` are the segment endpoints divided by the per-axis sweep radii
/// and `inv` scales a voxel coordinate into the same space, so the ellipsoid is
/// the unit ball and the prism the unit triangle. The cuboid shape uses
/// [`edit_box_segment_kernel`] instead.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn edit_segment_kernel(
    voxel_a: &mut [u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    leaf_masks: &mut [Atomic<u32>],
    coarse_masks: &mut [Atomic<u32>],
    brick_occupied: &mut [Atomic<u32>],
    dirty_bricks: &mut [Atomic<u32>],
    min_x: u32,
    min_y: u32,
    min_z: u32,
    max_x: u32,
    max_y: u32,
    max_z: u32,
    from_x: f32,
    from_y: f32,
    from_z: f32,
    to_x: f32,
    to_y: f32,
    to_z: f32,
    inv_x: f32,
    inv_y: f32,
    inv_z: f32,
    shape: u32,
    new_word: u32,
    new_ctype: u32,
    new_tmp: i32,
    new_tmp2: i32,
    new_tmp3: i32,
    hook: u32,
    table: &[u32],
) {
    let x = min_x + CUBE_POS_X * 8u32 + UNIT_POS_X;
    let y = min_y + CUBE_POS_Y * 8u32 + UNIT_POS_Y;
    let z = min_z + CUBE_POS_Z * 8u32 + UNIT_POS_Z;
    if x <= max_x && y <= max_y && z <= max_z {
        let (dx, dy, dz) = segment_offset(
            x as f32, y as f32, z as f32, from_x, from_y, from_z, to_x, to_y, to_z, inv_x, inv_y,
            inv_z,
        );
        let hit = if shape == 2u32 {
            fabsf(dx) <= 1.0f32
                && fabsf(dy) <= 1.0f32
                && fabsf(dz) <= 1.0f32
                && 2.0f32 * fabsf(dx) + dy <= 1.0f32
        } else {
            dx * dx + dy * dy + dz * dz <= 1.0f32
        };
        if hit {
            write_brush_voxel(
                voxel_a,
                aux_ctype,
                aux_tmp,
                aux_tmp2,
                aux_tmp3,
                leaf_masks,
                coarse_masks,
                brick_occupied,
                dirty_bricks,
                x,
                y,
                z,
                new_word,
                new_ctype,
                new_tmp,
                new_tmp2,
                new_tmp3,
                hook,
                table,
            );
        }
    }
}

/// Expanded-AABB test for a cuboid sweep over one AABB.
///
/// A swept box is not a box, so the segment is reduced to its world-space
/// endpoint AABB expanded by `radii`; that is the rectangular footprint the
/// box gesture wants and needs no closest-point solve.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn edit_box_segment_kernel(
    voxel_a: &mut [u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    leaf_masks: &mut [Atomic<u32>],
    coarse_masks: &mut [Atomic<u32>],
    brick_occupied: &mut [Atomic<u32>],
    dirty_bricks: &mut [Atomic<u32>],
    min_x: u32,
    min_y: u32,
    min_z: u32,
    max_x: u32,
    max_y: u32,
    max_z: u32,
    lo_x: f32,
    lo_y: f32,
    lo_z: f32,
    hi_x: f32,
    hi_y: f32,
    hi_z: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    new_word: u32,
    new_ctype: u32,
    new_tmp: i32,
    new_tmp2: i32,
    new_tmp3: i32,
    hook: u32,
    table: &[u32],
) {
    let x = min_x + CUBE_POS_X * 8u32 + UNIT_POS_X;
    let y = min_y + CUBE_POS_Y * 8u32 + UNIT_POS_Y;
    let z = min_z + CUBE_POS_Z * 8u32 + UNIT_POS_Z;
    if x <= max_x && y <= max_y && z <= max_z {
        let vx = x as f32;
        let vy = y as f32;
        let vz = z as f32;
        let hit = vx >= lo_x - rx
            && vx <= hi_x + rx
            && vy >= lo_y - ry
            && vy <= hi_y + ry
            && vz >= lo_z - rz
            && vz <= hi_z + rz;
        if hit {
            write_brush_voxel(
                voxel_a,
                aux_ctype,
                aux_tmp,
                aux_tmp2,
                aux_tmp3,
                leaf_masks,
                coarse_masks,
                brick_occupied,
                dirty_bricks,
                x,
                y,
                z,
                new_word,
                new_ctype,
                new_tmp,
                new_tmp2,
                new_tmp3,
                hook,
                table,
            );
        }
    }
}

/// Paints one packed voxel index per thread.
///
/// The flood-fill driver gathers an explicit connected set on the host (the
/// only read-back in the 3D edit path), so the device pass only has to stamp
/// those indices without re-deriving the region. Every stamped voxel refreshes
/// the same leaf/coarse/brick bits a brush stamp would.
#[cube(launch_unchecked)]
fn paint_indices_kernel(
    voxel_a: &mut [u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    leaf_masks: &mut [Atomic<u32>],
    coarse_masks: &mut [Atomic<u32>],
    brick_occupied: &mut [Atomic<u32>],
    dirty_bricks: &mut [Atomic<u32>],
    indices: &[u32],
    new_word: u32,
    new_ctype: u32,
    new_tmp: i32,
    new_tmp2: i32,
    new_tmp3: i32,
    hook: u32,
    table: &[u32],
) {
    let thread = ABSOLUTE_POS;
    if thread < indices.len() {
        let index = indices[thread];
        let x = index & 0xFFu32;
        let y = (index >> 16u32) & 0xFFu32;
        let z = (index >> 8u32) & 0xFFu32;
        write_brush_voxel(
            voxel_a,
            aux_ctype,
            aux_tmp,
            aux_tmp2,
            aux_tmp3,
            leaf_masks,
            coarse_masks,
            brick_occupied,
            dirty_bricks,
            x,
            y,
            z,
            new_word,
            new_ctype,
            new_tmp,
            new_tmp2,
            new_tmp3,
            hook,
            table,
        );
    }
}

/// Host-side description of one segment sweep.
///
/// Built once per segment: it owns the clamped AABB, the normalised endpoints
/// and the per-axis inverse radii, so the count and the launch cannot drift
/// apart.
#[derive(Clone, Copy)]
struct SegmentSweep {
    min: [u32; 3],
    max: [u32; 3],
    from_n: [f32; 3],
    to_n: [f32; 3],
    lo: [f32; 3],
    hi: [f32; 3],
    radii: [f32; 3],
    inv: [f32; 3],
    shape: BrushShape3d,
}

impl SegmentSweep {
    /// Validates the endpoints and radii and computes the AABB.
    ///
    /// Returns `None` for a non-finite endpoint, a negative or non-finite
    /// radius, three zero radii, or a sweep that misses the world.
    fn new(from: [f32; 3], to: [f32; 3], radii: [f32; 3], shape: BrushShape3d) -> Option<Self> {
        if !from.iter().chain(to.iter()).all(|value| value.is_finite()) {
            return None;
        }
        if !radii
            .iter()
            .all(|radius| radius.is_finite() && *radius >= 0.0)
        {
            return None;
        }
        if !radii.iter().any(|radius| *radius > 0.0) {
            return None;
        }

        let mut inv = [0.0f32; 3];
        let mut from_n = [0.0f32; 3];
        let mut to_n = [0.0f32; 3];
        let mut lo = [0.0f32; 3];
        let mut hi = [0.0f32; 3];
        for axis in 0..3 {
            inv[axis] = 1.0 / radii[axis].max(MIN_SWEEP_RADIUS);
            from_n[axis] = from[axis] * inv[axis];
            to_n[axis] = to[axis] * inv[axis];
            lo[axis] = from[axis].min(to[axis]);
            hi[axis] = from[axis].max(to[axis]);
        }

        let mut min = [0u32; 3];
        let mut max = [0u32; 3];
        for axis in 0..3 {
            let low = lo[axis] - radii[axis];
            let high = hi[axis] + radii[axis];
            if high < 0.0 || low > MAX_COORD {
                return None;
            }
            min[axis] = low.ceil().clamp(0.0, MAX_COORD) as u32;
            max[axis] = high.floor().clamp(0.0, MAX_COORD) as u32;
        }
        if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
            return None;
        }

        Some(Self {
            min,
            max,
            from_n,
            to_n,
            lo,
            hi,
            radii,
            inv,
            shape,
        })
    }

    /// Host mirror of the segment kernels' membership test.
    fn contains(&self, x: u32, y: u32, z: u32) -> bool {
        let vx = x as f32;
        let vy = y as f32;
        let vz = z as f32;
        if self.shape == BrushShape3d::Cuboid {
            let [rx, ry, rz] = self.radii;
            return vx >= self.lo[0] - rx
                && vx <= self.hi[0] + rx
                && vy >= self.lo[1] - ry
                && vy <= self.hi[1] + ry
                && vz >= self.lo[2] - rz
                && vz <= self.hi[2] + rz;
        }

        let nx = vx * self.inv[0];
        let ny = vy * self.inv[1];
        let nz = vz * self.inv[2];
        let abx = self.to_n[0] - self.from_n[0];
        let aby = self.to_n[1] - self.from_n[1];
        let abz = self.to_n[2] - self.from_n[2];
        let apx = nx - self.from_n[0];
        let apy = ny - self.from_n[1];
        let apz = nz - self.from_n[2];
        let denom = abx * abx + aby * aby + abz * abz;
        let mut t = 0.0f32;
        if denom > 0.0 {
            let dot = apx * abx + apy * aby + apz * abz;
            t = (dot / denom).clamp(0.0, 1.0);
        }
        let dx = nx - (self.from_n[0] + abx * t);
        let dy = ny - (self.from_n[1] + aby * t);
        let dz = nz - (self.from_n[2] + abz * t);
        if self.shape == BrushShape3d::Prism {
            let ax = dx.abs();
            ax <= 1.0 && dy.abs() <= 1.0 && dz.abs() <= 1.0 && 2.0 * ax + dy <= 1.0
        } else {
            dx * dx + dy * dy + dz * dz <= 1.0
        }
    }
}

/// Converts a Kelvin delta to the packed fixed-point domain, saturating.
///
/// `NaN` maps to zero, so a malformed delta is a no-op instead of corrupting
/// every temperature in the brush, and the result always fits the `i32` sum
/// with a `u16` temperature.
pub(crate) fn temp_delta_q(delta_k: f32) -> i32 {
    (delta_k * TEMP_Q_SCALE)
        .round()
        .clamp(-(TEMP_Q_MAX as f32), TEMP_Q_MAX as f32) as i32
}

/// Launch-size arguments covering the inclusive AABB.
fn groups(min: [u32; 3], max: [u32; 3]) -> (u32, u32, u32) {
    (
        (max[0] - min[0] + 1).div_ceil(CUBE),
        (max[1] - min[1] + 1).div_ceil(CUBE),
        (max[2] - min[2] + 1).div_ceil(CUBE),
    )
}

/// Paints (`word != 0`) or erases (`word == 0`) the brush.
///
/// Returns the number of voxels inside the brush; the GPU test mirrors the
/// host test, so the count is exact.
pub(crate) fn launch_edit_brush(
    engine: &Voxel3dEngine,
    fields: PaintFields,
    brush: Brush3d,
) -> u32 {
    let word = fields.word;
    let (min, max) = brush.bounds();
    if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
        return 0;
    }

    let mut count = 0u32;
    brush.for_each_voxel(|_| count += 1);
    if count == 0 {
        return 0;
    }
    let (groups_x, groups_y, groups_z) = groups(min, max);

    unsafe {
        edit_brush_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(groups_x, groups_y, groups_z),
            CubeDim::new_3d(CUBE, CUBE, CUBE),
            BufferArg::from_raw_parts(engine.bufs.voxel_a.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.bufs.aux_ctype.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.bufs.aux_tmp.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.bufs.aux_tmp2.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.bufs.aux_tmp3.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(
                engine.bufs.leaf_masks.clone(),
                tpt_core::voxel3d::LEAF_WORD_COUNT,
            ),
            BufferArg::from_raw_parts(engine.bufs.coarse_masks.clone(), super::COARSE_MASK_WORDS),
            BufferArg::from_raw_parts(
                engine.bufs.brick_occupied.clone(),
                tpt_core::voxel3d::BRICK_COUNT / 32,
            ),
            BufferArg::from_raw_parts(
                engine.bufs.dirty_bricks.clone(),
                tpt_core::voxel3d::BRICK_COUNT / 32,
            ),
            min[0],
            min[1],
            min[2],
            max[0],
            max[1],
            max[2],
            brush.center[0],
            brush.center[1],
            brush.center[2],
            brush.radii[0],
            brush.radii[1],
            brush.radii[2],
            brush.shape.code(),
            word,
            fields.ctype,
            fields.tmp,
            fields.tmp2,
            fields.tmp3,
            fields.hook,
            BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
        );
    }
    count
}

/// Paints `word` into every voxel index, one thread per index.
///
/// Returns the number of indices, or `0` for an empty list. The caller owns
/// the flood-fill set; the device only stamps it.
pub(crate) fn launch_paint_indices(
    engine: &Voxel3dEngine,
    fields: PaintFields,
    indices: &[u32],
) -> u32 {
    if indices.is_empty() {
        return 0;
    }
    let list = engine
        .client
        .create_from_slice(bytemuck::cast_slice(indices));
    let threads = 256u32;
    let cubes = (indices.len() as u32).div_ceil(threads);
    unsafe {
        paint_indices_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(threads),
            BufferArg::from_raw_parts(engine.bufs.voxel_a.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.bufs.aux_ctype.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.bufs.aux_tmp.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.bufs.aux_tmp2.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(engine.bufs.aux_tmp3.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(
                engine.bufs.leaf_masks.clone(),
                tpt_core::voxel3d::LEAF_WORD_COUNT,
            ),
            BufferArg::from_raw_parts(engine.bufs.coarse_masks.clone(), super::COARSE_MASK_WORDS),
            BufferArg::from_raw_parts(
                engine.bufs.brick_occupied.clone(),
                tpt_core::voxel3d::BRICK_COUNT / 32,
            ),
            BufferArg::from_raw_parts(
                engine.bufs.dirty_bricks.clone(),
                tpt_core::voxel3d::BRICK_COUNT / 32,
            ),
            BufferArg::from_raw_parts(list, indices.len()),
            fields.word,
            fields.ctype,
            fields.tmp,
            fields.tmp2,
            fields.tmp3,
            fields.hook,
            BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
        );
    }
    indices.len() as u32
}

/// Adds `delta_q` fixed-point Kelvin to every non-empty voxel in the brush.
///
/// Returns the number of voxels the brush covers: the host cannot know which
/// of them are non-empty without a read-back, so the count is the brush
/// volume, not the number of changed voxels.
pub(crate) fn launch_adjust_temp(engine: &Voxel3dEngine, brush: Brush3d, delta_q: i32) -> u32 {
    let (min, max) = brush.bounds();
    if min[0] > max[0] || min[1] > max[1] || min[2] > max[2] {
        return 0;
    }

    let mut count = 0u32;
    brush.for_each_voxel(|_| count += 1);
    if count == 0 || delta_q == 0 {
        return 0;
    }
    let (groups_x, groups_y, groups_z) = groups(min, max);

    unsafe {
        adjust_temp_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Static(groups_x, groups_y, groups_z),
            CubeDim::new_3d(CUBE, CUBE, CUBE),
            BufferArg::from_raw_parts(engine.bufs.voxel_a.clone(), VOXEL_COUNT),
            BufferArg::from_raw_parts(
                engine.bufs.dirty_bricks.clone(),
                tpt_core::voxel3d::BRICK_COUNT / 32,
            ),
            min[0],
            min[1],
            min[2],
            max[0],
            max[1],
            max[2],
            brush.center[0],
            brush.center[1],
            brush.center[2],
            brush.radii[0],
            brush.radii[1],
            brush.radii[2],
            brush.shape.code(),
            delta_q,
        );
    }
    count
}

/// Paints (`word != 0`) or erases (`word == 0`) the brush swept from `from` to
/// `to`.
///
/// One launch covers the whole segment. Returns the number of voxels the sweep
/// covers, computed with the host mirror of the kernel test.
pub(crate) fn launch_edit_segment(
    engine: &Voxel3dEngine,
    fields: PaintFields,
    from: [f32; 3],
    to: [f32; 3],
    radii: [f32; 3],
    shape: BrushShape3d,
) -> u32 {
    let word = fields.word;
    let Some(sweep) = SegmentSweep::new(from, to, radii, shape) else {
        return 0;
    };

    let mut count = 0u32;
    for y in sweep.min[1]..=sweep.max[1] {
        for z in sweep.min[2]..=sweep.max[2] {
            for x in sweep.min[0]..=sweep.max[0] {
                if sweep.contains(x, y, z) {
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        return 0;
    }
    let (groups_x, groups_y, groups_z) = groups(sweep.min, sweep.max);

    // The cuboid sweep only needs the endpoint AABB, so it uses the cheaper
    // kernel instead of doing the closest-point solve it would discard.
    if shape == BrushShape3d::Cuboid {
        unsafe {
            edit_box_segment_kernel::launch_unchecked(
                &engine.client,
                CubeCount::Static(groups_x, groups_y, groups_z),
                CubeDim::new_3d(CUBE, CUBE, CUBE),
                BufferArg::from_raw_parts(engine.bufs.voxel_a.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(engine.bufs.aux_ctype.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(engine.bufs.aux_tmp.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(engine.bufs.aux_tmp2.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(engine.bufs.aux_tmp3.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(
                    engine.bufs.leaf_masks.clone(),
                    tpt_core::voxel3d::LEAF_WORD_COUNT,
                ),
                BufferArg::from_raw_parts(
                    engine.bufs.coarse_masks.clone(),
                    super::COARSE_MASK_WORDS,
                ),
                BufferArg::from_raw_parts(
                    engine.bufs.brick_occupied.clone(),
                    tpt_core::voxel3d::BRICK_COUNT / 32,
                ),
                BufferArg::from_raw_parts(
                    engine.bufs.dirty_bricks.clone(),
                    tpt_core::voxel3d::BRICK_COUNT / 32,
                ),
                sweep.min[0],
                sweep.min[1],
                sweep.min[2],
                sweep.max[0],
                sweep.max[1],
                sweep.max[2],
                sweep.lo[0],
                sweep.lo[1],
                sweep.lo[2],
                sweep.hi[0],
                sweep.hi[1],
                sweep.hi[2],
                sweep.radii[0],
                sweep.radii[1],
                sweep.radii[2],
                word,
                fields.ctype,
                fields.tmp,
                fields.tmp2,
                fields.tmp3,
                fields.hook,
                BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
            );
        }
    } else {
        unsafe {
            edit_segment_kernel::launch_unchecked(
                &engine.client,
                CubeCount::Static(groups_x, groups_y, groups_z),
                CubeDim::new_3d(CUBE, CUBE, CUBE),
                BufferArg::from_raw_parts(engine.bufs.voxel_a.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(engine.bufs.aux_ctype.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(engine.bufs.aux_tmp.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(engine.bufs.aux_tmp2.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(engine.bufs.aux_tmp3.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(
                    engine.bufs.leaf_masks.clone(),
                    tpt_core::voxel3d::LEAF_WORD_COUNT,
                ),
                BufferArg::from_raw_parts(
                    engine.bufs.coarse_masks.clone(),
                    super::COARSE_MASK_WORDS,
                ),
                BufferArg::from_raw_parts(
                    engine.bufs.brick_occupied.clone(),
                    tpt_core::voxel3d::BRICK_COUNT / 32,
                ),
                BufferArg::from_raw_parts(
                    engine.bufs.dirty_bricks.clone(),
                    tpt_core::voxel3d::BRICK_COUNT / 32,
                ),
                sweep.min[0],
                sweep.min[1],
                sweep.min[2],
                sweep.max[0],
                sweep.max[1],
                sweep.max[2],
                sweep.from_n[0],
                sweep.from_n[1],
                sweep.from_n[2],
                sweep.to_n[0],
                sweep.to_n[1],
                sweep.to_n[2],
                sweep.inv[0],
                sweep.inv[1],
                sweep.inv[2],
                shape.code(),
                word,
                fields.ctype,
                fields.tmp,
                fields.tmp2,
                fields.tmp3,
                fields.hook,
                BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
            );
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_core::voxel3d::{LIFE_MAX, temp_to_q};

    /// Host mirror of [`axis_norm`].
    fn axis_norm_host(d: f32, r: f32) -> f32 {
        if r > 0.0 {
            let n = d / r;
            n * n
        } else {
            0.0
        }
    }

    /// Host mirror of [`axis_inside`].
    fn axis_inside_host(d: f32, r: f32) -> bool {
        r > 0.0 || d == 0.0
    }

    /// Host mirror of [`brush_hit`]: the device formula is repeated here
    /// because a `#[cube]` function cannot run on the host.
    fn brush_hit_host(
        dx: f32,
        dy: f32,
        dz: f32,
        [rx, ry, rz]: [f32; 3],
        shape: BrushShape3d,
    ) -> bool {
        let (ax, ay, az) = (dx.abs(), dy.abs(), dz.abs());
        let ellipsoid = axis_inside_host(dx, rx)
            && axis_inside_host(dy, ry)
            && axis_inside_host(dz, rz)
            && axis_norm_host(dx, rx) + axis_norm_host(dy, ry) + axis_norm_host(dz, rz) <= 1.0;
        let cuboid = ax <= rx && ay <= ry && az <= rz;
        let prism = cuboid && 2.0 * ax * ry + dy * rx <= rx * ry;
        (shape == BrushShape3d::Ellipsoid && ellipsoid)
            || (shape == BrushShape3d::Cuboid && cuboid)
            || (shape == BrushShape3d::Prism && prism)
    }

    #[test]
    fn device_formula_matches_the_host_brush() {
        let centers = [
            [10.0f32, 20.0, 30.0],
            [10.5, 20.5, 30.5],
            [0.0, 0.0, 0.0],
            [255.0, 255.0, 255.0],
        ];
        let radii = [
            [2.5f32, 3.5, 1.5],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.5, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ];
        for shape in [
            BrushShape3d::Ellipsoid,
            BrushShape3d::Cuboid,
            BrushShape3d::Prism,
        ] {
            for center in centers {
                for radius in radii {
                    let brush = Brush3d::with_shape(center, radius, shape);
                    for y in 0..=40 {
                        for z in 0..=40 {
                            for x in 0..=15 {
                                let dx = x as f32 - center[0];
                                let dy = y as f32 - center[1];
                                let dz = z as f32 - center[2];
                                assert_eq!(
                                    brush_hit_host(dx, dy, dz, radius, shape),
                                    brush.contains(x, y, z),
                                    "{shape:?} center {center:?} radii {radius:?} at ({x}, {y}, {z})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn zero_length_segment_matches_the_point_brush() {
        let centers = [[10.0f32, 20.0, 30.0], [128.5, 128.5, 128.5]];
        let radii = [[2.0f32, 4.0, 1.0], [2.0, 2.0, 2.0], [1.0, 2.0, 4.0]];
        for shape in [
            BrushShape3d::Ellipsoid,
            BrushShape3d::Cuboid,
            BrushShape3d::Prism,
        ] {
            for center in centers {
                for radius in radii {
                    let point = Brush3d::with_shape(center, radius, shape);
                    let sweep =
                        SegmentSweep::new(center, center, radius, shape).expect("non-empty sweep");
                    assert_eq!(sweep.min, point.bounds().0, "{shape:?}");
                    assert_eq!(sweep.max, point.bounds().1, "{shape:?}");
                    for y in 0..=40 {
                        for z in 0..=40 {
                            for x in 0..=15 {
                                let inside = (sweep.min[0]..=sweep.max[0]).contains(&x)
                                    && (sweep.min[1]..=sweep.max[1]).contains(&y)
                                    && (sweep.min[2]..=sweep.max[2]).contains(&z)
                                    && sweep.contains(x, y, z);
                                assert_eq!(
                                    inside,
                                    point.contains(x, y, z),
                                    "{shape:?} center {center:?} radii {radius:?} at ({x}, {y}, {z})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn segment_sweep_rejects_invalid_input() {
        let center = [10.0f32, 20.0, 30.0];
        for shape in [
            BrushShape3d::Ellipsoid,
            BrushShape3d::Cuboid,
            BrushShape3d::Prism,
        ] {
            assert!(
                SegmentSweep::new(center, center, [0.0; 3], shape).is_none(),
                "{shape:?}: three zero radii are empty"
            );
            assert!(
                SegmentSweep::new([f32::NAN, 20.0, 30.0], center, [1.0; 3], shape).is_none(),
                "{shape:?}: non-finite endpoint"
            );
            assert!(
                SegmentSweep::new(center, center, [1.0, -1.0, 1.0], shape).is_none(),
                "{shape:?}: negative radius"
            );
            assert!(
                SegmentSweep::new(center, center, [f32::INFINITY, 1.0, 1.0], shape).is_none(),
                "{shape:?}: non-finite radius"
            );
            assert!(
                SegmentSweep::new([-50.0, 20.0, 30.0], [-40.0, 20.0, 30.0], [1.0; 3], shape)
                    .is_none(),
                "{shape:?}: sweep outside the world"
            );
            assert!(
                SegmentSweep::new(center, center, [1.0; 3], shape).is_some(),
                "{shape:?}: valid sweep"
            );
        }
    }

    #[test]
    fn segment_bounds_cover_every_accepted_voxel() {
        let segments = [
            ([10.0f32, 20.0, 30.0], [12.0f32, 22.0, 30.0]),
            ([10.0, 20.0, 30.0], [10.0, 20.0, 35.0]),
            ([8.5, 18.5, 28.5], [13.5, 23.5, 33.5]),
        ];
        let radii = [[2.0f32, 2.0, 2.0], [2.0, 0.0, 2.0], [0.0, 2.0, 0.0]];
        for shape in [
            BrushShape3d::Ellipsoid,
            BrushShape3d::Cuboid,
            BrushShape3d::Prism,
        ] {
            for (from, to) in segments {
                for radius in radii {
                    let sweep =
                        SegmentSweep::new(from, to, radius, shape).expect("valid segment sweep");
                    // The scan window extends past the AABB by more than any
                    // radius, so an accepted voxel outside the AABB would be
                    // found.
                    let mut count = 0u32;
                    let mut outside = 0u32;
                    for y in 0..=40 {
                        for z in 0..=40 {
                            for x in 0..=15 {
                                if sweep.contains(x, y, z) {
                                    count += 1;
                                    let inside_bounds = (sweep.min[0]..=sweep.max[0]).contains(&x)
                                        && (sweep.min[1]..=sweep.max[1]).contains(&y)
                                        && (sweep.min[2]..=sweep.max[2]).contains(&z);
                                    if !inside_bounds {
                                        outside += 1;
                                    }
                                }
                            }
                        }
                    }
                    assert_eq!(outside, 0, "{shape:?} {from:?}->{to:?} {radius:?}");
                    let mut in_bounds = 0u32;
                    for y in sweep.min[1]..=sweep.max[1] {
                        for z in sweep.min[2]..=sweep.max[2] {
                            for x in sweep.min[0]..=sweep.max[0] {
                                if sweep.contains(x, y, z) {
                                    in_bounds += 1;
                                }
                            }
                        }
                    }
                    assert_eq!(in_bounds, count, "bounds loop must cover the sweep");
                }
            }
        }
    }

    #[test]
    fn delta_quantisation_saturates() {
        assert_eq!(temp_delta_q(200.0), 1600);
        assert_eq!(temp_delta_q(-200.0), -1600);
        assert_eq!(temp_delta_q(0.0), 0);
        assert_eq!(temp_delta_q(1.0e9), TEMP_Q_MAX);
        assert_eq!(temp_delta_q(-1.0e9), -TEMP_Q_MAX);
        assert_eq!(temp_delta_q(f32::NAN), 0);
    }

    #[test]
    fn temperature_rewrite_preserves_type_and_life() {
        let ty = 5u32;
        let temp = temp_to_q(293.15) as u32;
        let life = LIFE_MAX as u32;
        let word = ty | (temp << 8u32) | (life << 24u32);

        let changed = word & 0xFF00_00FFu32 | ((temp + 1600) << 8u32);
        assert_eq!(changed & 0xFF, ty);
        assert_eq!((changed >> 24) & 0xFF, life);
        assert_eq!((changed >> 8) & 0xFFFF, temp + 1600);
    }
}
