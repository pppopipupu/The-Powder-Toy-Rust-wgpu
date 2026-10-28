//! Device-side brush edits: decoration blending and single-field writes.
//!
//! Both tools used to pull the whole particle SoA and the touched map rectangle
//! back to the host before every stroke (`read_map_rect` +
//! `sync_particles_gpu_to_host`), edit the host shadow and upload the touched
//! slots again. On a 1024x768 canvas with 50 k particles that is a multi-megabyte
//! blocking copy per stroke, i.e. once per frame while dragging.
//!
//! The device already owns the authoritative `pmap`/`photons`/`part_*` arrays, so
//! the stroke is resolved there: the host uploads only the packed pixel list
//! (a few hundred words) and the kernels do the read-modify-write in place.

// Clippy's rewrite for these lints uses methods the device IR does not have
// (`is_multiple_of`, `clamp`, `Range::contains`, `saturating_sub`), and
// collapsing a nested `if` changes what the CubeCL macro expands, so the plain
// operators are deliberate.
#![allow(clippy::manual_clamp)]

use cubecl::prelude::*;

use tpt_core::commands::{DecoMode, ParticleField};
use tpt_core::constants::*;

use crate::buffers::SimBuffers;
use crate::reaction_kernel::shared::{launch_grid, pmap_at, pmap_slot};

/// `PT_NONE` marks a free slot; mirrored for the kernel bodies.
const C_PT_NONE: u32 = 0;
const _: () = assert!(PT_NONE == 0);

/// `ApplyDecoration`'s per-stroke blend strength.
const C_DECO_STRENGTH: f32 = 0.01;

/// Slot field written by [`set_field_kernel`]; mirrors [`ParticleField`]'s order.
const FIELD_TMP: u32 = 0;
const FIELD_TMP2: u32 = 1;
const FIELD_TMP3: u32 = 2;
const FIELD_TMP4: u32 = 3;
const FIELD_LIFE: u32 = 4;
const FIELD_CTYPE: u32 = 5;

#[cube]
fn deco_channel(word: u32, shift: u32) -> f32 {
    ((word >> shift) & 0xFFu32) as f32 / 255.0f32
}

#[cube]
fn deco_byte(value: f32) -> u32 {
    let mut scaled = value * 255.0f32 + 0.5f32;
    if scaled < 0.0f32 {
        scaled = 0.0f32;
    }
    if scaled > 255.0f32 {
        scaled = 255.0f32;
    }
    scaled as u32
}

/// `DECOSPACE_SRGB` transfer function of `Editing.cpp`.
#[cube]
fn srgb_to_linear(value: f32) -> f32 {
    let mut out = value / 12.92f32;
    if value > 0.04045f32 {
        out = ((value + 0.055f32) / 1.055f32).powf(2.4f32);
    }
    out
}

/// Inverse of [`srgb_to_linear`].
#[cube]
fn linear_to_srgb(value: f32) -> f32 {
    let mut out = value * 12.92f32;
    if value > 0.003_130_8f32 {
        out = 1.055f32 * value.powf(1.0f32 / 2.4f32) - 0.055f32;
    }
    out
}

/// Resolves the particle slot at a pixel, `0xFFFF_FFFF` when empty.
///
/// Mirrors `SimBuffers::particle_index_at`: the matter map first, then the
/// energy map, and a word without id bits means "no particle" (the packing
/// stores `id + 1`).
#[cube]
pub(crate) fn slot_at(
    pmap: &[u32],
    photons: &[u32],
    xres: u32,
    yres: u32,
    px: i32,
    py: i32,
) -> u32 {
    let mut word = pmap_at(pmap, xres, yres, px, py);
    if (word >> 9u32) == 0u32 {
        word = pmap_at(photons, xres, yres, px, py);
    }
    let mut slot = 0xFFFF_FFFFu32;
    if (word >> 9u32) != 0u32 {
        slot = pmap_slot(word) as u32;
    }
    slot
}

/// Blends one packed decoration word, in place.
///
/// Port of `Simulation::ApplyDecoration`'s channel maths: the smudge mode
/// averages the decorated neighbours in linear light, every other mode mixes
/// with [`C_DECO_STRENGTH`]. `neighbour_count == 0` in the smudge mode leaves
/// the particle untouched (the reference returns before writing).
#[cube]
fn blend_deco(
    current: u32,
    colour: u32,
    mode: u32,
    sum_a: f32,
    sum_r: f32,
    sum_g: f32,
    sum_b: f32,
    neighbour_count: u32,
) -> u32 {
    let mut ta = deco_channel(current, 24u32);
    let mut tr = deco_channel(current, 16u32);
    let mut tg = deco_channel(current, 8u32);
    let mut tb = deco_channel(current, 0u32);
    let col_a = deco_channel(colour, 24u32);
    let col_r = deco_channel(colour, 16u32);
    let col_g = deco_channel(colour, 8u32);
    let col_b = deco_channel(colour, 0u32);

    if mode == 0u32 {
        // `DECO_DRAW`
        ta = col_a;
        tr = col_r;
        tg = col_g;
        tb = col_b;
    } else if mode == 1u32 {
        // `DECO_CLEAR`
        ta = 0.0f32;
        tr = 0.0f32;
        tg = 0.0f32;
        tb = 0.0f32;
    } else if mode == 2u32 {
        // `DECO_ADD`
        tr += (col_r * C_DECO_STRENGTH) * col_a;
        tg += (col_g * C_DECO_STRENGTH) * col_a;
        tb += (col_b * C_DECO_STRENGTH) * col_a;
    } else if mode == 3u32 {
        // `DECO_SUBTRACT`
        tr -= (col_r * C_DECO_STRENGTH) * col_a;
        tg -= (col_g * C_DECO_STRENGTH) * col_a;
        tb -= (col_b * C_DECO_STRENGTH) * col_a;
    } else if mode == 4u32 {
        // `DECO_MULTIPLY`
        tr *= 1.0f32 + (col_r * C_DECO_STRENGTH) * col_a;
        tg *= 1.0f32 + (col_g * C_DECO_STRENGTH) * col_a;
        tb *= 1.0f32 + (col_b * C_DECO_STRENGTH) * col_a;
    } else if mode == 5u32 {
        // `DECO_DIVIDE`
        tr /= 1.0f32 + (col_r * C_DECO_STRENGTH) * col_a;
        tg /= 1.0f32 + (col_g * C_DECO_STRENGTH) * col_a;
        tb /= 1.0f32 + (col_b * C_DECO_STRENGTH) * col_a;
    } else {
        // `DECO_SMUDGE`
        if neighbour_count > 0u32 {
            let num = neighbour_count as f32;
            ta = linear_to_srgb(sum_a / num);
            tr = linear_to_srgb(sum_r / num);
            tg = linear_to_srgb(sum_g / num);
            tb = linear_to_srgb(sum_b / num);
        }
    }
    (deco_byte(ta) << 24u32) | (deco_byte(tr) << 16u32) | (deco_byte(tg) << 8u32) | deco_byte(tb)
}

#[cube(launch_unchecked)]
fn apply_deco_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_dcolour: &mut [u32],
    points: &[u32],
    count: u32,
    colour: u32,
    mode: u32,
    xres: u32,
    yres: u32,
    cell: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < count {
        let packed = points[i as usize];
        let px = (packed & 0xFFFFu32) as i32;
        let py = (packed >> 16u32) as i32;
        let slot = slot_at(pmap, photons, xres, yres, px, py);
        if slot != 0xFFFF_FFFFu32 {
            let id = slot as usize;
            if part_ty[id] != C_PT_NONE {
                let mut write = true;
                let mut sum_a = 0.0f32;
                let mut sum_r = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_b = 0.0f32;
                let mut neighbours = 0u32;
                if mode == 6u32 {
                    // `ApplyDecoration` only smudges outside the `CELL` border.
                    let in_region = px >= cell as i32
                        && py >= cell as i32
                        && (px as u32) < xres - cell
                        && (py as u32) < yres - cell;
                    if !in_region {
                        write = false;
                    }
                    // Neighbourhood `|rx| + |ry| > 2` inside `-2..=2`, matter
                    // map only, decorated neighbours only.
                    for k in 0..25u32 {
                        let rx = (k / 5u32) as i32 - 2i32;
                        let ry = (k % 5u32) as i32 - 2i32;
                        let ax = if rx < 0i32 { 0i32 - rx } else { rx };
                        let ay = if ry < 0i32 { 0i32 - ry } else { ry };
                        if ax + ay > 2i32 {
                            let word = pmap_at(pmap, xres, yres, px + rx, py + ry);
                            if (word >> 9u32) != 0u32 {
                                let nid = pmap_slot(word);
                                let dcolour = part_dcolour[nid];
                                if dcolour != 0u32 {
                                    sum_a += srgb_to_linear(deco_channel(dcolour, 24u32));
                                    sum_r += srgb_to_linear(deco_channel(dcolour, 16u32));
                                    sum_g += srgb_to_linear(deco_channel(dcolour, 8u32));
                                    sum_b += srgb_to_linear(deco_channel(dcolour, 0u32));
                                    neighbours += 1u32;
                                }
                            }
                        }
                    }
                    if neighbours == 0u32 {
                        write = false;
                    }
                }
                if write {
                    part_dcolour[id] = blend_deco(
                        part_dcolour[id],
                        colour,
                        mode,
                        sum_a,
                        sum_r,
                        sum_g,
                        sum_b,
                        neighbours,
                    );
                }
            }
        }
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn set_field_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    points: &[u32],
    count: u32,
    field: u32,
    value: i32,
    ctype_value: u32,
    xres: u32,
    yres: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < count {
        let packed = points[i as usize];
        let px = (packed & 0xFFFFu32) as i32;
        let py = (packed >> 16u32) as i32;
        let slot = slot_at(pmap, photons, xres, yres, px, py);
        if slot != 0xFFFF_FFFFu32 {
            let id = slot as usize;
            if part_ty[id] != C_PT_NONE {
                if field == FIELD_TMP {
                    part_tmp[id] = value;
                } else if field == FIELD_TMP2 {
                    part_tmp2[id] = value;
                } else if field == FIELD_TMP3 {
                    part_tmp3[id] = value;
                } else if field == FIELD_TMP4 {
                    part_tmp4[id] = value;
                } else if field == FIELD_LIFE {
                    part_life[id] = value;
                } else if field == FIELD_CTYPE {
                    part_ctype[id] = ctype_value;
                }
            }
        }
    }
}

/// Uploads a packed pixel list into the shared edit scratch buffer and returns
/// the number of points the kernels must process.
fn upload_points(bufs: &SimBuffers, points: &[(i32, i32)]) -> u32 {
    let count = points.len().min(SimBuffers::EDIT_POINT_CAPACITY);
    if count == 0 {
        return 0;
    }
    let packed: Vec<u32> = points[..count]
        .iter()
        .map(|&(x, y)| ((x as u32) & 0xFFFF) | (((y as u32) & 0xFFFF) << 16))
        .collect();
    crate::buffers::gpu_sync::upload(&bufs.client, &bufs.gpu_edit_points, &packed);
    count as u32
}

/// Blends the decoration colour of every particle under `points`, on the device.
pub fn apply_decoration(bufs: &mut SimBuffers, points: &[(i32, i32)], colour: u32, mode: DecoMode) {
    let count = upload_points(bufs, points);
    if count == 0 {
        return;
    }
    let (cubes, dim) = launch_grid(count as usize);
    unsafe {
        apply_deco_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), bufs.photons.len()),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), bufs.capacity),
            BufferArg::from_raw_parts(
                bufs.gpu_edit_points.clone(),
                SimBuffers::EDIT_POINT_CAPACITY,
            ),
            count,
            colour,
            mode as u32,
            bufs.config.xres,
            bufs.config.yres,
            CELL,
        );
    }
    // The device changed; the host shadow keeps its pre-stroke `dcolour` until
    // something explicitly reads it back.
    bufs.gpu_particles_dirty = true;
    bufs.gpu_dirty = true;
}

/// Writes one integer field of every particle under `points`, on the device.
pub fn set_particle_field(
    bufs: &mut SimBuffers,
    points: &[(i32, i32)],
    field: ParticleField,
    value: i64,
) {
    let count = upload_points(bufs, points);
    if count == 0 {
        return;
    }
    let (field_id, int_value) = match field {
        ParticleField::Tmp => (
            FIELD_TMP,
            value.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        ),
        ParticleField::Tmp2 => (
            FIELD_TMP2,
            value.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        ),
        ParticleField::Tmp3 => (
            FIELD_TMP3,
            value.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        ),
        ParticleField::Tmp4 => (
            FIELD_TMP4,
            value.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        ),
        ParticleField::Life => (
            FIELD_LIFE,
            value.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        ),
        ParticleField::Ctype => (FIELD_CTYPE, 0),
    };
    let ctype_value = if field == ParticleField::Ctype {
        value.clamp(0, PT_NUM as i64 - 1) as u32
    } else {
        0
    };
    let (cubes, dim) = launch_grid(count as usize);
    unsafe {
        set_field_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), bufs.photons.len()),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
            BufferArg::from_raw_parts(
                bufs.gpu_edit_points.clone(),
                SimBuffers::EDIT_POINT_CAPACITY,
            ),
            count,
            field_id,
            int_value,
            ctype_value,
            bufs.config.xres,
            bufs.config.yres,
        );
    }
    bufs.gpu_particles_dirty = true;
    bufs.gpu_dirty = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `0xAARRGGBB` with opaque alpha.
    const OPAQUE: u32 = 0xFF00_0000;

    /// Host mirror of [`blend_deco`] and [`apply_deco_kernel`]'s smudge gather.
    ///
    /// A `#[cube] fn` cannot be called from a host test, so the reference
    /// formula (`Editing.cpp:440-540`) is pinned here in plain Rust; the kernel
    /// must stay byte-identical to it.
    fn blend_deco_colour(
        current: u32,
        colour: u32,
        mode: DecoMode,
        neighbours: &[u32],
    ) -> Option<u32> {
        let mut ta = channel(current, 24);
        let mut tr = channel(current, 16);
        let mut tg = channel(current, 8);
        let mut tb = channel(current, 0);
        let col_a = channel(colour, 24);
        let col_r = channel(colour, 16);
        let col_g = channel(colour, 8);
        let col_b = channel(colour, 0);
        match mode {
            DecoMode::Draw => {
                ta = col_a;
                tr = col_r;
                tg = col_g;
                tb = col_b;
            }
            DecoMode::Clear => {
                ta = 0.0;
                tr = 0.0;
                tg = 0.0;
                tb = 0.0;
            }
            DecoMode::Add => {
                tr += (col_r * C_DECO_STRENGTH) * col_a;
                tg += (col_g * C_DECO_STRENGTH) * col_a;
                tb += (col_b * C_DECO_STRENGTH) * col_a;
            }
            DecoMode::Subtract => {
                tr -= (col_r * C_DECO_STRENGTH) * col_a;
                tg -= (col_g * C_DECO_STRENGTH) * col_a;
                tb -= (col_b * C_DECO_STRENGTH) * col_a;
            }
            DecoMode::Multiply => {
                tr *= 1.0 + (col_r * C_DECO_STRENGTH) * col_a;
                tg *= 1.0 + (col_g * C_DECO_STRENGTH) * col_a;
                tb *= 1.0 + (col_b * C_DECO_STRENGTH) * col_a;
            }
            DecoMode::Divide => {
                tr /= 1.0 + (col_r * C_DECO_STRENGTH) * col_a;
                tg /= 1.0 + (col_g * C_DECO_STRENGTH) * col_a;
                tb /= 1.0 + (col_b * C_DECO_STRENGTH) * col_a;
            }
            DecoMode::Smudge => {
                if neighbours.is_empty() {
                    return None;
                }
                let num = neighbours.len() as f32;
                let (mut tas, mut trs, mut tgs, mut tbs) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for &neighbour in neighbours {
                    tas += srgb_to_linear(channel(neighbour, 24));
                    trs += srgb_to_linear(channel(neighbour, 16));
                    tgs += srgb_to_linear(channel(neighbour, 8));
                    tbs += srgb_to_linear(channel(neighbour, 0));
                }
                ta = linear_to_srgb(tas / num);
                tr = linear_to_srgb(trs / num);
                tg = linear_to_srgb(tgs / num);
                tb = linear_to_srgb(tbs / num);
            }
        }
        Some((byte(ta) << 24) | (byte(tr) << 16) | (byte(tg) << 8) | byte(tb))
    }

    fn channel(word: u32, shift: u32) -> f32 {
        ((word >> shift) & 0xFF) as f32 / 255.0
    }

    fn byte(value: f32) -> u32 {
        (value * 255.0 + 0.5).clamp(0.0, 255.0) as u32
    }

    fn srgb_to_linear(value: f32) -> f32 {
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }

    fn linear_to_srgb(value: f32) -> f32 {
        if value <= 0.003_130_8 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        }
    }

    /// `DECO_DRAW` copies the tool colour, including its alpha.
    #[test]
    fn draw_replaces_the_whole_colour() {
        let colour = 0x8012_34AB;
        assert_eq!(
            blend_deco_colour(OPAQUE, colour, DecoMode::Draw, &[]),
            Some(colour)
        );
        assert_eq!(
            blend_deco_colour(0xFF12_34AB, 0x00FF_FFFF, DecoMode::Draw, &[]),
            Some(0x00FF_FFFF)
        );
    }

    /// `DECO_CLEAR` zeroes every channel, alpha included.
    #[test]
    fn clear_zeroes_the_colour() {
        assert_eq!(
            blend_deco_colour(0x8012_34AB, 0xFFFF_FFFF, DecoMode::Clear, &[]),
            Some(0)
        );
    }

    /// `DECO_ADD` adds `colour * 0.01 * alpha` and leaves alpha alone.
    #[test]
    fn add_scales_the_colour_by_strength_and_alpha() {
        assert_eq!(
            blend_deco_colour(0xFF20_4060, 0xFF80_4040, DecoMode::Add, &[]),
            Some(0xFF21_4161)
        );
        assert_eq!(
            blend_deco_colour(0x4020_4060, 0xFF80_4040, DecoMode::Add, &[]),
            Some(0x4021_4161)
        );
        assert_eq!(
            blend_deco_colour(0xFF20_4060, 0x00FF_FFFF, DecoMode::Add, &[]),
            Some(0xFF20_4060)
        );
    }

    /// `DECO_SUBTRACT` subtracts `colour * 0.01 * alpha`.
    #[test]
    fn subtract_scales_the_colour_by_strength_and_alpha() {
        assert_eq!(
            blend_deco_colour(0xFF20_4060, 0xFF80_4040, DecoMode::Subtract, &[]),
            Some(0xFF1F_3F5F)
        );
    }

    /// `DECO_MULTIPLY` scales by `1 + colour * 0.01 * alpha`.
    #[test]
    fn multiply_scales_up() {
        assert_eq!(
            blend_deco_colour(0xFFC8_8040, 0xFFFF_FFFF, DecoMode::Multiply, &[]),
            Some(0xFFCA_8141)
        );
        assert_eq!(
            blend_deco_colour(0xFFC8_8040, 0x00FF_FFFF, DecoMode::Multiply, &[]),
            Some(0xFFC8_8040)
        );
    }

    /// `DECO_DIVIDE` divides by `1 + colour * 0.01 * alpha`.
    #[test]
    fn divide_scales_down() {
        assert_eq!(
            blend_deco_colour(0xFFC8_8040, 0xFFFF_FFFF, DecoMode::Divide, &[]),
            Some(0xFFC6_7F3F)
        );
        assert_eq!(
            blend_deco_colour(0xFFC8_8040, 0x00FF_FFFF, DecoMode::Divide, &[]),
            Some(0xFFC8_8040)
        );
    }

    /// `DECO_SMUDGE` averages the neighbours in linear light.
    #[test]
    fn smudge_averages_the_neighbours() {
        assert_eq!(
            blend_deco_colour(
                0xFF00_0000,
                0xFFFF_FFFF,
                DecoMode::Smudge,
                &[0x8080_8080, 0x8080_8080]
            ),
            Some(0x8080_8080)
        );
    }

    /// A smudge with nothing decorated around it leaves the particle alone.
    #[test]
    fn smudge_without_neighbours_is_a_no_op() {
        assert_eq!(
            blend_deco_colour(0xFF00_0000, 0xFFFF_FFFF, DecoMode::Smudge, &[]),
            None
        );
    }

    /// `Editing.cpp:519-540`: the smudge result replaces every channel,
    /// including alpha, with the neighbour average — an undecorated pixel gains
    /// the neighbours' alpha instead of keeping its own.
    #[test]
    fn smudge_takes_the_neighbour_alpha() {
        assert_eq!(
            blend_deco_colour(0, 0xFFFF_FFFF, DecoMode::Smudge, &[0xFF00_0000]),
            Some(0xFF00_0000)
        );
    }

    /// Every mode clamps to a byte instead of wrapping.
    #[test]
    fn blend_clamps_to_a_byte() {
        assert_eq!(
            blend_deco_colour(0xFFFF_FFFF, 0xFFFF_FFFF, DecoMode::Add, &[]),
            Some(0xFFFF_FFFF)
        );
        assert_eq!(
            blend_deco_colour(0xFF01_0101, 0xFFFF_FFFF, DecoMode::Subtract, &[]),
            Some(0xFF00_0000)
        );
    }
}
