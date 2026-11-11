//! Device-side flood fill and property-tool writes.
//!
//! Both user actions used to rewind the whole simulation to the CPU:
//!
//! * `SimulationEngine::flood_fill` called `ensure_host_current()` (the full
//!   `pmap`/`photons`/SoA shadow, tens of megabytes), ran a host BFS, called
//!   `create_part` once per pixel and uploaded every allocation again;
//! * `AppState::apply_property_stroke` / `apply_property_fill` read the same
//!   shadow back before every stroke and fill just to resolve the particle ids
//!   under the brush.
//!
//! The device is authoritative, so both actions are resolved there. The host
//! uploads a seed pixel or a packed brush list (a few hundred words) and the
//! kernels below do the walk and the write in place.
//!
//! # Particle flood fill (`FloodParts`)
//!
//! `Simulation::FloodParts` (`Editing.cpp:879-1040`) walks the empty region
//! containing the clicked pixel with a scanline flood. Here the region is grown
//! by `flood_spread_kernel`: every marked, empty pixel marks its four empty
//! neighbours, so a pixel at Manhattan distance `d` is reached after `d`
//! rounds. The walk runs until a round marks nothing new (see
//! [`spread_until_stable`]), which is the reference's own stop condition: a
//! corridor that winds longer than `xres + yres` is still filled.
//!
//! The marked pixels are then turned into creation records with
//! `reaction_kernel::shared::request_create` and materialised by the regular
//! creation drain, so the flood uses the same allocation path as a reaction
//! (default properties, spawn window, queue cap). `SimBuffers::GPU_CREATE_COUNT`
//! records is therefore the hard fill cap: the surplus is dropped, exactly as a
//! full queue drops reaction creations. That cap is why the old
//! `FLOOD_FILL_BUDGET` pixel budget is gone — it was 200 000, an order of
//! magnitude above what the creation queue can ever materialise, so it no
//! longer bounded anything. The drain does not repeat `create_part`'s
//! `CreateAllowed` check (`SPAWN`, `STKM`, `FIGH`), the same limitation the
//! reaction-spawn path already has.
//!
//! # Property tool (`flood_prop`)
//!
//! `Simulation::flood_prop` (`Simulation.cpp:520-590`) rewrites one scalar
//! field of every particle of the clicked type connected to the seed. The port
//! additionally requires the candidate to hold the same value the seed holds
//! (see `apply.rs`), so the walk below compares the property through a
//! device-side snapshot (`gather_int_kernel` / `gather_float_kernel` /
//! `gather_word_kernel`) instead of binding every per-particle array.
//!
//! The fills/strokes are one-shot click actions: the scratch `marks` and
//! `scratch` buffers and the `params` word are allocated for the call and
//! dropped afterwards, so nothing here is on a per-frame path.

// Clippy's rewrite for these lints uses methods the device IR does not have
// (`is_multiple_of`, `clamp`, `Range::contains`, `saturating_sub`), and
// collapsing a nested `if` changes what the CubeCL macro expands, so the plain
// operators are deliberate.
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_is_multiple_of)]

use cubecl::prelude::*;
use cubecl::server::Handle;

use tpt_core::constants::*;

use crate::buffers::SimBuffers;
use crate::reaction_kernel::shared::{elem_word, launch_grid, pmap_at, pmap_slot, request_create};

/// `PT_NONE` marks a free slot; mirrored for the kernel bodies (rule 1).
const C_PT_NONE: u32 = 0;
const _: () = assert!(PT_NONE == 0);
const C_PT_NUM: u32 = PT_NUM as u32;
const _: () = assert!(C_PT_NUM as usize == PT_NUM);
/// `PT_SPRK`: a spark is a reaction state, not a fillable element.
const C_PT_SPRK: u32 = 12;
const _: () = assert!(PT_SPRK == 12);

/// `SimBuffers::GPU_CREATE_COUNT` mirrored for kernel bodies.
const _: () = assert!(SimBuffers::GPU_CREATE_COUNT == 16384);
/// `[count, record..]` word count of the shared creation queue.
const CREATE_QUEUE_WORDS: usize = SimBuffers::GPU_CREATE_COUNT * SimBuffers::CREATE_WORDS + 1;
/// `GpuElementProps` is uploaded as 32 `u32` words per element.
const _: () = assert!(core::mem::size_of::<tpt_core::elements::GpuElementProps>() == 128);
/// Sentinel slot id used by [`slot_at`].
const C_NO_SLOT: u32 = 0xFFFF_FFFF;
/// Bit set in [`write_target_pixel`]'s result when the thread owns no pixel.
const C_NO_PIXEL: u32 = 0x8000_0000;

/// Property ids, stable across the app/engine boundary and equal to
/// `PropertyKind::index` in `tpt-app` (the app converts through
/// [`PropertyField`]).
const C_FIELD_TMP: u32 = 0;
const C_FIELD_LIFE: u32 = 1;
const C_FIELD_CTYPE: u32 = 2;
const C_FIELD_TMP2: u32 = 3;
const C_FIELD_TMP3: u32 = 4;
const C_FIELD_TMP4: u32 = 5;
const C_FIELD_TYPE: u32 = 6;
const C_FIELD_TEMP: u32 = 7;
const C_FIELD_VX: u32 = 8;
const C_FIELD_VY: u32 = 9;
const C_FIELD_DCOLOUR: u32 = 10;

/// `WORD_PROPERTIES` of the packed element table (`shared.rs`).
const C_WORD_PROPERTIES: u32 = 1;
/// `TYPE_ENERGY`: the element lives in the `photons` layer.
const C_TYPE_ENERGY: u32 = 0x0000_0010;
const _: () = assert!(TYPE_ENERGY == 0x0000_0010);
/// `MIN_TEMP` / `MAX_TEMP`: the window `SimulationEngine::set_particle_property`
/// clamps a temperature write to.
const C_MIN_TEMP: f32 = 0.0;
const C_MAX_TEMP: f32 = 9999.0;
const _: () = assert!(MIN_TEMP == C_MIN_TEMP && MAX_TEMP == C_MAX_TEMP);

/// `[seed_ty, writes, reserved, reserved]` in the per-call params word.
const P_SEED_TY: u32 = 0;
const P_WRITES: u32 = 1;
const P_WORDS: usize = 4;

/// Mark value written by the flood kernels; any non-zero value means "visited".
const C_MARK: u32 = 1;

/// Seed a `u32` device local from a runtime anchor (AGENTS rule 4).
#[cube]
#[allow(clippy::erasing_op)]
fn seed_u32(anchor: u32, value: u32) -> u32 {
    anchor * 0u32 + value
}

/// `i32` variant of [`seed_u32`].
#[cube]
#[allow(clippy::erasing_op)]
fn seed_i32(anchor: i32, value: i32) -> i32 {
    anchor * 0i32 + value
}

/// Resolves the particle slot at a pixel, [`C_NO_SLOT`] when empty.
///
/// Mirrors `SimBuffers::particle_index_at`: the matter map first, then the
/// energy map, and a word without id bits means "no particle" (the packing
/// stores `id + 1`). Same helper as `edit_kernel::slot_at`.
#[cube]
fn slot_at(pmap: &[u32], photons: &[u32], xres: u32, yres: u32, px: i32, py: i32) -> u32 {
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

/// `SimulationEngine::set_particle_property`'s temperature clamp.
#[cube]
fn clamp_temperature(value: f32) -> f32 {
    let mut out = value;
    if out < 0.0f32 {
        out = 0.0f32;
    }
    if out > 9999.0f32 {
        out = 9999.0f32;
    }
    out
}

// ---------------------------------------------------------------------------
// Convergence of the one-shot floods
// ---------------------------------------------------------------------------

/// Spread rounds per convergence chunk; each chunk ends with one probe round.
const SPREAD_CHUNK: u32 = 32;

/// Safety cap on chunks. A 4-connected region needs at most one round per
/// pixel, and the fixed point is normally reached in a handful of rounds, so
/// this only bounds a buggy kernel; it is not a distance limit.
const SPREAD_MAX_CHUNKS: u32 = 4096;

/// Reads the progress counter written by a spread round.
///
/// A read error is reported as "still changing" so the caller's cap decides
/// when to stop instead of accepting a possibly truncated flood.
fn read_changed(bufs: &SimBuffers, changed: &Handle) -> u32 {
    let Ok(readback) = bufs.client.read(vec![changed.clone()]) else {
        return 1;
    };
    readback
        .first()
        .and_then(|bytes| bytes.as_chunks::<4>().0.first())
        .map(|chunk| u32::from_le_bytes(*chunk))
        .unwrap_or(1)
}

/// Runs `round` until a full round marks no new pixel.
///
/// The counter is cleared before a probe round so the chunk's earlier progress
/// does not count: only the probe's own writes decide whether the walk has
/// reached its fixed point. The flood is monotone in the marks, so a converged
/// probe means the whole connected region is marked.
fn spread_until_stable(bufs: &SimBuffers, changed: &Handle, mut round: impl FnMut()) {
    for _ in 0..SPREAD_MAX_CHUNKS {
        for _ in 0..SPREAD_CHUNK {
            round();
        }
        unsafe {
            clear_changed_kernel::launch_unchecked(
                &bufs.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(changed.clone(), 1),
            );
        }
        round();
        if read_changed(bufs, changed) == 0 {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Particle flood fill (`FloodParts`)
// ---------------------------------------------------------------------------

/// Zero the transient mark buffer.
#[cube(launch_unchecked)]
fn clear_marks_kernel(marks: &mut [u32], pixels: u32) {
    let i = ABSOLUTE_POS;
    if i < pixels as usize {
        marks[i] = 0u32;
    }
}

/// Reset the fixed-point progress counter before a probe round.
#[cube(launch_unchecked)]
fn clear_changed_kernel(changed: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        changed[0].store(0u32);
    }
}

/// Mark the seeded pixel when it is empty.
///
/// The host already checked the canvas bounds; the pixel-map read decides
/// whether there is anything to fill, exactly like `FloodParts`'s
/// `FloodFillPmapCheck(x, y, cm)`.
#[cube(launch_unchecked)]
fn flood_seed_kernel(pmap: &[u32], marks: &mut [u32], seed: u32, pixels: u32) {
    if ABSOLUTE_POS == 0usize {
        let s = seed as usize;
        if s < pixels as usize {
            if pmap[s] == 0u32 {
                marks[s] = C_MARK;
            }
        }
    }
}

/// One round of the empty-region walk: every marked pixel marks its four empty
/// neighbours.
///
/// All writers store [`C_MARK`], so the read/write race between neighbouring
/// threads only moves a pixel's discovery forward; it cannot corrupt the mark.
/// Every newly marked pixel bumps `changed`, which is how the host knows the
/// walk has reached its fixed point and can stop launching rounds.
#[cube(launch_unchecked)]
fn flood_spread_kernel(
    pmap: &[u32],
    marks: &mut [u32],
    changed: &mut [Atomic<u32>],
    pixels: u32,
    xres: u32,
    yres: u32,
) {
    let i = ABSOLUTE_POS;
    if i < pixels as usize {
        let mark = marks[i];
        if mark != 0u32 {
            let x = (i as u32) % xres;
            let y = (i as u32) / xres;
            if x > 0u32 {
                let n = i - 1usize;
                if pmap[n] == 0u32 && marks[n] == 0u32 {
                    marks[n] = mark;
                    changed[0].fetch_add(1u32);
                }
            }
            if x + 1u32 < xres {
                let n = i + 1usize;
                if pmap[n] == 0u32 && marks[n] == 0u32 {
                    marks[n] = mark;
                    changed[0].fetch_add(1u32);
                }
            }
            if y > 0u32 {
                let n = i - xres as usize;
                if pmap[n] == 0u32 && marks[n] == 0u32 {
                    marks[n] = mark;
                    changed[0].fetch_add(1u32);
                }
            }
            if y + 1u32 < yres {
                let n = i + xres as usize;
                if pmap[n] == 0u32 && marks[n] == 0u32 {
                    marks[n] = mark;
                    changed[0].fetch_add(1u32);
                }
            }
        }
    }
}

/// Turn every marked pixel into a creation record and clear its mark.
///
/// `request_create` caps the queue at `GPU_CREATE_COUNT` records and drops the
/// surplus, which is where the fill budget now lives. The occupancy re-check
/// keeps a pixel that filled in between the walk and this dispatch from being
/// overwritten.
#[cube(launch_unchecked)]
fn flood_create_kernel(
    pmap: &[u32],
    marks: &mut [u32],
    create_queue: &mut [Atomic<u32>],
    pixels: u32,
    xres: u32,
    ty: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < pixels {
        if marks[i as usize] != 0u32 {
            marks[i as usize] = 0u32;
            if pmap[i as usize] == 0u32 {
                request_create(
                    create_queue,
                    ty,
                    (i % xres) as i32,
                    (i / xres) as i32,
                    0u32,
                    0i32,
                    0u32,
                    0.0f32,
                    0i32,
                    0i32,
                    0i32,
                    0i32,
                    0u32,
                    0u32,
                    0.0f32,
                    0.0f32,
                );
            }
        }
    }
}

/// Flood-fill the empty region containing `(x, y)` with `elem_ty`, on the
/// device.
///
/// The host only validates the element id and the seed pixel; the region walk
/// and the creations happen on the device. The creations land in
/// `SimBuffers::GPU_CREATE_START..GPU_SPAWN_END`, so the host shadow does not
/// learn about them until something explicitly reads it back.
pub fn flood_fill_device(bufs: &mut SimBuffers, x: i32, y: i32, elem_ty: u32) {
    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    if xres == 0 || yres == 0 || elem_ty == C_PT_NONE || elem_ty >= C_PT_NUM {
        return;
    }
    // `FloodParts` refuses a spark (`Editing.cpp:888-893`): a spark only exists
    // on a conductor, and the creation drain cannot reproduce that check.
    if elem_ty == C_PT_SPRK {
        return;
    }
    if x < 0 || y < 0 || x >= xres as i32 || y >= yres as i32 {
        return;
    }
    // A pending host edit must reach the device before the walk reads the map,
    // and before the creations below would be overwritten by its upload.
    if bufs.host_dirty {
        bufs.sync_host_to_gpu();
    }

    let pixels = (xres * yres) as usize;
    let marks = bufs.client.empty(pixels * 4);
    let (cubes, dim) = launch_grid(pixels);
    let seed = y as u32 * xres + x as u32;

    // Arm the creation queue for the records [`flood_create_kernel`] pushes
    // below: empty it and rebuild the device free-slot window.
    crate::reaction_kernel::begin_pending_creates(bufs);
    let changed = bufs
        .client
        .create_from_slice(bytemuck::cast_slice(&[0u32]));

    unsafe {
        clear_marks_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(marks.clone(), pixels),
            pixels as u32,
        );
        flood_seed_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
            BufferArg::from_raw_parts(marks.clone(), pixels),
            seed,
            pixels as u32,
        );
    }
    // `FloodParts` has no distance cap; the walk's own fixed point is the only
    // stop condition, so a corridor that winds longer than `xres + yres` is
    // still filled completely.
    spread_until_stable(bufs, &changed, || unsafe {
        flood_spread_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
            BufferArg::from_raw_parts(marks.clone(), pixels),
            BufferArg::from_raw_parts(changed.clone(), 1),
            pixels as u32,
            xres,
            yres,
        );
    });
    unsafe {
        flood_create_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
            BufferArg::from_raw_parts(marks.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            pixels as u32,
            xres,
            elem_ty,
        );
    }

    // Allocate and initialise the queued particles through the same drain the
    // Game of Life births use (`drain_pending_creates`). The queue is armed
    // first so the flood's records start at index 0.
    let seed = (bufs.sim_tick as u32)
        .wrapping_mul(0x9E37_79B9)
        .rotate_left(13)
        ^ 0x85EB_CA6B;
    crate::reaction_kernel::drain_pending_creates(bufs, seed);

    bufs.gpu_spawn_used = true;
    bufs.gpu_dirty = true;
    bufs.gpu_particles_dirty = true;
    bufs.gpu_maps_dirty = true;
}

// ---------------------------------------------------------------------------
// Property tool: value snapshot
// ---------------------------------------------------------------------------

/// Slot field a property write targets.
///
/// [`Self::id`] returns the app's `PropertyKind::index` value, so the two
/// crates only share one integer contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyField {
    /// `tmp` scratch value.
    Tmp,
    /// Remaining life.
    Life,
    /// Clone / transition element.
    Ctype,
    /// `tmp2` scratch value.
    Tmp2,
    /// `tmp3` scratch value.
    Tmp3,
    /// `tmp4` scratch value.
    Tmp4,
    /// Element type.
    Type,
    /// Temperature.
    Temp,
    /// Horizontal velocity.
    Vx,
    /// Vertical velocity.
    Vy,
    /// Decoration colour.
    Dcolour,
}

impl PropertyField {
    /// Kernel id; see the `C_FIELD_*` constants.
    pub fn id(self) -> u32 {
        match self {
            Self::Tmp => C_FIELD_TMP,
            Self::Life => C_FIELD_LIFE,
            Self::Ctype => C_FIELD_CTYPE,
            Self::Tmp2 => C_FIELD_TMP2,
            Self::Tmp3 => C_FIELD_TMP3,
            Self::Tmp4 => C_FIELD_TMP4,
            Self::Type => C_FIELD_TYPE,
            Self::Temp => C_FIELD_TEMP,
            Self::Vx => C_FIELD_VX,
            Self::Vy => C_FIELD_VY,
            Self::Dcolour => C_FIELD_DCOLOUR,
        }
    }
}

/// Value a property write stores, kept integer/float apart like the app's
/// `PropertyValue` so an integer slot never round-trips through `f32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropertyValue {
    /// Integer property.
    Int(i64),
    /// Floating-point property.
    Float(f32),
}

impl PropertyValue {
    /// Bit pattern stored in an integer slot (two's complement), clamped the
    /// way `edit_kernel::set_particle_field` clamps the same fields.
    fn int_bits(self, field: PropertyField) -> u32 {
        let v = match self {
            Self::Int(v) => v,
            Self::Float(v) => v.round() as i64,
        };
        match field {
            // `ctype` names an element and has no sign.
            PropertyField::Ctype => v.clamp(0, C_PT_NUM as i64 - 1) as u32,
            _ => v.clamp(i32::MIN as i64, i32::MAX as i64) as i32 as u32,
        }
    }

    /// Float value of a `f32` slot.
    fn as_f32(self) -> f32 {
        match self {
            Self::Int(v) => v as f32,
            Self::Float(v) => v,
        }
    }

    /// Value of a `u32` slot (`type`, `dcolour`); `None` for an invalid element.
    fn as_word(self, field: PropertyField) -> Option<u32> {
        let v = match self {
            Self::Int(v) => v,
            Self::Float(v) => v.round() as i64,
        };
        match field {
            PropertyField::Type => {
                if (0..C_PT_NUM as i64).contains(&v) {
                    Some(v as u32)
                } else {
                    None
                }
            }
            PropertyField::Dcolour => Some(v.clamp(0, u32::MAX as i64) as u32),
            _ => None,
        }
    }
}

/// Snapshot `tmp`..`ctype` into the comparison scratch buffer.
#[cube(launch_unchecked)]
fn gather_int_kernel(
    part_tmp: &[i32],
    part_tmp2: &[i32],
    part_tmp3: &[i32],
    part_tmp4: &[i32],
    part_life: &[i32],
    part_ctype: &[u32],
    scratch: &mut [u32],
    slots: u32,
    field: u32,
) {
    let s = ABSOLUTE_POS;
    if s < slots as usize {
        let mut v = seed_u32(field, 0u32);
        if field == C_FIELD_TMP {
            v = part_tmp[s] as u32;
        } else if field == C_FIELD_LIFE {
            v = part_life[s] as u32;
        } else if field == C_FIELD_CTYPE {
            v = part_ctype[s];
        } else if field == C_FIELD_TMP2 {
            v = part_tmp2[s] as u32;
        } else if field == C_FIELD_TMP3 {
            v = part_tmp3[s] as u32;
        } else if field == C_FIELD_TMP4 {
            v = part_tmp4[s] as u32;
        }
        scratch[s] = v;
    }
}

/// Snapshot `temp`/`vx`/`vy` into the comparison scratch buffer.
#[cube(launch_unchecked)]
fn gather_float_kernel(
    part_temp: &[f32],
    part_vx: &[f32],
    part_vy: &[f32],
    scratch: &mut [u32],
    slots: u32,
    field: u32,
) {
    let s = ABSOLUTE_POS;
    if s < slots as usize {
        let mut v = seed_u32(field, 0u32);
        if field == C_FIELD_TEMP {
            v = u32::reinterpret(part_temp[s]);
        } else if field == C_FIELD_VX {
            v = u32::reinterpret(part_vx[s]);
        } else if field == C_FIELD_VY {
            v = u32::reinterpret(part_vy[s]);
        }
        scratch[s] = v;
    }
}

/// Snapshot `type`/`dcolour` into the comparison scratch buffer.
#[cube(launch_unchecked)]
fn gather_word_kernel(
    part_ty: &[u32],
    part_dcolour: &[u32],
    scratch: &mut [u32],
    slots: u32,
    field: u32,
) {
    let s = ABSOLUTE_POS;
    if s < slots as usize {
        let mut v = seed_u32(field, 0u32);
        if field == C_FIELD_TYPE {
            v = part_ty[s];
        } else if field == C_FIELD_DCOLOUR {
            v = part_dcolour[s];
        }
        scratch[s] = v;
    }
}

// ---------------------------------------------------------------------------
// Property tool: walk and write
// ---------------------------------------------------------------------------

/// Capture the seed particle's type and mark its pixel.
///
/// Runs once: the propagation kernels then compare against the captured type,
/// so no thread has to re-read the seed.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn property_seed_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    marks: &mut [u32],
    params: &mut [u32],
    xres: u32,
    yres: u32,
    seed_x: i32,
    seed_y: i32,
    slots: u32,
) {
    if ABSOLUTE_POS == 0usize {
        let slot = slot_at(pmap, photons, xres, yres, seed_x, seed_y);
        if slot != C_NO_SLOT {
            let s = slot as usize;
            if s < slots as usize && part_ty[s] != C_PT_NONE {
                params[P_SEED_TY as usize] = part_ty[s];
                marks[(seed_y as u32 * xres + seed_x as u32) as usize] = C_MARK;
            }
        }
    }
}

/// One round of the connected-region walk over particles that share the seed's
/// type.
///
/// `flood_prop` (`Simulation.cpp:520-597`) compares the element type only -
/// `FloodFillPmapCheck` (`Simulation.cpp:501-513`) never looks at the property
/// value, so a region holding mixed values is filled in one go.
#[cube(launch_unchecked)]
fn property_spread_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    marks: &mut [u32],
    params: &[u32],
    changed: &mut [Atomic<u32>],
    pixels: u32,
    xres: u32,
    yres: u32,
    slots: u32,
) {
    let i = ABSOLUTE_POS;
    if i < pixels as usize {
        if marks[i] != 0u32 {
            let ty = params[P_SEED_TY as usize];
            let x = (i as u32 % xres) as i32;
            let y = (i as u32 / xres) as i32;
            let mut k = seed_u32(xres, 0u32);
            while k < 4u32 {
                let mut nx = seed_i32(x, x);
                let mut ny = seed_i32(x, y);
                if k == 0u32 {
                    nx = x - 1i32;
                } else if k == 1u32 {
                    nx = x + 1i32;
                } else if k == 2u32 {
                    ny = y - 1i32;
                } else {
                    ny = y + 1i32;
                }
                let slot = slot_at(pmap, photons, xres, yres, nx, ny);
                if slot != C_NO_SLOT {
                    let n = slot as usize;
                    if n < slots as usize
                        && part_ty[n] == ty
                        && (nx as u32) < xres
                        && (ny as u32) < yres
                    {
                        let mi = (ny as u32 * xres + nx as u32) as usize;
                        if marks[mi] == 0u32 {
                            marks[mi] = C_MARK;
                            changed[0].fetch_add(1u32);
                        }
                    }
                }
                k += 1u32;
            }
        }
    }
}

/// Reads the pixel a write thread owns, packed `x | y << 16`.
///
/// `mode == 0` is the brush path (`points[i]`, the shared edit point list),
/// `mode == 1` the fill path (one thread per pixel, selected by `marks[i]`).
/// [`C_NO_PIXEL`] means "this thread owns no particle", so the caller skips it.
#[cube]
fn write_target_pixel(i: usize, mode: u32, points: &[u32], marks: &[u32], xres: u32) -> u32 {
    let mut packed = seed_u32(xres, C_NO_PIXEL);
    if mode == 0u32 {
        packed = points[i];
    } else if marks[i] != 0u32 {
        let x = (i as u32) % xres;
        let y = (i as u32) / xres;
        packed = (x & 0xFFFFu32) | (y << 16u32);
    }
    packed
}

/// Writes one `tmp`..`ctype` field of every selected particle.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn write_int_kernel(
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
    marks: &[u32],
    params: &mut [Atomic<u32>],
    count: u32,
    xres: u32,
    yres: u32,
    field: u32,
    value: u32,
    mode: u32,
) {
    let i = ABSOLUTE_POS;
    if i < count as usize {
        let packed = write_target_pixel(i, mode, points, marks, xres);
        if (packed & C_NO_PIXEL) == 0u32 {
            let px = (packed & 0xFFFFu32) as i32;
            let py = (packed >> 16u32) as i32;
            let slot = slot_at(pmap, photons, xres, yres, px, py);
            if slot != C_NO_SLOT {
                let id = slot as usize;
                if part_ty[id] != C_PT_NONE {
                    let bits = value as i32;
                    let mut wrote = true;
                    if field == C_FIELD_TMP {
                        part_tmp[id] = bits;
                    } else if field == C_FIELD_LIFE {
                        part_life[id] = bits;
                    } else if field == C_FIELD_CTYPE {
                        part_ctype[id] = value;
                    } else if field == C_FIELD_TMP2 {
                        part_tmp2[id] = bits;
                    } else if field == C_FIELD_TMP3 {
                        part_tmp3[id] = bits;
                    } else if field == C_FIELD_TMP4 {
                        part_tmp4[id] = bits;
                    } else {
                        wrote = false;
                    }
                    if wrote {
                        params[P_WRITES as usize].fetch_add(1u32);
                    }
                }
            }
        }
    }
}

/// Writes one `temp`/`vx`/`vy` field of every selected particle.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn write_float_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_temp: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    points: &[u32],
    marks: &[u32],
    params: &mut [Atomic<u32>],
    count: u32,
    xres: u32,
    yres: u32,
    field: u32,
    value: f32,
    mode: u32,
) {
    let i = ABSOLUTE_POS;
    if i < count as usize {
        let packed = write_target_pixel(i, mode, points, marks, xres);
        if (packed & C_NO_PIXEL) == 0u32 {
            let px = (packed & 0xFFFFu32) as i32;
            let py = (packed >> 16u32) as i32;
            let slot = slot_at(pmap, photons, xres, yres, px, py);
            if slot != C_NO_SLOT {
                let id = slot as usize;
                if part_ty[id] != C_PT_NONE {
                    let mut wrote = true;
                    if field == C_FIELD_TEMP {
                        part_temp[id] = clamp_temperature(value);
                    } else if field == C_FIELD_VX {
                        part_vx[id] = value;
                    } else if field == C_FIELD_VY {
                        part_vy[id] = value;
                    } else {
                        wrote = false;
                    }
                    if wrote {
                        params[P_WRITES as usize].fetch_add(1u32);
                    }
                }
            }
        }
    }
}

/// Writes `type` / `dcolour` of every selected particle.
///
/// A `type` write also moves the pixel's map word to the layer the new element
/// belongs to (`TYPE_ENERGY` decides `pmap` vs `photons`). The host path used to
/// rebuild the whole spatial map for one type change
/// (`SimulationEngine::set_particle_property`'s `moved` branch); updating the
/// one word keeps the device map consistent without that rewrite.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn write_word_kernel(
    pmap: &mut [u32],
    photons: &mut [u32],
    part_ty: &mut [u32],
    part_dcolour: &mut [u32],
    elem_props: &[u32],
    points: &[u32],
    marks: &[u32],
    params: &mut [Atomic<u32>],
    count: u32,
    xres: u32,
    yres: u32,
    field: u32,
    value: u32,
    mode: u32,
) {
    let i = ABSOLUTE_POS;
    if i < count as usize {
        let packed = write_target_pixel(i, mode, points, marks, xres);
        let px = (packed & 0xFFFFu32) as i32;
        let py = (packed >> 16u32) as i32;
        if (packed & C_NO_PIXEL) == 0u32
            && px >= 0i32
            && py >= 0i32
            && (px as u32) < xres
            && (py as u32) < yres
        {
            let index = (py as u32 * xres + px as u32) as usize;
            let mut word = pmap[index];
            let mut matter = (word >> 9u32) != 0u32;
            if (word >> 9u32) == 0u32 {
                word = photons[index];
                matter = false;
            }
            let mut slot = 0xFFFF_FFFFu32;
            if (word >> 9u32) != 0u32 {
                slot = pmap_slot(word) as u32;
            }
            if slot != C_NO_SLOT {
                let id = slot as usize;
                if part_ty[id] != C_PT_NONE {
                    let mut wrote = true;
                    if field == C_FIELD_TYPE {
                        let properties = elem_word(elem_props, value, C_WORD_PROPERTIES);
                        let energy = (properties & C_TYPE_ENERGY) != 0u32;
                        let new_word = ((id as u32 + 1u32) << 9u32) | value;
                        part_ty[id] = value;
                        if value == C_PT_NONE {
                            if matter {
                                pmap[index] = 0u32;
                            } else {
                                photons[index] = 0u32;
                            }
                        } else if matter {
                            if energy {
                                pmap[index] = 0u32;
                                photons[index] = new_word;
                            } else {
                                pmap[index] = new_word;
                            }
                        } else if energy {
                            photons[index] = new_word;
                        } else {
                            photons[index] = 0u32;
                            pmap[index] = new_word;
                        }
                    } else if field == C_FIELD_DCOLOUR {
                        part_dcolour[id] = value;
                    } else {
                        wrote = false;
                    }
                    if wrote {
                        params[P_WRITES as usize].fetch_add(1u32);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

/// Uploads a packed pixel list into the shared edit scratch buffer and returns
/// the number of points the kernels must process (`edit_kernel`'s shape).
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

/// Allocates the one-word placeholder bound where a kernel expects `marks`.
fn dummy_marks(bufs: &SimBuffers) -> Handle {
    bufs.client.empty(4)
}

/// Writes every brushed particle's `field` on the device.
///
/// The brush path only needs the pixel list; no particle SoA is read back.
pub fn set_property_stroke(
    bufs: &mut SimBuffers,
    points: &[(i32, i32)],
    field: PropertyField,
    value: PropertyValue,
) {
    if points.is_empty() {
        return;
    }
    // A pending host edit must reach the device before the write, or the next
    // upload would overwrite it.
    if bufs.host_dirty {
        bufs.sync_host_to_gpu();
    }
    let count = upload_points(bufs, points);
    if count == 0 {
        return;
    }
    let marks = dummy_marks(bufs);
    let params = bufs
        .client
        .create_from_slice(bytemuck::cast_slice(&[0u32; P_WORDS]));
    let field_id = field.id();
    let (cubes, dim) = launch_grid(count as usize);
    unsafe {
        match field {
            PropertyField::Tmp
            | PropertyField::Life
            | PropertyField::Ctype
            | PropertyField::Tmp2
            | PropertyField::Tmp3
            | PropertyField::Tmp4 => {
                write_int_kernel::launch_unchecked(
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
                    BufferArg::from_raw_parts(marks.clone(), 1),
                    BufferArg::from_raw_parts(params.clone(), P_WORDS),
                    count,
                    bufs.config.xres,
                    bufs.config.yres,
                    field_id,
                    value.int_bits(field),
                    0u32,
                );
            }
            PropertyField::Temp | PropertyField::Vx | PropertyField::Vy => {
                write_float_kernel::launch_unchecked(
                    &bufs.client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(dim),
                    BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
                    BufferArg::from_raw_parts(bufs.gpu_photons.clone(), bufs.photons.len()),
                    BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(
                        bufs.gpu_edit_points.clone(),
                        SimBuffers::EDIT_POINT_CAPACITY,
                    ),
                    BufferArg::from_raw_parts(marks.clone(), 1),
                    BufferArg::from_raw_parts(params.clone(), P_WORDS),
                    count,
                    bufs.config.xres,
                    bufs.config.yres,
                    field_id,
                    value.as_f32(),
                    0u32,
                );
            }
            PropertyField::Type | PropertyField::Dcolour => {
                let Some(word) = value.as_word(field) else {
                    return;
                };
                write_word_kernel::launch_unchecked(
                    &bufs.client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(dim),
                    BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
                    BufferArg::from_raw_parts(bufs.gpu_photons.clone(), bufs.photons.len()),
                    BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(
                        bufs.gpu_elem_props.clone(),
                        bufs.elem_props_host.len() * 32,
                    ),
                    BufferArg::from_raw_parts(
                        bufs.gpu_edit_points.clone(),
                        SimBuffers::EDIT_POINT_CAPACITY,
                    ),
                    BufferArg::from_raw_parts(marks.clone(), 1),
                    BufferArg::from_raw_parts(params.clone(), P_WORDS),
                    count,
                    bufs.config.xres,
                    bufs.config.yres,
                    field_id,
                    word,
                    0u32,
                );
            }
        }
    }
    bufs.gpu_dirty = true;
    bufs.gpu_particles_dirty = true;
    if field == PropertyField::Type {
        bufs.gpu_maps_dirty = true;
    }
}

/// Rewrites `field` of every particle connected to the seed pixel, on the
/// device, and returns the number of particles touched.
///
/// The walk matches the seed's element type *and* its current property value,
/// the port's deliberate extension of `Simulation::flood_prop` (which matches
/// the type only) so a fill never rewrites a particle the user did not point
/// at. Floating-point values compare bit-exactly.
pub fn flood_property(
    bufs: &mut SimBuffers,
    x: i32,
    y: i32,
    field: PropertyField,
    value: PropertyValue,
) -> usize {
    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    if xres == 0 || yres == 0 {
        return 0;
    }
    if x < 0 || y < 0 || x >= xres as i32 || y >= yres as i32 {
        return 0;
    }
    if bufs.host_dirty {
        bufs.sync_host_to_gpu();
    }

    let pixels = (xres * yres) as usize;
    let slots = bufs.used_slot_bound();
    let field_id = field.id();
    let target = match field {
        PropertyField::Type | PropertyField::Dcolour => match value.as_word(field) {
            Some(word) => word,
            None => return 0,
        },
        PropertyField::Temp | PropertyField::Vx | PropertyField::Vy => {
            u32::reinterpret(value.as_f32())
        }
        _ => value.int_bits(field),
    };

    let marks = bufs.client.empty(pixels * 4);
    let scratch = bufs.client.empty(slots * 4);
    let params = bufs
        .client
        .create_from_slice(bytemuck::cast_slice(&[0u32; P_WORDS]));
    let (cubes, dim) = launch_grid(pixels);

    // Snapshot the compared field once, so the walk and the seed bind a single
    // `scratch` word instead of every per-particle array.
    let (slot_cubes, slot_dim) = launch_grid(slots);
    unsafe {
        clear_marks_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(marks.clone(), pixels),
            pixels as u32,
        );
        match field {
            PropertyField::Tmp
            | PropertyField::Life
            | PropertyField::Ctype
            | PropertyField::Tmp2
            | PropertyField::Tmp3
            | PropertyField::Tmp4 => {
                gather_int_kernel::launch_unchecked(
                    &bufs.client,
                    CubeCount::Static(slot_cubes, 1, 1),
                    CubeDim::new_1d(slot_dim),
                    BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(scratch.clone(), slots),
                    slots as u32,
                    field_id,
                );
            }
            PropertyField::Temp | PropertyField::Vx | PropertyField::Vy => {
                gather_float_kernel::launch_unchecked(
                    &bufs.client,
                    CubeCount::Static(slot_cubes, 1, 1),
                    CubeDim::new_1d(slot_dim),
                    BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(scratch.clone(), slots),
                    slots as u32,
                    field_id,
                );
            }
            PropertyField::Type | PropertyField::Dcolour => {
                gather_word_kernel::launch_unchecked(
                    &bufs.client,
                    CubeCount::Static(slot_cubes, 1, 1),
                    CubeDim::new_1d(slot_dim),
                    BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(scratch.clone(), slots),
                    slots as u32,
                    field_id,
                );
            }
        }
        property_seed_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), bufs.photons.len()),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(marks.clone(), pixels),
            BufferArg::from_raw_parts(params.clone(), P_WORDS),
            xres,
            yres,
            x,
            y,
            slots as u32,
        );
        // `flood_prop` has no distance cap: run the walk to its fixed point.
        let changed = bufs
            .client
            .create_from_slice(bytemuck::cast_slice(&[0u32]));
        spread_until_stable(bufs, &changed, || {
            property_spread_kernel::launch_unchecked(
                &bufs.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(dim),
                BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
                BufferArg::from_raw_parts(bufs.gpu_photons.clone(), bufs.photons.len()),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
                BufferArg::from_raw_parts(marks.clone(), pixels),
                BufferArg::from_raw_parts(params.clone(), P_WORDS),
                BufferArg::from_raw_parts(changed.clone(), 1),
                pixels as u32,
                xres,
                yres,
                slots as u32,
            );
        });

        let write_count = pixels as u32;
        match field {
            PropertyField::Tmp
            | PropertyField::Life
            | PropertyField::Ctype
            | PropertyField::Tmp2
            | PropertyField::Tmp3
            | PropertyField::Tmp4 => {
                write_int_kernel::launch_unchecked(
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
                    BufferArg::from_raw_parts(marks.clone(), pixels),
                    BufferArg::from_raw_parts(params.clone(), P_WORDS),
                    write_count,
                    xres,
                    yres,
                    field_id,
                    value.int_bits(field),
                    1u32,
                );
            }
            PropertyField::Temp | PropertyField::Vx | PropertyField::Vy => {
                write_float_kernel::launch_unchecked(
                    &bufs.client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(dim),
                    BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
                    BufferArg::from_raw_parts(bufs.gpu_photons.clone(), bufs.photons.len()),
                    BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(
                        bufs.gpu_edit_points.clone(),
                        SimBuffers::EDIT_POINT_CAPACITY,
                    ),
                    BufferArg::from_raw_parts(marks.clone(), pixels),
                    BufferArg::from_raw_parts(params.clone(), P_WORDS),
                    write_count,
                    xres,
                    yres,
                    field_id,
                    value.as_f32(),
                    1u32,
                );
            }
            PropertyField::Type | PropertyField::Dcolour => {
                write_word_kernel::launch_unchecked(
                    &bufs.client,
                    CubeCount::Static(cubes, 1, 1),
                    CubeDim::new_1d(dim),
                    BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
                    BufferArg::from_raw_parts(bufs.gpu_photons.clone(), bufs.photons.len()),
                    BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), bufs.capacity),
                    BufferArg::from_raw_parts(
                        bufs.gpu_elem_props.clone(),
                        bufs.elem_props_host.len() * 32,
                    ),
                    BufferArg::from_raw_parts(
                        bufs.gpu_edit_points.clone(),
                        SimBuffers::EDIT_POINT_CAPACITY,
                    ),
                    BufferArg::from_raw_parts(marks.clone(), pixels),
                    BufferArg::from_raw_parts(params.clone(), P_WORDS),
                    write_count,
                    xres,
                    yres,
                    field_id,
                    target,
                    1u32,
                );
            }
        }
    }

    let writes = match bufs.client.read_one(params.clone()) {
        Ok(bytes) => {
            let raw: &[u8] = &bytes;
            let (words, _) = raw.as_chunks::<4>();
            words
                .get(P_WRITES as usize)
                .map_or(0, |w| u32::from_le_bytes(*w))
        }
        Err(_) => 0,
    };

    bufs.gpu_dirty = true;
    bufs.gpu_particles_dirty = true;
    if field == PropertyField::Type {
        bufs.gpu_maps_dirty = true;
    }
    writes as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_ids_match_the_app_pref_indices() {
        // `PropertyKind::index` in `tpt-app`; the app asserts the same pairing
        // from its side (`apply.rs`).
        assert_eq!(PropertyField::Tmp.id(), 0);
        assert_eq!(PropertyField::Life.id(), 1);
        assert_eq!(PropertyField::Ctype.id(), 2);
        assert_eq!(PropertyField::Tmp2.id(), 3);
        assert_eq!(PropertyField::Tmp3.id(), 4);
        assert_eq!(PropertyField::Tmp4.id(), 5);
        assert_eq!(PropertyField::Type.id(), 6);
        assert_eq!(PropertyField::Temp.id(), 7);
        assert_eq!(PropertyField::Vx.id(), 8);
        assert_eq!(PropertyField::Vy.id(), 9);
        assert_eq!(PropertyField::Dcolour.id(), 10);
    }

    #[test]
    fn integer_values_keep_two_complement_bits() {
        // A negative life has to survive the trip as the same bit pattern the
        // host SoA stores.
        assert_eq!(
            PropertyValue::Int(-7).int_bits(PropertyField::Life),
            (-7i32) as u32
        );
        assert_eq!(
            PropertyValue::Float(4.6).int_bits(PropertyField::Life),
            5u32
        );
        // `ctype` names an element: the edit kernel's clamp applies here too.
        assert_eq!(
            PropertyValue::Int(9999).int_bits(PropertyField::Ctype),
            (PT_NUM - 1) as u32
        );
        assert_eq!(PropertyValue::Int(-3).int_bits(PropertyField::Ctype), 0u32);
    }

    #[test]
    fn word_values_validate_the_type_slot() {
        assert_eq!(
            PropertyValue::Int(PT_SAND as i64).as_word(PropertyField::Type),
            Some(PT_SAND)
        );
        assert_eq!(
            PropertyValue::Int(PT_NUM as i64).as_word(PropertyField::Type),
            None
        );
        assert_eq!(PropertyValue::Int(-1).as_word(PropertyField::Type), None);
        assert_eq!(
            PropertyValue::Int(-1).as_word(PropertyField::Dcolour),
            Some(0)
        );
        assert_eq!(
            PropertyValue::Int(0x12_3456).as_word(PropertyField::Dcolour),
            Some(0x12_3456)
        );
    }
}
