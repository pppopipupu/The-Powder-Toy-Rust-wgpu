//! Device-side simulation tools (`src/simulation/simtools/*.cpp`).
//!
//! The reference tools edit the live simulation: `HEAT`/`COOL` look up the
//! occupant of every brush pixel and shift its temperature (`HEAT.cpp:16-28`),
//! `AIR`/`VAC` add to the pressure field (`AIR.cpp:17-23`), `WIND`/`CYCL` push
//! the velocity field (`WIND.cpp:20-29`, `CYCL.cpp:24-51`) and `MIX` swaps two
//! particles through `pmap` (`MIX.cpp:21-57`).
//!
//! The host shadow is only a snapshot of the last read-back, and the port never
//! refreshes it per frame. A host-side read-modify-write therefore computed the
//! new value from a stale old one and then re-uploaded the whole affected grid,
//! rewinding everything the device had done since that snapshot: holding `HEAT`
//! pinned the temperature at `snapshot + 2 K` and holding `AIR` overwrote the
//! evolved air field every tick. The edits run on the device instead, exactly
//! like the canvas edit kernels ([`crate::edit_kernel`]); the host uploads only
//! the packed brush footprint.

// CubeCL's expansions trip these lints (see `edit_kernel.rs`): the IR has no
// `clamp` method, and collapsing the nested guards changes the expansion.
#![allow(
    clippy::collapsible_if,
    clippy::manual_clamp,
    clippy::manual_is_multiple_of
)]

use cubecl::prelude::*;

use tpt_core::constants::*;

use crate::buffers::SimBuffers;
use crate::edit_kernel::slot_at;
use crate::reaction_kernel::shared::{elem_word, launch_grid, pmap_slot};

/// `CELL` mirror: a pixel maps to its air cell by `/ CELL` on both axes.
const C_CELL: u32 = 4;
const _: () = assert!(CELL == 4);

/// Pump ids that take the `strength * .1` branch (`HEAT.cpp:18-21`).
const C_PT_PUMP: u32 = 81;
const _: () = assert!(PT_PUMP == 81);
const C_PT_GPMP: u32 = 43;
const _: () = assert!(PT_GPMP == 43);

/// `PT_NONE` marks a free slot (`pmap` may name one for a frame).
const C_PT_NONE: u32 = 0;
const _: () = assert!(PT_NONE == 0);

/// `properties` word of [`tpt_core::elements::GpuElementProps`].
const C_PROPERTIES_WORD: u32 = 1;

/// `HEAT.cpp:16-28` / `COOL.cpp:16-28`: one thread per brush pixel adjusts the
/// temperature of the particle there (matter first, then energy), by
/// `strength * 2.0` (`* 0.1` for pumps), clamped to `MIN_TEMP..=MAX_TEMP`.
#[cube(launch_unchecked)]
fn heat_cool_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_temp: &mut [f32],
    points: &[u32],
    count: u32,
    delta: f32,
    pump_delta: f32,
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
            let ty = part_ty[id];
            // A map word may name a slot the device just freed; the reference's
            // `CallPerform` only ever sees a live particle.
            if ty != C_PT_NONE {
                let mut step = delta;
                if ty == C_PT_PUMP || ty == C_PT_GPMP {
                    step = pump_delta;
                }
                let mut temp = part_temp[id] + step;
                if temp > MAX_TEMP {
                    temp = MAX_TEMP;
                }
                if temp < MIN_TEMP {
                    temp = MIN_TEMP;
                }
                part_temp[id] = temp;
            }
        }
    }
}

/// `AIR.cpp`/`VAC.cpp`/`WIND.cpp`/`AMBP.cpp`/`AMBM.cpp`: one thread per air
/// cell adds `delta * count` to the pressure, heat or velocity field.
///
/// The brush pixel list is deduplicated on the host into `cell | count << 24`
/// so a cell is written by exactly one thread and the per-pixel additions of the
/// reference (`for (ui::Point off : brush)`) become one scaled add.
///
/// `clamp_velocity` is `0` for `WIND`, whose reference add is unbounded, and
/// `1` for `CYCL`, which clamps to `MIN_PRESSURE..=MAX_PRESSURE`
/// (`CYCL.cpp:37-45`).
#[cube(launch_unchecked)]
fn air_tool_kernel(
    cells: &[u32],
    count: u32,
    dpv: f32,
    dhv: f32,
    dvx: f32,
    dvy: f32,
    clamp_velocity: u32,
    air_pv: &mut [f32],
    air_hv: &mut [f32],
    air_vx: &mut [f32],
    air_vy: &mut [f32],
) {
    let i = ABSOLUTE_POS as u32;
    if i < count {
        let packed = cells[i as usize];
        let cell = (packed & 0x00FF_FFFFu32) as usize;
        let pixels = (packed >> 24u32) as f32;
        if dpv != 0.0f32 {
            let mut value = air_pv[cell] + dpv * pixels;
            if value > MAX_PRESSURE {
                value = MAX_PRESSURE;
            }
            if value < MIN_PRESSURE {
                value = MIN_PRESSURE;
            }
            air_pv[cell] = value;
        }
        if dhv != 0.0f32 {
            let mut value = air_hv[cell] + dhv * pixels;
            if value > MAX_TEMP {
                value = MAX_TEMP;
            }
            if value < MIN_TEMP {
                value = MIN_TEMP;
            }
            air_hv[cell] = value;
        }
        if dvx != 0.0f32 || dvy != 0.0f32 {
            let mut vx = air_vx[cell] + dvx * pixels;
            let mut vy = air_vy[cell] + dvy * pixels;
            if clamp_velocity != 0u32 {
                if vx > MAX_PRESSURE {
                    vx = MAX_PRESSURE;
                }
                if vx < MIN_PRESSURE {
                    vx = MIN_PRESSURE;
                }
                if vy > MAX_PRESSURE {
                    vy = MAX_PRESSURE;
                }
                if vy < MIN_PRESSURE {
                    vy = MIN_PRESSURE;
                }
            }
            air_vx[cell] = vx;
            air_vy[cell] = vy;
        }
    }
}

/// `CYCL.cpp:24-51`: one thread per cell rotates the cell's velocity about the
/// brush centre by `strength / 16`, clamping to `MIN_PRESSURE..=MAX_PRESSURE`.
///
/// The reference triggers only on the cell's top-left pixel (`x % CELL == 0 &&
/// y % CELL == 0`) and never on the centre cell itself, so the footprint is the
/// deduplicated corner list.
#[cube(launch_unchecked)]
fn cyclone_kernel(
    cells: &[u32],
    count: u32,
    centre_x: i32,
    centre_y: i32,
    scale: f32,
    xcells: u32,
    air_vx: &mut [f32],
    air_vy: &mut [f32],
) {
    let i = ABSOLUTE_POS as u32;
    if i < count {
        let cell = cells[i as usize] as i32;
        let x = (cell % xcells as i32) * C_CELL as i32;
        let y = (cell / xcells as i32) * C_CELL as i32;
        if x != centre_x || y != centre_y {
            let dvx = (centre_x - x) as f32;
            let dvy = (centre_y - y) as f32;
            let invsqr = 1.0f32 / (dvx * dvx + dvy * dvy).sqrt();
            let mut vx = air_vx[cell as usize] - scale * (-dvy) * invsqr;
            let mut vy = air_vy[cell as usize] - scale * dvx * invsqr;
            if vx > MAX_PRESSURE {
                vx = MAX_PRESSURE;
            }
            if vx < MIN_PRESSURE {
                vx = MIN_PRESSURE;
            }
            if vy > MAX_PRESSURE {
                vy = MAX_PRESSURE;
            }
            if vy < MIN_PRESSURE {
                vy = MIN_PRESSURE;
            }
            air_vx[cell as usize] = vx;
            air_vy[cell as usize] = vy;
        }
    }
}

/// `MIX.cpp:21-57`: swaps `pmap` entries and both particle positions.
///
/// The reference walks the brush in order and draws three `rng()` values per
/// occupied pixel (the 1-in-100 gate, then the two offsets), so the kernel runs
/// as a single thread. The rolls are generated on the host in the same order and
/// uploaded with the footprint; an empty pixel consumes none of them, exactly
/// like the reference's early `return 0`.
#[cube(launch_unchecked)]
fn mix_kernel(
    pmap: &mut [u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    elem_props: &[u32],
    points: &[u32],
    rolls: &[u32],
    count: u32,
    distance: i32,
    xres: u32,
    yres: u32,
) {
    if ABSOLUTE_POS == 0usize {
        for i in 0..count as usize {
            let packed = points[i];
            let x = (packed & 0xFFFFu32) as i32;
            let y = (packed >> 16u32) as i32;
            let pixel = ((y as u32) * xres + (x as u32)) as usize;
            let this_word = pmap[pixel];
            if (this_word >> 9u32) != 0u32 {
                if rolls[i * 3usize] % 100u32 == 0u32 {
                    if distance > 0i32 {
                        let this_ty = this_word & 0x1FFu32;
                        let props = elem_word(elem_props, this_ty, C_PROPERTIES_WORD);
                        if props & (TYPE_PART | TYPE_LIQUID | TYPE_GAS) != 0u32 {
                            let new_x = x + (rolls[i * 3usize + 1usize] % distance as u32) as i32
                                - distance / 2i32;
                            let new_y = y + (rolls[i * 3usize + 2usize] % distance as u32) as i32
                                - distance / 2i32;
                            if new_x >= 0i32
                                && new_y >= 0i32
                                && (new_x as u32) < xres
                                && (new_y as u32) < yres
                            {
                                let other_pixel = ((new_y as u32) * xres + (new_x as u32)) as usize;
                                let that_word = pmap[other_pixel];
                                if (that_word >> 9u32) != 0u32 {
                                    let that_ty = that_word & 0x1FFu32;
                                    let that_props =
                                        elem_word(elem_props, that_ty, C_PROPERTIES_WORD);
                                    // `MIX.cpp:46-47`: identical state, not element.
                                    if props & STATE_FLAGS == that_props & STATE_FLAGS {
                                        let this_id = pmap_slot(this_word);
                                        let that_id = pmap_slot(that_word);
                                        pmap[pixel] = that_word;
                                        part_x[that_id] = x as f32;
                                        part_y[that_id] = y as f32;
                                        pmap[other_pixel] = this_word;
                                        part_x[this_id] = new_x as f32;
                                        part_y[this_id] = new_y as f32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Uploads packed words into the shared edit scratch buffer.
///
/// Returns the number of entries the kernel must process; a footprint larger
/// than [`SimBuffers::EDIT_POINT_CAPACITY`] is truncated (a brush stamp never
/// comes close in practice).
fn upload_scratch(bufs: &SimBuffers, words: &[u32]) -> u32 {
    let count = words.len().min(SimBuffers::EDIT_POINT_CAPACITY);
    if count == 0 {
        return 0;
    }
    crate::buffers::gpu_sync::upload(&bufs.client, &bufs.gpu_edit_points, &words[..count]);
    count as u32
}

/// Packs a pixel list into `[x | y << 16]` words.
fn pack_points(points: &[(i32, i32)]) -> Vec<u32> {
    points
        .iter()
        .map(|&(x, y)| ((x as u32) & 0xFFFF) | (((y as u32) & 0xFFFF) << 16))
        .collect()
}

/// Applies `HEAT`/`COOL` over `points`: `delta` for a normal element,
/// `pump_delta` for `PT_PUMP`/`PT_GPMP`.
pub(crate) fn heat_cool(bufs: &mut SimBuffers, points: &[(i32, i32)], delta: f32, pump_delta: f32) {
    if points.is_empty() {
        return;
    }
    let words = pack_points(points);
    let count = upload_scratch(bufs, &words);
    if count == 0 {
        return;
    }
    let (cubes, dim) = launch_grid(count as usize);
    unsafe {
        heat_cool_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), bufs.photons.len()),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), bufs.capacity),
            BufferArg::from_raw_parts(
                bufs.gpu_edit_points.clone(),
                SimBuffers::EDIT_POINT_CAPACITY,
            ),
            count,
            delta,
            pump_delta,
            bufs.config.xres,
            bufs.config.yres,
        );
    }
    bufs.gpu_particles_dirty = true;
    bufs.gpu_dirty = true;
}

/// Adds `dpv`/`dhv`/`dvx`/`dvy` to the air cells of the footprint, once per
/// brush pixel.
///
/// A zero delta leaves that field untouched (the reference tool only writes its
/// own field).
pub(crate) fn air_tool(
    bufs: &mut SimBuffers,
    cells: &[(u32, u32)],
    dpv: f32,
    dhv: f32,
    dvx: f32,
    dvy: f32,
    clamp_velocity: bool,
) {
    if cells.is_empty() {
        return;
    }
    let words: Vec<u32> = cells
        .iter()
        .map(|&(cell, pixels)| (cell & 0x00FF_FFFF) | (pixels.min(0xFF) << 24))
        .collect();
    let count = upload_scratch(bufs, &words);
    if count == 0 {
        return;
    }
    let (cubes, dim) = launch_grid(count as usize);
    unsafe {
        air_tool_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(
                bufs.gpu_edit_points.clone(),
                SimBuffers::EDIT_POINT_CAPACITY,
            ),
            count,
            dpv,
            dhv,
            dvx,
            dvy,
            if clamp_velocity { 1u32 } else { 0u32 },
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), bufs.air_pv.len()),
            BufferArg::from_raw_parts(bufs.gpu_air_hv.clone(), bufs.air_hv.len()),
            BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), bufs.air_vx.len()),
            BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), bufs.air_vy.len()),
        );
    }
    bufs.gpu_air_dirty = true;
    bufs.gpu_dirty = true;
}

/// Rotates the velocity of every cell in `cells` about `centre` (`CYCL.cpp`).
pub(crate) fn cyclone(bufs: &mut SimBuffers, cells: &[u32], centre: (i32, i32), scale: f32) {
    if cells.is_empty() {
        return;
    }
    let count = upload_scratch(bufs, cells);
    if count == 0 {
        return;
    }
    let (cubes, dim) = launch_grid(count as usize);
    unsafe {
        cyclone_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(
                bufs.gpu_edit_points.clone(),
                SimBuffers::EDIT_POINT_CAPACITY,
            ),
            count,
            centre.0,
            centre.1,
            scale,
            bufs.config.xcells(),
            BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), bufs.air_vx.len()),
            BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), bufs.air_vy.len()),
        );
    }
    bufs.gpu_air_dirty = true;
    bufs.gpu_dirty = true;
}

/// Applies `MIX` over `points` with the host-generated RNG rolls (three per
/// pixel, in brush order).
pub(crate) fn mix(bufs: &mut SimBuffers, points: &[(i32, i32)], rolls: &[u32], distance: i32) {
    if points.is_empty() || distance <= 0 || rolls.len() < points.len() * 3 {
        return;
    }
    let words = pack_points(points);
    let count = upload_scratch(bufs, &words);
    if count == 0 {
        return;
    }
    let roll_words = count as usize * 3;
    crate::buffers::gpu_sync::upload(&bufs.client, &bufs.gpu_edit_rolls, &rolls[..roll_words]);
    // One thread: the reference walks the brush sequentially and every swap
    // changes the map the next pixel reads.
    unsafe {
        mix_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), bufs.pmap.len()),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), bufs.capacity),
            BufferArg::from_raw_parts(
                bufs.gpu_elem_props.clone(),
                bufs.elem_props_host.len().min(PT_NUM) * 32,
            ),
            BufferArg::from_raw_parts(
                bufs.gpu_edit_points.clone(),
                SimBuffers::EDIT_POINT_CAPACITY,
            ),
            BufferArg::from_raw_parts(bufs.gpu_edit_rolls.clone(), roll_words),
            count,
            distance,
            bufs.config.xres,
            bufs.config.yres,
        );
    }
    bufs.gpu_particles_dirty = true;
    bufs.gpu_maps_dirty = true;
    bufs.gpu_dirty = true;
}
