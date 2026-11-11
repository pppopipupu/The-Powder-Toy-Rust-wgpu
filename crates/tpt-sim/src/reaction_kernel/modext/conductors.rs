//! Conducting second-set materials: `WALL`, `CWIR`, `COPR`, `FNTC`, `FPTC`,
//! `ELEX`, `DMRN`.
//!
//! Ports
//!
//! * `WALL.cpp:48-53` — every tick stamps `1` / `0x8` into the coarse
//!   `bmap_blockair` / `bmap_blockairh` cell it occupies. The two grids are
//!   device-only scratch ([`SimBuffers::gpu_elem_block_air`] and
//!   [`SimBuffers::gpu_elem_block_airh`]): [`super::launch`] zeroes them at the
//!   start of the pass and `air_kernel::settle::air_block_masks_kernel` folds
//!   the stamps into the wall masks before they are cleared again, so a stamp
//!   lives exactly one frame.
//! * `CWIR.cpp:52-91` — the `tmp` / `tmp2` clamps, the custom melting point
//!   (`LAVA` carrying `ctype = CWIR`) and the custom-distance spark pickup.
//! * `COPR.cpp:50-115` — the superconductivity spark pickup at `+-8` pixels,
//!   the `tmp2` velocity freeze, the pressure flag, `O2` oxidation and the
//!   sparked-copper conversion.
//! * `FNTC.cpp:48-53` / `FPTC.cpp:48-53` — the `temp -= 10` self-cooling.
//! * `ELEX.cpp:46-59` — the random transmutation above `274.15` K.
//! * `DMRN.cpp` — no `Update`, `Create` or `Graphics` hook: demron only exists
//!   as an (indestructible, heat-insulating) registry row, so this module has
//!   nothing to run for it.
//!
//! The element passes are pixel parallel and slot-owned: a thread writes only
//! the slot of its own `pmap` pixel, and the `WALL` stamp is the one slot
//! parallel pass (it needs the particle's own position, not the pixel's top
//! particle). The one cross-slot write follows the reference exactly:
//! `COPR.cpp:104-111` turns an adjacent sparked `SPRK` whose `ctype` is `COPR`
//! back into `COPR`, and kills the adjacent `O2` it oxidised with.

// Same rationale as `pixel_mode.rs`: collapsing a nested `if` changes the
// CubeCL expansion, and the redundant `as u32` is the form the device IR takes.
#![allow(clippy::collapsible_if, clippy::unnecessary_cast)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::PassDims;
use super::super::combustion::{chance, rand_range};
use super::super::shared::{CREATE_QUEUE_WORDS, launch_grid, pmap_at, pmap_slot, request_replace};
use crate::buffers::SimBuffers;

/// Highest element id `ELEX.cpp:55` can transmute into.
///
/// The reference rolls `rng.between(1, (1 << PMAPBITS) - 1)` over the 511
/// `pmap` id values and then rejects everything `>= 253`, so roughly half its
/// rolls never convert. The port rolls the populated id range directly; ids
/// `239..=252` do not exist, `PT_MGNT` is the last registered element, so the
/// `elemid < 253` test is subsumed by the roll range.
const ELEX_MAX_ID: i32 = 238;
const _: () = assert!(PT_MGNT == ELEX_MAX_ID as u32);

/// Coarse air cell of a pixel: `(y / CELL) * xcells + (x / CELL)`.
#[cube]
fn cell_of(py: i32, px: i32, xcells: u32) -> usize {
    (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize
}

/// `WALL.cpp:48-53`: the elemental wall blocks pressure (`1`) and ambient heat
/// (`0x8`) in the coarse cell it sits in, every tick.
///
/// Slot parallel: a WALL particle writes the cell of its own position, so a
/// wall hidden under another particle still stamps its cell, exactly like the
/// reference's per-particle update loop.
#[cube(launch_unchecked)]
pub(crate) fn wall_block_stamp_kernel(
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    elem_block_air: &mut [u32],
    elem_block_airh: &mut [u32],
    used_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let i = ABSOLUTE_POS;
    if i < used_slots as usize {
        if part_ty[i] == PT_WALL {
            // `UPDATE_FUNC_ARGS` passes `(int)(parts[i].x + 0.5f)`.
            let px = i32::cast_from(part_x[i] + 0.5f32);
            let py = i32::cast_from(part_y[i] + 0.5f32);
            if px >= 0i32 && py >= 0i32 && (px as u32) < xres && (py as u32) < yres {
                let cell = cell_of(py, px, xcells);
                elem_block_air[cell] = 1u32;
                elem_block_airh[cell] = 0x8u32;
            }
        }
    }
}

/// `CWIR.cpp:52-91`: the custom wire.
///
/// `tmp` is the conduction reach (clamped to `0..=8`), `tmp2` the melting point
/// in Celsius (clamped to `1..=9725`) and `life` the spark the wire carries.
/// The pickup runs even for a wire that melted in the same tick, because the
/// reference falls through to it as well.
#[cube(launch_unchecked)]
pub(crate) fn cwir_update_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_temp: &[f32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_CWIR {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;

                // `CWIR.cpp:55-63`.
                if part_tmp[id] < 0i32 || part_tmp[id] > 8i32 {
                    part_tmp[id] = 8i32;
                }
                if part_tmp2[id] < 1i32 || part_tmp2[id] > 9725i32 {
                    part_tmp2[id] = 2000i32;
                }
                // `CWIR.cpp:65-69`: custom melting point.
                if part_temp[id] - 273.15f32 >= part_tmp2[id] as f32 {
                    part_ctype[id] = PT_CWIR;
                    part_ty[id] = PT_LAVA;
                }

                // `CWIR.cpp:70-89`: `tmp` cells of reach along the four axes.
                let cust = part_tmp[id];
                if part_life[id] == 0i32 {
                    for j in 0u32..4u32 {
                        let mut rx = 0i32;
                        let mut ry = 0i32;
                        if j == 0u32 {
                            rx = -cust;
                        } else if j == 1u32 {
                            rx = cust;
                        } else if j == 2u32 {
                            ry = -cust;
                        } else {
                            ry = cust;
                        }
                        let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            if part_ty[nid] == PT_SPRK
                                && part_life[nid] != 0i32
                                && part_life[nid] < 4i32
                            {
                                part_ty[id] = PT_SPRK;
                                part_life[id] = 4i32;
                                part_ctype[id] = PT_CWIR;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `COPR.cpp:50-115`: copper.
///
/// Below 173.15 K an unsparked crystal picks up a young spark up to eight
/// pixels away on each axis and becomes the spark itself; `tmp2` is set once
/// the copper has felt more than 4.3 pressure while hot (`tmp > 30`) and keeps
/// it moving (at `tmp2 == 0` the reference freezes its velocity); `O2` is
/// absorbed on a 1-in-80 roll (dropping the cell pressure to `-1` and raising
/// `tmp`); a `COPR` spark next to hot or charged copper turns back into solid
/// copper, and a spark whose `tmp > 30` sets the flag too.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn copr_update_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_temp: &[f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_pv: &mut [f32],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_COPR {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cell = cell_of(py, px, xcells);

                // `COPR.cpp:52-56`: the charged oxide keeps its velocity.
                if part_tmp2[id] == 0i32 {
                    part_vy[id] = 0.0f32;
                    part_vx[id] = 0.0f32;
                }

                // `COPR.cpp:57-76`: superconductivity picks up a young spark
                // eight pixels away on any axis.
                if part_temp[id] <= 173.15f32 {
                    if part_life[id] == 0i32 {
                        for j in 0u32..4u32 {
                            let mut rx = 0i32;
                            let mut ry = 0i32;
                            if j == 0u32 {
                                rx = -8i32;
                            } else if j == 1u32 {
                                rx = 8i32;
                            } else if j == 2u32 {
                                ry = -8i32;
                            } else {
                                ry = 8i32;
                            }
                            let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                if part_ty[nid] == PT_SPRK
                                    && part_life[nid] != 0i32
                                    && part_life[nid] < 4i32
                                {
                                    part_ty[id] = PT_SPRK;
                                    part_life[id] = 4i32;
                                    part_ctype[id] = PT_COPR;
                                }
                            }
                        }
                    }
                }

                // `COPR.cpp:78-113`: the 5x5 window.
                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            // `COPR.cpp:85-88`: a copper shocked by more than
                            // 4.3 pressure while hot keeps the flag.
                            if air_pv[cell] > 4.3f32 && part_tmp[id] > 30i32 {
                                part_tmp2[id] = 1i32;
                            }
                            let nty = part_ty[nid];
                            if nty == PT_O2 {
                                // `COPR.cpp:91-103`: oxygen uptake.
                                if part_tmp[id] < 100i32 {
                                    let roll = frame_seed
                                        ^ ((id as u32) << 8u32)
                                        ^ ((n as u32) << 1u32)
                                        ^ 0x00C0_0001u32;
                                    if chance(roll, 1u32, 80u32) {
                                        air_pv[cell] = -1.0f32;
                                        part_tmp[id] += 1i32;
                                        part_ty[nid] = PT_NONE;
                                    }
                                }
                            } else if nty == PT_SPRK {
                                // `COPR.cpp:104-111`: a sparked copper wire
                                // settles back once the copper is hot or
                                // charged.
                                if part_ctype[nid] == PT_COPR
                                    && (part_tmp[id] > 20i32 || part_temp[id] >= 573.15f32)
                                {
                                    part_ty[nid] = PT_COPR;
                                }
                                if part_tmp[nid] > 30i32 {
                                    part_tmp2[id] = 1i32;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `FNTC.cpp:48-53` / `FPTC.cpp:48-53`: both fast thermistors bleed 10 K per
/// tick while warmer than 295 K.
///
/// The sparked forms cool by 2.5 K instead (`Element_NTCT_update` through
/// `SPRK.cpp:94-99`), which the spark conduction pass owns.
#[cube(launch_unchecked)]
pub(crate) fn thermistor_cool_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_temp: &mut [f32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty == PT_FNTC || ty == PT_FPTC {
                if part_temp[id] > 295.0f32 {
                    part_temp[id] -= 10.0f32;
                }
            }
        }
    }
}

/// `ELEX.cpp:46-59`: `tmp` counts up to 120; once it is at least 115 and the
/// gas is above 274.15 K it becomes a uniformly random element, except the two
/// the reference rejects by literal id.
///
/// `ELEX.cpp:56` is `elemid != 230 && elemid != 217 && elemid < 253`; its
/// comment ("BFLM, GoL, WHEL, MIST") is stale from the vanilla table. The id is
/// the 0-based index in `elements/meson.build` (`NONE` is 0), so in this mod
/// `230` is `AMBE` and `217` is `RADN`; those two are excluded.
///
/// `create_part(i, x, y, elemid)` re-creates the particle in place
/// (`Simulation.cpp:1918-1945`): the slot starts over from the target's
/// `DefaultProperties` and its `Create` hook runs, so the request goes through
/// [`request_replace`] rather than a field-by-field rewrite.
#[cube(launch_unchecked)]
pub(crate) fn elex_update_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_tmp: &mut [i32],
    part_temp: &[f32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_ELEX {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;

                if part_tmp[id] < 120i32 {
                    part_tmp[id] += 1i32;
                }
                if part_tmp[id] >= 115i32 && part_temp[id] >= 274.15f32 {
                    let roll = frame_seed ^ ((id as u32) << 8u32) ^ PT_ELEX;
                    let elem = rand_range(roll, 1i32, ELEX_MAX_ID) as u32;
                    if elem != PT_AMBE && elem != PT_RADN {
                        request_replace(
                            create_queue,
                            id as u32,
                            elem,
                            px,
                            py,
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
    }
}

/// Launch the conductor group's passes for one tick.
pub(crate) fn launch(bufs: &SimBuffers, dims: &PassDims, seed: u32) {
    if dims.pixels == 0 || dims.cells == 0 || dims.xres == 0 || dims.yres == 0 {
        return;
    }

    if dims.used_slots > 0 {
        let (cubes, dim) = launch_grid(dims.used_slots);
        unsafe {
            wall_block_stamp_kernel::launch_unchecked(
                &bufs.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(dim),
                BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
                BufferArg::from_raw_parts(bufs.gpu_elem_block_air.clone(), dims.cells),
                BufferArg::from_raw_parts(bufs.gpu_elem_block_airh.clone(), dims.cells),
                dims.used_slots as u32,
                dims.xres,
                dims.yres,
                dims.xcells,
            );
        }
    }

    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        cwir_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
        copr_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
            seed,
        );
        thermistor_cool_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            dims.xres,
            dims.yres,
        );
        elex_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            seed,
        );
    }
}
