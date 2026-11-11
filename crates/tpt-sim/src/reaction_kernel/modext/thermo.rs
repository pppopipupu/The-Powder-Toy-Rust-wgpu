//! Thermal second-set elements: `CLNT`, `COND` (`THMO`), `AMBE`, `QGPP`,
//! `FUEL`.
//!
//! Ports the `Update` halves of `CLNT.cpp:50-64`, `COND.cpp:46-62`,
//! `AMBE.cpp:49-87`, `QGPP.cpp:51-69` and `FUEL.cpp:47-65`.
//!
//! None of the five creates a particle: the reference changes `QGPP` into
//! `SING` and `FUEL` into `PLSM` with a direct `parts[i].type` assignment, so
//! this pass mutates the slot in place and queues nothing. Everything is
//! pixel parallel over the spatial map, which is TPT's `pmap`; a neighbour
//! write is the same cross-pixel write the reference performs, and two
//! adjacent writers resolve by dispatch order exactly as TPT's slot order
//! does.

// The device IR has no `RangeInclusive::contains`; the two-sided comparisons
// are deliberate.
#![allow(clippy::manual_range_contains)]

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::PassDims;
use super::super::shared::{launch_grid, pmap_at, pmap_slot, stasis_frozen};
use crate::buffers::SimBuffers;

/// `Update` pass for `CLNT`, `COND`, `AMBE`, `QGPP` and `FUEL`.
///
/// * `CLNT` (`CLNT.cpp:50-64`) clamps `tmp` to `-273..=9725` (out of range
///   resets it to the `Create` value `22`) and then steps `temp` four degrees
///   towards `tmp + 273.15`. TPT tests the two `if`s in sequence against the
///   *updated* temperature, so a temperature within `[tmp, tmp + 4)` is
///   decremented and immediately incremented again; the two comparisons are
///   reproduced in the same order.
/// * `COND` (`COND.cpp:46-62`) writes its own temperature into every
///   non-`COND` occupant of the 5x5 window.
/// * `AMBE` (`AMBE.cpp:49-87`) clamps `tmp` to `1..=25` (anything else becomes
///   the `25` at `AMBE.cpp:51-52`). At `life == 10` it stamps its temperature
///   into the ambient-heat grid over the `[-ar ..= ar]` cell square and polls
///   the 5x5 window for other `AMBE`s: one in `1..=9` restarts its own
///   countdown at `9`, one at `0` is woken to `10`. Any other life counts down
///   to zero.
/// * `QGPP` (`QGPP.cpp:51-69`) drains `tmp` into `tmp2`; a cooled-out pixel at
///   `temp >= 474.15` becomes `SING` with `tmp = 55`, and one still at
///   `tmp <= 170` at that temperature sets its own air cell to `-2`.
/// * `FUEL` (`FUEL.cpp:47-65`) ignites above `345.15 K`: for every occupied
///   pixel of the 5x5 window it becomes `2000.15 K` `PLSM` with `life = 130`
///   and stamps `10` into the pressure cell at the same offset.
///
/// The reference indexes `sim->hv` / `sim->pv` with `(y / CELL) + ry` —
/// *pixel* offsets added to a *cell* index (`AMBE.cpp:66`, `FUEL.cpp:61`).
/// Those writes leave the cell grid at the canvas border; a device buffer
/// cannot, so an out-of-grid stamp is dropped instead of corrupting the next
/// row.
///
/// A particle frozen in an unpowered stasis wall never reaches `Update` at all
/// (`Simulation.cpp:2339-2341`); the shared life pass keeps that bit in
/// `part_flags`, so this pass tests it like the other element kernels do.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn thermo_update_kernel(
    pmap: &[u32],
    part_flags: &mut [u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    air_pv: &mut [f32],
    air_hv: &mut [f32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            // `Simulation.cpp:2339-2341`: a particle frozen in an unpowered
            // stasis wall never runs its `Update`.
            if stasis_frozen(part_flags, id) {
                terminate!();
            }
            let px = (idx as u32 % xres) as i32;
            let py = (idx as u32 / xres) as i32;
            let ty = part_ty[id];

            if ty == PT_CLNT {
                // `CLNT.cpp:52-53`: an out-of-range target falls back to the
                // `CLNT.cpp:88` create value.
                let mut target = part_tmp[id];
                if target > 9725i32 || target < -273i32 {
                    target = 22i32;
                    part_tmp[id] = target;
                }
                // `CLNT.cpp:55-62`: both tests read the temperature the previous
                // one may have just changed.
                let mut temp = part_temp[id];
                if temp - 273.15f32 >= target as f32 {
                    temp -= 4.00f32;
                }
                if temp - 273.15f32 < target as f32 {
                    temp += 4.00f32;
                }
                part_temp[id] = temp;
            } else if ty == PT_COND {
                // `COND.cpp:48-60`: 5x5 thermostat write. `rx`/`ry` skip the
                // centre; the target test uses the *particle* type, not the
                // spatial map, like `parts[ID(r)].type != PT_COND`.
                let temp = part_temp[id];
                for n in 0u32..25u32 {
                    let rx = (n % 5u32) as i32 - 2i32;
                    let ry = (n / 5u32) as i32 - 2i32;
                    if rx != 0i32 || ry != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                        if nv != 0u32 {
                            let nslot = pmap_slot(nv);
                            if part_ty[nslot] != PT_COND {
                                part_temp[nslot] = temp;
                            }
                        }
                    }
                }
            } else if ty == PT_AMBE {
                // `AMBE.cpp:51-53`: the effective area, clamped to 1..=25.
                let mut ar = part_tmp[id];
                if ar < 1i32 || ar > 25i32 {
                    ar = 25i32;
                    part_tmp[id] = ar;
                }
                if part_life[id] != 10i32 {
                    // `AMBE.cpp:54-58`: idle pixels count down to zero.
                    if part_life[id] > 0i32 {
                        part_life[id] -= 1i32;
                    }
                } else {
                    // `AMBE.cpp:61-68`: stamp the ambient heat. TPT checks the
                    // square in *pixel* space but offsets the *cell* index by
                    // the same `rx`/`ry`, so the stamp is `[-ar ..= ar]` cells
                    // around the AMBE's own cell (and clipped to the grid).
                    let span = (2i32 * ar + 1i32) as u32;
                    let temp = part_temp[id];
                    for n in 0u32..(span * span) {
                        let rx = (n % span) as i32 - ar;
                        let ry = (n / span) as i32 - ar;
                        let nx = px + rx;
                        let ny = py + ry;
                        if (rx != 0i32 || ry != 0i32)
                            && nx >= 0i32
                            && ny >= 0i32
                            && (nx as u32) < xres
                            && (ny as u32) < yres
                        {
                            let cx = (px as u32 / CELL) as i32 + rx;
                            let cy = (py as u32 / CELL) as i32 + ry;
                            if cx >= 0i32
                                && cy >= 0i32
                                && (cx as u32) < xcells
                                && (cy as u32) < ycells
                            {
                                air_hv[((cy as u32) * xcells + (cx as u32)) as usize] = temp;
                            }
                        }
                    }

                    // `AMBE.cpp:70-84`: an active neighbour at `life == 0`
                    // starts its own countdown at `10`; a neighbour already
                    // counting (`1..=9`) pulls this pixel back to `9`.
                    for n in 0u32..25u32 {
                        let rx = (n % 5u32) as i32 - 2i32;
                        let ry = (n / 5u32) as i32 - 2i32;
                        if rx != 0i32 || ry != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                            if nv != 0u32 && (nv & 0x1FFu32) == PT_AMBE {
                                let nslot = pmap_slot(nv);
                                // Fresh read each step: the `else if` above can
                                // have just woken this neighbour to `10`, and
                                // TPT sees that write immediately.
                                let nlife = part_life[nslot];
                                if nlife < 10i32 && nlife > 0i32 {
                                    part_life[id] = 9i32;
                                } else if nlife == 0i32 {
                                    part_life[nslot] = 10i32;
                                }
                            }
                        }
                    }
                }
            } else if ty == PT_QGPP {
                // `QGPP.cpp:53-57`: the `tmp` counter runs down into `tmp2`.
                if part_tmp[id] > 0i32 {
                    part_tmp[id] -= 1i32;
                    part_tmp2[id] += 1i32;
                }
                // `QGPP.cpp:59-67`: a drained, still hot pixel collapses into
                // `SING` (direct type write, no create hook) and resets the
                // counter to `55`; otherwise the hot plasma keeps evacuating
                // its own air cell to `-2`.
                if part_tmp[id] == 0i32 && part_temp[id] >= 474.15f32 {
                    part_tmp[id] = 55i32;
                    part_ty[id] = PT_SING;
                } else if part_tmp[id] <= 170i32 && part_temp[id] >= 474.15f32 {
                    let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                    air_pv[cidx] = -2.0f32;
                }
            } else if ty == PT_FUEL {
                // `FUEL.cpp:49-63`: the ignition test sits *inside* the
                // neighbour loop, so each occupied neighbour re-applies the
                // (idempotent) ignition and stamps its own pressure cell.
                for n in 0u32..25u32 {
                    let rx = (n % 5u32) as i32 - 2i32;
                    let ry = (n / 5u32) as i32 - 2i32;
                    if rx != 0i32 || ry != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + rx, py + ry);
                        // `R_TEMP + 50.0f + 273.15f` = 345.15 K.
                        if nv != 0u32 && part_temp[id] > R_TEMP + 50.0f32 + 273.15f32 {
                            part_temp[id] = 2000.15f32;
                            part_life[id] = 130i32;
                            part_ty[id] = PT_PLSM;
                            let cx = (px as u32 / CELL) as i32 + rx;
                            let cy = (py as u32 / CELL) as i32 + ry;
                            if cx >= 0i32
                                && cy >= 0i32
                                && (cx as u32) < xcells
                                && (cy as u32) < ycells
                            {
                                air_pv[((cy as u32) * xcells + (cx as u32)) as usize] = 10.0f32;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Launch the thermal second-set pass.
pub(crate) fn launch(bufs: &SimBuffers, dims: &PassDims, _seed: u32) {
    if dims.pixels == 0 || dims.xcells == 0 {
        return;
    }
    let ycells = (dims.cells / dims.xcells as usize) as u32;
    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        thermo_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_hv.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
            ycells,
        );
    }
}
