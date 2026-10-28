//! `PUMP.cpp` — the pressure pump.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{pmap_at, pmap_slot};

/// `PUMP.cpp:53-111`: the pressure pump.
///
/// A pump at `life == 10` is on. An unfiltered pump pulls every air cell in its
/// 3x3 window that is not on a diagonal (`!(rx && ry)`, so the centre and the
/// four orthogonal neighbours) a tenth of the way towards `temp - 273.15`. A
/// pump whose `tmp == 1` is in filter mode: it no longer pushes, and instead
/// writes the pressure `FILT` carries in `ctype` into the cell under that
/// filter. The radius-2 chain then copies the on/off state between adjacent
/// pumps.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pump_update_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    part_temp: &mut [f32],
    air_pv: &mut [f32],
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
            if part_ty[id] == PT_PUMP {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let life = part_life[id];

                if life != 10i32 {
                    // `PUMP.cpp:55-59`: a pump that is not fully on counts
                    // down to zero.
                    if life > 0i32 {
                        part_life[id] = life - 1i32;
                    }
                } else {
                    // `PUMP.cpp:62-65`: the target pressure is `temp-273.15`,
                    // so the temperature is clamped into the pressure scale.
                    let temp =
                        part_temp[id].clamp(273.15f32 + MIN_PRESSURE, 273.15f32 + MAX_PRESSURE);
                    part_temp[id] = temp;
                    let target = temp - 273.15f32;
                    let filter_mode = part_tmp[id] == 1i32;

                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if filter_mode {
                            // `PUMP.cpp:76-86`: filter mode writes the
                            // pressure serialised into the `FILT`'s `ctype`
                            // (`0x10000000 + pressure`) into that filter's air
                            // cell.
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nslot = pmap_slot(nv);
                                if part_ty[nslot] == PT_FILT {
                                    let raw = part_ctype[nslot];
                                    // `parts[ID(r)].ctype - 0x10000000` must
                                    // land in `0..=MAX_PRESSURE-MIN_PRESSURE`;
                                    // the guarded subtraction is that range
                                    // test without the reference's unsigned
                                    // wrap-around.
                                    if raw >= 0x1000_0000u32
                                        && raw - 0x1000_0000u32
                                            <= (MAX_PRESSURE - MIN_PRESSURE) as u32
                                    {
                                        let new_pressure = (raw - 0x1000_0000u32) as i32;
                                        let cidx = air_cell(px + dx, py + dy, xcells, ycells);
                                        air_pv[cidx] = new_pressure as f32 + MIN_PRESSURE;
                                    }
                                }
                            }
                        } else if dx == 0i32 || dy == 0i32 {
                            // `PUMP.cpp:71-75`: `!(rx && ry)` writes the
                            // centre and the four orthogonal cells.
                            let cidx = air_cell(px + dx, py + dy, xcells, ycells);
                            let pv = air_pv[cidx];
                            air_pv[cidx] = pv + 0.1f32 * (target - pv);
                        }
                    }

                    // `PUMP.cpp:90-105`: chain the on/off state across a
                    // radius-2 window. A neighbouring pump that is winding
                    // down (0 < life < 10) turns this pump off (life 9); a
                    // stopped neighbour (life 0) is switched on (life 10).
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nslot = pmap_slot(nv);
                                if part_ty[nslot] == PT_PUMP {
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
                }
            }
        }
    }
}

/// `PUMP.cpp`'s `sim->pv[(y+ry)/CELL][(x+rx)/CELL]`, clamped to the cell grid.
///
/// The reference indexes `pv` with unclamped pixel coordinates, which reads
/// past the pressure array at the canvas border; a device buffer cannot do
/// that, so the neighbour cell saturates to the nearest edge cell.
#[cube]
fn air_cell(x: i32, y: i32, xcells: u32, ycells: u32) -> usize {
    let mut cx = x / CELL as i32;
    let mut cy = y / CELL as i32;
    if cx < 0i32 {
        cx = 0i32;
    }
    if cy < 0i32 {
        cy = 0i32;
    }
    if cx >= xcells as i32 {
        cx = xcells as i32 - 1i32;
    }
    if cy >= ycells as i32 {
        cy = ycells as i32 - 1i32;
    }
    ((cy as u32) * xcells + (cx as u32)) as usize
}
