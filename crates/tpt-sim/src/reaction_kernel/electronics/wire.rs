//! `WIRE.cpp` — the WireWorld cellular automaton.
//!
//! The states are `0` (wire), `1` (spark head) and `2` (spark tail). The update
//! is double buffered: `Simulation.cpp:3862-3876` (`BeforeSim`) copies `ctype`
//! into `tmp` for every wire pixel before the element updates run, and
//! `WIRE.cpp:50-94` reads the previous generation from `tmp` and writes the
//! next one into `ctype`. The port keeps that split as two dispatches, because
//! a single kernel would race sibling threads reading a neighbour's `tmp`
//! while that neighbour overwrites it.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{pmap_at, pmap_slot};
use super::SPARK_LIFE;

/// `Simulation.cpp:3862-3876`: `BeforeSim`'s "make WIRE work" pass copies the
/// current `ctype` into `tmp` for every wire.
#[cube(launch_unchecked)]
pub(crate) fn wire_stage_kernel(
    part_ty: &[u32],
    part_tmp: &mut [i32],
    part_ctype: &[u32],
    used: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < used as usize && part_ty[idx] == PT_WIRE {
        // `parts[i].tmp = parts[i].ctype` (`Simulation.cpp:3873`).
        part_tmp[idx] = part_ctype[idx] as i32;
    }
}

/// `WIRE.cpp:50-94`: one WireWorld generation.
///
/// Slot-owned and pixel-parallel: the kernel only writes its own wire's
/// `ctype` plus, for the `NSCN` branch, the `NSCN` pixel's spark fields.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn wire_update_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &[i32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_WIRE {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                // `parts[i].tmp` is the previous generation; the update never
                // writes it.
                let prev = part_tmp[id];

                // `WIRE.cpp:61-69`: a head advances to a tail, a tail decays
                // back to plain wire. Everything else starts as state 0.
                let mut ctype = 0u32;
                if prev == 1i32 {
                    ctype = 2u32;
                }

                // `WIRE.cpp:86-87`: one or two wire neighbours holding a head
                // light this pixel.
                let mut count = 0i32;
                let mut stopped = false;
                for n in 0u32..9u32 {
                    let dx = (n % 3u32) as i32 - 1i32;
                    let dy = (n / 3u32) as i32 - 1i32;
                    if !stopped && (dx != 0i32 || dy != 0i32) {
                        let nx = px + dx;
                        let ny = py + dy;
                        let nv = pmap_at(pmap, xres, yres, nx, ny);
                        if nv != 0u32 {
                            let nslot = pmap_slot(nv);
                            let nty = part_ty[nslot];
                            if nty == PT_SPRK
                                && part_life[nslot] == 3i32
                                && part_ctype[nslot] == PT_PSCN
                            {
                                // `WIRE.cpp:79-83`: a `PSCN` spark at the
                                // third frame of its life drives the wire and
                                // returns immediately from the update.
                                ctype = 1u32;
                                stopped = true;
                            } else if nty == PT_NSCN && prev == 1i32 {
                                // `WIRE.cpp:84-85`: `create_part(-1, x+rx,
                                // y+ry, PT_SPRK)` on an `NSCN`. `create_part`
                                // (`Simulation.cpp:1851-1873`) only accepts an
                                // idle conductor (`life == 0`), converts it in
                                // place to a four-frame spark that remembers
                                // `NSCN`, and heats the good conductors by
                                // 10 K below 673 K.
                                if part_life[nslot] == 0i32 {
                                    part_ty[nslot] = PT_SPRK;
                                    part_ctype[nslot] = nty;
                                    part_life[nslot] = SPARK_LIFE;
                                    if part_temp[nslot] + 10.0f32 < 673.0f32 {
                                        part_temp[nslot] += 10.0f32;
                                    }
                                }
                            } else if nty == PT_WIRE && part_tmp[nslot] == 1i32 && prev == 0i32 {
                                // `WIRE.cpp:86-87`: count the neighbours that
                                // are heads, but only from an idle wire.
                                count += 1i32;
                            }
                        }
                    }
                }
                // `WIRE.cpp:91-92`: exactly one or two heads light the wire.
                if !stopped && (count == 1i32 || count == 2i32) {
                    ctype = 1u32;
                }
                part_ctype[id] = ctype;
            }
        }
    }
}
