//! `GLOW` — the pressure-reactive liquid.
//!
//! Ports `GLOW.cpp:52-96`: the water and gel reactions, then the `ctype`/`tmp`
//! words the renderer's `Graphics` function reads (`GLOW.cpp:99-116`). The pass
//! is pixel parallel and writes its own `part_ty`/`part_ctype`/`part_tmp`. The
//! one cross-slot write — `GLOW + WATR -> DEUT` and `GLOW + GEL -> RSST`
//! (`GLOW.cpp:65-80`) — is a `compare_exchange` on the neighbour's type: only
//! one glow can claim a given neighbour, so exactly one glow dies per converted
//! pixel and a losing glow continues its scan instead of killing itself.
//!
//! TPT returns from the update as soon as the water or gel branch fires, so the
//! `ctype`/`tmp` refresh is skipped that frame; the `dead` guard below
//! reproduces that.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::PassDims;
use super::shared::{hash_u32, iabs, launch_grid, mix_index, pmap_at, pmap_slot, stasis_frozen};
use crate::buffers::SimBuffers;

/// TPT's `rng.chance(n, d)`.
#[cube]
fn chance(roll: u32, n: u32, d: u32) -> bool {
    let mut hit = false;
    if d > 0u32 {
        hit = (roll % d) < n;
    }
    hit
}

/// `GLOW.cpp:52-96`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(super) fn glow_kernel(
    pmap: &[u32],
    part_ty: &mut [Atomic<u32>],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &mut [i32],
    part_flags: &mut [u32],
    part_vx: &[f32],
    part_vy: &[f32],
    air_pv: &[f32],
    air_vx: &[f32],
    air_vy: &[f32],
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
            if stasis_frozen(part_flags, id) {
                terminate!();
            }
            if part_ty[id].load() == PT_GLOW {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let roll = hash_u32(frame_seed ^ mix_index(idx as u32, 1_103_515_245u32));
                let mut dead = false;

                for n in 0u32..9u32 {
                    if !dead {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid].load();
                                if nty == PT_WATR && chance(roll ^ n, 1u32, 400u32) {
                                    // `GLOW + WATR -> DEUT`, one in four hundred.
                                    // The claim decides which glow dies when two
                                    // touch the same water: only the CAS winner
                                    // writes the neighbour and kills itself.
                                    if part_ty[nid].compare_exchange_weak(PT_WATR, PT_DEUT)
                                        == PT_WATR
                                    {
                                        part_life[nid] = 10i32;
                                        dead = true;
                                    }
                                } else if nty == PT_GEL {
                                    // `GLOW + GEL -> RSST`.
                                    if part_ty[nid].compare_exchange_weak(PT_GEL, PT_RSST) == PT_GEL
                                    {
                                        part_tmp[nid] = 0i32;
                                        dead = true;
                                    }
                                }
                            }
                        }
                    }
                }

                if dead {
                    part_ty[id].store(PT_NONE);
                } else {
                    // `ctype` = `int(pv * 16)` clamped at zero, `tmp` = the
                    // air plus the particle's own velocity, both truncated.
                    let mut ctype = (air_pv[cidx] * 16.0f32) as i32;
                    if ctype < 0i32 {
                        ctype = 0i32;
                    }
                    part_ctype[id] = ctype as u32;
                    let air = ((air_vx[cidx] + air_vy[cidx]) * 16.0f32) as i32;
                    let own = ((part_vx[id] + part_vy[id]) * 64.0f32) as i32;
                    part_tmp[id] = iabs(air) + iabs(own);
                }
            }
        }
    }
}

/// Launch the `GLOW` update for one reaction step.
pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    let (cubes, dim) = launch_grid(dims.pixels);
    unsafe {
        glow_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }
}
