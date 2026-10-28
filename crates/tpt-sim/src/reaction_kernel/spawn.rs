//! Device-side expansion of the `part_flags` spawn-request bits.
//!
//! A reaction pass that cannot bind the creation queue sets one of the
//! `FLAG_SPAWN_*` bits in its own `part_flags` word. The host used to
//! materialise those requests (`shared::apply_spawn_requests`), which forced a
//! full particle + map read-back and upload every frame a request was pending —
//! 23 ms per tick on a 1024x768 canvas with a plant or a pressure-emitting
//! `PLUT` in it. The bits are expanded on the device instead: these two kernels
//! translate them into creation records, and the ordinary
//! [`super::shared::drain_gpu_creates`] allocates them in the same frame.
//!
//! Split by binding count, not by semantics: the neighbour spawns need the
//! requester's pixel and the maps, the `STOR` / `NEUT` / `LIGHT` spawns need the
//! five words those three elements pack into.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::PassDims;
use super::shared::{
    CR_SET_CTYPE, CR_SET_LIFE, CR_SET_TEMP, CR_SET_TMP, CR_SET_TMP2, CR_SET_VX, CR_SET_VY,
    FLAG_SPAWN_CLONE, FLAG_SPAWN_LIGHT, FLAG_SPAWN_NEUT, FLAG_SPAWN_SELF, FLAG_SPAWN_STOR,
    FLAG_SPAWN_TMP, WORD_HIGH_TEMP_TRANSITION, elem_word, hash_u32, launch_grid, mix_index,
    pmap_at, request_create,
};
use crate::buffers::SimBuffers;

/// `min`/`max` without the `f32` methods, which CubeCL does not expand (rule 5).
#[cube]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut r = v;
    if r < lo {
        r = lo;
    }
    if r > hi {
        r = hi;
    }
    r
}

/// Element ids the kernels below compare against (rule 1: no host constants).
const PT_NONE_U32: u32 = 0;
const PT_NEUT_U32: u32 = 30;
const PT_PROT_U32: u32 = 173;
const PT_PHOT_U32: u32 = 29;
const PT_ELEC_U32: u32 = 31;
const PT_LIGH_U32: u32 = 107;
const _: () = assert!(PT_NONE == 0);
const _: () = assert!(PT_NEUT == 30);
const _: () = assert!(PT_PROT == 173);
const _: () = assert!(PT_PHOT == 29);
const _: () = assert!(PT_ELEC == 31);
const _: () = assert!(PT_LIGH == 107);
const PT_NUM_U32: u32 = 200;
const PT_LIFE_U32: u32 = 105;
const PT_LAVA_U32: u32 = 6;
/// `PIPE.cpp:283`: a `PPIP`'s border brick is created glowing.
const PT_PPIP_U32: u32 = 162;
/// See [`PT_PPIP_U32`].
const PT_BRCK_U32: u32 = 37;
const _: () = assert!(PT_LIFE == 105 && PT_LAVA == 6);
const _: () = assert!(PT_PPIP == 162 && PT_BRCK == 37);
const _: () = assert!(PT_NUM == 200);

/// The `skip`-th free pixel of the ring around `(x, y)`.
///
/// The scan starts from a position-derived offset so a field of clones does not
/// all grow towards the same corner; `skip` lets a caller that queues several
/// products in one frame ask for successive free pixels (the maps are only
/// rebuilt once per frame, so a repeated scan would return the same pixel).
///
/// Encodes `((x + 1) << 16) | (y + 1)`, or `-1` when the ring is full.
#[cube]
fn free_neighbor(
    pmap: &[u32],
    photons: &[u32],
    x: i32,
    y: i32,
    xres: u32,
    yres: u32,
    skip: u32,
) -> i32 {
    // Rule 4: seed the mutables from a runtime anchor so they stay device
    // variables instead of collapsing to the literal. Rule 10: the `x * 0`
    // idiom is what keeps them device-side, so silence `erasing_op`.
    #[allow(clippy::erasing_op)]
    let mut found = x * 0i32 - 1i32;
    #[allow(clippy::erasing_op)]
    let mut seen = (x as u32) * 0u32;
    let start = ((x as u32) * 73_856_093u32) ^ ((y as u32) * 19_349_663u32);
    let start = start % 9u32;
    for k in 0u32..9u32 {
        if found < 0i32 {
            let n = (start + k) % 9u32;
            let dx = (n % 3u32) as i32 - 1i32;
            let dy = (n / 3u32) as i32 - 1i32;
            if dx != 0i32 || dy != 0i32 {
                let nx = x + dx;
                let ny = y + dy;
                if nx >= 0i32
                    && ny >= 0i32
                    && (nx as u32) < xres
                    && (ny as u32) < yres
                    && pmap_at(pmap, xres, yres, nx, ny) == 0u32
                    && pmap_at(photons, xres, yres, nx, ny) == 0u32
                {
                    if seen == skip {
                        found = ((nx + 1i32) << 16i32) | (ny + 1i32);
                    }
                    seen += 1u32;
                }
            }
        }
    }
    found
}

/// `SPRK.cpp` `case PT_TESC` lightning life, `rng.between(0, 2 + power/15)` plus
/// `power/7`, capped at 60 (`electronics::tesc_ligh_life`).
#[cube]
fn ligh_life(power: i32, roll: u32) -> i32 {
    let p = clampf(power as f32, 0.0f32, 300.0f32) as i32;
    let span = 2i32 + p / 15i32;
    let mut life = (roll % (span as u32 + 1u32)) as i32 + p / 7i32;
    if life > 60i32 {
        life = 60i32;
    }
    life
}

/// Ask for one particle of `ty` in the `skip`-th free neighbour of `(x, y)`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn request_neighbor(
    create_queue: &mut [Atomic<u32>],
    pmap: &[u32],
    photons: &[u32],
    ty: u32,
    x: i32,
    y: i32,
    xres: u32,
    yres: u32,
    skip: u32,
    mask: u32,
    life: i32,
    ctype: u32,
    temp: f32,
    tmp: i32,
    tmp2: i32,
    tmp3: i32,
    tmp4: i32,
) {
    let spot = free_neighbor(pmap, photons, x, y, xres, yres, skip);
    if spot >= 0i32 && ty != PT_NONE_U32 && ty < PT_NUM_U32 {
        let nx = (spot >> 16i32) - 1i32;
        let ny = (spot & 0xFFFFi32) - 1i32;
        request_create(
            create_queue,
            ty,
            nx,
            ny,
            mask,
            life,
            ctype,
            temp,
            tmp,
            tmp2,
            tmp3,
            tmp4,
            0u32,
            0u32,
            0.0f32,
            0.0f32,
        );
    }
}

/// `FLAG_SPAWN_CLONE` / `SELF` / `TMP`: one particle in an adjacent free pixel.
///
/// Slot parallel over the used pool prefix. Only the request bits this kernel
/// owns are cleared, so the `STOR` / `NEUT` / `LIGHT` bits survive for
/// [`spawn_special_kernel`].
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_neighbor_kernel(
    part_flags: &mut [u32],
    part_ty: &[u32],
    part_ctype: &[u32],
    part_tmp: &[i32],
    part_x: &[f32],
    part_y: &[f32],
    pmap: &[u32],
    photons: &[u32],
    elem_words: &[u32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    n_slots: u32,
) {
    let id = ABSOLUTE_POS;
    if id < n_slots as usize {
        let flags = part_flags[id];
        let owned = FLAG_SPAWN_CLONE | FLAG_SPAWN_SELF | FLAG_SPAWN_TMP;
        if (flags & owned) != 0u32 {
            part_flags[id] = flags & !owned;
            let ty = part_ty[id];
            if ty != PT_NONE_U32 {
                let x = i32::cast_from(part_x[id] + 0.5f32);
                let y = i32::cast_from(part_y[id] + 0.5f32);
                if (flags & FLAG_SPAWN_CLONE) != 0u32 {
                    let ctype = part_ctype[id];
                    let payload_tmp = part_tmp[id];
                    // `CLNE.cpp:78-90`: the clone target decides what the child
                    // inherits. A `LIFE` payload passes the clone's `tmp` as the
                    // child's creation value, and a `LAVA` payload carries its
                    // own source element in `tmp` (only when that element melts
                    // back into lava, `CLNE.cpp:86-87`).
                    //
                    // `PIPE.cpp:282-284`: a `PPIP` sealing itself with a `BRCK`
                    // border creates the brick glowing (`parts[index].tmp = 1`);
                    // a plain `PIPE`'s border keeps the default `tmp = 0`.
                    let mut mask = 0u32;
                    let mut life = 0i32;
                    let mut child_ctype = 0u32;
                    let mut child_tmp = 0i32;
                    if ctype == PT_LIFE_U32 {
                        mask |= CR_SET_LIFE;
                        life = payload_tmp;
                    } else if ctype == PT_LAVA_U32
                        && payload_tmp > 0i32
                        && (payload_tmp as u32) < PT_NUM_U32
                    {
                        let melt =
                            elem_word(elem_words, payload_tmp as u32, WORD_HIGH_TEMP_TRANSITION);
                        if melt == PT_LAVA_U32 {
                            mask |= CR_SET_CTYPE;
                            child_ctype = payload_tmp as u32;
                        }
                    } else if ty == PT_PPIP_U32 && ctype == PT_BRCK_U32 {
                        mask |= CR_SET_TMP;
                        child_tmp = 1i32;
                    }
                    request_neighbor(
                        create_queue,
                        pmap,
                        photons,
                        ctype,
                        x,
                        y,
                        xres,
                        yres,
                        0u32,
                        mask,
                        life,
                        child_ctype,
                        0.0f32,
                        child_tmp,
                        0i32,
                        0i32,
                        0i32,
                    );
                }
                if (flags & FLAG_SPAWN_SELF) != 0u32 {
                    request_neighbor(
                        create_queue,
                        pmap,
                        photons,
                        ty,
                        x,
                        y,
                        xres,
                        yres,
                        0u32,
                        0u32,
                        0i32,
                        0u32,
                        0.0f32,
                        0i32,
                        0i32,
                        0i32,
                        0i32,
                    );
                }
                if (flags & FLAG_SPAWN_TMP) != 0u32 {
                    let spawn_ty = part_tmp[id];
                    if spawn_ty > 0i32 {
                        request_neighbor(
                            create_queue,
                            pmap,
                            photons,
                            spawn_ty as u32,
                            x,
                            y,
                            xres,
                            yres,
                            0u32,
                            0u32,
                            0i32,
                            0u32,
                            0.0f32,
                            0i32,
                            0i32,
                            0i32,
                            0i32,
                        );
                    }
                }
            }
        }
    }
}

/// `FLAG_SPAWN_STOR` / `NEUT` / `LIGHT`: the spawns that carry captured state.
///
/// Slot parallel over the used pool prefix. `STOR` also rewrites the requester
/// (its five capture words are consumed and it starts a ten frame cooldown), so
/// this kernel owns its own `part_life` / `part_tmp*` writes.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_special_kernel(
    part_flags: &mut [u32],
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_temp: &[f32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_vx: &[f32],
    part_vy: &[f32],
    pmap: &[u32],
    photons: &[u32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    n_slots: u32,
    frame_seed: u32,
) {
    let id = ABSOLUTE_POS;
    if id < n_slots as usize {
        let flags = part_flags[id];
        let owned = FLAG_SPAWN_STOR | FLAG_SPAWN_NEUT | FLAG_SPAWN_LIGHT;
        if (flags & owned) != 0u32 {
            part_flags[id] = flags & !owned;
            let ty = part_ty[id];
            if ty != PT_NONE_U32 {
                let x = i32::cast_from(part_x[id] + 0.5f32);
                let y = i32::cast_from(part_y[id] + 0.5f32);
                if (flags & FLAG_SPAWN_STOR) != 0u32 {
                    // `STOR.cpp:84-102`: replay the five captured words into a
                    // free neighbour, then go back to empty with a cooldown.
                    let stored_ty = part_tmp[id];
                    if stored_ty > 0i32 && (stored_ty as u32) < PT_NUM_U32 {
                        let mask = CR_SET_LIFE | CR_SET_CTYPE | CR_SET_TEMP | CR_SET_TMP;
                        request_neighbor(
                            create_queue,
                            pmap,
                            photons,
                            stored_ty as u32,
                            x,
                            y,
                            xres,
                            yres,
                            0u32,
                            mask,
                            part_tmp2[id],
                            part_tmp4[id] as u32,
                            part_temp[id],
                            part_tmp3[id],
                            0i32,
                            0i32,
                            0i32,
                        );
                    }
                    part_tmp[id] = 0i32;
                    part_tmp2[id] = 0i32;
                    part_tmp3[id] = 0i32;
                    part_tmp4[id] = 0i32;
                    part_life[id] = 10i32;
                }
                if (flags & FLAG_SPAWN_NEUT) != 0u32 {
                    // `POLO.cpp:62-92`: the energy layer at the requester's own
                    // pixel, only while that pixel holds no photon.
                    if x >= 0i32 && y >= 0i32 && (x as u32) < xres && (y as u32) < yres {
                        let px = (y as u32 * xres + x as u32) as usize;
                        if photons[px] == 0u32 {
                            let mut vx = 0.0f32;
                            let mut vy = 0.0f32;
                            let src = part_tmp3[id] - 1i32;
                            if src >= 0i32 && (src as usize) < n_slots as usize {
                                let s = src as usize;
                                let sty = part_ty[s];
                                if sty == PT_NEUT_U32
                                    || sty == PT_PROT_U32
                                    || sty == PT_PHOT_U32
                                    || sty == PT_ELEC_U32
                                {
                                    vx = part_vx[s];
                                    vy = part_vy[s];
                                }
                            }
                            request_create(
                                create_queue,
                                PT_NEUT_U32,
                                x,
                                y,
                                CR_SET_TEMP | CR_SET_VX | CR_SET_VY,
                                0i32,
                                0u32,
                                part_temp[id],
                                0i32,
                                0i32,
                                0i32,
                                0i32,
                                0u32,
                                0u32,
                                vx,
                                vy,
                            );
                            part_tmp3[id] = 0i32;
                        }
                    }
                }
                if (flags & FLAG_SPAWN_LIGHT) != 0u32 {
                    // `SPRK.cpp` `case PT_TESC`: one bolt per frame, aimed by the
                    // direction the coil stored in `tmp2`.
                    let power = part_tmp[id];
                    let dir = part_tmp2[id];
                    let roll = hash_u32(
                        frame_seed ^ mix_index(id as u32, 0x9E37_79B9u32) ^ (x as u32) ^ (y as u32),
                    );
                    let life = ligh_life(power, roll);
                    request_neighbor(
                        create_queue,
                        pmap,
                        photons,
                        PT_LIGH_U32,
                        x,
                        y,
                        xres,
                        yres,
                        0u32,
                        CR_SET_LIFE | CR_SET_TEMP | CR_SET_TMP | CR_SET_TMP2,
                        life,
                        0u32,
                        life as f32 * (power as f32) / 2.5f32,
                        dir,
                        1i32,
                        0i32,
                        0i32,
                    );
                }
            }
        }
    }
}

/// Expand every pending spawn request into creation records.
pub(super) fn expand_spawn_requests(bufs: &SimBuffers, dims: &PassDims) {
    let slots = bufs.used_slot_bound();
    let (cubes, dim) = launch_grid(slots);
    unsafe {
        spawn_neighbor_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(
                bufs.gpu_create_queue.clone(),
                super::shared::CREATE_QUEUE_WORDS,
            ),
            dims.xres,
            dims.yres,
            slots as u32,
        );
        spawn_special_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(
                bufs.gpu_create_queue.clone(),
                super::shared::CREATE_QUEUE_WORDS,
            ),
            dims.xres,
            dims.yres,
            slots as u32,
            dims.frame_seed,
        );
    }
}
