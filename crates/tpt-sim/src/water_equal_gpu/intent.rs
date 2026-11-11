//! Pass 3: Candidate move intent & arbitration for water equalisation.
//!
//! One thread per particle slot `id in 0..num_slots`.
//! Checks particle eligibility:
//! - Must be active liquid (`falldown == 2`).
//! - Not marked rigid (`FLAG_RIGID_PIXEL`) or non-moving (`FLAG_SKIPMOVE`).
//! - Passes the 1-in-200 stochastic gate (`flood_chance`).
//! - Pre-move `pmap` points to this particle slot.
//!
//! If eligible and the particle's column has an assigned target below this particle,
//! competes for that target pixel via `water_winner[target].fetch_min(id + 1)`.

use cubecl::prelude::*;
use tpt_core::constants::{FLAG_RIGID_PIXEL, FLAG_SKIPMOVE, PT_NUM, TYPE_LIQUID};

use crate::move_kernel::eval_move_at;

const PT_NUM_U32: u32 = PT_NUM as u32;
const FLAG_RIGID_PIXEL_U32: u32 = FLAG_RIGID_PIXEL;
const FLAG_SKIPMOVE_U32: u32 = FLAG_SKIPMOVE;

const _: () = assert!(PT_NUM_U32 as usize == PT_NUM);
const _: () = assert!(TYPE_LIQUID == 2);
const _: () = assert!(FLAG_RIGID_PIXEL == 0x10);
const _: () = assert!(FLAG_SKIPMOVE == 0x02);

#[cube]
fn mix_index(index: u32, salt: u32) -> u32 {
    index * salt
}

#[cube]
fn hash_u32(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16u32;
    h = mix_index(h, 2_246_822_507u32);
    h ^= h >> 13u32;
    h = mix_index(h, 3_266_489_909u32);
    h ^= h >> 16u32;
    h
}

#[cube]
fn flood_chance(frame_seed: u32, id: u32) -> bool {
    let mixed = frame_seed ^ mix_index(id, 0x9E37_79B9u32);
    hash_u32(mixed).is_multiple_of(200u32)
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn water_intent_kernel(
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_flags: &[u32],
    pmap: &[u32],
    elem_props: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    col_target: &mut [Atomic<u32>],
    water_winner: &mut [Atomic<u32>],
    num_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let id = ABSOLUTE_POS as u32;
    if id >= num_slots {
        terminate!();
    }
    let ty = part_ty[id as usize];
    if ty == 0u32 || ty >= PT_NUM_U32 {
        terminate!();
    }
    let falldown = elem_props[(ty as usize) * 32usize + 3usize];
    if falldown != 2u32 {
        terminate!();
    }
    let flags = part_flags[id as usize];
    if (flags & (FLAG_RIGID_PIXEL_U32 | FLAG_SKIPMOVE_U32)) != 0u32 {
        terminate!();
    }
    if !flood_chance(frame_seed, id) {
        terminate!();
    }

    let px = part_x[id as usize];
    let py = part_y[id as usize];
    let xi = i32::cast_from(px + 0.5f32);
    let yi = i32::cast_from(py + 0.5f32);
    if xi < 0i32 || xi >= (xres as i32) || yi < 0i32 || yi >= (yres as i32) {
        terminate!();
    }
    let idx = (yi as usize) * (xres as usize) + (xi as usize);
    let p = pmap[idx];
    if (p >> 9u32) != (id + 1u32) {
        terminate!();
    }

    let packed = col_target[xi as usize].fetch_max(0u32);
    if packed == 0u32 {
        terminate!();
    }
    let target = packed - 1u32;
    let dest_y = target / xres;
    // `flood_water` (`Simulation.cpp:735-739`) only teleports into an empty
    // pixel that also passes `eval_move`: walls, `can_move` and the VOID /
    // PVOD / INVIS filters all have to admit the particle.
    let dest_x = target % xres;
    let props = elem_props[(ty as usize) * 32usize + 1usize];
    let can_enter = eval_move_at(
        pmap,
        wall_bmap,
        wall_emap,
        move_lut,
        props,
        ty,
        dest_x as i32,
        dest_y as i32,
        xres as i32,
        yres as i32,
        xcells,
    ) != 0u32;
    if (yi as u32) < dest_y && pmap[target as usize] == 0u32 && can_enter {
        water_winner[target as usize].fetch_min(id + 1u32);
    }
}
