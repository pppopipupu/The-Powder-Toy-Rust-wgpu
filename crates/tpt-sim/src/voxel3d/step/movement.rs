//! Loose-voxel movement: claim, commit and destination masks (plan §4.4/§4.5).
//!
//! Movement is a three-pass pipeline over the simulated bricks:
//!
//! 1. `claim_clear` zeroes the claim words of the bricks in `sim_list`,
//! 2. `claim` lets every loose voxel pick at most one destination and publishes
//!    it with `atomicMax(claim[dest], key)`,
//! 3. `mark_bits` sets the leaf and coarse occupancy bits of every winning
//!    destination,
//! 4. `commit` moves the winners from `voxel_b` into `voxel_a`.
//!
//! The destination is a pure function of `(voxel_b, sim_bits, voxel_a, tick)`,
//! so `claim`, `mark_bits` and `commit` independently derive the same target
//! for the same mover. `mark_bits` therefore runs *before* `commit`: rewriting
//! `voxel_a` first would change the non-active-brick sample and could flip a
//! mover to a different target between the claim and the bits pass.
//!
//! The claim key is `(rank << 24) | index` with `rank` in `1..=127`. Keeping the
//! top bit clear preserves the plan's invariant that a rigid-body blocked key
//! (`0x80..`) always outranks a move key (§3.5); the rank makes a higher-ranked
//! mover win a contested cell and is re-derived deterministically from the
//! tick and the index.
//!
//! A target cell is free when its word is empty. Cells of bricks in `sim_bits`
//! are tested against `voxel_b` (this tick's state); every other brick is
//! tested against the authoritative `voxel_a`. Static voxels are never moved
//! because only the `ELEM_FLAG_LOOSE` phases participate.
//!
//! Clippy's `Range::contains` and `collapsible_if` rewrites use constructs the
//! CubeCL IR does not expand (see `AGENTS.md`, "Control Flow"), and collapsing
//! a nested `if` changes what the `#[cube]` macro generates. The boundary
//! helper also keeps `out = out + ...` on a runtime-seeded mutable, so the
//! assignment-pattern rewrite is disabled with them.
#![allow(
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::manual_range_contains
)]

use cubecl::prelude::*;

use super::super::Voxel3dBuffers;
use super::super::engine::Voxel3dEngine;
use super::super::gravity::NEWTON_SCALE;
use super::{
    C_BRICK_COUNT, C_ELEM_FLAGS, C_ELEM_WORDS, C_FLAG_FIRE, C_FLAG_GAS, C_FLAG_LIQUID,
    C_FLAG_LOOSE, C_NONE, C_PT_NEUT, C_PT_PHOT, C_TY_MASK, C_WORLD_DIM, brick_bit, brick_of_index,
    fabsf, hash_u32, mark_brick_bits, mark_leaf_coarse, mix_index, seed_i32, seed_u32,
};

/// Number of voxels along one brick edge.
const BRICK: u32 = 8;
/// Voxels in one brick.
const BRICK_VOXELS: u32 = 512;
/// Threads per brick: one per voxel.
const CUBE_DIM: u32 = 8;
/// Rank hash salt.
const RANK_SALT: u32 = 0x85EB_CA6Bu32;
/// Candidate-order hash salt.
const ORDER_SALT: u32 = 0x9E37_79B1u32;
/// Highest move rank; keeps bit 31 clear for the blocked-key ordering.
const MAX_RANK: u32 = 127;

const _: () = assert!(BRICK * 32u32 == C_WORLD_DIM);
const _: () = assert!(BRICK_VOXELS == 512);
const _: () = assert!(MAX_RANK < 0x80u32);

/// Whether `brick` is in this tick's simulation set.
#[cube]
fn sim_has(sim_bits: &[u32], brick: u32) -> u32 {
    let (word, bit) = brick_bit(brick);
    let mut out = seed_u32(brick, 0u32);
    if (sim_bits[word as usize] & bit) != 0u32 {
        out = 1u32;
    }
    out
}

/// Voxel word of `dest` as the movement passes must see it.
///
/// `voxel_b` is only valid for bricks in the simulation set; everything else
/// samples the authoritative array.
#[cube]
fn target_word(voxel_a: &mut [u32], voxel_b: &[u32], sim_bits: &[u32], dest: u32) -> u32 {
    let mut word = voxel_a[dest as usize];
    if sim_has(sim_bits, brick_of_index(dest)) != 0u32 {
        word = voxel_b[dest as usize];
    }
    word
}

/// TPT `EdgeMode` discriminants used by the kernels.
const EDGE_VOID: u32 = 0;
const EDGE_LOOP: u32 = 2;
/// Claim sentinel that removes a voxel which tried to leave a void world.
const C_VOID_KILL: u32 = 0xFFFF_FFFEu32;

/// Wraps one axis into `0..C_WORLD_DIM` for `EDGE_LOOP`.
#[cube]
fn wrap_axis(value: i32) -> u32 {
    let mut out = seed_i32(value, value);

    if out < 0i32 {
        out = out + C_WORLD_DIM as i32;
    }
    if out >= C_WORLD_DIM as i32 {
        out = out - C_WORLD_DIM as i32;
    }
    out as u32
}

/// Index of the neighbour `(index + dx, index + dy, index + dz)`.
///
/// `EDGE_LOOP` wraps the coordinate while `EDGE_SOLID` and `EDGE_VOID` reject
/// it with [`C_NONE`]. The void departure itself is decided by
/// [`leaves_world`] before the fallbacks run.
#[cube]
fn offset_index(index: u32, dx: i32, dy: i32, dz: i32, edge_mode: u32) -> u32 {
    let x = (index & 255u32) as i32 + dx;
    let z = ((index >> 8u32) & 255u32) as i32 + dz;
    let y = ((index >> 16u32) & 255u32) as i32 + dy;
    let mut out = seed_u32(index, C_NONE);
    if edge_mode == EDGE_LOOP {
        out = (wrap_axis(y) << 16u32) | (wrap_axis(z) << 8u32) | wrap_axis(x);
    } else if x >= 0i32
        && x < C_WORLD_DIM as i32
        && y >= 0i32
        && y < C_WORLD_DIM as i32
        && z >= 0i32
        && z < C_WORLD_DIM as i32
    {
        out = ((y as u32) << 16u32) | ((z as u32) << 8u32) | x as u32;
    }
    out
}

/// Whether `index + (dx, dy, dz)` leaves the world, ignoring `edge_mode`.
///
/// [`offset_index`] folds the wrap and the bounds check into one result, so a
/// caller that must tell "out of the world" from "occupied" asks here first.
#[cube]
fn leaves_world(index: u32, dx: i32, dy: i32, dz: i32) -> u32 {
    let x = (index & 255u32) as i32 + dx;
    let z = ((index >> 8u32) & 255u32) as i32 + dz;
    let y = ((index >> 16u32) & 255u32) as i32 + dy;
    let mut out = seed_u32(index, 0u32);
    if x < 0i32
        || x >= C_WORLD_DIM as i32
        || y < 0i32
        || y >= C_WORLD_DIM as i32
        || z < 0i32
        || z >= C_WORLD_DIM as i32
    {
        out = 1u32;
    }
    out
}

/// Returns the destination of one unit step if it is inside the world and
/// empty, otherwise [`C_NONE`].
#[cube]
fn try_step(
    voxel_a: &mut [u32],
    voxel_b: &[u32],
    sim_bits: &[u32],
    index: u32,
    dx: i32,
    dy: i32,
    dz: i32,
    edge_mode: u32,
) -> u32 {
    let dest = offset_index(index, dx, dy, dz, edge_mode);
    let mut out = seed_u32(index, C_NONE);
    if dest != C_NONE {
        if (target_word(voxel_a, voxel_b, sim_bits, dest) & C_TY_MASK) == 0u32 {
            out = dest;
        }
    }
    out
}

/// Converts `(axis, sign)` to a unit offset in `(dx, dy, dz)` order with
/// `x` = 0, `y` = 1 and `z` = 2.
#[cube]
fn axis_offset(axis: u32, sign: i32) -> (i32, i32, i32) {
    let mut dx = seed_i32(sign, 0i32);
    let mut dy = seed_i32(sign, 0i32);
    let mut dz = seed_i32(sign, 0i32);
    if axis == 0u32 {
        dx = sign;
    } else if axis == 1u32 {
        dy = sign;
    } else {
        dz = sign;
    }
    (dx, dy, dz)
}

/// The `slot`-th horizontal unit offset in a rotation order chosen by `rot`.
///
/// `rot` selects one of the four alternating orders `+a,-a,+b,-b`,
/// `-a,+a,-b,+b`, `+b,-b,+a,-a` and `-b,+b,-a,+a`. The per-voxel hash picks
/// `rot`, so powders and liquids do not all spread along the same axis.
#[cube]
#[allow(clippy::too_many_arguments)]
fn horiz_offset(
    rot: u32,
    slot: u32,
    ax: i32,
    ay: i32,
    az: i32,
    bx: i32,
    by: i32,
    bz: i32,
) -> (i32, i32, i32) {
    let mut use_b = seed_u32(slot, 0u32);
    let mut negate = seed_u32(slot, 0u32);
    if rot == 0u32 {
        if slot >= 2u32 {
            use_b = 1u32;
        }
        if (slot & 1u32) == 1u32 {
            negate = 1u32;
        }
    } else if rot == 1u32 {
        if slot >= 2u32 {
            use_b = 1u32;
        }
        if (slot & 1u32) == 0u32 {
            negate = 1u32;
        }
    } else if rot == 2u32 {
        if slot < 2u32 {
            use_b = 1u32;
        }
        if (slot & 1u32) == 1u32 {
            negate = 1u32;
        }
    } else {
        if slot < 2u32 {
            use_b = 1u32;
        }
        if (slot & 1u32) == 0u32 {
            negate = 1u32;
        }
    }

    let mut dx = seed_i32(ax, ax);
    let mut dy = seed_i32(ay, ay);
    let mut dz = seed_i32(az, az);
    if use_b != 0u32 {
        dx = bx;
        dy = by;
        dz = bz;
    }
    if negate != 0u32 {
        dx = -dx;
        dy = -dy;
        dz = -dz;
    }
    (dx, dy, dz)
}

/// Destination of one loose voxel, or [`C_NONE`].
///
/// Powders fall first. Liquids and gases occasionally try a horizontal step
/// before the fall/rise so a blob spreads while it moves. The per-voxel hash
/// also rotates the four horizontal directions. Everything stays a pure
/// function of its arguments so the three movement passes agree.
#[cube]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn choose_dest(
    voxel_a: &mut [u32],
    voxel_b: &[u32],
    sim_bits: &[u32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    grav_fz: &[f32],
    index: u32,
    ty: u32,
    flags: u32,
    gx: f32,
    gy: f32,
    gz: f32,
    tick: u32,
    edge_mode: u32,
) -> u32 {
    // `GetGravityField` (`Simulation.cpp:2088-2120`) adds the Newtonian force
    // to the particle's own gravity. `grav_f*` already holds `-grad(u)` on the
    // 4³ cell of this voxel; [`NEWTON_SCALE`] converts one TPT gravity unit to
    // the voxel-per-second-squared units of `Voxel3dConfig::gravity`.
    let cell = (((index >> 18u32) & 0x3Fu32) << 12u32)
        | (((index >> 10u32) & 0x3Fu32) << 6u32)
        | ((index >> 2u32) & 0x3Fu32);
    let ngx = gx + NEWTON_SCALE * grav_fx[cell as usize];
    let ngy = gy + NEWTON_SCALE * grav_fy[cell as usize];
    let ngz = gz + NEWTON_SCALE * grav_fz[cell as usize];

    let gxa = fabsf(ngx);
    let gya = fabsf(ngy);
    let gza = fabsf(ngz);

    let mut fall_axis = seed_u32(index, 1u32);
    let mut fall_sign = seed_i32(index as i32, 0i32);
    if gxa >= gya && gxa >= gza {
        fall_axis = 0u32;
        if ngx > 0.0f32 {
            fall_sign = 1i32;
        } else if ngx < 0.0f32 {
            fall_sign = -1i32;
        }
    } else if gya >= gza {
        fall_axis = 1u32;
        if ngy > 0.0f32 {
            fall_sign = 1i32;
        } else if ngy < 0.0f32 {
            fall_sign = -1i32;
        }
    } else {
        fall_axis = 2u32;
        if ngz > 0.0f32 {
            fall_sign = 1i32;
        } else if ngz < 0.0f32 {
            fall_sign = -1i32;
        }
    }

    let mut is_gas = seed_u32(index, 0u32);
    if (flags & (C_FLAG_GAS | C_FLAG_FIRE)) != 0u32 {
        is_gas = 1u32;
    }
    let mut is_liquid = seed_u32(index, 0u32);
    if (flags & C_FLAG_LIQUID) != 0u32 {
        is_liquid = 1u32;
    }
    let mut primary_sign = seed_i32(fall_sign, fall_sign);
    if is_gas != 0u32 {
        primary_sign = -fall_sign;
    }

    let mut perp_a = seed_u32(index, 0u32);
    let mut perp_b = seed_u32(index, 1u32);
    if fall_axis == 0u32 {
        perp_a = 1u32;
        perp_b = 2u32;
    } else if fall_axis == 1u32 {
        perp_a = 0u32;
        perp_b = 2u32;
    } else {
        perp_a = 0u32;
        perp_b = 1u32;
    }
    let (ax, ay, az) = axis_offset(perp_a, 1i32);
    let (bx, by, bz) = axis_offset(perp_b, 1i32);

    let h = hash_u32(mix_index(index, ORDER_SALT) ^ tick ^ (ty << 8u32));
    let rot = (h >> 2u32) & 3u32;

    let mut primary_first = seed_u32(index, 1u32);
    if is_liquid != 0u32 {
        if (h & 3u32) == 0u32 {
            primary_first = 0u32;
        }
    }
    if is_gas != 0u32 {
        if (h & 1u32) == 0u32 {
            primary_first = 0u32;
        }
    }

    let (px, py, pz) = axis_offset(fall_axis, primary_sign);
    let mut dest = seed_u32(index, C_NONE);

    // TPT moves the particle to its gravity step first and deletes it when that
    // position leaves a void world (`Simulation::move`, Simulation.cpp:1550-1556);
    // the sideways fallbacks only run after an in-world move was blocked
    // (`MovementPhase`, Simulation.cpp:3126-3200). Deciding the departure
    // before the fallbacks is what empties the boundary: a voxel on the bottom
    // face slid along the face forever instead of falling out.
    if edge_mode == EDGE_VOID && primary_sign != 0i32 {
        if leaves_world(index, px, py, pz) != 0u32 {
            dest = C_VOID_KILL;
        }
    }

    if dest == C_NONE && primary_first == 1u32 && primary_sign != 0i32 {
        dest = try_step(voxel_a, voxel_b, sim_bits, index, px, py, pz, edge_mode);
    }

    if dest == C_NONE {
        let (dx, dy, dz) = horiz_offset(rot, 0u32, ax, ay, az, bx, by, bz);
        dest = try_step(voxel_a, voxel_b, sim_bits, index, dx, dy, dz, edge_mode);
    }
    if dest == C_NONE {
        let (dx, dy, dz) = horiz_offset(rot, 1u32, ax, ay, az, bx, by, bz);
        dest = try_step(voxel_a, voxel_b, sim_bits, index, dx, dy, dz, edge_mode);
    }
    if dest == C_NONE {
        let (dx, dy, dz) = horiz_offset(rot, 2u32, ax, ay, az, bx, by, bz);
        dest = try_step(voxel_a, voxel_b, sim_bits, index, dx, dy, dz, edge_mode);
    }
    if dest == C_NONE {
        let (dx, dy, dz) = horiz_offset(rot, 3u32, ax, ay, az, bx, by, bz);
        dest = try_step(voxel_a, voxel_b, sim_bits, index, dx, dy, dz, edge_mode);
    }

    if dest == C_NONE && primary_sign != 0i32 {
        let (dx, dy, dz) = horiz_offset(rot, 0u32, ax, ay, az, bx, by, bz);
        dest = try_step(
            voxel_a,
            voxel_b,
            sim_bits,
            index,
            px + dx,
            py + dy,
            pz + dz,
            edge_mode,
        );
    }
    if dest == C_NONE && primary_sign != 0i32 {
        let (dx, dy, dz) = horiz_offset(rot, 1u32, ax, ay, az, bx, by, bz);
        dest = try_step(
            voxel_a,
            voxel_b,
            sim_bits,
            index,
            px + dx,
            py + dy,
            pz + dz,
            edge_mode,
        );
    }
    if dest == C_NONE && primary_sign != 0i32 {
        let (dx, dy, dz) = horiz_offset(rot, 2u32, ax, ay, az, bx, by, bz);
        dest = try_step(
            voxel_a,
            voxel_b,
            sim_bits,
            index,
            px + dx,
            py + dy,
            pz + dz,
            edge_mode,
        );
    }
    if dest == C_NONE && primary_sign != 0i32 {
        let (dx, dy, dz) = horiz_offset(rot, 3u32, ax, ay, az, bx, by, bz);
        dest = try_step(
            voxel_a,
            voxel_b,
            sim_bits,
            index,
            px + dx,
            py + dy,
            pz + dz,
            edge_mode,
        );
    }

    if dest == C_NONE && primary_first == 0u32 && primary_sign != 0i32 {
        dest = try_step(voxel_a, voxel_b, sim_bits, index, px, py, pz, edge_mode);
    }
    dest
}

/// Deterministic move rank of `index` on `tick`, in `1..=127`.
#[cube]
fn move_rank(tick: u32, index: u32) -> u32 {
    1u32 + (hash_u32(mix_index(index, RANK_SALT) ^ tick) % MAX_RANK)
}

/// Claim key of `index` on `tick`: `(rank << 24) | (index & 0xFF_FFFF)`.
#[cube]
fn move_key(tick: u32, index: u32) -> u32 {
    (move_rank(tick, index) << 24u32) | (index & 0x00FF_FFFFu32)
}

/// Local voxel coordinates of the thread's cell.
#[cube]
fn local_coords() -> (u32, u32, u32) {
    (UNIT_POS_X, UNIT_POS_Y, UNIT_POS_Z)
}

/// Clears the claim words of every simulated brick.
///
/// One thread per brick scans its 512 cells: claims are only ever written into
/// this tick's simulation set or into currently empty bricks, so clearing the
/// simulated bricks removes every claim the movement passes may read.
#[cube(launch_unchecked)]
fn claim_clear_kernel(claim: &mut [Atomic<u32>], sim_list: &[u32]) {
    let brick = sim_list[CUBE_POS_X as usize];
    let bx = brick & 31u32;
    let bz = (brick >> 5u32) & 31u32;
    let by = (brick >> 10u32) & 31u32;
    for cell in 0..BRICK_VOXELS {
        let lx = cell & 7u32;
        let lz = (cell >> 3u32) & 7u32;
        let ly = (cell >> 6u32) & 7u32;
        let x = bx * BRICK + lx;
        let y = by * BRICK + ly;
        let z = bz * BRICK + lz;
        let index = (y << 16u32) | (z << 8u32) | x;
        claim[index as usize].store(0u32);
    }
}

/// Publishes one claim per loose mover.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn claim_kernel(
    voxel_a: &mut [u32],
    voxel_b: &[u32],
    sim_bits: &[u32],
    sim_list: &[u32],
    table: &[u32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    grav_fz: &[f32],
    claim: &mut [Atomic<u32>],
    tick: u32,
    gx: f32,
    gy: f32,
    gz: f32,
    edge_mode: u32,
) {
    let brick = sim_list[CUBE_POS_X as usize];
    let (lx, ly, lz) = local_coords();
    let x = (brick & 31u32) * BRICK + lx;
    let y = ((brick >> 10u32) & 31u32) * BRICK + ly;
    let z = ((brick >> 5u32) & 31u32) * BRICK + lz;
    let index = (y << 16u32) | (z << 8u32) | x;

    let word = voxel_b[index as usize];
    let ty = word & C_TY_MASK;
    if ty != 0u32 {
        let flags = table[(ty as usize) * C_ELEM_WORDS + C_ELEM_FLAGS];
        // PHOT/NEUT are `TYPE_ENERGY` in C++ and move by their own velocity
        // update (`Simulation.cpp:2951-2957`), not through the powder/liquid
        // gravity path; [`super::radioactive`] advances them one voxel per
        // tick along their stored direction.
        if (flags & C_FLAG_LOOSE) != 0u32 && ty != C_PT_PHOT && ty != C_PT_NEUT {
            let dest = choose_dest(
                voxel_a, voxel_b, sim_bits, grav_fx, grav_fy, grav_fz, index, ty, flags, gx, gy,
                gz, tick, edge_mode,
            );
            if dest == C_VOID_KILL {
                claim[index as usize].store(C_VOID_KILL);
            } else if dest != C_NONE {
                claim[dest as usize].fetch_max(move_key(tick, index));
            }
        }
    }
}

/// Sets the leaf and coarse occupancy bits of every winning destination.
///
/// Runs before [`commit_kernel`] for the reason in the module docs: the
/// destination derivation samples `voxel_a` for bricks outside the simulation
/// set, and commit rewrites that array.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn mark_bits_kernel(
    voxel_a: &mut [u32],
    voxel_b: &[u32],
    sim_bits: &[u32],
    sim_list: &[u32],
    table: &[u32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    grav_fz: &[f32],
    claim: &mut [Atomic<u32>],
    leaf_masks: &mut [Atomic<u32>],
    coarse_masks: &mut [Atomic<u32>],
    tick: u32,
    gx: f32,
    gy: f32,
    gz: f32,
    edge_mode: u32,
) {
    let brick = sim_list[CUBE_POS_X as usize];
    let (lx, ly, lz) = local_coords();
    let x = (brick & 31u32) * BRICK + lx;
    let y = ((brick >> 10u32) & 31u32) * BRICK + ly;
    let z = ((brick >> 5u32) & 31u32) * BRICK + lz;
    let index = (y << 16u32) | (z << 8u32) | x;

    let word = voxel_b[index as usize];
    let ty = word & C_TY_MASK;
    if ty != 0u32 {
        let flags = table[(ty as usize) * C_ELEM_WORDS + C_ELEM_FLAGS];
        // See [`claim_kernel`]: energy voxels keep their own movement model.
        if (flags & C_FLAG_LOOSE) != 0u32 && ty != C_PT_PHOT && ty != C_PT_NEUT {
            let dest = choose_dest(
                voxel_a, voxel_b, sim_bits, grav_fx, grav_fy, grav_fz, index, ty, flags, gx, gy,
                gz, tick, edge_mode,
            );
            if dest != C_NONE && dest != C_VOID_KILL && dest != index {
                if claim[dest as usize].fetch_max(0u32) == move_key(tick, index) {
                    mark_leaf_coarse(leaf_masks, coarse_masks, dest);
                }
            }
        }
    }
}

/// Moves every winning voxel from `voxel_b` into `voxel_a`.
///
/// A cell that some mover claimed is skipped: the winner writes it. Cells
/// that were not claimed take their `voxel_b` word, which is zero for the
/// cells the thermal pass emptied. Source leaf bits are cleared by the mask
/// rebuild pass, which runs over the whole simulation set.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn commit_kernel(
    voxel_a: &mut [u32],
    voxel_b: &[u32],
    sim_bits: &[u32],
    sim_list: &[u32],
    table: &[u32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    grav_fz: &[f32],
    claim: &mut [Atomic<u32>],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    brick_occupied: &mut [Atomic<u32>],
    dirty_bricks: &mut [Atomic<u32>],
    tick: u32,
    gx: f32,
    gy: f32,
    gz: f32,
    edge_mode: u32,
) {
    let brick = sim_list[CUBE_POS_X as usize];
    let (lx, ly, lz) = local_coords();
    let x = (brick & 31u32) * BRICK + lx;
    let y = ((brick >> 10u32) & 31u32) * BRICK + ly;
    let z = ((brick >> 5u32) & 31u32) * BRICK + lz;
    let index = (y << 16u32) | (z << 8u32) | x;

    let word = voxel_b[index as usize];
    let ty = word & C_TY_MASK;
    let mut moved = seed_u32(index, 0u32);

    if ty != 0u32 {
        let flags = table[(ty as usize) * C_ELEM_WORDS + C_ELEM_FLAGS];
        // See [`claim_kernel`]: energy voxels keep their own movement model.
        if (flags & C_FLAG_LOOSE) != 0u32 && ty != C_PT_PHOT && ty != C_PT_NEUT {
            let dest = choose_dest(
                voxel_a, voxel_b, sim_bits, grav_fx, grav_fy, grav_fz, index, ty, flags, gx, gy,
                gz, tick, edge_mode,
            );
            if dest == C_VOID_KILL {
                voxel_a[index as usize] = 0u32;
                moved = 1u32;
            } else if dest != C_NONE {
                if claim[dest as usize].fetch_max(0u32) == move_key(tick, index) {
                    voxel_a[dest as usize] = word;
                    aux_ctype[dest as usize] = aux_ctype[index as usize];
                    aux_tmp[dest as usize] = aux_tmp[index as usize];
                    aux_tmp2[dest as usize] = aux_tmp2[index as usize];
                    aux_tmp3[dest as usize] = aux_tmp3[index as usize];
                    voxel_a[index as usize] = 0u32;
                    moved = 1u32;
                    mark_brick_bits(brick_occupied, dirty_bricks, brick_of_index(dest));
                }
            }
        }
    }

    if moved == 0u32 {
        // A claimed cell belongs to its winner; skip it even when this cell is
        // empty in `voxel_b`, otherwise the source thread would race the write.
        if claim[index as usize].fetch_max(0u32) == 0u32 {
            voxel_a[index as usize] = word;
        }
    }
}

/// Clears auxiliary words wherever the committed world is empty.
///
/// Movement copies aux into the destination and leaves the source's aux in
/// place until this pass, so every cleared cell (moved out, killed by the
/// reaction, or erased by a blob) starts the next tick with `ctype = tmp =
/// tmp2 = tmp3 = 0`, as a freshly created particle would.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn clear_aux_kernel(
    voxel_a: &[u32],
    sim_list: &[u32],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
) {
    let brick = sim_list[CUBE_POS_X as usize];
    let (lx, ly, lz) = local_coords();
    let x = (brick & 31u32) * BRICK + lx;
    let y = ((brick >> 10u32) & 31u32) * BRICK + ly;
    let z = ((brick >> 5u32) & 31u32) * BRICK + lz;
    let index = (y << 16u32) | (z << 8u32) | x;
    if voxel_a[index as usize] == 0u32 {
        aux_ctype[index as usize] = 0u32;
        aux_tmp[index as usize] = 0i32;
        aux_tmp2[index as usize] = 0i32;
        aux_tmp3[index as usize] = 0i32;
    }
}

/// Clears the claims of the simulated bricks.
pub(crate) fn clear_claims(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        claim_clear_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Dynamic(bufs.sim_count.clone().binding()),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.claim.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), C_BRICK_COUNT as usize),
        );
    }
}

/// Runs the claim pass.
pub(crate) fn claim(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    let tick = engine.tick as u32;
    let [gx, gy, gz] = engine.config.gravity;
    let edge_mode = engine.config.edge_mode as u32;
    unsafe {
        claim_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Dynamic(bufs.sim_count.clone().binding()),
            CubeDim::new_3d(CUBE_DIM, CUBE_DIM, CUBE_DIM),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.voxel_b.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.sim_bits.clone(), super::C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
            BufferArg::from_raw_parts(bufs.grav_force_x.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(bufs.grav_force_y.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(bufs.grav_force_z.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(bufs.claim.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            tick,
            gx,
            gy,
            gz,
            edge_mode,
        );
    }
}

/// Sets the occupancy bits of every winning destination.
pub(crate) fn mark_bits(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    let tick = engine.tick as u32;
    let [gx, gy, gz] = engine.config.gravity;
    let edge_mode = engine.config.edge_mode as u32;
    unsafe {
        mark_bits_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Dynamic(bufs.sim_count.clone().binding()),
            CubeDim::new_3d(CUBE_DIM, CUBE_DIM, CUBE_DIM),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.voxel_b.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.sim_bits.clone(), super::C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
            BufferArg::from_raw_parts(bufs.grav_force_x.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(bufs.grav_force_y.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(bufs.grav_force_z.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(bufs.claim.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.leaf_masks.clone(), tpt_core::voxel3d::LEAF_WORD_COUNT),
            BufferArg::from_raw_parts(bufs.coarse_masks.clone(), super::super::COARSE_MASK_WORDS),
            tick,
            gx,
            gy,
            gz,
            edge_mode,
        );
    }
}

/// Clears auxiliary words of empty voxels after the commit.
pub(crate) fn clear_aux(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        clear_aux_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Dynamic(bufs.sim_count.clone().binding()),
            CubeDim::new_3d(CUBE_DIM, CUBE_DIM, CUBE_DIM),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(bufs.aux_ctype.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp2.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp3.clone(), tpt_core::voxel3d::VOXEL_COUNT),
        );
    }
}

/// Moves the winners into `voxel_a`.
pub(crate) fn commit(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    let tick = engine.tick as u32;
    let [gx, gy, gz] = engine.config.gravity;
    let edge_mode = engine.config.edge_mode as u32;
    unsafe {
        commit_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Dynamic(bufs.sim_count.clone().binding()),
            CubeDim::new_3d(CUBE_DIM, CUBE_DIM, CUBE_DIM),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.voxel_b.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.sim_bits.clone(), super::C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
            BufferArg::from_raw_parts(bufs.grav_force_x.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(bufs.grav_force_y.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(bufs.grav_force_z.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(bufs.claim.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_ctype.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp2.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp3.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.brick_occupied.clone(), super::C_BRICK_WORDS as usize),
            BufferArg::from_raw_parts(bufs.dirty_bricks.clone(), super::C_BRICK_WORDS as usize),
            tick,
            gx,
            gy,
            gz,
            edge_mode,
        );
    }
}
