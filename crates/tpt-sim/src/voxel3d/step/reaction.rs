//! Thermal conduction and the faithful complex-element update pass.
//!
//! One 8x8x8 workgroup handles one simulated brick; every thread owns one
//! voxel. The pass reads the authoritative `voxel_a` and writes the complete
//! resulting word to `voxel_b`, including zeros for empty cells, so `voxel_b`
//! is a pure function of `voxel_a` (plan §3.6). The life pass
//! ([`super::life`]) runs before this pass, exactly like
//! `Simulation::RecalcFreeParticles` (`Simulation.cpp:3378-3425`), so the
//! element branches observe the post-decrement life counter.
//!
//! # Fidelity rules
//!
//! Every branch below is a direct port of the C++ element file named in its
//! comment. Neighbour effects that TPT implements with `create_part` or
//! `part_change_type` cannot write another voxel from this pass without
//! breaking the one-writer-per-cell rule, so they append a record to the
//! mutation queue; [`super::mutation`] drains the queue after the movement
//! commit and before the mask rebuild. Self changes are written to `voxel_b`
//! and to this cell's auxiliary words.
//!
//! `tmp` is `aux_tmp`, `tmp2` is `aux_tmp2`, `tmp3` is `aux_tmp3` and `ctype`
//! is `aux_ctype`. The packed word carries `ty`, temperature and life only.
//!
//! TPT's `pv` is the coarse 64³ pressure field in 1/1024 units. Every threshold
//! and impulse in the ported files is written with the original constant and
//! converted at the read site (`Air.cpp`, `SimulationConfig.h`); the field is
//! updated by [`super::super::air_kernel`] one tick later, like
//! `Air::update_air` running at the start of the next frame.
//!
//! # Known representational limits
//!
//! * `MAX_TEMP` is 9999 K in TPT, but the packed 16-bit `temp_q` field saturates
//!   at 8191.875 K (0.125 K units). A branch asking for `MAX_TEMP` writes the
//!   saturated value; no threshold used by this wave is above 3500 K except
//!   the unreachable O2/BREC branches.
//! * Particle velocity `vx/vy/vz` is not modelled by the 3D movement pass
//!   (the pass keeps its stochastic model), so `create_part` velocity writes
//!   are not stored. This is the documented scope exception in the task; no
//!   branch claims TPT velocity.
//! * H2 fusion (`H2.cpp:81-113`) and O2 high-temperature decay
//!   (`O2.cpp:80-123`) need NBLE/NEUT/PHOT/ELEC/GRVT, which are not part of
//!   this wave. Those branches are reported as blocked and are the only paths
//!   left out; the H2/O2 fire reactions below are implemented in full.
//!
//! Clippy's `Range::contains` and `collapsible_if` rewrites use constructs the
//! CubeCL IR does not expand, so both lints are disabled for this module.
#![allow(clippy::collapsible_if, clippy::manual_range_contains)]

use cubecl::prelude::*;

use super::super::Voxel3dBuffers;
use super::super::engine::Voxel3dEngine;
use super::super::mutation::{MODE_CREATE, MODE_KILL, MODE_REPLACE, push as push_mutation};
use super::{
    C_BRICK_COUNT, C_ELEM_BOIL, C_ELEM_BOIL_PRODUCT, C_ELEM_BURN_PRODUCT, C_ELEM_EXPLOSIVE,
    C_ELEM_FLAGS, C_ELEM_FLAMMABLE, C_ELEM_FREEZE, C_ELEM_FREEZE_PRODUCT, C_ELEM_HEAT_CONDUCT,
    C_ELEM_IGNITION, C_ELEM_MELT, C_ELEM_MELT_PRODUCT, C_ELEM_TPT_PROPS, C_ELEM_WORDS, C_FLAG_FIRE,
    C_FLAG_GAS, C_FLAG_LIQUID, C_FLAG_POWDER, C_FLAG_STATIC, C_LIFE_MASK, C_LIFE_SHIFT, C_NONE,
    C_PT_BANG, C_PT_BCOL, C_PT_BMTL, C_PT_BOMB, C_PT_BREC, C_PT_BRMT, C_PT_COAL, C_PT_DSTW,
    C_PT_EMBR, C_PT_EXOT, C_PT_FIRE, C_PT_FSEP, C_PT_FUSE, C_PT_GLAS, C_PT_H2, C_PT_ICEI,
    C_PT_IGNT, C_PT_LAVA, C_PT_LIGH, C_PT_NBLE, C_PT_NITR, C_PT_O2, C_PT_PLSM, C_PT_SALT,
    C_PT_SAND, C_PT_SLTW, C_PT_SMKE, C_PT_SNOW, C_PT_SPRK, C_PT_STNE, C_PT_THRM, C_PT_WATR,
    C_PT_WOOD, C_PT_WTRV, C_TEMP_MASK, C_TEMP_SHIFT, C_TPT_CARRIES_CTYPE, C_TPT_LIFE_DEC,
    C_TPT_LIFE_KILL, C_TPT_LIFE_KILL_DEC, C_TPT_SPARKSETTLE, C_TY_MASK, C_WORLD_DIM, clampf,
    hash_u32, mix_index, seed_f32, seed_i32, seed_u32,
};

/// Number of voxels along one brick edge.
const BRICK: i32 = 8;
/// Number of voxels along one world edge.
const WORLD: i32 = C_WORLD_DIM as i32;
/// Threads per brick: one per voxel.
const CUBE_DIM: u32 = 8;
/// Number of requests the explosion queue accepts.
const EXPLOSION_CAPACITY: u32 = 4_096;
/// Words per explosion request.
const EXPLOSION_STRIDE: u32 = 4;
/// Largest temperature representable by the packed word, `65535 / 8`.
const MAX_PACKED_TEMP_K: f32 = 8191.875;
/// Temperature of a voxel ignited by an adjacent fire (`FIRE.cpp:310-312`).
const FIRE_DEFAULT_TEMP_K: f32 = 695.15;
/// TPT `MAX_TEMP` (`ElementDefs.h:5`); packed values saturate below it.
const TPT_MAX_TEMP_K: f32 = 9999.0;
/// Fallback air conductivity used for empty neighbours.
const AIR_CONDUCT: f32 = 0.1;
/// Fraction of the neighbour difference exchanged per conduction edge.
const CONDUCT_RATE: f32 = 0.15;
/// RNG salt of the per-voxel roll chain.
const RNG_SALT: u32 = 0x85EB_CA6Bu32;
/// RNG salt of the per-neighbour roll chain.
const NEIGHBOUR_SALT: u32 = 0xC2B2_AE35u32;
/// Fixed-point units per TPT pressure unit (`pv` is 1/1024).
const PV_SCALE_F: f32 = 1024.0;

const _: () = assert!(CUBE_DIM * CUBE_DIM * CUBE_DIM == 512);
const _: () = assert!(BRICK as u32 * 32u32 == C_WORLD_DIM);
const _: () = assert!(EXPLOSION_STRIDE == 4u32);

/// Air-cell index of a world coordinate (`CELL = 4`, `Air.cpp` layout).
#[cube]
fn air_cell(x: u32, y: u32, z: u32) -> u32 {
    ((y >> 2u32) << 12u32) | ((z >> 2u32) << 6u32) | (x >> 2u32)
}

/// TPT `pv` at a world coordinate, in pressure units.
#[cube]
fn pv_at(pv: &mut [Atomic<i32>], x: u32, y: u32, z: u32) -> f32 {
    pv[air_cell(x, y, z) as usize].load() as f32 * (1.0f32 / PV_SCALE_F)
}

/// Adds a TPT pressure impulse (`pv += impulse`) to the cell of `(x, y, z)`.
#[cube]
fn pv_add(pv: &mut [Atomic<i32>], x: u32, y: u32, z: u32, impulse: f32) {
    pv[air_cell(x, y, z) as usize].fetch_add((impulse * PV_SCALE_F) as i32);
}

/// Packed word of `ty` at `temp_k` with `life`.
#[cube]
fn pack_word(ty: u32, temp_k: f32, life: u32) -> u32 {
    let mut q = seed_f32(temp_k, 0.0f32);
    if temp_k > 0.0f32 {
        q = (temp_k * 8.0f32).round();
    }
    let mut temp_q = q as u32;
    if temp_q > 65_535u32 {
        temp_q = 65_535u32;
    }
    let mut life_out = seed_u32(life, life);
    if life_out > C_LIFE_MASK {
        life_out = C_LIFE_MASK;
    }
    ty | (temp_q << C_TEMP_SHIFT) | (life_out << C_LIFE_SHIFT)
}

/// TPT `rng.chance(a, b)` with a deterministic per-roll seed.
#[cube]
fn chance(seed: u32, a: u32, b: u32) -> bool {
    (hash_u32(seed) % b) < a
}

/// TPT `rng.between(lo, hi)` with a deterministic per-roll seed.
#[cube]
fn between(seed: u32, lo: u32, hi: u32) -> u32 {
    lo + hash_u32(seed) % (hi - lo + 1u32)
}

/// World index of a neighbour, or [`C_NONE`] when outside the world.
#[cube]
fn world_index(x: i32, y: i32, z: i32) -> u32 {
    let mut out = seed_i32(x, C_NONE as i32);
    if x >= 0i32 && x < WORLD && y >= 0i32 && y < WORLD && z >= 0i32 && z < WORLD {
        out = (((y as u32) << 16u32) | ((z as u32) << 8u32) | x as u32) as i32;
    }
    out as u32
}

/// Appends one BOMB explosion request (`radius = -1`).
#[cube]
fn push_bomb_request(queue: &mut [u32], count: &mut [Atomic<u32>], x: u32, y: u32, z: u32) {
    let slot = count[0].fetch_add(1u32);
    if slot < EXPLOSION_CAPACITY {
        let entry = (slot * EXPLOSION_STRIDE) as usize;
        queue[entry] = x;
        queue[entry + 1usize] = y;
        queue[entry + 2usize] = z;
        queue[entry + 3usize] = u32::reinterpret(-1.0f32);
    }
}

/// Reads the air temperature of the 4³ cell containing `(x, y, z)`.
#[cube]
fn air_at(air_temp: &[f32], x: i32, y: i32, z: i32) -> f32 {
    air_temp
        [((y as u32 >> 2u32) << 12u32 | (z as u32 >> 2u32) << 6u32 | (x as u32 >> 2u32)) as usize]
}

/// One conduction edge plus the neighbour type.
#[cube]
#[allow(clippy::too_many_arguments)]
fn neighbour_info(
    voxel_a: &[u32],
    table: &[u32],
    air_temp: &[f32],
    x: i32,
    y: i32,
    z: i32,
    self_temp: f32,
    self_conduct: f32,
    air_enabled: u32,
    ambient_k: f32,
    dx: i32,
    dy: i32,
    dz: i32,
) -> (f32, u32) {
    let nx = x + dx;
    let ny = y + dy;
    let nz = z + dz;
    let mut delta = seed_f32(self_temp, 0.0f32);
    let mut nty = seed_u32(x as u32, 0u32);
    if nx >= 0i32 && nx < WORLD && ny >= 0i32 && ny < WORLD && nz >= 0i32 && nz < WORLD {
        let nindex = ((ny as u32) << 16u32) | ((nz as u32) << 8u32) | nx as u32;
        let nword = voxel_a[nindex as usize];
        let mut ntemp = seed_f32(self_temp, ambient_k);
        let mut nconduct = seed_f32(self_temp, AIR_CONDUCT);
        if nword != 0u32 {
            nty = nword & C_TY_MASK;
            ntemp = ((nword >> C_TEMP_SHIFT) & C_TEMP_MASK) as f32 * 0.125f32;
            let base = (nty as usize) * C_ELEM_WORDS;
            nconduct = f32::reinterpret(table[base + C_ELEM_HEAT_CONDUCT]);
        } else if air_enabled != 0u32 {
            ntemp = air_at(air_temp, nx, ny, nz);
        }
        let conduct = minf(self_conduct, nconduct);
        delta = (conduct / (conduct + 1.0f32)) * (ntemp - self_temp);
    }
    (delta, nty)
}

/// `min` without `f32::min`.
#[cube]
fn minf(a: f32, b: f32) -> f32 {
    let mut out = a;
    if b < a {
        out = b;
    }
    out
}

/// Applies conduction and every ported element branch to one occupied voxel.
#[cube]
#[allow(clippy::too_many_arguments)]
fn react_voxel(
    voxel_a: &[u32],
    table: &[u32],
    air_temp: &[f32],
    pv: &mut [Atomic<i32>],
    explosion_queue: &mut [u32],
    explosion_count: &mut [Atomic<u32>],
    mutation_queue: &mut [u32],
    mutation_count: &mut [Atomic<u32>],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    flood_queue: &mut [u32],
    flood_state: &mut [Atomic<u32>],
    word: u32,
    x: u32,
    y: u32,
    z: u32,
    tick: u32,
    ambient_k: f32,
    air_enabled: u32,
) -> u32 {
    let ty = word & C_TY_MASK;
    let index = (y << 16u32) | (z << 8u32) | x;
    let base = (ty as usize) * C_ELEM_WORDS;
    let flags = table[base + C_ELEM_FLAGS];
    let props = table[base + C_ELEM_TPT_PROPS];
    let self_temp = ((word >> C_TEMP_SHIFT) & C_TEMP_MASK) as f32 * 0.125f32;
    let self_conduct = f32::reinterpret(table[base + C_ELEM_HEAT_CONDUCT]);
    let life_in = (word >> C_LIFE_SHIFT) & C_LIFE_MASK;

    let xi = x as i32;
    let yi = y as i32;
    let zi = z as i32;

    // Thermal conduction over the six face neighbours; the effective
    // conductivity is the smaller of the pair so the exchange is symmetric.
    let (d0, _t0) = neighbour_info(
        voxel_a,
        table,
        air_temp,
        xi,
        yi,
        zi,
        self_temp,
        self_conduct,
        air_enabled,
        ambient_k,
        1i32,
        0i32,
        0i32,
    );
    let (d1, _t1) = neighbour_info(
        voxel_a,
        table,
        air_temp,
        xi,
        yi,
        zi,
        self_temp,
        self_conduct,
        air_enabled,
        ambient_k,
        -1i32,
        0i32,
        0i32,
    );
    let (d2, _t2) = neighbour_info(
        voxel_a,
        table,
        air_temp,
        xi,
        yi,
        zi,
        self_temp,
        self_conduct,
        air_enabled,
        ambient_k,
        0i32,
        1i32,
        0i32,
    );
    let (d3, _t3) = neighbour_info(
        voxel_a,
        table,
        air_temp,
        xi,
        yi,
        zi,
        self_temp,
        self_conduct,
        air_enabled,
        ambient_k,
        0i32,
        -1i32,
        0i32,
    );
    let (d4, _t4) = neighbour_info(
        voxel_a,
        table,
        air_temp,
        xi,
        yi,
        zi,
        self_temp,
        self_conduct,
        air_enabled,
        ambient_k,
        0i32,
        0i32,
        1i32,
    );
    let (d5, _t5) = neighbour_info(
        voxel_a,
        table,
        air_temp,
        xi,
        yi,
        zi,
        self_temp,
        self_conduct,
        air_enabled,
        ambient_k,
        0i32,
        0i32,
        -1i32,
    );
    let temp = clampf(
        self_temp + CONDUCT_RATE * (d0 + d1 + d2 + d3 + d4 + d5),
        0.0f32,
        MAX_PACKED_TEMP_K,
    );

    // Self state. Temperature always comes from the conduction solve; the
    // element branches may replace type/life/temperature and auxiliary words.
    let mut out_ty = seed_u32(ty, ty);
    let mut out_temp = seed_f32(temp, temp);
    let mut out_life = seed_u32(life_in, life_in);
    let mut out_ctype = seed_u32(ty, aux_ctype[index as usize]);
    let mut out_tmp = seed_i32(x as i32, aux_tmp[index as usize]);
    let mut out_tmp2 = seed_i32(x as i32, aux_tmp2[index as usize]);
    let mut out_tmp3 = seed_i32(x as i32, aux_tmp3[index as usize]);

    // TPT roll chain: each draw consumes one hash step. This keeps every
    // chance/between call deterministic per voxel and tick.
    let mut rng = seed_u32(
        index,
        hash_u32(mix_index(index, RNG_SALT) ^ tick ^ (ty << 8u32)),
    );

    // 1. Generic pressure explosion, before TransitionPhase
    //    (`Simulation.cpp:2790-2796`). NITR is the only element with bit 2.
    let explosive = table[base + C_ELEM_EXPLOSIVE];
    let flammable = table[base + C_ELEM_FLAMMABLE];
    let mut transitioned = seed_u32(ty, 0u32);
    if ty == C_PT_NITR && (explosive & 2u32) != 0u32 && pv_at(pv, x, y, z) > 2.5f32 {
        out_ty = C_PT_FIRE;
        out_life = between(rng, 180u32, 259u32);
        rng = hash_u32(rng);
        out_temp = FIRE_DEFAULT_TEMP_K + (flammable as f32) * 0.5f32;
        out_tmp = 0i32;
        out_ctype = 0u32;
        pv_add(pv, x, y, z, 0.25f32);
        transitioned = 1u32;
    }

    // 2. TransitionPhase (`Simulation.cpp:2470-2745`). Every threshold and
    //    product comes from the GPU element table, which copies each C++
    //    element's `HighTemperature`/`LowTemperature` slot. The three
    //    transitions that consume `ctype` (`ST` for ICEI/SNOW/SLTW and the
    //    LAVA(ctype) freeze) are applied before the generic table path.
    let old_is_gas = (flags & (C_FLAG_GAS | C_FLAG_FIRE)) != 0u32;
    if transitioned == 0u32 {
        let mut target = seed_u32(ty, 0u32);
        let mut keep_ctype = seed_u32(ty, out_ctype);

        // ICEI/SNOW `HighTemperatureTransition == ST`
        // (`Simulation.cpp:2545-2569`). The `fix for ice with ctype = 0`
        // substitutes water, whose own 273.15 K freezing point then decides
        // whether the sublimation happens.
        if (ty == C_PT_ICEI || ty == C_PT_SNOW) && temp >= 252.05f32 {
            let mut ct = keep_ctype;
            if ct == 0u32 || ct >= 256u32 || ct == C_PT_ICEI || ct == C_PT_SNOW {
                ct = C_PT_WATR;
            }
            let ct_base = (ct as usize) * C_ELEM_WORDS;
            let ct_freeze = f32::reinterpret(table[ct_base + C_ELEM_FREEZE]);
            let ct_freeze_product = table[ct_base + C_ELEM_FREEZE_PRODUCT];
            let mut ok = seed_u32(ct, 1u32);
            if ct_freeze_product == C_PT_ICEI || ct_freeze_product == C_PT_SNOW {
                if temp < ct_freeze {
                    ok = 0u32;
                }
            } else if temp < 273.15f32 {
                ok = 0u32;
            }
            if ok != 0u32 {
                target = ct;
                keep_ctype = 0u32;
                out_life = 0u32;
            }
        }
        // SLTW `HighTemperatureTransition == ST`
        // (`Simulation.cpp:2555-2558`): salt or steam, one in four.
        if ty == C_PT_SLTW && temp >= 383.0f32 {
            target = if chance(rng, 1u32, 4u32) {
                C_PT_SALT
            } else {
                C_PT_WTRV
            };
            rng = hash_u32(rng);
        }
        // BRMT keeps its `ctype == TUNG` special case in `BRMT.cpp:41-42`
        // (`Simulation.cpp:2563-2583`); TUNG is not part of this wave, so the
        // general high-temperature arm applies.
        if ty == C_PT_BRMT && temp >= 1273.0f32 {
            target = C_PT_LAVA;
            keep_ctype = C_PT_BMTL;
        }

        if target == 0u32 && ty != C_PT_LAVA {
            let ignition = f32::reinterpret(table[base + C_ELEM_IGNITION]);
            let burn_product = table[base + C_ELEM_BURN_PRODUCT];
            let melt = f32::reinterpret(table[base + C_ELEM_MELT]);
            let melt_product = table[base + C_ELEM_MELT_PRODUCT];
            let boil = f32::reinterpret(table[base + C_ELEM_BOIL]);
            let boil_product = table[base + C_ELEM_BOIL_PRODUCT];
            let freeze = f32::reinterpret(table[base + C_ELEM_FREEZE]);
            let freeze_product = table[base + C_ELEM_FREEZE_PRODUCT];
            let mut high_ok = seed_u32(ty, 1u32);
            if ty == C_PT_ICEI || ty == C_PT_SNOW || ty == C_PT_SLTW || ty == C_PT_BRMT {
                high_ok = 0u32;
            }
            if high_ok != 0u32 {
                if ignition > 0.0f32 && burn_product != 0u32 && temp >= ignition {
                    target = burn_product;
                } else if melt > 0.0f32 && melt_product != 0u32 && temp >= melt {
                    target = melt_product;
                } else if boil > 0.0f32 && boil_product != 0u32 && temp >= boil {
                    target = boil_product;
                }
            }
            if target == 0u32 && freeze > 0.0f32 && freeze_product != 0u32 && temp < freeze {
                target = freeze_product;
            }
        }

        if target != 0u32 {
            // `Simulation.cpp:2680-2689`: a transition to ICEI/LAVA/SNOW
            // stores the old type in ctype and every transition resets life.
            if target == C_PT_LAVA {
                let mut ctype = ty;
                if ctype == C_PT_BRMT {
                    ctype = C_PT_BMTL;
                } else if ctype == C_PT_SAND {
                    ctype = C_PT_GLAS;
                }
                keep_ctype = ctype;
                // `Simulation.cpp:2726-2736`: LAVA life tracks its heat.
                out_life = clamp_i((temp - 700.0f32) / 7.0f32, 0i32, 400i32) as u32;
            } else {
                out_life = 0u32;
            }
            // `Simulation.cpp:2695-2706`: entering FIRE clears tmp.
            if target == C_PT_FIRE {
                out_tmp = 0i32;
            }
            out_ty = target;
            out_ctype = keep_ctype;
            transitioned = 1u32;
        }
    }

    // ST gas impulse: `if new is gas and old is not gas, pv += 0.5`
    // (`Simulation.cpp:2708-2709`).
    if transitioned != 0u32 && !old_is_gas {
        let new_base = (out_ty as usize) * C_ELEM_WORDS;
        let new_flags = table[new_base + C_ELEM_FLAGS];
        if (new_flags & (C_FLAG_GAS | C_FLAG_FIRE)) != 0u32 {
            pv_add(pv, x, y, z, 0.5f32);
        }
    }
    // A transition to FIRE/PLSM resets the life to `rng 120..169`
    // (`Simulation.cpp:2723-2724`).
    if transitioned != 0u32 {
        if out_ty == C_PT_FIRE || out_ty == C_PT_PLSM {
            out_life = between(rng, 120u32, 169u32);
            rng = hash_u32(rng);
        }
    }

    // 3. LAVA's LowTemperatureTransition == ST changes type based on ctype
    //    (`Simulation.cpp:2622-2660`). This only runs when the particle was
    //    already LAVA at the start of the tick, exactly like one
    //    TransitionPhase call.
    if ty == C_PT_LAVA {
        let ctype = out_ctype;
        let pt = out_temp;
        let mut freeze_to = seed_u32(ctype, 0u32);
        if ctype == C_PT_BMTL {
            // LAVA(BMTL) refreezes below BMTL's HighTemperature.
            if pt < 1273.0f32 {
                freeze_to = C_PT_BMTL;
            }
        } else if ctype == C_PT_THRM {
            // LAVA(THRM) stays molten above 973 K; below it the metal
            // freezes as BMTL (`Simulation.cpp:2648-2660`).
            if pt < 973.0f32 {
                freeze_to = C_PT_BMTL;
            }
        } else if ctype == 0u32 || ctype == C_PT_STNE {
            if pt < 973.0f32 {
                freeze_to = C_PT_STNE;
            }
        } else if pt < 973.0f32 {
            freeze_to = ctype;
        }
        if freeze_to != 0u32 {
            out_ty = freeze_to;
            out_ctype = 0u32;
            out_tmp = 0i32;
            out_life = 0u32;
        } else {
            // LAVA life and the THRM heat reservoir
            // (`Simulation.cpp:2734-2745`).
            out_life = clamp_i((pt - 700.0f32) / 7.0f32, 0i32, 400i32) as u32;
            if out_ctype == C_PT_THRM && out_tmp > 0i32 {
                out_tmp -= 1i32;
                out_temp = 3500.0f32;
            }
        }
    }

    // 4. Element Update functions. In TPT the Update of the *new* type runs in
    //    the same tick as a transition (`Simulation.cpp:2390-2394`).
    let mut update_ty = seed_u32(out_ty, out_ty);

    // --- WATR + SALT (`WATR.cpp:49-67`). --------------------------------
    // Water next to salt dissolves one in fifty ticks; the salt itself turns
    // to salt water one time in three (`part_change_type` keeps its
    // temperature, life, ctype and tmp).
    if update_ty == C_PT_WATR {
        let mut dissolved = seed_u32(index, 0u32);
        let mut rng_salt = seed_u32(index, rng);
        for dx in -1i32..=1i32 {
            for dy in -1i32..=1i32 {
                for dz in -1i32..=1i32 {
                    if (dx != 0i32 || dy != 0i32 || dz != 0i32) && dissolved == 0u32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nword = voxel_a[ni as usize];
                            if (nword & C_TY_MASK) == C_PT_SALT && chance(rng_salt, 1u32, 50u32) {
                                out_ty = C_PT_SLTW;
                                dissolved = 1u32;
                                rng_salt = hash_u32(rng_salt);
                                if chance(rng_salt, 1u32, 3u32) {
                                    let ntemp =
                                        ((nword >> C_TEMP_SHIFT) & C_TEMP_MASK) as f32 * 0.125f32;
                                    let nlife = (nword >> C_LIFE_SHIFT) & C_LIFE_MASK;
                                    push_mutation(
                                        mutation_queue,
                                        mutation_count,
                                        ni,
                                        MODE_REPLACE,
                                        pack_word(C_PT_SLTW, ntemp, nlife),
                                        aux_ctype[ni as usize],
                                        aux_tmp[ni as usize],
                                        aux_tmp2[ni as usize],
                                        aux_tmp3[ni as usize],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        update_ty = out_ty;
    }

    // --- FIRE / PLSM burnout (`FIRE.cpp:57-105`). ---------------------------
    if update_ty == C_PT_FIRE || update_ty == C_PT_PLSM {
        let mut burnt = seed_u32(update_ty, 0u32);
        if out_life <= 1u32 {
            if update_ty == C_PT_PLSM {
                if out_ctype == C_PT_NBLE {
                    // `FIRE.cpp:62-66`: PLSM(NBLE) -> NBLE. Reachable only
                    // through H2 fusion (blocked, see module docs).
                    out_ty = C_PT_NBLE;
                    out_life = 0u32;
                    burnt = 1u32;
                } else if (out_tmp & 3i32) == 3i32 {
                    // `FIRE.cpp:67-73`: PLSM + O2 + H2 -> WTRV(FIRE).
                    out_ty = C_PT_WTRV;
                    out_life = 0u32;
                    out_ctype = C_PT_FIRE;
                    burnt = 1u32;
                }
            } else {
                if (out_tmp & 3i32) == 3i32 {
                    // `FIRE.cpp:79-88`: FIRE + O2 + H2 -> WTRV(FIRE).
                    out_ty = C_PT_WTRV;
                    out_life = 0u32;
                    out_ctype = C_PT_FIRE;
                    burnt = 1u32;
                } else if out_temp < 625.0f32 {
                    // `FIRE.cpp:89-94`: FIRE -> SMKE with life 250..269.
                    out_ty = C_PT_SMKE;
                    out_life = between(rng, 250u32, 269u32);
                    rng = hash_u32(rng);
                    burnt = 1u32;
                }
            }
        }
        let _ = burnt;
        update_ty = out_ty;
    }

    // --- WOOD (`WOOD.cpp:50-62`). ------------------------------------------
    if update_ty == C_PT_WOOD {
        // Peak temperature tracking, then vacuum pyrolysis.
        if out_temp > 450.0f32 && out_temp > out_tmp as f32 {
            out_tmp = out_temp as i32;
        }
        if out_temp > 773.0f32 && pv_at(pv, x, y, z) <= -10.0f32 {
            // `create_part(i, x, y, PT_BCOL)` resets BCOL's defaults
            // (`life = 110`, `tmp = 0`) and the wood's temperature is put
            // back afterwards.
            out_ty = C_PT_BCOL;
            out_life = 110u32;
            out_ctype = 0u32;
            out_tmp = 0i32;
            out_tmp2 = 0i32;
            out_tmp3 = 0i32;
        }
        update_ty = out_ty;
    }

    // --- COAL / BCOL (`COAL.cpp:55-77`). -----------------------------------
    if update_ty == C_PT_COAL || update_ty == C_PT_BCOL {
        let mut coal_return = seed_u32(update_ty, 0u32);
        if out_life == 0u32 {
            // `create_part(i, x, y, PT_FIRE)`: DefaultProperties + Create.
            out_ty = C_PT_FIRE;
            out_temp = FIRE_DEFAULT_TEMP_K;
            out_life = between(rng, 120u32, 169u32);
            rng = hash_u32(rng);
            out_ctype = 0u32;
            out_tmp = 0i32;
            out_tmp2 = 0i32;
            out_tmp3 = 0i32;
            coal_return = 1u32;
        } else if out_life < 100u32 {
            out_life -= 1u32;
            // `create_part(-1, x + rng.between(-1,1), y + rng.between(-1,1),
            // PT_FIRE)`; the 3D lift adds the same roll on z.
            let ox = between(rng, 0u32, 2u32) as i32 - 1i32;
            rng = hash_u32(rng);
            let oy = between(rng, 0u32, 2u32) as i32 - 1i32;
            rng = hash_u32(rng);
            let oz = between(rng, 0u32, 2u32) as i32 - 1i32;
            rng = hash_u32(rng);
            let target = world_index(xi + ox, yi + oy, zi + oz);
            if target != C_NONE {
                let fire_life = between(rng, 120u32, 169u32);
                rng = hash_u32(rng);
                push_mutation(
                    mutation_queue,
                    mutation_count,
                    target,
                    MODE_CREATE,
                    pack_word(C_PT_FIRE, FIRE_DEFAULT_TEMP_K, fire_life),
                    0u32,
                    0i32,
                    0i32,
                    0i32,
                );
            }
        }
        if coal_return == 0u32 && update_ty == C_PT_COAL {
            if pv_at(pv, x, y, z) > 4.3f32 && out_tmp > 40i32 {
                out_tmp = 39i32;
            } else if out_tmp < 40i32 && out_tmp > 0i32 {
                out_tmp -= 1i32;
            } else if out_tmp <= 0i32 {
                // `part_change_type(i, PT_BCOL)` keeps temperature, life,
                // tmp, tmp2 and ctype.
                out_ty = C_PT_BCOL;
                coal_return = 1u32;
            }
        }
        if coal_return == 0u32 && out_temp > out_tmp2 as f32 {
            out_tmp2 = out_temp as i32;
        }
        update_ty = out_ty;
    }

    // --- EMBR (`EMBR.cpp:56-77`). ------------------------------------------
    if update_ty == C_PT_EMBR {
        let mut kill = seed_u32(index, 0u32);
        for dx in -1i32..=1i32 {
            for dy in -1i32..=1i32 {
                for dz in -1i32..=1i32 {
                    if (dx != 0i32 || dy != 0i32 || dz != 0i32) && kill == 0u32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nword = voxel_a[ni as usize];
                            let nty = nword & C_TY_MASK;
                            if nty != 0u32 {
                                let nb = (nty as usize) * C_ELEM_WORDS;
                                let nflags = table[nb + C_ELEM_FLAGS];
                                let nprops = table[nb + C_ELEM_TPT_PROPS];
                                if (nflags & (C_FLAG_STATIC | C_FLAG_POWDER | C_FLAG_LIQUID))
                                    != 0u32
                                    && (nprops & C_TPT_SPARKSETTLE) == 0u32
                                {
                                    kill = 1u32;
                                }
                            }
                        }
                    }
                }
            }
        }
        if kill != 0u32 {
            out_ty = 0u32;
            out_life = 0u32;
        }
        update_ty = out_ty;
    }

    // --- BOMB (`BOMB.cpp:58-126`). -----------------------------------------
    if update_ty == C_PT_BOMB {
        let mut trigger = seed_u32(index, 0u32);
        for dx in -1i32..=1i32 {
            for dy in -1i32..=1i32 {
                for dz in -1i32..=1i32 {
                    if (dx != 0i32 || dy != 0i32 || dz != 0i32) && trigger == 0u32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nty = voxel_a[ni as usize] & C_TY_MASK;
                            if nty != 0u32 && nty != C_PT_BOMB && nty != C_PT_EMBR {
                                // The remaining protected types (DMND, CLNE,
                                // PCLN, BCLN, VIBR) do not exist in this build.
                                trigger = 1u32;
                            }
                        }
                    }
                }
            }
        }
        if trigger != 0u32 {
            // `kill_part(i)` then the radius-8 fill and radius-9 shell.
            out_ty = 0u32;
            out_life = 0u32;
            push_bomb_request(explosion_queue, explosion_count, x, y, z);
        }
        update_ty = out_ty;
    }

    // --- IGNT (`IGNT.cpp:48-92`). ------------------------------------------
    if update_ty == C_PT_IGNT {
        if out_tmp == 0i32 {
            let mut latch = seed_u32(index, 0u32);
            for dx in -1i32..=1i32 {
                for dy in -1i32..=1i32 {
                    for dz in -1i32..=1i32 {
                        if (dx != 0i32 || dy != 0i32 || dz != 0i32) && latch == 0u32 {
                            let ni = world_index(xi + dx, yi + dy, zi + dz);
                            if ni != C_NONE {
                                let nty = voxel_a[ni as usize] & C_TY_MASK;
                                let nlife = (voxel_a[ni as usize] >> C_LIFE_SHIFT) & C_LIFE_MASK;
                                if nty == C_PT_FIRE
                                    || nty == C_PT_PLSM
                                    || nty == C_PT_SPRK
                                    || nty == C_PT_LIGH
                                    || (nty == C_PT_IGNT && nlife == 1u32)
                                {
                                    latch = 1u32;
                                }
                            }
                        }
                    }
                }
            }
            if latch != 0u32 {
                out_tmp = 1i32;
            }
        } else if out_life > 0u32 {
            if chance(rng, 2u32, 3u32) {
                rng = hash_u32(rng);
                let ox = between(rng, 0u32, 2u32) as i32 - 1i32;
                rng = hash_u32(rng);
                let oy = between(rng, 0u32, 2u32) as i32 - 1i32;
                rng = hash_u32(rng);
                let oz = between(rng, 0u32, 2u32) as i32 - 1i32;
                rng = hash_u32(rng);
                let target = world_index(xi + ox, yi + oy, zi + oz);
                if target != C_NONE {
                    let mut ember_temp = out_temp - 273.15f32 + 400.0f32;
                    ember_temp = clampf(ember_temp, 0.0f32, TPT_MAX_TEMP_K);
                    push_mutation(
                        mutation_queue,
                        mutation_count,
                        target,
                        MODE_CREATE,
                        pack_word(C_PT_EMBR, ember_temp, 30u32),
                        0u32,
                        0i32,
                        0i32,
                        0i32,
                    );
                }
            } else {
                rng = hash_u32(rng);
                let ox = between(rng, 0u32, 2u32) as i32 - 1i32;
                rng = hash_u32(rng);
                let oy = between(rng, 0u32, 2u32) as i32 - 1i32;
                rng = hash_u32(rng);
                let oz = between(rng, 0u32, 2u32) as i32 - 1i32;
                rng = hash_u32(rng);
                let target = world_index(xi + ox, yi + oy, zi + oz);
                if target != C_NONE {
                    let fire_life = between(rng, 120u32, 169u32);
                    rng = hash_u32(rng);
                    push_mutation(
                        mutation_queue,
                        mutation_count,
                        target,
                        MODE_CREATE,
                        pack_word(C_PT_FIRE, FIRE_DEFAULT_TEMP_K, fire_life),
                        0u32,
                        0i32,
                        0i32,
                        0i32,
                    );
                }
            }
            out_life -= 1u32;
        }
    }

    // --- FUSE (`FUSE.cpp:59-91`). ------------------------------------------
    if update_ty == C_PT_FUSE {
        if out_life == 0u32 {
            // `create_part(i, PT_PLSM)` then life = 50.
            out_ty = C_PT_PLSM;
            out_temp = TPT_MAX_TEMP_K;
            out_life = 50u32;
            out_ctype = 0u32;
            out_tmp = 0i32;
            out_tmp2 = 0i32;
            out_tmp3 = 0i32;
            update_ty = out_ty;
        } else {
            if out_life < 40u32 {
                out_life -= 1u32;
                if chance(rng, 1u32, 100u32) {
                    rng = hash_u32(rng);
                    let ox = between(rng, 0u32, 2u32) as i32 - 1i32;
                    rng = hash_u32(rng);
                    let oy = between(rng, 0u32, 2u32) as i32 - 1i32;
                    rng = hash_u32(rng);
                    let oz = between(rng, 0u32, 2u32) as i32 - 1i32;
                    rng = hash_u32(rng);
                    let target = world_index(xi + ox, yi + oy, zi + oz);
                    if target != C_NONE {
                        push_mutation(
                            mutation_queue,
                            mutation_count,
                            target,
                            MODE_CREATE,
                            pack_word(C_PT_PLSM, TPT_MAX_TEMP_K, 50u32),
                            0u32,
                            0i32,
                            0i32,
                            0i32,
                        );
                    }
                }
            }
            if pv_at(pv, x, y, z) > 2.7f32 && out_tmp > 40i32 {
                out_tmp = 39i32;
            } else if out_tmp <= 0i32 {
                // `create_part(i, PT_FSEP)` resets to FSEP defaults: temp
                // 295.15 K, life 50, tmp 0.
                out_ty = C_PT_FSEP;
                out_temp = 295.15f32;
                out_life = 50u32;
                out_ctype = 0u32;
                out_tmp = 0i32;
                out_tmp2 = 0i32;
                out_tmp3 = 0i32;
                update_ty = out_ty;
            } else if out_tmp < 40i32 {
                out_tmp -= 1i32;
            }
            // `FUSE.cpp:78-94`: a spark neighbour, or any neighbour while the
            // fuse is above 973.15 K, ignites it with a 1/20 roll.
            if out_life > 40u32 {
                let mut ignite = seed_u32(index, 0u32);
                for dx in -2i32..=2i32 {
                    for dy in -2i32..=2i32 {
                        for dz in -2i32..=2i32 {
                            if (dx != 0i32 || dy != 0i32 || dz != 0i32) && ignite == 0u32 {
                                let ni = world_index(xi + dx, yi + dy, zi + dz);
                                if ni != C_NONE {
                                    let nty = voxel_a[ni as usize] & C_TY_MASK;
                                    if nty == C_PT_SPRK
                                        || (out_temp >= 973.15f32
                                            && chance(
                                                hash_u32(mix_index(ni, NEIGHBOUR_SALT) ^ rng),
                                                1u32,
                                                20u32,
                                            ))
                                    {
                                        ignite = 1u32;
                                    }
                                }
                            }
                        }
                    }
                }
                if ignite != 0u32 {
                    out_life = 39u32;
                }
            }
        }
    }

    // --- FSEP (`FSEP.cpp:53-81`). ------------------------------------------
    if update_ty == C_PT_FSEP {
        if out_life == 0u32 {
            out_ty = C_PT_PLSM;
            out_temp = TPT_MAX_TEMP_K;
            out_life = 50u32;
            out_ctype = 0u32;
            out_tmp = 0i32;
            out_tmp2 = 0i32;
            out_tmp3 = 0i32;
            update_ty = out_ty;
        } else if out_life < 40u32 {
            out_life -= 1u32;
            if chance(rng, 1u32, 10u32) {
                rng = hash_u32(rng);
                let ox = between(rng, 0u32, 2u32) as i32 - 1i32;
                rng = hash_u32(rng);
                let oy = between(rng, 0u32, 2u32) as i32 - 1i32;
                rng = hash_u32(rng);
                let oz = between(rng, 0u32, 2u32) as i32 - 1i32;
                rng = hash_u32(rng);
                let target = world_index(xi + ox, yi + oy, zi + oz);
                if target != C_NONE {
                    push_mutation(
                        mutation_queue,
                        mutation_count,
                        target,
                        MODE_CREATE,
                        pack_word(C_PT_PLSM, TPT_MAX_TEMP_K, 50u32),
                        0u32,
                        0i32,
                        0i32,
                        0i32,
                    );
                }
            }
        } else {
            // `FSEP.cpp:67-84`: a spark neighbour, or any neighbour above
            // 673.15 K, sets the fuse off with a 1/15 roll.
            let mut ignite = seed_u32(index, 0u32);
            for dx in -2i32..=2i32 {
                for dy in -2i32..=2i32 {
                    for dz in -2i32..=2i32 {
                        if (dx != 0i32 || dy != 0i32 || dz != 0i32)
                            && ignite == 0u32
                            && out_life > 40u32
                        {
                            let ni = world_index(xi + dx, yi + dy, zi + dz);
                            if ni != C_NONE {
                                let nty = voxel_a[ni as usize] & C_TY_MASK;
                                if nty == C_PT_SPRK
                                    || (out_temp >= 673.15f32
                                        && chance(
                                            hash_u32(mix_index(ni, NEIGHBOUR_SALT) ^ rng),
                                            1u32,
                                            15u32,
                                        ))
                                {
                                    ignite = 1u32;
                                }
                            }
                        }
                    }
                }
            }
            if ignite != 0u32 {
                out_life = 39u32;
            }
        }
    }

    // --- BRMT (`BRMT.cpp:49-77`). ------------------------------------------
    if update_ty == C_PT_BRMT {
        if out_temp > 523.15f32 {
            let mut temp_factor = 1000i32 - ((523.15f32 - out_temp) * 2.0f32) as i32;
            if temp_factor < 2i32 {
                temp_factor = 2i32;
            }
            for dx in -1i32..=1i32 {
                for dy in -1i32..=1i32 {
                    for dz in -1i32..=1i32 {
                        if dx != 0i32 || dy != 0i32 || dz != 0i32 {
                            let ni = world_index(xi + dx, yi + dy, zi + dz);
                            if ni != C_NONE {
                                let nty = voxel_a[ni as usize] & C_TY_MASK;
                                if nty == C_PT_BREC
                                    && chance(
                                        hash_u32(
                                            mix_index(ni, NEIGHBOUR_SALT)
                                                ^ rng
                                                ^ (temp_factor as u32),
                                        ),
                                        1u32,
                                        temp_factor as u32,
                                    )
                                {
                                    if chance(hash_u32(mix_index(ni, 0x27D4u32) ^ rng), 1u32, 2u32)
                                    {
                                        // `create_part(ID(r), PT_THRM)` resets
                                        // to THRM defaults.
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(C_PT_THRM, 295.15f32, 0u32),
                                            0u32,
                                            0i32,
                                            0i32,
                                            0i32,
                                        );
                                    } else {
                                        // `create_part(i, PT_THRM)`.
                                        out_ty = C_PT_THRM;
                                        out_temp = 295.15f32;
                                        out_life = 0u32;
                                        out_ctype = 0u32;
                                        out_tmp = 0i32;
                                        out_tmp2 = 0i32;
                                        out_tmp3 = 0i32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        update_ty = out_ty;
    }

    // --- BREC (`BREC.cpp:49-75`). ------------------------------------------
    if update_ty == C_PT_BREC {
        if out_life != 0u32 {
            if pv_at(pv, x, y, z) > 10.0f32 {
                let pressure = pv_at(pv, x, y, z);
                if out_temp > 9000.0f32 && pressure > 30.0f32 && chance(rng, 1u32, 200u32) {
                    // BREC -> EXOT. EXOT has no row in this wave and no
                    // creation path gives BREC a non-zero life (default 0),
                    // so this branch is unreachable from Create; it is ported
                    // verbatim and is not stubbed.
                    out_ty = C_PT_EXOT;
                    out_life = 1000u32;
                } else {
                    out_temp += pressure / 8.0f32;
                }
            }
        }
        update_ty = out_ty;
    }

    // --- BANG / TNT four-stage machine (`BANG.cpp:49-127`). ----------------
    if update_ty == C_PT_BANG {
        if out_tmp == 0i32 {
            let mut trigger = seed_u32(index, 0u32);
            if out_temp >= 673.0f32 {
                trigger = 1u32;
            } else {
                for dx in -1i32..=1i32 {
                    for dy in -1i32..=1i32 {
                        for dz in -1i32..=1i32 {
                            if (dx != 0i32 || dy != 0i32 || dz != 0i32) && trigger == 0u32 {
                                let ni = world_index(xi + dx, yi + dy, zi + dz);
                                if ni != C_NONE {
                                    let nty = voxel_a[ni as usize] & C_TY_MASK;
                                    if nty == C_PT_FIRE
                                        || nty == C_PT_PLSM
                                        || nty == C_PT_SPRK
                                        || nty == C_PT_LIGH
                                    {
                                        trigger = 1u32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if trigger != 0u32 {
                out_tmp = 1i32;
                // Stamp the trigger tick so the flood pass processes the
                // stage-1 seed on the *next* tick, matching TPT's Update
                // ordering (stage 1 floods to 2 in the following update).
                out_tmp2 = tick as i32;
                // Append to the persistent seed list; the serial flood pass
                // compacts it and walks the connected component next tick.
                let slot = flood_state[1].fetch_add(1u32);
                if slot < 16_777_216u32 {
                    flood_queue[slot as usize] = index;
                }
            }
        } else if out_tmp == 1i32 {
            // The flood pass ([`super::flood`]) owns the 1 -> 2 transition.
        } else if out_tmp == 2i32 {
            out_tmp = 3i32;
        } else {
            // Stage 3: explode (`BANG.cpp:90-125`).
            let otemp = out_temp - 273.15f32;
            pv_add(pv, x, y, z, 0.5f32);
            out_tmp = 0i32;
            if chance(rng, 1u32, 3u32) {
                rng = hash_u32(rng);
                if chance(rng, 1u32, 2u32) {
                    rng = hash_u32(rng);
                    // `create_part(i, PT_FIRE)` then temp override.
                    out_ty = C_PT_FIRE;
                    out_life = between(rng, 120u32, 169u32);
                    rng = hash_u32(rng);
                    out_temp = clampf(TPT_MAX_TEMP_K / 4.0f32 + otemp, 0.0f32, TPT_MAX_TEMP_K);
                    out_ctype = 0u32;
                } else {
                    rng = hash_u32(rng);
                    // `create_part(i, PT_SMKE); life = rng 500..549`.
                    out_ty = C_PT_SMKE;
                    out_life = between(rng, 500u32, 549u32);
                    rng = hash_u32(rng);
                    out_temp = clampf(TPT_MAX_TEMP_K / 4.0f32 + otemp, 0.0f32, TPT_MAX_TEMP_K);
                    out_ctype = 0u32;
                }
                out_tmp2 = 0i32;
                out_tmp3 = 0i32;
            } else {
                rng = hash_u32(rng);
                if chance(rng, 1u32, 15u32) {
                    rng = hash_u32(rng);
                    // `create_part(i, PT_EMBR)` with tmp 0, life 50 and the
                    // MAX_TEMP/3 + otemp override.
                    out_ty = C_PT_EMBR;
                    out_ctype = 0u32;
                    out_tmp = 0i32;
                    out_tmp2 = 0i32;
                    out_tmp3 = 0i32;
                    out_life = 50u32;
                    out_temp = clampf(TPT_MAX_TEMP_K / 3.0f32 + otemp, 0.0f32, TPT_MAX_TEMP_K);
                } else {
                    out_ty = 0u32;
                    out_life = 0u32;
                }
            }
        }
        update_ty = out_ty;
    }

    // --- H2 fire reaction (`H2.cpp:63-88`). --------------------------------
    // The `pv > 45` branch skips the fire reactions; the fusion branch below
    // pv > 50 is blocked on NBLE/NEUT/PHOT/ELEC and is reported, not stubbed.
    if update_ty == C_PT_H2 {
        let source_pv = pv_at(pv, x, y, z);
        let mut hydrogen_burned = seed_u32(index, 0u32);
        for dx in -2i32..=2i32 {
            for dy in -2i32..=2i32 {
                for dz in -2i32..=2i32 {
                    if (dx != 0i32 || dy != 0i32 || dz != 0i32) && hydrogen_burned == 0u32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nword = voxel_a[ni as usize];
                            let nty = nword & C_TY_MASK;
                            if source_pv <= 45.0f32 {
                                if nty == C_PT_FIRE {
                                    // FIRE temp 2473.15, or 3473 when its tmp
                                    // already has bit 1; FIRE tmp |= 1.
                                    let ntmp = aux_tmp[ni as usize];
                                    let new_temp = if (ntmp & 2i32) != 0i32 {
                                        3473.0f32
                                    } else {
                                        2473.15f32
                                    };
                                    let nlife = (nword >> C_LIFE_SHIFT) & C_LIFE_MASK;
                                    push_mutation(
                                        mutation_queue,
                                        mutation_count,
                                        ni,
                                        MODE_REPLACE,
                                        pack_word(C_PT_FIRE, new_temp, nlife),
                                        aux_ctype[ni as usize],
                                        ntmp | 1i32,
                                        aux_tmp2[ni as usize],
                                        aux_tmp3[ni as usize],
                                    );
                                    // `create_part(i, PT_FIRE)` + rng 0..99.
                                    out_ty = C_PT_FIRE;
                                    out_temp =
                                        FIRE_DEFAULT_TEMP_K + between(rng, 0u32, 99u32) as f32;
                                    rng = hash_u32(rng);
                                    out_life = between(rng, 120u32, 169u32);
                                    rng = hash_u32(rng);
                                    out_ctype = 0u32;
                                    out_tmp = 1i32;
                                    out_tmp2 = 0i32;
                                    out_tmp3 = 0i32;
                                    hydrogen_burned = 1u32;
                                } else if nty == C_PT_PLSM || nty == C_PT_LAVA {
                                    let ntmp = aux_tmp[ni as usize];
                                    let fire_ok = if nty == C_PT_PLSM {
                                        (ntmp & 4i32) == 0i32
                                    } else {
                                        aux_ctype[ni as usize] != C_PT_BMTL
                                    };
                                    if fire_ok {
                                        out_ty = C_PT_FIRE;
                                        out_temp =
                                            FIRE_DEFAULT_TEMP_K + between(rng, 0u32, 99u32) as f32;
                                        rng = hash_u32(rng);
                                        out_life = between(rng, 120u32, 169u32);
                                        rng = hash_u32(rng);
                                        out_ctype = 0u32;
                                        out_tmp = 1i32;
                                        out_tmp2 = 0i32;
                                        out_tmp3 = 0i32;
                                        hydrogen_burned = 1u32;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        update_ty = out_ty;
    }

    // --- O2 fire reaction (`O2.cpp:51-76`). --------------------------------
    if update_ty == C_PT_O2 {
        // The C++ loop keeps scanning after `create_part(i, PT_FIRE)` has
        // already replaced this particle; every later FIRE hit only re-runs the
        // same self conversion and adds one more temperature roll to the same
        // cell, so the 3D pass stops after the first conversion.
        let mut converted = seed_u32(index, 0u32);
        for dx in -2i32..=2i32 {
            for dy in -2i32..=2i32 {
                for dz in -2i32..=2i32 {
                    if (dx != 0i32 || dy != 0i32 || dz != 0i32) && converted == 0u32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nword = voxel_a[ni as usize];
                            let nty = nword & C_TY_MASK;
                            if nty == C_PT_FIRE {
                                let ntmp = aux_tmp[ni as usize];
                                let nlife = (nword >> C_LIFE_SHIFT) & C_LIFE_MASK;
                                let mut ntemp = ((nword >> C_TEMP_SHIFT) & C_TEMP_MASK) as f32
                                    * 0.125f32
                                    + between(rng, 0u32, 99u32) as f32;
                                rng = hash_u32(rng);
                                if (ntmp & 1i32) != 0i32 {
                                    ntemp = 3473.0f32;
                                }
                                push_mutation(
                                    mutation_queue,
                                    mutation_count,
                                    ni,
                                    MODE_REPLACE,
                                    pack_word(C_PT_FIRE, ntemp, nlife),
                                    aux_ctype[ni as usize],
                                    ntmp | 2i32,
                                    aux_tmp2[ni as usize],
                                    aux_tmp3[ni as usize],
                                );
                                // `create_part(i, PT_FIRE)` + rng 0..99, then
                                // `parts[i].tmp |= 2`.
                                out_ty = C_PT_FIRE;
                                out_temp = FIRE_DEFAULT_TEMP_K + between(rng, 0u32, 99u32) as f32;
                                rng = hash_u32(rng);
                                out_life = between(rng, 120u32, 169u32);
                                rng = hash_u32(rng);
                                out_ctype = 0u32;
                                out_tmp = 2i32;
                                out_tmp2 = 0i32;
                                out_tmp3 = 0i32;
                                converted = 1u32;
                            } else if nty == C_PT_PLSM {
                                let ntmp = aux_tmp[ni as usize];
                                if (ntmp & 4i32) == 0i32 {
                                    out_ty = C_PT_FIRE;
                                    out_temp =
                                        FIRE_DEFAULT_TEMP_K + between(rng, 0u32, 99u32) as f32;
                                    rng = hash_u32(rng);
                                    out_life = between(rng, 120u32, 169u32);
                                    rng = hash_u32(rng);
                                    out_ctype = 0u32;
                                    out_tmp = 2i32;
                                    out_tmp2 = 0i32;
                                    out_tmp3 = 0i32;
                                    converted = 1u32;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // O2's high-temperature branch (`O2.cpp:80-123`) guards on
    // `temp > 9973.15`, above the packed word's 8191.875 K maximum, so it is
    // unreachable in the 3D representation.

    // 5. FIRE/PLSM/LAVA's 5x5x5 neighbour loop (`FIRE.cpp:153-325`). The
    //    original `t` is used even when the burnout branch just changed the
    //    type, so the loop runs for the pre-update flame type.
    let mut flame_kind = seed_u32(ty, 0u32);
    if ty == C_PT_FIRE || ty == C_PT_PLSM || ty == C_PT_LAVA {
        flame_kind = ty;
    } else if out_ty == C_PT_FIRE || out_ty == C_PT_PLSM || out_ty == C_PT_LAVA {
        flame_kind = out_ty;
    }
    if flame_kind != 0u32 {
        // `surround_space` counts empty cells in the 3x3x3 neighbourhood
        // (`Simulation.cpp:2275-2285`).
        let mut surround_space = seed_u32(index, 0u32);
        for dx in -1i32..=1i32 {
            for dy in -1i32..=1i32 {
                for dz in -1i32..=1i32 {
                    if dx != 0i32 || dy != 0i32 || dz != 0i32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE && (voxel_a[ni as usize] & C_TY_MASK) == 0u32 {
                            surround_space += 1u32;
                        }
                    }
                }
            }
        }
        for dx in -2i32..=2i32 {
            for dy in -2i32..=2i32 {
                for dz in -2i32..=2i32 {
                    if dx != 0i32 || dy != 0i32 || dz != 0i32 {
                        let ni = world_index(xi + dx, yi + dy, zi + dz);
                        if ni != C_NONE {
                            let nword = voxel_a[ni as usize];
                            let nty = nword & C_TY_MASK;
                            if nty != 0u32 {
                                let nb = (nty as usize) * C_ELEM_WORDS;
                                let nprops = table[nb + C_ELEM_TPT_PROPS];
                                let nexplosive = table[nb + C_ELEM_EXPLOSIVE];
                                let nflammable = table[nb + C_ELEM_FLAMMABLE];
                                let nlife = (nword >> C_LIFE_SHIFT) & C_LIFE_MASK;
                                let ntemp =
                                    ((nword >> C_TEMP_SHIFT) & C_TEMP_MASK) as f32 * 0.125f32;
                                let nctype = aux_ctype[ni as usize];
                                let ntmp = aux_tmp[ni as usize];
                                let ntmp2 = aux_tmp2[ni as usize];
                                let ntmp3 = aux_tmp3[ni as usize];
                                let nseed = hash_u32(mix_index(ni, NEIGHBOUR_SALT) ^ rng ^ tick);

                                if nty == C_PT_THRM {
                                    // `FIRE.cpp:162-178`: THRM -> LAVA.
                                    let target_temp = 3500.0f32;
                                    if chance(nseed, 1u32, 500u32) {
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(C_PT_LAVA, target_temp, 0u32),
                                            C_PT_BMTL,
                                            0i32,
                                            ntmp2,
                                            ntmp3,
                                        );
                                        pv_add(
                                            pv,
                                            ni & 255u32,
                                            (ni >> 16u32) & 255u32,
                                            (ni >> 8u32) & 255u32,
                                            50.0f32,
                                        );
                                    } else {
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(C_PT_LAVA, target_temp, 400u32),
                                            C_PT_THRM,
                                            20i32,
                                            ntmp2,
                                            ntmp3,
                                        );
                                    }
                                } else if nty == C_PT_COAL || nty == C_PT_BCOL {
                                    // `FIRE.cpp:180-200`.
                                    if flame_kind == C_PT_FIRE || flame_kind == C_PT_PLSM {
                                        if nlife > 100u32 && chance(nseed, 1u32, 500u32) {
                                            push_mutation(
                                                mutation_queue,
                                                mutation_count,
                                                ni,
                                                MODE_REPLACE,
                                                pack_word(
                                                    nty,
                                                    nword as f32 * 0.0f32
                                                        + ((nword >> C_TEMP_SHIFT) & C_TEMP_MASK)
                                                            as f32
                                                            * 0.125f32,
                                                    99u32,
                                                ),
                                                nctype,
                                                ntmp,
                                                ntmp2,
                                                ntmp3,
                                            );
                                        }
                                    }
                                    // The LAVA(IRON)/LAVA(STNE) branches need
                                    // IRON/SLCN, which are outside this wave.
                                } else if nty == C_PT_ICEI || nty == C_PT_SNOW {
                                    // `FIRE.cpp:370-383`: ice/snow melts; fire
                                    // is extinguished, lava freezes to stone.
                                    push_mutation(
                                        mutation_queue,
                                        mutation_count,
                                        ni,
                                        MODE_REPLACE,
                                        pack_word(C_PT_WATR, ntemp, nlife),
                                        nctype,
                                        ntmp,
                                        ntmp2,
                                        ntmp3,
                                    );
                                    if flame_kind == C_PT_FIRE {
                                        out_ty = 0u32;
                                        out_life = 0u32;
                                    } else if flame_kind == C_PT_LAVA {
                                        out_ty = C_PT_STNE;
                                        out_life = 0u32;
                                        out_ctype = 0u32;
                                    }
                                } else if (nty == C_PT_WATR || nty == C_PT_DSTW || nty == C_PT_SLTW)
                                    && ntemp < 373.0f32
                                {
                                    // `FIRE.cpp:384-398`: water is destroyed;
                                    // fire is extinguished; lava freezes.
                                    push_mutation(
                                        mutation_queue,
                                        mutation_count,
                                        ni,
                                        MODE_KILL,
                                        0u32,
                                        0u32,
                                        0i32,
                                        0i32,
                                        0i32,
                                    );
                                    if flame_kind == C_PT_FIRE {
                                        out_ty = 0u32;
                                        out_life = 0u32;
                                    } else if flame_kind == C_PT_LAVA {
                                        out_ty = C_PT_STNE;
                                        out_life = 0u32;
                                        out_ctype = 0u32;
                                    }
                                } else {
                                    // Flammable ignition (`FIRE.cpp:304-321`).
                                    let target_pv = pv_at(
                                        pv,
                                        ni & 255u32,
                                        (ni >> 16u32) & 255u32,
                                        (ni >> 8u32) & 255u32,
                                    );
                                    let mut roll = (nflammable as f32 + target_pv * 10.0f32) as u32;
                                    if roll > 1000u32 {
                                        // `chance` with a numerator > denominator
                                        // always fires, matching the C++ `int`
                                        // cast and `rng.chance(int, 1000)`.
                                        roll = 1000u32;
                                    }
                                    if (surround_space > 0u32 || nexplosive != 0u32)
                                        && nflammable > 0u32
                                        && chance(nseed, roll, 1000u32)
                                    {
                                        // `part_change_type`, then temp, life,
                                        // tmp and ctype overrides.
                                        let fire_temp =
                                            FIRE_DEFAULT_TEMP_K + (nflammable as f32) * 0.5f32;
                                        let fire_life = between(nseed, 180u32, 259u32);
                                        push_mutation(
                                            mutation_queue,
                                            mutation_count,
                                            ni,
                                            MODE_REPLACE,
                                            pack_word(C_PT_FIRE, fire_temp, fire_life),
                                            0u32,
                                            0i32,
                                            ntmp2,
                                            ntmp3,
                                        );
                                        if nexplosive != 0u32 {
                                            pv_add(pv, x, y, z, 0.25f32);
                                        }
                                    }
                                }
                                if (nprops & C_TPT_CARRIES_CTYPE) != 0u32 {
                                    // Kept for symmetry with `CarriesTypeIn`;
                                    // no branch in this loop depends on it.
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 6. Write the resulting word and this cell's auxiliary words.
    if out_ty != 0u32 {
        if (props & C_TPT_LIFE_DEC) == 0u32
            && (props & C_TPT_LIFE_KILL) == 0u32
            && (props & C_TPT_LIFE_KILL_DEC) == 0u32
        {
            let _ = life_in;
        }
        aux_ctype[index as usize] = out_ctype;
        aux_tmp[index as usize] = out_tmp;
        aux_tmp2[index as usize] = out_tmp2;
        aux_tmp3[index as usize] = out_tmp3;
        pack_word(out_ty, out_temp, out_life)
    } else {
        aux_ctype[index as usize] = 0u32;
        aux_tmp[index as usize] = 0i32;
        aux_tmp2[index as usize] = 0i32;
        aux_tmp3[index as usize] = 0i32;
        0u32
    }
}

/// Integer clamp without `i32::clamp`.
#[cube]
fn clamp_i(value: f32, lo: i32, hi: i32) -> i32 {
    let mut out = value as i32;
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

/// One thread per voxel of every simulated brick.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn thermal_kernel(
    voxel_a: &[u32],
    voxel_b: &mut [u32],
    sim_list: &[u32],
    table: &[u32],
    air_temp: &[f32],
    pv: &mut [Atomic<i32>],
    explosion_queue: &mut [u32],
    explosion_count: &mut [Atomic<u32>],
    mutation_queue: &mut [u32],
    mutation_count: &mut [Atomic<u32>],
    aux_ctype: &mut [u32],
    aux_tmp: &mut [i32],
    aux_tmp2: &mut [i32],
    aux_tmp3: &mut [i32],
    flood_queue: &mut [u32],
    flood_state: &mut [Atomic<u32>],
    tick: u32,
    ambient_k: f32,
    air_enabled: u32,
) {
    let brick = sim_list[CUBE_POS_X as usize];
    let bx = brick & 31u32;
    let bz = (brick >> 5u32) & 31u32;
    let by = (brick >> 10u32) & 31u32;
    let x = bx * (BRICK as u32) + UNIT_POS_X;
    let y = by * (BRICK as u32) + UNIT_POS_Y;
    let z = bz * (BRICK as u32) + UNIT_POS_Z;
    let index = (y << 16u32) | (z << 8u32) | x;
    let word = voxel_a[index as usize];
    let mut out = seed_u32(word, 0u32);
    if word != 0u32 {
        out = react_voxel(
            voxel_a,
            table,
            air_temp,
            pv,
            explosion_queue,
            explosion_count,
            mutation_queue,
            mutation_count,
            aux_ctype,
            aux_tmp,
            aux_tmp2,
            aux_tmp3,
            flood_queue,
            flood_state,
            word,
            x,
            y,
            z,
            tick,
            ambient_k,
            air_enabled,
        );
    }
    voxel_b[index as usize] = out;
}

/// Runs the thermal/reaction pass for every brick in `sim_list`.
pub(crate) fn thermal(engine: &Voxel3dEngine) {
    let bufs: &Voxel3dBuffers = &engine.bufs;
    unsafe {
        thermal_kernel::launch_unchecked(
            &engine.client,
            CubeCount::Dynamic(bufs.sim_count.clone().binding()),
            CubeDim::new_3d(CUBE_DIM, CUBE_DIM, CUBE_DIM),
            BufferArg::from_raw_parts(bufs.voxel_a.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.voxel_b.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.sim_list.clone(), C_BRICK_COUNT as usize),
            BufferArg::from_raw_parts(engine.elements.table.clone(), C_ELEM_WORDS * 256),
            BufferArg::from_raw_parts(engine.air.t.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(engine.air.pv.clone(), tpt_core::voxel3d::AIR_COUNT),
            BufferArg::from_raw_parts(
                engine.explosion_queue.clone(),
                (EXPLOSION_CAPACITY * EXPLOSION_STRIDE) as usize,
            ),
            BufferArg::from_raw_parts(engine.explosion_count.clone(), 3),
            BufferArg::from_raw_parts(
                bufs.mutation_queue.clone(),
                (super::super::mutation::MUTATION_CAPACITY
                    * super::super::mutation::MUTATION_STRIDE) as usize,
            ),
            BufferArg::from_raw_parts(bufs.mutation_count.clone(), 3),
            BufferArg::from_raw_parts(bufs.aux_ctype.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp2.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.aux_tmp3.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.flood_queue.clone(), tpt_core::voxel3d::VOXEL_COUNT),
            BufferArg::from_raw_parts(bufs.flood_state.clone(), 2),
            engine.tick as u32,
            engine.config.ambient_temp_k,
            if engine.config.air_enabled {
                1u32
            } else {
                0u32
            },
        );
    }
}
