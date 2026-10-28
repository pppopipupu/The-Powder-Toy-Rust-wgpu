//! Commit stage of the movement pass and the spatial-map rebuild.
//!
//! * [`commit_move_kernel`] applies the arbitration winners: a particle only
//!   moves when it owns its target pixel; entering `PT_VOID` or a
//!   `WL_DESTROYALL` wall kills it, a displaceable occupant is swapped, and the
//!   same pass maintains `FLAG_STAGNANT` for liquids that could not move.
//! * [`clear_spatial_maps_kernel`] + [`scatter_spatial_maps_kernel`] rebuild
//!   `pmap` / `photons` from the post-move state, which keeps the frame
//!   GPU-resident (the host `rebuild_spatial_maps` is never needed here).
//!
//! The kernels spell constants out as literals (`4`, `0x1FF`, `0x7FFFFF`)
//! because kernel bodies cannot reference the host constants; element ids use
//! the `PT_NUM_U32` mirror and the LUT stride comes from
//! [`super::can_move_at`]. See the module docs of [`crate::move_kernel`] for the
//! mirror table.

use cubecl::prelude::*;

use tpt_core::constants::{
    MAX_TEMP, MIN_TEMP, PROP_NEUTABSORB, PROP_NEUTPENETRATE, PT_ANAR, PT_BCOL, PT_BHOL, PT_BVBR,
    PT_COAL, PT_DEUT, PT_ELEC, PT_FILT, PT_GBMB, PT_GPMP, PT_HSWC, PT_INVIS, PT_LCRY, PT_LITH,
    PT_NBHL, PT_NEUT, PT_NSCN, PT_NUM, PT_NWHL, PT_PHOT, PT_PIPE, PT_PPIP, PT_PSCN, PT_PVOD,
    PT_SEED, PT_SPRK, PT_SWCH, PT_TRON, PT_VIBR, PT_VOID, PT_WHOL, TYPE_ENERGY,
};

use super::can_move_at;
use super::{INTENT_HALF_MIRROR, INTENT_PIXEL_MASK};
use crate::reaction_kernel::shared::pmap_at;

/// `PT_NUM` as a `u32` for kernel-side range checks.
///
/// Kernel bodies cannot reference the host constant directly, so it is mirrored
/// here (same convention as `thermal_kernel`); every element-id guard below
/// uses this instead of a `64` literal.
const PT_NUM_U32: u32 = PT_NUM as u32;

/// `PT_PHOT` / `PT_LITH` / `PT_SEED` as `u32` for the reflection-mask rule
/// (`Simulation.cpp:3044-3056`). `PLNT_COLOUR` is the seed gene bit offset
/// (`PLNT.h`), mirrored here because kernel bodies cannot import it.
const PT_PHOT_U32: u32 = PT_PHOT;
const PT_LITH_U32: u32 = PT_LITH;
const PT_SEED_U32: u32 = PT_SEED;
/// Blocked-photon heat exchange mirrors (`Simulation.cpp:1220-1228`,
/// `SimulationData.cpp:338-341`).
const PT_COAL_U32: u32 = PT_COAL;
const PT_BCOL_U32: u32 = PT_BCOL;
const PT_FILT_U32: u32 = PT_FILT;
const PT_HSWC_U32: u32 = PT_HSWC;
const PT_PIPE_U32: u32 = PT_PIPE;
const PT_PPIP_U32: u32 = PT_PPIP;
/// `PFLAG_CAN_CONDUCT` (`PIPE.h:18`), the `tmp` bit a `PIPE`/`PPIP` needs to
/// conduct heat (`SimulationData.cpp:340`).
const PFLAG_CAN_CONDUCT: i32 = 0x0000_0001;
/// `PT_NEUT` mirrored for kernel bodies (rule 1).
const PT_NEUT_U32: u32 = PT_NEUT;
/// `Simulation.cpp:1568-1580` `photoelectric_effect` mirrors: an idle `PSCN`
/// with an `NSCN` beside it turns into a four-frame spark when a photon is
/// blocked by it.
const PT_PSCN_U32: u32 = PT_PSCN;
const PT_NSCN_U32: u32 = PT_NSCN;
const PT_SPRK_U32: u32 = PT_SPRK;
const ELECTRODE_SPARK_LIFE: i32 = 4;
const _: () = assert!(PT_PHOT == 29);
const _: () = assert!(PT_LITH == 190);
const _: () = assert!(PT_SEED == 194);
const _: () = assert!(PT_NEUT == 30);
const _: () = assert!(PT_PSCN == 50);
const _: () = assert!(PT_NSCN == 51);
const _: () = assert!(PT_SPRK == 12);

/// Resolves a `can_move == 3` entry (`Simulation.cpp:1126-1170`): the
/// destination decides whether the mover enters, shares the cell or is
/// blocked, using its own `life`/`tmp`/`ctype` and the local air pressure.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_can(
    can: u32,
    ty: u32,
    dest_ty: u32,
    other: usize,
    life: &mut [i32],
    tmp: &mut [i32],
    ctype: &mut [u32],
    pv: f32,
) -> u32 {
    let mut result = can;
    if result == 3u32 {
        if dest_ty == PT_LCRY && ty == PT_PHOT {
            // A charged crystal is opaque until its charge decays.
            result = if life[other] > 5i32 { 2u32 } else { 0u32 };
        } else if dest_ty == PT_GPMP && ty == PT_PHOT {
            result = if life[other] < 10i32 { 2u32 } else { 0u32 };
        } else if dest_ty == PT_INVIS {
            // Invisible wall: permeable only while the air pressure is beyond
            // the wall's resistance (`tmp`, default 4).
            let mut resistance = 4.0f32;
            if tmp[other] > 0i32 {
                resistance = tmp[other] as f32;
            }
            let mut outside = false;
            if pv < -resistance || pv > resistance {
                outside = true;
            }
            result = if outside { 2u32 } else { 0u32 };
        } else if dest_ty == PT_PVOD {
            let mut enter = false;
            if life[other] == 10i32 {
                enter = void_admits(ty, ctype[other], tmp[other]);
            }
            result = if enter { 1u32 } else { 0u32 };
        } else if dest_ty == PT_VOID {
            result = if void_admits(ty, ctype[other], tmp[other]) {
                1u32
            } else {
                0u32
            };
        } else if dest_ty == PT_SWCH {
            if ty == PT_TRON {
                result = if life[other] >= 10i32 { 2u32 } else { 0u32 };
            } else {
                result = 1u32;
            }
        } else {
            result = 1u32;
        }
    }
    result
}

/// `PT_VOID` / `PT_PVOD` filter (`Simulation.cpp:1149-1162`): an empty `ctype`
/// admits everything, otherwise a `tmp & 1` flag inverts the match.
#[cube]
fn void_admits(ty: u32, void_ctype: u32, void_tmp: i32) -> bool {
    let mut admit = true;
    if void_ctype != 0u32 {
        let matches = void_ctype == ty;
        let inverted = (void_tmp & 1i32) != 0i32;
        admit = if inverted { !matches } else { matches };
    }
    admit
}

/// `SimulationData::IsHeatInsulator` (`SimulationData.cpp:338-341`):
///
/// ```text
/// HeatConduct == 0
///   || (type == PT_HSWC && life != 10)
///   || ((type == PT_PIPE || type == PT_PPIP) && (tmp & PFLAG_CAN_CONDUCT) == 0)
/// ```
///
/// `heat_conduct` is the `HeatConduct` column (word 3 of the `move_lut`
/// record).
#[cube]
fn is_heat_insulator(ty: u32, life: i32, tmp: i32, heat_conduct: u32) -> bool {
    let mut insulator = heat_conduct == 0u32;
    if ty == PT_HSWC_U32 && life != 10i32 {
        insulator = true;
    }
    if (ty == PT_PIPE_U32 || ty == PT_PPIP_U32) && (tmp & PFLAG_CAN_CONDUCT) == 0i32 {
        insulator = true;
    }
    insulator
}

/// Applies every arbitration winner to the particle pool.
///
/// A particle moves only when `intent_target[i]` is set *and* it owns the pixel:
/// `cell_winner[target] & 0x7FFFFF == id + 1`. Entering `PT_VOID`/`PT_PVOD`
/// kills it, entering a displaceable occupant (`can_move == 1`) swaps the two
/// positions, otherwise the position is simply written. Occupants that won a
/// pixel themselves are left alone (see the swap annotation below).
///
/// A `WL_DESTROYALL` wall does **not** kill on entry: the reference kills a
/// particle standing inside one at the start of the next `UpdateParticles`
/// (`Simulation.cpp:2314-2330`), which is now
/// [`super::velocity::velocity_and_intent_kernel`]'s `wall_kills_particle`
/// pre-check. This lets the particle live one frame inside the wall, as in
/// `Simulation.cpp:1506-1537` + `2295-2330`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::manual_clamp)]
pub fn commit_move_kernel(
    part_ty: &mut [u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_flags: &mut [u32],
    part_ctype: &mut [u32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_temp: &mut [f32],
    intent_target: &[u32],
    cell_winner: &mut [Atomic<u32>],
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    air_pv: &[f32],
    move_lut: &[u32],
    photon_mask: &[u32],
    num_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    heat_enable: u32,
) {
    let i = ABSOLUTE_POS;
    if i >= num_slots as usize {
        terminate!();
    }
    let ty = part_ty[i];
    if ty == 0u32 {
        terminate!();
    }
    if ty >= PT_NUM_U32 {
        terminate!();
    }

    let raw_target = intent_target[i];
    if raw_target == 0xFFFF_FFFFu32 {
        // TPT sets `FLAG_STAGNANT` whenever a liquid could not move
        // (Simulation.cpp:3232, 3330, 3338); the intent pass then uses the
        // shorter `rt = 10` scan next frame. Rigid pixels are owned by the
        // rigid-body solver and never enter the liquid branch.
        let falldown = move_lut[(ty as usize) * 12usize + 1usize];
        if falldown >= 2u32 && (part_flags[i] & 0x10u32) == 0u32 {
            part_flags[i] |= 0x01u32;
        }
        terminate!();
    }
    // `INTENT_HALF_MIRROR` rides above the pixel field (`Simulation.cpp:
    // 1205-1206`); the blocked-photon marker uses bit 31 and is tested first.
    let half_mirror = (raw_target & INTENT_HALF_MIRROR) != 0u32;
    let target = raw_target & INTENT_PIXEL_MASK;
    if (raw_target & 0x8000_0000u32) != 0u32 {
        // A photon that could not enter the pixel in front of it (`Simulation.cpp:
        // 3044-3056`): the wavelengths the blocking element absorbs are cleared
        // from its `ctype`, and a photon left with no wavelength dies on its own
        // update (`PHOT.cpp:58`). The intent pass reports the blocking pixel in
        // the low bits; `pmap` still holds the pre-move state.
        if ty == PT_PHOT_U32 {
            let hit = (target & 0x7FFF_FFFFu32) as usize;
            let r = pmap[hit];
            if r >= 512u32 {
                let rty = r & 0x1FFu32;
                let rslot = ((r >> 9u32) - 1u32) as usize;
                let mut mask = photon_mask[rty as usize];
                if rty == PT_LITH_U32 {
                    // `LITH` reflects at the single wavelength it is tuned to.
                    let mut bin = (part_ctype[rslot] / 4u32) as i32;
                    if bin < 0i32 {
                        bin = 0i32;
                    }
                    if bin > 25i32 {
                        bin = 25i32;
                    }
                    mask = 0x1Fu32 << (bin as u32);
                } else if rty == PT_SEED_U32 {
                    // `SEED` reflects the wavelengths its colour genes do not
                    // use: red `1 << 25`, green `1 << 15`, blue `1 << 5`.
                    let colour = (part_ctype[rslot] >> 6u32) & 0x3Fu32;
                    if (colour & 48u32) == 0u32 {
                        mask |= 1u32 << 25u32;
                    }
                    if (colour & 12u32) == 0u32 {
                        mask |= 1u32 << 15u32;
                    }
                    if (colour & 3u32) == 0u32 {
                        mask |= 1u32 << 5u32;
                    }
                }
                part_ctype[i] &= mask;

                // `Simulation.cpp:1568-1580` `photoelectric_effect`: a photon
                // stopped by an idle `PSCN` sparks that electrode when an
                // `NSCN` sits on one of its four orthogonal sides. The check
                // reads the particle layer (`pmap`), not the energy layer, and
                // the new spark keeps the electrode type in `ctype` for four
                // frames (`parts[ID(r)].life = 4`).
                if rty == PT_PSCN_U32 && part_life[rslot] == 0i32 {
                    let hx = (hit as u32 % xres) as i32;
                    let hy = (hit as u32 / xres) as i32;
                    let mut electrode = false;
                    if (pmap_at(pmap, xres, yres, hx - 1i32, hy) & 0x1FFu32) == PT_NSCN_U32 {
                        electrode = true;
                    }
                    if (pmap_at(pmap, xres, yres, hx + 1i32, hy) & 0x1FFu32) == PT_NSCN_U32 {
                        electrode = true;
                    }
                    if (pmap_at(pmap, xres, yres, hx, hy - 1i32) & 0x1FFu32) == PT_NSCN_U32 {
                        electrode = true;
                    }
                    if (pmap_at(pmap, xres, yres, hx, hy + 1i32) & 0x1FFu32) == PT_NSCN_U32 {
                        electrode = true;
                    }
                    if electrode {
                        part_ctype[rslot] = PT_PSCN_U32;
                        part_ty[rslot] = PT_SPRK_U32;
                        part_life[rslot] = ELECTRODE_SPARK_LIFE;
                    }
                }

                // `try_move`'s blocked-energy heat exchange
                // (`Simulation.cpp:1220-1228`): a photon that cannot enter a
                // pixel equalises its temperature with the blocking particle
                // (and charges `COAL`/`BCOL` to the photon's temperature).
                // `heat_enable` is the reference's `!legacy_enable` guard.
                if heat_enable != 0u32 {
                    if rty == PT_COAL_U32 || rty == PT_BCOL_U32 {
                        part_temp[rslot] = part_temp[i];
                    }
                    let heat_conduct = move_lut[(rty as usize) * 12usize + 3usize];
                    if !is_heat_insulator(rty, part_life[rslot], part_tmp[rslot], heat_conduct)
                        && rty != PT_FILT_U32
                    {
                        let avg = max(
                            min((part_temp[rslot] + part_temp[i]) * 0.5f32, MAX_TEMP),
                            MIN_TEMP,
                        );
                        part_temp[i] = avg;
                        part_temp[rslot] = avg;
                    }
                }
            }
        }
        terminate!();
    }
    if target >= xres * yres {
        terminate!();
    }
    // A particle that found a destination is no longer stagnant
    // (`parts[i].flags &= ~FLAG_STAGNANT`, Simulation.cpp:2892).
    part_flags[i] &= 0xFFFF_FFFEu32;

    // Only the pixel's arbitration winner may move.
    //
    // `fetch_max(0)` is a no-op that returns the previous key. It is used
    // instead of a plain `load` so the compiled shader declares `cell_winner`
    // as `storage, read_write`: WGSL only allows atomic operations on read_write
    // storage buffers, and a load-only usage would be lowered to `storage, read`.
    let key = cell_winner[target as usize].fetch_max(0u32);
    if (key & 0x7FFFFFu32) != ((i as u32) + 1u32) {
        terminate!();
    }

    if half_mirror {
        // The half-silvered mirror forced `e = 2` (`Simulation.cpp:1205-1206`):
        // `try_move` returns 1 there, so `move` writes the position without ever
        // consulting `can_move` again. This is the port's equivalent — a plain
        // position write, no swap, no kill, no wavelength filter.
        part_x[i] = (target % xres) as f32;
        part_y[i] = (target / xres) as f32;
        terminate!();
    }

    let nx = (target % xres) as i32;
    let ny = (target / xres) as i32;
    let ty_idx = ty as usize;

    let wcell = (((ny as u32) / 4u32) * xcells + ((nx as u32) / 4u32)) as usize;
    let wall = wall_bmap[wcell];
    let dest = pmap[target as usize];
    let dest_ty = dest & 0x1FFu32;
    let dest_occ = dest >> 9u32;
    let mover_props = move_lut[ty_idx * 12usize];
    let dest_props = move_lut[(dest_ty as usize) * 12usize];

    // `Simulation.cpp:1126-1170`: resolve the runtime-decided entries.
    let mut can = 0u32;
    let mut other = i;
    if dest_occ != 0u32 {
        other = (dest_occ - 1u32) as usize;
        can = can_move_at(move_lut, ty_idx, dest_ty);
        can = resolve_can(
            can,
            ty,
            dest_ty,
            other,
            part_life,
            part_tmp,
            part_ctype,
            air_pv[wcell],
        );
    }

    // `try_move` (`Simulation.cpp:1461-1462`): a particle inside an unpowered
    // E-hole may not leave it for open space.
    let mut ehole_exit = false;
    if wall == 12u32 && (wall_emap[wcell] & 0xFFu32) == 0u32 {
        let cx = i32::cast_from(part_x[i] + 0.5f32) / 4i32;
        let cy = i32::cast_from(part_y[i] + 0.5f32) / 4i32;
        let here = ((cy as u32) * xcells + (cx as u32)) as usize;
        let here_hole = wall_bmap[here] == 12u32 && (wall_emap[here] & 0xFFu32) == 0u32;
        if here_hole {
            ehole_exit = true;
        }
    }

    if ehole_exit {
        // Blocked by the unpowered E-hole it is sitting in.
    } else if dest_occ != 0u32 && can == 0u32 {
        // `can == 0`: the destination is impassable; the particle stays.
    } else if dest_occ != 0u32 && can == 1u32 && other != i {
        // `try_move`'s target switch (`Simulation.cpp:1387-1432`): the
        // destination consumes the mover instead of swapping.
        if dest_ty == PT_VOID || dest_ty == PT_PVOD {
            part_ty[i] = 0u32;
        } else if dest_ty == PT_BHOL || dest_ty == PT_NBHL {
            // A hole swallows the particle and folds half its heat in.
            part_temp[other] = max(
                min(part_temp[other] + part_temp[i] / 2.0f32, MAX_TEMP),
                MIN_TEMP,
            );
            part_ty[i] = 0u32;
        } else if dest_ty == PT_WHOL || dest_ty == PT_NWHL {
            if ty == PT_ANAR {
                part_temp[other] = max(
                    min(
                        part_temp[other] - (MAX_TEMP - part_temp[i]) / 2.0f32,
                        MAX_TEMP,
                    ),
                    MIN_TEMP,
                );
                part_ty[i] = 0u32;
            }
        } else if dest_ty == PT_DEUT {
            if ty == PT_ELEC {
                // `DEUT` stores one electron of charge per hit.
                if part_life[other] < 6000i32 {
                    part_life[other] += 1i32;
                }
                part_temp[other] = 0.0f32;
                part_ty[i] = 0u32;
            }
        } else if dest_ty == PT_VIBR || dest_ty == PT_BVBR {
            if (mover_props & TYPE_ENERGY) != 0u32 {
                part_tmp[other] += 20i32;
                part_ty[i] = 0u32;
            }
        } else if ty == PT_NEUT_U32 && (dest_props & PROP_NEUTABSORB) != 0u32 {
            // `try_move`'s mover switch: an absorbing element eats the neutron
            // instead of swapping with it.
            part_ty[i] = 0u32;
        } else if ty == PT_NEUT_U32 {
            // `try_move`'s neutron penetration (`Simulation.cpp:1485-1510`).
            //
            // `s` is the particle under the neutron in its *own* pixel
            // (`pmap[y][x]`); when it carries `PROP_NEUTPENETRATE` the neutron
            // drags it: `s` takes the target pixel, the target occupant takes
            // the neutron's old pixel, and the neutron moves to the target.
            // Otherwise only the neutron moves and the target stays (the two
            // share the pixel, since the neutron lives in `photons`).
            //
            // The drag writes three slots from one thread; the parallel commit
            // already performs cross-slot writes for swaps, and the displaced
            // particles' own threads may race. Both writes are the values the
            // reference produces, so the result is one of the valid
            // interleavings.
            let own_x = i32::cast_from(part_x[i] + 0.5f32);
            let own_y = i32::cast_from(part_y[i] + 0.5f32);
            let own_cell = (((own_y as u32) / 4u32) * xcells + ((own_x as u32) / 4u32)) as usize;
            let own = pmap[(own_y * xres as i32 + own_x) as usize];
            let mut drag = false;
            if own != 0u32 {
                let under_ty = own & 0x1FFu32;
                let under_props = move_lut[(under_ty as usize) * 12usize];
                if (under_props & PROP_NEUTPENETRATE) != 0u32 {
                    drag = true;
                }
            }
            // `WL_ALLOWENERGY`: never drag a target into an energy-only wall.
            if drag && wall_bmap[own_cell] != 15u32 {
                if own != 0u32 {
                    let under = ((own >> 9u32) - 1u32) as usize;
                    part_x[under] = nx as f32;
                    part_y[under] = ny as f32;
                }
                part_x[other] = own_x as f32;
                part_y[other] = own_y as f32;
            }
            part_x[i] = nx as f32;
            part_y[i] = ny as f32;
        } else if ty == PT_GBMB && part_life[i] > 0i32 {
            // `GBMB` holds still while its fuse runs.
        } else {
            // Swap with the displaceable occupant (`try_move`'s swap,
            // `Simulation.cpp:1494-1505`).
            //
            // Required parallelisation deviation (M-COM-1): the reference swaps
            // unconditionally and then processes the displaced particle in its
            // own turn, so two particles can trade cells in one frame. Here both
            // particles are threads of the same dispatch: an unconditional swap
            // lets A read B's position while B writes it, which can leave both
            // particles on one pixel and the other pixel empty. The gate keeps
            // the swap only when the occupant is not moving this frame; a chain
            // of two moving particles resolves on the next frame instead.
            let other_raw = intent_target[other];
            let other_target = other_raw & INTENT_PIXEL_MASK;
            let mut other_moves = false;
            if other_raw != 0xFFFF_FFFFu32
                && (other_raw & 0x8000_0000u32) == 0u32
                && other_target < xres * yres
            {
                let other_key = cell_winner[other_target as usize].load();
                if (other_key & 0x7FFFFFu32) == ((other as u32) + 1u32) {
                    other_moves = true;
                }
            }
            if !other_moves {
                let ox = part_x[other];
                let oy = part_y[other];
                let sx = part_x[i];
                let sy = part_y[i];
                part_x[other] = sx;
                part_y[other] = sy;
                part_x[i] = ox;
                part_y[i] = oy;
            }
        }
    } else if dest_occ != 0u32 && can == 2u32 {
        // Share the pixel (energy passing through matter, `INVIS` under
        // pressure, ...).
        //
        // `try_move`'s `e == 2` table (`Simulation.cpp:1258-1381`: PHOT +
        // GLOW/FILT/C5/INVIS/BIZR/H2/GPMP, NEUT + GLAS cherenkov, ELEC + GLOW,
        // PROT + INVIS, BIZR + FILT) has already run in this frame: the pair
        // pass launches between the intent and this commit
        // ([`crate::reaction_kernel::energy_entry`]), before either slot is
        // moved, so its origin-dependent fields are read from the pre-move
        // `part_x`/`part_y`. This branch therefore only shares the pixel; the
        // mover's type/`ctype`/`tmp` fields may already carry the conversion.
        part_x[i] = nx as f32;
        part_y[i] = ny as f32;
    } else {
        part_x[i] = nx as f32;
        part_y[i] = ny as f32;
    }
}

/// Clears `pmap` and `photons` before [`scatter_spatial_maps_kernel`] refills
/// them, mirroring the `memset` at the top of TPT's `RecalcFreeParticles`
/// (Simulation.cpp:3362-3364).
#[cube(launch_unchecked)]
pub fn clear_spatial_maps_kernel(pmap: &mut [u32], photons: &mut [u32], num_pixels: u32) {
    let idx = ABSOLUTE_POS;
    if idx < num_pixels as usize {
        pmap[idx] = 0u32;
        photons[idx] = 0u32;
    }
}

/// Scatters every live particle into `pmap` / `photons`.
///
/// TPT rebuilds the maps serially (`RecalcFreeParticles`); here one thread per
/// particle slot writes its own pixel. Two particles can only collide on a
/// pixel when they already share it (pass-through energy), where the last
/// writer wins just like the serial reference.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn scatter_spatial_maps_kernel(
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    elem_props: &[u32],
    pmap: &mut [u32],
    photons: &mut [u32],
    num_slots: u32,
    xres: u32,
    yres: u32,
) {
    let i = ABSOLUTE_POS;
    if i >= num_slots as usize {
        terminate!();
    }
    let ty = part_ty[i];
    if ty == 0u32 {
        terminate!();
    }
    // Element ids must fit the packed tables (`PT_NUM_U32`, mirrored PT_NUM).
    if ty >= PT_NUM_U32 {
        terminate!();
    }
    let xr = xres as i32;
    let yr = yres as i32;
    let xi = i32::cast_from(part_x[i] + 0.5f32);
    let yi = i32::cast_from(part_y[i] + 0.5f32);
    if xi < 0i32 || xi >= xr || yi < 0i32 || yi >= yr {
        terminate!();
    }
    let idx = (yi * xr + xi) as u32;
    // `pmap_pack(i, ty)`: `(id + 1) << 9 | type`.
    let packed = (((i as u32) + 1u32) << 9u32) | (ty & 0x1FFu32);
    // `TYPE_ENERGY` particles live in `photons`, everything else in `pmap`
    // (Simulation.cpp:3375-3387). `properties` is word 1 of `GpuElementProps`.
    let props = elem_props[(ty as usize) * 32usize + 1usize];
    if (props & 0x10u32) != 0u32 {
        photons[idx as usize] = packed;
    } else {
        // `RecalcFreeParticles` (`Simulation.cpp:3375-3380`): particles may sit
        // inside INVIS/FILT, so those two never overwrite an occupied pixel;
        // every other matter particle does.
        if pmap[idx as usize] == 0u32 || (ty != PT_INVIS && ty != PT_FILT) {
            pmap[idx as usize] = packed;
        }
    }
}
