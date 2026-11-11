//! `Simulation::try_move`'s shared-pixel interactions (`e == 2`), resolved
//! **before** the commit pass writes the positions.
//!
//! TPT runs this table inside `try_move` (`Simulation.cpp:1258-1381`) when the
//! entry returns `e == 2` ("both particles occupy the same space"), i.e.
//! *before* `Simulation::move` stores `parts[i].x/y` and the map entry
//! (`Simulation.cpp:1529-1534`). The port keeps that order:
//! [`launch_pair_resolution`] runs between the intent and commit launches of
//! [`crate::move_kernel::step_movement_gpu`], reads the destination from
//! `intent_target[i]` (the velocity pass already ran `eval_move`/`PlanMove`) and
//! the matter occupant from the *pre-move* `pmap[dest]`. Resolving the pair
//! before either slot moves is what makes the origin-dependent fields right:
//!
//! * `PHOT + C5` stores the photon's sub-pixel offset in the crystal's `tmp2`
//!   (`Simulation.cpp:1309`),
//! * `PHOT + GLOW` aims its gain photon from the photon's pre-move float
//!   position and velocity (`Simulation.cpp:2022-2023`), and
//! * `PHOT + H2` rebuilds the hydrogen slot as `ELEC` on the *mover's origin*
//!   (`create_part(ID(r), x, y, PT_ELEC)`, `Simulation.cpp:1341`), leaving one
//!   particle per pixel instead of two on the destination.
//!
//! # Claiming and applying
//!
//! * Only the mover that won the destination pixel may resolve its pair; the
//!   check is the same `cell_winner` key the commit pass reads.
//! * The mover claims the occupant with an atomic CAS on the occupant's
//!   `part_ty` (the [`super::glow`] convention), so only one thread resolves a
//!   pair.
//! * `part_change_type` preserves `temp`/velocity, so those type changes are
//!   written to the slot directly. `create_part(ID(r), ...)` replaces the whole
//!   row, so the H2 arm applies that element's `DefaultProperties` and `Create`
//!   hook to the occupant slot (`Simulation.cpp:1941-1948`, `ELEC.cpp:159-165`).
//! * Brand-new particles (gain photon, Cherenkov photon) are normal pending
//!   create records.
//!
//! The arms' own conditions are exactly the `e == 2` decision for every
//! interaction (`PROP_PHOTPASS` / explicit `can_move=2` entries); the two
//! special cases are handled locally: `PT_GPMP` only fires while its `life` is
//! below the passing threshold, and `PT_INVIS` keeps its own air-pressure
//! test. A photon always passes the `eval_move` entry for `INVIS` because
//! `PROP_PHOTPASS` sets `can_move[PT_PHOT][PT_INVIS] = 2`
//! (`SimulationData.cpp:188-192`, overriding the generic `3`), independent of
//! the half-silvered-mirror path; the pressure test lives inside the arm
//! itself (`Simulation.cpp:1315-1326`).
//!
//! `step_movement_gpu` arms the queue before this launch and drains it after
//! the commit launch, so the pair's creations land on the post-move map and the
//! slot-targeted reset runs before the spatial maps are rebuilt.

use cubecl::prelude::*;
use tpt_core::constants::*;

use crate::buffers::SimBuffers;
use crate::move_kernel::{INTENT_PIXEL_MASK, NO_MOVE_INTENT};
use crate::reaction_kernel::neut::cherenkov_photon;
use crate::reaction_kernel::rays::{interact_wavelengths, seed_i32};
use crate::reaction_kernel::shared::{
    CR_NO_OCCUPANCY, CR_SET_CTYPE, CR_SET_TEMP, CR_SET_VX, CR_SET_VY, CREATE_QUEUE_WORDS, hash_u32,
    launch_grid, request_create,
};
use crate::reaction_kernel::special::chance1;

/// TPT `PT_GLOW` photon gain denominator (`Simulation.cpp:1266`).
const GLOW_GAIN_DENOM: u32 = 30;
/// `PT_GLOW`'s photon-gain timer, in frames (`Simulation.cpp:1269`).
const GLOW_GAIN_LIFE: i32 = 120;
/// `PT_ELEC` -> `PT_PHOT` conversion colour (`Simulation.cpp:1331`).
const ELEC_PHOT_WL: u32 = 0x3FFF_FFFF;
/// `PT_C5` only stores wavelengths above the visible band
/// (`Simulation.cpp:1305`).
const C5_WL_MASK: u32 = 0xFFFF_FFC0;
/// Transient occupant claim used by the pair CAS.
const PAIR_CLAIMED: u32 = 0xFFFF_FFFFu32;
/// `PT_NUM` mirrored for kernel-side range checks.
const PT_NUM_U32: u32 = PT_NUM as u32;
const _: () = assert!(PT_NUM_U32 as usize == PT_NUM);
/// `tpt_core::elements::SPAWN_DEFAULT_WORDS` mirrored for the H2 row reset.
const SPAWN_DEFAULT_WORDS: usize = tpt_core::elements::SPAWN_DEFAULT_WORDS;
const SPAWN_DEFAULT_WORDS_U32: u32 = SPAWN_DEFAULT_WORDS as u32;
const _: () = assert!(tpt_core::elements::SPAWN_DEFAULT_WORDS == 8);

/// Seeds a `let mut` float from a runtime anchor so a later runtime assignment
/// does not trip CubeCL's literal expansion (see the kernel rules in AGENTS.md).
#[cube]
fn seed_f32(anchor: f32, value: f32) -> f32 {
    anchor * 0.0f32 + value
}

/// Resolves one `e == 2` pair before the commit pass moves either slot.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)] // `.clamp()` is not available inside kernels
pub fn pair_resolve_kernel(
    pmap: &[u32],
    part_ty: &mut [Atomic<u32>],
    part_ctype: &mut [u32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_temp: &mut [f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    intent_target: &mut [u32],
    cell_winner: &mut [Atomic<u32>],
    air_pv: &[f32],
    create_queue: &mut [Atomic<u32>],
    spawn_defaults: &[u32],
    slot_count: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i >= slot_count as usize {
        terminate!();
    }
    let ty = part_ty[i].load();
    if ty == PT_NONE || ty >= PT_NUM_U32 {
        terminate!();
    }
    let mut tracked = false;
    if ty == PT_PHOT
        || ty == PT_NEUT
        || ty == PT_ELEC
        || ty == PT_PROT
        || ty == PT_BIZR
        || ty == PT_BIZRG
    {
        tracked = true;
    }
    if !tracked {
        terminate!();
    }

    let raw = intent_target[i];
    // `NO_MOVE_INTENT` and the blocked-photon marker (`Simulation.cpp:
    // 3044-3056`) are not shared-pixel entries.
    if raw == NO_MOVE_INTENT || (raw & 0x8000_0000u32) != 0u32 {
        terminate!();
    }
    let dest = (raw & INTENT_PIXEL_MASK) as usize;
    let pixels = (xres * yres) as usize;
    if dest >= pixels {
        terminate!();
    }

    let xf = part_x[i];
    let yf = part_y[i];
    let xi = (xf + 0.5f32) as i32;
    let yi = (yf + 0.5f32) as i32;
    let own = (yi * xres as i32 + xi) as u32;
    // `try_move`'s `if (x==nx && y==ny) return 1;` — no entry, no interaction.
    if own as usize == dest {
        terminate!();
    }

    // `r` is the occupant of the *pre-move* destination pixel.
    let m = pmap[dest];
    if m < 512u32 {
        terminate!();
    }
    let rty = m & 0x1FFu32;
    let r = ((m >> 9u32) - 1u32) as usize;
    if r >= slot_count as usize {
        terminate!();
    }

    // Only the pixel's arbitration winner may move (and therefore interact).
    let id_field = ((i as u32) + 1u32) & 0x07FF_FFFFu32;
    let winner = cell_winner[dest].fetch_max(0u32);
    if (winner & 0x07FF_FFFFu32) != id_field {
        terminate!();
    }

    // Every arm below is only reachable from `try_move` when the entry test
    // reported `e == 2`, and for these partner types that result is fixed by
    // `PROP_PHOTPASS` / the explicit `can_move = 2` entries — the only
    // conditional partners carry their condition in the arm itself (`GPMP`
    // `life`, `INVIS` air pressure). See the module docs.

    // Claim the occupant. Only the first mover may resolve a pair; the claim is
    // released (with the occupant's final type) at the end of the body, and
    // commit only runs after this kernel completes.
    if part_ty[r].compare_exchange_weak(rty, PAIR_CLAIMED) != rty {
        terminate!();
    }

    let seed = hash_u32(frame_seed ^ hash_u32(i as u32));
    let mut final_ty = rty;

    if ty == PT_PHOT {
        if rty == PT_GLOW {
            // `Simulation.cpp:1263-1272`: a photon crossing `GLOW` seeds its
            // laser, which emits a photon of its own from the perpendicular
            // pixel.
            if part_life[r] == 0i32 && chance1(seed, GLOW_GAIN_DENOM) {
                part_life[r] = GLOW_GAIN_LIFE;
                let mut lr = seed_i32(yi, -1i32);
                if chance1(hash_u32(seed ^ 0x68E3_1DA9u32), 2u32) {
                    lr = 1i32;
                }
                let fx = xf - (lr as f32) * 0.3f32 * part_vy[i];
                let fy = yf + (lr as f32) * 0.3f32 * part_vx[i];
                let gx = (fx + 0.5f32) as i32;
                let gy = (fy + 0.5f32) as i32;
                if gx >= 0i32 && gx < xres as i32 && gy >= 0i32 && gy < yres as i32 {
                    let g = pmap[(gy as u32 * xres + gx as u32) as usize];
                    if g >= 512u32 && (g & 0x1FFu32) == PT_GLOW {
                        let gslot = ((g >> 9u32) - 1u32) as usize;
                        // `create_gain_photon` (`Simulation.cpp:2046-2049`): the
                        // photon's wavelength is the temperature bin of the
                        // glow it left.
                        let mut bin = ((part_temp[gslot] - 273.0f32) * 0.25f32) as i32;
                        if bin < 0i32 {
                            bin = 0i32;
                        }
                        if bin > 25i32 {
                            bin = 25i32;
                        }
                        let wl = 0x1Fu32 << (bin as u32);
                        request_create(
                            create_queue,
                            PT_PHOT,
                            gx,
                            gy,
                            CR_NO_OCCUPANCY | CR_SET_CTYPE | CR_SET_TEMP | CR_SET_VX | CR_SET_VY,
                            0i32,
                            wl,
                            part_temp[gslot],
                            0i32,
                            0i32,
                            0i32,
                            0i32,
                            0u32,
                            0u32,
                            part_vx[i],
                            part_vy[i],
                        );
                    }
                }
            }
        } else if rty == PT_FILT {
            // `Simulation.cpp:1273-1275`.
            part_ctype[i] =
                interact_wavelengths(part_tmp[r], part_ctype[r], part_temp[r], part_ctype[i]);
        } else if rty == PT_C5 {
            if part_life[r] > 0i32 && (part_ctype[r] & part_ctype[i] & C5_WL_MASK) != 0u32 {
                // `Simulation.cpp:1277-1303`: an occupied `C5` collides,
                // blending wavelengths and momentum.
                let cvx = ((part_tmp[r] << 16i32) >> 16i32) as f32 / 255.0f32;
                let cvy = (part_tmp[r] >> 16i32) as f32 / 255.0f32;
                let ovx = part_vx[i];
                let ovy = part_vy[i];
                if ovx + cvx == 0.0f32 && ovy + cvy == 0.0f32 {
                    part_vx[i] = cvy;
                    part_vy[i] = -cvx;
                } else {
                    part_ctype[i] = (part_ctype[r] & part_ctype[i]) >> 6u32;
                    part_vx[i] = ovx + cvx;
                    part_vy[i] = ovy + cvy;
                    let speed = part_vx[i] * part_vx[i] + part_vy[i] * part_vy[i];
                    let scale = ((ovx * ovx + ovy * ovy) / speed).sqrt();
                    part_vx[i] *= scale;
                    part_vy[i] *= scale;
                }
                part_life[r] = 0i32;
                part_ctype[r] = 0u32;
            } else if part_ctype[r] == 0u32 && (part_ctype[i] & C5_WL_MASK) != 0u32 {
                // `Simulation.cpp:1304-1314`: an empty `C5` stores the photon
                // and its direction. `tmp2` is the photon's *pre-move*
                // sub-pixel offset — the pair resolves before the position
                // write, so `xf`/`yf` still hold the origin.
                part_life[r] = 1i32;
                part_ctype[r] = part_ctype[i];
                part_tmp[r] = (0xFFFFi32 & (part_vx[i] * 255.0f32) as i32)
                    | (0xFFFF_0000u32 as i32 & (part_vy[i] * 16_711_680.0f32) as i32);
                part_tmp2[r] = (0xFFFFi32 & ((xf - xi as f32) * 255.0f32) as i32)
                    | (0xFFFF_0000u32 as i32 & ((yf - yi as f32) * 16_711_680.0f32) as i32);
                part_ty[i].store(PT_NONE);
            }
        } else if rty == PT_INVIS {
            // `Simulation.cpp:1315-1326`: a photon passing an invisible wall
            // (the `PROP_PHOTPASS` entry is unconditional, so `e == 2` is
            // reached regardless of the half-mirror path) is scattered into
            // a neutron while the air pressure stays inside the wall's
            // resistance.
            let mut resistance = seed_f32(part_temp[i], 4.0f32);
            if part_tmp[r] > 0i32 {
                resistance = part_tmp[r] as f32;
            }
            let px = dest as u32 % xres;
            let py = dest as u32 / xres;
            let cell = (py / 4u32) * xcells + (px / 4u32);
            let pv = air_pv[cell as usize];
            if pv >= -resistance && pv <= resistance {
                part_ty[i].store(PT_NEUT);
                part_ctype[i] = 0u32;
            }
        } else if rty == PT_BIZR || rty == PT_BIZRG || rty == PT_BIZRS {
            // `Simulation.cpp:1326-1332`: PHOT + BIZR/BIZRG/BIZRS -> ELEC.
            part_ty[i].store(PT_ELEC);
            part_ctype[i] = 0u32;
        } else if rty == PT_H2 {
            if (part_tmp[i] & 1i32) == 0i32 {
                // `Simulation.cpp:1333-1343`: PHOT + H2 -> PROT + ELEC. The
                // photon becomes the proton (fields preserved), while the
                // hydrogen slot is replaced by a fresh electron on the mover's
                // *origin* (`create_part(ID(r), x, y, PT_ELEC)`).
                part_ty[i].store(PT_PROT);
                part_ctype[i] = 0u32;
                part_tmp2[i] = 1i32;
                final_ty = PT_ELEC;
                part_x[r] = xi as f32;
                part_y[r] = yi as f32;
                // The reference's `return 1` skips the hydrogen's own movement
                // this frame; the port cancels its intent instead of letting
                // commit undo the relocation.
                intent_target[r] = NO_MOVE_INTENT;
                // `create_part(ID(r), x, y, PT_ELEC)` copies the element's
                // `DefaultProperties`, then runs its `Create` hook
                // (`Simulation.cpp:1941-1948`, `ELEC.cpp:159-165`). The row is
                // rewritten here rather than queued: commit has not run yet, so
                // the write is the one commit would apply to this slot anyway.
                let db = (PT_ELEC * SPAWN_DEFAULT_WORDS_U32) as usize;
                part_temp[r] = f32::reinterpret(spawn_defaults[db]);
                part_life[r] = 680i32;
                part_ctype[r] = spawn_defaults[db + 2usize];
                part_tmp[r] = spawn_defaults[db + 3usize] as i32;
                part_tmp2[r] = spawn_defaults[db + 4usize] as i32;
                let roll = hash_u32(frame_seed ^ (((r as u32) + 1u32) << 7u32) ^ (PT_ELEC << 3u32));
                let a = (roll % 360u32) as f32 / 360.0f32 * core::f32::consts::TAU;
                part_vx[r] = 2.0f32 * a.cos();
                part_vy[r] = 2.0f32 * a.sin();
            }
        } else if rty == PT_GPMP && part_life[r] == 0i32 {
            // `Simulation.cpp:1344-1351`: PHOT + GPMP -> GRVT + GPMP.
            part_ty[i].store(PT_GRVT);
            part_tmp[i] = (part_temp[r] - 273.15f32) as i32;
        }
    } else if ty == PT_NEUT {
        // `Simulation.cpp:1353-1358`: NEUT + GLAS/BGLA -> NEUT + GLAS/BGLA +
        // PHOT (one in ten).
        if (rty == PT_GLAS || rty == PT_BGLA) && chance1(seed, 10u32) {
            // `create_cherenkov_photon` reads the *origin* pixel for the
            // glass that owns the temperature and re-checks the speed.
            let g = pmap[own as usize];
            if g >= 512u32 {
                let gty = g & 0x1FFu32;
                if gty == PT_GLAS || gty == PT_BGLA {
                    let vx = part_vx[i];
                    let vy = part_vy[i];
                    if vx * vx + vy * vy >= 1.44f32 * 1.44f32 {
                        let gslot = ((g >> 9u32) - 1u32) as usize;
                        cherenkov_photon(
                            create_queue,
                            xi,
                            yi,
                            vx,
                            vy,
                            part_temp[gslot],
                            hash_u32(seed ^ 0x9E37_79B9u32),
                        );
                    }
                }
            }
        }
    } else if ty == PT_ELEC {
        if rty == PT_GLOW {
            // `Simulation.cpp:1362-1368`: ELEC + GLOW -> PHOT.
            part_ty[i].store(PT_PHOT);
            part_ctype[i] = ELEC_PHOT_WL;
        }
    } else if ty == PT_PROT {
        if rty == PT_INVIS {
            // `Simulation.cpp:1370-1374`: PROT + INVIS -> NEUT.
            part_ty[i].store(PT_NEUT);
        }
    } else {
        // `PT_BIZR` / `PT_BIZRG`: `Simulation.cpp:1375-1379`.
        if rty == PT_FILT {
            part_ctype[i] =
                interact_wavelengths(part_tmp[r], part_ctype[r], part_temp[r], part_ctype[i]);
        }
    }

    part_ty[r].store(final_ty);
}

/// Launches the `e == 2` pair resolution for one movement step.
///
/// Runs after the intent launch and before the commit launch; the caller owns
/// the creation-queue begin/drain around the pair and the commit.
pub(crate) fn launch_pair_resolution(
    bufs: &SimBuffers,
    slot_count: usize,
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let pixels = (xres * yres) as usize;
    if slot_count == 0 || pixels == 0 || xcells == 0 {
        return;
    }
    let cells = (xcells * bufs.config.ycells()) as usize;
    let capacity = bufs.capacity;
    let (cubes, dim) = launch_grid(slot_count);
    unsafe {
        pair_resolve_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            BufferArg::from_raw_parts(bufs.intent_target.clone(), capacity),
            BufferArg::from_raw_parts(bufs.cell_winner.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            BufferArg::from_raw_parts(
                bufs.gpu_spawn_defaults.clone(),
                PT_NUM * tpt_core::elements::SPAWN_DEFAULT_WORDS,
            ),
            slot_count as u32,
            xres,
            yres,
            xcells,
            frame_seed,
        );
    }
}
