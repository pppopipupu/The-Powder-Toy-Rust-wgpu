//! GPU lock-free parallel particle movement.
//!
//! Nine CubeCL kernels run back to back on the same stream every frame:
//!
//! * `air_deposit_kernel` — the particle -> air velocity deposit
//!   (`Simulation.cpp:2334-2338`), before anything reads the air grid.
//! * `detect_charge_kernel` — `WL_DETECT` span charging
//!   (`Simulation.cpp:2330-2331`, `928-983`) and the span endpoints for the
//!   next pass.
//! * `emap_gate_kernel` — port-only pre-scan that tells the single-threaded
//!   closure whether an uncharged span and a charged wire both exist, so a
//!   canvas with nothing to propagate does not pay for a full-grid sweep.
//! * `emap_closure_kernel` — the vertical half of `set_emap`: fills the wire
//!   runs above and below every span the detector kernel charged.
//! * `strip_emap_endpoints_kernel` — port-only clear of the closure's transient
//!   span-endpoint markers, one thread per cell.
//! * [`clear_move_intents_kernel`] — unclaims every pixel by zeroing the
//!   `cell_winner` atomic array.
//! * [`velocity_and_intent_kernel`] — one thread per particle slot. Dead
//!   (`ty == 0`), rigid (`FLAG_RIGID_PIXEL`), `FLAG_SKIPMOVE` and
//!   stasis-frozen particles are skipped; off-screen and killing-wall
//!   particles are erased (`Simulation.cpp:2312-2330`); the rest are coupled
//!   to their coarse air cell (`air_vx`, `air_vy`), have their velocity
//!   integrated (`Loss`, `Advection`, `Gravity`, `Diffusion`) and then pick a
//!   destination pixel according to `Falldown`:
//!   * `0` gases / moving solids / `TYPE_ENERGY` — `PlanMove`, then the
//!     `X-only -> Y-only -> both` bounce (`Simulation.cpp:3138-3152`). A
//!     blocked photon reports the blocking pixel so the commit pass can apply
//!     its wavelength mask, and a photon crossing a glass surface is refracted
//!     or totally internally reflected (`Simulation.cpp:2966-3005`).
//!   * `>= 1` liquids and powders — the shared TPT `PlanMove` +
//!     `MovementPhase` chain (Simulation.cpp:2134-2256, 3138-3346): `fin`,
//!     `(fin_x, clear_y)`, `(clear_x, fin_y)`, the two diagonal velocity
//!     moves, and for liquids the lateral scan (`rt = 30`, or 10 when
//!     `FLAG_STAGNANT` is set and no foreign neighbour is present) with its
//!     vertical continuation; every fallback damps by `Collision` and the
//!     `clear` last resort closes the chain.
//!
//!   A particle that found a legal destination writes it to `intent_target[i]`
//!   and *competes* for that pixel with a single
//!   `cell_winner[target].fetch_max(priority_key)`. The key is
//!   `((weight_rank & 0x1FF) << 23) | ((id + 1) & 0x7FFFFF)`: `9` bits of
//!   element weight rank above `23` bits of slot id, so the heavier element
//!   always wins a contested pixel and ties are broken by slot id. The id
//!   field matches [`tpt_core::constants::pmap_pack`], which also spends `23`
//!   bits on `part_id + 1`; [`SimBuffers::MAX_PARTICLE_CAPACITY`] keeps every
//!   slot id inside that field.
//! * [`commit_move_kernel`] — a particle only moves when it won its pixel
//!   (`cell_winner[target] & 0x7FFFFF == id + 1`). Entering `PT_VOID` or
//!   `PT_PVOD` kills it, entering a displaceable occupant swaps the two
//!   positions (a neutron with `PROP_NEUTPENETRATE` drags the particle under
//!   it), everything else just writes the new position. The same pass
//!   maintains `FLAG_STAGNANT` for liquids that could not move.
//! * [`clear_spatial_maps_kernel`] + [`scatter_spatial_maps_kernel`] — rebuild
//!   `pmap` / `photons` from the post-move state, replacing the host
//!   `rebuild_spatial_maps` so the frame never leaves the device.
//!
//! Because every decision is derived from the *pre-move* snapshot (`pmap`,
//! positions, velocities) and the only shared mutable state is the atomic
//! winner array, the pass is order independent: no locks, no double occupancy,
//! no per-particle iteration order. The price is the documented deviations
//! listed in [`velocity`] and [`commit`] (the sequential reference lets later
//! particles see earlier particles' new state).
//!
//! # Edge mode
//!
//! `config.edge_mode` reaches the kernels as the `edge_mode` scalar and every
//! destination goes through [`do_move_at`], which is `Simulation::do_move`
//! (`Simulation.cpp:1508-1532`) plus `Simulation::move`'s out-of-band kill
//! (`Simulation.cpp:1556-1560`):
//!
//! * `EDGE_LOOP` folds the requested pixel into the `CELL .. res-CELL` band
//!   (`Simulation.cpp:1509-1521`) — and `PlanMove` folds its own samples and
//!   its final velocity target the same way (`Simulation.cpp:2191-2216`) — so
//!   a particle that leaves one edge reappears on the opposite one;
//! * `EDGE_SOLID` leaves the step blocked: the mode stamps `WL_WALL` on the
//!   coarse border ring ([`SimBuffers::apply_edge_mode`],
//!   `Simulation.cpp:783-796`), which `IsWallBlocking` turns into a rejected
//!   `eval_move`;
//! * `EDGE_VOID` lets the step succeed and then kills the particle because the
//!   new pixel is outside the movement band, which is what
//!   `Simulation::move`'s `kill_part` does.
//!
//! Nothing is read back: the mode is a host scalar and the arbitration is the
//! same `cell_winner` pass regardless of it.
//!
//! # Module map
//!
//! * [`velocity`] — the velocity kernel and its stage helpers: intent reset,
//!   gravity, the diffusion/velocity integration, the `PlanMove` clamp, the
//!   per-`Falldown` destination search and the photon refraction machinery.
//! * [`liquid`] — the shared `PlanMove` + `MovementPhase` chain for
//!   `Falldown >= 1` and the two lateral scans.
//! * [`commit`] — the commit kernel and the two spatial-map rebuild kernels.
//!
//! # Buffer tables
//!
//! Kernel bodies cannot reference host `const` items, so the numeric literals
//! in the kernels mirror [`tpt_core::constants`]:
//!
//! | literal | constant |
//! |---|---|
//! | `4` | [`CELL`] (pixels per air cell) |
//! | `0x1FF` / `9` | `pmap_pack` type mask / id shift |
//! | `0x7FFFFF` | `pmap_pack` id field (`part_id + 1`) and the winner id field |
//! | `0x01`, `0x02`, `0x08`, `0x10` | `TYPE_PART`, `TYPE_LIQUID`, `TYPE_GAS`, `TYPE_ENERGY` |
//! | `0x10`, `0x02`, `0x01` | `FLAG_RIGID_PIXEL`, `FLAG_SKIPMOVE`, `FLAG_STAGNANT` |
//! | `1`, `2`, `4`, `6`, `7`, `8`, `9`, `10`, `13`, `15`, `18` | `WL_WALLELEC`, `WL_EWALL`, `WL_STREAM`, `WL_ALLOWLIQUID`, `WL_DESTROYALL`, `WL_WALL`, `WL_ALLOWAIR`, `WL_ALLOWPOWDER`, `WL_ALLOWGAS`, `WL_ALLOWENERGY`, `WL_STASIS` |
//! | `2` | `ISTP` = `CELL / 2`, the velocity sampling step of `PlanMove` |
//! | `30` / `10` | `rt`, the liquid scan length (`Simulation.cpp:3196-3201`) |
//! | `0` / `1` / `2` | `EDGE_VOID` / `EDGE_SOLID` / `EDGE_LOOP` (`SimulationSettings.h:3-8`) |
//! | `23` / `75` | `PT_VOID` / `PT_PVOD` |
//! | `PT_NUM` | derived from the kernel-side `PT_NUM_U32` mirror |
//! | `PT_NUM * 12` | `CAN_MOVE_OFFSET`, the can-move matrix inside `move_lut` |

mod commit;
mod liquid;
mod velocity;

use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

use tpt_core::constants::*;
use tpt_core::elements::{ElementRegistry, GpuElementProps};

use crate::buffers::SimBuffers;
use crate::buffers::gpu_sync::upload;

pub use commit::{clear_spatial_maps_kernel, commit_move_kernel, scatter_spatial_maps_kernel};
pub use velocity::velocity_and_intent_kernel;

/// `intent_target` value meaning "this particle stays where it is".
pub const NO_MOVE_INTENT: u32 = 0xFFFF_FFFF;

/// `intent_target` flag: the half-silvered-mirror rule forced this entry to
/// share the pixel even though `eval_move` rejects it
/// (`Simulation.cpp:1205-1206`).
///
/// A photon inside `BMTL`, or one whose 1-in-2 roll passes the mirror, still
/// moves into the blocked pixel (`e = 2`). The bit marks that decision for the
/// pair pass and the commit pass, which both re-check `can_move` from the
/// pre-move state. `NO_MOVE_INTENT`, [`DIE_INTENT`] and the blocked-photon
/// marker (`0x8000_0000 | pixel`) all keep bit 30 clear, so the four states
/// stay disjoint.
pub const INTENT_HALF_MIRROR: u32 = 0x4000_0000;
/// Pixel field of `intent_target` (bits `0..30`, below the two intent flags).
pub const INTENT_PIXEL_MASK: u32 = 0x3FFF_FFFF;

/// Mask of the element weight-rank field of the winner key (`9` bits).
const RANK_MASK: u32 = 0x1FF;

/// Words per element in the packed `move_lut`:
/// `[properties, falldown, weight_rank, heat_conduct, advection, air_drag,
/// air_loss, loss, collision, gravity, diffusion, hot_air]`.
///
/// The first four words are the movement metadata; the remaining eight are the
/// float physics knobs the velocity kernel integrates (`Element` fields, see
/// `Simulation::UpdateParticles`, `Simulation.cpp:2361-2377`) plus the
/// `HeatConduct` the blocked-photon heat exchange needs
/// (`Simulation.cpp:1220-1228`, `SimulationData.cpp:338-341`). Both tables live
/// in one buffer because the velocity kernel is already at the 16 storage
/// buffer limit and still has to bind `part_ctype` / `part_tmp`.
const META_WORDS: usize = 12;
/// Start of the flat `PT_NUM * PT_NUM` can-move matrix inside `move_lut`.
///
/// Kernel bodies cannot see the host constant, so [`can_move_at`] mirrors the
/// offset and the row stride from [`PT_NUM_U32`] instead.
const CAN_MOVE_OFFSET: usize = PT_NUM * META_WORDS;
/// `PT_NUM` as a `u32` for kernel-side range checks.
///
/// Mirrored for the kernels the same way `thermal_kernel` does it; every
/// `move_lut` index and element-id guard is derived from this, so growing
/// `tpt_core::constants::PT_NUM` needs no literal edits here.
const PT_NUM_U32: u32 = PT_NUM as u32;

// `eval_move`'s `can_move == 3` switch (`Simulation.cpp:1126-1177`). Kernel
// bodies cannot name host items directly, so each id is mirrored through a
// `u32` constant; the `assert!`s keep the mirrors honest.
const PT_VOID_U32: u32 = PT_VOID;
const PT_PVOD_U32: u32 = PT_PVOD;
const PT_LCRY_U32: u32 = PT_LCRY;
const PT_GPMP_U32: u32 = PT_GPMP;
const PT_INVIS_U32: u32 = PT_INVIS;
const PT_SWCH_U32: u32 = PT_SWCH;
/// `PT_WOOD` / `PT_SAWD`, the `try_move` saw rule mirrors
/// (`Simulation.cpp:1211-1218`).
const PT_WOOD_U32: u32 = PT_WOOD;
const PT_SAWD_U32: u32 = PT_SAWD;
const W_EHOLE: u32 = WL_EHOLE as u32;
const _: () = assert!(PT_VOID == 23);
const _: () = assert!(PT_PVOD == 75);
const _: () = assert!(PT_LCRY == 58);
const _: () = assert!(PT_GPMP == 43);
const _: () = assert!(PT_INVIS == 124);
const _: () = assert!(PT_SWCH == 60);
const _: () = assert!(WL_EHOLE == 12);
const _: () = assert!(PT_WOOD == PT_WOOD_U32);
const _: () = assert!(PT_SAWD == PT_SAWD_U32);

/// `Simulation.cpp:1211-1218`: a move that `try_move` rejects because the
/// destination is `WOOD` saws that wood into `SAWD` when the mover's speed
/// exceeds `5` px/frame (`sqrt(vx^2+vy^2) > 5`, squared here to avoid a
/// square root).
///
/// The rule runs on the *rejected* probe, before any fallback probe succeeds,
/// so callers invoke it for every `DO_MOVE_BLOCKED` result. `pmap` is the
/// pre-move map, whose type field names the wood.
#[cube]
pub fn saw_wood_at(
    pmap: &[u32],
    part_ty: &mut [u32],
    nx: i32,
    ny: i32,
    vx: f32,
    vy: f32,
    xr: i32,
    yr: i32,
) {
    if nx >= 0i32 && ny >= 0i32 && nx < xr && ny < yr {
        let m = pmap[(ny * xr + nx) as usize];
        // `pmap_pack` leaves the id empty for an unoccupied pixel.
        if m >= 512u32 && (m & 0x1FFu32) == PT_WOOD_U32 && vx * vx + vy * vy > 25.0f32 {
            let slot = ((m >> 9u32) - 1u32) as usize;
            part_ty[slot] = PT_SAWD_U32;
        }
    }
}

// `can_move_at` cannot reference `META_WORDS`, so it spells the stride out as
// `12usize`; keep the mirror honest.
const _: () = assert!(META_WORDS == 12);
/// Words per element in the physics record:
/// `[advection, air_drag, air_loss, loss, collision, gravity, diffusion, hot_air]`.
///
/// The words live in the tail of `move_lut`; this length also sizes the
/// legacy `gpu_elem_phys` allocation, whose word 0 carries the `LOVE` gate.
const PHYS_WORDS: usize = 8;

/// Index of the first physics word inside one `move_lut` element record.
const PHYS_OFFSET: usize = 4;
const _: () = assert!(PHYS_OFFSET + PHYS_WORDS == META_WORDS);

/// Word index of `properties` / `falldown` / `weight_rank` / `HeatConduct`.
const META_PROPERTIES: usize = 0;
const META_FALLDOWN: usize = 1;
const META_RANK: usize = 2;
const META_HEAT_CONDUCT: usize = 3;

/// Words of the legacy standalone per-element physics table.
///
/// The velocity kernel no longer binds it (the physics words moved into
/// `move_lut`); the size is still exported because `buffers::layout` allocates
/// `gpu_elem_phys` with it.
pub(crate) const fn elem_phys_len() -> usize {
    PT_NUM * PHYS_WORDS
}

/// Words of the photon-mask table the commit kernel binds.
pub(crate) const fn photon_mask_len() -> usize {
    PT_NUM
}
/// Threads per cube for every movement kernel.
///
/// Every kernel is a flat `ABSOLUTE_POS` scan with no shared memory, so the
/// only constraint on the cube size is the backend limit (`256` on wgpu), and
/// a larger cube directly reduces the cube count: the dispatch dimension is
/// capped at `65 535` cubes, and a `2048 x 2048` canvas (4 194 304 pixels) or
/// a saturated `8 388 607`-slot pool needs `16 384` / `32 768` cubes at `256`
/// threads but `65 536` / `131 072` at `64`.
const BLOCK: u32 = 256;

/// Unclaims every pixel of the canvas.
///
/// `cell_winner` is `xres * yres` `Atomic<u32>` words holding
/// `(weight_rank << 23) | (part_id + 1)`; `0` means "nobody claimed this pixel".
#[cube(launch_unchecked)]
pub fn clear_move_intents_kernel(cell_winner: &mut [Atomic<u32>], num_pixels: u32) {
    let idx = ABSOLUTE_POS;
    if idx < num_pixels as usize {
        cell_winner[idx].store(0u32);
    }
}

/// A moving particle drags the air in its coarse cell.
///
/// `Simulation.cpp:2334-2338`:
///
/// ```text
/// vx[y/CELL][x/CELL] = vx[y/CELL][x/CELL] * AirLoss + AirDrag * parts[i].vx;
/// vy[y/CELL][x/CELL] = vy[y/CELL][x/CELL] * AirLoss + AirDrag * parts[i].vy;
/// ```
///
/// The reference applies this in particle-id order, so several particles in one
/// cell compose the recurrence `air <- air * AirLoss + AirDrag * v`. A parallel
/// pass cannot reproduce that composition without a per-cell ordered reduction;
/// the store below is the documented last-writer-wins approximation. `AirDrag`
/// is zero for nearly every element (`Element.cpp` default), so the common
/// frame writes nothing.
///
/// The pass runs before the velocity kernel because the same particle then
/// reads `vx[cell]` for its advection term (`Simulation.cpp:2383-2384`).
/// Particles the reference kills or freezes before the deposit (off-screen,
/// inside a killing wall, under unpowered `WL_STASIS`) do not write either.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn air_deposit_kernel(
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_vx: &[f32],
    part_vy: &[f32],
    part_flags: &[u32],
    move_lut: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    air_vx: &mut [f32],
    air_vy: &mut [f32],
    num_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let i = ABSOLUTE_POS;
    if i >= num_slots as usize {
        terminate!();
    }
    let ty = part_ty[i];
    if ty == 0u32 || ty >= PT_NUM_U32 {
        terminate!();
    }
    let flags = part_flags[i];
    // Rigid pixels are owned by the rigid-body solver and `FLAG_SKIPMOVE`
    // particles never enter the movement pass, so neither stirs the air.
    if (flags & 0x10u32) != 0u32 || (flags & 0x02u32) != 0u32 {
        terminate!();
    }
    let xi = i32::cast_from(part_x[i] + 0.5f32);
    let yi = i32::cast_from(part_y[i] + 0.5f32);
    if xi < 4i32 || yi < 4i32 || xi >= xres as i32 - 4i32 || yi >= yres as i32 - 4i32 {
        terminate!();
    }
    let cx = (xi as u32) / 4u32;
    let cy = (yi as u32) / 4u32;
    let cell = (cy * xcells + cx) as usize;
    if wall_bmap[cell] == 18u32 && (wall_emap[cell] & 0xFFu32) < 8u32 {
        terminate!();
    }
    // `move_lut` record: word 4 is the first physics word; `AirDrag` is +1 and
    // `AirLoss` is +2 (`[advection, air_drag, air_loss, loss, ...]`).
    let base = (ty as usize) * 12usize + 4usize;
    let air_drag = f32::reinterpret(move_lut[base + 1usize]);
    let air_loss = f32::reinterpret(move_lut[base + 2usize]);
    air_vx[cell] = air_vx[cell] * air_loss + air_drag * part_vx[i];
    air_vy[cell] = air_vy[cell] * air_loss + air_drag * part_vy[i];
}

/// `Simulation::is_wire_off` (`Simulation.cpp:861-864`): a wire wall that is
/// still uncharged.
///
/// ```text
/// (bmap==WL_DETECT || bmap==WL_EWALL || bmap==WL_ALLOWLIQUID
///  || bmap==WL_WALLELEC || bmap==WL_ALLOWALLELEC || bmap==WL_EHOLE
///  || bmap==WL_STASIS) && emap < 8
/// ```
#[cube]
fn is_wire_off(wall: u32, emap: u32) -> bool {
    is_wire_wall(wall) && (emap & EMAP_VALUE_MASK) < 8u32
}

/// `Simulation::is_wire` (`Simulation.cpp:861-864`): the wall is one of the
/// conductor walls, charged or not.
#[cube]
fn is_wire_wall(wall: u32) -> bool {
    wall == 3u32
        || wall == 2u32
        || wall == 6u32
        || wall == 1u32
        || wall == 5u32
        || wall == 12u32
        || wall == 18u32
}

/// Emap value `set_emap` writes (`Simulation.cpp:951`).
const EMAP_CHARGED: u32 = 16;
/// Transient marker set by `set_emap` on the two ends of the span it filled.
///
/// The recursive traversal needs the span endpoints to decide which rows above
/// and below continue the wire tree. The device closure kernel cannot keep a
/// stack, so the endpoints live in the top two bits of the emap word until the
/// closure finishes and clears them. `x1 == x2` (the reference's singleton
/// test) is both bits on the same cell.
const EMAP_ENDPOINT_LEFT: u32 = 0x80;
const EMAP_ENDPOINT_RIGHT: u32 = 0x40;
const EMAP_ENDPOINT: u32 = EMAP_ENDPOINT_LEFT | EMAP_ENDPOINT_RIGHT;
/// Value bits of an `emap` word (the endpoint markers are transient).
const EMAP_VALUE_MASK: u32 = 0x3F;
/// Cell was charged by the current `set_emap` closure.
#[cube]
fn is_charged_now(emap: u32) -> bool {
    (emap & EMAP_VALUE_MASK) == EMAP_CHARGED
}

/// Cell is one of the two endpoints of a span filled in this closure.
#[cube]
fn is_span_endpoint(emap: u32) -> bool {
    (emap & EMAP_ENDPOINT) != 0u32
}

/// `bmap` cell is a conductor wall, with bounds checks.
#[cube]
fn wire_at(wall_bmap: &[u32], x: i32, y: i32, xcells: u32, ycells: u32) -> bool {
    if x < 0i32 || y < 0i32 || x >= xcells as i32 || y >= ycells as i32 {
        false
    } else {
        is_wire_wall(wall_bmap[(y as u32 * xcells + x as u32) as usize])
    }
}

/// C++ `set_emap`'s "ladder" shortcut, upwards.
///
/// A single-cell charged span at `(x, target_row + 2)` jumps over a charged
/// three-wide cap at `target_row + 1` and charges the lone wire at
/// `target_row` (`Simulation.cpp:955-965`).
#[cube]
#[allow(clippy::too_many_arguments)]
fn ladder_up(
    wall_bmap: &[u32],
    source_is_singleton: bool,
    x: i32,
    target_row: i32,
    xcells: u32,
    ycells: u32,
) -> bool {
    source_is_singleton
        && wire_at(wall_bmap, x - 1i32, target_row + 1i32, xcells, ycells)
        && wire_at(wall_bmap, x, target_row + 1i32, xcells, ycells)
        && wire_at(wall_bmap, x + 1i32, target_row + 1i32, xcells, ycells)
        && !wire_at(wall_bmap, x - 1i32, target_row, xcells, ycells)
        && wire_at(wall_bmap, x, target_row, xcells, ycells)
        && !wire_at(wall_bmap, x + 1i32, target_row, xcells, ycells)
}

/// C++ `set_emap`'s "ladder" shortcut, downwards
/// (`Simulation.cpp:975-985`), symmetric to [`ladder_up`].
#[cube]
#[allow(clippy::too_many_arguments)]
fn ladder_down(
    wall_bmap: &[u32],
    source_is_singleton: bool,
    x: i32,
    target_row: i32,
    xcells: u32,
    ycells: u32,
) -> bool {
    source_is_singleton
        && wire_at(wall_bmap, x - 1i32, target_row - 1i32, xcells, ycells)
        && wire_at(wall_bmap, x, target_row - 1i32, xcells, ycells)
        && wire_at(wall_bmap, x + 1i32, target_row - 1i32, xcells, ycells)
        && !wire_at(wall_bmap, x - 1i32, target_row, xcells, ycells)
        && wire_at(wall_bmap, x, target_row, xcells, ycells)
        && !wire_at(wall_bmap, x + 1i32, target_row, xcells, ycells)
}

/// `WL_DETECT` charging: a particle standing in an uncharged detector fires it.
///
/// `Simulation.cpp:2330-2331`:
///
/// ```text
/// if (bmap[y/CELL][x/CELL]==WL_DETECT && emap[y/CELL][x/CELL]<8)
///     set_emap(x/CELL, y/CELL);
/// ```
///
/// The horizontal span fill is the first half of `set_emap`
/// (`Simulation.cpp:928-983`); it marks the span endpoints for
/// [`emap_closure_kernel`], which ports the recursive up/down children without
/// a device stack. The trigger order matches the reference: it runs after the
/// kill/stasis gates (the caller's gates) and before the velocity update, so a
/// particle entering a detector charges it for the next frame's electronics
/// pass.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn detect_charge_kernel(
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_flags: &[u32],
    wall_bmap: &[u32],
    wall_emap: &mut [u32],
    num_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let i = ABSOLUTE_POS;
    if i >= num_slots as usize {
        terminate!();
    }
    let ty = part_ty[i];
    if ty == 0u32 || ty >= PT_NUM_U32 {
        terminate!();
    }
    // Rigid pixels are a port construct, not reference particles.
    if (part_flags[i] & 0x10u32) != 0u32 {
        terminate!();
    }
    let xi = i32::cast_from(part_x[i] + 0.5f32);
    let yi = i32::cast_from(part_y[i] + 0.5f32);
    // `Simulation.cpp:2312-2317`: an off-margin particle is killed before the
    // detector check.
    if xi < 4i32 || yi < 4i32 || xi >= xres as i32 - 4i32 || yi >= yres as i32 - 4i32 {
        terminate!();
    }
    let cx = (xi as u32) / 4u32;
    let cy = (yi as u32) / 4u32;
    let row = cy * xcells;
    let cell = (row + cx) as usize;
    if wall_bmap[cell] != 3u32 {
        terminate!();
    }
    if (wall_emap[cell] & EMAP_VALUE_MASK) >= 8u32 {
        terminate!();
    }
    // `set_emap`: walk left and right over the connected wire span.
    let mut x1 = cx;
    let mut left = true;
    while left && x1 > 0u32 {
        let c = (row + x1 - 1u32) as usize;
        if is_wire_off(wall_bmap[c], wall_emap[c]) {
            x1 -= 1u32;
        } else {
            left = false;
        }
    }
    let mut x2 = cx;
    let mut right = true;
    while right && x2 + 1u32 < xcells {
        let c = (row + x2 + 1u32) as usize;
        if is_wire_off(wall_bmap[c], wall_emap[c]) {
            x2 += 1u32;
        } else {
            right = false;
        }
    }
    let mut x = x1;
    while x <= x2 {
        wall_emap[(row + x) as usize] = EMAP_CHARGED;
        x += 1u32;
    }
    // `set_emap` needs to know where the span ends when it looks for children;
    // the closure kernel reads these markers.
    wall_emap[(row + x1) as usize] |= EMAP_ENDPOINT_LEFT;
    wall_emap[(row + x2) as usize] |= EMAP_ENDPOINT_RIGHT;
}

/// Device gate for the wire closure.
///
/// [`emap_closure_kernel`] is a single-threaded sweep of the coarse grid. A
/// canvas with no wires, or with every wire already charged, has nothing to
/// propagate, but the sweep still has to walk every cell to discover that.
/// Both of the closure's activations need an uncharged wire span *and* a
/// charged wire to feed it, so this pass reduces those two bits on the device:
/// bit 0 = an uncharged wire span exists, bit 1 = some wire is charged. The
/// closure then returns immediately unless both bits are set.
///
/// The test is a superset of the closure's activation condition (it never
/// distinguishes *which* span feeds which), so it can only run the sweep when
/// it might change something, never skip a propagation.
#[cube(launch_unchecked)]
fn emap_gate_kernel(wall_bmap: &[u32], wall_emap: &[u32], gate: &mut [Atomic<u32>], cells: u32) {
    let idx = ABSOLUTE_POS as u32;
    if idx < cells {
        let wall = wall_bmap[idx as usize];
        let emap = wall_emap[idx as usize];
        if is_wire_off(wall, emap) {
            gate[0].fetch_or(1u32);
        }
        if is_charged_now(emap) {
            gate[0].fetch_or(2u32);
        }
    }
}

/// Clears the closure's transient span-endpoint markers.
///
/// The markers only exist for the duration of one closure pass; leaving them in
/// the emap would survive until the next host upload and the `emap & 0xFF`
/// checks in the thermal/render kernels would see `144` instead of `16`. One
/// thread per cell keeps this off the single-threaded closure (it used to be a
/// sequential pass over every cell, whether or not the closure ran).
#[cube(launch_unchecked)]
fn strip_emap_endpoints_kernel(wall_emap: &mut [u32], cells: u32) {
    let idx = ABSOLUTE_POS as u32;
    if idx < cells {
        wall_emap[idx as usize] &= !EMAP_ENDPOINT;
    }
}

/// Recursive half of `set_emap` (`Simulation.cpp:943-983`), without recursion.
///
/// The reference walks the wire tree depth-first: from a filled span it visits
/// the uncharged wire runs above and below, fills each one, and repeats. A
/// device kernel cannot recurse and has no local stack, so this single-thread
/// pass sweeps the coarse grid and fills every run that a neighbouring charged
/// span would reach, repeating until a sweep makes no change. The scan order
/// (top to bottom) fills downward branches in the same sweep and upward
/// branches on the next one, so the number of sweeps is bounded by the canvas
/// height.
///
/// That sweep is O(cells) on one thread, so it only runs when
/// [`emap_gate_kernel`] found both an uncharged span and a charged wire; the
/// gate word is consumed and cleared here.
#[cube(launch_unchecked)]
fn emap_closure_kernel(
    wall_bmap: &[u32],
    wall_emap: &mut [u32],
    gate: &mut [Atomic<u32>],
    xcells: u32,
    ycells: u32,
) {
    if ABSOLUTE_POS != 0usize {
        terminate!();
    }
    // Consume (and clear) the gate for the next tick. A missing bit means no
    // uncharged span can be reached from a charged one, so the sweep below
    // would not change a single cell.
    if gate[0].fetch_max(0u32) != 3u32 {
        gate[0].store(0u32);
        terminate!();
    }
    gate[0].store(0u32);

    let mut pass = 0u32;
    let max_passes = ycells + 2u32;
    let mut changed = true;
    while changed && pass < max_passes {
        changed = false;
        let mut y = 0u32;
        while y < ycells {
            let row = y * xcells;
            let mut x = 0u32;
            while x < xcells {
                let cell = (row + x) as usize;
                if !is_wire_off(wall_bmap[cell], wall_emap[cell]) {
                    x += 1u32;
                } else {
                    // The whole contiguous uncharged run is one `set_emap` span.
                    let x1 = x;
                    let mut x2 = x;
                    let mut grow = true;
                    while grow && x2 + 1u32 < xcells {
                        let next = (row + x2 + 1u32) as usize;
                        if is_wire_off(wall_bmap[next], wall_emap[next]) {
                            x2 += 1u32;
                        } else {
                            grow = false;
                        }
                    }

                    let mut activate = false;
                    // Child above: source span one row below (C++ upward branch).
                    if y + 1u32 < ycells {
                        let source_y = (y + 1u32) as i32;
                        let mut xx = x1;
                        while xx <= x2 && !activate {
                            let source = ((y + 1u32) * xcells + xx) as usize;
                            let source_emap = wall_emap[source];
                            let source_is_singleton =
                                (source_emap & EMAP_ENDPOINT) == EMAP_ENDPOINT;
                            if is_charged_now(source_emap)
                                // The reference's `if` ladder replaces the
                                // regular upward branch, so the cap row it
                                // would have charged must stay untouched.
                                && !ladder_up(
                                    wall_bmap,
                                    source_is_singleton,
                                    xx as i32,
                                    source_y - 2i32,
                                    xcells,
                                    ycells,
                                )
                                && (is_span_endpoint(source_emap)
                                    || y + 1u32 == ycells - 1u32
                                    || wire_at(wall_bmap, xx as i32 - 1i32, y as i32, xcells, ycells)
                                    || wire_at(wall_bmap, xx as i32 + 1i32, y as i32, xcells, ycells)
                                    || wire_at(
                                        wall_bmap,
                                        xx as i32 - 1i32,
                                        source_y + 1i32,
                                        xcells,
                                        ycells,
                                    )
                                    || !wire_at(
                                        wall_bmap,
                                        xx as i32,
                                        source_y + 1i32,
                                        xcells,
                                        ycells,
                                    )
                                    || wire_at(
                                        wall_bmap,
                                        xx as i32 + 1i32,
                                        source_y + 1i32,
                                        xcells,
                                        ycells,
                                    ))
                            {
                                activate = true;
                            }
                            xx += 1u32;
                        }
                    }

                    // Child below: source span one row above (C++ downward branch).
                    if !activate && y > 0u32 {
                        let source_y = (y - 1u32) as i32;
                        let mut xx = x1;
                        while xx <= x2 && !activate {
                            let source = ((y - 1u32) * xcells + xx) as usize;
                            let source_emap = wall_emap[source];
                            let source_is_singleton =
                                (source_emap & EMAP_ENDPOINT) == EMAP_ENDPOINT;
                            if is_charged_now(source_emap)
                                && !ladder_down(
                                    wall_bmap,
                                    source_is_singleton,
                                    xx as i32,
                                    (y + 1u32) as i32,
                                    xcells,
                                    ycells,
                                )
                                && (is_span_endpoint(source_emap)
                                    || wire_at(
                                        wall_bmap,
                                        xx as i32 - 1i32,
                                        y as i32,
                                        xcells,
                                        ycells,
                                    )
                                    || wire_at(
                                        wall_bmap,
                                        xx as i32 + 1i32,
                                        y as i32,
                                        xcells,
                                        ycells,
                                    )
                                    || wire_at(
                                        wall_bmap,
                                        xx as i32 - 1i32,
                                        source_y - 1i32,
                                        xcells,
                                        ycells,
                                    )
                                    || !wire_at(
                                        wall_bmap,
                                        xx as i32,
                                        source_y - 1i32,
                                        xcells,
                                        ycells,
                                    )
                                    || wire_at(
                                        wall_bmap,
                                        xx as i32 + 1i32,
                                        source_y - 1i32,
                                        xcells,
                                        ycells,
                                    ))
                            {
                                activate = true;
                            }
                            xx += 1u32;
                        }
                    }

                    // Ladder jumps: source two rows away, cap row in between.
                    if !activate {
                        let mut xx = x1;
                        while xx <= x2 && !activate {
                            let mut up_singleton = false;
                            if y + 2u32 < ycells {
                                up_singleton = (wall_emap[((y + 2u32) * xcells + xx) as usize]
                                    & EMAP_ENDPOINT)
                                    == EMAP_ENDPOINT;
                            }
                            let mut down_singleton = false;
                            if y >= 2u32 {
                                down_singleton = (wall_emap[((y - 2u32) * xcells + xx) as usize]
                                    & EMAP_ENDPOINT)
                                    == EMAP_ENDPOINT;
                            }
                            if ladder_up(
                                wall_bmap,
                                up_singleton,
                                xx as i32,
                                y as i32,
                                xcells,
                                ycells,
                            ) || ladder_down(
                                wall_bmap,
                                down_singleton,
                                xx as i32,
                                y as i32,
                                xcells,
                                ycells,
                            ) {
                                activate = true;
                            }
                            xx += 1u32;
                        }
                    }

                    if activate {
                        let mut xx = x1;
                        while xx <= x2 {
                            wall_emap[(row + xx) as usize] = EMAP_CHARGED;
                            xx += 1u32;
                        }
                        wall_emap[(row + x1) as usize] |= EMAP_ENDPOINT_LEFT;
                        wall_emap[(row + x2) as usize] |= EMAP_ENDPOINT_RIGHT;
                        changed = true;
                    }
                    x = x2 + 1u32;
                }
            }
            y += 1u32;
        }
        pass += 1u32;
    }
}

/// GPU mirror of `Simulation::eval_move` (`Simulation.cpp:1107-1191`):
/// `0` = blocked, `1` = swap / eat, `2` = share the pixel.
///
/// The `can_move == 3` switch is evaluated without the destination's
/// `life`/`tmp`/`ctype`/air pressure, which the velocity kernel cannot bind
/// inside its 16-buffer budget:
///
/// * `PT_VOID` / `PT_PVOD` resolve to `1` here, which is either the eat result
///   or the reference's `!eval` block — `PlanMove` rejects both, so the plan
///   never crosses a void (`Simulation.cpp:2226-2235`).
/// * `PT_LCRY` / `PT_GPMP` / `PT_INVIS` / `PT_SWCH` return `2` (the
///   pass-through state); their blocked states need the occupant's
///   `life`/`tmp`/pressure, so a liquid may plan through a closed shutter and
///   is then refused by the commit pass, which resolves `can == 3` with the
///   real state (`commit::resolve_can`).
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn eval_move_at(
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    props: u32,
    ty: u32,
    nx: i32,
    ny: i32,
    xr: i32,
    yr: i32,
    xcells: u32,
) -> u32 {
    let mut result = 0u32;
    if nx >= 0i32 && nx < xr && ny >= 0i32 && ny < yr {
        let t_idx = (ny * xr + nx) as u32;
        let d_val = pmap[t_idx as usize];
        let d_ty = d_val & 0x1FFu32;
        let wcell = (((ny as u32) / 4u32) * xcells + ((nx as u32) / 4u32)) as usize;
        let wall = wall_bmap[wcell];
        let emap = wall_emap[wcell];
        let mut can = can_move_at(move_lut, ty as usize, d_ty);
        if can == 3u32 {
            if d_ty == PT_VOID_U32 || d_ty == PT_PVOD_U32 {
                // `PT_VOID` / `PT_PVOD`: the eat result (`Simulation.cpp:1149-1162`).
                can = 1u32;
            } else if d_ty == PT_LCRY_U32
                || d_ty == PT_GPMP_U32
                || d_ty == PT_INVIS_U32
                || d_ty == PT_SWCH_U32
            {
                // `PT_LCRY` / `PT_GPMP` / `PT_INVIS` / `PT_SWCH`: pass state.
                can = 2u32;
            } else {
                // `eval_move`'s `default:` maps an unexpected 3 to 1.
                can = 1u32;
            }
        }
        result = can;
        if wall != 0u32 {
            if wall_blocks_particle(wall, emap, props) {
                result = 0u32;
            } else if wall == W_EHOLE && (emap & 0xFFu32) == 0u32 {
                // Closed E-hole: solids cannot pass, everything else shares the
                // pixel (`Simulation.cpp:1185-1189`).
                let dest_props = move_lut[(d_ty as usize) * 12usize];
                if (props & 0x4u32) == 0u32 && (dest_props & 0x4u32) == 0u32 {
                    result = 2u32;
                }
            }
        }
    }
    result
}

/// Bounds-checked `eval_move` predicate: `true` when a particle of type `ty`
/// may enter pixel `(nx, ny)` (`Simulation.cpp:1107`).
///
/// Out-of-bounds pixels are reported as blocked (`eval_move` returns `0` for
/// them as well), so callers never have to bounds-check first. This is the
/// `PlanMove` probe rule; `do_move` uses [`do_move_at`], which mirrors
/// `try_move`'s out-of-bounds short circuit instead.
#[cube]
#[allow(clippy::too_many_arguments)]
fn move_ok_at(
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    props: u32,
    ty: u32,
    nx: i32,
    ny: i32,
    xr: i32,
    yr: i32,
    xcells: u32,
) -> bool {
    eval_move_at(
        pmap, wall_bmap, wall_emap, move_lut, props, ty, nx, ny, xr, yr, xcells,
    ) != 0u32
}

/// `EDGE_VOID` / `EDGE_SOLID` / `EDGE_LOOP` (`SimulationSettings.h:3-8`).
///
/// `EDGE_LOOP` is the only mode the movement kernels treat specially: it folds
/// every destination into the `CELL .. res-CELL` band, exactly like `do_move`
/// (`Simulation.cpp:1509-1521`) and `PlanMove` (`Simulation.cpp:2191-2216`).
/// Without that fold a destination outside the band is a kill (`EDGE_VOID`, via
/// `Simulation::move`'s bounds check) or a rejected step (`EDGE_SOLID`, whose
/// stamped `WL_WALL` ring `IsWallBlocking` already rejects —
/// [`SimBuffers::apply_edge_mode`], `Simulation.cpp:783-796`).
pub const EDGE_VOID: u32 = 0;
pub const EDGE_SOLID: u32 = 1;
pub const EDGE_LOOP: u32 = 2;
const _: () = assert!(EDGE_VOID == 0 && EDGE_SOLID == 1 && EDGE_LOOP == 2);

/// `CELL` as the `i32` the movement kernels compare pixels against.
const CELL_I32: i32 = 4;
const _: () = assert!(CELL_I32 as u32 == CELL);

/// `intent_target` value meaning "this particle dies in this pass".
///
/// Used by [`do_move_at`]'s out-of-band kill, by `Simulation::move`'s margin
/// check, and by the photon refraction path when `get_wavelength_bin` or
/// `get_normal_interp` fails (`Simulation.cpp:2975-2985`).
pub const DIE_INTENT: u32 = 0xFFFF_FFFEu32;

/// [`do_move_at`] outcome: the destination is impassable, try the next probe.
pub const DO_MOVE_BLOCKED: u32 = 0;
/// [`do_move_at`] outcome: the particle moves to the returned pixel.
pub const DO_MOVE_OK: u32 = 1;
/// [`do_move_at`] outcome: `Simulation::move` kills the particle
/// (`Simulation.cpp:1556-1560`).
pub const DO_MOVE_DIE: u32 = 2;

/// `remainder_p` (`Simulation.cpp:44-47`): `fmod(x, y)` shifted into `[0, y)`.
///
/// ```text
/// return std::fmod(x, y) + (x>=0 ? 0 : y);
/// ```
#[cube]
fn remainder_p(x: f32, y: f32) -> f32 {
    let mut out = x - (x / y).floor() * y;
    if out < 0.0f32 {
        out += y;
    } else if out >= y {
        out -= y;
    }
    out
}

/// Integer form of `do_move`'s `EDGE_LOOP` wrap (`Simulation.cpp:1509-1521`):
/// a pixel outside the `CELL .. res-CELL` band folds onto the far side.
///
/// The reference wraps the float `nxf` and re-rounds it; with integer
/// positions the rounded pixel is what folds, so the destination band is the
/// same `CELL .. res-CELL` the float wrap produces.
#[cube]
fn wrap_loop_pixel(n: i32, res: u32) -> i32 {
    let lo = CELL_I32;
    let span = res as i32 - 2i32 * CELL_I32;
    let mut out = n;
    if span > 0i32 && (n < lo || n >= lo + span) {
        out = lo + i32::cast_from(remainder_p((n - lo) as f32, span as f32));
    }
    out
}

/// `Simulation::do_move` (`Simulation.cpp:1508-1532`) plus the out-of-band kill
/// `Simulation::move` performs after a successful `try_move`
/// (`Simulation.cpp:1541-1570`).
///
/// Returns `(status, x, y)`:
///
/// * `EDGE_LOOP` folds the requested pixel into the `CELL .. res-CELL` band
///   first and then tests it, so a particle that leaves one edge appears on the
///   opposite one;
/// * otherwise an out-of-canvas pixel is *success*, because `try_move` returns
///   `1` for it without consulting `eval_move`, and `move` then kills the
///   particle;
/// * an in-canvas pixel that passes `eval_move` but lies outside the
///   `CELL .. res-CELL` band is a kill under `EDGE_VOID` (`move`'s bounds
///   check), which is how `EDGE_VOID` erases a particle that reaches the
///   margin; under `EDGE_SOLID` the same pixel is [`DO_MOVE_BLOCKED`], because
///   the reference's stamped `WL_WALL` ring already rejects it through
///   `IsWallBlocking` (`Simulation.cpp:1076-1099`);
/// * anything `eval_move` rejects is [`DO_MOVE_BLOCKED`], so the caller tries
///   its next probe — this is why `EDGE_SOLID`'s ring stops the particle
///   instead of killing it.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn do_move_at(
    pmap: &[u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    props: u32,
    ty: u32,
    nx: i32,
    ny: i32,
    xr: i32,
    yr: i32,
    xcells: u32,
    edge_mode: u32,
) -> (u32, i32, i32) {
    let mut mx = nx;
    let mut my = ny;
    if edge_mode == EDGE_LOOP {
        mx = wrap_loop_pixel(nx, xr as u32);
        my = wrap_loop_pixel(ny, yr as u32);
    }
    // Seeded from a literal so the local stays a `NativeExpand<u32>`
    // (AGENTS.md rule 4); `DO_MOVE_BLOCKED` is the same value.
    let mut status = 0u32;
    if mx < 0i32 || my < 0i32 || mx >= xr || my >= yr {
        if edge_mode != EDGE_LOOP {
            status = DO_MOVE_DIE;
        }
    } else if move_ok_at(
        pmap, wall_bmap, wall_emap, move_lut, props, ty, mx, my, xr, yr, xcells,
    ) {
        let in_band = mx >= CELL_I32 && my >= CELL_I32 && mx < xr - CELL_I32 && my < yr - CELL_I32;
        status = DO_MOVE_OK;
        if !in_band && edge_mode != EDGE_LOOP {
            // Out of the movement band but still inside the canvas: `move`
            // kills the particle unless something blocked the step first.
            // `EDGE_SOLID`'s `WL_WALL` ring is exactly that blocker
            // (`IsWallBlocking`), so the step is rejected; the ring itself is
            // stamped by `SimBuffers::apply_edge_mode`, and this arm keeps the
            // same outcome when a caller changed `config.edge_mode` without
            // re-stamping it.
            status = DO_MOVE_BLOCKED;
            if edge_mode == EDGE_VOID {
                status = DO_MOVE_DIE;
            }
        }
    }
    (status, mx, my)
}

/// `can_move[moving_ty][dest_ty]` from the packed `move_lut`.
///
/// The matrix starts at `CAN_MOVE_OFFSET` (`PT_NUM * 12` words) and is stored
/// row-major, so both the offset and the row stride follow the kernel-side
/// [`PT_NUM_U32`] mirror instead of a hard-coded `768` / `64` literal.
#[cube]
pub fn can_move_at(move_lut: &[u32], moving_ty: usize, dest_ty: u32) -> u32 {
    // `12usize` is `META_WORDS`; both `PT_NUM` factors follow the mirror.
    let offset = (PT_NUM_U32 as usize) * 12usize;
    move_lut[offset + (moving_ty * (PT_NUM_U32 as usize) + dest_ty as usize)]
}

/// Mirrors [`tpt_core::walls::is_wall_blocking_particle`] for the GPU.
///
/// `wall` is the wall id stored in `wall_bmap` (already decoded, `0` = none) and
/// `emap` the electrical state in `wall_emap`.
///
/// `Simulation::IsWallBlocking` (`Simulation.cpp:1076-1099`) has **no**
/// `WL_DESTROYALL` clause: a destroy-all wall lets the particle in so the next
/// tick's wall-kill list ([`wall_kills_particle`]) can erase it.
#[cube]
pub fn wall_blocks_particle(wall: u32, emap: u32, props: u32) -> bool {
    let mut blocked = false;
    if wall == 8u32 || wall == 1u32 || wall == 9u32 {
        // WL_WALL, WL_WALLELEC, WL_ALLOWAIR (air only).
        blocked = true;
    } else if wall == 6u32 {
        // WL_ALLOWLIQUID: only liquids pass.
        blocked = (props & 0x2u32) == 0u32;
    } else if wall == 10u32 {
        // WL_ALLOWPOWDER: only powders pass.
        blocked = (props & 0x1u32) == 0u32;
    } else if wall == 13u32 {
        // WL_ALLOWGAS: only gases pass.
        blocked = (props & 0x8u32) == 0u32;
    } else if wall == 15u32 {
        // WL_ALLOWENERGY: only energy particles pass.
        blocked = (props & 0x10u32) == 0u32;
    } else if wall == 2u32 {
        // WL_EWALL: solid while unpowered, transparent once charged.
        blocked = (emap & 0xFFu32) == 0u32;
    } else if wall == 3u32 {
        // `Simulation.cpp:1095-1096`: a detector wall is solid to solids.
        blocked = (props & 0x4u32) != 0u32;
    }
    blocked
}

/// The wall ids that kill a particle standing inside them
/// (`Simulation.cpp:2314-2330`), mirroring the explicit list in
/// `UpdateParticles` rather than [`wall_blocks_particle`]: `WL_DESTROYALL` is
/// not in `IsWallBlocking`'s solid set but does kill on the next tick.
///
/// `WL_STKM`/`WL_STKM2`/`WL_FIGH` are exempt in the reference and handled by
/// the caller.
#[cube]
pub fn wall_kills_particle(wall: u32, emap: u32, props: u32) -> bool {
    let mut kill = false;
    if wall == 8u32 || wall == 1u32 || wall == 9u32 || wall == 7u32 {
        // WL_WALL, WL_WALLELEC, WL_ALLOWAIR, WL_DESTROYALL.
        kill = true;
    } else if wall == 6u32 {
        kill = (props & 0x2u32) == 0u32; // WL_ALLOWLIQUID, non-liquids.
    } else if wall == 10u32 {
        kill = (props & 0x1u32) == 0u32; // WL_ALLOWPOWDER, non-powders.
    } else if wall == 13u32 {
        kill = (props & 0x8u32) == 0u32; // WL_ALLOWGAS, non-gases.
    } else if wall == 15u32 {
        kill = (props & 0x10u32) == 0u32; // WL_ALLOWENERGY, non-energy.
    } else if wall == 2u32 {
        kill = (emap & 0xFFu32) == 0u32; // WL_EWALL, unpowered.
    }
    kill
}

/// `index * salt`, kept behind a call so the product is never constant-folded.
///
/// In the unexpanded (host) build of a `#[cube]` function `ABSOLUTE_POS` is the
/// literal `2`, so writing `(i as u32) * 2654435769u32` inline makes the
/// compiler evaluate `2 * 2654435769` and reject the build with
/// `arithmetic_overflow`. Integer multiplication in the IR is modular, so this
/// is the wrapping product on every backend.
#[cube]
fn mix_index(index: u32, salt: u32) -> u32 {
    index * salt
}

/// PCG-style integer hash used for the deterministic diffusion kick.
#[cube]
pub fn pcg_hash(seed: u32) -> u32 {
    let state = seed * 747796405u32 + 2891336453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277803737u32;
    (word >> 22u32) ^ word
}

/// Sizes shared by every launch of one movement step.
struct MoveDims {
    capacity: usize,
    num_pixels: usize,
    num_cells: usize,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
}

/// Runs one lock-free movement step.
///
/// Launches `air_deposit_kernel`, `detect_charge_kernel`,
/// `emap_closure_kernel`, [`clear_move_intents_kernel`],
/// [`velocity_and_intent_kernel`] and [`commit_move_kernel`]. The commit
/// kernel writes the new positions straight
/// into `gpu_pmap`, so the pass is fully GPU-resident: the host shadows are
/// left stale and only marked dirty ([`SimBuffers::gpu_particles_dirty`],
/// [`SimBuffers::gpu_maps_dirty`]) for a later explicit read-back.
///
/// `frame_seed` seeds the deterministic diffusion noise and the liquid search
/// direction, so the same seed always produces the same frame.
pub fn step_movement_gpu(bufs: &mut SimBuffers, registry: &ElementRegistry, frame_seed: u32) {
    let dims = match prepare_dims(bufs) {
        Some(dims) => dims,
        None => return,
    };

    let client = bufs.client.clone();

    // The three registry tables only change when Lua edits an element, so they
    // live in permanent device buffers: rebuilding them here allocated and
    // uploaded three buffers on every single tick.
    refresh_move_tables(bufs, registry);
    let lut_handle = bufs.gpu_move_lut.clone();
    let mask_handle = bufs.gpu_photon_mask.clone();

    launch_air_deposit(&client, bufs, &dims, &lut_handle);
    launch_detect_charge(&client, bufs, &dims);
    launch_emap_gate(&client, bufs, &dims);
    launch_emap_closure(&client, bufs, &dims);
    launch_emap_endpoint_strip(&client, bufs, &dims);
    launch_clear_intents(&client, bufs, &dims);
    launch_velocity_intents(&client, bufs, &dims, &lut_handle, frame_seed);
    // `try_move` resolves `e == 2` pairs before `move` writes the position
    // (`Simulation.cpp:1258-1381`, `:1529-1534`). The pass needs the creation
    // queue (gain/Cherenkov photons), so the queue is armed here and drained
    // after the commit has produced the post-move pixels those creations test
    // against.
    crate::reaction_kernel::begin_pending_creates(bufs);
    crate::reaction_kernel::energy_entry::launch_pair_resolution(
        bufs,
        bufs.used_slot_bound(),
        dims.xres,
        dims.yres,
        dims.xcells,
        frame_seed,
    );
    launch_commit_winners(&client, bufs, &dims, &lut_handle, &mask_handle);
    // The pair pass may have queued creations into the reserved window, so the
    // post-move bound has to cover it before the drain sizes and the spatial
    // rebuild scatters it (`reaction_kernel::step_reactions_gpu` does the same).
    bufs.gpu_spawn_used = true;
    crate::reaction_kernel::drain_pending_creates(bufs, frame_seed);
    launch_spatial_rebuild(&client, bufs, &dims);

    // The device maps are now authoritative and the host shadows are stale; the
    // engine can keep running GPU-resident until a subsystem explicitly asks for
    // a read-back (`sync_gpu_to_host`), which uses these flags to pick the
    // arrays it must refresh.
    bufs.gpu_dirty = true;
    bufs.gpu_particles_dirty = true;
    bufs.gpu_maps_dirty = true;
}

/// Validates the canvas and buffer lengths, returning the launch sizes.
///
/// `launch_unchecked` removes all bounds checks, so the host guarantees that
/// every bound buffer is at least as long as the kernel assumes.
fn prepare_dims(bufs: &SimBuffers) -> Option<MoveDims> {
    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    if xres == 0 || yres == 0 {
        return None;
    }
    let xcells = xres / CELL;
    let ycells = yres / CELL;
    let num_pixels = (xres as usize) * (yres as usize);
    let num_cells = (xcells as usize) * (ycells as usize);
    let capacity = bufs.capacity;
    if num_pixels == 0 || num_cells == 0 || capacity == 0 {
        return None;
    }

    if bufs.pmap.len() < num_pixels
        || bufs.photons.len() < num_pixels
        || bufs.air_vx.len() < num_cells
        || bufs.air_vy.len() < num_cells
        || bufs.air_pv.len() < num_cells
        || bufs.wall_bmap.len() < num_cells
        || bufs.wall_emap.len() < num_cells
        || bufs.part_ty.len() < capacity
        || bufs.part_x.len() < capacity
        || bufs.part_y.len() < capacity
        || bufs.part_vx.len() < capacity
        || bufs.part_vy.len() < capacity
        || bufs.part_flags.len() < capacity
        || bufs.part_temp.len() < capacity
        || bufs.part_ctype.len() < capacity
        || bufs.part_tmp.len() < capacity
        || bufs.elem_props_host.len() < PT_NUM
    {
        return None;
    }

    Some(MoveDims {
        capacity,
        num_pixels,
        num_cells,
        xres,
        yres,
        xcells,
        ycells,
    })
}

/// Unclaim every pixel.
fn launch_clear_intents(client: &Client, bufs: &SimBuffers, dims: &MoveDims) {
    unsafe {
        clear_move_intents_kernel::launch_unchecked(
            client,
            CubeCount::new_1d((dims.num_pixels as u32).div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.cell_winner.clone(), dims.num_pixels),
            dims.num_pixels as u32,
        );
    }
}

/// Apply the particle -> air velocity deposit (`Simulation.cpp:2334-2338`).
fn launch_air_deposit(client: &Client, bufs: &SimBuffers, dims: &MoveDims, lut_handle: &Handle) {
    let capacity = dims.capacity;
    let num_slots = bufs.used_slot_bound() as u32;
    unsafe {
        air_deposit_kernel::launch_unchecked(
            client,
            CubeCount::new_1d(num_slots.div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
            BufferArg::from_raw_parts(lut_handle.clone(), build_move_lut_len()),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), dims.num_cells),
            num_slots,
            dims.xres,
            dims.yres,
            dims.xcells,
        );
    }
}

/// Charge the `WL_DETECT` walls a live particle is standing in.
fn launch_detect_charge(client: &Client, bufs: &SimBuffers, dims: &MoveDims) {
    let capacity = dims.capacity;
    let num_slots = bufs.used_slot_bound() as u32;
    unsafe {
        detect_charge_kernel::launch_unchecked(
            client,
            CubeCount::new_1d(num_slots.div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.num_cells),
            num_slots,
            dims.xres,
            dims.yres,
            dims.xcells,
        );
    }
}

/// Fill the vertical wire tree around every span `detect_charge_kernel` started.
///
/// One thread because the traversal is inherently sequential: each filled span
/// decides which spans above and below it continue the tree, and the endpoint
/// markers are transient. The pass is a no-op map scan when no detector fired.
fn launch_emap_closure(client: &Client, bufs: &SimBuffers, dims: &MoveDims) {
    unsafe {
        emap_closure_kernel::launch_unchecked(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_emap_gate.clone(), 1),
            dims.xcells,
            dims.ycells,
        );
    }
}

/// Reduces the two closure preconditions on the device (see [`emap_gate_kernel`]).
fn launch_emap_gate(client: &Client, bufs: &SimBuffers, dims: &MoveDims) {
    unsafe {
        emap_gate_kernel::launch_unchecked(
            client,
            CubeCount::new_1d((dims.num_cells as u32).div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_emap_gate.clone(), 1),
            dims.num_cells as u32,
        );
    }
}

/// Drops the closure's transient endpoint markers, one thread per cell.
fn launch_emap_endpoint_strip(client: &Client, bufs: &SimBuffers, dims: &MoveDims) {
    unsafe {
        strip_emap_endpoints_kernel::launch_unchecked(
            client,
            CubeCount::new_1d((dims.num_cells as u32).div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.num_cells),
            dims.num_cells as u32,
        );
    }
}

/// Integrate velocity, pick a destination and arbitrate it.
fn launch_velocity_intents(
    client: &Client,
    bufs: &SimBuffers,
    dims: &MoveDims,
    lut_handle: &Handle,
    frame_seed: u32,
) {
    let capacity = dims.capacity;
    let num_slots = bufs.used_slot_bound() as u32;
    unsafe {
        velocity_and_intent_kernel::launch_unchecked(
            client,
            CubeCount::new_1d(num_slots.div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
            BufferArg::from_raw_parts(bufs.intent_target.clone(), capacity),
            BufferArg::from_raw_parts(bufs.cell_winner.clone(), dims.num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_air_vx.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_air_vy.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(lut_handle.clone(), build_move_lut_len()),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), capacity),
            num_slots,
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.ycells,
            bufs.config.gravity_mode as u32,
            bufs.config.custom_gravity_x,
            bufs.config.custom_gravity_y,
            u32::from(bufs.config.newton_grav),
            bufs.config.edge_mode as u32,
            frame_seed,
        );
    }
}

/// Apply the arbitration winners.
fn launch_commit_winners(
    client: &Client,
    bufs: &SimBuffers,
    dims: &MoveDims,
    lut_handle: &Handle,
    mask_handle: &Handle,
) {
    let capacity = dims.capacity;
    let num_slots = bufs.used_slot_bound() as u32;
    unsafe {
        commit_move_kernel::launch_unchecked(
            client,
            CubeCount::new_1d(num_slots.div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), capacity),
            BufferArg::from_raw_parts(bufs.intent_target.clone(), capacity),
            BufferArg::from_raw_parts(bufs.cell_winner.clone(), dims.num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), dims.num_cells),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.num_cells),
            BufferArg::from_raw_parts(lut_handle.clone(), build_move_lut_len()),
            BufferArg::from_raw_parts(mask_handle.clone(), PT_NUM),
            num_slots,
            dims.xres,
            dims.yres,
            dims.xcells,
            u32::from(bufs.config.heat_enable),
        );
    }
}

/// Rebuild `pmap` / `photons` from the post-move state.
///
/// The commit kernel cannot scatter safely (two winners may target one pixel
/// and swaps would race), so the maps are cleared and refilled by dedicated
/// passes. This is the GPU replacement for `SimBuffers::rebuild_spatial_maps`.
fn launch_spatial_rebuild(client: &Client, bufs: &SimBuffers, dims: &MoveDims) {
    let capacity = dims.capacity;
    let num_slots = bufs.used_slot_bound() as u32;
    let elem_words = core::mem::size_of::<GpuElementProps>() / core::mem::size_of::<u32>();
    unsafe {
        clear_spatial_maps_kernel::launch_unchecked(
            client,
            CubeCount::new_1d((dims.num_pixels as u32).div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.num_pixels),
            dims.num_pixels as u32,
        );
        scatter_spatial_maps_kernel::launch_unchecked(
            client,
            CubeCount::new_1d(num_slots.div_ceil(BLOCK)),
            CubeDim::new_1d(BLOCK),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), elem_words * PT_NUM),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.num_pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.num_pixels),
            num_slots,
            dims.xres,
            dims.yres,
        );
    }
}

/// Number of `u32` words the packed movement LUT occupies.
pub(crate) const fn build_move_lut_len() -> usize {
    CAN_MOVE_OFFSET + PT_NUM * PT_NUM
}

/// Builds the packed per-element table used by the movement kernels.
///
/// Layout, per element (`PT_NUM` entries of [`META_WORDS`] words):
/// `[properties, Falldown, weight_rank, HeatConduct, advection, air_drag,
/// air_loss, loss, collision, gravity, diffusion, hot_air]`, followed by the
/// flat `PT_NUM * PT_NUM` can-move matrix starting at [`CAN_MOVE_OFFSET`].
///
/// `weight_rank` orders the elements by `weight` (heaviest = `0x1FF`) so that
/// the atomic arbitration lets heavier particles claim contested pixels: sand
/// sinks through water, water sinks through gas.
///
/// The float physics words are the `Element` fields `UpdateParticles` applies
/// (`Simulation.cpp:2361-2377`); they are bit-cast into the same `u32` buffer
/// because the velocity kernel has no free storage-buffer slot for a second
/// table. `HeatConduct` is the `SimulationData::IsHeatInsulator` gate used by
/// the blocked-photon heat exchange (`Simulation.cpp:1220-1228`).
fn build_move_lut(registry: &ElementRegistry) -> Vec<u32> {
    let mut lut = vec![0u32; build_move_lut_len()];

    let mut order: Vec<u32> = (0..PT_NUM as u32).collect();
    order.sort_by_key(|&id| {
        let weight = registry
            .elements
            .get(id as usize)
            .map(|def| def.weight)
            .unwrap_or(0);
        (weight, id)
    });
    let last = (PT_NUM as u32).saturating_sub(1).max(1);
    let mut ranks = vec![0u32; PT_NUM];
    for (pos, &id) in order.iter().enumerate() {
        ranks[id as usize] = ((pos as u32) * RANK_MASK) / last;
    }

    for (ty, &rank) in ranks.iter().enumerate() {
        if let Some(def) = registry.elements.get(ty) {
            let base = ty * META_WORDS;
            lut[base + META_PROPERTIES] = def.properties;
            lut[base + META_FALLDOWN] = def.falldown;
            lut[base + META_RANK] = rank & RANK_MASK;
            lut[base + META_HEAT_CONDUCT] = def.heat_conduct;
            lut[base + PHYS_OFFSET] = def.advection.to_bits();
            lut[base + PHYS_OFFSET + 1] = def.air_drag.to_bits();
            lut[base + PHYS_OFFSET + 2] = def.air_loss.to_bits();
            lut[base + PHYS_OFFSET + 3] = def.loss.to_bits();
            lut[base + PHYS_OFFSET + 4] = def.collision.to_bits();
            lut[base + PHYS_OFFSET + 5] = def.gravity.to_bits();
            lut[base + PHYS_OFFSET + 6] = def.diffusion.to_bits();
            lut[base + PHYS_OFFSET + 7] = def.hot_air.to_bits();
        }
    }

    let can_move = registry.to_gpu_can_move_u32();
    for (dst, &src) in lut[CAN_MOVE_OFFSET..].iter_mut().zip(can_move.iter()) {
        *dst = src;
    }

    lut
}

/// Uploads the movement tables when the registry changed.
///
/// `SimBuffers::move_tables_stale` is set by `refresh_element_tables` and by the
/// engine's registry diff, so the per-tick path is a flag test.
pub fn refresh_move_tables(bufs: &mut SimBuffers, registry: &ElementRegistry) {
    if !bufs.move_tables_stale {
        return;
    }
    let move_lut = build_move_lut(registry);
    let photon_mask = build_photon_mask(registry);
    upload(&bufs.client, &bufs.gpu_move_lut, &move_lut);
    upload(&bufs.client, &bufs.gpu_photon_mask, &photon_mask);
    bufs.move_tables_stale = false;
}

/// Builds the per-element photon reflect mask table (`PT_NUM` words).
///
/// `Simulation.cpp:3044-3056` ANDs a reflected photon's `ctype` with the
/// `PhotonReflectWavelengths` of the pixel it hit, so the mask only ever needs
/// to be addressable by element id.
fn build_photon_mask(registry: &ElementRegistry) -> Vec<u32> {
    let mut mask = vec![0u32; PT_NUM];
    for (ty, slot) in mask.iter_mut().enumerate() {
        if let Some(def) = registry.elements.get(ty) {
            *slot = def.photon_reflect;
        }
    }
    mask
}
