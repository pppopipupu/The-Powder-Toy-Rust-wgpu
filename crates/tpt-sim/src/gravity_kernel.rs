//! Newtonian gravity GPU passes.
//!
//! Companion to [`crate::gravity_gpu`], which owns the device-side FFT Poisson
//! solve. This module ports the element side of The Powder Toy's Newtonian
//! gravity:
//!
//! | kernel | work |
//! |---|---|
//! | [`seed_gravity_tool_mass_kernel`] | folds the painted tool wells into the grid the solve reads |
//! | [`seed_gravity_mass_kernel`] | clears the per-cell accumulator before the element contributions |
//! | [`build_gravity_mass_kernel`] | atomic-adds every gravity element's mass (`GRVT`, `NBHL`, `NWHL`, `GBMB`, `GPMP`) |
//! | [`apply_grav_mass_override_kernel`] | replays the one-shot `gravIn.mass` assignments of `CEXP` / `GSNS` into the built grid |
//! | [`apply_gravity_kernel`] | applies `Loss`, then `newton_grav * grav_fx/fy[cell]` to every eligible particle's velocity |
//! | [`copy_gravity_mass_kernel`] | snapshots the accumulator for the next frame's dispatch |
//! | [`graviton_update_kernel`] | `GRVT` `tmp` clamp and the `GRAV`-style speed/life rule |
//! | [`gravity_solid_update_kernel`] | `GBMB` sticking and `GPMP` activation propagation |
//!
//! # Mass accumulation
//!
//! TPT writes `gravIn.mass` from each element's `Update`
//! (`elements/GRVT.cpp`, `NBHL.cpp`, `NWHL.cpp`, `GBMB.cpp`, `GPMP.cpp`). Here a
//! single slot-parallel kernel reproduces those writes. Because several
//! particles can share one coarse cell, the accumulator has to be atomic.
//! WebGPU only guarantees 32-bit *integer* atomics — the application requests
//! `wgpu::Features::empty()` — so the accumulator is an `Atomic<i32>` in units
//! of `1/65536` mass; the padded-mass kernel of [`crate::gravity_gpu`] converts
//! it back to `f32` inside the solve. The scale keeps every TPT contribution
//! (0.1 .. 51.2, and `0.2 * tmp` for `GRVT`) well inside the `i32` range.
//!
//! `CEXP` and `GSNS` instead *assign* `gravIn.mass[cell]` from their `Update`
//! (`CEXP.cpp:80`, `GSNS.cpp:104`), with `0` as a legitimate value. Those
//! writes go through [`set_grav_mass_override`] into the
//! [`SimBuffers::gpu_grav_mass_override`] channel plus its write-flag grid,
//! which [`apply_grav_mass_override_kernel`] folds into the accumulator after
//! the additive build. [`grav_mass_at`] is the matching read for kernels that
//! consume the solved grid.
//!
//! # Frame position
//!
//! TPT builds the mass grid during particle updates, dispatches the FFT at the
//! start of the next frame and clears the grid (`Simulation::BeforeSim`,
//! `Simulation.cpp:3737-3742`). `PGRV`/`NGRV` write the grid between frames and
//! must join that next dispatch, so [`step_gravity_gpu`] folds the painted
//! wells into the snapshot the solve reads before the FFT. The particle
//! cadence keeps two device grids: it solves from the snapshot taken by the
//! previous call, applies the force, then rebuilds the accumulator from the
//! current particle state and snapshots it.
//! [`step_gravity_element_updates`] is called from the reaction pass, before
//! the movement kernel, mirroring `UpdateParticles`.

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use tpt_core::constants::*;
use tpt_core::elements::ElementRegistry;

use crate::buffers::SimBuffers;
use crate::gravity_gpu::GravityFft;

/// Work units per cube. 256 is the portable WebGPU minimum for a 1-D cube.
const CUBE_DIM: u32 = 256;

/// Fixed-point scale of the device mass accumulator (see the module docs).
const MASS_SCALE: f32 = 65_536.0;

/// Inverse of [`MASS_SCALE`], used when reading the accumulator back.
pub(crate) const MASS_SCALE_INV: f32 = 1.0 / MASS_SCALE;

/// Largest magnitude [`set_grav_mass_override`] accepts, in TPT mass units.
///
/// Chosen so the fixed-point product still fits an `i32`:
/// `32767 * 65536 < i32::MAX`. TPT's element masses are smaller by orders of
/// magnitude (at most `51.2` for a black hole), so the clamp only guards a
/// corrupt or hostile `tmp`.
const MASS_OVERRIDE_MAX: f32 = 32_767.0;

/// Lower bound of the [`set_grav_mass_override`] clamp.
const MASS_OVERRIDE_MIN: f32 = -32_767.0;

/// TPT's per-element `NewtonianGravity` is `1.0` for every element
/// (`simulation/Element.cpp`) and no element overrides it.
const NEWTON_GRAVITY_SCALE: f32 = 1.0;

/// `WL_GRAV` mirror (`wall_bmap` stores the id in its low byte).
const W_GRAV: u32 = WL_GRAV as u32;

/// `PT_NUM` as a `u32` for the kernel-side element-id range guard.
///
/// The host constant is a `usize`; using it directly inside a `#[cube]` body
/// would expand to a `NativeExpand<usize>` operand against a `u32` element id.
const PT_NUM_U32: u32 = PT_NUM as u32;

// Device helpers (kept local so this module compiles on its own, like the
// other kernel modules; `reaction_kernel` has its own copies for its passes)

/// Clamp `v` into `[lo, hi]` (`f32::clamp` is unavailable inside a cube body).
#[cube]
fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut out = v;
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

/// Integer counterpart of [`clampf`].
#[cube]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    let mut out = v;
    if out < lo {
        out = lo;
    }
    if out > hi {
        out = hi;
    }
    out
}

/// Stores a one-shot mass override into `gravIn.mass` for the next solve.
///
/// `mass` is in TPT mass units (float); the accumulator is `1/65536` fixed
/// point. `override_set` carries the "written this tick" flag: TPT's element
/// updates *assign* `gravIn.mass[cell]`, so a forced `0` has to override the
/// particle contributions instead of being indistinguishable from "no write".
///
/// Callers (`CEXP.cpp:80`, `GSNS.cpp:104`) pass the coarse cell of the pixel
/// they occupy (`usize`, the index type the element kernels already use);
/// [`apply_grav_mass_override_kernel`] replays the assignment into the mass
/// grid once per gravity pass.
#[cube]
pub(crate) fn set_grav_mass_override(
    override_mass: &mut [i32],
    override_set: &mut [u32],
    cell: usize,
    mass: f32,
) {
    // Same half-away-from-zero rounding as `seed_gravity_mass_kernel`.
    let mut scaled = clampf(mass, MASS_OVERRIDE_MIN, MASS_OVERRIDE_MAX) * MASS_SCALE;
    if scaled >= 0.0f32 {
        scaled += 0.5f32;
    } else {
        scaled -= 0.5f32;
    }
    override_mass[cell] = i32::cast_from(scaled);
    override_set[cell] = 1u32;
}

/// Reads the solved mass grid for a coarse cell, in TPT mass units.
///
/// `cell` is the flat `cy * xcells + cx` index, like
/// [`set_grav_mass_override`]'s. `fetch_add(0)` returns the old value without
/// changing it, which is the portable non-destructive atomic read (CubeCL's
/// atomic min/max coverage is tier-dependent). `fetch_max(0)` must not be used
/// as a "read": it would clamp the negative hole / `NWHL` masses the snapshot
/// carries.
#[cube]
pub(crate) fn grav_mass_at(mass: &[Atomic<i32>], cell: usize) -> f32 {
    let raw = mass[cell].fetch_add(0i32);
    raw as f32 * MASS_SCALE_INV
}

/// Deterministic integer avalanche (the finaliser of `xxh3`).
#[cube]
fn hash_u32(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16u32;
    h *= 2_246_822_507u32;
    h ^= h >> 13u32;
    h *= 3_266_489_909u32;
    h ^= h >> 16u32;
    h
}

/// Uniform pseudo random number in `[0, 1)` derived from `seed`.
#[cube]
fn rand_unit(seed: u32) -> f32 {
    (hash_u32(seed) & 0x00FF_FFFFu32) as f32 * (1.0f32 / 16_777_216.0f32)
}

/// Packed spatial-map entry at `(x, y)`, or `0` when the pixel is outside the
/// canvas or empty.
#[cube]
fn pmap_at(pmap: &[u32], xres: u32, yres: u32, x: i32, y: i32) -> u32 {
    let mut v = 0u32;
    if x >= 0i32 && y >= 0i32 && (x as u32) < xres && (y as u32) < yres {
        v = pmap[((y as u32) * xres + (x as u32)) as usize];
    }
    v
}

/// Particle slot stored in a packed spatial-map word.
#[cube]
fn pmap_slot(packed: u32) -> usize {
    let raw = packed >> 9u32;
    let mut slot = 0usize;
    if raw > 0u32 {
        slot = (raw - 1u32) as usize;
    }
    slot
}

/// Clears the fixed-point mass accumulator before [`build_gravity_mass_kernel`]
/// refills it from the live particles.
///
/// The `PGRV`/`NGRV` tool wells are folded into the snapshot the solve reads in
/// [`seed_gravity_tool_mass_kernel`], so the device-side tool value is already
/// zero here. The unconditional store still matters: it wipes the previous
/// frame's accumulator before the additive build, exactly like the
/// `gravIn.mass` clear after `DispatchNewtonianGravity`
/// (`Simulation.cpp:3739-3743`).
///
/// Seeding and accumulating cannot share a launch: a thread that clears a cell
/// could race with a thread that already added to it.
#[cube(launch_unchecked)]
fn seed_gravity_mass_kernel(grav_mass: &mut [Atomic<i32>], tool_mass: &mut [f32], cells: u32) {
    let idx = ABSOLUTE_POS;
    if idx < cells as usize {
        let mut scaled = tool_mass[idx] * MASS_SCALE;
        if scaled >= 0.0f32 {
            scaled += 0.5f32;
        } else {
            scaled -= 0.5f32;
        }
        grav_mass[idx].store(i32::cast_from(scaled));
        tool_mass[idx] = 0.0f32;
    }
}

/// Folds the painted `PGRV`/`NGRV` wells into the grid the next FFT dispatch
/// reads.
///
/// `PGRV.cpp` / `NGRV.cpp` write `gravIn.mass[cell]` between frames and
/// `BeforeSim` dispatches (then clears) that grid in the very next frame
/// (`Simulation.cpp:3739-3743`); tool writes have no two-grid snapshot of
/// their own. The store is conditional on a non-zero tool cell so the particle
/// mass already accumulated in the snapshot's other cells survives, and the
/// tool word is consumed exactly like TPT's post-dispatch clear.
#[cube(launch_unchecked)]
fn seed_gravity_tool_mass_kernel(grav_prev: &mut [Atomic<i32>], tool_mass: &mut [f32], cells: u32) {
    let idx = ABSOLUTE_POS;
    if idx < cells as usize {
        let tool = tool_mass[idx];
        if tool != 0.0f32 {
            let mut scaled = tool * MASS_SCALE;
            if scaled >= 0.0f32 {
                scaled += 0.5f32;
            } else {
                scaled -= 0.5f32;
            }
            grav_prev[idx].store(i32::cast_from(scaled));
            tool_mass[idx] = 0.0f32;
        }
    }
}

/// Accumulates every gravity element's contribution into the coarse mass grid.
///
/// Mirrors the `gravIn.mass` writes in the element updates:
///
/// * `GRVT`: `0.2 * tmp` (`tmp` clamped to `[-100, 100]`, `GRVT.cpp`),
/// * `NBHL`: `restrict_flt(0.001 * tmp, 0.1, 51.2)`, or `0.1` when `tmp == 0`
///   (`NBHL.cpp`),
/// * `NWHL`: the negated black-hole mass (`NWHL.cpp`),
/// * `GBMB`: `20` while `life > 20`, `-80` while `1 <= life <= 20` (`GBMB.cpp`),
/// * `GPMP`: `0.2 * (temp - 273.15)` while `life == 10` (`GPMP.cpp`).
///
/// TPT assigns (`=`) the mass for `GRVT` / `GBMB` / `GPMP` and adds (`+=`) for
/// the holes; a GPU cannot express "last writer wins", so every contribution is
/// added, which is the intended behaviour when several particles share a cell.
/// Contributions are dropped where `grav_mask` is zero (`GravityImpl::Work`
/// masks the mass before the transform).
#[cube(launch_unchecked)]
fn build_gravity_mass_kernel(
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_life: &[i32],
    part_tmp: &[i32],
    part_temp: &[f32],
    grav_mask: &[u32],
    grav_mass: &mut [Atomic<i32>],
    num_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
) {
    let i = ABSOLUTE_POS;
    if i < num_slots as usize {
        let ty = part_ty[i];
        if ty != PT_NONE {
            let mut mass = 0.0f32;
            let mut contributes = false;

            if ty == PT_GRVT {
                // `GRVT.cpp`: tmp is clamped before it becomes mass.
                let tmp = clampi(part_tmp[i], -100i32, 100i32);
                mass = 0.2f32 * (tmp as f32);
                contributes = true;
            } else if ty == PT_NBHL || ty == PT_NWHL {
                let mut hole = 0.1f32;
                let tmp = part_tmp[i];
                if tmp != 0i32 {
                    // `restrict_flt(0.001f * tmp, 0.1f, 51.2f)`.
                    hole = clampf(0.001f32 * (tmp as f32), 0.1f32, 51.2f32);
                }
                // `NWHL.cpp` subtracts the same mass the black hole adds.
                if ty == PT_NWHL {
                    hole = -hole;
                }
                mass = hole;
                contributes = true;
            } else if ty == PT_GBMB {
                // `GBMB.cpp`: the bomb attracts while armed and pushes while
                // counting down.
                let life = part_life[i];
                if life > 20i32 {
                    mass = 20.0f32;
                    contributes = true;
                } else if life >= 1i32 {
                    mass = -80.0f32;
                    contributes = true;
                }
            } else if ty == PT_GPMP {
                // `GPMP.cpp`: only an activated pump contributes, scaled by its
                // temperature offset from ambient.
                if part_life[i] == 10i32 {
                    mass = 0.2f32 * (part_temp[i] - 273.15f32);
                    contributes = true;
                }
            }

            if contributes {
                let xi = i32::cast_from(part_x[i] + 0.5f32);
                let yi = i32::cast_from(part_y[i] + 0.5f32);
                if xi >= 0i32 && yi >= 0i32 && xi < xres as i32 && yi < yres as i32 {
                    let cx = (xi as u32) / CELL;
                    let cy = (yi as u32) / CELL;
                    if cx < xcells && cy < ycells {
                        let cell = (cy * xcells + cx) as usize;
                        if grav_mask[cell] != 0u32 {
                            // Round to the nearest fixed-point unit instead of
                            // truncating towards zero.
                            let mut scaled = mass * MASS_SCALE;
                            if scaled >= 0.0f32 {
                                scaled += 0.5f32;
                            } else {
                                scaled -= 0.5f32;
                            }
                            let quantised = i32::cast_from(scaled);
                            if quantised != 0i32 {
                                grav_mass[cell].fetch_add(quantised);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Replays this tick's one-shot element mass overrides into the freshly built
/// mass grid.
///
/// `CEXP` (`CEXP.cpp:80`) and `GSNS` (`GSNS.cpp:104`) assign
/// `gravIn.mass[cell]` outright, so the forced value replaces whatever the
/// other gravity elements accumulated in that cell rather than adding to it.
/// [`set_grav_mass_override`] recorded the value and its write flag during the
/// element pass; this kernel applies one tick of writes between
/// [`build_gravity_mass_kernel`] and [`copy_gravity_mass_kernel`], so the
/// injection joins the snapshot the next solve reads. Cell-parallel and
/// non-atomic: the channel holds one value per cell (the last element write
/// wins, like TPT's own assignments in the same cell), and the launch order
/// keeps this clear of the build's atomic adds.
#[cube(launch_unchecked)]
#[allow(clippy::collapsible_if)]
fn apply_grav_mass_override_kernel(
    grav_mass: &mut [Atomic<i32>],
    override_mass: &[i32],
    override_set: &[u32],
    cells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < cells as usize {
        if override_set[idx] != 0u32 {
            grav_mass[idx].store(override_mass[idx]);
        }
    }
}

/// Applies `Loss` and the Newtonian force to every eligible particle.
///
/// `Simulation.cpp:2361-2384`:
///
/// ```text
/// if (t != PT_SPNG || !(parts[i].flags&FLAG_MOVABLE)) {
///     parts[i].vx *= elements[t].Loss;
///     parts[i].vy *= elements[t].Loss;
/// }
/// parts[i].vx += elements[t].Advection*vx[cell] + neighbourhood.pGravX;
/// ```
///
/// `GetGravityField` adds `newtonGrav * gravOut.force[cell]` to `pGrav`, i.e.
/// **after** the `Loss` multiplication. The movement kernel therefore skips its
/// own `Loss` on the frames this kernel runs (`newton_grav != 0`), so the
/// Newtonian term is damped exactly like the reference instead of being
/// pre-damped.
///
/// `GetNeighbourhood` (`Simulation.cpp:2284-2293`) only evaluates `pGrav` for
/// `!(Properties & TYPE_SOLID)`, so solids receive `Loss` but no force. The
/// particles the movement kernel skips (rigid pixels, `FLAG_SKIPMOVE`,
/// unpowered `WL_STASIS`, off-screen) are skipped here too, matching the
/// reference's ordering (kill/stasis checks precede the velocity update).
///
/// The per-axis `[-16, 16]` clamp the port used is gone: `PlanMove` performs
/// the only velocity limit, the vector `MAX_VELOCITY` scale
/// (`Simulation.cpp:2170-2175`, `SimulationConfig.h:34`).
///
/// # Frame-position deviation
///
/// The reference runs this block inside `UpdateParticles`, i.e. after the
/// reaction/transition phase. The port's pipeline batches reactions and
/// movement (`Simulation::step`'s documented deviation), and the 16-buffer
/// budget of the velocity kernel leaves no room for the two force grids, so
/// `Loss` + the Newtonian add run here, before the reactions. The **relative**
/// order (`Loss` first, then the force, `Simulation.cpp:2365-2375` +
/// `2118-2124`) and the magnitude are the reference's; only the position in
/// the tick differs, and the velocity kernel skips its own `Loss` on these
/// frames so the damping is never applied twice.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn apply_gravity_kernel(
    part_ty: &[u32],
    part_x: &[f32],
    part_y: &[f32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_flags: &[u32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    num_slots: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    newton_grav: f32,
) {
    let i = ABSOLUTE_POS;
    if i < num_slots as usize {
        let ty = part_ty[i];
        if ty != PT_NONE && ty < PT_NUM_U32 {
            let flags = part_flags[i];
            // Rigid pixels belong to the rigid-body solver and never receive
            // the movement pass's velocity update. `FLAG_SKIPMOVE` is **not**
            // filtered here: the reference only implements it for `PHOT`
            // inside `MovementPhase` (`Simulation.cpp:2961-2965`), after the
            // velocity update has already run.
            let mut eligible = true;
            if (flags & 0x10u32) != 0u32 {
                eligible = false;
            }
            let xi = i32::cast_from(part_x[i] + 0.5f32);
            let yi = i32::cast_from(part_y[i] + 0.5f32);
            if xi < 0i32 || yi < 0i32 || xi >= xres as i32 || yi >= yres as i32 {
                eligible = false;
            }
            // `Simulation.cpp:2312-2317`: off-margin particles are killed by
            // the movement pass before their velocity would be updated.
            if xi < 4i32 || yi < 4i32 || xi >= xres as i32 - 4i32 || yi >= yres as i32 - 4i32 {
                eligible = false;
            }
            let cx = (xi as u32) / CELL;
            let cy = (yi as u32) / CELL;
            let cell = (cy * xcells + cx) as usize;
            // `Simulation.cpp:2335-2338`: unpowered stasis freezes the particle.
            if wall_bmap[cell] == 18u32 && (wall_emap[cell] & 0xFFu32) < 8u32 {
                eligible = false;
            }
            if eligible {
                // `move_lut` record: `[properties, falldown, rank, HeatConduct,
                // advection, air_drag, air_loss, loss, collision, gravity,
                // diffusion, hot_air]`; `loss` is the fourth physics word.
                let base = (ty as usize) * 12usize;
                let props = move_lut[base];
                let loss = f32::reinterpret(move_lut[base + 4usize + 3usize]);
                // `SPNG` with `FLAG_MOVABLE` keeps its velocity
                // (`Simulation.cpp:2365-2369`).
                let mut apply_loss = true;
                if ty == PT_SPNG && (flags & 0x08u32) != 0u32 {
                    apply_loss = false;
                }
                if apply_loss {
                    part_vx[i] *= loss;
                    part_vy[i] *= loss;
                }
                if (props & TYPE_SOLID) == 0u32 {
                    part_vx[i] += newton_grav * grav_fx[cell];
                    part_vy[i] += newton_grav * grav_fy[cell];
                }
            }
        }
    }
}

/// Copies the freshly built fixed-point mass grid into the double-buffered
/// "previous frame" grid that the next solve consumes.
///
/// `BeforeSim` dispatches the FFT and then clears `gravIn.mass`
/// (`Simulation.cpp:3737-3742`); the element `Update`s refill it during
/// `UpdateParticles`. The port keeps the same two-grid cadence without a
/// read-back: `gpu_grav_mass` is this frame's accumulation,
/// `gpu_grav_mass_prev` the snapshot the next frame's solve reads.
#[cube(launch_unchecked)]
fn copy_gravity_mass_kernel(src: &[i32], dst: &mut [Atomic<i32>], cells: u32) {
    let idx = ABSOLUTE_POS;
    if idx < cells as usize {
        dst[idx].store(src[idx]);
    }
}

/// `GRVT` update (`elements/GRVT.cpp`).
///
/// `GRVT` is `TYPE_ENERGY`, so it lives in `photons` rather than `pmap`; a
/// slot-parallel pass visits it regardless of the spatial maps. The speed/life
/// rule is the one TPT applies to `GRAV` (`elements/GRAV.cpp`):
/// `vx^2 + vy^2 >= 0.1` and a `1/512` chance grants `life = 48` to a particle
/// whose life reached zero.
///
/// This kernel runs after the generic life timers, which only kill
/// `PROP_LIFE_KILL_DEC` elements when the countdown reaches zero, so a
/// zero-life `GRVT` is still alive here (see [`crate::reaction_kernel`]).
#[cube(launch_unchecked)]
fn graviton_update_kernel(
    part_ty: &[u32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_vx: &[f32],
    part_vy: &[f32],
    num_slots: u32,
    frame_seed: u32,
) {
    let i = ABSOLUTE_POS;
    if i < num_slots as usize && part_ty[i] == PT_GRVT {
        // "at higher tmps they just go completely insane" (`GRVT.cpp`).
        part_tmp[i] = clampi(part_tmp[i], -100i32, 100i32);

        let vx = part_vx[i];
        let vy = part_vy[i];
        if vx * vx + vy * vy >= 0.1f32 {
            let roll = rand_unit(frame_seed ^ hash_u32(i as u32));
            if roll < (1.0f32 / 512.0f32) && part_life[i] == 0i32 {
                part_life[i] = 48i32;
            }
        }
    }
}

/// `GBMB` and `GPMP` updates (`elements/GBMB.cpp`, `elements/GPMP.cpp`).
///
/// Both are solids, so a pixel-parallel pass over `pmap` visits each of them
/// exactly once and the pixel index gives the neighbour coordinates TPT uses.
///
/// * `GBMB` with `life <= 0` sticks to the first neighbour that is not another
///   bomb, a clone or diamond and starts its 60-frame countdown. `BOMB` and
///   `PCLN` do not exist in this port, so only `GBMB` / `CLNE` / `DMND` are
///   excluded.
/// * `GPMP` with `life == 10` clamps its temperature and propagates activation
///   through the 5x5 neighbourhood: a neighbour in `1..10` sets *this* pump's
///   life to `9`, a neighbour at `0` is switched on (`life = 10`). Any other
///   pump counts down. The two writes can race between neighbouring pumps, but
///   both values are the ones TPT writes, so the result stays consistent.
#[cube(launch_unchecked)]
fn gravity_solid_update_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let packed = pmap[idx];
        if packed != 0u32 {
            let id = pmap_slot(packed);
            let ty = part_ty[id];
            let px = (idx as u32 % xres) as i32;
            let py = (idx as u32 / xres) as i32;

            if ty == PT_GBMB {
                // `GBMB.cpp`: a bomb that lost its countdown checks its 3x3
                // neighbourhood and sticks to the first non-bomb it finds.
                if part_life[id] <= 0i32 {
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        let neighbour = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if neighbour != 0u32 {
                            let nid = pmap_slot(neighbour);
                            let nty = part_ty[nid];
                            if nty != PT_NONE && nty != PT_GBMB && nty != PT_CLNE && nty != PT_DMND
                            {
                                part_life[id] = 60i32;
                            }
                        }
                    }
                }
            } else if ty == PT_GPMP {
                if part_life[id] == 10i32 {
                    // `GPMP.cpp` clamps an active pump to +/-256 K around
                    // the 273.15 K origin of the mass scale.
                    part_temp[id] =
                        clampf(part_temp[id], -256.0f32 + 273.15f32, 256.0f32 + 273.15f32);

                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let neighbour = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if neighbour != 0u32 {
                                let nid = pmap_slot(neighbour);
                                if part_ty[nid] == PT_GPMP {
                                    let nlife = part_life[nid];
                                    if nlife < 10i32 && nlife > 0i32 {
                                        part_life[id] = 9i32;
                                    } else if nlife == 0i32 {
                                        part_life[nid] = 10i32;
                                    }
                                }
                            }
                        }
                    }
                } else if part_life[id] > 0i32 {
                    let next_life = part_life[id] - 1i32;
                    part_life[id] = next_life;
                }
            }
        }
    }
}

/// Number of cubes and units per cube needed to cover `units` work items.
fn launch_grid(units: usize) -> (u32, u32) {
    let cubes = units.div_ceil(CUBE_DIM as usize).max(1) as u32;
    (cubes, CUBE_DIM)
}

/// Rebuilds the gravity propagation mask from the wall map.
///
/// Port of `Simulation::UpdateGravityMask` (`Simulation.cpp`): the mask starts
/// at zero everywhere, then a flood fill from the canvas border marks every
/// cell reachable through non-diagonal neighbours without crossing `WL_GRAV`.
/// Cells behind a gravity wall therefore keep mask `0`, exactly like TPT.
fn rebuild_gravity_mask(bufs: &mut SimBuffers, xcells: usize, ycells: usize) {
    let cells = xcells * ycells;
    if cells == 0 {
        return;
    }
    if bufs.grav_mask.len() != cells {
        bufs.grav_mask.resize(cells, 0);
    }
    bufs.grav_mask.fill(0);
    if bufs.wall_bmap.len() < cells {
        return;
    }

    /// Marks `idx` reachable when it is inside the grid, not a `WL_GRAV` wall
    /// and not already marked.
    fn seed(idx: usize, wall: &[u32], mask: &mut [u32], stack: &mut Vec<usize>) {
        if (wall[idx] & 0xFFu32) == W_GRAV || mask[idx] != 0u32 {
            return;
        }
        mask[idx] = u32::MAX;
        stack.push(idx);
    }

    let wall = &bufs.wall_bmap[..cells];
    let mask = &mut bufs.grav_mask[..cells];
    let mut stack: Vec<usize> = Vec::with_capacity(256);

    for x in 0..xcells {
        seed(x, wall, mask, &mut stack);
        seed((ycells - 1) * xcells + x, wall, mask, &mut stack);
    }
    for y in 1..ycells.saturating_sub(1) {
        seed(y * xcells, wall, mask, &mut stack);
        seed(y * xcells + xcells - 1, wall, mask, &mut stack);
    }

    while let Some(idx) = stack.pop() {
        let x = idx % xcells;
        let y = idx / xcells;
        if x > 0 {
            seed(idx - 1, wall, mask, &mut stack);
        }
        if x + 1 < xcells {
            seed(idx + 1, wall, mask, &mut stack);
        }
        if y > 0 {
            seed(idx - xcells, wall, mask, &mut stack);
        }
        if y + 1 < ycells {
            seed(idx + xcells, wall, mask, &mut stack);
        }
    }
}

/// Uploads the gravity propagation mask when a wall edit invalidated it.
///
/// TPT's `UpdateGravityMask` is independent of `sim->grav`: the zone overlay
/// (`Renderer::draw_grav_zones`) reads the mask with Newtonian gravity off, so
/// the render pass refreshes it as well.
pub(crate) fn ensure_gravity_mask(bufs: &mut SimBuffers) {
    let xcells = bufs.config.xcells() as usize;
    let ycells = bufs.config.ycells() as usize;
    let cells = xcells * ycells;
    if cells == 0
        || bufs.wall_bmap.len() < cells
        || bufs.grav_mask.len() < cells
        || !bufs.grav_mask_dirty
    {
        return;
    }
    rebuild_gravity_mask(bufs, xcells, ycells);
    bufs.client.write(
        &bufs.gpu_grav_mask,
        Bytes::from_elems(bufs.grav_mask[..cells].to_vec()),
    );
    bufs.grav_mask_dirty = false;
}

/// Runs the `GRVT` / `GBMB` / `GPMP` element updates for one frame.
///
/// Called from the reaction pass before movement, matching the order in
/// `Simulation::UpdateParticles` (life timers, then `Update`). The kernels only
/// touch their own element state; the mass grid is rebuilt later by
/// [`step_gravity_gpu`] from the resulting `life` / `tmp` / `temp` values.
pub fn step_gravity_element_updates(bufs: &mut SimBuffers, frame_seed: u32) {
    let capacity = bufs.capacity;
    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    if capacity == 0 || xres == 0 || yres == 0 {
        return;
    }
    let pixels = (xres as usize) * (yres as usize);
    if bufs.part_ty.len() < capacity
        || bufs.part_life.len() < capacity
        || bufs.part_tmp.len() < capacity
        || bufs.part_temp.len() < capacity
        || bufs.part_vx.len() < capacity
        || bufs.part_vy.len() < capacity
        || bufs.pmap.len() < pixels
    {
        return;
    }

    let slots = bufs.used_slot_bound();
    let (slot_cubes, slot_dim) = launch_grid(slots);
    let (pixel_cubes, pixel_dim) = launch_grid(pixels);

    unsafe {
        graviton_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
            slots as u32,
            frame_seed,
        );

        gravity_solid_update_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), capacity),
            xres,
            yres,
        );
    }

    // Device state changed; the reaction pass pulls it back right after this
    // call, but flagging it keeps the contract for any other caller.
    bufs.gpu_particles_dirty = true;
    bufs.gpu_dirty = true;
}

/// Runs one Newtonian gravity step: mask, FFT solve, velocity update and the
/// next frame's mass accumulation.
///
/// Nothing is read back: the mass grid, the mask and the force grids stay
/// device-resident, and the function never calls
/// [`SimBuffers::sync_gpu_to_host`].
///
/// Frame cadence, mirroring `Simulation::BeforeSim` (`Simulation.cpp:3737-3742`):
///
/// 0. fold any `PGRV`/`NGRV` tool wells into the snapshot the FFT reads (TPT
///    writes `gravIn.mass` between frames and dispatches it in the next
///    frame), then
/// 1. dispatch the FFT for the mass accumulated during the **previous** frame
///    (`gpu_grav_mass_prev`), then
/// 2. apply `Loss` + the force to the current particles
///    (`Simulation.cpp:2361-2384`), then
/// 3. rebuild `gpu_grav_mass` from the live particles (the element `Update`s
///    of the frame that just ran), replay this tick's `CEXP`/`GSNS` mass
///    overrides into it, then copy it into `gpu_grav_mass_prev`.
///
/// The first frame's snapshot is zero, exactly like a freshly cleared
/// `gravIn.mass`; a `GRVT`/`NBHL` created this frame therefore attracts from
/// the next frame on, as in the reference.
///
/// When `config.newton_grav` is disabled the whole pass is skipped, matching
/// `Simulation::EnableNewtonianGravity(false)`; the dirty mask flag survives so
/// the mask is rebuilt once gravity is switched back on, and a force field left
/// by the previous solve is zeroed on both sides.
pub fn step_gravity_gpu(
    bufs: &mut SimBuffers,
    registry: &ElementRegistry,
    solver: &mut GravityFft,
    _seed: u32,
) {
    if !bufs.config.newton_grav {
        bufs.zero_gravity_forces();
        return;
    }
    let xres = bufs.config.xres;
    let yres = bufs.config.yres;
    let xcells = bufs.config.xcells() as usize;
    let ycells = bufs.config.ycells() as usize;
    let capacity = bufs.capacity;
    let cells = xcells * ycells;
    let pixels = (xres as usize) * (yres as usize);
    if xres == 0 || yres == 0 || cells == 0 || capacity == 0 {
        return;
    }
    if bufs.part_ty.len() < capacity
        || bufs.part_x.len() < capacity
        || bufs.part_y.len() < capacity
        || bufs.part_life.len() < capacity
        || bufs.part_tmp.len() < capacity
        || bufs.part_temp.len() < capacity
        || bufs.part_vx.len() < capacity
        || bufs.part_vy.len() < capacity
        || bufs.part_flags.len() < capacity
        || bufs.pmap.len() < pixels
        || bufs.grav_mask.len() < cells
        || bufs.grav_mass.len() < cells
        || bufs.grav_mass_prev.len() < cells
        || bufs.grav_fx.len() < cells
        || bufs.grav_fy.len() < cells
        || bufs.wall_bmap.len() < cells
        || bufs.wall_emap.len() < cells
    {
        return;
    }

    // The movement LUT carries the `Loss` / `TYPE_SOLID` words this pass
    // reads; refresh it before the launch if the registry changed.
    crate::move_kernel::refresh_move_tables(bufs, registry);

    let client = bufs.client.clone();

    // Rebuild the propagation mask after a wall change (`gravWallChanged`).
    if bufs.grav_mask_dirty {
        rebuild_gravity_mask(bufs, xcells, ycells);
        client.write(
            &bufs.gpu_grav_mask,
            Bytes::from_elems(bufs.grav_mask[..cells].to_vec()),
        );
        bufs.grav_mask_dirty = false;
    }

    let (cell_cubes, cell_dim) = launch_grid(cells);
    let slots = bufs.used_slot_bound();
    let (slot_cubes, slot_dim) = launch_grid(slots);

    // 0. Painted `PGRV`/`NGRV` wells must affect THIS dispatch. TPT writes
    //    `gravIn.mass` between frames and `BeforeSim` dispatches it in the next
    //    frame (`Simulation.cpp:3739-3743`), with no snapshot of its own, so
    //    fold the wells into the grid the solve reads. The seed is conditional
    //    per cell, so the particle mass the snapshot holds everywhere else is
    //    not wiped. The host staging copy is cleared with the device grid,
    //    mirroring TPT's post-dispatch `gravIn.mass` clear.
    if bufs.grav_tool_mass_dirty {
        client.write(
            &bufs.gpu_grav_tool_mass,
            Bytes::from_elems(bufs.grav_tool_mass[..cells].to_vec()),
        );
        bufs.grav_tool_mass[..cells].fill(0.0);
        bufs.grav_tool_mass_dirty = false;
    }
    unsafe {
        seed_gravity_tool_mass_kernel::launch_unchecked(
            &client,
            CubeCount::Static(cell_cubes, 1, 1),
            CubeDim::new_1d(cell_dim),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass_prev.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_tool_mass.clone(), cells),
            cells as u32,
        );
    }

    // 1. Solve from the **previous** frame's accumulation, mirroring
    //    `BeforeSim`'s `DispatchNewtonianGravity()` before the `gravIn.mass`
    //    clear (`Simulation.cpp:3737-3742`). The element `Update`s of the
    //    frame that just ran refilled `gpu_grav_mass`; the double buffer holds
    //    the snapshot taken at the end of the previous gravity step. The first
    //    frame's snapshot is zero, exactly like a fresh `gravIn.mass`.
    let prev_mass = bufs.gpu_grav_mass_prev.clone();
    solver.solve(bufs, &prev_mass, MASS_SCALE_INV);
    bufs.grav_shadow_dirty = true;
    // `gravOut` now holds a solved field; if Newtonian gravity is switched off
    // before the next solve, `zero_gravity_forces` must erase it.
    bufs.grav_force_stale = true;

    // 2. `Loss` + Newtonian force, in the reference's order
    //    (`Simulation.cpp:2361-2384`).
    unsafe {
        apply_gravity_kernel::launch_unchecked(
            &client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_grav_fx.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_fy.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_bmap.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_wall_emap.clone(), cells),
            BufferArg::from_raw_parts(
                bufs.gpu_move_lut.clone(),
                crate::move_kernel::build_move_lut_len(),
            ),
            slots as u32,
            xres,
            yres,
            xcells as u32,
            NEWTON_GRAVITY_SCALE,
        );
    }

    // 3. Rebuild this frame's mass for the next dispatch. The tool grid was
    //    already consumed before the solve, so the seed now only clears the
    //    stale accumulator; the build then adds the live particles' mass.
    unsafe {
        seed_gravity_mass_kernel::launch_unchecked(
            &client,
            CubeCount::Static(cell_cubes, 1, 1),
            CubeDim::new_1d(cell_dim),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_tool_mass.clone(), cells),
            cells as u32,
        );

        build_gravity_mass_kernel::launch_unchecked(
            &client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), capacity),
            BufferArg::from_raw_parts(bufs.gpu_grav_mask.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass.clone(), cells),
            slots as u32,
            xres,
            yres,
            xcells as u32,
            ycells as u32,
        );

        // The element passes' one-shot `gravIn.mass` assignments replace the
        // contributions built above in their cell (`CEXP`, `GSNS`); the modext
        // clear kernel wipes the channel at the start of the next tick, so
        // exactly one tick of writes lands here.
        apply_grav_mass_override_kernel::launch_unchecked(
            &client,
            CubeCount::Static(cell_cubes, 1, 1),
            CubeDim::new_1d(cell_dim),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass_override.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass_override_set.clone(), cells),
            cells as u32,
        );

        // 4. `prev = current`, the double-buffer swap.
        copy_gravity_mass_kernel::launch_unchecked(
            &client,
            CubeCount::Static(cell_cubes, 1, 1),
            CubeDim::new_1d(cell_dim),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass.clone(), cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_mass_prev.clone(), cells),
            cells as u32,
        );
    }
    bufs.grav_mass_dirty = false;

    // Velocities changed on the device without a host read-back.
    bufs.gpu_particles_dirty = true;
    bufs.gpu_dirty = true;
}

/// The coarse grid and the mass kernel both use the shared [`CELL`] size.
const _: () = assert!(CELL == 4);

/// The [`set_grav_mass_override`] clamp keeps the fixed-point product inside an
/// `i32`: the rounding adds at most half a unit before the cast truncates.
const _: () = assert!(
    (MASS_OVERRIDE_MAX as i64) * (MASS_SCALE as i64) <= i32::MAX as i64,
    "the mass override clamp must fit the fixed-point accumulator"
);
const _: () = assert!(
    (MASS_OVERRIDE_MIN as i64) * (MASS_SCALE as i64) >= i32::MIN as i64,
    "the mass override clamp must fit the fixed-point accumulator"
);
