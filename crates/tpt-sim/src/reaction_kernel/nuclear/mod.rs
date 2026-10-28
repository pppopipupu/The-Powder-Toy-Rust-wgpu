//! Radioactive and nuclear element updates.
//!
//! Launched from [`crate::reaction_kernel::step_reactions_gpu`] after the
//! electrical layer and before wall conduction.
//!
//! Every kernel is pixel parallel and only reads the spatial maps, so no thread
//! ever writes a shared map entry. The one exception is [`shields::prot_fusion_kernel`],
//! which walks the particle pool: the proton that lost the `photons` word of a
//! shared pixel has no map entry, so no pixel thread ever reaches it. Type
//! changes, kills and neighbour field writes go through the SoA (`part_ty`,
//! `part_life`, ...) exactly like the other reaction passes. Creations that need
//! a full `create_part` — the shield rings, the antimatter photons, the `NBLE`
//! burst and the proton transmutations — push their own records through
//! `super::shared::request_create`; the few single-product requests of the
//! older kernels still ride a `FLAG_SPAWN_*` bit that [`super::spawn`] expands
//! into the same queue.
//!
//! # Submodules
//!
//! * [`decay`]: isotope decay (`ISOZ`/`ISZS`), noble-gas ionization (`NBLE`),
//!   virus infection/timers (`VIRS`), and platinum catalysis (`PTNM`).
//! * [`isotopes`]: pressure-driven heating (`URAN`/`PLUT`), annihilation (`AMTR`), and deuterium (`DEUT`).
//! * [`shields`]: shield mechanics (`SHLD1`..`SHLD4`) and proton reactions (`PROT`).
//! * [`converters`]: conversion (`CONV`) and spatial warping (`WARP`).
//!
//! | kernel | module | work | storage buffers |
//! |--------|--------|------|-----------------|
//! | [`isotopes::pressure_kernel`] | `isotopes` | `URAN` pressure heating, `PLUT` neutron release, `WARP` charge vent and electron, `INVS` pressure flag | 11 |
//! | [`decay::isotope_kernel`] | `decay` | `ISOZ`/`ISZS` photon decay, `PRTI`/`PRTO` channel index | 7 |
//! | [`decay::nble_kernel`] | `decay` | `NBLE` pressure/temperature ionization burst | 15 |
//! | [`decay::nble_burst_kernel`] | `decay` | `NBLE` burst products from the device spawn window | 16 |
//! | [`shields::shield_kernel`] | `shields` | `SHLD1`..`SHLD4` growth, upgrades and decay | 4 |
//! | [`isotopes::annihilate_kernel`] | `isotopes` | `AMTR` annihilation, `EMBR` spark settling | 6 |
//! | [`decay::virus_kernel`] | `decay` | `VIRS`/`VRSS`/`VRSG` infection, cure and death timers | 8 |
//! | [`converters::converter_kernel`] | `converters` | `CONV` conversion | 7 |
//! | [`converters::warp_kernel`] | `converters` | `WARP` displacement | 6 |
//! | [`isotopes::deut_kernel`] | `isotopes` | `DEUT` compression, absorption and diffusion | 8 |
//! | [`decay::ptnm_kernel`] | `decay` | `PTNM` fast conduction, `WTRV` catalysis, single-neighbour reactions | 8 |
//! | [`decay::ptnm_hydrogen_kernel`] | `decay` | `PTNM` hydrogen catalysis | 8 |
//! | [`shields::prot_under_kernel`] | `shields` | `PROT` reactions with the particle under it and the `tmp` transmutation | 8 |
//! | [`shields::prot_implode_kernel`] | `shields` | `PROT`-triggered `DEUT` implosion trigger | 8 |
//! | [`shields::prot_burst_kernel`] | `shields` | `DeutImplosion` protons from the device spawn window | 16 |
//! | [`shields::prot_seed_kernel`] | `shields` | `PROT` genome swap in the `SEED` it sits on | 7 |
//! | [`shields::prot_heat_kernel`] | `shields` | `PROT` ignition, temperature equalisation and conductor de-sparking | 7 |
//! | [`shields::prot_fusion_kernel`] | `shields` | proton-proton head-on collision energy transfer | 7 |
//!
//! # TPT references
//!
//! * `URAN.cpp:57-70` — pressure multiplies the temperature by `1 + pv/2000`
//!   (guarded by `!legacy_enable`).
//! * `PLUT.cpp:58-65` — `chance(1,100) && chance(int(5*pv),1000)` releases a
//!   neutron.
//! * `DEUT.cpp:54-136` — compression target from temperature and the Newtonian
//!   gravity field, absorption of neighbours, expansion into free space and the
//!   four random life trades.
//! * `WARP.cpp:54-85` — the `tmp2` charge vent and the position swap with a
//!   random neighbour.
//! * `CONV.cpp:51-102` — latching onto the first element it touches and the
//!   `tmp`/`tmp2` restriction channel.
//! * `ISOZ.cpp:69-77`, `ISZS.cpp:60-68` — pressure-gated decay into `PHOT`.
//! * `NBLE.cpp:50-93` — the 5273.15 K + 100 pressure ionization burst.
//! * `PRTI.cpp:64-68`, `PRTO.cpp:63-67` — the temperature channel index.
//! * `SHLD1.cpp:57-90`, `SHLD2.cpp:57-90`, `SHLD3.cpp:57-100`,
//!   `SHLD4.cpp:57-96` — spark growth, level upgrades and slow decay.
//! * `INVIS.cpp:64-73` — `tmp2` marks "invisible under pressure".
//! * `EMBR.cpp:54-75` — a spark dies when it settles against matter.
//! * `PROT.cpp:56-209` — the particle-under reactions, the `tmp` transmutation
//!   ladder, the `SEED` genome swap and the heat pump.
//! * `PROT.cpp:78-85,211-225` — `DeutImplosion` as triggered by a proton on
//!   `DEUT`: the `chance(4 - int(pv) + life/100, 200)` roll, the `life/50`
//!   proton burst clamped to `1..340`, the `temp + life*500` temperature and the
//!   `pv -= 6*CFDS*n` pressure loss.
//! * `PROT.cpp:191-206` — the proton-proton collision: two protons sharing a
//!   pixel must fly almost exactly head-on (direction difference inside
//!   `3.12659..3.15659` rad) and carry more than 10 units of squared speed
//!   between them, after which the proton that lost the `photons` word adds the
//!   combined energy to the winner's `tmp` and dies.
//! * `VIRS.cpp:55-149` — infection, the `tmp3` cure countdown, the `tmp4`
//!   death timer and the `PLSM` reaction (`VRSS`/`VRSG` share the update).
//! * `DRAY.cpp:52-196` — the duplicator ray.
//! * `PTNM.cpp:143-260` — fast conduction, shield growth, the temperature
//!   gated conversions and the `H2` / `WTRV` catalysis.
//!
//! # Deviations and deliberate approximations
//!
//! Each entry names the TPT source and the reason; everything else in those
//! files is ported above.
//!
//! * `BHOL.cpp` / `WHOL.cpp`: both files contain only the element definition —
//!   there is no `Update`/`Create`. The horizon is the `hot_air` term the air
//!   kernel applies plus the Newtonian mass the gravity pass accumulates, so
//!   this pass has nothing to do for `BHOL`/`WHOL` (`NBHL` is handled by
//!   [`crate::gravity_kernel`]).
//! * `PROT.cpp:78-85` (`DEUT` implosion): `DeutImplosion` calls
//!   `create_part(-3, x, y, PT_PROT)` `n` times in the same pixel, so all `n`
//!   products are live immediately and each overwrites the pixel's `photons`
//!   word (the last one owns it). [`shields::prot_implode_kernel`] parks the
//!   burst on the deuterium's slot and [`shields::prot_burst_kernel`] spawns it
//!   in the same frame from the device spawn window, mapping the last proton;
//!   the first proton reuses the deuterium's slot instead of a fresh one (the
//!   reference allocates one and then kills the deuterium, which is
//!   indistinguishable afterwards). A full spawn window drops the tail exactly
//!   like `create_part` returning `-1`.
//! * `PROT.cpp:191-206` (proton-proton collision): the reference adds the
//!   energy to `parts[ID(ahead)].tmp` while `ahead`'s own update is running in
//!   the same frame; the port adds it from the loser's thread with an atomic
//!   `fetch_add`, so a third proton sharing the pixel cannot lose a
//!   contribution (the reference's per-particle updates can). The branch
//!   creates no particle: the energy stays in `tmp` and
//!   [`shields::prot_under_kernel`] turns it into a heavier element on the next
//!   frame, exactly like the reference, unlike the `H2` fusion of
//!   [`super::combustion`], which queues its `NEUT`/`PHOT`/`ELEC` products
//!   directly (`H2.cpp:110-126`).
//! * `AMTR.cpp:52-77`: the antimatter walks its ring sequentially in its own
//!   thread, destroys each non-exempt contact and dies on the fourth one
//!   before touching that neighbour, which is the reference's order and abort.
//!   The photon is queued at the neighbour's pixel with `CR_NO_OCCUPANCY` and
//!   the neighbour's slot is killed, so the particle set matches TPT's in-place
//!   `create_part(ID(r), ...)`. A target that two antimatter particles both see
//!   before either kill lands can still be resolved twice; TPT's per-particle
//!   order lets the first one win. The pressure loss is booked in the
//!   antimatter's own cell (`sim->pv[y/CELL][x/CELL]`), like the reference.
//! * `NBLE.cpp:64-79`: TPT creates the neutron, the optional electron and the
//!   photon with `create_part(-3, x, y, ...)`, so each overwrites the pixel's
//!   single `photons` word and only the last allocation stays mapped. The port
//!   spawns them in the same call order from the device window in one thread
//!   ([`decay::nble_burst_kernel`]) and maps the last successful product, so
//!   the winner is fixed by creation order instead of by drain thread order.
//! * `PTNM` needs the *first* `H2` / `WTRV` neighbour for its pair reactions,
//!   which TPT keeps in locals across two passes. Both `PTNM` kernels park it in
//!   `tmp3` (unused by TPT's `PTNM`) for the duration of their own launch and
//!   clear it again, so no other pass observes the scratch value. The pair
//!   products TPT creates at the neighbour's own pixel (`PTNM.cpp:120-131`,
//!   `196`) go through the queue at that exact pixel with `CR_NO_OCCUPANCY`.
//! * A conversion that TPT performs with `create_part` (the slot restarts from
//!   the target's `DefaultProperties`) goes through
//!   [`super::neut::rebuild_type_in_place`], which seeds `temp`/`life`/`ctype`/
//!   `tmp`/`flags` and clears the velocity; the scratch words it cannot bind
//!   (`tmp2`..`tmp4`, `dcolour`) are finished by
//!   [`super::neut::launch_row_reset`] in the same step. A plain
//!   `part_ty[...] = X` matches `part_change_type`, which keeps the slot's
//!   fields. Fields TPT sets explicitly after a change (`SHLD` life, `SPRK`
//!   ctype, `PHOT` velocity, the `life = 680` of an implosion proton) are
//!   written at the call site.
//! * The map rebuild picks one of several same-pixel energy particles per
//!   frame, whereas TPT keeps the word on the last created one; the particles
//!   themselves (their slots, fields and movement) are identical.

mod converters;
mod decay;
mod isotopes;
mod shields;

pub(super) use converters::{converter_kernel, warp_kernel};
pub(super) use decay::{
    isotope_kernel, nble_burst_kernel, nble_kernel, ptnm_hydrogen_kernel, ptnm_kernel, virus_kernel,
};
pub(super) use isotopes::{annihilate_kernel, deut_kernel, pressure_kernel};
pub(super) use shields::{
    prot_burst_kernel, prot_fusion_kernel, prot_heat_kernel, prot_implode_kernel, prot_seed_kernel,
    prot_under_kernel, shield_kernel,
};

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::shared::{CREATE_QUEUE_WORDS, hash_u32, launch_grid, refresh_gpu_free_slots};
use super::{PassDims, SimBuffers};

/// `Simulation.h:24`: `int(MAX_TEMP - 73) / 100 + 2`, i.e. the number of
/// temperature channels a portal (and `WIFI`) distinguishes.
pub(super) const CHANNELS: i32 = ((MAX_TEMP as i32) - 73) / 100 + 2;

/// Type field of a packed `pmap`/`photons` word (`PMAPMASK`).
pub(super) const PMAP_TYPE: u32 = 0x1FF;

/// `restrict_flt(v, lo, hi)`.
#[cube]
pub(super) fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    let mut r = v;
    if r < lo {
        r = lo;
    }
    if r > hi {
        r = hi;
    }
    r
}

/// Seed a `u32` device local from a runtime anchor (AGENTS rule 4): a literal
/// initialiser that later receives a runtime value expands to a constant store.
#[cube]
#[allow(clippy::erasing_op)]
pub(super) fn seed_u32(anchor: u32, value: u32) -> u32 {
    anchor * 0u32 + value
}

/// Write one freshly created particle into `slot`: the element's
/// `DefaultProperties` row plus the element's `Create` hook, with the caller's
/// `temp`/`ctype`/`tmp` overrides.
///
/// `Simulation::create_part` (`Simulation.cpp:1941-1965`) copies the whole row,
/// sets `type`/`x`/`y`, then runs `Create`; the burst kernels that spawn
/// straight from the device window (instead of the creation queue) have to
/// reproduce both halves. The hook values mirror `shared::create_fields_kernel`
/// (`PROT.cpp:236-242`, `ELEC.cpp:159-165`, `NEUT.cpp:252-258`,
/// `PHOT.cpp:178-184`); every other burst product has no `Create` hook.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(super) fn init_spawned_slot(
    ty: u32,
    px: i32,
    py: i32,
    temp: f32,
    ctype: u32,
    tmp: i32,
    roll: u32,
    slot: usize,
    part_ty: &mut [u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_tmp3: &mut [i32],
    part_tmp4: &mut [i32],
    part_dcolour: &mut [u32],
    part_flags: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
) {
    part_ty[slot] = ty;
    part_x[slot] = px as f32;
    part_y[slot] = py as f32;
    part_temp[slot] = temp;
    part_ctype[slot] = ctype;
    part_tmp[slot] = tmp;
    part_tmp2[slot] = 0i32;
    part_tmp3[slot] = 0i32;
    part_tmp4[slot] = 0i32;
    part_dcolour[slot] = 0u32;
    part_flags[slot] = 0u32;
    part_life[slot] = 0i32;
    part_vx[slot] = 0.0f32;
    part_vy[slot] = 0.0f32;
    if ty == PT_NEUT {
        // `NEUT.cpp:252-258`: `life = rng.between(480, 959)`, `r = 128..255/127`
        // at a random angle.
        let h = hash_u32(roll);
        let angle = (h & 0xFFFFu32) as f32 * (core::f32::consts::TAU / 65_535.0f32);
        let radius = (128.0f32 + ((h >> 16u32) & 0x7Fu32) as f32) / 127.0f32;
        part_life[slot] = 480i32 + (roll % 480u32) as i32;
        part_vx[slot] = angle.cos() * radius;
        part_vy[slot] = angle.sin() * radius;
    } else if ty == PT_ELEC {
        // `ELEC.cpp:159-165`: `life = 680`, 2 px/frame at any angle.
        let angle = (roll % 360u32) as f32 / 360.0f32 * core::f32::consts::TAU;
        part_life[slot] = 680i32;
        part_vx[slot] = 2.0f32 * angle.cos();
        part_vy[slot] = 2.0f32 * angle.sin();
    } else if ty == PT_PROT {
        // `PROT.cpp:236-242`: `life = 680`, 2 px/frame in 36 directions.
        let angle = (roll % 36u32) as f32 * 0.17453f32;
        part_life[slot] = 680i32;
        part_vx[slot] = 2.0f32 * angle.cos();
        part_vy[slot] = 2.0f32 * angle.sin();
    } else if ty == PT_PHOT {
        // `PHOT.cpp:178-184`: 3 px/frame along one of eight directions; the hook
        // leaves `life` at the element default (zero).
        // The reference rounds its 45° step to five digits; `FRAC_PI_4` differs
        // by 1.8e-6, so the literal is kept as written.
        #[allow(clippy::approx_constant)]
        let angle = (roll % 8u32) as f32 * 0.78540f32;
        part_vx[slot] = 3.0f32 * angle.cos();
        part_vy[slot] = 3.0f32 * angle.sin();
    }
}

/// TPT's `rng.chance(n, d)`: `roll % d < n`, false for a non-positive `d`.
#[cube]
pub(super) fn chance(roll: u32, n: u32, d: u32) -> bool {
    let mut hit = false;
    if d > 0u32 {
        hit = (roll % d) < n;
    }
    hit
}

/// Launch every nuclear kernel for one reaction step.
pub(super) fn launch(bufs: &SimBuffers, dims: &PassDims) {
    // `nble_burst_kernel` and `prot_burst_kernel` allocate straight from the
    // device window, while `step_reactions_gpu` rebuilds that table only after
    // the reaction passes (for the creation drain). On the first reaction
    // frame the buffer is still zeroed, so those bursts found `avail == 0` and
    // dropped every product (`DeutImplosion` kept only the proton that reused
    // the deuterium's slot); on a later frame a slot a host edit claimed since
    // the last rebuild could be handed out twice. Collect against the pool as
    // it is now before any kernel below allocates.
    refresh_gpu_free_slots(bufs);
    let (pixel_cubes, pixel_dim) = launch_grid(dims.pixels);
    // TPT gates the `URAN` self-heating on `!legacy_enable`; the port models
    // legacy mode as "heat conduction off".
    let legacy = u32::from(!bufs.config.heat_enable);

    unsafe {
        pressure_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
            legacy,
        );
    }

    unsafe {
        nble_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(
                bufs.gpu_spawn_defaults.clone(),
                PT_NUM * tpt_core::elements::SPAWN_DEFAULT_WORDS,
            ),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    // `NBLE.cpp:64-79`'s same-pixel products, spawned in the reference's order
    // from the device window (the pixel's energy word goes to the last one).
    let (slot_cubes, slot_dim) = launch_grid(dims.used_slots);
    unsafe {
        nble_burst_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_free_slots.clone(), SimBuffers::GPU_SPAWN_COUNT + 2),
            dims.xres,
            dims.yres,
            dims.used_slots as u32,
            dims.frame_seed,
        );
    }

    unsafe {
        isotope_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    unsafe {
        shield_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }

    unsafe {
        annihilate_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    unsafe {
        virus_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    unsafe {
        converter_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }

    unsafe {
        warp_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }

    unsafe {
        deut_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_grav_fx.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_grav_fy.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    unsafe {
        ptnm_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }
    unsafe {
        ptnm_hydrogen_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }

    unsafe {
        prot_under_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_create_queue.clone(), CREATE_QUEUE_WORDS),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }
    unsafe {
        prot_seed_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.frame_seed,
        );
    }
    unsafe {
        prot_heat_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_elem_props.clone(), dims.elem_len),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
        );
    }
    // The collision is slot parallel: the displaced proton of a shared pixel
    // has no map entry. It runs after the transmutation so a proton that became
    // a heavier element this frame no longer matches.
    let (slot_cubes, slot_dim) = launch_grid(dims.used_slots);
    unsafe {
        prot_fusion_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            dims.xres,
            dims.yres,
            dims.used_slots as u32,
        );
    }
    // The implosion reuses the deuterium's slot, so it runs after the heat
    // pump has equalised that slot's temperature (`PROT.cpp:78-85`).
    unsafe {
        prot_implode_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(pixel_cubes, 1, 1),
            CubeDim::new_1d(pixel_dim),
            BufferArg::from_raw_parts(bufs.gpu_pmap.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_air_pv.clone(), dims.cells),
            dims.xres,
            dims.yres,
            dims.xcells,
            dims.frame_seed,
        );
    }
    // `DeutImplosion`'s `n` protons are created in the same frame from the
    // device window, so later passes of this step already see the whole burst.
    unsafe {
        prot_burst_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(slot_cubes, 1, 1),
            CubeDim::new_1d(slot_dim),
            BufferArg::from_raw_parts(bufs.gpu_photons.clone(), dims.pixels),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_x.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_y.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_life.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_ctype.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_temp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp2.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp3.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_tmp4.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_dcolour.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_flags.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vx.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_part_vy.clone(), dims.capacity),
            BufferArg::from_raw_parts(bufs.gpu_free_slots.clone(), SimBuffers::GPU_SPAWN_COUNT + 2),
            dims.xres,
            dims.yres,
            dims.used_slots as u32,
            dims.frame_seed,
        );
    }
    // `create_part` rebuilds the whole row for every in-place conversion
    // (`Simulation.cpp:1941`); this finishes the scratch words the conversion
    // kernels defer (see `neut::change_type_in_place`).
    super::neut::launch_row_reset(bufs, dims);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host mirror of [`prot_transmute_target`].
    fn prot_transmute_host(tmp: i32, under: u32) -> u32 {
        if tmp > 500000 {
            if under == PT_TUNG { PT_AMTR } else { PT_SING }
        } else if tmp > 700 {
            PT_PLUT
        } else if tmp > 420 {
            PT_URAN
        } else if tmp > 310 {
            PT_POLO
        } else if tmp > 250 {
            PT_PLSM
        } else if tmp > 100 {
            PT_O2
        } else if tmp > 50 {
            PT_CO2
        } else {
            PT_NBLE
        }
    }

    #[test]
    fn prot_transmutation_ladder_matches_tpt() {
        assert_eq!(prot_transmute_host(1, PT_DUST), PT_NBLE);
        assert_eq!(prot_transmute_host(51, PT_DUST), PT_CO2);
        assert_eq!(prot_transmute_host(101, PT_DUST), PT_O2);
        assert_eq!(prot_transmute_host(251, PT_DUST), PT_PLSM);
        assert_eq!(prot_transmute_host(311, PT_DUST), PT_POLO);
        assert_eq!(prot_transmute_host(421, PT_DUST), PT_URAN);
        assert_eq!(prot_transmute_host(701, PT_DUST), PT_PLUT);
        assert_eq!(prot_transmute_host(500001, PT_DUST), PT_SING);
        assert_eq!(prot_transmute_host(500001, PT_TUNG), PT_AMTR);
    }

    /// Host mirror of the implosion roll in [`prot_implode_kernel`]
    /// (`PROT.cpp:80`): `-(int(pv) - 4) + life/100`.
    fn implosion_roll(pv: f32, life: i32) -> i32 {
        4 - (pv as i32) + (life / 100)
    }

    /// Host mirror of `DeutImplosion` (`PROT.cpp:211-225`): the proton count
    /// clamped to `1..340` and the burst temperature.
    fn implosion_burst(life: i32, temp: f32) -> (i32, f32) {
        let n = (life / 50).clamp(1, 340);
        (n, (temp + life as f32 * 500.0f32).clamp(MIN_TEMP, MAX_TEMP))
    }

    #[test]
    fn deut_implosion_matches_the_reference() {
        // The roll grows with the life term and shrinks with pressure; a
        // non-positive numerator can never hit.
        assert_eq!(implosion_roll(0.0f32, 0), 4);
        assert_eq!(implosion_roll(3.9f32, 0), 1);
        assert_eq!(implosion_roll(4.0f32, 0), 0);
        assert!(implosion_roll(10.0f32, 0) < 0);
        assert_eq!(implosion_roll(0.0f32, 250), 6);
        // `life/50` protons, clamped to one at the bottom and 340 at the top.
        assert_eq!(implosion_burst(10, 300.0f32), (1, 5300.0f32));
        assert_eq!(implosion_burst(499, 300.0f32).0, 9);
        assert_eq!(implosion_burst(100_000, 300.0f32).0, 340);
        // `deut.temp + life*500` is clamped to the temperature range.
        assert_eq!(implosion_burst(499, 300.0f32).1, MAX_TEMP);
        assert_eq!(implosion_burst(10, MAX_TEMP).1, MAX_TEMP);
    }

    /// Host mirror of [`prot_burst_kernel`]'s allocation order: the deuterium's
    /// slot carries the first proton and the window slots carry the rest; the
    /// pixel's energy word goes to the last one (TPT's last
    /// `create_part(-3, x, y, PT_PROT)`).
    fn implosion_slots(n: u32, deut: u32, window: &[u32]) -> (Vec<u32>, u32) {
        let mut slots = vec![deut];
        slots.extend(window.iter().take(n.saturating_sub(1) as usize).copied());
        let owner = *slots.last().unwrap();
        (slots, owner)
    }

    #[test]
    fn implosion_burst_is_live_in_one_frame_and_the_last_proton_owns_the_pixel() {
        // Four protons (`life = 200`), all created by the same kernel launch.
        let (slots, owner) = implosion_slots(4, 7, &[11, 12, 13, 14]);
        assert_eq!(slots.len(), 4, "the burst must not be deferred a frame");
        assert_eq!(slots, vec![7, 11, 12, 13]);
        assert_eq!(owner, 13, "the last created proton owns the photons word");
        // A single-product burst leaves the deuterium's slot mapped.
        let (slots, owner) = implosion_slots(1, 7, &[11]);
        assert_eq!(slots, vec![7]);
        assert_eq!(owner, 7);
        // A full window drops the tail exactly like `create_part` returning -1.
        let (slots, owner) = implosion_slots(4, 7, &[11]);
        assert_eq!(slots, vec![7, 11]);
        assert_eq!(owner, 11);
    }

    /// Host mirror of `DeutImplosion`'s `pv -= (6.0f * CFDS) * n`
    /// (`PROT.cpp:225`) and the implosion temperature (`PROT.cpp:84`).
    fn implosion_pressure_drop(n: i32) -> f32 {
        -6.0f32 * (n as f32)
    }

    #[test]
    fn implosion_pressure_drop_scales_with_the_burst() {
        assert_eq!(implosion_pressure_drop(1), -6.0);
        assert_eq!(implosion_pressure_drop(4), -24.0);
        assert_eq!(implosion_pressure_drop(340), -2040.0);
    }

    /// Host mirror of the collision test in [`prot_fusion_kernel`]
    /// (`PROT.cpp:191-206`).
    #[allow(clippy::approx_constant)]
    fn proton_collision(vx1: f32, vy1: f32, vx2: f32, vy2: f32) -> bool {
        let velocity1 = vx1 * vx1 + vy1 * vy1;
        let velocity2 = vx2 * vx2 + vy2 * vy2;
        let direction1 = (-vy1).atan2(vx1);
        let direction2 = (-vy2).atan2(vx2);
        let mut difference = direction1 - direction2;
        if difference < 0.0f32 {
            difference += 6.28319f32;
        }
        difference > 3.12659f32 && difference < 3.15659f32 && velocity1 + velocity2 > 10.0f32
    }

    #[test]
    fn proton_collision_needs_a_fast_head_on_pair() {
        // Anti-parallel and fast: both axes qualify.
        assert!(proton_collision(4.0f32, 0.0f32, -4.0f32, 0.0f32));
        assert!(proton_collision(0.0f32, 4.0f32, 0.0f32, -4.0f32));
        // The direction test is `atan2(-vy, vx)`, so opposite velocities along
        // either axis give the same `pi` difference.
        assert!(proton_collision(3.0f32, 3.0f32, -3.0f32, -3.0f32));
        // Same direction: no collision.
        assert!(!proton_collision(4.0f32, 0.0f32, 4.0f32, 0.0f32));
        // Head-on but below the 10 unit energy gate.
        assert!(!proton_collision(2.0f32, 0.0f32, -2.0f32, 0.0f32));
        // 0.1 rad off anti-parallel: outside the 0.015 rad window.
        assert!(!proton_collision(4.0f32, 0.0f32, -3.98f32, -0.399f32));
    }

    /// Host mirror of [`super::seed_swap`]: TPT's four-word swap
    /// (`PROT.cpp:150-160`).
    fn seed_swap_host(mut genome: [i32; 4], j: u32, k: u32) -> [i32; 4] {
        let vj = genome[j as usize];
        let vk = genome[k as usize];
        genome[j as usize] = vk;
        genome[k as usize] = vj;
        genome
    }

    #[test]
    fn seed_gene_swap_exchanges_two_words() {
        assert_eq!(seed_swap_host([1, 2, 3, 4], 0, 2), [3, 2, 1, 4]);
        assert_eq!(seed_swap_host([1, 2, 3, 4], 1, 3), [1, 4, 3, 2]);
        // `j == k` degenerates to a no-op, exactly like TPT's pointer swap.
        assert_eq!(seed_swap_host([1, 2, 3, 4], 2, 2), [1, 2, 3, 4]);
    }
}
