//! Combustion and the local element updates that need particle state.
//!
//! Pixel parallel: the thread reads its own `pmap` entry, so the 5x5
//! neighbourhood is just a strided read of the same map.
//!
//! Besides fire / plasma / lava ignition and the `H2` / `O2` chemistry this
//! pass also runs:
//!
//! * the single-element `Update` functions of TPT's plain-matter batch that
//!   need `life`, `temp`, `tmp` or the air pressure — state that the other
//!   reaction passes do not bind: `GOO` pressure decay, `PLNT` growth, `YEST`
//!   budding, `COAL` smouldering, `SPNG` absorption/release and the
//!   `FOG`/`RIME` fog cycle;
//! * the explosive and ignition layer (`BOMB`, `FUSE`, `FSEP`, `DEST`, `BANG`,
//!   `IGNT`, `EMBR`, `FIRW`, `FWRK`, `LIGH`, `LITH`, `EXOT`, `SING`, `SOAP`,
//!   `GEL`) and the pressure gases (`ANAR`, `BOYL`, the cold branch of `C5`,
//!   the hot, over-pressured `O2` graviton burst).
//!
//! Purely neighbour-to-neighbour type changes live in [`super::fluid`]; the
//! generic transitions live in `thermal_kernel`.
//!
//! # TPT references
//!
//! * `BOMB.cpp:60-121` — 3x3 contact trigger, radius-8 kill/pressure disc and
//!   the bomb turning into `EMBR(tmp=2, life=2, MAX_TEMP)`.
//! * `C5.cpp:58-101` — the cold 5x5 trigger that turns `C5` into `CFLM`, and
//!   the `ctype` photon emission.
//! * `FUSE.cpp:65-92` — burn timer, `pv`/`tmp` pressure quench, `FSEP`
//!   conversion and spark/high-temperature ignition.
//! * `FSEP.cpp:58-84` — fuse powder countdown and its 400 K ignition point.
//! * `DEST.cpp:58-105` — random neighbour destruction, `MAX_TEMP` heating and
//!   the 60/80 unit pressure pulse.
//! * `BANG.cpp:50-122` — the four-stage `tmp` state machine, the connected-block
//!   `flood_prop` arming and the random `FIRE`/`SMKE`/`EMBR`/death result.
//! * `IGNT.cpp:58-91` — ignition cord trigger and its `EMBR`/`FIRE` emission.
//! * `ANAR.cpp:52-69` — anti-air dust turning into `CFLM` and cooling.
//! * `BOYL.cpp:55-92` — pressure equalisation over the 3x3 coarse cells, water
//!   fogging and the `O2` -> water reaction.
//! * `EMBR.cpp:56-70` — the spark dying against non-`PROP_SPARKSETTLE` matter.
//! * `FIRW.cpp:57-115` — firework launch kick and the `EMBR` burst.
//! * `FWRK.cpp:60-110` — heat/neutron launch, the random-colour burst and the
//!   45-frame life reset.
//! * `LIGH.cpp:70-240` — lightning heating, ignition, conductor sparking and
//!   the `DEUT`/`PLUT`/`HEAC`/`COAL` interactions.
//! * `LIGH.cpp:250-300` — the `tmp2` segment state machine.
//! * `LITH.cpp:58-260` — the stored-energy battery, water/`CO2` reactions,
//!   `SPRK`/`NSCN` charge exchange and the overcharge explosion.
//! * `EXOT.cpp:56-215` — pressure/`tmp2` diffusion, `PROT`/`LAVA`/`WARP`
//!   interactions and the `WARP` conversion thresholds.
//! * `SING.cpp:58-150` — the negative-pressure well, mass absorption and pop.
//! * `SOAP.cpp:104-290` — bubble flag, mate links, attach/detach and the
//!   neighbour spring.
//! * `GEL.cpp:60-140` — saturation, desaturation table and stickiness.
//! * `FIRE.cpp:176-190` + `Simulation.cpp:2623-2655,2731-2739` — thermite
//!   burning into 3500 K molten metal that holds its heat for 20 frames and
//!   freezes back into `BMTL`.
//! * `TRON.cpp:118-200,230-275,315-325` — head direction choice, the sight
//!   check against the wall map, tail growth and the create defaults.
//! * `EMP.cpp:99-260` — the pulse that breaks activated electronics.
//! * `CBNW.cpp:51-160` — the full carbonated-water update: degassing into
//!   `CO2` in low pressure, the `tmp2` speckle timer, the `tmp` blast countdown
//!   and the charge shared between touching soda pixels, plus rubber ignition
//!   and flame snuffing.
//! * `GLAS.cpp:50-96` — the pressure-differential shatter to `BGLA` against
//!   `int(pv * 64)` and the `LNTG` condensation on strengthened glass.
//!
//! # Kernel split
//!
//! Every kernel stays inside the sixteen-buffer ceiling (the base pass uses
//! fourteen), so the work is split by binding set across eleven child modules:
//!
//! | Module | Kernels | Purpose |
//! |--------|---------|---------|
//! | [`fire`] | [`combustion_reaction_kernel`] | `FIRE`, `PLSM`, `LAVA`, `H2`, `O2` chemistry & matter updates, `GOO` air drag, `SPNG` wring-out |
//! | [`fireworks`] | [`firework_kernel`] | `FIRW`, `FWRK` launch and burst |
//! | [`reactive`] | [`reactive_liquid_kernel`], [`exot_freeze_kernel`] | `LITH`, `EXOT`, `SING` reactive liquids |
//! | [`soap`] | [`soap_kernel`], [`soap_detach_kernel`], [`soap_spring_kernel`], [`gel_kernel`] | `SOAP` bubbles and `GEL` stickiness |
//! | [`lightning`] | [`ligh_heat_kernel`], [`ligh_prop_kernel`] | `LIGH` heating, propagation and `C5` photon |
//! | [`thermite`] | [`thermite_kernel`] | `THRM` thermite burning and molten lava |
//! | [`devices`] | [`tron_kernel`], [`emp_kernel`] | `TRON` snake and `EMP` electronics pulse |
//! | [`explosives`] | [`explosive_reaction_kernel`], [`fuse_kernel`] | `BOMB`, `DEST`, `BANG`, `EMBR` explosives and `FUSE` cords |
//! | [`gases`] | [`pressure_gas_kernel`], [`o2_burst_kernel`] | `ANAR`, `BOYL`, `C5` cold trigger, hot `O2` graviton burst |
//! | [`carbonated`] | [`carbonated_kernel`] | `CBNW` degassing, `tmp2` speckles, blast charge and rubber ignition |
//! | [`glass`] | [`glass_kernel`] | `GLAS` pressure-differential shatter and `LNTG` condensation |
//!
//! # Spawn contract
//!
//! Creations go through the device-side creation queue
//! ([`super::shared::request_create`]): a record carries the element id, the
//! pixel and any field overrides, and the drain seeds the rest from the
//! element's `DefaultProperties`, so the particle is what TPT's `create_part`
//! would have made. What the pass still cannot do:
//!
//! * `BOMB.cpp:88-121` creates `EMBR` in every pixel of a radius-9 disc; only
//!   the core `EMBR` exists here (the bomb itself becomes it).
//! * `FIRW.cpp:90-115` / `FWRK.cpp:78-100` create 40 `EMBR` per burst; the
//!   firework becomes a single `EMBR` with the same per-particle fields.
//! * `C5.cpp:83-98` gives the emitted photon its `ctype`, position offset and
//!   velocity; the spawned photon only has the registry defaults.
//! * `FIRW.cpp:104-115` / `FWRK.cpp:100-110` add 8 pressure units where the
//!   firework dies. [`firework_kernel`] is at eight bindings and cannot also
//!   carry `air_pv`, so the burst itself is pressure-neutral.
//! * `SING.cpp:100-140` creates up to 3019 `PHOT`/`NEUT`/`ELEC` on pop; the
//!   pop only contributes its pressure burst here.
//! * `TRON.cpp:194-232` copies the head state into the new head. The spawn
//!   contract cannot carry `tmp`/`tmp2`/`ctype`, so the head leaves its
//!   direction in `tmp` when it becomes a tail and the child inherits
//!   direction, hue and tail length from the adjacent tail with the largest
//!   `life` on its first frame. The child still lands on the host-chosen free
//!   neighbour rather than on the lane the sight check picked.
//! * `SOAP.cpp:110-180` walks and splices the bubble chain; this port keeps the
//!   links valid and applies the spring, but does not splice chains.
//! * `GEL.cpp:120-140` writes the neighbour's velocity; only the gel's own half
//!   of the sticky force is applied.
//! * `EMP.cpp` has no `Update`: TPT runs the pulse from the EMP *tool*. This
//!   port latches one pulse on a particle's `tmp` when an `EMP` appears, and
//!   evaluates the `triggerCount == 1` case of the binomial probabilities.
//! * `Simulation.cpp:2623-2655` freezes `LAVA(THRM)` into `BMTL` below 1273 K;
//!   `thermal_kernel.rs` owns that rule, and [`thermite_kernel`] covers the
//!   973..1273 K window where the lava is still liquid.
//! * `CFLM.cpp:57-60` (`Create`: life 50..199) and `TRON.cpp:315-325` have no
//!   `default_life` in the element table and both carry `PROP_LIFE_KILL`, so a
//!   painted particle of either dies on its first frame in `life.rs` before
//!   this pass can seed it.

pub(crate) mod carbonated;
pub(crate) mod devices;
pub(crate) mod explosives;
pub(crate) mod fire;
pub(crate) mod fireworks;
pub(crate) mod gases;
pub(crate) mod glass;
pub(crate) mod lightning;
pub(crate) mod reactive;
pub(crate) mod soap;
pub(crate) mod thermite;

pub(super) use self::carbonated::carbonated_kernel;
pub(super) use self::devices::{emp_kernel, tron_kernel};
pub(super) use self::explosives::{
    BANG_ARMED_NEXT_FRAME, BANG_REARMED, explosive_reaction_kernel, fuse_kernel,
};
pub(super) use self::fire::combustion_reaction_kernel;
pub(super) use self::fireworks::firework_kernel;
pub(super) use self::gases::{o2_burst_kernel, pressure_gas_kernel};
pub(super) use self::glass::glass_kernel;
pub(super) use self::lightning::{ligh_heat_kernel, ligh_prop_kernel};
pub(super) use self::reactive::{exot_freeze_kernel, reactive_liquid_kernel};
pub(super) use self::soap::{gel_kernel, soap_detach_kernel, soap_kernel, soap_spring_kernel};
pub(super) use self::thermite::thermite_kernel;

use cubecl::prelude::*;
use tpt_core::constants::*;
use tpt_core::elements::GpuElementProps;

pub(super) use super::shared::{
    CR_NO_OCCUPANCY, CR_SET_CTYPE, CR_SET_DCOLOUR, CR_SET_LIFE, CR_SET_TEMP, CR_SET_TMP,
    CR_SET_TMP2, CR_SET_VX, CR_SET_VY, FLAG_SPAWN_CLONE, FLAG_SPAWN_NEUT, FLAG_SPAWN_SELF,
    FLAG_SPAWN_TMP, WORD_EXPLOSIVE, WORD_FLAMMABLE, WORD_PROPERTIES, elem_word, hash_u32, iabs,
    insulated, mix_index, pmap_at, pmap_slot, rand_unit, request_create, request_create_plain,
    stasis_frozen,
};

/// `SPNG` absorption capacity (`SPNG.cpp` uses `limit = 50`).
pub(super) const SPONGE_CAPACITY: i32 = 50;

/// Word offset of [`GpuElementProps::heat_conduct`] (`IsHeatInsulator`).
pub(super) const WORD_HEAT_CONDUCT: u32 = 15;
/// Word offset of [`GpuElementProps::default_temp`] (`create_part` defaults).
pub(super) const WORD_DEFAULT_TEMP: u32 = 16;
/// Word offset of [`GpuElementProps::high_temp`] (used by the `LIGH`/`HEAC`).
pub(super) const WORD_HIGH_TEMP: u32 = 24;
/// Word offset of [`GpuElementProps::high_pressure`] (`FIRE.cpp:157`).
pub(super) const WORD_HIGH_PRESSURE: u32 = 20;

const _: () =
    assert!(core::mem::offset_of!(GpuElementProps, heat_conduct) == WORD_HEAT_CONDUCT as usize * 4);
const _: () =
    assert!(core::mem::offset_of!(GpuElementProps, default_temp) == WORD_DEFAULT_TEMP as usize * 4);
const _: () =
    assert!(core::mem::offset_of!(GpuElementProps, high_temp) == WORD_HIGH_TEMP as usize * 4);
const _: () = assert!(
    core::mem::offset_of!(GpuElementProps, high_pressure) == WORD_HIGH_PRESSURE as usize * 4
);

/// `SOAP.cpp:98`: water freezes into a solid bubble below this temperature.
pub(super) const SOAP_FREEZING: f32 = 248.15f32;
/// `FUSE.cpp:72`: pressure above this quenches a lit fuse (`pv > 2.7`).
pub(super) const FUSE_QUENCH_PRESSURE: f32 = 2.7f32;
/// `FUSE.cpp:86`: fuse ignition temperature (273.15 + 700).
pub(super) const FUSE_IGNITE_TEMP: f32 = 973.15f32;
/// `FSEP.cpp:76`: fuse-powder ignition temperature (273.15 + 400).
pub(super) const FSEP_IGNITE_TEMP: f32 = 673.15f32;
/// `BANG.cpp:53` / `IGNT.cpp:44`: TNT and ignition-cord trigger temperature.
pub(super) const TNT_IGNITE_TEMP: f32 = 673.0f32;
/// `DEST.cpp:77`: `MAX_TEMP / 4` (the fire/smoke explosion temperature).
pub(super) const DEST_FIRE_TEMP: f32 = MAX_TEMP / 4.0f32;
/// `DEST.cpp:96`: `MAX_TEMP / 3` (the ember explosion temperature).
pub(super) const DEST_EMBR_TEMP: f32 = MAX_TEMP / 3.0f32;
/// `LITH.cpp:96`: `273.15 + 700` is the `burnTimer` spark-over temperature.
pub(super) const LITH_BURN_START: i32 = 1013;
/// `LITH.cpp:90`: water quenches lithium above this burn timer.
pub(super) const LITH_WATER_BURN: i32 = 1016;
/// `LITH.cpp:86`: a burn timer above this spits `FIRE` into empty neighbours.
pub(super) const LITH_FIRE_SPIT_BURN: i32 = 1012;
/// `WTRV.cpp:57`: `273.15 + 12` K ignites rubber next to steam.
pub(super) const STEAM_IGNITE_K: f32 = 285.15f32;
/// `WTRV.cpp:76`: fire-steam above this sheds `temp / 1000` per frame.
pub(super) const STEAM_COOL_K: f32 = 1273.0f32;
/// `LITH.cpp:245`: overcharge threshold.
pub(super) const LITH_OVERCHARGE: i32 = 100;
/// `EXOT.cpp:159`: `tmp2` at which the exotic matter collapses into `WARP`.
pub(super) const EXOT_WARP_TMP2: i32 = 6000;
/// `LIGH.cpp:46`: the lightning power scale.
pub(super) const LIGHTING_POWER: f32 = 0.65f32;
/// `LIGH.cpp:70`: ambient-heat divisor (`hv += powderful / 50`).
pub(super) const LIGH_HV_DIV: f32 = 50.0f32;
/// `LIGH.cpp:206`: lightning segments shrink by this factor per generation
/// (`int(life/1.5 - rng.between(0, 1))`).
///
/// The kernel (`lightning.rs`) inlines the same literal in its segment roll, so
/// the mirror is referenced only by the host-side power-encoding test; it is
/// test-scoped to keep the library build free of dead constants.
#[cfg(test)]
pub(super) const LIGH_LIFE_SHRINK: f32 = 1.5f32;
/// `TRON.cpp:166`: a head that reached this life grows its tail.
pub(super) const TRON_GROW_LIFE: i32 = 100;
/// `TRON.cpp:120`: tail length a fresh `TRON` starts with.
pub(super) const TRON_TAIL_LEN: i32 = 4;

// `TRON.cpp:100-108`: the `tmp` bit field.
pub(super) const TRON_HEAD: i32 = 0x0001;
pub(super) const TRON_NOGROW: i32 = 0x0002;
pub(super) const TRON_WAIT: i32 = 0x0004;
pub(super) const TRON_NODIE: i32 = 0x0008;
pub(super) const TRON_DEATH: i32 = 0x0010;
pub(super) const TRON_NORANDOM: i32 = 0x1_0000;

/// `life` given to a freshly created `TRON` (`TRON.cpp:315-325`).
pub(super) const TRON_CREATE_LIFE: i32 = 5;
/// `TRON.cpp` keeps only these `tmp` bits on a head that just moved: `NODIE`,
/// `DEATH`, the direction (this port needs it for the spawned child) and the
/// 5-bit hue.
pub(super) const TRON_KEEP_MASK: i32 = 0xF878;

/// `BMTL`'s melting point, the temperature at which molten thermite freezes
/// back (`Simulation.cpp:2623`).
pub(super) const BMTL_HIGH_TEMP: f32 = 1273.0f32;
/// Temperature of burning thermite (`FIRE.cpp:185-190`).
pub(super) const THRM_BURN_TEMP: f32 = 3500.0f32;

/// `clamp(v, lo, hi)` for floats (CubeCL has no `clamp`).
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

/// `clamp(v, lo, hi)` for signed integers.
#[cube]
pub(super) fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    let mut r = v;
    if r < lo {
        r = lo;
    }
    if r > hi {
        r = hi;
    }
    r
}

/// TPT `sim->rng.chance(num, den)`: `between(0, den - 1) < num`.
#[cube]
pub(super) fn chance(seed: u32, num: u32, den: u32) -> bool {
    let mut hit = false;
    if den != 0u32 {
        hit = (hash_u32(seed) % den) < num;
    }
    hit
}

/// `rng.between(-1, 1)` for one axis of a cord's plasma offset (`FUSE.cpp:69`,
/// `FSEP.cpp:64`, `IGNT.cpp:79-88`).
#[cube]
pub(super) fn rng_offset(seed: u32) -> i32 {
    (hash_u32(seed ^ 0x2545_F491u32) % 3u32) as i32 - 1i32
}

/// `rng.between(-10, 10)` as a float: the ember's velocity kick
/// (`IGNT.cpp:81-82`).
#[cube]
pub(super) fn rng_kick(seed: u32) -> f32 {
    (hash_u32(seed ^ 0x68E9_50B1u32) % 21u32) as f32 - 10.0f32
}

/// Host mirror of the `#[cube]` avalanche in `shared::hash_u32`, so the two
/// device-side rolls above can be pinned by a test (AGENTS rule 11).
#[cfg(test)]
fn hash_u32_host(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16;
    h = h.wrapping_mul(2_246_822_507);
    h ^= h >> 13;
    h = h.wrapping_mul(3_266_489_909);
    h ^= h >> 16;
    h
}

/// Host mirror of [`rng_offset`].
#[cfg(test)]
fn rng_offset_host(seed: u32) -> i32 {
    (hash_u32_host(seed ^ 0x2545_F491) % 3) as i32 - 1
}

/// Host mirror of [`rng_kick`].
#[cfg(test)]
fn rng_kick_host(seed: u32) -> f32 {
    (hash_u32_host(seed ^ 0x68E9_50B1) % 21) as f32 - 10.0
}

/// `SEED.cpp:59-122`: seeds only germinate between 278.15 K and 343.15 K.
///
/// Written as two comparisons rather than a `RangeInclusive::contains` call
/// because this runs inside a kernel, where the host iterator method is not
/// available.
#[cube]
#[allow(clippy::manual_range_contains)]
pub(super) fn in_seed_germination_window(temp: f32) -> bool {
    temp >= 278.15f32 && 343.15f32 >= temp
}

/// The registry default temperature of `ty` (what `create_part` seeds).
#[cube]
pub(super) fn default_temp_of(words: &[u32], ty: u32) -> f32 {
    f32::reinterpret(elem_word(words, ty, WORD_DEFAULT_TEMP))
}

/// `SimulationData::IsHeatInsulator` (`SimulationData.cpp:338`): `HeatConduct == 0`.
#[cube]
pub(super) fn heat_insulator(words: &[u32], ty: u32) -> bool {
    elem_word(words, ty, WORD_HEAT_CONDUCT) == 0u32
}

/// `pv[cx, cy] += add` with canvas bounds (TPT's `pv` has a border cell).
#[cube]
pub(super) fn pv_add(pv: &mut [f32], xcells: u32, ycells: u32, cx: i32, cy: i32, add: f32) {
    if cx >= 0i32 && cy >= 0i32 && (cx as u32) < xcells && (cy as u32) < ycells {
        let i = ((cy as u32) * xcells + (cx as u32)) as usize;
        pv[i] += add;
    }
}

/// `pv[cx, cy]` clamped to the pressure range and stored back.
#[cube]
pub(super) fn pv_set_clamped(pv: &mut [f32], xcells: u32, ycells: u32, cx: i32, cy: i32, v: f32) {
    if cx >= 0i32 && cy >= 0i32 && (cx as u32) < xcells && (cy as u32) < ycells {
        let i = ((cy as u32) * xcells + (cx as u32)) as usize;
        pv[i] = clampf(v, MIN_PRESSURE, MAX_PRESSURE);
    }
}

/// Random `i32` in `lo..=hi` from `seed`.
#[cube]
pub(super) fn rand_range(seed: u32, lo: i32, hi: i32) -> i32 {
    let span = (hi - lo + 1i32) as u32;
    lo + (hash_u32(seed) % span) as i32
}

/// `gx` of the configured gravity direction index (0..7, 8 = none).
#[cube]
pub(super) fn gravity_gx(dir: u32) -> f32 {
    let mut gx = 0.0f32;
    if dir == 1u32 || dir == 3u32 {
        gx = -core::f32::consts::FRAC_1_SQRT_2;
    } else if dir == 2u32 {
        gx = -1.0f32;
    } else if dir == 5u32 || dir == 7u32 {
        gx = core::f32::consts::FRAC_1_SQRT_2;
    } else if dir == 6u32 {
        gx = 1.0f32;
    }
    gx
}

/// `gy` of the configured gravity direction index (0..7, 8 = none).
#[cube]
pub(super) fn gravity_gy(dir: u32) -> f32 {
    let mut gy = 0.0f32;
    if dir == 0u32 {
        gy = -1.0f32;
    } else if dir == 1u32 || dir == 7u32 {
        gy = -core::f32::consts::FRAC_1_SQRT_2;
    } else if dir == 3u32 || dir == 5u32 {
        gy = core::f32::consts::FRAC_1_SQRT_2;
    } else if dir == 4u32 {
        gy = 1.0f32;
    }
    gy
}

/// Host mirror of [`gravity_gx`] / [`gravity_gy`] for one direction index.
#[cfg(test)]
fn gravity_host(dir: u32) -> (f32, f32) {
    const D: f32 = core::f32::consts::FRAC_1_SQRT_2;
    match dir {
        0u32 => (0.0f32, -1.0f32),
        1u32 => (-D, -D),
        2u32 => (-1.0f32, 0.0f32),
        3u32 => (-D, D),
        4u32 => (0.0f32, 1.0f32),
        5u32 => (D, D),
        6u32 => (1.0f32, 0.0f32),
        7u32 => (D, -D),
        _ => (0.0f32, 0.0f32),
    }
}

/// Host mirror of [`tron_dir_x`].
#[cfg(test)]
fn tron_dir_host_x(dir: i32) -> i32 {
    if dir == 0 {
        -1
    } else if dir == 2 {
        1
    } else {
        0
    }
}

/// Host mirror of [`tron_dir_y`].
#[cfg(test)]
fn tron_dir_host_y(dir: i32) -> i32 {
    if dir == 1 {
        -1
    } else if dir == 3 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::super::electronics::tesc_ligh_life;
    use super::*;

    #[test]
    fn rng_offset_and_kick_stay_in_tpt_ranges() {
        // `rng.between(-1, 1)` and `rng.between(-10, 10)`: the cord offsets and
        // the ember kick must cover exactly those ranges and nothing wider.
        let mut offsets = [false; 3];
        let mut kicks = [false; 21];
        for n in 0u32..4096u32 {
            let seed = n.wrapping_mul(0x9E37_79B9);
            let o = rng_offset_host(seed);
            offsets[(o + 1) as usize] = true;
            let k = rng_kick_host(seed) as i32;
            kicks[(k + 10) as usize] = true;
        }
        assert_eq!(offsets, [true; 3]);
        assert_eq!(kicks, [true; 21]);
    }

    #[test]
    fn gravity_table_covers_all_nine_directions() {
        // 0..7 are the compass directions used by `special::gravity_dir_index`,
        // 8 is "undecided" and must fall back to the random kick.
        let expected = [
            (0.0f32, -1.0f32),
            (-1.0, -1.0),
            (-1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (1.0, -1.0),
            (0.0, 0.0),
        ];
        for (dir, (ex, ey)) in expected.iter().enumerate() {
            let (gx, gy) = gravity_host(dir as u32);
            let (ex, ey) = if dir == 8 {
                (*ex, *ey)
            } else {
                let len = (ex * ex + ey * ey).sqrt();
                (ex / len, ey / len)
            };
            assert!((gx - ex).abs() < 1e-6, "dir {dir} gx {gx} != {ex}");
            assert!((gy - ey).abs() < 1e-6, "dir {dir} gy {gy} != {ey}");
        }
        assert_eq!(gravity_host(8u32), (0.0, 0.0));
    }

    #[test]
    fn tron_direction_table_is_the_cpp_one() {
        // `TRON.cpp:130-131`: 0 left, 1 up, 2 right, 3 down.
        let table = [
            (0i32, (-1i32, 0i32)),
            (1, (0, -1)),
            (2, (1, 0)),
            (3, (0, 1)),
        ];
        for (dir, (dx, dy)) in table {
            assert_eq!(tron_dir_host_x(dir), dx);
            assert_eq!(tron_dir_host_y(dir), dy);
        }
    }

    #[test]
    fn tron_keep_mask_preserves_direction_and_hue() {
        let head = TRON_HEAD
            | TRON_WAIT
            | TRON_NOGROW
            | (2i32 << 5i32)
            | (17i32 << 11i32)
            | TRON_DEATH
            | TRON_NODIE;
        let tail = head & TRON_KEEP_MASK;
        assert_eq!(tail & TRON_HEAD, 0);
        assert_eq!(tail & TRON_WAIT, 0);
        assert_eq!(tail & TRON_NOGROW, 0);
        // The spawned child inherits the direction and the 5-bit hue, and a
        // crashed head keeps its death/nodie flags.
        assert_eq!((tail >> 5i32) & 3i32, 2);
        assert_eq!((tail >> 11i32) & 0x1F, 17);
        assert_ne!(tail & TRON_DEATH, 0);
        assert_ne!(tail & TRON_NODIE, 0);
        // The head bits of a freshly created `TRON` are exactly what
        // `TRON.cpp:315-325` sets.
        assert_eq!(TRON_HEAD, 1);
        assert_eq!(TRON_TAIL_LEN, 4);
        assert_eq!(TRON_CREATE_LIFE, 5);
    }

    #[test]
    fn ligh_power_encoding_shrinks_the_bolt() {
        // The segment stores `life * 7 / 1.5` in `tmp`; the host turns that
        // word into the child's life. Every generation must be shorter than its
        // parent (`LIGH.cpp:280-295`: `life /= 1.5`).
        for life in 2i32..=60i32 {
            let mut power = (life as f32 * (7.0f32 / LIGH_LIFE_SHRINK)) as i32;
            if power > 300i32 {
                power = 300i32;
            }
            let child = tesc_ligh_life(power, 0);
            assert!(child >= 1i32, "life {life} produced a dead bolt");
            assert!(child <= life, "life {life} grew to {child}");
        }
        // A one-life segment dies out instead of growing a new head.
        let power = (1.0f32 * (7.0f32 / LIGH_LIFE_SHRINK)) as i32;
        assert!(tesc_ligh_life(power, 0) <= 2i32);
        // The stored power never exceeds the coil limit.
        let mut power = (60.0f32 * (7.0f32 / LIGH_LIFE_SHRINK)) as i32;
        assert!(power <= 300i32);
        power = 400i32;
        assert_eq!(tesc_ligh_life(power, 0), tesc_ligh_life(300i32, 0));
    }

    #[test]
    fn thermite_constants_match_the_cpp() {
        // `Simulation.cpp:2623` freezes molten thermite at `BMTL`'s melting
        // point; `FIRE.cpp:185-190` burns it at 3500 K.
        assert_eq!(BMTL_HIGH_TEMP, 1273.0f32);
        assert_eq!(THRM_BURN_TEMP, 3500.0f32);
        assert_eq!(FUSE_QUENCH_PRESSURE, 2.7f32);
        assert_eq!(FUSE_IGNITE_TEMP, 973.15f32);
        assert_eq!(FSEP_IGNITE_TEMP, 673.15f32);
        assert_eq!(TNT_IGNITE_TEMP, 673.0f32);
        assert_eq!(SOAP_FREEZING, 248.15f32);
        assert_eq!(LITH_BURN_START, 1013);
        assert_eq!(LITH_WATER_BURN, 1016);
        assert_eq!(LITH_FIRE_SPIT_BURN, 1012);
        assert_eq!(STEAM_IGNITE_K, 273.15f32 + 12.0f32);
        assert_eq!(STEAM_COOL_K, 1273.0f32);
        assert_eq!(LITH_OVERCHARGE, 100);
        assert_eq!(EXOT_WARP_TMP2, 6000);
    }

    #[test]
    fn wood_pyrolysis_and_peak_temp() {
        fn update_wood(temp: f32, pv: f32, mut tmp: i32) -> (u32, i32) {
            let mut ty = PT_WOOD;
            if temp > 450.0f32 && temp > (tmp as f32) {
                tmp = temp as i32;
            }
            if temp > 773.0f32 && pv <= -10.0f32 {
                ty = PT_BCOL;
            }
            (ty, tmp)
        }
        let (ty1, tmp1) = update_wood(500.0, 0.0, 0);
        assert_eq!(ty1, PT_WOOD);
        assert_eq!(tmp1, 500);

        let (ty2, _) = update_wood(800.0, -15.0, 500);
        assert_eq!(ty2, PT_BCOL);

        let (ty3, _) = update_wood(800.0, 5.0, 500);
        assert_eq!(ty3, PT_WOOD);
    }

    #[test]
    fn brmt_temp_factor_formula() {
        // `BRMT.cpp:51-55`: auto tempFactor = int(1000 - ((523.15f-parts[i].temp)*2));
        fn brmt_tempfactor(temp: f32) -> u32 {
            let tf = 1000i32 - (((523.15f32 - temp) * 2.0f32) as i32);
            if tf < 2i32 { 2u32 } else { tf as u32 }
        }
        assert_eq!(brmt_tempfactor(523.15), 1000);
        assert!(brmt_tempfactor(600.0) > 1000);
        assert_eq!(brmt_tempfactor(0.0), 2);
    }

    #[test]
    fn exot_tmp_cycle() {
        let mut tmp = 1i32;
        tmp -= 1i32;
        if tmp <= 0i32 {
            tmp = 250i32;
        }
        assert_eq!(tmp, 250i32);
    }
}
