//! GPU element table and palette upload (plan §6.1).
//!
//! The simulation kernels read one [`GPU_ELEMENT_WORDS`]-word row per element
//! id from [`VoxelElementGpu::table`]; the renderer reads the matching RGBA
//! [`VoxelElementGpu::palette`]. Both are built by `tpt-core` so the host
//! reference code and the device agree on every constant.
//!
//! [`spawn_temp_k`] is the one piece of element policy that has to live on the
//! host: a painted voxel cannot inherit the ambient temperature for every
//! element. Lava that spawns at 293 K would freeze on its first tick, ice at
//! ambient would melt, and fire would not glow. The brush kernel receives the
//! resulting fixed-point temperature as an argument, so the kernel body never
//! embeds these thresholds.

use cubecl::client::Client;
use cubecl::server::Handle;
use tpt_core::constants::{
    PT_EXOT, PT_FIRE, PT_GPMP, PT_GRVT, PT_ICEI, PT_ISZS, PT_LAVA, PT_NEUT, PT_PHOT, PT_SING,
    PT_SMKE, PT_SNOW, PT_URAN, PT_WARP, PT_WTRV,
};
use tpt_core::voxel3d::{GPU_ELEMENT_ROWS, GPU_ELEMENT_WORDS, GPU_PALETTE_FLOATS};

/// Element-table value: temperature of freshly painted lava.
///
/// `LAVA.cpp:34`: `DefaultProperties.temp = R_TEMP + 1500.0f + 273.15f`
/// with `R_TEMP = 22` (`SimulationConfig.h:66`), i.e. 1795.15 K. The value is
/// far above the 973 K quench point, so painted lava stays molten until it is
/// cooled by a neighbour.
pub const SPAWN_LAVA_K: f32 = 1795.15;
/// Element-table value: temperature of freshly painted fire.
///
/// `FIRE.cpp:35`: `DefaultProperties.temp = R_TEMP + 400.0f + 273.15f` =
/// 695.15 K. The packed word can represent it exactly.
pub const SPAWN_FIRE_K: f32 = 695.15;
/// Element-table value: temperature of freshly painted ice.
///
/// `ICEI.cpp:31`: `DefaultProperties.temp = R_TEMP - 50.0f + 273.15f` =
/// 245.15 K, below the 252.05 K sublimation point.
pub const SPAWN_ICE_K: f32 = 245.15;
/// Element-table value: temperature of freshly painted snow.
///
/// `SNOW.cpp:32`: `DefaultProperties.temp = R_TEMP - 30.0f + 273.15f` =
/// 265.15 K. `ICEI.h`/`SNOW.h` share the 252.05 K sublimation transition, but
/// the `ST` branch (`Simulation.cpp:2545-2569`) keeps snow solid below the
/// 273.15 K freezing point of its water ctype.
pub const SPAWN_SNOW_K: f32 = 265.15;
/// Element-table value: temperature of freshly painted steam.
///
/// `WTRV.cpp:31`: `DefaultProperties.temp = R_TEMP + 100.0f + 273.15f` =
/// 395.15 K, comfortably above the 371 K condensation point.
pub const SPAWN_STEAM_K: f32 = 395.15;
/// Element-table value: temperature of freshly painted smoke.
///
/// `SMKE.cpp:31`: `DefaultProperties.temp = R_TEMP + 320.0f + 273.15f` =
/// 615.15 K, below its 625 K ignition transition.
pub const SPAWN_SMOKE_K: f32 = 615.15;
/// Element-table value: temperature of freshly painted uranium.
///
/// `URAN.cpp:32`: `DefaultProperties.temp = R_TEMP + 30.0f + 273.15f` =
/// 325.15 K.
pub const SPAWN_URAN_K: f32 = 325.15;
/// Element-table value: temperature of freshly painted photons.
///
/// `PHOT.cpp:35`: `DefaultProperties.temp = R_TEMP + 900.0f + 273.15f` =
/// 1195.15 K.
pub const SPAWN_PHOT_K: f32 = 1195.15;
/// Element-table value: temperature of freshly painted neutrons.
///
/// `NEUT.cpp:36`: `DefaultProperties.temp = R_TEMP + 4.0f + 273.15f` =
/// 299.15 K.
pub const SPAWN_NEUT_K: f32 = 299.15;
/// Element-table value: temperature of freshly painted isotope-Z solid.
///
/// `ISZS.cpp:31`: `DefaultProperties.temp = 140.0f`; the value is absolute
/// Kelvin, not an `R_TEMP` offset.
pub const SPAWN_ISZS_K: f32 = 140.0;

/// Number of ticks a freshly painted fire voxel starts with.
///
/// The engine prefers [`tpt_core::voxel3d::spawn_defaults`], whose `FIRE` hook
/// draws `rng.between(120, 169)` (`FIRE.cpp:421-424`); this constant is the
/// lower bound of that range for the fallback path.
pub const SPAWN_FIRE_LIFE: u32 = 120;
/// Life assigned to the smoke a fire decays into.
///
/// `FIRE.cpp:89-94`: `life = rng.between(250, 269)`; this is the lower bound
/// used by the fallback path.
pub const SPAWN_SMOKE_LIFE: u32 = 250;

/// Device copies of the element table and the renderer palette.
pub struct VoxelElementGpu {
    /// `GPU_ELEMENT_ROWS * GPU_ELEMENT_WORDS` `u32` words, indexed by element
    /// id. Float fields are `f32::to_bits` values.
    pub table: Handle,
    /// `GPU_ELEMENT_ROWS` `vec4<f32>` entries (`r, g, b, a`).
    pub palette: Handle,
}

impl VoxelElementGpu {
    /// Uploads the fifty-nine element table and palette.
    pub fn new(client: Client) -> Self {
        let table = tpt_core::voxel3d::gpu_table();
        let palette = tpt_core::voxel3d::gpu_palette();
        debug_assert_eq!(table.len(), GPU_ELEMENT_ROWS * GPU_ELEMENT_WORDS);
        debug_assert_eq!(palette.len(), GPU_PALETTE_FLOATS);
        Self {
            table: client.create_from_slice(bytemuck::cast_slice(&table)),
            palette: client.create_from_slice(bytemuck::cast_slice(&palette)),
        }
    }
}

/// Temperature a newly painted `ty` voxel starts at.
///
/// Elements without an explicit spawn temperature start at `ambient_k`, which
/// keeps inert solids and powders in thermal equilibrium with the world.
pub fn spawn_temp_k(ty: u32, ambient_k: f32) -> f32 {
    match ty {
        PT_LAVA => SPAWN_LAVA_K,
        PT_FIRE => SPAWN_FIRE_K,
        PT_ICEI => SPAWN_ICE_K,
        PT_SNOW => SPAWN_SNOW_K,
        PT_WTRV => SPAWN_STEAM_K,
        PT_SMKE => SPAWN_SMOKE_K,
        PT_URAN => SPAWN_URAN_K,
        PT_PHOT => SPAWN_PHOT_K,
        PT_NEUT => SPAWN_NEUT_K,
        PT_ISZS => SPAWN_ISZS_K,
        _ => ambient_k,
    }
}

/// Initial life counter of a newly painted `ty` voxel.
///
/// The engine prefers [`tpt_core::voxel3d::spawn_defaults`], which encodes the
/// element `Create` hooks. This fallback mirrors the lower bound of each
/// randomised `Create` range so a repainted voxel still lives long enough for
/// its update to run:
///
/// * `PHOT.cpp:50` — fixed `life = 680`.
/// * `NEUT.cpp:265-267` — `rng.between(480, 959)`.
/// * `GRVT.cpp:66-69` — `250 + rng.between(0, 199)`.
/// * `SING.cpp:154-156` — `rng.between(60, 109)`.
/// * `WARP.cpp:96-98` — `rng.between(70, 164)`.
/// * `EXOT.cpp:48` — fixed `life = 1000`.
/// * `GPMP.cpp:46` — fixed `life = 10`.
pub fn spawn_life(ty: u32) -> u32 {
    match ty {
        PT_FIRE => SPAWN_FIRE_LIFE,
        PT_PHOT => 680,
        PT_NEUT => 480,
        PT_GRVT => 250,
        PT_SING => 60,
        PT_WARP => 70,
        PT_EXOT => 1000,
        PT_GPMP => 10,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_core::constants::{PT_SAND, PT_SNOW, PT_STNE};
    use tpt_core::voxel3d::{VOXEL_ELEMENTS, element};

    #[test]
    fn every_spawn_temperature_stays_inside_its_own_transitions() {
        let ambient = 293.15f32;

        // `LAVA.cpp:34`: spawns at R_TEMP + 1500 + 273.15 and therefore stays
        // molten until a neighbour cools it below 973 K.
        let lava = element(PT_LAVA).unwrap();
        let lava_spawn = spawn_temp_k(PT_LAVA, ambient);
        assert_eq!(lava_spawn, SPAWN_LAVA_K);
        assert!(lava_spawn > lava.freeze_k + 2.0);

        // `ICEI.cpp:31`: spawns below the 252.05 K sublimation point.
        let ice = element(PT_ICEI).unwrap();
        let ice_spawn = spawn_temp_k(PT_ICEI, ambient);
        assert_eq!(ice_spawn, SPAWN_ICE_K);
        assert!(ice_spawn + 2.0 < ice.melt_k);

        // `SNOW.cpp:32`: spawns at 265.15 K. The `ST` branch of TransitionPhase
        // only sublimates it above the 273.15 K freezing point of its water
        // ctype, so the powder survives at its spawn temperature.
        assert_eq!(spawn_temp_k(PT_SNOW, ambient), SPAWN_SNOW_K);
        assert!(SPAWN_SNOW_K + 2.0 < 273.15);

        let steam = element(PT_WTRV).unwrap();
        let steam_spawn = spawn_temp_k(PT_WTRV, ambient);
        assert_eq!(steam_spawn, SPAWN_STEAM_K);
        assert!(steam_spawn > steam.freeze_k + 2.0);

        let fire = element(PT_FIRE).unwrap();
        let fire_spawn = spawn_temp_k(PT_FIRE, ambient);
        assert_eq!(fire_spawn, SPAWN_FIRE_K);
        assert!(fire_spawn > fire.freeze_k + 2.0);

        assert_eq!(spawn_temp_k(PT_SAND, ambient), ambient);
        assert_eq!(spawn_temp_k(PT_STNE, ambient), ambient);
        assert_eq!(spawn_life(PT_STNE), 0);
        assert_eq!(spawn_life(PT_FIRE), SPAWN_FIRE_LIFE);
        assert_eq!(spawn_life(PT_PHOT), 680);
    }

    #[test]
    fn every_element_spawns_on_the_safe_side_of_its_transitions() {
        let ambient = 293.15f32;
        for elem in &VOXEL_ELEMENTS {
            let spawn = spawn_temp_k(elem.id, ambient);
            // ICEI/SNOW encode the `HighTemperatureTransition == ST` branch in
            // their 252.05 K melt slot; that branch only fires once the ice
            // ctype's own freezing point is reached (273.15 K for water), so
            // the raw melt threshold is not the effective one for their spawn.
            if elem.melt_k > 0.0 && !matches!(elem.id, PT_ICEI | PT_SNOW) {
                assert!(
                    spawn + 2.0 < elem.melt_k,
                    "{} spawns at {spawn} K, at or above its {} K melt point",
                    elem.name,
                    elem.melt_k
                );
            }
            if elem.boil_k > 0.0 {
                assert!(
                    spawn + 2.0 < elem.boil_k,
                    "{} spawns at {spawn} K, at or above its {} K boiling point",
                    elem.name,
                    elem.boil_k
                );
            }
            if elem.freeze_k > 0.0 {
                assert!(
                    spawn > elem.freeze_k + 2.0,
                    "{} spawns at {spawn} K, at or below its {} K freeze point",
                    elem.name,
                    elem.freeze_k
                );
            }
            if elem.ignition_k > 0.0 {
                assert!(
                    spawn + 2.0 < elem.ignition_k,
                    "{} spawns at {spawn} K, at or above its {} K ignition point",
                    elem.name,
                    elem.ignition_k
                );
            }
        }

        // Exact `DefaultProperties.temp` values from the C++ element files.
        assert_eq!(spawn_temp_k(PT_LAVA, ambient), 1795.15);
        assert_eq!(spawn_temp_k(PT_FIRE, ambient), 695.15);
        assert_eq!(spawn_temp_k(PT_ICEI, ambient), 245.15);
        assert_eq!(spawn_temp_k(PT_SNOW, ambient), 265.15);
        assert_eq!(spawn_temp_k(PT_WTRV, ambient), 395.15);
        assert_eq!(spawn_temp_k(PT_SMKE, ambient), 615.15);
        assert_eq!(spawn_temp_k(PT_URAN, ambient), 325.15);
        assert_eq!(spawn_temp_k(PT_PHOT, ambient), 1195.15);
        assert_eq!(spawn_temp_k(PT_NEUT, ambient), 299.15);
        assert_eq!(spawn_temp_k(PT_ISZS, ambient), 140.0);

        // Every other 3D element starts at ambient unless `spawn_defaults`
        // (the C++ `DefaultProperties` table) provides an override.
        for elem in &VOXEL_ELEMENTS {
            if matches!(
                elem.id,
                PT_LAVA
                    | PT_FIRE
                    | PT_ICEI
                    | PT_SNOW
                    | PT_WTRV
                    | PT_SMKE
                    | PT_URAN
                    | PT_PHOT
                    | PT_NEUT
                    | PT_ISZS
            ) {
                continue;
            }
            assert_eq!(
                spawn_temp_k(elem.id, ambient),
                ambient,
                "{} should spawn at ambient temperature",
                elem.name
            );
        }
    }
}
