//! Wall types — the coarse `bmap` layer that overlays the particle grid.
//!
//! The simulation stores one wall id per `CELL x CELL` block (`WL_NUM` built-in
//! types plus empty). Walls never occupy particle slots; they only modify how
//! air, heat and particles behave inside the block. This module mirrors the
//! reference Powder Toy `wall_type` table (`SimulationData.cpp`) and the two
//! derived block maps built in `Air::ApproximateBlockAirMaps`:
//!
//! * `bmap_blockair` — air (and therefore pressure/velocity) cannot enter the
//!   block. See [`is_wall_blocking_air`].
//! * `bmap_blockairh` — ambient heat cannot enter the block, either because the
//!   wall blocks air or because it is a [`WL_GRAV`] wall. See
//!   [`is_wall_blocking_heat`].
//!
//! Particle blocking is not stored in a map: it depends on the matter state of
//! the particle that tries to occupy the block. See
//! [`is_wall_blocking_particle`].

use serde::{Deserialize, Serialize};

use crate::constants::*;

/// One wall type definition.
///
/// Field layout and semantics follow the reference `wall_type` struct. `colour`
/// and `eglow` are packed `0x00RRGGBB` values; `drawstyle` selects one of the
/// procedural wall patterns drawn by the render kernel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WallDef {
    /// Wall id, `0 .. WL_NUM` (`WL_ERASE` .. `WL_STASIS`).
    pub id: u8,
    /// Stable Powder Toy identifier, e.g. `DEFAULT_WL_WALL`.
    pub identifier: String,
    /// Human readable name, e.g. `WALL`.
    pub name: String,
    /// English description shown in the UI.
    pub description: String,
    /// Base colour, `0x00RRGGBB`.
    pub colour: u32,
    /// Colour added to the fire/glow buffer when the wall is electrified
    /// (`emap != 0`), `0x00RRGGBB`.
    pub eglow: u32,
    /// Procedural draw pattern: `0` = per-wall default, `1` = sparse dots,
    /// `2` = dot columns, `3` = solid fill, `4` = diagonal `colour`/`eglow`
    /// stripes.
    pub drawstyle: u8,
    /// Whether the wall type blocks air while unpowered.
    ///
    /// [`WL_EWALL`] is listed as blocking because it blocks air at `emap == 0`
    /// and becomes transparent once electrified; use [`is_wall_blocking_air`]
    /// for the authoritative, charge-aware answer.
    pub blocks_air: bool,
    /// Whether the wall type blocks ambient heat transfer.
    ///
    /// This is the air blocking set plus [`WL_GRAV`], which blocks heat while
    /// letting air through.
    pub blocks_heat: bool,
    /// Whether the wall conducts electricity (`Simulation::is_wire`).
    pub is_wire: bool,
}

/// Builds a wall definition, deriving the three blocking flags from the id so
/// that the table can never drift away from the predicates below.
fn wall_def(
    id: u8,
    identifier: &str,
    name: &str,
    description: &str,
    colour: u32,
    eglow: u32,
    drawstyle: u8,
) -> WallDef {
    WallDef {
        id,
        identifier: identifier.to_string(),
        name: name.to_string(),
        description: description.to_string(),
        colour,
        eglow,
        drawstyle,
        blocks_air: wall_blocks_air_when_unpowered(id),
        blocks_heat: wall_blocks_heat_when_unpowered(id),
        is_wire: wall_conducts(id),
    }
}

/// Static (unpowered) air blocking set, i.e. the wall types that can set
/// `bmap_blockair`.
fn wall_blocks_air_when_unpowered(id: u8) -> bool {
    matches!(id, WL_WALL | WL_WALLELEC | WL_BLOCKAIR | WL_EWALL)
}

/// Static (unpowered) heat blocking set, i.e. the wall types that can set the
/// `0x8` (fully insulating) bit of `bmap_blockairh`.
fn wall_blocks_heat_when_unpowered(id: u8) -> bool {
    id == WL_GRAV || wall_blocks_air_when_unpowered(id)
}

/// Wall types that take part in spark propagation (`Simulation::is_wire`).
fn wall_conducts(id: u8) -> bool {
    matches!(
        id,
        WL_DETECT
            | WL_EWALL
            | WL_ALLOWLIQUID
            | WL_WALLELEC
            | WL_ALLOWALLELEC
            | WL_EHOLE
            | WL_STASIS
    )
}

/// Returns the full built-in wall table, indexed by wall id.
///
/// The returned vector always has exactly [`WL_NUM`] entries and
/// `walls[WL_X as usize].id == WL_X`. Colours, glow colours, draw styles and
/// descriptions are copied from the reference Powder Toy wall table.
pub fn default_walls() -> Vec<WallDef> {
    let walls = vec![
        wall_def(
            WL_ERASE,
            "DEFAULT_WL_ERASE",
            "ERASE",
            "Erases walls.",
            0x0080_8080,
            0x0000_0000,
            0,
        ),
        wall_def(
            WL_WALLELEC,
            "DEFAULT_WL_CNDTW",
            "CONDUCTIVE WALL",
            "Blocks everything. Conductive.",
            0x00C0_C0C0,
            0x0010_1010,
            0,
        ),
        wall_def(
            WL_EWALL,
            "DEFAULT_WL_EWALL",
            "EWALL",
            "E-Wall. Becomes transparent when electricity is connected.",
            0x0080_8080,
            0x0080_8080,
            0,
        ),
        wall_def(
            WL_DETECT,
            "DEFAULT_WL_DTECT",
            "DETECTOR",
            "Detector. Generates electricity when a particle is inside.",
            0x00FF_8080,
            0x00FF_2008,
            1,
        ),
        wall_def(
            WL_STREAM,
            "DEFAULT_WL_STRM",
            "STREAMLINE",
            "Streamline. Creates a line that follows air movement.",
            0x0080_8080,
            0x0000_0000,
            0,
        ),
        wall_def(
            WL_FAN,
            "DEFAULT_WL_FAN",
            "FAN",
            "Fan. Accelerates air. Use the line tool to set direction and strength.",
            0x0080_80FF,
            0x0000_0000,
            1,
        ),
        wall_def(
            WL_ALLOWLIQUID,
            "DEFAULT_WL_LIQD",
            "LIQUID WALL",
            "Allows liquids, blocks all other particles. Conductive.",
            0x00C0_C0C0,
            0x0010_1010,
            2,
        ),
        wall_def(
            WL_DESTROYALL,
            "DEFAULT_WL_ABSRB",
            "ABSORB WALL",
            "Absorbs particles but lets air currents through.",
            0x0080_8080,
            0x0000_0000,
            1,
        ),
        wall_def(
            WL_WALL,
            "DEFAULT_WL_WALL",
            "WALL",
            "Basic wall, blocks everything.",
            0x0080_8080,
            0x0000_0000,
            3,
        ),
        wall_def(
            WL_ALLOWAIR,
            "DEFAULT_WL_AIR",
            "AIRONLY WALL",
            "Allows air, but blocks all particles.",
            0x003C_3C3C,
            0x0000_0000,
            1,
        ),
        wall_def(
            WL_ALLOWPOWDER,
            "DEFAULT_WL_POWDR",
            "POWDER WALL",
            "Allows powders, blocks all other particles.",
            0x0057_5757,
            0x0000_0000,
            1,
        ),
        wall_def(
            WL_ALLOWALLELEC,
            "DEFAULT_WL_CNDTR",
            "CONDUCTOR",
            "Conductor. Allows all particles to pass through and conducts electricity.",
            0x00FF_FF22,
            0x0010_1010,
            2,
        ),
        wall_def(
            WL_EHOLE,
            "DEFAULT_WL_EHOLE",
            "EHOLE",
            "E-Hole. absorbs particles, releases them when powered.",
            0x0024_2424,
            0x0010_1010,
            0,
        ),
        wall_def(
            WL_ALLOWGAS,
            "DEFAULT_WL_GAS",
            "GAS WALL",
            "Allows gases, blocks all other particles.",
            0x0057_9777,
            0x0000_0000,
            1,
        ),
        wall_def(
            WL_GRAV,
            "DEFAULT_WL_GRVTY",
            "GRAVITY WALL",
            "Gravity wall. Newtonian Gravity has no effect inside a box drawn with this.",
            0x00FF_EE00,
            0x00AA_9900,
            4,
        ),
        wall_def(
            WL_ALLOWENERGY,
            "DEFAULT_WL_ENRGY",
            "ENERGY WALL",
            "Allows energy particles, blocks all other particles.",
            0x00FF_AA00,
            0x00AA_5500,
            4,
        ),
        wall_def(
            WL_BLOCKAIR,
            "DEFAULT_WL_NOAIR",
            "AIRBLOCK WALL",
            "Allows all particles, but blocks air.",
            0x00DC_DCDC,
            0x0000_0000,
            1,
        ),
        wall_def(
            WL_ERASEALL,
            "DEFAULT_WL_ERASEA",
            "ERASEALL",
            "Erases walls, particles, and signs.",
            0x0080_8080,
            0x0000_0000,
            0,
        ),
        wall_def(
            WL_STASIS,
            "DEFAULT_WL_STASIS",
            "STASIS WALL",
            "Freezes particles inside the wall in place until powered.",
            0x0080_0080,
            0x0000_0000,
            0,
        ),
    ];
    debug_assert_eq!(walls.len(), WL_NUM);
    walls
}

/// Returns `true` if a particle with element properties `elem_props` is killed
/// by the wall in its block.
///
/// `emap_val` is the block's electrical charge (`0` = unpowered). This mirrors
/// the reference `Simulation::update_particles` kill check: solid walls and
/// state-selective walls only admit their configured matter state, and
/// [`WL_EWALL`] only blocks while unpowered. [`WL_ERASE`] is the "no wall"
/// value and never blocks; [`WL_STASIS`] freezes rather than kills.
#[inline]
pub fn is_wall_blocking_particle(wall_id: u8, emap_val: u8, elem_props: u32) -> bool {
    match wall_id {
        WL_WALL | WL_WALLELEC | WL_ALLOWAIR | WL_DESTROYALL => true,
        WL_ALLOWLIQUID => elem_props & TYPE_LIQUID == 0,
        WL_ALLOWPOWDER => elem_props & TYPE_PART == 0,
        WL_ALLOWGAS => elem_props & TYPE_GAS == 0,
        WL_ALLOWENERGY => elem_props & TYPE_ENERGY == 0,
        WL_EWALL => emap_val == 0,
        _ => false,
    }
}

/// Returns `true` if the wall prevents air from entering its block
/// (`bmap_blockair`).
///
/// [`WL_EWALL`] blocks air only while unpowered (`emap_val == 0`).
#[inline]
pub fn is_wall_blocking_air(wall_id: u8, emap_val: u8) -> bool {
    matches!(wall_id, WL_WALL | WL_WALLELEC | WL_BLOCKAIR) || (wall_id == WL_EWALL && emap_val == 0)
}

/// Returns `true` if the wall prevents ambient heat from entering its block
/// (`bmap_blockairh & 0x8`).
///
/// This is the air blocking set plus [`WL_GRAV`], which insulates heat while
/// still letting air through.
#[inline]
pub fn is_wall_blocking_heat(wall_id: u8, emap_val: u8) -> bool {
    wall_id == WL_GRAV || is_wall_blocking_air(wall_id, emap_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_indexed_by_wall_id() {
        let walls = default_walls();
        assert_eq!(walls.len(), WL_NUM);
        for (index, wall) in walls.iter().enumerate() {
            assert_eq!(wall.id, index as u8, "wall {index} has a mismatched id");
            assert!(!wall.identifier.is_empty());
            assert!(!wall.name.is_empty());
            assert!(!wall.description.is_empty());
        }
        assert_eq!(walls[WL_WALL as usize].identifier, "DEFAULT_WL_WALL");
        assert_eq!(walls[WL_GRAV as usize].colour, 0x00FF_EE00);
        assert_eq!(walls[WL_GRAV as usize].eglow, 0x00AA_9900);
        assert_eq!(walls[WL_STASIS as usize].colour, 0x0080_0080);
    }

    #[test]
    fn particle_blocking_follows_matter_state() {
        assert!(is_wall_blocking_particle(WL_WALL, 0, TYPE_PART));
        assert!(is_wall_blocking_particle(WL_DESTROYALL, 0, TYPE_ENERGY));
        assert!(!is_wall_blocking_particle(WL_ERASE, 0, TYPE_PART));
        assert!(!is_wall_blocking_particle(WL_STASIS, 0, TYPE_PART));

        assert!(!is_wall_blocking_particle(WL_ALLOWLIQUID, 0, TYPE_LIQUID));
        assert!(is_wall_blocking_particle(WL_ALLOWLIQUID, 0, TYPE_PART));
        assert!(!is_wall_blocking_particle(WL_ALLOWPOWDER, 0, TYPE_PART));
        assert!(is_wall_blocking_particle(WL_ALLOWPOWDER, 0, TYPE_GAS));
        assert!(!is_wall_blocking_particle(WL_ALLOWGAS, 0, TYPE_GAS));
        assert!(is_wall_blocking_particle(WL_ALLOWGAS, 0, TYPE_LIQUID));
        assert!(!is_wall_blocking_particle(WL_ALLOWENERGY, 0, TYPE_ENERGY));
        assert!(is_wall_blocking_particle(WL_ALLOWENERGY, 0, TYPE_SOLID));
        assert!(!is_wall_blocking_particle(WL_ALLOWALLELEC, 0, TYPE_PART));

        assert!(is_wall_blocking_particle(WL_EWALL, 0, TYPE_PART));
        assert!(!is_wall_blocking_particle(WL_EWALL, 1, TYPE_PART));
    }

    #[test]
    fn air_and_heat_blocking_match_reference() {
        assert!(is_wall_blocking_air(WL_WALL, 0));
        assert!(is_wall_blocking_air(WL_WALLELEC, 0));
        assert!(is_wall_blocking_air(WL_BLOCKAIR, 0));
        assert!(!is_wall_blocking_air(WL_ALLOWAIR, 0));
        assert!(!is_wall_blocking_air(WL_GRAV, 0));
        assert!(is_wall_blocking_air(WL_EWALL, 0));
        assert!(!is_wall_blocking_air(WL_EWALL, 8));

        assert!(is_wall_blocking_heat(WL_GRAV, 0));
        assert!(is_wall_blocking_heat(WL_WALL, 0));
        assert!(!is_wall_blocking_heat(WL_ALLOWAIR, 0));
        assert!(!is_wall_blocking_heat(WL_EWALL, 8));
    }

    #[test]
    fn metadata_flags_are_consistent_with_predicates() {
        for wall in default_walls() {
            assert_eq!(
                wall.blocks_air,
                is_wall_blocking_air(wall.id, 0),
                "blocks_air mismatch for {}",
                wall.identifier
            );
            assert_eq!(
                wall.blocks_heat,
                is_wall_blocking_heat(wall.id, 0),
                "blocks_heat mismatch for {}",
                wall.identifier
            );
        }

        let wires: Vec<u8> = default_walls()
            .into_iter()
            .filter(|wall| wall.is_wire)
            .map(|wall| wall.id)
            .collect();
        assert_eq!(
            wires,
            vec![
                WL_WALLELEC,
                WL_EWALL,
                WL_DETECT,
                WL_ALLOWLIQUID,
                WL_ALLOWALLELEC,
                WL_EHOLE,
                WL_STASIS
            ]
        );
    }
}
