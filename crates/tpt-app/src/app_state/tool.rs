//! Drawing tools bound to the mouse buttons.
//!
//! Besides the element/wall/brush tools this mirrors the reference game's tool
//! classes: `SimTool` (heat, air, gravity wells, ...), `PropertyTool`
//! (`DEFAULT_UI_PROPERTY`), `DecorationTool` (`DEFAULT_DECOR_*`), `SampleTool`
//! (`DEFAULT_UI_SAMPLE`) and the single-particle spawn tool.

use crate::i18n::{t, t_fmt};
use tpt_core::commands::{DecoMode, RigidSpawnShape, SimToolKind};

/// Tool currently assigned to a mouse button.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectedTool {
    /// Place particles of the given element id (`PT_*` constant).
    Element(u32),
    /// Draw the given wall id (`WL_*` constant).
    Wall(u8),
    /// Apply a non-drawing simulation tool (heat, wind, gravity, ...).
    SimTool(SimToolKind),
    /// Spawn a rigid body cluster of the given shape and element.
    RigidBody {
        /// Shape rasterized for the new rigid body.
        shape: RigidSpawnShape,
        /// Element id of the pixels making up the body.
        elem_ty: u32,
    },
    /// Erase particles and walls under the brush.
    Eraser,
    /// `DEFAULT_UI_PROPERTY`: rewrite `tmp`/`tmp2`/`life`/`ctype` of the
    /// particles under the brush.
    Property,
    /// Decoration tool (`DEFAULT_DECOR_*`): blend `dcolour` into the particles
    /// under the brush with the given [`DecoMode`]. Each reference decoration
    /// tool is one mode, so the palette binds one variant per mode.
    Sign {
        /// Blend mode the stroke applies.
        mode: DecoMode,
    },
    /// Text sign tool (`SignTool`): clicking the canvas places a sign and
    /// opens its editor; clicking a drawn sign edits it. Distinct from the
    /// decoration [`Self::Sign`] tool, which only rewrites `dcolour`.
    SignTool,
    /// `DEFAULT_UI_SAMPLE`: pick the element or wall under the cursor into the
    /// primary brush.
    Sample,
    /// Single-particle spawn tool (`sim.partCreate` semantics): one particle of
    /// the selected element per stroke sample, moving along the drag direction.
    Particle,
    /// `GOLTool` (`DEFAULT_UI_ADDLIFE`): paint `LIFE` cells that follow the
    /// given GoL `ctype` — a [`tpt_sim::gol_kernel::BUILTIN_GOL`] index or a
    /// packed `B/S` ruleset (see `GOLString.cpp`).
    Gol {
        /// `LIFE` `ctype`: builtin rule index (`0..NGOL`) or packed ruleset.
        rule: u32,
    },
}

impl Default for SelectedTool {
    fn default() -> Self {
        Self::Element(tpt_core::constants::PT_DUST)
    }
}

impl SelectedTool {
    /// Element id associated with the tool, if any.
    pub fn element_ty(self) -> Option<u32> {
        match self {
            Self::Element(ty) | Self::RigidBody { elem_ty: ty, .. } => Some(ty),
            Self::Gol { .. } => Some(tpt_core::constants::PT_LIFE),
            Self::Wall(_) | Self::SimTool(_) | Self::Eraser => None,
            Self::Property | Self::Sign { .. } | Self::SignTool | Self::Sample | Self::Particle => {
                None
            }
        }
    }

    /// Wall id associated with the tool, if any.
    pub fn wall_id(self) -> Option<u8> {
        match self {
            Self::Wall(id) => Some(id),
            _ => None,
        }
    }

    /// Whether this tool erases matter.
    pub fn is_eraser(self) -> bool {
        matches!(self, Self::Eraser)
    }

    /// Decoration mode this tool blends with, or `None` for every other tool.
    pub fn deco_mode(self) -> Option<DecoMode> {
        match self {
            Self::Sign { mode } => Some(mode),
            _ => None,
        }
    }

    /// Stable preference key, persisted as `Tool.Primary` / `Tool.Secondary`.
    ///
    /// The format mirrors the reference tool identifiers loosely
    /// (`element:17`, `wall:3`, `sim:pos_grav`, `rigid:box:11`, `eraser`, ...)
    /// and is versioned by [`Self::from_key`] rejecting unknown keys.
    pub fn to_key(self) -> String {
        match self {
            Self::Element(ty) => format!("element:{ty}"),
            Self::Wall(id) => format!("wall:{id}"),
            Self::SimTool(kind) => format!("sim:{}", sim_tool_key(kind)),
            Self::RigidBody { shape, elem_ty } => {
                let shape = match shape {
                    RigidSpawnShape::Box => "box",
                    RigidSpawnShape::Circle => "circle",
                };
                format!("rigid:{shape}:{elem_ty}")
            }
            Self::Eraser => String::from("eraser"),
            Self::Property => String::from("property"),
            Self::Sign { mode } => format!("sign:{}", deco_mode_key(mode)),
            Self::SignTool => String::from("sign_tool"),
            Self::Sample => String::from("sample"),
            Self::Particle => String::from("particle"),
            Self::Gol { rule } => format!("gol:{rule}"),
        }
    }

    /// Parses a key produced by [`Self::to_key`]. Unknown or malformed keys
    /// return `None`, so a hand-edited `prefs.json` falls back to the default
    /// tool instead of failing to start.
    pub fn from_key(key: &str) -> Option<Self> {
        let key = key.trim();
        let (head, rest) = match key.split_once(':') {
            Some((head, rest)) => (head, Some(rest)),
            None => (key, None),
        };
        match (head, rest) {
            ("eraser", None) => Some(Self::Eraser),
            ("property", None) => Some(Self::Property),
            // `sign` alone is the key written before the modes existed.
            ("sign", None) => Some(Self::Sign {
                mode: DecoMode::Draw,
            }),
            ("sign", Some(rest)) => deco_mode_from_key(rest).map(|mode| Self::Sign { mode }),
            ("sign_tool", None) => Some(Self::SignTool),
            ("sample", None) => Some(Self::Sample),
            ("particle", None) => Some(Self::Particle),
            ("gol", Some(rest)) => rest.parse().ok().map(|rule: u32| Self::Gol {
                rule: rule & tpt_sim::gol_kernel::GOL_RULE_MASK,
            }),
            ("element", Some(rest)) => rest.parse().ok().map(Self::Element),
            ("wall", Some(rest)) => rest.parse().ok().map(Self::Wall),
            ("sim", Some(rest)) => sim_tool_from_key(rest).map(Self::SimTool),
            ("rigid", Some(rest)) => {
                let (shape, ty) = rest.split_once(':')?;
                let shape = match shape {
                    "box" => RigidSpawnShape::Box,
                    "circle" => RigidSpawnShape::Circle,
                    _ => return None,
                };
                ty.parse()
                    .ok()
                    .map(|elem_ty| Self::RigidBody { shape, elem_ty })
            }
            _ => None,
        }
    }

    /// Human-readable label without needing the element registry.
    pub fn short_label(self) -> String {
        match self {
            Self::Element(ty) => t_fmt("app.tool_element", &[&ty.to_string()]),
            Self::Wall(id) => t_fmt("app.tool_wall", &[&id.to_string()]),
            Self::SimTool(kind) => sim_tool_key(kind).to_uppercase(),
            Self::RigidBody { elem_ty, .. } => {
                t_fmt("app.tool_rigid_body", &[&elem_ty.to_string()])
            }
            Self::Eraser => String::from("Eraser"),
            Self::Property => String::from("PROP"),
            Self::Sign { mode } => t_fmt("app.tool_decoration", &[mode.label()]),
            Self::SignTool => String::from("SIGN"),
            Self::Sample => String::from("SMPL"),
            Self::Particle => t("app.tool_particle").to_string(),
            Self::Gol { rule } => t_fmt(
                "app.tool_gol",
                &[&tpt_sim::gol_kernel::gol_rule_label(rule)],
            ),
        }
    }
}

/// Preference-key fragment of a simulation tool, mirroring the reference
/// tool identifiers (`PGRV`, `NGRV`, `HEAT`, ...).
pub fn sim_tool_key(kind: SimToolKind) -> &'static str {
    match kind {
        SimToolKind::Heat => "heat",
        SimToolKind::Cool => "cool",
        SimToolKind::Air => "air",
        SimToolKind::Vac => "vac",
        SimToolKind::Wind => "wind",
        SimToolKind::Mix => "mix",
        SimToolKind::Cycl => "cycl",
        SimToolKind::AmbHeatPlus => "amb_heat_plus",
        SimToolKind::AmbHeatMinus => "amb_heat_minus",
        SimToolKind::PosGrav => "pgrv",
        SimToolKind::NegGrav => "ngrv",
    }
}

/// Inverse of [`sim_tool_key`].
pub fn sim_tool_from_key(key: &str) -> Option<SimToolKind> {
    SimToolKind::ALL
        .into_iter()
        .find(|kind| sim_tool_key(*kind) == key)
}

/// Preference-key fragment of a decoration mode (`sign:add`, ...), mirroring
/// the reference tool identifiers (`DEFAULT_DECOR_ADD`, ...).
pub fn deco_mode_key(mode: DecoMode) -> &'static str {
    match mode {
        DecoMode::Draw => "set",
        DecoMode::Clear => "clr",
        DecoMode::Add => "add",
        DecoMode::Subtract => "sub",
        DecoMode::Multiply => "mul",
        DecoMode::Divide => "div",
        DecoMode::Smudge => "smdg",
    }
}

/// Inverse of [`deco_mode_key`].
pub fn deco_mode_from_key(key: &str) -> Option<DecoMode> {
    DecoMode::ALL
        .into_iter()
        .find(|mode| deco_mode_key(*mode) == key)
}

/// Mirrors the two bound tools into the preferences and writes `prefs.json`.
///
/// Called whenever a palette button rebinds a mouse button, so the selection
/// survives a restart. The write is a single small file, so it is only issued
/// on an actual rebind (or a property/spawn-setting edit). The bound
/// decoration mode is mirrored into `Decoration.Mode`, the key that seeds the
/// palette and the Options window.
pub fn persist_tool_settings(state: &mut crate::app_state::AppState) {
    state.prefs.primary_tool = state.primary_tool.to_key();
    state.prefs.secondary_tool = state.secondary_tool.to_key();
    sync_gravity_zones(state);
    if let Some(mode) = state
        .primary_tool
        .deco_mode()
        .or_else(|| state.secondary_tool.deco_mode())
    {
        state.prefs.deco_mode = mode.index();
    }
    state.prefs.save();
}

/// `GameModel::SetActiveTool`: the gravity-cell markers are shown exactly while
/// one of the bound tools is the gravity wall.
pub fn sync_gravity_zones(state: &mut crate::app_state::AppState) {
    let grav = tpt_core::constants::WL_GRAV;
    let enabled =
        state.primary_tool.wall_id() == Some(grav) || state.secondary_tool.wall_id() == Some(grav);
    state.engine.bufs.config.gravity_zones_enabled = enabled;
}

/// Packs a decoration colour into the `0xAARRGGBB` word stored in `dcolour`.
///
/// Matches `RGBA::Pack` on the reference `DecorationTool` colour: the alpha
/// byte is a blend factor rather than an opacity the renderer ignores.
pub fn pack_deco_colour([red, green, blue, alpha]: [u8; 4]) -> u32 {
    ((alpha as u32) << 24) | ((red as u32) << 16) | ((green as u32) << 8) | blue as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_survives_a_preference_round_trip() {
        let mut tools = vec![
            SelectedTool::Eraser,
            SelectedTool::Property,
            SelectedTool::SignTool,
            SelectedTool::Sample,
            SelectedTool::Particle,
            SelectedTool::Gol { rule: 0 },
            SelectedTool::Gol {
                rule: tpt_sim::gol_kernel::GOL_DEFAULT_RULE,
            },
            SelectedTool::Element(0),
            SelectedTool::Element(43),
            SelectedTool::Wall(0),
            SelectedTool::Wall(29),
            SelectedTool::RigidBody {
                shape: RigidSpawnShape::Box,
                elem_ty: 15,
            },
            SelectedTool::RigidBody {
                shape: RigidSpawnShape::Circle,
                elem_ty: 25,
            },
        ];
        tools.extend(SimToolKind::ALL.into_iter().map(SelectedTool::SimTool));
        tools.extend(
            DecoMode::ALL
                .into_iter()
                .map(|mode| SelectedTool::Sign { mode }),
        );
        for tool in tools {
            assert_eq!(
                SelectedTool::from_key(&tool.to_key()),
                Some(tool),
                "key {:?} did not round-trip",
                tool.to_key()
            );
        }
    }

    #[test]
    fn unknown_preference_keys_are_rejected() {
        for key in [
            "",
            "element",
            "element:",
            "wall:x",
            "sim:nope",
            "rigid:hex:1",
            "bogus:1",
        ] {
            assert_eq!(SelectedTool::from_key(key), None, "key {key:?}");
        }
    }

    #[test]
    fn sign_keys_carry_the_decoration_mode() {
        // A key written before the modes existed binds the default mode.
        assert_eq!(
            SelectedTool::from_key("sign"),
            Some(SelectedTool::Sign {
                mode: DecoMode::Draw
            })
        );
        assert_eq!(
            SelectedTool::from_key("sign:smdg"),
            Some(SelectedTool::Sign {
                mode: DecoMode::Smudge
            })
        );
        assert_eq!(SelectedTool::from_key("sign:nope"), None);

        for mode in DecoMode::ALL {
            assert_eq!(deco_mode_from_key(deco_mode_key(mode)), Some(mode));
        }
    }

    #[test]
    fn deco_colour_packs_argb() {
        assert_eq!(pack_deco_colour([0x12, 0x34, 0xAB, 0xFF]), 0xFF12_34AB);
        assert_eq!(pack_deco_colour([0, 0, 0, 0]), 0);
        assert_eq!(pack_deco_colour([0xFF, 0xFF, 0xFF, 0x80]), 0x80FF_FFFF);
    }
}
