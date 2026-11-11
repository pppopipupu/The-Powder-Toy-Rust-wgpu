//! Palette, colour helpers, element/tool naming and shared widgets.

use egui::{Color32, CornerRadius, Frame, Key, Margin, RichText, Stroke, Ui, vec2};
use tpt_core::commands::{DecoMode, RigidSpawnShape, SimToolKind};
use tpt_core::constants::{
    COLOUR_DEFAULT, COLOUR_DEST, COLOUR_HEAT, COLOUR_LIFE, DISPLAY_AIRH, DISPLAY_AIRP,
    DISPLAY_AIRV, DISPLAY_AIRW, DISPLAY_EFFE, MAX_TEMP, MIN_TEMP, RENDER_BASC, RENDER_EFFE,
};
use tpt_core::elements::ElementDef;
use tpt_core::particle::SimulationConfig;

use crate::app_state::{AppState, CanvasSizePreset, SelectedTool};
use crate::i18n::{t, t_fmt, try_t};

/// Background of the HUD panels.
pub(super) const BG_PANEL: Color32 = Color32::from_rgb(24, 26, 32);
/// Background around the simulation image.
pub(super) const BG_VIEWPORT: Color32 = Color32::from_rgb(9, 10, 12);
/// Highlight colour for titles and the cursor readout.
pub(super) const ACCENT: Color32 = Color32::from_rgb(255, 176, 64);
/// Muted label colour.
pub(super) const TEXT_DIM: Color32 = Color32::from_rgb(156, 162, 174);
/// Background of the main-menu cards.
pub(super) const BG_CARD: Color32 = Color32::from_rgb(31, 34, 41);
/// Subtle outline used by panels and cards.
pub(super) const STROKE_SOFT: Color32 = Color32::from_rgb(58, 63, 74);

/// Elements TPT marks `MenuVisible = 0`, by `SimulationData` identifier.
///
/// The reference game keeps these out of every menu section (`GameModel::
/// InitTools` never allocates a tool for them) while leaving them fully usable
/// from Lua and from saves. The palette filter reproduces that exactly:
///
/// `DYST MORT LIFE LOVE FRZW BIZG BIZS PSTS EQVE SPWN2 SPWN SHD2 SHD3 SHD4
/// LOLZ BRAY EMBR VRSS VRSG RFGL`
///
/// `LIFE` is hidden because `SC_LIFE` lists the GoL *rules* instead of the raw
/// element (see [`crate::ui_panels::bottom_bar`]).
pub(super) const MENU_HIDDEN: [&str; 20] = [
    "DEFAULT_PT_DYST",
    "DEFAULT_PT_MORT",
    "DEFAULT_PT_LIFE",
    "DEFAULT_PT_LOVE",
    "DEFAULT_PT_FRZW",
    "DEFAULT_PT_BIZRG",
    "DEFAULT_PT_BIZRS",
    "DEFAULT_PT_PSTS",
    "DEFAULT_PT_116",
    "DEFAULT_PT_SPAWN2",
    "DEFAULT_PT_SPAWN",
    "DEFAULT_PT_SHLD2",
    "DEFAULT_PT_SHLD3",
    "DEFAULT_PT_SHLD4",
    "DEFAULT_PT_LOLZ",
    "DEFAULT_PT_BRAY",
    "DEFAULT_PT_EMBR",
    "DEFAULT_PT_VRSS",
    "DEFAULT_PT_VRSG",
    "DEFAULT_PT_RFGL",
];

/// `false` for the elements TPT hides from the palette (`MenuVisible = 0`).
///
/// Hidden elements stay placeable through `SimCommand`s and Lua; only the
/// palette and the spawn-element dropdown skip them.
pub(super) fn menu_visible(definition: &ElementDef) -> bool {
    !MENU_HIDDEN.contains(&definition.identifier.as_str())
}

/// Non-particle simulation tools shown by the `Tools` category.
///
/// The labels are the tool names the port already displayed in English; tool
/// names are never localized (`AGENTS.md`), so they are literals instead of
/// locale keys. The reference identifiers are `HEAT`, `COOL`, `AIR`, `VAC`,
/// `WIND`, `MIX`, `CYCL`, `AMBP`, `AMBM`, `PGRV` and `NGRV`.
pub(super) const SIM_TOOL_BUTTONS: [(SimToolKind, &str, u32, &str); 11] = [
    (SimToolKind::Heat, "Heat", 0x00D9_4A2B, "tools.heat_desc"),
    (SimToolKind::Cool, "Cool", 0x003F_7CD6, "tools.cool_desc"),
    (SimToolKind::Air, "Air", 0x008A_9BA8, "tools.air_desc"),
    (SimToolKind::Vac, "Vacuum", 0x0044_4A52, "tools.vacuum_desc"),
    (SimToolKind::Wind, "Wind", 0x006F_B89A, "tools.wind_desc"),
    (SimToolKind::Mix, "Mix", 0x00B0_8AD0, "tools.mix_desc"),
    (
        SimToolKind::Cycl,
        "Cyclone",
        0x007A_6AC0,
        "tools.cyclone_desc",
    ),
    (
        SimToolKind::AmbHeatPlus,
        "Amb Heat +",
        0x00E0_8C3A,
        "tools.amb_heat_plus_desc",
    ),
    (
        SimToolKind::AmbHeatMinus,
        "Amb Heat -",
        0x003A_8CE0,
        "tools.amb_heat_minus_desc",
    ),
    (SimToolKind::PosGrav, "PGRV", 0x00CC_CCFF, "tools.pgrv_desc"),
    (SimToolKind::NegGrav, "NGRV", 0x00AA_CCFF, "tools.ngrv_desc"),
];

/// GUI tool classes (`Tool` subclasses in the reference game) shown next to
/// the simulation tools.
///
/// The labels are the reference tool identifiers (`DEFAULT_UI_PROPERTY`,
/// `DEFAULT_UI_SAMPLE`, ...), which stay verbatim in every language. The
/// decoration tools live in their own [`DECO_TOOL_BUTTONS`] row, exactly like
/// the reference `SC_DECO` menu section.
pub(super) const GUI_TOOL_BUTTONS: [(SelectedTool, &str, u32, &str); 4] = [
    (
        SelectedTool::Property,
        "PROP",
        0x00FE_A900,
        "tools.prop_desc",
    ),
    (SelectedTool::Sample, "SMPL", 0x0040_6080, "tools.smpl_desc"),
    (
        SelectedTool::Particle,
        "PART",
        0x00E0_C040,
        "tools.part_desc",
    ),
    (
        SelectedTool::SignTool,
        "SIGN",
        0x0020_4080,
        "tools.sign_desc",
    ),
];

/// Reference decoration tools (`SC_DECO`), one per blend mode.
///
/// The labels are `GameModel::InitTools`' tool names (`DEFAULT_DECOR_*`); the
/// port has no sign pixels to tint, so [`deco_tool_colour`] paints the button
/// the way `DecorationTool::GetIcon` would.
pub(super) const DECO_TOOL_BUTTONS: [(DecoMode, &str, &str); 7] = [
    (DecoMode::Draw, "SET", "tools.deco_set_desc"),
    (DecoMode::Clear, "CLR", "tools.deco_clear_desc"),
    (DecoMode::Add, "ADD", "tools.deco_add_desc"),
    (DecoMode::Subtract, "SUB", "tools.deco_sub_desc"),
    (DecoMode::Multiply, "MUL", "tools.deco_mul_desc"),
    (DecoMode::Divide, "DIV", "tools.deco_div_desc"),
    (DecoMode::Smudge, "SMDG", "tools.deco_smdg_desc"),
];

/// Palette colour of a decoration tool button.
///
/// `DRAW` and `CLEAR` preview the current decoration colour, the blending
/// modes use `DecorationTool::GetIcon`'s flat `0x323232` background.
pub(super) fn deco_tool_colour(mode: DecoMode, deco_colour: u32) -> u32 {
    match mode {
        DecoMode::Draw | DecoMode::Clear => deco_colour & 0x00FF_FFFF,
        _ => 0x0032_3232,
    }
}

/// The five display modes bound to the `1`–`5` hotkeys.
///
/// The toolbar dropdown was removed when the HUD settings moved into the
/// Options window, so only the hotkey application survives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DisplayPreset {
    Normal,
    FireGlow,
    Thermal,
    Pressure,
    Velocity,
}

impl DisplayPreset {
    /// Applies the preset to the simulation config.
    pub(super) fn apply(self, config: &mut SimulationConfig) {
        config.display_mode = match self {
            Self::Normal => 0,
            Self::FireGlow => DISPLAY_EFFE,
            Self::Thermal => DISPLAY_AIRH,
            Self::Pressure => DISPLAY_AIRP,
            Self::Velocity => DISPLAY_AIRV,
        };
        config.colour_mode = if self == Self::Thermal {
            COLOUR_HEAT
        } else {
            COLOUR_DEFAULT
        };
    }
}

/// The four extra render presets bound to `Shift+1/6/0/7`.
///
/// The reference indexes these as 10..13 in `Renderer::renderModePresets`
/// (`Renderer.cpp:1419-1437`); `GameView::OnKeyPress` (`GameView.cpp:1655-1669`)
/// maps the shifted digits to those indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RenderPreset {
    /// Preset 10: `Life Gradient Display` (base render, life colouring).
    LifeGradient,
    /// Preset 11: `Dynamic Heat Display` (auto heat-display limits).
    DynamicHeat,
    /// Preset 12: `Vorticity Display`.
    Vorticity,
    /// Preset 13: `Destructibility Display`.
    Destructibility,
}

impl RenderPreset {
    /// Preset selected by a shifted digit, matching the reference bindings.
    pub(super) fn for_shift_digit(key: Key) -> Option<Self> {
        match key {
            Key::Num1 => Some(Self::LifeGradient),
            Key::Num6 => Some(Self::DynamicHeat),
            Key::Num0 => Some(Self::Vorticity),
            Key::Num7 => Some(Self::Destructibility),
            _ => None,
        }
    }

    /// Index into `Renderer::renderModePresets`.
    #[cfg(test)]
    pub(super) fn index(self) -> usize {
        match self {
            Self::LifeGradient => 10,
            Self::DynamicHeat => 11,
            Self::Vorticity => 12,
            Self::Destructibility => 13,
        }
    }

    /// Localized preset name used by the HUD status tip
    /// (`GameController::LoadRenderPreset`'s `SetInfoTip(preset.Name)`).
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::LifeGradient => t("app.preset_life_gradient"),
            Self::DynamicHeat => t("app.preset_dynamic_heat"),
            Self::Vorticity => t("app.preset_vorticity"),
            Self::Destructibility => t("app.preset_destructibility"),
        }
    }

    /// Applies the preset's render / display / colour modes and heat-display
    /// limits (`RenderModel::LoadRenderPreset`, `RenderModel.cpp:50-58`).
    pub(super) fn apply(self, config: &mut SimulationConfig) {
        let (render_mode, display_mode, colour_mode, auto_limits) = match self {
            Self::LifeGradient => (RENDER_BASC, 0, COLOUR_LIFE, false),
            Self::DynamicHeat => (RENDER_BASC, DISPLAY_AIRH, COLOUR_HEAT, true),
            Self::Vorticity => (
                RENDER_EFFE | RENDER_BASC,
                DISPLAY_AIRW,
                COLOUR_DEFAULT,
                false,
            ),
            Self::Destructibility => (RENDER_BASC, 0, COLOUR_DEST, false),
        };
        config.render_mode = render_mode;
        config.display_mode = display_mode;
        config.colour_mode = colour_mode;
        // `HdispLimitAuto` only for the Dynamic Heat entry; the other presets
        // reset the explicit `MIN_TEMP`/`MAX_TEMP` defaults of
        // `RenderPreset` (`RenderPreset.h:6-15`).
        config.heat_display_limit_min = if auto_limits { None } else { Some(MIN_TEMP) };
        config.heat_display_limit_max = if auto_limits { None } else { Some(MAX_TEMP) };
    }
}

/// Draws a titled card and returns whatever the content closure returns.
pub(super) fn card<R>(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::new()
        .fill(BG_CARD)
        .inner_margin(Margin::symmetric(14, 12))
        .corner_radius(CornerRadius::same(10))
        .stroke(Stroke::new(1.0, STROKE_SOFT))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.label(
                RichText::new(title)
                    .size(18.0)
                    .strong()
                    .color(Color32::from_rgb(230, 233, 240)),
            );
            ui.add_space(2.0);
            ui.separator();
            ui.add_space(4.0);
            add(ui)
        })
        .inner
}

/// Label of a preset button, matching the wording used in the design docs.
pub(super) fn preset_button_label(preset: CanvasSizePreset) -> &'static str {
    match preset {
        CanvasSizePreset::Classic612x384 => t("tools.preset_612"),
        CanvasSizePreset::Medium800x600 => t("tools.preset_800"),
        CanvasSizePreset::Large1024x768 => t("tools.preset_1024"),
        CanvasSizePreset::HD1280x720 => t("tools.preset_1280"),
        CanvasSizePreset::Huge1600x900 => t("tools.preset_1600"),
        CanvasSizePreset::Custom => t("tools.preset_custom"),
    }
}

/// Formats a count with thin groups of three digits.
pub(super) fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(digit);
    }
    out
}

pub(super) fn element_color(colour: u32) -> Color32 {
    Color32::from_rgb(
        ((colour >> 16) & 0xFF) as u8,
        ((colour >> 8) & 0xFF) as u8,
        (colour & 0xFF) as u8,
    )
}

pub(super) fn contrasting_text(background: Color32) -> Color32 {
    let luma = 0.299 * background.r() as f32
        + 0.587 * background.g() as f32
        + 0.114 * background.b() as f32;
    if luma > 140.0 {
        Color32::from_rgb(16, 16, 16)
    } else {
        Color32::from_rgb(246, 246, 246)
    }
}

pub(super) fn element_def(state: &AppState, ty: u32) -> Option<&ElementDef> {
    if (ty as usize) < state.engine.registry.elements.len() {
        Some(state.engine.registry.get(ty))
    } else {
        None
    }
}

/// Display name of an element id.
pub(super) fn element_name(state: &AppState, ty: u32) -> String {
    element_def(state, ty)
        .map(|definition| definition.name.clone())
        .unwrap_or_else(|| t_fmt("tools.tool_element_fallback", &[&ty.to_string()]))
}

/// Localized description of an element id.
///
/// TPT shows `Element::Description` as the palette tooltip
/// (`src/simulation/elements/*.cpp`; `GameModel.cpp:1851-1852` copies it onto
/// the `Tool` and `GameView.cpp:1304-1331` renders it). Built-in descriptions
/// live in the `elements` locale module; a Lua-defined element has no key and
/// keeps its scripted `ElementDef::description`.
pub(super) fn element_description(state: &AppState, ty: u32) -> &str {
    let definition = state.engine.registry.get(ty);
    let key = format!("elements.{}", definition.identifier);
    try_t(&key).unwrap_or(definition.description.as_str())
}

/// Base colour of an element id as `0x00RRGGBB`.
pub(super) fn element_rgb(state: &AppState, ty: u32) -> u32 {
    element_def(state, ty)
        .map(|definition| definition.colour)
        .unwrap_or(0x0080_8080)
}

/// Display name of a wall id.
pub(super) fn wall_name(state: &AppState, id: u8) -> &str {
    state
        .engine
        .walls
        .iter()
        .find(|wall| wall.id == id)
        .map(|wall| wall.name.as_str())
        .unwrap_or(t("tools.tool_wall_fallback"))
}

/// Localized description of a wall id, matching [`element_description`].
///
/// Wall descriptions come from `default_walls()` and are keyed by the
/// `DEFAULT_WL_*` identifier in the `walls` locale module.
pub(super) fn wall_description(state: &AppState, id: u8) -> &str {
    match state.engine.walls.iter().find(|wall| wall.id == id) {
        Some(wall) => {
            let key = format!("walls.{}", wall.identifier);
            try_t(&key).unwrap_or(wall.description.as_str())
        }
        None => t("tools.tool_wall_fallback"),
    }
}

/// Air pressure of the cell containing pixel `(x, y)`.
pub(super) fn cell_pressure(state: &AppState, x: i32, y: i32) -> f32 {
    // Served by the inspector's cached probe; the raw device read happens at
    // most every 150 ms instead of once per frame.
    state.cell_pressure_at(x, y)
}

/// Id of the rigid body that owns the pixel at `(x, y)`, if any.
pub(super) fn rigid_id_at(state: &AppState, x: i32, y: i32) -> Option<u32> {
    state.engine.bufs.rigid_bodies.iter().find_map(|body| {
        if !body.active {
            return None;
        }
        body.pixels
            .iter()
            .any(|pixel| pixel.grid_x == x && pixel.grid_y == y)
            .then_some(body.id)
    })
}

/// Human-readable label of a bound tool.
pub(super) fn tool_label(state: &AppState, tool: SelectedTool) -> String {
    match tool {
        SelectedTool::Element(ty) => element_name(state, ty),
        SelectedTool::Wall(id) => t_fmt("tools.tool_wall", &[wall_name(state, id)]),
        SelectedTool::SimTool(kind) => sim_tool_label(kind).to_string(),
        SelectedTool::RigidBody { shape, elem_ty } => {
            format!(
                "{} {}",
                rigid_shape_label(shape),
                element_name(state, elem_ty)
            )
        }
        SelectedTool::Eraser => String::from("Eraser"),
        SelectedTool::Property => String::from("PROP"),
        SelectedTool::Sign { mode } => t_fmt("tools.tool_decoration", &[mode.label()]),
        SelectedTool::SignTool => String::from("SIGN"),
        SelectedTool::Sample => String::from("SMPL"),
        SelectedTool::Particle => t_fmt(
            "tools.tool_particle",
            &[&element_name(state, state.prefs.spawn_element)],
        ),
        SelectedTool::Gol { rule } => t_fmt(
            "tools.tool_gol",
            &[&tpt_sim::gol_kernel::gol_rule_label(rule)],
        ),
    }
}

/// Human-readable name of a simulation tool (never localized).
pub(super) fn sim_tool_label(kind: SimToolKind) -> &'static str {
    SIM_TOOL_BUTTONS
        .iter()
        .find(|(candidate, ..)| *candidate == kind)
        .map(|(_, label, ..)| *label)
        .unwrap_or("Sim Tool")
}

/// Human-readable name of a rigid-body shape.
pub(super) fn rigid_shape_label(shape: RigidSpawnShape) -> &'static str {
    match shape {
        RigidSpawnShape::Box => "Box",
        RigidSpawnShape::Circle => "Circle",
    }
}

/// Element spawn button painted with the element's authentic colour.
pub(super) fn element_button(
    ui: &mut Ui,
    definition: &ElementDef,
    description: &str,
    selected: bool,
) -> egui::Response {
    let background = element_color(definition.colour);
    let text = RichText::new(&definition.name)
        .color(contrasting_text(background))
        .strong();
    let mut button = egui::Button::new(text)
        .fill(background)
        .min_size(vec2(76.0, 24.0))
        .corner_radius(CornerRadius::same(4));
    if selected {
        button = button.stroke(Stroke::new(2.0, Color32::WHITE));
    }
    ui.add(button).on_hover_ui(|ui| {
        ui.label(RichText::new(&definition.name).strong());
        ui.label(description);
    })
}

/// Wall / sim-tool / rigid-body button with a solid background colour.
pub(super) fn tool_button(
    ui: &mut Ui,
    label: &str,
    colour: u32,
    tooltip: &str,
    selected: bool,
) -> egui::Response {
    let background = element_color(colour);
    let text = RichText::new(label)
        .color(contrasting_text(background))
        .strong();
    let mut button = egui::Button::new(text)
        .fill(background)
        .min_size(vec2(84.0, 24.0))
        .corner_radius(CornerRadius::same(4));
    if selected {
        button = button.stroke(Stroke::new(2.0, Color32::WHITE));
    }
    ui.add(button).on_hover_text(tooltip)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_core::particle::SimulationConfig;

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1 000");
        assert_eq!(thousands(235_008), "235 008");
        assert_eq!(thousands(1_234_567), "1 234 567");
    }

    #[test]
    fn contrast_picks_black_on_light_and_white_on_dark() {
        assert_eq!(
            contrasting_text(Color32::WHITE),
            Color32::from_rgb(16, 16, 16)
        );
        assert_eq!(
            contrasting_text(Color32::BLACK),
            Color32::from_rgb(246, 246, 246)
        );
    }

    #[test]
    fn display_presets_apply_to_config() {
        let mut config = SimulationConfig::new(128, 128);
        DisplayPreset::Thermal.apply(&mut config);
        assert_eq!(config.display_mode, DISPLAY_AIRH);
        assert_eq!(config.colour_mode, COLOUR_HEAT);
        DisplayPreset::Velocity.apply(&mut config);
        assert_eq!(config.display_mode, DISPLAY_AIRV);
        assert_eq!(config.colour_mode, COLOUR_DEFAULT);
        DisplayPreset::Normal.apply(&mut config);
        assert_eq!(config.display_mode, 0);
    }

    /// `GameView.cpp:1655-1669`: `Shift+1`, `Shift+6`, `Shift+0` and `Shift+7`
    /// load reference presets 10, 11, 12 and 13. The mapping is a literal
    /// index table, so it must not drift with the keyboard enum order.
    #[test]
    fn shift_digit_presets_match_reference_indices() {
        let mapping = [
            (Key::Num1, 10, RenderPreset::LifeGradient),
            (Key::Num6, 11, RenderPreset::DynamicHeat),
            (Key::Num0, 12, RenderPreset::Vorticity),
            (Key::Num7, 13, RenderPreset::Destructibility),
        ];
        for (key, index, preset) in mapping {
            assert_eq!(RenderPreset::for_shift_digit(key), Some(preset));
            assert_eq!(preset.index(), index);
        }
        for key in [
            Key::Num2,
            Key::Num3,
            Key::Num4,
            Key::Num5,
            Key::Num8,
            Key::Num9,
        ] {
            assert_eq!(RenderPreset::for_shift_digit(key), None);
        }
    }

    /// The four presets write the reference `renderModePresets` fields,
    /// including the Dynamic Heat auto limits (`Renderer.cpp:1419-1437`).
    #[test]
    fn render_presets_apply_reference_modes() {
        let mut config = SimulationConfig::new(128, 128);

        RenderPreset::LifeGradient.apply(&mut config);
        assert_eq!(config.render_mode, RENDER_BASC);
        assert_eq!(config.display_mode, 0);
        assert_eq!(config.colour_mode, COLOUR_LIFE);
        assert_eq!(config.heat_display_limit_min, Some(MIN_TEMP));

        RenderPreset::DynamicHeat.apply(&mut config);
        assert_eq!(config.display_mode, DISPLAY_AIRH);
        assert_eq!(config.colour_mode, COLOUR_HEAT);
        assert_eq!(config.heat_display_limit_min, None);
        assert_eq!(config.heat_display_limit_max, None);

        RenderPreset::Vorticity.apply(&mut config);
        assert_eq!(config.render_mode, RENDER_EFFE | RENDER_BASC);
        assert_eq!(config.display_mode, DISPLAY_AIRW);
        assert_eq!(config.colour_mode, COLOUR_DEFAULT);
        assert_eq!(config.heat_display_limit_max, Some(MAX_TEMP));

        RenderPreset::Destructibility.apply(&mut config);
        assert_eq!(config.render_mode, RENDER_BASC);
        assert_eq!(config.display_mode, 0);
        assert_eq!(config.colour_mode, COLOUR_DEST);
    }

    #[test]
    fn hidden_menu_list_matches_tpt_menu_visible_zero() {
        use tpt_core::elements::ElementRegistry;

        let registry = ElementRegistry::new_default();
        let mut names = Vec::with_capacity(MENU_HIDDEN.len());
        for identifier in MENU_HIDDEN {
            let definition = registry
                .elements
                .iter()
                .find(|definition| definition.identifier == identifier)
                .unwrap_or_else(|| panic!("{identifier} is not a registered element"));
            assert!(!menu_visible(definition), "{identifier} must be hidden");
            names.push(definition.name.as_str());
        }
        // `MenuVisible = 0` in the reference element table, in `PT_*` order.
        assert_eq!(
            names,
            [
                "DYST", "MORT", "LIFE", "LOVE", "FRZW", "BIZG", "BIZS", "PSTS", "EQVE", "SPWN2",
                "SPWN", "SHD2", "SHD3", "SHD4", "LOLZ", "BRAY", "EMBR", "VRSS", "VRSG", "RFGL",
            ]
        );
        // Everything else stays visible.
        assert!(menu_visible(registry.get(tpt_core::constants::PT_DUST)));
        assert!(menu_visible(registry.get(tpt_core::constants::PT_WATR)));
    }

    #[test]
    fn element_color_splits_rgb() {
        assert_eq!(
            element_color(0x0012_34AB),
            Color32::from_rgb(0x12, 0x34, 0xAB)
        );
    }
}
