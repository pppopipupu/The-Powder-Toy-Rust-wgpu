//! Bottom bar: brush controls and the element / wall / tool palette grid.

use egui::{RichText, Ui};
use tpt_core::commands::{BrushShape, RigidSpawnShape};
use tpt_core::constants::{
    PT_DMND, PT_GLAS, PT_ICEI, PT_METL, PT_NONE, PT_STNE, PT_WOOD, WL_ERASE,
};
use tpt_core::elements::MenuCategory;

use crate::app_state::{
    AppState, PropertyKind, SelectedTool, format_value, pack_deco_colour, persist_tool_settings,
    value_from_prefs,
};
use crate::brushes::MAX_BRUSH_RADIUS;
use crate::i18n::{t, t_fmt};

use super::shared::{
    DECO_TOOL_BUTTONS, GUI_TOOL_BUTTONS, SIM_TOOL_BUTTONS, TEXT_DIM, deco_tool_colour,
    element_button, element_description, element_name, element_rgb, menu_visible,
    rigid_shape_label, tool_button, wall_description,
};

/// TPT's `GOLTool` (`ToolGOL.cpp`): label `CUST`, colour `0xFEA900`.
const GOL_TOOL_LABEL: &str = "CUST";
/// `ToolGOL.cpp` button colour.
const GOL_TOOL_COLOUR: u32 = 0x00FE_A900;

/// Materials offered for rigid-body spawns.
pub(super) const RIGID_MATERIALS: [u32; 6] = [PT_WOOD, PT_METL, PT_STNE, PT_DMND, PT_GLAS, PT_ICEI];

pub(super) fn draw_bottom_bar(ui: &mut Ui, state: &mut AppState) {
    // The reference game binds the property tool to `Ctrl+P` (left button) and
    // `Ctrl+Shift+P` (right button) in `GameView::OnKeyPress`.
    handle_tool_shortcuts(ui, state);

    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(t("bottom_bar.brush")).strong());
        for (shape, label) in [
            (BrushShape::Circle, t("bottom_bar.brush_circle")),
            (BrushShape::Square, t("bottom_bar.brush_square")),
            (BrushShape::Triangle, t("bottom_bar.brush_triangle")),
        ] {
            if ui
                .selectable_label(state.brush_shape == shape, label)
                .clicked()
            {
                state.brush_shape = shape;
            }
        }

        ui.separator();
        ui.add(
            egui::Slider::new(&mut state.brush_rx, 0..=MAX_BRUSH_RADIUS)
                .text(t("bottom_bar.radius_x"))
                .max_decimals(0),
        );
        ui.add(
            egui::Slider::new(&mut state.brush_ry, 0..=MAX_BRUSH_RADIUS)
                .text(t("bottom_bar.radius_y"))
                .max_decimals(0),
        );
        if ui
            .button(t("bottom_bar.reset_5"))
            .on_hover_text(t("bottom_bar.reset_5_tip"))
            .clicked()
        {
            state.brush_rx = 5;
            state.brush_ry = 5;
            state.brush_shape = BrushShape::Circle;
        }

        ui.separator();
        ui.checkbox(&mut state.replace_mode, t("bottom_bar.replace_mode"))
            .on_hover_text(t("bottom_bar.replace_mode_tip"));
    });

    ui.separator();
    let height = ui.available_height().max(48.0);
    egui::ScrollArea::vertical()
        .id_salt("tpt_tool_grid")
        .auto_shrink([false, false])
        .max_height(height)
        .show(ui, |ui| draw_tool_grid(ui, state));
}

/// GoL rule the `GOLTool` button binds: the rule already bound to either
/// button when it is a `Gol` tool, else TPT's `CustomGOL.Rule` default.
fn gol_rule(state: &AppState) -> u32 {
    for tool in [state.primary_tool, state.secondary_tool] {
        if let SelectedTool::Gol { rule } = tool {
            return rule;
        }
    }
    tpt_sim::gol_kernel::GOL_DEFAULT_RULE
}

/// Renders the spawn buttons of the active category. Left click binds the
/// primary tool, right click binds the secondary tool.
fn draw_tool_grid(ui: &mut Ui, state: &mut AppState) {
    let mut assignment: Option<(bool, SelectedTool)> = None;
    let primary = state.primary_tool;
    let secondary = state.secondary_tool;

    match state.active_category {
        MenuCategory::Walls => {
            ui.horizontal_wrapped(|ui| {
                let selected = primary == SelectedTool::Eraser || secondary == SelectedTool::Eraser;
                let response = tool_button(
                    ui,
                    "Eraser",
                    0x0030_343C,
                    t("bottom_bar.eraser_desc"),
                    selected,
                );
                if response.clicked() {
                    assignment = Some((false, SelectedTool::Eraser));
                }
                if response.secondary_clicked() {
                    assignment = Some((true, SelectedTool::Eraser));
                }

                for wall in &state.engine.walls {
                    if wall.id == WL_ERASE {
                        continue;
                    }
                    let selected = primary == SelectedTool::Wall(wall.id)
                        || secondary == SelectedTool::Wall(wall.id);
                    let response = tool_button(
                        ui,
                        &wall.name,
                        wall.colour,
                        wall_description(state, wall.id),
                        selected,
                    );
                    if response.clicked() {
                        assignment = Some((false, SelectedTool::Wall(wall.id)));
                    }
                    if response.secondary_clicked() {
                        assignment = Some((true, SelectedTool::Wall(wall.id)));
                    }
                }
            });
        }
        MenuCategory::Tools => {
            ui.horizontal_wrapped(|ui| {
                for (kind, label, colour, description) in SIM_TOOL_BUTTONS {
                    let selected = primary == SelectedTool::SimTool(kind)
                        || secondary == SelectedTool::SimTool(kind);
                    let response = tool_button(ui, label, colour, t(description), selected);
                    if response.clicked() {
                        assignment = Some((false, SelectedTool::SimTool(kind)));
                    }
                    if response.secondary_clicked() {
                        assignment = Some((true, SelectedTool::SimTool(kind)));
                    }
                }
            });

            ui.add_space(4.0);
            ui.label(
                RichText::new(t("bottom_bar.tool_modes"))
                    .small()
                    .strong()
                    .color(TEXT_DIM),
            );
            ui.horizontal_wrapped(|ui| {
                for (tool, label, colour, description) in GUI_TOOL_BUTTONS {
                    let selected = primary == tool || secondary == tool;
                    let response = tool_button(ui, label, colour, t(description), selected);
                    if response.clicked() {
                        assignment = Some((false, tool));
                    }
                    if response.secondary_clicked() {
                        assignment = Some((true, tool));
                    }
                }
            });

            ui.add_space(4.0);
            ui.label(
                RichText::new(t("bottom_bar.decoration_heading"))
                    .small()
                    .strong()
                    .color(TEXT_DIM),
            );
            ui.horizontal_wrapped(|ui| {
                // `GameModel::InitTools` fills `SC_DECO` with one
                // `DecorationTool` per blend mode; the icon previews the
                // decoration colour for `SET`/`CLR`.
                let deco_colour = pack_deco_colour(state.prefs.deco_colour());
                for (mode, label, description) in DECO_TOOL_BUTTONS {
                    let tool = SelectedTool::Sign { mode };
                    let selected = primary == tool || secondary == tool;
                    let colour = deco_tool_colour(mode, deco_colour);
                    let response = tool_button(ui, label, colour, t(description), selected);
                    if response.clicked() {
                        assignment = Some((false, tool));
                    }
                    if response.secondary_clicked() {
                        assignment = Some((true, tool));
                    }
                }
            });

            ui.separator();
            draw_tool_settings(ui, state);
        }
        MenuCategory::RigidBodies => {
            ui.horizontal_wrapped(|ui| {
                for material in RIGID_MATERIALS {
                    let name = element_name(state, material);
                    let colour = element_rgb(state, material);
                    for shape in [RigidSpawnShape::Box, RigidSpawnShape::Circle] {
                        let tool = SelectedTool::RigidBody {
                            shape,
                            elem_ty: material,
                        };
                        let selected = primary == tool || secondary == tool;
                        let label = format!("{} {name}", rigid_shape_label(shape));
                        let response = tool_button(
                            ui,
                            &label,
                            colour,
                            t("bottom_bar.rigid_body_desc"),
                            selected,
                        );
                        if response.clicked() {
                            assignment = Some((false, tool));
                        }
                        if response.secondary_clicked() {
                            assignment = Some((true, tool));
                        }
                    }
                }
            });
        }
        MenuCategory::Life => {
            // `GameModel::InitTools` fills `SC_LIFE` with one `ElementTool` per
            // built-in GoL rule plus the `GOLTool`; the raw `LIFE` element
            // itself is `MenuVisible = 0`.
            ui.horizontal_wrapped(|ui| {
                for (index, rule) in tpt_sim::gol_kernel::BUILTIN_GOL.iter().enumerate() {
                    let tool = SelectedTool::Gol { rule: index as u32 };
                    let selected = primary == tool || secondary == tool;
                    let response =
                        tool_button(ui, rule.name, rule.colour, rule.description, selected);
                    if response.clicked() {
                        assignment = Some((false, tool));
                    }
                    if response.secondary_clicked() {
                        assignment = Some((true, tool));
                    }
                }
            });

            ui.add_space(4.0);
            ui.label(
                RichText::new(t("bottom_bar.custom_rule_heading"))
                    .small()
                    .strong()
                    .color(TEXT_DIM),
            );
            ui.horizontal_wrapped(|ui| {
                let current = gol_rule(state);
                let tool = SelectedTool::Gol { rule: current };
                let selected = primary == tool || secondary == tool;
                let response = tool_button(
                    ui,
                    GOL_TOOL_LABEL,
                    GOL_TOOL_COLOUR,
                    t("bottom_bar.custom_gol_desc"),
                    selected,
                );
                if response.clicked() {
                    assignment = Some((false, tool));
                }
                if response.secondary_clicked() {
                    assignment = Some((true, tool));
                }

                // Stand-in for `GOLWindow`: TPT validates the rule string with
                // `ParseGOLString`, registers the custom GOL and adds a tool
                // that paints `LIFE` of that ruleset.
                let id = egui::Id::unique("tpt_gol_rule_text");
                let mut text = ui
                    .ctx()
                    .data_mut(|data| data.get_temp::<String>(id))
                    .unwrap_or_else(|| {
                        tpt_sim::gol_kernel::serialise_gol_rule(
                            tpt_sim::gol_kernel::resolve_gol_rule(current),
                        )
                    });
                let edit = ui.add(
                    egui::TextEdit::singleline(&mut text)
                        .desired_width(90.0)
                        .hint_text(t("bottom_bar.rule_hint")),
                );
                if edit.changed() {
                    ui.ctx().data_mut(|data| data.insert_temp(id, text.clone()));
                }
                let parsed = tpt_sim::gol_kernel::parse_gol_string(text.trim());
                if ui
                    .add_enabled(
                        parsed.is_some(),
                        egui::Button::new(t("bottom_bar.set_rule")),
                    )
                    .on_hover_text(t("bottom_bar.set_rule_tip"))
                    .clicked()
                    && let Some(rule) = parsed
                {
                    assignment = Some((false, SelectedTool::Gol { rule }));
                    ui.ctx().data_mut(|data| data.remove_temp::<String>(id));
                }
                ui.label(
                    RichText::new(t_fmt(
                        "bottom_bar.current_rule",
                        &[&tpt_sim::gol_kernel::gol_rule_label(current)],
                    ))
                    .small()
                    .color(TEXT_DIM),
                );
            });
        }
        category => {
            ui.horizontal_wrapped(|ui| {
                for definition in &state.engine.registry.elements {
                    if !definition.enabled || definition.category != category {
                        continue;
                    }
                    if !menu_visible(definition, &state.engine.bufs.config) {
                        continue;
                    }
                    let selected = primary == SelectedTool::Element(definition.id)
                        || secondary == SelectedTool::Element(definition.id);
                    let response = element_button(
                        ui,
                        definition,
                        element_description(state, definition.id),
                        selected,
                    );
                    if response.clicked() {
                        assignment = Some((false, SelectedTool::Element(definition.id)));
                    }
                    if response.secondary_clicked() {
                        assignment = Some((true, SelectedTool::Element(definition.id)));
                    }
                }
            });
        }
    }

    if let Some((secondary_button, tool)) = assignment {
        if secondary_button {
            state.secondary_tool = tool;
        } else {
            state.primary_tool = tool;
        }
        // The single-particle tool spawns "the currently selected element", so
        // every element the palette binds becomes that element.
        if let SelectedTool::Element(elem_ty) = tool
            && elem_ty != PT_NONE
        {
            state.prefs.spawn_element = elem_ty;
        }
        persist_tool_settings(state);
        // Binding `PROP` opens its window, like `PropertyTool::Select`.
        if tool == SelectedTool::Property {
            state.open_property_window();
        }
    }
}

/// Applies the reference game's tool shortcuts.
///
/// `Ctrl+P` binds the property tool to the primary button and
/// `Ctrl+Shift+P` to the secondary one, exactly like `GameView::OnKeyPress`
/// (`SetActiveTool(0/1, "DEFAULT_UI_PROPERTY")`). Shortcuts are skipped while a
/// text box has focus so typing a save name never steals the tool.
fn handle_tool_shortcuts(ui: &Ui, state: &mut AppState) {
    if ui.ctx().text_edit_focused() {
        return;
    }
    let (ctrl, shift, pressed) = ui.input(|input| {
        (
            input.modifiers.ctrl || input.modifiers.command,
            input.modifiers.shift,
            input.key_pressed(egui::Key::P),
        )
    });
    if !(ctrl && pressed) {
        return;
    }
    if shift {
        state.secondary_tool = SelectedTool::Property;
    } else {
        state.primary_tool = SelectedTool::Property;
    }
    persist_tool_settings(state);
    state.open_property_window();
}

/// Property / decoration / spawn configuration rows of the tool palette.
///
/// The property tool's configuration lives in the floating property window
/// (`PropertyWindow`'s `Prop.Type` / `Prop.Value` prefs); this row shows the
/// active pair and opens that window. The decoration colour and the spawn
/// element are edited inline, so those tools can be configured without a modal
/// dialog.
fn draw_tool_settings(ui: &mut Ui, state: &mut AppState) {
    // Writes are coalesced: a slider drag touches the preference file once,
    // when the pointer is released, instead of once per frame.
    let mut changed = false;

    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(t("bottom_bar.property")).strong());
        // `PropertyWindow` owns the `Prop.Type` / `Prop.Value` pair now; the
        // button shows the configured property and opens the window (binding
        // the `PROP` tool or pressing Ctrl+P does the same).
        let kind = PropertyKind::from_index(state.prefs.prop_field);
        let value = value_from_prefs(kind, &state.prefs);
        let label = format!(
            "{} = {}",
            kind.label(),
            format_value(kind, value, state.prefs.temperature_scale)
        );
        if ui
            .button(label)
            .on_hover_text(t("bottom_bar.property_tip"))
            .clicked()
        {
            state.open_property_window();
        }

        ui.separator();
        ui.label(RichText::new(t("bottom_bar.decoration")).strong());
        let mut rgb = [
            state.prefs.deco_red,
            state.prefs.deco_green,
            state.prefs.deco_blue,
        ];
        if ui.color_edit_button_srgb(&mut rgb).changed() {
            state.prefs.deco_red = rgb[0];
            state.prefs.deco_green = rgb[1];
            state.prefs.deco_blue = rgb[2];
            changed = true;
        }
        let alpha = ui.add(
            egui::Slider::new(&mut state.prefs.deco_alpha, 0..=255)
                .text(t("bottom_bar.alpha"))
                .max_decimals(0),
        );
        if alpha.changed() {
            changed = true;
        }

        ui.separator();
        ui.label(RichText::new(t("bottom_bar.spawn_element")).strong());
        let spawn_name = element_name(state, state.prefs.spawn_element);
        egui::ComboBox::from_id_salt("tpt_spawn_element")
            .selected_text(spawn_name)
            .show_ui(ui, |ui| {
                for definition in &state.engine.registry.elements {
                    if !definition.enabled || !menu_visible(definition, &state.engine.bufs.config) {
                        continue;
                    }
                    let selected = definition.id == state.prefs.spawn_element;
                    if ui.selectable_label(selected, &definition.name).clicked() {
                        state.prefs.spawn_element = definition.id;
                        changed = true;
                    }
                }
            });
    });

    // A drag (slider, colour picker) keeps the setting dirty until the pointer
    // is released, so the preference file is written once per edit.
    let dirty_id = egui::Id::unique("tpt_tool_settings_dirty");
    let mut pending = ui
        .ctx()
        .data_mut(|data| data.get_temp::<bool>(dirty_id))
        .unwrap_or(false);
    if changed {
        pending = true;
        ui.ctx().data_mut(|data| data.insert_temp(dirty_id, true));
    }
    if pending && !ui.input(|input| input.pointer.any_down()) {
        // `persist_tool_settings` also refreshes the bound-tool keys, so a
        // sampled tool (bound without a palette click) is written too.
        persist_tool_settings(state);
        ui.ctx().data_mut(|data| data.insert_temp(dirty_id, false));
    }
}
