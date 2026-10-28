//! 3D palette: the right-hand category column and the bottom element bar.
//!
//! The right sidebar keeps only the category tabs and the bound-tool
//! summary, matching the 2D `side_bar` column. All spawn actions (elements
//! plus the `Tools` actions: Eraser, Heat, Cool, Stone Box, Stone Floor) live
//! in the bottom bar, mirroring the 2D `bottom_bar` tool grid: selecting the
//! `Tools` category shows the tool buttons and the Stone Box settings there.
//!
//! The element table comes straight from `tpt_core::voxel3d::VOXEL_ELEMENTS`,
//! so the panel cannot drift from the simulation table. Draw modes mirror
//! `DrawMode` (freehand, line, rectangle, connected-empty fill) plus the
//! volume-box and temperature tools.

use egui::{RichText, Ui};
use tpt_core::voxel3d::{
    BrushShape3d, VOXEL_CATEGORIES, VOXEL_ELEMENTS, VoxelCategory, VoxelElement,
};

use crate::app_state::{AppState, VoxelDrawMode, VoxelTool};
use crate::i18n::{t, t_fmt};

use super::shared::{TEXT_DIM, tool_button};

/// Localized category label.
fn category_key(category: VoxelCategory) -> &'static str {
    match category {
        VoxelCategory::Electronics => "voxel3d.group_electronics",
        VoxelCategory::Powered => "voxel3d.group_powered",
        VoxelCategory::Sensors => "voxel3d.group_sensors",
        VoxelCategory::Force => "voxel3d.group_force",
        VoxelCategory::Explosives => "voxel3d.group_explosives",
        VoxelCategory::Gases => "voxel3d.group_gases",
        VoxelCategory::Liquids => "voxel3d.group_liquids",
        VoxelCategory::Powders => "voxel3d.group_powders",
        VoxelCategory::Solids => "voxel3d.group_solids",
        VoxelCategory::Radioactive => "voxel3d.group_radioactive",
        VoxelCategory::Special => "voxel3d.group_special",
        VoxelCategory::Life => "voxel3d.group_life",
        VoxelCategory::Tools => "voxel3d.group_tools",
    }
}

/// Display colour of an element button, from its albedo.
fn element_rgb(element: &VoxelElement) -> u32 {
    let channel = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u32;
    (channel(element.albedo[0]) << 16)
        | (channel(element.albedo[1]) << 8)
        | channel(element.albedo[2])
}

/// Table entry for an element id.
fn element_of(ty: u32) -> Option<&'static VoxelElement> {
    VOXEL_ELEMENTS.iter().find(|element| element.id == ty)
}

/// Element display name, falling back to the numeric id.
pub(super) fn voxel_element_name(ty: u32) -> String {
    element_of(ty)
        .map(|element| element.name.to_string())
        .unwrap_or_else(|| t_fmt("voxel3d.element_fallback", &[&ty.to_string()]))
}

/// Display name of the currently bound tool.
fn voxel_tool_name(state: &AppState) -> String {
    // Tool labels are TPT identifiers and stay verbatim in every language.
    match state.voxel_tool {
        VoxelTool::Element(ty) => voxel_element_name(ty),
        VoxelTool::Eraser => String::from("Eraser"),
        VoxelTool::Heat => String::from("Heat"),
        VoxelTool::Cool => String::from("Cool"),
        VoxelTool::StoneBox => String::from("Stone Box"),
    }
}

/// Localized draw-mode label.
fn draw_mode_name(mode: VoxelDrawMode) -> String {
    match mode {
        VoxelDrawMode::Brush => t("voxel3d.draw_brush"),
        VoxelDrawMode::Line => t("voxel3d.draw_line"),
        VoxelDrawMode::Rect => t("voxel3d.draw_rect"),
        VoxelDrawMode::Fill => t("voxel3d.draw_fill"),
    }
    .to_string()
}

/// Right panel: category tabs and the bound-tool summary, matching the 2D
/// `side_bar` column. Spawn buttons live in the bottom bar.
pub(super) fn draw_voxel_palette(ui: &mut Ui, state: &mut AppState) {
    ui.label(
        RichText::new(t("voxel3d.categories"))
            .small()
            .strong()
            .color(TEXT_DIM),
    );
    ui.add_space(4.0);

    for category in VOXEL_CATEGORIES {
        let width = ui.available_width();
        if ui
            .add_sized(
                [width, 24.0],
                egui::Button::selectable(
                    state.voxel_category == category,
                    t(category_key(category)),
                ),
            )
            .clicked()
        {
            state.voxel_category = category;
        }
    }

    ui.separator();
    ui.label(
        RichText::new(t("voxel3d.bound_tools"))
            .small()
            .strong()
            .color(TEXT_DIM),
    );
    let tool = voxel_tool_name(state);
    ui.label(
        RichText::new(t_fmt("voxel3d.bound_tool", &[&tool]))
            .small()
            .color(TEXT_DIM),
    );
    let mode = draw_mode_name(state.voxel_draw_mode);
    ui.label(
        RichText::new(t_fmt("voxel3d.bound_draw", &[&mode]))
            .small()
            .color(TEXT_DIM),
    );
}

/// Bottom panel: brush controls, draw modes and the selected category's
/// spawn grid. The `Tools` category mirrors the 2D bottom bar's tool grid.
pub(super) fn draw_voxel_bottom_bar(ui: &mut Ui, state: &mut AppState) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(t("voxel3d.brush")).strong());
        for (shape, key) in [
            (BrushShape3d::Ellipsoid, "voxel3d.shape_ellipsoid"),
            (BrushShape3d::Cuboid, "voxel3d.shape_cuboid"),
            (BrushShape3d::Prism, "voxel3d.shape_prism"),
        ] {
            if ui
                .selectable_label(state.voxel_brush_shape == shape, t(key))
                .clicked()
            {
                state.voxel_brush_shape = shape;
            }
        }

        ui.separator();
        for (axis, label, lock_label) in [
            (0usize, "voxel3d.axis_x", "voxel3d.lock_x"),
            (1, "voxel3d.axis_y", "voxel3d.lock_y"),
            (2, "voxel3d.axis_z", "voxel3d.lock_z"),
        ] {
            ui.label(t(label));
            ui.add(
                egui::DragValue::new(&mut state.voxel_brush_radii[axis])
                    .range(1..=64)
                    .speed(0.2),
            );
            ui.checkbox(&mut state.voxel_brush_locks[axis], t(lock_label));
        }

        ui.separator();
        ui.label(
            RichText::new(t("voxel3d.draw_mode"))
                .small()
                .color(TEXT_DIM),
        );
        for mode in [
            VoxelDrawMode::Brush,
            VoxelDrawMode::Line,
            VoxelDrawMode::Rect,
            VoxelDrawMode::Fill,
        ] {
            if ui
                .selectable_label(state.voxel_draw_mode == mode, draw_mode_name(mode))
                .clicked()
            {
                state.voxel_draw_mode = mode;
            }
        }
    });

    ui.separator();
    let height = ui.available_height().max(48.0);
    egui::ScrollArea::vertical()
        .id_salt("tpt_voxel_element_grid")
        .auto_shrink([false, false])
        .max_height(height)
        .show(ui, |ui| {
            if state.voxel_category == VoxelCategory::Tools {
                draw_tool_grid(ui, state);
                ui.add_space(4.0);
                draw_box_settings(ui, state);
            } else {
                draw_element_grid(ui, state);
            }
        });
}

/// Element buttons of the active category.
fn draw_element_grid(ui: &mut Ui, state: &mut AppState) {
    let elements: Vec<&'static VoxelElement> = VOXEL_ELEMENTS
        .iter()
        .filter(|element| element.category == state.voxel_category)
        .collect();
    if elements.is_empty() {
        ui.label(
            RichText::new(t("voxel3d.category_empty"))
                .small()
                .color(TEXT_DIM),
        );
        return;
    }
    ui.horizontal_wrapped(|ui| {
        for element in elements {
            let selected = state.voxel_tool == VoxelTool::Element(element.id);
            let name = element.name;
            if tool_button(
                ui,
                name,
                element_rgb(element),
                &t_fmt("voxel3d.element_tooltip", &[name]),
                selected,
            )
            .clicked()
            {
                state.voxel_tool = VoxelTool::Element(element.id);
            }
        }
    });
}

/// Tool buttons of the `Tools` category, in the bottom bar.
///
/// Labels stay verbatim TPT names; tooltips stay localized. This mirrors the
/// 2D `bottom_bar` tool grid's wrapped button row.
fn draw_tool_grid(ui: &mut Ui, state: &mut AppState) {
    let entries = [
        (
            "Eraser",
            0x0030_343C,
            t("voxel3d.eraser_tooltip"),
            Some(VoxelTool::Eraser),
        ),
        (
            "Heat",
            0x00E0_6030,
            t("voxel3d.heat_tooltip"),
            Some(VoxelTool::Heat),
        ),
        (
            "Cool",
            0x0030_70E0,
            t("voxel3d.cool_tooltip"),
            Some(VoxelTool::Cool),
        ),
        (
            "Stone Box",
            0x0080_8890,
            t("voxel3d.stone_box_tooltip"),
            Some(VoxelTool::StoneBox),
        ),
        (
            "Stone Floor",
            0x0060_6870,
            t("voxel3d.stone_floor_tooltip"),
            None,
        ),
    ];
    ui.horizontal_wrapped(|ui| {
        for (label, colour, tooltip, tool) in entries {
            let selected = matches!(tool, Some(bound) if state.voxel_tool == bound);
            if tool_button(ui, label, colour, tooltip, selected).clicked() {
                match tool {
                    Some(bound) => state.voxel_tool = bound,
                    None => state.paint_voxel_floor(),
                }
            }
        }
    });
}

/// Stone Box material and half-extent controls, shown under the bottom bar's
/// `Tools` category. Collapsed by default so the tool grid stays compact.
fn draw_box_settings(ui: &mut Ui, state: &mut AppState) {
    egui::CollapsingHeader::new(t("voxel3d.box_settings"))
        .default_open(false)
        .show(ui, |ui| {
            ui.label(RichText::new(t("voxel3d.box_size")).small().color(TEXT_DIM));
            ui.horizontal_wrapped(|ui| {
                for (axis, label) in [
                    (0usize, "voxel3d.axis_x"),
                    (1, "voxel3d.axis_y"),
                    (2, "voxel3d.axis_z"),
                ] {
                    ui.label(t(label));
                    ui.add(
                        egui::DragValue::new(&mut state.voxel_box_half[axis])
                            .range(1..=16)
                            .speed(0.2),
                    );
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label(t("voxel3d.box_element"));
                let mut element = state.voxel_box_element;
                egui::ComboBox::from_id_salt("tpt_voxel_box_element")
                    .selected_text(voxel_element_name(element))
                    .width(96.0)
                    .show_ui(ui, |ui| {
                        for candidate in VOXEL_ELEMENTS {
                            ui.selectable_value(&mut element, candidate.id, candidate.name);
                        }
                    });
                state.voxel_box_element = element;
            });
        });
}
