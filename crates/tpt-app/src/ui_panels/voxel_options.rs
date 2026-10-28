//! 3D page of the floating Options window.
//!
//! Kept separate from `options_window` so the 2D settings pages and the 3D
//! renderer settings stay independently readable. Changes are applied by
//! `AppState::apply_prefs`, which pushes them into `Renderer3d` and the voxel
//! engine when either exists.

use egui::Ui;
use tpt_core::particle::EdgeMode;

use crate::app_state::AppState;
use crate::i18n::t;
use crate::prefs::{MAX_VOXEL_AIR_ITERATIONS, MIN_VOXEL_AIR_ITERATIONS, VOXEL_RENDER_SCALES};

use super::widgets::{option_checkbox, option_drag_u32, option_dropdown, option_info};

/// Maps a combo-box discriminant back to the edge enumeration.
fn edge_mode(value: u32) -> EdgeMode {
    match value {
        1 => EdgeMode::Solid,
        2 => EdgeMode::Loop,
        _ => EdgeMode::Void,
    }
}

/// Draws the 3D render/edge/air controls and returns whether anything changed.
pub(super) fn draw_voxel3d_options(ui: &mut Ui, state: &mut AppState) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(t("voxel3d.render_scale"));
        for scale in VOXEL_RENDER_SCALES {
            let label = if (scale - 0.5).abs() < f32::EPSILON {
                t("voxel3d.scale_half")
            } else if (scale - 0.75).abs() < f32::EPSILON {
                t("voxel3d.scale_three_quarters")
            } else {
                t("voxel3d.scale_full")
            };
            let selected = (state.prefs.voxel_render_scale - scale).abs() < 1.0e-4;
            if ui.selectable_label(selected, label).clicked() {
                state.prefs.voxel_render_scale = scale;
                changed = true;
            }
        }
    });
    option_info(ui, t("voxel3d.render_scale_note"));

    changed |= option_dropdown(
        ui,
        "tpt_voxel_ao_rays",
        t("voxel3d.ao_rays"),
        &mut state.prefs.voxel_ao_rays,
        &[
            (0, t("voxel3d.ao_off")),
            (2, t("voxel3d.ao_two")),
            (4, t("voxel3d.ao_four")),
        ],
    );
    option_info(ui, t("voxel3d.ao_rays_note"));

    changed |= option_checkbox(
        ui,
        &mut state.prefs.voxel_shadows,
        t("voxel3d.shadows"),
        "",
        t("voxel3d.shadows_note"),
    );

    let mut edge = state.voxel_edge_mode as u32;
    if option_dropdown(
        ui,
        "tpt_voxel_edge_mode",
        t("voxel3d.edge_mode"),
        &mut edge,
        &[
            (EdgeMode::Void as u32, t("voxel3d.edge_void")),
            (EdgeMode::Solid as u32, t("voxel3d.edge_solid")),
            (EdgeMode::Loop as u32, t("voxel3d.edge_loop")),
        ],
    ) {
        state.set_voxel_edge_mode(edge_mode(edge));
        changed = true;
    }
    option_info(ui, t("voxel3d.edge_mode_tooltip"));

    changed |= option_drag_u32(
        ui,
        t("voxel3d.air_iterations"),
        &mut state.prefs.voxel_air_iterations,
        MIN_VOXEL_AIR_ITERATIONS..=MAX_VOXEL_AIR_ITERATIONS,
    );
    option_info(ui, t("voxel3d.air_iterations_note"));
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_mode_round_trips_through_its_discriminant() {
        for mode in [EdgeMode::Void, EdgeMode::Solid, EdgeMode::Loop] {
            assert_eq!(edge_mode(mode as u32), mode);
        }
    }
}
