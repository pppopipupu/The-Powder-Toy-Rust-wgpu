//! Floating Options window (TPT's `OptionsView.cpp` layout and wording).
//!
//! This window is the single home for simulation, graphics and application
//! settings; the HUD only keeps transport controls and the brush palette, and
//! render/display/colour modes live in the Render window (`RenderView.cpp`).
//!
//! The controls are grouped into five pages, selected by a tab strip:
//!
//! * **Graphics** — the simulation/rendering framerate caps, the
//!   restart-scoped window settings, the viewport scale factor and the live
//!   blurry-scaling switch.
//! * **3D Sandbox** — render scale, ambient-occlusion rays, shadows and air
//!   solve iterations of the dedicated 3D renderer.
//! * **Simulation** — heat, ambient heat, Newtonian gravity, water
//!   equalisation, the Eulerian air model, edge behaviour and the gravity
//!   model (including the custom direction vector).
//! * **Advanced** — tool, decoration, settings-folder and undo-history
//!   settings.
//! * **Key bindings** — the pointer and keyboard shortcuts the port binds,
//!   as a read-only action/binding table.
//!
//! The labels and hints follow the C++ view's strings.

use std::path::{Path, PathBuf};

use egui::{Color32, CornerRadius, RichText, Sense, Ui, vec2};
use tpt_core::commands::DecoMode;
use tpt_core::constants::{
    AIRC_BOUSSINESQ, AIRC_LEGACY, AIRC_NONE, MAX_PRESSURE, MAX_TEMP, MIN_PRESSURE, MIN_TEMP,
    TEMPSCALE_CELSIUS, TEMPSCALE_FAHRENHEIT, TEMPSCALE_KELVIN,
};

use crate::app_state::{
    AppState, SAVES_DIR, SelectedTool, temperature_from_scale, temperature_suffix,
    temperature_to_scale,
};
use crate::i18n::{Language, t, t_fmt};
use crate::prefs::{DEFAULT_DRAW_FPS, FpsLimit, MAX_UNDO_HISTORY_LIMIT};

use super::shared::TEXT_DIM;
use super::widgets::{
    draw_fps_control, fps_limit_control, option_checkbox, option_drag_f32, option_drag_u32,
    option_dropdown, option_info,
};

/// Largest absolute ambient air velocity accepted by the options window.
const MAX_EDGE_VELOCITY: f32 = 1.0e4;

/// Window scale factors offered by the Graphics page (`OptionsView.cpp:343-359`
/// builds the list from the desktop size; the port offers the full range the
/// viewport can use).
const WINDOW_SCALE_OPTIONS: [(u32, &str); 8] = [
    (1, "1"),
    (2, "2"),
    (3, "3"),
    (4, "4"),
    (5, "5"),
    (6, "6"),
    (7, "7"),
    (8, "8"),
];

/// egui memory key holding the result of the last data-directory migration.
fn migration_status_key() -> egui::Id {
    egui::Id::unique("tpt_migration_status")
}

/// Platform data directory shared between installs
/// (`Platform::DefaultDdir`, `DdirCommon.cpp`).
///
/// TPT stores its shared data below the platform's per-application directory;
/// the port uses the same location so the migration button lands where the
/// reference build would.
fn shared_data_dir() -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(|dir| PathBuf::from(dir).join("The Powder Toy"))
    } else if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|dir| {
            PathBuf::from(dir)
                .join("Library")
                .join("Application Support")
                .join("The Powder Toy")
        })
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|dir| PathBuf::from(dir).join(".local/share"))
            })
            .map(|dir| dir.join("the-powder-toy"))
    }
}

/// Result of copying one source tree into the shared directory.
#[derive(Default)]
struct MigrationReport {
    copied: u64,
    skipped: u64,
    failed: u64,
}

/// Copies `src` to `dst`, leaving any file that already exists untouched.
///
/// Directories are descended recursively, so `saves/` keeps its subfolders.
/// Failures are counted instead of aborting: one unreadable file must not stop
/// the rest of the migration.
fn copy_tree(src: &Path, dst: &Path, report: &mut MigrationReport) {
    if src.is_dir() {
        if std::fs::create_dir_all(dst).is_err() {
            report.failed += 1;
            return;
        }
        let Ok(entries) = std::fs::read_dir(src) else {
            report.failed += 1;
            return;
        };
        for entry in entries.flatten() {
            copy_tree(&entry.path(), &dst.join(entry.file_name()), report);
        }
    } else if src.is_file() {
        if dst.exists() {
            report.skipped += 1;
            return;
        }
        if let Some(parent) = dst.parent()
            && std::fs::create_dir_all(parent).is_err()
        {
            report.failed += 1;
            return;
        }
        match std::fs::copy(src, dst) {
            Ok(_) => report.copied += 1,
            Err(_) => report.failed += 1,
        }
    }
}

/// Copies the user data TPT's migration button moves (`OptionsView.cpp:455-466`)
/// into `shared`: `saves/`, `stamps/`, `screenshots/`, `recordings/` and
/// `prefs.json`.
fn migrate_to_shared_data(shared: &Path) -> MigrationReport {
    let mut report = MigrationReport::default();
    if std::fs::create_dir_all(shared).is_err() {
        report.failed += 1;
        return report;
    }
    for dir in [
        crate::app_state::SAVES_DIR,
        crate::app_state::STAMPS_DIR,
        crate::gpu_ctx::screenshots::SCREENSHOTS_DIR,
        crate::gpu_ctx::screenshots::RECORDINGS_DIR,
    ] {
        copy_tree(Path::new(dir), &shared.join(dir), &mut report);
    }
    copy_tree(
        Path::new(crate::prefs::PREFS_FILE),
        &shared.join(crate::prefs::PREFS_FILE),
        &mut report,
    );
    report
}

/// Draws the floating Options window when [`AppState::options_open`] is set.
///
/// Also owns the application-level `Ctrl+O` shortcut, so it works on the main
/// menu as well as in game without `ui_panels::mod` having to know about it.
pub(super) fn draw_options_window(ctx: &egui::Context, state: &mut AppState) {
    handle_shortcuts(ctx, state);

    if !state.options_open {
        return;
    }

    let mut open = true;
    let mut changed = false;
    let mut reset_limits = false;
    let mut open_folder = false;
    let mut close_requested = false;

    let mut tab = options_tab(ctx);

    egui::Window::new(t("options.title"))
        .open(&mut open)
        .resizable(true)
        .default_width(500.0)
        .default_height(640.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                for candidate in OptionsTab::ALL {
                    if ui
                        .add(egui::Button::selectable(
                            tab == candidate,
                            candidate.label(),
                        ))
                        .clicked()
                    {
                        tab = candidate;
                    }
                }
            });
            ui.separator();

            egui::ScrollArea::vertical()
                .id_salt(tab.id_salt())
                .max_height(560.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    changed |= match tab {
                        OptionsTab::Graphics => section(ui, t("options.section_graphics"), |ui| {
                            graphics_section(ui, state, &mut reset_limits)
                        }),
                        OptionsTab::Simulation => {
                            section(ui, t("options.section_simulation"), |ui| {
                                simulation_section(ui, state)
                            })
                        }
                        OptionsTab::Voxel3d => section(ui, t("options.section_voxel3d"), |ui| {
                            super::voxel_options::draw_voxel3d_options(ui, state)
                        }),
                        OptionsTab::Advanced => section(ui, t("options.section_advanced"), |ui| {
                            advanced_section(ui, state, &mut open_folder)
                        }),
                        OptionsTab::Keybinds => {
                            section(ui, t("options.section_keybinds"), keybinds_section)
                        }
                    };
                });

            ui.separator();
            if ui.button(t("options.ok")).clicked() {
                close_requested = true;
            }
        });

    ctx.data_mut(|data| data.insert_temp(tab_key(), tab));

    state.options_open = open && !close_requested;

    if reset_limits {
        state.prefs.fps_limit = FpsLimit::default();
        state.prefs.draw_fps = DEFAULT_DRAW_FPS;
        changed = true;
    }
    if open_folder {
        open_data_folder();
    }
    if changed {
        state.apply_prefs();
    }
}

/// Handles the application-level shortcut the Options view owns: `Ctrl+O`
/// toggles this window (TPT has no keyboard binding for its Settings button;
/// this is the port's mnemonic). It is ignored while a text field has focus so
/// typing a save name never toggles the window.
fn handle_shortcuts(ctx: &egui::Context, state: &mut AppState) {
    // An open dialog owns the keyboard (`MakeActiveWindow`), so no shortcut
    // fires underneath it.
    if !state.modals.is_empty() {
        return;
    }
    if ctx.text_edit_focused() {
        return;
    }
    let shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::O);
    if ctx.input_mut(|input| input.consume_shortcut(&shortcut)) {
        if state.options_open {
            state.options_open = false;
        } else {
            state.open_options();
        }
    }
}

/// Draws one titled group and returns whether any control in it changed.
fn section(ui: &mut Ui, title: &str, add: impl FnOnce(&mut Ui) -> bool) -> bool {
    ui.add_space(4.0);
    ui.label(RichText::new(title).size(16.0).strong());
    ui.separator();
    let changed = add(ui);
    ui.add_space(6.0);
    changed
}

/// Pages of the Settings window, one per option group.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OptionsTab {
    Graphics,
    Simulation,
    Voxel3d,
    Advanced,
    Keybinds,
}

impl OptionsTab {
    const ALL: [Self; 5] = [
        Self::Graphics,
        Self::Simulation,
        Self::Voxel3d,
        Self::Advanced,
        Self::Keybinds,
    ];

    /// Locale key of the tab label, also used as the page heading.
    fn label(self) -> &'static str {
        match self {
            Self::Graphics => t("options.section_graphics"),
            Self::Simulation => t("options.section_simulation"),
            Self::Voxel3d => t("options.section_voxel3d"),
            Self::Advanced => t("options.section_advanced"),
            Self::Keybinds => t("options.section_keybinds"),
        }
    }

    /// Scroll-area salt, so each page remembers its own scroll position.
    fn id_salt(self) -> &'static str {
        match self {
            Self::Graphics => "tpt_options_graphics",
            Self::Simulation => "tpt_options_simulation",
            Self::Voxel3d => "tpt_options_voxel3d",
            Self::Advanced => "tpt_options_advanced",
            Self::Keybinds => "tpt_options_keybinds",
        }
    }
}

/// egui memory key holding the selected Settings page.
fn tab_key() -> egui::Id {
    egui::Id::unique("tpt_options_tab")
}

/// Selected page, defaulting to Graphics until the user switches.
fn options_tab(ctx: &egui::Context) -> OptionsTab {
    ctx.data_mut(|data| data.get_temp::<OptionsTab>(tab_key()))
        .unwrap_or(OptionsTab::Graphics)
}

/// Action/binding rows of the Key bindings page.
///
/// Mirrors the handlers that actually run: `ui_panels::handle_hotkeys` for
/// the simulation keys and display presets, the viewport's pointer/navigation
/// input, the tool palette's `Ctrl+P` shortcut and `options_window`'s
/// `Ctrl+O`. Modifier-drag rows list the gesture that starts the stroke.
const KEYBINDS: &[(&str, &str)] = &[
    ("options.controls_primary_tool", "options.bind_lmb"),
    ("options.controls_secondary_tool", "options.bind_rmb"),
    ("options.controls_sample_element", "options.bind_mmb"),
    ("options.controls_pan", "options.bind_mmb_drag"),
    ("options.controls_zoom", "options.bind_ctrl_wheel"),
    ("options.controls_brush_size", "options.bind_wheel"),
    ("options.controls_brush_size", "options.bind_brackets"),
    ("options.controls_line", "options.bind_shift_drag"),
    ("options.controls_rectangle", "options.bind_ctrl_drag"),
    ("options.controls_snap", "options.bind_shift_alt"),
    ("options.controls_fill", "options.bind_alt_click"),
    ("options.controls_pause", "options.bind_space"),
    ("options.controls_step", "options.bind_f"),
    ("options.controls_undo", "options.bind_undo"),
    ("options.controls_redo", "options.bind_redo"),
    ("options.controls_zoom_scope", "options.bind_z"),
    ("options.controls_display_presets", "options.bind_presets"),
    ("options.controls_property", "options.bind_ctrl_p"),
    (
        "options.controls_property_secondary",
        "options.bind_ctrl_shift_p",
    ),
    ("options.controls_options", "options.bind_ctrl_o"),
    ("options.controls_grave_console", "options.bind_grave"),
    ("options.controls_view_pan", "options.bind_wasd_arrows"),
    ("options.controls_view_zoom", "options.bind_wheel"),
    ("options.controls_reset_view", "options.bind_home"),
];

/// 3D sandbox action/binding rows, appended to the key table under a heading.
const VOXEL_KEYBINDS: &[(&str, &str)] = &[
    ("options.controls_voxel_paint", "options.bind_lmb"),
    ("options.controls_voxel_line", "options.bind_shift_drag"),
    ("options.controls_voxel_rectangle", "options.bind_ctrl_drag"),
    ("options.controls_voxel_pick", "options.bind_alt_lmb"),
    ("options.controls_voxel_orbit", "options.bind_rmb_drag"),
    ("options.controls_voxel_pan", "options.bind_mmb_drag"),
    ("options.controls_voxel_zoom", "options.bind_wheel"),
    ("options.controls_voxel_brush_size", "options.bind_brackets"),
    (
        "options.controls_voxel_brush_fine",
        "options.bind_alt_brackets",
    ),
    ("options.controls_voxel_move", "options.bind_wasd_arrows"),
];

/// Key bindings block: a read-only two-column table of action -> binding.
///
/// The key under Esc (`GraveExitsConsole`) is handled in gameplay/Options
/// code, so its row names the preference gate the same way the other rows name
/// their handlers.
fn keybinds_section(ui: &mut Ui) -> bool {
    egui::Grid::new("tpt_options_keybinds")
        .num_columns(2)
        .spacing([24.0, 6.0])
        .striped(true)
        .show(ui, |ui| {
            for &(action, binding) in KEYBINDS {
                ui.label(t(action));
                ui.label(RichText::new(t(binding)).strong());
                ui.end_row();
            }
            ui.label(RichText::new(t("options.voxel3d_keybinds")).strong());
            ui.label("");
            ui.end_row();
            for &(action, binding) in VOXEL_KEYBINDS {
                ui.label(t(action));
                ui.label(RichText::new(t(binding)).strong());
                ui.end_row();
            }
        });
    false
}

/// Graphics block: framerate caps and the restart-scoped window settings.
fn graphics_section(ui: &mut Ui, state: &mut AppState, reset_limits: &mut bool) -> bool {
    let mut changed = false;
    changed |= fps_limit_control(ui, &mut state.prefs.fps_limit);
    changed |= draw_fps_control(ui, &mut state.prefs.draw_fps);
    ui.horizontal(|ui| {
        if ui.button(t("options.reset")).clicked() {
            *reset_limits = true;
        }
        ui.label(
            RichText::new(t("options.hint_reset"))
                .small()
                .color(TEXT_DIM),
        );
    });

    ui.separator();
    changed |= option_checkbox(
        ui,
        &mut state.prefs.resizable,
        t("options.resizable"),
        t("options.resizable_note"),
        t("options.restart_notice"),
    );
    changed |= option_checkbox(
        ui,
        &mut state.prefs.fullscreen,
        t("options.fullscreen"),
        t("options.fullscreen_note"),
        t("options.restart_notice"),
    );
    changed |= option_checkbox(
        ui,
        &mut state.prefs.blurry_scaling,
        t("options.blurry_scaling"),
        t("options.blurry_scaling_note"),
        "",
    );

    // `OptionsView.cpp:340-362`: the window scale factor drives the
    // simulation viewport blit. TPT's companion `ForceIntegerScaling` has no
    // port: egui samples the simulation texture through its own rect, so a
    // whole-multiple snap would only letterbox small canvases.
    changed |= option_dropdown(
        ui,
        "tpt_window_scale",
        t("options.scale"),
        &mut state.prefs.scale,
        &WINDOW_SCALE_OPTIONS,
    );
    option_info(ui, t("options.scale_info"));

    ui.separator();
    ui.horizontal(|ui| {
        ui.label(t("options.language"));
        let mut language = state.prefs.language;
        egui::ComboBox::from_id_salt("tpt_language")
            .selected_text(language.label())
            .width(150.0)
            .show_ui(ui, |ui| {
                for candidate in Language::ALL {
                    ui.selectable_value(&mut language, candidate, candidate.label());
                }
            });
        if language != state.prefs.language {
            state.prefs.language = language;
            crate::i18n::set_language(language);
            changed = true;
        }
        ui.label(
            RichText::new(t("options.language_note"))
                .small()
                .color(TEXT_DIM),
        );
    });
    changed
}

/// Simulation block: heat, air, gravity, edge behaviour and temperature scale.
fn simulation_section(ui: &mut Ui, state: &mut AppState) -> bool {
    let mut changed = false;
    changed |= option_checkbox(
        ui,
        &mut state.prefs.heat_enable,
        t("options.heat_simulation"),
        "",
        t("options.heat_simulation_info"),
    );
    changed |= option_checkbox(
        ui,
        &mut state.prefs.newtonian_gravity,
        t("options.newtonian_gravity"),
        "",
        t("options.newtonian_gravity_info"),
    );
    changed |= option_checkbox(
        ui,
        &mut state.prefs.ambient_heat,
        t("options.ambient_heat"),
        "",
        t("options.ambient_heat_info"),
    );
    changed |= option_checkbox(
        ui,
        &mut state.prefs.water_equal_test,
        t("options.water_equalisation"),
        "",
        t("options.water_equalisation_info"),
    );

    changed |= option_dropdown(
        ui,
        "tpt_air_mode",
        t("options.air_simulation_mode"),
        &mut state.prefs.air_mode,
        &[
            (0, t("options.air_mode_on")),
            (1, t("options.air_mode_pressure_off")),
            (2, t("options.air_mode_velocity_off")),
            (3, t("options.air_mode_off")),
            (4, t("options.air_mode_no_update")),
        ],
    );

    // Ambient air temperature is edited in the selected scale
    // and stored in Kelvin, like TPT does.
    let scale = state.prefs.temperature_scale;
    let mut display_temp = temperature_to_scale(state.prefs.ambient_air_temp, scale);
    let min_display = temperature_to_scale(MIN_TEMP, scale);
    let max_display = temperature_to_scale(MAX_TEMP, scale);
    ui.horizontal(|ui| {
        ui.label(t("options.ambient_air_temp"));
        let response = ui.add(
            egui::DragValue::new(&mut display_temp)
                .speed(0.1)
                .range(min_display..=max_display)
                .max_decimals(2)
                .suffix(temperature_suffix(scale)),
        );
        if response.changed() {
            let kelvin = temperature_from_scale(display_temp, scale);
            if kelvin.is_finite() {
                state.prefs.ambient_air_temp = kelvin.clamp(MIN_TEMP, MAX_TEMP);
                changed = true;
            }
        }
        preview_swatch(ui, heat_colour(state.prefs.ambient_air_temp));
    });

    ui.horizontal(|ui| {
        ui.label(t("options.ambient_air_pressure"));
        if ui
            .add(
                egui::DragValue::new(&mut state.prefs.edge_pressure)
                    .speed(0.1)
                    .range(MIN_PRESSURE..=MAX_PRESSURE)
                    .max_decimals(2),
            )
            .changed()
        {
            changed = true;
        }
        preview_swatch(ui, pressure_colour(state.prefs.edge_pressure));
    });

    ui.horizontal(|ui| {
        ui.label(t("options.ambient_air_velocity"));
        if ui
            .add(
                egui::DragValue::new(&mut state.prefs.edge_velocity_x)
                    .speed(0.1)
                    .range(-MAX_EDGE_VELOCITY..=MAX_EDGE_VELOCITY)
                    .max_decimals(2)
                    .prefix(t("options.axis_x")),
            )
            .changed()
        {
            changed = true;
        }
        if ui
            .add(
                egui::DragValue::new(&mut state.prefs.edge_velocity_y)
                    .speed(0.1)
                    .range(-MAX_EDGE_VELOCITY..=MAX_EDGE_VELOCITY)
                    .max_decimals(2)
                    .prefix(t("options.axis_y")),
            )
            .changed()
        {
            changed = true;
        }
    });

    changed |= option_drag_f32(
        ui,
        t("options.vorticity_confinement"),
        &mut state.prefs.vorticity_coeff,
        0.01,
        0.0..=1.0,
        2,
    );

    changed |= option_dropdown(
        ui,
        "tpt_convection_mode",
        t("options.air_heat_convection_mode"),
        &mut state.prefs.convection_mode,
        &[
            (AIRC_NONE, t("options.convection_none")),
            (AIRC_LEGACY, t("options.convection_legacy")),
            (AIRC_BOUSSINESQ, t("options.convection_boussinesq")),
        ],
    );

    changed |= option_dropdown(
        ui,
        "tpt_gravity_mode",
        t("options.gravity_simulation_mode"),
        &mut state.prefs.gravity_mode,
        &[
            (0, t("options.gravity_mode_vertical")),
            (1, t("options.gravity_mode_off")),
            (2, t("options.gravity_mode_radial")),
            (3, t("options.gravity_mode_custom")),
        ],
    );
    if state.prefs.gravity_mode == 3 {
        ui.horizontal(|ui| {
            ui.label(t("options.custom_gravity"));
            if ui
                .add(
                    egui::DragValue::new(&mut state.prefs.custom_gravity_x)
                        .speed(0.1)
                        .range(-MAX_EDGE_VELOCITY..=MAX_EDGE_VELOCITY)
                        .max_decimals(2)
                        .prefix(t("options.axis_x")),
                )
                .changed()
            {
                changed = true;
            }
            if ui
                .add(
                    egui::DragValue::new(&mut state.prefs.custom_gravity_y)
                        .speed(0.1)
                        .range(-MAX_EDGE_VELOCITY..=MAX_EDGE_VELOCITY)
                        .max_decimals(2)
                        .prefix(t("options.axis_y")),
                )
                .changed()
            {
                changed = true;
            }
        });
        option_info(ui, t("options.custom_gravity_info"));
    }

    changed |= option_dropdown(
        ui,
        "tpt_edge_mode",
        t("options.edge_mode"),
        &mut state.prefs.edge_mode,
        &[
            (0, t("options.edge_mode_void")),
            (1, t("options.edge_mode_solid")),
            (2, t("options.edge_mode_loop")),
        ],
    );
    changed |= option_dropdown(
        ui,
        "tpt_temperature_scale",
        t("options.temperature_scale"),
        &mut state.prefs.temperature_scale,
        &[
            (TEMPSCALE_KELVIN, t("options.temp_scale_kelvin")),
            (TEMPSCALE_CELSIUS, t("options.temp_scale_celsius")),
            (TEMPSCALE_FAHRENHEIT, t("options.temp_scale_fahrenheit")),
        ],
    );
    changed |= option_checkbox(
        ui,
        &mut state.prefs.pretty_powder,
        t("options.pretty_powder"),
        "",
        t("options.pretty_powder_info"),
    );
    changed
}

/// Advanced block: category, tool, decoration and undo settings.
fn advanced_section(ui: &mut Ui, state: &mut AppState, open_folder: &mut bool) -> bool {
    let mut changed = false;
    changed |= option_checkbox(
        ui,
        &mut state.prefs.mouse_click_required,
        t("options.sticky_categories"),
        "",
        t("options.sticky_categories_info"),
    );
    changed |= option_checkbox(
        ui,
        &mut state.prefs.include_pressure,
        t("options.include_pressure"),
        "",
        t("options.include_pressure_info"),
    );
    changed |= option_checkbox(
        ui,
        &mut state.prefs.grave_exits_console,
        t("options.grave_exits_console"),
        "",
        t("options.grave_exits_console_info"),
    );
    // `OptionsView.cpp:413-415`: only compiled in builds with a platform
    // clipboard; the port always offers it and falls back to the in-process
    // clipboard if the platform request fails.
    changed |= option_checkbox(
        ui,
        &mut state.prefs.native_clipboard,
        t("options.native_clipboard"),
        t("options.native_clipboard_note"),
        t("options.native_clipboard_info"),
    );
    // The blend mode belongs to the tool (`DecorationTool`'s `ToolID`), so the
    // control shows the mode bound to either button and retargets both.
    let mut deco_mode = state
        .primary_tool
        .deco_mode()
        .or_else(|| state.secondary_tool.deco_mode())
        .unwrap_or_else(|| state.prefs.deco_mode())
        .index();
    if option_dropdown(
        ui,
        "tpt_deco_mode",
        t("options.deco_mode"),
        &mut deco_mode,
        &[
            (DecoMode::Draw.index(), t("options.deco_mode_draw")),
            (DecoMode::Clear.index(), t("options.deco_mode_clear")),
            (DecoMode::Add.index(), t("options.deco_mode_add")),
            (DecoMode::Subtract.index(), t("options.deco_mode_subtract")),
            (DecoMode::Multiply.index(), t("options.deco_mode_multiply")),
            (DecoMode::Divide.index(), t("options.deco_mode_divide")),
            (DecoMode::Smudge.index(), t("options.deco_mode_smudge")),
        ],
    ) {
        state.prefs.deco_mode = deco_mode;
        let mode = DecoMode::from_index(deco_mode);
        for tool in [&mut state.primary_tool, &mut state.secondary_tool] {
            if let SelectedTool::Sign { mode: bound } = tool {
                *bound = mode;
            }
        }
        changed = true;
    }

    ui.horizontal(|ui| {
        ui.label(t("options.deco_colour"));
        let [red, green, blue, alpha] = state.prefs.deco_colour();
        let mut colour = Color32::from_rgba_unmultiplied(red, green, blue, alpha);
        if ui.color_edit_button_srgba(&mut colour).changed() {
            let [red, green, blue, alpha] = colour.to_srgba_unmultiplied();
            state.prefs.deco_red = red;
            state.prefs.deco_green = green;
            state.prefs.deco_blue = blue;
            state.prefs.deco_alpha = alpha;
            changed = true;
        }
        for (prefix, channel) in [
            (t("options.channel_r"), &mut state.prefs.deco_red),
            (t("options.channel_g"), &mut state.prefs.deco_green),
            (t("options.channel_b"), &mut state.prefs.deco_blue),
            (t("options.channel_a"), &mut state.prefs.deco_alpha),
        ] {
            if ui
                .add(
                    egui::DragValue::new(channel)
                        .speed(1.0)
                        .range(0..=255)
                        .prefix(prefix),
                )
                .changed()
            {
                changed = true;
            }
        }
    });

    ui.horizontal(|ui| {
        if ui
            .button(t("options.open_data_folder"))
            .on_hover_text(t("options.open_data_folder_desc"))
            .clicked()
        {
            *open_folder = true;
        }
        if ui
            .button(t("options.migrate_shared"))
            .on_hover_text(t("options.migrate_shared_note"))
            .clicked()
        {
            let status = match shared_data_dir() {
                Some(shared) => {
                    let report = migrate_to_shared_data(&shared);
                    let copied = report.copied.to_string();
                    let skipped = report.skipped.to_string();
                    let failed = report.failed.to_string();
                    let path = shared.display().to_string();
                    if report.failed > 0 {
                        t_fmt(
                            "options.migration_partial",
                            &[&copied, &skipped, &failed, &path],
                        )
                    } else {
                        t_fmt(
                            "options.migration_done",
                            &[&copied, &skipped, &path],
                        )
                    }
                }
                None => t("options.migration_no_shared").to_string(),
            };
            ui.ctx().data_mut(|data| {
                data.insert_temp(migration_status_key(), status);
            });
        }
    });
    if let Some(status) = ui
        .ctx()
        .data_mut(|data| data.get_temp::<String>(migration_status_key()))
    {
        ui.label(RichText::new(status).small().color(TEXT_DIM));
    }

    changed |= option_drag_u32(
        ui,
        t("options.undo_history_limit"),
        &mut state.prefs.undo_history_limit,
        1..=MAX_UNDO_HISTORY_LIMIT,
    );
    option_info(ui, t("options.undo_history_limit_info"));
    changed
}

/// Small colour rectangle used by the ambient air previews.
fn preview_swatch(ui: &mut Ui, colour: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(22.0, 16.0), Sense::hover());
    ui.painter()
        .rect_filled(rect, CornerRadius::same(3), colour);
}

/// Approximate blue-to-red heat gradient for the temperature preview.
fn heat_colour(kelvin: f32) -> Color32 {
    let t = ((kelvin - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)).clamp(0.0, 1.0);
    if t < 0.5 {
        let k = t * 2.0;
        Color32::from_rgb(
            (k * 255.0) as u8,
            (k * 80.0) as u8,
            ((1.0 - k) * 255.0) as u8,
        )
    } else {
        let k = (t - 0.5) * 2.0;
        Color32::from_rgb(255, (80.0 + k * 175.0) as u8, 0)
    }
}

/// TPT's pressure preview: positive is red, negative is blue.
fn pressure_colour(pressure: f32) -> Color32 {
    if pressure > 0.0 {
        let k = (pressure / 8.0).clamp(0.0, 1.0);
        Color32::from_rgb((k * 255.0) as u8, 0, 0)
    } else {
        let k = (-pressure / 8.0).clamp(0.0, 1.0);
        Color32::from_rgb(0, 0, (k * 255.0) as u8)
    }
}

/// Opens the data directory (saves and `prefs.json`) in the platform file
/// manager. Failures are logged and otherwise ignored.
fn open_data_folder() {
    let path = Path::new(SAVES_DIR);
    let _ = std::fs::create_dir_all(path);

    let result = if cfg!(target_os = "windows") {
        std::process::Command::new("explorer").arg(path).spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(path).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(path).spawn()
    };

    if let Err(err) = result {
        eprintln!("[tpt-app] could not open '{SAVES_DIR}': {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_scale_round_trips() {
        for kelvin in [0.0f32, 273.15, 300.0, 1000.0] {
            for scale in [TEMPSCALE_KELVIN, TEMPSCALE_CELSIUS, TEMPSCALE_FAHRENHEIT] {
                let value = temperature_to_scale(kelvin, scale);
                let back = temperature_from_scale(value, scale);
                assert!((back - kelvin).abs() < 0.01, "{kelvin} K in scale {scale}");
            }
        }
    }

    #[test]
    fn temperature_suffix_matches_scale() {
        assert_eq!(temperature_suffix(TEMPSCALE_KELVIN), " K");
        assert_eq!(temperature_suffix(TEMPSCALE_CELSIUS), " °C");
        assert_eq!(temperature_suffix(TEMPSCALE_FAHRENHEIT), " °F");
    }
}
