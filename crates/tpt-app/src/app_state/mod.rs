//! Application-level state for The Powder Toy (Rust + wgpu + CubeCL).
//!
//! This module owns the two top-level screens (main menu / in-game), the
//! new-canvas dialog, the currently selected drawing tools, and all
//! simulation-adjacent state (engine, undo history, Lua runtime, brush and
//! FPS bookkeeping). Rendering and input live in `main.rs`, `brushes.rs`
//! and `ui_panels`; this module is deliberately UI-framework agnostic.
//!
//! The type definitions live here; the [`AppState`] behaviour lives in
//! `commands` (lifecycle, save/load, stepping, preferences) and `inspector`
//! (cached device probes and wall samples), the canvas dialog types in
//! `canvas_dialog`, the bound-tool enum in `tool`, the canvas sign editor in
//! `signs`, the stamp files in `stamps`, the decoration colour picker in
//! `colour` and the 3D sandbox state (mode, palette, undo log, save browser)
//! in `voxel`.

mod canvas_dialog;
mod colour;
mod commands;
mod inspector;
mod property;
pub mod selection;
mod signs;
mod stamps;
mod tool;
pub(crate) mod voxel;

pub use canvas_dialog::{CanvasSizePreset, NewCanvasDialog, format_save_time};
pub use colour::ColourPicker;
pub use property::{
    PropertyKind, PropertyValue, PropertyWindow, format_value, temperature_from_scale,
    temperature_suffix, temperature_to_scale, value_from_prefs,
};
pub use signs::{SignEditor, clamp_sign_text};
pub use stamps::{STAMPS_DIR, StampBrowser};
pub use tool::{
    SelectedTool, pack_deco_colour, persist_tool_settings, sim_tool_from_key, sim_tool_key,
};
pub use voxel::{
    GameMode, VOXEL_SAVE_EXTENSION, VOXEL_UNDO_LIMIT, Voxel3dBrickSnapshot, Voxel3dUndoHistory,
    VoxelDrawMode, VoxelTool,
};

use std::path::PathBuf;
use std::time::Instant;

use tpt_core::commands::BrushShape;
use tpt_core::elements::MenuCategory;
use tpt_core::particle::EdgeMode;
use tpt_core::voxel3d::{BrushShape3d, VoxelCategory};
use tpt_lua::LuaScriptingEngine;
use tpt_save::UndoHistory;
use tpt_sim::SimulationEngine;
use tpt_sim::voxel3d::Voxel3dEngine;
use tpt_sim::voxel3d::pick::PickHit;

use crate::prefs::Prefs;
use crate::render3d::{Camera, Renderer3d};

/// Directory (relative to the process working directory) holding `.tptrs` saves.
pub const SAVES_DIR: &str = "saves";
/// File extension used by the native save format.
pub const SAVE_EXTENSION: &str = "tptrs";
/// Number of frame-time samples used for the rolling FPS estimate.
pub const FPS_SAMPLE_WINDOW: usize = 120;
/// Upper bound on the simulation steps one [`AppState::tick_frame`] call may
/// run to catch up with the configured TPS.
///
/// High TPS targets (240, 1000) need more than one step per event-loop
/// iteration; the bound keeps a slow step from turning the loop into an
/// unbounded burst that would stall input and presentation.
pub const MAX_CATCHUP_STEPS: u32 = 16;
/// Number of console commands kept for Up/Down recall.
///
/// Ports the 25-entry cap of `ConsoleModel::AddLastCommand`
/// (`ConsoleModel.cpp:43-51`).
pub const CONSOLE_HISTORY_LIMIT: usize = 25;

/// Up/Down command history of the console.
///
/// Ports `ConsoleModel` (`ConsoleModel.cpp:5-51`): [`Self::index`] points one
/// past the last entry while a new command is being edited, [`Self::draft`]
/// preserves that new command when the user walks back into the history, and
/// a repeated command collapses into its newest position instead of being
/// recorded twice.
#[derive(Debug, Default)]
pub struct CommandHistory {
    entries: Vec<String>,
    index: usize,
    draft: String,
}

impl CommandHistory {
    /// Appends an executed command and resets recall to the new-command slot.
    pub fn record(&mut self, command: &str) {
        let command = command.trim();
        if command.is_empty() {
            self.reset();
            return;
        }
        // `ConsoleView.cpp:45-49` saves the in-progress text before walking
        // back; the port keeps the recall list short by collapsing a repeated
        // command into its newest position instead of appending another copy.
        if let Some(index) = self.entries.iter().position(|entry| entry == command) {
            self.entries.remove(index);
        }
        self.entries.push(command.to_string());
        if self.entries.len() > CONSOLE_HISTORY_LIMIT {
            self.entries.remove(0);
        }
        self.reset();
    }

    /// Recalls the previous entry, saving `current` as the draft on the first
    /// step back. Stays on the oldest entry at the start of the list.
    pub fn previous(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        if self.index >= self.entries.len() {
            self.draft = current.to_string();
        }
        self.index = self.index.saturating_sub(1).min(self.entries.len() - 1);
        self.entries.get(self.index).cloned()
    }

    /// Moves towards newer entries; stepping past the newest restores the
    /// draft that was being edited before recall started.
    // The recall cursor is two-way (`previous`/`next`), so `Iterator` is the
    // wrong abstraction here.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<String> {
        if self.index >= self.entries.len() {
            return None;
        }
        self.index += 1;
        if self.index >= self.entries.len() {
            Some(self.draft.clone())
        } else {
            self.entries.get(self.index).cloned()
        }
    }

    /// Number of recorded commands.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no command has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn reset(&mut self) {
        self.index = self.entries.len();
        self.draft.clear();
    }
}

/// Action the console is waiting for when a modal closes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PendingConsoleAction {
    /// `!quit` confirmation.
    Quit,
    /// `!save` without a name asks for one through a text prompt.
    SaveAs,
    /// `!load` without a name asks for one through a text prompt.
    LoadAs,
}

/// Console-owned state that is not part of the Lua runtime.
#[derive(Debug, Default)]
pub struct ConsoleState {
    /// Commands executed this session, recalled with Up/Down.
    pub history: CommandHistory,
    /// Modal opened by the console and the action to run when it closes.
    pub pending: Option<(u64, PendingConsoleAction)>,
}

/// One reusable dialog. Ports the shared `gui/dialogues` windows:
/// `ConfirmPrompt.cpp` (confirm/cancel), `TextPrompt.cpp` (text field with
/// Okay/Cancel) and `ErrorMessage.cpp` (title, message, Dismiss).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModalKind {
    /// Cancel/confirm pair with a message.
    Confirm {
        title: String,
        message: String,
        confirm_label: String,
    },
    /// One text field; the submitted string is returned in
    /// [`ModalOutcome::Submitted`].
    Prompt {
        title: String,
        message: String,
        input: String,
        hint: String,
        submit_label: String,
    },
    /// Error box with the failing message and a single dismiss button.
    Error { title: String, message: String },
}

/// A dialog plus its stack identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Modal {
    id: u64,
    kind: ModalKind,
    /// Set once the first frame moved keyboard focus inside the dialog, so
    /// later frames let the user tab between its widgets.
    focus_requested: bool,
}

impl Modal {
    /// Stable identity of this dialog.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Content and controls of this dialog.
    pub fn kind(&self) -> &ModalKind {
        &self.kind
    }

    /// Mutable content, for the dialog renderer's text field.
    pub fn kind_mut(&mut self) -> &mut ModalKind {
        &mut self.kind
    }

    /// Returns whether the renderer still has to move focus into the dialog.
    pub fn take_focus_request(&mut self) -> bool {
        if self.focus_requested {
            false
        } else {
            self.focus_requested = true;
            true
        }
    }
}

/// What the user chose in the newest modal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModalOutcome {
    /// The confirm button was pressed.
    Confirmed,
    /// Cancel was pressed or Escape was hit.
    Cancelled,
    /// A text prompt was submitted; carries the entered string.
    Submitted(String),
    /// The error box was dismissed.
    Dismissed,
}

/// Identity and outcome of a closed modal, consumed by the panel that opened it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModalResult {
    /// [`Modal::id`] of the dialog that closed.
    pub id: u64,
    /// What the user chose.
    pub outcome: ModalOutcome,
}

/// Stack of reusable dialogs.
///
/// The newest dialog is the only one drawn and owns the keyboard, mirroring
/// `MakeActiveWindow` in the C++ `ui::Window` stack.
#[derive(Debug, Default)]
pub struct ModalStack {
    entries: Vec<Modal>,
    next_id: u64,
}

impl ModalStack {
    /// Pushes a cancel/confirm prompt and returns its identity.
    pub fn open_confirm(
        &mut self,
        title: impl Into<String>,
        message: impl Into<String>,
        confirm_label: impl Into<String>,
    ) -> u64 {
        self.push(ModalKind::Confirm {
            title: title.into(),
            message: message.into(),
            confirm_label: confirm_label.into(),
        })
    }

    /// Pushes a one-field text prompt and returns its identity.
    pub fn open_prompt(
        &mut self,
        title: impl Into<String>,
        message: impl Into<String>,
        input: impl Into<String>,
        hint: impl Into<String>,
        submit_label: impl Into<String>,
    ) -> u64 {
        self.push(ModalKind::Prompt {
            title: title.into(),
            message: message.into(),
            input: input.into(),
            hint: hint.into(),
            submit_label: submit_label.into(),
        })
    }

    /// Pushes an error box and returns its identity.
    pub fn open_error(
        &mut self,
        title: impl Into<String>,
        message: impl Into<String>,
    ) -> u64 {
        self.push(ModalKind::Error {
            title: title.into(),
            message: message.into(),
        })
    }

    fn push(&mut self, kind: ModalKind) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(Modal {
            id,
            kind,
            focus_requested: false,
        });
        id
    }

    /// Mutable newest dialog.
    pub fn top_mut(&mut self) -> Option<&mut Modal> {
        self.entries.last_mut()
    }

    /// Number of stacked dialogs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no dialog is open.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Pops `id` when it is the newest dialog, so a stale result can never
    /// close a newer prompt.
    pub fn close(&mut self, id: u64) -> bool {
        if self.entries.last().is_some_and(|modal| modal.id == id) {
            self.entries.pop();
            true
        } else {
            false
        }
    }
}

/// Which top-level screen the application is currently showing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AppScreen {
    /// Title screen with the new-simulation and load-save cards.
    #[default]
    MainMenu,
    /// Active simulation with HUD, toolbars and canvas.
    InGame,
}

/// Complete application state shared by the renderer, HUD and input handlers.
pub struct AppState {
    /// Active top-level screen.
    pub screen: AppScreen,
    /// Main-menu new-canvas / load-save dialog state.
    pub new_canvas: NewCanvasDialog,
    /// `./stamps` browser and the `Ctrl+S`/`K`/`L` stamp state.
    pub stamps: stamps::StampBrowser,
    /// GPU-backed simulation (particles, air, walls, rigid bodies).
    pub engine: SimulationEngine,
    /// Undo/redo snapshot history for the current simulation.
    pub undo: UndoHistory,
    /// Floating sign text editor (placement or edit of one sign).
    pub sign_editor: SignEditor,
    /// Lua scripting runtime and console log.
    pub lua: LuaScriptingEngine,

    /// Active simulation flavour (classic powders or the 3D sandbox).
    pub mode: GameMode,
    /// Lazily created 3D voxel simulation; stays resident after leaving 3D.
    pub voxel3d: Option<Box<Voxel3dEngine>>,
    /// Deferred 3D renderer, created with the application device on demand.
    pub voxel_renderer: Option<Renderer3d>,
    /// Boundary behaviour of loose voxels in the 3D sandbox.
    pub voxel_edge_mode: EdgeMode,
    /// Orbit camera of the 3D viewport.
    pub voxel_camera: Camera,
    /// Element / eraser / heat / rigid-box tool selected in the 3D palette.
    pub voxel_tool: VoxelTool,
    /// Gesture used when the primary button is pressed.
    pub voxel_draw_mode: VoxelDrawMode,
    /// Active element category tab in the 3D palette.
    pub voxel_category: VoxelCategory,
    /// Brush radii along x, y and z in voxels (`1..=64` each).
    pub voxel_brush_radii: [u32; 3],
    /// Rasterisation shape of the 3D brush.
    pub voxel_brush_shape: BrushShape3d,
    /// Per-axis scale locks for `[` / `]` resizing.
    pub voxel_brush_locks: [bool; 3],
    /// Rigid box half extents in voxels (`1..=16` per axis).
    pub voxel_box_half: [u32; 3],
    /// Element id of the rigid box tool.
    pub voxel_box_element: u32,
    /// Bounded brick undo/redo log of the 3D sandbox.
    pub voxel_undo: Voxel3dUndoHistory,
    /// Last brick count sampled from the device, refreshed every 250 ms.
    pub voxel_active_bricks: Option<u32>,
    /// Timestamp of the last active-brick probe.
    pub voxel_stats_at: Instant,
    /// Most recent voxel picked by a click, for the viewport crosshair.
    pub voxel_last_pick: Option<PickHit>,
    /// Cached hover pick used by the brush ghost.
    pub voxel_hover_pick: Option<PickHit>,
    /// Pointer position the cached hover pick belongs to.
    pub voxel_hover_pointer: Option<[f32; 2]>,
    /// Timestamp of the last hover pick.
    pub voxel_hover_at: Instant,
    /// Centre of the first brush stamp of the current stroke.
    pub voxel_stroke_start: Option<[f32; 3]>,
    /// Centre of the latest brush stamp of the current stroke.
    pub voxel_stroke_last: Option<[f32; 3]>,
    /// Draw mode captured when the stroke started.
    pub voxel_stroke_mode: Option<VoxelDrawMode>,
    /// Bricks captured for the current stroke's undo entry.
    pub voxel_stroke_snapshot: Voxel3dBrickSnapshot,
    /// Bricks already captured in [`Self::voxel_stroke_snapshot`].
    pub voxel_stroke_bricks: std::collections::HashSet<u32>,
    /// True while a paint stroke owns the current undo snapshot.
    pub voxel_stroke_active: bool,
    /// Save name shown by the 3D toolbar.
    pub voxel_save_name_input: String,
    /// `.tpt3` files found in [`SAVES_DIR`] for the main-menu browser.
    pub voxel_saves: Vec<PathBuf>,
    /// Physical pixel size of the 3D viewport reported by the last UI pass.
    pub voxel_viewport_size: Option<(u32, u32)>,

    /// Persistent user preferences, loaded from `prefs.json` on startup.
    pub prefs: Prefs,
    /// Whether the floating Options window is open.
    pub options_open: bool,
    /// Whether the floating Render settings window is open.
    pub render_window_open: bool,
    /// Floating decoration colour picker (`ColourPickerActivity`) plus the
    /// eight editable presets from `GameModel::colourPresets`.
    pub colour_picker: ColourPicker,
    /// Floating property window (`PropertyWindow`): field, value and open
    /// state, seeded from `Prop.Type` / `Prop.Value`.
    pub property_window: PropertyWindow,
    /// Timestamp of the most recent [`Self::tick_frame`] call, used to measure
    /// the wall-clock time that the step accumulator consumes.
    pub last_sim_step_instant: Instant,
    /// Unspent simulation time in seconds, in units of the configured step
    /// interval.
    ///
    /// Carrying the remainder between calls is what makes the configured TPS
    /// exact: a plain "one step per event-loop iteration" gate quantises to the
    /// loop rate divided by the interval (a 180 Hz loop yields 45 TPS for a
    /// 60 TPS target). Clamped to [`MAX_CATCHUP_STEPS`] intervals so a stall
    /// never triggers a long burst.
    pub sim_step_debt: f32,
    /// Number of undo snapshots currently retained by [`Self::undo`].
    pub undo_limit: usize,

    /// Tool bound to the primary (left) mouse button.
    pub primary_tool: SelectedTool,
    /// Tool bound to the secondary (right) mouse button.
    pub secondary_tool: SelectedTool,
    /// Brush rasterization shape.
    pub brush_shape: BrushShape,
    /// Brush radius along x, in simulation pixels.
    pub brush_rx: i32,
    /// Brush radius along y, in simulation pixels.
    pub brush_ry: i32,
    /// When true, drawing replaces existing particles instead of stacking.
    pub replace_mode: bool,
    /// Element category shown in the bottom toolbar.
    pub active_category: MenuCategory,

    /// Whether the floating Lua console window is open.
    pub show_lua_console: bool,
    /// Script source currently in the console editor.
    pub lua_input: String,
    /// File stem used by the save dialog.
    pub save_name_input: String,

    /// Simulation-space position under the mouse cursor, if inside the canvas.
    pub hover_pos: Option<(i32, i32)>,
    /// Simulation-space anchor of the in-progress drag, if any.
    pub drag_start: Option<(i32, i32)>,
    /// Last simulation-space position painted, used to interpolate strokes.
    pub last_cursor_sim: Option<(i32, i32)>,
    /// Secondary button plus cursor position of a held simulation tool.
    ///
    /// `GameView::OnTick` (`GameView.cpp:1754-1758`) re-draws from `lastPoint`
    /// to `currentPoint` every tick while the button is down, which is what
    /// makes the continuous tools (heat, cool, air, vacuum, wind, mix,
    /// cyclone) act while the cursor stands still.
    pub held_sim_tool: Option<(bool, (i32, i32))>,
    /// Canvas zoom factor applied by the HUD.
    pub zoom: f32,
    /// Navigation mode: the wheel zooms, `WASD`/the arrow keys pan and the
    /// canvas ignores paint input.
    ///
    /// TPT has no equivalent; the port adds it because the brush is bound to
    /// both mouse buttons, so panning a zoomed canvas otherwise means dragging
    /// the middle button while the tool stays armed.
    pub move_view: bool,
    /// Pending region selection (`SelectMode`), copy/cut/paste state.
    pub select_mode: selection::SelectMode,
    /// First corner of the rubber band.
    pub select_start: Option<(i32, i32)>,
    /// Second corner of the rubber band.
    pub select_end: Option<(i32, i32)>,
    /// Last captured region (`GameController::clipboard`).
    pub clipboard: Option<selection::Region>,
    /// A clipboard region being placed under the cursor.
    pub placement: Option<selection::Placement>,
    /// `zoomCursorFixed`: a click inside the magnification window pins the
    /// scope, so releasing the zoom key keeps it open and the brackets go back
    /// to resizing the brush (`GameView::OnMouseUp`).
    pub zoom_cursor_fixed: bool,

    /// Rolling presented-frames-per-second estimate.
    pub fps: f32,
    /// Timestamp of the previous presented frame.
    pub last_frame_instant: std::time::Instant,
    /// Recent presented-frame durations in seconds, newest at the back.
    pub frame_times: std::collections::VecDeque<f32>,
    /// Rolling simulation-steps-per-second estimate (logical TPS).
    pub tps: f32,
    /// Timestamps of the simulation steps inside the last second, used by
    /// [`Self::tps`]; entries older than a second are pruned every frame.
    pub step_instants: std::collections::VecDeque<std::time::Instant>,

    /// Short user-facing status line (save/load/undo feedback).
    pub status_message: String,
    /// Whether [`Self::status_message`] describes an error.
    pub status_is_error: bool,

    /// Set by `tpt.exit`; the winit event loop observes it and calls
    /// `ActiveEventLoop::exit`.
    pub exit_requested: bool,

    /// Reusable dialog layer shared by the console and the panels.
    pub modals: ModalStack,
    /// Command history and the prompt the console is waiting on.
    pub console: ConsoleState,
    /// Outcome of the modal closed on the previous frame. The panel that
    /// opened the dialog consumes it by matching [`ModalResult::id`].
    pub modal_result: Option<ModalResult>,
}

impl AppState {
    /// Queues a canvas screenshot for the next rendered frame.
    ///
    /// `GameView.cpp:1449-1462` (`SDL_SCANCODE_P` / `SDL_SCANCODE_F2`) sets
    /// `doScreenshot`, which `GameView::Render` turns into
    /// `TakeScreenshot(0, 0)`. The read-back itself needs the wgpu device, so
    /// the request travels through the screenshot mailbox to
    /// [`crate::gpu_ctx::GpuContext::encode_sim_blit`].
    pub fn take_screenshot(&mut self) {
        crate::gpu_ctx::screenshots::request_screenshot();
    }

    /// Drains completed captures and pending `tpt.record` requests into the
    /// HUD status line.
    ///
    /// Called once per UI frame from [`crate::ui_panels::draw_app_ui`]; the
    /// status strings are localized in `locales/*/app.json`, matching the
    /// reference's on-screen feedback (`GameView.cpp:2340-2347`).
    pub fn poll_screenshot_events(&mut self) {
        use crate::gpu_ctx::screenshots::{self, CaptureOutcome};
        use crate::i18n::{t, t_fmt};

        while let Some(outcome) = screenshots::take_outcome() {
            match outcome {
                CaptureOutcome::Screenshot(Ok(path)) => self.set_status(
                    t_fmt(
                        "app.status_screenshot_saved",
                        &[&path.display().to_string()],
                    ),
                    false,
                ),
                CaptureOutcome::Screenshot(Err(err)) => {
                    self.set_status(t_fmt("app.status_screenshot_failed", &[&err]), true);
                }
                CaptureOutcome::Recording(Ok(path)) => self.set_status(
                    t_fmt(
                        "app.status_recording_saved",
                        &[&path.display().to_string()],
                    ),
                    false,
                ),
                CaptureOutcome::Recording(Err(err)) => {
                    self.set_status(t_fmt("app.status_recording_failed", &[&err]), true);
                }
            }
        }

        // `tpt.record` lives in `tpt-lua`, which cannot name `tpt-app` types
        // (dependency cycle), so it appends requests to a Lua global (one
        // per line) that this per-frame drain consumes.
        let globals = self.lua.lua.globals();
        let Ok(queue) = globals.get::<String>(screenshots::RECORD_REQUEST_GLOBAL) else {
            return;
        };
        let _ = globals.set(screenshots::RECORD_REQUEST_GLOBAL, false);

        for request in queue.lines().filter(|line| !line.is_empty()) {
            if request == "stop" {
                // A successful stop finishes asynchronously on the recorder
                // thread; its `CaptureOutcome::Recording` prints the final
                // path when the APNG has been written.
                if screenshots::stop_recording().is_err() {
                    self.set_status(t("app.status_recording_not_active"), true);
                }
                continue;
            }
            if !request.starts_with("start") {
                continue;
            }

            let requested = request
                .strip_prefix("start")
                .and_then(|rest| rest.strip_prefix(screenshots::RECORD_REQUEST_SEPARATOR))
                .map(PathBuf::from);
            match screenshots::start_recording(requested.as_deref()) {
                Ok(path) => self.set_status(
                    t_fmt(
                        "app.status_recording_started",
                        &[&path.display().to_string()],
                    ),
                    false,
                ),
                Err(err) => {
                    self.set_status(t_fmt("app.status_recording_failed", &[&err]), true);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_recall_walks_back_to_the_oldest_and_restores_the_draft() {
        let mut history = CommandHistory::default();
        history.record("print(1)");
        history.record("print(2)");

        assert_eq!(history.previous("draft"), Some("print(2)".into()));
        assert_eq!(history.previous("draft"), Some("print(1)".into()));
        // At the oldest entry the recall stays put instead of wrapping.
        assert_eq!(history.previous("draft"), Some("print(1)".into()));

        assert_eq!(history.next(), Some("print(2)".into()));
        assert_eq!(history.next(), Some("draft".into()));
        assert_eq!(history.next(), None);
    }

    #[test]
    fn history_duplicates_collapse_and_newest_entries_win() {
        let mut history = CommandHistory::default();
        history.record("!reset");
        history.record("!save demo");
        // A non-consecutive repeat moves to the newest slot instead of
        // appearing twice in the recall list.
        history.record("!reset");

        assert_eq!(history.len(), 2);
        assert_eq!(history.previous(""), Some("!reset".into()));
        assert_eq!(history.previous(""), Some("!save demo".into()));
    }

    #[test]
    fn history_is_capped_at_the_console_limit() {
        let mut history = CommandHistory::default();
        for index in 0..CONSOLE_HISTORY_LIMIT + 5 {
            history.record(&format!("lua {index}"));
        }
        assert_eq!(history.len(), CONSOLE_HISTORY_LIMIT);

        let newest = format!("lua {}", CONSOLE_HISTORY_LIMIT + 4);
        assert_eq!(history.previous(""), Some(newest));
        for _ in 0..CONSOLE_HISTORY_LIMIT {
            let _ = history.previous("");
        }
        // The five oldest entries were dropped.
        assert_eq!(history.previous(""), Some("lua 5".into()));
    }

    #[test]
    fn history_resets_recall_after_the_next_command() {
        let mut history = CommandHistory::default();
        history.record("!quit");
        assert_eq!(history.previous("draft"), Some("!quit".into()));
        history.record("!clear");
        assert_eq!(history.previous("draft"), Some("!clear".into()));
    }

    #[test]
    fn modal_stack_only_closes_the_newest_modal() {
        let mut stack = ModalStack::default();
        let first = stack.open_confirm("Quit", "Are you sure?", "Confirm");
        let second = stack.open_error("Error", "failed");

        assert_eq!(stack.len(), 2);
        assert!(!stack.close(first), "the error box owns the stack");
        assert!(stack.close(second));
        assert!(stack.close(first));
        assert!(stack.is_empty());
    }
}
