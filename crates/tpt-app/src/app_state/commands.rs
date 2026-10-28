//! [`AppState`](super::AppState) behaviour: lifecycle, save/load, stepping,
//! preferences.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::Path;
use std::time::{Duration, Instant};

use cubecl::client::Client;
use tpt_core::commands::BrushShape;
use tpt_core::constants::{CELL, DEFAULT_XRES, DEFAULT_YRES, PT_DUST, PT_STNE};
use tpt_core::elements::MenuCategory;
use tpt_core::particle::{EdgeMode, GpuParticle, SimulationConfig};
use tpt_core::voxel3d::{BrushShape3d, VoxelCategory};
use tpt_lua::LuaScriptingEngine;
use tpt_save::{SaveData, UndoHistory};
use tpt_sim::SimulationEngine;
use tpt_sim::buffers::PixelProbe;

use crate::i18n::{t, t_fmt};
use crate::prefs::{FpsLimit, MAX_SANE_DRAW_FPS, MAX_UNDO_HISTORY_LIMIT, MIN_SANE_DRAW_FPS, Prefs};

use super::property::{temperature_suffix, temperature_to_scale};
use super::{
    AppScreen, AppState, ColourPicker, ConsoleState, FPS_SAMPLE_WINDOW, GameMode,
    MAX_CATCHUP_STEPS, ModalStack, NewCanvasDialog, PropertyWindow, SAVE_EXTENSION, SAVES_DIR,
    SelectedTool, SignEditor, StampBrowser, Voxel3dUndoHistory, VoxelDrawMode, VoxelTool,
};

/// Shortest gap between two inspector device probes while the cursor moves.
const INSPECTOR_PROBE_MIN_INTERVAL: Duration = Duration::from_millis(40);
/// A probe older than this is always refreshed, even for a stationary cursor.
const INSPECTOR_PROBE_MAX_AGE: Duration = Duration::from_millis(150);

/// Last device probe rendered by the HUD inspector.
struct InspectorCache {
    x: i32,
    y: i32,
    cell: (u32, u32),
    sampled_at: Instant,
    probe: Option<PixelProbe>,
}

// The initialiser is already a `const` block; clippy 1.99 reports it anyway.
thread_local! {
    /// The inspector is drawn from this cache instead of reading the device
    /// once per frame; the simulation thread owns it for the whole UI frame.
    #[allow(clippy::missing_const_for_thread_local)]
    static INSPECTOR_CACHE: RefCell<Option<InspectorCache>> =
        const { RefCell::new(None::<InspectorCache>) };
}

/// Coarse `CELL x CELL` block of pixel `(x, y)`.
fn inspector_cell(x: i32, y: i32) -> (u32, u32) {
    (
        x.div_euclid(CELL as i32) as u32,
        y.div_euclid(CELL as i32) as u32,
    )
}

/// Drops the cached probe (canvas changed, save loaded, simulation cleared).
fn clear_inspector_cache() {
    INSPECTOR_CACHE.with(|cache| *cache.borrow_mut() = None);
}

/// Wall report for the HUD inspector and the property window's sample row.
///
/// Mirrors `SimulationSample`'s wall fields (`Simulation::GetSample` in
/// `Editing.cpp` sets `WallType` from `bmap` and keeps the block's air state
/// next to it) and the way `GameView::Draw` renders them: the wall's name
/// followed by the block's pressure.
///
/// The reference stopped giving walls a temperature, so there is no wall
/// temperature to report; [`Self::air_temperature`] is the temperature of the
/// `CELL x CELL` block the wall occupies (`SimulationSample::AirTemperature`).
#[derive(Clone, Debug, PartialEq)]
pub struct WallSample {
    /// Wall id (`WL_*`).
    pub id: u8,
    /// Name from the wall table (`GameController::WallName`).
    pub name: String,
    /// Base colour of the wall type, `0x00RRGGBB`.
    pub colour: u32,
    /// Electrical state of the block (`wall_emap`); non-zero when the wall is
    /// carrying a charge.
    pub emap: u8,
    /// Air temperature of the block, in Kelvin.
    pub air_temperature: f32,
    /// Air pressure of the block.
    pub pressure: f32,
}

impl WallSample {
    /// One-line report, in the style of the HUD's sample readout.
    ///
    /// `temperature_scale` is a `TEMPSCALE_*`; the same scale the inspector uses
    /// for particle temperatures.
    pub fn summary(&self, temperature_scale: u32) -> String {
        let temperature = temperature_to_scale(self.air_temperature, temperature_scale);
        let mut text = t_fmt(
            "app.wall_summary",
            &[
                &self.name,
                &format!("{:06X}", self.colour & 0x00FF_FFFF),
                &format!("{:.1}", temperature),
                temperature_suffix(temperature_scale),
            ],
        );
        if self.emap != 0 {
            text.push_str(&t_fmt("app.wall_electrified", &[&self.emap.to_string()]));
        }
        text.push_str(&t_fmt(
            "app.wall_pressure",
            &[&format!("{:+.2}", self.pressure)],
        ));
        text
    }
}

impl AppState {
    /// Builds the application state around a freshly created CubeCL client.
    ///
    /// The initial engine uses the classic 612 x 384 canvas and the
    /// application starts on the main menu; [`Self::start_new_canvas`] or
    /// [`Self::load_from_file`] switches to the in-game screen.
    pub fn new(client: Client) -> Self {
        let config = SimulationConfig::new(DEFAULT_XRES, DEFAULT_YRES);
        Self::from_engine(SimulationEngine::new(client, config))
    }

    /// Builds the application state around an already-created simulation
    /// engine. The engine keeps its current canvas and particle contents.
    pub fn from_engine(engine: SimulationEngine) -> Self {
        let lua = LuaScriptingEngine::new()
            .unwrap_or_else(|err| panic!("failed to initialize the Lua scripting engine: {err}"));

        // Preferences drive the new-canvas defaults and the live renderer
        // configuration, so load them before the state is assembled.
        let prefs = Prefs::load();
        crate::i18n::set_language(prefs.language);
        let undo_limit =
            (prefs.undo_history_limit as usize).clamp(1, MAX_UNDO_HISTORY_LIMIT as usize);

        // The bound tools are preference keys; an unknown key (first start, or
        // a hand-edited file) falls back to dust and the eraser.
        let primary_tool =
            SelectedTool::from_key(&prefs.primary_tool).unwrap_or(SelectedTool::Element(PT_DUST));
        let secondary_tool =
            SelectedTool::from_key(&prefs.secondary_tool).unwrap_or(SelectedTool::Eraser);

        let mut new_canvas = NewCanvasDialog::new();
        new_canvas.edge_mode = prefs.edge_mode();
        new_canvas.gravity_mode = prefs.gravity_mode();
        new_canvas.ambient_heat = prefs.ambient_heat;
        new_canvas.refresh_saves_list();

        let property_window = PropertyWindow::new(&prefs);
        let game_mode = prefs.game_mode;

        let mut state = Self {
            screen: AppScreen::MainMenu,
            new_canvas,
            stamps: StampBrowser::default(),
            engine,
            undo: UndoHistory::new(undo_limit),
            sign_editor: SignEditor::default(),
            lua,
            mode: game_mode,
            voxel3d: None,
            voxel_renderer: None,
            voxel_edge_mode: EdgeMode::Void,
            voxel_camera: super::voxel::default_voxel_camera(),
            voxel_tool: VoxelTool::Element(PT_STNE),
            voxel_draw_mode: VoxelDrawMode::Brush,
            voxel_category: VoxelCategory::Powders,
            voxel_brush_radii: [4, 4, 4],
            voxel_brush_shape: BrushShape3d::Ellipsoid,
            voxel_brush_locks: [false; 3],
            voxel_box_half: [4, 4, 4],
            voxel_box_element: PT_STNE,
            voxel_undo: Voxel3dUndoHistory::default(),
            voxel_active_bricks: None,
            voxel_stats_at: Instant::now(),
            voxel_last_pick: None,
            voxel_hover_pick: None,
            voxel_hover_pointer: None,
            voxel_hover_at: Instant::now(),
            voxel_stroke_start: None,
            voxel_stroke_last: None,
            voxel_stroke_mode: None,
            voxel_stroke_snapshot: Default::default(),
            voxel_stroke_bricks: std::collections::HashSet::new(),
            voxel_stroke_active: false,
            voxel_save_name_input: t("voxel3d.default_save_name").to_string(),
            voxel_saves: Vec::new(),
            voxel_viewport_size: None,
            prefs,
            options_open: false,
            render_window_open: false,
            colour_picker: ColourPicker::default(),
            property_window,
            last_sim_step_instant: Instant::now(),
            sim_step_debt: 0.0,
            undo_limit,
            primary_tool,
            secondary_tool,
            brush_shape: BrushShape::Circle,
            brush_rx: 5,
            brush_ry: 5,
            replace_mode: false,
            active_category: MenuCategory::Powders,
            show_lua_console: false,
            lua_input: String::new(),
            save_name_input: t("app.default_save_name").to_string(),
            hover_pos: None,
            drag_start: None,
            last_cursor_sim: None,
            held_sim_tool: None,
            zoom: 1.0,
            move_view: false,
            zoom_cursor_fixed: false,
            select_mode: crate::app_state::selection::SelectMode::None,
            select_start: None,
            select_end: None,
            clipboard: None,
            placement: None,
            fps: 0.0,
            last_frame_instant: Instant::now(),
            frame_times: VecDeque::with_capacity(FPS_SAMPLE_WINDOW),
            tps: 0.0,
            step_instants: VecDeque::new(),
            status_message: String::new(),
            status_is_error: false,
            exit_requested: false,
            modals: ModalStack::default(),
            console: ConsoleState::default(),
            modal_result: None,
        };
        state
            .lua
            .console_log
            .push(t("app.lua_console_ready").to_string());

        // Mirror the persisted settings onto the engine before the first frame.
        state.prefs.apply_to_config(&mut state.engine.bufs.config);
        state.seed_engine_caps();
        state.refresh_voxel_saves();
        crate::app_state::tool::sync_gravity_zones(&mut state);
        clear_inspector_cache();
        state
    }

    /// Starts a blank simulation at the requested size, applying the
    /// new-canvas dialog's edge, gravity and ambient-heat settings.
    ///
    /// Returns the final canvas dimensions after alignment to the cell grid
    /// (multiples of `CELL`, minimum 64 x 64).
    pub fn start_new_canvas(&mut self, xres: u32, yres: u32) -> (u32, u32) {
        // Starting a powder canvas always leaves the 3D sandbox; the 3D world
        // stays resident and can be resumed from the title screen.
        self.mode = GameMode::Powder2D;
        self.prefs.game_mode = GameMode::Powder2D;
        let aligned = SimulationConfig::new(xres, yres);

        // Settings first: `resize` reseeds the air field from
        // `config.ambient_air_temp` and uploads the whole canvas. Applying them
        // afterwards (and then clearing) meant a second full reset plus a
        // second full upload, which made starting a big canvas visibly slow.
        {
            let config = &mut self.engine.bufs.config;
            // Start from the persisted defaults, then let the new-canvas
            // dialog override the three settings it exposes.
            self.prefs.apply_to_config(config);
            config.edge_mode = self.new_canvas.edge_mode;
            config.gravity_mode = self.new_canvas.gravity_mode;
            config.aheat_enable = self.new_canvas.ambient_heat;
            config.paused = false;
        }
        crate::app_state::tool::sync_gravity_zones(self);
        // `resize_canvas` already rebuilds every device buffer and drops the
        // particles, rigid bodies, air, walls and glow, so a separate
        // `clear()` only repeated the host reset and the upload.
        self.engine.resize_canvas(aligned.xres, aligned.yres);
        self.engine.frame_count = 0;
        self.last_sim_step_instant = Instant::now();
        self.sim_step_debt = 0.0;

        let (width, height) = (self.engine.bufs.config.xres, self.engine.bufs.config.yres);

        self.undo = UndoHistory::new(self.undo_limit);
        self.engine.set_signs(Vec::new());
        self.sign_editor = SignEditor::default();
        self.screen = AppScreen::InGame;
        self.hover_pos = None;
        self.drag_start = None;
        self.last_cursor_sim = None;
        self.zoom = 1.0;
        self.save_name_input = t("app.default_save_name").to_string();
        self.fps = 0.0;
        self.frame_times.clear();
        self.last_frame_instant = Instant::now();
        self.tps = 0.0;
        self.step_instants.clear();
        clear_inspector_cache();

        self.new_canvas.status_msg = Some(t_fmt(
            "app.status_new_simulation",
            &[&width.to_string(), &height.to_string()],
        ));
        self.new_canvas.save_load_error = None;
        (width, height)
    }

    /// Starts a simulation using the size currently selected in the
    /// new-canvas dialog. Returns the final dimensions so the caller can
    /// resize the simulation texture.
    pub fn start_new_simulation(&mut self) -> Option<(u32, u32)> {
        let (width, height) = self.new_canvas.resolved_size();
        let dims = self.start_new_canvas(width, height);
        self.set_status(
            t_fmt(
                "app.status_new_simulation_started",
                &[&dims.0.to_string(), &dims.1.to_string()],
            ),
            false,
        );
        Some(dims)
    }

    /// Records a short user-facing status message for the HUD.
    pub fn set_status(&mut self, message: impl Into<String>, is_error: bool) {
        self.status_message = message.into();
        self.status_is_error = is_error;
    }

    /// Saves the current simulation as `<name>.tptrs` under [`SAVES_DIR`].
    ///
    /// The returned string is a human-readable status message suitable for
    /// the HUD; errors are returned as user-facing strings as well.
    pub fn save_current(&mut self, name: &str) -> Result<String, String> {
        let stem = normalize_save_stem(name)?;
        let saves_dir = Path::new(SAVES_DIR);
        std::fs::create_dir_all(saves_dir)
            .map_err(|err| t_fmt("app.error_create_saves_dir", &[SAVES_DIR, &err.to_string()]))?;

        let path = saves_dir.join(format!("{stem}.{SAVE_EXTENSION}"));
        // `GameController.cpp:1241` (`OpenLocalSaveWindow`) passes
        // `GetIncludePressure()` into `Simulation::Save`, so the preference
        // also gates the on-disk payload, not just copy/stamp.
        let data =
            SaveData::capture_from_with_pressure(&mut self.engine, self.prefs.include_pressure);
        data.save_to_path(&path)?;

        let save_target = format!("{stem}.{SAVE_EXTENSION}");
        let msg = t_fmt("app.status_saved", &[&save_target]);
        self.save_name_input = stem.clone();
        self.new_canvas.status_msg = Some(msg.clone());
        self.new_canvas.save_load_error = None;
        self.new_canvas.refresh_saves_list();
        self.set_status(msg.clone(), false);
        Ok(msg)
    }

    /// Saves using the current contents of [`Self::save_name_input`], updating
    /// the HUD status line on both success and failure.
    pub fn save_current_from_input(&mut self) -> Result<String, String> {
        let name = self.save_name_input.clone();
        match self.save_current(&name) {
            Ok(message) => {
                self.set_status(message.clone(), false);
                Ok(message)
            }
            Err(err) => {
                self.set_status(err.clone(), true);
                self.new_canvas.status_msg = Some(err.clone());
                self.new_canvas.save_load_error = Some(err.clone());
                Err(err)
            }
        }
    }

    /// Loads a `.tptrs` save into the engine and switches to the in-game
    /// screen. Returns the canvas dimensions stored in the save so the caller
    /// can resize the simulation texture.
    pub fn load_from_file(&mut self, path: &Path) -> Result<(u32, u32), String> {
        let data = SaveData::load_from_path(path)?;

        // `GameController::LoadSaveFile` / `OpenSearch`
        // (`GameController.cpp:1224-1225,1339`) push a `HistorySnapshot`
        // before the loaded save replaces the current simulation, so Ctrl+Z
        // returns to the pre-load canvas instead of losing it. Only the
        // engine state is replaced: the history itself is kept.
        self.push_undo_snapshot();
        data.restore_into(&mut self.engine);
        self.mode = GameMode::Powder2D;
        self.prefs.game_mode = GameMode::Powder2D;
        self.sign_editor = SignEditor::default();
        self.engine.frame_count = 0;
        self.last_sim_step_instant = Instant::now();
        self.sim_step_debt = 0.0;
        let (width, height) = self.canvas_size();

        self.screen = AppScreen::InGame;
        self.hover_pos = None;
        self.drag_start = None;
        self.last_cursor_sim = None;
        self.zoom = 1.0;
        self.fps = 0.0;
        self.frame_times.clear();
        self.last_frame_instant = Instant::now();
        self.tps = 0.0;
        self.step_instants.clear();
        clear_inspector_cache();

        self.new_canvas.edge_mode = data.config.edge_mode;
        self.new_canvas.gravity_mode = data.config.gravity_mode;
        self.new_canvas.ambient_heat = data.config.aheat_enable;
        let path_str = path.display().to_string();
        let msg = t_fmt("app.status_loaded", &[&path_str]);
        self.new_canvas.status_msg = Some(msg.clone());
        self.new_canvas.save_load_error = None;
        self.save_name_input = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("untitled")
            .to_string();
        self.set_status(msg, false);

        Ok((width, height))
    }

    /// Loads a save file for the main-menu browser, reporting failures through
    /// the HUD status line instead of propagating them.
    pub fn load_save_file(&mut self, path: &Path) -> Option<(u32, u32)> {
        match self.load_from_file(path) {
            Ok(dims) => Some(dims),
            Err(err) => {
                self.set_status(err.clone(), true);
                self.new_canvas.status_msg = Some(err.clone());
                self.new_canvas.save_load_error = Some(err);
                None
            }
        }
    }

    /// Advances the simulation by as many steps as the configured simulation
    /// framerate allows and refreshes the logical-TPS estimate.
    ///
    /// The wall-clock time since the previous call accumulates in
    /// [`Self::sim_step_debt`] and the remainder carries over, so the step rate
    /// matches the live cap ([`SimulationEngine::fps_cap`]) regardless of how
    /// often the event loop runs.
    /// A plain "one step per call, when the interval has elapsed" gate
    /// quantises to `loop rate / ceil(interval * loop rate)`: a 180 Hz loop
    /// with a 60 TPS target lands on 4 iterations per step, i.e. 45 TPS.
    ///
    /// With [`FpsLimit::Uncapped`] one step runs per call, matching TPT's
    /// "advance with the frame". At most [`MAX_CATCHUP_STEPS`] steps run per
    /// call, so a long stall never triggers an unbounded burst.
    pub fn tick_frame(&mut self) {
        // Pruned every call, so a paused or capped simulation decays to 0 TPS
        // instead of freezing at its last value.
        self.sample_tps();
        self.refresh_voxel_stats();

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_sim_step_instant).as_secs_f32();
        self.last_sim_step_instant = now;

        // Lua's `tpt.fpsCap` writes the engine mirror; fold a change back
        // into `Prefs` (which persists it) even while the simulation is paused.
        self.persist_script_caps();

        if self.is_paused() {
            // `GameView::OnTick` (`GameView.cpp:1739-1768`) runs the held tool
            // pass on every game tick and has no pause check, so a held HEAT,
            // COOL, AIR or VACUUM keeps acting on a frozen canvas in the
            // reference. Only the simulation step is paused; the tools write
            // the simulation state directly. Pace it at the same game tick as
            // the running path instead of the (much higher) present rate.
            let limit = FpsLimit::from_cap_value(self.engine.fps_cap);
            match limit.interval_secs() {
                Some(interval) => {
                    let steps = steps_due(&mut self.sim_step_debt, elapsed, interval);
                    for _ in 0..steps {
                        self.repeat_held_sim_tool();
                    }
                }
                None => self.repeat_held_sim_tool(),
            }
            // A paused canvas still has to show host edits: the reference tools
            // and brushes write the live `Simulation`, so its renderer picks
            // them up without a tick.
            self.engine.flush_host_edits();
            return;
        }

        // `2` means uncapped (TPT), and the engine mirror is the live value
        // shared with the Options window through `Prefs`.
        let limit = FpsLimit::from_cap_value(self.engine.fps_cap);
        let Some(interval) = limit.interval_secs() else {
            self.repeat_held_sim_tool();
            self.step_simulation_once();
            self.step_instants.push_back(Instant::now());
            return;
        };

        let steps = steps_due(&mut self.sim_step_debt, elapsed, interval);
        for _ in 0..steps {
            self.repeat_held_sim_tool();
            self.step_simulation_once();
        }
        if steps > 0 {
            let stamp = Instant::now();
            for _ in 0..steps {
                self.step_instants.push_back(stamp);
            }
        }
    }

    /// Re-applies the held simulation tool, once per tick.
    ///
    /// `GameView::OnTick` (`GameView.cpp:1754-1758`) draws from `lastPoint` to
    /// `currentPoint` on every tick while the button is down, which is what
    /// makes the continuous tools (heat, cool, air, vacuum, wind, mix,
    /// cyclone) act while the cursor stands still. A single press-stamp of
    /// `HEAT` is `+2 K` and of `AIR` `+0.05` pressure, so without the repeat
    /// the tools look dead.
    fn repeat_held_sim_tool(&mut self) {
        if self.mode != GameMode::Powder2D || self.screen != AppScreen::InGame {
            return;
        }
        let Some((is_secondary, position)) = self.held_sim_tool else {
            return;
        };
        crate::brushes::apply_tool_stroke(
            self,
            position,
            position,
            is_secondary,
            false,
            false,
            false,
        );
    }

    /// Records a presented frame and refreshes the rolling FPS estimate.
    ///
    /// Called only when a frame is actually rendered and presented, so `fps`
    /// reports the present rate rather than the event-loop iteration rate.
    pub fn sample_presented_frame(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame_instant).as_secs_f32();
        self.last_frame_instant = now;

        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        // Ignore pathological stalls (window drags, breakpoints) so a single
        // long frame does not poison the rolling average.
        let sample = dt.min(1.0);
        self.frame_times.push_back(sample);
        while self.frame_times.len() > FPS_SAMPLE_WINDOW {
            self.frame_times.pop_front();
        }
        let total: f32 = self.frame_times.iter().sum();
        self.fps = if total > 0.0 {
            self.frame_times.len() as f32 / total
        } else {
            0.0
        };
    }

    /// Recomputes [`Self::tps`] from the simulation steps taken in the last
    /// second, dropping the ones that fell out of the window.
    fn sample_tps(&mut self) {
        let now = Instant::now();
        while self
            .step_instants
            .front()
            .is_some_and(|t| now.duration_since(*t).as_secs_f32() > 1.0)
        {
            self.step_instants.pop_front();
        }
        let Some(oldest) = self.step_instants.front().copied() else {
            self.tps = 0.0;
            return;
        };
        // `len - 1` steps span the interval between the oldest and the newest;
        // the floor keeps a fresh window from spiking on two close samples.
        let span = now.duration_since(oldest).as_secs_f32().clamp(0.2, 1.0);
        self.tps = (self.step_instants.len().saturating_sub(1)) as f32 / span;
    }

    /// Runs one simulation step plus the Lua tick callbacks.
    ///
    /// The interactive loop goes through [`Self::tick_frame`], which paces the
    /// call against the configured TPS. Smoke and bench harnesses that want a
    /// fixed number of frames call this directly.
    pub fn step_simulation_once(&mut self) {
        if self.mode == GameMode::Sandbox3D {
            // The voxel world carries its own GPU tick; Lua ticks are a 2D
            // feature and do not run here.
            self.ensure_voxel_engine().step();
            return;
        }
        self.engine.step();
        self.lua.fire_tick(&mut self.engine);
    }

    /// Whether the simulation is currently paused.
    pub fn is_paused(&self) -> bool {
        match self.mode {
            GameMode::Sandbox3D => self
                .voxel3d
                .as_ref()
                .map(|engine| engine.config.paused)
                .unwrap_or(false),
            GameMode::Powder2D => self.engine.bufs.config.paused,
        }
    }

    /// Sets the pause flag on the active simulation.
    pub fn set_paused(&mut self, paused: bool) {
        match self.mode {
            GameMode::Sandbox3D => {
                if let Some(engine) = self.voxel3d.as_mut() {
                    engine.config.paused = paused;
                }
            }
            GameMode::Powder2D => self.engine.bufs.config.paused = paused,
        }
        // Treat unpausing as a fresh step interval instead of a backlog.
        self.last_sim_step_instant = Instant::now();
        self.sim_step_debt = 0.0;
    }

    /// Toggles pause and returns the new pause state.
    pub fn toggle_pause(&mut self) -> bool {
        let paused = !self.is_paused();
        self.set_paused(paused);
        paused
    }

    /// Steps exactly one simulation frame regardless of the pause flag.
    /// Used by the HUD "Step Frame" button / `F` hotkey.
    pub fn step_single_frame(&mut self) {
        if self.mode == GameMode::Sandbox3D {
            if let Some(engine) = self.voxel3d.as_mut() {
                let was_paused = engine.config.paused;
                if was_paused {
                    engine.config.paused = false;
                }
                engine.step();
                if was_paused {
                    engine.config.paused = true;
                }
            }
            return;
        }
        // `SimulationEngine::step` is a no-op while paused, so temporarily
        // clear the flag to advance exactly one frame and restore it after.
        let was_paused = self.engine.bufs.config.paused;
        if was_paused {
            self.engine.bufs.config.paused = false;
        }
        self.engine.step();
        if was_paused {
            self.engine.bufs.config.paused = true;
        }
        self.lua.fire_tick(&mut self.engine);
    }

    /// Alias for [`Self::step_single_frame`] used by the HUD.
    pub fn step_frame(&mut self) {
        self.step_single_frame();
    }

    /// Pushes the current simulation onto the undo stack.
    ///
    /// `SaveData::capture_from` flushes pending host-side brush edits before
    /// reading the GPU state back, so this is safe immediately after a stroke.
    ///
    /// The read-back also clears the host dirty flags and the per-slot upload
    /// list, so the stroke that follows the snapshot cannot trigger a stale
    /// whole-pool upload: it uploads only the slots it creates or frees.
    pub fn push_undo_snapshot(&mut self) {
        // The 3D sandbox snapshots the bricks a brush touches, not a whole
        // simulation snapshot; `begin_voxel_stroke` owns that path.
        if self.mode == GameMode::Sandbox3D {
            return;
        }
        self.undo.push_snapshot(&mut self.engine);
    }

    /// Clears all particles, walls, air and rigid bodies while keeping the
    /// current canvas size and configuration. The pre-clear state is pushed
    /// onto the undo stack.
    pub fn clear_sim(&mut self) {
        if self.mode == GameMode::Sandbox3D {
            self.ensure_voxel_engine().reset();
            self.voxel_undo.clear();
            self.voxel_last_pick = None;
            self.voxel_active_bricks = Some(0);
            self.set_status(t("app.status_canvas_cleared"), false);
            return;
        }
        self.push_undo_snapshot();
        self.engine.clear();
        // `Simulation::clear_sim` drops the sign list with the scene.
        self.engine.set_signs(Vec::new());
        self.close_sign_editor();
        self.set_status(t("app.status_canvas_cleared"), false);
    }

    /// Restores the previous undo snapshot. Returns whether anything changed.
    pub fn do_undo(&mut self) -> bool {
        if self.mode == GameMode::Sandbox3D {
            return self.undo_voxel();
        }
        let changed = self.undo.undo(&mut self.engine);
        if changed {
            self.close_sign_editor();
            self.set_status(t("app.status_undo"), false);
        } else {
            self.set_status(t("app.status_nothing_to_undo"), false);
        }
        changed
    }

    /// Reapplies the most recently undone snapshot. Returns whether anything
    /// changed.
    pub fn do_redo(&mut self) -> bool {
        if self.mode == GameMode::Sandbox3D {
            return self.redo_voxel();
        }
        let changed = self.undo.redo(&mut self.engine);
        if changed {
            self.close_sign_editor();
            self.set_status(t("app.status_redo"), false);
        } else {
            self.set_status(t("app.status_nothing_to_redo"), false);
        }
        changed
    }

    /// Rescans the saves directory for the main-menu load browser.
    pub fn refresh_saved_files(&mut self) {
        self.new_canvas.refresh_saves_list();
        self.refresh_voxel_saves();
    }

    /// Applies the current [`Prefs`] to the live engine configuration, the
    /// rendering cap and the undo history, then writes `prefs.json`.
    ///
    /// Window-level preferences (resizable, fullscreen, blurry scaling, ...)
    /// are persisted only: the winit window lives in `main.rs` and applies
    /// those on startup.
    pub fn apply_prefs(&mut self) {
        self.prefs.apply_to_config(&mut self.engine.bufs.config);
        self.seed_engine_caps();
        self.prefs.game_mode = self.mode;
        self.prefs.primary_tool = self.primary_tool.to_key();
        self.prefs.secondary_tool = self.secondary_tool.to_key();

        let limit =
            (self.prefs.undo_history_limit as usize).clamp(1, MAX_UNDO_HISTORY_LIMIT as usize);
        if limit != self.undo_limit {
            // `UndoHistory` has no setter, so rebuild it. Changing the limit
            // intentionally discards the current snapshots.
            self.undo = UndoHistory::new(limit);
            self.undo_limit = limit;
        }

        // Keep the new-canvas dialog in sync with the persisted defaults.
        self.new_canvas.edge_mode = self.prefs.edge_mode();
        self.new_canvas.gravity_mode = self.prefs.gravity_mode();
        self.new_canvas.ambient_heat = self.prefs.ambient_heat;

        self.apply_voxel_prefs();
        self.prefs.save();
    }

    /// Opens the quit confirmation prompt once, mirroring `Engine::ConfirmExit`
    /// (`Engine.cpp:57-68`): while `confirmingExit` is set, repeated requests
    /// (a second close click, another Ctrl+Q) do not stack another prompt.
    /// Opens the Options window, first mirroring the live configuration into
    /// the preference fields so the controls show the current values.
    pub fn open_options(&mut self) {
        self.prefs.capture_from_config(&self.engine.bufs.config);
        self.options_open = true;
    }

    /// Opens the Render settings window, first mirroring the live renderer
    /// configuration into the preference fields.
    pub fn open_render_window(&mut self) {
        self.prefs.capture_from_config(&self.engine.bufs.config);
        self.render_window_open = true;
    }

    /// Rendering cap in frames per second, read from the live engine mirror.
    ///
    /// `0` means "follow the display"; the port has no display-derived mode, so
    /// it maps to the 240 fps ceiling the render loop enforces. Lua's
    /// `tpt.drawCap` writes the same mirror.
    pub fn draw_fps(&self) -> u32 {
        match self.engine.draw_cap {
            0 => MAX_SANE_DRAW_FPS,
            fps => fps,
        }
    }

    /// Copies the persisted caps into the engine mirror Lua reads.
    fn seed_engine_caps(&mut self) {
        self.engine.fps_cap = self.prefs.fps_limit.to_cap_value();
        self.engine.draw_cap = self.prefs.draw_fps;
        self.engine.caps_dirty = false;
    }

    /// Folds `tpt.fpsCap` / `tpt.drawCap` changes into [`Prefs`] and persists
    /// them.
    ///
    /// The Lua bridge can only reach the engine, but the Options window and
    /// `prefs.json` live in [`Prefs`]; this keeps the preference canonical so a
    /// script-set cap survives a restart. Called once per tick and on exit.
    pub fn persist_script_caps(&mut self) {
        if !self.engine.caps_dirty {
            return;
        }
        self.engine.caps_dirty = false;
        let fps = self.engine.fps_cap;
        let next_fps = FpsLimit::from_cap_value(fps);
        // `0` = follow the display; the port has no display-derived mode, so
        // persist the 240 fps ceiling it actually renders at.
        let next_draw = match self.engine.draw_cap {
            0 => MAX_SANE_DRAW_FPS,
            fps => fps.clamp(MIN_SANE_DRAW_FPS, MAX_SANE_DRAW_FPS),
        };
        let changed = next_fps != self.prefs.fps_limit || next_draw != self.prefs.draw_fps;
        self.prefs.fps_limit = next_fps;
        self.prefs.draw_fps = next_draw;
        // Normalise the mirror so `tpt.fpsCap()`/`tpt.drawCap()` report the
        // effective value, but keep `0` ("display") as the live draw value.
        self.engine.fps_cap = self.prefs.fps_limit.to_cap_value();
        if self.engine.draw_cap != 0 {
            self.engine.draw_cap = self.prefs.draw_fps;
        }
        // A tick handler that sets the same cap every tick must not rewrite
        // `prefs.json` every tick.
        if changed {
            self.prefs.save();
        }
    }

    /// Number of live particles in the simulation.
    pub fn active_particles(&self) -> usize {
        // `count_active_particles` walks the whole pool shadow, which is up to
        // `MAX_PARTICLE_CAPACITY` entries; the HUD asks for this every frame.
        // `active_count` is maintained incrementally and refreshed from the
        // device on the same schedule as the other HUD counters.
        self.engine.bufs.particle_count()
    }

    /// Number of active rigid bodies in the simulation.
    pub fn rigid_body_count(&self) -> usize {
        self.engine
            .bufs
            .rigid_bodies
            .iter()
            .filter(|body| body.active)
            .count()
    }

    /// Returns the particle occupying the given simulation pixel, if any.
    ///
    /// The HUD runs every frame, so the device probe is cached: it refreshes at
    /// most every 150 ms, and at most every 40 ms while the cursor crosses
    /// cells. A blocking probe per frame stalled the render loop.
    pub fn particle_at(&self, x: i32, y: i32) -> Option<GpuParticle> {
        self.probe_at(x, y).and_then(|probe| probe.particle)
    }

    /// Cached pixel probe (particle, wall, electrical state and pressure).
    ///
    /// Out-of-canvas coordinates return `None` without touching the device.
    pub fn probe_at(&self, x: i32, y: i32) -> Option<PixelProbe> {
        let cell = inspector_cell(x, y);
        INSPECTOR_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(cached) = cache.as_ref() {
                let age = cached.sampled_at.elapsed();
                let same_pixel = cached.x == x && cached.y == y;
                let same_cell = cached.cell == cell;
                let reuse = age < INSPECTOR_PROBE_MAX_AGE
                    && (same_pixel || (same_cell && age < INSPECTOR_PROBE_MIN_INTERVAL));
                if reuse {
                    return cached.probe.clone();
                }
            }
            let probe = self.engine.bufs.probe_pixel(x, y);
            *cache = Some(InspectorCache {
                x,
                y,
                cell,
                sampled_at: Instant::now(),
                probe: probe.clone(),
            });
            probe
        })
    }

    /// Air pressure of the cell containing pixel `(x, y)`, from the same cached
    /// probe the inspector uses (`0` outside the canvas).
    pub fn cell_pressure_at(&self, x: i32, y: i32) -> f32 {
        if let Some(probe) = self.probe_at(x, y) {
            return probe.pressure;
        }
        let config = &self.engine.bufs.config;
        if x < 0 || y < 0 || x as u32 >= config.xres || y as u32 >= config.yres {
            return 0.0;
        }
        let index =
            (y as usize / CELL as usize) * config.xcells() as usize + (x as usize / CELL as usize);
        self.engine.bufs.air_pv.get(index).copied().unwrap_or(0.0)
    }

    /// Returns the wall id occupying the given simulation pixel, if any.
    pub fn wall_at(&self, x: i32, y: i32) -> Option<u8> {
        self.engine.bufs.wall_at(x, y)
    }

    /// Wall occupying `(x, y)` together with the air state of its block.
    ///
    /// The wall counterpart of [`Self::particle_at`]: the sample tool reports
    /// walls this way (`SimulationSample::WallType`), and the property window
    /// shows the report when a sample lands on a wall instead of a particle.
    ///
    /// Read from the host shadow (`wall_bmap`, `wall_emap`, `air_hv`, `air_pv`),
    /// the same source [`Self::cell_pressure_at`] falls back to, so the HUD can
    /// call it every frame without a device read-back.
    pub fn wall_sample_at(&self, x: i32, y: i32) -> Option<WallSample> {
        let id = self.wall_at(x, y)?;
        let bufs = &self.engine.bufs;
        let cell = bufs.cell_index(x, y)?;
        let (name, colour) = self
            .engine
            .walls
            .iter()
            .find(|wall| wall.id == id)
            .map(|wall| (wall.name.clone(), wall.colour))
            .unwrap_or_else(|| (t("app.wall_fallback").to_string(), 0x00A0_A0A0));
        Some(WallSample {
            id,
            name,
            colour,
            emap: (bufs.wall_emap.get(cell).copied().unwrap_or(0) & 0xFF) as u8,
            air_temperature: bufs
                .air_hv
                .get(cell)
                .copied()
                .unwrap_or(bufs.config.ambient_air_temp),
            pressure: bufs.air_pv.get(cell).copied().unwrap_or(0.0),
        })
    }

    /// Resolves the effective heat display limits (min, max) in Kelvin.
    /// If explicit, returns the explicit values. If "auto", calculates the min/max
    /// of active particles inside the configured auto area.
    pub fn resolve_heat_display_limits(&self) -> (f32, f32) {
        let cfg = &self.engine.bufs.config;
        let mut min = cfg.heat_display_limit_min;
        let mut max = cfg.heat_display_limit_max;
        if min.is_none() || max.is_none() {
            let ax = cfg.heat_display_auto_x;
            let ay = cfg.heat_display_auto_y;
            let aw = cfg.heat_display_auto_w;
            let ah = cfg.heat_display_auto_h;
            let mut auto_min = 9999.0f32;
            let mut auto_max = 0.0f32;
            let mut found = false;
            let bufs = &self.engine.bufs;
            // Live slots are sparse, so the scan bound is the slot watermark,
            // not the live count.
            let active = bufs.used_slot_bound();
            for i in 0..active.min(bufs.part_ty.len()) {
                if bufs.part_ty[i] != 0 {
                    let px = bufs.part_x[i] as i32;
                    let py = bufs.part_y[i] as i32;
                    if px >= ax && px < ax + aw && py >= ay && py < ay + ah {
                        let temp = bufs.part_temp[i];
                        if temp < auto_min {
                            auto_min = temp;
                        }
                        if temp > auto_max {
                            auto_max = temp;
                        }
                        found = true;
                    }
                }
            }
            if !found {
                auto_min = 0.0;
                auto_max = 9999.0;
            }
            if min.is_none() {
                min = Some(auto_min);
            }
            if max.is_none() {
                max = Some(auto_max);
            }
        }
        let eff_min = min.unwrap_or(0.0).clamp(0.0, 9999.0);
        let eff_max = max.unwrap_or(9999.0).clamp(0.0, 9999.0).max(eff_min + 1.0);
        (eff_min, eff_max)
    }

    /// Current canvas dimensions in simulation pixels.
    pub fn canvas_size(&self) -> (u32, u32) {
        (self.engine.bufs.config.xres, self.engine.bufs.config.yres)
    }

    /// Ports `GameController::AdjustZoomSize`: the scope is 5..=64 pixels wide
    /// and the magnification keeps the window near 256 pixels
    /// (`ZFACTOR = 256 / size`, never below 3).
    pub fn adjust_zoom_size(&mut self, delta: i32, logarithmic: bool) {
        let cfg = &mut self.engine.bufs.config;
        let current = cfg.zoom_scope_size.clamp(5, 64);
        let step = if logarithmic {
            (current / 10).max(1)
        } else {
            1
        };
        let size = (current + delta.signum() * step).clamp(5, 64);
        cfg.zoom_scope_size = size;
        cfg.zoom_factor = (256 / size).max(3);
    }

    /// Ports `GameController::SetZoomPosition`: the scope is centred on the
    /// cursor and clamped inside the canvas, and the magnification window sits
    /// in the opposite half of the canvas so it never covers the cursor.
    pub fn set_zoom_position(&mut self, x: i32, y: i32) {
        let cfg = &mut self.engine.bufs.config;
        let size = cfg.zoom_scope_size.clamp(5, 64);
        let factor = cfg.zoom_factor.max(1);
        let half = size / 2;
        let max_x = (cfg.xres as i32 - size).max(0);
        let max_y = (cfg.yres as i32 - size).max(0);
        cfg.zoom_scope_x = (x - half).clamp(0, max_x);
        cfg.zoom_scope_y = (y - half).clamp(0, max_y);
        cfg.zoom_window_x = if x < cfg.xres as i32 / 2 {
            cfg.xres as i32 - size * factor
        } else {
            0
        }
        .max(0);
        cfg.zoom_window_y = 0;
    }

    /// Returns the tool bound to the primary or secondary mouse button.
    pub fn selected_tool(&self, secondary: bool) -> SelectedTool {
        if secondary {
            self.secondary_tool
        } else {
            self.primary_tool
        }
    }
}

/// Validates and normalizes a user-provided save name to a bare file stem.
pub(super) fn normalize_save_stem(name: &str) -> Result<String, String> {
    let trimmed = name.trim();
    let suffix = format!(".{SAVE_EXTENSION}");
    let stem = if trimmed.len() >= suffix.len()
        && trimmed
            .get(trimmed.len() - suffix.len()..)
            .is_some_and(|tail| tail.eq_ignore_ascii_case(&suffix))
    {
        &trimmed[..trimmed.len() - suffix.len()]
    } else {
        trimmed
    }
    .trim();

    if stem.is_empty() {
        return Err(t("app.error_save_name_empty").to_string());
    }
    if stem == "." || stem == ".." {
        return Err(t("app.error_save_name_invalid").to_string());
    }
    if stem.chars().any(|ch| {
        matches!(ch, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || ch.is_control()
    }) {
        return Err(t("app.error_save_name_reserved_chars").to_string());
    }
    if stem.chars().count() > 96 {
        return Err(t("app.error_save_name_too_long").to_string());
    }
    Ok(stem.to_string())
}

/// Adds `elapsed` seconds of wall clock to `debt` and returns how many whole
/// step intervals it covers, consuming them.
///
/// The remainder stays in `debt`, so the average step rate equals
/// `1 / interval` even when the event-loop period is not a multiple of it. A
/// plain "one step per call once the interval has elapsed" gate quantises to
/// `loop rate / ceil(interval * loop rate)`: at a 180 Hz loop with a 60 TPS
/// target, three calls only accumulate 16.6 ms of a 16.67 ms interval, so every
/// step waits for the fourth call and the simulation runs at 45 TPS. Carrying
/// the remainder removes that quantisation, and the [`MAX_CATCHUP_STEPS`] clamp
/// keeps a stall from turning into an unbounded burst.
fn steps_due(debt: &mut f32, elapsed: f32, interval: f32) -> u32 {
    let elapsed = if elapsed.is_finite() {
        elapsed.max(0.0)
    } else {
        0.0
    };
    *debt = (*debt + elapsed).min(interval * MAX_CATCHUP_STEPS as f32);
    let mut steps = 0;
    while *debt >= interval && steps < MAX_CATCHUP_STEPS {
        *debt -= interval;
        steps += 1;
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacing_holds_the_configured_tps_at_a_finer_loop_rate() {
        let interval = 1.0 / 60.0;
        let mut debt = 0.0;
        let mut steps = 0;
        // Three seconds of a 180 Hz event loop. The old one-step-per-call gate
        // produced 135 steps (45 TPS) for this loop.
        for _ in 0..540 {
            steps += steps_due(&mut debt, 1.0 / 180.0, interval);
        }
        assert!(
            (steps as i32 - 180).abs() <= 1,
            "expected ~180 steps for 3 s at 60 TPS, got {steps}"
        );
    }

    #[test]
    fn high_tps_runs_multiple_steps_per_call() {
        let interval = 1.0 / 240.0;
        let mut debt = 0.0;
        assert_eq!(steps_due(&mut debt, 1.0 / 60.0, interval), 4);
    }

    #[test]
    fn a_stall_never_runs_more_than_the_catch_up_bound() {
        let interval = 1.0 / 60.0;
        let mut debt = 0.0;
        assert_eq!(steps_due(&mut debt, 5.0, interval), MAX_CATCHUP_STEPS);
        assert_eq!(steps_due(&mut debt, 0.0, interval), 0);
    }
}
