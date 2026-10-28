//! Lua 5.4 scripting for The Powder Toy (Rust + wgpu + CubeCL rewrite).
//!
//! The crate exposes a TPT-compatible scripting surface on top of
//! [`tpt_sim::SimulationEngine`]:
//!
//! | Lua namespace | Backing Rust data |
//! |---------------|-------------------|
//! | `sim` / `simulation` | [`tpt_sim::SimBuffers`] (particles, air, walls, rigid bodies) |
//! | `elem` / `elements` | [`tpt_core::elements::ElementRegistry`] |
//! | `ren` / `renderer` | [`tpt_core::particle::SimulationConfig`] display/colour mode |
//! | `evt` / `event` | [`LuaScriptingEngine`] callback lists |
//! | `tpt` | misc helpers (`print`, pause, step hooks, create/delete) |
//!
//! # Engine access model
//!
//! Lua callbacks are `'static` closures owned by the Lua state, so they cannot
//! borrow the simulation directly. Instead every callback shares a
//! `LuaBridgeState` behind a shared `Rc<RefCell<..>>`. The Lua state is
//! `!Send`, so the scripting layer never leaves the thread that owns it and the
//! bridge needs no locking. For the duration of a host-driven Lua call
//! ([`LuaScriptingEngine::exec_script`],
//! [`LuaScriptingEngine::fire_tick`],
//! [`LuaScriptingEngine::fire_mouse_down`],
//! [`LuaScriptingEngine::fire_key_press`]) the bridge holds a non-owning
//! pointer to the engine that the host handed over. Callbacks therefore:
//!
//! * **read** live particle/air/wall/element state through the bridge,
//! * **write** it through the same bridge (so a script observes its own
//!   writes), and
//! * **queue** [`SimCommand`](tpt_core::commands::SimCommand)s, which the
//!   bridge dispatches immediately (or hands back to the host when no engine is
//!   attached).
//!
//! When the call returns, the host flushes the bridge: queued commands are
//! applied, newly registered callbacks are moved into the engine's handler
//! lists, modified element definitions rebuild the movement table, and dirty
//! particle/air/wall data is re-uploaded to the GPU.
//!
//! The device is authoritative, so host getters pull exactly what they address
//! before reading: one particle slot (`SimBuffers::read_particle_slot`), one
//! pixel rectangle (`SimBuffers::read_map_rect`) or one coarse grid
//! (`sync_air_gpu_to_host` / `sync_gravity_gpu_to_host`). There is deliberately
//! no whole-simulation read-back before a call, so a callback that touches
//! nothing transfers nothing and a per-tick handler costs only the data it
//! actually asks for.
//!
//! # Coordinates
//!
//! Particle functions (`sim.partCreate`, `sim.partKill`, `sim.partID`,
//! `sim.partPosition`) use **pixel** coordinates, exactly like the classic
//! TPT API. The Eulerian air and wall accessors (`sim.pressure`,
//! `sim.velocity`, `sim.ambientHeat`, `sim.wallProperty`) use **cell**
//! coordinates: one cell spans `tpt_core::constants::CELL` (4) pixels, so
//! valid ranges are `0..xres/4` and `0..yres/4`. Element ids are indices into
//! [`tpt_core::elements::ElementRegistry`] (`elem.DEFAULT_PT_*` constants).

pub mod api;
mod bridge;

use std::rc::Rc;

use mlua::{Function, IntoLuaMulti, Lua, RegistryKey, Value};
use tpt_sim::SimulationEngine;

use crate::api::evt::LuaEvent;
use crate::api::{install_globals, is_truthy};
use crate::bridge::{AttachGuard, Bridge, borrow_bridge};

/// Maximum number of lines kept in [`LuaScriptingEngine::console_log`].
///
/// Older lines are dropped once the limit is exceeded, which keeps a runaway
/// `print` loop from growing the log without bound.
const MAX_CONSOLE_LINES: usize = 4096;

/// Lua scripting engine: owns the Lua state, the console log, the registered
/// event callbacks and the bridge used by all Lua-facing functions.
pub struct LuaScriptingEngine {
    /// The Lua state scripts are executed in.
    pub lua: Lua,
    /// Console output produced by `print`, `tpt.log` and script errors.
    pub console_log: Vec<String>,
    /// Callbacks registered for `evt.TICK` / `tpt.register_step`.
    pub tick_handlers: Vec<RegistryKey>,
    /// Callbacks registered for `evt.MOUSEDOWN`.
    pub mouse_handlers: Vec<RegistryKey>,
    /// Callbacks registered for `evt.KEYPRESS`.
    pub key_handlers: Vec<RegistryKey>,
    bridge: Bridge,
}

impl LuaScriptingEngine {
    /// Creates a Lua state with the full TPT scripting API installed.
    pub fn new() -> Result<Self, String> {
        let lua = Lua::new();
        let bridge = Bridge::default();
        install_globals(&lua, &bridge)
            .map_err(|error| format!("failed to initialise the Lua API: {error}"))?;
        Ok(Self {
            lua,
            console_log: Vec::new(),
            tick_handlers: Vec::new(),
            mouse_handlers: Vec::new(),
            key_handlers: Vec::new(),
            bridge,
        })
    }

    /// Runs `code` and returns the console output it produced.
    ///
    /// On failure the returned error contains both the script error and any
    /// output produced before the failure, so the console can show context.
    pub fn exec_script(
        &mut self,
        code: &str,
        engine: &mut SimulationEngine,
    ) -> Result<String, String> {
        let (result, log) =
            self.run_attached(engine, |lua| lua.load(code).set_name("=lua_script").exec());
        let output = log.join("\n");
        match result {
            Ok(()) => Ok(output),
            Err(error) => {
                let message = error.to_string();
                Err(if output.is_empty() {
                    message
                } else {
                    format!("{output}\n{message}")
                })
            }
        }
    }

    /// Fires every callback registered for `evt.TICK`.
    pub fn fire_tick(&mut self, engine: &mut SimulationEngine) {
        if self.tick_handlers.is_empty() {
            return;
        }
        self.dispatch_event(engine, LuaEvent::Tick, ());
    }

    /// Fires every callback registered for `evt.MOUSEDOWN`.
    ///
    /// Returns `true` when a callback returned a truthy value, which the app
    /// uses to suppress the default brush behaviour for that click.
    pub fn fire_mouse_down(
        &mut self,
        x: i32,
        y: i32,
        button: u32,
        engine: &mut SimulationEngine,
    ) -> bool {
        if self.mouse_handlers.is_empty() {
            return false;
        }
        self.dispatch_event(engine, LuaEvent::MouseDown, (x, y, button))
    }

    /// Fires every callback registered for `evt.KEYPRESS`.
    ///
    /// Returns `true` when a callback returned a truthy value, which the app
    /// uses to suppress the default keyboard shortcut for that key.
    pub fn fire_key_press(&mut self, key: &str, engine: &mut SimulationEngine) -> bool {
        if self.key_handlers.is_empty() {
            return false;
        }
        self.dispatch_event(engine, LuaEvent::KeyPress, key.to_owned())
    }

    /// Runs `f` with the engine attached to the bridge and flushes the bridge
    /// afterwards. Returns the closure result together with the console lines
    /// produced during the call.
    fn run_attached<R>(
        &mut self,
        engine: &mut SimulationEngine,
        f: impl FnOnce(&Lua) -> mlua::Result<R>,
    ) -> (mlua::Result<R>, Vec<String>) {
        // No blanket read-back here: every getter pulls exactly the device data
        // it needs (one particle slot, one pixel rectangle, one grid), so a
        // script that touches nothing transfers nothing. The old whole-pool
        // read-back copied ~32 MB on every run and every event.
        if let Err(error) = borrow_bridge(&self.bridge).and_then(|mut state| state.attach(engine)) {
            return (Err(error), Vec::new());
        }
        let guard = AttachGuard {
            bridge: Rc::clone(&self.bridge),
        };
        let result = f(&self.lua);
        let log = self.flush_bridge(engine);
        drop(guard);
        (result, log)
    }

    /// Invokes every handler registered for `event` with `args`.
    ///
    /// Handlers are temporarily moved out of the engine so a callback may
    /// register further callbacks without deadlocking or mutating the list
    /// being iterated.
    fn dispatch_event<A>(&mut self, engine: &mut SimulationEngine, event: LuaEvent, args: A) -> bool
    where
        A: IntoLuaMulti + Clone,
    {
        let mut handlers = self.take_handlers(event);
        let mut handled = false;
        let mut errors: Vec<String> = Vec::new();
        let (result, _log) = self.run_attached(engine, |lua| -> mlua::Result<()> {
            handlers.retain(|key| {
                if let Ok(func) = lua.registry_value::<Function>(key) {
                    !crate::api::evt::is_handler_unregistered(lua, event, &func)
                } else {
                    false
                }
            });
            for key in &handlers {
                let function: Function = match lua.registry_value(key) {
                    Ok(function) => function,
                    Err(error) => {
                        errors.push(error.to_string());
                        continue;
                    }
                };
                if crate::api::evt::is_handler_unregistered(lua, event, &function) {
                    continue;
                }
                match function.call::<Value>(args.clone()) {
                    Ok(value) => {
                        if is_truthy(&value) {
                            handled = true;
                        }
                    }
                    Err(error) => errors.push(error.to_string()),
                }
            }
            handlers.retain(|key| {
                if let Ok(func) = lua.registry_value::<Function>(key) {
                    !crate::api::evt::is_handler_unregistered(lua, event, &func)
                } else {
                    false
                }
            });
            Ok(())
        });
        self.set_handlers(event, handlers);
        for error in errors {
            self.push_console_line(format!("[lua] {error}"));
        }
        if let Err(error) = result {
            self.push_console_line(format!("[lua] {error}"));
        }
        handled
    }

    fn take_handlers(&mut self, event: LuaEvent) -> Vec<RegistryKey> {
        match event {
            LuaEvent::Tick => std::mem::take(&mut self.tick_handlers),
            LuaEvent::MouseDown => std::mem::take(&mut self.mouse_handlers),
            LuaEvent::KeyPress => std::mem::take(&mut self.key_handlers),
        }
    }

    fn set_handlers(&mut self, event: LuaEvent, handlers: Vec<RegistryKey>) {
        let target = match event {
            LuaEvent::Tick => &mut self.tick_handlers,
            LuaEvent::MouseDown => &mut self.mouse_handlers,
            LuaEvent::KeyPress => &mut self.key_handlers,
        };
        // Handlers registered while the event was being fired were flushed
        // into `target` by `flush_bridge`; keep them after the handlers that
        // were already registered when the event started.
        let newly_registered = std::mem::take(target);
        target.extend(handlers);
        target.extend(newly_registered);
        target.retain(|key| {
            if let Ok(func) = self.lua.registry_value::<Function>(key) {
                !crate::api::evt::is_handler_unregistered(&self.lua, event, &func)
            } else {
                false
            }
        });
    }

    /// Appends a line to the console, trimming the oldest lines past the cap.
    pub(crate) fn push_console_line(&mut self, line: String) {
        self.console_log.push(line);
        self.trim_console();
    }

    /// Appends the console lines produced by a Lua call.
    pub(crate) fn extend_console(&mut self, lines: Vec<String>) {
        if lines.is_empty() {
            return;
        }
        self.console_log.extend(lines);
        self.trim_console();
    }

    fn trim_console(&mut self) {
        let excess = self.console_log.len().saturating_sub(MAX_CONSOLE_LINES);
        if excess > 0 {
            self.console_log.drain(..excess);
        }
    }
}
