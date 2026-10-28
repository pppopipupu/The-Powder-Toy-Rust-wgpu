//! Host <-> Lua bridge shared by every Lua-facing callback.
//!
//! Lua callbacks are `'static` closures owned by the Lua state, so they cannot
//! borrow the simulation directly. Every callback therefore holds a [`Bridge`]
//! and reaches the engine through the non-owning pointer the host installs for
//! the duration of a Lua call. `mlua`'s [`Lua`](mlua::Lua) state is `!Send`, so
//! the bridge is shared on a single thread with `Rc`/`RefCell`; touching it
//! re-entrantly is reported as an error instead of deadlocking on (or panicking
//! inside) the `RefCell`.

use std::cell::{RefCell, RefMut};
use std::rc::Rc;

use mlua::RegistryKey;
use tpt_core::commands::SimCommand;
use tpt_sim::SimulationEngine;

use crate::LuaScriptingEngine;
use crate::api::evt::LuaEvent;

/// Shared handle to the bridge state, cloned into every Lua callback.
pub(crate) type Bridge = Rc<RefCell<LuaBridgeState>>;

/// Mutable borrow of the bridge state handed out by [`borrow_bridge`].
pub(crate) type BridgeRef<'a> = RefMut<'a, LuaBridgeState>;

/// Error message used when Lua code tries to reach the simulation outside of a
/// host-driven Lua call.
const NO_ENGINE: &str = "the simulation is not accessible outside of a running script";

/// Error message used when a Lua call re-enters the bridge it already holds.
const REENTRANT: &str = "re-entrant Lua execution is not supported";

/// State shared between the host and every Lua callback.
///
/// The bridge is the only place that knows how to reach the simulation; see
/// the crate-level docs for the lifetime rules of `LuaBridgeState::engine`.
#[derive(Default)]
pub(crate) struct LuaBridgeState {
    /// Non-owning pointer to the engine, installed by the host for the
    /// duration of a Lua call and cleared before that call returns.
    engine: Option<*mut SimulationEngine>,
    /// [`SimCommand`]s queued by Lua and awaiting dispatch to the engine.
    pub(crate) pending_commands: Vec<SimCommand>,
    /// Console lines produced during the current call.
    pub(crate) pending_log: Vec<String>,
    /// Callbacks registered during the current call, moved into the engine's
    /// handler lists when the call finishes.
    pub(crate) pending_registrations: Vec<(LuaEvent, RegistryKey)>,
    /// Element definitions were modified.
    pub(crate) elements_dirty: bool,
    /// Particle pool / air grid / wall grid data was modified.
    pub(crate) particles_dirty: bool,
}

impl LuaBridgeState {
    /// Installs `engine` for the duration of a Lua call.
    pub(crate) fn attach(&mut self, engine: &mut SimulationEngine) -> Result<(), mlua::Error> {
        if self.engine.is_some() {
            return Err(mlua::Error::runtime(REENTRANT));
        }
        self.engine = Some(engine as *mut SimulationEngine);
        Ok(())
    }

    /// Removes the engine pointer installed by [`LuaBridgeState::attach`].
    fn detach(&mut self) {
        self.engine = None;
    }

    /// Whether an engine pointer is currently attached.
    #[allow(dead_code)]
    pub(crate) fn is_attached(&self) -> bool {
        self.engine.is_some()
    }

    /// Runs `f` with exclusive access to the attached engine.
    ///
    /// # Safety contract
    ///
    /// The pointer installed by [`LuaBridgeState::attach`] is derived from the
    /// `&mut SimulationEngine` owned by the host entry point. The host never
    /// touches that borrow while the pointer is installed, `with_engine` takes
    /// `&mut self` (so only one `&mut SimulationEngine` exists at a time), and
    /// the pointer is cleared before the host borrow is used again.
    pub(crate) fn with_engine<R>(
        &mut self,
        f: impl FnOnce(&mut SimulationEngine) -> R,
    ) -> Result<R, mlua::Error> {
        let Some(pointer) = self.engine else {
            return Err(mlua::Error::runtime(NO_ENGINE));
        };
        // SAFETY: see the doc comment above.
        let engine = unsafe { &mut *pointer };
        Ok(f(engine))
    }

    /// Queues `command` and dispatches it to the attached engine.
    ///
    /// Dispatching immediately keeps the Lua-visible state consistent: a
    /// script can inspect the effect of its own writes in the next statement.
    /// When no engine is attached the command stays queued for the host.
    pub(crate) fn dispatch(&mut self, command: SimCommand) -> Result<(), mlua::Error> {
        self.pending_commands.push(command);
        self.drain_commands()
    }

    /// Applies every queued command to the attached engine.
    fn drain_commands(&mut self) -> Result<(), mlua::Error> {
        if self.pending_commands.is_empty() {
            return Ok(());
        }
        let Some(pointer) = self.engine else {
            return Ok(());
        };
        // SAFETY: identical invariant to `with_engine`; `pointer` is a copy, so
        // draining `pending_commands` does not alias the engine borrow.
        let engine = unsafe { &mut *pointer };
        // Commands such as `DecorateAt` run their read-modify-write on the
        // device. Upload the slots the script wrote first, otherwise a particle
        // created earlier in the same script is still only in the host shadow
        // and the kernel cannot see it. This transfers just the dirty slots,
        // never a whole-pool copy.
        if engine.bufs.host_dirty {
            engine.bufs.sync_host_to_gpu();
        }
        for command in self.pending_commands.drain(..) {
            engine.apply_command(command);
        }
        Ok(())
    }

    /// Appends a line to the console buffer of the running call.
    pub(crate) fn push_log(&mut self, line: String) {
        self.pending_log.push(line);
    }
}

/// Clears the engine pointer if a host -> Lua call unwinds before the normal
/// flush runs.
pub(crate) struct AttachGuard {
    pub(crate) bridge: Bridge,
}

impl Drop for AttachGuard {
    fn drop(&mut self) {
        // No borrow can be outstanding here: the callback frame that held it
        // is dropped before this guard unwinds past the host entry point.
        if let Ok(mut state) = self.bridge.try_borrow_mut() {
            state.detach();
        }
    }
}

/// Borrows the bridge state mutably, reporting a re-entrant call as a Lua error
/// instead of panicking inside the `RefCell`.
pub(crate) fn borrow_bridge(bridge: &Bridge) -> Result<BridgeRef<'_>, mlua::Error> {
    bridge
        .try_borrow_mut()
        .map_err(|_| mlua::Error::runtime(REENTRANT))
}

impl LuaScriptingEngine {
    /// Applies everything the callbacks left in the bridge to `engine` and
    /// detaches the bridge. Returns the console lines produced by the call.
    pub(crate) fn flush_bridge(&mut self, engine: &mut SimulationEngine) -> Vec<String> {
        let pending = borrow_bridge(&self.bridge).ok().map(|mut state| {
            // Detach before the host touches its `&mut SimulationEngine`
            // again; commands are normally dispatched synchronously by the
            // callback that queued them, so `leftovers` is empty in practice.
            state.detach();
            (
                std::mem::take(&mut state.pending_log),
                std::mem::take(&mut state.pending_registrations),
                std::mem::take(&mut state.elements_dirty),
                std::mem::take(&mut state.particles_dirty),
                std::mem::take(&mut state.pending_commands),
            )
        });
        let Some((logs, registrations, elements_dirty, particles_dirty, leftovers)) = pending
        else {
            let line = format!("[lua] {REENTRANT}");
            self.push_console_line(line.clone());
            return vec![line];
        };

        for command in leftovers {
            engine.apply_command(command);
        }
        for (event, key) in registrations {
            match event {
                LuaEvent::Tick => self.tick_handlers.push(key),
                LuaEvent::MouseDown => self.mouse_handlers.push(key),
                LuaEvent::KeyPress => self.key_handlers.push(key),
            }
        }
        if elements_dirty {
            engine.registry.rebuild_can_move();
        }
        // Upload only what the script actually touched. The old
        // `mark_all_dirty` forced a whole-pool + grid upload for a single
        // `sim.partProperty` write; the buffers already track per-slot and
        // per-grid dirt, so `sync_host_to_gpu` transfers exactly that (and does
        // nothing at all when the script only read). `host_dirty` also catches
        // the grid setters (`sim.pressure`, walls), which mark the buffer flags
        // directly instead of the bridge's coarse ones.
        if particles_dirty || engine.bufs.host_dirty {
            engine.bufs.sync_host_to_gpu();
        }

        let returned = logs.clone();
        self.extend_console(logs);
        returned
    }
}
