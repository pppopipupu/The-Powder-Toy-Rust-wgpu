//! The `tpt` table (misc helpers).

use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use mlua::{Function, Lua, Table, Value};
use tpt_sim::SimulationEngine;

use crate::api::evt::LuaEvent;
use crate::api::interface::queue_message;
use crate::api::sim::allocate_particle;
use crate::bridge::{Bridge, borrow_bridge};

/// Directory (relative to the working directory) holding screenshots.
const SCREENSHOTS_DIR: &str = "screenshots";
/// Screenshot file stem. Mirrors `crates/tpt-app/src/screenshots.rs`.
const SCREENSHOT_PREFIX: &str = "screenshot_";
/// Screenshot extension (`TakeScreenshot(0, 0)` always takes the PNG branch,
/// `GameView.cpp:998-1004`).
const SCREENSHOT_EXTENSION: &str = "png";
/// Lua global drained by the application. Must match
/// `tpt_app::gpu_ctx::screenshots::RECORD_REQUEST_GLOBAL`; the two crates
/// cannot share the constant because `tpt-app` depends on `tpt-lua`.
const RECORD_REQUEST_GLOBAL: &str = "__tpt_record_request";
/// Separator between the `"start"` verb and an explicit output path. Must
/// match `tpt_app::gpu_ctx::screenshots::RECORD_REQUEST_SEPARATOR`.
const RECORD_REQUEST_SEPARATOR: char = '\u{1}';

/// Script manager URL (`LuaMisc.cpp:34`).
const SCRIPT_MANAGER_URL: &str = "https://starcatcher.us/scripts/main.lua";

/// The built-in compatibility chunk (`LuaMisc.cpp:175-180`).
///
/// The reference embeds `src/lua/luascripts/compat.lua` through the build's
/// data-file generator; this port keeps a verbatim copy next to the `tpt`
/// table so `tpt.compatChunk()` returns the same source.
const COMPAT_CHUNK: &str = include_str!("../compat.lua");

/// `CommandInterface::debugFlags` mirror.
///
/// TPT keeps the flags on `GameController` and the renderer reads them every
/// frame; this port stores them in Lua-side state so `tpt.debug()` round-trips
/// and the application can read [`LuaScriptingEngine::debug_flags`].
#[derive(Clone, Default)]
pub struct DebugFlags(Rc<std::cell::RefCell<i32>>);

impl DebugFlags {
    /// Returns the current debug flags (`LuaMisc.cpp:181-194`).
    pub fn get(&self) -> i32 {
        *self.0.borrow()
    }

    /// Replaces the debug flags.
    pub fn set(&self, flags: i32) {
        *self.0.borrow_mut() = flags;
    }
}

impl crate::LuaScriptingEngine {
    /// Returns the `tpt.debug` flag mirror of this Lua state.
    pub fn debug_flags(&self) -> DebugFlags {
        self.lua
            .app_data_ref::<DebugFlags>()
            .expect("the tpt table is installed with the Lua state")
            .clone()
    }
}

const DEBUG_CONSTANTS: &[(&str, u32)] = &[
    ("DEBUG_PARTS", 1),
    ("DEBUG_ELEMENTPOP", 2),
    ("DEBUG_LINES", 4),
    ("DEBUG_PARTICLE", 8),
    ("DEBUG_SURFNORM", 16),
    ("DEBUG_SIMHUD", 32),
    ("DEBUG_RENHUD", 64),
    ("DEBUG_AIRVEL", 128),
    ("DEBUG_FRAMETIME", 256),
];

/// Builds the `tpt` table. `print` is reused for `tpt.log` so both share the
/// same console buffer.
pub(super) fn build_tpt_table(lua: &Lua, bridge: &Bridge, print: Function) -> mlua::Result<Table> {
    let tpt = lua.create_table()?;
    tpt.set("log", print)?;

    for (name, val) in DEBUG_CONSTANTS {
        tpt.set(*name, *val)?;
    }

    // tpt.version table
    let version = lua.create_table()?;
    version.set("major", 98)?;
    version.set("minor", 2)?;
    version.set("build", 360)?;
    version.set("upstreamMajor", 98)?;
    version.set("upstreamMinor", 2)?;
    version.set("upstreamBuild", 360)?;
    version.set("modid", 0)?;
    version.set("snapshot", false)?;
    version.set("beta", false)?;
    version.set("vcstag", "master")?;
    version.set("packagemode", "release")?;
    tpt.set("version", version)?;

    // tpt.getUserName() -> string (`LuaMisc.cpp:13-25`)
    //
    // `gameModel->GetUser()` is the signed-in TPT user; this port has no
    // account system, so the OS user name is the closest equivalent (the
    // reference returns "" when there is no user).
    tpt.set(
        "getUserName",
        lua.create_function(|_, ()| -> mlua::Result<String> {
            Ok(std::env::var("USERNAME")
                .or_else(|_| std::env::var("USER"))
                .unwrap_or_default())
        })?,
    )?;

    // tpt.installScriptManager([url]) (`LuaMisc.cpp:27-35`, `62-144`)
    //
    // Downloads the script manager, writes it as `autorun.lua` and runs it in
    // this Lua state (the reference's `Autorun`). The optional `url` is a port
    // extension so a local mirror can be used; the reference URL is HTTPS and
    // the local-only `http` subset has no TLS, so it reports the unreachable
    // error.
    tpt.set(
        "installScriptManager",
        lua.create_function(|lua, url: Option<String>| -> mlua::Result<()> {
            let url = url.unwrap_or_else(|| String::from(SCRIPT_MANAGER_URL));
            match crate::api::http::fetch_text(&url) {
                Ok(script) => {
                    if std::fs::write("autorun.lua", &script).is_err() {
                        let message = String::from("Failed to get script manager: Unable to write to autorun.lua");
                        return queue_message(lua, "Install script manager", &message, true);
                    }
                    if let Err(error) = lua.load(&script).set_name("autorun.lua").exec() {
                        let message = format!("Failed to run script manager: {error}");
                        return queue_message(lua, "Install script manager", &message, true);
                    }
                    queue_message(
                        lua,
                        "Install script manager",
                        "Script manager successfully installed",
                        false,
                    )
                }
                Err(error) => {
                    let message = format!("Failed to get script manager: {error}");
                    queue_message(lua, "Install script manager", &message, true)
                }
            }
        })?,
    )?;

    // tpt.screenshot([captureUI, fileType]) -> written path
    //
    // `LuaMisc::screenshot` (`LuaMisc.cpp:147-161`) calls
    // `GameController::TakeScreenshot` and returns the file name, or nothing
    // when writing failed. This port supports PNG captures of the simulation
    // canvas only: `captureUI` and the BMP/PPM file types have no equivalent
    // here, so asking for them raises instead of silently writing a different
    // image. `SimulationEngine::render_rgba` is the host equivalent of the
    // reference's `rendererFrame` (`GameView.cpp:952`), which is exactly what
    // `TakeScreenshot(0, 0)` encodes.
    tpt.set(
        "screenshot",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: (Option<i32>, Option<i32>)| -> mlua::Result<String> {
                let (capture_ui, file_type) = args;
                if capture_ui.unwrap_or(0) != 0 {
                    return Err(mlua::Error::runtime(
                        "tpt.screenshot cannot capture the UI in this port",
                    ));
                }
                match file_type.unwrap_or(0) {
                    0 => {}
                    1 | 2 => {
                        return Err(mlua::Error::runtime(
                            "tpt.screenshot only supports PNG (fileType 0) in this port",
                        ));
                    }
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "unknown screenshot file type {other}"
                        )));
                    }
                }

                let mut state = borrow_bridge(&bridge)?;
                state
                    .with_engine(write_screenshot)?
                    .map_err(mlua::Error::runtime)
            }
        })?,
    )?;

    // tpt.record("start" | "stop" | path)
    //
    // The reference's `record(bool)` (`LuaMisc.cpp:163-173`) toggles
    // `GameView`'s PPM recorder and returns the recording folder. This port
    // records rendered frames and assembles an APNG when recording stops:
    // `"start"` uses the lowest free `recordings/recording_<n>.apng`,
    // `"stop"` finishes that file, and any other string starts a recording to
    // the given path.
    //
    // Capturing needs the application's GPU read-back, so the Lua side only
    // queues the request in `RECORD_REQUEST_GLOBAL`; `AppState` drains it on
    // the next UI frame. `true` means the request was accepted, not that the
    // file exists yet.
    tpt.set(
        "record",
        lua.create_function(|lua, arg: Option<String>| -> mlua::Result<bool> {
            let Some(arg) = arg else {
                return Err(mlua::Error::runtime(
                    "tpt.record expects \"start\", \"stop\" or an output path",
                ));
            };
            let request = match arg.as_str() {
                "start" => String::from("start"),
                "stop" => String::from("stop"),
                "" => {
                    return Err(mlua::Error::runtime(
                        "tpt.record expects \"start\", \"stop\" or an output path",
                    ));
                }
                path => format!("start{RECORD_REQUEST_SEPARATOR}{path}"),
            };
            // Append instead of overwrite: a script may call
            // `tpt.record("start"); ... tpt.record("stop")` in one run, and
            // the application drains the whole queue on its next UI frame.
            let mut queued = match lua.globals().get::<Value>(RECORD_REQUEST_GLOBAL) {
                Ok(Value::String(text)) => text.to_string_lossy(),
                _ => String::new(),
            };
            queued.push_str(&request);
            queued.push('\n');
            lua.globals().set(RECORD_REQUEST_GLOBAL, queued)?;
            Ok(true)
        })?,
    )?;

    // tpt.debug([flags]) -> flags | nil (`LuaMisc.cpp:181-194`)
    //
    // TPT reads/writes `GameController::debugFlags`; this port mirrors the
    // value in Lua-side state that the application reads through
    // [`LuaScriptingEngine::debug_flags`].
    let debug_flags = DebugFlags::default();
    lua.set_app_data(debug_flags.clone());
    tpt.set(
        "debug",
        lua.create_function(move |_, flags: Option<i32>| -> mlua::Result<Value> {
            match flags {
                Some(flags) => {
                    debug_flags.set(flags);
                    Ok(Value::Nil)
                }
                None => Ok(Value::Integer(debug_flags.get() as i64)),
            }
        })?,
    )?;

    // tpt.fpsCap([cap]) -> cap | nil
    //
    // The live value lives on the engine (`SimulationEngine::fps_cap`); the
    // application owns the persisted preference, seeds this mirror at startup
    // and saves it back when `caps_dirty` is set. TPT's sentinel `2` means
    // "uncapped" (`LuaMisc.cpp:196`).
    tpt.set(
        "fpsCap",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, cap: Option<f64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| match cap {
                    Some(cap) => {
                        if cap < 2.0 {
                            return Err(mlua::Error::runtime("fps cap too small"));
                        }
                        engine.fps_cap = cap as f32;
                        engine.caps_dirty = true;
                        Ok(Value::Nil)
                    }
                    None => Ok(Value::Number(if engine.fps_cap <= 2.0 {
                        2.0
                    } else {
                        engine.fps_cap as f64
                    })),
                })?
            }
        })?,
    )?;

    // tpt.drawCap([cap]) -> cap | "display" | nil
    //
    // Like `fpsCap`, the live value is `SimulationEngine::draw_cap` (`0` =
    // follow the display); the application persists changes flagged by
    // `caps_dirty` and maps `0` to the port's 240 fps ceiling.
    tpt.set(
        "drawCap",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, cap: Option<Value>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                match cap {
                    None => {
                        let current = state.with_engine(|engine| engine.draw_cap)?;
                        if current == 0 {
                            Ok(Value::String(lua.create_string("display")?))
                        } else {
                            Ok(Value::Integer(current as i64))
                        }
                    }
                    Some(Value::String(text)) => {
                        if text.as_bytes() == b"display" {
                            state.with_engine(|engine| {
                                engine.draw_cap = 0;
                                engine.caps_dirty = true;
                            })?;
                            Ok(Value::Nil)
                        } else {
                            Err(mlua::Error::runtime("draw cap too small"))
                        }
                    }
                    Some(Value::Integer(fps)) if fps > 0 => {
                        let fps = fps.min(u32::MAX as i64) as u32;
                        state.with_engine(|engine| {
                            engine.draw_cap = fps;
                            engine.caps_dirty = true;
                        })?;
                        Ok(Value::Nil)
                    }
                    Some(Value::Number(fps)) if fps > 0.0 => {
                        let fps = fps.min(u32::MAX as f64) as u32;
                        state.with_engine(|engine| {
                            engine.draw_cap = fps;
                            engine.caps_dirty = true;
                        })?;
                        Ok(Value::Nil)
                    }
                    Some(_) => Err(mlua::Error::runtime("draw cap too small")),
                }
            }
        })?,
    )?;

    // tpt.compatChunk() -> the built-in compatibility script
    //
    // `LuaMisc.cpp:175-180` pushes the byte array generated from
    // `src/lua/luascripts/compat.lua`; the reference loads the same chunk at
    // Lua-state creation (`LuaScriptInterface.cpp:184-188`) so old scripts can
    // keep using the pre-rewrite names. This port ships the identical chunk as
    // `crates/tpt-lua/src/compat.lua` for callers that load it themselves.
    tpt.set(
        "compatChunk",
        lua.create_function(|lua, ()| -> mlua::Result<mlua::LuaString> {
            lua.create_string(COMPAT_CHUNK)
        })?,
    )?;

    // tpt.register_step(fn)
    tpt.set(
        "register_step",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, function: Function| -> mlua::Result<()> {
                let key = lua.create_registry_value(function)?;
                borrow_bridge(&bridge)?
                    .pending_registrations
                    .push((LuaEvent::Tick, key));
                Ok(())
            }
        })?,
    )?;

    // tpt.set_pause([paused]) -> bool
    tpt.set(
        "set_pause",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, paused: Option<bool>| -> mlua::Result<bool> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if let Some(paused) = paused {
                        engine.bufs.config.paused = paused;
                    }
                    engine.bufs.config.paused
                })
            }
        })?,
    )?;

    // tpt.create(x, y, ty)
    tpt.set(
        "create",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (x, y, ty): (f64, f64, u32)| -> mlua::Result<()> {
                let mut state = borrow_bridge(&bridge)?;
                let id = state
                    .with_engine(|engine| allocate_particle(engine, x as i32, y as i32, ty))?;
                if id.is_some() {
                    // `allocate_particle` / `kill_at` update the pixel-map word
                    // of the slot they touch, so only that slot is uploaded.
                    state.particles_dirty = true;
                }
                Ok(())
            }
        })?,
    )?;

    // tpt.delete(x, y)
    tpt.set(
        "delete",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (x, y): (f64, f64)| -> mlua::Result<()> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    crate::api::sim::refresh_pixel_occupant(engine, x as i32, y as i32);
                    engine.bufs.kill_at(x as i32, y as i32);
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    Ok(tpt)
}

/// Returns `dir/<SCREENSHOT_PREFIX><n>.<SCREENSHOT_EXTENSION>` for the lowest
/// free `n`, mirroring `GameView::TakeScreenshot`'s index rule
/// (`GameView.cpp:1449-1462`) and `crates/tpt-app/src/screenshots.rs`.
fn next_screenshot_path(dir: &Path) -> PathBuf {
    let mut index = 1u32;
    loop {
        let candidate = dir.join(format!("{SCREENSHOT_PREFIX}{index}.{SCREENSHOT_EXTENSION}"));
        if !candidate.exists() {
            return candidate;
        }
        index = index.saturating_add(1);
    }
}

/// Renders the simulation frame and writes it as a PNG, returning the path.
///
/// `GameView::TakeScreenshot(0, 0)` encodes `rendererFrame`
/// (`GameView.cpp:943-1005`). The interactive window normally renders on the
/// GPU and blits on the device (`GpuContext::encode_sim_blit`), but the Lua
/// entry point runs synchronously inside a script call, so it uses the host
/// render path (`SimulationEngine::render_rgba`) to produce the same RGBA
/// bytes without needing the application's `GpuContext`.
fn write_screenshot(engine: &mut SimulationEngine) -> Result<String, String> {
    let (width, height) = {
        let config = engine.config();
        (config.xres.max(1), config.yres.max(1))
    };
    let rgba = engine.render_rgba();

    let dir = Path::new(SCREENSHOTS_DIR);
    std::fs::create_dir_all(dir)
        .map_err(|err| format!("could not create '{}': {err}", dir.display()))?;
    let path = next_screenshot_path(dir);

    let file = std::fs::File::create(&path)
        .map_err(|err| format!("could not create '{}': {err}", path.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|err| format!("could not write PNG header: {err}"))?;
    writer
        .write_image_data(rgba)
        .map_err(|err| format!("could not write PNG data: {err}"))?;
    writer
        .finish()
        .map_err(|err| format!("could not finish PNG: {err}"))?;

    Ok(path.display().to_string())
}
