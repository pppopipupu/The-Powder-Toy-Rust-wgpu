//! The `ren` table (and the classic `renderer` alias).

use std::cell::RefCell;
use std::rc::Rc;

use mlua::{Lua, Table, Value};
use tpt_core::constants::*;

use crate::bridge::{Bridge, borrow_bridge};

const PMODE: u32 = 0x0000_0FFF;
const PMODE_NONE: u32 = 0x0000_0000;
const PMODE_FLAT: u32 = 0x0000_0001;
const PMODE_BLOB: u32 = 0x0000_0002;
const PMODE_BLUR: u32 = 0x0000_0004;
const PMODE_GLOW: u32 = 0x0000_0008;
const PMODE_SPARK: u32 = 0x0000_0010;
const PMODE_FLARE: u32 = 0x0000_0020;
const PMODE_LFLARE: u32 = 0x0000_0040;
const PMODE_ADD: u32 = 0x0000_0080;
const PMODE_BLEND: u32 = 0x0000_0100;
const PSPEC_STICKMAN: u32 = 0x0000_0200;

const OPTIONS: u32 = 0x0000_F000;
const NO_DECO: u32 = 0x0000_1000;
const DECO_FIRE: u32 = 0x0000_2000;

const FIREMODE: u32 = 0x00FF_0000;
const FIRE_ADD: u32 = 0x0001_0000;
const FIRE_BLEND: u32 = 0x0002_0000;
const FIRE_SPARK: u32 = 0x0004_0000;

const EFFECT: u32 = 0xFF00_0000;
const EFFECT_GRAVIN: u32 = 0x0100_0000;
const EFFECT_GRAVOUT: u32 = 0x0200_0000;
const EFFECT_LINES: u32 = 0x0400_0000;
const EFFECT_DBGLINES: u32 = 0x0800_0000;

const DISPLAY_AIR: u32 = DISPLAY_AIRC | DISPLAY_AIRP | DISPLAY_AIRV | DISPLAY_AIRH | DISPLAY_AIRW;
// `DISPLAY_EFFE` comes from `tpt_core::constants` (0x40, `ElementGraphics.h:56`);
// the local table used to shadow it with 0, so a script toggling the bit cleared
// every other display mode instead.

struct RenderPreset {
    render_mode: u32,
    display_mode: u32,
    colour_mode: u32,
}

const RENDER_PRESETS: &[RenderPreset] = &[
    RenderPreset {
        render_mode: RENDER_EFFE | RENDER_BASC,
        display_mode: DISPLAY_AIRC,
        colour_mode: 0,
    },
    RenderPreset {
        render_mode: RENDER_EFFE | RENDER_BASC,
        display_mode: DISPLAY_AIRV,
        colour_mode: 0,
    },
    RenderPreset {
        render_mode: RENDER_EFFE | RENDER_BASC,
        display_mode: DISPLAY_AIRP,
        colour_mode: 0,
    },
    RenderPreset {
        render_mode: RENDER_EFFE | RENDER_BASC,
        display_mode: DISPLAY_PERS,
        colour_mode: 0,
    },
    RenderPreset {
        render_mode: RENDER_FIRE | RENDER_SPRK | RENDER_EFFE | RENDER_BASC,
        display_mode: 0,
        colour_mode: 0,
    },
    RenderPreset {
        render_mode: RENDER_FIRE | RENDER_SPRK | RENDER_EFFE | RENDER_BLOB,
        display_mode: 0,
        colour_mode: 0,
    },
    RenderPreset {
        render_mode: RENDER_BASC,
        display_mode: DISPLAY_AIRH,
        colour_mode: COLOUR_HEAT,
    },
    RenderPreset {
        render_mode: RENDER_FIRE
            | RENDER_SPRK
            | RENDER_GLOW
            | RENDER_BLUR
            | RENDER_EFFE
            | RENDER_BASC,
        display_mode: DISPLAY_WARP,
        colour_mode: 0,
    },
    RenderPreset {
        render_mode: RENDER_BASC,
        display_mode: 0,
        colour_mode: 0,
    },
    RenderPreset {
        render_mode: RENDER_BASC,
        display_mode: 0,
        colour_mode: COLOUR_GRAD,
    },
    RenderPreset {
        render_mode: RENDER_BASC,
        display_mode: 0,
        colour_mode: COLOUR_LIFE,
    },
    RenderPreset {
        render_mode: RENDER_BASC,
        display_mode: DISPLAY_AIRH,
        colour_mode: 0,
    },
    RenderPreset {
        render_mode: RENDER_EFFE | RENDER_BASC,
        display_mode: DISPLAY_AIRW,
        colour_mode: 0,
    },
    RenderPreset {
        render_mode: RENDER_BASC,
        display_mode: 0,
        colour_mode: COLOUR_DEST,
    },
];

const RENDER_CONSTANTS: &[(&str, u32)] = &[
    ("PMODE", PMODE),
    ("PMODE_NONE", PMODE_NONE),
    ("PMODE_FLAT", PMODE_FLAT),
    ("PMODE_BLOB", PMODE_BLOB),
    ("PMODE_BLUR", PMODE_BLUR),
    ("PMODE_GLOW", PMODE_GLOW),
    ("PMODE_SPARK", PMODE_SPARK),
    ("PMODE_FLARE", PMODE_FLARE),
    ("PMODE_LFLARE", PMODE_LFLARE),
    ("PMODE_ADD", PMODE_ADD),
    ("PMODE_BLEND", PMODE_BLEND),
    ("PSPEC_STICKMAN", PSPEC_STICKMAN),
    ("OPTIONS", OPTIONS),
    ("NO_DECO", NO_DECO),
    ("DECO_FIRE", DECO_FIRE),
    ("FIREMODE", FIREMODE),
    ("FIRE_ADD", FIRE_ADD),
    ("FIRE_BLEND", FIRE_BLEND),
    ("FIRE_SPARK", FIRE_SPARK),
    ("EFFECT", EFFECT),
    ("EFFECT_GRAVIN", EFFECT_GRAVIN),
    ("EFFECT_GRAVOUT", EFFECT_GRAVOUT),
    ("EFFECT_LINES", EFFECT_LINES),
    ("EFFECT_DBGLINES", EFFECT_DBGLINES),
    ("RENDER_EFFE", RENDER_EFFE),
    ("RENDER_FIRE", RENDER_FIRE),
    ("RENDER_GLOW", RENDER_GLOW),
    ("RENDER_BLUR", RENDER_BLUR),
    ("RENDER_BLOB", RENDER_BLOB),
    ("RENDER_BASC", RENDER_BASC),
    ("RENDER_NONE", RENDER_NONE),
    ("COLOUR_HEAT", COLOUR_HEAT),
    ("COLOUR_LIFE", COLOUR_LIFE),
    ("COLOUR_GRAD", COLOUR_GRAD),
    ("COLOUR_BASC", COLOUR_BASC),
    ("COLOUR_DEST", COLOUR_DEST),
    ("COLOUR_DEFAULT", COLOUR_DEFAULT),
    ("DISPLAY_AIRC", DISPLAY_AIRC),
    ("DISPLAY_AIRP", DISPLAY_AIRP),
    ("DISPLAY_AIRV", DISPLAY_AIRV),
    ("DISPLAY_AIRH", DISPLAY_AIRH),
    ("DISPLAY_AIR", DISPLAY_AIR),
    ("DISPLAY_WARP", DISPLAY_WARP),
    ("DISPLAY_PERS", DISPLAY_PERS),
    ("DISPLAY_EFFE", DISPLAY_EFFE),
    ("DISPLAY_AIRW", DISPLAY_AIRW),
];

/// Lua-side renderer switches that have no GPU equivalent.
///
/// `ren.separateThread` used to select TPT's threaded CPU renderer
/// (`LuaRenderer.cpp:236-247`); `ren.depth3d` is a deprecated toggle
/// (`LuaRenderer.cpp:154-158`). This port stores both values so scripts keep
/// their state round-tripping instead of raising.
#[derive(Clone)]
pub struct RenderLuaState(Rc<RefCell<RenderLuaSwitches>>);

#[derive(Default)]
struct RenderLuaSwitches {
    /// `GameModel::threadedRendering`, seeded from
    /// `Renderer.SeparateThread` (`GameModel.cpp:115`).
    separate_thread: bool,
    /// Deprecated 3D toggle.
    depth3d: bool,
}

impl Default for RenderLuaState {
    fn default() -> Self {
        Self(Rc::new(RefCell::new(RenderLuaSwitches {
            separate_thread: true,
            depth3d: false,
        })))
    }
}

impl RenderLuaState {
    /// Returns `ren.separateThread()`.
    pub fn separate_thread(&self) -> bool {
        self.0.borrow().separate_thread
    }

    /// Sets `ren.separateThread(value)`.
    pub fn set_separate_thread(&self, value: bool) {
        self.0.borrow_mut().separate_thread = value;
    }

    /// Returns `ren.depth3d()`.
    pub fn depth3d(&self) -> bool {
        self.0.borrow().depth3d
    }

    /// Sets `ren.depth3d(value)`.
    pub fn set_depth3d(&self, value: bool) {
        self.0.borrow_mut().depth3d = value;
    }
}

impl crate::LuaScriptingEngine {
    /// Returns the `ren` switch mirror of this Lua state.
    pub fn render_lua_state(&self) -> RenderLuaState {
        self.lua
            .app_data_ref::<RenderLuaState>()
            .expect("the ren table is installed with the Lua state")
            .clone()
    }
}

pub(super) fn build_render_table(lua: &Lua, bridge: &Bridge) -> mlua::Result<Table> {
    let ren = lua.create_table()?;

    for (name, val) in RENDER_CONSTANTS {
        ren.set(*name, *val)?;
    }

    let render_state = RenderLuaState::default();
    lua.set_app_data(render_state.clone());

    // ren.renderMode([mode]) -> mode | nil
    ren.set(
        "renderMode",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, mode: Option<u32>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if let Some(mode) = mode {
                        engine.bufs.config.render_mode = mode;
                        Ok(Value::Nil)
                    } else {
                        Ok(Value::Integer(engine.bufs.config.render_mode as i64))
                    }
                })?
            }
        })?,
    )?;

    // ren.displayMode([mode]) -> mode | nil
    ren.set(
        "displayMode",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, mode: Option<u32>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if let Some(mode) = mode {
                        engine.bufs.config.display_mode = mode;
                        Ok(Value::Nil)
                    } else {
                        Ok(Value::Integer(engine.bufs.config.display_mode as i64))
                    }
                })?
            }
        })?,
    )?;

    // ren.colorMode([mode]) -> mode | nil
    ren.set(
        "colorMode",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, mode: Option<u32>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if let Some(mode) = mode {
                        engine.bufs.config.colour_mode = mode;
                        Ok(Value::Nil)
                    } else {
                        Ok(Value::Integer(engine.bufs.config.colour_mode as i64))
                    }
                })?
            }
        })?,
    )?;

    // ren.fireSize([intensity]) -> intensity | nil
    ren.set(
        "fireSize",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, intensity: Option<f64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if let Some(intensity) = intensity {
                        engine.bufs.config.fire_intensity = intensity as f32;
                        Ok(Value::Nil)
                    } else {
                        Ok(Value::Number(engine.bufs.config.fire_intensity as f64))
                    }
                })?
            }
        })?,
    )?;

    // ren.useDisplayPreset([cmode])
    ren.set(
        "useDisplayPreset",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, cmode: Option<i32>| -> mlua::Result<Value> {
                let mut cmode = cmode.unwrap_or(3);
                if (0..=10).contains(&cmode) {
                    cmode = (cmode + 1) % 11;
                }
                if cmode < 0 || (cmode as usize) >= RENDER_PRESETS.len() {
                    return Err(mlua::Error::runtime("Invalid display preset"));
                }
                let preset = &RENDER_PRESETS[cmode as usize];
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    engine.bufs.config.render_mode = preset.render_mode;
                    engine.bufs.config.display_mode = preset.display_mode;
                    engine.bufs.config.colour_mode = preset.colour_mode;
                    if cmode == 11 {
                        engine.bufs.config.heat_display_limit_min = None;
                        engine.bufs.config.heat_display_limit_max = None;
                    } else {
                        engine.bufs.config.heat_display_limit_min = Some(MIN_TEMP);
                        engine.bufs.config.heat_display_limit_max = Some(MAX_TEMP);
                    }
                })?;
                Ok(Value::Nil)
            }
        })?,
    )?;

    // ren.decorations([bool]) -> bool | nil
    ren.set(
        "decorations",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<Value>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    let b = value_to_bool(&val);
                    state.with_engine(|engine| {
                        engine.bufs.config.decorations = b;
                    })?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.decorations)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // ren.grid([val]) -> bool | nil
    ren.set(
        "grid",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<Value>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    state.with_engine(|engine| match val {
                        Value::Integer(i) => {
                            if i > 0 {
                                engine.bufs.config.grid = true;
                                engine.bufs.config.grid_size = i as i32;
                            } else {
                                engine.bufs.config.grid = false;
                                engine.bufs.config.grid_size = 0;
                            }
                        }
                        Value::Number(n) => {
                            let i = n as i32;
                            if i > 0 {
                                engine.bufs.config.grid = true;
                                engine.bufs.config.grid_size = i;
                            } else {
                                engine.bufs.config.grid = false;
                                engine.bufs.config.grid_size = 0;
                            }
                        }
                        Value::Boolean(b) => {
                            engine.bufs.config.grid = b;
                            if b && engine.bufs.config.grid_size <= 0 {
                                engine.bufs.config.grid_size = 1;
                            }
                        }
                        _ => {}
                    })?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.grid)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // ren.gridCheckerboard([bool]) -> bool | nil
    ren.set(
        "gridCheckerboard",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<Value>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    let b = value_to_bool(&val);
                    state.with_engine(|engine| {
                        engine.bufs.config.grid_checkerboard = b;
                    })?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.grid_checkerboard)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // ren.hud([bool]) -> bool | nil
    ren.set(
        "hud",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<Value>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    let b = value_to_bool(&val);
                    state.with_engine(|engine| {
                        engine.bufs.config.hud = b;
                    })?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.hud)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // ren.debugHud([bool]) -> bool | nil
    ren.set(
        "debugHud",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<Value>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    let b = value_to_bool(&val);
                    state.with_engine(|engine| {
                        engine.bufs.config.debug_hud = b;
                    })?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.debug_hud)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // ren.showBrush([bool]) -> bool | nil
    ren.set(
        "showBrush",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<Value>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    let b = value_to_bool(&val);
                    state.with_engine(|engine| {
                        engine.bufs.config.show_brush = b;
                    })?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.show_brush)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // ren.zoomEnabled([bool]) -> bool | nil
    ren.set(
        "zoomEnabled",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<Value>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    let b = value_to_bool(&val);
                    state.with_engine(|engine| {
                        engine.bufs.config.zoom_enabled = b;
                    })?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.zoom_enabled)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // ren.zoomWindow([x, y, factor]) -> x, y, factor, window_size | nil
    ren.set(
        "zoomWindow",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: mlua::MultiValue| -> mlua::Result<mlua::MultiValue> {
                let mut state = borrow_bridge(&bridge)?;
                if args.is_empty() {
                    let (x, y, factor, scope_size) = state.with_engine(|engine| {
                        (
                            engine.bufs.config.zoom_window_x,
                            engine.bufs.config.zoom_window_y,
                            engine.bufs.config.zoom_factor,
                            engine.bufs.config.zoom_scope_size,
                        )
                    })?;
                    Ok(mlua::MultiValue::from_vec(vec![
                        Value::Integer(x as i64),
                        Value::Integer(y as i64),
                        Value::Integer(factor as i64),
                        Value::Integer((scope_size * factor) as i64),
                    ]))
                } else {
                    let mut iter = args.into_iter();
                    let x = match iter.next() {
                        Some(Value::Integer(i)) => i as i32,
                        Some(Value::Number(n)) => n as i32,
                        _ => 0,
                    };
                    let y = match iter.next() {
                        Some(Value::Integer(i)) => i as i32,
                        Some(Value::Number(n)) => n as i32,
                        _ => 0,
                    };
                    let f = match iter.next() {
                        Some(Value::Integer(i)) => i as i32,
                        Some(Value::Number(n)) => n as i32,
                        _ => 0,
                    };
                    if f <= 0 {
                        return Err(mlua::Error::runtime("Zoom factor must be greater than 0"));
                    }
                    state.with_engine(|engine| {
                        let xres = engine.bufs.config.xres as i32;
                        let yres = engine.bufs.config.yres as i32;
                        let s = engine.bufs.config.zoom_scope_size;
                        if x < 0 || y < 0 || s * f + x > xres || s * f + y > yres {
                            return Err(mlua::Error::runtime("Zoom window outside of bounds"));
                        }
                        engine.bufs.config.zoom_window_x = x;
                        engine.bufs.config.zoom_window_y = y;
                        engine.bufs.config.zoom_factor = f;
                        Ok(())
                    })??;
                    Ok(mlua::MultiValue::new())
                }
            }
        })?,
    )?;

    // ren.zoomScope([x, y, size]) -> x, y, size | nil
    ren.set(
        "zoomScope",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: mlua::MultiValue| -> mlua::Result<mlua::MultiValue> {
                let mut state = borrow_bridge(&bridge)?;
                if args.is_empty() {
                    let (x, y, scope_size) = state.with_engine(|engine| {
                        (
                            engine.bufs.config.zoom_scope_x,
                            engine.bufs.config.zoom_scope_y,
                            engine.bufs.config.zoom_scope_size,
                        )
                    })?;
                    Ok(mlua::MultiValue::from_vec(vec![
                        Value::Integer(x as i64),
                        Value::Integer(y as i64),
                        Value::Integer(scope_size as i64),
                    ]))
                } else {
                    let mut iter = args.into_iter();
                    let x = match iter.next() {
                        Some(Value::Integer(i)) => i as i32,
                        Some(Value::Number(n)) => n as i32,
                        _ => 0,
                    };
                    let y = match iter.next() {
                        Some(Value::Integer(i)) => i as i32,
                        Some(Value::Number(n)) => n as i32,
                        _ => 0,
                    };
                    let s = match iter.next() {
                        Some(Value::Integer(i)) => i as i32,
                        Some(Value::Number(n)) => n as i32,
                        _ => 0,
                    };
                    if s <= 0 {
                        return Err(mlua::Error::runtime(
                            "Zoom scope size must be greater than 0",
                        ));
                    }
                    state.with_engine(|engine| {
                        let xres = engine.bufs.config.xres as i32;
                        let yres = engine.bufs.config.yres as i32;
                        let f = engine.bufs.config.zoom_factor;
                        let wx = engine.bufs.config.zoom_window_x;
                        let wy = engine.bufs.config.zoom_window_y;
                        let window_edge_right = f * s + wx;
                        let window_edge_bottom = f * s + wy;
                        if x < 0 || y < 0 || x + s > xres || y + s > yres {
                            return Err(mlua::Error::runtime("Zoom scope outside of bounds"));
                        }
                        if window_edge_right > xres || window_edge_bottom > yres {
                            return Err(mlua::Error::runtime("Zoom window outside of bounds"));
                        }
                        engine.bufs.config.zoom_scope_x = x;
                        engine.bufs.config.zoom_scope_y = y;
                        engine.bufs.config.zoom_scope_size = s;
                        Ok(())
                    })??;
                    Ok(mlua::MultiValue::new())
                }
            }
        })?,
    )?;

    // ren.heatDisplayLimits([min, max]) -> min, max | nil
    ren.set(
        "heatDisplayLimits",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, args: mlua::MultiValue| -> mlua::Result<mlua::MultiValue> {
                let mut state = borrow_bridge(&bridge)?;
                if args.is_empty() {
                    let (min, max) = state.with_engine(|engine| {
                        (
                            engine.bufs.config.heat_display_limit_min,
                            engine.bufs.config.heat_display_limit_max,
                        )
                    })?;
                    let to_val = |opt: Option<f32>| -> mlua::Result<Value> {
                        match opt {
                            Some(v) => Ok(Value::Number(v as f64)),
                            None => Ok(Value::String(lua.create_string("auto")?)),
                        }
                    };
                    Ok(mlua::MultiValue::from_vec(vec![to_val(min)?, to_val(max)?]))
                } else {
                    let mut iter = args.into_iter();
                    let parse_limit = |val: Option<Value>| -> Option<f32> {
                        match val {
                            Some(Value::String(s)) => {
                                if s.to_str().map(|str| str == "auto").unwrap_or(false) {
                                    None
                                } else {
                                    s.to_str().ok().and_then(|str| str.parse::<f32>().ok())
                                }
                            }
                            Some(Value::Number(n)) => Some(n as f32),
                            Some(Value::Integer(i)) => Some(i as f32),
                            _ => None,
                        }
                    };
                    let min = parse_limit(iter.next());
                    let max = parse_limit(iter.next());
                    state.with_engine(|engine| {
                        engine.bufs.config.heat_display_limit_min = min;
                        engine.bufs.config.heat_display_limit_max = max;
                    })?;
                    Ok(mlua::MultiValue::new())
                }
            }
        })?,
    )?;

    // ren.heatDisplayAutoArea([x, y, w, h]) -> x, y, w, h | nil
    ren.set(
        "heatDisplayAutoArea",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: mlua::MultiValue| -> mlua::Result<mlua::MultiValue> {
                let mut state = borrow_bridge(&bridge)?;
                if args.is_empty() {
                    let (x, y, w, h) = state.with_engine(|engine| {
                        (
                            engine.bufs.config.heat_display_auto_x,
                            engine.bufs.config.heat_display_auto_y,
                            engine.bufs.config.heat_display_auto_w,
                            engine.bufs.config.heat_display_auto_h,
                        )
                    })?;
                    Ok(mlua::MultiValue::from_vec(vec![
                        Value::Integer(x as i64),
                        Value::Integer(y as i64),
                        Value::Integer(w as i64),
                        Value::Integer(h as i64),
                    ]))
                } else {
                    let mut iter = args.into_iter();
                    let get_int = |v: Option<Value>| -> i32 {
                        match v {
                            Some(Value::Integer(i)) => i as i32,
                            Some(Value::Number(n)) => n as i32,
                            _ => 0,
                        }
                    };
                    let x = get_int(iter.next());
                    let y = get_int(iter.next());
                    let w = get_int(iter.next());
                    let h = get_int(iter.next());
                    state.with_engine(|engine| {
                        engine.bufs.config.heat_display_auto_x = x;
                        engine.bufs.config.heat_display_auto_y = y;
                        engine.bufs.config.heat_display_auto_w = w;
                        engine.bufs.config.heat_display_auto_h = h;
                    })?;
                    Ok(mlua::MultiValue::new())
                }
            }
        })?,
    )?;

    // ren.separateThread([threaded]) -> threaded | nil
    //
    // `LuaRenderer.cpp:236-247`: getter/setter for
    // `GameModel::threadedRendering`. GPU compute runs on the device queue, so
    // the value only mirrors what a script asked for.
    ren.set(
        "separateThread",
        lua.create_function({
            let render_state = render_state.clone();
            move |_, threaded: Option<bool>| -> mlua::Result<Value> {
                match threaded {
                    Some(threaded) => {
                        render_state.set_separate_thread(threaded);
                        Ok(Value::Nil)
                    }
                    None => Ok(Value::Boolean(render_state.separate_thread())),
                }
            }
        })?,
    )?;

    // ren.depth3d([enabled]) -> enabled | nil
    //
    // The reference always raises (`LuaRenderer.cpp:154-158`); the port keeps
    // accepting and storing the toggle as the assignment requires.
    ren.set(
        "depth3d",
        lua.create_function({
            let render_state = render_state.clone();
            move |_, enabled: Option<bool>| -> mlua::Result<Value> {
                match enabled {
                    Some(enabled) => {
                        render_state.set_depth3d(enabled);
                        Ok(Value::Nil)
                    }
                    None => Ok(Value::Boolean(render_state.depth3d())),
                }
            }
        })?,
    )?;

    Ok(ren)
}

fn value_to_bool(val: &Value) -> bool {
    match val {
        Value::Boolean(b) => *b,
        Value::Integer(i) => *i != 0,
        Value::Number(n) => *n != 0.0,
        _ => false,
    }
}
