//! Lua-visible API tables (`sim`, `elem`, `ren`, `evt`, `tpt`, `interface`,
//! `graphics`, `tools`, `bit`, `bz2`, `http`, `socket`, `platform`, `fs`).
//!
//! Each namespace lives in its own submodule and is installed by
//! [`install_globals`] together with the classic `simulation` / `elements` /
//! `renderer` / `event` / `ui` / `gfx` / `plat` aliases. The value-conversion
//! helpers shared by every namespace stay here so the submodules only own their
//! table.

pub(crate) mod bit;
pub(crate) mod bz2;
pub(crate) mod elem;
pub(crate) mod elem_constants;
pub(crate) mod evt;
pub(crate) mod fs;
pub mod graphics;
pub(crate) mod http;
pub mod interface;
pub(crate) mod platform;
pub(crate) mod ren;
pub(crate) mod sim;
pub(crate) mod socket;
pub(crate) mod tpt;
pub mod tools;
use mlua::{Lua, MultiValue, Value, Variadic};
use tpt_sim::buffers::SimBuffers;

use crate::api::bit::build_bit_table;
use crate::api::bz2::build_bz2_table;
use crate::api::elem::build_elem_table;
use crate::api::evt::build_event_table;
use crate::api::fs::build_filesystem_table;
use crate::api::graphics::build_graphics_table;
use crate::api::http::build_http_table;
use crate::api::interface::build_interface_table;
use crate::api::platform::build_platform_table;
use crate::api::ren::build_render_table;
use crate::api::socket::build_socket_table;
use crate::api::tools::build_tools_table;
use crate::api::sim::build_sim_table;
use crate::api::tpt::build_tpt_table;
use crate::bridge::{Bridge, borrow_bridge};

/// Installs the TPT globals: `print`, `sim`, `elem`, `ren`, `evt`, `tpt`,
/// `interface`, `graphics`, `tools`, `bit`, `bz2`, `http`, `socket`,
/// `platform` and `fileSystem`, plus the classic aliases (`simulation`,
/// `elements`, `renderer`, `event`, `ui`, `gfx`, `plat`, `fs`).
pub(crate) fn install_globals(lua: &Lua, bridge: &Bridge) -> mlua::Result<()> {
    let globals = lua.globals();

    let print = lua.create_function({
        let bridge = bridge.clone();
        move |_, args: Variadic<Value>| -> mlua::Result<()> {
            let line = format_arguments(&args);
            borrow_bridge(&bridge)?.push_log(line);
            Ok(())
        }
    })?;
    globals.set("print", print.clone())?;

    // `LuaScriptInterface::LuaScriptInterface` installs the namespaces in this
    // order (`LuaScriptInterface.cpp:154-168`).
    globals.set("bit", build_bit_table(lua)?)?;
    globals.set("bz2", build_bz2_table(lua)?)?;

    let sim = build_sim_table(lua, bridge)?;
    globals.set("sim", sim.clone())?;
    globals.set("simulation", sim)?;

    let elem = build_elem_table(lua, bridge)?;
    globals.set("elem", elem.clone())?;
    globals.set("elements", elem)?;

    let evt = build_event_table(lua, bridge)?;
    globals.set("evt", evt.clone())?;
    globals.set("event", evt)?;

    let file_system = build_filesystem_table(lua)?;
    globals.set("fileSystem", file_system.clone())?;
    globals.set("fs", file_system)?;

    let graphics = build_graphics_table(lua)?;
    globals.set("graphics", graphics.clone())?;
    globals.set("gfx", graphics)?;

    globals.set("http", build_http_table(lua)?)?;

    let interface = build_interface_table(lua)?;
    globals.set("interface", interface.clone())?;
    globals.set("ui", interface)?;

    globals.set("tpt", build_tpt_table(lua, bridge, print)?)?;

    let plat = build_platform_table(lua)?;
    globals.set("platform", plat.clone())?;
    globals.set("plat", plat)?;

    let ren = build_render_table(lua, bridge)?;
    globals.set("ren", ren.clone())?;
    globals.set("renderer", ren)?;

    globals.set("socket", build_socket_table(lua)?)?;

    globals.set("tools", build_tools_table(lua, bridge)?)?;

    Ok(())
}

/// A scalar value read from an element definition or a particle slot.
///
/// Values are collected while the bridge holds the engine and converted into
/// real Lua values afterwards, so no Lua allocation happens inside the
/// engine-access window.
pub(crate) enum PropValue {
    Str(String),
    Int(i64),
    Num(f64),
}

impl PropValue {
    /// Converts the collected scalar into a Lua value.
    pub(crate) fn into_value(self, lua: &Lua) -> mlua::Result<Value> {
        Ok(match self {
            PropValue::Str(text) => Value::String(lua.create_string(text.as_bytes())?),
            PropValue::Int(number) => Value::Integer(number),
            PropValue::Num(number) => Value::Number(number),
        })
    }
}

/// Returns whether a Lua value is truthy (only `nil` and `false` are falsy).
pub(crate) fn is_truthy(value: &Value) -> bool {
    !matches!(value, Value::Nil | Value::Boolean(false))
}

/// Pulls the whole particle pool back for a getter that walks every slot.
///
/// `sim.parts`, `sim.elementCount` and `sim.hash` inspect the entire pool, so
/// the one-slot read used by `sim.partProperty` cannot serve them. The read is
/// skipped while a host edit is pending: the upload at the end of the Lua call
/// makes the host shadow authoritative, and reading the device here would
/// overwrite the script's own writes with pre-edit values.
pub(crate) fn refresh_particles(bufs: &mut SimBuffers) {
    if bufs.gpu_particles_dirty && !bufs.host_particles_dirty {
        bufs.sync_particles_gpu_to_host();
    }
}

/// Pulls the `pmap`/`photons` rectangle a pixel getter reads.
///
/// `sim.partID`, `sim.pmap` and `sim.neighbors` only need the pixels they
/// address; the rest of the map stays on the device. A pending host map edit
/// wins for the same reason as [`refresh_particles`].
pub(crate) fn refresh_map_rect(bufs: &mut SimBuffers, x0: i32, y0: i32, x1: i32, y1: i32) {
    if bufs.gpu_maps_dirty && !bufs.host_maps_dirty {
        bufs.read_map_rect(x0, y0, x1, y1);
    }
}

/// Pulls the Eulerian air grids for `sim.pressure` / `sim.ambientHeat` /
/// `sim.velocity`.
///
/// The fan-velocity and wall getters are host-authoritative (no device pass
/// writes those grids), so they do not call this.
pub(crate) fn refresh_air(bufs: &mut SimBuffers) {
    if bufs.gpu_air_dirty && !bufs.host_air_dirty {
        bufs.sync_air_gpu_to_host();
    }
}

/// Refreshes [`SimBuffers::active_count`] from the device reduction.
///
/// The kernels create and kill particles without touching the host counter, so
/// `sim.partCount` asks the device for the exact number instead of reading the
/// whole `part_ty` array back. A pending host edit (creation, kill) is counted
/// by `active_count` already and must not be replaced by the pre-upload device
/// count.
pub(crate) fn refresh_particle_count(bufs: &mut SimBuffers) {
    if bufs.gpu_particles_dirty && !bufs.host_particles_dirty {
        bufs.refresh_active_count_device();
    }
}

/// Converts a numeric Lua value to `f64`, accepting numeric strings and
/// booleans the way TPT's Lua API does.
pub(crate) fn value_to_f64(value: &Value) -> mlua::Result<f64> {
    match value {
        Value::Integer(number) => Ok(*number as f64),
        Value::Number(number) => Ok(*number),
        Value::Boolean(flag) => Ok(if *flag { 1.0 } else { 0.0 }),
        Value::String(text) => {
            let text = text.to_string_lossy();
            text.trim()
                .parse::<f64>()
                .map_err(|_| mlua::Error::runtime(format!("expected a number, got \"{text}\"")))
        }
        other => Err(mlua::Error::runtime(format!(
            "expected a number, got {}",
            other.type_name()
        ))),
    }
}

/// Converts a numeric Lua value to `i64`.
pub(crate) fn value_to_i64(value: &Value) -> mlua::Result<i64> {
    match value {
        Value::Integer(number) => Ok(*number),
        other => Ok(value_to_f64(other)? as i64),
    }
}

/// Converts a Lua value to an owned `String`.
pub(crate) fn value_to_string(value: &Value) -> mlua::Result<String> {
    match value {
        Value::String(text) => Ok(text.to_string_lossy()),
        other => Err(mlua::Error::runtime(format!(
            "expected a string, got {}",
            other.type_name()
        ))),
    }
}

/// Builds a two-value Lua return (used by `sim.partPosition` and
/// `sim.velocity`).
pub(crate) fn pair(first: f32, second: f32) -> MultiValue {
    MultiValue::from_vec(vec![
        Value::Number(first as f64),
        Value::Number(second as f64),
    ])
}

/// Formats `print` / `tpt.log` arguments the way Lua's `print` does.
fn format_arguments(args: &Variadic<Value>) -> String {
    let mut line = String::new();
    for (index, value) in args.iter().enumerate() {
        if index > 0 {
            line.push('\t');
        }
        line.push_str(&format_value(value));
    }
    line
}

/// Formats a single Lua value for the console.
fn format_value(value: &Value) -> String {
    match value {
        Value::Nil => "nil".to_owned(),
        Value::Boolean(flag) => flag.to_string(),
        Value::Integer(number) => number.to_string(),
        Value::Number(number) => format!("{number}"),
        Value::String(text) => text.to_string_lossy(),
        other => format!("<{}>", other.type_name()),
    }
}
