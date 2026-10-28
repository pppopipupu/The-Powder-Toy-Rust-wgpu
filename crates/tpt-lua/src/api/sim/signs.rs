//! `sim.signs`: the canvas text signs (`LuaSimulation.cpp:1786-1960`).
//!
//! The reference exposes a proxied table rather than a plain list: `#sim.signs`
//! counts, `sim.signs[i]` is a per-sign proxy whose `text`/`x`/`y`/`ju` are
//! read-write, and `sim.signs.new` / `sim.signs.delete` mutate the list. The
//! proxy is rebuilt on every access, so a write always lands on the live
//! `SimulationEngine` list instead of a stale copy.

use std::rc::Rc;

use mlua::{Lua, Table, Value};

use tpt_core::sign::{JUSTIFY_MODES, clamp_justification, sanitise_text};

use crate::api::value_to_i64;
use crate::bridge::{Bridge, borrow_bridge};

/// Registers `sim.signs`.
pub(crate) fn register_signs(lua: &Lua, bridge: &Bridge, sim: &Table) -> mlua::Result<()> {
    let signs = lua.create_table()?;

    // sim.signs.new(text, x, y [, ju]) -> index | nil
    signs.set(
        "new",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (text, x, y, ju): (mlua::LuaString, i64, i64, Option<i64>)| {
                let mut state = borrow_bridge(&bridge)?;
                let text = text.to_string_lossy().to_string();
                let ju = ju.unwrap_or(i64::from(tpt_core::sign::JUSTIFY_MIDDLE));
                if !(0..i64::from(JUSTIFY_MODES)).contains(&ju) {
                    return Err(mlua::Error::runtime("Invalid justification"));
                }
                let (xres, yres) = state
                    .with_engine(|engine| (engine.bufs.config.xres, engine.bufs.config.yres))?;
                if x < 0 || x as u32 >= xres {
                    return Err(mlua::Error::runtime("Invalid X coordinate"));
                }
                if y < 0 || y as u32 >= yres {
                    return Err(mlua::Error::runtime("Invalid Y coordinate"));
                }
                let index = state.with_engine(|engine| {
                    engine
                        .add_sign(x as i32, y as i32, &text, ju as u8)
                        .map(|index| index as i64)
                })?;
                Ok(index)
            }
        })?,
    )?;

    // sim.signs.delete(index)
    signs.set(
        "delete",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, index: i64| {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if index >= 1 && (index as usize) <= engine.signs().len() {
                        engine.remove_sign(index as usize - 1);
                    }
                })?;
                Ok(())
            }
        })?,
    )?;

    // `#sim.signs`
    let meta = lua.create_table()?;
    meta.set(
        "__len",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, _: Table| {
                let mut state = borrow_bridge(&bridge)?;
                let count = state.with_engine(|engine| engine.signs().len())?;
                Ok(count as i64)
            }
        })?,
    )?;

    // `sim.signs[i]`
    meta.set(
        "__index",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, (_, key): (Table, Value)| -> mlua::Result<Value> {
                let index = match key {
                    Value::Integer(index) => index,
                    Value::Number(number) => number as i64,
                    _ => return Ok(Value::Nil),
                };
                let mut state = borrow_bridge(&bridge)?;
                let sign = state.with_engine(|engine| {
                    engine
                        .signs()
                        .get(index.saturating_sub(1) as usize)
                        .cloned()
                })?;
                match sign {
                    Some(_) => Ok(Value::Table(sign_proxy(lua, &bridge, index)?)),
                    None => Ok(Value::Nil),
                }
            }
        })?,
    )?;
    signs.set_metatable(Some(meta))?;

    sim.set("signs", signs)?;
    Ok(())
}

/// Builds the per-sign proxy `sim.signs[index]`.
///
/// The proxy carries the raw `id` field like the reference (`signsIndex` reads
/// it back with `lua_rawget`), so `tostring`-style inspection and the
/// reference's own error paths still work.
fn sign_proxy(lua: &Lua, bridge: &Bridge, index: i64) -> mlua::Result<Table> {
    let proxy = lua.create_table()?;
    proxy.set("id", index)?;

    let meta = lua.create_table()?;
    meta.set(
        "__index",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, (_, key): (Table, String)| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                let sign = state.with_engine(|engine| {
                    engine
                        .signs()
                        .get(index.saturating_sub(1) as usize)
                        .cloned()
                })?;
                let Some(sign) = sign else {
                    return Ok(Value::Nil);
                };
                Ok(match key.as_str() {
                    "text" => Value::String(lua.create_string(sign.text.as_str())?),
                    "x" => Value::Integer(i64::from(sign.x)),
                    "y" => Value::Integer(i64::from(sign.y)),
                    "ju" => Value::Integer(i64::from(sign.ju)),
                    _ => Value::Nil,
                })
            }
        })?,
    )?;
    meta.set(
        "__newindex",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (_, key, value): (Table, String, Value)| -> mlua::Result<()> {
                // Convert before touching the engine: `with_engine` runs a
                // closure that cannot raise an mlua error.
                enum NewValue {
                    Text(String),
                    X(i32),
                    Y(i32),
                    Ju(u8),
                }
                let update = match key.as_str() {
                    "text" => match &value {
                        Value::String(text) => {
                            NewValue::Text(sanitise_text(&text.to_string_lossy()))
                        }
                        other => {
                            return Err(mlua::Error::runtime(format!(
                                "sign.text must be a string, got {other:?}"
                            )));
                        }
                    },
                    "x" => NewValue::X(value_to_i64(&value)? as i32),
                    "y" => NewValue::Y(value_to_i64(&value)? as i32),
                    "ju" => NewValue::Ju(clamp_justification(value_to_i64(&value)? as u8)),
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "sign property '{other}' is read-only"
                        )));
                    }
                };
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    let Some(sign) = engine.signs.get_mut(index.saturating_sub(1) as usize) else {
                        return;
                    };
                    match update {
                        NewValue::Text(text) => sign.text = text,
                        NewValue::X(x) => sign.x = x,
                        NewValue::Y(y) => sign.y = y,
                        NewValue::Ju(ju) => sign.ju = ju,
                    }
                })?;
                Ok(())
            }
        })?,
    )?;
    proxy.set_metatable(Some(meta))?;
    Ok(proxy)
}
