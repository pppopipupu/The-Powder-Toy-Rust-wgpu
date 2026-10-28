//! The `evt` table (and the classic `event` alias).

use mlua::{Function, Lua, Table, Value};

use crate::bridge::{Bridge, borrow_bridge};

/// Events that can be registered from Lua and fired by the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LuaEvent {
    Tick = 8,
    MouseDown = 4,
    KeyPress = 2,
}

impl LuaEvent {
    /// Converts the numeric `evt.*` constant used by Lua into a [`LuaEvent`].
    ///
    /// Accepts both TPT variant indices and the legacy port numbers.
    pub(crate) fn from_id(id: u32) -> Result<Self, mlua::Error> {
        match id {
            8 | 0 => Ok(Self::Tick),
            4 | 1 => Ok(Self::MouseDown),
            2 | 3 => Ok(Self::KeyPress),
            other => Err(mlua::Error::runtime(format!(
                "unsupported or invalid event type {other}"
            ))),
        }
    }
}

const UNREGISTERED_REGISTRY_KEY: &str = "__tpt_unregistered_event_handlers";

pub(crate) fn mark_handler_unregistered(
    lua: &Lua,
    event: LuaEvent,
    func: &Function,
) -> mlua::Result<()> {
    let unreg_table: Table = match lua.named_registry_value(UNREGISTERED_REGISTRY_KEY) {
        Ok(t) => t,
        Err(_) => {
            let t = lua.create_table()?;
            lua.set_named_registry_value(UNREGISTERED_REGISTRY_KEY, t.clone())?;
            t
        }
    };
    let event_table: Table = match unreg_table.get(event as u32) {
        Ok(t) => t,
        Err(_) => {
            let t = lua.create_table()?;
            unreg_table.set(event as u32, t.clone())?;
            t
        }
    };
    event_table.set(func.clone(), true)?;
    Ok(())
}

pub(crate) fn unmark_handler_unregistered(
    lua: &Lua,
    event: LuaEvent,
    func: &Function,
) -> mlua::Result<()> {
    if let Ok(unreg_table) = lua.named_registry_value::<Table>(UNREGISTERED_REGISTRY_KEY)
        && let Ok(event_table) = unreg_table.get::<Table>(event as u32)
    {
        let _ = event_table.set(func.clone(), Value::Nil);
    }
    Ok(())
}

pub(crate) fn is_handler_unregistered(lua: &Lua, event: LuaEvent, func: &Function) -> bool {
    if let Ok(unreg_table) = lua.named_registry_value::<Table>(UNREGISTERED_REGISTRY_KEY)
        && let Ok(event_table) = unreg_table.get::<Table>(event as u32)
        && let Ok(val) = event_table.get::<Value>(func.clone())
    {
        return matches!(val, Value::Boolean(true));
    }
    false
}

const EVENT_CONSTANTS: &[(&str, u32)] = &[
    ("TEXTINPUT", 0),
    ("TEXTEDITING", 1),
    ("KEYPRESS", 2),
    ("KEYRELEASE", 3),
    ("MOUSEDOWN", 4),
    ("MOUSEUP", 5),
    ("MOUSEMOVE", 6),
    ("MOUSEWHEEL", 7),
    ("TICK", 8),
    ("BLUR", 9),
    ("CLOSE", 10),
    ("BEFORESIM", 11),
    ("AFTERSIM", 12),
    ("BEFORESIMDRAW", 13),
    ("AFTERSIMDRAW", 14),
];

/// Builds the `evt` table.
pub(super) fn build_event_table(lua: &Lua, bridge: &Bridge) -> mlua::Result<Table> {
    let evt = lua.create_table()?;

    for (name, id) in EVENT_CONSTANTS {
        evt.set(*name, *id)?;
    }

    // evt.register(eventType, function) -> function
    evt.set(
        "register",
        lua.create_function({
            let bridge = bridge.clone();
            move |lua, (event_id, function): (u32, Function)| -> mlua::Result<Function> {
                let event = LuaEvent::from_id(event_id)?;
                unmark_handler_unregistered(lua, event, &function)?;
                let key = lua.create_registry_value(function.clone())?;
                borrow_bridge(&bridge)?
                    .pending_registrations
                    .push((event, key));
                Ok(function)
            }
        })?,
    )?;

    // evt.unregister(eventType, function)
    evt.set(
        "unregister",
        lua.create_function(
            |lua, (event_id, function): (u32, Function)| -> mlua::Result<Value> {
                let event = LuaEvent::from_id(event_id)?;
                mark_handler_unregistered(lua, event, &function)?;
                Ok(Value::Nil)
            },
        )?,
    )?;

    // evt.getModifiers() -> number
    evt.set(
        "getModifiers",
        lua.create_function(|_, ()| -> mlua::Result<f64> { Ok(0.0) })?,
    )?;

    Ok(evt)
}
