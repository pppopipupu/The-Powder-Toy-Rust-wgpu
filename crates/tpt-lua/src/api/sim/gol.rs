//! Game of Life rules and simulation speed controls.

use mlua::{Lua, MultiValue, Table, Value};
use std::rc::Rc;
use tpt_sim::CustomGolRule;
use tpt_sim::gol_kernel::{BUILTIN_GOL, parse_gol_string, serialise_gol_rule};

use crate::api::{value_to_i64, value_to_string};
use crate::bridge::{Bridge, borrow_bridge};

pub(crate) fn register_gol(lua: &Lua, bridge: &Bridge, sim: &Table) -> mlua::Result<()> {
    // sim.listDefaultGol() -> table
    sim.set(
        "listDefaultGol",
        lua.create_function(|lua, ()| -> mlua::Result<Table> {
            let table = lua.create_table()?;
            for (i, gol) in BUILTIN_GOL.iter().enumerate() {
                let entry = lua.create_table()?;
                entry.set("name", gol.name)?;
                entry.set("rulestr", serialise_gol_rule(gol.ruleset))?;
                entry.set("rule", gol.ruleset as i64)?;
                entry.set("color1", gol.colour as i64)?;
                entry.set("color2", gol.colour2 as i64)?;
                table.raw_set(i + 1, entry)?;
            }
            Ok(table)
        })?,
    )?;

    // sim.listCustomGol() -> table
    sim.set(
        "listCustomGol",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, ()| -> mlua::Result<Table> {
                let mut state = borrow_bridge(&bridge)?;
                let custom = state.with_engine(|engine| engine.custom_gol.clone())?;
                let table = lua.create_table()?;
                for (i, gol) in custom.iter().enumerate() {
                    let entry = lua.create_table()?;
                    entry.set("name", gol.name.clone())?;
                    entry.set("rulestr", serialise_gol_rule(gol.ruleset))?;
                    entry.set("rule", gol.ruleset as i64)?;
                    entry.set("color1", gol.colour as i64)?;
                    entry.set("color2", gol.colour2 as i64)?;
                    table.raw_set(i + 1, entry)?;
                }
                Ok(table)
            }
        })?,
    )?;

    // sim.addCustomGol(rule, name, color1, color2)
    sim.set(
        "addCustomGol",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                if vec.len() < 4 {
                    return Err(mlua::Error::runtime("expected rule, name, color1, color2"));
                }
                let rule = match &vec[0] {
                    Value::Integer(i) => {
                        let r = *i as u32;
                        let s = serialise_gol_rule(r);
                        match parse_gol_string(&s) {
                            Some(parsed) => parsed,
                            None => return Err(mlua::Error::runtime("Invalid rule provided")),
                        }
                    }
                    Value::String(s) => match parse_gol_string(&s.to_str()?) {
                        Some(parsed) => parsed,
                        None => return Err(mlua::Error::runtime("Invalid rule provided")),
                    },
                    _ => return Err(mlua::Error::runtime("Invalid rule provided")),
                };

                let name = value_to_string(&vec[1])?;
                if name.is_empty() {
                    return Err(mlua::Error::runtime("Invalid name provided"));
                }
                let color1 = value_to_i64(&vec[2])? as u32;
                let color2 = value_to_i64(&vec[3])? as u32;

                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if engine.custom_gol.iter().any(|g| g.ruleset == rule) {
                        return Err(mlua::Error::runtime("This Custom GoL rule already exists"));
                    }
                    if engine.custom_gol.iter().any(|g| g.name == name) {
                        return Err(mlua::Error::runtime("Duplicate name, cannot add"));
                    }
                    engine.custom_gol.push(CustomGolRule {
                        name,
                        ruleset: rule,
                        colour: color1,
                        colour2: color2,
                    });
                    Ok(())
                })??;

                Ok(())
            }
        })?,
    )?;

    // sim.removeCustomGol(name) -> bool
    sim.set(
        "removeCustomGol",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, name: String| -> mlua::Result<bool> {
                let mut state = borrow_bridge(&bridge)?;
                let res = state.with_engine(|engine| {
                    let clean_name = name.strip_prefix("DEFAULT_PT_LIFECUST_").unwrap_or(&name);
                    if let Some(pos) = engine
                        .custom_gol
                        .iter()
                        .position(|g| g.name == clean_name || g.name == name)
                    {
                        engine.custom_gol.remove(pos);
                        true
                    } else {
                        false
                    }
                })?;
                Ok(res)
            }
        })?,
    )?;

    // sim.golSpeedRatio([speed]) -> int | ()
    sim.set(
        "golSpeedRatio",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, speed: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(s) = speed {
                    if s < 1 {
                        return Err(mlua::Error::runtime("GSPEED must be at least 1"));
                    }
                    state.with_engine(|engine| engine.gspeed = s as u32)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.gspeed)?;
                    Ok(Value::Integer(cur as i64))
                }
            }
        })?,
    )?;

    Ok(())
}
