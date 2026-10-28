//! The `elem` table (and the classic `elements` alias).

use std::rc::Rc;

use mlua::{Lua, Table, Value};
use tpt_core::constants::*;
use tpt_core::elements::{ElementRegistry, has_graphics_override};
use tpt_sim::SimulationEngine;

use super::elem_constants::{ELEMENT_CONSTANTS, TYPE_CONSTANTS};
use crate::api::{PropValue, value_to_f64, value_to_i64, value_to_string};
use crate::bridge::{Bridge, borrow_bridge};

/// One writable element property, shared by `elem.property` reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ElementProperty {
    Name,
    Description,
    Colour,
    Weight,
    Gravity,
    Flammable,
    Explosive,
    HeatConduct,
    DefaultTemp,
    HighTemperature,
    HighTemperatureTransition,
    LowTemperature,
    LowTemperatureTransition,
    Properties,
    Falldown,
    Hardness,
    Advection,
    Loss,
    AirDrag,
    AirLoss,
    Collision,
    Diffusion,
    HotAir,
    Meltable,
    LowPressure,
    LowPressureTransition,
    HighPressure,
    HighPressureTransition,
}

impl ElementProperty {
    fn parse(name: &str) -> mlua::Result<Self> {
        Ok(match name.to_ascii_lowercase().as_str() {
            "name" => Self::Name,
            "description" => Self::Description,
            "colour" | "color" => Self::Colour,
            "weight" => Self::Weight,
            "gravity" => Self::Gravity,
            "flammable" => Self::Flammable,
            "explosive" => Self::Explosive,
            "heatconduct" | "heat_conduct" => Self::HeatConduct,
            "defaulttemp" | "default_temp" | "temperature" => Self::DefaultTemp,
            "hightemperature" | "high_temperature" => Self::HighTemperature,
            "hightemperaturetransition" | "high_temperature_transition" => {
                Self::HighTemperatureTransition
            }
            "lowtemperature" | "low_temperature" => Self::LowTemperature,
            "lowtemperaturetransition" | "low_temperature_transition" => {
                Self::LowTemperatureTransition
            }
            "properties" => Self::Properties,
            "falldown" => Self::Falldown,
            "hardness" => Self::Hardness,
            "advection" => Self::Advection,
            "loss" => Self::Loss,
            "airdrag" | "air_drag" => Self::AirDrag,
            "airloss" | "air_loss" => Self::AirLoss,
            "collision" => Self::Collision,
            "diffusion" => Self::Diffusion,
            "hotair" | "hot_air" => Self::HotAir,
            "meltable" => Self::Meltable,
            "lowpressure" | "low_pressure" => Self::LowPressure,
            "lowpressuretransition" | "low_pressure_transition" => Self::LowPressureTransition,
            "highpressure" | "high_pressure" => Self::HighPressure,
            "highpressuretransition" | "high_pressure_transition" => Self::HighPressureTransition,
            _ => {
                return Err(mlua::Error::runtime(format!(
                    "unknown element property \"{name}\""
                )));
            }
        })
    }
}

const ALL_ELEMENT_PROPERTIES: &[(&str, ElementProperty)] = &[
    ("Name", ElementProperty::Name),
    ("Colour", ElementProperty::Colour),
    ("Description", ElementProperty::Description),
    ("Weight", ElementProperty::Weight),
    ("Gravity", ElementProperty::Gravity),
    ("Flammable", ElementProperty::Flammable),
    ("Explosive", ElementProperty::Explosive),
    ("HeatConduct", ElementProperty::HeatConduct),
    ("Temperature", ElementProperty::DefaultTemp),
    ("DefaultTemp", ElementProperty::DefaultTemp),
    ("HighTemperature", ElementProperty::HighTemperature),
    (
        "HighTemperatureTransition",
        ElementProperty::HighTemperatureTransition,
    ),
    ("LowTemperature", ElementProperty::LowTemperature),
    (
        "LowTemperatureTransition",
        ElementProperty::LowTemperatureTransition,
    ),
    ("Properties", ElementProperty::Properties),
    ("Falldown", ElementProperty::Falldown),
    ("Hardness", ElementProperty::Hardness),
    ("Advection", ElementProperty::Advection),
    ("Loss", ElementProperty::Loss),
    ("AirDrag", ElementProperty::AirDrag),
    ("AirLoss", ElementProperty::AirLoss),
    ("Collision", ElementProperty::Collision),
    ("Diffusion", ElementProperty::Diffusion),
    ("HotAir", ElementProperty::HotAir),
    ("Meltable", ElementProperty::Meltable),
    ("LowPressure", ElementProperty::LowPressure),
    (
        "LowPressureTransition",
        ElementProperty::LowPressureTransition,
    ),
    ("HighPressure", ElementProperty::HighPressure),
    (
        "HighPressureTransition",
        ElementProperty::HighPressureTransition,
    ),
];

/// Builds the `elem` table.
pub(super) fn build_elem_table(lua: &Lua, bridge: &Bridge) -> mlua::Result<Table> {
    let elem = lua.create_table()?;

    for (name, id) in ELEMENT_CONSTANTS {
        elem.set(*name, *id)?;
    }
    for (name, id) in TYPE_CONSTANTS {
        elem.set(*name, *id)?;
    }

    // elem.allocate(group, name) -> id | nil
    elem.set(
        "allocate",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (group, name): (String, String)| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                let id =
                    state.with_engine(|engine| engine.registry.allocate_custom(&group, &name))?;
                if id.is_some() {
                    state.elements_dirty = true;
                }
                Ok(match id {
                    Some(id) => Value::Integer(id as i64),
                    None => Value::Nil,
                })
            }
        })?,
    )?;

    // elem.getByName(name) -> id | nil
    elem.set(
        "getByName",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, name: String| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                let id = state.with_engine(|engine| engine.registry.find_by_name(&name))?;
                Ok(match id {
                    Some(id) => Value::Integer(id as i64),
                    None => Value::Nil,
                })
            }
        })?,
    )?;

    // elem.exists(id) -> bool
    elem.set(
        "exists",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, id: u32| -> mlua::Result<bool> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    id > 0
                        && (id as usize) < engine.registry.elements.len()
                        && engine.registry.elements[id as usize].enabled
                })
            }
        })?,
    )?;

    // elem.element(id, [props]) -> table | nil
    elem.set(
        "element",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, (id, props): (u32, Option<Table>)| -> mlua::Result<Value> {
                if id >= PT_NUM as u32 {
                    return Err(mlua::Error::runtime("Invalid element"));
                }
                let mut state = borrow_bridge(&bridge)?;
                if let Some(props) = props {
                    state.with_engine(|engine| {
                        if id as usize >= engine.registry.elements.len()
                            || (id != 0 && !engine.registry.elements[id as usize].enabled)
                        {
                            return Err(mlua::Error::runtime("Invalid element"));
                        }
                        for pair in props.pairs::<String, Value>() {
                            let (key, val) = pair?;
                            if key.eq_ignore_ascii_case("defaultproperties") {
                                if let Value::Table(def_tbl) = val
                                    && let Ok(temp) = def_tbl.get::<Value>("temp")
                                {
                                    let def = engine.registry.get_mut(id).unwrap();
                                    def.default_temp = value_to_f64(&temp)? as f32;
                                }
                            } else if let Ok(prop) = ElementProperty::parse(&key) {
                                write_element_property(engine, id, prop, &val)?;
                            }
                        }
                        Ok(())
                    })??;
                    state.elements_dirty = true;
                    Ok(Value::Nil)
                } else {
                    let (
                        props_map,
                        identifier,
                        default_temp,
                        default_life,
                        default_tmp,
                        default_ctype,
                    ) = state.with_engine(|engine| {
                        if id as usize >= engine.registry.elements.len()
                            || (id != 0 && !engine.registry.elements[id as usize].enabled)
                        {
                            return Err(mlua::Error::runtime("Invalid element"));
                        }
                        let mut values = Vec::new();
                        for (name, prop) in ALL_ELEMENT_PROPERTIES {
                            values.push((*name, read_element_property(engine, id, *prop)));
                        }
                        let def = engine.registry.get(id);
                        Ok((
                            values,
                            def.identifier.clone(),
                            def.default_temp,
                            def.default_life,
                            def.default_tmp,
                            def.default_ctype,
                        ))
                    })??;
                    let tbl = lua.create_table()?;
                    for (name, val) in props_map {
                        tbl.set(name, val.into_value(lua)?)?;
                    }
                    tbl.set("Identifier", identifier)?;
                    let def_props = lua.create_table()?;
                    def_props.set("temp", default_temp)?;
                    def_props.set("life", default_life)?;
                    def_props.set("tmp", default_tmp)?;
                    def_props.set("ctype", default_ctype)?;
                    tbl.set("DefaultProperties", def_props)?;
                    Ok(Value::Table(tbl))
                }
            }
        })?,
    )?;

    // elem.loadDefault([id])
    elem.set(
        "loadDefault",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, id: Option<u32>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if let Some(id) = id {
                        if id >= PT_NUM as u32 {
                            return Err(mlua::Error::runtime("Invalid element"));
                        }
                        let default_registry = ElementRegistry::new_default();
                        if let Some(default_def) = default_registry.elements.get(id as usize)
                            && let Some(target) = engine.registry.elements.get_mut(id as usize)
                        {
                            *target = default_def.clone();
                        }
                        engine.registry.rebuild_can_move();
                    } else {
                        engine.registry = ElementRegistry::new_default();
                    }
                    Ok(())
                })??;
                state.elements_dirty = true;
                Ok(Value::Nil)
            }
        })?,
    )?;

    // elem.free(id)
    elem.set(
        "free",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, id: u32| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if id == 0
                        || id as usize >= engine.registry.elements.len()
                        || !engine.registry.elements[id as usize].enabled
                    {
                        return Err(mlua::Error::runtime("Invalid element"));
                    }
                    let def = &mut engine.registry.elements[id as usize];
                    if (id as usize) < BUILTIN_ELEMENT_COUNT
                        || def.identifier.starts_with("DEFAULT_PT_")
                    {
                        return Err(mlua::Error::runtime("Cannot free default elements"));
                    }
                    def.enabled = false;
                    engine.registry.rebuild_can_move();
                    Ok(())
                })??;
                state.elements_dirty = true;
                Ok(Value::Nil)
            }
        })?,
    )?;

    register_element_property(lua, bridge, &elem)?;

    Ok(elem)
}

/// Registers `elem.property(id, name, [value])`.
fn register_element_property(lua: &Lua, bridge: &Bridge, elem: &Table) -> mlua::Result<()> {
    elem.set(
        "property",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, (id, name, value): (u32, String, Option<Value>)| -> mlua::Result<Value> {
                let property = ElementProperty::parse(&name)?;
                let mut state = borrow_bridge(&bridge)?;
                let is_write = value.is_some();
                let result = state.with_engine(|engine| {
                    if id as usize >= engine.registry.elements.len() {
                        return Err(mlua::Error::runtime(format!(
                            "elem.property: invalid element id {id}"
                        )));
                    }
                    if let Some(value) = &value {
                        write_element_property(engine, id, property, value)?;
                    }
                    Ok(read_element_property(engine, id, property))
                })??;
                if is_write {
                    state.elements_dirty = true;
                }
                result.into_value(lua)
            }
        })?,
    )?;
    Ok(())
}

/// Reads an element definition field. The element id must be validated by the
/// caller.
fn read_element_property(
    engine: &SimulationEngine,
    id: u32,
    property: ElementProperty,
) -> PropValue {
    let definition = engine.registry.get(id);
    match property {
        ElementProperty::Name => PropValue::Str(definition.name.clone()),
        ElementProperty::Description => PropValue::Str(definition.description.clone()),
        ElementProperty::Colour => PropValue::Int(definition.colour as i64),
        ElementProperty::Weight => PropValue::Int(definition.weight as i64),
        ElementProperty::Gravity => PropValue::Num(definition.gravity as f64),
        ElementProperty::Flammable => PropValue::Int(definition.flammable as i64),
        ElementProperty::Explosive => PropValue::Int(definition.explosive as i64),
        ElementProperty::HeatConduct => PropValue::Int(definition.heat_conduct as i64),
        ElementProperty::DefaultTemp => PropValue::Num(definition.default_temp as f64),
        ElementProperty::HighTemperature => PropValue::Num(definition.high_temp as f64),
        ElementProperty::HighTemperatureTransition => {
            PropValue::Int(definition.high_temp_transition as i64)
        }
        ElementProperty::LowTemperature => PropValue::Num(definition.low_temp as f64),
        ElementProperty::LowTemperatureTransition => {
            PropValue::Int(definition.low_temp_transition as i64)
        }
        ElementProperty::Properties => PropValue::Int(definition.properties as i64),
        ElementProperty::Falldown => PropValue::Int(definition.falldown as i64),
        ElementProperty::Hardness => PropValue::Int(definition.hardness as i64),
        ElementProperty::Advection => PropValue::Num(definition.advection as f64),
        ElementProperty::Loss => PropValue::Num(definition.loss as f64),
        ElementProperty::AirDrag => PropValue::Num(definition.air_drag as f64),
        ElementProperty::AirLoss => PropValue::Num(definition.air_loss as f64),
        ElementProperty::Collision => PropValue::Num(definition.collision as f64),
        ElementProperty::Diffusion => PropValue::Num(definition.diffusion as f64),
        ElementProperty::HotAir => PropValue::Num(definition.hot_air as f64),
        ElementProperty::Meltable => PropValue::Int(definition.meltable as i64),
        ElementProperty::LowPressure => PropValue::Num(definition.low_pressure as f64),
        ElementProperty::LowPressureTransition => {
            PropValue::Int(definition.low_pressure_transition as i64)
        }
        ElementProperty::HighPressure => PropValue::Num(definition.high_pressure as f64),
        ElementProperty::HighPressureTransition => {
            PropValue::Int(definition.high_pressure_transition as i64)
        }
    }
}

/// Writes an element definition field.
fn write_element_property(
    engine: &mut SimulationEngine,
    id: u32,
    property: ElementProperty,
    value: &Value,
) -> mlua::Result<()> {
    let definition = engine
        .registry
        .get_mut(id)
        .ok_or_else(|| mlua::Error::runtime(format!("elem.property: invalid element id {id}")))?;
    match property {
        ElementProperty::Name => definition.name = value_to_string(value)?,
        ElementProperty::Description => definition.description = value_to_string(value)?,
        ElementProperty::Colour => definition.colour = value_to_i64(value)? as u32,
        ElementProperty::Weight => definition.weight = value_to_i64(value)? as i32,
        ElementProperty::Gravity => definition.gravity = value_to_f64(value)? as f32,
        ElementProperty::Flammable => definition.flammable = value_to_i64(value)? as i32,
        ElementProperty::Explosive => definition.explosive = value_to_i64(value)? as i32,
        ElementProperty::HeatConduct => definition.heat_conduct = value_to_i64(value)? as u32,
        ElementProperty::DefaultTemp => definition.default_temp = value_to_f64(value)? as f32,
        ElementProperty::HighTemperature => definition.high_temp = value_to_f64(value)? as f32,
        ElementProperty::HighTemperatureTransition => {
            definition.high_temp_transition = value_to_i64(value)? as i32
        }
        ElementProperty::LowTemperature => definition.low_temp = value_to_f64(value)? as f32,
        ElementProperty::LowTemperatureTransition => {
            definition.low_temp_transition = value_to_i64(value)? as i32
        }
        ElementProperty::Properties => definition.properties = value_to_i64(value)? as u32,
        ElementProperty::Falldown => {
            let falldown = value_to_i64(value)?.clamp(0, 2) as u32;
            definition.falldown = falldown;
            // Keep the matter-state bits in `properties` in sync so movement
            // and collision rules see the new state immediately.
            let state = match falldown {
                1 => TYPE_PART,
                2 => TYPE_LIQUID,
                _ => TYPE_GAS,
            };
            definition.properties = (definition.properties & !STATE_FLAGS) | state;
        }
        ElementProperty::Hardness => definition.hardness = value_to_i64(value)? as i32,
        ElementProperty::Advection => definition.advection = value_to_f64(value)? as f32,
        ElementProperty::Loss => definition.loss = value_to_f64(value)? as f32,
        ElementProperty::AirDrag => definition.air_drag = value_to_f64(value)? as f32,
        ElementProperty::AirLoss => definition.air_loss = value_to_f64(value)? as f32,
        ElementProperty::Collision => definition.collision = value_to_f64(value)? as f32,
        ElementProperty::Diffusion => definition.diffusion = value_to_f64(value)? as f32,
        ElementProperty::HotAir => definition.hot_air = value_to_f64(value)? as f32,
        ElementProperty::Meltable => definition.meltable = value_to_i64(value)? as i32,
        ElementProperty::LowPressure => definition.low_pressure = value_to_f64(value)? as f32,
        ElementProperty::LowPressureTransition => {
            definition.low_pressure_transition = value_to_i64(value)? as i32
        }
        ElementProperty::HighPressure => definition.high_pressure = value_to_f64(value)? as f32,
        ElementProperty::HighPressureTransition => {
            definition.high_pressure_transition = value_to_i64(value)? as i32
        }
    }
    // `Element::defaultGraphics` recomputes its mode from `Properties` and
    // `Colour` on every frame, so an element that keeps it (every element
    // without a `Graphics = &...` override) must follow a property edit.
    if !has_graphics_override(id) {
        definition.apply_default_graphics();
    }
    Ok(())
}
