//! Simulation physics toggles, configs and state management.

use mlua::{Lua, MultiValue, Table, Value};
use std::rc::Rc;
use tpt_core::constants::{MAX_PRESSURE, MAX_TEMP, MIN_PRESSURE, MIN_TEMP};
use tpt_core::particle::{AirMode, EdgeMode, GravityMode};

use crate::api::{pair, refresh_particles, value_to_f64, value_to_i64};
use crate::bridge::{Bridge, borrow_bridge};

pub(crate) fn register_state(lua: &Lua, bridge: &Bridge, sim: &Table) -> mlua::Result<()> {
    // sim.ambientHeatSim([bool]) -> bool | ()
    sim.set(
        "ambientHeatSim",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<bool>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    state.with_engine(|engine| engine.bufs.config.aheat_enable = val)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.aheat_enable)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // sim.heatSim([bool]) -> bool | ()
    sim.set(
        "heatSim",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<bool>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    state.with_engine(|engine| engine.bufs.config.heat_enable = val)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.heat_enable)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // sim.newtonianGravity([bool]) -> bool | ()
    sim.set(
        "newtonianGravity",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<bool>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    state.with_engine(|engine| engine.bufs.config.newton_grav = val)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.newton_grav)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // sim.paused([bool]) -> bool | ()
    sim.set(
        "paused",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<bool>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    state.with_engine(|engine| engine.bufs.config.paused = val)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.paused)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // sim.gravityMode([mode]) -> int | ()
    sim.set(
        "gravityMode",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(m) = arg {
                    state.with_engine(|engine| {
                        engine.bufs.config.gravity_mode = match m {
                            1 => GravityMode::Off,
                            2 => GravityMode::Radial,
                            3 => GravityMode::Custom,
                            _ => GravityMode::Vertical,
                        };
                    })?;
                    Ok(Value::Nil)
                } else {
                    let cur =
                        state.with_engine(|engine| match engine.bufs.config.gravity_mode {
                            GravityMode::Vertical => 0i64,
                            GravityMode::Off => 1i64,
                            GravityMode::Radial => 2i64,
                            GravityMode::Custom => 3i64,
                        })?;
                    Ok(Value::Integer(cur))
                }
            }
        })?,
    )?;

    // sim.customGravity([gx, gy] | [gy]) -> (gx, gy) | ()
    sim.set(
        "customGravity",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<MultiValue> {
                let vec = args.into_vec();
                let mut state = borrow_bridge(&bridge)?;
                if vec.is_empty() {
                    let (gx, gy) = state.with_engine(|engine| {
                        (
                            engine.bufs.config.custom_gravity_x,
                            engine.bufs.config.custom_gravity_y,
                        )
                    })?;
                    Ok(pair(gx, gy))
                } else if vec.len() == 1 {
                    let gy = value_to_f64(&vec[0])? as f32;
                    state.with_engine(|engine| {
                        engine.bufs.config.custom_gravity_x = 0.0;
                        engine.bufs.config.custom_gravity_y = gy;
                    })?;
                    Ok(MultiValue::new())
                } else {
                    let gx = value_to_f64(&vec[0])? as f32;
                    let gy = value_to_f64(&vec[1])? as f32;
                    state.with_engine(|engine| {
                        engine.bufs.config.custom_gravity_x = gx;
                        engine.bufs.config.custom_gravity_y = gy;
                    })?;
                    Ok(MultiValue::new())
                }
            }
        })?,
    )?;

    // sim.edgeMode([mode]) -> int | ()
    sim.set(
        "edgeMode",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(m) = arg {
                    state.with_engine(|engine| {
                        engine.bufs.config.edge_mode = match m {
                            1 => EdgeMode::Solid,
                            2 => EdgeMode::Loop,
                            _ => EdgeMode::Void,
                        };
                    })?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| match engine.bufs.config.edge_mode {
                        EdgeMode::Void => 0i64,
                        EdgeMode::Solid => 1i64,
                        EdgeMode::Loop => 2i64,
                    })?;
                    Ok(Value::Integer(cur))
                }
            }
        })?,
    )?;

    // sim.airMode([mode]) -> int | ()
    sim.set(
        "airMode",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(m) = arg {
                    state.with_engine(|engine| {
                        engine.bufs.config.air_mode = match m {
                            1 => AirMode::PressureOff,
                            2 => AirMode::VelocityOff,
                            3 => AirMode::Off,
                            4 => AirMode::NoUpdate,
                            _ => AirMode::On,
                        };
                    })?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| match engine.bufs.config.air_mode {
                        AirMode::On => 0i64,
                        AirMode::PressureOff => 1i64,
                        AirMode::VelocityOff => 2i64,
                        AirMode::Off => 3i64,
                        AirMode::NoUpdate => 4i64,
                    })?;
                    Ok(Value::Integer(cur))
                }
            }
        })?,
    )?;

    // sim.waterEqualization([mode]) -> int | ()
    sim.set(
        "waterEqualization",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(m) = arg {
                    state.with_engine(|engine| engine.bufs.config.water_equal_test = m != 0)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.water_equal_test)?;
                    Ok(Value::Integer(if cur { 1 } else { 0 }))
                }
            }
        })?,
    )?;

    // sim.ambientAirTemp([temp]) -> float | ()
    sim.set(
        "ambientAirTemp",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<f64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(t) = arg {
                    let clamped = (t as f32).clamp(MIN_TEMP, MAX_TEMP);
                    state.with_engine(|engine| engine.bufs.config.ambient_air_temp = clamped)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.ambient_air_temp)?;
                    Ok(Value::Number(cur as f64))
                }
            }
        })?,
    )?;

    // sim.edgePressure([pressure]) -> float | ()
    sim.set(
        "edgePressure",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<f64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(p) = arg {
                    let clamped = (p as f32).clamp(MIN_PRESSURE, MAX_PRESSURE);
                    state.with_engine(|engine| engine.bufs.config.edge_pressure = clamped)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.edge_pressure)?;
                    Ok(Value::Number(cur as f64))
                }
            }
        })?,
    )?;

    // sim.edgeVelocity([vx, vy]) -> (vx, vy) | ()
    sim.set(
        "edgeVelocity",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<MultiValue> {
                let vec = args.into_vec();
                let mut state = borrow_bridge(&bridge)?;
                if vec.is_empty() {
                    let (vx, vy) = state.with_engine(|engine| {
                        (
                            engine.bufs.config.edge_velocity_x,
                            engine.bufs.config.edge_velocity_y,
                        )
                    })?;
                    Ok(pair(vx, vy))
                } else {
                    let vx = (value_to_f64(&vec[0])? as f32).clamp(-16.0, 16.0);
                    let vy = if vec.len() >= 2 {
                        (value_to_f64(&vec[1])? as f32).clamp(-16.0, 16.0)
                    } else {
                        0.0
                    };
                    state.with_engine(|engine| {
                        engine.bufs.config.edge_velocity_x = vx;
                        engine.bufs.config.edge_velocity_y = vy;
                    })?;
                    Ok(MultiValue::new())
                }
            }
        })?,
    )?;

    // sim.vorticityCoeff([coeff]) -> float | ()
    sim.set(
        "vorticityCoeff",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<f64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(c) = arg {
                    let clamped = (c as f32).clamp(0.0, 1.0);
                    state.with_engine(|engine| engine.bufs.config.vorticity_coeff = clamped)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.vorticity_coeff)?;
                    Ok(Value::Number(cur as f64))
                }
            }
        })?,
    )?;

    // sim.convectionMode([mode]) -> int | ()
    sim.set(
        "convectionMode",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(m) = arg {
                    if !(0..=2).contains(&m) {
                        return Err(mlua::Error::runtime("invalid convection mode"));
                    }
                    state.with_engine(|engine| engine.bufs.config.convection_mode = m as u32)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.convection_mode)?;
                    Ok(Value::Integer(cur as i64))
                }
            }
        })?,
    )?;

    // sim.gravityGrid([val]) -> int | ()
    sim.set(
        "gravityGrid",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(v) = arg {
                    state.with_engine(|engine| engine.gravity_grid = v as u32)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.gravity_grid)?;
                    Ok(Value::Integer(cur as i64))
                }
            }
        })?,
    )?;

    // sim.decoSpace([index]) -> int | ()
    sim.set(
        "decoSpace",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(v) = arg {
                    if !(0..4).contains(&v) {
                        return Err(mlua::Error::runtime(format!(
                            "Invalid deco space index {v}"
                        )));
                    }
                    state.with_engine(|engine| engine.deco_space = v as u32)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.deco_space)?;
                    Ok(Value::Integer(cur as i64))
                }
            }
        })?,
    )?;

    // sim.temperatureScale([scale]) -> int | ()
    sim.set(
        "temperatureScale",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(s) = arg {
                    if !(0..3).contains(&s) {
                        return Err(mlua::Error::runtime("Invalid temperature scale"));
                    }
                    state.with_engine(|engine| engine.bufs.config.temperature_scale = s as u32)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.temperature_scale)?;
                    Ok(Value::Integer(cur as i64))
                }
            }
        })?,
    )?;

    // sim.prettyPowders([val]) -> int | ()
    sim.set(
        "prettyPowders",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(v) = arg {
                    state.with_engine(|engine| engine.bufs.config.pretty_powder = v != 0)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.bufs.config.pretty_powder)?;
                    Ok(Value::Integer(if cur { 1 } else { 0 }))
                }
            }
        })?,
    )?;

    // sim.replaceModeFlags([flags]) -> int | ()
    sim.set(
        "replaceModeFlags",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(f) = arg {
                    let flags = f as u32;
                    if (flags & !0x03) != 0 {
                        return Err(mlua::Error::runtime("Invalid flags"));
                    }
                    if (flags & 0x01 != 0) && (flags & 0x02 != 0) {
                        return Err(mlua::Error::runtime(
                            "Cannot set replace mode and specific delete at the same time",
                        ));
                    }
                    state.with_engine(|engine| engine.replace_mode_flags = flags)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.replace_mode_flags)?;
                    Ok(Value::Integer(cur as i64))
                }
            }
        })?,
    )?;

    // sim.frameRender([frames]) -> int | ()
    sim.set(
        "frameRender",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(f) = arg {
                    if f < 0 {
                        return Err(mlua::Error::runtime(
                            "Can't simulate a negative number of frames",
                        ));
                    }
                    state.with_engine(|engine| engine.queued_frames = f as i32)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.queued_frames)?;
                    Ok(Value::Integer(cur as i64))
                }
            }
        })?,
    )?;

    // sim.lastUpdatedID() -> int | nil
    sim.set(
        "lastUpdatedID",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, ()| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                let id = state.with_engine(|engine| engine.last_updated_id)?;
                Ok(match id {
                    Some(id) => Value::Integer(id as i64),
                    None => Value::Nil,
                })
            }
        })?,
    )?;

    // sim.updateUpTo([upTo]) -> ()
    sim.set(
        "updateUpTo",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<i64>| -> mlua::Result<()> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    let up_to = arg.unwrap_or((engine.bufs.capacity as i64) - 1);
                    if up_to < -1 || up_to >= engine.bufs.capacity as i64 {
                        return Err(mlua::Error::runtime("ID not in valid range"));
                    }
                    engine.step();
                    Ok(())
                })??;
                // The step ran on the device, so the host shadow is stale and
                // must not be rebuilt into a map (the next getter pulls what it
                // needs, and the device keeps its own authoritative maps).
                Ok(())
            }
        })?,
    )?;

    // sim.randomSeed([s1, s2, s3, s4]) -> (s1, s2, s3, s4) | ()
    sim.set(
        "randomSeed",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<MultiValue> {
                let vec = args.into_vec();
                let mut state = borrow_bridge(&bridge)?;
                if vec.len() >= 4 {
                    let s1 = value_to_i64(&vec[0])? as u32;
                    let s2 = value_to_i64(&vec[1])? as u32;
                    let s3 = value_to_i64(&vec[2])? as u32;
                    let s4 = value_to_i64(&vec[3])? as u32;
                    state.with_engine(|engine| {
                        engine.rng_state = [s1, s2, s3, s4];
                    })?;
                    Ok(MultiValue::new())
                } else {
                    let s = state.with_engine(|engine| engine.rng_state)?;
                    let mut vals = MultiValue::new();
                    vals.push_back(Value::Integer(s[0] as i64));
                    vals.push_back(Value::Integer(s[1] as i64));
                    vals.push_back(Value::Integer(s[2] as i64));
                    vals.push_back(Value::Integer(s[3] as i64));
                    Ok(vals)
                }
            }
        })?,
    )?;

    // sim.hash() -> int
    sim.set(
        "hash",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, ()| -> mlua::Result<i64> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    // The hash folds the particle SoA, so the pool has to be
                    // current; only the first 1024 slots are hashed.
                    refresh_particles(&mut engine.bufs);
                    engine.compute_hash() as i64
                })
            }
        })?,
    )?;

    // sim.ensureDeterminism([bool]) -> bool | ()
    sim.set(
        "ensureDeterminism",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, arg: Option<bool>| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                if let Some(val) = arg {
                    state.with_engine(|engine| engine.ensure_determinism = val)?;
                    Ok(Value::Nil)
                } else {
                    let cur = state.with_engine(|engine| engine.ensure_determinism)?;
                    Ok(Value::Boolean(cur))
                }
            }
        })?,
    )?;

    // sim.resetSpark() -> ()
    sim.set(
        "resetSpark",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, ()| -> mlua::Result<()> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| engine.reset_spark())?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.resetTemp([onlyConductors]) -> ()
    sim.set(
        "resetTemp",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, only_conductors: Option<bool>| -> mlua::Result<()> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| engine.reset_temp(only_conductors.unwrap_or(false)))?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.clearSim() -> ()
    sim.set(
        "clearSim",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, ()| -> mlua::Result<()> {
                let mut state = borrow_bridge(&bridge)?;
                // `clear` resets every array and uploads it; the host and
                // device maps are already empty, so no rebuild is queued.
                state.with_engine(|engine| engine.clear())?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.clearRect(x, y, w, h) -> ()
    sim.set(
        "clearRect",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (x, y, w, h): (i32, i32, i32, i32)| -> mlua::Result<()> {
                let mut state = borrow_bridge(&bridge)?;
                // `clear_rect` kills slots individually (each updates its
                // pixel-map word) and uploads what it touched.
                state.with_engine(|engine| engine.clear_rect(x, y, w, h))?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.adjustCoords(x, y) -> (x, y)
    sim.set(
        "adjustCoords",
        lua.create_function(
            |_, (x, y): (Option<i32>, Option<i32>)| -> mlua::Result<MultiValue> {
                let x = x.unwrap_or(0);
                let y = y.unwrap_or(0);
                Ok(pair(x as f32, y as f32))
            },
        )?,
    )?;

    Ok(())
}
