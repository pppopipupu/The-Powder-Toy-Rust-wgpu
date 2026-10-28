//! Particle property accessors and field constants.

use mlua::{Lua, MultiValue, Table, Value};
use std::rc::Rc;
use tpt_core::constants::PT_NONE;

use crate::api::{value_to_f64, value_to_i64};
use crate::bridge::{Bridge, borrow_bridge};

pub const FIELD_TYPE: i64 = 0;
pub const FIELD_LIFE: i64 = 1;
pub const FIELD_CTYPE: i64 = 2;
pub const FIELD_X: i64 = 3;
pub const FIELD_Y: i64 = 4;
pub const FIELD_VX: i64 = 5;
pub const FIELD_VY: i64 = 6;
pub const FIELD_TEMP: i64 = 7;
pub const FIELD_FLAGS: i64 = 8;
pub const FIELD_TMP: i64 = 9;
pub const FIELD_TMP2: i64 = 10;
pub const FIELD_TMP3: i64 = 11;
pub const FIELD_TMP4: i64 = 12;
pub const FIELD_DCOLOUR: i64 = 13;

pub(crate) fn register_properties(lua: &Lua, bridge: &Bridge, sim: &Table) -> mlua::Result<()> {
    // Register FIELD_* constants
    sim.set("FIELD_TYPE", FIELD_TYPE)?;
    sim.set("FIELD_LIFE", FIELD_LIFE)?;
    sim.set("FIELD_CTYPE", FIELD_CTYPE)?;
    sim.set("FIELD_X", FIELD_X)?;
    sim.set("FIELD_Y", FIELD_Y)?;
    sim.set("FIELD_VX", FIELD_VX)?;
    sim.set("FIELD_VY", FIELD_VY)?;
    sim.set("FIELD_TEMP", FIELD_TEMP)?;
    sim.set("FIELD_FLAGS", FIELD_FLAGS)?;
    sim.set("FIELD_TMP", FIELD_TMP)?;
    sim.set("FIELD_TMP2", FIELD_TMP2)?;
    sim.set("FIELD_TMP3", FIELD_TMP3)?;
    sim.set("FIELD_TMP4", FIELD_TMP4)?;
    sim.set("FIELD_DCOLOUR", FIELD_DCOLOUR)?;
    sim.set("FIELD_PAVG0", FIELD_TMP3)?;
    sim.set("FIELD_PAVG1", FIELD_TMP4)?;
    sim.set("FIELD_DCOLOR", FIELD_DCOLOUR)?;

    // sim.partProperty(id, field, [value])
    sim.set(
        "partProperty",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<Value> {
                let mut iter = args.into_iter();
                let Some(id_val) = iter.next() else {
                    return Err(mlua::Error::runtime("expected particle ID"));
                };
                let id = value_to_i64(&id_val)?;
                let Some(field_val) = iter.next() else {
                    return Err(mlua::Error::runtime("expected property name or field ID"));
                };
                let new_value = iter.next();

                let field_id = match &field_val {
                    Value::Integer(i) => *i,
                    Value::Number(n) => *n as i64,
                    Value::String(s) => {
                        let name = s.to_str()?;
                        resolve_field_name(&name)?
                    }
                    _ => {
                        return Err(mlua::Error::runtime(
                            "Field ID must be an name (string) or identifier (integer)",
                        ));
                    }
                };

                if !(0..=13).contains(&field_id) {
                    return Err(mlua::Error::runtime(format!(
                        "Invalid field ID ({field_id})"
                    )));
                }

                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if id < 0 || id as usize >= engine.bufs.capacity {
                        return Ok(Value::Nil);
                    }
                    let slot = id as usize;
                    // `sim.partProperty` addresses one slot; pull just that slot
                    // back instead of the whole pool (see
                    // `SimBuffers::read_particle_slot`).
                    engine.bufs.read_particle_slot(slot);
                    if engine.bufs.part_ty[slot] == PT_NONE {
                        return Ok(Value::Nil);
                    }
                    let old_x = engine.bufs.part_x[slot];
                    let old_y = engine.bufs.part_y[slot];

                    if let Some(val) = new_value {
                        let mut moved = false;
                        match field_id {
                            FIELD_TYPE => {
                                let ty = value_to_i64(&val)? as u32;
                                engine.bufs.part_ty[slot] = ty;
                                moved = true;
                            }
                            FIELD_LIFE => engine.bufs.part_life[slot] = value_to_i64(&val)? as i32,
                            FIELD_CTYPE => {
                                engine.bufs.part_ctype[slot] = value_to_i64(&val)? as u32
                            }
                            FIELD_X => {
                                let x = value_to_f64(&val)? as f32;
                                engine.bufs.part_x[slot] = x
                                    .clamp(0.0, (engine.bufs.config.xres.saturating_sub(1)) as f32);
                                moved = true;
                            }
                            FIELD_Y => {
                                let y = value_to_f64(&val)? as f32;
                                engine.bufs.part_y[slot] = y
                                    .clamp(0.0, (engine.bufs.config.yres.saturating_sub(1)) as f32);
                                moved = true;
                            }
                            FIELD_VX => engine.bufs.part_vx[slot] = value_to_f64(&val)? as f32,
                            FIELD_VY => engine.bufs.part_vy[slot] = value_to_f64(&val)? as f32,
                            FIELD_TEMP => engine.bufs.part_temp[slot] = value_to_f64(&val)? as f32,
                            FIELD_FLAGS => {
                                engine.bufs.part_flags[slot] = value_to_i64(&val)? as u32
                            }
                            FIELD_TMP => engine.bufs.part_tmp[slot] = value_to_i64(&val)? as i32,
                            FIELD_TMP2 => engine.bufs.part_tmp2[slot] = value_to_i64(&val)? as i32,
                            FIELD_TMP3 => engine.bufs.part_tmp3[slot] = value_to_i64(&val)? as i32,
                            FIELD_TMP4 => engine.bufs.part_tmp4[slot] = value_to_i64(&val)? as i32,
                            FIELD_DCOLOUR => {
                                engine.bufs.part_dcolour[slot] = value_to_i64(&val)? as u32
                            }
                            _ => {}
                        }
                        engine.bufs.mark_particle_slot_dirty(slot);
                        if moved {
                            // One slot changes pixel; the device rebuilds its
                            // own maps from the particle arrays every frame.
                            engine.bufs.retarget_particle_pixel(slot, old_x, old_y);
                        }
                        Ok(Value::Nil)
                    } else {
                        Ok(match field_id {
                            FIELD_TYPE => Value::Integer(engine.bufs.part_ty[slot] as i64),
                            FIELD_LIFE => Value::Integer(engine.bufs.part_life[slot] as i64),
                            FIELD_CTYPE => Value::Integer(engine.bufs.part_ctype[slot] as i64),
                            FIELD_X => Value::Number(engine.bufs.part_x[slot] as f64),
                            FIELD_Y => Value::Number(engine.bufs.part_y[slot] as f64),
                            FIELD_VX => Value::Number(engine.bufs.part_vx[slot] as f64),
                            FIELD_VY => Value::Number(engine.bufs.part_vy[slot] as f64),
                            FIELD_TEMP => Value::Number(engine.bufs.part_temp[slot] as f64),
                            FIELD_FLAGS => Value::Integer(engine.bufs.part_flags[slot] as i64),
                            FIELD_TMP => Value::Integer(engine.bufs.part_tmp[slot] as i64),
                            FIELD_TMP2 => Value::Integer(engine.bufs.part_tmp2[slot] as i64),
                            FIELD_TMP3 => Value::Integer(engine.bufs.part_tmp3[slot] as i64),
                            FIELD_TMP4 => Value::Integer(engine.bufs.part_tmp4[slot] as i64),
                            FIELD_DCOLOUR => Value::Integer(engine.bufs.part_dcolour[slot] as i64),
                            _ => Value::Nil,
                        })
                    }
                })?
            }
        })?,
    )?;

    Ok(())
}

fn resolve_field_name(name: &str) -> mlua::Result<i64> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "type" => FIELD_TYPE,
        "life" => FIELD_LIFE,
        "ctype" => FIELD_CTYPE,
        "x" => FIELD_X,
        "y" => FIELD_Y,
        "vx" => FIELD_VX,
        "vy" => FIELD_VY,
        "temp" => FIELD_TEMP,
        "flags" => FIELD_FLAGS,
        "tmp" => FIELD_TMP,
        "tmp2" => FIELD_TMP2,
        "tmp3" | "pavg0" => FIELD_TMP3,
        "tmp4" | "pavg1" => FIELD_TMP4,
        "dcolour" | "dcolor" => FIELD_DCOLOUR,
        _ => return Err(mlua::Error::runtime(format!("Unknown field ({name})"))),
    })
}
