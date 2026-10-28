//! Particle lifecycle and query functions.

use mlua::{Lua, MultiValue, Table, Value};
use std::cell::RefCell;
use std::rc::Rc;
use tpt_core::constants::{PT_NONE, PT_NUM};

use crate::api::{
    pair, refresh_map_rect, refresh_particle_count, refresh_particles, value_to_f64, value_to_i64,
};
use crate::bridge::{Bridge, borrow_bridge};
use tpt_sim::buffers::CreationSource;

use super::refresh_pixel_occupant;

pub(crate) fn register_lifecycle(lua: &Lua, bridge: &Bridge, sim: &Table) -> mlua::Result<()> {
    // sim.partCreate(newID, x, y, ty, [v]) or sim.partCreate(x, y, ty)
    sim.set(
        "partCreate",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<Value> {
                let vec = args.into_vec();
                let mut state = borrow_bridge(&bridge)?;

                if vec.len() == 3 {
                    // sim.partCreate(x, y, ty) -> id | nil
                    let x = value_to_i64(&vec[0])? as i32;
                    let y = value_to_i64(&vec[1])? as i32;
                    let ty = value_to_i64(&vec[2])? as u32;

                    let id = state.with_engine(|engine| {
                        let xres = engine.bufs.config.xres as i32;
                        let yres = engine.bufs.config.yres as i32;
                        if x < 0 || y < 0 || x >= xres || y >= yres {
                            return None;
                        }
                        if ty == PT_NONE || ty as usize >= engine.registry.elements.len() {
                            return None;
                        }
                        refresh_pixel_occupant(engine, x, y);
                        engine.bufs.create_part(
                            x,
                            y,
                            ty,
                            CreationSource::Program,
                            0,
                            &engine.registry,
                        )
                    })?;

                    if id.is_some() {
                        state.particles_dirty = true;
                    }
                    Ok(match id {
                        Some(id) => Value::Integer(id as i64),
                        None => Value::Nil,
                    })
                } else if vec.len() >= 4 {
                    // sim.partCreate(newID, x, y, ty, [v]) -> id | -1
                    let new_id = value_to_i64(&vec[0])? as i32;
                    let x = value_to_i64(&vec[1])? as i32;
                    let y = value_to_i64(&vec[2])? as i32;
                    let ty = value_to_i64(&vec[3])? as u32;
                    let v = if vec.len() >= 5 {
                        Some(value_to_i64(&vec[4])? as i32)
                    } else {
                        None
                    };

                    let res = state.with_engine(|engine| {
                        let capacity = engine.bufs.capacity as i32;
                        if new_id >= capacity || new_id < -3 {
                            return -1i64;
                        }
                        if new_id >= 0 {
                            // The requested slot's live state decides whether
                            // it may be overwritten; pull just that slot.
                            engine.bufs.read_particle_slot(new_id as usize);
                            if engine.bufs.part_ty[new_id as usize] == PT_NONE {
                                return -1i64;
                            }
                        }
                        let xres = engine.bufs.config.xres as i32;
                        let yres = engine.bufs.config.yres as i32;
                        if x < 0 || y < 0 || x >= xres || y >= yres {
                            return -1i64;
                        }
                        if ty == PT_NONE || ty as usize >= engine.registry.elements.len() {
                            return -1i64;
                        }

                        refresh_pixel_occupant(engine, x, y);
                        if new_id >= 0 {
                            let slot = new_id as usize;
                            let old_x = engine.bufs.part_x[slot];
                            let old_y = engine.bufs.part_y[slot];
                            engine.bufs.part_ty[slot] = ty;
                            engine.bufs.part_x[slot] = x as f32;
                            engine.bufs.part_y[slot] = y as f32;
                            if let Some(val) = v {
                                engine.bufs.part_ctype[slot] = val as u32;
                            }
                            engine.bufs.mark_particle_slot_dirty(slot);
                            engine.bufs.retarget_particle_pixel(slot, old_x, old_y);
                            slot as i64
                        } else if let Some(slot) =
                            engine.bufs.allocate_particle(x, y, ty, &engine.registry)
                        {
                            if let Some(val) = v {
                                engine.bufs.part_ctype[slot] = val as u32;
                            }
                            slot as i64
                        } else {
                            -1i64
                        }
                    })?;

                    if res >= 0 {
                        state.particles_dirty = true;
                    }
                    Ok(Value::Integer(res))
                } else {
                    Err(mlua::Error::runtime("not enough arguments to partCreate"))
                }
            }
        })?,
    )?;

    // sim.partKill(id) or sim.partKill(x, y)
    sim.set(
        "partKill",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<()> {
                let vec = args.into_vec();
                let mut state = borrow_bridge(&bridge)?;
                if vec.len() == 1 {
                    let id = value_to_i64(&vec[0])?;
                    state.with_engine(|engine| {
                        if id >= 0 && (id as usize) < engine.bufs.capacity {
                            engine.bufs.read_particle_slot(id as usize);
                            engine.bufs.kill_particle(id as usize);
                        }
                    })?;
                    state.particles_dirty = true;
                } else if vec.len() >= 2 {
                    let x = value_to_f64(&vec[0])? as i32;
                    let y = value_to_f64(&vec[1])? as i32;
                    state.with_engine(|engine| {
                        refresh_pixel_occupant(engine, x, y);
                        engine.bufs.kill_at(x, y);
                    })?;
                    state.particles_dirty = true;
                }
                Ok(())
            }
        })?,
    )?;

    // sim.partCount() -> integer
    sim.set(
        "partCount",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, ()| -> mlua::Result<i64> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    refresh_particle_count(&mut engine.bufs);
                    engine.particle_count() as i64
                })
            }
        })?,
    )?;

    // sim.partExists(id) -> bool
    sim.set(
        "partExists",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, id: i64| -> mlua::Result<bool> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if id < 0 || id as usize >= engine.bufs.capacity {
                        return false;
                    }
                    let slot = id as usize;
                    engine.bufs.read_particle_slot(slot);
                    engine.bufs.part_ty[slot] != PT_NONE
                })
            }
        })?,
    )?;

    // sim.partChangeType(id, new_type)
    sim.set(
        "partChangeType",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (id, new_type): (i64, u32)| -> mlua::Result<()> {
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    if id >= 0 && (id as usize) < engine.bufs.capacity {
                        let slot = id as usize;
                        engine.bufs.read_particle_slot(slot);
                        if engine.bufs.part_ty[slot] != PT_NONE {
                            if new_type == PT_NONE {
                                engine.bufs.kill_particle(slot);
                            } else if (new_type as usize) < engine.registry.elements.len() {
                                let old_x = engine.bufs.part_x[slot];
                                let old_y = engine.bufs.part_y[slot];
                                engine.bufs.part_ty[slot] = new_type;
                                let props = engine.registry.get(new_type);
                                engine.bufs.part_life[slot] = props.default_life;
                                engine.bufs.part_ctype[slot] = props.default_ctype;
                                engine.bufs.part_tmp[slot] = props.default_tmp;
                                engine.bufs.retarget_particle_pixel(slot, old_x, old_y);
                                engine.bufs.mark_particle_slot_dirty(slot);
                            }
                        }
                    }
                })?;
                state.particles_dirty = true;
                Ok(())
            }
        })?,
    )?;

    // sim.partPosition(id, [x, y])
    sim.set(
        "partPosition",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<MultiValue> {
                let vec = args.into_vec();
                let mut state = borrow_bridge(&bridge)?;
                if vec.is_empty() {
                    return Ok(MultiValue::new());
                }
                let id = value_to_i64(&vec[0])?;

                if vec.len() >= 3 {
                    // Set position: sim.partPosition(id, x, y)
                    let new_x = value_to_f64(&vec[1])? as f32;
                    let new_y = value_to_f64(&vec[2])? as f32;
                    state.with_engine(|engine| {
                        if id >= 0 && (id as usize) < engine.bufs.capacity {
                            let slot = id as usize;
                            engine.bufs.read_particle_slot(slot);
                            if engine.bufs.part_ty[slot] != PT_NONE {
                                let old_x = engine.bufs.part_x[slot];
                                let old_y = engine.bufs.part_y[slot];
                                let xres = engine.bufs.config.xres.saturating_sub(1) as f32;
                                let yres = engine.bufs.config.yres.saturating_sub(1) as f32;
                                engine.bufs.part_x[slot] = new_x.clamp(0.0, xres);
                                engine.bufs.part_y[slot] = new_y.clamp(0.0, yres);
                                engine.bufs.retarget_particle_pixel(slot, old_x, old_y);
                                engine.bufs.mark_particle_slot_dirty(slot);
                            }
                        }
                    })?;
                    state.particles_dirty = true;
                    Ok(MultiValue::new())
                } else {
                    // Get position: sim.partPosition(id) -> x, y or nil, nil
                    let pos = state.with_engine(|engine| {
                        if id >= 0 && (id as usize) < engine.bufs.capacity {
                            let slot = id as usize;
                            engine.bufs.read_particle_slot(slot);
                            if engine.bufs.part_ty[slot] != PT_NONE {
                                return Some((engine.bufs.part_x[slot], engine.bufs.part_y[slot]));
                            }
                        }
                        None
                    })?;
                    Ok(match pos {
                        Some((x, y)) => pair(x, y),
                        None => {
                            let mut vals = MultiValue::new();
                            vals.push_back(Value::Nil);
                            vals.push_back(Value::Nil);
                            vals
                        }
                    })
                }
            }
        })?,
    )?;

    // sim.partID(x, y) -> id | nil
    sim.set(
        "partID",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (x, y): (f64, f64)| -> mlua::Result<Value> {
                let mut state = borrow_bridge(&bridge)?;
                let id = state.with_engine(|engine| {
                    let px = x as i32;
                    let py = y as i32;
                    refresh_map_rect(&mut engine.bufs, px, py, px, py);
                    engine.bufs.particle_index_at(px, py)
                })?;
                Ok(match id {
                    Some(id) => Value::Integer(id as i64),
                    None => Value::Nil,
                })
            }
        })?,
    )?;

    // sim.partNeighbors(x, y, r, [t]) -> table
    sim.set(
        "partNeighbors",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, (x, y, r, t): (i32, i32, i32, Option<u32>)| -> mlua::Result<Table> {
                let mut state = borrow_bridge(&bridge)?;
                let ids = state.with_engine(|engine| {
                    let mut result = Vec::new();
                    let xres = engine.bufs.config.xres as i32;
                    let yres = engine.bufs.config.yres as i32;
                    refresh_map_rect(&mut engine.bufs, x - r, y - r, x + r, y + r);
                    for ry in -r..=r {
                        for rx in -r..=r {
                            if rx == 0 && ry == 0 {
                                continue;
                            }
                            let px = x + rx;
                            let py = y + ry;
                            if px >= 0
                                && py >= 0
                                && px < xres
                                && py < yres
                                && let Some(slot) = engine.bufs.particle_index_at(px, py)
                            {
                                // Only the candidates' types are needed, so
                                // each hit pulls its own slot instead of
                                // the whole pool.
                                engine.bufs.read_particle_slot(slot);
                                let ty = engine.bufs.part_ty[slot];
                                if ty != PT_NONE && t.is_none_or(|target| ty == target) {
                                    result.push(slot as i64);
                                }
                            }
                        }
                    }
                    result
                })?;

                let table = lua.create_table()?;
                for (i, id) in ids.into_iter().enumerate() {
                    table.raw_set(i + 1, id)?;
                }
                Ok(table)
            }
        })?,
    )?;

    // sim.parts() -> iterator
    sim.set(
        "parts",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, ()| -> mlua::Result<mlua::Function> {
                // The iterator walks every slot, so the pool has to be current
                // once; a per-slot read per `next()` would be far worse.
                {
                    let mut state = borrow_bridge(&bridge)?;
                    state.with_engine(|engine| refresh_particles(&mut engine.bufs))?;
                }
                let cursor = Rc::new(RefCell::new(0usize));
                let bridge = Rc::clone(&bridge);
                lua.create_function(move |_, ()| -> mlua::Result<Value> {
                    let mut cur = cursor.borrow_mut();
                    let mut state = borrow_bridge(&bridge)?;
                    state.with_engine(|engine| {
                        let cap = engine.bufs.capacity;
                        while *cur < cap {
                            let slot = *cur;
                            *cur += 1;
                            if engine.bufs.part_ty[slot] != PT_NONE {
                                return Value::Integer(slot as i64);
                            }
                        }
                        Value::Nil
                    })
                })
            }
        })?,
    )?;

    // sim.neighbors(cx, cy, [rx=2], [ry=2], [t=PT_NONE]) -> iterator
    sim.set(
        "neighbors",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |lua, args: MultiValue| -> mlua::Result<mlua::Function> {
                let vec = args.into_vec();
                if vec.len() < 2 {
                    return Err(mlua::Error::runtime("expected cx, cy"));
                }
                let cx = value_to_i64(&vec[0])? as i32;
                let cy = value_to_i64(&vec[1])? as i32;
                let rx = if vec.len() >= 3 {
                    value_to_i64(&vec[2])? as i32
                } else {
                    2
                };
                let ry = if vec.len() >= 4 {
                    value_to_i64(&vec[3])? as i32
                } else {
                    2
                };
                let t = if vec.len() >= 5 {
                    value_to_i64(&vec[4])? as u32
                } else {
                    PT_NONE
                };

                {
                    let mut state = borrow_bridge(&bridge)?;
                    state.with_engine(|engine| {
                        let xres = engine.bufs.config.xres as i32;
                        let yres = engine.bufs.config.yres as i32;
                        if cx < 0 || cy < 0 || cx >= xres || cy >= yres {
                            return Err(mlua::Error::runtime("Invalid position"));
                        }
                        if rx < 0 || ry < 0 || rx >= xres || ry >= yres {
                            return Err(mlua::Error::runtime("Invalid radius"));
                        }
                        Ok(())
                    })??;
                }

                let cur_y = Rc::new(RefCell::new((cy - ry).max(0)));
                let cur_x = Rc::new(RefCell::new((cx - rx).max(0)));
                let map_row = Rc::new(RefCell::new(i32::MIN));
                let bridge = Rc::clone(&bridge);

                lua.create_function(move |_, ()| -> mlua::Result<MultiValue> {
                    let mut x = cur_x.borrow_mut();
                    let mut y = cur_y.borrow_mut();
                    let mut read_row = map_row.borrow_mut();
                    let mut state = borrow_bridge(&bridge)?;

                    state.with_engine(|engine| {
                        let xres = engine.bufs.config.xres as i32;
                        let yres = engine.bufs.config.yres as i32;
                        while *y <= cy + ry && *y < yres {
                            let px = *x;
                            let py = *y;
                            // One row of the map at a time: the iterator may be
                            // drained across several `next()` calls and never
                            // needs the pixels it has already passed.
                            if py != *read_row {
                                refresh_map_rect(&mut engine.bufs, cx - rx, py, cx + rx, py);
                                *read_row = py;
                            }
                            *x += 1;
                            if *x > cx + rx || *x >= xres {
                                *x = (cx - rx).max(0);
                                *y += 1;
                            }
                            if px == cx && py == cy {
                                continue;
                            }
                            if let Some(slot) = engine.bufs.particle_index_at(px, py) {
                                engine.bufs.read_particle_slot(slot);
                                let ty = engine.bufs.part_ty[slot];
                                if ty != PT_NONE && (t == PT_NONE || ty == t) {
                                    let mut vals = MultiValue::new();
                                    vals.push_back(Value::Integer(slot as i64));
                                    vals.push_back(Value::Integer(px as i64));
                                    vals.push_back(Value::Integer(py as i64));
                                    return vals;
                                }
                            }
                        }
                        MultiValue::new()
                    })
                })
            }
        })?,
    )?;

    // sim.elementCount([elem=0]) -> integer
    sim.set(
        "elementCount",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, elem: Option<i64>| -> mlua::Result<i64> {
                let elem = elem.unwrap_or(0);
                if elem < 0 || elem >= PT_NUM as i64 {
                    return Err(mlua::Error::runtime(format!("Invalid element ID ({elem})")));
                }
                let target = elem as u32;
                let mut state = borrow_bridge(&bridge)?;
                state.with_engine(|engine| {
                    refresh_particles(&mut engine.bufs);
                    let mut count = 0i64;
                    for slot in 0..engine.bufs.capacity {
                        if engine.bufs.part_ty[slot] == target {
                            count += 1;
                        }
                    }
                    count
                })
            }
        })?,
    )?;

    // sim.canMove(from, to, [setting]) -> integer | ()
    sim.set(
        "canMove",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, args: MultiValue| -> mlua::Result<Value> {
                let vec = args.into_vec();
                if vec.len() < 2 {
                    return Err(mlua::Error::runtime(
                        "expected movingElement and destinationElement",
                    ));
                }
                let from = value_to_i64(&vec[0])?;
                let to = value_to_i64(&vec[1])?;
                if from < 0 || from >= PT_NUM as i64 {
                    return Err(mlua::Error::runtime(format!("Invalid element ID ({from})")));
                }
                if to < 0 || to >= PT_NUM as i64 {
                    return Err(mlua::Error::runtime(format!("Invalid element ID ({to})")));
                }
                let idx = (from as usize) * PT_NUM + (to as usize);
                let mut state = borrow_bridge(&bridge)?;

                if vec.len() >= 3 {
                    let setting = value_to_i64(&vec[2])? as u32;
                    state.with_engine(|engine| {
                        if idx < engine.bufs.can_move_host.len() {
                            engine.bufs.can_move_host[idx] = setting & 0x7F;
                        }
                    })?;
                    Ok(Value::Nil)
                } else {
                    let val = state.with_engine(|engine| {
                        if idx < engine.bufs.can_move_host.len() {
                            engine.bufs.can_move_host[idx] as i64
                        } else {
                            0i64
                        }
                    })?;
                    Ok(Value::Integer(val))
                }
            }
        })?,
    )?;

    // sim.spawnRigidBox(x, y, half_w, half_h, elem_ty) -> rigid_id
    sim.set(
        "spawnRigidBox",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_,
                  (x, y, half_w, half_h, elem_ty): (f32, f32, i32, i32, u32)|
                  -> mlua::Result<u32> {
                let mut state = borrow_bridge(&bridge)?;
                let id = state.with_engine(|engine| {
                    engine.spawn_rigid_body(
                        x,
                        y,
                        half_w,
                        half_h,
                        tpt_core::commands::RigidSpawnShape::Box,
                        elem_ty,
                    )
                })?;
                state.particles_dirty = true;
                Ok(id)
            }
        })?,
    )?;

    Ok(())
}
