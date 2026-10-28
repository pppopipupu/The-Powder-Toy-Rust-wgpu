//! Undo/redo snapshots, saves and stamps.

use mlua::{Lua, MultiValue, Table, Value};
use std::cell::RefCell;
use std::rc::Rc;
use tpt_core::constants::PT_NONE;
use tpt_save::{SaveData, UndoHistory};

use crate::api::{pair, refresh_map_rect, value_to_i64};
use crate::bridge::{Bridge, borrow_bridge};

struct StampEntry {
    name: String,
    data: SaveData,
    _origin: (i32, i32),
}

struct SaveState {
    undo: UndoHistory,
    stamps: Vec<StampEntry>,
    next_stamp_id: usize,
    initial_save: Option<SaveData>,
}

pub(crate) fn register_saves(lua: &Lua, bridge: &Bridge, sim: &Table) -> mlua::Result<()> {
    let save_state = Rc::new(RefCell::new(SaveState {
        undo: UndoHistory::new(16),
        stamps: Vec::new(),
        next_stamp_id: 1,
        initial_save: None,
    }));

    // sim.takeSnapshot()
    sim.set(
        "takeSnapshot",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            let state_ref = Rc::clone(&save_state);
            move |_, ()| -> mlua::Result<()> {
                let mut bridge_state = borrow_bridge(&bridge)?;
                bridge_state.with_engine(|engine| {
                    state_ref.borrow_mut().undo.push_snapshot(engine);
                })?;
                Ok(())
            }
        })?,
    )?;

    // sim.historyRestore() -> bool
    sim.set(
        "historyRestore",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            let state_ref = Rc::clone(&save_state);
            move |_, ()| -> mlua::Result<bool> {
                let mut bridge_state = borrow_bridge(&bridge)?;
                let res =
                    bridge_state.with_engine(|engine| state_ref.borrow_mut().undo.undo(engine))?;
                bridge_state.particles_dirty = true;
                Ok(res)
            }
        })?,
    )?;

    // sim.historyForward() -> bool
    sim.set(
        "historyForward",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            let state_ref = Rc::clone(&save_state);
            move |_, ()| -> mlua::Result<bool> {
                let mut bridge_state = borrow_bridge(&bridge)?;
                let res =
                    bridge_state.with_engine(|engine| state_ref.borrow_mut().undo.redo(engine))?;
                bridge_state.particles_dirty = true;
                Ok(res)
            }
        })?,
    )?;

    // sim.saveStamp(x, y, w, h, [includePressure=1]) -> string
    sim.set(
        "saveStamp",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            let state_ref = Rc::clone(&save_state);
            move |_, args: MultiValue| -> mlua::Result<String> {
                let vec = args.into_vec();
                let x = if !vec.is_empty() {
                    value_to_i64(&vec[0])? as i32
                } else {
                    0
                };
                let y = if vec.len() >= 2 {
                    value_to_i64(&vec[1])? as i32
                } else {
                    0
                };
                let w = if vec.len() >= 3 {
                    value_to_i64(&vec[2])? as i32
                } else {
                    611
                };
                let h = if vec.len() >= 4 {
                    value_to_i64(&vec[3])? as i32
                } else {
                    383
                };

                let mut bridge_state = borrow_bridge(&bridge)?;
                let stamp_data = bridge_state.with_engine(SaveData::capture_from)?;

                let mut st = state_ref.borrow_mut();
                let name = format!("stamp_{:04x}", st.next_stamp_id);
                st.next_stamp_id += 1;

                // Keep only particles within the bounds [x..x+w, y..y+h]
                let mut filtered_parts = Vec::new();
                for p in &stamp_data.particles {
                    let px = p.x as i32;
                    let py = p.y as i32;
                    if px >= x && px <= x + w && py >= y && py <= y + h {
                        let mut p_copy = *p;
                        p_copy.x -= x as f32;
                        p_copy.y -= y as f32;
                        filtered_parts.push(p_copy);
                    }
                }

                let mut stamp = stamp_data;
                stamp.particles = filtered_parts;

                st.stamps.push(StampEntry {
                    name: name.clone(),
                    data: stamp,
                    _origin: (x, y),
                });

                Ok(name)
            }
        })?,
    )?;

    // sim.loadStamp(stamp, [x, y, hflip, rotation, includePressure]) -> 1 | (nil, error)
    sim.set(
        "loadStamp",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            let state_ref = Rc::clone(&save_state);
            move |lua, args: MultiValue| -> mlua::Result<MultiValue> {
                let vec = args.into_vec();
                if vec.is_empty() {
                    let mut vals = MultiValue::new();
                    vals.push_back(Value::Nil);
                    vals.push_back(Value::String(lua.create_string("does not exist")?));
                    return Ok(vals);
                }

                let target_x = if vec.len() >= 2 {
                    value_to_i64(&vec[1])? as i32
                } else {
                    0
                };
                let target_y = if vec.len() >= 3 {
                    value_to_i64(&vec[2])? as i32
                } else {
                    0
                };

                let st = state_ref.borrow();
                let stamp_entry = match &vec[0] {
                    Value::String(s) => {
                        let name = s.to_str()?;
                        st.stamps.iter().find(|e| e.name == *name)
                    }
                    Value::Integer(i) => {
                        let idx = *i as usize;
                        st.stamps.get(idx)
                    }
                    _ => None,
                };

                let Some(entry) = stamp_entry else {
                    let mut vals = MultiValue::new();
                    vals.push_back(Value::Nil);
                    vals.push_back(Value::String(lua.create_string("does not exist")?));
                    return Ok(vals);
                };

                let particles = entry.data.particles.clone();
                drop(st);

                let mut bridge_state = borrow_bridge(&bridge)?;
                bridge_state.with_engine(|engine| {
                    let xres = engine.bufs.config.xres as i32;
                    let yres = engine.bufs.config.yres as i32;

                    // Occupancy checks and the per-slot map-word updates need
                    // the destination pixels current, not the whole pool.
                    if let Some((min_x, min_y, max_x, max_y)) =
                        particles
                            .iter()
                            .fold(None::<(i32, i32, i32, i32)>, |bounds, p| {
                                let (x, y) = (target_x + p.x as i32, target_y + p.y as i32);
                                Some(match bounds {
                                    Some((x0, y0, x1, y1)) => {
                                        (x0.min(x), y0.min(y), x1.max(x), y1.max(y))
                                    }
                                    None => (x, y, x, y),
                                })
                            })
                    {
                        refresh_map_rect(&mut engine.bufs, min_x, min_y, max_x, max_y);
                    }

                    for p in particles {
                        let nx = target_x + p.x as i32;
                        let ny = target_y + p.y as i32;
                        if nx >= 0
                            && ny >= 0
                            && nx < xres
                            && ny < yres
                            && p.ty != PT_NONE
                            && let Some(slot) =
                                engine
                                    .bufs
                                    .allocate_particle(nx, ny, p.ty, &engine.registry)
                        {
                            engine.bufs.part_life[slot] = p.life;
                            engine.bufs.part_ctype[slot] = p.ctype;
                            engine.bufs.part_temp[slot] = p.temp;
                            engine.bufs.part_vx[slot] = p.vx;
                            engine.bufs.part_vy[slot] = p.vy;
                            engine.bufs.part_dcolour[slot] = p.dcolour;
                            engine.bufs.part_tmp[slot] = p.tmp;
                            engine.bufs.part_tmp2[slot] = p.tmp2;
                            engine.bufs.part_tmp3[slot] = p.tmp3;
                            engine.bufs.part_tmp4[slot] = p.tmp4;
                            engine.bufs.part_flags[slot] = p.flags;
                            engine.bufs.mark_particle_slot_dirty(slot);
                        }
                    }
                })?;

                bridge_state.particles_dirty = true;

                let mut vals = MultiValue::new();
                vals.push_back(Value::Integer(1));
                Ok(vals)
            }
        })?,
    )?;

    // sim.deleteStamp(stamp) -> 0 | -1
    sim.set(
        "deleteStamp",
        lua.create_function({
            let state_ref = Rc::clone(&save_state);
            move |_, arg: Value| -> mlua::Result<i64> {
                let mut st = state_ref.borrow_mut();
                match arg {
                    Value::String(s) => {
                        let name = s.to_str()?;
                        if let Some(idx) = st.stamps.iter().position(|e| e.name == *name) {
                            st.stamps.remove(idx);
                            Ok(0)
                        } else {
                            Ok(-1)
                        }
                    }
                    Value::Integer(i) => {
                        let idx = i as usize;
                        if idx < st.stamps.len() {
                            st.stamps.remove(idx);
                            Ok(0)
                        } else {
                            Ok(-1)
                        }
                    }
                    _ => Ok(-1),
                }
            }
        })?,
    )?;

    // sim.listStamps() -> table
    sim.set(
        "listStamps",
        lua.create_function({
            let state_ref = Rc::clone(&save_state);
            move |lua, ()| -> mlua::Result<Table> {
                let st = state_ref.borrow();
                let table = lua.create_table()?;
                for (i, entry) in st.stamps.iter().enumerate() {
                    table.raw_set(i + 1, entry.name.clone())?;
                }
                Ok(table)
            }
        })?,
    )?;

    // sim.loadSave(saveID, [instant], [history])
    sim.set(
        "loadSave",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, (save_id, _, _): (Value, Option<i64>, Option<i64>)| -> mlua::Result<()> {
                let mut bridge_state = borrow_bridge(&bridge)?;
                let path_str = match &save_id {
                    Value::String(s) => s.to_str()?.to_string(),
                    Value::Integer(i) => format!("{i}.tptrs"),
                    _ => return Ok(()),
                };
                if let Ok(data) = SaveData::load_from_path(&path_str) {
                    bridge_state.with_engine(|engine| {
                        data.restore_into(engine);
                    })?;
                    bridge_state.particles_dirty = true;
                }
                Ok(())
            }
        })?,
    )?;

    // sim.reloadSave()
    sim.set(
        "reloadSave",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            let state_ref = Rc::clone(&save_state);
            move |_, ()| -> mlua::Result<()> {
                let mut bridge_state = borrow_bridge(&bridge)?;
                let st = state_ref.borrow();
                if let Some(data) = &st.initial_save {
                    let data = data.clone();
                    drop(st);
                    bridge_state.with_engine(|engine| {
                        data.restore_into(engine);
                    })?;
                    bridge_state.particles_dirty = true;
                } else {
                    drop(st);
                    bridge_state.with_engine(|engine| {
                        engine.clear();
                    })?;
                    bridge_state.particles_dirty = true;
                }
                Ok(())
            }
        })?,
    )?;

    // sim.getSaveID() -> (id, version) | nil
    sim.set(
        "getSaveID",
        lua.create_function({
            let bridge = Rc::clone(bridge);
            move |_, ()| -> mlua::Result<MultiValue> {
                let mut bridge_state = borrow_bridge(&bridge)?;
                let id_pair = bridge_state.with_engine(|engine| engine.current_save_id)?;
                Ok(match id_pair {
                    Some((id, ver)) => pair(id as f32, ver as f32),
                    None => MultiValue::new(),
                })
            }
        })?,
    )?;

    Ok(())
}
