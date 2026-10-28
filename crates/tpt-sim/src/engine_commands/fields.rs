//! Per-particle field and property editing commands, plus the single-slot read
//! used by Lua's per-particle getters and by [`SimulationEngine::set_particle_property`].

use cubecl::bytes::Bytes;
use tpt_core::commands::ParticleField;
use tpt_core::constants::*;

use crate::SimulationEngine;
use crate::buffers::{FULL_PARTICLE_UPLOAD, SimBuffers};

/// Reads word `index` of a device read-back, tolerating short/failed replies.
fn read_word(readback: &[Bytes], index: usize) -> u32 {
    let Some(bytes) = readback.get(index) else {
        return 0;
    };
    let raw: &[u8] = bytes;
    match raw.get(..4) {
        Some(word) => u32::from_le_bytes([word[0], word[1], word[2], word[3]]),
        None => 0,
    }
}

impl SimBuffers {
    /// Pulls the 14 fields of particle slot `id` back from the device.
    ///
    /// Lua addresses particles by slot (`sim.partProperty`, `sim.partPosition`,
    /// `sim.partExists`), so the pixel-rectangle read ([`Self::read_map_rect`])
    /// cannot serve them and the whole-pool read
    /// ([`Self::sync_particles_gpu_to_host`]) is far too much for one field.
    /// Only the requested slot crosses the bus; the pool bookkeeping
    /// (`free_list`/`next_slot`/`active_count`) is deliberately left alone,
    /// because it is only meaningful once a full read-back has run
    /// [`Self::reconcile_particle_pool`].
    ///
    /// A pending host edit of the same slot wins: it has not been uploaded yet,
    /// so the device still holds the pre-edit values and reading it back would
    /// undo the script's own write.
    ///
    /// Returns `false` when the slot is out of range or the read failed; the
    /// caller then observes the previous host values.
    pub fn read_particle_slot(&mut self, id: usize) -> bool {
        if id >= self.capacity {
            return false;
        }
        // Nothing ran on the device since the last read-back (or every slot was
        // uploaded), so the host shadow is already authoritative for this slot.
        if !self.gpu_particles_dirty {
            return true;
        }
        if self.host_particles_dirty
            && (self.host_dirty_slots.contains(&FULL_PARTICLE_UPLOAD)
                || self.host_dirty_slots.contains(&(id as u32)))
        {
            return true;
        }

        let word = core::mem::size_of::<u32>();
        let offset = |handle: cubecl::server::Handle| {
            handle
                .offset_start((id * word) as u64)
                .offset_end(((self.capacity - id - 1) * word) as u64)
        };
        let handles = vec![
            offset(self.gpu_part_ty.clone()),
            offset(self.gpu_part_life.clone()),
            offset(self.gpu_part_ctype.clone()),
            offset(self.gpu_part_x.clone()),
            offset(self.gpu_part_y.clone()),
            offset(self.gpu_part_vx.clone()),
            offset(self.gpu_part_vy.clone()),
            offset(self.gpu_part_temp.clone()),
            offset(self.gpu_part_flags.clone()),
            offset(self.gpu_part_tmp.clone()),
            offset(self.gpu_part_tmp2.clone()),
            offset(self.gpu_part_tmp3.clone()),
            offset(self.gpu_part_tmp4.clone()),
            offset(self.gpu_part_dcolour.clone()),
        ];
        let Ok(readback) = self.client.read(handles) else {
            return false;
        };
        self.part_ty[id] = read_word(&readback, 0);
        self.part_life[id] = read_word(&readback, 1) as i32;
        self.part_ctype[id] = read_word(&readback, 2);
        self.part_x[id] = f32::from_bits(read_word(&readback, 3));
        self.part_y[id] = f32::from_bits(read_word(&readback, 4));
        self.part_vx[id] = f32::from_bits(read_word(&readback, 5));
        self.part_vy[id] = f32::from_bits(read_word(&readback, 6));
        self.part_temp[id] = f32::from_bits(read_word(&readback, 7));
        self.part_flags[id] = read_word(&readback, 8);
        self.part_tmp[id] = read_word(&readback, 9) as i32;
        self.part_tmp2[id] = read_word(&readback, 10) as i32;
        self.part_tmp3[id] = read_word(&readback, 11) as i32;
        self.part_tmp4[id] = read_word(&readback, 12) as i32;
        self.part_dcolour[id] = read_word(&readback, 13);
        true
    }

    /// Moves the host `pmap`/`photons` word owned by slot `id` from
    /// `(old_x, old_y)` to the slot's current `part_x`/`part_y`.
    ///
    /// Slot-scoped edits keep the host pixel map coherent without
    /// [`Self::rebuild_spatial_maps`], which would remap the whole pool from
    /// host data that may lag the device. The device rebuilds its own maps from
    /// the particle arrays every frame, so only the host shadow needs this patch
    /// for Lua's pixel getters in the same call.
    pub fn retarget_particle_pixel(&mut self, id: usize, old_x: f32, old_y: f32) {
        if id >= self.capacity {
            return;
        }
        let old_pixel = self.pixel_index(old_x.round() as i32, old_y.round() as i32);
        if let Some(pixel) = old_pixel {
            if pmap_id(self.pmap[pixel]) == Some(id as u32) {
                self.pmap[pixel] = 0;
            }
            if pmap_id(self.photons[pixel]) == Some(id as u32) {
                self.photons[pixel] = 0;
            }
        }
        let ty = self.part_ty[id];
        if ty == PT_NONE {
            return;
        }
        let new_x = self.part_x[id].round() as i32;
        let new_y = self.part_y[id].round() as i32;
        let Some(pixel) = self.pixel_index(new_x, new_y) else {
            return;
        };
        let packed = pmap_pack(id as u32, ty);
        if self.is_energy_type(ty) {
            self.photons[pixel] = packed;
        } else {
            self.pmap[pixel] = packed;
        }
    }
}

impl SimulationEngine {
    /// Rewrite one integer field of every particle under the brushed pixels.
    ///
    /// Used by the property tool (`tmp`/`tmp2`/`life`/`ctype`) and the
    /// decoration tool (`dcolour`). Writing a single slot only uploads that
    /// slot, so a stroke never rewinds the rest of the pool.
    pub(super) fn set_particle_field_at(
        &mut self,
        points: &[(i32, i32)],
        field: ParticleField,
        value: i64,
    ) {
        if points.is_empty() {
            return;
        }
        // Device-side write: no read-back, no per-slot re-upload, so a stroke
        // cannot rewind `temp`/`vx`/`tmp` of the untouched pool.
        crate::edit_kernel::set_particle_field(&mut self.bufs, points, field, value);
    }

    /// Set one numeric property of a particle (Lua `partProperty` / debug tool).
    pub(super) fn set_particle_property(&mut self, part_id: u32, field: &str, value: f32) {
        let id = part_id as usize;
        if id >= self.bufs.capacity {
            return;
        }
        // A targeted edit must start from current data, but only for this slot:
        // the device may have run ahead of the host shadow since the last frame
        // and a whole-pool read-back here would dominate a property stroke.
        self.bufs.read_particle_slot(id);
        if self.bufs.part_ty[id] == PT_NONE {
            return;
        }
        let old_x = self.bufs.part_x[id];
        let old_y = self.bufs.part_y[id];
        let mut moved = false;
        match field.to_ascii_lowercase().as_str() {
            "temp" | "temperature" => {
                self.bufs.part_temp[id] = value.clamp(MIN_TEMP, MAX_TEMP);
            }
            "life" => self.bufs.part_life[id] = value as i32,
            "ctype" => self.bufs.part_ctype[id] = value.max(0.0) as u32,
            "vx" => self.bufs.part_vx[id] = value,
            "vy" => self.bufs.part_vy[id] = value,
            "x" => {
                self.bufs.part_x[id] = value;
                moved = true;
            }
            "y" => {
                self.bufs.part_y[id] = value;
                moved = true;
            }
            "type" | "ty" => {
                let ty = value.max(0.0) as u32;
                if (ty as usize) < self.registry.elements.len() {
                    self.bufs.part_ty[id] = ty;
                    moved = true;
                }
            }
            "dcolour" | "dcolor" => self.bufs.part_dcolour[id] = value.max(0.0) as u32,
            "flags" => self.bufs.part_flags[id] = value.max(0.0) as u32,
            "tmp" => self.bufs.part_tmp[id] = value as i32,
            "tmp2" => self.bufs.part_tmp2[id] = value as i32,
            _ => return,
        }
        if moved {
            // Only this slot's pixel-map word changes; the device rebuilds its
            // maps from the particle arrays every frame.
            self.bufs.retarget_particle_pixel(id, old_x, old_y);
        }
        self.bufs.mark_particle_slot_dirty(id);
    }
}
