//! Particle pool bookkeeping: slot allocation, freeing, import / export and the
//! full reset used by the save system.
//!
//! Live slots are sparse: a slot is free when `part_ty[id] == PT_NONE`, and
//! [`SimBuffers::reconcile_particle_pool`] rebuilds `free_list`, `active_count`
//! and `next_slot` from that invariant after a device read-back.

use tpt_core::constants::*;
use tpt_core::elements::{CtypeDraw, ElementRegistry};
use tpt_core::particle::{GpuParticle, GravityMode};

use crate::gol_kernel::{NGOL, gol_colours};

use super::SimBuffers;

/// `SPRK`'s life when it starts (`Simulation.cpp:1870`), also the refractory
/// cooldown a discharged conductor keeps (`SPRK.cpp:62-88`).
const SPARK_LIFE: i32 = 4;
/// `MAX_FIGHTERS` (`Stickman.h:3`): how many `FIGH` rigs a canvas may hold.
const MAX_FIGHTERS: usize = 100;
/// Bits the particle id occupies in a packed `pmap` word, i.e. where
/// `ctype`'s value payload starts (`Element::ctypeDrawVInCtype`).
const PMAP_ID_BITS: u32 = 9;
/// `PLNT_DIR` (`PLNT.h`): the bit offset of the plant's growth direction.
const PLNT_DIR_BITS: u32 = 3;

/// Which `create_part` entry point a creation came from, TPT's `p` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreationSource {
    /// `create_part(-1, ...)`: reactions, Lua and the flood fill. Refuses an
    /// occupied pixel unless `eval_move` lets the two share the cell, and a
    /// wall-blocked empty one (`Simulation.cpp:1901-1912`).
    Program,
    /// `create_part(-2, ...)`: the brush. Runs `CtypeDraw` on the occupant,
    /// refuses walls and sparks `INST` networks.
    Brush,
    /// `create_part(<occupant slot>, ...)`, i.e. `CreatePartFlags`'s
    /// `REPLACE_MODE`: the occupant is replaced without the `-2` checks. TPT
    /// reuses the occupant's slot; the port frees it and allocates again, so
    /// only the slot id can differ.
    Replace,
    /// `create_part(-3, ...)`: skips the occupancy, wall and layer checks and
    /// never takes a spark over a conductor.
    SkipChecks,
}

impl SimBuffers {
    /// Rebuild `free_list`, `active_count` and `next_slot` from `part_ty`.
    ///
    /// The free list is kept ascending so [`SimBuffers::allocate_particle`] hands
    /// out the lowest free slot first.
    pub fn reconcile_particle_pool(&mut self) {
        self.free_list.clear();
        let mut live = 0usize;
        let mut highest = 0usize;
        let scan = self.used_slot_bound();
        for id in 0..scan {
            if self.part_ty[id] == PT_NONE {
                if !(Self::GPU_SPAWN_START..Self::GPU_SPAWN_END).contains(&id)
                    && (self.part_flags[id] & FLAG_PORTAL_HELD) == 0
                {
                    self.free_list.push(id);
                }
            } else {
                live += 1;
                highest = id + 1;
            }
        }
        self.free_list.retain(|&id| id < highest);
        self.free_list.reverse();
        self.active_count = live;
        self.next_slot = highest;
    }

    /// Record that the host fields of particle slot `id` changed.
    ///
    /// The slot-scoped uploader (`upload_particles`) transfers only these
    /// sub-ranges, so a paint stroke never overwrites device slots it did not
    /// touch with a stale host shadow.
    #[inline]
    pub fn mark_particle_slot_dirty(&mut self, id: usize) {
        if id >= self.capacity {
            return;
        }
        self.host_dirty = true;
        self.host_particles_dirty = true;
        self.host_dirty_slots.push(id as u32);
    }

    /// Reserve room for `additional` new particles before a multi-allocation
    /// command (brush segment, flood fill, reaction spawn batch).
    ///
    /// Growing the pool reallocates every device handle, which requires a
    /// read-back when the device holds newer state; doing it once up front keeps
    /// that cost out of the per-particle allocation loop. Returns `true` when
    /// the pool can hold the requested number of particles.
    pub fn reserve_particles(&mut self, additional: usize) -> bool {
        let free = self.free_list.len() + self.capacity.saturating_sub(self.next_slot);
        if free >= additional {
            return true;
        }
        let wanted = self
            .active_count
            .saturating_add(additional)
            .min(Self::MAX_PARTICLE_CAPACITY);
        self.ensure_particle_capacity(wanted);
        let free = self.free_list.len() + self.capacity.saturating_sub(self.next_slot);
        free >= additional
    }

    /// Allocate a particle at pixel `(x, y)`.
    ///
    /// The slot is initialised from the element definition: `life`, `ctype`,
    /// `temp` and, for `PHOT`/`NEUT`/`ELEC` (unit travel direction) or
    /// `LAVA`/`FIRE`/`ACID` (small jitter), a velocity. Returns the slot id, or
    /// `None` when the position is out of bounds or the element is unknown.
    pub fn allocate_particle(
        &mut self,
        x: i32,
        y: i32,
        ty: u32,
        registry: &ElementRegistry,
    ) -> Option<usize> {
        if ty == PT_NONE || ty as usize >= registry.elements.len() {
            return None;
        }
        let pixel = self.pixel_index(x, y)?;
        // The caller is responsible for making the host maps current for the
        // pixels it is about to touch (`read_map_rect` or `ensure_host_current`).
        // A hidden full read-back here used to cost a device sync per brush
        // frame, which dominated the frame time.

        // Never orphan an occupant of the same layer: replacing a pixel kills it.
        let layer = if self.is_energy_type(ty) {
            self.photons[pixel]
        } else {
            self.pmap[pixel]
        };
        // A spark inherits the conductor it was placed on, so it can restore it
        // when it discharges; grab the type before the occupant is freed.
        let mut prev_ty = PT_NONE;
        if let Some(occupant) = pmap_id(layer) {
            let occupant = occupant as usize;
            if occupant < self.capacity {
                prev_ty = self.part_ty[occupant];
            }
            self.kill_particle(occupant);
        }

        if self.free_list.is_empty() && self.active_count == self.capacity {
            self.ensure_particle_capacity(self.capacity.saturating_mul(2).max(1));
        }
        let id = match self.free_list.pop() {
            Some(slot) => slot,
            None => {
                if self.next_slot >= Self::GPU_SPAWN_START && self.next_slot < Self::GPU_SPAWN_END {
                    self.next_slot = Self::GPU_SPAWN_END;
                }
                if self.next_slot >= self.capacity {
                    self.ensure_particle_capacity(self.capacity.saturating_mul(2).max(1));
                }
                if self.next_slot >= Self::GPU_SPAWN_START && self.next_slot < Self::GPU_SPAWN_END {
                    self.next_slot = Self::GPU_SPAWN_END;
                }
                if self.next_slot >= self.capacity {
                    // The pool is saturated at `MAX_PARTICLE_CAPACITY`; the pixel
                    // stays empty instead of indexing past the buffers.
                    return None;
                }
                let slot = self.next_slot;
                self.next_slot += 1;
                slot
            }
        };

        let def = registry.get(ty);
        let temp = if def.default_temp.abs() > f32::EPSILON {
            def.default_temp
        } else {
            DEFAULT_TEMP_K
        };
        let (vx, vy) = Self::initial_velocity(id as u32, x, y, ty);

        self.part_ty[id] = ty;
        self.part_life[id] = def.default_life;
        self.part_ctype[id] = def.default_ctype;
        self.part_x[id] = x as f32;
        self.part_y[id] = y as f32;
        self.part_vx[id] = vx;
        self.part_vy[id] = vy;
        self.part_temp[id] = temp;
        self.part_flags[id] = 0;
        self.part_tmp[id] = def.default_tmp;
        self.part_tmp2[id] = def.default_tmp2;
        self.part_tmp3[id] = def.default_tmp3;
        self.part_tmp4[id] = def.default_tmp4;
        self.part_dcolour[id] = 0;
        self.active_count += 1;

        if ty == PT_SPRK {
            // Sparks are short-lived and remember the conductor underneath so
            // the reaction kernel can settle them back into it.
            self.part_life[id] = SPARK_LIFE;
            if prev_ty != PT_NONE && (registry.get(prev_ty).properties & PROP_CONDUCTS) != 0 {
                self.part_ctype[id] = prev_ty;
            }
        }

        let packed = pmap_pack(id as u32, ty);
        if self.is_energy_type(ty) {
            self.photons[pixel] = packed;
        } else {
            self.pmap[pixel] = packed;
        }

        // Only this slot and the pixels it wrote need to reach the device; the
        // map word is uploaded from `host_dirty_slots` below. Any occupant this
        // allocation killed recorded its own slot already.
        self.mark_particle_slot_dirty(id);
        Some(id)
    }

    /// TPT `Simulation::create_part` (`Simulation.cpp:1843-1985`).
    ///
    /// `v` is the creation variable: the UI packs it into the tool id
    /// (`ElementTool.cpp:39`) and Lua passes it as `sim.partCreate`'s last
    /// argument. `FILT` and `TESC` store it, and `CtypeDraw`'s `ValueIn*` hooks
    /// splice it into the occupant's `ctype`/`tmp`.
    ///
    /// Returns the slot that holds the particle afterwards. A spark never
    /// allocates a slot: it takes over the conductor at the pixel and returns
    /// that particle (`Simulation.cpp:1852-1878`), which is also why painting
    /// `SPRK` on empty space or on a charged conductor does nothing.
    pub fn create_part(
        &mut self,
        x: i32,
        y: i32,
        ty: u32,
        source: CreationSource,
        v: i32,
        registry: &ElementRegistry,
    ) -> Option<usize> {
        if ty == PT_NONE || ty as usize >= registry.elements.len() {
            return None;
        }
        // The port's "Enable Cracker1000 mod elements" switch: a user placement
        // must not create a mod element while it is off. `SkipChecks` is the
        // save and region loader, which has to keep restoring particles that
        // already exist; their simulation is never gated.
        if source != CreationSource::SkipChecks
            && is_mod_element(ty)
            && !self.config.mod_elements_enabled
        {
            return None;
        }
        let pixel = self.pixel_index(x, y)?;
        let energy = self.is_energy_type(ty);

        if ty == PT_SPRK && source != CreationSource::SkipChecks {
            let occupant = pmap_id(self.pmap[pixel]).map(|id| id as usize);
            // `!(p == -2 && elements[TYP(pmap[y][x])].CtypeDraw)`: painting a
            // spark onto an element that records `ctype` writes the spark into
            // that particle instead of sparking it (the `CtypeDraw` branch
            // below).
            let brush_writes_ctype = source == CreationSource::Brush
                && occupant.is_some_and(|id| {
                    id < self.capacity
                        && registry.get(self.part_ty[id]).ctype_draw != CtypeDraw::None
                });
            if !brush_writes_ctype {
                let occ = occupant?;
                if occ >= self.capacity {
                    return None;
                }
                let occ_ty = self.part_ty[occ];
                if occ_ty == PT_NONE {
                    return None;
                }
                if occ_ty == PT_WIRE {
                    // `WIRE` records the painted element and never sparks.
                    self.part_ctype[occ] = PT_DUST;
                    self.mark_particle_slot_dirty(occ);
                    return Some(occ);
                }
                let conducts =
                    occ_ty == PT_INST || (registry.get(occ_ty).properties & PROP_CONDUCTS) != 0;
                // `parts[index].life != 0` refuses a conductor that is already
                // charged, so a second stroke over a live spark is a no-op.
                if !conducts || self.part_life[occ] != 0 {
                    return None;
                }
                if source == CreationSource::Brush && occ_ty == PT_INST {
                    self.flood_inst(x, y, registry);
                    return Some(occ);
                }
                self.part_ty[occ] = PT_SPRK;
                self.part_life[occ] = SPARK_LIFE;
                self.part_ctype[occ] = occ_ty;
                self.pmap[pixel] = pmap_pack(occ as u32, PT_SPRK);
                // `Simulation.cpp:1875-1877`: the good conductors heat up a
                // little when a charge starts, unless they are already hot.
                if self.part_temp[occ] + 10.0 < 673.0
                    && matches!(
                        occ_ty,
                        PT_METL
                            | PT_BMTL
                            | PT_BRMT
                            | PT_PSCN
                            | PT_NSCN
                            | PT_ETRD
                            | PT_NBLE
                            | PT_IRON
                    )
                {
                    self.part_temp[occ] += 10.0;
                }
                self.mark_particle_slot_dirty(occ);
                return Some(occ);
            }
        }

        if source == CreationSource::Replace {
            // `CreatePartFlags` calls `create_part(ID(occupant), ...)`, which
            // starts by clearing the old map words and resetting the slot.
            self.kill_at(x, y);
        }

        if source == CreationSource::Brush {
            if let Some(occ) = pmap_id(self.pmap[pixel]).map(|id| id as usize) {
                if occ < self.capacity {
                    self.ctype_draw(occ, ty, v, registry);
                }
                return None;
            }
            if self.is_wall_blocking(x, y, ty, registry) {
                return None;
            }
            // `photons[y][x] && (elements[t].Properties & TYPE_ENERGY)`.
            if energy && self.photons[pixel] != 0 {
                return None;
            }
        }

        if source == CreationSource::Program {
            // `Simulation.cpp:1901-1912`: a programmatic creation refuses an
            // occupied pixel unless the occupant shares the cell with the new
            // particle, and refuses a wall-blocked empty one.
            if self.pmap[pixel] != 0 {
                if self.eval_move(ty, x, y, registry) != 2 {
                    return None;
                }
            } else if self.wall_at(x, y).is_some() && self.eval_move(ty, x, y, registry) == 0 {
                return None;
            }
        }

        if !self.create_allowed(ty) {
            return None;
        }

        let id = self.allocate_particle(x, y, ty, registry)?;
        self.run_create_hook(id, x, y, ty, v);
        Some(id)
    }

    /// `create_part`'s `CtypeDraw` dispatch (`Element.cpp:252-290`,
    /// `Simulation.cpp:1885-1893`).
    ///
    /// Returns `true` when the occupant took the painted element into its
    /// `ctype` (or `tmp`), which is what makes painting `FILT`, `CONV`, `VOID`
    /// or `CLNE` over an existing particle work at all.
    fn ctype_draw(&mut self, occ: usize, t: u32, v: i32, registry: &ElementRegistry) -> bool {
        let from = self.part_ty[occ];
        if from == PT_NONE || t as usize >= registry.elements.len() {
            return false;
        }
        let hook = registry.get(from).ctype_draw;
        if hook == CtypeDraw::None {
            return false;
        }
        // `Element::basicCtypeDraw` refuses a no-op paint and the elements that
        // opt out of `ctype` drawing with `PROP_NOCTYPEDRAW`.
        if from == t || (registry.get(t).properties & PROP_NOCTYPEDRAW) != 0 {
            return false;
        }
        let accepted = match hook {
            CtypeDraw::None | CtypeDraw::Basic | CtypeDraw::ValueInCtype => true,
            CtypeDraw::ValueInCtypeLigh => true,
            CtypeDraw::ValueInTmp => true,
            CtypeDraw::Cray => true,
            CtypeDraw::Stor => (registry.get(t).properties & TYPE_SOLID) == 0,
            CtypeDraw::Pcln => !matches!(t, PT_PSCN | PT_NSCN | PT_SPRK),
            CtypeDraw::Pstn => t != PT_FRME,
        };
        if !accepted {
            return false;
        }
        self.part_ctype[occ] = t;
        match hook {
            // `Element::ctypeDrawVInCtype` keeps the value bits above the id.
            CtypeDraw::ValueInCtype | CtypeDraw::ValueInCtypeLigh | CtypeDraw::Cray => {
                self.part_ctype[occ] |= (v as u32) << PMAP_ID_BITS;
            }
            // `Element::ctypeDrawVInTmp` stores the value in `tmp` for `LIFE`.
            CtypeDraw::ValueInTmp | CtypeDraw::Pcln if t == PT_LIFE => {
                self.part_tmp[occ] = v;
            }
            _ => {}
        }
        if hook == CtypeDraw::Cray {
            if t == PT_LIGH {
                self.part_ctype[occ] |= 30u32 << PMAP_ID_BITS;
            }
            self.part_temp[occ] = registry.get(t).default_temp;
        }
        // `CSNS.cpp:163-175` and `PROJ.cpp:115-124`: the ctype sensor and the
        // projectile accept a painted element like `ctypeDrawVInCtype`, and a
        // painted `LIGH` additionally gets `PMAPID(30)`, i.e. the bolt's
        // default 30-unit power in the bits above the element id.
        if hook == CtypeDraw::ValueInCtypeLigh && t == PT_LIGH {
            self.part_ctype[occ] |= 30u32 << PMAP_ID_BITS;
        }
        self.mark_particle_slot_dirty(occ);
        true
    }

    /// `Simulation::IsWallBlocking` (`Simulation.cpp:1076-1099`).
    fn is_wall_blocking(&self, x: i32, y: i32, ty: u32, registry: &ElementRegistry) -> bool {
        let Some(wall) = self.wall_at(x, y) else {
            return false;
        };
        let props = registry.get(ty).properties;
        match wall {
            WL_ALLOWGAS => props & TYPE_GAS == 0,
            WL_ALLOWENERGY => props & TYPE_ENERGY == 0,
            WL_ALLOWLIQUID => props & TYPE_LIQUID == 0,
            WL_ALLOWPOWDER => props & TYPE_PART == 0,
            WL_ALLOWAIR | WL_WALL | WL_WALLELEC => true,
            WL_EWALL => self.emap_at(x, y) == 0,
            WL_DETECT => props & TYPE_SOLID != 0,
            _ => false,
        }
    }

    /// `Simulation::eval_move` (`Simulation.cpp:1107-1194`): `0` when a `ty`
    /// particle cannot enter `(x, y)`, `1` when it swaps with the occupant and
    /// `2` when both share the cell.
    ///
    /// `create_part` uses it to decide whether a programmatic creation may
    /// replace an occupant; the runtime cases read the host shadow, which can
    /// trail the device by one tick.
    fn eval_move(&self, ty: u32, x: i32, y: i32, registry: &ElementRegistry) -> u32 {
        let Some(pixel) = self.pixel_index(x, y) else {
            return 0;
        };
        let occ = pmap_id(self.pmap[pixel]).map(|id| id as usize);
        let occ_ty = match occ {
            Some(id) if id < self.capacity => self.part_ty[id],
            _ => PT_NONE,
        };
        if ty as usize >= PT_NUM || occ_ty as usize >= PT_NUM {
            return 0;
        }
        let mut result = u32::from(registry.can_move_val(ty, occ_ty));
        if result == 3 {
            let id = occ.unwrap_or(0);
            result = match occ_ty {
                PT_LCRY if ty == PT_PHOT => {
                    if self.part_life[id] > 5 {
                        2
                    } else {
                        0
                    }
                }
                PT_GPMP if ty == PT_PHOT => {
                    if self.part_life[id] < 10 {
                        2
                    } else {
                        0
                    }
                }
                PT_INVIS => {
                    let resistance = if self.part_tmp[id] > 0 {
                        self.part_tmp[id] as f32
                    } else {
                        4.0
                    };
                    let pv = self.cell_index(x, y).map_or(0.0, |c| self.air_pv[c]);
                    if pv < -resistance || pv > resistance {
                        2
                    } else {
                        0
                    }
                }
                PT_PVOD => {
                    if self.part_life[id] != 10 {
                        0
                    } else if self.void_admits(id, ty) {
                        1
                    } else {
                        0
                    }
                }
                PT_VOID => {
                    if self.void_admits(id, ty) {
                        1
                    } else {
                        0
                    }
                }
                PT_SWCH if ty == PT_TRON => {
                    if self.part_life[id] >= 10 {
                        2
                    } else {
                        0
                    }
                }
                _ => 1,
            };
        }
        if self.is_wall_blocking(x, y, ty, registry) {
            return 0;
        }
        if self.wall_at(x, y) == Some(WL_EHOLE)
            && self.emap_at(x, y) == 0
            && (registry.get(ty).properties & TYPE_SOLID) == 0
            && (registry.get(occ_ty).properties & TYPE_SOLID) == 0
        {
            return 2;
        }
        result
    }

    /// `VOID.cpp:35-42` / `PVOD.cpp:44-51`: an untyped void eats everything,
    /// a typed one eats only the types its `tmp` filter selects.
    fn void_admits(&self, id: usize, ty: u32) -> bool {
        let filter = self.part_ctype[id];
        filter == 0 || ((filter == ty) != (self.part_tmp[id] & 1 != 0))
    }

    /// `Simulation::GetGravityField` (`Simulation.cpp:2088-2125`) with
    /// `particleGrav = 1` and `newtonGrav = 1`: the configured field direction
    /// at `(x, y)` plus the Newtonian force of the last solve.
    fn gravity_field(&self, x: i32, y: i32) -> (f32, f32) {
        let (mut gx, mut gy) = match self.config.gravity_mode {
            GravityMode::Vertical => (0.0, 1.0),
            GravityMode::Off => (0.0, 0.0),
            GravityMode::Radial => {
                let dx = (x - self.config.xres as i32 / 2) as f32;
                let dy = (y - self.config.yres as i32 / 2) as f32;
                if dx == 0.0 && dy == 0.0 {
                    (0.0, 0.0)
                } else {
                    let d = 0.01 - (dx * dx + dy * dy).sqrt();
                    (dx / d, dy / d)
                }
            }
            GravityMode::Custom => (self.config.custom_gravity_x, self.config.custom_gravity_y),
        };
        if self.config.newton_grav
            && let Some(cell) = self.cell_index(x, y)
        {
            gx += self.grav_fx[cell];
            gy += self.grav_fy[cell];
        }
        (gx, gy)
    }

    /// `Element::CreateAllowed` (`Simulation.cpp:1899-1903`).
    ///
    /// The counts come from the host shadow, so the check is exact for the
    /// brush (a stroke refreshes the shadow when it starts) and can only be
    /// stale for a particle the device created since the last read-back.
    fn create_allowed(&self, ty: u32) -> bool {
        match ty {
            // `SPAWN.cpp:50-53` / `SPAWN2.cpp`: one spawn point at a time.
            PT_SPAWN | PT_SPAWN2 => !self.any_live(ty),
            // `STKM.cpp:81-84`: a second stickman is refused while one lives.
            PT_STKM | PT_STKM2 => !self.any_live(PT_STKM) && !self.any_live(PT_STKM2),
            // `FIGH.cpp:175-178`.
            PT_FIGH => self.live_count(PT_FIGH) < MAX_FIGHTERS,
            _ => true,
        }
    }

    /// Number of live particles of element `ty` in the host shadow.
    fn live_count(&self, ty: u32) -> usize {
        let scan = self.used_slot_bound();
        (0..scan).filter(|&id| self.part_ty[id] == ty).count()
    }

    fn any_live(&self, ty: u32) -> bool {
        let scan = self.used_slot_bound();
        (0..scan).any(|id| self.part_ty[id] == ty)
    }

    /// TPT `Simulation::FloodINST` (`Simulation.cpp:586-700`): spark a whole
    /// `INST` wire network from one pixel.
    ///
    /// A span fill, not a BFS: each popped pixel is widened to the left and
    /// right over sparkable `INST` pixels, the whole span becomes sparks, and
    /// the rows above and below are queued. The one-pixel "wire crossing"
    /// exception (a vertical run passing over a horizontal one without
    /// connecting) is part of the layout, so it is ported literally.
    fn flood_inst(&mut self, x: i32, y: i32, registry: &ElementRegistry) -> bool {
        let xres = self.config.xres as i32;
        let yres = self.config.yres as i32;
        if x < 0 || y < 0 || x >= xres || y >= yres {
            return false;
        }
        let cell = CELL as i32;
        let sparkable = |bufs: &Self, x: i32, y: i32| -> bool {
            if x < 0 || y < 0 || x >= xres || y >= yres {
                return false;
            }
            let idx = (y as usize) * (xres as usize) + x as usize;
            match pmap_id(bufs.pmap[idx]) {
                Some(id) => {
                    let id = id as usize;
                    id < bufs.capacity && bufs.part_ty[id] == PT_INST && bufs.part_life[id] == 0
                }
                None => false,
            }
        };
        let is_inst = |bufs: &Self, x: i32, y: i32| -> bool {
            if x < 0 || y < 0 || x >= xres || y >= yres {
                return false;
            }
            let idx = (y as usize) * (xres as usize) + x as usize;
            match pmap_id(bufs.pmap[idx]) {
                Some(id) => {
                    let id = id as usize;
                    id < bufs.capacity
                        && (bufs.part_ty[id] == PT_INST
                            || (bufs.part_ty[id] == PT_SPRK && bufs.part_ctype[id] == PT_INST))
                }
                None => false,
            }
        };
        if !sparkable(self, x, y) {
            return false;
        }
        let mut created = false;
        let mut stack = vec![(x, y)];
        while let Some((mut cx, cy)) = stack.pop() {
            let mut x1 = cx;
            let mut x2 = cx;
            while x1 >= cell && sparkable(self, x1 - 1, cy) {
                x1 -= 1;
            }
            while x2 < xres - cell && sparkable(self, x2 + 1, cy) {
                x2 += 1;
            }
            cx = x1;
            while cx <= x2 {
                if self
                    .create_part(cx, cy, PT_SPRK, CreationSource::Program, 0, registry)
                    .is_some()
                {
                    created = true;
                }
                cx += 1;
            }

            if cy > cell
                && x1 == x2
                && is_inst(self, x1 - 1, cy - 1)
                && is_inst(self, x1, cy - 1)
                && is_inst(self, x1 + 1, cy - 1)
                && !is_inst(self, x1 - 1, cy - 2)
                && is_inst(self, x1, cy - 2)
                && !is_inst(self, x1 + 1, cy - 2)
            {
                // Travelling vertically up, skipping a horizontal line.
                if sparkable(self, x1, cy - 2) {
                    stack.push((x1, cy - 2));
                }
            } else if cy > cell {
                let mut sx = x1;
                while sx <= x2 {
                    if sparkable(self, sx, cy - 1)
                        && (sx == x1
                            || sx == x2
                            || cy >= yres - cell - 1
                            || !is_inst(self, sx, cy + 1)
                            || is_inst(self, sx + 1, cy + 1)
                            || is_inst(self, sx - 1, cy + 1))
                    {
                        stack.push((sx, cy - 1));
                    }
                    sx += 1;
                }
            }

            if cy < yres - cell - 1
                && x1 == x2
                && is_inst(self, x1 - 1, cy + 1)
                && is_inst(self, x1, cy + 1)
                && is_inst(self, x1 + 1, cy + 1)
                && !is_inst(self, x1 - 1, cy + 2)
                && is_inst(self, x1, cy + 2)
                && !is_inst(self, x1 + 1, cy + 2)
            {
                // Travelling vertically down, skipping a horizontal line.
                if sparkable(self, x1, cy + 2) {
                    stack.push((x1, cy + 2));
                }
            } else if cy < yres - cell - 1 {
                let mut sx = x1;
                while sx <= x2 {
                    if sparkable(self, sx, cy + 1)
                        && (sx == x1
                            || sx == x2
                            || cy < 0
                            || !is_inst(self, sx, cy - 1)
                            || is_inst(self, sx + 1, cy - 1)
                            || is_inst(self, sx - 1, cy - 1))
                    {
                        stack.push((sx, cy + 1));
                    }
                    sx += 1;
                }
            }
        }
        created
    }

    /// The element `Create` hooks (`ELEMENT_CREATE_FUNC_ARGS`).
    ///
    /// Each hook seeds a random timer, a packed genome or a pressure snapshot
    /// that a static [`ElementDef`] default cannot carry. The rolls come from
    /// [`hash3`] keyed on the pixel and slot, so replaying a stroke reproduces
    /// the same scene; the bounds are TPT's `rng.between` ranges.
    fn run_create_hook(&mut self, id: usize, x: i32, y: i32, ty: u32, v: i32) {
        let roll = hash3(x as u32, y as u32, id as u32);
        let roll2 = hash3(id as u32, y as u32, x as u32);
        // `int(sim->pv[y/CELL][x/CELL] * 64)`: the cell pressure snapshot the
        // glass-family hooks store in `tmp3`.
        let pv64 = (self.pressure_at(x, y) * 64.0) as i32;
        match ty {
            // `CFLM.cpp:69-72` / `PLSM.cpp:47-50`: `life = rng.between(50, 199)`.
            PT_HFLM | PT_PLSM => self.part_life[id] = 50 + (roll % 150) as i32,
            // `FIRE.cpp:64-67`: `life = rng.between(120, 169)`.
            PT_FIRE => self.part_life[id] = 120 + (roll % 50) as i32,
            // `LAVA.cpp:73-76`: `life = rng.between(240, 359)`.
            PT_LAVA => self.part_life[id] = 240 + (roll % 120) as i32,
            // `WARP.cpp:47-50`: `life = rng.between(70, 164)`.
            PT_WARP => self.part_life[id] = 70 + (roll % 95) as i32,
            // `SING.cpp:44-47`: `life = rng.between(60, 109)`.
            PT_SING => self.part_life[id] = 60 + (roll % 50) as i32,
            // `NEUT.cpp:252-258`: `life = rng.between(480, 959)`.
            PT_NEUT => self.part_life[id] = 480 + (roll % 480) as i32,
            // `ELEC.cpp:44-50`: `life = 680`, 2 px/frame at any angle.
            PT_ELEC => {
                self.part_life[id] = 680;
                let a = (roll % 360) as f32 * (core::f32::consts::PI / 180.0);
                self.part_vx[id] = 2.0 * a.cos();
                self.part_vy[id] = 2.0 * a.sin();
            }
            // `PROT.cpp:236-242`: `life = 680`, 2 px/frame in 10 degree steps.
            PT_PROT => {
                self.part_life[id] = 680;
                let a = (roll % 36) as f32 * 0.17453;
                self.part_vx[id] = 2.0 * a.cos();
                self.part_vy[id] = 2.0 * a.sin();
            }
            // `GRVT.cpp:44-50`: `life = 250..449`, 2 px/frame at any angle.
            PT_GRVT => {
                self.part_life[id] = 250 + (roll % 200) as i32;
                let a = (roll2 % 360) as f32 * (core::f32::consts::PI / 180.0);
                self.part_vx[id] = 2.0 * a.cos();
                self.part_vy[id] = 2.0 * a.sin();
            }
            // `PHOT.cpp:176-184`: 3 px/frame along one of eight directions, and
            // a `FILT` at the pixel tints the wavelength it starts with.
            PT_PHOT => {
                let a = (roll % 8) as f32 * core::f32::consts::FRAC_PI_4;
                self.part_vx[id] = 3.0 * a.cos();
                self.part_vy[id] = 3.0 * a.sin();
                if let Some(filt) = self
                    .pixel_index(x, y)
                    .and_then(|pixel| pmap_id(self.pmap[pixel]))
                    .map(|f| f as usize)
                    && filt < self.capacity
                    && filt != id
                    && self.part_ty[filt] == PT_FILT
                {
                    let wl = filt_interact_wavelengths(
                        self.part_tmp[filt],
                        self.part_ctype[filt],
                        self.part_temp[filt],
                        self.part_ctype[id],
                        roll,
                    );
                    self.part_ctype[id] = wl;
                }
            }
            // `LIGH.cpp:312-336`: `v` is the bolt power (the brush packs
            // `radiusX + radiusY` into it), `life * 150` is its temperature and
            // `tmp` aims the bolt along the gravity field: the field direction
            // plus up to 20 degrees of jitter, and a random direction when the
            // field is weaker than 0.04.
            PT_LIGH => {
                let life = if v >= 0 { v.min(55) } else { 30 };
                self.part_life[id] = life;
                self.part_temp[id] = life as f32 * 150.0;
                let (mut gx, mut gy) = self.gravity_field(x, y);
                let gsize = gx * gx + gy * gy;
                if gsize < 0.0016 {
                    let angle = (roll % 6283) as f32 * 0.001;
                    let weak = gsize.sqrt();
                    gx += angle.cos() * (0.04 - weak);
                    gy += angle.sin() * (0.04 - weak);
                }
                let aim = ((-gy).atan2(gx) * 180.0 / core::f32::consts::PI) as i32;
                let jitter = ((roll / 6283) % 41) as i32 - 20;
                self.part_tmp[id] = (aim + jitter + 360) % 360;
                self.part_tmp2[id] = 4;
            }
            // `LIFE.cpp:93-119`: `ctype` is the ruleset (a builtin index or a
            // packed custom rule), `dcolour`/`tmp` are its two colours and
            // `tmp2` the state count.
            PT_LIFE => {
                let rule = (if v >= 0 { v as u32 } else { 0 }) & 0x1F_FFFF;
                self.part_ctype[id] = rule;
                if rule < NGOL as u32 {
                    let (c1, c2) = gol_colours(rule);
                    self.part_dcolour[id] = c1;
                    self.part_tmp[id] = c2;
                }
                self.part_tmp2[id] = (((rule >> 17) & 0xF) + 1) as i32;
            }
            // `CLST.cpp:47-50`: `tmp = rng.between(0, 6)`.
            PT_CLST => self.part_tmp[id] = (roll % 7) as i32,
            // `CRMC.cpp:47-50`: `tmp2 = rng.between(0, 4)`.
            PT_CRMC => self.part_tmp2[id] = (roll % 5) as i32,
            // `PQRT.cpp` / `ROCK.cpp`: `tmp2 = rng.between(0, 10)`.
            PT_PQRT | PT_ROCK => self.part_tmp2[id] = (roll % 11) as i32,
            // `QRTZ.cpp:44-50`: the growth phase and the pressure it started at.
            PT_QRTZ => {
                self.part_tmp2[id] = (roll % 11) as i32;
                self.part_tmp3[id] = pv64;
            }
            // `GLAS.cpp` / `TUNG.cpp`: the pressure the pixel was created under.
            PT_GLAS | PT_TUNG => self.part_tmp3[id] = pv64,
            // `PTNM.cpp:44-50`: a 1/15 chance to start as plutonium.
            PT_PTNM => {
                if roll.is_multiple_of(15) {
                    self.part_tmp[id] = 1;
                }
            }
            // `SLCN.cpp:59-66`: the sparkle phase word.
            PT_SLCN => self.part_tmp[id] = 0x10_0000 + (roll % 0x90_0000) as i32,
            // `TESC.cpp:44-52`: the coil's power comes from the creation value.
            PT_TESC => {
                if v >= 0 {
                    self.part_tmp[id] = v.min(300);
                }
            }
            // `FILT.cpp:44-47`: the filter mode comes from the creation value.
            PT_FILT => self.part_tmp[id] = v,
            // `TRON.cpp:194-205`: head marker, random direction and hue, a four
            // pixel tail and five life.
            PT_TRON => {
                let hue = (roll % 360) as i32;
                let dir = ((roll / 360) % 4) as i32;
                self.part_tmp[id] = 1 | (dir << 5) | (hue << 7);
                self.part_tmp2[id] = 4;
                self.part_life[id] = 5;
            }
            // `VIRS.cpp:48`, `VRSS.cpp:48`, `VRSG.cpp:48`: every virus is
            // created with a full `tmp4` death timer.
            PT_VIRS | PT_VRSS | PT_VRSG => self.part_tmp4[id] = 250,
            // `SEED.cpp:47-76`: the genome is four random 15-bit words and the
            // growth direction points against gravity. Without a gravity field
            // TPT picks the direction at random too (`PLNT.cpp:406-434`).
            PT_SEED => {
                let down = roll2 % 8;
                let up = (down + 4) % 8;
                self.part_ctype[id] = 0b1110_1100_0000 | (up << PLNT_DIR_BITS);
                self.part_tmp[id] = (roll & 0x7FFF) as i32;
                self.part_tmp2[id] = ((roll >> 7) & 0x7FFF) as i32;
                self.part_tmp3[id] = ((roll >> 14) & 0x7FFF) as i32;
                self.part_tmp4[id] = ((roll >> 21) & 0x7FFF) as i32;
            }
            // The second element set (`PT_FNTC`..`PT_MGNT`, ids `195..=238`).
            // `CLNT.cpp:86-89`: the thermostat's cooling target, in Celsius.
            PT_CLNT => self.part_tmp[id] = 22,
            // `CWIR.cpp:93-96`: the wire's melting point, in Kelvin.
            PT_CWIR => self.part_tmp2[id] = 2000,
            // `CEXP.cpp:91-97`: `temp` is the blast temperature, `life` the
            // blast pressure, `tmp` the fuse length and the charge starts as
            // plasma rather than as a solid.
            PT_CEXP => {
                self.part_temp[id] = 9720.0;
                self.part_ctype[id] = PT_PLSM;
                self.part_tmp[id] = 100;
                self.part_life[id] = 256;
            }
            // `LED.cpp:95-98`: a fresh LED is white so the decoration colour
            // immediately shows.
            PT_LED => self.part_dcolour[id] = 0xFFFF_FFFF,
            // `GRPH.cpp:169-173`: `tmp2 = rng.between(0, 5)`, ignition countdown.
            PT_GRPH => {
                self.part_tmp2[id] = (roll % 6) as i32;
                self.part_tmp3[id] = 100;
            }
            // `DFOM.cpp:160-165`: the foam remembers its spawn pixel and a 40
            // frame coalesce timer.
            PT_DFOM => {
                self.part_tmp[id] = x;
                self.part_tmp2[id] = y;
                self.part_tmp3[id] = 40;
            }
            // `MISL.cpp:148-152`: target `x = rng.between(1, 610)`,
            // `y = rng.between(1, 380)`.
            PT_MISL => {
                self.part_tmp[id] = 1 + (roll % 610) as i32;
                self.part_tmp2[id] = 1 + (roll % 380) as i32;
            }
            // `PRMT.cpp:107-110`: `tmp2 = rng.between(0, 10)`.
            PT_PRMT => self.part_tmp2[id] = (roll % 11) as i32,
            // `UVRD.cpp:129-135`: five frames of travel at 2 px/frame along a
            // random `rng.between(0, 359)` degree heading.
            PT_UVRD => {
                self.part_life[id] = 5;
                let a = (roll % 360) as f32 * (core::f32::consts::PI / 180.0);
                self.part_vx[id] = 2.0 * a.cos();
                self.part_vy[id] = 2.0 * a.sin();
            }
            // `ALMP.cpp:158-161`: `tmp2 = rng.between(0, 6)`.
            PT_ALMP => self.part_tmp2[id] = (roll % 7) as i32,
            // `BFLM.cpp:103-106`: `life = rng.between(80, 120)`.
            PT_BFLM => self.part_life[id] = 80 + (roll % 41) as i32,
            _ => {}
        }
    }

    fn initial_velocity(id: u32, x: i32, y: i32, ty: u32) -> (f32, f32) {
        match ty {
            PT_NEUT => {
                let h = hash3(id, x as u32, y as u32);
                let angle = (h & 0xFFFF) as f32 / 65_535.0 * core::f32::consts::TAU;
                // `NEUT.cpp:255-257`: `r = rng.between(128, 255) / 127.0f`, so a
                // fresh neutron leaves at 1.008..2.008 px/frame.
                let radius = if ty == PT_NEUT {
                    1.0 + ((h >> 16) & 0xFF) as f32 / 127.0
                } else {
                    1.0
                };
                (angle.cos() * radius, angle.sin() * radius)
            }
            // `MORT.cpp:46`: "CHOO CHOO", a constant 2 px/frame sideways.
            PT_MORT => (2.0, 0.0),
            PT_LAVA | PT_FIRE | PT_ACID => {
                let h = hash3(id, y as u32, x as u32);
                let jx = ((h & 0xFF) as f32 / 255.0 - 0.5) * 0.2;
                let jy = (((h >> 8) & 0xFF) as f32 / 255.0 - 0.5) * 0.2;
                (jx, jy)
            }
            _ => (0.0, 0.0),
        }
    }

    /// Free particle `id` and clear every spatial map entry that pointed at it.
    pub fn kill_particle(&mut self, id: usize) {
        if id >= self.capacity || self.part_ty[id] == PT_NONE {
            return;
        }
        if let Some(pixel) = self.pixel_index(
            self.part_x[id].round() as i32,
            self.part_y[id].round() as i32,
        ) {
            if pmap_id(self.pmap[pixel]) == Some(id as u32) {
                self.pmap[pixel] = 0;
            }
            if pmap_id(self.photons[pixel]) == Some(id as u32) {
                self.photons[pixel] = 0;
            }
        }

        self.part_ty[id] = PT_NONE;
        self.part_life[id] = 0;
        self.part_ctype[id] = 0;
        // `part_x` / `part_y` keep the slot's last pixel: dead slots are skipped
        // by `ty == PT_NONE` everywhere, and the slot-scoped uploader uses the
        // position to address the map word this kill invalidated.
        self.part_vx[id] = 0.0;
        self.part_vy[id] = 0.0;
        self.part_temp[id] = DEFAULT_TEMP_K;
        self.part_flags[id] = 0;
        self.part_tmp[id] = 0;
        self.part_tmp2[id] = 0;
        self.part_tmp3[id] = 0;
        self.part_tmp4[id] = 0;
        self.part_dcolour[id] = 0;

        self.free_list.push(id);
        self.active_count = self.active_count.saturating_sub(1);
        self.mark_particle_slot_dirty(id);
    }

    /// Kill whatever particle occupies pixel `(x, y)`.
    pub fn kill_at(&mut self, x: i32, y: i32) {
        if let Some(id) = self.particle_index_at(x, y) {
            self.kill_particle(id);
        }
    }

    /// Copy particle `id` out of the SoA arrays.
    pub fn export_particle(&self, id: usize) -> GpuParticle {
        if id >= self.capacity {
            return GpuParticle::default();
        }
        GpuParticle {
            ty: self.part_ty[id],
            life: self.part_life[id],
            ctype: self.part_ctype[id],
            x: self.part_x[id],
            y: self.part_y[id],
            vx: self.part_vx[id],
            vy: self.part_vy[id],
            temp: self.part_temp[id],
            flags: self.part_flags[id],
            tmp: self.part_tmp[id],
            tmp2: self.part_tmp2[id],
            tmp3: self.part_tmp3[id],
            tmp4: self.part_tmp4[id],
            dcolour: self.part_dcolour[id],
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Write particle `id` back into the SoA arrays (save restore / Lua edits).
    ///
    /// Also maintains `free_list`, `next_slot` and `active_count` when the slot
    /// transitions from free to occupied.
    pub fn import_particle(&mut self, id: usize, p: &GpuParticle) {
        if id >= self.capacity {
            self.ensure_particle_capacity(id + 1);
        }
        let was_free = self.part_ty[id] == PT_NONE;

        self.part_ty[id] = p.ty;
        self.part_life[id] = p.life;
        self.part_ctype[id] = p.ctype;
        self.part_x[id] = p.x;
        self.part_y[id] = p.y;
        self.part_vx[id] = p.vx;
        self.part_vy[id] = p.vy;
        self.part_temp[id] = p.temp;
        self.part_flags[id] = p.flags;
        self.part_tmp[id] = p.tmp;
        self.part_tmp2[id] = p.tmp2;
        self.part_tmp3[id] = p.tmp3;
        self.part_tmp4[id] = p.tmp4;
        self.part_dcolour[id] = p.dcolour;

        if was_free && p.ty != PT_NONE {
            if let Some(pos) = self.free_list.iter().position(|slot| *slot == id) {
                self.free_list.swap_remove(pos);
            }
            if id >= self.next_slot {
                self.next_slot = id + 1;
            }
            self.active_count += 1;
        } else if !was_free && p.ty == PT_NONE {
            self.free_list.push(id);
            self.active_count = self.active_count.saturating_sub(1);
        }

        self.mark_particle_slot_dirty(id);
        self.mark_maps_dirty();
    }

    /// Reset the pool to `n` live particles laid out contiguously in `0..n`.
    ///
    /// Used by the save system after importing `0..n` slots.
    pub fn set_particle_count(&mut self, n: usize) {
        if n > self.capacity {
            self.ensure_particle_capacity(n);
        }
        let n = n.min(self.capacity);
        for id in n..self.next_slot.min(self.capacity) {
            self.part_ty[id] = PT_NONE;
        }
        self.active_count = n;
        self.next_slot = n;
        self.free_list = (n..self.capacity)
            .filter(|id| !(Self::GPU_SPAWN_START..Self::GPU_SPAWN_END).contains(id))
            .rev()
            .collect();
        // The cleared tail is not tracked per slot; uploading the whole (now
        // authoritative) shadow is cheaper than listing it.
        self.host_dirty_slots.clear();
        self.mark_particles_dirty();
        self.mark_maps_dirty();
    }

    /// Kill every particle and clear the spatial maps.
    pub fn clear_all_particles(&mut self) {
        self.part_ty.fill(PT_NONE);
        self.part_life.fill(0);
        self.part_ctype.fill(0);
        self.part_x.fill(0.0);
        self.part_y.fill(0.0);
        self.part_vx.fill(0.0);
        self.part_vy.fill(0.0);
        self.part_temp.fill(DEFAULT_TEMP_K);
        self.part_equil.fill(-1.0);
        self.part_flags.fill(0);
        self.part_tmp.fill(0);
        self.part_tmp2.fill(0);
        self.part_tmp3.fill(0);
        self.part_tmp4.fill(0);
        self.part_dcolour.fill(0);

        self.pmap.fill(0);
        self.photons.fill(0);
        self.pmap_count.fill(0);
        self.active_count = 0;
        self.next_slot = 0;
        self.free_list.clear();
        self.pending_live_count = None;
        self.reset_device_soa();
        // Every array was rewritten, so the next upload is a whole-pool one.
        self.host_dirty_slots.clear();
        self.mark_particles_dirty();
        self.mark_maps_dirty();
    }

    /// Reset the whole simulation: particles, air, walls, glow and rigid bodies.
    pub fn reset_all(&mut self) {
        self.clear_all_particles();
        self.air_vx.fill(0.0);
        self.air_vy.fill(0.0);
        self.air_pv.fill(0.0);
        let ambient = self.config.ambient_air_temp;
        self.air_hv.fill(ambient);
        self.air_ovx.fill(0.0);
        self.air_ovy.fill(0.0);
        self.air_opv.fill(0.0);
        self.air_ohv.fill(ambient);
        self.fan_vx.fill(0.0);
        self.fan_vy.fill(0.0);
        self.wall_bmap.fill(0);
        self.wall_emap.fill(0);
        self.block_air.fill(0);
        self.block_airh.fill(0);
        // `Simulation::ClearSim` ends with `SetEdgeMode(edgeMode)`
        // (`Simulation.cpp:1073`), which re-stamps the `EDGE_SOLID` ring the
        // `fill(0)` above just erased. It also refreshes the derived
        // `block_air` / `block_airh` masks for that ring.
        self.apply_edge_mode();
        self.fire_r.fill(0);
        self.fire_g.fill(0);
        self.fire_b.fill(0);
        self.framebuf_rgba.fill(0xFF00_0000);
        self.emp_decor = 0;
        self.fire_tick = self.sim_tick.wrapping_sub(1);
        self.grav_mass.fill(0.0);
        self.grav_mass_prev.fill(0.0);
        self.grav_tool_mass.fill(0.0);
        // The device copy may still hold a painted well that was never solved.
        self.grav_tool_mass_dirty = true;
        self.grav_mask.fill(u32::MAX);
        self.grav_fx.fill(0.0);
        self.grav_fy.fill(0.0);
        // `Simulation::ClearSim` calls `ResetNewtonianGravity({}, {})`
        // (`Simulation.cpp:1067`, `3961-3973`), which zeroes the force output;
        // upload that zero field when a previous solve left one on the device.
        self.zero_gravity_forces();
        self.grav_mask_dirty = true;
        self.grav_mass_dirty = true;
        self.rigid_bodies.clear();
        self.next_rigid_id = 0;
        self.mark_all_dirty();
    }
}

/// Deterministic 3-word mix used for spawn jitter.
#[inline]
/// `Element_FILT_interactWavelengths` (`FILT.cpp:83-131`): the wavelength a
/// `FILT` leaves on a particle that enters it.
///
/// `tmp` selects the mode, and a `ctype` with no wavelength bits falls back to
/// the temperature band (`Element_FILT_getWavelengths`, `FILT.cpp:133-146`).
/// The mode 9 jitter is the only randomness, so it takes the caller's roll.
fn filt_interact_wavelengths(tmp: i32, ctype: u32, temp: f32, orig_wl: u32, roll: u32) -> u32 {
    const MASK: u32 = 0x3FFF_FFFF;
    let filt_wl = if ctype & MASK != 0 {
        ctype & MASK
    } else {
        let bin = (((temp - 273.0) * 0.025) as i32).clamp(0, 25) as u32;
        0x1F << bin
    };
    match tmp {
        // Assign Colour.
        0 => filt_wl,
        // Filter Colour.
        1 => orig_wl & filt_wl,
        // Add Colour.
        2 => orig_wl | filt_wl,
        // Filter Colour the other way.
        3 => orig_wl & !filt_wl,
        // Red shift.
        4 => (orig_wl << (((temp - 273.0) * 0.025) as i32).max(1)) & MASK,
        // Blue shift.
        5 => (orig_wl >> (((temp - 273.0) * 0.025) as i32).max(1)) & MASK,
        // No change.
        6 => orig_wl,
        // XOR colours.
        7 => orig_wl ^ filt_wl,
        // Invert colours.
        8 => (!orig_wl) & MASK,
        // Jitter each colour channel by up to two wavelengths.
        9 => {
            let t1 = (orig_wl & 0xFF) as i32 + (roll % 5) as i32 - 2;
            let t2 = ((orig_wl & 0xFF00) >> 8) as i32 + ((roll / 5) % 5) as i32 - 2;
            let t3 = ((orig_wl & 0xFF_0000) >> 16) as i32 + ((roll / 25) % 5) as i32 - 2;
            (orig_wl & 0xFF00_0000)
                | (((t3 as u32) & 0xFF) << 16)
                | (((t2 as u32) & 0xFF) << 8)
                | ((t1 as u32) & 0xFF)
        }
        // Red shift by the filter's lowest set wavelength.
        10 => {
            let lsb = filt_wl.isolate_lowest_one();
            (orig_wl as u64 * lsb as u64) as u32 & MASK
        }
        // Blue shift by the filter's lowest set wavelength.
        _ => {
            let lsb = filt_wl.isolate_lowest_one();
            orig_wl.checked_div(lsb).unwrap_or(orig_wl) & MASK
        }
    }
}

fn hash3(a: u32, b: u32, c: u32) -> u32 {
    let mut h =
        a.wrapping_mul(0x9E37_79B9) ^ b.wrapping_mul(0x85EB_CA6B) ^ c.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    h
}
