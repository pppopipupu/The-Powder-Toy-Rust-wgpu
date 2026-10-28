//! `Voxel3dEngine`: the 3D world simulation orchestration.
//!
//! The engine owns the persistent world ([`Voxel3dBuffers`]), the element
//! table, the coarse air field and the rigid-body solver, and drives them in
//! the order fixed by plan §4.1. One [`Voxel3dEngine::step`] only enqueues GPU
//! work: no pass reads a value back, so the host never stalls on the device.
//!
//! # Tick order
//!
//! 1. reset the two dispatch counters, snapshot `brick_occupied`, fold
//!    `dirty` in and compact the simulation brick list,
//! 2. gather and solve the air field (skipped when `air_enabled` is false),
//! 3. thermal conduction and reactions (A -> B),
//! 4. clear claims, claim destinations, mark the winners' masks, commit into A,
//! 5. apply the explosion queue and the `SPRK` conduction pass,
//! 6. rigid-body substeps, loose-voxel displacement and blocked-cell claims,
//! 7. rebuild the leaf/coarse masks over `sim_bits | dirty`,
//! 8. store this tick's activity as the previous-frame bitset.
//!
//! `place_sphere`/`erase_sphere` and `let`-style queries are outside the tick:
//! painting launches one brush or segment kernel, [`Voxel3dEngine::pick`] and
//! the read helpers are the only paths that copy a word back to the host.

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use tpt_core::voxel3d::{
    AIR_COUNT, BRICK_COUNT, Brush3d, BrushShape3d, VOXEL_COUNT, Voxel, Voxel3dConfig, element,
    spawn_defaults, temp_to_q,
};

use super::air::AirState;
use super::air_kernel::PV_SCALE;
use super::buffers::Voxel3dBuffers;
use super::edit::{self, HOOK_FIRE, HOOK_LAVA, HOOK_PLSM, PaintFields};
use super::elements::{VoxelElementGpu, spawn_life, spawn_temp_k};
use super::pick::{PickHit, pick};
use super::rigid::Rigid3dState;
use super::step;
use super::{flood, gravity, mutation};

/// Number of explosion requests the queue holds.
pub const EXPLOSION_CAPACITY: usize = 4_096;
/// Words per explosion request (`x, y, z, f32-bits(radius)`).
pub const EXPLOSION_STRIDE: usize = 4;

const _: () = assert!(BRICK_COUNT == 32_768);
const _: () = assert!(EXPLOSION_STRIDE == 4);

/// Copies one device word into a one-word read-back buffer.
#[cube(launch_unchecked)]
fn read_word_kernel(words: &[u32], index: u32, out: &mut [u32]) {
    if ABSOLUTE_POS == 0usize {
        out[0] = words[index as usize];
    }
}

/// The complete 3D sandbox simulation.
pub struct Voxel3dEngine {
    /// CubeCL client that owns the device and every allocation below.
    pub client: Client,
    /// Simulation settings; `paused` short-circuits [`Self::step`].
    pub config: Voxel3dConfig,
    /// The 256³ world state and its masks/lists.
    pub bufs: Voxel3dBuffers,
    /// Element table and renderer palette.
    pub elements: VoxelElementGpu,
    /// 64³ coarse air/pressure field.
    pub air: AirState,
    /// Number of completed ticks.
    pub tick: u64,
    /// Rigid-body solver (host descriptions, device state, render buffers).
    pub rigid: Rigid3dState,
    /// Explosion requests written by the reaction pass.
    pub explosion_queue: Handle,
    /// `[count, 1, 1]` indirect dispatch arguments for the blast pass.
    pub explosion_count: Handle,
    /// Brick list consumed by the mask rebuild (`sim_bits | dirty_bricks`).
    pub mask_list: Handle,
    /// `[count, 1, 1]` indirect dispatch arguments for the mask rebuild.
    pub mask_count: Handle,
}

/// Host copy of the four auxiliary arrays: `(ctype, tmp, tmp2, tmp3)`.
pub type VoxelAuxWords = (Vec<u32>, Vec<i32>, Vec<i32>, Vec<i32>);

/// One undo record: the voxel words and every auxiliary array of the bricks a
/// brush touched.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VoxelBrickSnapshot {
    /// `(brick index, 512 packed voxel words)`.
    pub voxels: Vec<(u32, Vec<u32>)>,
    /// `(brick index, 512 ctype words)`.
    pub ctype: Vec<(u32, Vec<u32>)>,
    /// `(brick index, 512 tmp words)`.
    pub tmp: Vec<(u32, Vec<i32>)>,
    /// `(brick index, 512 tmp2 words)`.
    pub tmp2: Vec<(u32, Vec<i32>)>,
    /// `(brick index, 512 tmp3 words)`.
    pub tmp3: Vec<(u32, Vec<i32>)>,
}

impl VoxelBrickSnapshot {
    /// Whether the snapshot carries no brick.
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Drops every recorded brick.
    pub fn clear(&mut self) {
        self.voxels.clear();
        self.ctype.clear();
        self.tmp.clear();
        self.tmp2.clear();
        self.tmp3.clear();
    }

    /// Appends another snapshot, keeping one entry per brick.
    ///
    /// The caller captures only bricks it has not recorded yet, so a plain
    /// append is enough here; the method exists to keep the five parallel
    /// arrays in lockstep.
    pub fn extend(&mut self, other: Self) {
        self.voxels.extend(other.voxels);
        self.ctype.extend(other.ctype);
        self.tmp.extend(other.tmp);
        self.tmp2.extend(other.tmp2);
        self.tmp3.extend(other.tmp3);
    }
}

impl Voxel3dEngine {
    /// Allocates the complete 3D world on `client`.
    ///
    /// The persistent buffers start empty, the air field starts in its neutral
    /// state and the element table/palette are uploaded once.
    pub fn new(client: Client, config: Voxel3dConfig) -> Self {
        let bufs = Voxel3dBuffers::new(client.clone());
        let elements = VoxelElementGpu::new(client.clone());
        let mut air = AirState::new(client.clone());
        air.iterations = config.air_iterations;
        let rigid = Rigid3dState::new(&client);
        let explosion_queue = client.empty(EXPLOSION_CAPACITY * EXPLOSION_STRIDE * 4);
        let explosion_count = count_handle(&client);
        let mask_list = client.empty(BRICK_COUNT * 4);
        let mask_count = count_handle(&client);
        Self {
            client,
            config,
            bufs,
            elements,
            air,
            tick: 0,
            rigid,
            explosion_queue,
            explosion_count,
            mask_list,
            mask_count,
        }
    }

    /// Returns the engine to its freshly created state.
    ///
    /// Clears the voxel arrays and masks, resets the air field, the rigid
    /// solver and the tick counter. User-triggered; never called per tick.
    pub fn reset(&mut self) {
        self.tick = 0;
        self.bufs.reset();
        self.air = AirState::new(self.client.clone());
        self.air.iterations = self.config.air_iterations;
        self.rigid = Rigid3dState::new(&self.client);
        self.explosion_queue = self.client.empty(EXPLOSION_CAPACITY * EXPLOSION_STRIDE * 4);
        self.explosion_count = count_handle(&self.client);
        self.mask_list = self.client.empty(BRICK_COUNT * 4);
        self.mask_count = count_handle(&self.client);
    }

    /// Advances the world by one tick.
    ///
    /// Returns immediately when `config.paused` is set. Every pass below is a
    /// device launch; the method never blocks on the device.
    pub fn step(&mut self) {
        if self.config.paused {
            return;
        }
        self.tick = self.tick.wrapping_add(1);
        self.air.iterations = self.config.air_iterations;

        step::active::reset_counts(self);
        step::active::snapshot(self);
        step::active::compact(self);

        if self.config.air_enabled {
            self.air
                .gather(&self.bufs, &self.bufs.sim_list, &self.bufs.sim_count);
            self.air
                .solve(&self.bufs, &self.bufs.sim_list, &self.bufs.sim_count);
        }

        // Newtonian gravity (`Simulation::GetGravityField`,
        // `Simulation.cpp:2088-2120`): the force consumed by this tick's
        // movement comes from the mass gathered on the previous tick, exactly
        // like TPT's asynchronous FFT dispatch. The accumulator is then cleared
        // (`Simulation::BeforeSim`) and refilled from the current voxel state
        // once the element passes have run.
        gravity::solve(self);
        gravity::clear_mass(self);

        step::life::apply(self);
        step::reaction::thermal(self);
        // SC_SPECIAL and the nuclear pass both write self state in place and
        // queue neighbour edits, exactly like `Simulation.cpp`'s element loop
        // (C++ element pass runs before movement). Gamma/nuclear and special
        // elements run before movement so their self changes move this tick.
        step::radioactive::apply(self);
        step::special::apply(self);
        gravity::gather(self);
        step::movement::clear_claims(self);
        step::movement::claim(self);
        step::movement::mark_bits(self);
        step::movement::commit(self);
        step::movement::clear_aux(self);
        step::blast::apply(self);
        // `SPRK.cpp` conduction: runs after the movement commit and before the
        // mutation drain, so a sparked neighbour is written in the same tick.
        step::electric::apply(self);
        self.rigid.step(&self.client, &self.bufs, self.tick);
        // Element and blast records become visible in one ordered drain, then
        // the TNT flood advances connected components for the next reaction
        // pass. Both write `voxel_a` and mark dirty bricks before the masks.
        mutation::drain(self);
        flood::apply(self);

        step::active::compact_mask_list(self);
        step::mask::rebuild(self);
        step::active::snapshot_prev(self);
    }

    /// Paints a brush of `ty` and returns the number of covered voxels.
    ///
    /// Returns `0` for an unknown element id or an empty brush. The new voxels
    /// start at the element's spawn temperature so lava does not freeze and
    /// ice does not melt on the first tick.
    pub fn place_brush(&mut self, ty: u32, brush: Brush3d) -> u32 {
        if element(ty).is_none() {
            return 0;
        }
        let fields = self.spawn_fields(ty);
        edit::launch_edit_brush(self, fields, brush)
    }

    /// Erases a brush and returns the number of covered voxels.
    pub fn erase_brush(&mut self, brush: Brush3d) -> u32 {
        edit::launch_edit_brush(self, PaintFields::default(), brush)
    }

    /// Paints `ty` into every packed voxel index.
    ///
    /// The host flood-fill tool owns the connected set and its bounds; the
    /// device stamps the list in one dispatch and updates the same leaf,
    /// coarse and brick activity bits a brush edit does. Returns the number
    /// of indices painted, or `0` for an unknown element or an empty list.
    pub fn paint_indices(&mut self, ty: u32, indices: &[u32]) -> u32 {
        if element(ty).is_none() {
            return 0;
        }
        let fields = self.spawn_fields(ty);
        edit::launch_paint_indices(self, fields, indices)
    }

    /// Adds `delta_k` Kelvin to every non-empty voxel inside `brush`.
    ///
    /// The fixed-point temperature is clamped to the packed `u16` range while
    /// the element id and 8-bit life are preserved, and each touched
    /// brick is marked dirty. Returns the number of voxels the brush covers
    /// (zero for an empty brush or a zero delta): the host cannot know how
    /// many of the covered voxels are non-empty without a read-back.
    pub fn adjust_brush_temp(&mut self, brush: Brush3d, delta_k: f32) -> u32 {
        let delta_q = edit::temp_delta_q(delta_k);
        edit::launch_adjust_temp(self, brush, delta_q)
    }

    /// Paints `ty` along the segment from `from` to `to`, one launch per
    /// segment.
    ///
    /// Ellipsoid and prism sweeps test each voxel against the closest point on
    /// the segment in a space normalised by `radii`. A cuboid sweep uses the
    /// segment's expanded AABB in voxel space instead of a true swept box, so
    /// the box tool keeps its rectangular footprint. Returns `0` for an
    /// unknown element id or an empty sweep.
    pub fn place_brush_segment(
        &mut self,
        ty: u32,
        from: [f32; 3],
        to: [f32; 3],
        radii: [f32; 3],
        shape: BrushShape3d,
    ) -> u32 {
        if element(ty).is_none() {
            return 0;
        }
        let fields = self.spawn_fields(ty);
        edit::launch_edit_segment(self, fields, from, to, radii, shape)
    }

    /// Erases the brush swept from `from` to `to`.
    ///
    /// See [`Self::place_brush_segment`] for the sweep metric.
    pub fn erase_brush_segment(
        &mut self,
        from: [f32; 3],
        to: [f32; 3],
        radii: [f32; 3],
        shape: BrushShape3d,
    ) -> u32 {
        edit::launch_edit_segment(self, PaintFields::default(), from, to, radii, shape)
    }

    /// Paints a sphere of `ty` and returns the number of covered voxels.
    ///
    /// Delegates to [`Self::place_brush`] with an isotropic ellipsoid.
    pub fn place_sphere(&mut self, ty: u32, center: [f32; 3], radius: f32) -> u32 {
        self.place_brush(ty, Brush3d::new(center, radius))
    }

    /// Erases a sphere and returns the number of covered voxels.
    ///
    /// Delegates to [`Self::erase_brush`] with an isotropic ellipsoid.
    pub fn erase_sphere(&mut self, center: [f32; 3], radius: f32) -> u32 {
        self.erase_brush(Brush3d::new(center, radius))
    }

    /// Packed word and auxiliary fields of a freshly painted voxel of `ty`.
    ///
    /// TPT's `create_part` copies `DefaultProperties` (temperature, life,
    /// ctype, tmp…) and then calls the element's `Create` hook; the values for
    /// the ported elements come from [`spawn_defaults`] so the host never
    /// re-invents a default. Elements outside that table keep the previous
    /// ambient-temperature spawn policy.
    fn spawn_fields(&self, ty: u32) -> PaintFields {
        let mut fields = PaintFields::default();
        if let Some(defaults) = spawn_defaults(ty) {
            let temp_q = temp_to_q(defaults.temp_k) as u32;
            fields.word = ty | (temp_q << 8u32) | ((defaults.life & 0xFF) << 24u32);
            fields.ctype = defaults.ctype;
            fields.tmp = defaults.tmp;
            fields.tmp2 = defaults.tmp2;
            fields.tmp3 = defaults.tmp3;
            fields.hook = match defaults.random_life {
                1 => HOOK_FIRE,
                2 => HOOK_PLSM,
                3 => HOOK_LAVA,
                _ => 0,
            };
        } else {
            fields.word = self.spawn_word(ty);
        }
        fields
    }

    /// Packed word of a freshly painted voxel of `ty`.
    fn spawn_word(&self, ty: u32) -> u32 {
        let temp_q = temp_to_q(spawn_temp_k(ty, self.config.ambient_temp_k)) as u32;
        let life = spawn_life(ty);
        ty | (temp_q << 8u32) | (life << 24u32)
    }

    /// Casts a pick ray and returns the first voxel.
    ///
    /// Performs one tiny read-back; call it on a click, not per frame.
    pub fn pick(&self, origin: [f32; 3], dir: [f32; 3]) -> Option<PickHit> {
        pick(self, origin, dir)
    }

    /// Copies the authoritative 256³ voxel array to the host.
    ///
    /// This is the save path, not a per-frame query: the transfer is 64 MiB
    /// and blocks on the device queue. Returns `None` when the read-back fails
    /// or the backend returns a short buffer.
    pub fn read_world_words(&self) -> Option<Vec<u32>> {
        let readback = self.client.read(vec![self.bufs.voxel_a.clone()]).ok()?;
        let raw = readback.first()?;
        if raw.len() < VOXEL_COUNT * 4 {
            return None;
        }
        let (words, _) = raw.as_chunks::<4>();
        Some(words.iter().map(|word| u32::from_le_bytes(*word)).collect())
    }

    /// Copies the six 64³ air fields to the host as `vx | vy | vz | pv | t | solid`.
    ///
    /// Field 4 is the authoritative TPT `pv` (the reaction and blast passes
    /// write it); the Jacobi `p` and its scratch partner are derived state and
    /// are intentionally not saved.
    ///
    /// Like [`Self::read_world_words`], this is a save/load helper only.
    pub fn read_air_fields(&self) -> Option<Vec<f32>> {
        let handles = vec![
            self.air.vx.clone(),
            self.air.vy.clone(),
            self.air.vz.clone(),
            self.air.pv.clone(),
            self.air.t.clone(),
            self.air.solid.clone(),
        ];
        let readback = self.client.read(handles).ok()?;
        let mut words = Vec::with_capacity(6 * AIR_COUNT);
        for (field, raw) in readback.iter().enumerate() {
            if raw.len() < AIR_COUNT * 4 {
                return None;
            }
            let (chunks, _) = raw.as_chunks::<4>();
            if field == 3 {
                // `pv` is `i32` fixed point; the save format stores pressure
                // units so a file never depends on `PV_SCALE`.
                words.extend(
                    chunks
                        .iter()
                        .map(|word| i32::from_le_bytes(*word) as f32 / PV_SCALE as f32),
                );
            } else {
                words.extend(chunks.iter().map(|word| f32::from_le_bytes(*word)));
            }
        }
        Some(words)
    }

    /// Copies the four auxiliary arrays to the host for a save.
    ///
    /// This is the save path; each array is 64 MiB, so it must not run per
    /// frame. Returns `None` when a read-back fails or is short.
    pub fn read_aux_words(&self) -> Option<VoxelAuxWords> {
        let handles = vec![
            self.bufs.aux_ctype.clone(),
            self.bufs.aux_tmp.clone(),
            self.bufs.aux_tmp2.clone(),
            self.bufs.aux_tmp3.clone(),
        ];
        let readback = self.client.read(handles).ok()?;
        if readback.len() != 4 {
            return None;
        }
        let mut words = Vec::with_capacity(4);
        for raw in &readback {
            if raw.len() < VOXEL_COUNT * 4 {
                return None;
            }
            let (chunks, _) = raw.as_chunks::<4>();
            words.push(chunks);
        }
        let ctype = words[0]
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let tmp = words[1]
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .collect();
        let tmp2 = words[2]
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .collect();
        let tmp3 = words[3]
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .collect();
        Some((ctype, tmp, tmp2, tmp3))
    }

    /// Replaces the world and its auxiliary arrays from a saved snapshot.
    ///
    /// Empty auxiliary slices mean "all zero". The arrays are optional so a
    /// version-2 save can still be loaded after upgrading.
    #[allow(clippy::too_many_arguments)]
    pub fn load_world_with_aux(
        &mut self,
        voxels: &[u32],
        aux_ctype: &[u32],
        aux_tmp: &[i32],
        aux_tmp2: &[i32],
        aux_tmp3: &[i32],
        air: &[f32],
    ) -> Result<(), String> {
        for (name, len) in [
            ("ctype", aux_ctype.len()),
            ("tmp", aux_tmp.len()),
            ("tmp2", aux_tmp2.len()),
            ("tmp3", aux_tmp3.len()),
        ] {
            if len != 0 && len != VOXEL_COUNT {
                return Err(format!(
                    "voxel3d: expected 0 or {VOXEL_COUNT} {name} words, got {len}"
                ));
            }
        }
        self.load_world(voxels, air)?;
        if !aux_ctype.is_empty() {
            self.client
                .write(&self.bufs.aux_ctype, Bytes::from_elems(aux_ctype.to_vec()));
        }
        if !aux_tmp.is_empty() {
            self.client
                .write(&self.bufs.aux_tmp, Bytes::from_elems(aux_tmp.to_vec()));
        }
        if !aux_tmp2.is_empty() {
            self.client
                .write(&self.bufs.aux_tmp2, Bytes::from_elems(aux_tmp2.to_vec()));
        }
        if !aux_tmp3.is_empty() {
            self.client
                .write(&self.bufs.aux_tmp3, Bytes::from_elems(aux_tmp3.to_vec()));
        }
        Ok(())
    }

    /// Replaces the world with a saved voxel array and air fields.
    ///
    /// The save path clears the scratch state and every mask, uploads the
    /// authoritative array, then rebuilds the leaf/coarse pyramid from that
    /// array on the device. `air` is either empty (neutral air) or exactly
    /// [`AIR_COUNT`] `* 6` floats in `vx | vy | vz | pv | t | solid` order.
    pub fn load_world(&mut self, voxels: &[u32], air: &[f32]) -> Result<(), String> {
        if voxels.len() != VOXEL_COUNT {
            return Err(format!(
                "voxel3d: expected {VOXEL_COUNT} voxels, got {}",
                voxels.len()
            ));
        }
        if !air.is_empty() && air.len() != 6 * AIR_COUNT {
            return Err(format!(
                "voxel3d: expected 0 or {} air floats, got {}",
                6 * AIR_COUNT,
                air.len()
            ));
        }

        self.bufs.reset();
        self.client
            .write(&self.bufs.voxel_a, Bytes::from_elems(voxels.to_vec()));
        self.tick = 0;
        self.rigid = Rigid3dState::new(&self.client);
        self.client
            .write(&self.explosion_count, Bytes::from_elems(vec![0u32, 1, 1]));

        if air.is_empty() {
            self.air = AirState::new(self.client.clone());
        } else {
            let fields = [
                &self.air.vx,
                &self.air.vy,
                &self.air.vz,
                &self.air.pv,
                &self.air.t,
                &self.air.solid,
            ];
            for (field, handle) in fields.into_iter().enumerate() {
                let start = field * AIR_COUNT;
                if field == 3 {
                    // `pv` is `i32` fixed point; the save stores pressure
                    // units, matching `read_air_fields`.
                    let fixed: Vec<i32> = air[start..start + AIR_COUNT]
                        .iter()
                        .map(|value| (value * PV_SCALE as f32) as i32)
                        .collect();
                    self.client.write(handle, Bytes::from_elems(fixed));
                } else {
                    self.client.write(
                        handle,
                        Bytes::from_elems(air[start..start + AIR_COUNT].to_vec()),
                    );
                }
            }
            // The Jacobi pressure is scratch: `pv` is authoritative and the
            // next solve rebuilds `p` from the velocity divergence. Leaving an
            // old field in place would make the first loaded frame use the
            // previous world's pressure.
            let zero = vec![0.0f32; AIR_COUNT];
            self.client
                .write(&self.air.p, Bytes::from_elems(zero.clone()));
            self.client.write(&self.air.p_tmp, Bytes::from_elems(zero));
        }
        self.air.iterations = self.config.air_iterations;

        // Rebuild the occupancy pyramid from the uploaded array. Setting the
        // occupied bits makes the compaction visit every brick; the mask pass
        // then leaves exactly the bits that the new voxel array requires.
        let all_bricks = vec![u32::MAX; BRICK_COUNT / 32];
        self.client
            .write(&self.bufs.brick_occupied, Bytes::from_elems(all_bricks));
        step::active::reset_counts(self);
        step::active::snapshot(self);
        step::active::compact(self);
        step::active::compact_mask_list(self);
        step::mask::rebuild(self);
        // `snapshot` left `active_bits` at the provisional all-ones value.
        // The next tick's snapshot replaces it from the rebuilt occupancy.
        let zero = vec![0u32; BRICK_COUNT / 32];
        self.client
            .write(&self.bufs.prev_active_bits, Bytes::from_elems(zero));
        Ok(())
    }

    /// Reads the number of occupied bricks from the activity snapshot.
    ///
    /// Used by the HUD on a throttled schedule; it copies only the 1,024-word
    /// `active_bits` bitset, not the voxel array.
    pub fn read_active_bricks(&self) -> Option<u32> {
        let readback = self.client.read(vec![self.bufs.active_bits.clone()]).ok()?;
        let raw = readback.first()?;
        let (words, _) = raw.as_chunks::<4>();
        Some(
            words
                .iter()
                .map(|word| u32::from_le_bytes(*word).count_ones())
                .sum(),
        )
    }

    /// Copies a set of 8³ bricks and their auxiliary words out of the device.
    ///
    /// The undo log must restore `ctype`/`tmp`/`tmp2`/`tmp3` along with the
    /// voxel word: erasing a voxel clears the aux words, and a later repaint of
    /// the same cell starts from that cleared state. A voxel-only snapshot
    /// would therefore rewind the type but leave a stale aux word behind.
    pub fn read_brick_snapshot(&self, bricks: &[u32]) -> Option<VoxelBrickSnapshot> {
        let voxels = self.read_bricks(bricks)?;
        let ctype = self.read_aux_bricks(&self.bufs.aux_ctype, bricks)?;
        let tmp = self.read_aux_bricks(&self.bufs.aux_tmp, bricks)?;
        let tmp2 = self.read_aux_bricks(&self.bufs.aux_tmp2, bricks)?;
        let tmp3 = self.read_aux_bricks(&self.bufs.aux_tmp3, bricks)?;
        let tmp = tmp
            .into_iter()
            .map(|(brick, words)| (brick, words.into_iter().map(|word| word as i32).collect()))
            .collect();
        let tmp2 = tmp2
            .into_iter()
            .map(|(brick, words)| (brick, words.into_iter().map(|word| word as i32).collect()))
            .collect();
        let tmp3 = tmp3
            .into_iter()
            .map(|(brick, words)| (brick, words.into_iter().map(|word| word as i32).collect()))
            .collect();
        Some(VoxelBrickSnapshot {
            voxels,
            ctype,
            tmp,
            tmp2,
            tmp3,
        })
    }

    /// Copies one auxiliary array for the requested bricks.
    fn read_aux_bricks(&self, aux: &Handle, bricks: &[u32]) -> Option<Vec<(u32, Vec<u32>)>> {
        if bricks.is_empty() {
            return Some(Vec::new());
        }
        if bricks.iter().any(|&brick| brick as usize >= BRICK_COUNT) {
            return None;
        }
        let list = self.client.create_from_slice(bytemuck::cast_slice(bricks));
        let words = bricks.len() * 512;
        let out = self.client.empty(words * 4);
        let threads = 256u32;
        let cubes = (words as u32).div_ceil(threads);
        unsafe {
            copy_aux_bricks_out_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(threads),
                BufferArg::from_raw_parts(aux.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(list, bricks.len()),
                BufferArg::from_raw_parts(out.clone(), words),
                bricks.len() as u32,
            );
        }
        let readback = self.client.read(vec![out]).ok()?;
        let raw = readback.first()?;
        if raw.len() < words * 4 {
            return None;
        }
        let (chunks, _) = raw.as_chunks::<4>();
        let words: Vec<u32> = chunks
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        Some(
            bricks
                .iter()
                .enumerate()
                .map(|(i, &brick)| (brick, words[i * 512..(i + 1) * 512].to_vec()))
                .collect(),
        )
    }

    /// Copies a set of 8³ bricks out of the authoritative array.
    ///
    /// Each entry is `(brick_index, 512 words)`. Used by the 3D undo log; the
    /// transfer is `bricks.len() * 2 KiB`, never the whole world.
    pub fn read_bricks(&self, bricks: &[u32]) -> Option<Vec<(u32, Vec<u32>)>> {
        if bricks.is_empty() {
            return Some(Vec::new());
        }
        if bricks.iter().any(|&brick| brick as usize >= BRICK_COUNT) {
            return None;
        }
        let list = self.client.create_from_slice(bytemuck::cast_slice(bricks));
        let words = bricks.len() * 512;
        let out = self.client.empty(words * 4);
        let threads = 256u32;
        let cubes = (words as u32).div_ceil(threads);
        unsafe {
            copy_bricks_out_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(threads),
                BufferArg::from_raw_parts(self.bufs.voxel_a.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(list, bricks.len()),
                BufferArg::from_raw_parts(out.clone(), words),
                bricks.len() as u32,
            );
        }
        let readback = self.client.read(vec![out]).ok()?;
        let raw = readback.first()?;
        if raw.len() < words * 4 {
            return None;
        }
        let (chunks, _) = raw.as_chunks::<4>();
        let words: Vec<u32> = chunks
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        Some(
            bricks
                .iter()
                .enumerate()
                .map(|(i, &brick)| (brick, words[i * 512..(i + 1) * 512].to_vec()))
                .collect(),
        )
    }

    /// Writes 512-word brick payloads back into the authoritative array.
    ///
    /// The masks are updated conservatively; the next tick's mask rebuild
    /// makes them exact. This is the undo path and is never called per frame.
    pub fn restore_bricks(&mut self, bricks: &[(u32, Vec<u32>)]) {
        if bricks.is_empty()
            || bricks
                .iter()
                .any(|(brick, words)| *brick as usize >= BRICK_COUNT || words.len() != 512)
        {
            return;
        }
        let list: Vec<u32> = bricks.iter().map(|(brick, _)| *brick).collect();
        let mut payload = Vec::with_capacity(bricks.len() * 512);
        for (_, words) in bricks {
            payload.extend_from_slice(words);
        }
        let list = self.client.create_from_slice(bytemuck::cast_slice(&list));
        let payload = self
            .client
            .create_from_slice(bytemuck::cast_slice(&payload));
        let words = bricks.len() * 512;
        let threads = 256u32;
        let cubes = (words as u32).div_ceil(threads);
        unsafe {
            restore_bricks_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(threads),
                BufferArg::from_raw_parts(self.bufs.voxel_a.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(list, bricks.len()),
                BufferArg::from_raw_parts(payload, words),
                BufferArg::from_raw_parts(
                    self.bufs.leaf_masks.clone(),
                    tpt_core::voxel3d::LEAF_WORD_COUNT,
                ),
                BufferArg::from_raw_parts(self.bufs.coarse_masks.clone(), super::COARSE_MASK_WORDS),
                BufferArg::from_raw_parts(self.bufs.brick_occupied.clone(), BRICK_COUNT / 32),
                BufferArg::from_raw_parts(self.bufs.dirty_bricks.clone(), BRICK_COUNT / 32),
                bricks.len() as u32,
            );
        }
    }

    /// Restores a brick snapshot produced by [`Self::read_brick_snapshot`].
    pub fn restore_brick_snapshot(&mut self, snapshot: &VoxelBrickSnapshot) {
        self.restore_bricks(&snapshot.voxels);
        self.restore_aux_bricks(&self.bufs.aux_ctype.clone(), &snapshot.ctype);
        let tmp: Vec<(u32, Vec<u32>)> = snapshot
            .tmp
            .iter()
            .map(|(brick, words)| (*brick, words.iter().map(|word| *word as u32).collect()))
            .collect();
        let tmp2: Vec<(u32, Vec<u32>)> = snapshot
            .tmp2
            .iter()
            .map(|(brick, words)| (*brick, words.iter().map(|word| *word as u32).collect()))
            .collect();
        let tmp3: Vec<(u32, Vec<u32>)> = snapshot
            .tmp3
            .iter()
            .map(|(brick, words)| (*brick, words.iter().map(|word| *word as u32).collect()))
            .collect();
        self.restore_aux_bricks(&self.bufs.aux_tmp.clone(), &tmp);
        self.restore_aux_bricks(&self.bufs.aux_tmp2.clone(), &tmp2);
        self.restore_aux_bricks(&self.bufs.aux_tmp3.clone(), &tmp3);
    }

    /// Writes one auxiliary array's brick payloads back to the device.
    fn restore_aux_bricks(&self, aux: &Handle, bricks: &[(u32, Vec<u32>)]) {
        if bricks.is_empty()
            || bricks
                .iter()
                .any(|(brick, words)| *brick as usize >= BRICK_COUNT || words.len() != 512)
        {
            return;
        }
        let list: Vec<u32> = bricks.iter().map(|(brick, _)| *brick).collect();
        let mut payload = Vec::with_capacity(bricks.len() * 512);
        for (_, words) in bricks {
            payload.extend_from_slice(words);
        }
        let list = self.client.create_from_slice(bytemuck::cast_slice(&list));
        let payload = self
            .client
            .create_from_slice(bytemuck::cast_slice(&payload));
        let words = bricks.len() * 512;
        let threads = 256u32;
        let cubes = (words as u32).div_ceil(threads);
        unsafe {
            restore_aux_bricks_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(threads),
                BufferArg::from_raw_parts(aux.clone(), VOXEL_COUNT),
                BufferArg::from_raw_parts(list, bricks.len()),
                BufferArg::from_raw_parts(payload, words),
                bricks.len() as u32,
            );
        }
    }

    /// Reads one voxel from the authoritative array.
    ///
    /// `None` only when the coordinate is outside the world or the read-back
    /// fails; an empty cell returns `Some(Voxel::EMPTY)`. This is a debug/test
    /// helper and stalls the queue, unlike [`Self::step`].
    pub fn read_voxel(&self, x: u32, y: u32, z: u32) -> Option<Voxel> {
        let dim = self.config.dim.min(tpt_core::voxel3d::VOXEL_DIM);
        if x >= dim || y >= dim || z >= dim {
            return None;
        }
        let index = (y << 16u32) | (z << 8u32) | x;
        let word = self.read_word(&self.bufs.voxel_a, index, VOXEL_COUNT)?;
        Some(Voxel::from_word(word))
    }

    /// Whether `brick` currently contains at least one voxel.
    pub fn read_brick_occupied(&self, brick: usize) -> bool {
        if brick >= BRICK_COUNT {
            return false;
        }
        let word = (brick / 32) as u32;
        let mask = 1u32 << (brick & 31);
        match self.read_word(&self.bufs.brick_occupied, word, BRICK_COUNT / 32) {
            Some(value) => value & mask != 0,
            None => false,
        }
    }

    /// Spawns a rigid box and returns its body slot.
    ///
    /// Delegates to [`Rigid3dState::spawn_box`]; the body is uploaded once and
    /// then lives entirely on the device.
    pub fn spawn_box(&mut self, center: [f32; 3], half_extents: [u32; 3], ty: u32) -> usize {
        self.rigid.spawn_box(&self.client, center, half_extents, ty)
    }

    /// Copies one word out of a read-only buffer.
    fn read_word(&self, source: &Handle, index: u32, words: usize) -> Option<u32> {
        let out = self.client.create_from_slice(bytemuck::cast_slice(&[0u32]));
        unsafe {
            read_word_kernel::launch_unchecked(
                &self.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(source.clone(), words),
                index,
                BufferArg::from_raw_parts(out.clone(), 1),
            );
        }
        let bytes = self.client.read(vec![out]).ok()?;
        let raw = bytes.first()?;
        if raw.len() < 4 {
            return None;
        }
        Some(u32::from_le_bytes(raw[..4].try_into().ok()?))
    }
}

/// Copies one auxiliary array's 512-word brick payloads into `out`.
#[cube(launch_unchecked)]
fn copy_aux_bricks_out_kernel(aux: &[u32], bricks: &[u32], out: &mut [u32], count: u32) {
    let i = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if i < count * 512u32 {
        let slot = i / 512u32;
        let local = i % 512u32;
        let brick = bricks[slot as usize];
        let bx = brick & 31u32;
        let bz = (brick >> 5u32) & 31u32;
        let by = (brick >> 10u32) & 31u32;
        let x = (bx << 3u32) + (local & 7u32);
        let z = (bz << 3u32) + ((local >> 3u32) & 7u32);
        let y = (by << 3u32) + ((local >> 6u32) & 7u32);
        let index = (y << 16u32) | (z << 8u32) | x;
        out[i as usize] = aux[index as usize];
    }
}

/// Writes sparse auxiliary brick payloads back into one array.
#[cube(launch_unchecked)]
fn restore_aux_bricks_kernel(aux: &mut [u32], bricks: &[u32], payload: &[u32], count: u32) {
    let i = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if i < count * 512u32 {
        let slot = i / 512u32;
        let local = i % 512u32;
        let brick = bricks[slot as usize];
        let bx = brick & 31u32;
        let bz = (brick >> 5u32) & 31u32;
        let by = (brick >> 10u32) & 31u32;
        let x = (bx << 3u32) + (local & 7u32);
        let z = (bz << 3u32) + ((local >> 3u32) & 7u32);
        let y = (by << 3u32) + ((local >> 6u32) & 7u32);
        let index = (y << 16u32) | (z << 8u32) | x;
        aux[index as usize] = payload[i as usize];
    }
}

/// Copies one 512-word brick payload per brick into `out`.
#[cube(launch_unchecked)]
fn copy_bricks_out_kernel(voxel_a: &[u32], bricks: &[u32], out: &mut [u32], count: u32) {
    let i = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if i < count * 512u32 {
        let slot = i / 512u32;
        let local = i % 512u32;
        let brick = bricks[slot as usize];
        let bx = brick & 31u32;
        let bz = (brick >> 5u32) & 31u32;
        let by = (brick >> 10u32) & 31u32;
        let x = (bx << 3u32) + (local & 7u32);
        let z = (bz << 3u32) + ((local >> 3u32) & 7u32);
        let y = (by << 3u32) + ((local >> 6u32) & 7u32);
        let index = (y << 16u32) | (z << 8u32) | x;
        out[i as usize] = voxel_a[index as usize];
    }
}

/// Writes sparse brick payloads back into the authoritative array.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn restore_bricks_kernel(
    voxel_a: &mut [u32],
    bricks: &[u32],
    payload: &[u32],
    leaf_masks: &mut [Atomic<u32>],
    coarse_masks: &mut [Atomic<u32>],
    brick_occupied: &mut [Atomic<u32>],
    dirty_bricks: &mut [Atomic<u32>],
    count: u32,
) {
    let i = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    if i < count * 512u32 {
        let slot = i / 512u32;
        let local = i % 512u32;
        let brick = bricks[slot as usize];
        let bx = brick & 31u32;
        let bz = (brick >> 5u32) & 31u32;
        let by = (brick >> 10u32) & 31u32;
        let x = (bx << 3u32) + (local & 7u32);
        let z = (bz << 3u32) + ((local >> 3u32) & 7u32);
        let y = (by << 3u32) + ((local >> 6u32) & 7u32);
        let index = (y << 16u32) | (z << 8u32) | x;
        let word = payload[i as usize];
        voxel_a[index as usize] = word;
        if word != 0u32 {
            super::step::mark_leaf_coarse(leaf_masks, coarse_masks, index);
        }
        super::step::mark_brick_bits(brick_occupied, dirty_bricks, brick);
    }
}

/// `[0, 1, 1]` dispatch arguments: one dimension grows, the other two stay 1.
fn count_handle(client: &Client) -> Handle {
    client.create_from_slice(bytemuck::cast_slice(&[0u32, 1u32, 1u32]))
}
