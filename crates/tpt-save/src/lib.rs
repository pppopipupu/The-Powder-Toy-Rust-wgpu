//! Save / load and undo-redo support for The Powder Toy (Rust + wgpu + CubeCL).
//!
//! # On-disk format (`.tptrs`)
//!
//! ```text
//! offset 0   : 8  bytes  magic  = b"TPTRS001"
//! offset 8.. : N  bytes  zlib(bincode(SaveData))
//! ```
//!
//! `SaveData` is a complete, self describing snapshot of one simulation: the
//! [`SimulationConfig`] (canvas size, physics modes, display modes), every
//! active particle (`ty != 0`) exported from the GPU particle pool, the full
//! Eulerian air / fan / wall grids, and all rigid bodies. Payloads written with
//! `Simulation.IncludePressure` off omit the four air grids
//! ([`SaveData::retain_air_pressure`]), matching C++ saves without a pressure
//! map. Because the canvas size travels with the snapshot,
//! [`SaveData::restore_into`] can restore a save into an engine of any
//! dimension: the engine is resized first and the grids are re-fitted to the
//! new cell count. Canvas text signs ([`Sign`]) travel in the same payload even
//! though they are UI annotations rather than simulation state.
//!
//! Undo/redo snapshots are stored in exactly the same compressed form, which
//! keeps the history small (a 612x384 scene with a few thousand particles
//! compresses to tens of kilobytes) and makes the history trivially serializable
//! if a session ever needs to be persisted.

pub mod voxel3d;

use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex, Weak, mpsc};
use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;

use tpt_core::particle::{GpuParticle, RigidBody2D, SimulationConfig};
use tpt_sim::SimulationEngine;

/// Magic header written in front of every `.tptrs` payload.
pub const TPT_RS_MAGIC: &[u8; 8] = b"TPTRS001";

/// Version of the [`SaveData`] layout produced by this build.
///
/// Version 2 appends [`SaveData::signs`]. Version 1 payloads are still
/// accepted by [`SaveData::from_bytes`], which decodes them through
/// `SaveDataV1`; bincode is not self-describing, so the layout change is a
/// format bump even though the added field is serde-defaulted.
pub const SAVE_VERSION: u32 = 2;

/// Largest canvas edge accepted when decoding a save file.
///
/// This is purely a corrupt-header guard: the app's own canvas presets stay far
/// below it, while a garbage length would otherwise trigger a huge allocation.
pub const MAX_CANVAS_DIM: u32 = 16384;

/// File extension used by the app's save browser.
pub const SAVE_EXTENSION: &str = "tptrs";

/// Canvas signs travel with the save; they are defined in `tpt-core` because
/// the simulation owns them (`Simulation::signs`), not the application.
pub use tpt_core::sign::{MAX_SIGN_TEXT, MAX_SIGNS, Sign};

/// A complete snapshot of a running simulation.
///
/// Every field is plain data so the whole structure round-trips through
/// `bincode` without any custom serialization code.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SaveData {
    /// Layout version; see [`SAVE_VERSION`].
    pub version: u32,
    /// Canvas size plus all simulation / display modes.
    pub config: SimulationConfig,
    /// All active particles (`ty != 0`) exported from the GPU pool.
    pub particles: Vec<GpuParticle>,
    /// Air x-velocity per cell (`xcells * ycells`).
    pub air_vx: Vec<f32>,
    /// Air y-velocity per cell (`xcells * ycells`).
    pub air_vy: Vec<f32>,
    /// Air pressure per cell (`xcells * ycells`).
    ///
    /// A payload written with `Simulation.IncludePressure` off carries no air
    /// grids at all (see [`SaveData::retain_air_pressure`]); [`Self::restore_into`]
    /// then loads calm air through `fit_f32`.
    pub air_pv: Vec<f32>,
    /// Ambient heat per cell (`xcells * ycells`).
    pub air_hv: Vec<f32>,
    /// Fan x-velocity per cell (`xcells * ycells`).
    pub fan_vx: Vec<f32>,
    /// Fan y-velocity per cell (`xcells * ycells`).
    pub fan_vy: Vec<f32>,
    /// Wall bitmap per cell, verbatim from `SimBuffers::wall_bmap`.
    pub wall_bmap: Vec<u32>,
    /// Wall electric map per cell, verbatim from `SimBuffers::wall_emap`.
    pub wall_emap: Vec<u32>,
    /// Every rigid body, including its constituent pixels.
    pub rigid_bodies: Vec<RigidBody2D>,
    /// Canvas text signs, drawn as an overlay; never part of the physics.
    #[serde(default)]
    pub signs: Vec<Sign>,
}

/// Grid of exactly `len` elements: copies `src` when the cell counts match,
/// otherwise falls back to a zeroed grid.
fn fit_f32(src: &[f32], len: usize) -> Vec<f32> {
    if src.len() == len {
        src.to_vec()
    } else {
        // A grid captured at a different cell count cannot be mapped onto the
        // new canvas without knowing its original dimensions, so fall back to a
        // calm (zeroed) grid instead of smearing stale values across the scene.
        vec![0.0; len]
    }
}

/// Grid of exactly `len` elements: copies `src` when the cell counts match,
/// otherwise falls back to a zeroed grid.
fn fit_u32(src: &[u32], len: usize) -> Vec<u32> {
    if src.len() == len {
        src.to_vec()
    } else {
        vec![0u32; len]
    }
}

/// Reads only the leading `version` word of a payload.
///
/// bincode's `deserialize` allows trailing bytes, so a one-field probe decodes
/// from the start of a payload laid out by any [`SaveData`] version (the
/// version is always the first field).
#[derive(serde::Deserialize)]
struct SaveDataVersion {
    version: u32,
}

/// The version 1 [`SaveData`] layout, before [`SaveData::signs`] existed.
///
/// bincode is not self-describing: appending a field changes the field count
/// the deserializer expects, so a version 1 payload cannot be decoded as the
/// current struct even with `#[serde(default)]`. Decoding through this mirror
/// keeps old saves loadable. Field order must match the old layout exactly.
#[derive(serde::Deserialize, serde::Serialize)]
struct SaveDataV1 {
    version: u32,
    config: SimulationConfig,
    particles: Vec<GpuParticle>,
    air_vx: Vec<f32>,
    air_vy: Vec<f32>,
    air_pv: Vec<f32>,
    air_hv: Vec<f32>,
    fan_vx: Vec<f32>,
    fan_vy: Vec<f32>,
    wall_bmap: Vec<u32>,
    wall_emap: Vec<u32>,
    rigid_bodies: Vec<RigidBody2D>,
}

impl From<SaveDataV1> for SaveData {
    fn from(legacy: SaveDataV1) -> Self {
        Self {
            version: legacy.version,
            config: legacy.config,
            particles: legacy.particles,
            air_vx: legacy.air_vx,
            air_vy: legacy.air_vy,
            air_pv: legacy.air_pv,
            air_hv: legacy.air_hv,
            fan_vx: legacy.fan_vx,
            fan_vy: legacy.fan_vy,
            wall_bmap: legacy.wall_bmap,
            wall_emap: legacy.wall_emap,
            rigid_bodies: legacy.rigid_bodies,
            // Version 1 had no signs; the feature did not exist.
            signs: Vec::new(),
        }
    }
}

impl SaveData {
    /// Captures the current state of `engine` into a snapshot.
    ///
    /// Wall and decoration edits only touch the host shadows and set
    /// `host_dirty`; a brush stroke uploads its own slots right away. A pending
    /// edit is flushed *before* the read-back, otherwise `sync_gpu_to_host`
    /// would overwrite it with the stale device contents and the snapshot would
    /// silently lose the edit.
    pub fn capture_from(engine: &mut SimulationEngine) -> Self {
        if engine.bufs.host_dirty {
            engine.bufs.sync_host_to_gpu();
        }
        engine.bufs.ensure_host_current();
        // Rigid-body poses are device-resident; pull them before the snapshot
        // so a saved/undone pose is exactly the frame the user sees.
        engine.sync_rigid_bodies_to_host();

        let bufs = &engine.bufs;
        let mut particles = Vec::new();
        let scan = bufs.used_slot_bound().min(bufs.part_ty.len());
        for id in 0..scan {
            // `ty == 0` marks a free slot in the pool; only live particles are
            // persisted so that save files stay compact.
            if bufs.part_ty[id] != 0 {
                particles.push(bufs.export_particle(id));
            }
        }

        Self {
            version: SAVE_VERSION,
            config: bufs.config.clone(),
            particles,
            air_vx: bufs.air_vx.clone(),
            air_vy: bufs.air_vy.clone(),
            air_pv: bufs.air_pv.clone(),
            air_hv: bufs.air_hv.clone(),
            fan_vx: bufs.fan_vx.clone(),
            fan_vy: bufs.fan_vy.clone(),
            wall_bmap: bufs.wall_bmap.clone(),
            wall_emap: bufs.wall_emap.clone(),
            rigid_bodies: bufs.rigid_bodies.clone(),
            signs: engine.signs().to_vec(),
        }
    }

    /// Captures a save / copy / stamp payload, honouring `includePressure`.
    ///
    /// `Simulation::Save` (`Simulation.cpp:341`, pressure block at
    /// `Simulation.cpp:430-456`) only copies the pressure, velocity and
    /// ambient-heat grids when `includePressure` is set; `GameController`
    /// passes `GetIncludePressure()` into every save, copy and stamp call
    /// (`GameController.cpp:473,496,1241,1451,1494`). Undo snapshots keep
    /// using [`Self::capture_from`], because restoring history must not drop
    /// the air grid behind the user's back.
    pub fn capture_from_with_pressure(
        engine: &mut SimulationEngine,
        include_pressure: bool,
    ) -> Self {
        Self::capture_from(engine).retain_air_pressure(include_pressure)
    }

    /// Drops the air grids when `include_pressure` is false.
    ///
    /// The four gated grids match the C++ `includePressure` block: pressure,
    /// both velocity components and ambient heat. Fan velocities and walls are
    /// unconditional there, so they stay. A dropped grid is not serialized at
    /// all; [`Self::restore_into`] zero-fills it through `fit_f32`, which is
    /// exactly the calm air the C++ loader produces for a save without a
    /// pressure map.
    pub fn retain_air_pressure(mut self, include_pressure: bool) -> Self {
        if !include_pressure {
            self.air_vx.clear();
            self.air_vy.clear();
            self.air_pv.clear();
            self.air_hv.clear();
        }
        self
    }

    /// Restores this snapshot into `engine`, resizing the simulation whenever
    /// the saved canvas dimensions differ from the engine's current ones.
    ///
    /// The engine ends up in a fully host/GPU-consistent state: spatial maps are
    /// rebuilt and the host shadows are uploaded to the GPU.
    pub fn restore_into(&self, engine: &mut SimulationEngine) {
        // `SimulationConfig::new` normalises the dimensions (multiple of CELL,
        // at least 64x64); the remaining fields are taken verbatim from the
        // save so physics/display modes survive a load.
        let mut config = SimulationConfig::new(self.config.xres, self.config.yres);
        config.paused = self.config.paused;
        config.aheat_enable = self.config.aheat_enable;
        config.pretty_powder = self.config.pretty_powder;
        config.edge_mode = self.config.edge_mode;
        config.gravity_mode = self.config.gravity_mode;
        config.air_mode = self.config.air_mode;
        config.ambient_air_temp = self.config.ambient_air_temp;
        config.edge_pressure = self.config.edge_pressure;
        config.display_mode = self.config.display_mode;
        config.colour_mode = self.config.colour_mode;
        // The mod-element switch is a global preference (`prefs.json`), not
        // save state, so a load or an undo must not silently re-enable it.
        config.mod_elements_enabled = engine.bufs.config.mod_elements_enabled;

        if engine.bufs.config.xres != config.xres || engine.bufs.config.yres != config.yres {
            engine.resize_canvas(config.xres, config.yres);
        }
        engine.bufs.config = config;

        // Free every slot before importing: the pool may still hold particles
        // from the scene we are replacing. `clear_all_particles` also resets
        // `active_count` and the free list.
        engine.bufs.clear_all_particles();
        // The pool saturates at `SimBuffers::MAX_PARTICLE_CAPACITY`: the packed
        // spatial maps (and the movement winner key) store `part_id + 1` in 23
        // bits. A snapshot with more live particles than that - only possible
        // for a corrupt or hand-crafted file, since no canvas above the pool
        // cap can be filled - is imported up to the cap and the rest dropped.
        let wanted = self.particles.len();
        if wanted > 0 {
            engine.bufs.ensure_particle_capacity(wanted);
        }
        let mut slot = 0usize;
        for particle in &self.particles {
            if particle.ty == 0 {
                continue;
            }
            if slot >= engine.bufs.capacity {
                break;
            }
            engine.bufs.import_particle(slot, particle);
            slot += 1;
        }
        // Publish the live prefix and rebuild the free list for the tail.
        engine.bufs.set_particle_count(slot);

        let cells = engine.bufs.config.num_cells();
        engine.bufs.air_vx = fit_f32(&self.air_vx, cells);
        engine.bufs.air_vy = fit_f32(&self.air_vy, cells);
        engine.bufs.air_pv = fit_f32(&self.air_pv, cells);
        engine.bufs.air_hv = fit_f32(&self.air_hv, cells);
        // The previous-frame air buffers are scratch space; they must not carry
        // stale values from the pre-load scene into the first air step. Mirror
        // `SimBuffers::reset_all`: previous ambient heat starts at ambient.
        let ambient = engine.bufs.config.ambient_air_temp;
        engine.bufs.air_ovx = vec![0.0; cells];
        engine.bufs.air_ovy = vec![0.0; cells];
        engine.bufs.air_opv = vec![0.0; cells];
        engine.bufs.air_ohv = vec![ambient; cells];
        engine.bufs.fan_vx = fit_f32(&self.fan_vx, cells);
        engine.bufs.fan_vy = fit_f32(&self.fan_vy, cells);
        engine.bufs.wall_bmap = fit_u32(&self.wall_bmap, cells);
        engine.bufs.wall_emap = fit_u32(&self.wall_emap, cells);
        // `block_air` / `block_airh` are derived from the wall grids and are
        // recomputed by `rebuild_spatial_maps()` below.

        engine.bufs.rigid_bodies = self.rigid_bodies.clone();
        let mut next_id = 1u32;
        for body in &engine.bufs.rigid_bodies {
            next_id = next_id.max(body.id.saturating_add(1));
        }
        engine.bufs.next_rigid_id = next_id;

        // Glow buffers are not part of the snapshot; drop any glow from the
        // scene being replaced so it cannot bleed into the loaded one.
        engine.bufs.fire_r.fill(0);
        engine.bufs.fire_g.fill(0);
        engine.bufs.fire_b.fill(0);

        engine.bufs.rebuild_spatial_maps();
        engine.set_signs(self.signs.clone());
        // Every array was replaced wholesale; mark all categories dirty so the
        // upload also refreshes air, walls and maps on the device.
        engine.bufs.mark_all_dirty();
        engine.bufs.sync_host_to_gpu();
    }

    /// Serializes and compresses this snapshot into the `.tptrs` byte layout.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let payload = bincode::serialize(self)
            .map_err(|err| format!("failed to serialize save data: {err}"))?;

        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(&payload)
            .map_err(|err| format!("failed to compress save data: {err}"))?;
        let compressed = encoder
            .finish()
            .map_err(|err| format!("failed to finish compressing save data: {err}"))?;

        let mut out = Vec::with_capacity(TPT_RS_MAGIC.len() + compressed.len());
        out.extend_from_slice(TPT_RS_MAGIC);
        out.extend_from_slice(&compressed);
        Ok(out)
    }

    /// Decodes a `.tptrs` byte buffer produced by [`SaveData::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() <= TPT_RS_MAGIC.len() {
            return Err("save data is too short to contain a header".to_string());
        }
        if &bytes[..TPT_RS_MAGIC.len()] != TPT_RS_MAGIC {
            return Err("not a Powder Toy (Rust) save file: bad magic header".to_string());
        }

        let mut decoder = ZlibDecoder::new(&bytes[TPT_RS_MAGIC.len()..]);
        let mut payload = Vec::new();
        decoder
            .read_to_end(&mut payload)
            .map_err(|err| format!("failed to decompress save data: {err}"))?;

        // Peek at the leading version word before decoding the body: the
        // field count differs between layout versions, so the version decides
        // which struct the remainder of the payload must be decoded as.
        let probe: SaveDataVersion = bincode::deserialize(&payload)
            .map_err(|err| format!("failed to deserialize save data: {err}"))?;
        if probe.version == 0 || probe.version > SAVE_VERSION {
            return Err(format!(
                "unsupported save file version {} (this build supports 1..={})",
                probe.version, SAVE_VERSION
            ));
        }

        let data: SaveData = if probe.version == 1 {
            let legacy: SaveDataV1 = bincode::deserialize(&payload)
                .map_err(|err| format!("failed to deserialize version 1 save data: {err}"))?;
            legacy.into()
        } else {
            bincode::deserialize(&payload)
                .map_err(|err| format!("failed to deserialize save data: {err}"))?
        };

        if data.config.xres == 0
            || data.config.yres == 0
            || data.config.xres > MAX_CANVAS_DIM
            || data.config.yres > MAX_CANVAS_DIM
        {
            return Err(format!(
                "save file has an invalid canvas size {}x{}",
                data.config.xres, data.config.yres
            ));
        }

        Ok(data)
    }

    /// Writes this snapshot to `path`, creating parent directories as needed.
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create save directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes)
            .map_err(|err| format!("failed to write save file {}: {err}", path.display()))
    }

    /// Reads and decodes a `.tptrs` file from `path`.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|err| format!("failed to read save file {}: {err}", path.display()))?;
        Self::from_bytes(&bytes)
    }
}

/// Payload stored inside one undo/redo stack entry.
///
/// New snapshots start uncompressed so capturing on mouse-down does not stall
/// the UI thread with synchronous zlib compression on large canvases, and a
/// background worker replaces the raw snapshot with its compressed `.tptrs`
/// payload while freeing the uncompressed buffers.
#[derive(Debug)]
enum UndoPayload {
    Raw(Arc<SaveData>),
    Compressed(Vec<u8>),
}

/// One undo/redo snapshot entry, compressed asynchronously off the UI thread.
#[derive(Clone, Debug)]
pub struct UndoSnapshot(Arc<Mutex<UndoPayload>>);

impl From<Vec<u8>> for UndoSnapshot {
    fn from(bytes: Vec<u8>) -> Self {
        Self(Arc::new(Mutex::new(UndoPayload::Compressed(bytes))))
    }
}

impl PartialEq for UndoSnapshot {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

static UNDO_COMPRESS_TX: LazyLock<mpsc::Sender<Weak<Mutex<UndoPayload>>>> = LazyLock::new(|| {
    let (tx, rx) = mpsc::channel::<Weak<Mutex<UndoPayload>>>();
    let _ = std::thread::Builder::new()
        .name("tpt-undo-compress".to_string())
        .spawn(move || {
            while let Ok(weak) = rx.recv() {
                let Some(arc) = weak.upgrade() else {
                    continue;
                };
                let raw = {
                    let Ok(guard) = arc.lock() else {
                        continue;
                    };
                    match &*guard {
                        UndoPayload::Raw(data) => Some(data.clone()),
                        UndoPayload::Compressed(_) => None,
                    }
                };
                let Some(raw) = raw else {
                    continue;
                };
                if let Ok(bytes) = raw.to_bytes()
                    && let Ok(mut guard) = arc.lock()
                    && matches!(&*guard, UndoPayload::Raw(_))
                {
                    *guard = UndoPayload::Compressed(bytes);
                }
            }
        });
    tx
});

impl UndoSnapshot {
    fn from_save_data(data: SaveData) -> Self {
        let arc = Arc::new(Mutex::new(UndoPayload::Raw(Arc::new(data))));
        let _ = UNDO_COMPRESS_TX.send(Arc::downgrade(&arc));
        Self(arc)
    }


    fn decode(&self) -> Result<SaveData, String> {
        let payload = {
            let guard = self.0.lock().map_err(|_| "undo snapshot lock poisoned".to_string())?;
            match &*guard {
                UndoPayload::Raw(data) => return Ok((**data).clone()),
                UndoPayload::Compressed(bytes) => bytes.clone(),
            }
        };
        SaveData::from_bytes(&payload)
    }
}

/// Bounded undo / redo history holding [`SaveData`] snapshots.
///
/// `undo_stack` holds earlier states (most recent last) and `redo_stack` holds
/// states that were undone, so `undo`/`redo` behave like a classic linear
/// history: a fresh [`UndoHistory::push_snapshot`] discards the redo branch.
#[derive(Clone, Debug)]
pub struct UndoHistory {
    /// Snapshots of past states; the last entry is the most recent.
    pub undo_stack: Vec<UndoSnapshot>,
    /// Snapshots of undone states; the last entry is the most recent.
    pub redo_stack: Vec<UndoSnapshot>,
    /// Upper bound on the number of snapshots kept in each stack.
    pub max_snapshots: usize,
}

impl UndoHistory {
    /// Creates a history bounded to `max_snapshots` entries per stack.
    ///
    /// A bound of zero is clamped to one so that undo remains meaningful.
    pub fn new(max_snapshots: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_snapshots: max_snapshots.max(1),
        }
    }

    /// Records the current engine state as an undo point and clears redo.
    pub fn push_snapshot(&mut self, engine: &mut SimulationEngine) {
        let snapshot = UndoSnapshot::from_save_data(SaveData::capture_from(engine));
        self.push_snapshot_entry(snapshot);
    }

    fn push_snapshot_entry(&mut self, snapshot: UndoSnapshot) {
        self.redo_stack.clear();
        self.undo_stack.push(snapshot);
        self.trim_undo();
    }

    /// Restores the most recent undo point. Returns `false` when empty.
    pub fn undo(&mut self, engine: &mut SimulationEngine) -> bool {
        let Some(snapshot) = self.undo_stack.pop() else {
            return false;
        };
        let data = match snapshot.decode() {
            Ok(data) => data,
            Err(_) => {
                // Keep the stack consistent if the payload is corrupt.
                self.undo_stack.push(snapshot);
                return false;
            }
        };

        self.push_redo(engine);
        data.restore_into(engine);
        true
    }

    /// Re-applies the most recently undone state. Returns `false` when empty.
    pub fn redo(&mut self, engine: &mut SimulationEngine) -> bool {
        let Some(snapshot) = self.redo_stack.pop() else {
            return false;
        };
        let data = match snapshot.decode() {
            Ok(data) => data,
            Err(_) => {
                self.redo_stack.push(snapshot);
                return false;
            }
        };

        self.push_undo(engine);
        data.restore_into(engine);
        true
    }

    fn push_redo(&mut self, engine: &mut SimulationEngine) {
        self.redo_stack
            .push(UndoSnapshot::from_save_data(SaveData::capture_from(engine)));
        self.trim_redo();
    }

    fn push_undo(&mut self, engine: &mut SimulationEngine) {
        self.undo_stack
            .push(UndoSnapshot::from_save_data(SaveData::capture_from(engine)));
        self.trim_undo();
    }

    /// Whether [`UndoHistory::undo`] would restore a state.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Whether [`UndoHistory::redo`] would restore a state.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Drops all recorded history.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    fn trim_undo(&mut self) {
        let overflow = self.undo_stack.len().saturating_sub(self.max_snapshots);
        if overflow > 0 {
            self.undo_stack.drain(..overflow);
        }
    }

    fn trim_redo(&mut self) {
        let overflow = self.redo_stack.len().saturating_sub(self.max_snapshots);
        if overflow > 0 {
            self.redo_stack.drain(..overflow);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &SaveData) -> SaveData {
        let bytes = data.to_bytes().expect("serialize");
        SaveData::from_bytes(&bytes).expect("deserialize")
    }

    fn sample_config() -> SimulationConfig {
        SimulationConfig::new(128, 96)
    }

    fn sample_particle(x: f32, y: f32, ty: u32) -> GpuParticle {
        GpuParticle {
            ty,
            life: 42,
            ctype: 2,
            x,
            y,
            vx: 0.5,
            vy: -0.25,
            temp: 295.15,
            flags: 0,
            tmp: 1,
            tmp2: 2,
            tmp3: 3,
            tmp4: 4,
            dcolour: 0x0011_2233,
            _pad0: 0,
            _pad1: 0,
        }
    }

    fn sample_save() -> SaveData {
        let config = sample_config();
        let cells = config.num_cells();
        SaveData {
            version: SAVE_VERSION,
            particles: vec![
                sample_particle(10.0, 20.0, 1),
                sample_particle(11.0, 21.0, 2),
            ],
            air_vx: vec![0.25; cells],
            air_vy: vec![-0.5; cells],
            air_pv: vec![1.5; cells],
            air_hv: vec![295.15; cells],
            fan_vx: vec![0.0; cells],
            fan_vy: vec![0.0; cells],
            wall_bmap: vec![0; cells],
            wall_emap: vec![0; cells],
            rigid_bodies: Vec::new(),
            signs: vec![
                Sign {
                    x: 32,
                    y: 48,
                    text: "Hot zone".to_string(),
                    ju: tpt_core::sign::JUSTIFY_LEFT,
                },
                Sign {
                    x: 90,
                    y: 12,
                    text: "Max 45 chars".to_string(),
                    ju: tpt_core::sign::JUSTIFY_RIGHT,
                },
            ],
            config,
        }
    }

    #[test]
    fn magic_header_is_written() {
        let bytes = sample_save().to_bytes().expect("serialize");
        assert_eq!(&bytes[..8], TPT_RS_MAGIC);
    }

    #[test]
    fn roundtrip_preserves_state() {
        let original = sample_save();
        let restored = roundtrip(&original);
        assert_eq!(restored.version, SAVE_VERSION);
        assert_eq!(restored.config.xres, original.config.xres);
        assert_eq!(restored.config.yres, original.config.yres);
        assert_eq!(restored.particles.len(), 2);
        assert_eq!(restored.particles[1].ty, 2);
        assert_eq!(restored.air_pv.len(), original.config.num_cells());
    }

    #[test]
    fn roundtrip_preserves_signs() {
        let original = sample_save();
        let restored = roundtrip(&original);
        assert_eq!(restored.signs, original.signs);
        assert_eq!(restored.signs[0].x, 32);
        assert_eq!(restored.signs[0].y, 48);
        assert_eq!(restored.signs[0].text, "Hot zone");
        assert_eq!(restored.signs[1].text, "Max 45 chars");
        assert_eq!(restored.signs[0].ju, tpt_core::sign::JUSTIFY_LEFT);
        assert_eq!(restored.signs[1].ju, tpt_core::sign::JUSTIFY_RIGHT);
    }

    #[test]
    fn version_one_payload_loads_without_signs() {
        let sample = sample_save();
        let legacy = SaveDataV1 {
            version: 1,
            config: sample.config.clone(),
            particles: sample.particles.clone(),
            air_vx: sample.air_vx.clone(),
            air_vy: sample.air_vy.clone(),
            air_pv: sample.air_pv.clone(),
            air_hv: sample.air_hv.clone(),
            fan_vx: sample.fan_vx.clone(),
            fan_vy: sample.fan_vy.clone(),
            wall_bmap: sample.wall_bmap.clone(),
            wall_emap: sample.wall_emap.clone(),
            rigid_bodies: sample.rigid_bodies.clone(),
        };
        let payload = bincode::serialize(&legacy).expect("serialize version 1");
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&payload).expect("compress version 1");
        let mut bytes = TPT_RS_MAGIC.to_vec();
        bytes.extend_from_slice(&encoder.finish().expect("finish compression"));

        let restored = SaveData::from_bytes(&bytes).expect("decode version 1");
        assert_eq!(restored.version, 1);
        assert!(restored.signs.is_empty());
        assert_eq!(restored.particles, sample.particles);
        assert_eq!(restored.air_pv, sample.air_pv);
    }

    #[test]
    fn payload_with_pressure_keeps_air_grids() {
        let original = sample_save();
        let kept = original.clone().retain_air_pressure(true);
        assert_eq!(kept.air_vx, original.air_vx);
        assert_eq!(kept.air_vy, original.air_vy);
        assert_eq!(kept.air_pv, original.air_pv);
        assert_eq!(kept.air_hv, original.air_hv);
    }

    #[test]
    fn payload_without_pressure_drops_air_grids() {
        let original = sample_save();
        let cells = original.config.num_cells();
        let dropped = original.clone().retain_air_pressure(false);
        assert!(dropped.air_vx.is_empty());
        assert!(dropped.air_vy.is_empty());
        assert!(dropped.air_pv.is_empty());
        assert!(dropped.air_hv.is_empty());
        // Only the air grids are gated: walls, particles, fan velocities and
        // signs stay in the payload (`Simulation.cpp:430-456`).
        assert_eq!(dropped.wall_bmap.len(), cells);
        assert_eq!(dropped.fan_vx, original.fan_vx);
        assert_eq!(dropped.particles, original.particles);
        assert_eq!(dropped.signs, original.signs);
    }

    #[test]
    fn pressureless_payload_roundtrips() {
        let dropped = sample_save().retain_air_pressure(false);
        let restored = roundtrip(&dropped);
        // The empty grids survive serialization; `restore_into` zero-fills
        // them through `fit_f32` when the payload is loaded.
        assert!(restored.air_vx.is_empty());
        assert!(restored.air_vy.is_empty());
        assert!(restored.air_pv.is_empty());
        assert!(restored.air_hv.is_empty());
        assert_eq!(restored.particles, dropped.particles);
        assert_eq!(restored.signs, dropped.signs);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = sample_save().to_bytes().expect("serialize");
        bytes[0] = b'X';
        assert!(SaveData::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_truncated_input() {
        let bytes = sample_save().to_bytes().expect("serialize");
        assert!(SaveData::from_bytes(&bytes[..5]).is_err());
        assert!(SaveData::from_bytes(&bytes[..bytes.len() - 4]).is_err());
    }

    #[test]
    fn rejects_future_version() {
        let mut data = sample_save();
        data.version = SAVE_VERSION + 1;
        let bytes = data.to_bytes().expect("serialize");
        assert!(SaveData::from_bytes(&bytes).is_err());
    }

    #[test]
    fn rejects_version_zero() {
        let mut data = sample_save();
        data.version = 0;
        let bytes = data.to_bytes().expect("serialize");
        assert!(SaveData::from_bytes(&bytes).is_err());
    }

    #[test]
    fn payload_is_zlib_compressed() {
        let bytes = sample_save().to_bytes().expect("serialize");
        // zlib streams start with CMF 0x78 (deflate, 32K window).
        assert_eq!(bytes[TPT_RS_MAGIC.len()], 0x78);
        assert!(bytes.len() < bincode::serialize(&sample_save()).expect("bincode").len());
    }

    #[test]
    fn undo_history_bounds_stacks() {
        // Pure data-structure behaviour: no engine required for trimming.
        let mut history = UndoHistory::new(2);
        let s1 = UndoSnapshot::from(vec![1]);
        let s2 = UndoSnapshot::from(vec![2]);
        let s3 = UndoSnapshot::from(vec![3]);
        history.undo_stack.push(s1);
        history.undo_stack.push(s2.clone());
        history.undo_stack.push(s3);
        history.trim_undo();
        assert_eq!(history.undo_stack.len(), 2);
        assert_eq!(history.undo_stack[0], s2);
        assert!(history.can_undo());
        history.clear();
        assert!(!history.can_undo());
        assert!(!history.can_redo());
    }
}
