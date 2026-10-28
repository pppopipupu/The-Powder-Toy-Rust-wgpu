//! Save / load support for the 3D voxel mode (`.tpt3`).
//!
//! # On-disk format
//!
//! ```text
//! offset 0   : 8  bytes  magic  = b"TPT3RS01"
//! offset 8.. : N  bytes  zlib(bincode(Voxel3dSaveData))
//! ```
//!
//! The payload stores host-side data only: the packed world grid
//! ([`VOXEL_COUNT`] `u32` voxels), the coarse air grid ([`AIR_COUNT`] `f32`
//! cells per layer, or no air at all), the [`Voxel3dConfig`] and the rigid
//! bodies with their local voxel grids. No GPU buffer layout leaks into the
//! file, so reorganising the simulation or renderer buffers never invalidates
//! an existing save.

use std::io::{Read, Write};
use std::path::Path;

use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;

use tpt_core::voxel3d::{AIR_COUNT, BODY_VOXELS, MAX_BODIES, VOXEL_COUNT, Voxel3dConfig};

/// Magic header written in front of every `.tpt3` payload.
pub const TPT3_MAGIC: &[u8; 8] = b"TPT3RS01";

/// Version of the [`Voxel3dSaveData`] layout produced by this build.
///
/// Version 2 adds the `edge_mode` field to [`Voxel3dConfig`]. Version 3 adds
/// the four per-voxel auxiliary arrays (`ctype`, `tmp`, `tmp2`, `tmp3`).
/// Version 4 stores the authoritative TPT pressure `pv` in the air field's
/// fourth slot instead of the derived Jacobi pressure.
pub const VOXEL3D_SAVE_VERSION: u32 = 4;

/// One rigid body as it travels in a `.tpt3` snapshot.
///
/// The local grid is a packed copy of the body's `32^3` voxel block; it is
/// empty for bodies that carry no voxel payload. All other fields are the
/// host-side pose and mass properties needed to re-create the GPU body.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Voxel3dBodySave {
    /// World-space position of the body centre.
    pub pos: [f32; 3],
    /// Orientation quaternion `(x, y, z, w)`.
    pub quat: [f32; 4],
    /// Linear velocity in voxels per second.
    pub lin_vel: [f32; 3],
    /// Angular velocity in radians per second.
    pub ang_vel: [f32; 3],
    /// Half extents of the body's box in voxels.
    pub half_extents: [f32; 3],
    /// Element type stored in the body's local voxels.
    pub ty: u32,
    /// Packed local voxel grid, at most [`BODY_VOXELS`] entries.
    pub local_voxels: Vec<u32>,
}

/// Complete snapshot of the 3D voxel world.
///
/// `version` is always written as [`VOXEL3D_SAVE_VERSION`]. The layout
/// invariants are checked by [`Voxel3dSaveData::to_bytes`] and
/// [`Voxel3dSaveData::from_bytes`], so invalid data never reaches disk and a
/// corrupt file is rejected with a descriptive error.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Voxel3dSaveData {
    /// Layout version of this payload.
    pub version: u32,
    /// Simulation and display settings of the 3D mode.
    pub config: Voxel3dConfig,
    /// Packed world grid; must hold [`VOXEL_COUNT`] voxels.
    pub voxels: Vec<u32>,
    /// TPT `ctype` per voxel; empty means all zero.
    #[serde(default)]
    pub aux_ctype: Vec<u32>,
    /// TPT `tmp` per voxel; empty means all zero.
    #[serde(default)]
    pub aux_tmp: Vec<i32>,
    /// TPT `tmp2` per voxel; empty means all zero.
    #[serde(default)]
    pub aux_tmp2: Vec<i32>,
    /// TPT `tmp3` per voxel; empty means all zero.
    #[serde(default)]
    pub aux_tmp3: Vec<i32>,
    /// Coarse air grid (`vx | vy | vz | pv | t | solid`); empty, or a multiple
    /// of [`AIR_COUNT`] cells. `pv` is the authoritative TPT pressure.
    pub air: Vec<f32>,
    /// Rigid bodies, at most [`MAX_BODIES`].
    pub rigid_bodies: Vec<Voxel3dBodySave>,
}

impl Voxel3dSaveData {
    /// Builds a snapshot from host-side data.
    ///
    /// The lengths are validated when the snapshot is written or read back;
    /// construction itself is infallible so callers can assemble a save from
    /// live simulation buffers without error plumbing.
    pub fn new(
        config: Voxel3dConfig,
        voxels: Vec<u32>,
        air: Vec<f32>,
        rigid_bodies: Vec<Voxel3dBodySave>,
    ) -> Self {
        Self {
            version: VOXEL3D_SAVE_VERSION,
            config,
            voxels,
            aux_ctype: Vec::new(),
            aux_tmp: Vec::new(),
            aux_tmp2: Vec::new(),
            aux_tmp3: Vec::new(),
            air,
            rigid_bodies,
        }
    }

    /// Builds a snapshot including the four per-voxel auxiliary arrays.
    ///
    /// An empty array is stored as "all zero" and is accepted by
    /// [`Self::validate`]; otherwise every array must hold exactly
    /// [`VOXEL_COUNT`] entries.
    #[allow(clippy::too_many_arguments)]
    pub fn with_aux(
        config: Voxel3dConfig,
        voxels: Vec<u32>,
        aux_ctype: Vec<u32>,
        aux_tmp: Vec<i32>,
        aux_tmp2: Vec<i32>,
        aux_tmp3: Vec<i32>,
        air: Vec<f32>,
        rigid_bodies: Vec<Voxel3dBodySave>,
    ) -> Self {
        Self {
            version: VOXEL3D_SAVE_VERSION,
            config,
            voxels,
            aux_ctype,
            aux_tmp,
            aux_tmp2,
            aux_tmp3,
            air,
            rigid_bodies,
        }
    }

    /// Creates a snapshot of an empty world.
    ///
    /// The world grid is zeroed and the air grid is left empty, which means
    /// "save no air state": the loader starts from a zeroed coarse field.
    pub fn empty(config: Voxel3dConfig) -> Self {
        Self::new(config, vec![0; VOXEL_COUNT], Vec::new(), Vec::new())
    }

    /// Checks every layout invariant of the payload.
    fn validate(&self) -> Result<(), String> {
        if self.version != VOXEL3D_SAVE_VERSION {
            return Err(format!(
                "unsupported voxel save version {} (this build supports {})",
                self.version, VOXEL3D_SAVE_VERSION
            ));
        }
        if self.voxels.len() != VOXEL_COUNT {
            return Err(format!(
                "voxel save holds {} voxels, expected {}",
                self.voxels.len(),
                VOXEL_COUNT
            ));
        }
        if !self.aux_ctype.is_empty() && self.aux_ctype.len() != VOXEL_COUNT {
            return Err(format!(
                "voxel save holds {} ctype words, expected 0 or {}",
                self.aux_ctype.len(),
                VOXEL_COUNT
            ));
        }
        for (name, len) in [
            ("tmp", self.aux_tmp.len()),
            ("tmp2", self.aux_tmp2.len()),
            ("tmp3", self.aux_tmp3.len()),
        ] {
            if len != 0 && len != VOXEL_COUNT {
                return Err(format!(
                    "voxel save holds {len} {name} words, expected 0 or {VOXEL_COUNT}"
                ));
            }
        }
        if !self.air.is_empty() && !self.air.len().is_multiple_of(AIR_COUNT) {
            return Err(format!(
                "voxel save air grid has {} cells, expected a multiple of {}",
                self.air.len(),
                AIR_COUNT
            ));
        }
        if self.rigid_bodies.len() > MAX_BODIES {
            return Err(format!(
                "voxel save holds {} rigid bodies, at most {} are supported",
                self.rigid_bodies.len(),
                MAX_BODIES
            ));
        }
        for (index, body) in self.rigid_bodies.iter().enumerate() {
            if body.local_voxels.len() > BODY_VOXELS {
                return Err(format!(
                    "rigid body {index} local grid holds {} voxels, at most {} are supported",
                    body.local_voxels.len(),
                    BODY_VOXELS
                ));
            }
        }
        Ok(())
    }

    /// Encodes this snapshot as a `.tpt3` byte buffer.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate()?;

        let payload = bincode::serialize(self)
            .map_err(|err| format!("failed to serialize voxel save data: {err}"))?;

        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(&payload)
            .map_err(|err| format!("failed to compress voxel save data: {err}"))?;
        let compressed = encoder
            .finish()
            .map_err(|err| format!("failed to finish compressing voxel save data: {err}"))?;

        let mut out = Vec::with_capacity(TPT3_MAGIC.len() + compressed.len());
        out.extend_from_slice(TPT3_MAGIC);
        out.extend_from_slice(&compressed);
        Ok(out)
    }

    /// Decodes a `.tpt3` byte buffer produced by [`Voxel3dSaveData::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() <= TPT3_MAGIC.len() {
            return Err("voxel save is too short to contain a header".to_string());
        }
        if &bytes[..TPT3_MAGIC.len()] != TPT3_MAGIC {
            return Err("not a Powder Toy 3D voxel save: bad magic header".to_string());
        }

        let mut decoder = ZlibDecoder::new(&bytes[TPT3_MAGIC.len()..]);
        let mut payload = Vec::new();
        decoder
            .read_to_end(&mut payload)
            .map_err(|err| format!("failed to decompress voxel save data: {err}"))?;

        let data: Self = bincode::deserialize(&payload)
            .map_err(|err| format!("failed to deserialize voxel save data: {err}"))?;

        data.validate()?;
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
                    "failed to create voxel save directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes)
            .map_err(|err| format!("failed to write voxel save file {}: {err}", path.display()))
    }

    /// Reads and decodes a `.tpt3` file from `path`.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|err| format!("failed to read voxel save file {}: {err}", path.display()))?;
        Self::from_bytes(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use tpt_core::particle::EdgeMode;

    /// Serializes without layout validation so malformed payloads can be fed
    /// to `from_bytes`.
    fn encode_raw(data: &Voxel3dSaveData) -> Vec<u8> {
        let payload = bincode::serialize(data).expect("serialize");
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&payload).expect("compress");
        let compressed = encoder.finish().expect("finish compression");
        let mut bytes = TPT3_MAGIC.to_vec();
        bytes.extend_from_slice(&compressed);
        bytes
    }

    fn sample_body(local_voxels: Vec<u32>) -> Voxel3dBodySave {
        Voxel3dBodySave {
            pos: [1.5, 2.5, -3.0],
            quat: [0.0, 0.0, 0.0, 1.0],
            lin_vel: [0.5, -0.25, 0.125],
            ang_vel: [1.0, -2.0, 3.0],
            half_extents: [0.5, 1.0, 1.5],
            ty: 7,
            local_voxels,
        }
    }

    #[test]
    fn empty_roundtrip_and_path_save() {
        let config = Voxel3dConfig {
            paused: true,
            ambient_temp_k: 310.0,
            gravity: [0.0, -3.5, 0.25],
            ..Voxel3dConfig::default()
        };
        let data = Voxel3dSaveData::empty(config);

        let bytes = data.to_bytes().expect("serialize");
        let loaded = Voxel3dSaveData::from_bytes(&bytes).expect("deserialize");
        assert_eq!(loaded, data);
        assert_eq!(loaded.config, config);
        assert_eq!(loaded.voxels.len(), VOXEL_COUNT);
        assert!(loaded.voxels.iter().all(|voxel| *voxel == 0));
        assert!(loaded.air.is_empty());
        assert!(loaded.rigid_bodies.is_empty());

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "tpt3_save_roundtrip_{}_{nanos}.tpt3",
            std::process::id()
        ));
        data.save_to_path(&path).expect("save to path");
        let from_path = Voxel3dSaveData::load_from_path(&path).expect("load from path");
        assert_eq!(from_path, data);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn populated_roundtrip_and_config_survives() {
        let config = Voxel3dConfig {
            dim: 256,
            paused: false,
            ambient_temp_k: 293.15,
            gravity: [1.0, -9.81, -0.5],
            edge_mode: EdgeMode::Solid,
            air_enabled: true,
            air_iterations: 20,
            sun_dir: [0.25, 0.5, 0.75],
            exposure: 1.75,
            render_scale: 0.75,
            ao_rays: 4,
            shadows: false,
        };
        let mut voxels = vec![0; VOXEL_COUNT];
        voxels[0] = 0x0000_0001;
        voxels[VOXEL_COUNT / 2] = 0xDEAD_BEEF;
        voxels[VOXEL_COUNT - 1] = 0x1234_5678;

        let mut air = vec![0.0f32; AIR_COUNT];
        air[0] = 293.15;
        air[AIR_COUNT / 2] = -4.5;
        air[AIR_COUNT - 1] = 1.0e6;

        let bodies = vec![sample_body(vec![0x0102_0304; 512]), sample_body(Vec::new())];
        let data = Voxel3dSaveData::new(config, voxels, air, bodies);

        let bytes = data.to_bytes().expect("serialize");
        let loaded = Voxel3dSaveData::from_bytes(&bytes).expect("deserialize");
        assert_eq!(loaded, data);
        assert_eq!(loaded.config, config);
        assert_eq!(loaded.air[AIR_COUNT / 2], -4.5);
        assert_eq!(loaded.rigid_bodies.len(), 2);
        assert_eq!(loaded.rigid_bodies[0].local_voxels.len(), 512);
    }

    #[test]
    fn aux_arrays_roundtrip() {
        let mut voxels = vec![0u32; VOXEL_COUNT];
        voxels[7] = 0x0000_0005;
        let mut ctype = vec![0u32; VOXEL_COUNT];
        ctype[7] = 0xDEAD_BEEF;
        let mut tmp = vec![0i32; VOXEL_COUNT];
        tmp[7] = -123;
        let mut tmp2 = vec![0i32; VOXEL_COUNT];
        tmp2[7] = 456;
        let mut tmp3 = vec![0i32; VOXEL_COUNT];
        tmp3[7] = i32::MIN;
        let data = Voxel3dSaveData::with_aux(
            Voxel3dConfig::default(),
            voxels,
            ctype,
            tmp,
            tmp2,
            tmp3,
            Vec::new(),
            Vec::new(),
        );
        let bytes = data.to_bytes().expect("serialize");
        let loaded = Voxel3dSaveData::from_bytes(&bytes).expect("deserialize");
        assert_eq!(loaded, data);
        assert_eq!(loaded.aux_ctype[7], 0xDEAD_BEEF);
        assert_eq!(loaded.aux_tmp[7], -123);
        assert_eq!(loaded.aux_tmp2[7], 456);
        assert_eq!(loaded.aux_tmp3[7], i32::MIN);
    }

    #[test]
    fn rejects_bad_magic_and_short_input() {
        assert!(Voxel3dSaveData::from_bytes(&[]).is_err());
        assert!(Voxel3dSaveData::from_bytes(TPT3_MAGIC).is_err());

        let mut bytes = b"NOTTPT03".to_vec();
        bytes.extend_from_slice(&[0u8; 16]);
        let err = Voxel3dSaveData::from_bytes(&bytes).unwrap_err();
        assert!(err.contains("bad magic"), "{err}");
    }

    #[test]
    fn rejects_unsupported_version() {
        let data = Voxel3dSaveData {
            version: VOXEL3D_SAVE_VERSION + 1,
            config: Voxel3dConfig::default(),
            voxels: Vec::new(),
            aux_ctype: Vec::new(),
            aux_tmp: Vec::new(),
            aux_tmp2: Vec::new(),
            aux_tmp3: Vec::new(),
            air: Vec::new(),
            rigid_bodies: Vec::new(),
        };
        let err = Voxel3dSaveData::from_bytes(&encode_raw(&data)).unwrap_err();
        assert!(err.contains("unsupported voxel save version"), "{err}");
        assert!(
            err.contains(&(VOXEL3D_SAVE_VERSION + 1).to_string()),
            "{err}"
        );
    }

    #[test]
    fn rejects_wrong_voxel_count() {
        let data = Voxel3dSaveData {
            version: VOXEL3D_SAVE_VERSION,
            config: Voxel3dConfig::default(),
            voxels: vec![0; VOXEL_COUNT - 1],
            aux_ctype: Vec::new(),
            aux_tmp: Vec::new(),
            aux_tmp2: Vec::new(),
            aux_tmp3: Vec::new(),
            air: Vec::new(),
            rigid_bodies: Vec::new(),
        };
        // Keep the test cheap: the decoded grid is checked before any other
        // payload is built, so only the serialized copy is large.
        let err = Voxel3dSaveData::from_bytes(&encode_raw(&data)).unwrap_err();
        assert!(err.contains("expected"), "{err}");
        assert!(err.contains(&VOXEL_COUNT.to_string()), "{err}");
    }

    #[test]
    fn to_bytes_rejects_invalid_layout() {
        let mut data =
            Voxel3dSaveData::new(Voxel3dConfig::default(), Vec::new(), Vec::new(), Vec::new());
        let err = data.to_bytes().unwrap_err();
        assert!(err.contains("voxels"), "{err}");

        data.voxels = vec![0; VOXEL_COUNT];
        data.air = vec![0.0; AIR_COUNT + 1];
        let err = data.to_bytes().unwrap_err();
        assert!(err.contains("multiple"), "{err}");
    }

    #[test]
    fn rejects_invalid_air_and_body_layout() {
        let mut data = Voxel3dSaveData::empty(Voxel3dConfig::default());

        data.air = vec![0.0; AIR_COUNT + 1];
        let err = Voxel3dSaveData::from_bytes(&encode_raw(&data)).unwrap_err();
        assert!(err.contains("multiple"), "{err}");
        data.air = Vec::new();

        data.rigid_bodies = vec![sample_body(Vec::new()); MAX_BODIES + 1];
        let err = Voxel3dSaveData::from_bytes(&encode_raw(&data)).unwrap_err();
        assert!(err.contains("rigid bodies"), "{err}");
        assert!(err.contains(&MAX_BODIES.to_string()), "{err}");

        data.rigid_bodies = vec![sample_body(vec![0; BODY_VOXELS + 1])];
        let err = Voxel3dSaveData::from_bytes(&encode_raw(&data)).unwrap_err();
        assert!(err.contains("local grid"), "{err}");
        assert!(err.contains(&BODY_VOXELS.to_string()), "{err}");
    }
}
