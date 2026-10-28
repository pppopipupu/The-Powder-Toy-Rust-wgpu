//! Per-tick 3D statistics. Filled by the sim engine slice.

/// Lightweight counters reported by the HUD.
#[derive(Clone, Copy, Debug, Default)]
pub struct Voxel3dStats {
    /// Active bricks dispatched this tick.
    pub active_bricks: u32,
    /// Live voxels, when the engine has a device-side count.
    pub active_voxels: u32,
    /// Simulation milliseconds of the last tick.
    pub sim_ms: f32,
}
