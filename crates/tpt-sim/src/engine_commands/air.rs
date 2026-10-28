//! Whole-canvas air actions that are not brush tools.
//!
//! These are the keyboard-triggered air commands of `GameView`: they rewrite
//! whole Eulerian fields at once instead of painting a brush footprint.

use crate::SimulationEngine;

/// `Air::Invert`'s per-field body (`Air.cpp:499-513`): negate every cell.
pub(super) fn negate_in_place(field: &mut [f32]) {
    for value in field {
        *value = -*value;
    }
}

/// `Air::ClearAirH` (`Air.cpp:71-74`): fill the ambient heat field with the
/// configured ambient air temperature.
pub(super) fn reset_ambient_heat_field(field: &mut [f32], ambient_air_temp: f32) {
    field.fill(ambient_air_temp);
}

impl SimulationEngine {
    /// `Air::Invert` (`Air.cpp:499-513`, `GameController::InvertAirSim`):
    /// negate pressure and both velocity components of every cell.
    ///
    /// The reference loops `XCELLS x YCELLS`, which is exactly the padded host
    /// arrays here; the flow/velocity kernels consume the inverted fields on
    /// the next tick.
    pub(super) fn invert_air(&mut self) {
        negate_in_place(&mut self.bufs.air_pv);
        negate_in_place(&mut self.bufs.air_vx);
        negate_in_place(&mut self.bufs.air_vy);
        self.bufs.host_air_dirty = true;
        self.bufs.host_dirty = true;
    }

    /// `Air::ClearAirH` (`Air.cpp:71-74`, `GameController::ResetAHeat`): reset
    /// every ambient heat cell to `ambientAirTemp`.
    pub(super) fn reset_ambient_heat(&mut self) {
        reset_ambient_heat_field(&mut self.bufs.air_hv, self.bufs.config.ambient_air_temp);
        self.bufs.host_air_dirty = true;
        self.bufs.host_dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::{negate_in_place, reset_ambient_heat_field};

    #[test]
    fn invert_negates_pressure_and_both_velocity_arrays() {
        let mut pv = vec![1.0f32, -2.5, 0.0, 4.0];
        let mut vx = vec![-3.0f32, 0.5, 0.0, -1.25];
        let mut vy = vec![7.0f32, -0.0, 2.0, -8.5];

        negate_in_place(&mut pv);
        negate_in_place(&mut vx);
        negate_in_place(&mut vy);

        assert_eq!(pv, vec![-1.0, 2.5, -0.0, -4.0]);
        assert_eq!(vx, vec![3.0, -0.5, -0.0, 1.25]);
        assert_eq!(vy, vec![-7.0, 0.0, -2.0, 8.5]);
    }

    #[test]
    fn reset_ambient_heat_fills_every_cell() {
        // `Air::ClearAirH` (`Air.cpp:71-74`) fills with `ambientAirTemp`.
        let mut hv = vec![300.0f32, 120.0, 512.0];
        reset_ambient_heat_field(&mut hv, 293.15);
        assert_eq!(hv, vec![293.15; 3]);
    }
}
