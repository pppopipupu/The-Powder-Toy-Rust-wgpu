//! The device FFT Poisson solve must reproduce the point-mass convolution.
//!
//! `gravity_gpu` replaces the CPU solver with a padded radix-2 FFT on the
//! device, so the only thing worth pinning is the result: the force grids it
//! writes must match a direct `O(n^2)` convolution of the same mass grid with
//! `-M_GRAV * d / |d|^3`.

use super::*;

/// Straight `O(n^2)` convolution with the point-mass kernel:
/// `force(p) = sum_q mass(q) * (-M_GRAV) * (p - q) / |p - q|^3`.
///
/// The FFT path stores the kernel with `scale = -M_GRAV / (4 * NCELL)` and runs
/// an *unnormalised* inverse transform, whose `4 * NCELL` factor cancels that
/// division — exactly as FFTW does in `Fft.cpp`. The direct reference therefore
/// uses the physical `-M_GRAV` scale.
fn brute_force(xcells: usize, ycells: usize, mass: &[f32], mask: &[u32]) -> (Vec<f32>, Vec<f32>) {
    let scale = -6.673e-1f32;
    let mut force_x = vec![0.0f32; xcells * ycells];
    let mut force_y = vec![0.0f32; xcells * ycells];
    for py in 0..ycells {
        for px in 0..xcells {
            let mut sx = 0.0f32;
            let mut sy = 0.0f32;
            for qy in 0..ycells {
                for qx in 0..xcells {
                    let i = qy * xcells + qx;
                    if mask[i] == 0 || mass[i] == 0.0 {
                        continue;
                    }
                    let dx = px as f32 - qx as f32;
                    let dy = py as f32 - qy as f32;
                    let d2 = dx * dx + dy * dy;
                    if d2 == 0.0 {
                        continue;
                    }
                    let inv = scale / (d2 * d2.sqrt());
                    sx += mass[i] * dx * inv;
                    sy += mass[i] * dy * inv;
                }
            }
            let i = py * xcells + px;
            if mask[i] != 0 {
                force_x[i] = sx;
                force_y[i] = sy;
            }
        }
    }
    (force_x, force_y)
}

fn relative_error(actual: &[f32], expected: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, e) in actual.iter().zip(expected.iter()) {
        let d = *a as f64 - *e as f64;
        num += d * d;
        den += (*e as f64) * (*e as f64);
    }
    if den == 0.0 {
        return num.sqrt() as f32;
    }
    (num.sqrt() / den.sqrt()) as f32
}

#[test]
fn test_gravity_fft_matches_brute_force() {
    let _guard = gpu_lock();
    let mut engine = create_test_engine(96, 96);
    engine.bufs.config.newton_grav = true;
    let xcells = engine.bufs.config.xcells() as usize;
    let ycells = engine.bufs.config.ycells() as usize;
    let cells = xcells * ycells;

    // A handful of point masses, including negative (white-hole-like) ones,
    // seeded through the tool grid the `PGRV`/`NGRV` tools write to.
    let mut mass = vec![0.0f32; engine.bufs.grav_tool_mass.len()];
    mass[xcells + 2] = 3.0;
    mass[4 * xcells + 5] = -2.0;
    mass[6 * xcells + 1] = 1.5;
    let tool_mass = mass[..cells].to_vec();
    engine.bufs.grav_tool_mass = mass;
    engine.bufs.grav_tool_mass_dirty = true;

    // `PGRV.cpp:16` writes `gravIn.mass` between frames and `BeforeSim`
    // dispatches that grid in the very next frame, clearing it immediately
    // afterwards (`Simulation.cpp:3739-3744`). A tool well therefore has no
    // second snapshot: it must be solved by this step and must not survive
    // into the next one.
    step(&mut engine);
    assert!(
        engine.bufs.sync_gravity_gpu_to_host(),
        "the gravity shadows must come back from the device"
    );

    let mask = engine.bufs.grav_mask[..cells].to_vec();
    let (want_x, want_y) = brute_force(xcells, ycells, &tool_mass, &mask);
    let got_x = &engine.bufs.grav_fx[..cells];
    let got_y = &engine.bufs.grav_fy[..cells];
    let err_x = relative_error(got_x, &want_x);
    let err_y = relative_error(got_y, &want_y);
    assert!(
        err_x < 0.02 && err_y < 0.02,
        "device FFT differs from the direct convolution: rel err x {err_x}, y {err_y}"
    );

    // The accumulator is cleared after the dispatch: the well was consumed by
    // this solve, so the host shadow must not still hold it.
    assert!(
        engine.bufs.grav_mass[xcells + 2].abs() < 1e-3,
        "the mass accumulator must be cleared after the dispatch, got {}",
        engine.bufs.grav_mass[xcells + 2]
    );

    // The field must actually point towards the strongest source.
    let probe = xcells + 3;
    assert!(
        got_x[probe] < 0.0,
        "a cell to the right of a positive mass is pulled left, got {}",
        got_x[probe]
    );
}
