//! Fireworks launch and burst kernels.

use cubecl::prelude::*;
use tpt_core::constants::*;

use crate::move_kernel::eval_move_at;

use super::*;

/// Pixel `x` of a linear pixel index.
#[cube]
fn px_of(idx: usize, xres: u32) -> i32 {
    (idx as u32 % xres) as i32
}

/// Pixel `y` of a linear pixel index.
#[cube]
fn py_of(idx: usize, xres: u32) -> i32 {
    (idx as u32 / xres) as i32
}

/// Deterministic per-pixel RNG seed for one of a firework's draws.
///
/// TPT draws every kick from the shared `sim->rng`, whose result depends on the
/// update order; a pixel-parallel pass has to derive an independent draw from
/// the pixel, the frame seed and a per-call `salt` instead. Both products go
/// through [`mix_index`] so the host build cannot constant-fold them (the
/// `ABSOLUTE_POS` literal of the unexpanded build would otherwise overflow).
#[cube]
fn seed_of(idx: usize, frame_seed: u32, salt: u32) -> u32 {
    hash_u32(frame_seed ^ mix_index(idx as u32, 0x9E37_79B9u32) ^ mix_index(salt, 0x85EB_CA6Bu32))
}

/// `Renderer::firwTableAt` (`Renderer.cpp:1607-1615`, `Gradient.cpp:3-30`):
/// the 200-entry magenta→blue→cyan→green→yellow→red gradient.
///
/// `Gradient` walks `point = i / resolution`, blends each pair of neighbouring
/// stops with `RGB::Blend` using `alpha = uint8(f * 0xFF)`, and leaves the
/// entries before the first stop black (there are none here).
#[cube]
fn firw_colour(index: u32) -> u32 {
    let point = index as f32 / 200.0f32;
    let mut left = 0xFF00FFu32;
    let mut right = 0x0000FFu32;
    let mut lo = 0.0f32;
    if point >= 0.8f32 {
        left = 0xFFFF00u32;
        right = 0xFF0000u32;
        lo = 0.8f32;
    } else if point >= 0.6f32 {
        left = 0x00FF00u32;
        right = 0xFFFF00u32;
        lo = 0.6f32;
    } else if point >= 0.4f32 {
        left = 0x00FFFFu32;
        right = 0x00FF00u32;
        lo = 0.4f32;
    } else if point >= 0.2f32 {
        left = 0x0000FFu32;
        right = 0x00FFFFu32;
        lo = 0.2f32;
    }
    let f = (point - lo) / 0.2f32;
    let alpha = (f * 255.0f32) as u32;
    let mut out = 0u32;
    for shift in 0u32..3u32 {
        let sh = shift * 8u32;
        let l = (left >> sh) & 0xFFu32;
        let r = (right >> sh) & 0xFFu32;
        let c = (alpha * r + (255u32 - alpha) * l) / 255u32;
        out |= c << sh;
    }
    out
}

/// `FIRW.cpp:96-114` / `FWRK.cpp:102-124`: the 40-ember burst.
///
/// Every ember leaves the same pixel (`create_part(-3)`), inherits half the
/// parent's velocity plus a random 2.0..4.95 px/frame kick, and takes the
/// burst colour, `tmp = 1`, a 70..109 life and a 5750..6249 K temperature.
#[cube]
#[allow(clippy::too_many_arguments)]
fn queue_firework_burst(
    create_queue: &mut [Atomic<u32>],
    px: i32,
    py: i32,
    parent_vx: f32,
    parent_vy: f32,
    col: u32,
    dcolour: u32,
    seed: u32,
) {
    for n in 0u32..40u32 {
        let r = hash_u32(seed ^ mix_index(n, 0x9E37_79B9u32));
        let magnitude = (40u32 + (hash_u32(r ^ 0x01u32) % 60u32)) as f32 * 0.05f32;
        let angle = rand_unit(hash_u32(r ^ 0x02u32)) * core::f32::consts::TAU;
        let life = 70i32 + (hash_u32(r ^ 0x03u32) % 40u32) as i32;
        let temp = 5750.0f32 + (hash_u32(r ^ 0x04u32) % 500u32) as f32;
        request_create(
            create_queue,
            PT_EMBR,
            px,
            py,
            CR_NO_OCCUPANCY
                | CR_SET_CTYPE
                | CR_SET_TMP
                | CR_SET_LIFE
                | CR_SET_TEMP
                | CR_SET_DCOLOUR
                | CR_SET_VX
                | CR_SET_VY,
            life,
            col,
            temp,
            1i32,
            0i32,
            0i32,
            0i32,
            dcolour,
            0u32,
            parent_vx * 0.5f32 + angle.cos() * magnitude,
            parent_vy * 0.5f32 + angle.sin() * magnitude,
        );
    }
}

/// Word offset of [`GpuElementProps::gravity`].
const WORD_GRAVITY: u32 = 9;
const _: () = assert!(
    core::mem::offset_of!(tpt_core::elements::GpuElementProps, gravity)
        == WORD_GRAVITY as usize * 4
);

/// `GravityMode` discriminants (`tpt_core::particle`).
const GRAV_VERTICAL: u32 = 0;
const GRAV_RADIAL: u32 = 2;
const GRAV_CUSTOM: u32 = 3;

/// `Simulation::GetGravityField` (`Simulation.cpp:2088-2124`) for a particle
/// of gravity `particle_grav`: vertical uses the element's own gravity, radial
/// points at the canvas centre (the reference's `XCNTR`/`YCNTR`) and custom
/// uses the configured vector. `Off` leaves both components zero.
#[cube]
#[allow(clippy::too_many_arguments)]
fn config_gravity(
    mode: u32,
    custom_gx: f32,
    custom_gy: f32,
    particle_grav: f32,
    px: i32,
    py: i32,
    xres: u32,
    yres: u32,
) -> (f32, f32) {
    let mut gx = 0.0f32;
    let mut gy = 0.0f32;
    if mode == GRAV_VERTICAL {
        gy = particle_grav;
    } else if mode == GRAV_RADIAL {
        let dx = (px - (xres / 2u32) as i32) as f32;
        let dy = (py - (yres / 2u32) as i32) as f32;
        if dx != 0.0f32 || dy != 0.0f32 {
            // `0.01 - hypot(dx, dy)` is the reference's signed denominator,
            // which makes the vector point inwards.
            let d = 0.01f32 - (dx * dx + dy * dy).sqrt();
            gx = particle_grav * (dx / d);
            gy = particle_grav * (dy / d);
        }
    } else if mode == GRAV_CUSTOM {
        gx = particle_grav * custom_gx;
        gy = particle_grav * custom_gy;
    }
    (gx, gy)
}

/// `FWRK.cpp:60-63` / `FIRW.cpp:70-74`: the configured gravity field plus the
/// random half-strength kick used when it is near zero.
#[cube]
#[allow(clippy::too_many_arguments)]
fn firework_gravity(
    mode: u32,
    custom_gx: f32,
    custom_gy: f32,
    particle_grav: f32,
    px: i32,
    py: i32,
    xres: u32,
    yres: u32,
    seed: u32,
) -> (f32, f32) {
    let (mut gx, mut gy) = config_gravity(
        mode,
        custom_gx,
        custom_gy,
        particle_grav,
        px,
        py,
        xres,
        yres,
    );
    if gx * gx + gy * gy < 0.001f32 {
        let angle = rand_unit(seed) * core::f32::consts::TAU;
        gx += angle.sin() * particle_grav * 0.5f32;
        gy += angle.cos() * particle_grav * 0.5f32;
    }
    (gx, gy)
}

/// `FIRW` and `FWRK`: the launch kick and the `EMBR` burst.
///
/// The launch probe is `Simulation::eval_move` (`FWRK.cpp:78`) against the
/// movement tables, and the kick uses `GetGravityField`'s configured vector
/// (vertical/radial/custom, `FWRK.cpp:60-63` / `FIRW.cpp:70-74`). Both bursts
/// queue their 40 embers and add the reference's 8 pressure units
/// (`FIRW.cpp:115` / `FWRK.cpp:125`).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn firework_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_ctype: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_dcolour: &mut [u32],
    air_pv: &mut [f32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    move_lut: &[u32],
    elem_words: &[u32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
    grav_mode: u32,
    custom_gx: f32,
    custom_gy: f32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty == PT_FIRW {
                // `FIRW.cpp:57-115`.
                let mut tmp = part_tmp[id];
                let mut life = part_life[id];
                let mut done = false;
                if tmp <= 0i32 {
                    let mut ignite = false;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(
                                pmap,
                                xres,
                                yres,
                                px_of(idx, xres) + dx,
                                py_of(idx, xres) + dy,
                            );
                            if nv != 0u32 {
                                let nty = part_ty[pmap_slot(nv)];
                                if nty == PT_FIRE || nty == PT_PLSM || nty == PT_THDR {
                                    ignite = true;
                                }
                            }
                        }
                    }
                    if ignite {
                        // `FIRW.cpp:69-77`: `GetGravityField` with the
                        // element's own gravity, then the half-strength random
                        // kick when the field is near zero.
                        let firw_grav =
                            f32::reinterpret(elem_word(elem_words, PT_FIRW, WORD_GRAVITY));
                        let (gx, gy) = firework_gravity(
                            grav_mode,
                            custom_gx,
                            custom_gy,
                            firw_grav,
                            px_of(idx, xres),
                            py_of(idx, xres),
                            xres,
                            yres,
                            seed_of(idx, frame_seed, 0x1u32),
                        );
                        tmp = 1i32;
                        life = 20i32 + rand_range(seed_of(idx, frame_seed, 0x2u32), 0i32, 9i32);
                        let len = (gx * gx + gy * gy).sqrt();
                        let mult = (life + 20i32) as f32 * 0.2f32 / len;
                        part_vx[id] -= gx * mult;
                        part_vy[id] -= gy * mult;
                    }
                } else if tmp == 1i32 {
                    if life <= 0i32 {
                        tmp = 2i32;
                    }
                } else {
                    // `FIRW.cpp:90-117`: one palette colour per burst, 40
                    // embers on the same pixel, then the pressure pulse and
                    // the parent's death.
                    let col = firw_colour(hash_u32(seed_of(idx, frame_seed, 0x3u32)) % 200u32);
                    let pvx = part_vx[id];
                    let pvy = part_vy[id];
                    queue_firework_burst(
                        create_queue,
                        px_of(idx, xres),
                        py_of(idx, xres),
                        pvx,
                        pvy,
                        col,
                        part_dcolour[id],
                        seed_of(idx, frame_seed, 0x4u32),
                    );
                    let cidx = (((py_of(idx, xres) as u32) / CELL) * xcells
                        + ((px_of(idx, xres) as u32) / CELL))
                        as usize;
                    air_pv[cidx] += 8.0f32;
                    part_ty[id] = PT_NONE;
                    done = true;
                }
                if !done {
                    part_tmp[id] = tmp;
                    part_life[id] = life;
                }
            } else if ty == PT_FWRK {
                // `FWRK.cpp:60-110`.
                let mut done = false;
                let mut launched = false;
                let life = part_life[id];
                let mut surround = false;
                for n in 0u32..9u32 {
                    let dx = (n % 3u32) as i32 - 1i32;
                    let dy = (n / 3u32) as i32 - 1i32;
                    if (dx != 0i32 || dy != 0i32)
                        && pmap_at(
                            pmap,
                            xres,
                            yres,
                            px_of(idx, xres) + dx,
                            py_of(idx, xres) + dy,
                        ) == 0u32
                    {
                        surround = true;
                    }
                }
                let temp = part_temp[id];
                let heat_launch = surround
                    && temp > 400.0f32
                    && chance(
                        seed_of(idx, frame_seed, 0x8u32),
                        (9.0f32 + temp / 40.0f32) as u32,
                        100_000u32,
                    );
                if life == 0i32 && (heat_launch || part_ctype[id] == PT_DUST) {
                    // `FWRK.cpp:60-78`: the configured gravity field, the
                    // half-strength random kick, and an `eval_move` probe one
                    // pixel opposite gravity before the launch.
                    let fwrk_grav = f32::reinterpret(elem_word(elem_words, PT_FWRK, WORD_GRAVITY));
                    let (mut gx, mut gy) = firework_gravity(
                        grav_mode,
                        custom_gx,
                        custom_gy,
                        fwrk_grav,
                        px_of(idx, xres),
                        py_of(idx, xres),
                        xres,
                        yres,
                        seed_of(idx, frame_seed, 0x9u32),
                    );
                    let agx = if gx < 0.0f32 { -gx } else { gx };
                    let agy = if gy < 0.0f32 { -gy } else { gy };
                    let gmax = if agy > agx { agy } else { agx };
                    let probe_x = (px_of(idx, xres) as f32 - gx / gmax + 0.5f32) as i32;
                    let probe_y = (py_of(idx, xres) as f32 - gy / gmax + 0.5f32) as i32;
                    let fwrk_props = elem_word(elem_words, PT_FWRK, WORD_PROPERTIES);
                    if eval_move_at(
                        pmap,
                        wall_bmap,
                        wall_emap,
                        move_lut,
                        fwrk_props,
                        PT_FWRK,
                        probe_x,
                        probe_y,
                        xres as i32,
                        yres as i32,
                        xcells,
                    ) != 0u32
                    {
                        let len = (gx * gx + gy * gy).sqrt();
                        let mult = 15.0f32 / len;
                        let r1 =
                            rand_range(seed_of(idx, frame_seed, 0xAu32), -100i32, 100i32) as f32;
                        gx += gx * r1 * 0.002f32;
                        gy += gy * r1 * 0.002f32;
                        let r2 =
                            rand_range(seed_of(idx, frame_seed, 0xBu32), -100i32, 100i32) as f32;
                        gx += -gy * r2 * 0.005f32;
                        gy += gx * r2 * 0.005f32;
                        part_vx[id] -= gx * mult;
                        part_vy[id] -= gy * mult;
                        part_life[id] =
                            18i32 + rand_range(seed_of(idx, frame_seed, 0xCu32), 0i32, 9i32);
                        part_ctype[id] = 0u32;
                        launched = true;
                    }
                }
                if !launched && life < 3i32 && life > 0i32 {
                    // `FWRK.cpp:96-127`: a random RGB burst colour, 40 embers
                    // on the same pixel, the pressure pulse and the parent's
                    // death.
                    let r = rand_range(seed_of(idx, frame_seed, 0xDu32), 11i32, 255i32) as u32;
                    let g = rand_range(seed_of(idx, frame_seed, 0xEu32), 11i32, 255i32) as u32;
                    let b = rand_range(seed_of(idx, frame_seed, 0xFu32), 11i32, 255i32) as u32;
                    let col = (r << 16u32) | (g << 8u32) | b;
                    let pvx = part_vx[id];
                    let pvy = part_vy[id];
                    queue_firework_burst(
                        create_queue,
                        px_of(idx, xres),
                        py_of(idx, xres),
                        pvx,
                        pvy,
                        col,
                        part_dcolour[id],
                        seed_of(idx, frame_seed, 0x10u32),
                    );
                    let cidx = (((py_of(idx, xres) as u32) / CELL) * xcells
                        + ((px_of(idx, xres) as u32) / CELL))
                        as usize;
                    air_pv[cidx] += 8.0f32;
                    part_ty[id] = PT_NONE;
                    done = true;
                }
                if !done && part_life[id] >= 45i32 {
                    part_life[id] = 0i32;
                }
            }
        }
    }
}
