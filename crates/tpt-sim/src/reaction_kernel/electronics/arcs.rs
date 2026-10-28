//! Tesla arcs, electrode arcs and delays.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{
    CR_SET_LIFE, CR_SET_TEMP, CR_SET_TMP, CR_SET_TMP2, hash_u32, iabs, insulated, pmap_at,
    pmap_slot, rand_unit, request_create,
};
use super::SPARK_LIFE;
use super::bolt::{FLAG_ETRD_BOLT, pack_coords_cube};

/// `ETRD.cpp:74`: the ordered neighbour table (`deltaPos`) only covers offsets
/// with `|dx| + |dy| <= 12`; the pool-scan fallback (`ETRD.cpp:130-160`) has no
/// such limit and reaches the whole canvas up to the `tmp2` maximum.
const ETRD_SEARCH_RADIUS: i32 = 12;
/// `DLAY.cpp`: the delay countdown clamps the temperature to freezing point.
const DLAY_MIN_TEMP: f32 = 274.15;

/// `|v|` for floats (CubeCL has no `f32::abs` in the portable subset).
#[cube]
fn fabsf(v: f32) -> f32 {
    let mut r = v;
    if v < 0.0f32 {
        r = -v;
    }
    r
}

/// Integer square root used by the pool-scan fallback, where the distance can
/// exceed the 12 px fast-path radius. Exact floor, unlike an `f32::sqrt` cast
/// for squared distances above the 24-bit mantissa.
#[cube]
fn isqrt_i32(v_in: i32) -> i32 {
    let mut v = v_in;
    // Seeded from a runtime anchor (AGENTS rule 4); `bit` starts with the
    // highest even power of two that fits in the 31-bit positive range.
    #[allow(clippy::erasing_op)]
    let mut r = v_in * 0i32;
    #[allow(clippy::erasing_op)]
    let mut bit = v_in * 0i32 + (1i32 << 30i32);
    for _ in 0u32..16u32 {
        let t = r + bit;
        if v >= t {
            v -= t;
            r = (r >> 1i32) + bit;
        } else {
            r >>= 1i32;
        }
        bit >>= 2i32;
    }
    r
}

/// `SPRK.cpp` `case PT_TESC` bolt life:
/// `rng.between(0, 2 + tmp/15) + tmp/7`, capped at 60 (the reference's
/// `parts[i].life > 60` guard names the coil, so this cap mirrors the port's
/// existing `electronics::tesc_ligh_life`, not a second random draw).
#[cube]
#[allow(clippy::manual_clamp)] // `.clamp()` is not available inside kernels
fn tesc_ligh_life(power: i32, roll: u32) -> i32 {
    let mut p = power;
    if p > 300i32 {
        p = 300i32;
    }
    if p < 0i32 {
        p = 0i32;
    }
    let span = 2i32 + p / 15i32;
    let mut life = (roll % (span as u32 + 1u32)) as i32 + p / 7i32;
    if life > 60i32 {
        life = 60i32;
    }
    life
}

/// Euclidean integer hypotenuse `int(hypot(dx, dy))`, matching TPT's
/// `int(std::hypot(d.X, d.Y))` for offsets up to radius 16.
#[cube]
fn ihypot(dx: i32, dy: i32) -> i32 {
    let d2 = dx * dx + dy * dy;
    // Seeded from a runtime expression on purpose: CubeCL mis-expands an
    // assignment to a local that started as a host literal (see the kernel
    // rules in `AGENTS.md`), so the zero is computed from `dx`.
    #[allow(clippy::erasing_op)]
    let mut r = dx * 0i32;
    for i in 1i32..18i32 {
        if i * i <= d2 {
            r = i;
        }
    }
    r
}

/// Direction of a neighbour offset in degrees, matching TPT's
/// `atan2(-ry, rx) * 360 / pi` for the eight primary directions.
#[cube]
fn octant_degrees(dx: i32, dy: i32) -> i32 {
    let mut deg = 0i32;
    if dx > 0i32 {
        if dy < 0i32 {
            deg = 45i32;
        } else if dy > 0i32 {
            deg = 315i32;
        }
    } else if dx < 0i32 {
        if dy < 0i32 {
            deg = 135i32;
        } else if dy > 0i32 {
            deg = 225i32;
        } else {
            deg = 180i32;
        }
    } else if dy < 0i32 {
        deg = 90i32;
    } else if dy > 0i32 {
        deg = 270i32;
    }
    deg
}

/// `SPRK.cpp` `case PT_TESC` / `case PT_ETRD` and `DLAY.cpp`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_clamp)] // `.clamp()` is not available inside kernels
pub(crate) fn arc_and_delay_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    create_queue: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
    xcells: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty != PT_NONE {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;

                if ty == PT_SPRK && part_ctype[id] == PT_TESC {
                    // `SPRK.cpp:130-170` `case PT_TESC`: the coil's `tmp` is
                    // its power. `TESC.cpp` `create` already clamps a painted
                    // coil to 300; the update clamps again.
                    let mut power = part_tmp[id];
                    if power > 300i32 {
                        power = 300i32;
                    }
                    if power < 0i32 {
                        power = 0i32;
                    }
                    part_tmp[id] = power;
                    if power > 4i32 {
                        // One independent `chance(1, power*power/20 + 6)` trial
                        // per empty neighbour; every success creates its own
                        // bolt (the old `fired` latch stopped after the first).
                        let denom = (power * power / 20i32 + 6i32) as f32;
                        let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                        for n in 0u32..9u32 {
                            let dx = (n % 3u32) as i32 - 1i32;
                            let dy = (n / 3u32) as i32 - 1i32;
                            if (dx != 0i32 || dy != 0i32)
                                && pmap_at(pmap, xres, yres, px + dx, py + dy) == 0u32
                            {
                                let roll_seed = frame_seed ^ ((id as u32) << 8u32) ^ n;
                                let roll = rand_unit(roll_seed);
                                if roll * denom < 1.0f32 {
                                    // `create_part(-1, x+rx*2, y+ry*2, PT_LIGH)`:
                                    // the empty neighbour was only the path
                                    // test, the bolt starts one cell further
                                    // out. `LIGH.tmp` is the angle,
                                    // `tmp2 = 1` the deferred bending mode.
                                    let life =
                                        tesc_ligh_life(power, hash_u32(roll_seed ^ 0x9E37_79B9u32));
                                    request_create(
                                        create_queue,
                                        PT_LIGH,
                                        px + dx * 2i32,
                                        py + dy * 2i32,
                                        CR_SET_LIFE | CR_SET_TEMP | CR_SET_TMP | CR_SET_TMP2,
                                        life,
                                        0u32,
                                        life as f32 * (power as f32) / 2.5f32,
                                        octant_degrees(dx, dy),
                                        1i32,
                                        0i32,
                                        0i32,
                                        0u32,
                                        0u32,
                                        0.0f32,
                                        0.0f32,
                                    );
                                    // Slight self-cooling, once per bolt
                                    // (`SPRK.cpp:157`).
                                    let mut temp = part_temp[id]
                                        - ((power * 2) as f32 + part_temp[id] / 5.0f32);
                                    if temp < MIN_TEMP {
                                        temp = MIN_TEMP;
                                    }
                                    if temp > MAX_TEMP {
                                        temp = MAX_TEMP;
                                    }
                                    part_temp[id] = temp;
                                    // `SPRK.cpp:158-164`: the bolt creation
                                    // damps the cell pressure by half a unit,
                                    // snapping a cell at or below 0.5 to zero.
                                    let pv = air_pv[cidx];
                                    if fabsf(pv) != 0.0f32 {
                                        if fabsf(pv) <= 0.5f32 {
                                            air_pv[cidx] = 0.0f32;
                                        } else if pv > 0.0f32 {
                                            air_pv[cidx] = pv - 0.5f32;
                                        } else {
                                            air_pv[cidx] = pv + 0.5f32;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if ty == PT_SPRK && part_ctype[id] == PT_ETRD && part_life[id] == 1i32 {
                    // `SPRK.cpp:100-113`: relay into the nearest idle electrode.
                    // The search is `Element_ETRD_nearestSparkablePart`
                    // (`ETRD.cpp:93-160`). `tmp` is the minimum distance, `tmp2`
                    // the (optional) maximum; the neighbour table only reaches
                    // 12 px, the pool-scan fallback reaches the whole canvas.
                    let min_dist = part_tmp[id];
                    let tmp2 = part_tmp2[id];
                    let max_distance = ((xres * xres + yres * yres) as f32).sqrt() as i32;
                    let mut found_distance = max_distance;
                    if tmp2 != 0i32 && tmp2 < max_distance {
                        found_distance = tmp2;
                    }
                    // `ETRD.cpp:101-102`: invalid range if a maximum is set
                    // below the minimum.
                    let mut valid = true;
                    if tmp2 != 0i32 && min_dist > tmp2 {
                        valid = false;
                    }
                    let mut best = id as i32;
                    let mut best_x = px;
                    let mut best_y = py;
                    if valid {
                        // `ETRD.cpp:113-129` fast path: the ordered `deltaPos`
                        // table (`|dx| + |dy| <= 12`), nearest candidate wins.
                        let mut best_dist = found_distance;
                        let span = 2i32 * ETRD_SEARCH_RADIUS + 1i32;
                        for n in 0u32..((span * span) as u32) {
                            let dx = (n as i32 % span) - ETRD_SEARCH_RADIUS;
                            let dy = (n as i32 / span) - ETRD_SEARCH_RADIUS;
                            let manhattan = iabs(dx) + iabs(dy);
                            if (dx != 0i32 || dy != 0i32) && manhattan <= ETRD_SEARCH_RADIUS {
                                let dist = ihypot(dx, dy);
                                if dist > min_dist && dist < best_dist {
                                    let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                    if nv != 0u32 {
                                        let nslot = pmap_slot(nv);
                                        if part_ty[nslot] == PT_ETRD
                                            && part_life[nslot] == 0i32
                                            && nslot != id
                                        {
                                            best = nslot as i32;
                                            best_dist = dist;
                                            best_x = px + dx;
                                            best_y = py + dy;
                                        }
                                    }
                                }
                            }
                        }
                        // `ETRD.cpp:130-160`: the pool scan has no 12 px cap.
                        // An `ETRD` always owns a `pmap` pixel, so walking the
                        // map is equivalent to walking `parts[0..active]`.
                        if best == id as i32 {
                            let mut scan_dist = found_distance;
                            let pixels = xres * yres;
                            for k in 0u32..pixels {
                                let nv = pmap[k as usize];
                                if nv != 0u32 {
                                    let nslot = pmap_slot(nv);
                                    if part_ty[nslot] == PT_ETRD
                                        && part_life[nslot] == 0i32
                                        && nslot != id
                                    {
                                        let nx = (k % xres) as i32;
                                        let ny = (k / xres) as i32;
                                        let ddx = nx - px;
                                        let ddy = ny - py;
                                        let dist = isqrt_i32(ddx * ddx + ddy * ddy);
                                        if dist < scan_dist && dist > min_dist {
                                            scan_dist = dist;
                                            best = nslot as i32;
                                            best_x = nx;
                                            best_y = ny;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if best != id as i32 {
                        // `SPRK.cpp:105-106`: `parts_avg(i, nearp, PT_INSL)`
                        // must be neither `INSL` nor `RSSS`. The reference picks
                        // the nearest electrode first and only then rejects the
                        // arc, so the insulation test is not part of the search.
                        if !insulated(pmap, xres, yres, best_x, best_y, px, py) {
                            // The source relays: it becomes `ETRD` again with a
                            // 20 frame cooldown while the target electrode lights.
                            part_ty[id] = PT_ETRD;
                            part_ctype[id] = PT_NONE;
                            part_life[id] = 20i32;
                            part_ty[best as usize] = PT_SPRK;
                            part_ctype[best as usize] = PT_ETRD;
                            part_life[best as usize] = 9i32;
                            part_tmp[id] = pack_coords_cube(px, py);
                            part_tmp2[id] = pack_coords_cube(best_x, best_y);
                            part_flags[id] |= FLAG_ETRD_BOLT;
                        }
                    }
                } else if ty == PT_DLAY {
                    // `DLAY.cpp`: the delay length is the temperature above
                    // freezing point.
                    let old_life = part_life[id];
                    let mut life = old_life;
                    if life > 0i32 {
                        life -= 1i32;
                    }
                    if part_temp[id] <= DLAY_MIN_TEMP {
                        part_temp[id] = DLAY_MIN_TEMP;
                    }
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nx = px + dx;
                            let ny = py + dy;
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv != 0u32 && !insulated(pmap, xres, yres, nx, ny, px, py) {
                                let nslot = pmap_slot(nv);
                                let nty = part_ty[nslot];
                                if nty == PT_SPRK
                                    && life == 0i32
                                    && part_life[nslot] > 0i32
                                    && part_life[nslot] < SPARK_LIFE
                                    && part_ctype[nslot] == PT_PSCN
                                {
                                    life = (part_temp[id] - 273.15f32 + 0.5f32) as i32;
                                } else if nty == PT_DLAY {
                                    if life == 0i32 {
                                        if part_life[nslot] > 0i32 {
                                            life = part_life[nslot];
                                            // The neighbour may already have
                                            // counted down this frame.
                                            if nslot > id {
                                                life -= 1i32;
                                            }
                                        }
                                    } else if part_life[nslot] == 0i32 {
                                        let mut handed = life;
                                        if nslot > id {
                                            handed += 1i32;
                                        }
                                        part_life[nslot] = handed;
                                    }
                                } else if nty == PT_NSCN && old_life == 1i32 {
                                    // A relayed pulse leaves through the
                                    // n-type silicon.
                                    part_ty[nslot] = PT_SPRK;
                                    part_ctype[nslot] = PT_NSCN;
                                    part_life[nslot] = SPARK_LIFE;
                                }
                            }
                        }
                    }
                    part_life[id] = life;
                }
            }
        }
    }
}
