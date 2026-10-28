//! Pressure and life sensors, and source ignition.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{
    WORD_PROPERTIES, elem_word, hash_u32, iabs, insulated, pmap_at, pmap_slot,
};
use super::{SENSOR_MAX_RANGE, SENSOR_SCAN, SPARK_LIFE};

/// `PSNS.cpp`: the `FILT` pressure broadcast (`tmp == 1`).
///
/// `PSNS` has no trigger state of its own, so [`source_ignition_kernel`]
/// re-evaluates the pressure comparison; this pass only serialises the cell
/// pressure into an adjacent `FILT` chain as a wavelength. It is split from
/// [`light_sensor_kernel`] so each launch stays inside the eight storage
/// buffer budget.
#[cube(launch_unchecked)]
pub(crate) fn sensor_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    air_pv: &[f32],
    part_ctype: &mut [u32],
    part_tmp: &[i32],
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_PSNS && part_tmp[id] == 1i32 {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let wl = 0x1000_0000u32 + ((air_pv[cidx] - MIN_PRESSURE) + 0.5f32) as i32 as u32;
                for n in 0u32..9u32 {
                    let dx = (n % 3u32) as i32 - 1i32;
                    let dy = (n / 3u32) as i32 - 1i32;
                    if dx != 0i32 || dy != 0i32 {
                        let mut cx = px + dx;
                        let mut cy = py + dy;
                        let mut walking = true;
                        for _ in 0u32..64u32 {
                            if walking {
                                let cv = pmap_at(pmap, xres, yres, cx, cy);
                                if cv == 0u32 || (cv & 0x1FFu32) != PT_FILT {
                                    walking = false;
                                } else {
                                    part_ctype[pmap_slot(cv)] = wl;
                                    cx += dx;
                                    cy += dy;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

const PT_LSNS_U32: u32 = 184;
const _: () = assert!(PT_LSNS == 184);
const PT_FILT_U32: u32 = 134;
const _: () = assert!(PT_FILT == 134);

/// `LSNS.cpp`: scan the `tmp2` radius and latch `life = 1` on a life sample
/// above (or, in `tmp == 2` mode, below) `temp - 273.15`.
///
/// Modes 1 and 3 are `FILT` serialisation/deserialisation and are handled by
#[cube(launch_unchecked)]
pub(crate) fn light_sensor_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_life: &mut [i32],
    part_tmp: &[i32],
    part_tmp2: &mut [i32],
    part_temp: &[f32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_LSNS {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                // `LSNS.cpp` declares `DefaultProperties.tmp2 = 2`, but the
                // spawn path seeds `tmp2` with `0`; a zero radius could never
                // detect anything, so `0` means "the TPT default".
                let mut rd = part_tmp2[id];
                if rd > SENSOR_MAX_RANGE as i32 {
                    rd = SENSOR_MAX_RANGE as i32;
                    part_tmp2[id] = rd;
                }
                if rd <= 0i32 {
                    rd = 2i32;
                    part_tmp2[id] = rd;
                }
                let threshold = part_temp[id] - 273.15f32;
                let mode = part_tmp[id];
                // Recompute the trigger every frame, so a stale `life` cannot
                // keep sparking after the condition disappeared.
                let mut triggered = 0i32;
                if mode != 1i32 && mode != 3i32 {
                    for n in 0u32..(SENSOR_SCAN * SENSOR_SCAN) {
                        let rxi = (n % SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                        let ryi = (n / SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                        if triggered == 0i32
                            && rxi >= -rd
                            && rxi <= rd
                            && ryi >= -rd
                            && ryi <= rd
                            && (rxi != 0i32 || ryi != 0i32)
                        {
                            let nx = px + rxi;
                            let ny = py + ryi;
                            let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                            if nv == 0u32
                                && nx >= 0i32
                                && ny >= 0i32
                                && (nx as u32) < xres
                                && (ny as u32) < yres
                            {
                                nv = photons[((ny as u32) * xres + (nx as u32)) as usize];
                            }
                            if nv != 0u32 {
                                let nslot = pmap_slot(nv);
                                let nty = part_ty[nslot];
                                // `LSNS.cpp` never reacts to metal; another sensor is
                                // skipped as well so the parallel scan stays
                                // deterministic (TPT scans sequentially).
                                if nty != PT_METL && nty != PT_LSNS {
                                    let nlife = part_life[nslot] as f32;
                                    let hit = if mode == 2i32 {
                                        nlife <= threshold
                                    } else {
                                        nlife > threshold
                                    };
                                    if hit && !insulated(pmap, xres, yres, nx, ny, px, py) {
                                        triggered = 1i32;
                                    }
                                }
                            }
                        }
                    }
                }
                part_life[id] = triggered;
            }
        }
    }
}

/// `LSNS.cpp`: `FILT` serialisation (`tmp == 1`) and deserialisation (`tmp == 3`).
///
/// * Mode 1: scans the `tmp2` radius for the first particle that is neither
///   `LSNS` nor `FILT` with `life >= 0`, and writes `0x1000_0000 + life` into
///   adjacent `FILT` chains.
/// * Mode 3: scans the `tmp2` radius for a `FILT`, reads its `ctype` as a
///   wavelength, and writes `ctype - 0x1000_0000` into adjacent non-`FILT`
///   neighbours.
#[cube(launch_unchecked)]
pub(crate) fn lsns_filter_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &[u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &[i32],
    part_tmp2: &[i32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_LSNS_U32 {
                let mode = part_tmp[id];
                if mode == 1i32 || mode == 3i32 {
                    let px = (idx as u32 % xres) as i32;
                    let py = (idx as u32 / xres) as i32;

                    let mut rd = part_tmp2[id];
                    if rd > SENSOR_MAX_RANGE as i32 {
                        rd = SENSOR_MAX_RANGE as i32;
                    }
                    if rd <= 0i32 {
                        rd = 2i32;
                    }

                    if mode == 1i32 {
                        let mut found = false;
                        let mut captured_life = 0i32;

                        for n in 0u32..(SENSOR_SCAN * SENSOR_SCAN) {
                            if !found {
                                let rxi = (n % SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                                let ryi = (n / SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                                if rxi >= -rd
                                    && rxi <= rd
                                    && ryi >= -rd
                                    && ryi <= rd
                                    && (rxi != 0i32 || ryi != 0i32)
                                {
                                    let nx = px + rxi;
                                    let ny = py + ryi;
                                    let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                                    if nv == 0u32
                                        && nx >= 0i32
                                        && ny >= 0i32
                                        && (nx as u32) < xres
                                        && (ny as u32) < yres
                                    {
                                        nv = photons[((ny as u32) * xres + (nx as u32)) as usize];
                                    }
                                    if nv != 0u32 {
                                        let nslot = pmap_slot(nv);
                                        let nty = part_ty[nslot];
                                        if nty != PT_LSNS_U32 && nty != PT_FILT_U32 {
                                            let l = part_life[nslot];
                                            if l >= 0i32 {
                                                captured_life = l;
                                                found = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if found {
                            let wl = 0x1000_0000u32 + (captured_life as u32);
                            for n in 0u32..9u32 {
                                let dx = (n % 3u32) as i32 - 1i32;
                                let dy = (n / 3u32) as i32 - 1i32;
                                if dx != 0i32 || dy != 0i32 {
                                    let mut cx = px + dx;
                                    let mut cy = py + dy;
                                    let mut walking = true;
                                    for _ in 0u32..64u32 {
                                        if walking {
                                            let cv = pmap_at(pmap, xres, yres, cx, cy);
                                            if cv == 0u32 || (cv & 0x1FFu32) != PT_FILT_U32 {
                                                walking = false;
                                            } else {
                                                part_ctype[pmap_slot(cv)] = wl;
                                                cx += dx;
                                                cy += dy;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Mode 3: deserialisation.
                        let mut found = false;
                        let mut deser_life = 0u32;

                        for n in 0u32..(SENSOR_SCAN * SENSOR_SCAN) {
                            if !found {
                                let rxi = (n % SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                                let ryi = (n / SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                                if rxi >= -rd
                                    && rxi <= rd
                                    && ryi >= -rd
                                    && ryi <= rd
                                    && (rxi != 0i32 || ryi != 0i32)
                                {
                                    let nx = px + rxi;
                                    let ny = py + ryi;
                                    let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                                    if nv == 0u32
                                        && nx >= 0i32
                                        && ny >= 0i32
                                        && (nx as u32) < xres
                                        && (ny as u32) < yres
                                    {
                                        nv = photons[((ny as u32) * xres + (nx as u32)) as usize];
                                    }
                                    if nv != 0u32 {
                                        let nslot = pmap_slot(nv);
                                        let nty = part_ty[nslot];
                                        if nty == PT_FILT_U32 {
                                            deser_life = part_ctype[nslot];
                                            found = true;
                                        }
                                    }
                                }
                            }
                        }

                        if found {
                            let unpacked_life = (deser_life as i32) - 0x1000_0000i32;
                            for n in 0u32..9u32 {
                                let dx = (n % 3u32) as i32 - 1i32;
                                let dy = (n / 3u32) as i32 - 1i32;
                                if dx != 0i32 || dy != 0i32 {
                                    let nx = px + dx;
                                    let ny = py + dy;
                                    let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                                    if nv == 0u32
                                        && nx >= 0i32
                                        && ny >= 0i32
                                        && (nx as u32) < xres
                                        && (ny as u32) < yres
                                    {
                                        nv = photons[((ny as u32) * xres + (nx as u32)) as usize];
                                    }
                                    if nv != 0u32 {
                                        let nslot = pmap_slot(nv);
                                        let nty = part_ty[nslot];
                                        if nty != PT_FILT_U32 {
                                            part_life[nslot] = unpacked_life;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#[cfg(test)]
pub(super) fn lsns_filt_serialize_value(life: i32) -> u32 {
    0x1000_0000u32.wrapping_add(life as u32)
}

#[cfg(test)]
pub(super) fn lsns_filt_deserialize_value(ctype: u32) -> i32 {
    (ctype as i32).wrapping_sub(0x1000_0000i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsns_filt_life_serialization_roundtrip() {
        for life in [0, 1, 4, 10, 255, 1000, -5] {
            let ctype = lsns_filt_serialize_value(life);
            let recovered = lsns_filt_deserialize_value(ctype);
            assert_eq!(recovered, life);
        }
    }
}

/// Conductors turn into sparks when a source demands it.
///
/// `BTRY.cpp:48-76` sparks every conductor in the battery's radius-2 diamond
/// (`abs(rx)+abs(ry) < 4`) that is idle and neither insulated nor one of the
/// water family / thermistors / `INWR`. `PSNS.cpp` and `LSNS.cpp` do the same
/// for their own conditions. Expressed receiver-side that is one branch per
/// conductor pixel, so no thread ever writes another pixel's element and no
/// `pmap` entry has to be updated. Also runs the thermistor self cooling of
/// `NTCT.cpp` / `PTCT.cpp` and `SPRK.cpp`'s `case PT_NBLE`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn source_ignition_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    air_pv: &mut [f32],
    elem_words: &[u32],
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

                if ty == PT_SPRK {
                    // `SPRK.cpp` `case PT_NBLE`: the spark detonates the gas.
                    if part_ctype[id] == PT_NBLE && part_life[id] <= 1i32 {
                        let tmp = part_tmp[id];
                        if (tmp & 1i32) == 0i32 {
                            let mut new_life =
                                50i32 + ((hash_u32((id as u32) ^ frame_seed) % 150u32) as i32);
                            if new_life > 199i32 {
                                new_life = 199i32;
                            }
                            let mut new_tmp = tmp;
                            if part_temp[id] > 5273.15f32 {
                                new_tmp |= 4i32;
                            }
                            part_ty[id] = PT_PLSM;
                            part_ctype[id] = PT_NBLE;
                            part_life[id] = new_life;
                            part_tmp[id] = new_tmp;
                            part_temp[id] = 3500.0f32;
                            let cidx =
                                (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                            air_pv[cidx] += 1.0f32;
                        }
                    }
                } else {
                    if ty == PT_NTCT || ty == PT_PTCT {
                        // `NTCT.cpp` / `PTCT.cpp`: both bleed heat above 295 K.
                        if part_temp[id] > 295.0f32 {
                            part_temp[id] -= 2.5f32;
                        }
                    }
                    // Sources never feed the water family, the thermistors or
                    // insulated wire directly (`BTRY.cpp` and friends).
                    let excluded = ty == PT_WATR
                        || ty == PT_SLTW
                        || ty == PT_NTCT
                        || ty == PT_PTCT
                        || ty == PT_INWR;
                    if !excluded
                        && part_life[id] == 0i32
                        && (elem_word(elem_words, ty, WORD_PROPERTIES) & PROP_CONDUCTS) != 0u32
                    {
                        let mut fired = false;
                        for n in 0u32..25u32 {
                            if !fired {
                                let dx = (n % 5u32) as i32 - 2i32;
                                let dy = (n / 5u32) as i32 - 2i32;
                                if dx != 0i32 || dy != 0i32 {
                                    let nx = px + dx;
                                    let ny = py + dy;
                                    let nv = pmap_at(pmap, xres, yres, nx, ny);
                                    if nv != 0u32 && !insulated(pmap, xres, yres, nx, ny, px, py) {
                                        let nslot = pmap_slot(nv);
                                        let nty = part_ty[nslot];
                                        // `BTRY.cpp:56` / `:65-69`: a
                                        // battery in the radius-2 diamond
                                        // (`abs(rx)+abs(ry) < 4`) sparks the
                                        // conductor, recording its own type in
                                        // `ctype` and a fresh four-frame life.
                                        if nty == PT_BTRY && iabs(dx) + iabs(dy) < 4i32 {
                                            fired = true;
                                        } else if nty == PT_PSNS {
                                            let mode = part_tmp[nslot];
                                            let limit = part_temp[nslot] - 273.15f32;
                                            let ncidx = (((ny as u32) / CELL) * xcells
                                                + ((nx as u32) / CELL))
                                                as usize;
                                            let npv = air_pv[ncidx];
                                            if (mode == 0i32 && npv > limit)
                                                || (mode == 2i32 && npv < limit)
                                            {
                                                fired = true;
                                            }
                                        } else if nty == PT_LSNS && part_life[nslot] == 1i32 {
                                            fired = true;
                                        }
                                    }
                                }
                            }
                        }
                        if fired {
                            part_ty[id] = PT_SPRK;
                            part_ctype[id] = ty;
                            part_life[id] = SPARK_LIFE;
                        }
                    }
                }
            }
        }
    }
}

/// `parts_avg(i, j, PT_INSL)` — the type average the temperature and velocity
/// sensors compare against `INSL` / `RSSS` (`Simulation.cpp:2087-2101`).
#[cube]
fn parts_avg_type(a: u32, b: u32) -> u32 {
    let mut result = 0u32;
    let mut count = 0u32;
    if a != PT_INSL {
        result += a;
        count += 1u32;
    }
    if b != PT_INSL {
        result += b;
        count += 1u32;
    }
    if count == 0u32 {
        PT_INSL
    } else if count == 1u32 {
        result
    } else {
        result / 2u32
    }
}

/// `TSNS.cpp:52-80` — a latched sensor sparks every conducting neighbour.
///
/// Shared by [`temp_sensor_kernel`] and [`velocity_sensor_kernel`]: both
/// elements carry the identical `if (parts[i].life)` block.
#[cube]
fn spark_neighbours(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    elem_words: &[u32],
    id: usize,
    px: i32,
    py: i32,
    xres: u32,
    yres: u32,
) {
    let self_ty = part_ty[id];
    for n in 0u32..9u32 {
        let dx = (n % 3u32) as i32 - 1i32;
        let dy = (n / 3u32) as i32 - 1i32;
        if dx != 0i32 || dy != 0i32 {
            let nx = px + dx;
            let ny = py + dy;
            let mut nv = pmap_at(pmap, xres, yres, nx, ny);
            if nv == 0u32 && nx >= 0i32 && ny >= 0i32 && (nx as u32) < xres && (ny as u32) < yres {
                nv = photons[((ny as u32) * xres + (nx as u32)) as usize];
            }
            if nv != 0u32 {
                let nslot = pmap_slot(nv);
                let nty = part_ty[nslot];
                let pavg = parts_avg_type(self_ty, nty);
                let props = elem_word(elem_words, nty, WORD_PROPERTIES);
                if pavg != PT_INSL
                    && pavg != PT_RSSS
                    && (props & PROP_CONDUCTS) != 0u32
                    && nty != PT_WATR
                    && nty != PT_SLTW
                    && nty != PT_NTCT
                    && nty != PT_PTCT
                    && nty != PT_INWR
                    && part_life[nslot] == 0i32
                {
                    part_life[nslot] = SPARK_LIFE;
                    part_ctype[nslot] = nty;
                    part_ty[nslot] = PT_SPRK;
                }
            }
        }
    }
}

/// `TSNS.cpp`: temperature sensor.
///
/// `tmp` is the mode: `0` (default) arms the latch while a non-`METL`,
/// non-`TSNS` particle inside the `tmp2` radius is hotter than the sensor,
/// `2` while one is colder, and `1` serialises the nearest temperature into
/// adjacent `FILT` chains. The latch is consumed by
/// [`spark_neighbours`] on the next frame, exactly as `TSNS.cpp` orders it.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn temp_sensor_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &[f32],
    part_tmp: &[i32],
    part_tmp2: &mut [i32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_TSNS {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let mut rd = part_tmp2[id];
                if rd > SENSOR_MAX_RANGE as i32 {
                    rd = SENSOR_MAX_RANGE as i32;
                    part_tmp2[id] = rd;
                }
                if rd <= 0i32 {
                    rd = 2i32;
                    part_tmp2[id] = rd;
                }
                if part_life[id] != 0i32 {
                    part_life[id] = 0i32;
                    spark_neighbours(
                        pmap, photons, part_ty, part_life, part_ctype, elem_words, id, px, py,
                        xres, yres,
                    );
                }

                let mode = part_tmp[id];
                let self_temp = part_temp[id];
                let mut armed = 0i32;
                let mut set_filt = false;
                let mut photon_wl = 0i32;
                for n in 0u32..(SENSOR_SCAN * SENSOR_SCAN) {
                    let rxi = (n % SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                    let ryi = (n / SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                    if rxi >= -rd
                        && rxi <= rd
                        && ryi >= -rd
                        && ryi <= rd
                        && (rxi != 0i32 || ryi != 0i32)
                    {
                        let nx = px + rxi;
                        let ny = py + ryi;
                        let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                        if nv == 0u32
                            && nx >= 0i32
                            && ny >= 0i32
                            && (nx as u32) < xres
                            && (ny as u32) < yres
                        {
                            nv = photons[((ny as u32) * xres + (nx as u32)) as usize];
                        }
                        if nv != 0u32 {
                            let nslot = pmap_slot(nv);
                            let nty = part_ty[nslot];
                            let ntemp = part_temp[nslot];
                            if mode == 0i32 && nty != PT_TSNS && nty != PT_METL && ntemp > self_temp
                            {
                                armed = 1i32;
                            }
                            if mode == 2i32 && nty != PT_TSNS && nty != PT_METL && ntemp < self_temp
                            {
                                armed = 1i32;
                            }
                            if mode == 1i32 && nty != PT_TSNS && nty != PT_FILT {
                                set_filt = true;
                                photon_wl = ntemp as i32;
                            }
                        }
                    }
                }
                if armed != 0i32 {
                    part_life[id] = 1i32;
                }

                if set_filt {
                    let wl = 0x1000_0000u32 + photon_wl as u32;
                    for n in 0u32..9u32 {
                        let dx = (n % 3u32) as i32 - 1i32;
                        let dy = (n / 3u32) as i32 - 1i32;
                        if dx != 0i32 || dy != 0i32 {
                            let mut cx = px + dx;
                            let mut cy = py + dy;
                            let mut walking = true;
                            for _ in 0u32..64u32 {
                                if walking {
                                    let cv = pmap_at(pmap, xres, yres, cx, cy);
                                    if cv == 0u32 || (cv & 0x1FFu32) != PT_FILT {
                                        walking = false;
                                    } else {
                                        part_ctype[pmap_slot(cv)] = wl;
                                        cx += dx;
                                        cy += dy;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `VSNS.cpp`: velocity sensor.
///
/// The latch mirrors [`temp_sensor_kernel`] but compares
/// `sqrt(vx^2 + vy^2)` against `temp - 273.15` (below it for `tmp == 2`, above
/// it otherwise). `tmp == 1` serialises the fastest non-solid neighbour's
/// speed into adjacent `FILT` chains and `tmp == 3` deserialises a `FILT`
/// wavelength back onto one non-solid neighbour's velocity vector.
///
/// The deserialisation scan keeps TPT's `rx`-outer/`ry`-inner order: its
/// `break` exits only the inner row, so one neighbour per column can be
/// written (up to three per frame). The port reproduces that with a per-column
/// `written` flag instead of a single `done` guard.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn velocity_sensor_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &[f32],
    part_tmp: &[i32],
    part_tmp2: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_VSNS {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let mut rd = part_tmp2[id];
                if rd > SENSOR_MAX_RANGE as i32 {
                    rd = SENSOR_MAX_RANGE as i32;
                    part_tmp2[id] = rd;
                }
                if rd <= 0i32 {
                    rd = 2i32;
                    part_tmp2[id] = rd;
                }
                if part_life[id] != 0i32 {
                    part_life[id] = 0i32;
                    spark_neighbours(
                        pmap, photons, part_ty, part_life, part_ctype, elem_words, id, px, py,
                        xres, yres,
                    );
                }

                let mode = part_tmp[id];
                let threshold = part_temp[id] - 273.15f32;
                let mut armed = 0i32;
                let mut serialise = false;
                let mut deserialise = false;
                let mut vs = 0.0f32;
                for n in 0u32..(SENSOR_SCAN * SENSOR_SCAN) {
                    let rxi = (n % SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                    let ryi = (n / SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                    if rxi >= -rd
                        && rxi <= rd
                        && ryi >= -rd
                        && ryi <= rd
                        && (rxi != 0i32 || ryi != 0i32)
                    {
                        let nx = px + rxi;
                        let ny = py + ryi;
                        let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                        if nv == 0u32
                            && nx >= 0i32
                            && ny >= 0i32
                            && (nx as u32) < xres
                            && (ny as u32) < yres
                        {
                            nv = photons[((ny as u32) * xres + (nx as u32)) as usize];
                        }
                        if nv != 0u32 {
                            let nslot = pmap_slot(nv);
                            let nty = part_ty[nslot];
                            let solid =
                                (elem_word(elem_words, nty, WORD_PROPERTIES) & TYPE_SOLID) != 0u32;
                            let vx = part_vx[nslot];
                            let vy = part_vy[nslot];
                            let vm = (vx * vx + vy * vy).sqrt();
                            if mode == 1i32 {
                                if nty != PT_VSNS && nty != PT_FILT && !solid {
                                    serialise = true;
                                    vs = vm;
                                }
                            } else if mode == 3i32 {
                                if nty == PT_FILT {
                                    let vel = (part_ctype[nslot] as i32) - 0x1000_0000i32;
                                    if vel >= 0i32 && (vel as f32) < 1.0e4f32 {
                                        deserialise = true;
                                        vs = vel as f32;
                                    }
                                }
                            } else if mode == 2i32 {
                                if !solid && vm <= threshold {
                                    armed = 1i32;
                                }
                            } else if !solid && vm > threshold {
                                armed = 1i32;
                            }
                        }
                    }
                }
                if armed != 0i32 {
                    part_life[id] = 1i32;
                }

                let wl = 0x1000_0000u32 + (vs + 0.5f32) as i32 as u32;
                // `VSNS.cpp:127-190`: `rx` is the outer loop and `ry` the
                // inner one; the deserialisation `break` leaves only the inner
                // row, so each column can write one neighbour.
                let mut column_written = false;
                for n in 0u32..9u32 {
                    let dx = (n / 3u32) as i32 - 1i32;
                    let dy = (n % 3u32) as i32 - 1i32;
                    if n % 3u32 == 0u32 {
                        column_written = false;
                    }
                    if (dx != 0i32 || dy != 0i32) && !column_written {
                        let nx = px + dx;
                        let ny = py + dy;
                        let mut nv = pmap_at(pmap, xres, yres, nx, ny);
                        if nv == 0u32
                            && nx >= 0i32
                            && ny >= 0i32
                            && (nx as u32) < xres
                            && (ny as u32) < yres
                        {
                            nv = photons[((ny as u32) * xres + (nx as u32)) as usize];
                        }
                        if nv != 0u32 {
                            if serialise {
                                let mut cx = nx;
                                let mut cy = ny;
                                let mut walking = true;
                                for _ in 0u32..64u32 {
                                    if walking {
                                        let cv = pmap_at(pmap, xres, yres, cx, cy);
                                        if cv == 0u32 || (cv & 0x1FFu32) != PT_FILT {
                                            walking = false;
                                        } else {
                                            part_ctype[pmap_slot(cv)] = wl;
                                            cx += dx;
                                            cy += dy;
                                        }
                                    }
                                }
                            }
                            if deserialise && !column_written {
                                let nslot = pmap_slot(nv);
                                let nty = part_ty[nslot];
                                let solid = (elem_word(elem_words, nty, WORD_PROPERTIES)
                                    & TYPE_SOLID)
                                    != 0u32;
                                if nty != PT_FILT && !solid {
                                    let vx = part_vx[nslot];
                                    let vy = part_vy[nslot];
                                    let vm = (vx * vx + vy * vy).sqrt();
                                    if vm > 0.0f32 {
                                        part_vx[nslot] = vx * vs / vm;
                                        part_vy[nslot] = vy * vs / vm;
                                    }
                                    column_written = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
