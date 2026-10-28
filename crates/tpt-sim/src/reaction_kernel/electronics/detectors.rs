//! `DTEC` and `LDTC` particle and light detectors.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::super::shared::{WORD_PROPERTIES, elem_word, insulated, pmap_at, pmap_slot};
use super::{SENSOR_MAX_RANGE, SENSOR_SCAN, SPARK_LIFE};

/// `DTEC.cpp` and `LDTC.cpp`.
///
/// Both scan for particles and then spark an adjacent conductor, so they keep
/// their whole behaviour here: the sensor state and the ignition it causes are
/// computed by the same thread.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn detector_kernel(
    pmap: &[u32],
    photons: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_tmp: &[i32],
    part_tmp2: &[i32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ty = part_ty[id];
            if ty == PT_DTEC {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                // `DTEC.cpp` declares `DefaultProperties.tmp2 = 2`, but the
                // spawn path seeds `tmp2` with `0`; a zero radius could never
                // detect anything, so `0` means "the TPT default".
                let mut rd = part_tmp2[id];
                if rd > SENSOR_MAX_RANGE as i32 {
                    rd = SENSOR_MAX_RANGE as i32;
                }
                if rd <= 0i32 {
                    rd = 2i32;
                }
                let ctype = part_ctype[id];
                let dtec_tmp = part_tmp[id];
                let mut triggered = 0i32;
                let mut set_filt = false;
                // Runtime-seeded zero (see the kernel rules in `AGENTS.md`): a
                // host literal is mis-expanded when the value is assigned
                // inside the scan loop below.
                #[allow(clippy::erasing_op)]
                let mut photon_wl = part_ctype[id] * 0u32;
                for n in 0u32..(SENSOR_SCAN * SENSOR_SCAN) {
                    let rxi = (n % SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                    let ryi = (n / SENSOR_SCAN) as i32 - SENSOR_MAX_RANGE as i32;
                    if (triggered == 0i32 || !set_filt)
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
                            let nty = nv & 0x1FFu32;
                            // `DTEC.cpp` line 98: compare against ctype (and tmp for PT_LIFE).
                            if ctype != PT_NONE
                                && nty == ctype
                                && (ctype != PT_LIFE
                                    || dtec_tmp == 0i32
                                    || part_ctype[nslot] == (dtec_tmp as u32))
                            {
                                triggered = 1i32;
                            }
                            // `DTEC.cpp` lines 100-107: photon/ray wavelength detection.
                            if !set_filt
                                && (nty == PT_PHOT
                                    || (nty == PT_BRAY && part_tmp[nslot] != 2i32)
                                    || nty == PT_BIZR
                                    || nty == PT_BIZRG
                                    || nty == PT_BIZRS)
                            {
                                set_filt = true;
                                photon_wl = part_ctype[nslot];
                            }
                        }
                    }
                }
                part_life[id] = triggered;
                if triggered == 1i32 {
                    ignite_adjacent_conductors(
                        pmap, part_ty, part_life, part_ctype, elem_words, xres, yres, px, py,
                    );
                }
                // `DTEC.cpp` lines 108-134: copy detected wavelength into adjacent FILT chains.
                if set_filt {
                    for m in 0u32..9u32 {
                        let fdx = (m % 3u32) as i32 - 1i32;
                        let fdy = (m / 3u32) as i32 - 1i32;
                        if fdx != 0i32 || fdy != 0i32 {
                            let mut cx = px + fdx;
                            let mut cy = py + fdy;
                            let mut walking = true;
                            for _ in 0u32..64u32 {
                                if walking {
                                    let cv = pmap_at(pmap, xres, yres, cx, cy);
                                    if cv == 0u32 || (cv & 0x1FFu32) != PT_FILT {
                                        walking = false;
                                    } else {
                                        part_ctype[pmap_slot(cv)] = photon_wl;
                                        cx += fdx;
                                        cy += fdy;
                                    }
                                }
                            }
                        }
                    }
                }
            } else if ty == PT_LDTC {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let ctype = part_ctype[id];
                let detect_length = part_tmp[id];
                let flags = part_tmp2[id];
                let ignore_energy = (flags & 2i32) != 0i32;
                let invert = (flags & 1i32) != 0i32;
                let skip = part_life[id];
                for n in 0u32..9u32 {
                    let dx = (n % 3u32) as i32 - 1i32;
                    let dy = (n / 3u32) as i32 - 1i32;
                    if dx != 0i32 || dy != 0i32 {
                        let cv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if cv != 0u32 {
                            let cslot = pmap_slot(cv);
                            let ct = cv & 0x1FFu32;
                            let conductor = (elem_word(elem_words, ct, WORD_PROPERTIES)
                                & PROP_CONDUCTS)
                                != 0u32
                                && ct != PT_WATR
                                && ct != PT_SLTW
                                && ct != PT_NTCT
                                && ct != PT_PTCT
                                && ct != PT_INWR
                                && part_life[cslot] == 0i32;
                            if conductor {
                                let step_x = -dx;
                                let step_y = -dy;
                                let mut cx = px + step_x * (skip + 1i32);
                                let mut cy = py + step_y * (skip + 1i32);
                                let mut fired = false;
                                for _ in 0u32..1024u32 {
                                    if !fired {
                                        let inside = cx >= 0i32
                                            && cy >= 0i32
                                            && (cx as u32) < xres
                                            && (cy as u32) < yres;
                                        let past_limit = detect_length != 0i32
                                            && (step_x * (cx - px) > skip + detect_length
                                                || step_y * (cy - py) > skip + detect_length);
                                        if !inside || past_limit {
                                            fired = true;
                                        } else {
                                            let mut rv =
                                                pmap[((cy as u32) * xres + (cx as u32)) as usize];
                                            if rv == 0u32 && !ignore_energy {
                                                rv = photons
                                                    [((cy as u32) * xres + (cx as u32)) as usize];
                                            }
                                            let mut stop = false;
                                            if rv != 0u32 {
                                                let rt = rv & 0x1FFu32;
                                                let matches = rt == ctype;
                                                let pass = ctype == PT_NONE || (invert != matches);
                                                if pass {
                                                    // The beam found its target:
                                                    // spark the conductor on the
                                                    // near side of the detector.
                                                    part_ty[cslot] = PT_SPRK;
                                                    part_life[cslot] = SPARK_LIFE;
                                                    part_ctype[cslot] = ct;
                                                    stop = true;
                                                } else {
                                                    // `LDTC.cpp` keeps scanning
                                                    // past a non-match only when
                                                    // the flag is set.
                                                    stop = (flags & 8i32) == 0i32;
                                                }
                                            }
                                            if stop {
                                                fired = true;
                                            }
                                            cx += step_x;
                                            cy += step_y;
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

/// Spark every adjacent conductor that is idle and not on the source exclusion
/// list (`BTRY.cpp`, `PSNS.cpp`, `LSNS.cpp`, `DTEC.cpp` share it).
#[cube]
#[allow(clippy::too_many_arguments)] // cube kernels cannot take a struct
fn ignite_adjacent_conductors(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    px: i32,
    py: i32,
) {
    for n in 0u32..25u32 {
        let dx = (n % 5u32) as i32 - 2i32;
        let dy = (n / 5u32) as i32 - 2i32;
        if dx != 0i32 || dy != 0i32 {
            let nx = px + dx;
            let ny = py + dy;
            let nv = pmap_at(pmap, xres, yres, nx, ny);
            if nv != 0u32 {
                let nslot = pmap_slot(nv);
                let nty = part_ty[nslot];
                if nty != PT_NONE
                    && nty != PT_WATR
                    && nty != PT_SLTW
                    && nty != PT_NTCT
                    && nty != PT_PTCT
                    && nty != PT_INWR
                    && (elem_word(elem_words, nty, WORD_PROPERTIES) & PROP_CONDUCTS) != 0u32
                    && part_life[nslot] == 0i32
                    && !insulated(pmap, xres, yres, nx, ny, px, py)
                {
                    part_ty[nslot] = PT_SPRK;
                    part_ctype[nslot] = nty;
                    part_life[nslot] = SPARK_LIFE;
                }
            }
        }
    }
}
