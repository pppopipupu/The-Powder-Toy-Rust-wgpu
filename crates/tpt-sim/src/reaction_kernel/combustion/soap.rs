//! Soap bubbles and gel kernels.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::*;

/// `SOAP.cpp:8-16` `Element_SOAP_neighourLoop`: under `EDGE_LOOP` the vector
/// between two linked bubbles is folded into
/// `-(RES/2 - CELL) .. RES/2 - CELL`, so a bubble chain that spans the seam
/// keeps pulling its two ends together instead of stretching across the whole
/// canvas.
///
/// ```text
/// if (sim->edgeMode == EDGE_LOOP)
/// {
///     if (dx >= (XRES / 2 - CELL)) dx -= XRES - 2 * CELL;
///     if (dy >= (YRES / 2 - CELL)) dy -= YRES - 2 * CELL;
///     if (dx < -(XRES / 2 - CELL)) dx += XRES - 2 * CELL;
///     if (dy < -(YRES / 2 - CELL)) dy += YRES - 2 * CELL;
/// }
/// ```
///
/// `XRES / 2 - CELL` is integer division, so `res / 2 - 4` is the exact bound
/// the reference compares against.
#[cube]
fn soap_neighbour_loop(dx: f32, dy: f32, xres: u32, yres: u32, edge_mode: u32) -> (f32, f32) {
    let mut out_x = dx;
    let mut out_y = dy;
    if edge_mode == EDGE_LOOP {
        let half_x = (xres / 2u32) as i32 - CELL as i32;
        let half_y = (yres / 2u32) as i32 - CELL as i32;
        let span_x = (xres - 2u32 * CELL) as f32;
        let span_y = (yres - 2u32 * CELL) as f32;
        if out_x >= half_x as f32 {
            out_x -= span_x;
        }
        if out_y >= half_y as f32 {
            out_y -= span_y;
        }
        if out_x < -(half_x as f32) {
            out_x += span_x;
        }
        if out_y < -(half_y as f32) {
            out_y += span_y;
        }
    }
    (out_x, out_y)
}

/// `SOAP` bubble flag, mate links, attach and detach (`SOAP.cpp:104-230`).
///
/// Each bubble resolves its own link, and a broken chain converges over the
/// following frame; the pop-on-contact half, walls included, is
/// [`soap_detach_kernel`].
///
/// `SOAP.cpp:33`: `Gravity = 0.1f`, the scale of the local field the `OIL` foam
/// coupling (`SOAP.cpp:274-292`) mixes against.
const SOAP_GRAVITY: f32 = 0.1;

/// `EDGE_LOOP` (`SimulationSettings.h:7`) mirrored for the kernel body.
const EDGE_LOOP: u32 = 2;
const _: () = assert!(EDGE_LOOP == 2);

#[cube(launch_unchecked)]
pub(crate) fn soap_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_ctype: &mut [u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    air_pv: &mut [f32],
    grav_fx: &[f32],
    grav_fy: &[f32],
    gravity_mode: u32,
    custom_x: f32,
    custom_y: f32,
    newton: u32,
    xres: u32,
    yres: u32,
    xcells: u32,
    n_slots: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_SOAP {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let mut ctype = part_ctype[id];
                let mut tmp = part_tmp[id];
                let mut tmp2 = part_tmp2[id];
                let mut life = part_life[id];

                if (ctype & 1u32) != 0u32 {
                    let mut ok = true;
                    if tmp < 0i32 || (tmp as u32) >= n_slots {
                        ok = false;
                    }
                    if tmp2 < 0i32 || (tmp2 as u32) >= n_slots {
                        ok = false;
                    }
                    if !ok {
                        ctype = 0u32;
                        tmp = 0i32;
                        tmp2 = 0i32;
                    } else {
                        // `SOAP.cpp:110-150`: pop when the bubble froze out or
                        // its life ran out. TPT walks the whole chain; this
                        // port only clears the two links of this pixel.
                        if part_temp[id] > SOAP_FREEZING && life <= 0i32 {
                            let mut detach = (ctype & 6u32) != 6u32;
                            if !detach {
                                let p = tmp as usize;
                                if part_ty[p] == PT_SOAP && part_tmp[p] == id as i32 {
                                    detach = true;
                                }
                            }
                            if detach {
                                if (ctype & 2u32) != 0u32 {
                                    let p = tmp as usize;
                                    if part_ty[p] == PT_SOAP && (part_ctype[p] & 4u32) != 0u32 {
                                        part_ctype[p] ^= 4u32;
                                    }
                                }
                                if (ctype & 4u32) != 0u32 {
                                    let p = tmp2 as usize;
                                    if part_ty[p] == PT_SOAP && (part_ctype[p] & 2u32) != 0u32 {
                                        part_ctype[p] ^= 2u32;
                                    }
                                }
                                ctype = 0u32;
                                tmp = 0i32;
                                tmp2 = 0i32;
                            }
                        }
                        if (ctype & 2u32) == 0u32 {
                            // `SOAP.cpp:90-108`, `SOAP.cpp:168-184`: attach to adjacent bubble
                            let mut attached = false;
                            for n in 0u32..25u32 {
                                let dx = (n % 5u32) as i32 - 2i32;
                                let dy = (n / 5u32) as i32 - 2i32;
                                if (dx != 0i32 || dy != 0i32) && !attached {
                                    let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                                    if nv != 0u32 {
                                        let nid = pmap_slot(nv);
                                        let nctype = part_ctype[nid];
                                        if part_ty[nid] == PT_SOAP && (nctype & 1u32) != 0u32 {
                                            if (nctype & 4u32) == 0u32 {
                                                ctype |= 2u32;
                                                tmp = nid as i32;
                                                part_ctype[nid] = nctype | 4u32;
                                                part_tmp2[nid] = id as i32;
                                                attached = true;
                                            } else if (nctype & 2u32) == 0u32 {
                                                ctype |= 4u32;
                                                tmp2 = nid as i32;
                                                part_ctype[nid] = nctype | 2u32;
                                                part_tmp[nid] = id as i32;
                                                attached = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // `SOAP.cpp:268-272`: a pressure swing turns unfoamed soap into a bubble with life = 10.
                    let pv = air_pv[cidx];
                    if pv > 0.5f32 || pv < -0.5f32 {
                        ctype = 1u32;
                        life = 10i32;
                    }
                    // `SOAP.cpp:274-292`: soap next to oil drags both velocities
                    // towards the average of the soap's gravity-relative speed
                    // and the oil's speed, which is what makes foam wobble.
                    let mut gx = 0.0f32;
                    let mut gy = 0.0f32;
                    if gravity_mode == 0u32 {
                        gy = SOAP_GRAVITY;
                    } else if gravity_mode == 2u32 {
                        let dx = (px as f32) - (xres as f32 * 0.5f32).floor();
                        let dy = (py as f32) - (yres as f32 * 0.5f32).floor();
                        if dx != 0.0f32 || dy != 0.0f32 {
                            let d = 0.01f32 - (dx * dx + dy * dy).sqrt();
                            gx = SOAP_GRAVITY * (dx / d);
                            gy = SOAP_GRAVITY * (dy / d);
                        }
                    } else if gravity_mode == 3u32 {
                        gx = SOAP_GRAVITY * custom_x;
                        gy = SOAP_GRAVITY * custom_y;
                    }
                    if newton != 0u32 {
                        gx += grav_fx[cidx];
                        gy += grav_fy[cidx];
                    }
                    for n in 0u32..25u32 {
                        let dx = (n % 5u32) as i32 - 2i32;
                        let dy = (n / 5u32) as i32 - 2i32;
                        if dx != 0i32 || dy != 0i32 {
                            let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                            if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                if part_ty[nid] == PT_OIL {
                                    let ax = ((part_vx[id] - gx) * 0.5f32 + part_vx[nid]) / 2.0f32;
                                    let ay = ((part_vy[id] - gy) * 0.5f32 + part_vy[nid]) / 2.0f32;
                                    part_vx[id] = ax;
                                    part_vx[nid] = ax;
                                    part_vy[id] = ay;
                                    part_vy[nid] = ay;
                                }
                            }
                        }
                    }
                }

                part_ctype[id] = ctype;
                part_tmp[id] = tmp;
                part_tmp2[id] = tmp2;
                part_life[id] = life;
            }
        }
    }
}

/// `SOAP` pop-on-contact (`SOAP.cpp:183-215`).
///
/// Split from [`soap_kernel`] so both stay inside the eight-buffer budget: the
/// gas test needs the element property table. A bubble also pops against a
/// wall, which is why the coarse wall grid is bound here.
#[cube(launch_unchecked)]
pub(crate) fn soap_detach_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp: &mut [i32],
    part_tmp2: &mut [i32],
    part_ctype: &mut [u32],
    elem_words: &[u32],
    wall_bmap: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
    ycells: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            let ctype = part_ctype[id];
            if part_ty[id] == PT_SOAP
                && (ctype & 1u32) != 0u32
                && part_life[id] <= 0i32
                && part_temp[id] > SOAP_FREEZING
            {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let mut detach = false;
                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if (dx != 0i32 || dy != 0i32) && !detach {
                        let nx = px + dx;
                        let ny = py + dy;
                        if nx >= 0i32 && ny >= 0i32 && (nx as u32) < xres && (ny as u32) < yres {
                            let cell = ((ny as u32) / 4u32) * xcells + ((nx as u32) / 4u32);
                            let walled = cell < xcells * ycells && wall_bmap[cell as usize] != 0u32;
                            let nv = pmap_at(pmap, xres, yres, nx, ny);
                            if walled {
                                // `SOAP.cpp:196`: a wall is as solid as a rock.
                                detach = true;
                            } else if nv != 0u32 {
                                let nid = pmap_slot(nv);
                                let nty = part_ty[nid];
                                let props = elem_word(elem_words, nty, WORD_PROPERTIES);
                                if nty != PT_NONE
                                    && nty != PT_SOAP
                                    && nty != PT_GLAS
                                    && (props & TYPE_GAS) == 0u32
                                {
                                    detach = true;
                                }
                            }
                        }
                    }
                }
                if detach {
                    let tmp = part_tmp[id];
                    let tmp2 = part_tmp2[id];
                    if (ctype & 2u32) != 0u32
                        && tmp >= 0i32
                        && part_ty[tmp as usize] == PT_SOAP
                        && (part_ctype[tmp as usize] & 4u32) != 0u32
                    {
                        part_ctype[tmp as usize] ^= 4u32;
                    }
                    if (ctype & 4u32) != 0u32
                        && tmp2 >= 0i32
                        && part_ty[tmp2 as usize] == PT_SOAP
                        && (part_ctype[tmp2 as usize] & 2u32) != 0u32
                    {
                        part_ctype[tmp2 as usize] ^= 2u32;
                    }
                    part_ctype[id] = 0u32;
                    part_tmp[id] = 0i32;
                    part_tmp2[id] = 0i32;
                }
            }
        }
    }
}

/// `SOAP` bubble spring (`SOAP.cpp:196-230`), slot parallel.
///
/// TPT adds the reaction force to both ends and to the third chain pixel. This
/// port applies only the particle's own half of the two forces, which keeps
/// every velocity write slot-local; the bubble still contracts. The
/// `temp > FREEZING` guard of the damping is dropped because `temp` is not
/// bound here.
///
/// Both mate vectors go through [`soap_neighbour_loop`] before the force is
/// computed, so under `EDGE_LOOP` a bubble pair straddling the seam springs
/// across the wrap (`SOAP.cpp:8-16`).
#[cube(launch_unchecked)]
pub(crate) fn soap_spring_kernel(
    part_ty: &[u32],
    part_tmp: &[i32],
    part_ctype: &[u32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    part_x: &[f32],
    part_y: &[f32],
    n_slots: u32,
    xres: u32,
    yres: u32,
    edge_mode: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n_slots as usize && part_ty[i] == PT_SOAP && (part_ctype[i] & 1u32) != 0u32 {
        part_vy[i] = (part_vy[i] - 0.1f32) * 0.5f32;
        part_vx[i] *= 0.5f32;
        let tmp = part_tmp[i];
        if (part_ctype[i] & 2u32) != 0u32 && tmp >= 0i32 && (tmp as u32) < n_slots {
            let p = tmp as usize;
            if part_ty[p] == PT_SOAP {
                let wrapped = soap_neighbour_loop(
                    part_x[i] - part_x[p],
                    part_y[i] - part_y[p],
                    xres,
                    yres,
                    edge_mode,
                );
                let dx = wrapped.0;
                let dy = wrapped.1;
                let d = 9.0f32 / (dx * dx + dy * dy + 9.0f32) - 0.5f32;
                part_vx[i] += dx * d;
                part_vy[i] += dy * d;
                let tmp2 = part_tmp[p];
                if (part_ctype[p] & 3u32) == 3u32 && tmp2 >= 0i32 && (tmp2 as u32) < n_slots {
                    let q = tmp2 as usize;
                    if part_ty[q] == PT_SOAP {
                        let wrapped2 = soap_neighbour_loop(
                            part_x[q] - part_x[p],
                            part_y[q] - part_y[p],
                            xres,
                            yres,
                            edge_mode,
                        );
                        let dx2 = wrapped2.0;
                        let dy2 = wrapped2.1;
                        let d2 = 81.0f32 / (dx2 * dx2 + dy2 * dy2 + 81.0f32) - 0.5f32;
                        part_vx[i] += dx2 * d2 * 0.5f32;
                        part_vy[i] += dy2 * d2 * 0.5f32;
                    }
                }
            }
        }
    }
}

/// `GEL` saturation, desaturation and stickiness (`GEL.cpp:60-140`).
///
/// The neighbour half of the sticky force is not written (cross-slot velocity
/// writes race with the movement pass), so a gel pulls itself onto its
/// neighbours but does not drag them.
#[cube(launch_unchecked)]
pub(crate) fn gel_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_tmp: &mut [i32],
    part_vx: &mut [f32],
    part_vy: &mut [f32],
    elem_words: &[u32],
    xres: u32,
    yres: u32,
    frame_seed: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx < (xres * yres) as usize {
        let v = pmap[idx];
        if v != 0u32 {
            let id = pmap_slot(v);
            if part_ty[id] == PT_GEL {
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 0x1B87_3593u32));
                let mut sat = clampi(part_tmp[id], 0i32, 100i32);
                let denom = (sat * 10i32 + 500i32) as u32;

                for n in 0u32..25u32 {
                    let dx = (n % 5u32) as i32 - 2i32;
                    let dy = (n / 5u32) as i32 - 2i32;
                    if dx != 0i32 || dy != 0i32 {
                        let nv = pmap_at(pmap, xres, yres, px + dx, py + dy);
                        if nv != 0u32 {
                            let nid = pmap_slot(nv);
                            let nty = part_ty[nid];
                            if nty != PT_NONE {
                                let mut gel_neighbour = false;
                                if nty == PT_WATR || nty == PT_DSTW || nty == PT_FRZW {
                                    if sat < 100i32 && chance(seed ^ n, 500u32, denom) {
                                        sat += 1i32;
                                        part_ty[nid] = PT_NONE;
                                    }
                                } else if nty == PT_PSTE {
                                    if sat < 100i32 && chance(seed ^ n, 20u32, denom) {
                                        sat += 1i32;
                                        part_ty[nid] = PT_CLST;
                                    }
                                } else if nty == PT_SLTW {
                                    if sat < 100i32 && chance(seed ^ n, 50u32, denom) {
                                        sat += 1i32;
                                        if chance(seed ^ n ^ 0x1u32, 3u32, 4u32) {
                                            part_ty[nid] = PT_NONE;
                                        } else {
                                            part_ty[nid] = PT_SALT;
                                        }
                                    }
                                } else if nty == PT_CBNW {
                                    if sat < 100i32 && chance(seed ^ n, 100u32, denom) {
                                        sat += 1i32;
                                        part_ty[nid] = PT_CO2;
                                    }
                                } else if nty == PT_SPNG {
                                    if part_life[nid] > 0i32
                                        && sat < 100i32
                                        && part_life[nid] + 1i32 > sat
                                    {
                                        part_life[nid] -= 1i32;
                                        sat += 1i32;
                                    } else if sat > 0i32 && part_life[nid] + 1i32 < sat {
                                        part_life[nid] += 1i32;
                                        sat -= 1i32;
                                    }
                                } else if nty == PT_GEL {
                                    if part_tmp[nid] + 1i32 < sat {
                                        part_tmp[nid] += 1i32;
                                        sat -= 1i32;
                                    }
                                    gel_neighbour = true;
                                } else if nty == PT_BASE && sat > 0i32 && part_life[nid] > 1i32 {
                                    part_life[nid] -= 1i32;
                                    sat -= 1i32;
                                }

                                // `GEL.cpp:113-140`: stickiness. `dx`/`dy` are
                                // the pixel offsets, exactly TPT's
                                // `parts[i].x - parts[r].x`.
                                let dxf = -(dx as f32);
                                let dyf = -(dy as f32);
                                let props = elem_word(elem_words, nty, WORD_PROPERTIES);
                                let falldown = elem_word(elem_words, nty, 3u32);
                                let near = iabs(dx) < 2i32 && iabs(dy) < 2i32;
                                if dxf * dxf + dyf * dyf > 1.5f32
                                    && (gel_neighbour || falldown == 0u32 || near)
                                {
                                    let nd = dxf * dxf + dyf * dyf - 0.5f32;
                                    let mut per = 5.0f32
                                        * (1.0f32 - sat as f32 / 100.0f32)
                                        * (nd / (dxf * dxf + dyf * dyf + nd) - 0.5f32);
                                    if (props & TYPE_LIQUID) != 0u32 {
                                        per *= 0.1f32;
                                    }
                                    part_vx[id] += dxf * per;
                                    part_vy[id] += dyf * per;
                                }
                            }
                        }
                    }
                }
                part_tmp[id] = sat;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// Host mirror of the `OIL` coupling inside [`soap_kernel`]
    /// (`SOAP.cpp:284-289`): the soap's gravity-relative speed is halved, then
    /// averaged with the oil's speed, and the result goes to both particles.
    fn foam_couple(soap_vx: f32, oil_vx: f32, gx: f32) -> f32 {
        ((soap_vx - gx) * 0.5f32 + oil_vx) / 2.0f32
    }

    #[test]
    fn foam_coupling_matches_the_reference() {
        // Soap at 2 px/frame next to still oil: 0.5 * 2 / 2 = 0.5.
        assert_eq!(foam_couple(2.0, 0.0, 0.0), 0.5);
        assert_eq!(foam_couple(0.0, 0.0, 0.0), 0.0);
        // The soap term is measured against the local gravity, so a field that
        // points against the motion raises the blended speed.
        assert_eq!(foam_couple(0.0, 0.0, -1.0), 0.25);
    }
}
