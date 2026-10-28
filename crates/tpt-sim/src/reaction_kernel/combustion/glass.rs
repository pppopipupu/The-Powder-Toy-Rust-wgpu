//! Glass: the pressure-differential shatter and the liquid-nitrogen
//! condensation on chemically strengthened glass.

use cubecl::prelude::*;
use tpt_core::constants::*;

use super::*;

/// `GLAS.cpp:50-90`: the glass `Update`.
///
/// Pixel parallel: the pressure history lives in the particle's `tmp3`, and the
/// condensation branch samples one of the four orthogonal neighbours.
///
/// `GLAS.cpp:59-63` reads `pv` as `int(pv * 64)` and compares it against the
/// value the previous frame stored; the `create` hook seeds the first value
/// (`GLAS.cpp:92-95`), which the host already mirrors when a glass pixel is
/// painted (`buffers/particles.rs:827`).
///
/// The `GLAS.cpp:80` creation goes through the device queue; TPT only spends
/// the pressure unit and the 200 K when `create_part` returns a slot, so the
/// queue call is made only after the free-pixel and bounds test, and a full
/// queue drops the child exactly like `create_part` returning `-1`.
#[cube(launch_unchecked)]
pub(crate) fn glass_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_temp: &mut [f32],
    part_tmp3: &mut [i32],
    part_flags: &mut [u32],
    air_pv: &mut [f32],
    air_hv: &[f32],
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
            if part_ty[id] == PT_GLAS {
                if stasis_frozen(part_flags, id) {
                    terminate!();
                }
                let px = (idx as u32 % xres) as i32;
                let py = (idx as u32 / xres) as i32;
                let cidx = (((py as u32) / CELL) * xcells + ((px as u32) / CELL)) as usize;
                let seed = hash_u32(frame_seed ^ mix_index(idx as u32, 2_654_435_761u32));

                // `GLAS.cpp:52-64`: a pressure jump beyond the strength of the
                // pane shatters it into `BGLA`. `life` is the chemical
                // strengthening (molten salt feeds it in `FIRE.cpp:285-289`);
                // the base strength is 16.
                let press = (air_pv[cidx] * 64.0f32) as i32;
                let diff = press - part_tmp3[id];
                let mut strength = part_life[id] / 120i32 + 16i32;
                if strength < 16i32 {
                    strength = 16i32;
                }
                if diff > strength || diff < -strength {
                    // `GLAS.cpp:59-63`: `part_change_type` leaves the fields
                    // alone, so `tmp3` is still written below.
                    part_ty[id] = PT_BGLA;
                }
                part_tmp3[id] = press;

                // `GLAS.cpp:66-87`: liquid nitrogen condenses on chemically
                // strengthened glass in a cold, pressurised cell.
                if strength > 200i32
                    && part_temp[id] < 77.0f32
                    && air_hv[cidx] < 77.0f32
                    && air_pv[cidx] > 5.0f32
                    && chance(seed, 1u32, 100u32)
                {
                    // `GLAS.cpp:70-73`: one of the four orthogonal neighbours.
                    let adj = rand_range(seed ^ 0x10u32, 0i32, 3i32);
                    let sign = 1i32 - 2i32 * (adj % 2i32);
                    let rx = sign * (1i32 - adj / 2i32);
                    let ry = sign * (adj / 2i32);
                    let nx = px + rx;
                    let ny = py + ry;
                    // `GLAS.cpp:75-77`: an empty spot around the glass. TPT's
                    // border cells are empty, but `create_part` still refuses
                    // an off-canvas pixel, so the bounds test is explicit here.
                    if nx >= 0i32
                        && ny >= 0i32
                        && nx < xres as i32
                        && ny < yres as i32
                        && pmap_at(pmap, xres, yres, nx, ny) == 0u32
                    {
                        // `GLAS.cpp:79-85`: the created nitrogen costs the cell
                        // one pressure unit and heats the pane by 200 K.
                        request_create_plain(create_queue, PT_LN2, nx, ny);
                        air_pv[cidx] -= 1.0f32;
                        part_temp[id] += 200.0f32;
                    }
                }
            }
        }
    }
}
