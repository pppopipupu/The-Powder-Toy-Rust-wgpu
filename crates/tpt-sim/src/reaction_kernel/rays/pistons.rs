use cubecl::prelude::*;

use super::{C_GPU_ALLOC_NONE, C_PT_FRME, C_PT_NONE, C_PT_PSCN, C_PT_PSTN, C_PT_SPRK};
use super::{seed_i32, seed_u32};
use crate::reaction_kernel::shared::{gpu_alloc_slot, pmap_at, pmap_slot};

const FLAG_EXT_OBSTACLE: i32 = 1;
const FLAG_EXT_ARMLIMIT: i32 = 2;
const FLAG_RET_OBSTACLE: i32 = 4;
const FLAG_RET_TOOSHORT: i32 = 8;

const PISTON_INACTIVE: u32 = 0;
const PISTON_RETRACT: u32 = 1;
const PISTON_EXTEND: u32 = 2;
const MAX_FRAME: i32 = 15;

/// TPT `PSTN.cpp` temperature force: `floor((temp - 268.15) / 10.0)`.
#[cube]
pub fn pstn_temp_force(temp: f32) -> i32 {
    let v = (temp - 268.15f32) / 10.0f32;
    let truncated = v as i32;
    if (truncated as f32) > v {
        truncated - 1i32
    } else {
        truncated
    }
}

/// TPT `PSTN.cpp:195-198`: darkens `dcolour` by `0x3C` (60) per RGB channel.
#[cube]
#[allow(clippy::implicit_saturating_sub)]
pub fn pstn_darken_dcolour(colour: u32) -> u32 {
    if colour == 0u32 {
        0u32
    } else {
        let a = colour & 0xFF00_0000u32;
        let r = colour & 0x00FF_0000u32;
        let g = colour & 0x0000_FF00u32;
        let b = colour & 0x0000_00FFu32;

        let new_r = if r >= 0x003C_0000u32 {
            r - 0x003C_0000u32
        } else {
            0u32
        };
        let new_g = if g >= 0x0000_3C00u32 {
            g - 0x0000_3C00u32
        } else {
            0u32
        };
        let new_b = if b >= 0x0000_003Cu32 {
            b - 0x0000_003Cu32
        } else {
            0u32
        };

        a | new_r | new_g | new_b
    }
}

/// `Simulation::IsWallBlocking(x, y, 0)` (`Simulation.cpp:1076-1099`) for the
/// `type == 0` that `PSTN.cpp:245` passes to `CanMoveStack`.
///
/// TPT only fixed 4x4-cell wall ids block a stack: every "allow" wall rejects
/// type 0 because `elements[0]` has no type bits, `WL_WALL`/`WL_WALLELEC`/
/// `WL_ALLOWAIR` always block, and `WL_EWALL` blocks until it is electrified.
/// `WL_DETECT` only blocks solids (type 0 is not one), and the unlisted ids
/// (`WL_STREAM`, `WL_FAN`, `WL_GRAV`, ...) let the stack through.
#[cube]
fn wall_blocks_stack(wall: u32, emap: u32) -> bool {
    let mut blocked = wall == 1u32
        || wall == 6u32
        || wall == 8u32
        || wall == 9u32
        || wall == 10u32
        || wall == 13u32
        || wall == 15u32;
    if wall == 2u32 {
        blocked = (emap & 0xFFu32) == 0u32;
    }
    blocked
}

/// TPT `CanMoveStack`: returns number of empty spaces available along `(dx, dy)`.
#[cube]
fn can_move_stack_spaces(
    pmap: &[u32],
    start_x: i32,
    start_y: i32,
    dx: i32,
    dy: i32,
    max_size: i32,
    amount: i32,
    block: u32,
    wall_bmap: &[u32],
    wall_emap: &[u32],
    xcells: u32,
    xres: u32,
    yres: u32,
) -> i32 {
    let mut spaces = 0i32;
    let mut pushed = 0i32;
    for step in 0u32..64u32 {
        if spaces >= amount || (pushed + spaces) >= (max_size + amount) {
            break;
        }
        let cx = start_x + dx * (step as i32);
        let cy = start_y + dy * (step as i32);
        if cx < 0i32 || cy < 0i32 || (cx as u32) >= xres || (cy as u32) >= yres {
            break;
        }
        // `PSTN.cpp:245`: a wall stops the stack exactly where the block
        // element would.
        let cell = (((cy as u32) / 4u32) * xcells + ((cx as u32) / 4u32)) as usize;
        if wall_blocks_stack(wall_bmap[cell] & 0xFFu32, wall_emap[cell] & 0xFFu32) {
            break;
        }
        let p = pmap_at(pmap, xres, yres, cx, cy);
        let ty = p & 0x1FFu32;
        if block != 0u32 && ty == block {
            break;
        }
        if p == 0u32 {
            spaces += 1i32;
        } else if pushed < max_size {
            pushed += 1i32;
        } else {
            break;
        }
    }
    spaces
}

/// TPT `MoveStack` on extend: shifts particles forward along `(dx, dy)` into empty spaces.
#[cube]
fn shift_stack_forward(
    pmap: &[u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    start_x: i32,
    start_y: i32,
    dx: i32,
    dy: i32,
    max_size: i32,
    amount: i32,
    block: u32,
    wall_bmap: &[u32],
    wall_emap: &[u32],
    xcells: u32,
    xres: u32,
    yres: u32,
) {
    let mut total_steps = 0u32;
    let mut spaces_found = 0i32;
    let mut pushed_count = 0i32;
    for step in 0u32..64u32 {
        if spaces_found >= amount || (pushed_count + spaces_found) >= (max_size + amount) {
            break;
        }
        let cx = start_x + dx * (step as i32);
        let cy = start_y + dy * (step as i32);
        if cx < 0i32 || cy < 0i32 || (cx as u32) >= xres || (cy as u32) >= yres {
            break;
        }
        // Mirror `CanMoveStack`'s wall stop so the particles behind it stay
        // put: TPT moves exactly the list that scan returned (`PSTN.cpp:245`).
        let cell = (((cy as u32) / 4u32) * xcells + ((cx as u32) / 4u32)) as usize;
        if wall_blocks_stack(wall_bmap[cell] & 0xFFu32, wall_emap[cell] & 0xFFu32) {
            break;
        }
        let p = pmap_at(pmap, xres, yres, cx, cy);
        let ty = p & 0x1FFu32;
        if block != 0u32 && ty == block {
            break;
        }
        if p == 0u32 {
            spaces_found += 1i32;
            total_steps = step + 1u32;
        } else if pushed_count < max_size {
            pushed_count += 1i32;
            total_steps = step + 1u32;
        } else {
            break;
        }
    }

    for step in 0u32..64u32 {
        if step >= total_steps {
            break;
        }
        let cx = start_x + dx * (step as i32);
        let cy = start_y + dy * (step as i32);
        let p = pmap_at(pmap, xres, yres, cx, cy);
        if p != 0u32 {
            let mut empty_ahead = 0i32;
            for ahead in 0u32..64u32 {
                if ahead > step && ahead < total_steps {
                    let ax = start_x + dx * (ahead as i32);
                    let ay = start_y + dy * (ahead as i32);
                    if pmap_at(pmap, xres, yres, ax, ay) == 0u32 {
                        empty_ahead += 1i32;
                    }
                }
            }
            if empty_ahead > 0i32 {
                let slot = pmap_slot(p);
                part_x[slot] += (dx * empty_ahead) as f32;
                part_y[slot] += (dy * empty_ahead) as f32;
            }
        }
    }
}

/// TPT `MoveStack` on retract: pulls particles backward along `(dx, dy)`.
#[cube]
fn pull_stack_backward(
    pmap: &[u32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    start_x: i32,
    start_y: i32,
    dx: i32,
    dy: i32,
    max_size: i32,
    amount: i32,
    block: u32,
    sticky: bool,
    xres: u32,
    yres: u32,
) {
    for step in 0u32..64u32 {
        if (step as i32) >= max_size {
            break;
        }
        let cx = start_x + dx * (step as i32);
        let cy = start_y + dy * (step as i32);
        if cx < 0i32 || cy < 0i32 || (cx as u32) >= xres || (cy as u32) >= yres {
            break;
        }
        let p = pmap_at(pmap, xres, yres, cx, cy);
        let ty = p & 0x1FFu32;
        if p == 0u32 || (block != 0u32 && ty == block) || (!sticky && ty != C_PT_FRME) {
            break;
        }
        let slot = pmap_slot(p);
        part_x[slot] -= (dx * amount) as f32;
        part_y[slot] -= (dy * amount) as f32;
    }
}

/// `PSTN` piston update pass (`PSTN.cpp`).
#[cube(launch_unchecked)]
#[allow(
    clippy::manual_clamp,
    clippy::too_many_arguments,
    clippy::approx_constant,
    unused_assignments
)]
pub fn pstn_kernel(
    pmap: &[u32],
    free_slots: &mut [Atomic<u32>],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &[u32],
    part_temp: &[f32],
    part_tmp: &[i32],
    part_tmp2: &[i32],
    part_tmp3: &[i32],
    part_x: &mut [f32],
    part_y: &mut [f32],
    part_dcolour: &mut [u32],
    wall_bmap: &[u32],
    wall_emap: &[u32],
    xres: u32,
    yres: u32,
    xcells: u32,
) {
    let idx = ABSOLUTE_POS as u32;
    if idx >= xres * yres {
        terminate!();
    }
    let p = pmap[idx as usize];
    if (p & 0x1FFu32) != C_PT_PSTN {
        terminate!();
    }
    let i = pmap_slot(p);
    if part_life[i] != 0i32 {
        terminate!();
    }
    let px = (idx % xres) as i32;
    let py = (idx / xres) as i32;

    let mut state = seed_u32(part_ty[i], PISTON_INACTIVE);
    for off_idx in 0u32..8u32 {
        let mut rx = 0i32;
        let mut ry = 0i32;
        if off_idx == 0u32 {
            rx = -2;
            ry = 0;
        } else if off_idx == 1u32 {
            rx = -1;
            ry = 0;
        } else if off_idx == 2u32 {
            rx = 1;
            ry = 0;
        } else if off_idx == 3u32 {
            rx = 2;
            ry = 0;
        } else if off_idx == 4u32 {
            rx = 0;
            ry = -2;
        } else if off_idx == 5u32 {
            rx = 0;
            ry = -1;
        } else if off_idx == 6u32 {
            rx = 0;
            ry = 1;
        } else {
            rx = 0;
            ry = 2;
        }

        let sp = pmap_at(pmap, xres, yres, px + rx, py + ry);
        if (sp & 0x1FFu32) == C_PT_SPRK {
            let sprk = pmap_slot(sp);
            if part_life[sprk] == 3i32 {
                if part_ctype[sprk] == C_PT_PSCN {
                    state = PISTON_EXTEND;
                } else {
                    state = PISTON_RETRACT;
                }
            }
        }
    }

    if state == PISTON_INACTIVE {
        terminate!();
    }

    let max_size = if part_tmp[i] > 0i32 {
        part_tmp[i]
    } else {
        31i32
    };
    let arm_limit = if part_tmp2[i] > 0i32 {
        part_tmp2[i]
    } else {
        255i32
    };
    let block = part_ctype[i] & 0x1FFu32;
    let flags = part_tmp3[i];

    let mut moved_piston = false;

    for dir_idx in 0u32..4u32 {
        if moved_piston {
            break;
        }
        let mut dx = 0i32;
        let mut dy = 0i32;
        if dir_idx == 0u32 {
            dx = 0;
            dy = -1;
        } else if dir_idx == 1u32 {
            dx = 0;
            dy = 1;
        } else if dir_idx == 2u32 {
            dx = -1;
            dy = 0;
        } else {
            dx = 1;
            dy = 0;
        }

        let n_p = pmap_at(pmap, xres, yres, px + dx, py + dy);
        if (n_p & 0x1FFu32) == C_PT_PSTN && part_life[pmap_slot(n_p)] == 0i32 {
            let mut piston_count = seed_i32(part_life[i], -1i32);
            let mut arm_count = seed_i32(part_life[i], 0i32);
            let mut piston_end_x = seed_i32(px, 0i32);
            let mut piston_end_y = seed_i32(py, 0i32);
            let mut found_end = false;

            for step in 0u32..256u32 {
                let cx = px + dx * (step as i32);
                let cy = py + dy * (step as i32);
                if cx < 0i32 || cy < 0i32 || (cx as u32) >= xres || (cy as u32) >= yres {
                    break;
                }
                let cur_p = pmap_at(pmap, xres, yres, cx, cy);
                if (cur_p & 0x1FFu32) == C_PT_PSTN {
                    let slot = pmap_slot(cur_p);
                    if part_life[slot] != 0i32 {
                        arm_count += 1i32;
                    } else if arm_count > 0i32 {
                        piston_end_x = cx;
                        piston_end_y = cy;
                        found_end = true;
                        break;
                    } else {
                        piston_count += pstn_temp_force(part_temp[slot]);
                    }
                } else if step == 0u32 {
                    piston_count += pstn_temp_force(part_temp[i]);
                } else {
                    piston_end_x = cx;
                    piston_end_y = cy;
                    found_end = true;
                    break;
                }
            }

            if found_end {
                if state == PISTON_EXTEND {
                    let mut can_extend = true;
                    if arm_count + piston_count > arm_limit {
                        if (flags & FLAG_EXT_ARMLIMIT) != 0i32 {
                            can_extend = false;
                        } else {
                            piston_count = arm_limit - arm_count;
                        }
                    }
                    if can_extend && piston_count > 0i32 {
                        let cancel_on_obstacle = (flags & FLAG_EXT_OBSTACLE) != 0i32;
                        let end_p = pmap_at(pmap, xres, yres, piston_end_x, piston_end_y);
                        let is_frame = (end_p & 0x1FFu32) == C_PT_FRME;

                        if is_frame {
                            let new_x = if dy != 0i32 { 1i32 } else { 0i32 };
                            let new_y = if dx != 0i32 { 1i32 } else { 0i32 };
                            let mut effective_amount = piston_count;
                            let mut max_right = seed_i32(piston_end_x, MAX_FRAME);
                            let mut max_left = seed_i32(piston_end_x, MAX_FRAME);

                            for c in 0i32..MAX_FRAME {
                                let cx = piston_end_x + c * new_x;
                                let cy = piston_end_y + c * new_y;
                                let cp = pmap_at(pmap, xres, yres, cx, cy);
                                if (cp & 0x1FFu32) == C_PT_FRME {
                                    let sp = can_move_stack_spaces(
                                        pmap,
                                        cx,
                                        cy,
                                        dx,
                                        dy,
                                        max_size,
                                        effective_amount,
                                        block,
                                        wall_bmap,
                                        wall_emap,
                                        xcells,
                                        xres,
                                        yres,
                                    );
                                    if sp < effective_amount {
                                        if cancel_on_obstacle {
                                            effective_amount = 0i32;
                                            break;
                                        }
                                        effective_amount = sp;
                                    }
                                } else {
                                    max_right = c;
                                    break;
                                }
                            }

                            if effective_amount > 0i32 {
                                for c in 1i32..MAX_FRAME {
                                    let cx = piston_end_x - c * new_x;
                                    let cy = piston_end_y - c * new_y;
                                    let cp = pmap_at(pmap, xres, yres, cx, cy);
                                    if (cp & 0x1FFu32) == C_PT_FRME {
                                        let sp = can_move_stack_spaces(
                                            pmap,
                                            cx,
                                            cy,
                                            dx,
                                            dy,
                                            max_size,
                                            effective_amount,
                                            block,
                                            wall_bmap,
                                            wall_emap,
                                            xcells,
                                            xres,
                                            yres,
                                        );
                                        if sp < effective_amount {
                                            if cancel_on_obstacle {
                                                effective_amount = 0i32;
                                                break;
                                            }
                                            effective_amount = sp;
                                        }
                                    } else {
                                        max_left = c;
                                        break;
                                    }
                                }
                            }

                            if effective_amount > 0i32 {
                                for c in 1i32..max_right {
                                    let cx = piston_end_x + c * new_x;
                                    let cy = piston_end_y + c * new_y;
                                    shift_stack_forward(
                                        pmap,
                                        part_x,
                                        part_y,
                                        cx,
                                        cy,
                                        dx,
                                        dy,
                                        max_size,
                                        effective_amount,
                                        block,
                                        wall_bmap,
                                        wall_emap,
                                        xcells,
                                        xres,
                                        yres,
                                    );
                                }
                                for c in 1i32..max_left {
                                    let cx = piston_end_x - c * new_x;
                                    let cy = piston_end_y - c * new_y;
                                    shift_stack_forward(
                                        pmap,
                                        part_x,
                                        part_y,
                                        cx,
                                        cy,
                                        dx,
                                        dy,
                                        max_size,
                                        effective_amount,
                                        block,
                                        wall_bmap,
                                        wall_emap,
                                        xcells,
                                        xres,
                                        yres,
                                    );
                                }
                                shift_stack_forward(
                                    pmap,
                                    part_x,
                                    part_y,
                                    piston_end_x,
                                    piston_end_y,
                                    dx,
                                    dy,
                                    max_size,
                                    effective_amount,
                                    block,
                                    wall_bmap,
                                    wall_emap,
                                    xcells,
                                    xres,
                                    yres,
                                );

                                for j in 0i32..effective_amount {
                                    let nr = gpu_alloc_slot(free_slots);
                                    if nr != C_GPU_ALLOC_NONE {
                                        let s = nr as usize;
                                        part_ty[s] = C_PT_PSTN;
                                        part_life[s] = 1i32;
                                        part_x[s] = (piston_end_x + dx * j) as f32;
                                        part_y[s] = (piston_end_y + dy * j) as f32;
                                        part_dcolour[s] = pstn_darken_dcolour(part_dcolour[i]);
                                    }
                                }
                                moved_piston = true;
                            }
                        } else {
                            // Direct stack push (no FRME)
                            let sp = can_move_stack_spaces(
                                pmap,
                                piston_end_x,
                                piston_end_y,
                                dx,
                                dy,
                                max_size,
                                piston_count,
                                block,
                                wall_bmap,
                                wall_emap,
                                xcells,
                                xres,
                                yres,
                            );
                            let mut effective_amount = piston_count;
                            if sp < effective_amount {
                                if cancel_on_obstacle {
                                    effective_amount = 0i32;
                                } else {
                                    effective_amount = sp;
                                }
                            }

                            if effective_amount > 0i32 {
                                shift_stack_forward(
                                    pmap,
                                    part_x,
                                    part_y,
                                    piston_end_x,
                                    piston_end_y,
                                    dx,
                                    dy,
                                    max_size,
                                    effective_amount,
                                    block,
                                    wall_bmap,
                                    wall_emap,
                                    xcells,
                                    xres,
                                    yres,
                                );
                                for j in 0i32..effective_amount {
                                    let nr = gpu_alloc_slot(free_slots);
                                    if nr != C_GPU_ALLOC_NONE {
                                        let s = nr as usize;
                                        part_ty[s] = C_PT_PSTN;
                                        part_life[s] = 1i32;
                                        part_x[s] = (piston_end_x + dx * j) as f32;
                                        part_y[s] = (piston_end_y + dy * j) as f32;
                                        part_dcolour[s] = pstn_darken_dcolour(part_dcolour[i]);
                                    }
                                }
                                moved_piston = true;
                            }
                        }
                    }
                } else if state == PISTON_RETRACT {
                    let mut can_retract = true;
                    if piston_count > arm_count {
                        if (flags & FLAG_RET_TOOSHORT) != 0i32 {
                            can_retract = false;
                        } else {
                            piston_count = arm_count;
                        }
                    }
                    if can_retract && arm_count > 0i32 && piston_count > 0i32 {
                        let cancel_on_obstacle = (flags & FLAG_RET_OBSTACLE) != 0i32;
                        let end_p = pmap_at(pmap, xres, yres, piston_end_x, piston_end_y);
                        let is_frame = (end_p & 0x1FFu32) == C_PT_FRME;

                        if is_frame {
                            let new_x = if dy != 0i32 { 1i32 } else { 0i32 };
                            let new_y = if dx != 0i32 { 1i32 } else { 0i32 };
                            let mut effective_amount = piston_count;
                            let mut max_right = seed_i32(piston_end_x, MAX_FRAME);
                            let mut max_left = seed_i32(piston_end_x, MAX_FRAME);

                            for c in 1i32..MAX_FRAME {
                                let cx = piston_end_x + c * new_x;
                                let cy = piston_end_y + c * new_y;
                                let cp = pmap_at(pmap, xres, yres, cx, cy);
                                if (cp & 0x1FFu32) == C_PT_FRME {
                                    let sp = can_move_stack_spaces(
                                        pmap,
                                        cx,
                                        cy,
                                        -dx,
                                        -dy,
                                        max_size,
                                        effective_amount,
                                        block,
                                        wall_bmap,
                                        wall_emap,
                                        xcells,
                                        xres,
                                        yres,
                                    );
                                    if sp < effective_amount {
                                        if cancel_on_obstacle {
                                            effective_amount = 0i32;
                                            break;
                                        }
                                        effective_amount = sp;
                                    }
                                } else {
                                    max_right = c;
                                    break;
                                }
                            }

                            if effective_amount > 0i32 {
                                for c in 1i32..MAX_FRAME {
                                    let cx = piston_end_x - c * new_x;
                                    let cy = piston_end_y - c * new_y;
                                    let cp = pmap_at(pmap, xres, yres, cx, cy);
                                    if (cp & 0x1FFu32) == C_PT_FRME {
                                        let sp = can_move_stack_spaces(
                                            pmap,
                                            cx,
                                            cy,
                                            -dx,
                                            -dy,
                                            max_size,
                                            effective_amount,
                                            block,
                                            wall_bmap,
                                            wall_emap,
                                            xcells,
                                            xres,
                                            yres,
                                        );
                                        if sp < effective_amount {
                                            if cancel_on_obstacle {
                                                effective_amount = 0i32;
                                                break;
                                            }
                                            effective_amount = sp;
                                        }
                                    } else {
                                        max_left = c;
                                        break;
                                    }
                                }
                            }

                            if effective_amount > 0i32 {
                                for j in 1i32..=effective_amount {
                                    let ax = piston_end_x - dx * j;
                                    let ay = piston_end_y - dy * j;
                                    let ap = pmap_at(pmap, xres, yres, ax, ay);
                                    if (ap & 0x1FFu32) == C_PT_PSTN {
                                        let s = pmap_slot(ap);
                                        part_ty[s] = C_PT_NONE;
                                    }
                                }

                                for c in 1i32..max_right {
                                    let cx = piston_end_x + c * new_x;
                                    let cy = piston_end_y + c * new_y;
                                    let frme_slot = pmap_slot(pmap_at(pmap, xres, yres, cx, cy));
                                    let sticky = part_tmp[frme_slot] == 0i32;
                                    pull_stack_backward(
                                        pmap,
                                        part_x,
                                        part_y,
                                        cx,
                                        cy,
                                        dx,
                                        dy,
                                        max_size,
                                        effective_amount,
                                        block,
                                        sticky,
                                        xres,
                                        yres,
                                    );
                                }
                                for c in 1i32..max_left {
                                    let cx = piston_end_x - c * new_x;
                                    let cy = piston_end_y - c * new_y;
                                    let frme_slot = pmap_slot(pmap_at(pmap, xres, yres, cx, cy));
                                    let sticky = part_tmp[frme_slot] == 0i32;
                                    pull_stack_backward(
                                        pmap,
                                        part_x,
                                        part_y,
                                        cx,
                                        cy,
                                        dx,
                                        dy,
                                        max_size,
                                        effective_amount,
                                        block,
                                        sticky,
                                        xres,
                                        yres,
                                    );
                                }
                                let center_slot = pmap_slot(pmap_at(
                                    pmap,
                                    xres,
                                    yres,
                                    piston_end_x,
                                    piston_end_y,
                                ));
                                let sticky = part_tmp[center_slot] == 0i32;
                                pull_stack_backward(
                                    pmap,
                                    part_x,
                                    part_y,
                                    piston_end_x,
                                    piston_end_y,
                                    dx,
                                    dy,
                                    max_size,
                                    effective_amount,
                                    block,
                                    sticky,
                                    xres,
                                    yres,
                                );
                                moved_piston = true;
                            }
                        } else {
                            // Direct stack retract (no FRME)
                            for j in 1i32..=piston_count {
                                let ax = piston_end_x - dx * j;
                                let ay = piston_end_y - dy * j;
                                let ap = pmap_at(pmap, xres, yres, ax, ay);
                                if (ap & 0x1FFu32) == C_PT_PSTN {
                                    let s = pmap_slot(ap);
                                    part_ty[s] = C_PT_NONE;
                                }
                            }
                            pull_stack_backward(
                                pmap,
                                part_x,
                                part_y,
                                piston_end_x,
                                piston_end_y,
                                dx,
                                dy,
                                max_size,
                                piston_count,
                                block,
                                true,
                                xres,
                                yres,
                            );
                            moved_piston = true;
                        }
                    }
                }
            }
        }
    }
}
