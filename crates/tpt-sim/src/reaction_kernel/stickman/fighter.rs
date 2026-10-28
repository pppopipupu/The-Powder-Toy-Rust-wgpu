use cubecl::prelude::*;

use super::is_obstacle;
use super::{C_STKM_PLAYER1, C_STKM_PLAYER2};

/// Fighter command AI (`FIGH.cpp:60-145`).
#[cube]
#[allow(unused_assignments)]
pub(super) fn figh_ai_select_command(
    pmap: &[u32],
    xres: u32,
    yres: u32,
    x: f32,
    y: f32,
    stkm_legs: &mut [f32],
    stkm_meta: &mut [u32],
    fid: usize,
) -> u32 {
    let p1_spwn = stkm_meta[C_STKM_PLAYER1 * 8usize + 0usize] != 0u32;
    let p2_spwn = stkm_meta[C_STKM_PLAYER2 * 8usize + 0usize] != 0u32;

    let p1_x = stkm_legs[C_STKM_PLAYER1 * 16usize + 2usize];
    let p1_y = stkm_legs[C_STKM_PLAYER1 * 16usize + 3usize];
    let p2_x = stkm_legs[C_STKM_PLAYER2 * 16usize + 2usize];
    let p2_y = stkm_legs[C_STKM_PLAYER2 * 16usize + 3usize];

    let mut tarx = 0.0f32;
    let mut tary = 0.0f32;
    let mut has_target = false;

    if p2_spwn {
        if p1_spwn {
            let d1 = (p1_x - x) * (p1_x - x) + (p1_y - y) * (p1_y - y);
            let d2 = (p2_x - x) * (p2_x - x) + (p2_y - y) * (p2_y - y);
            if d1 <= d2 {
                tarx = p1_x;
                tary = p1_y;
            } else {
                tarx = p2_x;
                tary = p2_y;
            }
        } else {
            tarx = p2_x;
            tary = p2_y;
        }
        has_target = true;
    } else if p1_spwn {
        tarx = p1_x;
        tary = p1_y;
        has_target = true;
    }

    let lb = fid * 16usize;
    let figh_l4 = stkm_legs[lb + 4usize];
    let figh_l5 = stkm_legs[lb + 5usize];
    let figh_l6 = stkm_legs[lb + 6usize];
    let figh_l12 = stkm_legs[lb + 12usize];
    let figh_l13 = stkm_legs[lb + 13usize];
    let figh_l14 = stkm_legs[lb + 14usize];

    let rocket_boots = stkm_meta[fid * 8usize + 5usize] != 0u32;
    let pcomm = stkm_meta[fid * 8usize + 2usize];

    let mut comm = 0u32;

    if has_target {
        let dist2 = (tarx - x) * (tarx - x) + (tary - y) * (tary - y);
        if dist2 < 600.0f32 {
            comm = 0x08u32;
        } else if tarx < x {
            let blocked = is_obstacle(
                pmap,
                xres,
                yres,
                (figh_l4 - 10.0f32) as i32,
                (figh_l5 + 6.0f32) as i32,
            ) || is_obstacle(
                pmap,
                xres,
                yres,
                (figh_l4 - 10.0f32) as i32,
                (figh_l5 + 3.0f32) as i32,
            );
            if rocket_boots || !blocked {
                comm = 0x01u32;
            } else {
                comm = 0x02u32;
            }

            if rocket_boots {
                if tary < y {
                    comm |= 0x04u32;
                }
            } else {
                let obs1 = is_obstacle(
                    pmap,
                    xres,
                    yres,
                    (figh_l4 - 4.0f32) as i32,
                    (figh_l5 - 1.0f32) as i32,
                );
                let obs2 = is_obstacle(
                    pmap,
                    xres,
                    yres,
                    (figh_l12 - 4.0f32) as i32,
                    (figh_l13 - 1.0f32) as i32,
                );
                let clear_ahead = !is_obstacle(
                    pmap,
                    xres,
                    yres,
                    (2.0f32 * figh_l4 - figh_l6) as i32,
                    (figh_l5 + 5.0f32) as i32,
                );
                if obs1 || obs2 || clear_ahead {
                    comm |= 0x04u32;
                }
            }
        } else {
            let blocked = is_obstacle(
                pmap,
                xres,
                yres,
                (figh_l12 + 10.0f32) as i32,
                (figh_l13 + 6.0f32) as i32,
            ) || is_obstacle(
                pmap,
                xres,
                yres,
                (figh_l12 + 10.0f32) as i32,
                (figh_l13 + 3.0f32) as i32,
            );
            if rocket_boots || !blocked {
                comm = 0x02u32;
            } else {
                comm = 0x01u32;
            }

            if rocket_boots {
                if tary < y {
                    comm |= 0x04u32;
                }
            } else {
                let obs1 = is_obstacle(
                    pmap,
                    xres,
                    yres,
                    (figh_l4 + 4.0f32) as i32,
                    (figh_l5 - 1.0f32) as i32,
                );
                let obs2 = is_obstacle(
                    pmap,
                    xres,
                    yres,
                    (figh_l12 + 4.0f32) as i32,
                    (figh_l13 - 1.0f32) as i32,
                );
                let clear_ahead = !is_obstacle(
                    pmap,
                    xres,
                    yres,
                    (2.0f32 * figh_l12 - figh_l14) as i32,
                    (figh_l13 + 5.0f32) as i32,
                );
                if obs1 || obs2 || clear_ahead {
                    comm |= 0x04u32;
                }
            }
        }
    } else {
        let left_obs = is_obstacle(
            pmap,
            xres,
            yres,
            (figh_l4 - 4.0f32) as i32,
            (figh_l5 - 1.0f32) as i32,
        );
        let right_obs = is_obstacle(
            pmap,
            xres,
            yres,
            (figh_l12 + 4.0f32) as i32,
            (figh_l13 - 1.0f32) as i32,
        );
        if left_obs && !right_obs {
            comm = 0x02u32;
        } else if right_obs && !left_obs {
            comm = 0x01u32;
        } else if left_obs && right_obs {
            comm = 0x04u32;
        } else if (pcomm & 0x01u32) != 0u32 {
            comm = 0x01u32;
        } else {
            comm = 0x02u32;
        }
    }

    comm
}

#[cfg(test)]
#[allow(unused_assignments, clippy::too_many_arguments)]
pub(crate) fn figh_ai_select_command_host(
    pmap: &[u32],
    xres: u32,
    yres: u32,
    x: f32,
    y: f32,
    stkm_legs: &[f32],
    stkm_meta: &[u32],
    fid: usize,
) -> u32 {
    let p1_spwn = stkm_meta[C_STKM_PLAYER1 * 8 + 0] != 0;
    let p2_spwn = stkm_meta[C_STKM_PLAYER2 * 8 + 0] != 0;

    let p1_x = stkm_legs[C_STKM_PLAYER1 * 16 + 2];
    let p1_y = stkm_legs[C_STKM_PLAYER1 * 16 + 3];
    let p2_x = stkm_legs[C_STKM_PLAYER2 * 16 + 2];
    let p2_y = stkm_legs[C_STKM_PLAYER2 * 16 + 3];

    let mut tarx = 0.0f32;
    let mut tary = 0.0f32;
    let mut has_target = false;

    if p2_spwn {
        if p1_spwn {
            let d1 = (p1_x - x) * (p1_x - x) + (p1_y - y) * (p1_y - y);
            let d2 = (p2_x - x) * (p2_x - x) + (p2_y - y) * (p2_y - y);
            if d1 <= d2 {
                tarx = p1_x;
                tary = p1_y;
            } else {
                tarx = p2_x;
                tary = p2_y;
            }
        } else {
            tarx = p2_x;
            tary = p2_y;
        }
        has_target = true;
    } else if p1_spwn {
        tarx = p1_x;
        tary = p1_y;
        has_target = true;
    }

    let is_obs = |px: i32, py: i32| -> bool {
        if px < 0 || px >= xres as i32 || py < 0 || py >= yres as i32 {
            true
        } else {
            (pmap[(py as u32 * xres + px as u32) as usize] & 0x1FF) != 0
        }
    };

    let lb = fid * 16;
    let figh_l4 = stkm_legs[lb + 4];
    let figh_l5 = stkm_legs[lb + 5];
    let figh_l6 = stkm_legs[lb + 6];
    let figh_l12 = stkm_legs[lb + 12];
    let figh_l13 = stkm_legs[lb + 13];
    let figh_l14 = stkm_legs[lb + 14];

    let rocket_boots = stkm_meta[fid * 8 + 5] != 0;
    let pcomm = stkm_meta[fid * 8 + 2];

    let mut comm = 0u32;

    if has_target {
        let dist2 = (tarx - x) * (tarx - x) + (tary - y) * (tary - y);
        if dist2 < 600.0 {
            comm = 0x08;
        } else if tarx < x {
            let blocked = is_obs((figh_l4 - 10.0) as i32, (figh_l5 + 6.0) as i32)
                || is_obs((figh_l4 - 10.0) as i32, (figh_l5 + 3.0) as i32);
            if rocket_boots || !blocked {
                comm = 0x01;
            } else {
                comm = 0x02;
            }

            if rocket_boots {
                if tary < y {
                    comm |= 0x04;
                }
            } else {
                let obs1 = is_obs((figh_l4 - 4.0) as i32, (figh_l5 - 1.0) as i32);
                let obs2 = is_obs((figh_l12 - 4.0) as i32, (figh_l13 - 1.0) as i32);
                let clear_ahead = !is_obs((2.0 * figh_l4 - figh_l6) as i32, (figh_l5 + 5.0) as i32);
                if obs1 || obs2 || clear_ahead {
                    comm |= 0x04;
                }
            }
        } else {
            let blocked = is_obs((figh_l12 + 10.0) as i32, (figh_l13 + 6.0) as i32)
                || is_obs((figh_l12 + 10.0) as i32, (figh_l13 + 3.0) as i32);
            if rocket_boots || !blocked {
                comm = 0x02;
            } else {
                comm = 0x01;
            }

            if rocket_boots {
                if tary < y {
                    comm |= 0x04;
                }
            } else {
                let obs1 = is_obs((figh_l4 + 4.0) as i32, (figh_l5 - 1.0) as i32);
                let obs2 = is_obs((figh_l12 + 4.0) as i32, (figh_l13 - 1.0) as i32);
                let clear_ahead =
                    !is_obs((2.0 * figh_l12 - figh_l14) as i32, (figh_l13 + 5.0) as i32);
                if obs1 || obs2 || clear_ahead {
                    comm |= 0x04;
                }
            }
        }
    } else {
        let left_obs = is_obs((figh_l4 - 4.0) as i32, (figh_l5 - 1.0) as i32);
        let right_obs = is_obs((figh_l12 + 4.0) as i32, (figh_l13 - 1.0) as i32);
        if left_obs && !right_obs {
            comm = 0x02;
        } else if right_obs && !left_obs {
            comm = 0x01;
        } else if left_obs && right_obs {
            comm = 0x04;
        } else if (pcomm & 0x01) != 0 {
            comm = 0x01;
        } else {
            comm = 0x02;
        }
    }

    comm
}
