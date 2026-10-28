use cubecl::prelude::*;

use super::temp_channel;
use super::{PORTAL_CHANNELS_U32, PT_INWR_U32, PT_NSCN_U32, PT_PSCN_U32, PT_SPRK_U32, PT_WIFI_U32};
use crate::reaction_kernel::shared::{pmap_at, pmap_slot};

/// Step wireless channels: advance `active_next` into `active_now`, and reset `active_next`.
#[allow(
    clippy::manual_clamp,
    clippy::too_many_arguments,
    clippy::approx_constant
)]
#[cube(launch_unchecked)]
pub fn wifi_step_kernel(wireless: &mut [Atomic<u32>]) {
    let ch = ABSOLUTE_POS;
    if ch >= PORTAL_CHANNELS_U32 as usize {
        terminate!();
    }
    let next = wireless[ch * 2usize + 1usize].load();
    wireless[ch * 2usize].store(next);
    wireless[ch * 2usize + 1usize].store(0u32);
}

/// Wireless transmitter: scans adjacent sparks and activates channel for next frame.
#[allow(
    clippy::manual_clamp,
    clippy::too_many_arguments,
    clippy::approx_constant
)]
#[cube(launch_unchecked)]
pub fn wifi_transmit_kernel(
    pmap: &[u32],
    part_ty: &[u32],
    part_life: &[i32],
    part_ctype: &[u32],
    part_temp: &[f32],
    part_tmp: &mut [i32],
    wireless: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 {
        terminate!();
    }
    let ty = v & 0x1FFu32;
    if ty != PT_WIFI_U32 {
        terminate!();
    }
    let s = pmap_slot(v);
    if part_ty[s] != PT_WIFI_U32 {
        terminate!();
    }
    let ch = temp_channel(part_temp[s]);
    part_tmp[s] = ch as i32;
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;

    for n in 0u32..9u32 {
        let dx = (n % 3u32) as i32 - 1i32;
        let dy = (n / 3u32) as i32 - 1i32;
        if dx != 0i32 || dy != 0i32 {
            let nx = px + dx;
            let ny = py + dy;
            let nv = pmap_at(pmap, xres, yres, nx, ny);
            if nv != 0u32 {
                let nty = nv & 0x1FFu32;
                if nty == PT_SPRK_U32 {
                    let ns = pmap_slot(nv);
                    if part_ty[ns] == PT_SPRK_U32
                        && part_ctype[ns] != PT_NSCN_U32
                        && part_life[ns] >= 3i32
                    {
                        wireless[(ch * 2u32 + 1u32) as usize].store(1u32);
                    }
                }
            }
        }
    }
}

/// Wireless receiver: conductors check adjacent `WIFI` on active channels and spark.
#[allow(
    clippy::manual_clamp,
    clippy::too_many_arguments,
    clippy::approx_constant
)]
#[cube(launch_unchecked)]
pub fn wifi_receive_kernel(
    pmap: &[u32],
    part_ty: &mut [u32],
    part_life: &mut [i32],
    part_ctype: &mut [u32],
    part_temp: &[f32],
    wireless: &mut [Atomic<u32>],
    xres: u32,
    yres: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= (xres * yres) as usize {
        terminate!();
    }
    let v = pmap[idx];
    if v == 0u32 {
        terminate!();
    }
    let ty = v & 0x1FFu32;
    if ty != PT_NSCN_U32 && ty != PT_PSCN_U32 && ty != PT_INWR_U32 {
        terminate!();
    }
    let s = pmap_slot(v);
    if part_ty[s] != ty || part_life[s] != 0i32 {
        terminate!();
    }
    let px = (idx as u32 % xres) as i32;
    let py = (idx as u32 / xres) as i32;

    let mut sparked = false;
    for n in 0u32..9u32 {
        let dx = (n % 3u32) as i32 - 1i32;
        let dy = (n / 3u32) as i32 - 1i32;
        if (dx != 0i32 || dy != 0i32) && !sparked {
            let nx = px + dx;
            let ny = py + dy;
            let nv = pmap_at(pmap, xres, yres, nx, ny);
            if nv != 0u32 {
                let nty = nv & 0x1FFu32;
                // `WIFI.cpp:63-84`: no insulation test — any adjacent spark with
                // `life >= 3` and a non-`NSCN` `ctype` arms the channel.
                if nty == PT_WIFI_U32 {
                    let ws = pmap_slot(nv);
                    if part_ty[ws] == PT_WIFI_U32 {
                        let ch = temp_channel(part_temp[ws]);
                        if wireless[(ch * 2u32) as usize].load() != 0u32 {
                            part_ctype[s] = ty;
                            part_ty[s] = PT_SPRK_U32;
                            part_life[s] = 4i32;
                            sparked = true;
                        }
                    }
                }
            }
        }
    }
}
