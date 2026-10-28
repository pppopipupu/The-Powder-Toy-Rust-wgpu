//! Rendering entry points: the smoke-test read-back path, the GPU-only path
//! the interactive viewport uses, plus the host-sync helper they share.

use crate::{SimulationEngine, render_kernel};

impl SimulationEngine {
    /// Render the current state into the packed RGBA8 framebuffer on the host.
    ///
    /// The returned slice is `xres * yres * 4` bytes, ready for
    /// `queue.write_texture` on an `Rgba8Unorm` texture.
    ///
    /// This reads the whole GPU framebuffer back to the host and MUST NOT be
    /// used by the interactive window: that path renders with [`Self::render_gpu`]
    /// and blits `bufs.gpu_framebuf` straight into the presentation texture on
    /// the device (`GpuContext::encode_sim_blit`), so no pixel data crosses the
    /// bus. Only `--headless-smoke` and the GPU parity tests call this.
    ///
    /// Any pending host edit (brush stroke, Lua spawn, save restore) is uploaded
    /// first, so the canvas is correct even while the simulation is paused.
    pub fn render_rgba(&mut self) -> &[u8] {
        self.sync_element_tables();
        if self.bufs.host_dirty {
            self.bufs.sync_host_to_gpu();
        }
        render_kernel::render_simulation_gpu(&mut self.bufs, &self.registry)
    }

    /// Render the current state into the GPU framebuffer without reading it back.
    ///
    /// This is the interactive path: the app copies `bufs.gpu_framebuf` straight
    /// into the presentation texture with a compute pass, so no pixel data ever
    /// crosses the PCIe bus.
    pub fn render_gpu(&mut self) {
        self.sync_element_tables();
        if self.bufs.host_dirty {
            self.bufs.sync_host_to_gpu();
        }
        render_kernel::render_gpu_only(&mut self.bufs, &self.registry);
    }

    /// Makes the host shadows match the device before host code inspects them.
    ///
    /// Used by the Lua bridge, save capture, undo snapshots, the HUD inspector
    /// and the flood fill tool; the renderer and the simulation step never call
    /// it on the hot path.
    pub fn ensure_host_current(&mut self) {
        self.bufs.ensure_host_current();
    }
}
