# The Powder Toy (Rust + wgpu + CubeCL)

English | [简体中文](README_ZH.md)

> **Work in Progress:** This project is still under active development. Some upstream content may not yet be ported, and performance issues may occur.

A GPU-accelerated Rust port of *The Powder Toy*. Simulation passes and fancy rendering execute as CubeCL compute kernels on `wgpu`, paired with a `winit` and `egui-wgpu` desktop front end.

## Overview

- **Device-resident GPU pipeline**: Eulerian air/ambient heat, thermal conduction, phase/pressure transitions, element reactions, lock-free particle movement, Game of Life (`LIFE`), FFT Newtonian gravity, water equalisation, and fancy display rendering run on a single shared `wgpu::Device`/`Queue`, blitted straight into an `egui` texture without per-frame CPU/GPU round-trips.
- **Full TPT element and wall set**: Ports the 195-slot built-in element registry (`PT_NONE` plus 194 elements), 19 wall types, and portal/wireless/piston/stickman mechanics.
- **Workspace crates**: `tpt-core` (shared constants, element/wall definitions, 3D voxel model), `tpt-sim` (GPU buffers, CubeCL kernels, rigid bodies, 3D voxel engine), `tpt-save` (`.tptrs` / `.tpt3` serialization, undo history), and `tpt-app` (desktop window, UI, preferences, smoke/bench runners).

## Changes from Upstream The Powder Toy

- **Rust + `wgpu` + CubeCL engine & `egui` UI**: Replaces the C++/SDL CPU cellular automaton and SDL UI with Rust compute kernels (`wgpu` / `CubeCL`) and `winit` + `egui-wgpu`.
- **Batched lock-free simulation**: Element reactions are batched by particle type and separated from the lock-free atomic-arbitration movement pass instead of C++'s slot-serial interleaved loop (neighbor chain reactions and cyclic liquid swaps lag by one tick).
- **Configurable canvas sizes**: Supports presets from `612x384` up to `1600x900` and custom sizes (`128..=2048` in the menu, 4 px cell aligned) rather than a fixed `612x384` grid.
- **2D pixel-cluster rigid bodies**: Adds a `RigidBodies` category (`RigidBody2D`) simulating destructible particle clusters on the GPU with sub-stepped contacts and per-pixel thermal/chemical fracture back into free particles.
- **3D voxel sandbox (WIP)**: Adds an independent `256x256x256` GPU voxel sandbox with sparse active-brick indirect dispatch, a `64^3` air/pressure grid, 6-DOF rigid voxel bodies, and volumetric DDA rendering.
- **Local-first saves, APNG & i18n**: Uses local zlib-compressed `.tptrs` (2D) / `.tpt3` (3D) snapshots and local `./saves/` / `./stamps/` instead of `.cps` and online community save servers; encodes recordings directly as `.apng` instead of frame-by-frame PPM images; adds English and Simplified Chinese (`zh-CN`) UI localization.

## Build and Run

Requirements: Rust 1.99.1+, a `wgpu`-capable GPU/driver (Vulkan, DX12, or Metal), and a C toolchain.

```bash
cargo run --release -p tpt-app
cargo run --release -p tpt-app -- --headless-smoke
cargo test --workspace --release
```
