# Repository Guidelines

## Hard Constraints

- **The local C++ checkout is the behavioural source of truth.** Every element behaviour, UI, preference/setting and game mechanic MUST be ported from `C:\Users\pppop\Desktop\azuki\The-Powder-Toy` (read-only sibling checkout): `src/simulation/` (`Simulation.cpp`, `SimulationData.cpp`, `elements/*.cpp`, `SimTool.cpp`, `Sign.cpp`), `src/gui/options/` and `src/prefs/` for UI and settings, `src/graphics/` for rendering. Cite the source file in the code comment that implements the behaviour. Observable mechanics and visuals MUST match TPT; save format (`.tptrs`) and frame timing are the port's own.
- **NEVER invent particle mechanics.** No guessed thresholds, lifetimes, reaction partners, colours, scan orders, `ctype`/`tmp` semantics or transitions. A behaviour not present in the C++ source is not implemented; when the source is ambiguous, port the literal code path, not a plausible approximation.
- **NEVER simplify particle mechanics.** Port the full 195-element table and every `Update`/`Graphics`/`Create` branch, including `ST`, `ctype`, `tmp`, `life` and special-case transitions. No reduced tables, collapsed passes, omitted branches or "close enough" values.
- **NEVER clone or download the upstream repository.** No `git clone`, `gh repo clone`, tarball, mirror or network lookup. The sibling checkout is the only reference and MUST NOT be modified.
- **NEVER ping-pong state between GPU and CPU.** Simulation and rendering MUST stay device-resident. No per-tick or per-frame full-state readback, staging copy or host-shadow refresh. Upload only what the host changed (`host_dirty_slots`, dirty categories). `Client::read`/`map_async` are allowed only for narrow bounded slices that host bookkeeping genuinely needs (spawn window, save/undo capture, Lua grid getter, cached HUD probe, periodic HUD count, tests), never for the simulation or render state each frame.
- **Prefer WGSL/CubeCL device code.** New simulation and render work MUST be a device kernel, authored in CubeCL (WGSL backend) or raw WGSL. NEVER add a CPU reference path to the runtime and NEVER compute on the CPU then upload the result. CPU code is limited to host-side command application, `create_part` and Lua edits, uploaded once.
- **NEVER put shortcut hints outside Settings.** Key bindings, chords, mouse buttons and gestures are named only in the Settings window's Key bindings page (`ui_panels/options_window.rs`, opened by the Settings button on the top bar or title screen; rows in `KEYBINDS`/`VOXEL_KEYBINDS`). Panels, tooltips, menus, tool descriptions, HUD text and hints describe the action alone and MUST NOT name a key, chord, button or gesture. A new binding gets a row there and appears nowhere else; existing hints outside that page MUST be deleted.
- **Element, wall and tool names are NEVER localized.** Locale JSON translates prose only. Element names, wall names, tool labels (`Air`, `Heat`, `Cool`, `Vacuum`, `Wind`, `Cyclone`, `Mix`, `Box`, `Circle`, `Eraser`, `Sample`, `Sign`, `Property`, `Amb Heat +/-`) and the three-letter tool codes (`SET`, `SMPL`, `PGRV`, ...) stay verbatim in every language, matching TPT. Existing translated names MUST be reverted to the TPT form.

## Project Overview

Rust + `wgpu` + `CubeCL` + `egui-wgpu` + `mlua` port of The Powder Toy (`tpt-app`). GPU compute kernels replace the CPU cellular automaton; egui replaces the SDL UI.

```text
tpt-app  --> { tpt-core, tpt-sim, tpt-save, tpt-lua }
tpt-save --> { tpt-core, tpt-sim }
tpt-lua  --> { tpt-core, tpt-sim, tpt-save }
tpt-sim  --> { tpt-core }
tpt-core --> (pure data and constants; no GPU, windowing or I/O dependencies)
```

## Architecture Invariants

- **One GPU context, zero-copy viewport.** A single `wgpu::Device`/`Queue` is shared by CubeCL, `egui_wgpu` and the simulation. `sim_texture` is bound directly as an `egui::TextureId`, and the device render blits `gpu_framebuf` into it. Request `adapter.limits()` so kernels can bind up to 16 storage buffers.
- **Tick order** is defined by `SimulationEngine::step` and its comments (authoritative): air and ambient heat, thermal/phase transitions, element reactions, GoL, gravity, movement, rigid bodies, water equalisation, render. Every pass is device-resident; each module's docs describe its mechanics.
- **Coordinates.** Pixel space (`y * xres + x`) is used by particles, maps, `SimCommand`s and Lua particle APIs. Coarse `CELL = 4` space (`cy * xcells + cx`) is used by air, walls and `PlaceWalls`, whose cells are already divided by `CELL` (do not divide again). `y` grows downward; brush rows are emitted bottom-to-top so powders stack in one frame.
- **Host/GPU sync.** The host SoA is stale during interactive play. NEVER upload the whole particle pool from it; write only `host_dirty_slots` (a full upload is limited to a fresh or reallocated pool and a just-finished read-back). Device-created particles live in `GPU_SPAWN_START..GPU_SPAWN_END`; `used_slot_bound()` sizes slot-parallel launches. Host map edits upload only the touched pixel sub-ranges.
- **Creation and destruction.** `SimBuffers::create_part` (ports `Simulation::create_part`) and `kill_particle` are the only host-side paths. Canvas edits that only modify existing particles run on the device (`edit_kernel.rs`, `flood_kernel.rs`). Device spawns go through the creation queue; passes without room record `part_flags` bits expanded later in the same tick.
- **Portal-held slots are dead but not free.** `FLAG_PORTAL_HELD` marks them; NEVER return them to `free_list` or the device spawn window.
- **`pmap`/`photons`.** Matter particles occupy `pmap`; energy particles (`PHOT`, `NEUT`, `ELEC`) occupy `photons`; both are rebuilt on the device each frame. The `pmap` id field is 23 bits, so the pool never exceeds `0x7F_FFFF` slots. Use the Rust `TYPE_*`/`PROP_*`/`FLAG_*` constants; NEVER copy numeric values from the C++ headers.

## CubeCL Kernel Rules

- Mirror host constants as literals with a `const _: () = assert!(...)` guard; a kernel body cannot read a host constant.
- Route hash multiplications through a `#[cube] fn mix_index` helper; inline constants can trip `arithmetic_overflow` in the host build.
- `GpuElementProps` is 32 `u32` words; recover floats with `f32::reinterpret`.
- Seed runtime-assigned `let mut` values from runtime expressions (`seed_i32`/`seed_u32`), never from a host literal.
- Use `terminate!()` instead of `return`; `continue` is rejected (invert it into a guard); `break` is fine. Use `clampf`/`iabs`, not `.abs()`/`.clamp()`/`.max()`.
- NEVER use `wrapping_*`; integer arithmetic is already modular. Write `(!x) + 1u32` for wrapping negation.
- Read an atomic without storing via `fetch_max(0u32)`.
- Stay within 8 storage buffers where possible; 16 is the hard ceiling.
- NEVER assign to a mutable local declared outside a nested loop; accumulate into the thread's own buffer word instead.
- The `x * 0` runtime-seed idiom needs `#[allow(clippy::erasing_op)]` and a comment; a literal brings the mis-expansion back.
- Host mirrors of kernel formulas MUST be `#[cfg(test)]`-gated.

## Rigid Body Engine

- Bodies own `RigidPixel`s that also exist in the pool with `FLAG_RIGID_PIXEL | FLAG_SKIPMOVE`. The steady state is device-only; the host re-initialises a body only when its description changes.
- Treat `(other_props & TYPE_SOLID != 0) || (flags & FLAG_RIGID_PIXEL != 0)` as a hard contact. Loose powder is a hard support only below its yield strength (`unit_weight * radius`).
- Contacts are symmetric and fixed-point: queue the receiver's equal-and-opposite share with `fetch_add` and fold it in after every sender finishes, so scheduling cannot change the total.
- The rasteriser never places two rigid pixels in one cell; arbitration is deterministic (stationary beats mover, then lowest global pixel index).
- Fluid cells give buoyancy and drag, not rigid support; the rasteriser displaces free particles it replaces.
- `sync_rigid_bodies_to_host()` is the only host-transform refresh; save capture calls it explicitly.

## Lua Bridge

- `LuaScriptingEngine` shares `Rc<RefCell<LuaBridgeState>>` with `'static` closures; `run_attached` installs the engine through an RAII guard and rejects re-entry.
- Built-in elements are `0..195`; custom elements are `195..PT_NUM`. `elem.DEFAULT_PT_*` covers all 195 TPT names.
- `sim.signs` is a live proxy, never a copy: `#`, indexing, `new` and `delete` mutate `SimulationEngine::signs`.
- `ren` toggles are `SimulationConfig` fields shared by scripts and UI; `separateThread` and `depth3d` stay documented no-ops.

## UI Text, i18n and Comments

- Every localizable user-visible string lives in `crates/tpt-app/locales/<language>/<module>.json` and is read via `i18n::t("module.key")` (`t_fmt` for placeholders). A new module needs both language files and a row in `i18n::MODULES`.
- Not localized: element/wall/tool names (see Hard Constraints), Lua API names, `.tptprefs` keys and CLI flags.
- Comments explain why, cite the C++ source, and never restate the next line.

## Error Handling and Window Lifecycle

- Library crates return `Result<T, String>` with descriptive messages.
- Kernel drivers validate slice lengths before `launch_unchecked`.
- In `app_window.rs`, process `textures_delta.set` and extract/clear `textures_delta.free` before any early return on `surface.get_current_texture()`.
- Poll unbounded before `configure`; a first-frame shader compile can exceed the idle timeout.

## Key Directories

- `crates/tpt-core/src/`: pure data: constants, element registry, walls, commands, signs, voxel model.
- `crates/tpt-sim/src/`: `SimBuffers` host/GPU state, all kernels, gravity FFT, rigid solver, water equalisation, canvas-edit kernels, `SimulationEngine` and `engine_commands/`. `reaction_kernel/` is one module per pass; large passes are directories of focused kernels, each file around 700 lines or less.
- `crates/tpt-save/src/`: `.tptrs` serializer and undo history.
- `crates/tpt-lua/src/`: bridge and `sim`/`elem`/`ren`/`evt`/`tpt` APIs.
- `crates/tpt-app/src/`: winit/egui application, `app_state/`, `ui_panels/`, `render3d/`, `brushes.rs`, `i18n.rs`, `prefs/`.
- 3D sandbox: `tpt-core/src/voxel3d/`, `tpt-sim/src/voxel3d/`, `tpt-app/src/render3d/`; the same device-resident rules apply.

## Commands

```bash
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets
cargo build --release -p tpt-app
cargo run --release -p tpt-app
cargo run --release -p tpt-app -- --headless-smoke
cargo run --release -p tpt-app -- --windowed-smoke 30
cargo run --release -p tpt-app -- --voxel-windowed-smoke 5
cargo run --release -p tpt-app -- --bench
cargo test --workspace --release
cargo test -p tpt-sim --test gpu_sim_tests
cargo test -p tpt-app --bins
```

Use release builds for GPU runs. A local build produces the Windows binary only.

Pushing any tag runs `.github/workflows/release.yml`: it builds
`x86_64-pc-windows-msvc` (static CRT), `x86_64-unknown-linux-gnu` and
`aarch64-apple-darwin`, smoke-tests the Linux binary on a software Vulkan device,
and publishes `tpt-app-<tag>-<platform>` archives on a GitHub release.

## Testing

- GPU integration: `crates/tpt-sim/tests/gpu_sim_tests.rs`, serialised by `GPU_LOCK`; covers movement, thermal/phase, reactions, rendering, rigid contacts, save, Lua and dispatch limits.
- Save: `crates/tpt-save/tests/save_tests.rs`. Lua: `crates/tpt-lua/tests/lua_tests.rs`. 3D core: `crates/tpt-sim/tests/gpu_voxel3d_core_tests.rs`.
- `tpt-app` is binary-only: use `cargo test -p tpt-app --bins`.
- Smoke paths: `--headless-smoke`, `--windowed-smoke`, `--voxel-windowed-smoke`, `--bench`.

## Runtime and Tooling

- Toolchain: edition 2024, `rust-version = "1.95"`. No `rust-toolchain.toml` is committed, so a checkout builds with the host's default toolchain. This workstation's default is `stable-x86_64-pc-windows-gnu` and keeps its MinGW-w64 linker paths in `~/.cargo/config.toml`, never in the repository; CI builds `x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu` and `aarch64-apple-darwin` (`.github/workflows/release.yml`). Do not switch this workstation to `-msvc`: it has no Visual Studio Build Tools (`mlua` vendors Lua 5.4 and needs `cc`).
- NEVER remove `-Wl,--no-gc-sections` from `.cargo/config.toml`: on `x86_64-pc-windows-gnu` GNU `ld` garbage-collects `linkme`'s kernel registry and every launch panics. MSVC (`/INCLUDE:` directives) and the ELF/Mach-O targets keep the registry without the flag, which is why CI does not carry it.
- Keep `[profile.release]` at `lto = false` and `codegen-units = 16`; whole-program LTO and a single codegen unit make the CubeCL-heavy build disproportionately slow.
- Workspace `cubecl*` dependencies use `default-features = false`; their defaults enable `persistence` and build the libSQL fork.
- Initialize devices with `required_limits: adapter.limits()`.
- Initialize CubeCL with `MemoryConfiguration::ExclusivePages`; the default allocator makes wgpu reject a buffer used as both indirect and storage.
