# The Powder Toy (Rust + wgpu + CubeCL)

[English](README.md) | 简体中文

> **开发提示：** 本项目仍处于开发中，可能存在未移植完整的原版内容以及性能问题，且不兼容原版存档。

基于 GPU 加速的 *The Powder Toy* Rust 移植版。物理模拟与特效渲染均以 CubeCL 计算着色器形式运行在 `wgpu` 上，桌面端采用 `winit` 与 `egui-wgpu` 构建。

## 项目简介

- **全流程 GPU 驻留管线**：欧拉流体气压/风速与环境温度、热传导与相变/压变、元素反应、无锁并行粒子移动、康威生命游戏（`LIFE`）、FFT 牛顿引力、水面均压与特效渲染均在同一个共享 `wgpu::Device`/`Queue` 上运行，渲染结果直接 blit 至 `egui` 纹理，无逐帧 CPU/GPU 往返拷贝。
- **完整元素与墙体体系**：移植原版 195 项内置元素表（`PT_NONE` 与 194 种内置元素）、19 种墙体以及传送门/无线电/活塞/火柴人机制。
- **Workspace 模块划分**：`tpt-core`（共享常量、元素与墙体定义、3D 体素模型）、`tpt-sim`（GPU 缓冲、CubeCL 内核、刚体、3D 体素引擎）、`tpt-save`（`.tptrs` / `.tpt3` 存档与撤销历史）与 `tpt-app`（桌面窗口、UI、偏好设置与冒烟/基准测试入口）。

## 相比原版 The Powder Toy 的主要更改

- **Rust + `wgpu` + CubeCL 引擎与 `egui` 界面**：将原版 C++/SDL 的 CPU 元胞自动机与 SDL 界面重写为基于 `wgpu`/`CubeCL` 的 GPU 计算管线和 `winit` + `egui-wgpu` 前端。
- **批处理无锁并行模拟**：元素反应按类型分批执行，并与无锁原子仲裁移动阶段分离，而非原版 C++ 的按槽位串行交错循环（相邻链式反应与液体环形交换存在一帧的固有延迟）。
- **可调画布尺寸**：除经典 `612x384` 外，提供最高 `1600x900` 预设及 `128..=2048` 自定义尺寸（按 4 px 网格对齐），不再固定于 `612x384`。
- **2D 像素簇刚体**：新增 `RigidBodies` 分类（`RigidBody2D`），在 GPU 上模拟可破坏像素簇刚体的多子步碰撞与浮力，受热或化学腐蚀破裂后可还原为自由粒子。
- **3D 体素沙盒模式（开发中）**：新增独立的 `256x256x256` GPU 体素沙盒，支持基于活跃 brick 的稀疏间接派发、`64^3` 气压流场、6 自由度体素刚体以及体渲染/DDA 光线追踪。
- **纯本地存档、APNG 录制与双语界面**：使用本地 zlib 压缩的 `.tptrs`（2D）/ `.tpt3`（3D）快照与本地 `./saves/`、`./stamps/` 目录（不兼容原版 `.cps` / `.stm` 格式及在线存档服务器）；录屏直接输出 `.apng` 而非逐帧 PPM 图片；新增中英文（`en` / `zh-CN`）界面本地化。

## 构建与运行

前置要求：Rust 1.99.1+、支持 `wgpu` 的图形驱动（Vulkan / DX12 / Metal）以及 C 编译工具链。

```bash
cargo run --release -p tpt-app
cargo run --release -p tpt-app -- --headless-smoke
cargo test --workspace --release
```
