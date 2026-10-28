//! Canvas screenshots and APNG recordings.
//!
//! Ports `GameView::TakeScreenshot` (`GameView.cpp:943-1005`) and
//! `GameView::Record` (`GameView.cpp:1013-1029`, `GameView.cpp:2310-2317`,
//! `GameView.cpp:2340-2347`), with two deliberate differences:
//!
//! * the reference writes one PPM per recorded frame into
//!   `recordings/<unix time>/`; this port encodes an animated PNG
//!   (`recordings/recording_<n>.apng`) when recording stops, as required by
//!   the port's `tpt.record("start" | "stop" | path)` contract;
//! * screenshots are named `screenshots/screenshot_<n>.png` with the lowest
//!   free index (`GameView.cpp:1449-1462`).
//!
//! # Data flow
//!
//! The UI layer (`ui_panels`) and the Lua runtime have no access to the wgpu
//! device, while [`crate::gpu_ctx::GpuContext`] — the only owner of
//! `sim_texture` — does not receive `AppState`. The two are connected by the
//! small process-wide mailbox in this module:
//!
//! * `ui_panels` calls [`request_screenshot`] from the `P`/`F2` shortcut;
//! * [`CapturePump::after_frame`], invoked from
//!   [`crate::gpu_ctx::GpuContext::encode_sim_blit`], reads `sim_texture`
//!   back into a staging buffer once per rendered frame, writes the PNG
//!   (screenshots) or hands the pixels to the recorder thread (recordings),
//!   and pushes a [`CaptureOutcome`];
//! * `AppState::poll_screenshot_events` drains the outcomes into the HUD
//!   status line.
//!
//! The render loop only ever copies device memory; PNG/DEFLATE encoding runs
//! on the recorder thread, and at most [`MAX_PENDING_FRAMES`] read-backs are
//! in flight, so recording can never stall presentation.

use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

/// Directory (relative to the process working directory) holding screenshots.
///
/// `GameView::TakeScreenshot` writes to the working directory; the port keeps
/// captures in their own folder so `Client.cpp`'s screenshot scan
/// (`Client.cpp:569`) has a single location to look at.
pub const SCREENSHOTS_DIR: &str = "screenshots";
/// Directory holding recorded APNG files.
pub const RECORDINGS_DIR: &str = "recordings";
/// File stem of a screenshot: `screenshot_<n>.png`.
pub const SCREENSHOT_PREFIX: &str = "screenshot_";
/// Extension of a screenshot.
pub const SCREENSHOT_EXTENSION: &str = "png";
/// File stem of a recording: `recording_<n>.apng`.
pub const RECORDING_PREFIX: &str = "recording_";
/// Extension of a recording.
pub const RECORDING_EXTENSION: &str = "apng";

/// Lua global drained by [`AppState::poll_screenshot_events`].
///
/// `tpt.record` lives in the `tpt-lua` crate, which cannot name
/// `tpt-app` types (that would be a dependency cycle). It therefore stores
/// the request in a Lua global that the application polls once per UI frame.
///
/// The value is either `"stop"`, `"start"` or
/// `"start" + RECORD_REQUEST_SEPARATOR + <output path>`.
///
/// [`AppState::poll_screenshot_events`]: crate::app_state::AppState::poll_screenshot_events
pub const RECORD_REQUEST_GLOBAL: &str = "__tpt_record_request";
/// Separator between the `"start"` verb and the requested output path.
pub const RECORD_REQUEST_SEPARATOR: char = '\u{1}';

/// Default delay of a single-frame recording, in milliseconds.
const DEFAULT_FRAME_DELAY_MS: u16 = 100;
/// Upper bound on compressed frame data kept for one recording.
///
/// A long recording is assembled in memory when it stops, so the worker caps
/// what it retains instead of letting a forgotten recording exhaust the
/// process. Compressed frames are small (`png`'s fast deflate on a sparse
/// canvas is tens of kilobytes per frame), so 512 MiB holds hours of capture.
const MAX_RECORDING_BYTES: usize = 512 * 1024 * 1024;

/// Maximum number of read-backs queued while recording.
///
/// Recording captures at most one frame per rendered frame; when the GPU is
/// behind, frames are dropped instead of growing an unbounded queue.
const MAX_PENDING_FRAMES: usize = 3;
/// IDAT/fdAT payload size. Well below the PNG chunk limit and large enough to
/// avoid per-chunk allocator traffic.
const PNG_CHUNK_PAYLOAD: usize = 64 * 1024;
/// PNG file signature (`\x89PNG\r\n\x1a\n`).
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

/// Returns `dir/<prefix><n>.<extension>` for the lowest `n` not on disk.
///
/// `n` starts at 1, matching the reference's `screenshotIndex(1)`
/// (`GameView.cpp:198`) and `Format::Width(recordingIndex++, 6)`
/// (`GameView.cpp:2314`). Scanning the directory is what makes a screenshot
/// taken after the previous files were deleted reuse a low index instead of
/// accumulating ever-growing numbers.
pub fn next_free_path(dir: &Path, prefix: &str, extension: &str) -> PathBuf {
    let mut index = 1u32;
    loop {
        let candidate = dir.join(format!("{prefix}{index}.{extension}"));
        if !candidate.exists() {
            return candidate;
        }
        index = index.saturating_add(1);
    }
}

/// Lowest free `screenshots/screenshot_<n>.png` in `dir`.
pub fn next_screenshot_path(dir: &Path) -> PathBuf {
    next_free_path(dir, SCREENSHOT_PREFIX, SCREENSHOT_EXTENSION)
}

/// Lowest free `recording_<n>.apng` in `dir`.
pub fn next_recording_path(dir: &Path) -> PathBuf {
    next_free_path(dir, RECORDING_PREFIX, RECORDING_EXTENSION)
}

/// Creates `dir` and any missing parents (`Platform::MakeDirectory`).
///
/// Screenshot and recording directories are created on demand, exactly like
/// `GameView::Record` creates `recordings/` before the first frame
/// (`GameView.cpp:1024-1025`).
pub fn ensure_dir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|err| format!("could not create '{}': {err}", dir.display()))
}

/// Encodes tightly packed RGBA8 pixels as a PNG file at `path`.
pub fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let expected = (width as usize) * (height as usize) * 4;
    if rgba.len() != expected {
        return Err(format!(
            "pixel buffer has {} bytes, expected {expected}",
            rgba.len()
        ));
    }
    let file = std::fs::File::create(path)
        .map_err(|err| format!("could not create '{}': {err}", path.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|err| format!("could not write PNG header: {err}"))?;
    writer
        .write_image_data(rgba)
        .map_err(|err| format!("could not write PNG data: {err}"))?;
    writer
        .finish()
        .map_err(|err| format!("could not finish PNG: {err}"))
}

/// Result of one capture, consumed by the HUD status line.
#[derive(Debug)]
pub enum CaptureOutcome {
    /// A one-off screenshot (`P`/`F2`).
    Screenshot(Result<PathBuf, String>),
    /// A recording written when `tpt.record("stop")` finished the APNG.
    Recording(Result<PathBuf, String>),
}

/// Process-wide request/result mailbox shared by the UI and the GPU context.
#[derive(Default)]
struct Mailbox {
    /// A bare `P`/`F2` is waiting for the next rendered frame.
    screenshot_requested: bool,
    /// Completed captures awaiting a HUD status line.
    outcomes: VecDeque<CaptureOutcome>,
}

fn mailbox() -> &'static Mutex<Mailbox> {
    static MAILBOX: LazyLock<Mutex<Mailbox>> = LazyLock::new(|| Mutex::new(Mailbox::default()));
    &MAILBOX
}

fn lock_mailbox() -> std::sync::MutexGuard<'static, Mailbox> {
    mailbox()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Requests one screenshot of the simulation canvas.
pub fn request_screenshot() {
    lock_mailbox().screenshot_requested = true;
}

/// Pushes a completed capture for [`crate::app_state::AppState`] to display.
fn push_outcome(outcome: CaptureOutcome) {
    lock_mailbox().outcomes.push_back(outcome);
}

/// Takes the next completed capture, if any.
pub fn take_outcome() -> Option<CaptureOutcome> {
    lock_mailbox().outcomes.pop_front()
}

fn take_screenshot_request() -> bool {
    std::mem::take(&mut lock_mailbox().screenshot_requested)
}

/// Active recorder shared by the GPU pump and the Lua-request drainer.
struct Recorder {
    /// Channel into the APNG worker thread.
    sender: SyncSender<RecordMsg>,
    /// File the worker writes when the recording stops.
    output: PathBuf,
}

fn recorder() -> &'static Mutex<Option<Recorder>> {
    static RECORDER: LazyLock<Mutex<Option<Recorder>>> = LazyLock::new(|| Mutex::new(None));
    &RECORDER
}

fn lock_recorder() -> std::sync::MutexGuard<'static, Option<Recorder>> {
    recorder()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Whether an APNG recording is currently collecting frames.
pub fn is_recording() -> bool {
    lock_recorder().is_some()
}

/// Starts recording rendered frames into `requested` (or the lowest free
/// `recordings/recording_<n>.apng`).
///
/// Returns the output path. Calling this while a recording is already active
/// keeps the current recording and returns its path, mirroring
/// `GameView::Record(true)` treating a repeated start as a no-op
/// (`GameView.cpp:1020`).
pub fn start_recording(requested: Option<&Path>) -> Result<PathBuf, String> {
    let mut recorder = lock_recorder();
    if let Some(active) = recorder.as_ref() {
        return Ok(active.output.clone());
    }

    let output = match requested {
        Some(path) => {
            let mut path = path.to_path_buf();
            if path.extension().is_none() {
                path.set_extension(RECORDING_EXTENSION);
            }
            path
        }
        None => next_recording_path(Path::new(RECORDINGS_DIR)),
    };
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        ensure_dir(parent)?;
    }

    // Bounded so a recorder thread that falls behind cannot grow an unbounded
    // queue of raw frames; `push_recorded_frame` drops frames instead.
    let (sender, receiver) = mpsc::sync_channel(MAX_PENDING_FRAMES + 1);
    let worker_output = output.clone();
    std::thread::Builder::new()
        .name("tpt-apng-recorder".to_string())
        .spawn(move || run_recorder(receiver, worker_output))
        .map_err(|err| format!("could not start the recording thread: {err}"))?;

    *recorder = Some(Recorder {
        sender,
        output: output.clone(),
    });
    Ok(output)
}

/// Stops the active recording; the APNG is assembled and written on the
/// recorder thread, which reports completion through [`take_outcome`].
///
/// Returns the output path so the caller can report it immediately. An error
/// means no recording was active.
pub fn stop_recording() -> Result<PathBuf, String> {
    let mut recorder = lock_recorder();
    let Some(active) = recorder.take() else {
        return Err(String::from("no recording is in progress"));
    };
    // A closed or full channel also makes the worker finish (a disconnected
    // sender ends its receive loop), so a failed send is not an error: the
    // worker already has every queued frame.
    let _ = active.sender.try_send(RecordMsg::Stop);
    Ok(active.output)
}

/// A frame handed from the GPU pump to the recorder thread.
enum RecordMsg {
    Frame {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        captured_at: Instant,
    },
    Stop,
}

/// Copies a frame to the active recorder, if any.
fn push_recorded_frame(rgba: Vec<u8>, width: u32, height: u32, captured_at: Instant) {
    let recorder = lock_recorder();
    let Some(active) = recorder.as_ref() else {
        return;
    };
    // A closed or full channel means no frame can be queued; dropping the
    // frame keeps the render thread free instead of waiting for the encoder.
    let _ = active.sender.try_send(RecordMsg::Frame {
        rgba,
        width,
        height,
        captured_at,
    });
}

/// One compressed frame kept by the recorder until the animation is written.
struct EncodedFrame {
    /// Concatenated IDAT payload of the standalone PNG for this frame.
    idat: Vec<u8>,
    /// Time the frame was captured, used to derive the APNG frame delay.
    captured_at: Instant,
}

/// Worker body: compress each frame as it arrives, then assemble the APNG.
fn run_recorder(receiver: Receiver<RecordMsg>, output: PathBuf) {
    let mut frames: Vec<EncodedFrame> = Vec::new();
    let mut size: Option<(u32, u32)> = None;
    let mut ihdr: Option<Vec<u8>> = None;
    let mut last_error: Option<String> = None;
    let mut stored_bytes = 0usize;
    let mut limit_hit = false;

    while let Ok(msg) = receiver.recv() {
        match msg {
            RecordMsg::Frame {
                rgba,
                width,
                height,
                captured_at,
            } => {
                if limit_hit {
                    continue;
                }
                match size {
                    Some(current) if current != (width, height) => {
                        // The canvas was resized mid-recording; mixing sizes in
                        // one APNG is invalid, so drop the odd frame.
                        continue;
                    }
                    None => {
                        size = Some((width, height));
                    }
                    Some(_) => {}
                }
                match encode_compressed_frame(width, height, &rgba) {
                    Ok((frame_ihdr, idat)) => {
                        if stored_bytes.saturating_add(idat.len()) > MAX_RECORDING_BYTES {
                            limit_hit = true;
                            continue;
                        }
                        stored_bytes += idat.len();
                        if ihdr.is_none() {
                            ihdr = Some(frame_ihdr);
                        }
                        frames.push(EncodedFrame { idat, captured_at });
                    }
                    Err(err) => last_error = Some(err),
                }
            }
            RecordMsg::Stop => break,
        }
    }

    let outcome = if limit_hit {
        Err(format!(
            "recording exceeded the {MAX_RECORDING_BYTES}-byte limit; stop it earlier"
        ))
    } else {
        match (size, ihdr) {
            (Some((width, height)), Some(ihdr)) if !frames.is_empty() => {
                match write_apng(&output, width, height, &ihdr, &frames) {
                    Ok(()) => Ok(output),
                    Err(err) => Err(err),
                }
            }
            _ => Err(last_error.unwrap_or_else(|| String::from("no frames were captured"))),
        }
    };
    push_outcome(CaptureOutcome::Recording(outcome));
}

/// Compresses one RGBA frame and returns its IHDR payload plus IDAT payload.
fn encode_compressed_frame(
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut png = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        // `GameView` records at the render rate; the ultra-fast deflate mode
        // keeps the recorder thread far ahead of the 240 fps present ceiling.
        encoder.set_deflate_compression(png::DeflateCompression::FdeflateUltraFast);
        let mut writer = encoder
            .write_header()
            .map_err(|err| format!("could not write frame header: {err}"))?;
        writer
            .write_image_data(rgba)
            .map_err(|err| format!("could not write frame data: {err}"))?;
        writer
            .finish()
            .map_err(|err| format!("could not finish frame: {err}"))?;
    }
    split_png(&png)
}

/// Splits a standalone PNG into its IHDR payload and concatenated IDAT payload.
fn split_png(png: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    if !png.starts_with(&PNG_SIGNATURE) {
        return Err(String::from("frame is not a PNG"));
    }
    let mut offset = PNG_SIGNATURE.len();
    let mut ihdr = None;
    let mut idat = Vec::new();
    while offset + 12 <= png.len() {
        let length = u32::from_be_bytes([
            png[offset],
            png[offset + 1],
            png[offset + 2],
            png[offset + 3],
        ]) as usize;
        let end = offset + 12 + length;
        if end > png.len() {
            return Err(String::from("truncated PNG frame"));
        }
        let kind = &png[offset + 4..offset + 8];
        let data = &png[offset + 8..offset + 8 + length];
        match kind {
            b"IHDR" => ihdr = Some(data.to_vec()),
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        offset = end;
    }
    match (ihdr, idat.is_empty()) {
        (Some(ihdr), false) => Ok((ihdr, idat)),
        _ => Err(String::from("frame is missing IHDR or IDAT")),
    }
}

/// Writes an APNG containing every recorded frame.
///
/// The chunk layout follows the APNG specification: `IHDR`, `acTL`, then for
/// each frame an `fcTL` followed by its compressed data (`IDAT` for the first
/// frame because it doubles as the default image, `fdAT` afterwards).
fn write_apng(
    path: &Path,
    width: u32,
    height: u32,
    ihdr: &[u8],
    frames: &[EncodedFrame],
) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|err| format!("could not create '{}': {err}", path.display()))?;
    let mut out = BufWriter::new(file);
    let io_error = |err: std::io::Error| format!("could not write '{}': {err}", path.display());

    out.write_all(&PNG_SIGNATURE).map_err(io_error)?;
    write_chunk(&mut out, b"IHDR", ihdr).map_err(io_error)?;

    let mut animation = [0u8; 8];
    animation[..4].copy_from_slice(&(frames.len() as u32).to_be_bytes());
    // 0 = loop forever, matching how the reference recording replays frames.
    animation[4..].copy_from_slice(&0u32.to_be_bytes());
    write_chunk(&mut out, b"acTL", &animation).map_err(io_error)?;

    let delays = frame_delays(frames);
    let mut sequence = 0u32;
    for (index, frame) in frames.iter().enumerate() {
        let (delay_num, delay_den) = delays[index];
        write_chunk(
            &mut out,
            b"fcTL",
            &frame_control(sequence, width, height, delay_num, delay_den),
        )
        .map_err(io_error)?;
        sequence = sequence.wrapping_add(1);

        if index == 0 {
            for chunk in frame.idat.chunks(PNG_CHUNK_PAYLOAD) {
                write_chunk(&mut out, b"IDAT", chunk).map_err(io_error)?;
            }
        } else {
            for chunk in frame.idat.chunks(PNG_CHUNK_PAYLOAD) {
                let mut data = Vec::with_capacity(4 + chunk.len());
                data.extend_from_slice(&sequence.to_be_bytes());
                data.extend_from_slice(chunk);
                write_chunk(&mut out, b"fdAT", &data).map_err(io_error)?;
                sequence = sequence.wrapping_add(1);
            }
        }
    }

    write_chunk(&mut out, b"IEND", &[]).map_err(io_error)?;
    out.flush().map_err(io_error)
}

/// Display duration of every frame, derived from the capture timestamps.
fn frame_delays(frames: &[EncodedFrame]) -> Vec<(u16, u16)> {
    let mut delays = Vec::with_capacity(frames.len());
    let mut previous = DEFAULT_FRAME_DELAY_MS;
    for index in 0..frames.len() {
        let millis = if index + 1 < frames.len() {
            frames[index + 1]
                .captured_at
                .saturating_duration_since(frames[index].captured_at)
                .as_millis()
        } else {
            u128::from(previous)
        };
        let millis = millis.clamp(1, u16::MAX as u128) as u16;
        delays.push((millis, 1000));
        previous = millis;
    }
    delays
}

/// 26-byte `fcTL` payload: sequence, size, position, delay, dispose, blend.
fn frame_control(
    sequence: u32,
    width: u32,
    height: u32,
    delay_num: u16,
    delay_den: u16,
) -> [u8; 26] {
    let mut data = [0u8; 26];
    data[0..4].copy_from_slice(&sequence.to_be_bytes());
    data[4..8].copy_from_slice(&width.to_be_bytes());
    data[8..12].copy_from_slice(&height.to_be_bytes());
    // x_offset / y_offset stay 0: every frame is the full canvas.
    data[20..22].copy_from_slice(&delay_num.to_be_bytes());
    data[22..24].copy_from_slice(&delay_den.to_be_bytes());
    // dispose_op = APNG_DISPOSE_OP_NONE (0), blend_op = APNG_BLEND_OP_SOURCE (0).
    data
}

/// Writes one length/type/data/CRC chunk.
fn write_chunk<W: Write>(out: &mut W, kind: &[u8; 4], data: &[u8]) -> std::io::Result<()> {
    out.write_all(&(data.len() as u32).to_be_bytes())?;
    out.write_all(kind)?;
    out.write_all(data)?;
    out.write_all(&crc32_parts(&[kind.as_slice(), data]).to_be_bytes())
}

/// IEEE CRC-32 (the polynomial PNG uses) over the concatenation of `parts`.
fn crc32_parts(parts: &[&[u8]]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for part in parts {
        for &byte in *part {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
    }
    !crc
}

/// GPU read-back with at most one frame-per-rendered-frame and bounded
/// in-flight work.
///
/// Owned by [`crate::gpu_ctx::GpuContext`] and driven from
/// `encode_sim_blit`; see the module docs for the full data flow.
#[derive(Default)]
pub struct CapturePump {
    /// Read-backs whose `map_async` callback has not fired yet.
    pending: Vec<PendingFrame>,
    /// Staging buffers recycled after their mapping is released.
    free: Vec<wgpu::Buffer>,
    /// Size of every buffer in [`Self::free`].
    buffer_bytes: u64,
    /// Set by [`Self::begin_frame`]; cleared by the first
    /// [`Self::after_frame`] of that frame.
    ///
    /// `present_frame` calls `set_sim_filter` exactly once per presented
    /// frame, while `encode_sim_blit` also runs from the canvas-resize helper.
    /// The token makes the pump capture at most one frame per rendered frame
    /// and keeps the resize helper's blank, freshly allocated texture out of
    /// recordings and screenshots.
    frame_open: bool,
}

/// One asynchronous `sim_texture` -> staging buffer read-back.
struct PendingFrame {
    buffer: wgpu::Buffer,
    receiver: Receiver<Result<(), wgpu::BufferAsyncError>>,
    width: u32,
    height: u32,
    bytes_per_row: u32,
    captured_at: Instant,
}

impl CapturePump {
    /// Marks the start of a presented frame; the first [`Self::after_frame`]
    /// afterwards is the only one that captures.
    ///
    /// Called from `GpuContext::set_sim_filter`, which `present_frame` invokes
    /// once per presented frame before encoding any command buffer.
    pub fn begin_frame(&mut self) {
        self.frame_open = true;
    }

    /// Drops a token left open by a frame that never reached
    /// `encode_sim_blit` (for example the 3D sandbox).
    ///
    /// `GpuContext::resize_sim_texture` calls this before the caller's
    /// resize blit, so that blit cannot capture a blank, freshly allocated
    /// texture even when the previous frame was not a 2D frame.
    pub fn cancel_frame(&mut self) {
        self.frame_open = false;
    }

    /// Polls completed read-backs, then serves the pending screenshot request
    /// and records one frame if a recording is active.
    ///
    /// Called from `GpuContext::encode_sim_blit`. Only the first call after
    /// [`Self::begin_frame`] does anything, so a frame that also resizes the
    /// canvas still records exactly one (non-blank) frame.
    /// Nothing here waits for the GPU: recording read-backs are mapped
    /// asynchronously, and only a user-requested screenshot (`P`/`F2`) pays
    /// for a single `poll(Wait)`, exactly like the reference's synchronous
    /// `VideoBuffer::ToPNG` + `WriteFile` pair (`GameView.cpp:1001-1004`).
    pub fn after_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        (width, height): (u32, u32),
    ) {
        if !std::mem::take(&mut self.frame_open) || width == 0 || height == 0 {
            return;
        }
        self.poll(device);

        if take_screenshot_request() {
            self.capture_screenshot(device, queue, texture, width, height);
        }
        if is_recording() && self.pending.len() < MAX_PENDING_FRAMES {
            self.encode_recording_frame(device, queue, texture, width, height);
        }
    }

    /// Drains finished read-backs without blocking.
    fn poll(&mut self, device: &wgpu::Device) {
        if self.pending.is_empty() {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let mut index = 0;
        while index < self.pending.len() {
            match self.pending[index].receiver.try_recv() {
                Ok(result) => {
                    let frame = self.pending.swap_remove(index);
                    self.finish_frame(frame, result);
                }
                Err(TryRecvError::Empty) => index += 1,
                Err(TryRecvError::Disconnected) => {
                    let frame = self.pending.swap_remove(index);
                    self.recycle(frame.buffer);
                }
            }
        }
    }

    /// Copies a completed read-back into the recorder (or drops it).
    fn finish_frame(&mut self, frame: PendingFrame, result: Result<(), wgpu::BufferAsyncError>) {
        if result.is_ok()
            && let Ok(view) = frame.buffer.slice(..).get_mapped_range()
        {
            let rgba = strip_row_padding(&view, frame.width, frame.height, frame.bytes_per_row);
            drop(view);
            frame.buffer.unmap();
            push_recorded_frame(rgba, frame.width, frame.height, frame.captured_at);
            self.recycle(frame.buffer);
            return;
        }
        frame.buffer.unmap();
        self.recycle(frame.buffer);
    }

    /// Encodes an asynchronous copy of `texture` for the recorder.
    fn encode_recording_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) {
        let bytes_per_row = padded_bytes_per_row(width);
        let buffer = self.take_buffer(device, width, height);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tpt-record-readback"),
        });
        encode_texture_copy(&mut encoder, texture, &buffer, width, height, bytes_per_row);
        queue.submit(Some(encoder.finish()));

        let (sender, receiver) = mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.pending.push(PendingFrame {
            buffer,
            receiver,
            width,
            height,
            bytes_per_row,
            captured_at: Instant::now(),
        });
    }

    /// Reads `texture` back immediately and writes the requested PNG.
    fn capture_screenshot(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) {
        let path = next_screenshot_path(Path::new(SCREENSHOTS_DIR));
        let result = (|| -> Result<(), String> {
            ensure_dir(Path::new(SCREENSHOTS_DIR))?;
            let rgba = read_texture_blocking(device, queue, texture, width, height)?;
            write_png(&path, width, height, &rgba)
        })();
        push_outcome(CaptureOutcome::Screenshot(result.map(|()| path)));
    }

    /// Returns a staging buffer of the size required for `width` x `height`.
    fn take_buffer(&mut self, device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
        let size = u64::from(padded_bytes_per_row(width)) * u64::from(height);
        if size != self.buffer_bytes {
            self.free.clear();
            self.buffer_bytes = size;
        }
        self.free.pop().unwrap_or_else(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tpt-capture-readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
    }

    /// Returns an unmapped staging buffer to the pool.
    fn recycle(&mut self, buffer: wgpu::Buffer) {
        if self.free.len() < MAX_PENDING_FRAMES + 1 {
            self.free.push(buffer);
        }
    }
}

/// Reads a texture back to tightly packed RGBA8 bytes, blocking until done.
fn read_texture_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let bytes_per_row = padded_bytes_per_row(width);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tpt-screenshot-readback"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tpt-screenshot-readback"),
    });
    encode_texture_copy(&mut encoder, texture, &buffer, width, height, bytes_per_row);
    queue.submit(Some(encoder.finish()));

    let (sender, receiver) = mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    if let Err(err) = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    }) {
        return Err(format!("GPU wait for the screenshot failed: {err}"));
    }
    let result = receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .map_err(|err| format!("GPU read-back did not complete: {err}"))?;
    result.map_err(|err| format!("GPU read-back failed: {err:?}"))?;

    let view = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|err| format!("GPU read-back view failed: {err}"))?;
    let rgba = strip_row_padding(&view, width, height, bytes_per_row);
    drop(view);
    buffer.unmap();
    Ok(rgba)
}

/// Copies a texture into a `COPY_DST | MAP_READ` staging buffer.
fn encode_texture_copy(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    bytes_per_row: u32,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

/// `width * 4` rounded up to wgpu's 256-byte copy-row alignment.
fn padded_bytes_per_row(width: u32) -> u32 {
    let row = width.saturating_mul(4);
    row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        .saturating_mul(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

/// Copies the visible rows of a padded mapped range into one tight buffer.
fn strip_row_padding(mapped: &[u8], width: u32, height: u32, bytes_per_row: u32) -> Vec<u8> {
    let row_bytes = (width as usize) * 4;
    let stride = bytes_per_row as usize;
    let mut rgba = vec![0u8; row_bytes * height as usize];
    for row in 0..height as usize {
        let source = row * stride;
        if source + row_bytes > mapped.len() {
            break;
        }
        rgba[row * row_bytes..(row + 1) * row_bytes]
            .copy_from_slice(&mapped[source..source + row_bytes]);
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Creates a unique empty directory for one test and removes it on drop.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let path = std::env::temp_dir().join(format!(
                "tpt-screenshots-{name}-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create test dir");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn screenshot_index_is_the_lowest_free_number() {
        let dir = TestDir::new("index");
        assert_eq!(
            next_screenshot_path(&dir.0).file_name().unwrap(),
            "screenshot_1.png"
        );

        std::fs::write(dir.0.join("screenshot_1.png"), b"one").unwrap();
        std::fs::write(dir.0.join("screenshot_3.png"), b"three").unwrap();
        assert_eq!(
            next_screenshot_path(&dir.0).file_name().unwrap(),
            "screenshot_2.png",
            "the gap at 2 must be filled before reusing 4"
        );

        std::fs::write(dir.0.join("screenshot_2.png"), b"two").unwrap();
        assert_eq!(
            next_screenshot_path(&dir.0).file_name().unwrap(),
            "screenshot_4.png"
        );
    }

    #[test]
    fn screenshot_directory_is_created_on_demand() {
        let root = TestDir::new("dir");
        let dir = root.0.join("screenshots");
        assert!(!dir.exists());

        ensure_dir(&dir).expect("first call creates the directory");
        assert!(dir.is_dir());

        // A second call is a no-op rather than an `AlreadyExists` error, so a
        // failed capture retried on the next frame cannot wedge screenshots.
        ensure_dir(&dir).expect("second call is a no-op");
    }

    #[test]
    fn recording_name_uses_the_same_free_index_rule() {
        let dir = TestDir::new("recording");
        assert_eq!(
            next_recording_path(&dir.0).file_name().unwrap(),
            "recording_1.apng"
        );
        std::fs::write(dir.0.join("recording_1.apng"), b"one").unwrap();
        assert_eq!(
            next_recording_path(&dir.0).file_name().unwrap(),
            "recording_2.apng"
        );
    }

    #[test]
    fn apng_round_trips_every_recorded_frame() {
        let dir = TestDir::new("apng");
        let path = dir.0.join("recording.apng");
        let (width, height) = (2u32, 2u32);
        let start = Instant::now();
        let mut frames = Vec::new();
        let mut ihdr = None;
        for (index, shade) in [0u8, 64, 200].into_iter().enumerate() {
            let rgba: Vec<u8> = (0..(width * height))
                .flat_map(|_| [shade, shade, shade, 0xFF])
                .collect();
            let (frame_ihdr, idat) =
                encode_compressed_frame(width, height, &rgba).expect("encode frame");
            if ihdr.is_none() {
                ihdr = Some(frame_ihdr);
            }
            frames.push(EncodedFrame {
                idat,
                captured_at: start + std::time::Duration::from_millis(16 * (index as u64 + 1)),
            });
        }

        write_apng(&path, width, height, ihdr.as_deref().unwrap(), &frames).expect("write APNG");

        let decoder = png::Decoder::new(std::io::BufReader::new(
            std::fs::File::open(&path).expect("open APNG"),
        ));
        let mut reader = decoder.read_info().expect("read APNG header");
        assert_eq!(
            reader
                .info()
                .animation_control
                .as_ref()
                .map(|actl| actl.num_frames),
            Some(frames.len() as u32),
            "acTL must describe every frame"
        );
        let mut buffer = vec![0u8; reader.output_buffer_size().expect("frame size")];
        for shade in [0u8, 64, 200] {
            let info = reader.next_frame(&mut buffer).expect("decode frame");
            assert_eq!((info.width, info.height), (width, height));
            assert!(
                buffer.iter().step_by(4).all(|&channel| channel == shade),
                "frame data must decode to the recorded pixels"
            );
        }
    }
}
