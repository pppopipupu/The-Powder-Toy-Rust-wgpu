//! Device-side live-particle count.
//!
//! The kernels create and kill particles without telling the host, so the
//! incrementally maintained [`SimBuffers::active_count`] drifts. The HUD needs
//! the real number, and reading `part_ty` back to count on the CPU moved up to
//! 6 MB per call on a 1024x768 canvas plus a 1.5 M iteration loop on the main
//! thread, every half second. The reduction runs on the device instead and the
//! host reads one word.

// Clippy's rewrite for these lints uses methods the device IR does not have
// (`is_multiple_of`, `clamp`, `Range::contains`, `saturating_sub`), and
// collapsing a nested `if` changes what the CubeCL macro expands, so the plain
// operators are deliberate.
#![allow(clippy::collapsible_if)]

use cubecl::prelude::*;

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, mpsc};
use std::task::{Context, Poll, Wake, Waker};

use crate::buffers::SimBuffers;

/// `PT_NONE` is the free-slot marker; mirrored as a literal because a `#[cube]`
/// kernel cannot read host constants.
const C_PT_NONE: u32 = 0;
const _: () = assert!(tpt_core::constants::PT_NONE == 0);

const CUBE_DIM: u32 = 256;

#[cube(launch_unchecked)]
fn reset_live_kernel(live: &mut [Atomic<u32>]) {
    if ABSOLUTE_POS == 0usize {
        live[0].store(0u32);
    }
}

#[cube(launch_unchecked)]
fn count_live_kernel(part_ty: &[u32], live: &mut [Atomic<u32>], bound: u32) {
    let idx = ABSOLUTE_POS as u32;
    if idx < bound {
        if part_ty[idx as usize] != C_PT_NONE {
            live[0].fetch_add(1u32);
        }
    }
}

/// Count the live slots in `0..used_slot_bound()` on the device and return the
/// result, or `None` when the read-back fails.
///
/// The count is exact for every slot the host already knows about plus the
/// device spawn window, which is the same range the old trimmed read-back
/// covered.
fn launch_live_count_kernels(bufs: &SimBuffers, bound: usize) {
    let cubes = (bound as u32).div_ceil(CUBE_DIM).max(1);
    unsafe {
        reset_live_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(bufs.gpu_live_count.clone(), 1),
        );
        count_live_kernel::launch_unchecked(
            &bufs.client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(CUBE_DIM),
            BufferArg::from_raw_parts(bufs.gpu_part_ty.clone(), bufs.capacity),
            BufferArg::from_raw_parts(bufs.gpu_live_count.clone(), 1),
            bound as u32,
        );
    }
}

struct ThreadWaker(std::thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

type BoxReadTask = Pin<Box<dyn Future<Output = ()> + Send>>;

static READBACK_TX: LazyLock<mpsc::Sender<BoxReadTask>> = LazyLock::new(|| {
    let (tx, rx) = mpsc::channel::<BoxReadTask>();
    let _ = std::thread::Builder::new()
        .name("tpt-gpu-readback".to_string())
        .spawn(move || {
            let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
            let mut cx = Context::from_waker(&waker);
            let mut active: Vec<BoxReadTask> = Vec::new();
            loop {
                if active.is_empty() {
                    let Ok(task) = rx.recv() else {
                        break;
                    };
                    active.push(task);
                }
                while let Ok(task) = rx.try_recv() {
                    active.push(task);
                }
                let mut i = 0;
                while i < active.len() {
                    if active[i].as_mut().poll(&mut cx).is_ready() {
                        drop(active.swap_remove(i));
                    } else {
                        i += 1;
                    }
                }
                if !active.is_empty() && rx.try_recv().is_err() {
                    std::thread::park();
                }
            }
        });
    tx
});

/// Non-blocking handle for an in-flight `Client::read_async` operation.
///
/// Always runs the underlying CubeCL future to completion on a background
/// thread even if the caller drops the handle early, so pooled CubeCL staging
/// buffers are never left mapped across `Queue::submit`.
pub struct PendingRead<T> {
    done: Arc<AtomicBool>,
    slot: Arc<Mutex<Option<T>>>,
}

impl<T: Send + 'static> PendingRead<T> {
    /// Kicks off `fut` immediately on the caller thread so its GPU copy is
    /// queued right away, then hands any pending wait to the readback worker.
    pub fn spawn(mut fut: Pin<Box<dyn Future<Output = T> + Send>>) -> Self {
        let mut cx = Context::from_waker(Waker::noop());
        if let Poll::Ready(val) = fut.as_mut().poll(&mut cx) {
            return Self {
                done: Arc::new(AtomicBool::new(true)),
                slot: Arc::new(Mutex::new(Some(val))),
            };
        }
        let done = Arc::new(AtomicBool::new(false));
        let slot = Arc::new(Mutex::new(None));
        let done_tx = done.clone();
        let slot_tx = slot.clone();
        let task: BoxReadTask = Box::pin(async move {
            let val = fut.await;
            if let Ok(mut guard) = slot_tx.lock() {
                *guard = Some(val);
            }
            done_tx.store(true, Ordering::Release);
        });
        let _ = READBACK_TX.send(task);
        Self { done, slot }
    }

    /// Returns `Some(result)` once the async readback has finished.
    pub fn try_take(&self) -> Option<T> {
        if !self.done.load(Ordering::Acquire) {
            return None;
        }
        self.slot.try_lock().ok()?.take()
    }

    /// Whether the async readback has completed.
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }
}

pub(crate) type PendingLiveCount = PendingRead<Option<usize>>;

pub fn refresh_live_count(bufs: &SimBuffers) -> Option<usize> {
    let bound = bufs.used_slot_bound().min(bufs.capacity);
    if bound == 0 {
        return Some(0);
    }
    launch_live_count_kernels(bufs, bound);
    let readback = bufs.client.read(vec![bufs.gpu_live_count.clone()]).ok()?;
    let raw: &[u8] = readback.first()?;
    let word = u32::from_le_bytes(raw.get(..4)?.try_into().ok()?);
    Some(word as usize)
}

/// Polls any in-flight live-count readback without blocking and refreshes
/// `bufs.active_count` if it finished.
pub fn poll_live_count(bufs: &mut SimBuffers) {
    let Some(pending) = bufs.pending_live_count.as_ref() else {
        return;
    };
    if let Some(result) = pending.try_take() {
        bufs.pending_live_count = None;
        if let Some(count) = result {
            bufs.active_count = count;
        }
    }
}

/// Launches a non-blocking live-count reduction if none is already in flight.
pub fn start_live_count_async(bufs: &mut SimBuffers) {
    poll_live_count(bufs);
    if bufs.pending_live_count.is_some() {
        return;
    }
    let bound = bufs.used_slot_bound().min(bufs.capacity);
    if bound == 0 {
        bufs.active_count = 0;
        return;
    }
    launch_live_count_kernels(bufs, bound);
    let client = bufs.client.clone();
    let handle = bufs.gpu_live_count.clone();
    let pending = PendingRead::spawn(Box::pin(async move {
        let readback = client.read_async(vec![handle]).await.ok()?;
        let raw: &[u8] = readback.first()?;
        let word = u32::from_le_bytes(raw.get(..4)?.try_into().ok()?);
        Some(word as usize)
    }));
    if let Some(result) = pending.try_take() {
        if let Some(count) = result {
            bufs.active_count = count;
        }
    } else {
        bufs.pending_live_count = Some(pending);
    }
}
