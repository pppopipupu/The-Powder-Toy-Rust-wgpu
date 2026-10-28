//! Batch 2: every remaining Powder Toy element definition.
//!
//! TPT ids `0..=195` (minus the reserved slot 146) are covered by the table:
//! the classic and batch 1 slots above keep their historical registry indices,
//! and the definitions appended here continue the dense slot numbering from
//! `PT_PLUT` onwards. Each entry names its TPT id and the `.cpp` it was
//! transcribed from, so the mapping to `ElementNumbers.h` stays checkable.
//!
//! Values are copied from `../The-Powder-Toy/src/simulation/elements/<IDENT>.cpp`
//! with `CFDS == 1` (one particle per pixel). Elements that own a custom
//! `Update` in TPT keep the same data-driven fields here but behave as their
//! base material until that update is ported, so the data (colours, transition
//! points, weights) is faithful while the behaviour stays data-driven.

use super::super::ElementDef;

/// Appends the batch 2 definitions in The Powder Toy's element order.
pub(crate) fn push(v: &mut Vec<ElementDef>) {
    super::batch2a::push(v);
    super::batch2b::push(v);
    super::batch2c::push(v);
}
