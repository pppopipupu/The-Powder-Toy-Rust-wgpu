//! The built-in element table, split by menu category.
//!
//! Each module holds the definitions of one (or two closely related) menu
//! categories and exposes a `push_*` function that appends them to the table.
//! `builtin_elements` calls the sections in the original monolithic table
//! order, so the emitted sequence of `PT_*` slots is unchanged by the split.
//!
//! [`batch1`] and [`batch2`] are the exceptions: they hold the elements whose
//! ids were appended above the classic ones (`PT_GOO ..= PT_SEED`) and
//! therefore keep TPT's element order instead of grouping by category.

pub(super) mod batch1;
pub(super) mod batch2;
pub(super) mod batch2a;
pub(super) mod batch2b;
pub(super) mod batch2c;
pub(super) mod energy;
pub(super) mod gases;
pub(super) mod liquids;
pub(super) mod powders;
pub(super) mod solids;
pub(super) mod special;
