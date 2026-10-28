//! Brick-aligned leaf masks and the brick occupancy bitmap.
//!
//! A brick is an 8^3 voxel region containing 64 leaf blocks of 2^3 voxels.
//! Its 64 occupancy bits live in exactly two `u32` words, so mask rebuilds and
//! claim commits never share an atomic word with a neighbouring brick.

use crate::voxel3d::{LEAF_WORDS_PER_BRICK, brick_index, local_block};

/// Bits per brick occupancy word.
const BRICK_WORD_BITS: usize = u32::BITS as usize;
/// Shift separating the two leaf-mask words owned by one brick.
const LEAF_WORD_SHIFT: usize = 5;
/// Bit mask within one leaf-mask word.
const LEAF_WORD_MASK: usize = 31;

/// Coarse occupancy levels, from the finest 4^3 level to the single 128^3 bit.
///
/// The bits of each level are stored back-to-back in `coarse_masks`; the bases
/// below are the word indices where each level starts.
pub const COARSE_LEVELS: [u32; 6] = [4, 8, 16, 32, 64, 128];
/// Number of coarse cells along one axis for each level.
pub const COARSE_LEVEL_BITS: [u32; 6] = [64, 32, 16, 8, 4, 2];
/// First `u32` of each coarse level in `coarse_masks`.
pub const COARSE_LEVEL_BASES: [usize; 6] = [0, 8192, 9216, 9344, 9360, 9362];
/// Total `u32` words in the coarse pyramid.
pub const COARSE_WORD_COUNT: usize = 9363;

const _: () = assert!(COARSE_LEVEL_BASES[5] + 1 == COARSE_WORD_COUNT);

/// Word index of the coarse occupancy bit that covers voxel `(x, y, z)`.
///
/// `level` is an index into [`COARSE_LEVELS`] (`0` = 4^3, `5` = 128^3).
#[inline]
pub fn coarse_word_index(level: usize, x: u32, y: u32, z: u32) -> usize {
    let shift = match level {
        0 => 2,
        1 => 3,
        2 => 4,
        3 => 5,
        4 => 6,
        _ => 7,
    };
    let bits = COARSE_LEVEL_BITS[level];
    let cell = ((y >> shift) * bits + (z >> shift)) * bits + (x >> shift);
    COARSE_LEVEL_BASES[level] + (cell >> 5) as usize
}

/// Bit mask of the coarse occupancy bit that covers voxel `(x, y, z)`.
#[inline]
pub fn coarse_bit_mask(level: usize, x: u32, y: u32, z: u32) -> u32 {
    let shift = match level {
        0 => 2,
        1 => 3,
        2 => 4,
        3 => 5,
        4 => 6,
        _ => 7,
    };
    let bits = COARSE_LEVEL_BITS[level];
    let cell = ((y >> shift) * bits + (z >> shift)) * bits + (x >> shift);
    1u32 << (cell & 31)
}

/// Index of the leaf-mask word containing voxel `(x, y, z)`.
///
/// Each brick owns `[brick * 2, brick * 2 + 1]`, so adjacent bricks never
/// write the same word.
#[inline]
pub fn leaf_word_index(x: u32, y: u32, z: u32) -> usize {
    let block = local_block(x, y, z);
    brick_index(x, y, z) * LEAF_WORDS_PER_BRICK + (block >> LEAF_WORD_SHIFT)
}

/// Bit mask of voxel `(x, y, z)` inside its leaf-mask word.
#[inline]
pub fn leaf_bit_mask(x: u32, y: u32, z: u32) -> u32 {
    1u32 << (local_block(x, y, z) & LEAF_WORD_MASK)
}

/// Word of the `brick_occupied` bitmap containing `brick`.
#[inline]
pub fn brick_word_index(brick: usize) -> usize {
    brick / BRICK_WORD_BITS
}

/// Bit mask of `brick` inside its `brick_occupied` word.
#[inline]
pub fn brick_bit_mask(brick: usize) -> u32 {
    1u32 << (brick & (BRICK_WORD_BITS - 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel3d::{BRICK_COUNT, LEAF_WORD_COUNT, VOXEL_COUNT, VOXEL_DIM};

    #[test]
    fn voxel_indices_are_unique() {
        let mut seen = vec![0u64; VOXEL_COUNT.div_ceil(u64::BITS as usize)];
        for y in 0..VOXEL_DIM {
            for z in 0..VOXEL_DIM {
                for x in 0..VOXEL_DIM {
                    let index = crate::voxel3d::voxel_index(x, y, z);
                    assert!(index < VOXEL_COUNT);
                    let word = index >> 6;
                    let bit = 1u64 << (index & 63);
                    assert_eq!(seen[word] & bit, 0, "duplicate voxel index {index}");
                    seen[word] |= bit;
                }
            }
        }
        let total: u32 = seen.iter().map(|word| word.count_ones()).sum();
        assert_eq!(total as usize, VOXEL_COUNT);
    }

    #[test]
    fn every_brick_owns_exactly_two_disjoint_leaf_words() {
        let mut masks = vec![0u32; LEAF_WORD_COUNT];
        let mut owners = vec![usize::MAX; LEAF_WORD_COUNT];

        for brick in 0..BRICK_COUNT {
            let b = brick as u32;
            let bx = b & 31;
            let bz = (b >> 5) & 31;
            let by = (b >> 10) & 31;
            let base_word = brick * LEAF_WORDS_PER_BRICK;

            for block in 0u32..64 {
                let lx = block & 3;
                let lz = (block >> 2) & 3;
                let ly = (block >> 4) & 3;
                let x = bx * 8 + lx * 2;
                let y = by * 8 + ly * 2;
                let z = bz * 8 + lz * 2;

                assert_eq!(brick_index(x, y, z), brick);
                assert_eq!(local_block(x, y, z) as u32, block);

                let word = leaf_word_index(x, y, z);
                assert!(word == base_word || word == base_word + 1);
                assert_eq!(word, base_word + (block as usize >> 5));
                assert_eq!(leaf_bit_mask(x, y, z), 1u32 << (block & 31));

                assert!(
                    owners[word] == usize::MAX || owners[word] == brick,
                    "leaf word {word} shared by bricks {} and {brick}",
                    owners[word]
                );
                owners[word] = brick;

                let bit = leaf_bit_mask(x, y, z);
                assert_eq!(
                    masks[word] & bit,
                    0,
                    "leaf bit for block {block} of brick {brick} set twice"
                );
                masks[word] |= bit;
            }
        }

        for brick in 0..BRICK_COUNT {
            let base_word = brick * LEAF_WORDS_PER_BRICK;
            assert_eq!(owners[base_word], brick);
            assert_eq!(owners[base_word + 1], brick);
            assert_eq!(masks[base_word], u32::MAX);
            assert_eq!(masks[base_word + 1], u32::MAX);
        }
        assert!(owners.iter().all(|&owner| owner != usize::MAX));
    }

    #[test]
    fn coarse_levels_cover_the_pyramid_without_overlap() {
        assert_eq!(COARSE_LEVELS, [4, 8, 16, 32, 64, 128]);
        assert_eq!(COARSE_WORD_COUNT, 9363);
        for level in 0..COARSE_LEVELS.len() {
            let end = if level + 1 < COARSE_LEVEL_BASES.len() {
                COARSE_LEVEL_BASES[level + 1]
            } else {
                COARSE_WORD_COUNT
            };
            let words = end - COARSE_LEVEL_BASES[level];
            let bits = COARSE_LEVEL_BITS[level];
            assert_eq!(words, (bits * bits * bits).div_ceil(32) as usize);

            let step = COARSE_LEVELS[level];
            let word = coarse_word_index(level, 0, 0, 0);
            let mask = coarse_bit_mask(level, 0, 0, 0);
            assert_eq!(word, COARSE_LEVEL_BASES[level]);
            assert_eq!(mask, 1);
            assert_eq!(word, coarse_word_index(level, step - 1, step - 1, step - 1));
            assert_eq!(mask, coarse_bit_mask(level, step - 1, step - 1, step - 1));

            let next = step;
            assert_ne!(
                (
                    coarse_word_index(level, 0, 0, 0),
                    coarse_bit_mask(level, 0, 0, 0)
                ),
                (
                    coarse_word_index(level, next, 0, 0),
                    coarse_bit_mask(level, next, 0, 0)
                ),
            );
        }
    }

    #[test]
    fn brick_occupancy_bits_are_unique() {
        assert_eq!(brick_word_index(0), 0);
        assert_eq!(brick_word_index(31), 0);
        assert_eq!(brick_word_index(32), 1);
        assert_eq!(brick_word_index(BRICK_COUNT - 1), 1023);
        assert_eq!(brick_bit_mask(0), 1);
        assert_eq!(brick_bit_mask(31), 1 << 31);
        assert_eq!(brick_bit_mask(32), 1);

        let mut words = vec![0u32; BRICK_COUNT.div_ceil(BRICK_WORD_BITS)];
        for brick in 0..BRICK_COUNT {
            let word = brick_word_index(brick);
            let bit = brick_bit_mask(brick);
            assert!(word < words.len());
            assert_eq!(bit.count_ones(), 1);
            assert_eq!(
                words[word] & bit,
                0,
                "duplicate occupancy bit for brick {brick}"
            );
            words[word] |= bit;
        }
        assert_eq!(words.len(), BRICK_COUNT / BRICK_WORD_BITS);
        assert!(words.iter().all(|&word| word == u32::MAX));
    }
}
