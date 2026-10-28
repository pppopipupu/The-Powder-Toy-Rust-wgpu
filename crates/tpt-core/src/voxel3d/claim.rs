//! Movement-claim encoding for the atomic `claim` buffer.
//!
//! A claim word packs a 24-bit owner voxel index and an 8-bit tag (plan §3.5):
//!
//! ```text
//! bits  0..24  owner  voxel linear index
//! bits 24..32  tag    move tag 1..=127, or tag | 0x80 for a rigid-body block
//! ```
//!
//! Move keys always have bit 31 clear; blocked keys have bit 31 set and owner
//! `0xFFFFFF`, so `atomicMax` lets a body win over every mover. The tag keeps
//! stale blocked keys from pinning a cell after the body has moved away.

/// Number of low bits holding the owner voxel index.
pub const CLAIM_OWNER_BITS: u32 = 24;
/// Mask selecting the owner voxel index.
pub const CLAIM_OWNER_MASK: u32 = (1u32 << CLAIM_OWNER_BITS) - 1;
/// Shift separating the tag from the owner index.
pub const CLAIM_TAG_SHIFT: u32 = CLAIM_OWNER_BITS;
/// Mask selecting the full 8-bit tag.
pub const CLAIM_TAG_MASK: u32 = 0xFF;
/// Mask selecting the 7-bit move tag.
pub const CLAIM_MOVE_TAG_MASK: u32 = 0x7F;
/// High tag bit marking a rigid-body block.
pub const CLAIM_BLOCKED_BIT: u32 = 0x80;
/// Number of ticks after which a move tag repeats.
pub const CLAIM_MOVE_TAG_PERIOD: u64 = CLAIM_MOVE_TAG_MASK as u64;

/// Move tag for `tick`: `tick % 127 + 1`, so tags occupy `1..=127` and never
/// carry the blocked bit.
#[inline]
pub fn move_tag(tick: u64) -> u32 {
    ((tick % CLAIM_MOVE_TAG_PERIOD) + 1) as u32
}

/// Claim key written by a voxel at `owner` moving with `tag`.
#[inline]
pub fn move_claim_key(owner: u32, tag: u32) -> u32 {
    debug_assert!(tag <= CLAIM_MOVE_TAG_MASK, "move tag must fit in 7 bits");
    ((tag & CLAIM_MOVE_TAG_MASK) << CLAIM_TAG_SHIFT) | (owner & CLAIM_OWNER_MASK)
}

/// Claim key written by a rigid body blocking a world cell with `tag`.
///
/// The owner is all ones and the blocked tag bit is set, so the key outranks
/// every move key. With `tag == 127` the literal value is `0xFFFFFFFF`.
#[inline]
pub fn blocked_claim_key(tag: u32) -> u32 {
    debug_assert!(tag <= CLAIM_MOVE_TAG_MASK, "blocked tag must fit in 7 bits");
    (((tag & CLAIM_MOVE_TAG_MASK) | CLAIM_BLOCKED_BIT) << CLAIM_TAG_SHIFT) | CLAIM_OWNER_MASK
}

/// Extracts the full 8-bit tag from a claim word.
#[inline]
pub fn claim_tag(claim: u32) -> u32 {
    claim >> CLAIM_TAG_SHIFT
}

/// Extracts the owner voxel index from a claim word.
#[inline]
pub fn claim_owner(claim: u32) -> u32 {
    claim & CLAIM_OWNER_MASK
}

/// Whether a claim word is a rigid-body block rather than a move key.
#[inline]
pub fn claim_is_blocked(claim: u32) -> bool {
    claim_tag(claim) & CLAIM_BLOCKED_BIT != 0
}

/// Whether `claim` is a move key for the current `tag`.
///
/// Tag zero is not a move tag, so an untouched (`0`) claim never matches.
#[inline]
pub fn claim_matches_move(claim: u32, tag: u32) -> bool {
    let tag = tag & CLAIM_MOVE_TAG_MASK;
    tag != 0 && !claim_is_blocked(claim) && claim_tag(claim) == tag
}

/// Whether `claim` is a rigid-body block for the current `tag`.
#[inline]
pub fn claim_matches_blocked(claim: u32, tag: u32) -> bool {
    claim_is_blocked(claim)
        && (claim_tag(claim) & CLAIM_MOVE_TAG_MASK) == (tag & CLAIM_MOVE_TAG_MASK)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNERS: [u32; 4] = [0, 1, 0x12_3456, CLAIM_OWNER_MASK];

    #[test]
    fn move_tags_stay_in_range_and_wrap() {
        assert_eq!(move_tag(0), 1);
        assert_eq!(move_tag(1), 2);
        assert_eq!(move_tag(126), 127);
        assert_eq!(move_tag(127), 1);
        assert_eq!(move_tag(128), 2);
        assert_eq!(move_tag(253), 127);
        assert_eq!(move_tag(254), 1);
        assert_eq!(move_tag(255), 2);

        for tick in 0..1024 {
            let tag = move_tag(tick);
            assert!((1..=CLAIM_MOVE_TAG_MASK).contains(&tag));
        }
    }

    #[test]
    fn claim_fields_round_trip() {
        for tag in 1..=CLAIM_MOVE_TAG_MASK {
            for owner in OWNERS {
                let key = move_claim_key(owner, tag);
                assert_eq!(claim_owner(key), owner);
                assert_eq!(claim_tag(key), tag);
                assert!(!claim_is_blocked(key));
                assert!(claim_matches_move(key, tag));
                assert!(!claim_matches_blocked(key, tag));

                let blocked = blocked_claim_key(tag);
                assert_eq!(claim_owner(blocked), CLAIM_OWNER_MASK);
                assert_eq!(claim_tag(blocked), tag | CLAIM_BLOCKED_BIT);
                assert!(claim_is_blocked(blocked));
                assert!(claim_matches_blocked(blocked, tag));
                assert!(!claim_matches_move(blocked, tag));
            }
        }

        assert_eq!(move_claim_key(0x12_3456, 5), 0x0512_3456);
        assert_eq!(move_claim_key(u32::MAX, 1), 0x01FF_FFFF);
        assert_eq!(blocked_claim_key(1), 0x81FF_FFFF);
        assert_eq!(blocked_claim_key(127), u32::MAX);
    }

    #[test]
    fn stale_and_empty_claims_do_not_match() {
        let key = move_claim_key(7, 3);
        assert!(!claim_matches_move(key, 2));
        assert!(!claim_matches_move(key, 4));
        assert!(!claim_matches_move(0, 1));
        assert!(!claim_matches_move(0, 0));
        assert!(!claim_matches_blocked(0, 0));

        let blocked = blocked_claim_key(3);
        assert!(!claim_matches_blocked(blocked, 2));
        assert!(!claim_matches_blocked(blocked, 4));
        assert!(!claim_matches_move(blocked, 3));
    }

    #[test]
    fn blocked_keys_outrank_every_move_key() {
        for blocked_tag in 1..=CLAIM_MOVE_TAG_MASK {
            let blocked = blocked_claim_key(blocked_tag);
            assert!(blocked > 0x7FFF_FFFF);
            for tag in 1..=CLAIM_MOVE_TAG_MASK {
                for owner in OWNERS {
                    assert!(blocked > move_claim_key(owner, tag));
                }
            }
        }
        for tag in 1..CLAIM_MOVE_TAG_MASK {
            assert!(blocked_claim_key(tag) < blocked_claim_key(tag + 1));
        }
    }
}
