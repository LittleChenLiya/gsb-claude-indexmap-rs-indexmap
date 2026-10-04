//! Planning and applying reorderings of an [`IndexMap`].
//!
//! [`IndexMap::reorder_plan`] computes a minimal sequence of index moves
//! that transforms the current key order into a desired target order,
//! without touching any key-value bindings. The plan is computed read-only;
//! [`IndexMap::apply_reorder_plan`] later verifies a fingerprint of the
//! original key order before executing the moves, so a stale plan (the map
//! was reordered in between) is rejected instead of silently misapplied.

use super::IndexMap;
use crate::Equivalent;
use alloc::vec::Vec;
use core::fmt;
use core::hash::{BuildHasher, Hash, Hasher};

/// A single move operation of a [`ReorderPlan`].
///
/// The indices are relative to the map state *at that step* of the plan:
/// each operation is meant to be executed with
/// [`IndexMap::move_index`], and later operations are indexed against the
/// map as reshaped by all preceding operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoveOp {
    /// Index of the key-value pair to move, in the current (per-step) order.
    pub from: usize,
    /// Index the pair should occupy after this step.
    pub to: usize,
}

impl fmt::Display for MoveOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "move {} -> {}", self.from, self.to)
    }
}

/// A precomputed plan that reorders an [`IndexMap`] into a target key order.
///
/// Created by [`IndexMap::reorder_plan`]; executed by
/// [`IndexMap::apply_reorder_plan`]. The plan carries a fingerprint of the
/// key order it was computed from, and applying it fails with
/// [`ReorderError::StalePlan`] if the key order of the map has changed since.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReorderPlan {
    fingerprint: u64,
    moves: Vec<MoveOp>,
}

impl ReorderPlan {
    /// Fingerprint of the original key order this plan was computed from.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The ordered move operations of this plan.
    ///
    /// Executing them in order with [`IndexMap::move_index`] (each indexed
    /// against the result of the previous steps) reproduces the target
    /// order, leaving every key-value binding intact.
    pub fn moves(&self) -> &[MoveOp] {
        &self.moves
    }

    /// Number of move operations in the plan.
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    /// Returns `true` if the plan performs no moves (already in order).
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }
}

/// Errors returned by [`IndexMap::reorder_plan`] and
/// [`IndexMap::apply_reorder_plan`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReorderError {
    /// The target key sequence has a different length than the map.
    LengthMismatch {
        /// Number of keys in the map.
        expected: usize,
        /// Number of keys in the target sequence.
        found: usize,
    },
    /// The target sequence mentions the same key more than once.
    DuplicateKey {
        /// Index of the repeated key within the target sequence.
        index: usize,
    },
    /// The target sequence mentions a key that is not in the map.
    MissingKey {
        /// Index of the unknown key within the target sequence.
        index: usize,
    },
    /// The plan fingerprint does not match the current key order of the
    /// map, i.e. the map was reordered after the plan was computed.
    StalePlan {
        /// Fingerprint recorded in the plan.
        expected: u64,
        /// Fingerprint of the current key order of the map.
        found: u64,
    },
}

impl fmt::Display for ReorderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ReorderError::LengthMismatch { expected, found } => write!(
                f,
                "target key order has {found} keys, but the map has {expected}"
            ),
            ReorderError::DuplicateKey { index } => {
                write!(f, "target key order repeats a key at index {index}")
            }
            ReorderError::MissingKey { index } => {
                write!(f, "target key order has an unknown key at index {index}")
            }
            ReorderError::StalePlan { expected, found } => write!(
                f,
                "reorder plan is stale: expected fingerprint {expected:#x}, found {found:#x}"
            ),
        }
    }
}

impl core::error::Error for ReorderError {}

/// FNV-1a hasher used for order fingerprints (deterministic across runs).
struct FnvHasher(u64);

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
        }
    }
}

fn fnv_mix(acc: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *acc = (*acc ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
}

impl<K, V, S> IndexMap<K, V, S>
where
    K: Hash,
    S: BuildHasher,
{
    /// Computes a fingerprint of the current key order.
    fn order_fingerprint(&self) -> u64 {
        let mut acc = FNV_OFFSET;
        fnv_mix(&mut acc, self.len() as u64);
        for key in self.keys() {
            let mut hasher = FnvHasher(FNV_OFFSET);
            key.hash(&mut hasher);
            fnv_mix(&mut acc, hasher.finish());
        }
        acc
    }

    /// Computes a [`ReorderPlan`] that transforms the current key order into
    /// the `target` key order, without modifying the map.
    ///
    /// The `target` sequence must contain exactly the same keys as the map,
    /// each exactly once; a length mismatch, a missing key, or a repeated
    /// key is rejected with a [`ReorderError`] before any planning happens.
    ///
    /// The plan keeps a longest subsequence of keys that already appear in
    /// the target relative order untouched (a longest common subsequence
    /// of the current and target orders), so the number of moves is
    /// minimal: `len - lcs`. Among equally long subsequences, the one
    /// favouring the earliest keys of the current order is chosen, making
    /// the plan deterministic. Each [`MoveOp`] is indexed against the map
    /// state after the preceding operations have been applied.
    ///
    /// Planning is read-only. Execute the result with
    /// [`IndexMap::apply_reorder_plan`], which first checks that the key
    /// order of the map still matches the plan fingerprint.
    ///
    /// # Examples
    ///
    /// ```
    /// use indexmap::map::MoveOp;
    /// use indexmap::IndexMap;
    ///
    /// let mut map = IndexMap::from([("a", 1), ("b", 2), ("c", 3), ("d", 4)]);
    /// let plan = map.reorder_plan(&["b", "c", "a", "d"]).unwrap();
    ///
    /// // Only "a" has to move, from index 0 to index 2.
    /// assert_eq!(plan.moves(), &[MoveOp { from: 0, to: 2 }]);
    ///
    /// map.apply_reorder_plan(&plan).unwrap();
    /// assert_eq!(map.keys().copied().collect::<Vec<_>>(), ["b", "c", "a", "d"]);
    /// // Values are still bound to their original keys.
    /// assert_eq!(map["a"], 1);
    /// ```
    ///
    /// A target that is not a permutation of the current keys is rejected:
    ///
    /// ```
    /// use indexmap::map::ReorderError;
    /// use indexmap::IndexMap;
    ///
    /// let map = IndexMap::from([("a", 1), ("b", 2)]);
    /// assert_eq!(
    ///     map.reorder_plan(&["a", "a"]),
    ///     Err(ReorderError::DuplicateKey { index: 1 }),
    /// );
    /// assert_eq!(
    ///     map.reorder_plan(&["a", "z"]),
    ///     Err(ReorderError::MissingKey { index: 1 }),
    /// );
    /// ```
    pub fn reorder_plan<Q>(&self, target: &[Q]) -> Result<ReorderPlan, ReorderError>
    where
        Q: Hash + Equivalent<K>,
    {
        let n = self.len();
        if target.len() != n {
            return Err(ReorderError::LengthMismatch {
                expected: n,
                found: target.len(),
            });
        }

        // `perm[i]` is the current index of the key that should end up at
        // position `i`; the target is valid iff `perm` is a permutation.
        let mut perm = Vec::with_capacity(n);
        let mut seen = alloc::vec![false; n];
        for (index, key) in target.iter().enumerate() {
            let Some(current) = self.get_index_of(key) else {
                return Err(ReorderError::MissingKey { index });
            };
            if seen[current] {
                return Err(ReorderError::DuplicateKey { index });
            }
            seen[current] = true;
            perm.push(current);
        }

        // Keys not in a longest increasing subsequence of `perm` (a longest
        // common subsequence of the current and target orders) must move;
        // everything else keeps its relative position. The patience-based
        // scan keeps the smallest possible tail value at every length, so
        // ties are resolved in favour of the earliest current keys.
        let mut tails: Vec<usize> = Vec::with_capacity(n);
        let mut prev = alloc::vec![usize::MAX; n];
        for (i, &value) in perm.iter().enumerate() {
            let len = tails.partition_point(|&t| perm[t] < value);
            if len > 0 {
                prev[i] = tails[len - 1];
            }
            if len == tails.len() {
                tails.push(i);
            } else {
                tails[len] = i;
            }
        }
        let mut keep = alloc::vec![false; n];
        if let Some(&last) = tails.last() {
            let mut i = last;
            while i != usize::MAX {
                keep[perm[i]] = true;
                i = prev[i];
            }
        }

        // Emit the moves, simulating each step so every operation is
        // indexed against the map state produced by the previous ones.
        //
        // Non-kept keys are relocated one by one, in target order. A key
        // with target index `i` is reinserted immediately after the `i`-th
        // fixed key of the working order, where "fixed" means kept or
        // already relocated: every key preceding it in the target order is
        // already fixed, so once all relocations are done the working order
        // equals the target order. Later relocations only remove not-yet
        // fixed keys, which cannot disturb this alignment.
        let mut work: Vec<usize> = (0..n).collect();
        let mut position: Vec<usize> = (0..n).collect();
        let mut fixed = keep.clone();
        let mut moves = Vec::new();
        for (i, &value) in perm.iter().enumerate() {
            if keep[value] {
                continue;
            }
            let from = position[value];
            work.remove(from);
            for (p, &w) in work.iter().enumerate().skip(from) {
                position[w] = p;
            }
            // Insert right after the `i`-th fixed key of the working order.
            let to = if i == 0 {
                0
            } else {
                let mut count = 0;
                let mut to = work.len();
                for (index, &w) in work.iter().enumerate() {
                    if fixed[w] {
                        count += 1;
                        if count == i {
                            to = index + 1;
                            break;
                        }
                    }
                }
                to
            };
            work.insert(to, value);
            for (p, &w) in work.iter().enumerate().skip(to) {
                position[w] = p;
            }
            fixed[value] = true;
            if from != to {
                moves.push(MoveOp { from, to });
            }
        }
        debug_assert_eq!(work, perm);

        Ok(ReorderPlan {
            fingerprint: self.order_fingerprint(),
            moves,
        })
    }

    /// Executes a [`ReorderPlan`] previously computed by
    /// [`IndexMap::reorder_plan`].
    ///
    /// The current key order of the map must still match the order the plan
    /// was computed from; this is verified through the plan fingerprint, and
    /// a mismatch is rejected with [`ReorderError::StalePlan`] without
    /// modifying the map. Otherwise the move operations are executed in
    /// order, leaving every key-value binding intact.
    ///
    /// # Examples
    ///
    /// ```
    /// use indexmap::map::ReorderError;
    /// use indexmap::IndexMap;
    ///
    /// let mut map = IndexMap::from([("a", 1), ("b", 2), ("c", 3), ("d", 4)]);
    /// let plan = map.reorder_plan(&["b", "c", "a", "d"]).unwrap();
    ///
    /// // An external reorder invalidates the plan.
    /// map.move_index(0, 3);
    /// assert!(matches!(
    ///     map.apply_reorder_plan(&plan),
    ///     Err(ReorderError::StalePlan { .. })
    /// ));
    /// ```
    pub fn apply_reorder_plan(&mut self, plan: &ReorderPlan) -> Result<(), ReorderError> {
        let found = self.order_fingerprint();
        if found != plan.fingerprint {
            return Err(ReorderError::StalePlan {
                expected: plan.fingerprint,
                found,
            });
        }
        for &MoveOp { from, to } in &plan.moves {
            self.move_index(from, to);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "reorder/tests.rs"]
mod tests;
