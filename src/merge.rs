//! Explicit multi-map merging with order preservation.
//!
//! [`merge_ordered`] merges any number of [`IndexMap`] inputs into one new
//! map, without ever letting a later map blindly overwrite an earlier one.
//!
//! ## Merge rules
//!
//! 1. **Relative order constraints are kept.** Each input map contributes
//!    constraints "key *i* precedes key *i+1*" for every adjacent pair in its
//!    own iteration order. The merged map's key order is a topological
//!    ordering of the union of all these constraints.
//! 2. **Ties are fixed by earliest source.** Whenever several keys are
//!    unconstrained relative to each other, the key whose first occurrence
//!    is in the earliest input map (and earliest position within it) comes
//!    first. This makes the output fully deterministic.
//! 3. **Common keys are merged by the caller.** If a key appears in more
//!    than one input, its values are folded left-to-right in source order
//!    through the caller-provided callback. The callback runs exactly
//!    `occurrences - 1` times per key (once for the classic two-map case).
//! 4. **Cycles are rejected before any callback runs.** If the combined
//!    order constraints form a cycle (e.g. one map says `a` before `b`
//!    while another says `b` before `a`), [`MergeError::Cycle`] is returned
//!    and the merge callback is never invoked.
//! 5. **All-or-nothing values.** If the merge callback fails, the error is
//!    returned as [`MergeError::Merge`] and no partially built map escapes.
//! 6. **Inputs are never mutated**, source order and key identity are
//!    preserved (result keys are cloned from the earliest source that
//!    contains them), and values are never required to be sortable — only
//!    `Clone`.
//!
//! ## Local check
//!
//! Run only this feature's tests with:
//!
//! ```text
//! cargo test --test merge
//! ```

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::hash::{BuildHasher, Hash};

use crate::IndexMap;

/// Error returned by [`merge_ordered`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeError<E> {
    /// The input maps' relative key orders contradict each other and form a
    /// cycle. Detected before the merge callback is ever invoked.
    Cycle,
    /// The caller-provided merge callback failed. No partial map is
    /// returned and the inputs are left unchanged.
    Merge(E),
}

impl<E: core::fmt::Display> core::fmt::Display for MergeError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MergeError::Cycle => {
                f.write_str("input key orders form a cycle and cannot be merged")
            }
            MergeError::Merge(e) => write!(f, "value merge callback failed: {e}"),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for MergeError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            MergeError::Cycle => None,
            MergeError::Merge(e) => Some(e),
        }
    }
}

/// Merge several [`IndexMap`]s into one, preserving relative key order.
///
/// See the [module documentation](self) for the full merge rules.
///
/// * `maps` — input maps in source order; earlier sources win ties.
/// * `merge` — callback combining values of a shared key, called as
///   `merge(key, accumulated, next)` once per additional occurrence, folded
///   in source order. It is only invoked after the key order has been
///   proven acyclic.
///
/// Returns the merged map, or a [`MergeError`] without touching the inputs.
///
/// # Examples
///
/// ```
/// use indexmap::{indexmap, merge_ordered};
///
/// let m1 = indexmap! {"a" => 1, "b" => 2};
/// let m2 = indexmap! {"b" => 20, "c" => 3};
/// let merged = merge_ordered(&[&m1, &m2], |_k, x, y| Ok::<_, ()>(x + y)).unwrap();
/// assert_eq!(merged.keys().copied().collect::<Vec<_>>(), ["a", "b", "c"]);
/// assert_eq!(merged["b"], 22);
/// ```
pub fn merge_ordered<K, V, S, F, E>(
    maps: &[&IndexMap<K, V, S>],
    mut merge: F,
) -> Result<IndexMap<K, V, S>, MergeError<E>>
where
    K: Eq + Hash + Clone,
    V: Clone,
    S: BuildHasher + Default,
    F: FnMut(&K, &V, &V) -> Result<V, E>,
{
    // Phase 1: collect distinct keys (in first-seen source order), their
    // per-source values, and the adjacency constraints of each input.
    let mut ids: IndexMap<&K, usize, S> = IndexMap::with_hasher(S::default());
    let mut keys: Vec<&K> = Vec::new();
    let mut values: Vec<Vec<&V>> = Vec::new();
    let mut rank: Vec<(usize, usize)> = Vec::new(); // (first source, position in it)
    let mut edges: Vec<(usize, usize)> = Vec::new();

    for (src, map) in maps.iter().enumerate() {
        let mut prev: Option<usize> = None;
        for (pos, (k, v)) in map.iter().enumerate() {
            let id = match ids.get(&k) {
                Some(&id) => id,
                None => {
                    let id = keys.len();
                    ids.insert(k, id);
                    keys.push(k);
                    values.push(Vec::new());
                    rank.push((src, pos));
                    id
                }
            };
            values[id].push(v);
            if let Some(p) = prev {
                if p != id && !edges.contains(&(p, id)) {
                    edges.push((p, id));
                }
            }
            prev = Some(id);
        }
    }

    // Phase 2: Kahn's topological sort. Among all currently unconstrained
    // keys, always pick the one with the earliest (source, position) rank,
    // so ties are deterministically fixed by the earliest source.
    let n = keys.len();
    let mut adj: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    let mut indeg = alloc::vec![0usize; n];
    for &(u, v) in &edges {
        adj[u].push(v);
        indeg[v] += 1;
    }
    let mut ready: BinaryHeap<Reverse<(usize, usize, usize)>> = BinaryHeap::new();
    for (id, &d) in indeg.iter().enumerate() {
        if d == 0 {
            ready.push(Reverse((rank[id].0, rank[id].1, id)));
        }
    }
    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(Reverse((_, _, id))) = ready.pop() {
        order.push(id);
        for &m in &adj[id] {
            indeg[m] -= 1;
            if indeg[m] == 0 {
                ready.push(Reverse((rank[m].0, rank[m].1, m)));
            }
        }
    }
    // Reject cycles *before* any value callback runs.
    if order.len() != n {
        return Err(MergeError::Cycle);
    }

    // Phase 3: fold values through the caller's callback and build the
    // result. A callback failure discards the whole result — no partial
    // map is returned and the inputs were never modified.
    let mut result: IndexMap<K, V, S> = IndexMap::with_capacity_and_hasher(n, S::default());
    for id in order {
        let mut occurrences = values[id].iter();
        let first = occurrences.next().expect("every key has at least one value");
        let mut acc = (*first).clone();
        for v in occurrences {
            acc = merge(keys[id], &acc, v).map_err(MergeError::Merge)?;
        }
        result.insert(keys[id].clone(), acc);
    }
    Ok(result)
}
