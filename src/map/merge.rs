//! Explicit, multi-source ordered map merger.
//!
//! [`Merger`] combines several [`IndexMap`]s that each carry their own key
//! order into a single `IndexMap`. It is *not* a "later map overwrites the
//! earlier" merge: every input's relative key-order constraints are preserved,
//! and values of keys shared between sources are combined by a caller-supplied
//! function.
//!
//! # Merge rules
//!
//! 1. **Order constraints.** Each input map contributes the constraint that
//!    every key must appear *before* the key that follows it in that input.
//!    All constraints from all sources are collected first.
//! 2. **Cycle rejection.** If the constraints contradict each other (e.g. one
//!    source orders `a` before `b` while another orders `b` before `a`), the
//!    merge fails with [`MergeError::Cycle`]. This is detected *before* any
//!    value callback is invoked.
//! 3. **Tie-breaking by earliest source.** When two keys are not related by
//!    any constraint, the key whose *first* occurrence is in the earliest
//!    source is emitted first. Within a single source the input order is
//!    preserved by the constraints themselves.
//! 4. **Value merging.** A key that appears in only one source keeps that
//!    value with no callback. A key shared by several sources is folded in
//!    source order: the callback receives the key, the accumulated value and
//!    the next value, and returns the merged value (or an error).
//! 5. **Atomicity.** If the callback returns an error, the merge fails with
//!    [`MergeError::Merge`] and no partial map is returned. The input maps are
//!    only borrowed, so they are never modified.
//! 6. **No sorting required.** Keys and values are never compared by value;
//!    only hash/equality is used to identify shared keys, so values do not need
//!    to be [`Ord`] or even [`PartialOrd`].
//!
//! # Examples
//!
//! ```
//! use indexmap::map::Merger;
//! use indexmap::IndexMap;
//!
//! let a: IndexMap<&str, i32> = [("a", 1), ("b", 2)].into_iter().collect();
//! let b: IndexMap<&str, i32> = [("b", 3), ("c", 4)].into_iter().collect();
//!
//! let merged = Merger::new()
//!     .push(&a)
//!     .push(&b)
//!     .merge(|_key, v1, v2| Ok::<i32, ()>(v1 + v2))
//!     .unwrap();
//!
//! let keys: Vec<_> = merged.keys().copied().collect();
//! assert_eq!(keys, vec!["a", "b", "c"]);
//! assert_eq!(merged["b"], 2 + 3);
//! ```
//!
//! # Running the tests
//!
//! The merge-specific tests live in this module. From the crate root:
//!
//! ```sh
//! cargo test --lib map::merge
//! cargo test --doc map::merge
//! ```
//!
//! The first runs the unit tests (including the `a,b` + `b,c` and `a,b` +
//! `b,a` acceptance cases); the second runs the documentation example.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::fmt;
use core::hash::{BuildHasher, Hash};

use crate::IndexMap;

/// The error type returned by [`Merger::merge`].
#[derive(Debug)]
pub enum MergeError<E> {
    /// The input maps contain contradictory ordering constraints, so no single
    /// order can satisfy every source.
    ///
    /// This is reported before any value callback is invoked.
    Cycle,
    /// The caller-supplied value merge callback returned an error.
    ///
    /// No partial map is returned; the inputs are untouched.
    Merge(E),
}

impl<E: fmt::Display> fmt::Display for MergeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MergeError::Cycle => write!(f, "ordering cycle detected between input maps"),
            MergeError::Merge(e) => write!(f, "value merge callback failed: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl<E> std::error::Error for MergeError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MergeError::Cycle => None,
            MergeError::Merge(e) => Some(e),
        }
    }
}

/// An explicit, multi-source ordered map merger.
///
/// See the [module documentation][self] for the full merge rules.
pub struct Merger<'a, K, V, S> {
    sources: Vec<&'a IndexMap<K, V, S>>,
}

impl<'a, K, V, S> Default for Merger<'a, K, V, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, K, V, S> Merger<'a, K, V, S> {
    /// Create an empty merger.
    pub fn new() -> Self {
        Merger {
            sources: Vec::new(),
        }
    }

    /// Add a source map. Sources are merged in the order they are pushed; the
    /// first source is the earliest one.
    pub fn push(mut self, map: &'a IndexMap<K, V, S>) -> Self {
        self.sources.push(map);
        self
    }
}

impl<'a, K, V, S> Merger<'a, K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    /// Merge all pushed sources into a single [`IndexMap`].
    ///
    /// The `merge` callback is invoked for every key shared by more than one
    /// source, in source order, receiving the key, the accumulated value and
    /// the next value. It must return the merged value or an error.
    ///
    /// # Errors
    ///
    /// * [`MergeError::Cycle`] if the input ordering constraints contradict
    ///   each other. This is detected before any callback runs.
    /// * [`MergeError::Merge`] if a callback returns an error. No partial map
    ///   is returned and the inputs are not modified.
    pub fn merge<F, E>(self, mut merge: F) -> Result<IndexMap<K, V, S>, MergeError<E>>
    where
        F: FnMut(&K, V, V) -> Result<V, E>,
    {
        // --- 1. Intern keys and build the ordering graph. -----------------
        //
        // Each source contributes edges between consecutive keys:
        // `keys[i] -> keys[i+1]`, meaning `keys[i]` must come first.
        let mut key_ids: IndexMap<&K, u32, S> = IndexMap::with_hasher(S::default());
        let mut keys: Vec<&K> = Vec::new();
        let mut first_source: Vec<u32> = Vec::new();
        let mut first_pos: Vec<u32> = Vec::new();
        let mut adjacency: Vec<Vec<u32>> = Vec::new();
        let mut in_degree: Vec<u32> = Vec::new();

        for (si, map) in self.sources.iter().enumerate() {
            let mut prev: Option<u32> = None;
            for (pos, (key, _value)) in map.iter().enumerate() {
                let id = match key_ids.get(key) {
                    Some(&id) => id,
                    None => {
                        let id = keys.len() as u32;
                        key_ids.insert(key, id);
                        keys.push(key);
                        first_source.push(si as u32);
                        first_pos.push(pos as u32);
                        adjacency.push(Vec::new());
                        in_degree.push(0);
                        id
                    }
                };
                if let Some(p) = prev {
                    // Duplicate edges (same consecutive pair in several
                    // sources) are counted in `in_degree` and decremented the
                    // same number of times during the topological walk.
                    adjacency[p as usize].push(id);
                    in_degree[id as usize] += 1;
                }
                prev = Some(id);
            }
        }

        let n = keys.len();

        // --- 2. Topological sort with earliest-source tie-break. -----------
        //
        // Kahn's algorithm. When several keys are available, the one with the
        // earliest first source (and earliest position within that source) is
        // chosen. A `BTreeSet` acts as a deterministic min-priority queue.
        let mut ready: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
        for id in 0..n as u32 {
            if in_degree[id as usize] == 0 {
                ready.insert((first_source[id as usize], first_pos[id as usize], id));
            }
        }

        let mut order: Vec<u32> = Vec::with_capacity(n);
        while let Some(&key) = ready.iter().next() {
            ready.remove(&key);
            let (_fs, _fp, id) = key;
            order.push(id);
            for &succ in &adjacency[id as usize] {
                in_degree[succ as usize] -= 1;
                if in_degree[succ as usize] == 0 {
                    ready.insert((
                        first_source[succ as usize],
                        first_pos[succ as usize],
                        succ,
                    ));
                }
            }
        }

        if order.len() < n {
            // Contradictory constraints: a cycle. Reject before any callback.
            return Err(MergeError::Cycle);
        }

        // --- 3. Merge values in the confirmed order. -----------------------
        //
        // Only now, after the order is proven acyclic, do we invoke the value
        // callback. The result is built from scratch; inputs are only read.
        let mut result = IndexMap::with_capacity_and_hasher(n, S::default());
        for &id in &order {
            let key = keys[id as usize];
            let mut values = self.sources.iter().filter_map(|m| m.get(key));
            // Every interned key came from at least one source.
            let first = values
                .next()
                .expect("interned key must exist in at least one source");
            let mut acc = first.clone();
            for value in values {
                acc = merge(key, acc, value.clone()).map_err(MergeError::Merge)?;
            }
            result.insert(key.clone(), acc);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn kv<'a>(items: &'a [(&'a str, i32)]) -> IndexMap<&'a str, i32> {
        items.iter().copied().collect()
    }

    fn keys_of<'a>(map: &'a IndexMap<&'a str, i32>) -> Vec<&'a str> {
        map.keys().copied().collect()
    }

    // A simple infallible merge callback, so the error type can be inferred.
    fn sum(_k: &&str, a: i32, b: i32) -> Result<i32, ()> {
        Ok(a + b)
    }

    #[test]
    fn merges_two_sources_in_order_with_one_callback() {
        let a = kv(&[("a", 1), ("b", 2)]);
        let b = kv(&[("b", 3), ("c", 4)]);
        let mut calls = 0;
        let mut merged_keys = Vec::new();

        let result = Merger::new()
            .push(&a)
            .push(&b)
            .merge(|key, v1, v2| -> Result<i32, ()> {
                calls += 1;
                merged_keys.push(*key);
                Ok(v1 + v2)
            })
            .unwrap();

        assert_eq!(keys_of(&result), vec!["a", "b", "c"]);
        assert_eq!(result["a"], 1);
        assert_eq!(result["b"], 5);
        assert_eq!(result["c"], 4);
        assert_eq!(calls, 1);
        assert_eq!(merged_keys, vec!["b"]);

        // Inputs are untouched.
        assert_eq!(keys_of(&a), vec!["a", "b"]);
        assert_eq!(keys_of(&b), vec!["b", "c"]);
        assert_eq!(a["b"], 2);
        assert_eq!(b["b"], 3);
    }

    #[test]
    fn rejects_cycle_before_any_callback() {
        let a = kv(&[("a", 1), ("b", 2)]);
        let b = kv(&[("b", 3), ("a", 4)]);
        let mut calls = 0;

        let result = Merger::new()
            .push(&a)
            .push(&b)
            .merge(|_key, v1, v2| -> Result<i32, ()> {
                calls += 1;
                Ok(v1 + v2)
            });

        assert!(matches!(result, Err(MergeError::Cycle)));
        assert_eq!(calls, 0);

        // Inputs are untouched.
        assert_eq!(keys_of(&a), vec!["a", "b"]);
        assert_eq!(keys_of(&b), vec!["b", "a"]);
    }

    #[test]
    fn value_callback_failure_returns_no_map_and_keeps_inputs() {
        let a = kv(&[("a", 1), ("b", 2)]);
        let b = kv(&[("b", 3), ("c", 4)]);

        let result = Merger::new()
            .push(&a)
            .push(&b)
            .merge(|_key, _v1, _v2| -> Result<i32, &'static str> {
                Err("boom")
            });

        assert!(matches!(result, Err(MergeError::Merge("boom"))));

        // Inputs are untouched.
        assert_eq!(a.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(), vec![("a", 1), ("b", 2)]);
        assert_eq!(b.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(), vec![("b", 3), ("c", 4)]);
    }

    #[test]
    fn merges_three_sources_folding_in_source_order() {
        let a = kv(&[("a", 1), ("b", 2)]);
        let b = kv(&[("b", 3), ("c", 4)]);
        let c = kv(&[("c", 5), ("d", 6)]);
        let mut calls = 0;

        let result = Merger::new()
            .push(&a)
            .push(&b)
            .push(&c)
            .merge(|_key, acc, next| -> Result<i32, ()> {
                calls += 1;
                Ok(acc * 10 + next)
            })
            .unwrap();

        assert_eq!(keys_of(&result), vec!["a", "b", "c", "d"]);
        // b: 2 then 3 -> 23 ; c: 4 then 5 -> 45
        assert_eq!(result["b"], 23);
        assert_eq!(result["c"], 45);
        assert_eq!(calls, 2);
    }

    #[test]
    fn tie_break_by_earliest_source() {
        // No constraints between {a,b} and {c,d}: the earliest source wins.
        let a = kv(&[("a", 1), ("b", 2)]);
        let b = kv(&[("c", 3), ("d", 4)]);
        let result = Merger::new()
            .push(&a)
            .push(&b)
            .merge(sum)
            .unwrap();
        assert_eq!(keys_of(&result), vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn single_and_empty_sources() {
        let a = kv(&[("a", 1), ("b", 2)]);
        let empty: IndexMap<&str, i32> = IndexMap::new();

        let result = Merger::new()
            .push(&a)
            .push(&empty)
            .merge(sum)
            .unwrap();
        assert_eq!(keys_of(&result), vec!["a", "b"]);

        let only_empty = Merger::new()
            .push(&empty)
            .merge(sum)
            .unwrap();
        assert!(only_empty.is_empty());
    }

    #[test]
    fn disjoint_sources_keep_everything() {
        let a = kv(&[("a", 1)]);
        let b = kv(&[("b", 2)]);
        let c = kv(&[("c", 3)]);
        let result = Merger::new()
            .push(&a)
            .push(&b)
            .push(&c)
            .merge(sum)
            .unwrap();
        assert_eq!(keys_of(&result), vec!["a", "b", "c"]);
    }

    #[test]
    fn rejects_three_way_cycle_without_callback() {
        let a = kv(&[("a", 1), ("b", 2)]);
        let b = kv(&[("b", 3), ("c", 4)]);
        let c = kv(&[("c", 5), ("a", 6)]);
        let mut calls = 0;
        let result = Merger::new()
            .push(&a)
            .push(&b)
            .push(&c)
            .merge(|_k, v1, v2| -> Result<i32, ()> {
                calls += 1;
                Ok(v1 + v2)
            });
        assert!(matches!(result, Err(MergeError::Cycle)));
        assert_eq!(calls, 0);
    }

    #[test]
    fn preserves_source_order_within_each_input() {
        // source1 orders b before a; source2 orders a before c.
        // Result must be b, a, c (b<a from s1, a<c from s2).
        let s1 = kv(&[("b", 1), ("a", 2)]);
        let s2 = kv(&[("a", 3), ("c", 4)]);
        let result = Merger::new()
            .push(&s1)
            .push(&s2)
            .merge(sum)
            .unwrap();
        assert_eq!(keys_of(&result), vec!["b", "a", "c"]);
    }
}
