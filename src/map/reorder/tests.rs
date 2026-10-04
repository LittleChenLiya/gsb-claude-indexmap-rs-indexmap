use super::*;
use std::collections::{HashMap, VecDeque};
use std::string::ToString;

#[test]
fn reorder_plan_abcd_to_bcad() {
    let mut map = IndexMap::from([("a", 1), ("b", 2), ("c", 3), ("d", 4)]);
    let plan = map.reorder_plan(&["b", "c", "a", "d"]).unwrap();

    // A single move of "a" from index 0 to index 2 suffices.
    assert_eq!(plan.moves(), &[MoveOp { from: 0, to: 2 }]);
    assert_eq!(plan.len(), 1);
    assert!(!plan.is_empty());

    // Planning is read-only: the map is untouched so far.
    assert_eq!(map.keys().copied().collect::<Vec<_>>(), ["a", "b", "c", "d"]);

    map.apply_reorder_plan(&plan).unwrap();
    assert_eq!(map.keys().copied().collect::<Vec<_>>(), ["b", "c", "a", "d"]);

    // Values are still bound to their original keys.
    assert_eq!(map["a"], 1);
    assert_eq!(map["b"], 2);
    assert_eq!(map["c"], 3);
    assert_eq!(map["d"], 4);
}

#[test]
fn reorder_plan_identity_is_empty() {
    let mut map = IndexMap::from([("a", 1), ("b", 2)]);
    let plan = map.reorder_plan(&["a", "b"]).unwrap();
    assert!(plan.is_empty());
    map.apply_reorder_plan(&plan).unwrap();
    assert_eq!(map.keys().copied().collect::<Vec<_>>(), ["a", "b"]);
}

#[test]
fn reorder_plan_prefers_earliest_keys_on_ties() {
    // Reversing three keys has three equally long subsequences to keep
    // ("a", "b", or "c"); the tie is resolved for the earliest key "a".
    let mut map = IndexMap::from([("a", 1), ("b", 2), ("c", 3)]);
    let plan = map.reorder_plan(&["c", "b", "a"]).unwrap();
    assert_eq!(
        plan.moves(),
        &[MoveOp { from: 2, to: 0 }, MoveOp { from: 2, to: 1 }]
    );
    map.apply_reorder_plan(&plan).unwrap();
    assert_eq!(map.keys().copied().collect::<Vec<_>>(), ["c", "b", "a"]);
}

#[test]
fn reorder_plan_rejects_invalid_targets() {
    let map = IndexMap::from([("a", 1), ("b", 2), ("c", 3)]);
    assert_eq!(
        map.reorder_plan(&["a", "b"]),
        Err(ReorderError::LengthMismatch {
            expected: 3,
            found: 2
        })
    );
    assert_eq!(
        map.reorder_plan(&["a", "b", "b"]),
        Err(ReorderError::DuplicateKey { index: 2 })
    );
    assert_eq!(
        map.reorder_plan(&["a", "b", "z"]),
        Err(ReorderError::MissingKey { index: 2 })
    );
}

#[test]
fn reorder_plan_empty_map() {
    let mut map = IndexMap::<&str, i32>::new();
    let empty: [&str; 0] = [];
    let plan = map.reorder_plan(&empty).unwrap();
    assert!(plan.is_empty());
    assert_eq!(plan.len(), 0);
    map.apply_reorder_plan(&plan).unwrap();
    assert!(map.is_empty());

    // An empty target on a non-empty map is a length mismatch.
    let map = IndexMap::from([("a", 1)]);
    assert_eq!(
        map.reorder_plan(&empty),
        Err(ReorderError::LengthMismatch {
            expected: 1,
            found: 0
        })
    );
}

#[test]
fn apply_rejects_stale_plan() {
    let mut map = IndexMap::from([("a", 1), ("b", 2), ("c", 3)]);
    let plan = map.reorder_plan(&["c", "a", "b"]).unwrap();

    // The baseline order is changed externally before the plan is applied.
    map.move_index(0, 2);
    let before: Vec<_> = map.keys().copied().collect();
    match map.apply_reorder_plan(&plan) {
        Err(ReorderError::StalePlan { expected, found }) => {
            assert_eq!(expected, plan.fingerprint());
            assert_ne!(found, expected);
        }
        other => panic!("expected a stale-plan rejection, got {other:?}"),
    }
    // A rejected plan leaves the map untouched.
    assert_eq!(map.keys().copied().collect::<Vec<_>>(), before);
}

#[test]
fn reorder_error_display() {
    let err = ReorderError::LengthMismatch {
        expected: 3,
        found: 2,
    };
    assert_eq!(
        err.to_string(),
        "target key order has 2 keys, but the map has 3"
    );
    assert_eq!(
        ReorderError::DuplicateKey { index: 2 }.to_string(),
        "target key order repeats a key at index 2"
    );
    assert_eq!(
        ReorderError::MissingKey { index: 1 }.to_string(),
        "target key order has an unknown key at index 1"
    );
    assert_eq!(
        MoveOp { from: 3, to: 1 }.to_string(),
        "move 3 -> 1"
    );
}

/// Minimum number of `move_index`-style moves to turn `0..n` into every
/// permutation of it, computed by exhaustive breadth-first search.
fn brute_force_min_moves(n: usize) -> HashMap<Vec<usize>, usize> {
    let start: Vec<usize> = (0..n).collect();
    let mut dist = HashMap::new();
    dist.insert(start.clone(), 0usize);
    let mut queue = VecDeque::from([start]);
    while let Some(state) = queue.pop_front() {
        let d = dist[&state];
        for from in 0..n {
            for to in 0..n {
                if from == to {
                    continue;
                }
                let mut next = state.clone();
                let x = next.remove(from);
                next.insert(to, x);
                if !dist.contains_key(&next) {
                    dist.insert(next.clone(), d + 1);
                    queue.push_back(next);
                }
            }
        }
    }
    dist
}

#[test]
fn reorder_plan_is_minimal_on_small_permutations() {
    for n in 0..=6usize {
        let dist = brute_force_min_moves(n);
        for (perm, &min) in &dist {
            let mut map: IndexMap<usize, usize> = (0..n).map(|i| (i, i * 100)).collect();
            let plan = map.reorder_plan(perm).unwrap();

            // The plan uses the minimum number of moves.
            assert_eq!(plan.len(), min, "n={n} perm={perm:?}");

            // Executing the plan step by step reaches the target order.
            let mut work: Vec<usize> = (0..n).collect();
            for op in plan.moves() {
                let x = work.remove(op.from);
                work.insert(op.to, x);
            }
            assert_eq!(&work, perm, "n={n} perm={perm:?}");

            // Applying the plan agrees, and key-value bindings survive.
            map.apply_reorder_plan(&plan).unwrap();
            assert_eq!(&map.keys().copied().collect::<Vec<_>>(), perm);
            for (k, v) in &map {
                assert_eq!(*k * 100, *v);
            }
        }
    }
}
