//! Tests for the explicit multi-map merger (`indexmap::merge_ordered`).
//!
//! Run locally with: `cargo test --test merge`

use indexmap::{indexmap, merge_ordered, MergeError};

/// (a, b) merged with (b, c) yields (a, b, c); the shared key `b` is
/// merged through the callback exactly once, and neither input changes.
#[test]
fn merges_two_maps_preserving_relative_order() {
    let m1 = indexmap! {"a" => 1, "b" => 2};
    let m2 = indexmap! {"b" => 20, "c" => 3};
    let before1 = m1.clone();
    let before2 = m2.clone();

    let mut calls = 0;
    let merged = merge_ordered(&[&m1, &m2], |_k, x, y| {
        calls += 1;
        Ok::<_, ()>(x + y)
    })
    .unwrap();

    assert_eq!(calls, 1, "shared key `b` must be merged exactly once");
    assert_eq!(
        merged.keys().copied().collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
    assert_eq!(merged["a"], 1);
    assert_eq!(merged["b"], 22, "common key uses the caller's merge, not overwrite");
    assert_eq!(merged["c"], 3);

    // Inputs are not modified.
    assert_eq!(m1, before1);
    assert_eq!(m2, before2);
}

/// (a, b) merged with (b, a) is an order cycle: rejected with
/// `MergeError::Cycle` before the value callback ever runs.
#[test]
fn order_cycle_is_rejected_before_any_callback() {
    let m1 = indexmap! {"a" => 1, "b" => 2};
    let m2 = indexmap! {"b" => 3, "a" => 4};

    let mut calls = 0;
    let err = merge_ordered(&[&m1, &m2], |_k, x, y| {
        calls += 1;
        Ok::<_, ()>(x + y)
    })
    .unwrap_err();

    assert_eq!(err, MergeError::Cycle);
    assert_eq!(calls, 0, "cycle must be detected before value callbacks");
}

/// A failing merge callback yields `MergeError::Merge`, no partial map
/// escapes, and the inputs are left untouched.
#[test]
fn callback_failure_returns_no_partial_map_and_keeps_inputs() {
    let m1 = indexmap! {"a" => 1, "b" => 2};
    let m2 = indexmap! {"b" => 20, "c" => 3};
    let before1 = m1.clone();
    let before2 = m2.clone();

    let err = merge_ordered(&[&m1, &m2], |_k, _x, _y| Err::<i32, _>("boom")).unwrap_err();

    assert_eq!(err, MergeError::Merge("boom"));
    assert_eq!(m1, before1);
    assert_eq!(m2, before2);
}

/// Keys unconstrained relative to each other are ordered by their first
/// occurrence in the earliest source map.
#[test]
fn ties_are_broken_by_earliest_source() {
    let m1 = indexmap! {"a" => 1, "c" => 3};
    let m2 = indexmap! {"b" => 2};

    let merged = merge_ordered(&[&m1, &m2], |_k, x, _y| Ok::<_, ()>(*x)).unwrap();

    // `b` is unconstrained w.r.t. `a` and `c`, but comes from a later
    // source, so it sorts after both.
    assert_eq!(
        merged.keys().copied().collect::<Vec<_>>(),
        ["a", "c", "b"]
    );
}

/// A key shared by three maps folds its values in source order.
#[test]
fn shared_key_folds_across_sources_in_order() {
    let m1 = indexmap! {"k" => String::from("1")};
    let m2 = indexmap! {"k" => String::from("2")};
    let m3 = indexmap! {"k" => String::from("3")};

    let mut calls = 0;
    let merged = merge_ordered(&[&m1, &m2, &m3], |_k, acc, next| {
        calls += 1;
        Ok::<_, ()>(format!("{acc}+{next}"))
    })
    .unwrap();

    assert_eq!(calls, 2);
    assert_eq!(merged["k"], "1+2+3");
}

/// Values never need to be sortable (or ordered) for the merge to work.
#[test]
fn values_need_not_be_sortable() {
    #[derive(Clone, PartialEq, Debug)]
    struct Blob(Vec<i32>); // deliberately no Ord

    let m1 = indexmap! {"x" => Blob(vec![1]), "y" => Blob(vec![2])};
    let m2 = indexmap! {"y" => Blob(vec![3])};

    let merged = merge_ordered(&[&m1, &m2], |_k, a, b| {
        Ok::<_, ()>(Blob([a.0.clone(), b.0.clone()].concat()))
    })
    .unwrap();

    assert_eq!(merged.keys().copied().collect::<Vec<_>>(), ["x", "y"]);
    assert_eq!(merged["x"], Blob(vec![1]));
    assert_eq!(merged["y"], Blob(vec![2, 3]));
}

/// Merging disjoint maps keeps every key, ordered by earliest source.
#[test]
fn disjoint_maps_concatenate_by_source_order() {
    let m1 = indexmap! {"b" => 1, "a" => 2};
    let m2 = indexmap! {"d" => 3, "c" => 4};

    let merged = merge_ordered(&[&m1, &m2], |_k, x, _y| Ok::<_, ()>(*x)).unwrap();

    assert_eq!(
        merged.keys().copied().collect::<Vec<_>>(),
        ["b", "a", "d", "c"]
    );
}
