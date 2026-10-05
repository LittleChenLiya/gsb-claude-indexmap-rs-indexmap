//! Acceptance tests for the explicit entry-handle layer (`indexmap::tracked`).
//!
//! Focused run: `cargo test --test tracked_handles`

use indexmap::tracked::TrackedMap;

/// 取 b 句柄后移除 a 仍定位 b：shift_remove 会改变索引，但句柄不靠旧索引。
#[test]
fn handle_survives_shift_remove_of_other_item() {
    let mut map = TrackedMap::new();
    map.insert("a", 10);
    let b = map.insert_handle("b", 20);
    map.insert("c", 30);

    assert_eq!(map.index_of_handle(&b), Some(1));
    assert_eq!(map.shift_remove(&"a"), Some(10));

    // b 被挪到索引 0，但同一句柄仍定位同一个真实项。
    assert_eq!(map.index_of_handle(&b), Some(0));
    assert_eq!(map.get_by_handle(&b), Some(&20));
    assert_eq!(b.key(), &"b");
}

#[test]
fn handle_survives_swap_remove_of_other_item() {
    let mut map = TrackedMap::new();
    map.insert("a", 10);
    let b = map.insert_handle("b", 20);
    map.insert("c", 30);
    map.insert("d", 40);

    // swap-remove 会把最后一项挪进被删位置；b 句柄仍指向自己。
    assert_eq!(map.swap_remove(&"a"), Some(10));
    assert_eq!(map.get_by_handle(&b), Some(&20));
    assert_eq!(map.get(&"d"), Some(&40));

    assert_eq!(map.swap_remove_handle(&b), Some(20));
    assert_eq!(map.get_by_handle(&b), None);
}

/// 重排只换次序不换项身份。
#[test]
fn sorting_keeps_identity() {
    let mut map = TrackedMap::new();
    map.insert("a", 1);
    let b = map.insert_handle("b", 2);
    map.insert("c", 3);

    // 按值逆序：c, b, a
    map.sort_unstable_by(|_k1, v1, _k2, v2| v2.cmp(v1));
    assert_eq!(map.keys().copied().collect::<Vec<_>>(), vec!["c", "b", "a"]);
    assert_eq!(map.index_of_handle(&b), Some(1));
    assert_eq!(map.get_by_handle(&b), Some(&2));

    // 再按键正序排回来，句柄依旧有效。
    map.sort_by(|k1, _v1, k2, _v2| k1.cmp(k2));
    assert_eq!(map.keys().copied().collect::<Vec<_>>(), vec!["a", "b", "c"]);
    assert_eq!(map.index_of_handle(&b), Some(1));
    assert!(map.is_handle_valid(&b));

    // 通过句柄可变访问，改值不改变身份。
    *map.get_by_handle_mut(&b).unwrap() = 22;
    assert_eq!(map.get_by_handle(&b), Some(&22));
}

/// 删 b 再插 b 是新代次：旧句柄必须失效，新句柄才能用。
#[test]
fn remove_and_reinsert_invalidates_old_handle() {
    let mut map = TrackedMap::new();
    map.insert("a", 1);
    let b = map.insert_handle("b", 2);
    map.insert("c", 3);

    assert!(map.is_handle_valid(&b));
    assert_eq!(map.shift_remove_handle(&b), Some(2));
    assert!(!map.is_handle_valid(&b));

    let b2 = map.insert_handle("b", 200);
    assert_eq!(map.get(&"b"), Some(&200));
    assert_eq!(map.get_by_handle(&b), None);
    assert!(!map.is_handle_valid(&b));
    assert_eq!(map.get_by_handle(&b2), Some(&200));
    assert!(map.is_handle_valid(&b2));

    // 旧句柄也不能被拿来删东西。
    assert_eq!(map.shift_remove_handle(&b), None);
    assert_eq!(map.len(), 3);

    // 两个句柄不是同一个身份。
    assert_ne!(b, b2);
}

/// 别的容器不接受：即使键相同、顺序相同，跨容器句柄一律拒绝。
#[test]
fn foreign_container_handles_are_rejected() {
    let mut m1 = TrackedMap::new();
    let mut m2 = TrackedMap::new();
    m1.insert("a", 1);
    let h1 = m1.insert_handle("b", 2);
    m1.insert("c", 3);

    m2.insert("a", 1);
    let h2 = m2.insert_handle("b", 2);
    m2.insert("c", 3);

    assert_eq!(m1.index_of_handle(&h2), None);
    assert_eq!(m2.index_of_handle(&h1), None);
    assert_eq!(m1.get_by_handle(&h2), None);
    assert_eq!(m2.get_by_handle(&h1), None);
    assert!(!m1.is_handle_valid(&h2));
    assert_eq!(m1.swap_remove_handle(&h2), None);
    assert_eq!(m1.len(), 3);
    assert_eq!(m2.get_by_handle(&h2), Some(&2));
}

/// 移动容器本身不会使句柄失效（身份锚定在堆上的容器分配）。
#[test]
fn moving_container_keeps_handles_valid() {
    let mut map = TrackedMap::new();
    let b = map.insert_handle("b", 2);
    let moved = map;
    assert!(moved.is_handle_valid(&b));
    assert_eq!(moved.get_by_handle(&b), Some(&2));
}

/// 值不要求 Clone：用不可克隆的值走完整个句柄流程。
#[test]
fn values_need_not_be_clone() {
    struct NotClone(i32);

    let mut map = TrackedMap::new();
    let h = map.insert_handle("k", NotClone(7));
    map.insert("other", NotClone(8));

    map.shift_remove(&"other");
    map.sort_by(|k1, _, k2, _| k1.cmp(k2));

    assert_eq!(map.get_by_handle(&h).unwrap().0, 7);
    let v = map.swap_remove_handle(&h).unwrap();
    assert_eq!(v.0, 7);
}

/// 三键例子：完整走一遍验收叙事。
#[test]
fn three_key_acceptance_scenario() {
    let mut map: TrackedMap<&str, i32> = TrackedMap::new();
    map.insert("a", 1);
    let b = map.insert_handle("b", 2);
    map.insert("c", 3);

    // 取 b 句柄后移除 a，仍定位 b（只是索引变了）。
    map.shift_remove(&"a");
    assert_eq!(map.index_of_handle(&b), Some(0));
    assert_eq!(map.get_by_handle(&b), Some(&2));

    // 重排不换身份。
    map.sort_unstable_by(|k1, _, k2, _| k2.cmp(k1));
    assert_eq!(map.keys().copied().collect::<Vec<_>>(), vec!["c", "b"]);
    assert_eq!(map.get_by_handle(&b), Some(&2));

    // 删 b 再插 b：旧句柄失效。
    map.shift_remove(&"b");
    map.insert("b", 20);
    assert_eq!(map.get_by_handle(&b), None);

    // 跨容器拒绝。
    let mut other = TrackedMap::<&str, i32>::new();
    let other_b = other.insert_handle("b", 2);
    assert_eq!(map.get_by_handle(&other_b), None);
    assert_eq!(other.get_by_handle(&other_b), Some(&2));
}

/// 只替换值不算新代次，身份不变；clear 之后旧句柄一律失效。
#[test]
fn replacing_value_keeps_identity_but_clear_invalidates() {
    let mut map = TrackedMap::new();
    let b = map.insert_handle("b", 2);
    map.insert("b", 22);
    assert_eq!(map.get_by_handle(&b), Some(&22));

    let again = map.insert_handle("b", 222);
    assert_eq!(again, b);
    assert_eq!(map.get_by_handle(&b), Some(&222));

    map.clear();
    assert!(!map.is_handle_valid(&b));
    map.insert("b", 2);
    assert_eq!(map.get_by_handle(&b), None);
}

/// handle_of 与 entry 视图发出的句柄遵循同样的代次规则。
#[test]
fn handle_of_and_entry_api() {
    let mut map = TrackedMap::new();
    let (h, v) = map.entry("x").or_insert_handle(1);
    assert_eq!(*v, 1);
    *v = 9;
    assert_eq!(map.get_by_handle(&h), Some(&9));
    assert_eq!(map.handle_of(&"x"), Some(h.clone()));

    map.entry("x").and_modify(|n| *n += 1).or_insert(0);
    assert_eq!(map.get(&"x"), Some(&10));
    assert!(map.is_handle_valid(&h));
}
