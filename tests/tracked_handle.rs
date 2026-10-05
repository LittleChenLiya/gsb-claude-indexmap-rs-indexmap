//! 显式 Entry 句柄层的验收测试（三键示例）。
//!
//! 运行方式（定向命令）：
//!
//! ```sh
//! cargo test --test tracked_handle
//! ```
//!
//! 本测试使用三个键 `a`、`b`、`c` 演示并验收以下要求：
//!
//! 1. 取 `b` 句柄后移除 `a`，仍能定位 `b`（不依赖旧下标）；
//! 2. `sort` / `shift_remove` / `swap_remove` 等重排或移除不改变项身份；
//! 3. 删除 `b` 后再插入 `b` 是新代次，旧句柄失效；
//! 4. 跨容器拒绝：别的容器不接受本容器的句柄；
//! 5. 值类型不要求 `Clone`。

use indexmap::tracked::TrackedIndexMap;

/// 三键示例：`a`、`b`、`c`。
///
/// 覆盖验收要求的完整流程。
#[test]
fn three_key_example() {
    let mut map = TrackedIndexMap::new();
    map.insert("a", 1);
    map.insert("b", 2);
    map.insert("c", 3);

    // 取 b 的句柄
    let hb = map.handle(&"b").unwrap();

    // 【验收 1】移除 a 后仍能通过句柄定位 b
    map.shift_remove(&"a");
    assert_eq!(map.get_handle(&hb), Some(&2));

    // 【验收 2】重排不改变项身份，句柄仍有效
    map.sort_keys();
    assert_eq!(map.get_handle(&hb), Some(&2));

    // swap_remove c 后 b 仍可定位
    map.swap_remove(&"c");
    assert_eq!(map.get_handle(&hb), Some(&2));

    // 【验收 3】删除 b 再插入 b：旧句柄失效
    map.shift_remove(&"b");
    assert_eq!(map.get_handle(&hb), None);
    map.insert("b", 20);
    assert_eq!(map.get_handle(&hb), None);

    // 新句柄可以定位新值
    let hb2 = map.handle(&"b").unwrap();
    assert_eq!(map.get_handle(&hb2), Some(&20));

    // 【验收 4】跨容器拒绝
    let mut other = TrackedIndexMap::new();
    other.insert("b", 2);
    assert_eq!(other.get_handle(&hb), None);
    assert_eq!(other.get_handle(&hb2), None);

    // 原容器也不接受别的容器的句柄
    let hb_other = other.handle(&"b").unwrap();
    assert_eq!(map.get_handle(&hb_other), None);
}

/// 验收 1 的独立用例：取 b 句柄后移除 a，仍定位 b。
#[test]
fn handle_survives_removal_of_other_keys() {
    let mut map = TrackedIndexMap::new();
    map.insert("a", 1);
    map.insert("b", 2);
    map.insert("c", 3);

    let hb = map.handle(&"b").unwrap();

    // 分别用 shift_remove 和 swap_remove 移除 a，b 的句柄都应有效
    map.shift_remove(&"a");
    assert_eq!(map.get_handle(&hb), Some(&2));

    map.insert("a", 1);
    map.swap_remove(&"a");
    assert_eq!(map.get_handle(&hb), Some(&2));
}

/// 验收 2 的独立用例：sort / reverse 不换项身份。
#[test]
fn reorder_keeps_identity() {
    let mut map = TrackedIndexMap::new();
    map.insert("a", 1);
    map.insert("b", 2);
    map.insert("c", 3);

    let handles: Vec<_> = ["a", "b", "c"]
        .iter()
        .map(|k| map.handle(k).unwrap())
        .collect();

    // 按值降序排序
    map.sort_by(|k1, _, k2, _| k2.cmp(k1));
    for (h, expected) in handles.iter().zip([1, 2, 3].iter()) {
        assert_eq!(map.get_handle(h), Some(expected));
    }

    // 反转
    map.reverse();
    for (h, expected) in handles.iter().zip([1, 2, 3].iter()) {
        assert_eq!(map.get_handle(h), Some(expected));
    }

    // sort_unstable_keys
    map.sort_unstable_keys();
    for (h, expected) in handles.iter().zip([1, 2, 3].iter()) {
        assert_eq!(map.get_handle(h), Some(expected));
    }
}

/// 验收 3 的独立用例：删除再插入同键是新代次，旧句柄失效。
#[test]
fn delete_and_reinsert_changes_generation() {
    let mut map = TrackedIndexMap::new();
    map.insert("b", 2);

    let hb = map.handle(&"b").unwrap();
    assert_eq!(map.get_handle(&hb), Some(&2));

    // 删除后、重新插入前：句柄已失效
    map.shift_remove(&"b");
    assert_eq!(map.get_handle(&hb), None);

    // 重新插入同键：新代次，旧句柄仍然失效
    map.insert("b", 20);
    assert_eq!(map.get_handle(&hb), None);

    // 新句柄有效
    let hb2 = map.handle(&"b").unwrap();
    assert_eq!(map.get_handle(&hb2), Some(&20));

    // 再次删除并插入，新句柄也失效
    map.swap_remove(&"b");
    assert_eq!(map.get_handle(&hb2), None);
    map.insert("b", 200);
    assert_eq!(map.get_handle(&hb2), None);
    let hb3 = map.handle(&"b").unwrap();
    assert_eq!(map.get_handle(&hb3), Some(&200));
}

/// 验收 4 的独立用例：跨容器拒绝。
#[test]
fn cross_container_rejection() {
    let mut map = TrackedIndexMap::new();
    map.insert("b", 2);
    let hb = map.handle(&"b").unwrap();

    // 另一个空容器
    let mut other = TrackedIndexMap::new();
    assert_eq!(other.get_handle(&hb), None);

    // 另一个容器有相同的键和值
    other.insert("b", 2);
    assert_eq!(other.get_handle(&hb), None);

    // 反向也拒绝
    let hb_other = other.handle(&"b").unwrap();
    assert_eq!(map.get_handle(&hb_other), None);

    // 克隆体也是不同容器，原句柄在克隆体上无效
    let cloned = map.clone();
    assert_eq!(cloned.get_handle(&hb), None);
    // 克隆体自己的句柄在克隆体上有效
    let hb_cloned = cloned.handle(&"b").unwrap();
    assert_eq!(cloned.get_handle(&hb_cloned), Some(&2));
}

/// 验收 5：值类型不要求 `Clone`。
#[test]
fn values_do_not_require_clone() {
    // 不实现 Clone 的值类型
    struct NonClone(i32);

    let mut map = TrackedIndexMap::new();
    map.insert("a", NonClone(1));
    map.insert("b", NonClone(2));

    let ha = map.handle(&"a").unwrap();
    let hb = map.handle(&"b").unwrap();

    assert_eq!(map.get_handle(&ha).unwrap().0, 1);
    assert_eq!(map.get_handle(&hb).unwrap().0, 2);

    // 通过句柄可变修改值
    map.get_mut_handle(&ha).unwrap().0 = 10;
    assert_eq!(map.get_handle(&ha).unwrap().0, 10);

    // 移除 b 后 a 的句柄仍有效
    map.shift_remove(&"b");
    assert_eq!(map.get_handle(&ha).unwrap().0, 10);
}

/// 通过句柄可变访问后，重排不影响句柄有效性。
#[test]
fn mutable_access_then_reorder() {
    let mut map = TrackedIndexMap::new();
    map.insert("a", 1);
    map.insert("b", 2);
    map.insert("c", 3);

    let hb = map.handle(&"b").unwrap();
    *map.get_mut_handle(&hb).unwrap() += 10;
    assert_eq!(map.get_handle(&hb), Some(&12));

    map.sort_by(|k1, _, k2, _| k1.cmp(k2));
    assert_eq!(map.get_handle(&hb), Some(&12));

    map.reverse();
    assert_eq!(map.get_handle(&hb), Some(&12));
}
