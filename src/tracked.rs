//! 显式 Entry 句柄层：为 [`IndexMap`] 提供绑定「容器 + 键代次」的句柄。
//!
//! 普通 [`IndexMap`] 通过键或下标访问元素。下标会随着 [`sort`][IndexMap::sort_by]、
//! [`shift_remove`][IndexMap::shift_remove]、[`swap_remove`][IndexMap::swap_remove]
//! 等操作而改变，键本身也可能被删除后重新插入。本模块在不改动普通 `IndexMap`
//! 接口的前提下，提供一层包装：
//!
//! - 每个容器拥有唯一的容器标识（container id）；
//! - 每个槽位（entry slot）拥有随插入递增的代次（generation）；
//! - [`Handle`] 同时绑定容器标识、键与代次。
//!
//! 因此：
//!
//! - 对 `b` 取句柄后，即使移除了 `a`，仍能通过句柄定位到 `b`；
//! - 删除 `b` 后再插入 `b` 是新代次，旧句柄失效；
//! - 别的容器无法接受本容器的句柄（容器标识不匹配）；
//! - `sort` / `reverse` 等重排不改变任何槽位的代次，句柄仍然有效。
//!
//! 包装层不暴露未跟踪的可变内部结构：所有结构性变更（插入、移除、清空）都必须
//! 经过本层的方法，代次随之维护；值类型 `V` 不要求 [`Clone`]。
//!
//! # 三键示例
//!
//! ```
//! use indexmap::tracked::TrackedIndexMap;
//!
//! let mut map = TrackedIndexMap::new();
//! map.insert("a", 1);
//! map.insert("b", 2);
//! map.insert("c", 3);
//!
//! // 取 b 的句柄
//! let hb = map.handle(&"b").unwrap();
//!
//! // 移除 a 后仍能通过句柄定位 b
//! map.shift_remove(&"a");
//! assert_eq!(map.get_handle(&hb), Some(&2));
//!
//! // 重排不改变项身份，句柄仍有效
//! map.sort_keys();
//! assert_eq!(map.get_handle(&hb), Some(&2));
//!
//! // swap_remove c 后 b 仍可定位
//! map.swap_remove(&"c");
//! assert_eq!(map.get_handle(&hb), Some(&2));
//!
//! // 删除 b 再插入 b：旧句柄失效
//! map.shift_remove(&"b");
//! map.insert("b", 20);
//! assert_eq!(map.get_handle(&hb), None);
//!
//! // 新句柄可以定位新值
//! let hb2 = map.handle(&"b").unwrap();
//! assert_eq!(map.get_handle(&hb2), Some(&20));
//!
//! // 跨容器拒绝
//! let mut other = TrackedIndexMap::new();
//! other.insert("b", 2);
//! assert_eq!(other.get_handle(&hb), None);
//! assert_eq!(other.get_handle(&hb2), None);
//! ```

use core::cmp::Ordering;
use core::fmt;
use core::hash::{BuildHasher, Hash};
use core::ops::{Index, IndexMut};
use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::{Equivalent, IndexMap};

/// 全局容器标识计数器，保证每个 [`TrackedIndexMap`] 拥有唯一的容器标识。
static NEXT_CONTAINER_ID: AtomicU64 = AtomicU64::new(1);

fn next_container_id() -> u64 {
    NEXT_CONTAINER_ID.fetch_add(1, AtomicOrdering::Relaxed)
}

/// 槽位：值与代次。
///
/// 这是 `TrackedIndexMap` 内部使用的不透明类型，字段不对外暴露。
#[derive(Clone)]
pub struct Slot<V> {
    value: V,
    generation: u64,
}

/// 绑定了「容器 + 键 + 代次」的显式句柄。
///
/// 句柄在创建后不依赖下标：即使容器被重排（[`sort`][TrackedIndexMap::sort_by]、
/// [`reverse`][TrackedIndexMap::reverse]）或其他元素被移除，只要该键的槽位仍然
/// 是同一个代次，句柄就仍然有效。
///
/// 删除键后重新插入会产生新代次，此时旧句柄失效；句柄也无法在别的容器上使用。
#[derive(Clone)]
pub struct Handle<K> {
    container_id: u64,
    key: K,
    generation: u64,
}

impl<K> Handle<K> {
    /// 返回句柄绑定的键。
    pub fn key(&self) -> &K {
        &self.key
    }
}

impl<K: fmt::Debug> fmt::Debug for Handle<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("container_id", &self.container_id)
            .field("key", &self.key)
            .field("generation", &self.generation)
            .finish()
    }
}

/// 提供显式 Entry 句柄层的有序映射包装。
///
/// 详见[模块文档][crate::tracked]。
#[cfg(feature = "std")]
pub struct TrackedIndexMap<K, V, S = std::hash::RandomState> {
    inner: IndexMap<K, Slot<V>, S>,
    container_id: u64,
    next_generation: u64,
}

/// 提供显式 Entry 句柄层的有序映射包装。
///
/// 详见[模块文档][crate::tracked]。
#[cfg(not(feature = "std"))]
pub struct TrackedIndexMap<K, V, S> {
    inner: IndexMap<K, Slot<V>, S>,
    container_id: u64,
    next_generation: u64,
}

#[cfg(feature = "std")]
impl<K, V> TrackedIndexMap<K, V> {
    /// 创建一个新的空映射。
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// 创建一个能容纳 `n` 个元素的空映射。
    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        Self::with_capacity_and_hasher(n, <_>::default())
    }
}

impl<K, V, S> TrackedIndexMap<K, V, S> {
    /// 使用给定的 `hash_builder` 创建一个新的空映射。
    pub fn with_hasher(hash_builder: S) -> Self {
        TrackedIndexMap {
            inner: IndexMap::with_hasher(hash_builder),
            container_id: next_container_id(),
            next_generation: 1,
        }
    }

    /// 使用给定的 `hash_builder` 创建一个能容纳 `n` 个元素的空映射。
    pub fn with_capacity_and_hasher(n: usize, hash_builder: S) -> Self {
        TrackedIndexMap {
            inner: IndexMap::with_capacity_and_hasher(n, hash_builder),
            container_id: next_container_id(),
            next_generation: 1,
        }
    }

    /// 返回映射中的元素个数。
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 返回映射是否为空。
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// 返回映射的容量。
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// 返回映射使用的哈希构造器。
    pub fn hasher(&self) -> &S {
        self.inner.hasher()
    }

    /// 清空映射。
    ///
    /// 清空后所有旧句柄都将失效（槽位已不存在，且代次计数也会递增）。
    pub fn clear(&mut self) {
        self.inner.clear();
        self.next_generation += 1;
    }
}

impl<K, V, S> TrackedIndexMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// 插入一个键值对。
    ///
    /// - 若键已存在，则原地替换值并返回旧值，槽位代次不变（项身份不变）；
    /// - 若键不存在，则创建新槽位并分配新代次。
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(slot) = self.inner.get_mut(&key) {
            let generation = slot.generation;
            let old = core::mem::replace(slot, Slot { value, generation });
            return Some(old.value);
        }
        let generation = self.next_generation;
        self.next_generation += 1;
        self.inner.insert(key, Slot { value, generation });
        None
    }

    /// 返回键对应值的引用。
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        self.inner.get(key).map(|slot| &slot.value)
    }

    /// 返回键对应值的可变引用。
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        self.inner.get_mut(key).map(|slot| &mut slot.value)
    }

    /// 返回键对应的键值对引用。
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        self.inner
            .get_key_value(key)
            .map(|(k, slot)| (k, &slot.value))
    }

    /// 为 `key` 创建一个显式句柄。
    ///
    /// 若键不存在则返回 `None`。句柄绑定当前容器与该键槽位的代次。
    pub fn handle(&self, key: &K) -> Option<Handle<K>>
    where
        K: Clone,
    {
        let (k, slot) = self.inner.get_key_value(key)?;
        Some(Handle {
            container_id: self.container_id,
            key: k.clone(),
            generation: slot.generation,
        })
    }

    /// 通过句柄定位值。
    ///
    /// 以下情况返回 `None`：
    ///
    /// - 句柄来自别的容器（容器标识不匹配）；
    /// - 句柄绑定的键已不存在；
    /// - 键被删除后重新插入，槽位代次已更新（旧句柄失效）。
    pub fn get_handle(&self, handle: &Handle<K>) -> Option<&V> {
        if handle.container_id != self.container_id {
            return None;
        }
        let slot = self.inner.get(&handle.key)?;
        if slot.generation != handle.generation {
            return None;
        }
        Some(&slot.value)
    }

    /// 通过句柄定位值的可变引用。
    ///
    /// 失效规则与 [`get_handle`](Self::get_handle) 相同。
    pub fn get_mut_handle(&mut self, handle: &Handle<K>) -> Option<&mut V> {
        if handle.container_id != self.container_id {
            return None;
        }
        let slot = self.inner.get_mut(&handle.key)?;
        if slot.generation != handle.generation {
            return None;
        }
        Some(&mut slot.value)
    }

    /// 移除键对应的值（与 [`swap_remove`](Self::swap_remove) 语义相同）。
    ///
    /// 被移除键的旧句柄将失效。
    #[deprecated(note = "`remove` 会扰动映射顺序，建议显式使用 `swap_remove` 或 `shift_remove`")]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        self.swap_remove(key)
    }

    /// 移除键对应的值，并用映射末尾元素填补其位置。
    ///
    /// 不影响其他键的槽位代次：对 `b` 的句柄在移除 `a` 后仍然有效。
    pub fn swap_remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        self.inner.swap_remove(key).map(|slot| slot.value)
    }

    /// 移除键对应的值，并保持其余元素的相对顺序。
    ///
    /// 不影响其他键的槽位代次。
    pub fn shift_remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: ?Sized + Hash + Equivalent<K>,
    {
        self.inner.shift_remove(key).map(|slot| slot.value)
    }

    /// 移除并返回映射末尾的键值对。
    pub fn pop(&mut self) -> Option<(K, V)> {
        self.inner.pop().map(|(k, slot)| (k, slot.value))
    }

    /// 按比较函数对键值对排序。
    ///
    /// 重排不改变任何槽位的代次，已有句柄仍然有效。
    pub fn sort_by<F>(&mut self, mut cmp: F)
    where
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        self.inner
            .sort_by(|k1, s1, k2, s2| cmp(k1, &s1.value, k2, &s2.value));
    }

    /// 按键的默认顺序排序。
    ///
    /// 重排不改变任何槽位的代次。
    pub fn sort_keys(&mut self)
    where
        K: Ord,
    {
        self.inner.sort_keys();
    }

    /// 按键的默认顺序排序（不保证相等元素的顺序）。
    ///
    /// 重排不改变任何槽位的代次。
    pub fn sort_unstable_keys(&mut self)
    where
        K: Ord,
    {
        self.inner.sort_unstable_keys();
    }

    /// 按 sort-key 提取函数排序。
    ///
    /// 重排不改变任何槽位的代次。
    pub fn sort_by_key<T, F>(&mut self, mut sort_key: F)
    where
        T: Ord,
        F: FnMut(&K, &V) -> T,
    {
        self.inner.sort_by_key(|k, s| sort_key(k, &s.value));
    }

    /// 反转映射中元素的顺序。
    ///
    /// 重排不改变任何槽位的代次。
    pub fn reverse(&mut self) {
        self.inner.reverse();
    }

    /// 返回键值对的迭代器。
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.inner.iter().map(|(k, slot)| (k, &slot.value))
    }

    /// 返回键值对的可变迭代器。
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.inner
            .iter_mut()
            .map(|(k, slot)| (k, &mut slot.value))
    }

    /// 返回键的迭代器。
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.inner.keys()
    }

    /// 返回值的迭代器。
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.inner.values().map(|slot| &slot.value)
    }

    /// 返回值的可变迭代器。
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.inner.values_mut().map(|slot| &mut slot.value)
    }
}

impl<K, V, S> Default for TrackedIndexMap<K, V, S>
where
    S: Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, S> Clone for TrackedIndexMap<K, V, S>
where
    K: Clone,
    V: Clone,
    S: Clone,
{
    fn clone(&self) -> Self {
        // 克隆体是一个全新的容器：分配新的容器标识，
        // 从句柄的角度看，原容器的句柄在克隆体上无效。
        TrackedIndexMap {
            inner: self.inner.clone(),
            container_id: next_container_id(),
            next_generation: self.next_generation,
        }
    }
}

impl<K, V, S> fmt::Debug for TrackedIndexMap<K, V, S>
where
    K: fmt::Debug + Hash + Eq,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S, Q> Index<&Q> for TrackedIndexMap<K, V, S>
where
    Q: ?Sized + Hash + Equivalent<K>,
    K: Hash + Eq,
    S: BuildHasher,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, V, S, Q> IndexMut<&Q> for TrackedIndexMap<K, V, S>
where
    Q: ?Sized + Hash + Equivalent<K>,
    K: Hash + Eq,
    S: BuildHasher,
{
    fn index_mut(&mut self, key: &Q) -> &mut V {
        self.get_mut(key).expect("no entry found for key")
    }
}

impl<K, V, S> Extend<(K, V)> for TrackedIndexMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V, S> FromIterator<(K, V)> for TrackedIndexMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Self::with_hasher(S::default());
        map.extend(iter);
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn handle_survives_removal_of_other_keys() {
        let mut map = TrackedIndexMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        let hb = map.handle(&"b").unwrap();
        map.shift_remove(&"a");
        assert_eq!(map.get_handle(&hb), Some(&2));

        map.swap_remove(&"c");
        assert_eq!(map.get_handle(&hb), Some(&2));
    }

    #[test]
    fn handle_invalidated_by_delete_and_reinsert() {
        let mut map = TrackedIndexMap::new();
        map.insert("b", 2);

        let hb = map.handle(&"b").unwrap();
        assert_eq!(map.get_handle(&hb), Some(&2));

        map.shift_remove(&"b");
        assert_eq!(map.get_handle(&hb), None);

        map.insert("b", 20);
        assert_eq!(map.get_handle(&hb), None);

        let hb2 = map.handle(&"b").unwrap();
        assert_eq!(map.get_handle(&hb2), Some(&20));
    }

    #[test]
    fn handle_rejected_by_other_container() {
        let mut map = TrackedIndexMap::new();
        map.insert("b", 2);
        let hb = map.handle(&"b").unwrap();

        let mut other = TrackedIndexMap::new();
        other.insert("b", 2);
        assert_eq!(other.get_handle(&hb), None);

        // 即使另一个容器也有相同的键和值，容器标识不同，仍然拒绝。
        let hb_other = other.handle(&"b").unwrap();
        assert_eq!(map.get_handle(&hb_other), None);
    }

    #[test]
    fn sort_and_reverse_preserve_identity() {
        let mut map = TrackedIndexMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        let handles: Vec<_> = ["a", "b", "c"]
            .iter()
            .map(|k| map.handle(k).unwrap())
            .collect();

        map.sort_by(|k1, _, k2, _| k2.cmp(k1));
        for (h, expected) in handles.iter().zip([1, 2, 3].iter()) {
            assert_eq!(map.get_handle(h), Some(expected));
        }

        map.reverse();
        for (h, expected) in handles.iter().zip([1, 2, 3].iter()) {
            assert_eq!(map.get_handle(h), Some(expected));
        }
    }

    #[test]
    fn occupied_insert_keeps_generation() {
        let mut map = TrackedIndexMap::new();
        map.insert("b", 2);
        let hb = map.handle(&"b").unwrap();

        assert_eq!(map.insert("b", 20), Some(2));
        assert_eq!(map.get_handle(&hb), Some(&20));
    }

    #[test]
    fn works_without_clone_on_values() {
        // 一个不实现 Clone 的值类型
        struct NonClone(i32);

        let mut map = TrackedIndexMap::new();
        map.insert("a", NonClone(1));
        let h = map.handle(&"a").unwrap();
        assert_eq!(map.get_handle(&h).unwrap().0, 1);
        map.get_mut_handle(&h).unwrap().0 = 42;
        assert_eq!(map.get_handle(&h).unwrap().0, 42);
    }
}
