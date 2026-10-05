//! Explicit entry handles with stable item identity for [`IndexMap`].
//!
//! This module offers a thin wrapper around [`IndexMap`], [`TrackedMap`], that
//! hands out [`EntryHandle`]s. A handle identifies a *real item* by three
//! things:
//!
//! 1. the container it was issued by (an identity tag on the heap allocation
//!    behind the wrapper, so moving the [`TrackedMap`] does not invalidate
//!    outstanding handles);
//! 2. the item's key; and
//! 3. the key's *generation* -- a counter that is bumped every time a key is
//!    inserted as a genuinely new item.
//!
//! Because every lookup through a handle resolves the current index of the key
//! at use time, reordering the map does not change item identity: handles keep
//! working after [`sort_by`](TrackedMap::sort_by),
//! [`shift_remove`](TrackedMap::shift_remove) of other items or
//! [`swap_remove`](TrackedMap::swap_remove) of other items, even though the
//! numeric indices of the items may change. A handle is **not** a cached
//! index.
//!
//! Removing a key and inserting that same key again starts a new generation,
//! so handles obtained before the removal become stale and resolve to `None`.
//! A handle issued by one [`TrackedMap`] is never accepted by another one.
//!
//! The wrapper never hands out an untracked mutable reference to the inner
//! map, and stored values do not need to implement [`Clone`] (only the
//! internally shared key is cheaply reference-counted).
//!
//! The ordinary [`IndexMap`] interface is untouched; all of this lives in this
//! separate module.
//!
//! # Three-key example
//!
//! ```
//! use indexmap::tracked::TrackedMap;
//!
//! let mut map: TrackedMap<&str, i32> = TrackedMap::new();
//! map.insert("a", 1);
//! let b = map.insert_handle("b", 2);
//! map.insert("c", 3);
//!
//! // Removing `a` shifts `b` to a different numeric index, but the handle
//! // still reaches the same real item:
//! map.shift_remove(&"a");
//! assert_eq!(map.get_by_handle(&b), Some(&2));
//! assert_eq!(map.index_of_handle(&b), Some(0));
//!
//! // Reordering never changes identity either:
//! map.sort_unstable_by(|k1, _, k2, _| k2.cmp(k1));
//! assert_eq!(map.index_of_handle(&b), Some(1));
//! assert_eq!(map.get_by_handle(&b), Some(&2));
//!
//! // Delete `b` and insert that key anew: the old generation is gone, so the
//! // old handle can no longer locate anything.
//! map.shift_remove(&"b");
//! map.insert("b", 20);
//! assert_eq!(map.get_by_handle(&b), None);
//!
//! // A handle belonging to another container is rejected outright, even with
//! // a matching key and generation.
//! let mut other = TrackedMap::<&str, i32>::new();
//! let other_b = other.insert_handle("b", 2);
//! assert_eq!(map.get_by_handle(&other_b), None);
//! ```
//!
//! # Running the focused tests
//!
//! ```text
//! cargo test --test tracked_handles
//! cargo test --doc tracked
//! ```

use crate::IndexMap;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{BuildHasher, Hash};

#[cfg(feature = "std")]
use std::hash::RandomState;

/// Per-slot bookkeeping kept in the same order as the entries of the inner map.
#[derive(Clone, Copy, Debug)]
struct Slot {
    generation: u64,
}

/// The stable heap home of a tracked map's data.
///
/// Boxing this means its address stays fixed for the whole lifetime of the
/// [`TrackedMap`], even when the wrapper itself is moved.
struct Inner<K, V, S> {
    map: IndexMap<Arc<K>, V, S>,
    slots: Vec<Slot>,
    next_generation: u64,
}

/// An [`IndexMap`] wrapper that issues identity-based [`EntryHandle`]s.
///
/// See the [module documentation](self) for details.
#[cfg(feature = "std")]
pub struct TrackedMap<K, V, S = RandomState> {
    inner: Box<Inner<K, V, S>>,
}

/// An [`IndexMap`] wrapper that issues identity-based [`EntryHandle`]s.
///
/// See the [module documentation](self) for details.
#[cfg(not(feature = "std"))]
pub struct TrackedMap<K, V, S> {
    inner: Box<Inner<K, V, S>>,
}

impl<K, V, S> TrackedMap<K, V, S>
where
    S: BuildHasher,
{
    /// Creates an empty `TrackedMap` with the given hasher.
    pub fn with_hasher(hash_builder: S) -> Self {
        Self::with_capacity_and_hasher(0, hash_builder)
    }

    /// Creates an empty `TrackedMap` with the specified capacity and hasher.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            inner: Box::new(Inner {
                map: IndexMap::with_capacity_and_hasher(capacity, hash_builder),
                slots: Vec::with_capacity(capacity),
                next_generation: 0,
            }),
        }
    }

    /// Returns the container's identity tag, derived from the stable address
    /// of its heap allocation. The address is only ever compared, never
    /// dereferenced.
    fn owner_tag(&self) -> usize {
        let inner: &Inner<K, V, S> = &self.inner;
        (inner as *const Inner<K, V, S> as *const ()) as usize
    }

    /// Allocates the generation of the next genuinely new key.
    fn fresh_generation(&mut self) -> u64 {
        let generation = self.inner.next_generation;
        self.inner.next_generation = self.inner.next_generation.wrapping_add(1);
        generation
    }

    /// Builds a handle for an existing slot.
    fn handle_at(&self, index: usize) -> EntryHandle<K> {
        let key = Arc::clone(self.inner.map.get_index(index).expect("valid index").0);
        EntryHandle {
            owner: self.owner_tag(),
            generation: self.inner.slots[index].generation,
            key,
        }
    }

    /// Resolves a handle to the index of its live item, rejecting stale and
    /// foreign handles.
    fn live_index(&self, handle: &EntryHandle<K>) -> Option<usize>
    where
        K: Eq + Hash,
    {
        if handle.owner != self.owner_tag() {
            return None;
        }
        let index = self.inner.map.get_index_of(&*handle.key)?;
        if self.inner.slots[index].generation != handle.generation {
            return None;
        }
        Some(index)
    }

    /// Returns `true` if the handle still points at a live item in *this* map.
    pub fn is_handle_valid(&self, handle: &EntryHandle<K>) -> bool
    where
        K: Eq + Hash,
    {
        self.live_index(handle).is_some()
    }

    /// Number of items currently tracked.
    pub fn len(&self) -> usize {
        self.inner.map.len()
    }

    /// Returns `true` if the map contains no items.
    pub fn is_empty(&self) -> bool {
        self.inner.map.is_empty()
    }

    /// Returns the map's reserved capacity.
    pub fn capacity(&self) -> usize {
        self.inner.map.capacity()
    }

    /// Removes all items. All outstanding handles become invalid.
    pub fn clear(&mut self) {
        self.inner.map.clear();
        self.inner.slots.clear();
    }

    /// Looks up a value by key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        Arc<K>: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.map.get(key)
    }

    /// Looks up a value by key, mutably.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        Arc<K>: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.map.get_mut(key)
    }

    /// Looks up the key-value pair by key.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        Arc<K>: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner
            .map
            .get_key_value(key)
            .map(|(stored, value)| (&**stored, value))
    }

    /// Returns `true` if the map contains the given key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Arc<K>: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.map.contains_key(key)
    }

    /// Returns a handle for an existing key, if present.
    pub fn handle_of<Q>(&self, key: &Q) -> Option<EntryHandle<K>>
    where
        Arc<K>: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.inner.map.get_index_of(key)?;
        Some(self.handle_at(index))
    }

    /// Resolves a handle to its current numeric index.
    pub fn index_of_handle(&self, handle: &EntryHandle<K>) -> Option<usize>
    where
        K: Eq + Hash,
    {
        self.live_index(handle)
    }

    /// Fetches the value of a handle's item.
    pub fn get_by_handle(&self, handle: &EntryHandle<K>) -> Option<&V>
    where
        K: Eq + Hash,
    {
        let index = self.live_index(handle)?;
        self.inner.map.get_index(index).map(|(_, value)| value)
    }

    /// Fetches the value of a handle's item, mutably.
    pub fn get_by_handle_mut(&mut self, handle: &EntryHandle<K>) -> Option<&mut V>
    where
        K: Eq + Hash,
    {
        let index = self.live_index(handle)?;
        self.inner.map.get_index_mut(index).map(|(_, value)| value)
    }

    /// Inserts a key-value pair, replacing the value if the key exists, and
    /// returns a handle to the item. Item identity is unchanged when only the
    /// value of a live key is replaced.
    pub fn insert_handle(&mut self, key: K, value: V) -> EntryHandle<K>
    where
        K: Eq + Hash,
    {
        match self.entry(key) {
            Entry::Occupied(mut entry) => {
                entry.insert(value);
                entry.into_handle()
            }
            Entry::Vacant(entry) => entry.insert_handle(value).0,
        }
    }

    /// Removes the handle's item with shift semantics, returning its value.
    pub fn shift_remove_handle(&mut self, handle: &EntryHandle<K>) -> Option<V>
    where
        K: Eq + Hash,
    {
        let index = self.live_index(handle)?;
        self.inner.slots.remove(index);
        self.inner
            .map
            .shift_remove_index(index)
            .map(|(_, value)| value)
    }

    /// Removes the handle's item with swap semantics, returning its value.
    pub fn swap_remove_handle(&mut self, handle: &EntryHandle<K>) -> Option<V>
    where
        K: Eq + Hash,
    {
        let index = self.live_index(handle)?;
        self.inner.slots.swap_remove(index);
        self.inner
            .map
            .swap_remove_index(index)
            .map(|(_, value)| value)
    }

    /// Removes a key with shift semantics, returning its value.
    pub fn shift_remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Arc<K>: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.inner.map.get_index_of(key)?;
        self.inner.slots.remove(index);
        self.inner
            .map
            .shift_remove_index(index)
            .map(|(_, value)| value)
    }

    /// Removes a key with swap semantics, returning its value.
    pub fn swap_remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Arc<K>: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let index = self.inner.map.get_index_of(key)?;
        self.inner.slots.swap_remove(index);
        self.inner
            .map
            .swap_remove_index(index)
            .map(|(_, value)| value)
    }

    /// Sorts the map's entries by key-value pair, preserving the identity of
    /// every item. This is a stable sort.
    pub fn sort_by<F>(&mut self, mut cmp: F)
    where
        K: Eq + Hash,
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        let before: Vec<Arc<K>> = self.inner.map.keys().cloned().collect();
        self.inner.map.sort_by(|ak, av, bk, bv| cmp(ak, av, bk, bv));
        self.relabel_slots(before);
    }

    /// Sorts the map's entries by key-value pair, preserving the identity of
    /// every item. This sort is not stable.
    pub fn sort_unstable_by<F>(&mut self, mut cmp: F)
    where
        K: Eq + Hash,
        F: FnMut(&K, &V, &K, &V) -> Ordering,
    {
        let before: Vec<Arc<K>> = self.inner.map.keys().cloned().collect();
        self.inner
            .map
            .sort_unstable_by(|ak, av, bk, bv| cmp(ak, av, bk, bv));
        self.relabel_slots(before);
    }

    /// Re-aligns the generation slots with the map after a reorder.
    ///
    /// `before[i]` is the key that used to be at index `i`.
    fn relabel_slots(&mut self, before: Vec<Arc<K>>)
    where
        K: Eq + Hash,
    {
        let count = before.len();
        let old_slots = core::mem::take(&mut self.inner.slots);
        let mut new_slots = vec![Slot { generation: 0 }; count];
        for (old_index, key) in before.into_iter().enumerate() {
            let new_index = self
                .inner
                .map
                .get_index_of(&key)
                .expect("every sorted key is still present");
            new_slots[new_index] = old_slots[old_index];
        }
        self.inner.slots = new_slots;
    }

    /// Iterates over key-value pairs in insertion order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&K, &V)> {
        self.inner.map.iter().map(|(key, value)| (&**key, value))
    }

    /// Iterates over key-value pairs in insertion order, with mutable values.
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = (&K, &mut V)> {
        self.inner
            .map
            .iter_mut()
            .map(|(key, value)| (&**key, value))
    }

    /// Iterates over the keys in insertion order.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> {
        self.inner.map.keys().map(|key| &**key)
    }

    /// Iterates over the values in insertion order.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.inner.map.values()
    }

    /// Iterates over the values in insertion order, mutably.
    pub fn values_mut(&mut self) -> impl ExactSizeIterator<Item = &mut V> {
        self.inner.map.values_mut()
    }
}

impl<K, V, S> TrackedMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Inserts a key-value pair. Returns the previous value if the key already
    /// existed; replacing a value keeps the item's existing identity.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let key = Arc::new(key);
        match self.inner.map.get_index_of(&key) {
            Some(_) => self.inner.map.insert(key, value),
            None => {
                let generation = self.fresh_generation();
                self.inner.slots.push(Slot { generation });
                self.inner.map.insert(key, value)
            }
        }
    }

    /// Gets the entry for in-place manipulation, vacant or occupied.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S> {
        let key = Arc::new(key);
        match self.inner.map.get_index_of(&key) {
            Some(index) => Entry::Occupied(OccupiedEntry {
                map: self,
                index,
                key,
            }),
            None => Entry::Vacant(VacantEntry { map: self, key }),
        }
    }
}

#[cfg(feature = "std")]
impl<K, V> TrackedMap<K, V, RandomState> {
    /// Creates an empty map with the default hasher.
    pub fn new() -> Self {
        Self::with_hasher(RandomState::default())
    }

    /// Creates an empty map with at least the specified capacity, using the
    /// default hasher.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::default())
    }
}

impl<K, V, S> Default for TrackedMap<K, V, S>
where
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, S> fmt::Debug for TrackedMap<K, V, S>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.inner.map.iter().map(|(key, value)| (&**key, value)))
            .finish()
    }
}

/// An identity-based handle to an item in a [`TrackedMap`].
///
/// Handles can be freely cloned and compared; they remain valid through
/// reordering of their container but become invalid when their key is removed
/// and re-inserted (a new generation) or when used against another container.
pub struct EntryHandle<K> {
    owner: usize,
    generation: u64,
    key: Arc<K>,
}

impl<K> Clone for EntryHandle<K> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner,
            generation: self.generation,
            key: Arc::clone(&self.key),
        }
    }
}

impl<K: fmt::Debug> fmt::Debug for EntryHandle<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EntryHandle")
            .field("generation", &self.generation)
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl<K: PartialEq> PartialEq for EntryHandle<K> {
    fn eq(&self, other: &Self) -> bool {
        self.owner == other.owner && self.generation == other.generation && *self.key == *other.key
    }
}

impl<K: Eq> Eq for EntryHandle<K> {}

impl<K> EntryHandle<K> {
    /// Returns the key this handle refers to.
    pub fn key(&self) -> &K {
        &self.key
    }
}

/// A view into an occupied or vacant entry of a [`TrackedMap`].
pub enum Entry<'a, K, V, S> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V, S>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V, S>),
}

impl<'a, K, V, S> Entry<'a, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Returns a reference to the entry's key.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(entry) => entry.key(),
            Entry::Vacant(entry) => entry.key(),
        }
    }

    /// Inserts a value if vacant, then returns a mutable reference to it.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default),
        }
    }

    /// Inserts a lazily produced value if vacant, then returns a mutable
    /// reference to the value.
    pub fn or_insert_with<Deliver>(self, default: Deliver) -> &'a mut V
    where
        Deliver: FnOnce() -> V,
    {
        match self {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(default()),
        }
    }

    /// Ensures a value is in place and returns its handle together with a
    /// mutable reference to the value.
    pub fn or_insert_handle(self, default: V) -> (EntryHandle<K>, &'a mut V) {
        match self {
            Entry::Occupied(entry) => {
                let index = entry.index;
                let handle = entry.handle();
                let value = entry
                    .map
                    .inner
                    .map
                    .get_index_mut(index)
                    .expect("valid index")
                    .1;
                (handle, value)
            }
            Entry::Vacant(entry) => entry.insert_handle(default),
        }
    }

    /// Provides mutable in-place access to an occupied entry.
    pub fn and_modify<Deliver>(self, f: Deliver) -> Self
    where
        Deliver: FnOnce(&mut V),
    {
        match self {
            Entry::Occupied(mut entry) => {
                f(entry.get_mut());
                Entry::Occupied(entry)
            }
            vacant => vacant,
        }
    }
}

impl<'a, K, V, S> fmt::Debug for Entry<'a, K, V, S>
where
    K: fmt::Debug + Eq + Hash,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Entry::Occupied(entry) => f.debug_tuple("Occupied").field(entry).finish(),
            Entry::Vacant(entry) => f.debug_tuple("Vacant").field(entry).finish(),
        }
    }
}

/// An occupied entry of a [`TrackedMap`].
pub struct OccupiedEntry<'a, K, V, S> {
    map: &'a mut TrackedMap<K, V, S>,
    index: usize,
    key: Arc<K>,
}

impl<'a, K, V, S> OccupiedEntry<'a, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Returns the entry's key.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Returns the entry's current index.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns a handle to this item.
    pub fn handle(&self) -> EntryHandle<K> {
        EntryHandle {
            owner: self.map.owner_tag(),
            generation: self.map.inner.slots[self.index].generation,
            key: Arc::clone(&self.key),
        }
    }

    /// Consumes the entry and returns a handle to this item.
    pub fn into_handle(self) -> EntryHandle<K> {
        let OccupiedEntry { map, index, key } = self;
        EntryHandle {
            owner: map.owner_tag(),
            generation: map.inner.slots[index].generation,
            key,
        }
    }

    /// Returns a reference to the value.
    pub fn get(&self) -> &V {
        self.map
            .inner
            .map
            .get_index(self.index)
            .expect("valid index")
            .1
    }

    /// Returns a mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut V {
        self.map
            .inner
            .map
            .get_index_mut(self.index)
            .expect("valid index")
            .1
    }

    /// Consumes the entry and returns a mutable reference to the value.
    pub fn into_mut(self) -> &'a mut V {
        self.map
            .inner
            .map
            .get_index_mut(self.index)
            .expect("valid index")
            .1
    }

    /// Replaces the value and returns the previous one.
    pub fn insert(&mut self, value: V) -> V {
        core::mem::replace(self.get_mut(), value)
    }

    /// Removes the item with shift semantics and returns its value.
    pub fn shift_remove(self) -> V {
        let OccupiedEntry { map, index, key: _ } = self;
        map.inner.slots.remove(index);
        map.inner
            .map
            .shift_remove_index(index)
            .expect("valid index")
            .1
    }

    /// Removes the item with swap semantics and returns its value.
    pub fn swap_remove(self) -> V {
        let OccupiedEntry { map, index, key: _ } = self;
        map.inner.slots.swap_remove(index);
        map.inner
            .map
            .swap_remove_index(index)
            .expect("valid index")
            .1
    }
}

impl<K: fmt::Debug + Eq + Hash, V: fmt::Debug, S: BuildHasher> fmt::Debug
    for OccupiedEntry<'_, K, V, S>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OccupiedEntry")
            .field("key", &self.key)
            .field("index", &self.index)
            .field("value", self.get())
            .finish()
    }
}

/// A vacant entry of a [`TrackedMap`].
pub struct VacantEntry<'a, K, V, S> {
    map: &'a mut TrackedMap<K, V, S>,
    key: Arc<K>,
}

impl<'a, K, V, S> VacantEntry<'a, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    /// Returns the key that would be inserted.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Inserts a value and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        self.insert_handle(value).1
    }

    /// Inserts a value and returns the new item's handle together with a
    /// mutable reference to the value.
    pub fn insert_handle(self, value: V) -> (EntryHandle<K>, &'a mut V) {
        let VacantEntry { map, key } = self;
        let generation = map.fresh_generation();
        map.inner.slots.push(Slot { generation });
        let index = map.inner.map.len();
        map.inner.map.insert(Arc::clone(&key), value);
        let handle = EntryHandle {
            owner: map.owner_tag(),
            generation,
            key,
        };
        let value = map.inner.map.get_index_mut(index).expect("just inserted").1;
        (handle, value)
    }
}

impl<K: fmt::Debug, V, S> fmt::Debug for VacantEntry<'_, K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VacantEntry")
            .field("key", &self.key)
            .finish()
    }
}
