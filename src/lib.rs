//! An implementation of a [vantage-point tree][vp-tree] backed by a vector.
//!
//! [vp-tree]: https://en.wikipedia.org/wiki/Vantage-point_tree
#![warn(missing_docs)]

extern crate rand;
extern crate smallvec;

use rand::Rng;

use smallvec::SmallVec;

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::{cmp, fmt, iter, mem};

mod select;

mod print;

const NO_NODE: usize = ::std::usize::MAX;

#[cfg(test)]
type SplitStack = Vec<usize>;
#[cfg(not(test))]
type SplitStack = SmallVec<[usize; 3]>;

/// An implementation of a vantage-point tree backed by a vector.
///
/// Only bulk insert/removals are provided in order to keep the tree balanced.
pub struct VpTree<T, D> {
    nodes: Vec<Node>,
    items: Vec<T>,
    dist_fn: D,
}

impl<T: fmt::Debug, D: VpDist<T>> VpTree<T, D> {
    pub fn new<I: IntoIterator<Item = T>>(items: I, dist_fn: D) -> Self {
        Self::from_vec(items.into_iter().collect(), dist_fn)
    }

    pub fn from_vec(items: Vec<T>, dist_fn: D) -> Self {
        let mut self_ = VpTree {
            nodes: Vec::with_capacity(items.len()),
            items: items,
            dist_fn: dist_fn,
        };

        self_.full_rebuild();

        self_
    }

    /// Rebuild the full tree.
    ///
    /// This is only necessary if the one or more properties of a contained
    /// item which determine their distance via `D: VpDist<T>` was somehow changed without
    /// the tree being rebuilt, or a panic occurred during a mutation and was caught.
    pub fn full_rebuild(&mut self) {
        self.nodes.clear();

        let len = self.items.len();
        let nodes_cap = self.nodes.capacity();

        if len > nodes_cap {
            self.nodes.reserve(len - nodes_cap);
        }

        self.rebuild(NO_NODE, 0, len);
    }

    /// Rebuild the tree in [start, end)
    fn rebuild(&mut self, parent_idx: usize, start: usize, end: usize) -> usize {
        if start == end { return NO_NODE; }

        if start + 1 == end {
            return self.push_node(start, parent_idx, 0);
        }

        let pivot_idx = rand::thread_rng().gen_range(start, end);
        self.items.swap(start, pivot_idx);

        let threshold = {
            let (pivot, items) = self.items.split_first_mut().unwrap();

            // Without this reborrow, the closure will try to borrow all of `self`.
            let dist_fn = &self.dist_fn;

            // This function will partition around the median element so we don't need
            // as second partial-sort run
            let median_thresh_item = select::median_inplace_by(
                items,
                |left, right| dist_fn.dist(pivot, left).cmp(&dist_fn.dist(pivot, right))
            );


            dist_fn.dist(pivot, median_thresh_item)
        };

        println!("Items (pivot_idx: {}): {:?}", pivot_idx, &self.items[start .. end]);

        let split_idx = start + 1 + (end - (start + 1)) / 2;

        let self_idx = self.push_node(start, parent_idx, threshold);

        let left_idx = self.rebuild(self_idx, start + 1, split_idx + 1);
        let right_idx = self.rebuild(self_idx, split_idx + 1, end);

        self.nodes[self_idx].left = left_idx;
        self.nodes[self_idx].right = right_idx;

        self_idx
    }

    fn push_node(&mut self, idx: usize, parent_idx: usize, threshold: u64) -> usize {
        let self_idx = self.nodes.len();

        self.nodes.push(Node {
            idx: idx,
            parent: parent_idx,
            left: NO_NODE,
            right: NO_NODE,
            threshold: threshold,
        });

        self_idx
    }

    #[inline(always)]
    fn sanity_check(&self) {
        assert!(self.nodes.len() == self.items.len(), "`VpTree` was .")
    }

    /// Add `new_items` to the tree and rebuild it.
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, new_items: I) {
        self.nodes.clear();
        self.items.extend(new_items);
        self.full_rebuild();
    }

    /// Apply a new distance function and rebuild the tree, returning the transformed type.
    ///
    /// The tree will be rebuilt before it is returned.
    pub fn dist_fn<D_: VpDist<T>>(self, dist_fn: D_) -> VpTree<T, D_> {
        let mut self_ = VpTree {
            nodes: self.nodes,
            items: self.items,
            dist_fn: dist_fn,
        };

        self_.full_rebuild();

        self_
    }

    /// Iterate over the contained items, dropping them if `ret_fn` returns `false`,
    /// keeping them otherwise.
    ///
    /// The tree will be rebuilt afterwards.
    pub fn retain<F>(&mut self, ret_fn: F) where F: FnMut(&T) -> bool {
        self.nodes.clear();
        self.items.retain(ret_fn);
        self.full_rebuild();
    }

    /// Get a slice of the items.
    ///
    /// ##Note
    /// It is a logic error for an item to be modified in such a way that the item's distance
    /// to any other item, as determined by `D: VpDist<T>`, changes while it is in the tree
    /// without the tree being rebuilt.
    /// This is normally only possible through Cell, RefCell, global state, I/O, or unsafe code.
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Get a scoped mutable slice to the contained items.
    ///
    /// The tree will be rebuilt after `mut_fn` returns, in assumption that it will modify one or
    /// more of the contained items such that their distance to others,
    /// as determined by `D: VpDist<T>`, changes.
    pub fn with_mut_items<F>(&mut self, mut_fn: F) where F: FnOnce(&mut [T]) {
        self.nodes.clear();
        mut_fn(&mut self.items);
        self.full_rebuild();
    }

    /// Return an iterator over the items within `radius` distance of `origin`.
    ///
    /// ##Note
    /// `origin` may or may not be in the tree, but if it is, it will be returned in this iterator.
    ///
    /// ##Panics
    /// If the tree was not rebuilt properly. This would only happen if a panic occurred during
    /// rebuild or one of the mutating functions here, and then was caught.
    ///
    /// Call `.full_rebuild()` to restore the tree.
    pub fn neighbors<'t, 'o>(&'t self, origin: &'o T, radius: u64) -> Neighbors<'t, 'o, T, D> {
        self.sanity_check();

        Neighbors {
            tree: self,
            origin: origin,
            splits: Default::default(),
            current_node: if self.nodes.len() > 0 { 0 } else { NO_NODE },
            radius: radius,
        }
    }

    /// Get a vector of the `k` nearest neighbors to `origin`, sorted in ascending order
    /// by the distance.
    pub fn k_nearest<'t, 'o>(&'t self, origin: &'o T, mut k: usize) -> Vec<Neighbor<'t, T>> where T: Eq {
        if k == 0 {
            return Vec::new();
        }

        let mut neighbors = self.neighbors(origin, ::std::u64::MAX);

        let mut heap = BinaryHeap::with_capacity(k * 2);

        while let Some(neighbor) = neighbors.next() {
            if neighbor.item == origin { k += 1; }
            if heap.len() == k { heap.pop(); }
            heap.push(neighbor);
            if heap.len() == k {
                neighbors.radius = heap.peek().unwrap().dist;
            }
        }

        let mut vec = heap.into_sorted_vec();
        vec.retain(|n| origin != n.item);
        vec
    }

    /// Consume `self` and return the vector of items.
    ///
    /// These items may have been rearranged from the order which they were inserted.
    pub fn into_vec(self) -> Vec<T> {
        self.items
    }
}

impl<T: fmt::Debug, D: VpDist<T>> fmt::Debug for VpTree<T, D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(writeln!(f, "VpTree {{ len: {} }}", self.items.len()));

        if self.nodes.len() == 0 { return f.write_str("[Empty]\n");}

        try!(writeln!(f, "Items: {:?}\nStructure:", self.items));

        print::TreePrinter::new(self).print(f)
    }
}

#[derive(Debug)]
struct Node {
    idx: usize,
    parent: usize,
    left: usize,
    right: usize,
    threshold: u64,
}

pub struct Neighbors<'t, 'o, T: 't + 'o, D: 't> {
    tree: &'t VpTree<T, D>,
    origin: &'o T,
    splits: SplitStack,
    current_node: usize,
    radius: u64,
}

impl<'t, 'o, T: 't + 'o, D: 't> Iterator for Neighbors<'t, 'o, T, D> where D: VpDist<T> {
    type Item = Neighbor<'t, T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut already_seen = false;

            if self.current_node == NO_NODE {
                if let Some(last_split) = self.splits.pop() {
                    self.current_node = last_split;
                    already_seen = true;
                } else {
                    break;
                }
            }

            let cur_node = &self.tree.nodes[self.current_node];

            if cur_node.left == NO_NODE && cur_node.right == NO_NODE {
                self.current_node = NO_NODE;
                continue;
            }

            let item = &self.tree.items[cur_node.idx];

            let dist_to_cur = self.tree.dist_fn.dist(&self.origin, item);

            let go_left = dist_to_cur.saturating_sub(self.radius) <= cur_node.threshold;
            let go_right = dist_to_cur.saturating_add(self.radius) >= cur_node.threshold;

            self.current_node = if dist_to_cur <= cur_node.threshold {
                if go_left && !already_seen {
                    if go_right {
                        self.push_split();
                    }

                    cur_node.left
                } else if go_right {
                    cur_node.right
                } else {
                    NO_NODE
                }
            } else {
                if go_right && !already_seen {
                    if go_left {
                        self.push_split();
                    }

                     cur_node.right
                } else if go_left {
                    cur_node.left
                } else {
                    NO_NODE
                }
            };

            if dist_to_cur <= self.radius && !already_seen {
                return Some(Neighbor {
                    item: item,
                    dist: dist_to_cur
                });
            }
        }

        None
    }
}

impl<'t, 'o, T: 't + 'o, D: 't> Neighbors<'t, 'o, T, D>{
    fn push_split(&mut self) {
        self.splits.push(self.current_node);
        let cur_node = &self.tree.nodes[self.current_node];

        println!("Push split: items[{}]", cur_node.idx);
    }
}

/// Wrapper of an item and a distance, returned by `Neighbors`.
pub struct Neighbor<'t, T: 't> {
    /// The item that this entry concerns.
    pub item: &'t T,
    /// The distance between `item` and the origin passed to `VpTree::neighbors()` or
    /// `VpTree::k_nearest()`.
    pub dist: u64,
}

/// Returns the comparison of the distances only.
impl<'t, T: 't> PartialOrd for Neighbor<'t, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns the comparison of the distances only.
impl<'t, T: 't> Ord for Neighbor<'t, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}

/// Returns the equality of the distances only.
impl<'t, T: 't> PartialEq for Neighbor<'t, T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

/// Returns the equality of the distances only.
impl<'t, T: 't> Eq for Neighbor<'t, T> {}

pub trait VpDist<T> {
    fn dist(&self, left: &T, right: &T) -> u64;
}

impl<T, F> VpDist<T> for F where F: Fn(&T, &T) -> u64 {
    fn dist(&self, left: &T, right: &T) -> u64 {
        (self)(left, right)
    }
}

#[cfg(test)]
mod test {
    use super::VpTree;

    const MAX_TREE_VAL: i32 = 8;
    const TREE_MEDIAN: i32 = 4;
    const RADIUS: u64 = 2;
    const NEIGHBORS: &'static [i32] = &[2, 3, 5, 6];

    fn setup_tree() -> VpTree<i32, fn(&i32, &i32) -> u64> {
        fn dist(left: &i32, right: &i32) -> u64 {
            (left - right).abs() as u64
        }

        VpTree::new(0i32 .. MAX_TREE_VAL, dist)
    }

    //#[test]
    fn test_k_nearest() {
        let tree = setup_tree();
        let nearest: Vec<_> = tree.k_nearest(&TREE_MEDIAN, NEIGHBORS.len())
            .into_iter().map(|n| *n.item).collect();
        assert_eq!(&*nearest, NEIGHBORS);
    }

    #[test]
    fn test_neighbors() {
        let tree = setup_tree();
        println!("{:?}", tree);
        let mut neighbors: Vec<_> = tree.neighbors(&TREE_MEDIAN, RADIUS)
            .filter(|n| n.item != &TREE_MEDIAN).map(|n| *n.item).collect();
        neighbors.sort();
        assert_eq!(&*neighbors, NEIGHBORS)
    }
}
