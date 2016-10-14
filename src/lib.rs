// Copyright 2016 Austin Bonander
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
// Implementation adapted from C++ code at http://stevehanov.ca/blog/index.php?id=130
// (accessed October 14, 2016). No copyright or license information is provided,
// so the original code is assumed to be public domain.
//! An implementation of a [vantage-point tree][vp-tree] backed by a vector.
//!
//! [vp-tree]: https://en.wikipedia.org/wiki/Vantage-point_tree
#![warn(missing_docs)]

extern crate rand;

use rand::Rng;


use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::{cmp, fmt, iter, mem};

mod select;

mod print;

const NO_NODE: usize = ::std::usize::MAX;

/// An implementation of a vantage-point tree backed by a vector.
///
/// Only bulk insert/removals are provided in order to keep the tree balanced.
pub struct VpTree<T, D> {
    nodes: Vec<Node>,
    items: Vec<T>,
    dist_fn: D,
}

impl<T, D: DistFn<T>> VpTree<T, D> {
    /// Collect the results of `items` into the tree, and build the tree using `dist_fn`.
    ///
    /// If coming straight from a vector, use `from_vec` to avoid a copy.
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

        let median_idx = (end - (start + 1)) / 2;

        let threshold = {
            let (pivot, items) = self.items.split_first_mut().unwrap();

            // Without this reborrow, the closure will try to borrow all of `self`.
            let dist_fn = &self.dist_fn;

            // This function will partition around the median element
            let median_thresh_item = select::qselect_inplace_by(
                items, median_idx,
                |left, right| dist_fn.dist(pivot, left).cmp(&dist_fn.dist(pivot, right))
            );

            dist_fn.dist(pivot, median_thresh_item)
        };

        let left_start = start + 1;

        let split_idx = left_start + median_idx + 1;

        let self_idx = self.push_node(start, parent_idx, threshold);

        let left_idx = self.rebuild(self_idx, left_start, split_idx);

        let right_idx = self.rebuild(self_idx, split_idx, end);

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
        assert!(self.nodes.len() == self.items.len(), "Attempting to traverse `VpTree` when it is
        in an invalid state. This can happen if a panic was thrown while it was being mutated and
        then caught outside.")
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
    pub fn dist_fn<D_: DistFn<T>>(self, dist_fn: D_) -> VpTree<T, D_> {
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
    /// ## Note
    /// It is a logic error for an item to be modified in such a way that the item's distance
    /// to any other item, as determined by `D: VpDist<T>`, changes while it is in the tree
    /// without the tree being rebuilt.
    /// This is normally only possible through Cell, RefCell, global state, I/O, or unsafe code.
    ///
    /// If you wish to mutate one or more of the contained items, use `.with_mut_items()` instead.
    pub fn items(&self) -> &[T] {
        &self.items
    }

    /// Get a scoped mutable slice to the contained items.
    ///
    /// The tree will be rebuilt after `mut_fn` returns, in assumption that it will modify one or
    /// more of the contained items such that their distance to others,
    /// as determined by `D: VpDist<T>`, changes.
    ///
    /// ## Note
    /// If a panic is initiated in `mut_fn` and then caught outside this method,
    /// the tree will need to be manually rebuilt with `.full_rebuild()`.
    pub fn with_mut_items<F>(&mut self, mut_fn: F) where F: FnOnce(&mut [T]) {
        self.nodes.clear();
        mut_fn(&mut self.items);
        self.full_rebuild();
    }

    /// Get a vector of the `k` nearest neighbors to `origin`, sorted in ascending order
    /// by the distance.
    ///
    /// ## Note
    /// If `origin` is contained within the tree, which is allowed by the API contract,
    /// it will be returned. In this case, it may be preferable to start with a higher `k` and
    /// filter out duplicate entries.
    ///
    /// ## Panics
    /// If the tree was in an invalid state
    pub fn k_nearest<'t, O: Borrow<T>>(&'t self, origin: O, mut k: usize) -> Vec<Neighbor<'t, T>> {
        self.sanity_check();

        let origin = origin.borrow();

        KnnVisitor::new(self, origin, k)
            .visit_all()
            .into_vec()
    }

    fn root(&self) -> usize {
        if self.nodes.len() > 0 {
            0
        } else {
            NO_NODE
        }
    }

    /// Consume `self` and return the vector of items.
    ///
    /// These items may have been rearranged from the order which they were inserted.
    pub fn into_vec(self) -> Vec<T> {
        self.items
    }
}

impl<T: fmt::Debug, D: DistFn<T>> fmt::Debug for VpTree<T, D> {
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

struct KnnVisitor<'t, 'o, T: 't + 'o, D: 't> {
    tree: &'t VpTree<T, D>,
    origin: &'o T,
    heap: BinaryHeap<Neighbor<'t, T>>,
    k: usize,
    radius: u64,
}

impl<'t, 'o, T: 't + 'o, D: 't> KnnVisitor<'t, 'o, T, D> where D: DistFn<T> {
    fn new(tree: &'t VpTree<T, D>, origin: &'o T, k: usize) -> Self {
        KnnVisitor {
            tree: tree,
            origin: origin,
            heap: if k > 0 { BinaryHeap::with_capacity(k + 2) } else { BinaryHeap::new() },
            k: k,
            radius: ::std::u64::MAX,
        }
    }
    fn visit_all(mut self) -> Self {
        if self.k > 0 && self.tree.nodes.len() > 0 {
            self.visit(0);
        }

        self
    }

    fn visit(&mut self, node_idx: usize, ){
        if node_idx == NO_NODE {
            return;
        }

        let cur_node = &self.tree.nodes[node_idx];

        let item = &self.tree.items[cur_node.idx];

        let dist_to_cur = self.tree.dist_fn.dist(&self.origin, item);

        if dist_to_cur < self.radius {
            if self.heap.len() == self.k {
                self.heap.pop();
            }

            self.heap.push(Neighbor {
                item: item,
                dist: dist_to_cur
            });

            if self.heap.len() == self.k {
                self.radius = self.heap.peek().unwrap().dist;
            }
        }

        let go_left = dist_to_cur.saturating_sub(self.radius) <= cur_node.threshold;
        let go_right = dist_to_cur.saturating_add(self.radius) >= cur_node.threshold;

        if dist_to_cur <= cur_node.threshold {
            if go_left {
                self.visit(cur_node.left);
            }

            if go_right {
                self.visit(cur_node.right);
            }
        } else {
            if go_right {
                self.visit(cur_node.right);
            }

            if go_left {
                self.visit(cur_node.left);
            }
        };
    }

    fn into_vec(self) -> Vec<Neighbor<'t, T>> {
        self.heap.into_sorted_vec()
    }
}

#[derive(Debug, Clone)]
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

/// Describes a type which can act as a distance-function for `T`.
///
/// Implemented for `Fn(&T, &T) -> u64`
pub trait DistFn<T> {
    /// Return the distance between `left` and `right`.
    ///
    /// ## Note
    /// It is a logic error for this method to return different values for the same operands,
    /// regardless of order (i.e. it is required to be idempotent and commutative).
    fn dist(&self, left: &T, right: &T) -> u64;
}

/// Simply calls `(self)(left, right)`
impl<T, F> DistFn<T> for F where F: Fn(&T, &T) -> u64 {
    fn dist(&self, left: &T, right: &T) -> u64 {
        (self)(left, right)
    }
}

#[cfg(test)]
mod test {
    use super::VpTree;

    const MAX_TREE_VAL: i32 = 8;
    const ORIGIN: i32 = 4;
    const NEIGHBORS: &'static [i32] = &[2, 3, 4, 5, 6];

    fn setup_tree() -> VpTree<i32, fn(&i32, &i32) -> u64> {
        fn dist(left: &i32, right: &i32) -> u64 {
            (left - right).abs() as u64
        }

        VpTree::new(0i32 .. MAX_TREE_VAL, dist)
    }

    #[test]
    fn test_k_nearest() {
        let tree = setup_tree();

        println!("Tree: {:?}", tree);

        let nearest: Vec<_> = tree.k_nearest(&ORIGIN, NEIGHBORS.len())
            .into_iter().collect();

        println!("Nearest: {:?}", nearest);

        for neighbor in nearest {
            assert!(NEIGHBORS.contains(&neighbor.item), "Was not expecting {:?}", neighbor);
        }
    }
}
