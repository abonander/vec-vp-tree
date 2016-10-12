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

    /// Rebuild the tree
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

        let split_idx = start + (end - (start + 1)) / 2;

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

    /// Add `new_items` to the tree and rebuild it.
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, new_items: I) {
        self.nodes.clear();
        self.items.extend(new_items);
        self.full_rebuild();
    }

    /// Apply a new distance function and rebuild the tree, returning the transformed type.
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
        self.items.retain(ret_fn);
        self.full_rebuild();
    }

    pub fn items(&self) -> &[T] {
        &self.items
    }

    pub fn neighbors<'t, 'o>(&'t self, origin: &'o T, radius: u64) -> Neighbors<'t, 'o, T, D> {
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
    pub fn k_nearest<'t, 'o>(&'t self, origin: &'o T, k: usize) -> Vec<Neighbor<'t, T>> where T: Eq {
        if k == 0 {
            return Vec::new();
        }

        let mut neighbors = self.neighbors(origin, ::std::u64::MAX);

        if k == 1 {
            let mut ret = Vec::new();

            while let Some(neighbor) = neighbors.next() {
                if ret.len() == 1 && neighbor < ret[0] {
                    neighbors.radius = ret[0].dist;
                    ret[0] = neighbor;
                } else {
                    ret.push(neighbor);
                }
            }

            return ret;
        }

        let mut heap = BinaryHeap::with_capacity(k * 2);

        while let Some(neighbor) = neighbors.next() {
            heap.push(neighbor);
            if heap.len() == k {
                neighbors.radius = heap.pop().unwrap().dist;
            }
        }

        heap.into_sorted_vec()
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
            if self.current_node == NO_NODE {
                if let Some(&last_split) = self.splits.last() {
                    self.current_node = last_split;
                } else {
                    break;
                }
            }

            let cur_node = &self.tree.nodes[self.current_node];

            let item = &self.tree.items[cur_node.idx];

            let dist_to_cur = self.tree.dist_fn.dist(&self.origin, item);

            let go_left = dist_to_cur.saturating_sub(self.radius) <= cur_node.threshold;
            let go_right = dist_to_cur.saturating_add(self.radius) >= cur_node.threshold;

            let already_seen = if let Some(last_split) = self.splits.pop() {
                let already_seen = self.current_node == last_split;

                if !already_seen {
                    self.splits.push(last_split);
                }

                already_seen
            } else {
                false
            };

            if dist_to_cur <= cur_node.threshold {
                if go_left && !already_seen {
                    if go_right {
                        self.push_split();
                    }

                    self.current_node = cur_node.left;
                }

                if go_right {
                    self.current_node = cur_node.right;
                }
            } else {
                if go_right && !already_seen {
                    if go_left {
                        self.push_split();
                    }

                    self.current_node = cur_node.right;
                }

                if go_left {
                    self.current_node = cur_node.left;
                }
            }

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

impl<'t, 'o, T: 't + 'o, D: 't> Neighbors<'t, 'o, T, D> {
    fn push_split(&mut self) {
        self.splits.push(self.current_node);
        println!("Push split: {}", self.current_node);
    }
}

/// Wrapper of an item and a distance, returned by `Neighbors`.
pub struct Neighbor<'t, T: 't> {
    pub item: &'t T,
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

    fn setup_tree() -> VpTree<i32, fn(&i32, &i32) -> u64> {
        fn dist(left: &i32, right: &i32) -> u64 {
            (left - right).abs() as u64
        }

        VpTree::new(0i32 .. MAX_TREE_VAL, dist)
    }

    //#[test]
    fn test_k_nearest() {
        let tree = setup_tree();
        let nearest: Vec<_> = tree.k_nearest(&TREE_MEDIAN, 4).into_iter().map(|n| *n.item).collect();
        assert_eq!(nearest, vec![6, 7, 9, 10]);
    }

    #[test]
    fn test_neighbors() {
        let tree = setup_tree();
        println!("{:?}", tree);
        let mut neighbors: Vec<_> = tree.neighbors(&TREE_MEDIAN, 2).map(|n| *n.item).collect();
        neighbors.sort();
        assert_eq!(neighbors, vec![6, 7, 9, 10])
    }
}
