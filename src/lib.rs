extern crate rand;

use rand::Rng;

mod select;

const NO_NODE: usize = ::std::usize::MAX;

pub struct VpTree<T, Df> {
    nodes: Vec<Node>,
    items: Vec<T>,
    dist_fn: Df,
}

impl<T, Df> VpTree<T, Df> where Df: FnMut(&T, &T) -> u64 {
    pub fn new<I: IntoIterator<Item = T>>(items: I, dist_fn: Df) -> Self {
        Self::from_vec(items.into_iter().collect(), dist_fn)
    }

    pub fn from_vec(items: Vec<T>, dist_fn: Df) -> Self {
        let mut self_ = VpTree {
            nodes: Vec::with_capacity(items.len()),
            items: items,
            dist_fn: dist_fn,
        };

        self_.full_rebuild();

        self_
    }

    fn full_rebuild(&mut self) {
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
        if start + 1 == end { return NO_NODE; }

        if start + 2 == end {
            self.push_node(parent_idx, start, 0);
        }

        let pivot_idx = rand::thread_rng().gen_range(start, end);
        self.items.swap(start, pivot_idx);

        let threshold = {
            let (pivot, items) = self.items.split_first_mut().unwrap();

            // Without this reborrow, the closure will try to borrow all of `self`.
            let mut dist_fn = &mut self.dist_fn;

            // This function will partition around the median element so we don't need
            // as second partial-sort run
            let median_thresh_item = select::median_inplace_by(
                items,
                |left, right| dist_fn(pivot, left).cmp(&dist_fn(pivot, right))
            );

            dist_fn(pivot, median_thresh_item)
        };

        let self_idx = self.push_node(parent_idx, start, threshold);

        let left_idx = self.rebuild(self_idx, start + 1, pivot_idx + 1);
        let right_idx = self.rebuild(self_idx, pivot_idx + 1, end);

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
        self.items.extend(new_items);
        self.full_rebuild();
    }

    /// Apply a new distance function and rebuild the tree, returning the transformed type.
    pub fn dist_fn<Df_>(self, dist_fn: Df_) -> VpTree<T, Df_>
        where Df_: FnMut(&T, &T) -> u64 {
        let mut self_ = VpTree {
            nodes: self.nodes,
            items: self.items,
            dist_fn: dist_fn,
        };

        self_.full_rebuild();

        self_
    }

    pub fn items(&self) -> &[T] {
        &self.items
    }

    pub fn neighbors<'t, 'o>(&'t self, origin: &'o T, radius: u64) -> Neighbors<'t, 'o, T, Df>
    where T: Eq {
        Neighbors {
            tree: self,
            origin: origin,
            current_node: if self.nodes.len() > 0 { 0 } else { NO_NODE },
            radius: radius
        }
    }
}

struct Node {
    idx: usize,
    parent: usize,
    left: usize,
    right: usize,
    threshold: u64,
}

pub struct Neighbors<'t, 'o, T: 't + 'o, Df: 't> {
    tree: &'t VpTree<T, Df>,
    origin: &'o T,
    current_node: usize,
    radius: u64,
}

impl<'t, 'o, T: 't + 'o, Df: 't> Iterator for Neighbors<'t, 'o, T, Df>
where Df: Fn(&T, &T) -> u64, T: Eq {
    type Item = (&'t T, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_node == NO_NODE {
            return None;
        }

        if self.tree.nodes.len() == 0 { return None; }

        let cur_node = &self.tree.nodes[self.current_node];

        let dist_to_cur = (self.tree.dist_fn)(&self.origin, &self.tree.items[cur_node.idx]);

        if dist_to_cur <= cur_node.threshold {
            if cur_node.left == NO_NODE {

            }
        }

        if  dist_to_cur <= self.radius {

        }
    }
}