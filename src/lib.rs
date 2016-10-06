extern crate algo;
extern crate rand;

use algo::select;

use rand::Rng;

pub struct VpTree<T, Df> {
    nodes: Vec<Node>,
    data: Vec<T>,
    dist_fn: Df,
}

impl<T, Df> VpTree<T, Df> where Df: FnMut(&T, &T) -> u64 {
    pub fn new<I: IntoIterator<Item = T>>(items: I, dist_fn: Df) -> Self {
        Self::from_vec(items.into_iter().collect(), dist_fn)
    }

    pub fn from_vec(items: Vec<T>, dist_fn: Df) -> Self {
        let mut self_ = VpTree {
            nodes: Vec::with_capacity(items.len()),
            data: items,
            dist_fn: dist_fn,
        };

        self_.full_rebuild();

        self_
    }

    fn full_rebuild(&mut self) {
        self.nodes.clear();
        self.rebuild(0, self.items.len());
    }

    /// Rebuild the tree in [start, end)
    fn rebuild(&mut self, start: usize, end: usize) {
        if start == end - 1 { return; }

        let pivot_idx = rand::thread_rng().gen_range(start, end);
        self.items.swap(start, pivot_idx);

        let threshold = {
            let (pivot, items) = self.items.split_first_mut().unwrap();

            let mut dist_fn = &mut self.dist_fn;

            let median_thresh_item = select::median_inplace_by(
                items,
                |left, right| dist_fn(&pivot, left).cmp(dist_fn(&pivot, right))
            );

            dist_fn(pivot, median_thresh_item)
        };


    }
}

struct Node {
    idx: usize,
    left: usize,
    right: usize,
    threshold: u64,
}