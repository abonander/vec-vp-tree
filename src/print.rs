// Copyright 2016 Austin Bonander
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use super::{VpTree, Node, NO_NODE};

use std::fmt::Write;
use std::fmt;

pub struct TreePrinter<'t, T: 't, D: 't> {
    tree: &'t VpTree<T, D>,
    nodes: Vec<NodeInfo<'t, T>>,
}

impl<'t, T: 't, D: 't> TreePrinter<'t, T, D> where T: fmt::Debug {
    pub fn new(tree: &'t VpTree<T, D>) -> Self {
        TreePrinter {
            tree: tree,
            nodes: Vec::new(),
        }
    }

    pub fn print(mut self, f: &mut fmt::Formatter) -> fmt::Result {
        self.visit_node(0, 0);
        self.print_levels(f)
    }

    fn visit_node(&mut self, node_idx: usize, tree_idx: usize) {
        if node_idx == NO_NODE { return; }

        let (left, right) = {
            let node_info = get_node(&mut self.nodes, tree_idx);

            let node: &Node = &self.tree.nodes[node_idx];

            node_info.item = Some(&self.tree.items[node.idx]);
            node_info.threshold = node.threshold;

            (node.left, node.right)
        };

        self.visit_node(left, 2 * tree_idx + 1);
        self.visit_node(right, 2 * tree_idx + 2);
    }

    fn print_levels(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut start = 0;
        let mut len = 1;

        while start + len <= self.nodes.len() {
            let line = &self.nodes[start .. start + len];

            for pair in line.chunks(2) {
                try!(f.write_str("[ "));

                for node in pair {
                    try!(write!(f, "{:?} ", node));
                }

                try!(f.write_str("] "));
            }

            try!(f.write_char('\n'));


            start += len;
            len <<= 1;
        }

        Ok(())
    }
}

fn get_node<'n, 't, T: 't>(nodes: &'n mut Vec<NodeInfo<'t, T>>, node_idx: usize) -> &'n mut NodeInfo<'t, T> {
    if node_idx >= nodes.len() {
        let new_size = (node_idx + 1).next_power_of_two();

        nodes.resize(new_size, NodeInfo::default());
    }

    &mut nodes[node_idx]
}

struct NodeInfo<'t, T: 't> {
    item: Option<&'t T>,
    threshold: u64,
}

impl<'t, T: 't> Clone for NodeInfo<'t, T> {
    fn clone(&self) -> Self {
        NodeInfo {
            item: self.item,
            threshold: self.threshold,
        }
    }
}

impl<'t, T: 't> Copy for NodeInfo<'t, T> {}

impl<'t, T: 't> Default for NodeInfo<'t, T> {
    fn default() -> Self {
        NodeInfo {
            item: None,
            threshold: 0
        }
    }
}

impl<'t, T: fmt::Debug + 't> fmt::Debug for NodeInfo<'t, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(item) = self.item {
            write!(f, "{:?}({})", item, self.threshold)
        } else {
            f.write_char('_')
        }
    }
}
