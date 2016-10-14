// Copyright 2016 Austin Bonander
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

extern crate vec_vp_tree;

use vec_vp_tree::VpTree;

fn main() {
    let vp_tree = VpTree::new(0i32 .. 8, |left: &i32, right: &i32| (left - right).abs() as u64);
    println!("{:?}", vp_tree);
}