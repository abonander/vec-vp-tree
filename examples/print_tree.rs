extern crate vec_vp_tree;

use vec_vp_tree::VpTree;

fn main() {
    let vp_tree = VpTree::new(0i32 .. 8, |left: &i32, right: &i32| (left - right).abs() as u64);
    println!("{:?}", vp_tree);
}