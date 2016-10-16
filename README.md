`vec-vp-tree` [![Build Status](https://travis-ci.org/abonander/vec-vp-tree.svg?branch=master)](https://travis-ci.org/abonander/vec-vp-tree) [![On Crates.io](https://img.shields.io/crates/v/vec-vp-tree.svg)](https://crates.io/crates/vec-vp-tree)
=========

A [vantage-point tree][vp-tree] implementation backed by a vector for good performance with **zero** lines of `unsafe`
code.

Provides bulk insert/removal operations (to maintain balance without too much extra bookkeeping), 
and the [*k*-Nearest Neighbors (k-NN)][knn] algorithm.

Multiple distance functions are provided by default, including Hamming and Levenshtein distance functions for strings.

[vp-tree]: https://en.wikipedia.org/wiki/Vantage-point_tree
[knn]: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

###[Documentation](http://docs.rs/multipart/)

License
-------

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.