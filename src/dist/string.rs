// Copyright 2016 Austin Bonander
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//! Distance functions for strings (`strsim` feature, enabled by default)

use strsim;

use super::{DistFn, KnownDist};

use std::cmp;

const JARO_SCALE_FACTOR: f64 = 100.;

/// Hamming distance function for strings. Default as the fastest.
///
/// Returns the number of characters that are different between the two strings.
///
/// If the strings are not the same length, adds the length difference as well.
pub struct Hamming;

/// Levenshtein distance (edit-distance) function for strings.
///
/// Returns the the minimum number of insertions, deletions, and substitutions required to change
/// one string into the other.
pub struct Levenshtein;

/// Damerau-Levenshtein distance function for strings.
///
/// Returns the the minimum number of insertions, deletions, and substitutions required to change
/// one string into the other, also counts swaps of adjacent characters.
pub struct DamerauLevenshtein;

/// Distance function for strings using the Jaro similarity factor.
///
/// Returns a number in `[0, 100]`, inversely proportional to the similarity between
/// the two strings, where 0 is exactly the same, and 100 is not at all similar.
pub struct Jaro;

/// Distance function for strings using the Jaro similarity factor,
/// optimized for when strings are expected to have a common prefix.
///
/// Returns a number in `[0, 100]`, inversely proportional to the similarity between
/// the two strings, where 0 is exactly the same, and 100 is not at all similar.
pub struct JaroWinkler;

macro_rules! impl_dist_fn {
    ($($ty:ty = $distfn:path),*) => (
        $(
            impl<'a> DistFn<&'a str> for $ty {
                fn dist(&self, left: &&'a str, right: &&'a str) -> u64 {
                    $distfn(left, right) as u64
                }
            }

            impl DistFn<String> for $ty {
                fn dist(&self, left: &String, right: &String) -> u64 {
                    $distfn(left, right) as u64
                }
            }
        )*
    )
}

impl_dist_fn! {
    Hamming = hamming_dist,
    Levenshtein = strsim::levenshtein,
    DamerauLevenshtein = strsim::damerau_levenshtein,
    Jaro = jaro_factor,
    JaroWinkler = jaro_winkler_factor
}

impl<'a> KnownDist for &'a str {
    /// The fastest distance function for strings.
    type DistFn = Hamming;
    fn dist_fn() -> Hamming { Hamming }
}

impl KnownDist for String {
    /// The fastest distance function for strings.
    type DistFn = Hamming;
    fn dist_fn() -> Hamming { Hamming }
}

fn hamming_dist(left: &str, right: &str) -> u64 {
    let len = cmp::min(left.len(), right.len());
    let diff = cmp::max(left.len(), right.len()) - len;

    (strsim::hamming(&left[..len], &right[..len]).unwrap() + diff) as u64
}

fn jaro_factor(left: &str, right: &str) -> u64 {
    strsim::jaro(left, right).mul_add(JARO_SCALE_FACTOR, -JARO_SCALE_FACTOR).round() as u64
}

fn jaro_winkler_factor(left: &str, right: &str) -> u64 {
    strsim::jaro_winkler(left, right).mul_add(JARO_SCALE_FACTOR, -JARO_SCALE_FACTOR).round() as u64
}

