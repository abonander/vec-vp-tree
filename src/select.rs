// Copyright 2016 Austin Bonander
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
use std::cmp::Ordering;
use std::mem;

// The threshold below which qselect_inplace_by() should just sort the slice.
const SORT_THRESH: usize = 6;

/// Use the Quickselect algorithm, given an ordering function,
/// to select the `k`th smallest element from `data`. As part of the
/// algorithm, `k` is moved to its final sorted position and the rest of the array is (at least)
/// partially sorted.
///
/// ##Panics
/// If `k` is greater than `data.len()`.
pub fn qselect_inplace_by<T, F>(data: &mut [T], k: usize, mut ord_fn: F) -> &mut T
where F: FnMut(&T, &T) -> Ordering {

    let len = data.len();

    assert!(k < len,
            "Called qselect_inplace with k = {} and data length: {}",
            k,
            len);

    if len < SORT_THRESH {
        data.sort_by(&mut ord_fn);
        return &mut data[k];
    }

    let pivot_idx = partition_by(data, &mut ord_fn);

    if k == pivot_idx {
        &mut data[pivot_idx]
    } else if k < pivot_idx {
        qselect_inplace_by(&mut data[..pivot_idx], k, ord_fn)
    } else {
        qselect_inplace_by(&mut data[pivot_idx + 1..], k - pivot_idx - 1, ord_fn)
    }
}

fn partition_around<T, F>(data: &mut [T], pivot_idx: usize, mut ord_fn: F) -> usize
    where F: FnMut(&T, &T) -> Ordering
{

    let len = data.len();

    data.swap(pivot_idx, len - 1);

    let mut curr = 0;

    for i in 0..len - 1 {
        if ord_fn(&data[i], &data[len - 1]) == Ordering::Less {
            data.swap(i, curr);
            curr += 1;
        }
    }

    data.swap(curr, len - 1);

    curr
}

/// Given an ordering function, pick an arbitrary pivot and partition the slice,
/// returning the pivot.
fn partition_by<T, F: FnMut(&T, &T) -> Ordering>(data: &mut [T], mut ord_fn: F) -> usize {
    let len = data.len();

    let pivot_idx = {
        let first = (&data[0], 0);
        let mid = (&data[len / 2], len / 2);
        let last = (&data[len - 1], len - 1);

        median_of_3_by(&first, &mid, &last, |left, right| ord_fn(left.0, right.0)).1
    };

    partition_around(data, pivot_idx, ord_fn)
}

/// Given an ordering function, of the three values passed, return the median.
fn median_of_3_by<T, F: FnMut(&T, &T) -> Ordering>(mut x: T,
                                                       mut y: T,
                                                       mut z: T,
                                                       mut ord_fn: F)
                                                       -> T {
    in_order_by(&mut x, &mut y, &mut ord_fn);
    in_order_by(&mut x, &mut z, &mut ord_fn);
    in_order_by(&mut y, &mut z, &mut ord_fn);

    y
}

/// Given an ordering function, if `x > y`, swap `x` and `y`.
#[inline]
fn in_order_by<T, F: FnMut(&T, &T) -> Ordering>(x: &mut T, y: &mut T, mut ord_fn: F) {
    if ord_fn(x, y) == Ordering::Greater {
        mem::swap(x, y);
    }
}
