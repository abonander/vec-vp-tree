// Copyright 2016 Austin Bonander
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
// Distance-function trait and helper types for `VpTree`.

pub mod num;

#[cfg(feature = "strsim")]
pub mod string;

/// Describes a type which can act as a distance-function for `T`.
///
/// Implemented for `Fn(&T, &T) -> u64`.
///
/// Default implementations are provided for common numeric types.
pub trait DistFn<T> {
    /// Return the distance between `left` and `right`.
    ///
    /// ## Note
    /// It is a logic error for this method to return different values for the same operands,
    /// regardless of order (i.e. it is required to be idempotent and commutative).
    fn dist(&self, left: &T, right: &T) -> u64;
}

/// Simply calls `(self)(left, right)`
impl<T, F> DistFn<T> for F
    where F: Fn(&T, &T) -> u64
{
    fn dist(&self, left: &T, right: &T) -> u64 {
        (self)(left, right)
    }
}

/// Trait describing a type where a default distance function is known.
pub trait KnownDist: Sized {
    /// The known distance function for `Self`.
    type DistFn: DistFn<Self> + Sized;

    /// Return an instance of `DistFn`.
    fn dist_fn() -> Self::DistFn;
}
