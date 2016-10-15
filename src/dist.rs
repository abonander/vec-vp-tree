// Copyright 2016 Austin Bonander
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//! Distance-function trait and helper types for `VpTree`.

/// Describes a type which can act as a distance-function for `T`.
///
/// Implemented for `Fn(&T, &T) -> u64`.
///
/// ## Exaple (Unsigned Integers)
///
/// ```rust
/// |left, right| if left < right { right - left } else { left - right } as u64
/// ```
///
/// ## Example (Signed Integers)
///
/// ```rust
/// |left, right| (left - right).abs() as u64
/// ```
pub trait DistFn<T> {
    /// Return the distance between `left` and `right`.
    ///
    /// ## Note
    /// It is a logic error for this method to return different values for the same operands,
    /// regardless of order (i.e. it is required to be idempotent and commutative).
    fn dist(&self, left: &T, right: &T) -> u64;
}

/// Simply calls `(self)(left, right)`
impl<T, F> DistFn<T> for F where F: Fn(&T, &T) -> u64 {
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

/// Structs implementing `DistFn`.
pub mod impls {
    use super::{DistFn, KnownDist};

    /// Implements `DistFn` for signed integers.
    pub struct SignedDist;

    macro_rules! impl_signed_dist {
        ($($ty:ty),*) => (
            $(
                impl DistFn<$ty> for SignedDist {
                    /// Returns `(left - right).abs() as u64`
                    fn dist(&self, left: &$ty, right: &$ty) -> u64 {
                        (left - right).abs() as u64
                    }
                }

                impl KnownDist for $ty {
                    type DistFn = SignedDist;

                    fn dist_fn() -> SignedDist { SignedDist }
                }
            )*
        )
    }

    impl_signed_dist! { i8, i16, i32, i64, isize }

    /// Implements `DistFn` for unsigned integers.
    pub struct UnsignedDist;

    macro_rules! impl_unsigned_dist {
        ($($ty:ty),*) => (
            $(
                impl DistFn<$ty> for UnsignedDist {
                    /// Returns ` if left < right { left - right } else { right - left } as u64`
                    fn dist(&self, left: &$ty, right: &$ty) -> u64 {
                        let dist = if left < right { left - right } else { right - left };
                        dist as u64
                    }
                }

                impl KnownDist for $ty {
                    type DistFn = UnsignedDist;

                    fn dist_fn() -> UnsignedDist { UnsignedDist }
                }
            )*
        )
    }

    impl_unsigned_dist! { u8, u16, u32, u64, usize }

    /// Implements `DistFn` for floating-point numbers, which takes the absolute value of the
    /// difference and rounds to the nearest integer before casting to `u64`.
    pub struct FloatDist;

    /// Implements `DistFn` for floating-point numbers, which multiplies the difference by the
    /// contained value before taking the absolute value, rounding to the nearest integer,
    /// and casting to `u64`.
    pub struct ScaledFloatDist<T>(pub T);

    macro_rules! impl_float_dist {
        ($($ty:ty),*) => (
            $(
                impl DistFn<$ty> for FloatDist {
                    /// Returns `(left - right).abs().round() as u64`.
                    fn dist(&self, left: &$ty, right: &$ty) -> u64 {
                        (left - right).abs().round() as u64
                    }
                }

                impl KnownDist<$ty> for FloatDist {
                    type DistFn = FloatDist;

                    fn dist_fn() -> FloatDist { FloatDist }
                }

                impl DistFn<$ty> for ScaledFloatDist<$ty> {
                    /// Returns `((left - right) * self.0).abs().round() as u64`
                    fn dist(&self, left: &$ty, right: &$ty) -> u64 {
                        ((left - right) * self.0).abs().round() as u64
                    }
                }
            )*
        )
    }
}
