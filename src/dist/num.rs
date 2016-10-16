// Copyright 2016 Austin Bonander
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Distance functions for numeric types.

use super::{DistFn, KnownDist};

/// Distance function for signed integers.
///
/// Returns `(left - right).abs() as u64`
#[derive(Copy, Clone)]
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

/// Distance function for unsigned integers.
///
/// Returns ` if left < right { left - right } else { right - left } as u64`
#[derive(Copy, Clone)]
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

/// Implements `DistFn` for floating-point numbers.
///
/// Returns `(left - right).abs().round() as u64`.
#[derive(Copy, Clone)]
pub struct FloatDist;

/// Implements `DistFn` for floating-point numbers with a scaling factor.
///
/// Returns `((left - right) * self.0).abs().round() as u64`
#[derive(Copy, Clone)]
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

                impl KnownDist for $ty {
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

impl_float_dist! { f32, f64 }
