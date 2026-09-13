//! Shared kernel traits and combinators.
//!
//! Core volume traits live in [`traits`], boundary traits in
//! [`boundary_traits`]; statically dispatched tensor sums in [`tensor_sum`],
//! dynamic sums/sets in [`residual_sums`], facet-term containers in
//! [`boundary_terms`], and sum-factorized assembly loops in
//! [`tensor_assemble`].

pub mod boundary_terms;
pub mod boundary_traits;
pub mod residual_sums;
pub mod tensor_assemble;
pub mod tensor_sum;
pub mod traits;

pub use boundary_terms::*;
pub use boundary_traits::*;
pub use residual_sums::*;
pub use tensor_assemble::*;
pub use tensor_sum::*;
pub use traits::*;
