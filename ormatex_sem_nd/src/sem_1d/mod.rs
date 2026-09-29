//! 1D GLL spectral-element problem: weak, tensor, and bilinear paths.
//!
//! Assembly is split by family: [`weak`], [`tensor`], [`bilinear`];
//! residual operators live in [`operators`], shared traits in
//! `crate::sem_traits`. The [`tensor`] volume pass is a thin 1D adapter over
//! the shared [`crate::common::tensor_pass`] skeleton used by both 1D and 2D.

pub mod bilinear;
pub mod operators;
pub mod problem;
pub mod tensor;
pub mod weak;

pub use operators::{
    SEM1DMixedResidualOperator, SEM1DResidualExecution, SEM1DResidualOperator,
    SEM1DTensorResidualOperator,
};
pub use problem::{BoundaryPoint, DofReduction1D, SEM1DProblem};
