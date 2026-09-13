//! 2D GLL spectral-element problem: weak, tensor, and bilinear paths.
//!
//! Assembly is split by family: [`weak`], [`tensor`], [`bilinear`];
//! residual operators live in [`operators`], shared traits in
//! `crate::sem_traits`.

pub mod bilinear;
pub mod operators;
pub mod problem;
pub mod tensor;
pub mod weak;

pub use operators::{
    SEM2DMixedResidualOperator, SEM2DResidualExecution, SEM2DResidualOperator,
    SEM2DTensorResidualOperator,
};
pub use problem::{DofReduction2D, SEM2DProblem};
