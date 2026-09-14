//! Weak-form drift-flux endpoint kernels (1D only).
//!
//! Volumes are tensor-only by design; this module holds the single outflow
//! endpoint term the 1D tensor operator consumes via `with_state_boundary`
//! (same pairing as
//! [`KernelAdvectionOutflow1D`](crate::kernels::KernelAdvectionOutflow1D)).

pub mod drift_outflow_1d;

pub use drift_outflow_1d::DriftOutflow1D;
