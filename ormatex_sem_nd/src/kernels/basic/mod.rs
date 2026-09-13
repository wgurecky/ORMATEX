//! Basic kernels common to different physics problems.
//!
//! Weak-form volumes and boundaries live in [`weak`], sum-factorized
//! counterparts in [`tensor`]. The EDAC family lives in [`super::edac`];
//! shared traits and combinators live in [`super::common`].

pub mod kernel_neumann_flux;
pub mod kernel_robin_convection;
pub mod tensor;
pub mod weak;

pub use kernel_neumann_flux::NeumannFlux;
pub use kernel_robin_convection::RobinConvection;
pub use tensor::{
    TensorKernelAdvDiff, TensorKernelAdvDiff2D, TensorKernelAdvDiffSUPG, TensorKernelAdvDiffSUPG2D,
    TensorKernelAdvection2D, TensorKernelAdvectionOutflow2D, TensorKernelBoussinesq2D,
    TensorKernelConservationLaw1D, TensorKernelDiffusion, TensorKernelDiffusion2D,
    TensorKernelEnergyAdvectionDiffusion2D, TensorKernelLinearReaction, TensorKernelMass,
    TensorKernelVolumeSource,
};
pub use weak::{
    KernelAdvDiff, KernelAdvDiff2D, KernelAdvDiffSUPG, KernelAdvDiffSUPG2D, KernelAdvection2D,
    KernelAdvectionOutflow1D, KernelAdvectionOutflow2D, KernelBoussinesq2D, KernelConservationLaw1D, KernelDiffusion,
    KernelDiffusion2D, KernelEnergyAdvectionDiffusion2D, KernelLinearReaction, KernelMass,
    KernelVolumeSource,
};
