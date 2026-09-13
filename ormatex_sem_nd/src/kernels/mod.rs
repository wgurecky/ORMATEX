//! Reusable finite-element kernels.
//!
//! Layout: backend-first directories. Basic physics kernels shared across
//! problems live in [`basic`] (weak volumes/boundaries in `basic::weak`,
//! sum-factorized counterparts in `basic::tensor`), the EDAC family in
//! [`edac`] (likewise split, with shared parameters at its root). Shared
//! traits and combinators live in [`common`].

pub mod basic;
pub mod common;
pub mod edac;

pub use edac::{
    EdacNavierStokes2DConfig, KernelEdacDirectionalDoNothing2D, KernelEdacDongOutflow2D,
    KernelEdacMomentumConvection2D, KernelEdacMomentumConvectionSplit2D,
    KernelEdacNavierStokes2D, KernelEdacNavierStokesSplit2D, KernelEdacNoSlipWall2D,
    KernelEdacPressureAdvection2D, KernelEdacPressureAdvectionSplit2D,
    KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D,
    KernelEdacSlipWall2D, KernelEdacSplitBoundaryFlux2D, SmagorinskyLilly2D,
    TensorKernelEdacDirectionalDoNothing2D, TensorKernelEdacDongOutflow2D,
    TensorKernelEdacMomentumConvection2D, TensorKernelEdacMomentumConvectionSplit2D,
    TensorKernelEdacNavierStokes2D, TensorKernelEdacNavierStokesSplit2D,
    TensorKernelEdacNoSlipWall2D, TensorKernelEdacPressureAdvection2D,
    TensorKernelEdacPressureAdvectionSplit2D, TensorKernelEdacPressureDiffusion2D,
    TensorKernelEdacPressureDivergence2D, TensorKernelEdacPressureGradient2D,
    TensorKernelEdacSlipWall2D, TensorKernelEdacSplitBoundaryFlux2D,
    TensorKernelEdacViscousStress2D, KernelEdacViscousStress2D,
};
pub use basic::{
    KernelAdvDiff,     KernelAdvDiff2D, KernelAdvDiffSUPG, KernelAdvDiffSUPG2D, KernelAdvection2D,
    KernelAdvectionOutflow1D, KernelAdvectionOutflow2D, KernelBoussinesq2D, KernelConservationLaw1D, KernelDiffusion,
    KernelDiffusion2D, KernelEnergyAdvectionDiffusion2D, KernelLinearReaction, KernelMass,
    KernelVolumeSource, NeumannFlux, RobinConvection, TensorKernelAdvDiff, TensorKernelAdvDiff2D,
    TensorKernelAdvDiffSUPG, TensorKernelAdvDiffSUPG2D, TensorKernelAdvection2D,
    TensorKernelAdvectionOutflow2D, TensorKernelBoussinesq2D, TensorKernelConservationLaw1D,
    TensorKernelDiffusion, TensorKernelDiffusion2D, TensorKernelEnergyAdvectionDiffusion2D,
    TensorKernelLinearReaction, TensorKernelMass, TensorKernelVolumeSource,
};
pub use common::{
    BilinearForm, BoundaryIntegrator, FluxKernel1D, LinearForm, ResidualKernel, ResidualKernelSet,
    ResidualKernelSum, StateBoundaryIntegrator, StateBoundaryTerms, StateTensorBoundaryIntegrator,
    StateTensorBoundaryTerms, TensorResidualKernel, TensorResidualKernelSet,
    TensorResidualKernelSum,
};
