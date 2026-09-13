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

pub use basic::{
    KernelAdvDiff, KernelAdvDiff2D, KernelAdvDiffSUPG, KernelAdvDiffSUPG2D, KernelAdvection2D,
    KernelAdvectionOutflow1D, KernelAdvectionOutflow2D, KernelBoussinesq2D,
    KernelConservationLaw1D, KernelDiffusion, KernelDiffusion2D, KernelEnergyAdvectionDiffusion1D,
    KernelEnergyAdvectionDiffusion2D, KernelLinearReaction, KernelMass, KernelVolumeSource,
    NeumannFlux, RobinConvection, TensorKernelAdvDiff, TensorKernelAdvDiff2D,
    TensorKernelAdvDiffSUPG, TensorKernelAdvDiffSUPG2D, TensorKernelAdvection2D,
    TensorKernelAdvectionOutflow2D, TensorKernelBoussinesq2D, TensorKernelConservationLaw1D,
    TensorKernelDiffusion, TensorKernelDiffusion2D, TensorKernelEnergyAdvectionDiffusion1D,
    TensorKernelEnergyAdvectionDiffusion2D, TensorKernelLinearReaction, TensorKernelMass,
    TensorKernelVolumeSource,
};
pub use common::{
    BilinearForm, BoundaryIntegrator, FluxKernel1D, LinearForm, ResidualKernel, ResidualKernelSet,
    ResidualKernelSum, StateBoundaryIntegrator, StateBoundaryTerms, StateTensorBoundaryIntegrator,
    StateTensorBoundaryTerms, TensorResidualKernel, TensorResidualKernelSet,
    TensorResidualKernelSum,
};
pub use edac::{
    EdacNavierStokes1DConfig, EdacNavierStokes2DConfig, KernelEdacDirectionalDoNothing2D,
    KernelEdacDongOutflow2D, KernelEdacMomentumConvection2D, KernelEdacMomentumConvectionSplit1D,
    KernelEdacMomentumConvectionSplit2D, KernelEdacNavierStokes2D, KernelEdacNavierStokesSplit1D,
    KernelEdacNavierStokesSplit2D, KernelEdacNoSlipWall2D, KernelEdacPressureAdvection2D,
    KernelEdacPressureAdvectionSplit1D, KernelEdacPressureAdvectionSplit2D,
    KernelEdacPressureDiffusion1D, KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence1D,
    KernelEdacPressureDivergence2D, KernelEdacPressureGradient1D, KernelEdacPressureGradient2D,
    KernelEdacSlipWall2D, KernelEdacSplitBoundaryFlux2D, KernelEdacViscousStress1D,
    KernelEdacViscousStress2D, SmagorinskyLilly2D, TensorKernelEdacDirectionalDoNothing2D,
    TensorKernelEdacDongOutflow2D, TensorKernelEdacMomentumConvection2D,
    TensorKernelEdacMomentumConvectionSplit1D, TensorKernelEdacMomentumConvectionSplit2D,
    TensorKernelEdacNavierStokes2D, TensorKernelEdacNavierStokesSplit1D,
    TensorKernelEdacNavierStokesSplit2D, TensorKernelEdacNoSlipWall2D,
    TensorKernelEdacPressureAdvection2D, TensorKernelEdacPressureAdvectionSplit1D,
    TensorKernelEdacPressureAdvectionSplit2D, TensorKernelEdacPressureDiffusion1D,
    TensorKernelEdacPressureDiffusion2D, TensorKernelEdacPressureDivergence1D,
    TensorKernelEdacPressureDivergence2D, TensorKernelEdacPressureGradient1D,
    TensorKernelEdacPressureGradient2D, TensorKernelEdacSlipWall2D,
    TensorKernelEdacSplitBoundaryFlux2D, TensorKernelEdacViscousStress1D,
    TensorKernelEdacViscousStress2D,
};
