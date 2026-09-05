//! Reusable finite-element kernels.

pub mod kernel_adv_diff_1d;
pub mod kernel_adv_diff_2d;
pub mod kernel_adv_diff_supg_1d;
pub mod kernel_adv_diff_supg_2d;
pub mod kernel_advection_2d;
pub mod kernel_boussinesq_2d;
pub mod kernel_common;
pub mod kernel_conservation_law_1d;
pub mod kernel_diffusion_1d;
pub mod kernel_diffusion_2d;
pub mod kernel_edac_dong_outflow_2d;
pub mod kernel_edac_navier_stokes_2d;
pub mod kernel_edac_navier_stokes_2d_com;
pub mod kernel_edac_navier_stokes_2d_com_split;
pub mod kernel_edac_wall_2d;
pub mod kernel_linear_reaction;
pub mod kernel_mass;
pub mod kernel_neumann_flux;
pub mod kernel_robin_convection;
pub mod kernel_smagorinsky_lilly_2d;
pub mod kernel_volume_source;

pub use kernel_adv_diff_1d::{KernelAdvDiff, TensorKernelAdvDiff};
pub use kernel_adv_diff_2d::{KernelAdvDiff2D, TensorKernelAdvDiff2D};
pub use kernel_adv_diff_supg_1d::{KernelAdvDiffSUPG, TensorKernelAdvDiffSUPG};
pub use kernel_adv_diff_supg_2d::{KernelAdvDiffSUPG2D, TensorKernelAdvDiffSUPG2D};
pub use kernel_advection_2d::{KernelAdvection2D, TensorKernelAdvection2D};
pub use kernel_boussinesq_2d::{
    KernelBoussinesq2D, KernelEnergyAdvectionDiffusion2D, TensorKernelBoussinesq2D,
    TensorKernelEnergyAdvectionDiffusion2D,
};
pub use kernel_common::{
    BilinearForm, BoundaryIntegrator, FluxKernel1D, LinearForm, ResidualKernel, ResidualKernelSet,
    ResidualKernelSum, StateBoundaryIntegrator, StateBoundaryTerms, StateTensorBoundaryIntegrator,
    StateTensorBoundaryTerms, TensorResidualKernel, TensorResidualKernelSet,
    TensorResidualKernelSum,
};
pub use kernel_conservation_law_1d::{KernelConservationLaw1D, TensorKernelConservationLaw1D};
pub use kernel_diffusion_1d::{KernelDiffusion, TensorKernelDiffusion};
pub use kernel_diffusion_2d::{KernelDiffusion2D, TensorKernelDiffusion2D};
pub use kernel_edac_dong_outflow_2d::{
    KernelEdacDirectionalDoNothing2D, KernelEdacDongOutflow2D, KernelEdacSplitBoundaryFlux2D,
    TensorKernelEdacDirectionalDoNothing2D, TensorKernelEdacDongOutflow2D,
    TensorKernelEdacSplitBoundaryFlux2D,
};
pub use kernel_edac_navier_stokes_2d::{KernelEdacNavierStokes2D, TensorKernelEdacNavierStokes2D};
pub use kernel_edac_navier_stokes_2d_com::{
    EdacNavierStokes2DConfig, KernelEdacMomentumConvection2D, KernelEdacPressureAdvection2D,
    KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D,
    KernelEdacViscousStress2D, TensorKernelEdacMomentumConvection2D,
    TensorKernelEdacPressureAdvection2D, TensorKernelEdacPressureDiffusion2D,
    TensorKernelEdacPressureDivergence2D, TensorKernelEdacPressureGradient2D,
    TensorKernelEdacViscousStress2D,
};
pub use kernel_edac_navier_stokes_2d_com_split::{
    KernelEdacMomentumConvectionSplit2D, KernelEdacPressureAdvectionSplit2D,
    TensorKernelEdacMomentumConvectionSplit2D, TensorKernelEdacPressureAdvectionSplit2D,
};
pub use kernel_edac_wall_2d::{
    KernelEdacNoSlipWall2D, KernelEdacSlipWall2D, TensorKernelEdacNoSlipWall2D,
    TensorKernelEdacSlipWall2D,
};
pub use kernel_linear_reaction::{KernelLinearReaction, TensorKernelLinearReaction};
pub use kernel_mass::{KernelMass, TensorKernelMass};
pub use kernel_neumann_flux::NeumannFlux;
pub use kernel_robin_convection::RobinConvection;
pub use kernel_smagorinsky_lilly_2d::SmagorinskyLilly2D;
pub use kernel_volume_source::{KernelVolumeSource, TensorKernelVolumeSource};
