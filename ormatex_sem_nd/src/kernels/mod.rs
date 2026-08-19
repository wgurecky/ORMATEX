//! Reusable finite-element kernels.

pub mod kernel_adv_diff_1d;
pub mod kernel_adv_diff_2d;
pub mod kernel_adv_diff_supg_1d;
pub mod kernel_adv_diff_supg_2d;
pub mod kernel_advection_2d;
pub mod kernel_common;
pub mod kernel_conservation_law_1d;
pub mod kernel_diffusion_1d;
pub mod kernel_diffusion_2d;
pub mod kernel_edac_navier_stokes_2d;
pub mod kernel_edac_navier_stokes_2d_com;
pub mod kernel_linear_reaction;
pub mod kernel_mass;
pub mod kernel_neumann_flux;
pub mod kernel_robin_convection;
pub mod kernel_smagorinsky_lilly_2d;
pub mod kernel_volume_source;

pub use kernel_adv_diff_1d::KernelAdvDiff;
pub use kernel_adv_diff_2d::KernelAdvDiff2D;
pub use kernel_adv_diff_supg_1d::KernelAdvDiffSUPG;
pub use kernel_adv_diff_supg_2d::KernelAdvDiffSUPG2D;
pub use kernel_advection_2d::KernelAdvection2D;
pub use kernel_common::{
    BilinearForm, BoundaryIntegrator, FluxKernel1D, LinearForm, ResidualKernel, ResidualKernelSum,
};
pub use kernel_conservation_law_1d::KernelConservationLaw1D;
pub use kernel_diffusion_1d::KernelDiffusion;
pub use kernel_diffusion_2d::KernelDiffusion2D;
pub use kernel_edac_navier_stokes_2d::KernelEdacNavierStokes2D;
pub use kernel_edac_navier_stokes_2d_com::{
    EdacNavierStokes2DConfig, KernelEdacMomentumConvection2D, KernelEdacPressureAdvection2D,
    KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D,
    KernelEdacViscousStress2D,
};
pub use kernel_linear_reaction::KernelLinearReaction;
pub use kernel_mass::KernelMass;
pub use kernel_neumann_flux::NeumannFlux;
pub use kernel_robin_convection::RobinConvection;
pub use kernel_smagorinsky_lilly_2d::SmagorinskyLilly2D;
pub use kernel_volume_source::KernelVolumeSource;
