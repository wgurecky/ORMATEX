//! Weak-form EDAC kernels (`[u, v, p]`).
//!
//! Sum-factorized counterparts live in [`super::tensor`]; shared parameters
//! in [`super::config`].

pub mod directional_do_nothing;
pub mod dong_outflow;
pub mod momentum_convection;
pub mod momentum_convection_split;
pub mod momentum_convection_split_1d;
pub mod navier_stokes;
pub mod navier_stokes_split;
pub mod navier_stokes_split_1d;
pub mod no_slip_wall;
pub mod pressure_advection;
pub mod pressure_advection_split;
pub mod pressure_advection_split_1d;
pub mod pressure_diffusion;
pub mod pressure_diffusion_1d;
pub mod pressure_divergence;
pub mod pressure_divergence_1d;
pub mod pressure_gradient;
pub mod pressure_gradient_1d;
pub mod slip_wall;
pub mod split_boundary_flux;
pub mod viscous_stress;
pub mod viscous_stress_1d;

pub use directional_do_nothing::KernelEdacDirectionalDoNothing2D;
pub use dong_outflow::KernelEdacDongOutflow2D;
pub use momentum_convection::KernelEdacMomentumConvection2D;
pub use momentum_convection_split::KernelEdacMomentumConvectionSplit2D;
pub use momentum_convection_split_1d::KernelEdacMomentumConvectionSplit1D;
pub use navier_stokes::KernelEdacNavierStokes2D;
pub use navier_stokes_split::KernelEdacNavierStokesSplit2D;
pub use navier_stokes_split_1d::KernelEdacNavierStokesSplit1D;
pub use no_slip_wall::KernelEdacNoSlipWall2D;
pub use pressure_advection::KernelEdacPressureAdvection2D;
pub use pressure_advection_split::KernelEdacPressureAdvectionSplit2D;
pub use pressure_advection_split_1d::KernelEdacPressureAdvectionSplit1D;
pub use pressure_diffusion::KernelEdacPressureDiffusion2D;
pub use pressure_diffusion_1d::KernelEdacPressureDiffusion1D;
pub use pressure_divergence::KernelEdacPressureDivergence2D;
pub use pressure_divergence_1d::KernelEdacPressureDivergence1D;
pub use pressure_gradient::KernelEdacPressureGradient2D;
pub use pressure_gradient_1d::KernelEdacPressureGradient1D;
pub use slip_wall::KernelEdacSlipWall2D;
pub use split_boundary_flux::KernelEdacSplitBoundaryFlux2D;
pub use viscous_stress::KernelEdacViscousStress2D;
pub use viscous_stress_1d::KernelEdacViscousStress1D;
