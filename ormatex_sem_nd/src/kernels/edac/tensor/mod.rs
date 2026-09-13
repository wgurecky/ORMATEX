//! Sum-factorized tensor-product EDAC kernels (`[u, v, p]`).
//!
//! Weak-form originals live in [`super::weak`]; shared parameters in
//! [`super::config`]. Split-form volumes pair with a split-flux boundary
//! (see [`super::weak`] boundary docs).

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

pub use directional_do_nothing::TensorKernelEdacDirectionalDoNothing2D;
pub use dong_outflow::TensorKernelEdacDongOutflow2D;
pub use momentum_convection::TensorKernelEdacMomentumConvection2D;
pub use momentum_convection_split::TensorKernelEdacMomentumConvectionSplit2D;
pub use momentum_convection_split_1d::TensorKernelEdacMomentumConvectionSplit1D;
pub use navier_stokes::TensorKernelEdacNavierStokes2D;
pub use navier_stokes_split::TensorKernelEdacNavierStokesSplit2D;
pub use navier_stokes_split_1d::TensorKernelEdacNavierStokesSplit1D;
pub use no_slip_wall::TensorKernelEdacNoSlipWall2D;
pub use pressure_advection::TensorKernelEdacPressureAdvection2D;
pub use pressure_advection_split::TensorKernelEdacPressureAdvectionSplit2D;
pub use pressure_advection_split_1d::TensorKernelEdacPressureAdvectionSplit1D;
pub use pressure_diffusion::TensorKernelEdacPressureDiffusion2D;
pub use pressure_diffusion_1d::TensorKernelEdacPressureDiffusion1D;
pub use pressure_divergence::TensorKernelEdacPressureDivergence2D;
pub use pressure_divergence_1d::TensorKernelEdacPressureDivergence1D;
pub use pressure_gradient::TensorKernelEdacPressureGradient2D;
pub use pressure_gradient_1d::TensorKernelEdacPressureGradient1D;
pub use slip_wall::TensorKernelEdacSlipWall2D;
pub use split_boundary_flux::TensorKernelEdacSplitBoundaryFlux2D;
pub use viscous_stress::TensorKernelEdacViscousStress2D;
pub use viscous_stress_1d::TensorKernelEdacViscousStress1D;
