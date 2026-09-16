//! Sum-factorized tensor-product drift-flux kernels.
//!
//! Four-field 2D state `[u, v, p, alpha]`, three-field 1D state
//! `[u, p, alpha]`; shared parameters in [`super::config`] /
//! [`super::config_1d`], closures in [`super::closures`]. Tensor-only by
//! design (no weak-form counterparts); split-form advection volumes pair with
//! the drift split-flux boundary.

pub mod directional_do_nothing_drift;
pub mod drift_flux;
pub mod drift_flux_1d;
pub mod free_surface;
pub mod gravity_buoyancy;
pub mod gravity_buoyancy_1d;
pub mod momentum_convection_split;
pub mod momentum_convection_split_1d;
pub mod pressure_advection_split;
pub mod pressure_advection_split_1d;
pub mod pressure_diffusion;
pub mod pressure_diffusion_1d;
pub mod pressure_divergence_mixture;
pub mod pressure_divergence_mixture_1d;
pub mod pressure_gradient_mixture;
pub mod pressure_gradient_mixture_1d;
pub mod split_boundary_flux_drift;
pub mod turbulent_dispersion_1d;
pub mod turbulent_dispersion_2d;
pub mod viscous_stress_mixture;
pub mod viscous_stress_mixture_1d;
pub mod void_advection_split;
pub mod void_advection_split_1d;

pub use directional_do_nothing_drift::TensorDriftDirectionalDoNothing2D;
pub use drift_flux::TensorDriftFlux2D;
pub use drift_flux_1d::TensorDriftFlux1D;
pub use free_surface::TensorDriftFreeSurface2D;
pub use gravity_buoyancy::TensorDriftGravity2D;
pub use gravity_buoyancy_1d::TensorDriftGravity1D;
pub use momentum_convection_split::TensorDriftMomentumConvectionSplit2D;
pub use momentum_convection_split_1d::TensorDriftMomentumConvectionSplit1D;
pub use pressure_advection_split::TensorDriftPressureAdvectionSplit2D;
pub use pressure_advection_split_1d::TensorDriftPressureAdvectionSplit1D;
pub use pressure_diffusion::TensorDriftPressureDiffusion2D;
pub use pressure_diffusion_1d::TensorDriftPressureDiffusion1D;
pub use pressure_divergence_mixture::TensorDriftPressureDivergence2D;
pub use pressure_divergence_mixture_1d::TensorDriftPressureDivergence1D;
pub use pressure_gradient_mixture::TensorDriftPressureGradient2D;
pub use pressure_gradient_mixture_1d::TensorDriftPressureGradient1D;
pub use split_boundary_flux_drift::TensorDriftSplitBoundaryFlux2D;
pub use turbulent_dispersion_1d::TensorDriftTurbulentDispersion1D;
pub use turbulent_dispersion_2d::TensorDriftTurbulentDispersion2D;
pub use viscous_stress_mixture::TensorDriftViscousStress2D;
pub use viscous_stress_mixture_1d::TensorDriftViscousStress1D;
pub use void_advection_split::TensorDriftVoidAdvectionSplit2D;
pub use void_advection_split_1d::TensorDriftVoidAdvectionSplit1D;
