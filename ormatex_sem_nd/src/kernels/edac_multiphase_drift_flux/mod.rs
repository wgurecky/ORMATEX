//! EDAC drift-flux kernels for dispersed vapor-in-liquid flow (`[u, v, p, alpha]`).
//!
//! Model (Ishii 1975 drift-flux on the EDAC base; dispersed-bubbly regime):
//!
//! ```text
//! d_t u_m + (u_m.grad) u_m = -grad(p)/rho_m + div(tau_m)/rho_m(scaled)
//!                             + (rho_m - rho_l)/rho_m g          (momentum)
//! d_t p + u_m.grad p + rho_m c0^2 div(u_m) = div(k grad(p))      (mixture EDAC)
//! d_t a + div(a (C0 u_m + V_drift)) = div(D_td grad(a))           (void)
//! ```
//!
//! with `rho_m = a rho_g + (1-a) rho_l` (linear), `mu_m` likewise,
//! `tau_m = 2 (nu_m + nu_t) S` (`nu_t` Smagorinsky-Lilly in 2D, laminar in 1D),
//! `u_g = C0 j + V_gj` (Ishii-Zuber `V_gj`, Dix `C0` optional), buoyant gravity
//! vanishing at `a = 0`, and constant-coefficient turbulent dispersion in 2D.
//! Compose the PDE from the per-term kernels in [`tensor`]; split-form
//! advection volumes pair with [`tensor::TensorDriftSplitBoundaryFlux2D`] and
//! the drift directional-do-nothing outlet in 2D, and with the weak
//! [`weak::DriftOutflow1D`] endpoint in 1D (the 1D tensor path has no
//! facet-kernel support; same pairing as
//! [`KernelAdvectionOutflow1D`](crate::kernels::KernelAdvectionOutflow1D)).
//!
//! Citations live on [`closures`]; the drift (momentum) stress
//! `-div(a(1-a) rho_g rho_l/rho_m V_gj V_gj)` is neglected in this phase.

pub mod closures;
pub mod config;
pub mod config_1d;
pub mod tensor;
pub mod weak;

pub use closures::{DistributionParameter, IshiiZuberParams, MIN_LIQUID_FRACTION};
pub use config::{DriftFlux2DConfig, ALPHA_2D};
pub use config_1d::{DriftFlux1DConfig, ALPHA_1D};
pub use tensor::{
    TensorDriftDirectionalDoNothing2D, TensorDriftFlux1D,     TensorDriftFlux2D, TensorDriftFreeSurface2D, TensorDriftGravity1D,
    TensorDriftGravity2D, TensorDriftMomentumConvectionSplit1D,
    TensorDriftMomentumConvectionSplit2D, TensorDriftPressureAdvectionSplit1D,
    TensorDriftPressureAdvectionSplit2D, TensorDriftPressureDiffusion1D,
    TensorDriftPressureDiffusion2D, TensorDriftPressureDivergence1D,
    TensorDriftPressureDivergence2D, TensorDriftPressureGradient1D, TensorDriftPressureGradient2D,
    TensorDriftSplitBoundaryFlux2D, TensorDriftTurbulentDispersion1D,
    TensorDriftTurbulentDispersion2D, TensorDriftViscousStress1D, TensorDriftViscousStress2D,
    TensorDriftVoidAdvectionSplit1D, TensorDriftVoidAdvectionSplit2D,
};
pub use weak::DriftOutflow1D;
