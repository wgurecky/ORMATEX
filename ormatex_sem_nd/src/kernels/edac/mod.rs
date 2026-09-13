//! Entropically damped artificial compressibility (EDAC) kernels (`[u, v, p]`).
//!
//! Layout: weak-form volumes and boundaries in [`weak`], sum-factorized
//! counterparts in [`tensor`], shared parameters in [`config`].
//!
//! Split-form pairing rule: split advection volumes integrate the
//! conservative half by parts, so they require a
//! [`SplitBoundaryFlux`](weak::split_boundary_flux::KernelEdacSplitBoundaryFlux2D)
//! default with [`Dong`](weak::dong_outflow::KernelEdacDongOutflow2D) or
//! [`directional-do-nothing`](weak::directional_do_nothing::KernelEdacDirectionalDoNothing2D)
//! `.with_split_flux(true)` outlets (tensor backends use the [`tensor`]
//! mirrors). Walls use the no-slip/slip closures.

pub mod config;
pub mod smagorinsky_lilly;
pub mod tensor;
pub mod weak;

pub use config::EdacNavierStokes2DConfig;
pub use smagorinsky_lilly::SmagorinskyLilly2D;
pub use weak::{
    KernelEdacDirectionalDoNothing2D, KernelEdacDongOutflow2D, KernelEdacMomentumConvection2D,
    KernelEdacMomentumConvectionSplit2D, KernelEdacNavierStokes2D, KernelEdacNavierStokesSplit2D,
    KernelEdacNoSlipWall2D, KernelEdacPressureAdvection2D, KernelEdacPressureAdvectionSplit2D,
    KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D,
    KernelEdacSlipWall2D, KernelEdacSplitBoundaryFlux2D, KernelEdacViscousStress2D,
};
pub use tensor::{
    TensorKernelEdacDirectionalDoNothing2D, TensorKernelEdacDongOutflow2D,
    TensorKernelEdacMomentumConvection2D, TensorKernelEdacMomentumConvectionSplit2D,
    TensorKernelEdacNavierStokes2D, TensorKernelEdacNavierStokesSplit2D,
    TensorKernelEdacNoSlipWall2D, TensorKernelEdacPressureAdvection2D,
    TensorKernelEdacPressureAdvectionSplit2D, TensorKernelEdacPressureDiffusion2D,
    TensorKernelEdacPressureDivergence2D, TensorKernelEdacPressureGradient2D,
    TensorKernelEdacSlipWall2D, TensorKernelEdacSplitBoundaryFlux2D,
    TensorKernelEdacViscousStress2D,
};
