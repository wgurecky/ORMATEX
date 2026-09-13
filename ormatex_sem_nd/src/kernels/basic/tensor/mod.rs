//! Sum-factorized tensor-product kernels (`TensorResidualKernel`).
//!
//! Each newtype wraps its [`super::weak`] counterpart and delegates shared
//! parameters; weak-form originals live in [`super::weak`].

pub mod adv_diff_1d;
pub mod adv_diff_2d;
pub mod adv_diff_supg_1d;
pub mod adv_diff_supg_2d;
pub mod advection_2d;
pub mod advection_outflow_2d;
pub mod boussinesq_2d;
pub mod conservation_law_1d;
pub mod diffusion_1d;
pub mod diffusion_2d;
pub mod energy_advection_diffusion_2d;
pub mod linear_reaction;
pub mod mass;
pub mod volume_source;

pub use adv_diff_1d::TensorKernelAdvDiff;
pub use adv_diff_2d::TensorKernelAdvDiff2D;
pub use adv_diff_supg_1d::TensorKernelAdvDiffSUPG;
pub use adv_diff_supg_2d::TensorKernelAdvDiffSUPG2D;
pub use advection_2d::TensorKernelAdvection2D;
pub use advection_outflow_2d::TensorKernelAdvectionOutflow2D;
pub use boussinesq_2d::TensorKernelBoussinesq2D;
pub use conservation_law_1d::TensorKernelConservationLaw1D;
pub use diffusion_1d::TensorKernelDiffusion;
pub use diffusion_2d::TensorKernelDiffusion2D;
pub use energy_advection_diffusion_2d::TensorKernelEnergyAdvectionDiffusion2D;
pub use linear_reaction::TensorKernelLinearReaction;
pub use mass::TensorKernelMass;
pub use volume_source::TensorKernelVolumeSource;
