//! Weak-form volume kernels (`ResidualKernel` / `BilinearForm` / `LinearForm`).
//!
//! Sum-factorized counterparts live in [`super::tensor`].

pub mod adv_diff_1d;
pub mod adv_diff_2d;
pub mod adv_diff_supg_1d;
pub mod adv_diff_supg_2d;
pub mod advection_2d;
pub mod advection_outflow_1d;
pub mod advection_outflow_2d;
pub mod boussinesq_2d;
pub mod conservation_law_1d;
pub mod diffusion_1d;
pub mod diffusion_2d;
pub mod energy_advection_diffusion_1d;
pub mod energy_advection_diffusion_2d;
pub mod linear_reaction;
pub mod mass;
pub mod volume_source;

pub use adv_diff_1d::KernelAdvDiff;
pub use adv_diff_2d::KernelAdvDiff2D;
pub use adv_diff_supg_1d::KernelAdvDiffSUPG;
pub use adv_diff_supg_2d::KernelAdvDiffSUPG2D;
pub use advection_2d::KernelAdvection2D;
pub use advection_outflow_1d::KernelAdvectionOutflow1D;
pub use advection_outflow_2d::KernelAdvectionOutflow2D;
pub use boussinesq_2d::KernelBoussinesq2D;
pub use conservation_law_1d::KernelConservationLaw1D;
pub use diffusion_1d::KernelDiffusion;
pub use diffusion_2d::KernelDiffusion2D;
pub use energy_advection_diffusion_1d::KernelEnergyAdvectionDiffusion1D;
pub use energy_advection_diffusion_2d::KernelEnergyAdvectionDiffusion2D;
pub use linear_reaction::KernelLinearReaction;
pub use mass::KernelMass;
pub use volume_source::KernelVolumeSource;
