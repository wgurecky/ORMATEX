//! Spectral-element finite-element infrastructure built on `ndmesh`.

// Link the BLAS/LAPACK implementation into binaries using this crate.
#[cfg(target_os = "linux")]
extern crate blas_src;
#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(target_os = "linux")]
extern crate lapack_src;
#[cfg(target_os = "macos")]
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;

pub mod common;
pub mod fields;
pub mod io;
pub mod jacobian;
pub mod kernels;
pub mod material;
mod op;
pub mod regions;
pub mod sem_1d;
pub mod sem_2d;
pub mod sem_traits;
mod simd;

pub use common::{
    dirichlet_values_with_precedence, BoundaryContributions, BoundaryFacet, CellState, FacetCtx,
    LocalCtx, ShapeFn, StateBoundaryContributions, TensorCtx, TensorFacetCtx,
};
pub use fields::{FieldRegistry, FieldValues};
pub use io::gmsh::{gmsh_quad_data, gmsh_quad_mesh, GmshQuadData, QuadMesh};
pub use jacobian::{
    CompleteResidualOperator, MatrixFreeJacobianProblem, MatrixFreeJacobianSource,
    MatrixFreeMinvJacobian, OwnedMinvJacobian, ParallelOwnedMinvJacobian,
};
pub use kernels::{
    BilinearForm, BoundaryIntegrator, DistributionParameter, DriftFlux1DConfig, DriftFlux2DConfig,
    DriftOutflow1D, EdacNavierStokes1DConfig, EdacNavierStokes2DConfig, FluxKernel1D,
    IshiiZuberParams, KernelAdvDiff, KernelAdvDiff2D, KernelAdvDiffSUPG, KernelAdvDiffSUPG2D,
    KernelAdvection2D, KernelAdvectionOutflow1D, KernelAdvectionOutflow2D, KernelBoussinesq2D,
    KernelConservationLaw1D, KernelDiffusion, KernelDiffusion2D, KernelEdacDirectionalDoNothing2D,
    KernelEdacDongOutflow2D, KernelEdacMomentumConvection2D, KernelEdacMomentumConvectionSplit1D,
    KernelEdacMomentumConvectionSplit2D, KernelEdacNavierStokes2D, KernelEdacNavierStokesSplit1D,
    KernelEdacNavierStokesSplit2D, KernelEdacNoSlipWall2D, KernelEdacPressureAdvection2D,
    KernelEdacPressureAdvectionSplit1D, KernelEdacPressureAdvectionSplit2D,
    KernelEdacPressureDiffusion1D, KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence1D,
    KernelEdacPressureDivergence2D, KernelEdacPressureGradient1D, KernelEdacPressureGradient2D,
    KernelEdacSlipWall2D, KernelEdacSplitBoundaryFlux2D, KernelEdacViscousStress1D,
    KernelEdacViscousStress2D, KernelEnergyAdvectionDiffusion1D, KernelEnergyAdvectionDiffusion2D,
    KernelLinearReaction, KernelMass, KernelVolumeSource, LinearForm, NeumannFlux, ResidualKernel,
    ResidualKernelSet, ResidualKernelSum, RobinConvection, SmagorinskyLilly2D,
    StateBoundaryIntegrator, StateBoundaryTerms, StateTensorBoundaryIntegrator,
    StateTensorBoundaryTerms, TensorDriftDirectionalDoNothing2D, TensorDriftFlux1D,
    TensorDriftFlux2D, TensorDriftFreeSurface2D, TensorDriftGravity1D, TensorDriftGravity2D,
    TensorDriftMomentumConvectionSplit1D, TensorDriftMomentumConvectionSplit2D,
    TensorDriftPressureAdvectionSplit1D, TensorDriftPressureAdvectionSplit2D,
    TensorDriftPressureDiffusion1D, TensorDriftPressureDiffusion2D,
    TensorDriftPressureDivergence1D, TensorDriftPressureDivergence2D,
    TensorDriftPressureGradient1D, TensorDriftPressureGradient2D, TensorDriftSplitBoundaryFlux2D,
    TensorDriftTurbulentDispersion1D, TensorDriftTurbulentDispersion2D, TensorDriftViscousStress1D,
    TensorDriftViscousStress2D, TensorDriftVoidAdvectionSplit1D, TensorDriftVoidAdvectionSplit2D,
    TensorKernelAdvDiff, TensorKernelAdvDiff2D, TensorKernelAdvDiffSUPG, TensorKernelAdvDiffSUPG2D,
    TensorKernelAdvection2D, TensorKernelAdvectionOutflow2D, TensorKernelBoussinesq2D,
    TensorKernelConservationLaw1D, TensorKernelDiffusion, TensorKernelDiffusion2D,
    TensorKernelEdacDirectionalDoNothing2D, TensorKernelEdacDongOutflow2D,
    TensorKernelEdacMomentumConvection2D, TensorKernelEdacMomentumConvectionSplit1D,
    TensorKernelEdacMomentumConvectionSplit2D, TensorKernelEdacNavierStokes2D,
    TensorKernelEdacNavierStokesSplit1D, TensorKernelEdacNavierStokesSplit2D,
    TensorKernelEdacNoSlipWall2D, TensorKernelEdacPressureAdvection2D,
    TensorKernelEdacPressureAdvectionSplit1D, TensorKernelEdacPressureAdvectionSplit2D,
    TensorKernelEdacPressureDiffusion1D, TensorKernelEdacPressureDiffusion2D,
    TensorKernelEdacPressureDivergence1D, TensorKernelEdacPressureDivergence2D,
    TensorKernelEdacPressureGradient1D, TensorKernelEdacPressureGradient2D,
    TensorKernelEdacSlipWall2D, TensorKernelEdacSplitBoundaryFlux2D,
    TensorKernelEdacViscousStress1D, TensorKernelEdacViscousStress2D,
    TensorKernelEnergyAdvectionDiffusion1D, TensorKernelEnergyAdvectionDiffusion2D,
    TensorKernelLinearReaction, TensorKernelMass, TensorKernelVolumeSource, TensorResidualKernel,
    TensorResidualKernelSet, TensorResidualKernelSum, ALPHA_1D, ALPHA_2D,
};
pub use material::{
    Coefficient, ConstantCoefficient, FrozenFacetField, FrozenQuadratureField, FrozenVelocity2D,
    MaterialContext, MaterialProperty, RegionCoefficient,
};
pub use op::ParCsrJacobian;
pub use regions::{CellMeta, FacetMeta, MeshMetadata, PhysicalRegion, PhysicalSelector};
pub use sem_1d::{
    BoundaryPoint, DofReduction1D, SEM1DMixedResidualOperator, SEM1DProblem,
    SEM1DResidualExecution, SEM1DResidualOperator, SEM1DTensorResidualOperator,
};
pub use sem_2d::{
    DofReduction2D, SEM2DMixedResidualOperator, SEM2DProblem, SEM2DResidualExecution,
    SEM2DResidualOperator, SEM2DTensorResidualOperator,
};
pub use sem_traits::{BilinearOps, TensorResidualOps, WeakResidualOps};
