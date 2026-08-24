//! Spectral-element finite-element infrastructure built on `ndmesh`.

pub mod common;
pub mod fields;
pub mod gmsh;
pub mod jacobian;
pub mod kernels;
pub mod material;
pub mod sem_1d;
pub mod sem_2d;

pub use common::{
    BoundaryContributions, BoundaryFacet, CellState, FacetCtx, LocalCtx, ShapeFn,
    StateBoundaryContributions,
};
pub use fields::{FieldRegistry, FieldValues};
pub use gmsh::{gmsh_quad_data, gmsh_quad_mesh, GmshQuadData, QuadMesh};
pub use jacobian::{
    CompleteResidualOperator, MatrixFreeJacobianProblem, MatrixFreeJacobianSource,
    MatrixFreeMinvJacobian, OwnedMinvJacobian,
};
pub use kernels::{
    BilinearForm, BoundaryIntegrator, EdacNavierStokes2DConfig, FluxKernel1D, KernelAdvDiff,
    KernelAdvDiff2D, KernelAdvDiffSUPG, KernelAdvDiffSUPG2D, KernelAdvection2D,
    KernelConservationLaw1D, KernelDiffusion, KernelDiffusion2D, KernelEdacDongOutflow2D,
    KernelEdacMomentumConvection2D, KernelEdacMomentumConvectionSplit2D, KernelEdacNavierStokes2D,
    KernelEdacPressureAdvection2D, KernelEdacPressureAdvectionSplit2D,
    KernelEdacPressureDiffusion2D, KernelEdacPressureDivergence2D, KernelEdacPressureGradient2D,
    KernelEdacSplitBoundaryFlux2D, KernelEdacViscousStress2D, KernelLinearReaction, KernelMass,
    KernelVolumeSource, LinearForm, NeumannFlux, ResidualKernel, ResidualKernelSum,
    RobinConvection, SmagorinskyLilly2D, StateBoundaryIntegrator, StateBoundaryTerms,
};
pub use material::{
    CellMeta, Coefficient, ConstantCoefficient, FacetMeta, MaterialContext, MaterialProperty,
    MeshMetadata, PhysicalRegion, RegionCoefficient,
};
pub use sem_1d::{BoundaryPoint, DofReduction1D, SEM1DProblem, SEM1DResidualOperator};
pub use sem_2d::{DofReduction2D, SEM2DProblem, SEM2DResidualOperator};
