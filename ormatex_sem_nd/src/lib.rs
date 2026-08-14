//! Spectral-element finite-element infrastructure built on `ndmesh`.

pub mod common;
pub mod gmsh;
pub mod jacobian;
pub mod kernels;
pub mod material;
pub mod sem_1d;
pub mod sem_2d;

pub use common::{BoundaryContributions, BoundaryFacet, CellState, FacetCtx, LocalCtx, ShapeFn};
pub use gmsh::{gmsh_quad_data, gmsh_quad_mesh, GmshQuadData, QuadMesh};
pub use jacobian::{MatrixFreeJacobianProblem, MatrixFreeMinvJacobian, OwnedMinvJacobian};
pub use kernels::{
    BilinearForm, BoundaryIntegrator, FluxKernel1D, KernelAdvDiff, KernelAdvDiff2D,
    KernelAdvDiffSUPG, KernelAdvDiffSUPG2D, KernelConservationLaw1D, KernelDiffusion,
    KernelDiffusion2D, KernelLinearReaction, KernelMass, KernelVolumeSource, LinearForm,
    NeumannFlux, ResidualKernel, RobinConvection,
};
pub use material::{
    CellMeta, Coefficient, ConstantCoefficient, FacetMeta, MaterialContext, MaterialProperty,
    MeshMetadata, PhysicalRegion, RegionCoefficient,
};
pub use sem_1d::{BoundaryPoint, DofReduction1D, SEM1DProblem};
pub use sem_2d::{DofReduction2D, SEM2DProblem};
