//! Shared finite-element infrastructure for the `ex_nd_*` examples.
//!
//! Holds the kernel abstraction (`LocalCtx`, `ShapeFn`, `BilinearForm`,
//! `LinearForm`, `BoundaryIntegrator`) plus the lumped-mass ODE system
//! (`AdvDiffSys`, `MinvKLinOp`) so both the 1D and 2D examples can consume
//! the same kernel API.  Per-example FE problem structs (mesh assembly, BC
//! reduction, dof LUTs) live in the example files themselves -- the common
//! surface here is exactly what a kernel author needs.
//!
//! Design (Phase 1.5, MOOSE-style `integrand` ergonomics):
//!   * `BilinearForm::integrand` -- bare physics at one quadrature point for
//!     one `(test, trial)` pair, NO weight/jdet factor.  The provided
//!     `assemble_local` default owns the test x trial x q triple loop and
//!     multiplies by `wts[q] * jdets[q]`.  Override `assemble_local` for
//!     max-control kernels (SIMD fusion, cell-level SUPG tau precompute).
//!   * `LinearForm::integrand` -- RHS physics at one Q point for one test
//!     dof, with the same provided-default-impl pattern.  For source terms
//!     and volumetric forcing.
//!   * `BoundaryIntegrator` -- Neumann/Robin facet kernels. The 2D problem
//!     owns quadrilateral boundary traversal and DOF reduction.

use faer::dyn_stack::{MemStack, StackReq};
use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::Triplet;
use faer::sparse::{SparseColMat, SparseColMatRef};
use ndelement::{
    ciarlet::LagrangeElementFamily,
    traits::{ElementFamily, FiniteElement},
    types::ReferenceCellType,
};
use ndfunctionspace::{traits::FunctionSpace, FunctionSpaceImpl};
use ndmesh::traits::{Entity, Geometry, GeometryMap, Mesh, Point, Topology};
use ormatex::ode_sys::OdeSys;
use quadraturerules::{single_integral_quadrature, Domain, QuadratureRule};
use rlst::{rlst_dynamic_array, DynArray};

// =============================================================================
// Per-cell-type cached data
// =============================================================================
//
// One entry per `ReferenceCellType` present in `mesh.entity_types(tdim)`.
// Carries the quadrature rule, the reference-cell tabulation (with `nderivs
// = 1`), and per-cell inverse-jacobians / determinants, so `assemble` can
// loop over cell types uniformly without knowing which quadrature rule or
// reference element each cell type used.  Shared between the 1D and 2D
// examples' FE problem structs.

/// Cached per-cell-type quadrature + tabulation + geometry-map data.
///
/// One entry per `ReferenceCellType` present in
/// `mesh.entity_types(mesh.topology_dim())`.  Built once by the FE
/// problem's constructor and reused for every `assemble` call.
pub struct CellData {
    /// Reference-cell quadrature weights, length `npts`.  Already
    /// normalized so `sum(wts) == reference-cell measure` (1 for unit
    /// interval/square, 1/2 for unit triangle).
    pub wts: Vec<f64>,

    /// Number of quadrature points on the cell (`wts.len()`).
    pub npts: usize,

    /// Number of local dofs on this cell type (`element.dim()`).
    pub ndofs: usize,

    /// Reference-cell tabulation with `nderivs=1` (values + first
    /// partials), shape `[deriv_count, npts, ndofs, ncomp=1]` per
    /// `element.tabulate_array_shape`.
    pub table: DynArray<f64, 4>,

    /// Reference basis values, packed `[ndofs, npts]`. Scalar Lagrange values
    /// are identical on every cell, so assembly borrows this cache directly.
    pub reference_values: Vec<f64>,

    /// Collocated GLL quadrature point for each local nodal DOF. Used by the
    /// lumped-mass fast path; `nodal_quadrature[dof]` indexes `wts`.
    pub nodal_quadrature: Vec<usize>,

    /// Per-cell inverse jacobians, `DynArray<f64, 4>` shape
    /// `[tdim, gdim, npts, ncells]`, indexed `[td, gd, q, c]`.
    pub jinv_cache: DynArray<f64, 4>,

    /// Jacobian determinants, flat `[ncells * npts]`, row-major `(c, q)`.
    pub jdets_cache: Vec<f64>,

    /// Physical quadrature coordinates, flat `[ncells * npts * gdim]` with
    /// coordinates contiguous for each point: `(c, q, gd)`.
    pub physical_points_cache: Vec<f64>,

    /// Reference-cell quadrature points, shape `[tdim, npts]`.
    /// Debug-only: stored for inspection post-construction, not read by
    /// `assemble`.  Kept `#[allow(dead_code)]` so callers that want to
    /// inspect quadrature node positions can do so.
    #[allow(dead_code)]
    pub pts: DynArray<f64, 2>,
}

// =============================================================================
// Kernel abstraction
//
// Designed to extend cleanly to 2D/3D, mixed tri/quad meshes, and vector FE:
//   * `LocalCtx` carries per-cell geometry (tdim, gdim), quadrature data
//     (wts, jdets), and already-physical (pushed-forward + relayouted)
//     values and gradients as `&[f64]` slices with documented strides.
//   * `ShapeFn` is a per-dof view offered in two reading styles: scalar-at-q
//     (`v(q)`, `grad(q, d)`) for readable physics, and slice (`v_slice()`,
//     `grad_slice(d)`) for vectorization.  Both read the same backing buffer.
//   * `BilinearForm::assemble_local` writes the full local element matrix
//     `[ndofs * ndofs * ncomp * ncomp]` into a caller-owned buffer that is
//     reused across cells.  No per-call `Vec`, no per-call heap allocation;
//     the kernel owns the test x trial x quadrature triple loop so it fuses
//     and inlines per concrete form (the trait object `dyn` is dropped).
//
// nderivs := 1 (values + first partials).  The reference tabulation
// `self.table[deriv, q, dof, comp]` has `deriv` slots: 0 = value,
// 1..=tdim = d/dx_d.  Physical gradients are precomputed in `assemble`
// (per example's FE problem struct):
//
//     grad_phys[dof, comp, gd, q] = sum_{td=0..tdim} jinv[td, gd, q] * grad_ref[1+td, q, dof, comp]
//
// (J^{-T} . grad_ref).  For 1D (tdim=gdim=1) this collapses to
// `jinv[0,0,q] * table[1,q,dof,0]`.
//
// `push_forward` would have been the idiomatic path, but nd's `IdentityMap`
// (used by `LagrangeElementFamily`) `unimplemented!()`'s for `nderivs > 0`,
// so the manual transform is kept in each example's `assemble`; the indexing
// is tdim/gdim-generic so 2D/3D does not require revisiting it.
// =============================================================================

/// Per-cell assembly context handed to a `BilinearForm` kernel.
///
/// All tabulation data is already physical (pushed-forward) and laid out
/// with `npts` as the contiguous (last) axis so kernels can borrow real
/// `&[f64]` slices.  Layouts (row-major):
///   * `values`: `[ndofs, ncomp, npts]`, indexed
///     `values[(i * ncomp + c) * npts + q]`.
///   * `grads` : `[ndofs, ncomp, gdim, npts]`, indexed
///     `grads[((i * ncomp + c) * gdim + d) * npts + q]`.
pub struct LocalCtx<'a> {
    /// Topological dimension of the cell -- the dimension of the *reference*
    /// element and of the mesh's connectivity graph.  Equals `gdim` for an
    /// embedded mesh where each cell's geometry lives in the same dimension
    /// it tiles (e.g. tri in 2D space, tet in 3D space), but can be smaller
    /// than `gdim` for cells embedded in a higher-dimensional ambient space
    /// (e.g. a 1D curve mesh in 3D space, or a 2D surface mesh in 3D).
    /// Range: 1D Interval = 1, 2D Tri/Quad = 2, 3D Tet/Hex = 3.
    /// Kernels use this to size the reference-derivative loop when summing
    /// `grad_phys . grad_phys` dot products.
    pub tdim: usize,

    /// Geometric (ambient) dimension of the mesh -- the number of Cartesian
    /// coordinates of each physical vertex.  Equals `tdim` for the common
    /// embedded case, but exceeds it for lower-dimensional manifolds in
    /// higher-dimensional space (curve in 3D, surface in 3D).
    /// Range: 1D mesh = 1, 2D mesh = 2, 3D mesh = 3.
    /// Physical gradients live in `R^gdim`; kernels size the gradient-component
    /// loop with this.  The 1D advection-diffusion kernel asserts
    /// `gdim == 1`; a 2D/3D variant reads a vector `vel` (length `gdim`).
    pub gdim: usize,

    /// Number of vector components per DOF -- the value-size of the finite
    /// element's push-forward.  `ncomp == 1` for scalar fields (Lagrange
    /// on any cell type); `ncomp == gdim` for `H(div)`/`H(curl)` fields
    /// (Raviart-Thomas, Nedelec) once those are adopted; `ncomp == gdim`
    /// or more for displacement-based vector FE (elasticity).
    /// Scalar kernels `assert_eq!(ctx.ncomp, 1)`.  The shape is designed so
    /// vector kernels can extend the test/trial loop with a `0..ncomp` inner
    /// loop without trait changes.
    pub ncomp: usize,

    /// Number of quadrature points on this cell.  Equal to the length of
    /// `wts` and `jdets`, and the contiguous last-axis length of `values`
    /// and `grads` (i.e. `ShapeFn::v_slice()` returns a slice of this length).
    /// Determined by the per-cell-type quadrature rule chosen by the FE
    /// problem -- for mass-lumped Lagrange it is `p + 1` GLL points (interval/
    /// quad) or 3 vertex nodes (P1 triangle); for a generic 2D/3D rule it is
    /// rule-specific (e.g. XiaoGimbutas on triangles).
    pub npts: usize,

    /// Number of local degrees of freedom on this cell -- the count of
    /// dofs returned by `space.entity_closure_dofs(cell_type, c)`.  For
    /// Lagrange P_p: `p+1` on an Interval, `(p+1)(p+2)/2` on a Triangle,
    /// `(p+1)^2` on a Quadrilateral, etc.  The kernel writes the local
    /// matrix `out: &mut [f64]` of length `ndofs * ndofs` (scalar) or
    /// `ndofs * ndofs * ncomp * ncomp` (vector FE).
    pub ndofs: usize,

    /// Quadrature weights on the *reference* cell, length `npts`.
    /// Already normalized so `sum(wts) == reference-cell measure` (e.g. 1
    /// for the unit interval, 1 for the unit square, 1/2 for the unit
    /// triangle -- the example's FE problem applies this area scaling when
    /// building `wts`).  The kernel combines `wts[q] * jdets[q]` to get the
    /// physical integral weight at point `q` -- DO NOT apply a second
    /// jacobian scaling.
    pub wts: &'a [f64],

    /// Jacobian determinant of the cell's reference->physical map at each
    /// quadrature point, length `npts`.  For an Interval on [0,1] with
    /// `nx` cells this is the cell length `1/nx` at every Q point; for a
    /// deformed 2D/3D cell it varies per Q point.  Always positive for a
    /// valid (orientation-preserving) mesh.  Multiply with `wts[q]` to
    /// form the physical integral weight.
    pub jdets: &'a [f64],

    /// Physical quadrature coordinates, packed `[npts, gdim]`. Access with
    /// `ctx.point(q)` for spatially varying material and source kernels.
    pub points: &'a [f64],

    /// Physical values of all basis functions on this cell, row-major
    /// `[ndofs, ncomp, npts]` with `npts` contiguous.  Already pushed
    /// forward (so for Lagrange on `IdentityMap` this is the reference
    /// tabulation; for Piola-mapped elements it is the transformed value).
    /// Access via `ctx.test(i, comp).v(q)` / `.v_slice()`; do not index
    /// directly.
    pub values: &'a [f64],

    /// Physical gradients of all basis functions on this cell, row-major
    /// `[ndofs, ncomp, gdim, npts]` with `npts` contiguous.  Computed as
    /// `J^{-T} . d phi/d xi` per Q point (for each `tdim` slot summed into
    /// `gdim` physical components).  Access via `ctx.test(i, comp).grad(q, d)`
    /// / `.grad_slice(d)`; do not index directly.
    pub grads: &'a [f64],
}

impl<'a> LocalCtx<'a> {
    pub fn point(&self, q: usize) -> &'a [f64] {
        &self.points[q * self.gdim..(q + 1) * self.gdim]
    }

    /// View of the shape function for test dof `i`, component `comp`.
    /// For Galerkin test == trial; `trial(i, comp)` returns an identical view.
    /// `test`/`trial` are separate so a future non-Galerkin `LocalCtx` (carrying
    /// two tabulations) can keep this signature.
    pub fn test(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        ShapeFn {
            npts: self.npts,
            ncomp: self.ncomp,
            gdim: self.gdim,
            values: self.values,
            grads: self.grads,
            i,
            comp,
        }
    }
    pub fn trial(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        self.test(i, comp)
    }
}

/// Per-dof view over tabulation data.  Exposes two reading styles:
/// scalar-at-q (`v(q)`, `grad(q, d)`) for readable physics, and slice
/// (`v_slice()`, `grad_slice(d)`) for vectorization.  Both read the same
/// backing slices.
///
/// Held-by-value (stack-only, no allocations): stores the tabulation
/// strides + borrowed `values`/`grads` slices + dof index + component
/// index.  `LocalCtx::test`/`::trial` and `FacetCtx::test`/`::trial` both
/// construct it from their own fields, so the same `ShapeFn` type serves
/// cell- and facet-kernel ergonomics without a trait abstraction.
pub struct ShapeFn<'a> {
    pub npts: usize,
    pub ncomp: usize,
    pub gdim: usize,
    pub values: &'a [f64],
    pub grads: &'a [f64],
    pub i: usize,
    pub comp: usize,
}

impl<'a> ShapeFn<'a> {
    /// Physical value at quadrature point `q`.
    pub fn v(&self, q: usize) -> f64 {
        self.values[(self.i * self.ncomp + self.comp) * self.npts + q]
    }
    /// Physical gradient component `gd` at quadrature point `q`.
    pub fn grad(&self, q: usize, gd: usize) -> f64 {
        self.grads[((self.i * self.ncomp + self.comp) * self.gdim + gd) * self.npts + q]
    }
    /// Slice of values at all quadrature points.  Length `npts`, contiguous.
    pub fn v_slice(&self) -> &'a [f64] {
        let start = (self.i * self.ncomp + self.comp) * self.npts;
        &self.values[start..start + self.npts]
    }
    /// Slice of gradient component `gd` at all quadrature points.  Length
    /// `npts`, contiguous.  Different `gd` are independent slices.
    pub fn grad_slice(&self, gd: usize) -> &'a [f64] {
        let start = ((self.i * self.ncomp + self.comp) * self.gdim + gd) * self.npts;
        &self.grads[start..start + self.npts]
    }
}

/// Per-cell bilinear-form kernel.
///
/// **MOOSE-style `integrand` API**: kernel authors implement only
/// `integrand` -- the physics at one quadrature point for one
/// `(test_i, trial_i)` pair, WITHOUT quadrature weights or jacobian
/// determinants.  The provided `assemble_local` default owns the test x
/// trial x quadrature triple loop, multiplies by `wts[q] * jdets[q]`, and
/// accumulates into `out` (row-major `[ndofs * ndofs * ncomp * ncomp]`).
///
/// Override `assemble_local` for max-control kernels (SIMD-fused Q loop,
/// cell-level SUPG `tau` precomputation, etc.); the default impl is
/// sufficient for almost all physics kernels.
pub trait BilinearForm {
    /// Physics integrand at one quadrature point for one (test, trial) pair.
    /// Return the bare integrand -- NO `wts[q] * jdets[q]` factor; the
    /// framework's default `assemble_local` multiplies those in.
    /// `test_i` / `trial_i` are local cell dof indices (`0..ndofs`).
    fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize, trial_i: usize) -> f64;

    /// Default fused local-matrix assembly.  Iterates test x trial x q in
    /// the same order as the original hand-fused kernels, weighting
    /// `integrand` by `wts[q] * jdets[q]`.  Output `out: &mut [f64]` of
    /// length `ndofs * ndofs` (scalar) or `ndofs * ndofs * ncomp * ncomp`
    /// (vector FE when overridden).  Override only for max-control kernels.
    fn assemble_local(&self, ctx: &LocalCtx, out: &mut [f64]) {
        let n = ctx.ndofs;
        for ti in 0..n {
            for si in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q] * ctx.jdets[q] * self.integrand(ctx, q, ti, si);
                }
                out[ti * n + si] = acc;
            }
        }
    }
}

/// Per-cell linear-form kernel (RHS contribution).  Mirrors `BilinearForm`:
/// authors implement only `integrand` (physics at one Q point for one test
/// dof, no weight/jdet factor); the default `assemble_local_rhs` owns the
/// test x quadrature loop and weights via `wts[q] * jdets[q]`.
pub trait LinearForm {
    /// RHS integrand at one quadrature point for one test dof.  Bare
    /// physics -- NO `wts[q] * jdets[q]` factor; the framework multiplies
    /// those in.  `test_i` is a local cell dof index (`0..ndofs`).
    fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize) -> f64;

    /// Default fused local-RHS assembly, weighting `integrand` by
    /// `wts[q] * jdets[q]`.  Output `out: &mut [f64]` of length `ndofs`.
    /// Override only for max-control kernels.
    fn assemble_local_rhs(&self, ctx: &LocalCtx, out: &mut [f64]) {
        for ti in 0..ctx.ndofs {
            let mut acc = 0.0;
            for q in 0..ctx.npts {
                acc += ctx.wts[q] * ctx.jdets[q] * self.integrand(ctx, q, ti);
            }
            out[ti] = acc;
        }
    }
}

// -----------------------------------------------------------------------------
// kernels
// -----------------------------------------------------------------------------

/// Mass kernel: integral of `u * v`.  Dimension-agnostic (mass has no
/// derivatives so `tdim`/`gdim` are irrelevant).
pub struct KernelMass {}

impl KernelMass {
    pub fn new() -> Self {
        Self {}
    }
}
impl BilinearForm for KernelMass {
    fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize, trial_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelMass: scalar only (ncomp==1)");
        ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}

/// 1D advection-diffusion kernel: `nu * grad_u . grad_v - vel * u * grad_v`.
///
/// Diffusion sums over `gdim` so the same kernel runs in 1D/2D/3D.  The
/// advection term uses scalar `vel` against `grad_v[0]` -- this is the
/// existing 1D form.  2D/3D advection needs a vector velocity (a physics
/// decision: flux form vs convective form), so it asserts `gdim==1` here
/// rather than silently producing a wrong extension.  Use `KernelAdvDiff2D`
/// for 2D physics.
pub struct KernelAdvDiff {
    pub nu: f64,
    pub vel: f64,
}

impl KernelAdvDiff {
    pub fn new(nu: f64, vel: f64) -> Self {
        Self { nu, vel }
    }
}
impl BilinearForm for KernelAdvDiff {
    fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize, trial_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiff: scalar only (ncomp==1)");
        assert_eq!(
            ctx.gdim, 1,
            "KernelAdvDiff: 1D only (use KernelAdvDiff2D for 2D)"
        );
        let gv = ctx.test(test_i, 0).grad(q, 0);
        let gu = ctx.trial(trial_i, 0).grad(q, 0);
        let u = ctx.trial(trial_i, 0).v(q);
        self.nu * gu * gv - self.vel * u * gv
    }
}

/// 2D advection-diffusion kernel: `nu * grad_u . grad_v - (vel_x, vel_y) . grad_v * u`.
///
/// Sums the diffusion and advection dot products over `gdim==2`.  Same MOOSE
/// `integrand` ergonomics as the 1D variant.
pub struct KernelAdvDiff2D {
    pub nu: f64,
    pub vel: [f64; 2],
}

impl KernelAdvDiff2D {
    pub fn new(nu: f64, vel: [f64; 2]) -> Self {
        Self { nu, vel }
    }
}
impl BilinearForm for KernelAdvDiff2D {
    fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize, trial_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiff2D: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 2, "KernelAdvDiff2D: 2D only (gdim==2)");
        let gv = ctx.test(test_i, 0);
        let gu = ctx.trial(trial_i, 0);
        let u = ctx.trial(trial_i, 0).v(q);
        let grad_dot: f64 = (0..2).map(|d| gu.grad(q, d) * gv.grad(q, d)).sum();
        let advect: f64 = (0..2).map(|d| self.vel[d] * u * gv.grad(q, d)).sum();
        self.nu * grad_dot - advect
    }
}

/// 1D advection-diffusion kernel with SUPG stabilization.
/// Replaces the test function v -> v + tau * vel * grad_v, which adds the
/// term `tau * (vel * grad_u) * (vel * grad_v)` to the standard Galerkin
/// form.  Per quadrature point:
///   `q = (nu + tau * vel^2) * grad_u * grad_v - vel * u * grad_v`
pub struct KernelAdvDiffSUPG {
    pub nu: f64,
    pub vel: f64,
    pub tau: f64,
}

impl KernelAdvDiffSUPG {
    pub fn new(nu: f64, vel: f64, tau: f64) -> Self {
        Self { nu, vel, tau }
    }
}
impl BilinearForm for KernelAdvDiffSUPG {
    fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize, trial_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiffSUPG: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 1, "KernelAdvDiffSUPG: 1D only");
        let supg = self.tau * self.vel * self.vel;
        let gv = ctx.test(test_i, 0).grad(q, 0);
        let gu = ctx.trial(trial_i, 0).grad(q, 0);
        let u = ctx.trial(trial_i, 0).v(q);
        (self.nu + supg) * gu * gv - self.vel * u * gv
    }
}

/// Constant volumetric source `f(x) = val` -- demo `LinearForm` kernel.
pub struct KernelVolumeSource {
    pub val: f64,
}

impl KernelVolumeSource {
    pub fn new(val: f64) -> Self {
        Self { val }
    }
}

impl LinearForm for KernelVolumeSource {
    fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelVolumeSource: scalar only (ncomp==1)");
        self.val * ctx.test(test_i, 0).v(q)
    }
}

// =============================================================================
// Boundary integrator.
//
// The unit-square quadrilateral path below runs a facet-loop assembly path
// independent of cell-kernel assembly:
//   * Neumann: facet-local RHS contribution, weighted by `wts[q] * jfacet_det[q]`
//     (surface measure).
//   * Robin:   additionally a facet-local mass-like matrix block, also
//     weighted by `wts[q] * jfacet_det[q]`, added into the global `K`.
//
// The trait adopts the same MOOSE-style `integrand` ergonomics as
// `BilinearForm` / `LinearForm`: physics authors implement `integrand_rhs`
// (and optionally `integrand_mat` for Robin); the framework owns the facet
// test x quadrature loop and the surface-measure weighting.  Override the
// `assemble_*` provided methods for max-control kernels.
//
// It currently covers straight quadrilateral facets; curved geometry and
// other facet types need their own geometry path.
// =============================================================================

/// Reduced boundary RHS and Robin matrix assembled on one square side.
pub struct BoundaryContributions {
    pub rhs: Vec<f64>,
    pub mat: SparseColMat<usize, f64>,
}

/// Geometry used to select a natural-boundary kernel. `index` is the local
/// `ndmesh` interval-facet index; Gmsh importers can use it to look up a
/// physical-curve tag.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryFacet {
    pub index: usize,
    pub midpoint: [f64; 2],
}

/// Assemble selected natural-boundary kernels on straight quadrilateral facets.
/// This is called by `FiniteElement2DProblem::assemble_boundary`; callers only
/// provide the selector closure through that method.
pub(crate) fn assemble_quad_boundaries<'a, M, D, F>(
    mesh: &M,
    family: &LagrangeElementFamily<f64>,
    p: usize,
    reduced_size: usize,
    target_dof: D,
    mut select_kernel: F,
) -> BoundaryContributions
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
    D: Fn(usize) -> Option<usize>,
    F: FnMut(BoundaryFacet) -> Option<&'a dyn BoundaryIntegrator>,
{
    assert!(p >= 1, "boundary quadrature requires p >= 1");
    assert_eq!(
        mesh.topology_dim(),
        2,
        "unit-square boundary assembly is 2D only"
    );
    assert_eq!(
        mesh.geometry_dim(),
        2,
        "unit-square boundary assembly is 2D-in-2D only"
    );
    let space = FunctionSpaceImpl::new(mesh, family);
    let (qpts, wts) = single_integral_quadrature(
        QuadratureRule::GaussLobattoLegendre,
        Domain::Interval,
        p - 1,
    )
    .unwrap();
    let npts = wts.len();
    debug_assert_eq!(npts, p + 1);
    let xs: Vec<f64> = (0..npts).map(|q| qpts[2 * q + 1]).collect();
    let mut rhs = vec![0.0; reduced_size];
    let mut triplets = Vec::new();
    let mut coord = [0.0; 2];

    for facet in mesh.entity_iter(ReferenceCellType::Interval) {
        let facet_index = facet.local_index();
        let topology = facet.topology();
        let mut cells = topology.connected_entity_iter(ReferenceCellType::Quadrilateral);
        let Some(cell_index) = cells.next() else {
            continue;
        };
        if cells.next().is_some() {
            continue;
        }
        let mut midpoint = [0.0; 2];
        let mut endpoints = [[0.0; 2]; 2];
        let mut count = 0;
        for point in facet.geometry().points() {
            point.coords(&mut coord);
            if count < 2 {
                endpoints[count] = coord;
            }
            midpoint[0] += coord[0];
            midpoint[1] += coord[1];
            count += 1;
        }
        assert_eq!(count, 2, "quadrilateral boundary facets must be intervals");
        midpoint[0] /= 2.0;
        midpoint[1] /= 2.0;
        let Some(kernel) = select_kernel(BoundaryFacet {
            index: facet_index,
            midpoint,
        }) else {
            continue;
        };

        let cell = mesh
            .entity(ReferenceCellType::Quadrilateral, cell_index)
            .expect("boundary facet owner must be a quadrilateral");
        let local_facet = cell
            .topology()
            .sub_entity_iter(ReferenceCellType::Interval)
            .position(|index| index == facet_index)
            .expect("boundary facet missing from owning quadrilateral");

        let mut pts = rlst_dynamic_array!(f64, [2, npts]);
        for q in 0..npts {
            let (xi, eta) = match local_facet {
                0 => (xs[q], 0.0),
                1 => (0.0, xs[q]),
                2 => (1.0, xs[q]),
                3 => (xs[q], 1.0),
                _ => panic!("quadrilateral has an invalid local interval index {local_facet}"),
            };
            *pts.get_mut([0, q]).unwrap() = xi;
            *pts.get_mut([1, q]).unwrap() = eta;
        }
        let element = family.element(ReferenceCellType::Quadrilateral);
        let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(0, npts));
        element.tabulate(&pts, 0, &mut table);
        let cell_dofs = space
            .entity_closure_dofs(ReferenceCellType::Quadrilateral, cell_index)
            .unwrap();
        let facet_dofs = space
            .entity_closure_dofs(ReferenceCellType::Interval, facet_index)
            .unwrap();
        let nf = facet_dofs.len();
        let mut values = vec![0.0; nf * npts];
        for (i, &facet_dof) in facet_dofs.iter().enumerate() {
            let cell_i = cell_dofs
                .iter()
                .position(|&d| d == facet_dof)
                .expect("facet dof missing from owning quadrilateral");
            for q in 0..npts {
                values[i * npts + q] = *table.get([0, q, cell_i, 0]).unwrap();
            }
        }
        let length = ((endpoints[1][0] - endpoints[0][0]).powi(2)
            + (endpoints[1][1] - endpoints[0][1]).powi(2))
        .sqrt();
        assert!(length > 0.0, "boundary facet has zero length");
        let mut cell_center = [0.0; 2];
        let mut cell_point_count = 0;
        for point in cell.geometry().points() {
            point.coords(&mut coord);
            cell_center[0] += coord[0];
            cell_center[1] += coord[1];
            cell_point_count += 1;
        }
        assert!(
            cell_point_count >= 4,
            "quadrilateral owner has too few geometry points"
        );
        cell_center[0] /= cell_point_count as f64;
        cell_center[1] /= cell_point_count as f64;
        let mut normal = [
            -(endpoints[1][1] - endpoints[0][1]) / length,
            (endpoints[1][0] - endpoints[0][0]) / length,
        ];
        if normal[0] * (midpoint[0] - cell_center[0]) + normal[1] * (midpoint[1] - cell_center[1])
            < 0.0
        {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
        }
        let jfacet_det = vec![length; npts];
        let gmap = mesh.geometry_map(ReferenceCellType::Quadrilateral, 1, &pts);
        let mut physical_points = rlst_dynamic_array!(f64, [2, npts]);
        gmap.physical_points(cell_index, &mut physical_points);
        let mut points = vec![0.0; npts * 2];
        for q in 0..npts {
            points[2 * q] = *physical_points.get([0, q]).unwrap();
            points[2 * q + 1] = *physical_points.get([1, q]).unwrap();
        }
        let ctx = FacetCtx {
            tdim: 1,
            gdim: 2,
            ncomp: 1,
            npts,
            ndofs: nf,
            wts: &wts,
            jfacet_det: &jfacet_det,
            points: &points,
            normal: &normal,
            values: &values,
            grads: &[],
        };
        let mut local_rhs = vec![0.0; nf];
        let mut local_mat = vec![0.0; nf * nf];
        kernel.assemble_facet_rhs(&ctx, &mut local_rhs);
        kernel.assemble_facet_mat(&ctx, &mut local_mat);
        for (local_i, &full_i) in facet_dofs.iter().enumerate() {
            let Some(reduced_i) = target_dof(full_i) else {
                continue;
            };
            rhs[reduced_i] += local_rhs[local_i];
            for (local_j, &full_j) in facet_dofs.iter().enumerate() {
                if let Some(reduced_j) = target_dof(full_j) {
                    let value = local_mat[local_i * nf + local_j];
                    if value != 0.0 {
                        triplets.push(Triplet::new(reduced_i, reduced_j, value));
                    }
                }
            }
        }
    }
    BoundaryContributions {
        rhs,
        mat: SparseColMat::try_new_from_triplets(reduced_size, reduced_size, &triplets).unwrap(),
    }
}

/// Per-facet boundary-assembly context (mirrors `LocalCtx` for facets).
#[allow(dead_code)]
pub struct FacetCtx<'a> {
    pub tdim: usize,
    pub gdim: usize,
    pub ncomp: usize,
    pub npts: usize,
    pub ndofs: usize,
    pub wts: &'a [f64],
    /// Facet jacobian determinant (surface measure) at each Q point,
    /// length `npts`.  Multiply with `wts[q]` for the physical facet
    /// integral weight.
    pub jfacet_det: &'a [f64],
    /// Physical quadrature coordinates, packed `[npts, gdim]`.
    pub points: &'a [f64],
    /// Outward unit normal of this facet in `R^gdim`, length `gdim`
    /// (constant across Q points for a flat facet; a future curved-facet
    /// extension would carry it per Q point).
    pub normal: &'a [f64],
    pub values: &'a [f64], // [ndofs, ncomp, npts], npts-contiguous
    pub grads: &'a [f64],  // [ndofs, ncomp, gdim, npts]
}

impl<'a> FacetCtx<'a> {
    pub fn point(&self, q: usize) -> &'a [f64] {
        &self.points[q * self.gdim..(q + 1) * self.gdim]
    }

    /// View of the shape function for test dof `i`, component `comp`.
    /// Mirrors `LocalCtx::test` so facet kernels read shape values via the
    /// same `.v(q)` / `.v_slice()` API as cell kernels.
    pub fn test(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        ShapeFn {
            npts: self.npts,
            ncomp: self.ncomp,
            gdim: self.gdim,
            values: self.values,
            grads: self.grads,
            i,
            comp,
        }
    }
    pub fn trial(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        self.test(i, comp)
    }
}

/// Boundary integrator trait.  MOOSE-style `integrand` ergonomics: physics
/// authors implement `integrand_rhs` (required) and `integrand_mat`
/// (optional, for Robin) -- the framework's provided `assemble_*` methods
/// own the facet test x quadrature loop and surface-measure weighting via
/// `wts[q] * jfacet_det[q]`.  Override `assemble_*` for max-control kernels.
#[allow(dead_code)]
pub trait BoundaryIntegrator {
    /// Neumann flux integrand at one facet quadrature point for one test
    /// dof.  Bare physics -- NO `wts[q] * jfacet_det[q]` factor; the
    /// framework's default `assemble_facet_rhs` multiplies those in.
    fn integrand_rhs(&self, ctx: &FacetCtx, q: usize, test_i: usize) -> f64;

    /// Robin matrix integrand at one facet quadrature point for one
    /// `(test_i, trial_i)` pair.  Bare physics -- NO `wts[q] *
    /// jfacet_det[q]` factor.  Default returns 0.0 (pure Neumann); override
    /// for Robin.
    fn integrand_mat(&self, _ctx: &FacetCtx, _q: usize, _test_i: usize, _trial_i: usize) -> f64 {
        0.0
    }

    /// Default fused facet-RHS assembly, weighting `integrand_rhs` by
    /// `wts[q] * jfacet_det[q]`.  Output `rhs: &mut [f64]` of length `ndofs`.
    fn assemble_facet_rhs(&self, ctx: &FacetCtx, rhs: &mut [f64]) {
        for ti in 0..ctx.ndofs {
            let mut acc = 0.0;
            for q in 0..ctx.npts {
                acc += ctx.wts[q] * ctx.jfacet_det[q] * self.integrand_rhs(ctx, q, ti);
            }
            rhs[ti] = acc;
        }
    }

    /// Default fused facet-matrix assembly (Robin), weighting
    /// `integrand_mat` by `wts[q] * jfacet_det[q]`.  Output `mat: &mut
    /// [f64]` of length `ndofs * ndofs`.  No-op when `integrand_mat` is
    /// left at its default 0.0 (pure Neumann).
    fn assemble_facet_mat(&self, ctx: &FacetCtx, mat: &mut [f64]) {
        let n = ctx.ndofs;
        for ti in 0..n {
            for si in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q] * ctx.jfacet_det[q] * self.integrand_mat(ctx, q, ti, si);
                }
                mat[ti * n + si] = acc;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Boundary integrator kernels
// -----------------------------------------------------------------------------

/// Constant prescribed Neumann flux `g` into the domain.  Contributes only
/// to the RHS (`integrand_mat` left at default 0.0).  Sign convention:
/// positive `g` represents flux INTO the domain -- the example's system
/// assembly adds the resulting `b` to the RHS as `+M^{-1} b`.
///
/// Spatially varying kernels can query `ctx.point(q)` for the physical facet
/// quadrature coordinate; this constant form covers the current example.
pub struct NeumannFlux {
    pub g: f64,
}

impl NeumannFlux {
    pub fn new(g: f64) -> Self {
        Self { g }
    }
}

impl BoundaryIntegrator for NeumannFlux {
    fn integrand_rhs(&self, ctx: &FacetCtx, q: usize, test_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "NeumannFlux: scalar only (ncomp==1)");
        // g * v(q) -- constant flux tested against the test function.
        self.g * ctx.test(test_i, 0).v(q)
    }
}

/// Robin (convective) boundary: `h (T_amb - T)`.  Splits into a RHS
/// contribution `h * T_amb * v` (driving term) and a matrix contribution
/// `h * u * v` added into `K` (the part that makes the pure-Neumann problem
/// non-singular).  Sign convention matches `NeumannFlux`: positive `h` is
/// heat transfer OUT of the domain when `T > T_amb` (cooling).
pub struct RobinConvection {
    pub h: f64,
    pub t_amb: f64,
}

impl RobinConvection {
    pub fn new(h: f64, t_amb: f64) -> Self {
        Self { h, t_amb }
    }
}

impl BoundaryIntegrator for RobinConvection {
    fn integrand_rhs(&self, ctx: &FacetCtx, q: usize, test_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "RobinConvection: scalar only (ncomp==1)");
        // h * T_amb * v(q) -- the ambient driving term.
        self.h * self.t_amb * ctx.test(test_i, 0).v(q)
    }
    fn integrand_mat(&self, ctx: &FacetCtx, q: usize, test_i: usize, trial_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "RobinConvection: scalar only (ncomp==1)");
        // h * u(q) * v(q) -- the Robin mass-like addition to K.
        self.h * ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}

// =============================================================================
// Lumped-mass ODE system: du/dt = -M^{-1} K u  (M diagonal).
// =============================================================================

/// Linear operator applying the system Jacobian `J = -M^{-1} K` (the
/// derivative of `frhs = -M^{-1} K x`), used as the `fjac` return.  `M` is
/// diagonal (mass-lumped), so `M^{-1}` action is an element-wise scale by
/// `m_inv[i] = 1 / M[i,i]`.
#[derive(Debug)]
pub struct MinvKLinOp<'a> {
    pub k: SparseColMatRef<'a, usize, f64>,
    pub m_inv: &'a [f64],
}

impl<'a> LinOp<f64> for MinvKLinOp<'a> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::empty()
    }
    fn nrows(&self) -> usize {
        self.k.nrows()
    }
    fn ncols(&self) -> usize {
        self.k.ncols()
    }
    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        // J = -M^{-1} K  ->  J v = -(m_inv .* (K v))
        let kv = self.k * rhs;
        let n = self.m_inv.len();
        for j in 0..out.ncols() {
            for i in 0..n {
                out[(i, j)] = -self.m_inv[i] * kv[(i, j)];
            }
        }
    }
    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // real f64: conjugate == self
        self.apply(out, rhs, par, stack);
    }
}

/// Linear semi-discrete advection-diffusion system `du/dt = -M^{-1} K u`.
///
/// `M` is stored only via its diagonal reciprocals `m_inv` (mass-lumped);
/// `K` is stored as a sparse matrix.  `fjac` returns a `MinvKLinOp` that
/// applies `M^{-1} K`; `fmass` is left at the default `None` since the
/// `dirk_step` residual is not mass-consistent.
pub struct AdvDiffSys {
    pub k: SparseColMat<usize, f64>,
    pub m_inv: Vec<f64>,
}

impl AdvDiffSys {
    pub fn new(m: SparseColMat<usize, f64>, k: SparseColMat<usize, f64>) -> Self {
        let n = k.nrows();
        assert!(
            m.nrows() == n && m.ncols() == n,
            "M and K shape mismatch: M={}x{}, K={}x{}",
            m.nrows(),
            m.ncols(),
            n,
            n,
        );
        assert!(
            m.compute_nnz() == n,
            "expected lumped-diagonal M (nnz==n={}), got nnz={}; \
             check quadrature order (must be 2p-1 for mass lumping)",
            n,
            m.compute_nnz(),
        );
        let mut m_inv = vec![0.0_f64; n];
        for i in 0..n {
            let m_ii = m[(i, i)];
            assert!(m_ii.abs() > 1e-30, "zero diagonal entry M[{}, {}]", i, i);
            m_inv[i] = 1.0 / m_ii;
        }
        Self { k, m_inv }
    }

    /// Apply `M^{-1} K` to `x` -> returns `Mat<n, ncols>`.
    pub fn apply_minv_k(&self, x: MatRef<f64>) -> Mat<f64> {
        let kv = self.k.as_ref() * x;
        let n = self.m_inv.len();
        let mut out = kv.clone();
        for j in 0..out.ncols() {
            for i in 0..n {
                out[(i, j)] = self.m_inv[i] * kv[(i, j)];
            }
        }
        out
    }
}

impl<'a> OdeSys<'a> for AdvDiffSys {
    fn frhs(&self, _t: f64, x: MatRef<f64>) -> Mat<f64> {
        faer::Scale(-1.0) * self.apply_minv_k(x)
    }

    fn fjac<'b>(&'a self, _t: f64, _x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        Box::new(MinvKLinOp {
            k: self.k.as_ref(),
            m_inv: &self.m_inv,
        })
    }
}
