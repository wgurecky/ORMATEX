/// 1D reaction-advection-diffusion problem using the nd
/// crate to build a finite element discritization of space
/// and to construct the advection and diffusion operators
/// used in the system dynamics definition.
///
/// Boundary conditions (selectable, default `Periodic`):
///   * `Periodic`  -- dofs 0 and `nx*p` identified (cyclic), size `nx*p`.
///   * `Sides`     -- per-side Dirichlet (eliminated) or Outflow (free); each of
///                    the left (x=0) and right (x=1) boundaries can be chosen
///                    independently.  Size = `nx*p + 1` minus (# Dirichlet sides).
///
/// The example builds the periodic mass `M` and advection-diffusion `K`
/// operators, precomputes `A = M^{-1} K`, and integrates the linear
/// semi-discrete system `du/dt = -A u` with the implicit-Euler DIRK
/// integrator from `ormatex::ode_implicit`.  The initial condition is a
/// periodic Gaussian bump centred at `x = 0.5`.
///
use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
use faer::matrix_free::LinOp;
use faer::dyn_stack::{MemStack, StackReq};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;

// nd
use ndelement::{
    ciarlet::{CiarletElement, LagrangeElementFamily, LagrangeVariant},
    map::IdentityMap,
    traits::{FiniteElement, MappedFiniteElement},
    types::{Continuity, ReferenceCellType},
};
use ndfunctionspace::{FunctionSpaceImpl, traits::FunctionSpace};
use ndmesh::{
    shapes::unit_interval,
    traits::{Entity, GeometryMap, Mesh},
    SingleElementMesh,
};
use quadraturerules::{Domain, QuadratureRule, single_integral_quadrature};
use rlst::{DynArray, rlst_dynamic_array};

/// Per-side boundary condition.  `Dirichlet` eliminates the end DOF
/// (homogeneous zero); `Outflow` keeps it (free / natural).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideBC {
    Dirichlet,
    Outflow,
}

/// Boundary condition selector.  Default is `Periodic`.
///
/// `Sides` lets the user pick `Dirichlet` or `Outflow` independently at the
/// left (x=0) and right (x=1) boundaries.
#[derive(Clone, Copy, Debug)]
pub enum BoundaryCondition {
    /// identify dofs 0 and nx (cyclic), system size nx
    Periodic,
    /// per-side Dirichlet/Outflow selection
    Sides {
        left: SideBC,
        right: SideBC,
    },
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        BoundaryCondition::Periodic
    }
}

impl BoundaryCondition {
    /// both sides Dirichlet
    pub fn dirichlet() -> Self {
        BoundaryCondition::Sides {
            left: SideBC::Dirichlet,
            right: SideBC::Dirichlet,
        }
    }
    /// both sides Outflow (free)
    pub fn outflow() -> Self {
        BoundaryCondition::Sides {
            left: SideBC::Outflow,
            right: SideBC::Outflow,
        }
    }
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
// 1..=tdim = d/dx_d.  Physical gradients are precomputed in `assemble`:
//
//     grad_phys[dof, comp, gd, q] = sum_{td=0..tdim} jinv[td, gd, q] * grad_ref[1+td, q, dof, comp]
//
// (J^{-T} . grad_ref).  For 1D (tdim=gdim=1) this collapses to
// `jinv[0,0,q] * table[1,q,dof,0]` -- the existing hand-rolled transform,
// lifted here from per-cell scratch to a per-call reused relayout buffer
// whose layout makes `npts` contiguous so `v_slice()/grad_slice(d)` return
// real `&[f64]` with `u[q]`-style indexing.
//
// `push_forward` would have been the idiomatic path, but nd's `IdentityMap`
// (used by `LagrangeElementFamily`) `unimplemented!()`'s for `nderivs > 0`,
// so the manual transform is kept; the indexing is now tdim/gdim-generic so
// 2D/3D does not require revisiting it.
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
    /// loop with this.  The advection-diffusion kernel currently asserts
    /// `gdim == 1`; a 2D/3D variant would dispatch on `gdim` and read a
    /// vector `vel` (length `gdim`) into the dot product.
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
    /// Determined by the per-cell-type quadrature rule chosen in `with_bc`
    /// -- for mass-lumped Lagrange it is `p + 1` GLL points; for a generic
    /// 2D/3D rule it is rule-specific (e.g. Xiao-Gimbutas on triangles).
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
    /// for the unit interval, 1/2 for the unit triangle).  The kernel
    /// combines `wts[q] * jdets[q]` to get the physical integral weight
    /// at point `q` --DO NOT apply a second jacobian scaling.
    pub wts: &'a [f64],

    /// Jacobian determinant of the cell's reference->physical map at each
    /// quadrature point, length `npts`.  For an Interval on [0,1] with
    /// `nx` cells this is the cell length `1/nx` at every Q point; for a
    /// deformed 2D/3D cell it varies per Q point.  Always positive for a
    /// valid (orientation-preserving) mesh.  Multiply with `wts[q]` to
    /// form the physical integral weight.
    pub jdets: &'a [f64],

    /// Physical values of all basis functions on this cell, row-major
    /// `[ndofs, ncomp, npts]` with `npts` contiguous.  Already pushed
    /// forward (so for Lagrange on `IdentityMap` this is the reference
    /// tabulation; for Piola-mapped elements it is the transformed value).
    /// Access via `ctx.test(i, comp).v(q)` / `.v_slice()`; do not index
    /// directly.
    values: &'a [f64],

    /// Physical gradients of all basis functions on this cell, row-major
    /// `[ndofs, ncomp, gdim, npts]` with `npts` contiguous.  Computed as
    /// `J^{-T} · ∂φ/∂ξ` per Q point (for each `tdim` slot summed into
    /// `gdim` physical components).  Access via `ctx.test(i, comp).grad(q, d)`
    /// / `.grad_slice(d)`; do not index directly.
    grads: &'a [f64],
}

impl<'a> LocalCtx<'a> {
    /// View of the shape function for test dof `i`, component `comp`.
    /// For Galerkin test == trial; `trial(i, comp)` returns an identical view.
    /// `test`/`trial` are separate so a future non-Galerkin `LocalCtx` (carrying
    /// two tabulations) can keep this signature.
    pub fn test(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        ShapeFn { ctx: self, i, comp }
    }
    pub fn trial(&'a self, i: usize, comp: usize) -> ShapeFn<'a> {
        ShapeFn { ctx: self, i, comp }
    }
}

/// Per-dof view over `LocalCtx` tabulation.  Exposes two reading styles.
pub struct ShapeFn<'a> {
    ctx: &'a LocalCtx<'a>,
    i: usize,
    comp: usize,
}

impl<'a> ShapeFn<'a> {
    /// Physical value at quadrature point `q`.
    pub fn v(&self, q: usize) -> f64 {
        self.ctx.values[(self.i * self.ctx.ncomp + self.comp) * self.ctx.npts + q]
    }
    /// Physical gradient component `gd` at quadrature point `q`.
    pub fn grad(&self, q: usize, gd: usize) -> f64 {
        self.ctx.grads[((self.i * self.ctx.ncomp + self.comp) * self.ctx.gdim + gd)
            * self.ctx.npts + q]
    }
    /// Slice of values at all quadrature points.  Length `npts`, contiguous.
    pub fn v_slice(&self) -> &'a [f64] {
        let start = (self.i * self.ctx.ncomp + self.comp) * self.ctx.npts;
        &self.ctx.values[start..start + self.ctx.npts]
    }
    /// Slice of gradient component `gd` at all quadrature points.  Length
    /// `npts`, contiguous.  Different `gd` are independent slices.
    pub fn grad_slice(&self, gd: usize) -> &'a [f64] {
        let start = ((self.i * self.ctx.ncomp + self.comp) * self.ctx.gdim + gd)
            * self.ctx.npts;
        &self.ctx.grads[start..start + self.ctx.npts]
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
///
/// `assemble` is generic over `K: BilinearForm` (no `dyn`) so each form's
/// inner loop is monomorphized and inlinable -- the Basix `tabulate_tensor`
/// win without the codegen build step.
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

// -----------------------------------------------------------------------------
// kernels
// -----------------------------------------------------------------------------

// mass kernel: integral of u * v.  Dimension-agnostic (mass has no derivatives
// so `tdim`/`gdim` are irrelevant).
struct KernelMass {}

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

// advection-diffusion kernel: nu * grad_u . grad_v - vel * u * grad_v
//
// Diffusion sums over `gdim` so the same kernel runs in 1D/2D/3D.  The
// advection term uses scalar `vel` against `grad_v[0]` -- this is the
// existing 1D form.  2D/3D advection needs a vector velocity (a physics
// decision: flux form vs convective form), so it asserts `gdim==1` here
// rather than silently producing a wrong extension.  A `KernelAdvDiffN`
// with `vel: Vec<f64>` can drop in when 2D/3D physics is needed.
struct KernelAdvDiff {
    nu: f64,
    vel: f64,
}

impl KernelAdvDiff {
    pub fn new(nu: f64, vel: f64) -> Self {
        Self { nu, vel }
    }
}
impl BilinearForm for KernelAdvDiff {
    fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize, trial_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiff: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 1, "KernelAdvDiff: 1D only (use a 2D/3D variant for gdim>1)");
        let gv = ctx.test(test_i, 0).grad(q, 0);
        let gu = ctx.trial(trial_i, 0).grad(q, 0);
        let u = ctx.trial(trial_i, 0).v(q);
        self.nu * gu * gv - self.vel * u * gv
    }
}

// advection-diffusion kernel with SUPG stabilization.
// Replaces the test function v -> v + tau * vel * grad_v, which adds the
// term tau * (vel * grad_u) * (vel * grad_v) to the standard Galerkin form.
// Per quadrature point:
//   q = (nu + tau * vel^2) * grad_u * grad_v - vel * u * grad_v
//
// Same 1D-only assertion as `KernelAdvDiff`.
struct KernelAdvDiffSUPG {
    nu: f64,
    vel: f64,
    tau: f64,
}

impl KernelAdvDiffSUPG {
    pub fn new(nu: f64, vel: f64, tau: f64) -> Self {
        Self { nu, vel, tau }
    }
}
impl BilinearForm for KernelAdvDiffSUPG {
    fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize, trial_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelAdvDiffSUPG: scalar only (ncomp==1)");
        assert_eq!(ctx.gdim, 1, "KernelAdvDiffSUPG: 1D only (use a 2D/3D variant for gdim>1)");
        let supg = self.tau * self.vel * self.vel;
        let gv = ctx.test(test_i, 0).grad(q, 0);
        let gu = ctx.trial(trial_i, 0).grad(q, 0);
        let u = ctx.trial(trial_i, 0).v(q);
        (self.nu + supg) * gu * gv - self.vel * u * gv
    }
}

// -----------------------------------------------------------------------------
// LinearForm (RHS / source terms) -- sibling of `BilinearForm`, same MOOSE-style
// `integrand` ergonomics.  A linear form contributes to the global RHS:
//   rhs[test_dof] = sum_cells sum_q (wts[q] * jdets[q] * f(q, test_i))
// e.g. volumetric source terms, body forces.  Boundary flux (Neumann) lives
// on the `BoundaryIntegrator` facet trait below, not here.
// -----------------------------------------------------------------------------

/// Per-cell linear-form kernel (RHS contribution).  Mirrors `BilinearForm`:
/// authors implement only `integrand` (physics at one Q point for one test
/// dof, no weight/jdet factor); the default `assemble_local_rhs` owns the
/// test x quadrature loop and weights via `wts[q] * jdets[q]`.
///
/// Future wiring (not in Phase 1): the ODE `frhs` will combine
/// `-apply_minv_k(x)` with a per-cell `LinearForm` pass for forcing/source
/// terms.  The trait surface here is exercised by `KernelVolumeSource` so
/// the API is concrete; the example ODE still runs with zero forcing to
/// preserve bit-identical simulation output.
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

/// Constant volumetric source `f(x) = val`.  Concrete demo exercising the
/// `LinearForm` trait surface (no callers in the example ODE -- marked
/// `#[allow(dead_code)]` like `BoundaryIntegrator`).
#[allow(dead_code)]
struct KernelVolumeSource {
    val: f64,
}

impl KernelVolumeSource {
    pub fn new(val: f64) -> Self {
        Self { val }
    }
}

impl LinearForm for KernelVolumeSource {
    fn integrand(&self, ctx: &LocalCtx, q: usize, test_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelVolumeSource: scalar only (ncomp==1)");
        // f * v(q): constant source tested against the test function
        self.val * ctx.test(test_i, 0).v(q)
    }
}

/// Linear operator applying the system Jacobian `J = -M^{-1} K` (the
/// derivative of `frhs = -M^{-1} K x`), used as the `fjac` return.  `M` is
/// diagonal (mass-lumped), so `M^{-1}` action is an element-wise scale by
/// `m_inv[i] = 1 / M[i,i]`.
#[derive(Debug)]
struct MinvKLinOp<'a> {
    k: SparseColMatRef<'a, usize, f64>,
    m_inv: &'a [f64],
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
    fn apply(&self, mut out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, _par: Par, _stack: &mut MemStack) {
        // J = -M^{-1} K  ->  J v = -(m_inv .* (K v))
        let kv = self.k * rhs;
        let n = self.m_inv.len();
        for j in 0..out.ncols() {
            for i in 0..n {
                out[(i, j)] = -self.m_inv[i] * kv[(i, j)];
            }
        }
    }
    fn conj_apply(&self, out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, stack: &mut MemStack) {
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
struct AdvDiffSys {
    k: SparseColMat<usize, f64>,
    m_inv: Vec<f64>,
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

    /// Apply `M^{-1} K` to `x` → returns `Mat<n, ncols>`.
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

/// defines 1D finite element domain using the nd crate (https://codeberg.org/nd-project/nd)
///
/// Only owned data is stored here.  The function space and geometry map
/// both borrow the mesh, so they are re-created inside each build call
/// (mirrors the nd `test_mass_matrix.rs` example's local-scope pattern).
struct FiniteElement1DProblem {
    nx: usize,
    /// polynomial degree of the Lagrange element (P_p)
    p: usize,
    mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>,
    family: LagrangeElementFamily<f64>,
    wts: Vec<f64>,
    // reference-cell tabulation with nderivs = 1 (values + d/dx)
    table: DynArray<f64, 4>,
    bc: BoundaryCondition,
    // global dof index of the right boundary vertex (x=1).
    // Queried from the function space so BC reduction is robust to dof
    // ordering (nd numbers vertex dofs before edge-interior dofs).
    // `left_end_dof` was previously a field but is now consumed only in
    // `with_bc` (folded into `dof_lut`); the right endpoint is still needed
    // by `dof_positions` to set its x-coordinate to 0.0 under Periodic.
    right_end_dof: usize,
    // per-cell, per-quadrature-point geometry-map data, computed once in
    // with_bc().  Column c corresponds to the cell with local_index() == c.
    // Caching avoids recomputing the jacobians on every assemble() call.
    jinv_cache: DynArray<f64, 4>, // [tdim, gdim, npts, ncells]
    jdets_cache: Vec<f64>,        // [ncells * npts], row-major (c, q)
    // Precomputed full -> Option<reduced> DOF map (BC reduction LUT).
    // Built once in `with_bc`; `target_dof` becomes O(1), `apply_bc` O(nnz).
    dof_lut: Vec<Option<usize>>,
}

impl FiniteElement1DProblem {
    #[allow(dead_code)]
    pub fn new(nx: usize, p: usize) -> Self {
        Self::with_bc(nx, p, BoundaryCondition::default())
    }

    pub fn with_bc(nx: usize, p: usize, bc: BoundaryCondition) -> Self {
        assert!(p >= 1, "polynomial degree p must be >= 1");
        // from: https://codeberg.org/nd-project/nd/src/branch/main/ndfunctionspace/examples/test_mass_matrix.rs example
        // but with the unit interval rather than a triangle
        let mesh = unit_interval(nx);

        // finite element family -- degree p selectable at runtime
        let family =
            LagrangeElementFamily::<f64>::new(p, Continuity::Standard, LagrangeVariant::GLL);

        // need a function space to inspect the element superdegree and endpoint dofs
        let tmp_space = FunctionSpaceImpl::new(&mesh, &family);
        let element = &tmp_space.elements()[0];

        // endpoint dof indices: vertex 0 (x=0) and vertex nx (x=1).
        let left_end_dof = *tmp_space
            .entity_dofs(ReferenceCellType::Point, 0)
            .unwrap()
            .first()
            .unwrap();
        let right_end_dof = *tmp_space
            .entity_dofs(ReferenceCellType::Point, nx)
            .unwrap()
            .first()
            .unwrap();

        let (qpts, w) = single_integral_quadrature(
            QuadratureRule::GaussLobattoLegendre,
            Domain::Interval,
            // order = p-1 returns p+1 GLL points = interpolation nodes, which
            // lumps the mass matrix to diagonal (classic SEM tradeoff).
            // NB: the quadraturerules crate's `order` = n_points - 2.
            element.lagrange_superdegree().saturating_sub(1),
        )
        .unwrap();
        let npts = w.len();
        // interval quadrature points are stored as barycentric pairs (α, β);
        // the physical coordinate on [0, 1] is the β component (index 2*i+1).
        // Using the interpolation nodes (= GLL points) as quadrature points
        // lumps the mass matrix to diagonal.
        let mut pts = rlst_dynamic_array!(f64, [1, npts]);
        for i in 0..npts {
            *pts.get_mut([0, i]).unwrap() = qpts[2 * i + 1];
        }
        // weights already sum to 1 (reference length 1) -- no /2 scaling.
        let wts = w;

        let mut table = DynArray::<f64, 4>::from_shape(element.tabulate_array_shape(1, npts));
        element.tabulate(&pts, 1, &mut table);

        // --- compute per-cell geometry-map data once (jacobians, inverses, dets)
        // the mesh never changes, so these are cached for reuse in every assemble().
        let ncells = mesh.entity_count(ReferenceCellType::Interval);
        let gdim = mesh.geometry_dim();
        let tdim = mesh.topology_dim();
        let gmap = mesh.geometry_map(ReferenceCellType::Interval, 1, &pts);

        let mut jac_scratch = rlst_dynamic_array!(f64, [gdim, tdim, npts]);
        let mut jinv_scratch = rlst_dynamic_array!(f64, [tdim, gdim, npts]);
        let mut jdet_scratch = vec![0.0_f64; npts];

        let mut jinv_cache = rlst_dynamic_array!(f64, [tdim, gdim, npts, ncells]);
        let mut jdets_cache = vec![0.0_f64; ncells * npts];

        for cell in mesh.entity_iter(ReferenceCellType::Interval) {
            let c = cell.local_index();
            debug_assert!(c < ncells, "cell local_index out of cache range");
            gmap.jacobians_inverses_dets(
                c,
                &mut jac_scratch,
                &mut jinv_scratch,
                &mut jdet_scratch,
            );
            for q in 0..npts {
                *jinv_cache.get_mut([0, 0, q, c]).unwrap() =
                    *jinv_scratch.get([0, 0, q]).unwrap();
                jdets_cache[c * npts + q] = jdet_scratch[q];
            }
        }

        // --- precompute BC reduction LUT (full dof -> Option<reduced>).
        // Building it once here lets `target_dof` be O(1) and `apply_bc` O(nnz)
        // instead of the previous O(full^2) scan.
        let n_full = nx * p + 1;
        let dof_lut: Vec<Option<usize>> = (0..n_full)
            .map(|full| Self::slow_target_dof(bc, left_end_dof, right_end_dof, full))
            .collect();

        Self {
            nx,
            p,
            mesh,
            family,
            wts,
            table,
            bc,
            right_end_dof,
            jinv_cache,
            jdets_cache,
            dof_lut,
        }
    }

    /// Number of full (unreduced) DOFs on the mesh.
    /// For continuous P_p on nx intervals: nx*p + 1 nodes.
    fn full_size(&self) -> usize {
        self.nx * self.p + 1
    }

    /// true if the left and right boundary DOFs are identified (Periodic only).
    fn is_periodic(&self) -> bool {
        matches!(self.bc, BoundaryCondition::Periodic)
    }

    /// O(full) computation of `full -> Option<reduced>` used once in `with_bc`
    /// to populate `dof_lut`.  Mirrors the previous `target_dof` body: count
    /// eliminated DOFs (and the periodic right endpoint, which is identified
    /// with the left) below `full` to get a contiguous reduced index.
    fn slow_target_dof(
        bc: BoundaryCondition,
        left_end_dof: usize,
        right_end_dof: usize,
        full: usize,
    ) -> Option<usize> {
        let is_elim = |f: usize| match bc {
            BoundaryCondition::Periodic => false,
            BoundaryCondition::Sides { left, right } => {
                (f == left_end_dof && left == SideBC::Dirichlet)
                    || (f == right_end_dof && right == SideBC::Dirichlet)
            }
        };
        let is_periodic = matches!(bc, BoundaryCondition::Periodic);
        if is_elim(full) {
            return None;
        }
        if is_periodic && full == right_end_dof {
            return Some(left_end_dof);
        }
        let mut offset = 0;
        for i in 0..full {
            if is_elim(i) {
                offset += 1;
            } else if is_periodic && i == right_end_dof {
                offset += 1;
            }
        }
        Some(full - offset)
    }

    /// Number of DOFs after applying the boundary condition.
    fn reduced_size(&self) -> usize {
        if self.is_periodic() {
            // cyclic identify the two endpoints
            return self.full_size() - 1;
        }
        // start with all DOFs, subtract each Dirichlet side
        let mut n = self.full_size();
        if let BoundaryCondition::Sides { left, right } = self.bc {
            if left == SideBC::Dirichlet {
                n -= 1;
            }
            if right == SideBC::Dirichlet {
                n -= 1;
            }
        }
        n
    }

    /// Map a full DOF index to a reduced DOF index, or `None` if the DOF is
    /// eliminated (Dirichlet endpoint).  For Periodic, the right endpoint dof
    /// is identified with the left endpoint dof (mapped to the same index).
    ///
    /// O(1) lookup into `dof_lut` (built once in `with_bc`).
    fn target_dof(&self, full: usize) -> Option<usize> {
        self.dof_lut[full]
    }

    /// Reduce a full assembled sparse matrix to the bc-reduced sparse size by
    /// summing identified-DOF entries and dropping eliminated DOFs.
    fn apply_bc(&self, full: SparseColMatRef<'_, usize, f64>) -> SparseColMat<usize, f64> {
        let nr = self.reduced_size();
        let (sym, vals) = full.parts();
        let col_ptr = sym.col_ptr();
        let row_idx = sym.row_idx();
        let ncols = sym.ncols();
        let mut new_triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
        for j in 0..ncols {
            for k in col_ptr[j]..col_ptr[j + 1] {
                let i = row_idx[k];
                let v = vals[k];
                if let Some(ii) = self.target_dof(i) {
                    if let Some(jj) = self.target_dof(j) {
                        new_triplets.push(Triplet::new(ii, jj, v));
                    }
                }
            }
        }
        SparseColMat::try_new_from_triplets(nr, nr, &new_triplets).unwrap()
    }

    /// Physical x-coordinate of each reduced DOF, in reduced-index order.
    /// For the uniform `unit_interval`, full DOF `k` sits at `x = k/(nx*p)`.
    /// The periodic right endpoint (full = nx*p) maps back to `x = 0`.
    pub fn dof_positions(&self) -> Vec<f64> {
        let nr = self.reduced_size();
        let mut xs = vec![f64::NAN; nr];
        let denom = (self.nx * self.p) as f64;
        for full in 0..self.full_size() {
            if let Some(r) = self.target_dof(full) {
                let x = full as f64 / denom;
                // periodic identification: right endpoint -> left endpoint's x (0.0)
                if self.is_periodic() && full == self.right_end_dof {
                    xs[r] = 0.0;
                } else {
                    xs[r] = x;
                }
            }
        }
        debug_assert!(xs.iter().all(|x| !x.is_nan()), "unfilled dof position");
        xs
    }

    /// Assemble a sparse matrix using `kernel.assemble_local` per cell.
    ///
    /// Per cell:
    ///   1. the reference-cell tabulation (`self.table`, shape
    ///      `[deriv_count, npts, ndofs, ncomp]`) is relayouted + pushed
    ///      forward into per-cell buffers with `npts` contiguous (so kernels
    ///      can borrow real `&[f64]` slices via `ShapeFn::v_slice()`);
    ///   2. the kernel writes the full `ndofs * ndofs` local matrix into a
    ///      caller-owned scratch buffer (no per-call `Vec`, no per-call heap);
    ///   3. the local matrix is scattered into full-index triplets (BC
    ///      reduction happens later in `apply_bc`).
    ///
    /// The per-cell jacobian data (jinv, jdets) is read from caches filled
    /// once in `with_bc()`, so only the tabulation relayout + kernel work
    /// runs per call.
    ///
    /// `K: BilinearForm` (no `dyn`) monomorphizes the inner triple loop per
    /// concrete form -- the Basix `tabulate_tensor` win without codegen.
    pub fn assemble<K: BilinearForm>(&self, kernel: &K) -> SparseColMat<usize, f64> {
        let space = FunctionSpaceImpl::new(&self.mesh, &self.family);
        let npts = self.wts.len();
        let n = space.process_size();
        let gdim = self.mesh.geometry_dim();
        let tdim = self.mesh.topology_dim();

        // Per-call reused scratch (sized once, reused for every cell).
        // For Phase 1 (single Interval cell type) all cells share `ndofs`,
        // so a single max is enough; for future mixed meshes this already
        // sizes to the max `element.dim()` across cell types.
        let max_ndofs = space.elements().iter().map(|e| e.dim()).max().unwrap_or(0);
        let mut values_buf = vec![0.0_f64; max_ndofs * npts];       // [ndofs, ncomp=1, npts]
        let mut grads_buf = vec![0.0_f64; max_ndofs * gdim * npts]; // [ndofs, ncomp=1, gdim, npts]
        let mut local_mat = vec![0.0_f64; max_ndofs * max_ndofs];   // [ndofs, ndofs] (ncomp=1)
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();

        for cell in self.mesh.entity_iter(ReferenceCellType::Interval) {
            let c = cell.local_index();
            let dofs = space
                .entity_closure_dofs(ReferenceCellType::Interval, c)
                .unwrap();
            let ndofs = dofs.len();
            debug_assert!(ndofs <= max_ndofs, "cell ndofs exceeds scratch buffer");

            // --- (1) relayout tabulation -> physical values + grads, npts-contiguous.
            //     values: [ndofs, npts],              from table[0,      q, dof, 0]
            //     grads : [ndofs, gdim, npts],        from J^{-T} . table[1+td, q, dof, 0]
            // ncomp = 1 (scalar Lagrange for Phase 1). For 2D/3D the same code
            // runs unchanged (tdim/gdim loop bounds come from the mesh).
            {
                let vals_slice = &mut values_buf[..ndofs * npts];
                let grads_slice = &mut grads_buf[..ndofs * gdim * npts];
                for (dof_i, _full_dof) in dofs.iter().enumerate() {
                    for q in 0..npts {
                        vals_slice[dof_i * npts + q] =
                            *self.table.get([0, q, dof_i, 0]).unwrap();
                    }
                    for q in 0..npts {
                        for gd in 0..gdim {
                            let mut acc = 0.0;
                            for td in 0..tdim {
                                let grad_ref =
                                    *self.table.get([1 + td, q, dof_i, 0]).unwrap();
                                let jinv_td_gd =
                                    *self.jinv_cache.get([td, gd, q, c]).unwrap();
                                acc += jinv_td_gd * grad_ref;
                            }
                            grads_slice[(dof_i * gdim + gd) * npts + q] = acc;
                        }
                    }
                }
            }

            // --- (2) dispatch per-cell kernel into the local matrix.
            let ctx = LocalCtx {
                tdim,
                gdim,
                ncomp: 1,
                npts,
                ndofs,
                wts: &self.wts[..],
                jdets: &self.jdets_cache[c * npts..(c + 1) * npts],
                values: &values_buf[..ndofs * npts],
                grads: &grads_buf[..ndofs * gdim * npts],
            };
            {
                let mat_slice = &mut local_mat[..ndofs * ndofs];
                mat_slice.fill(0.0);
                kernel.assemble_local(&ctx, mat_slice);
            }

            // --- (3) scatter local matrix into full-index triplets (apply_bc
            //     reduces row/col indices later).  The cutoff below matches
            //     the original `assemble`: Lagrange basis at GLL nodes is
            //     exactly 0/1 in exact arithmetic but FP noise ~1e-17 would
            //     otherwise be stored as explicit nnz.
            for (ti, &test_dof) in dofs.iter().enumerate() {
                for (si, &trial_dof) in dofs.iter().enumerate() {
                    let entry = local_mat[ti * ndofs + si];
                    if entry.abs() > 1e-12 {
                        triplets.push(Triplet::new(test_dof, trial_dof, entry));
                    }
                }
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    pub fn build_mass(&self) -> SparseColMat<usize, f64> {
        self.apply_bc(self.assemble(&KernelMass::new()).as_ref())
    }

    pub fn build_adv_diff(&self, nu: f64, vel: f64) -> SparseColMat<usize, f64> {
        self.apply_bc(self.assemble(&KernelAdvDiff::new(nu, vel)).as_ref())
    }
}

pub fn main() {
    // --- problem parameters --------------------------------------------------
    let nx = 64;
    let p = 2;
    let nu = 0.001;   // diffusion coefficient
    let vel = 0.5;    // advection velocity; T=2.0 is one full revolution
    let sigma = 0.05; // Gaussian bump width
    let x0 = 0.5;    // bump centre
    let dt = 0.01;
    let nsteps = 200;
    let snapshot_every = 20;

    // --- build the periodic FE problem --------------------------------------
    let problem = FiniteElement1DProblem::with_bc(nx, p, BoundaryCondition::Periodic);

    // sparse reduced mass M (diagonal, mass-lumped) and advection-diffusion K
    let m_sparse = problem.build_mass();
    let k_sparse = problem.build_adv_diff(nu, vel);
    let n = m_sparse.nrows();
    println!(
        "reduced ndofs = {}, mass nnz = {}, adv_diff nnz = {}",
        n,
        m_sparse.compute_nnz(),
        k_sparse.compute_nnz()
    );

    // parity check: SUPG with tau=0 must reproduce the plain adv-diff kernel
    // (one-time off-hot-path to_dense; K is unaffected by mass lumping)
    let k_plain_dense = k_sparse.to_dense();
    let k_supg0 = problem
        .apply_bc(problem.assemble(&KernelAdvDiffSUPG::new(nu, vel, 0.0)).as_ref())
        .to_dense();
    let mut max_diff = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            max_diff = max_diff.max((k_plain_dense[(i, j)] - k_supg0[(i, j)]).abs());
        }
    }
    println!("SUPG(tau=0) vs plain adv-diff max diff: {:.3e}", max_diff);
    assert!(
        max_diff < 1e-12,
        "SUPG(tau=0) should equal plain adv-diff kernel"
    );

    // build the ODE system: du/dt = -M^{-1} K u (M is diagonal → O(n) M^{-1} action)
    let sys = AdvDiffSys::new(m_sparse, k_sparse);

    // sanity: A·1 = M^{-1} K·1 should be ~0 (constant in null space, periodic)
    let ones = Mat::<f64>::from_fn(n, 1, |_, _| 1.0);
    let a_one = sys.apply_minv_k(ones.as_ref());
    let max_row_sum = (0..n).map(|i| a_one[(i, 0)].abs()).fold(0.0_f64, f64::max);
    println!("A = M^-1 K . ones max (periodic null-space check): {:.3e}", max_row_sum);

    // --- initial condition: periodic Gaussian bump ---------------------------
    let xs = problem.dof_positions();
    let mut y0 = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        y0[(i, 0)] = periodic_gaussian(xs[i], x0, sigma);
    }

    // --- integrate with implicit Euler (DIRK + implicit_euler tableau) --------
    let mut solver = DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::implicit_euler(), 1e-12, 1e-12);

    // snapshot store: (t, Vec<f64> profile)
    let mut snapshots: Vec<(f64, Vec<f64>)> = Vec::new();
    let mut y = y0.clone();
    snapshots.push((0.0, (0..n).map(|i| y[(i, 0)]).collect()));
    for step in 1..=nsteps {
        let res = solver.step(&sys, dt).unwrap();
        let t = res.t;
        y = res.y.clone();
        // state of the integrator advances y; fetch from res
        if step % snapshot_every == 0 || step == nsteps {
            snapshots.push((t, (0..n).map(|i| y[(i, 0)]).collect()));
        }
        solver.accept_step(res);
    }

    // --- verification: at t=2.0 the bump advects back to x0 (modulo diffusion)
    let mut max_err = 0.0_f64;
    for i in 0..n {
        let u_init = periodic_gaussian(xs[i], x0, sigma);
        let u_final = y[(i, 0)];
        max_err = max_err.max((u_final - u_init).abs());
    }
    println!("final t = {:.3}, max |u_final - u_initial|: {:.3e}", solver.time(), max_err);

    // --- CSV output ----------------------------------------------------------
    let out_path = "target/ex_nd_1d_out.csv";
    let mut f = File::create(out_path).expect("failed to create output csv");
    writeln!(f, "t,x,u").unwrap();
    for (t, profile) in &snapshots {
        for (i, u) in profile.iter().enumerate() {
            writeln!(f, "{:.6},{:.6},{:.9e}", t, xs[i], u).unwrap();
        }
    }
    println!("wrote {} snapshots x {} dofs to {}", snapshots.len(), n, out_path);
}

/// Periodic Gaussian evaluated at position `x` on [0,1) centred at `x0`.
fn periodic_gaussian(x: f64, x0: f64, sigma: f64) -> f64 {
    let two_s2 = 2.0 * sigma * sigma;
    // wrap the periodic distance to [-0.5, 0.5)
    let mut dx = x - x0;
    if dx > 0.5 {
        dx -= 1.0;
    } else if dx < -0.5 {
        dx += 1.0;
    }
    (-(dx * dx) / two_s2).exp()
}

// =============================================================================
// Boundary integrator (extension stub, NOT implemented in Phase 1).
//
// When 2D/3D Neumann/Robin flux BCs land, a `BoundaryIntegrator` trait will
// run a facet-loop assembly path independent of the cell-kernel `assemble`:
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
// Phase 1 leaves this as documented trait + ctx only -- no callers, no
// implementations, no build path.  Implementing it requires a facet iterator
// over the mesh's codimension-1 boundary, facet-per-cell-type quadrature
// rules, and the surface jacobian / precomputed normals.  Those land with
// the first real 2D/3D flux-BC problem, not before (YAGNI).
// =============================================================================

/// Per-facet boundary-assembly context (mirrors `LocalCtx` for facets).
///
/// Fields:
///   * `tdim`/`gdim` -- topological / ambient mesh dimension; on a facet,
///     the facet's intrinsic dimension is `tdim - 1` (codimension-1 boundary).
///   * `ncomp` -- vector components per dof (1 for scalar physics).
///   * `npts` -- number of facet quadrature points.
///   * `ndofs` -- number of local dofs supported on the facet (cell closure
///     dofs restricted to the facet's closure).
///   * `wts` -- reference-facet quadrature weights, length `npts`.
///   * `jfacet_det` -- facet jacobian determinant (surface measure) at each
///     Q point, length `npts`.  Multiply with `wts[q]` for the physical
///     facet integral weight.
///   * `normal` -- the outward unit normal of this facet in `R^gdim`,
///     length `gdim` (constant across Q points for a flat facet; a future
///     curved-facet extension would carry it per Q point).
///   * `values`/`grads` -- pushed-forward tabulation restricted to the
///     facet's closure dofs, with the same layouts as `LocalCtx`.  Accessed
///     via FacetCtx's own `test`/`trial` view methods (TODO when the facet
///     path is wired up; for Phase 1 the raw slices remain private).
#[allow(dead_code)]
pub struct FacetCtx<'a> {
    pub tdim: usize,
    pub gdim: usize,
    pub ncomp: usize,
    pub npts: usize,
    pub ndofs: usize,
    pub wts: &'a [f64],
    pub jfacet_det: &'a [f64],  // surface measure per Q point
    pub normal: &'a [f64],      // [gdim] -- one outward normal for this facet
    values: &'a [f64],          // [ndofs, ncomp, npts], npts-contiguous
    grads: &'a [f64],           // [ndofs, ncomp, gdim, npts]
}

/// Boundary integrator trait.  MOOSE-style `integrand` ergonomics: physics
/// authors implement `integrand_rhs` (required) and `integrand_mat`
/// (optional, for Robin) -- the framework's provided `assemble_*` methods
/// own the facet test x quadrature loop and surface-measure weighting via
/// `wts[q] * jfacet_det[q]`.  Override `assemble_*` for max-control kernels.
///
/// See the Phase 1 stub block above for the deferred wiring plan.
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
                    acc += ctx.wts[q] * ctx.jfacet_det[q]
                        * self.integrand_mat(ctx, q, ti, si);
                }
                mat[ti * n + si] = acc;
            }
        }
    }
}
