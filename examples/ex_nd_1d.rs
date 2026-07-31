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

/// per-quadrature-point integrand kernels
/// `u, v` are physical trial/test values and `grad_u, grad_v`
/// physical trial/test gradients at each quadrature point.
pub trait BilinearForm {
    fn q_f(&self, u: &[f64], v: &[f64], grad_u: &[f64], grad_v: &[f64]) -> Vec<f64>;
}

// mass kernel:  u * v
struct KernelMass {}

impl KernelMass {
    pub fn new() -> Self {
        Self {}
    }
}
impl BilinearForm for KernelMass {
    fn q_f(&self, u: &[f64], v: &[f64], _grad_u: &[f64], _grad_v: &[f64]) -> Vec<f64> {
        u.iter().zip(v.iter()).map(|(a, b)| *a * *b).collect()
    }
}

// advection diffusion kernel:  nu * grad_u . grad_v - vel * u * grad_v
struct KernelAdvDiff {
    // diffusion coeff
    nu: f64,
    // velocity
    vel: f64,
}

impl KernelAdvDiff {
    pub fn new(nu: f64, vel: f64) -> Self {
        Self { nu, vel }
    }
}
impl BilinearForm for KernelAdvDiff {
    fn q_f(&self, u: &[f64], _v: &[f64], grad_u: &[f64], grad_v: &[f64]) -> Vec<f64> {
        u.iter()
            .zip(grad_u.iter())
            .zip(grad_v.iter())
            .map(|((u_i, gu_i), gv_i)| {
                self.nu * (*gu_i) * (*gv_i) - self.vel * (*u_i) * (*gv_i)
            })
            .collect()
    }
}

// advection-diffusion kernel with SUPG stabilization.
// Replaces the test function v -> v + tau * vel * grad_v, which adds the
// term tau * (vel * grad_u) * (vel * grad_v) to the standard Galerkin form.
// Per quadrature point:
//   q = (nu + tau * vel^2) * grad_u * grad_v - vel * u * grad_v
struct KernelAdvDiffSUPG {
    // diffusion coeff
    nu: f64,
    // velocity
    vel: f64,
    // SUPG stabilization parameter
    tau: f64,
}

impl KernelAdvDiffSUPG {
    pub fn new(nu: f64, vel: f64, tau: f64) -> Self {
        Self { nu, vel, tau }
    }
}
impl BilinearForm for KernelAdvDiffSUPG {
    fn q_f(&self, u: &[f64], _v: &[f64], grad_u: &[f64], grad_v: &[f64]) -> Vec<f64> {
        let supg = self.tau * self.vel * self.vel;
        u.iter()
            .zip(grad_u.iter())
            .zip(grad_v.iter())
            .map(|((u_i, gu_i), gv_i)| {
                (self.nu + supg) * (*gu_i) * (*gv_i) - self.vel * (*u_i) * (*gv_i)
            })
            .collect()
    }
}

/// Linear operator applying `M^{-1} K` to a vector (or matrix of columns),
/// used as the `fjac` return.  `M` is diagonal (mass-lumped), so `M^{-1}`
/// action is an element-wise scale by `m_inv[i] = 1 / M[i,i]`.
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
        let kv = self.k * rhs;
        let n = self.m_inv.len();
        for j in 0..out.ncols() {
            for i in 0..n {
                out[(i, j)] = self.m_inv[i] * kv[(i, j)];
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
    pts: DynArray<f64, 2>,
    wts: Vec<f64>,
    // reference-cell tabulation with nderivs = 1 (values + d/dx)
    table: DynArray<f64, 4>,
    bc: BoundaryCondition,
    // global dof indices of the two boundary vertices (x=0, x=1).
    // queried from the function space so BC reduction is robust to dof
    // ordering (nd numbers vertex dofs before edge-interior dofs).
    left_end_dof: usize,
    right_end_dof: usize,
    // per-cell, per-quadrature-point geometry-map data, computed once in
    // with_bc().  Column c corresponds to the cell with local_index() == c.
    // Caching avoids recomputing the jacobians on every assemble() call.
    jinv_cache: DynArray<f64, 4>, // [tdim, gdim, npts, ncells]
    jdets_cache: Vec<f64>,        // [ncells * npts], row-major (c, q)
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
            // 2p-1 order returns p+1 GLL points = interpolation nodes, which
            // lumps the mass matrix to diagonal (classic SEM tradeoff).
            2 * element.lagrange_superdegree().saturating_sub(1),
        )
        .unwrap();
        let npts = w.len();
        // interval quadrature points are stored as barycentric pairs (x, 1-x);
        // the reference interval is [0, 1] (topological dim 1).
        let mut pts = rlst_dynamic_array!(f64, [1, npts]);
        for i in 0..npts {
            *pts.get_mut([0, i]).unwrap() = qpts[2 * i];
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

        Self {
            nx,
            p,
            mesh,
            family,
            pts,
            wts,
            table,
            bc,
            left_end_dof,
            right_end_dof,
            jinv_cache,
            jdets_cache,
        }
    }

    /// Number of full (unreduced) DOFs on the mesh.
    /// For continuous P_p on nx intervals: nx*p + 1 nodes.
    fn full_size(&self) -> usize {
        self.nx * self.p + 1
    }

    /// true if the given full DOF is an eliminated (Dirichlet) endpoint.
    fn is_eliminated(&self, full: usize) -> bool {
        match self.bc {
            BoundaryCondition::Periodic => false,
            BoundaryCondition::Sides { left, right } => {
                (full == self.left_end_dof && left == SideBC::Dirichlet)
                    || (full == self.right_end_dof && right == SideBC::Dirichlet)
            }
        }
    }

    /// true if the left and right boundary DOFs are identified (Periodic only).
    fn is_periodic(&self) -> bool {
        matches!(self.bc, BoundaryCondition::Periodic)
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
    fn target_dof(&self, full: usize) -> Option<usize> {
        if self.is_eliminated(full) {
            return None;
        }
        if self.is_periodic() && full == self.right_end_dof {
            // identify right endpoint with the left endpoint
            return Some(self.left_end_dof);
        }
        // count eliminated DOFs and the (periodic) identified right endpoint
        // that come before `full` to get a contiguous reduced index.
        let mut offset = 0;
        for i in 0..full {
            if self.is_eliminated(i) {
                offset += 1;
            } else if self.is_periodic() && i == self.right_end_dof {
                offset += 1;
            }
        }
        Some(full - offset)
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

    /// Assemble a sparse matrix using `kernel` per quadrature point.
    /// `phys_grad` selects whether physical gradients (`nderivs=1` entries,
    /// transformed by the cell inverse jacobian) are passed to the kernel.
    ///
    /// The per-cell jacobian data (jinv, jdets) is read from caches filled once
    /// in `with_bc()`, so only the tabulation + kernel work runs per call.
    pub fn assemble(&self, kernel: &dyn BilinearForm, phys_grad: bool) -> SparseColMat<usize, f64> {
        let space = FunctionSpaceImpl::new(&self.mesh, &self.family);

        let npts = self.wts.len();
        let n = space.process_size();
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();

        let mut u = vec![0.0_f64; npts];
        let mut v = vec![0.0_f64; npts];
        let mut gu = vec![0.0_f64; npts];
        let mut gv = vec![0.0_f64; npts];

        for cell in self.mesh.entity_iter(ReferenceCellType::Interval) {
            let c = cell.local_index();
            let dofs = space
                .entity_closure_dofs(ReferenceCellType::Interval, cell.local_index())
                .unwrap();

            for (test_i, test_dof) in dofs.iter().enumerate() {
                for q in 0..npts {
                    v[q] = *self.table.get([0, q, test_i, 0]).unwrap();
                    if phys_grad {
                        gv[q] = *self.jinv_cache.get([0, 0, q, c]).unwrap()
                            * *self.table.get([1, q, test_i, 0]).unwrap();
                    }
                }
                for (trial_i, trial_dof) in dofs.iter().enumerate() {
                    for q in 0..npts {
                        u[q] = *self.table.get([0, q, trial_i, 0]).unwrap();
                        if phys_grad {
                            gu[q] = *self.jinv_cache.get([0, 0, q, c]).unwrap()
                                * *self.table.get([1, q, trial_i, 0]).unwrap();
                        }
                    }
                    let qf = kernel.q_f(&u, &v, &gu, &gv);
                    let entry = (0..npts)
                        .map(|q| self.wts[q] * self.jdets_cache[c * npts + q] * qf[q])
                        .sum::<f64>();
                    triplets.push(Triplet::new(*test_dof, *trial_dof, entry));
                }
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    pub fn build_mass(&self) -> SparseColMat<usize, f64> {
        self.apply_bc(self.assemble(&KernelMass::new(), false).as_ref())
    }

    pub fn build_adv_diff(&self, nu: f64, vel: f64) -> SparseColMat<usize, f64> {
        self.apply_bc(self.assemble(&KernelAdvDiff::new(nu, vel), true).as_ref())
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
    let m_sparse = problem.apply_bc(problem.assemble(&KernelMass::new(), false).as_ref());
    let k_sparse = problem.apply_bc(problem.assemble(&KernelAdvDiff::new(nu, vel), true).as_ref());
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
        .apply_bc(problem.assemble(&KernelAdvDiffSUPG::new(nu, vel, 0.0), true).as_ref())
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
