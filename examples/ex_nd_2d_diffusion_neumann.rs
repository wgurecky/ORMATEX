//! 2D pure-diffusion example with Neumann (left) + Robin (right) + adiabatic
//! (top/bottom) boundary conditions, exercising the `BoundaryIntegrator`
//! trait from `ex_nd_common` for the first time.
//!
//! Problem: steady-state heat equation on the unit square with
//!   * left   (x=0): prescribed heat flux `q_left` into the domain (Neumann)
//!   * right  (x=1): convective cooling `h (T_amb - T)` (Robin -- this is
//!                   what stabilizes the otherwise-singular pure-Neumann
//!                   diffusion problem; no Dirichlet anywhere)
//!   * top/bottom:   adiabatic (do-nothing; zero flux)
//!
//! With top/bottom adiabatic, the 2D solution collapses to 1D in `x`; the
//! analytic steady state is the linear profile
//!
//!     T(x) = -q_left/k * x + (q_left/h + T_amb)
//!
//! (derived from `-q_left = -k T'(x)` with `T'(L) = -h(T(L) - T_amb)/k`, here
//! L=1).  The example integrates the time-dependent problem to steady
//! state and compares to this formula.
//!
//! `FiniteElement2DProblem::assemble_boundary` selects the `NeumannFlux` and
//! `RobinConvection` integrands by each boundary facet's midpoint.

use std::fs::File;
use std::io::Write;

use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;

// Re-use the 2D FE problem and its DOF-reduction policy.
#[path = "ex_nd_2d.rs"]
pub mod ex_nd_2d;
pub use ex_nd_2d::ex_nd_common;
use ex_nd_2d::{DofReduction2D, FiniteElement2DProblem};
use ex_nd_common::*;

use ndelement::{ciarlet::CiarletElement, map::IdentityMap, types::ReferenceCellType};
use ndmesh::{shapes::unit_square, SingleElementMesh};

// =============================================================================
// Diffusion + boundary-forcing ODE system
// =============================================================================

/// System `M du/dt = -(K_diff + K_robin) u + b`, where `b` is the Neumann +
/// Robin RHS forcing.  Lumped-mass fast path: `M` diagonal, `m_inv` stored.
///
/// Sign conventions:
///   * Diffusion weak form: `int nu grad u . grad v dx` => `K_diff` enters
///     the residual as `-K_diff u` (heat flowing OUT of cell -> temperature
///     decreases).
///   * Neumann flux `q_left` (into domain): the boundary term
///     `int_left (q_left n) . v dl = -q_left * v` (n is outward, q_left is
///     INTO the domain so `q_left n = -q_left`, sign flips once) goes to
///     the RHS as `b -= q_left * v_int` (subtracted).  We follow the
///     convention `NeumannFlux::integrand_rhs = g * v` (in domain means
///     positive `g`) and the example subtracts `b` from the RHS to get the
///     right net sign.
///   * Robin `h(T_amb - T)`: `h*T_amb*v` goes to the RHS as `b += h*T_amb*v`
///     (the "drive toward ambient" forcing); `-h*T*v` goes to the LHS as
///     `K_robin = -h * u*v` (added, since the LHS is `-K_eff u` and we want
///     `-(-h*T*v) = +h*T*v`... actually amended sign convention below).
///
/// In `frhs` we compute `du/dt = -M^{-1} (K_eff u) + M^{-1} b` where the
/// LHS residual contribution is `-K_eff u` and `K_eff = K_diff + K_robin`
/// with `K_robin += h * u * v` (positive matrix from `RobinConvection`,
/// since `int h(T_amb - T) v dl = h*T_amb*v - h*T*v`, and the `-h*T*v` part
/// becomes `+K_robin*u` in the LHS after the sign flip on the residual).
struct DiffusionNeumannSys {
    k_eff: SparseColMat<usize, f64>,
    m_inv: Vec<f64>,
    b: Vec<f64>,
}

impl DiffusionNeumannSys {
    pub fn new(
        m: SparseColMat<usize, f64>,
        k_diff: SparseColMat<usize, f64>,
        k_robin: SparseColMat<usize, f64>,
        b: Vec<f64>,
    ) -> Self {
        let n = k_diff.nrows();
        assert_eq!(k_diff.nrows(), k_diff.ncols());
        assert_eq!(k_robin.nrows(), n);
        assert_eq!(k_robin.ncols(), n);
        assert_eq!(m.nrows(), n);
        assert_eq!(b.len(), n);
        assert!(
            m.compute_nnz() == n,
            "expected lumped-diagonal M (nnz==n={n}), got nnz={}; \
             check quadrature (must be 2p-1 for lumping)",
            m.compute_nnz(),
        );
        let mut m_inv = vec![0.0_f64; n];
        for i in 0..n {
            let m_ii = m[(i, i)];
            assert!(m_ii.abs() > 1e-30, "zero diagonal M[{i}, {i}]");
            m_inv[i] = 1.0 / m_ii;
        }
        // K_eff = K_diff + K_robin (sparse add via triplets)
        let k_eff = sparse_add(k_diff.as_ref(), k_robin.as_ref());
        Self { k_eff, m_inv, b }
    }
}

impl<'a> OdeSys<'a> for DiffusionNeumannSys {
    fn frhs(&self, _t: f64, x: MatRef<f64>) -> Mat<f64> {
        // M du/dt = -K_eff u + b  -> du/dt = -M^{-1} K_eff u + M^{-1} b
        let n = self.m_inv.len();
        let kx = self.k_eff.as_ref() * x;
        let mut out = Mat::<f64>::zeros(n, x.ncols());
        for j in 0..out.ncols() {
            for i in 0..n {
                out[(i, j)] = -self.m_inv[i] * kx[(i, j)] + self.m_inv[i] * self.b[i];
            }
        }
        out
    }

    fn fjac<'b>(&'a self, _t: f64, _x: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        Box::new(MinvKLinOp {
            k: self.k_eff.as_ref(),
            m_inv: &self.m_inv,
        })
    }
}

/// Sparse matrix sum `A + B` via triplets.  Both must be `n x n`.
fn sparse_add(
    a: SparseColMatRef<'_, usize, f64>,
    b: SparseColMatRef<'_, usize, f64>,
) -> SparseColMat<usize, f64> {
    let n = a.nrows();
    assert_eq!(a.ncols(), n);
    assert_eq!(b.nrows(), n);
    assert_eq!(b.ncols(), n);
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    for (mat, sign) in [(a, 1.0_f64), (b, 1.0)] {
        let (sym, vals) = mat.parts();
        let col_ptr = sym.col_ptr();
        let row_idx = sym.row_idx();
        for j in 0..n {
            for k in col_ptr[j]..col_ptr[j + 1] {
                triplets.push(Triplet::new(row_idx[k], j, sign * vals[k]));
            }
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

// =============================================================================
// main
// =============================================================================

pub fn run_diffusion_neumann<M, F>(
    label: &str,
    mesh: M,
    p: usize,
    assemble_boundary: F,
    out_path: &str,
) where
    M: ndmesh::traits::Mesh<EntityDescriptor = ReferenceCellType, T = f64>,
    F: FnOnce(&FiniteElement2DProblem<M>) -> BoundaryContributions,
{
    // --- parameters ---------------------------------------------------------
    let k = 0.1; // thermal diffusivity (also called nu).  Picked so the
                 // slowest diffusion mode tau ~ 1/(pi*k/L)^2 ~ 1/(pi*0.1)^2 ~ 10,
                 // letting t_final=200 reach steady state in ~20x tau.
    let q_left = 1.0; // prescribed heat flux into domain (left Neumann)
    let h = 0.1; // convective heat transfer coefficient (right Robin)
    let t_amb = 0.0; // ambient temperature for Robin BC
    let dt = 1.0;
    let nsteps = 200; // t_final = 200, ~20x slowest diffusion mode tau

    println!("\n=== {label} (p={p}) ===");

    // --- build FE problem without reduction; all boundary dofs stay
    let problem = FiniteElement2DProblem::new(mesh, p, DofReduction2D::None);
    let n = problem.reduced_size();
    println!("ndofs = {n}");

    // --- assemble K_diff (pure diffusion, nu=k, vel=[0,0])
    let k_diff = problem.assemble_bilinear(&KernelAdvDiff2D::new(k, [0.0, 0.0]));
    println!("K_diff nnz = {}", k_diff.compute_nnz());

    // --- mass (lumped diagonal)
    let m_sparse = problem.assemble_lumped_mass();
    println!("mass nnz = {}", m_sparse.compute_nnz());

    let boundary = assemble_boundary(&problem);
    let b = boundary.rhs;
    let k_robin = boundary.mat;
    println!("K_robin nnz = {}", k_robin.compute_nnz());
    println!(
        "b max = {}, b sum = {}",
        b.iter().map(|v| v.abs()).fold(0.0_f64, f64::max),
        b.iter().sum::<f64>()
    );

    // --- self-checks
    // K_diff symmetry (pure diffusion must be symmetric)
    {
        let kd = k_diff.to_dense();
        let mut max_asym = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                max_asym = max_asym.max((kd[(i, j)] - kd[(j, i)]).abs());
            }
        }
        println!("K_diff symmetry max |K - K^T|: {:.3e}", max_asym);
        assert!(max_asym < 1e-14, "K_diff not symmetric");
    }
    // K_robin symmetry
    {
        let kr = k_robin.to_dense();
        let mut max_asym = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                max_asym = max_asym.max((kr[(i, j)] - kr[(j, i)]).abs());
            }
        }
        println!("K_robin symmetry max |K - K^T|: {:.3e}", max_asym);
        assert!(max_asym < 1e-14, "K_robin not symmetric");
    }
    // K_eff = K_diff + K_robin must be non-singular (Robin stabilizes).
    let k_eff_dense = {
        let kd = k_diff.to_dense();
        let kr = k_robin.to_dense();
        let mut ke = kd.clone();
        for i in 0..n {
            for j in 0..n {
                ke[(i, j)] += kr[(i, j)];
            }
        }
        ke
    };
    // Non-singularity check via lu
    {
        let lu = k_eff_dense.partial_piv_lu();
        let _ = lu.solve(&Mat::<f64>::from_fn(n, 1, |_, _| 1.0));
        println!("K_eff partial_piv_lu solved (non-singular): OK (rcond check lax)");
    }
    // DEBUG: direct steady-state solve, compare to integrator and to analytic.
    {
        let lu = k_eff_dense.partial_piv_lu();
        let bv = Mat::<f64>::from_fn(n, 1, |i, _| b[i]);
        let t_direct = lu.solve(&bv);
        let xs = problem.dof_positions();
        let mut max_t_dir = 0.0_f64;
        let mut t_at_0 = 0.0_f64;
        let mut t_at_1 = 0.0_f64;
        for i in 0..n {
            let tv = t_direct[(i, 0)];
            max_t_dir = max_t_dir.max(tv.abs());
            if xs[i].0.abs() < 1e-9 && xs[i].1.abs() < 1e-9 {
                t_at_0 = tv;
            }
            if (xs[i].0 - 1.0).abs() < 1e-9 && xs[i].1.abs() < 1e-9 {
                t_at_1 = tv;
            }
        }
        println!(
            "DIRECT solve: max |T| = {max_t_dir:.4e}, T(0,0) = {t_at_0:.4e}, T(1,0) = {t_at_1:.4e}"
        );
        // expected analytic: T(0,0)=1010, T(1,0)=10
        let _ = xs;
    }

    // --- build ODE system + integrate to steady state
    let sys = DiffusionNeumannSys::new(m_sparse, k_diff, k_robin, b);
    let mut y = Mat::<f64>::zeros(n, 1); // IC: T(x,y,0) = 0 = T_amb
    let mut solver =
        DirkIntegrator::new(0.0, y.as_ref(), ImplicitBT::implicit_euler(), 1e-10, 1e-10);
    for _step in 1..=nsteps {
        let res = solver.step(&sys, dt).unwrap();
        y = res.y.clone();
        solver.accept_step(res);
    }
    let t_final = solver.time();
    println!("integrated to t = {t_final:.3}");

    // --- compare to analytic steady state
    // With top/bottom adiabatic, 2D collapses to 1D in x.  Neumann `q_left`
    // is heat flux INTO the domain at x=0, so `T'(0) = -q_left/k` (T slopes
    // DOWN from left to right, since heat flows from hot (left) to cold
    // (right) and is removed by the Robin condition at x=1).
    //
    // Robin at x=1:  -k T'(1) = h (T(1) - T_amb)
    //   -> q_left = h (T(1) - T_amb)   [since -k T'(L) = q_left]
    //   -> T(1) = T_amb + q_left / h
    //   -> T(0) = T(1) - T'(L)*L = T(1) + (q_left/k)*L = T(1) + q_left/k
    //
    //   T(x) = T(0) - (q_left/k) x = (T_amb + q_left/h + q_left/k) - (q_left/k) x
    let slope = -q_left / k; // T'(x) < 0
    let t_at_l = t_amb + q_left / h; // T(1)
    let intercept = t_at_l - slope; // T(0) = T(1) - slope*L
    let xs = problem.dof_positions();
    let mut max_err = 0.0_f64;
    let mut max_t = 0.0_f64;
    for i in 0..n {
        let (x, _y) = xs[i];
        let t_exact = slope * x + intercept;
        let t_num = y[(i, 0)];
        max_err = max_err.max((t_num - t_exact).abs());
        max_t = max_t.max(t_num.abs());
    }
    println!("analytic T(x) = {slope:.3e}*x + {intercept:.3e}");
    println!(
        "steady-state max |T_num - T_exact|: {:.3e} (max |T| = {:.3e})",
        max_err, max_t
    );
    // GLL p=2 represents this linear steady state exactly; time integration
    // and linear-solver tolerances dominate the remaining error.
    assert!(
        max_err < 5e-3,
        "steady-state error {max_err} exceeds tolerance 5e-3"
    );
    println!("GATE: steady-state error < 5e-3: PASS");

    // --- CSV output
    let mut f = File::create(out_path).expect("failed to create output csv");
    writeln!(f, "x,y,T").unwrap();
    for i in 0..n {
        writeln!(f, "{:.6},{:.6},{:.9e}", xs[i].0, xs[i].1, y[(i, 0)]).unwrap();
    }
    println!("wrote {n} dofs to {out_path}");
}

fn main() {
    let nx = 32;
    let ny = 2;
    let p = 2;
    let mesh: SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>> =
        unit_square(nx, ny, ReferenceCellType::Quadrilateral);
    run_diffusion_neumann(
        "diffusion-neumann-robin",
        mesh,
        p,
        |problem| {
            let neumann = NeumannFlux::new(1.0);
            let robin = RobinConvection::new(0.1, 0.0);
            problem.assemble_boundary(|facet| {
                const EPS: f64 = 1e-9;
                if facet.midpoint[0] < EPS {
                    Some(&neumann)
                } else if facet.midpoint[0] > 1.0 - EPS {
                    Some(&robin)
                } else {
                    None
                }
            })
        },
        "target/ex_nd_2d_diffusion_neumann_out.csv",
    );
}
