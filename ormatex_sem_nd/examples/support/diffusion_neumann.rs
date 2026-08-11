use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
use ndelement::types::ReferenceCellType;
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    BoundaryContributions, DofReduction2D, KernelAdvDiff2D, MeshMetadata, SEM2DProblem,
};

use super::linear_system::MinvKLinOp;

fn sparse_add(
    a: SparseColMatRef<'_, usize, f64>,
    b: SparseColMatRef<'_, usize, f64>,
) -> SparseColMat<usize, f64> {
    let n = a.nrows();
    let mut triplets = Vec::new();
    for mat in [a, b] {
        let (symbolic, values) = mat.parts();
        let columns = symbolic.col_ptr();
        let rows = symbolic.row_idx();
        for column in 0..n {
            for entry in columns[column]..columns[column + 1] {
                triplets.push(Triplet::new(rows[entry], column, values[entry]));
            }
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

pub fn run_diffusion_neumann<M, F>(
    label: &str,
    mesh: M,
    p: usize,
    assemble_boundary: F,
    out_path: &str,
) where
    M: ndmesh::traits::Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    F: FnOnce(&SEM2DProblem<M>) -> BoundaryContributions,
{
    run_diffusion_neumann_with_metadata(
        label,
        mesh,
        p,
        MeshMetadata::default(),
        assemble_boundary,
        out_path,
    );
}

pub fn run_diffusion_neumann_with_metadata<M, F>(
    label: &str,
    mesh: M,
    p: usize,
    metadata: MeshMetadata,
    assemble_boundary: F,
    out_path: &str,
) where
    M: ndmesh::traits::Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    F: FnOnce(&SEM2DProblem<M>) -> BoundaryContributions,
{
    let k = 0.1;
    let q_left = 1.0;
    let h = 0.1;
    let t_amb = 0.0;
    let dt = 1.0;
    let nsteps = 200;
    println!("\n=== {label} (p={p}) ===");

    let problem = SEM2DProblem::new_with_metadata(mesh, p, DofReduction2D::None, metadata);
    let n = problem.reduced_size();
    let k_diff = problem.assemble_bilinear(&KernelAdvDiff2D::new(k, [0.0, 0.0]));
    let mass = problem.assemble_lumped_mass();
    let boundary = assemble_boundary(&problem);
    let b = boundary.rhs;
    let k_eff = sparse_add(k_diff.as_ref(), boundary.mat.as_ref());

    let system = DiffusionNeumannSys::new(mass, k_eff, b);
    let mut y = Mat::<f64>::zeros(n, 1);
    let mut solver =
        DirkIntegrator::new(0.0, y.as_ref(), ImplicitBT::implicit_euler(), 1e-10, 1e-10);
    for _ in 0..nsteps {
        let step = solver.step(&system, dt).unwrap();
        y = step.y.clone();
        solver.accept_step(step);
    }

    let positions = problem.dof_positions();
    let slope = -q_left / k;
    let intercept = t_amb + q_left / h - slope;
    let mut max_error = 0.0_f64;
    for row in 0..n {
        max_error = max_error.max((y[(row, 0)] - (slope * positions[row].0 + intercept)).abs());
    }
    assert!(
        max_error < 5e-3,
        "steady-state error {max_error} exceeds tolerance"
    );
    let mut output = File::create(out_path).expect("failed to create output csv");
    writeln!(output, "x,y,T").unwrap();
    for row in 0..n {
        writeln!(
            output,
            "{:.6},{:.6},{:.9e}",
            positions[row].0,
            positions[row].1,
            y[(row, 0)]
        )
        .unwrap();
    }
}

struct DiffusionNeumannSys {
    k_eff: SparseColMat<usize, f64>,
    m_inv: Vec<f64>,
    b: Vec<f64>,
}

impl DiffusionNeumannSys {
    fn new(mass: SparseColMat<usize, f64>, k_eff: SparseColMat<usize, f64>, b: Vec<f64>) -> Self {
        let n = k_eff.nrows();
        assert_eq!(k_eff.ncols(), n);
        assert_eq!(mass.nrows(), n);
        assert_eq!(mass.ncols(), n);
        assert_eq!(b.len(), n);
        assert_eq!(
            mass.compute_nnz(),
            n,
            "expected lumped diagonal mass matrix"
        );
        let m_inv = (0..n)
            .map(|i| {
                let value = mass[(i, i)];
                assert!(value.abs() > 1e-30, "zero mass diagonal at {i}");
                1.0 / value
            })
            .collect();
        Self { k_eff, m_inv, b }
    }
}

impl<'a> OdeSys<'a> for DiffusionNeumannSys {
    fn frhs(&self, _t: f64, x: MatRef<f64>) -> Mat<f64> {
        let kx = self.k_eff.as_ref() * x;
        Mat::from_fn(x.nrows(), x.ncols(), |row, column| {
            -self.m_inv[row] * kx[(row, column)] + self.m_inv[row] * self.b[row]
        })
    }

    fn fjac<'b>(
        &'a self,
        _t: f64,
        x: MatRef<'b, f64>,
    ) -> Box<dyn faer::matrix_free::LinOp<f64> + 'a> {
        let _ = x;
        Box::new(MinvKLinOp {
            k: self.k_eff.as_ref(),
            m_inv: &self.m_inv,
        })
    }
}
