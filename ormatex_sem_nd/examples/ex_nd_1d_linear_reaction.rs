//! Three-species periodic 1D linear reaction-advection-diffusion.

use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ndmesh::shapes::unit_interval;
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_pade::PadeExpm;
use ormatex::ode_epirk::EpirkIntegrator;
use ormatex::ode_sys::IntegrateSys;
use ormatex_sem_nd::{
    DofReduction1D, FieldRegistry, KernelAdvDiff, KernelLinearReaction, SEM1DProblem,
};

#[path = "support/linear_system.rs"]
mod linear_system;
use linear_system::{sparse_add, LinearOdeSys};

const RATES: [(usize, usize, f64); 5] = [
    (0, 0, -0.1),
    (1, 0, 0.1),
    (1, 1, -10.0),
    (2, 1, 10.0),
    (2, 2, -0.01),
];

fn reaction_matrix() -> SparseColMat<usize, f64> {
    let triplets: Vec<_> = RATES
        .iter()
        .map(|&(row, column, value)| Triplet::new(row, column, value))
        .collect();
    SparseColMat::try_new_from_triplets(3, 3, &triplets).unwrap()
}

fn block_diagonal(scalar: SparseColMat<usize, f64>, fields: usize) -> SparseColMat<usize, f64> {
    let n = scalar.nrows();
    assert_eq!(scalar.ncols(), n);
    let (symbolic, values) = scalar.as_ref().parts();
    let columns = symbolic.col_ptr();
    let rows = symbolic.row_idx();
    let mut triplets = Vec::new();
    for field in 0..fields {
        for column in 0..n {
            for entry in columns[column]..columns[column + 1] {
                triplets.push(Triplet::new(
                    field * n + rows[entry],
                    field * n + column,
                    values[entry],
                ));
            }
        }
    }
    SparseColMat::try_new_from_triplets(fields * n, fields * n, &triplets).unwrap()
}

fn run_epi3(system: &LinearOdeSys, y0: MatRef<'_, f64>, dt: f64, nsteps: usize) -> Mat<f64> {
    let expmv = Box::new(PadeExpm::new(12));
    let krylov = KrylovExpm::new(expmv, 30, 100, 1e-12, Some(2));
    let mut solver = EpirkIntegrator::new(0.0, y0, "epi3".to_string(), krylov);
    for _ in 0..nsteps {
        let step = solver.step(system, dt).unwrap();
        solver.accept_step(step);
    }
    solver.state()
}

fn main() {
    let nx = 64;
    let p = 2;
    let diffusivity = 0.002;
    let velocity = 0.5;
    let dt = 0.01;
    let nsteps = 100;

    let problem = SEM1DProblem::new(
        unit_interval(nx, 1),
        p,
        FieldRegistry::new(["c0", "c1", "c2"]),
        DofReduction1D::Periodic { facets: [0, nx] },
    );
    let n = problem.reduced_size();
    let transport = block_diagonal(
        problem.assemble_bilinear(0.0, &KernelAdvDiff::new(diffusivity, velocity)),
        3,
    );
    let reaction = problem.assemble_bilinear(
        0.0,
        &KernelLinearReaction::with_field_names(reaction_matrix(), ["c0", "c1", "c2"]),
    );
    let operator = sparse_add(transport.as_ref(), reaction.as_ref());
    let mass = problem.assemble_lumped_mass();
    let system = LinearOdeSys::new(mass, operator, vec![0.0; 3 * n]);

    let positions = problem.dof_positions();
    let mut y0 = Mat::<f64>::zeros(3 * n, 1);
    for i in 0..n {
        y0[(i, 0)] = periodic_gaussian(positions[i], 0.5, 0.05);
    }
    assert!(y0.get(n..3 * n, ..).norm_max() == 0.0);

    let state = run_epi3(&system, y0.as_ref(), dt, nsteps);
    let c0 = problem.field_values("c0", state.as_ref()).unwrap();
    let c1 = problem.field_values("c1", state.as_ref()).unwrap();
    let c2 = problem.field_values("c2", state.as_ref()).unwrap();
    let mut output = File::create("target/ex_nd_1d_linear_reaction_out.csv")
        .expect("failed to create output csv");
    writeln!(output, "x,c0,c1,c2").unwrap();
    for (((x, c0_value), (_, c1_value)), (_, c2_value)) in c0
        .positions
        .iter()
        .zip(&c0.values)
        .zip(c1.positions.iter().zip(&c1.values))
        .zip(c2.positions.iter().zip(&c2.values))
    {
        writeln!(
            output,
            "{:.6},{:.9e},{:.9e},{:.9e}",
            x, c0_value, c1_value, c2_value,
        )
        .unwrap();
    }
}

fn periodic_gaussian(x: f64, center: f64, width: f64) -> f64 {
    (-4..=4)
        .map(|image| {
            let dx = x - center - image as f64;
            (-(dx * dx) / (4.0 * width * width)).exp()
        })
        .sum()
}
