//! Sequential 1D flow + 3-species reaction-advection-diffusion on the same mesh.
//!
//! Two problems share `unit_interval(nx)` with `P=2`:
//! fluid `[u, rho]` (tensor isothermal Euler, periodic) and species
//! `[c0, c1, c2]` (frozen-velocity tensor adv-diff + linear reaction,
//! inlet Dirichlet `c0=1, c1=c2=0` on the left endpoint, directional
//! advective outflow on the right endpoint).
//! An outer loop alternates fluid substeps, frozen-velocity refresh, and
//! species substeps. Tensor kernels + matrix-free Jacobian by default
//! (`--assembled-jacobian` opts into the assembled backend).

use std::fs::File;
use std::io::Write;

use faer::matrix_free::LinOp;
use faer::prelude::*;
use faer::sparse::{SparseColMat, Triplet};
use ndmesh::shapes::unit_interval;
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    BilinearOps, ConstantCoefficient, DofReduction1D, FieldRegistry, KernelAdvectionOutflow1D,
    MatrixFreeMinvJacobian, ParallelOwnedMinvJacobian, StateBoundaryTerms,
    TensorKernelAdvDiff, TensorKernelConservationLaw1D, TensorKernelLinearReaction,
    TensorResidualKernelSet, SEM1DProblem,
};

#[path = "support/isothermal_euler.rs"]
mod isothermal_euler;
#[path = "support/linear_system.rs"]
mod linear_system;
use isothermal_euler::IsothermalEuler1D;
use linear_system::lumped_inverse_mass;

type IntervalMesh = ndmesh::SingleElementMesh<
    f64,
    ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
>;

// Same rates as `ex_nd_1d_linear_reaction`.
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

fn species_kernel(
    frozen: ormatex_sem_nd::FrozenQuadratureField,
    diffusivity: f64,
) -> TensorResidualKernelSet<'static, 1> {
    // ponytail: one frozen snapshot shared by 3 species via Arc clones.
    let t0 = TensorKernelAdvDiff::with_coefficients(
        ConstantCoefficient(diffusivity),
        frozen.clone(),
    )
    .with_field_name("c0");
    let t1 = TensorKernelAdvDiff::with_coefficients(
        ConstantCoefficient(diffusivity),
        frozen.clone(),
    )
    .with_field_name("c1");
    let t2 = TensorKernelAdvDiff::with_coefficients(ConstantCoefficient(diffusivity), frozen)
        .with_field_name("c2");
    let reaction = TensorKernelLinearReaction::with_field_names(
        reaction_matrix(),
        ["c0", "c1", "c2"],
    );
    TensorResidualKernelSet::from_kernel(t0)
        .with(t1)
        .with(t2)
        .with(reaction)
}

struct FluidSystem<'a> {
    problem: &'a SEM1DProblem<IntervalMesh>,
    kernel: TensorKernelConservationLaw1D<IsothermalEuler1D>,
    m_inv: Vec<f64>,
    assembled: bool,
}

struct SpeciesSystem<'a> {
    problem: &'a SEM1DProblem<IntervalMesh>,
    kernel: TensorResidualKernelSet<'a, 1>,
    terms: StateBoundaryTerms,
    m_inv: Vec<f64>,
    assembled: bool,
}

fn minv_residual(m_inv: &[f64], residual: Vec<f64>) -> Mat<f64> {
    Mat::from_fn(m_inv.len(), 1, |row, _| -m_inv[row] * residual[row])
}

impl<'a> OdeSys<'a> for FluidSystem<'a> {
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .residual(state);
        minv_residual(&self.m_inv, residual)
    }
    fn fjac<'b>(
        &'a self,
        t: f64,
        state: MatRef<'b, f64>,
    ) -> Box<dyn LinOp<f64> + 'a> {
        let operator = self.problem.tensor_residual_operator(&self.kernel).at_time(t);
        if self.assembled {
            Box::new(ParallelOwnedMinvJacobian::new(
                operator.assemble_jacobian(state),
                &self.m_inv,
            ))
        } else {
            Box::new(MatrixFreeMinvJacobian::new(
                operator,
                state.to_owned(),
                &self.m_inv,
            ))
        }
    }
}

impl<'a> OdeSys<'a> for SpeciesSystem<'a> {
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .with_state_boundary(&self.terms)
            .residual(state);
        minv_residual(&self.m_inv, residual)
    }
    fn fjac<'b>(
        &'a self,
        t: f64,
        state: MatRef<'b, f64>,
    ) -> Box<dyn LinOp<f64> + 'a> {
        let operator = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .with_state_boundary(&self.terms);
        if self.assembled {
            Box::new(ParallelOwnedMinvJacobian::new(
                operator.assemble_jacobian(state),
                &self.m_inv,
            ))
        } else {
            Box::new(MatrixFreeMinvJacobian::new(
                operator,
                state.to_owned(),
                &self.m_inv,
            ))
        }
    }
}

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let assembled = args.iter().any(|a| a == "--assembled-jacobian");
    let outer: usize = parse_flag(&args, "--outer")
        .map(|s| s.parse().expect("invalid --outer"))
        .unwrap_or(20);
    let fluid_sub: usize = parse_flag(&args, "--fluid-steps")
        .map(|s| s.parse().expect("invalid --fluid-steps"))
        .unwrap_or(2);
    let species_sub: usize = parse_flag(&args, "--species-steps")
        .map(|s| s.parse().expect("invalid --species-steps"))
        .unwrap_or(2);
    let dt: f64 = parse_flag(&args, "--dt")
        .map(|s| s.parse().expect("invalid --dt"))
        .unwrap_or(0.002);

    let nx = 64;
    let p = 2;
    let diffusivity = 0.002;

    let fluid_problem = SEM1DProblem::new(
        unit_interval(nx, 1),
        p,
        FieldRegistry::new(["u", "rho"]),
        DofReduction1D::Periodic { facets: [0, nx] },
    );
    // Inlet = left endpoint facet 0: c0=1, c1=c2=0.
    let species_problem = SEM1DProblem::new(
        unit_interval(nx, 1),
        p,
        FieldRegistry::new(["c0", "c1", "c2"]),
        DofReduction1D::FieldSpecific {
            reductions: vec![
                DofReduction1D::Dirichlet {
                    facets: vec![(0, 1.0)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(0, 0.0)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(0, 0.0)],
                },
            ],
        },
    );
    assert_eq!(fluid_problem.cell_count(), species_problem.cell_count());
    assert_eq!(
        fluid_problem.quadrature_points_per_cell(),
        species_problem.quadrature_points_per_cell()
    );

    let fluid_kernel =
        TensorKernelConservationLaw1D::new(IsothermalEuler1D::new(1.0));
    let fluid_system = FluidSystem {
        problem: &fluid_problem,
        kernel: fluid_kernel,
        m_inv: lumped_inverse_mass(fluid_problem.assemble_lumped_mass().as_ref()),
        assembled,
    };
    let species_m_inv = lumped_inverse_mass(species_problem.assemble_lumped_mass().as_ref());

    // Initial fluid: small acoustic perturbation; species: zeros (inlet via BC).
    let n = fluid_problem.reduced_size();
    let x = fluid_problem.dof_positions();
    let fluid_state = {
        let mut s = Mat::<f64>::zeros(2 * n, 1);
        for i in 0..n {
            let phase = 2.0 * std::f64::consts::PI * x[i];
            s[(i, 0)] = 0.02 * phase.sin();
            s[(n + i, 0)] = 1.0 + 0.02 * phase.cos();
        }
        s
    };
    let mut species_solver = DirkIntegrator::new(
        0.0,
        Mat::<f64>::zeros(species_problem.system_size(), 1).as_ref(),
        ImplicitBT::implicit_euler(),
        1e-10,
        1e-10,
    );
    let mut fluid_solver = DirkIntegrator::new(
        0.0,
        fluid_state.as_ref(),
        ImplicitBT::implicit_euler(),
        1e-10,
        1e-10,
    );

    for _ in 0..outer {
        for _ in 0..fluid_sub {
            let step = fluid_solver.step(&fluid_system, dt).unwrap();
            fluid_solver.accept_step(step);
        }
        let frozen = fluid_problem
            .sample_quadrature_field(fluid_solver.state().as_ref(), "u");
        frozen.assert_compatible(
            species_problem.cell_count(),
            species_problem.quadrature_points_per_cell(),
        );
        // Directional advective outflow at the right endpoint: lets species
        // leave the domain, assumes zero concentration for re-entering flow.
        // Without it the dropped volume boundary flux acts as a closed wall
        // and species pile up at the outlet.
        let outlet_vel = fluid_problem.sample_endpoint_value(
            fluid_solver.state().as_ref(),
            "u",
            nx,
        );
        let outflow = KernelAdvectionOutflow1D::with_field_names(
            outlet_vel,
            ["c0", "c1", "c2"],
        );
        let species_system = SpeciesSystem {
            problem: &species_problem,
            kernel: species_kernel(frozen, diffusivity),
            terms: StateBoundaryTerms::new().with_entities([nx], outflow),
            m_inv: species_m_inv.clone(),
            assembled,
        };
        for _ in 0..species_sub {
            let step = species_solver.step(&species_system, dt).unwrap();
            species_solver.accept_step(step);
        }
    }

    let fluid_final = fluid_solver.state();
    let species_final = species_solver.state();
    let u = fluid_problem.field_values("u", fluid_final.as_ref()).unwrap();
    let rho = fluid_problem.field_values("rho", fluid_final.as_ref()).unwrap();
    let c0 = species_problem
        .field_values("c0", species_final.as_ref())
        .unwrap();
    let c1 = species_problem
        .field_values("c1", species_final.as_ref())
        .unwrap();
    let c2 = species_problem
        .field_values("c2", species_final.as_ref())
        .unwrap();
    std::fs::create_dir_all("target").expect("failed to create output directory");
    let mut output = File::create("target/frozen_velocity_species_1d.csv")
        .expect("failed to create output csv");
    writeln!(output, "x,u,rho,c0,c1,c2").unwrap();
    for i in 0..u.positions.len() {
        writeln!(
            output,
            "{:.6},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
            u.positions[i], u.values[i], rho.values[i], c0.values[i], c1.values[i], c2.values[i],
        )
        .unwrap();
    }
    println!(
        "frozen 1D: outer={outer} dofs(fluid={}, species={}) assembled={assembled} -> target/frozen_velocity_species_1d.csv",
        fluid_problem.system_size(),
        species_problem.system_size(),
    );
}
