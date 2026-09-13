//! Coupled 1D EDAC pipe flow with volumetric heating (`[u, p, T]`).
//!
//! A uniform flow enters at `u0` with fixed inlet temperature; a uniform
//! volumetric source heats the fluid and the outlet is adiabatic (natural
//! zero-flux, matching the non-conservative energy form). The tensor EDAC
//! split kernels, the state-coupled 1D energy kernel, and a named volume
//! source are composed monolithically and marched to steady state with
//! implicit Euler. The temperature profile is checked against the analytic
//! solution in `support/heated_pipe.rs`.

use std::fs::File;
use std::io::Write;

use faer::matrix_free::LinOp;
use faer::prelude::*;
use ndmesh::shapes::unit_interval;
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    BilinearOps, DofReduction1D, EdacNavierStokes1DConfig, FieldRegistry, MatrixFreeMinvJacobian,
    ParallelOwnedMinvJacobian, SEM1DProblem, TensorKernelEdacMomentumConvectionSplit1D,
    TensorKernelEdacPressureAdvectionSplit1D, TensorKernelEdacPressureDiffusion1D,
    TensorKernelEdacPressureDivergence1D, TensorKernelEdacPressureGradient1D,
    TensorKernelEdacViscousStress1D, TensorKernelEnergyAdvectionDiffusion1D,
    TensorKernelVolumeSource, TensorResidualKernelSet, TensorResidualKernelSum,
};

#[path = "support/heated_pipe.rs"]
mod heated_pipe;
#[path = "support/linear_system.rs"]
mod linear_system;
use heated_pipe::HeatedPipeParams;
use linear_system::lumped_inverse_mass;

type IntervalMesh = ndmesh::SingleElementMesh<
    f64,
    ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
>;

fn coupled_kernel(
    config: EdacNavierStokes1DConfig,
    alpha: f64,
    heat: f64,
) -> TensorResidualKernelSet<'static, 1> {
    let edac = TensorResidualKernelSum::from_kernel(
        TensorKernelEdacMomentumConvectionSplit1D::new(config),
    )
    .with(TensorKernelEdacPressureGradient1D::new(config))
    .with(TensorKernelEdacViscousStress1D::new(config))
    .with(TensorKernelEdacPressureDivergence1D::new(config))
    .with(TensorKernelEdacPressureAdvectionSplit1D::new(config))
    .with(TensorKernelEdacPressureDiffusion1D::new(config));

    TensorResidualKernelSet::from_kernel(edac)
        .with(TensorKernelEnergyAdvectionDiffusion1D::new(alpha))
        .with(TensorKernelVolumeSource::with_field_names(heat, ["T"]))
}

struct HeatedPipeSystem<'a> {
    problem: &'a SEM1DProblem<IntervalMesh>,
    kernel: TensorResidualKernelSet<'a, 1>,
    m_inv: Vec<f64>,
    assembled: bool,
}

impl<'a> OdeSys<'a> for HeatedPipeSystem<'a> {
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t)
            .residual(state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(&'a self, t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
        let operator = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(t);
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

fn main() {
    let assembled = std::env::args().any(|arg| arg == "--assembled-jacobian");
    let nx = 32;
    let p = 2;
    let params = HeatedPipeParams {
        length: 1.0,
        inlet_velocity: 0.1,
        thermal_diffusivity: 0.01,
        heat_source: 0.5,
        inlet_temperature: 0.0,
    };
    let config = EdacNavierStokes1DConfig::new(1.0, 0.01, 10.0);
    let dt = 0.05;
    let nsteps = 800;

    // Inlet: fixed velocity and temperature; outlet: pressure datum, adiabatic
    // (natural) temperature.
    let problem = SEM1DProblem::new(
        unit_interval(nx, 1),
        p,
        FieldRegistry::new(["u", "p", "T"]),
        DofReduction1D::FieldSpecific {
            reductions: vec![
                DofReduction1D::Dirichlet {
                    facets: vec![(0, params.inlet_velocity)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(nx, 0.0)],
                },
                DofReduction1D::Dirichlet {
                    facets: vec![(0, params.inlet_temperature)],
                },
            ],
        },
    );
    let system = HeatedPipeSystem {
        problem: &problem,
        kernel: coupled_kernel(config, params.thermal_diffusivity, params.heat_source),
        m_inv: lumped_inverse_mass(problem.assemble_lumped_mass().as_ref()),
        assembled,
    };

    // Start from rest at the inlet temperature and march to steady state.
    let mut y0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let t_field = problem.field_id("T").unwrap();
    let t_offset = problem.field_offset(t_field);
    for row in 0..problem.field_reduced_size(t_field) {
        y0[(t_offset + row, 0)] = params.inlet_temperature;
    }
    let mut solver =
        DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::implicit_euler(), 1e-10, 1e-10);
    for _ in 0..nsteps {
        let step = solver.step(&system, dt).unwrap();
        solver.accept_step(step);
    }
    let state = solver.state();

    let u = problem.field_values("u", state.as_ref()).unwrap();
    let t = problem.field_values("T", state.as_ref()).unwrap();
    let mut max_error = 0.0f64;
    let mut max_velocity_error = 0.0f64;
    for (x, value) in t.positions.iter().zip(&t.values) {
        assert!(value.is_finite(), "non-finite pipe temperature");
        max_error = max_error.max((value - params.analytic_temperature(*x)).abs());
    }
    for value in &u.values {
        max_velocity_error = max_velocity_error.max((value - params.inlet_velocity).abs());
    }
    assert!(
        max_velocity_error < 2e-2,
        "pipe velocity not uniform: {max_velocity_error}"
    );
    assert!(
        max_error < 2e-2,
        "heated-pipe temperature error too large: {max_error}"
    );

    std::fs::create_dir_all("target").expect("failed to create output directory");
    let mut output =
        File::create("target/ex_nd_1d_edac_heated_pipe.csv").expect("failed to create output csv");
    writeln!(output, "x,u,T,T_analytic").unwrap();
    let p_field = problem.field_values("p", state.as_ref()).unwrap();
    for (((x, u_value), (_, t_value)), (_, _)) in u
        .positions
        .iter()
        .zip(&u.values)
        .zip(t.positions.iter().zip(&t.values))
        .zip(p_field.positions.iter().zip(&p_field.values))
    {
        writeln!(
            output,
            "{x:.6},{u_value:.9e},{t_value:.9e},{:.9e}",
            params.analytic_temperature(*x),
        )
        .unwrap();
    }
    println!(
        "heated pipe: max velocity error={max_velocity_error:.3e}, max temperature error={max_error:.3e}"
    );
}
