//! Coupled 1D EDAC + energy pipe regression against the analytic profile.
//!
//! Coarse, fast version of `ex_nd_1d_edac_heated_pipe`: monolithic `[u, p, T]`
//! solve with volumetric heating marched to steady state, checked against
//! `support/heated_pipe.rs`. The heating term only affects the answer through
//! the named volume source, so this also covers its set composition.

use faer::matrix_free::LinOp;
use faer::prelude::*;
use ndmesh::shapes::unit_interval;
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    BilinearOps, DofReduction1D, EdacNavierStokes1DConfig, FieldRegistry,
    ParallelOwnedMinvJacobian, SEM1DProblem, TensorKernelEdacMomentumConvectionSplit1D,
    TensorKernelEdacPressureAdvectionSplit1D, TensorKernelEdacPressureDiffusion1D,
    TensorKernelEdacPressureDivergence1D, TensorKernelEdacPressureGradient1D,
    TensorKernelEdacViscousStress1D, TensorKernelEnergyAdvectionDiffusion1D,
    TensorKernelVolumeSource, TensorResidualKernel, TensorResidualKernelSet,
    TensorResidualKernelSum,
};

#[path = "../examples/support/heated_pipe.rs"]
mod heated_pipe;
#[path = "../examples/support/linear_system.rs"]
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
        // ponytail: assembled direct solve; the coarse test mesh makes this
        // far cheaper than matrix-free Krylov.
        Box::new(ParallelOwnedMinvJacobian::new(
            self.problem
                .tensor_residual_operator(&self.kernel)
                .at_time(t)
                .assemble_jacobian(state),
            &self.m_inv,
        ))
    }
}

#[test]
fn volume_source_carries_heated_field_name() {
    let kernel = TensorKernelVolumeSource::with_field_names(0.5, ["T"]);
    assert_eq!(
        <TensorKernelVolumeSource as TensorResidualKernel<1>>::field_names(&kernel),
        Some(vec!["T".to_owned()])
    );
}

#[test]
fn heated_pipe_matches_analytic_temperature() {
    let nx = 16;
    let params = HeatedPipeParams {
        length: 1.0,
        inlet_velocity: 0.1,
        thermal_diffusivity: 0.01,
        heat_source: 0.5,
        inlet_temperature: 0.0,
    };
    let config = EdacNavierStokes1DConfig::new(1.0, 0.01, 10.0);
    let problem = SEM1DProblem::new(
        unit_interval(nx, 1),
        2,
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
    };
    let mut y0 = Mat::<f64>::zeros(problem.system_size(), 1);
    let t_field = problem.field_id("T").unwrap();
    let t_offset = problem.field_offset(t_field);
    for row in 0..problem.field_reduced_size(t_field) {
        y0[(t_offset + row, 0)] = params.inlet_temperature;
    }
    let mut solver =
        DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::implicit_euler(), 1e-10, 1e-10);
    for _ in 0..200 {
        let step = solver.step(&system, 0.2).unwrap();
        solver.accept_step(step);
    }
    let state = solver.state();

    let u = problem.field_values("u", state.as_ref()).unwrap();
    let t = problem.field_values("T", state.as_ref()).unwrap();
    let mut max_velocity_error = 0.0f64;
    for value in &u.values {
        max_velocity_error = max_velocity_error.max((value - params.inlet_velocity).abs());
    }
    assert!(
        max_velocity_error < 5e-3,
        "pipe velocity not uniform: {max_velocity_error}"
    );
    let mut max_error = 0.0f64;
    for (x, value) in t.positions.iter().zip(&t.values) {
        max_error = max_error.max((value - params.analytic_temperature(*x)).abs());
    }
    assert!(
        max_error < 5e-3,
        "heated-pipe temperature error too large: {max_error}"
    );
}
