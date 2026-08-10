//! Smooth periodic 1D isothermal Euler in primitive variables `[u, rho]`.

use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap};
use ndmesh::{shapes::unit_interval, SingleElementMesh};
use ormatex::ode_implicit::DirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex::tableau_implicit::ImplicitBT;
use ormatex_sem_nd::{
    DofReduction1D, KernelConservationLaw1D, MatrixFreeMinvJacobian, ResidualKernel, SEM1DProblem,
};

#[path = "support/isothermal_euler.rs"]
mod isothermal_euler;
use isothermal_euler::IsothermalEuler1D;

type IntervalMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

struct IsothermalEulerSystem<'a> {
    problem: &'a SEM1DProblem<IntervalMesh>,
    kernel: KernelConservationLaw1D<IsothermalEuler1D>,
    m_inv: Vec<f64>,
}

impl<'a> IsothermalEulerSystem<'a> {
    fn new(
        problem: &'a SEM1DProblem<IntervalMesh>,
        kernel: KernelConservationLaw1D<IsothermalEuler1D>,
    ) -> Self {
        let mass = problem.assemble_system_lumped_mass(kernel.nfields());
        let m_inv = (0..mass.nrows())
            .map(|i| {
                let value = mass[(i, i)];
                assert!(value > 0.0, "zero lumped mass at {i}");
                1.0 / value
            })
            .collect();
        Self {
            problem,
            kernel,
            m_inv,
        }
    }
}

impl<'a> OdeSys<'a> for IsothermalEulerSystem<'a> {
    fn frhs(&self, _t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .assemble_system_residual_at(_t, &self.kernel, state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(
        &'a self,
        t: f64,
        state: MatRef<'b, f64>,
    ) -> Box<dyn faer::matrix_free::LinOp<f64> + 'a> {
        Box::new(MatrixFreeMinvJacobian::new_at(
            t,
            self.problem,
            &self.kernel,
            state.to_owned(),
            None,
            &self.m_inv,
        ))
    }
}

fn main() {
    let nx = 32;
    let p = 2;
    let sound_speed = 1.0;
    let rho0 = 1.0;
    let amplitude = 0.02;
    let dt = 0.002;
    let nsteps = 25;

    let mesh = unit_interval(nx);
    let problem = SEM1DProblem::new(mesh, p, DofReduction1D::Periodic { facets: [0, nx] });
    let kernel = KernelConservationLaw1D::new(IsothermalEuler1D::new(sound_speed));
    let system = IsothermalEulerSystem::new(&problem, kernel);
    let n = problem.reduced_size();
    let x = problem.dof_positions();
    let mut y0 = Mat::<f64>::zeros(2 * n, 1);
    for i in 0..n {
        let phase = 2.0 * std::f64::consts::PI * x[i];
        y0[(i, 0)] = amplitude * phase.sin();
        y0[(n + i, 0)] = rho0 + amplitude * phase.cos();
    }

    let mut solver =
        DirkIntegrator::new(0.0, y0.as_ref(), ImplicitBT::implicit_euler(), 1e-10, 1e-10);
    let mut y = y0;
    for _ in 0..nsteps {
        let step = solver.step(&system, dt).unwrap();
        y = step.y.clone();
        solver.accept_step(step);
    }

    let mut output = File::create("target/ex_nd_1d_isothermal_euler_out.csv")
        .expect("failed to create output csv");
    writeln!(output, "x,u,rho").unwrap();
    for i in 0..n {
        writeln!(
            output,
            "{:.6},{:.9e},{:.9e}",
            x[i],
            y[(i, 0)],
            y[(n + i, 0)]
        )
        .unwrap();
    }
}
