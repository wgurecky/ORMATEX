//! Smooth periodic 1D isothermal Euler in primitive variables `[u, rho]`.

use std::fs::File;
use std::io::Write;

use faer::prelude::*;
use ndelement::{ciarlet::CiarletElement, map::IdentityMap};
use ndmesh::{shapes::unit_interval, SingleElementMesh};
use ormatex::ode_sys::OdeSys;
use ormatex_sem_nd::{
    DofReduction1D, FieldRegistry, MatrixFreeMinvJacobian, ParallelOwnedMinvJacobian, SEM1DProblem,
    TensorKernelConservationLaw1D,
};

#[path = "support/isothermal_euler.rs"]
mod isothermal_euler;
#[path = "support/linear_system.rs"]
mod linear_system;
use isothermal_euler::IsothermalEuler1D;
use linear_system::{implicit_euler_final_state, lumped_inverse_mass};

type IntervalMesh = SingleElementMesh<f64, CiarletElement<f64, IdentityMap, f64>>;

struct IsothermalEulerSystem<'a> {
    problem: &'a SEM1DProblem<IntervalMesh>,
    kernel: TensorKernelConservationLaw1D<IsothermalEuler1D>,
    m_inv: Vec<f64>,
    assembled: bool,
}

impl<'a> IsothermalEulerSystem<'a> {
    fn new(
        problem: &'a SEM1DProblem<IntervalMesh>,
        kernel: TensorKernelConservationLaw1D<IsothermalEuler1D>,
    ) -> Self {
        let mass = problem.assemble_lumped_mass();
        let m_inv = lumped_inverse_mass(mass.as_ref());
        Self {
            problem,
            kernel,
            m_inv,
            assembled: std::env::args().any(|arg| arg == "--assembled-jacobian"),
        }
    }
}

impl<'a> OdeSys<'a> for IsothermalEulerSystem<'a> {
    fn frhs(&self, _t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .tensor_residual_operator(&self.kernel)
            .at_time(_t)
            .residual(state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(
        &'a self,
        t: f64,
        state: MatRef<'b, f64>,
    ) -> Box<dyn faer::matrix_free::LinOp<f64> + 'a> {
        if self.assembled {
            return Box::new(ParallelOwnedMinvJacobian::new(
                self.problem
                    .tensor_residual_operator(&self.kernel)
                    .at_time(t)
                    .assemble_jacobian(state),
                &self.m_inv,
            ));
        }
        Box::new(MatrixFreeMinvJacobian::new(
            self.problem
                .tensor_residual_operator(&self.kernel)
                .at_time(t),
            state.to_owned(),
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

    let mesh = unit_interval(nx, 1);
    let problem = SEM1DProblem::new(
        mesh,
        p,
        FieldRegistry::new(["u", "rho"]),
        DofReduction1D::Periodic { facets: [0, nx] },
    );
    let kernel = TensorKernelConservationLaw1D::new(IsothermalEuler1D::new(sound_speed));
    let system = IsothermalEulerSystem::new(&problem, kernel);
    let n = problem.reduced_size();
    let x = problem.dof_positions();
    let mut y0 = Mat::<f64>::zeros(2 * n, 1);
    for i in 0..n {
        let phase = 2.0 * std::f64::consts::PI * x[i];
        y0[(i, 0)] = amplitude * phase.sin();
        y0[(n + i, 0)] = rho0 + amplitude * phase.cos();
    }

    let y = implicit_euler_final_state(&system, y0.as_ref(), dt, nsteps, 1e-10);
    let u = problem.field_values("u", y.as_ref()).unwrap();
    let rho = problem.field_values("rho", y.as_ref()).unwrap();

    let mut output = File::create("target/ex_nd_1d_isothermal_euler_out.csv")
        .expect("failed to create output csv");
    writeln!(output, "x,u,rho").unwrap();
    for ((x, u_value), (_, rho_value)) in u
        .positions
        .iter()
        .zip(&u.values)
        .zip(rho.positions.iter().zip(&rho.values))
    {
        writeln!(output, "{:.6},{:.9e},{:.9e}", x, u_value, rho_value,).unwrap();
    }
}
