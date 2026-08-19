use faer::matrix_free::LinOp;
use faer::prelude::*;
use ormatex::matexp_krylov::KrylovExpm;
use ormatex::matexp_pade::PadeExpm;
use ormatex::ode_epirk::EpirkIntegrator;
use ormatex::ode_sys::{IntegrateSys, OdeSys};
use ormatex_sem_nd::{MatrixFreeMinvJacobian, QuadMesh, ResidualKernel, SEM2DProblem};

use super::linear_system::lumped_inverse_mass;

pub struct FluidSystem<'a, K> {
    pub problem: &'a SEM2DProblem<QuadMesh>,
    pub kernel: K,
    m_inv: Vec<f64>,
}

impl<'a, K> FluidSystem<'a, K> {
    pub fn new(problem: &'a SEM2DProblem<QuadMesh>, kernel: K) -> Self {
        let mass = problem.assemble_system_lumped_mass();
        Self {
            problem,
            kernel,
            m_inv: lumped_inverse_mass(mass.as_ref()),
        }
    }
}

impl<'a, K> OdeSys<'a> for FluidSystem<'a, K>
where
    K: ResidualKernel + Sync + Send,
{
    fn frhs(&self, t: f64, state: MatRef<f64>) -> Mat<f64> {
        let residual = self
            .problem
            .assemble_system_residual_at(t, &self.kernel, state);
        Mat::from_fn(self.m_inv.len(), 1, |row, _| {
            -self.m_inv[row] * residual[row]
        })
    }

    fn fjac<'b>(&'a self, t: f64, state: MatRef<'b, f64>) -> Box<dyn LinOp<f64> + 'a> {
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

pub fn epi3(state0: MatRef<'_, f64>) -> EpirkIntegrator<KrylovExpm> {
    let expmv = Box::new(PadeExpm::new(12));
    let krylov = KrylovExpm::new(expmv, 30, 100, 1e-12, Some(2));
    EpirkIntegrator::new(0.0, state0, "epi3".to_string(), krylov)
}

pub fn advance(
    system: &FluidSystem<'_, impl ResidualKernel + Sync + Send>,
    state0: MatRef<'_, f64>,
    dt: f64,
    nsteps: usize,
) -> Mat<f64> {
    let mut integrator = epi3(state0);
    for step in 0..nsteps {
        let result = integrator
            .step(system, dt)
            .unwrap_or_else(|error| panic!("EDAC step {step} failed: {}", error.msg));
        integrator.accept_step(result);
    }
    integrator.state()
}
