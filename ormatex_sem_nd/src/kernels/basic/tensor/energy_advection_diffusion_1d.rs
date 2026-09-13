use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::energy_advection_diffusion_1d::KernelEnergyAdvectionDiffusion1D;

/// Tensor 1D energy advection-diffusion with velocity-coupled transport.
///
/// Mathematics: rectangular coupling with 2 inputs (`u`, `T`) and 1 output
/// (`T`); the triple is `(u*dx(T), alpha*dx(T), 0)` and the action
/// differentiates both the transporting velocity and `T`. Weak counterpart:
/// [`KernelEnergyAdvectionDiffusion1D`].
/// Tensor-product state-coupled energy kernel (sum-factorized counterpart).
pub struct TensorKernelEnergyAdvectionDiffusion1D(pub KernelEnergyAdvectionDiffusion1D);

impl TensorKernelEnergyAdvectionDiffusion1D {
    pub fn new(thermal_diffusivity: f64) -> Self {
        Self(KernelEnergyAdvectionDiffusion1D::new(thermal_diffusivity))
    }
}

impl TensorResidualKernel<1> for TensorKernelEnergyAdvectionDiffusion1D {
    fn nfields(&self) -> usize {
        1
    }

    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn input_nfields(&self) -> usize {
        2
    }

    fn output_nfields(&self) -> usize {
        1
    }

    fn input_field_names(&self) -> Option<Vec<String>> {
        Some(["u", "T"].into_iter().map(str::to_owned).collect())
    }

    fn output_field_names(&self) -> Option<Vec<String>> {
        Some(["T"].into_iter().map(str::to_owned).collect())
    }

    fn tensor_residual(
        &self,
        _ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        assert_eq!(state.nfields, 2, "energy state must contain [u, T]");
        [
            state.value(0, q) * state.grad(1, q, 0),
            self.0.thermal_diffusivity * state.grad(1, q, 0),
            0.0,
        ]
    }

    fn tensor_jacobian_action(
        &self,
        _ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        assert_eq!(state.nfields, 2, "energy state must contain [u, T]");
        assert_eq!(direction.nfields, 2, "energy direction must contain [u, T]");
        [
            direction.value(0, q) * state.grad(1, q, 0)
                + state.value(0, q) * direction.grad(1, q, 0),
            self.0.thermal_diffusivity * direction.grad(1, q, 0),
            0.0,
        ]
    }
}
