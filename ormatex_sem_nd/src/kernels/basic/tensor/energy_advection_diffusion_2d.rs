use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::energy_advection_diffusion_2d::KernelEnergyAdvectionDiffusion2D;

/// Tensor energy advection-diffusion with velocity-coupled transport.
///
/// Mathematics: rectangular coupling with 3 inputs (`u`, `v`, `T`) and 1
/// output (`T`); the triple is `(u*dx(T) + v*dy(T), alpha*dx(T),
/// alpha*dy(T))` and the action differentiates both the transporting
/// velocity and `T`. Weak counterpart:
/// [`KernelEnergyAdvectionDiffusion2D`].
/// Tensor-product state-coupled energy kernel (sum-factorized counterpart).
pub struct TensorKernelEnergyAdvectionDiffusion2D(pub KernelEnergyAdvectionDiffusion2D);

impl TensorKernelEnergyAdvectionDiffusion2D {
    pub fn new(thermal_diffusivity: f64) -> Self {
        Self(KernelEnergyAdvectionDiffusion2D::new(thermal_diffusivity))
    }
}

impl TensorResidualKernel<2> for TensorKernelEnergyAdvectionDiffusion2D {
    fn nfields(&self) -> usize {
        1
    }

    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn input_nfields(&self) -> usize {
        3
    }

    fn output_nfields(&self) -> usize {
        1
    }

    fn input_field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v", "T"].into_iter().map(str::to_owned).collect())
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
        assert_eq!(state.nfields, 3, "energy state must contain [u, v, T]");
        [
            state.value(0, q) * state.grad(2, q, 0) + state.value(1, q) * state.grad(2, q, 1),
            self.0.thermal_diffusivity * state.grad(2, q, 0),
            self.0.thermal_diffusivity * state.grad(2, q, 1),
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
        assert_eq!(state.nfields, 3, "energy state must contain [u, v, T]");
        assert_eq!(
            direction.nfields, 3,
            "energy direction must contain [u, v, T]"
        );
        [
            direction.value(0, q) * state.grad(2, q, 0)
                + direction.value(1, q) * state.grad(2, q, 1)
                + state.value(0, q) * direction.grad(2, q, 0)
                + state.value(1, q) * direction.grad(2, q, 1),
            self.0.thermal_diffusivity * direction.grad(2, q, 0),
            self.0.thermal_diffusivity * direction.grad(2, q, 1),
        ]
    }
}
