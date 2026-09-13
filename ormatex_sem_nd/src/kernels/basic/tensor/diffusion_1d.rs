use crate::common::{CellState, TensorCtx};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::diffusion_1d::KernelDiffusion;

/// Tensor diffusion `d/dx(nu du/dx)` for a scalar field.
///
/// Mathematics: the `(f0, f1x, f1y)` triple is `(0, nu*u', 0)`; the Jacobian
/// action is `(0, nu*du' + dnu*du*u', 0)` with `nu` and its state derivative
/// from the material context. Weak counterpart: [`KernelDiffusion`].
/// Tensor-product 1D diffusion kernel (sum-factorized counterpart).
pub struct TensorKernelDiffusion(pub KernelDiffusion);

impl TensorKernelDiffusion {
    pub fn new(nu: f64) -> Self {
        Self(KernelDiffusion::new(nu))
    }
    pub fn with_coefficient<N>(nu: N) -> Self
    where
        N: MaterialProperty<f64> + 'static,
    {
        Self(KernelDiffusion::with_coefficient(nu))
    }
}

impl TensorResidualKernel<1> for TensorKernelDiffusion {
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        [
            0.0,
            self.0.nu.eval(&ctx.material_context(Some(state), q)) * state.grad(0, q, 0),
            0.0,
        ]
    }
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let material = ctx.material_context(Some(state), q);
        [
            0.0,
            self.0.nu.eval(&material) * direction.grad(0, q, 0)
                + self.0.nu.derivative(&material, 0).unwrap_or(0.0)
                    * direction.value(0, q)
                    * state.grad(0, q, 0),
            0.0,
        ]
    }

}
