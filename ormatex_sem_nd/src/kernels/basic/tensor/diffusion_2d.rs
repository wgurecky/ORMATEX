use crate::common::{CellState, TensorCtx};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::diffusion_2d::KernelDiffusion2D;

/// Tensor diffusion `-div(nu grad(u))` for a scalar field.
///
/// Mathematics: the triple is `(0, nu*dx(u), nu*dy(u))`; the action adds the
/// `dnu` material-derivative terms. Serves the `BilinearForm` fast path.
/// Weak counterpart: [`KernelDiffusion2D`].
/// Tensor-product 2D diffusion kernel (sum-factorized counterpart).
pub struct TensorKernelDiffusion2D(pub KernelDiffusion2D);

impl TensorKernelDiffusion2D {
    pub fn new(nu: f64) -> Self {
        Self(KernelDiffusion2D::new(nu))
    }
    pub fn with_coefficient<N>(nu: N) -> Self
    where
        N: MaterialProperty<f64> + 'static,
    {
        Self(KernelDiffusion2D::with_coefficient(nu))
    }
}

impl TensorResidualKernel<2> for TensorKernelDiffusion2D {
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let nu = self.0.nu.eval(&ctx.material_context(Some(state), q));
        [0.0, nu * state.grad(0, q, 0), nu * state.grad(0, q, 1)]
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
        let nu = self.0.nu.eval(&material);
        let dnu = self.0.nu.derivative(&material, 0).unwrap_or(0.0);
        let du = direction.value(0, q);
        [
            0.0,
            nu * direction.grad(0, q, 0) + dnu * du * state.grad(0, q, 0),
            nu * direction.grad(0, q, 1) + dnu * du * state.grad(0, q, 1),
        ]
    }

}
