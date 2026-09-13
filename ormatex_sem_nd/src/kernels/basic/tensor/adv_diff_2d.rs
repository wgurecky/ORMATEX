use crate::common::{CellState, TensorCtx};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::adv_diff_2d::KernelAdvDiff2D;

/// Tensor advection-diffusion for a scalar field.
///
/// Mathematics: the triple is `(0, nu*dx(u) - velx*u, nu*dy(u) - vely*u)`;
/// the action carries the `dnu`/`dvel` material-derivative product-rule
/// terms. Weak counterpart: [`KernelAdvDiff2D`].
/// Tensor-product 2D advection-diffusion kernel (sum-factorized counterpart).
pub struct TensorKernelAdvDiff2D(pub KernelAdvDiff2D);

impl TensorKernelAdvDiff2D {
    pub fn new(nu: f64, vel: [f64; 2]) -> Self {
        Self(KernelAdvDiff2D::new(nu, vel))
    }
    pub fn with_coefficients<N, V>(nu: N, vel: [V; 2]) -> Self
    where
        N: MaterialProperty<f64> + 'static,
        V: MaterialProperty<f64> + 'static,
    {
        Self(KernelAdvDiff2D::with_coefficients(nu, vel))
    }
    pub fn with_field_name(self, name: impl Into<String>) -> Self {
        Self(self.0.with_field_name(name))
    }
}

impl TensorResidualKernel<2> for TensorKernelAdvDiff2D {
    fn field_names(&self) -> Option<Vec<String>> {
        <KernelAdvDiff2D as crate::kernels::common::ResidualKernel>::field_names(&self.0)
    }
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let material = ctx.material_context(Some(state), q);
        let value = state.value(0, q);
        let nu = self.0.nu.eval(&material);
        [
            0.0,
            nu * state.grad(0, q, 0) - self.0.vel[0].eval(&material) * value,
            nu * state.grad(0, q, 1) - self.0.vel[1].eval(&material) * value,
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
        let value = state.value(0, q);
        let dv = direction.value(0, q);
        let nu = self.0.nu.eval(&material);
        let dnu = self.0.nu.derivative(&material, 0).unwrap_or(0.0);
        [
            0.0,
            nu * direction.grad(0, q, 0) + dnu * dv * state.grad(0, q, 0)
                - (self.0.vel[0].eval(&material)
                    + self.0.vel[0].derivative(&material, 0).unwrap_or(0.0) * value)
                    * dv,
            nu * direction.grad(0, q, 1) + dnu * dv * state.grad(0, q, 1)
                - (self.0.vel[1].eval(&material)
                    + self.0.vel[1].derivative(&material, 0).unwrap_or(0.0) * value)
                    * dv,
        ]
    }

}
