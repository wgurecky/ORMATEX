use crate::common::{CellState, TensorCtx};
use crate::material::MaterialProperty;

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::advection_2d::KernelAdvection2D;

/// Tensor conservative advection for a scalar field.
///
/// Mathematics: the triple is `(0, -velx*u, -vely*u)`, i.e. the `-(vel u)`
/// flux whose weak form is `-(vel*u).grad(v)`; the action adds the `dvel`
/// material-derivative terms. Weak counterpart:
/// [`KernelAdvection2D`].
/// Tensor-product 2D advection kernel (sum-factorized counterpart).
pub struct TensorKernelAdvection2D(pub KernelAdvection2D);

impl TensorKernelAdvection2D {
    pub fn new(vel: [f64; 2]) -> Self {
        Self(KernelAdvection2D::new(vel))
    }
    pub fn with_velocity<V>(vel: [V; 2]) -> Self
    where
        V: MaterialProperty<f64> + 'static,
    {
        Self(KernelAdvection2D::with_velocity(vel))
    }
}

impl TensorResidualKernel<2> for TensorKernelAdvection2D {
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        _equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let material = ctx.material_context(Some(state), q);
        let value = state.value(0, q);
        [
            0.0,
            -self.0.vel[0].eval(&material) * value,
            -self.0.vel[1].eval(&material) * value,
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
        [
            0.0,
            -(self.0.vel[0].eval(&material)
                + self.0.vel[0].derivative(&material, 0).unwrap_or(0.0) * value)
                * dv,
            -(self.0.vel[1].eval(&material)
                + self.0.vel[1].derivative(&material, 0).unwrap_or(0.0) * value)
                * dv,
        ]
    }

}
