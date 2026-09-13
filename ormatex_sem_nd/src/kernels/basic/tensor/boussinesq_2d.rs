use crate::common::{CellState, TensorCtx};

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::boussinesq_2d::KernelBoussinesq2D;

/// Tensor Boussinesq buoyancy coupling `T` into momentum.
///
/// Mathematics: rectangular coupling with 1 input (`T`) and 2 outputs
/// (`u`, `v`); the triple is `(-buoyancy*g[i]*(T - T_ref), 0, 0)` and the
/// action `(-buoyancy*g[i]*dT, 0, 0)`. Weak counterpart:
/// [`KernelBoussinesq2D`].
/// Tensor-product Boussinesq buoyancy kernel (sum-factorized counterpart).
pub struct TensorKernelBoussinesq2D(pub KernelBoussinesq2D);

impl TensorKernelBoussinesq2D {
    pub fn new(buoyancy: f64, gravity: [f64; 2]) -> Self {
        Self(KernelBoussinesq2D::new(buoyancy, gravity))
    }

    pub fn with_reference_temperature(mut self, value: f64) -> Self {
        self.0 = self.0.with_reference_temperature(value);
        self
    }
}

impl TensorResidualKernel<2> for TensorKernelBoussinesq2D {
    fn nfields(&self) -> usize {
        2
    }

    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn input_nfields(&self) -> usize {
        1
    }

    fn output_nfields(&self) -> usize {
        2
    }

    fn input_field_names(&self) -> Option<Vec<String>> {
        Some(["T"].into_iter().map(str::to_owned).collect())
    }

    fn output_field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v"].into_iter().map(str::to_owned).collect())
    }

    fn tensor_residual(
        &self,
        _ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        assert_eq!(state.nfields, 1, "Boussinesq state must contain [T]");
        assert!(equation < 2);
        [
            -self.0.buoyancy * self.0.gravity[equation] * self.0.temperature(state, q),
            0.0,
            0.0,
        ]
    }

    fn tensor_jacobian_action(
        &self,
        _ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        assert_eq!(state.nfields, 1, "Boussinesq state must contain [T]");
        assert_eq!(
            direction.nfields, 1,
            "Boussinesq direction must contain [T]"
        );
        assert!(equation < 2);
        [
            -self.0.buoyancy * self.0.gravity[equation] * direction.value(0, q),
            0.0,
            0.0,
        ]
    }

}
