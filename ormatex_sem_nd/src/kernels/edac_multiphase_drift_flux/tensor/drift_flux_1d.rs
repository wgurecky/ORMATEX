//! Conservative Ishii-Zuber drift-flux tensor kernel (1D).
//!
//! Mathematics: `d(F(a) sin(theta))/dx` for equation 2 with hindered drift
//! flux `F = a*V_gj(a)`; conservative triple `(0, -F sin(theta), 0)` with the
//! exact `dF/da` action. The pipe angle `theta` is a space-dependent
//! `MaterialProperty` (radians from horizontal).
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig, ALPHA_1D,
};
use crate::material::{ConstantCoefficient, MaterialProperty};

/// Tensor conservative drift flux (owns equation 2).
pub struct TensorDriftFlux1D {
    pub config: DriftFlux1DConfig,
    pub theta: Box<dyn MaterialProperty<f64>>,
}

impl TensorDriftFlux1D {
    pub fn new<T>(config: DriftFlux1DConfig, theta: T) -> Self
    where
        T: MaterialProperty<f64> + 'static,
    {
        Self {
            config,
            theta: Box::new(theta),
        }
    }

    /// Horizontal pipe (`theta = 0`, no axial drift); gravity still acts in
    /// the momentum balance only if a tilted `theta` is supplied there.
    pub fn horizontal(config: DriftFlux1DConfig) -> Self {
        Self::new(config, ConstantCoefficient(0.0_f64))
    }

    fn theta_at(&self, ctx: &TensorCtx<'_>, q: usize) -> f64 {
        self.theta.eval(&ctx.material_context(None, q))
    }
}

impl TensorResidualKernel<1> for TensorDriftFlux1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == ALPHA_1D
    }
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != ALPHA_1D {
            return [0.0; 3];
        }
        let a = state.value(ALPHA_1D, q);
        let f = self.config.axial_drift_flux(a, self.theta_at(ctx, q));
        [0.0, -f, 0.0]
    }
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != ALPHA_1D {
            return [0.0; 3];
        }
        let a = state.value(ALPHA_1D, q);
        let df = self
            .config
            .axial_drift_flux_derivative(a, self.theta_at(ctx, q));
        [0.0, -df * direction.value(ALPHA_1D, q), 0.0]
    }
}
