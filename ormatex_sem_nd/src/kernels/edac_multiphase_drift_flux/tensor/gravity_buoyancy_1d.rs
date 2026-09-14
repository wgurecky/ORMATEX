//! Buoyant gravity body-force tensor kernel (1D).
//!
//! Mathematics: `(rho_m - rho_l)/rho_m * g_axial(x)` for equation 0 with
//! axial gravity `g_axial = -|g| sin(theta(x))`; the pipe angle `theta` is a
//! space-dependent `MaterialProperty` (radians from horizontal) so the 1D
//! model still feels inclination. Vanishes at `alpha = 0`.
use crate::common::{CellState, TensorCtx};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac_multiphase_drift_flux::config_1d::{
    drift_field_names_1d, DriftFlux1DConfig, ALPHA_1D,
};
use crate::material::{ConstantCoefficient, MaterialProperty};

/// Tensor buoyant gravity source (owns equation 0).
pub struct TensorDriftGravity1D {
    pub config: DriftFlux1DConfig,
    pub theta: Box<dyn MaterialProperty<f64>>,
}

impl TensorDriftGravity1D {
    pub fn new<T>(config: DriftFlux1DConfig, theta: T) -> Self
    where
        T: MaterialProperty<f64> + 'static,
    {
        Self {
            config,
            theta: Box::new(theta),
        }
    }

    /// Horizontal pipe (`theta = 0`, no axial gravity).
    pub fn horizontal(config: DriftFlux1DConfig) -> Self {
        Self::new(config, ConstantCoefficient(0.0_f64))
    }

    fn gravity_at(&self, ctx: &TensorCtx<'_>, q: usize) -> f64 {
        let theta = self.theta.eval(&ctx.material_context(None, q));
        self.config.axial_gravity(theta)
    }
}

impl TensorResidualKernel<1> for TensorDriftGravity1D {
    fn nfields(&self) -> usize {
        3
    }
    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names_1d()
    }
    fn owns_equation(&self, equation: usize) -> bool {
        equation == 0
    }
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 0 {
            return [0.0; 3];
        }
        let rho = self.config.mixture_density(state.value(ALPHA_1D, q));
        [
            (rho - self.config.rho_l) / rho * self.gravity_at(ctx, q),
            0.0,
            0.0,
        ]
    }
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        if equation != 0 {
            return [0.0; 3];
        }
        let rho = self.config.mixture_density(state.value(ALPHA_1D, q));
        let drho = self.config.mixture_density_derivative();
        let factor = drho * self.config.rho_l / (rho * rho) * self.gravity_at(ctx, q);
        [factor * direction.value(ALPHA_1D, q), 0.0, 0.0]
    }
}
