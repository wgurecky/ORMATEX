//! Buoyant gravity body-force tensor kernel (1D).
//!
//! Mathematics: `(rho_m - rho_l)/rho_m * g_axial(x)` for equation 0 with
//! axial gravity `g_axial = -|g| sin(theta(x))`; the pipe angle `theta` is a
//! space-dependent `MaterialProperty` (radians from horizontal) so the 1D
//! model still feels inclination. Vanishes at `alpha = 0`.
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
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

    /// Lane-packed buoyant-gravity residual for all lanes.
    ///
    /// The axial gravity is evaluated per lane from that lane's context,
    /// exactly like the scalar path (the angle callback is state-independent,
    /// so the lane state is not passed, mirroring `gravity_at`).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `equation` - momentum equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0` - lane value slots. Overwritten.
    /// * `f1x` - lane x-flux slots. Overwritten.
    /// * `f1y` - lane y-flux slots. Overwritten with `0.0` (unused in 1D).
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        if equation != 0 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let alpha = state.value(ALPHA_1D, q);
        for l in 0..LANES {
            let theta = self.theta.eval(&ctxs[l].lane_material_context(None, l, q));
            let rho = self.config.mixture_density(alpha[l]);
            f0[l] = (rho - self.config.rho_l) / rho * self.config.axial_gravity(theta);
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed buoyant-gravity Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - momentum equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0` - lane linearized value slots. Overwritten.
    /// * `f1x` - lane linearized x-flux slots. Overwritten.
    /// * `f1y` - lane linearized y-flux slots. Overwritten with `0.0` (unused in 1D).
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        if equation != 0 {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let drho = self.config.mixture_density_derivative();
        let alpha = state.value(ALPHA_1D, q);
        let da = direction.value(ALPHA_1D, q);
        for l in 0..LANES {
            let theta = self.theta.eval(&ctxs[l].lane_material_context(None, l, q));
            let rho = self.config.mixture_density(alpha[l]);
            let factor = drho * self.config.rho_l / (rho * rho) * self.config.axial_gravity(theta);
            f0[l] = factor * da[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
