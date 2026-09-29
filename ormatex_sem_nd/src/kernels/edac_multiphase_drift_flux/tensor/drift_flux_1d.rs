//! Conservative Ishii-Zuber drift-flux tensor kernel (1D).
//!
//! Mathematics: `d(F(a) sin(theta))/dx` for equation 2 with hindered drift
//! flux `F = a*V_gj(a)`; conservative triple `(0, -F sin(theta), 0)` with the
//! exact `dF/da` action. The pipe angle `theta` is a space-dependent
//! `MaterialProperty` (radians from horizontal).
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
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

    /// Lane-packed conservative drift-flux residual for all lanes.
    ///
    /// The pipe angle is evaluated per lane from that lane's context, exactly
    /// like the scalar path (the callback is state-independent, so the lane
    /// state is not passed, mirroring `theta_at`).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `equation` - void equation shared by all lanes.
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
        if equation != ALPHA_1D {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let mut theta = [0.0; LANES];
        for l in 0..LANES {
            theta[l] = self.theta.eval(&ctxs[l].lane_material_context(None, l, q));
        }
        let alpha = state.value(ALPHA_1D, q);
        let f = self.config.axial_drift_flux_lanes(alpha, &theta);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = -f[l];
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed conservative drift-flux Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - void equation shared by all lanes.
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
        if equation != ALPHA_1D {
            *f0 = [0.0; LANES];
            *f1x = [0.0; LANES];
            *f1y = [0.0; LANES];
            return;
        }
        let mut theta = [0.0; LANES];
        for l in 0..LANES {
            theta[l] = self.theta.eval(&ctxs[l].lane_material_context(None, l, q));
        }
        let alpha = state.value(ALPHA_1D, q);
        let df = self.config.axial_drift_flux_derivative_lanes(alpha, &theta);
        let da = direction.value(ALPHA_1D, q);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = -df[l] * da[l];
            f1y[l] = 0.0;
        }
    }
}
