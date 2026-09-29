use crate::common::{LaneState, Lanes, TensorCtx, LANES};

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
    #[inline]
    fn nfields(&self) -> usize {
        2
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    #[inline]
    fn input_nfields(&self) -> usize {
        1
    }

    #[inline]
    fn output_nfields(&self) -> usize {
        2
    }

    #[inline]
    fn input_field_names(&self) -> Option<Vec<String>> {
        Some(["T"].into_iter().map(str::to_owned).collect())
    }

    #[inline]
    fn output_field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v"].into_iter().map(str::to_owned).collect())
    }

    /// Lane-packed buoyancy residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed `[T]` solution.
    /// * `equation` - momentum equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
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
        assert_eq!(state.nfields, 1, "Boussinesq state must contain [T]");
        assert!(equation < 2);
        let t = state.value(0, q);
        let coeff = -self.0.buoyancy * self.0.gravity[equation];
        for l in 0..LANES {
            f0[l] = coeff * (t[l] - self.0.reference_temperature);
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed buoyancy Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed `[T]` linearization point (checked, unused).
    /// * `direction` - lane-packed `[T]` direction.
    /// * `equation` - momentum equation shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
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
        assert_eq!(state.nfields, 1, "Boussinesq state must contain [T]");
        assert_eq!(
            direction.nfields, 1,
            "Boussinesq direction must contain [T]"
        );
        assert!(equation < 2);
        let dt = direction.value(0, q);
        let coeff = -self.0.buoyancy * self.0.gravity[equation];
        for l in 0..LANES {
            f0[l] = coeff * dt[l];
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
