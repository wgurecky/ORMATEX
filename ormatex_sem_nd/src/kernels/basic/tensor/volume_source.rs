use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::volume_source::KernelVolumeSource;

/// Tensor constant volumetric source, GDIM-generic.
///
/// Mathematics: the triple is `(-val, 0, 0)` matching the `ResidualKernel`
/// sign convention (sources move to the LHS); the action is zero. Weak
/// counterpart: [`KernelVolumeSource`].
/// Tensor-product volumetric source kernel (sum-factorized counterpart).
pub struct TensorKernelVolumeSource(pub KernelVolumeSource);

impl TensorKernelVolumeSource {
    pub fn new(val: f64) -> Self {
        Self(KernelVolumeSource::new(val))
    }

    pub fn with_field_names<I, S>(val: f64, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self(KernelVolumeSource::with_field_names(val, names))
    }
}

impl<const GDIM: usize> TensorResidualKernel<GDIM> for TensorKernelVolumeSource {
    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        <KernelVolumeSource as crate::kernels::common::ResidualKernel>::field_names(&self.0)
    }

    /// Lane-packed source residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed solution (unused).
    /// * `equation` - unused (single output).
    /// * `q` - quadrature-point index shared by all lanes (unused).
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorCtx<'_>],
        _state: &LaneState<'_>,
        _equation: usize,
        _q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        for l in 0..LANES {
            f0[l] = -self.0.val;
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed source Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed linearization point (unused).
    /// * `direction` - lane-packed Gateaux direction (unused).
    /// * `equation` - unused (single output).
    /// * `q` - quadrature-point index shared by all lanes (unused).
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorCtx<'_>],
        _state: &LaneState<'_>,
        _direction: &LaneState<'_>,
        _equation: usize,
        _q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        for l in 0..LANES {
            f0[l] = 0.0;
            f1x[l] = 0.0;
            f1y[l] = 0.0;
        }
    }
}
