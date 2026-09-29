use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::energy_advection_diffusion_2d::KernelEnergyAdvectionDiffusion2D;

/// Tensor energy advection-diffusion with velocity-coupled transport.
///
/// Mathematics: rectangular coupling with 3 inputs (`u`, `v`, `T`) and 1
/// output (`T`); the triple is `(u*dx(T) + v*dy(T), alpha*dx(T),
/// alpha*dy(T))` and the action differentiates both the transporting
/// velocity and `T`. Weak counterpart:
/// [`KernelEnergyAdvectionDiffusion2D`].
/// Tensor-product state-coupled energy kernel (sum-factorized counterpart).
pub struct TensorKernelEnergyAdvectionDiffusion2D(pub KernelEnergyAdvectionDiffusion2D);

impl TensorKernelEnergyAdvectionDiffusion2D {
    pub fn new(thermal_diffusivity: f64) -> Self {
        Self(KernelEnergyAdvectionDiffusion2D::new(thermal_diffusivity))
    }
}

impl TensorResidualKernel<2> for TensorKernelEnergyAdvectionDiffusion2D {
    #[inline]
    fn nfields(&self) -> usize {
        1
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    #[inline]
    fn input_nfields(&self) -> usize {
        3
    }

    #[inline]
    fn output_nfields(&self) -> usize {
        1
    }

    #[inline]
    fn input_field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v", "T"].into_iter().map(str::to_owned).collect())
    }

    #[inline]
    fn output_field_names(&self) -> Option<Vec<String>> {
        Some(["T"].into_iter().map(str::to_owned).collect())
    }

    /// Lane-packed energy residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed `[u, v, T]` solution.
    /// * `equation` - unused (single output).
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        _equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        assert_eq!(state.nfields, 3, "energy state must contain [u, v, T]");
        let u = state.value(0, q);
        let v = state.value(1, q);
        let gx = state.grad(2, q, 0);
        let gy = state.grad(2, q, 1);
        for l in 0..LANES {
            f0[l] = u[l] * gx[l] + v[l] * gy[l];
            f1x[l] = self.0.thermal_diffusivity * gx[l];
            f1y[l] = self.0.thermal_diffusivity * gy[l];
        }
    }

    /// Lane-packed energy Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed `[u, v, T]` linearization point.
    /// * `direction` - lane-packed `[u, v, T]` direction.
    /// * `equation` - unused (single output).
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0`/`f1x`/`f1y` - lane output slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        _equation: usize,
        q: usize,
        f0: &mut Lanes,
        f1x: &mut Lanes,
        f1y: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        assert_eq!(state.nfields, 3, "energy state must contain [u, v, T]");
        assert_eq!(
            direction.nfields, 3,
            "energy direction must contain [u, v, T]"
        );
        let u = state.value(0, q);
        let v = state.value(1, q);
        let gx = state.grad(2, q, 0);
        let gy = state.grad(2, q, 1);
        let du = direction.value(0, q);
        let dv = direction.value(1, q);
        let dgx = direction.grad(2, q, 0);
        let dgy = direction.grad(2, q, 1);
        for l in 0..LANES {
            f0[l] = du[l] * gx[l] + dv[l] * gy[l] + u[l] * dgx[l] + v[l] * dgy[l];
            f1x[l] = self.0.thermal_diffusivity * dgx[l];
            f1y[l] = self.0.thermal_diffusivity * dgy[l];
        }
    }
}
