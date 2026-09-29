use crate::common::{LaneState, Lanes, TensorCtx, LANES};

use crate::kernels::common::TensorResidualKernel;

use crate::kernels::basic::weak::energy_advection_diffusion_1d::KernelEnergyAdvectionDiffusion1D;

/// Tensor 1D energy advection-diffusion with velocity-coupled transport.
///
/// Mathematics: rectangular coupling with 2 inputs (`u`, `T`) and 1 output
/// (`T`); the triple is `(u*dx(T), alpha*dx(T), 0)` and the action
/// differentiates both the transporting velocity and `T`. Weak counterpart:
/// [`KernelEnergyAdvectionDiffusion1D`].
/// Tensor-product state-coupled energy kernel (sum-factorized counterpart).
pub struct TensorKernelEnergyAdvectionDiffusion1D(pub KernelEnergyAdvectionDiffusion1D);

impl TensorKernelEnergyAdvectionDiffusion1D {
    pub fn new(thermal_diffusivity: f64) -> Self {
        Self(KernelEnergyAdvectionDiffusion1D::new(thermal_diffusivity))
    }
}

impl TensorResidualKernel<1> for TensorKernelEnergyAdvectionDiffusion1D {
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
        2
    }

    #[inline]
    fn output_nfields(&self) -> usize {
        1
    }

    #[inline]
    fn input_field_names(&self) -> Option<Vec<String>> {
        Some(["u", "T"].into_iter().map(str::to_owned).collect())
    }

    #[inline]
    fn output_field_names(&self) -> Option<Vec<String>> {
        Some(["T"].into_iter().map(str::to_owned).collect())
    }

    /// Lane-packed energy residual for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed `[u, T]` solution.
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
        assert_eq!(state.nfields, 2, "energy state must contain [u, T]");
        let u = state.value(0, q);
        let g = state.grad(1, q, 0);
        for l in 0..LANES {
            f0[l] = u[l] * g[l];
            f1x[l] = self.0.thermal_diffusivity * g[l];
            f1y[l] = 0.0;
        }
    }

    /// Lane-packed energy Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`] (unused).
    /// * `state` - lane-packed `[u, T]` linearization point.
    /// * `direction` - lane-packed `[u, T]` direction.
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
        assert_eq!(state.nfields, 2, "energy state must contain [u, T]");
        assert_eq!(direction.nfields, 2, "energy direction must contain [u, T]");
        let u = state.value(0, q);
        let g = state.grad(1, q, 0);
        let du = direction.value(0, q);
        let dg = direction.grad(1, q, 0);
        for l in 0..LANES {
            f0[l] = du[l] * g[l] + u[l] * dg[l];
            f1x[l] = self.0.thermal_diffusivity * dg[l];
            f1y[l] = 0.0;
        }
    }
}
