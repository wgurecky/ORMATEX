//! Tensor Dong OBC-C outflow for EDAC (`[u, v, p]`).
//!
//! Mathematics: `u_n u / 2` consistency flux for the weak conservative half
//! of split momentum advection, Dong's smoothed OBC-C traction
//! (`tanh` switch on `u.n / (delta * U0)`), and the matching `u_n p / 2`
//! flux for split pressure advection. With `split_flux`, pairs with
//! split-form volume kernels; otherwise with conservative-form volumes.
//! Needs the SplitBoundaryFlux default on remaining facets.
use crate::common::{LaneState, Lanes, TensorFacetCtx, LANES};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac::config::fluid_field_names;
use crate::kernels::edac::weak::dong_outflow::KernelEdacDongOutflow2D;

/// Tensor-product Dong OBC-C boundary kernel for monolithic EDAC.
pub struct TensorKernelEdacDongOutflow2D {
    pub rho: f64,
    pub delta: f64,
    pub velocity_scale: f64,
    pub split_flux: bool,
}

impl TensorKernelEdacDongOutflow2D {
    pub fn new(rho: f64, delta: f64, velocity_scale: f64) -> Self {
        let kernel = KernelEdacDongOutflow2D::new(rho, delta, velocity_scale);
        Self {
            rho: kernel.rho,
            delta: kernel.delta,
            velocity_scale: kernel.velocity_scale,
            split_flux: kernel.split_flux,
        }
    }

    pub fn with_split_flux(mut self) -> Self {
        self.split_flux = true;
        self
    }

    pub(crate) fn as_weak(&self) -> KernelEdacDongOutflow2D {
        KernelEdacDongOutflow2D {
            rho: self.rho,
            delta: self.delta,
            velocity_scale: self.velocity_scale,
            split_flux: self.split_flux,
        }
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorKernelEdacDongOutflow2D {
    #[inline]
    fn nfields(&self) -> usize {
        3
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    /// Lane-packed Dong residual for all lanes.
    ///
    /// Per lane `l` computes exactly the scalar expression with
    /// `&ctxs[l]` (same operations in the same order; the `tanh` switch
    /// needs no lane-divergent branch).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed facet solution.
    /// * `equation` - output equation shared by all lanes.
    /// * `q` - facet quadrature-point index shared by all lanes.
    /// * `out` - lane trace-flux slots. Overwritten.
    #[inline]
    fn tensor_residual(
        &self,
        ctxs: &[TensorFacetCtx<'_>],
        state: &LaneState<'_>,
        equation: usize,
        q: usize,
        out: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let weak = self.as_weak();
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let p = state.value(2, q);
        for l in 0..LANES {
            let normal = ctxs[l].normal;
            let velocity = [u0[l], u1[l]];
            let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
            out[l] = match equation {
                0 | 1 => {
                    let dong_flux = weak.dong_flux(normal, velocity);
                    (if self.split_flux {
                        0.5 * normal_velocity * velocity[equation]
                    } else {
                        0.0
                    }) - p[l] * normal[equation] / self.rho
                        - dong_flux[equation]
                }
                2 => {
                    if self.split_flux {
                        0.5 * normal_velocity * p[l]
                    } else {
                        0.0
                    }
                }
                _ => unreachable!(),
            };
        }
    }

    /// Lane-packed Dong Jacobian action for all lanes.
    ///
    /// Per lane `l` computes exactly the scalar expression with
    /// `&ctxs[l]` (same operations in the same order, including the
    /// two-term velocity-derivative sum).
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - output equation shared by all lanes.
    /// * `q` - facet quadrature-point index shared by all lanes.
    /// * `out` - lane linearized trace-flux slots. Overwritten.
    #[inline]
    fn tensor_jacobian_action(
        &self,
        ctxs: &[TensorFacetCtx<'_>],
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        equation: usize,
        q: usize,
        out: &mut Lanes,
    ) {
        debug_assert_eq!(ctxs.len(), LANES);
        let weak = self.as_weak();
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let du0 = direction.value(0, q);
        let du1 = direction.value(1, q);
        let dp = direction.value(2, q);
        for l in 0..LANES {
            let normal = ctxs[l].normal;
            let velocity = [u0[l], u1[l]];
            let direction_velocity = [du0[l], du1[l]];
            let normal_velocity = normal[0] * velocity[0] + normal[1] * velocity[1];
            let direction_normal_velocity =
                normal[0] * direction_velocity[0] + normal[1] * direction_velocity[1];
            out[l] = match equation {
                0 | 1 => {
                    let split = if self.split_flux {
                        0.5 * (direction_normal_velocity * velocity[equation]
                            + normal_velocity * direction_velocity[equation])
                    } else {
                        0.0
                    };
                    split
                        - dp[l] * normal[equation] / self.rho
                        - (0..2)
                            .map(|unknown| {
                                direction_velocity[unknown]
                                    * weak.dong_flux_derivative(normal, velocity, equation, unknown)
                            })
                            .sum::<f64>()
                }
                2 => {
                    if self.split_flux {
                        0.5 * (direction_normal_velocity * state.value(2, q)[l]
                            + normal_velocity * dp[l])
                    } else {
                        0.0
                    }
                }
                _ => unreachable!(),
            };
        }
    }
}
