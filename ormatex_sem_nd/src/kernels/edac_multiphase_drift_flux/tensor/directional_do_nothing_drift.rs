//! Tensor directional do-nothing outflow for drift-flux EDAC (`[u, v, p, alpha]`).
//!
//! Mathematics: borrows the base EDAC outflow — `-p n / rho_l` traction plus
//! the `max(-u.n, 0)` backflow penalty — with the reference liquid density so
//! the `alpha -> 0` limit matches the single-phase kernel exactly; with
//! `split_flux` it also supplies the conservative-half fluxes for split
//! momentum/pressure advection. The void equation gets the passive-scalar
//! split outflow `1/2 (u.n) alpha` (zero when `split_flux` is off).
use crate::common::{LaneState, Lanes, TensorFacetCtx, LANES};
use crate::kernels::common::StateTensorBoundaryIntegrator;
use crate::kernels::edac::weak::directional_do_nothing::{
    directional_jacobian_action, directional_residual,
};
use crate::kernels::edac_multiphase_drift_flux::config::{drift_field_names, ALPHA_2D};

/// Tensor directional do-nothing outflow for drift-flux EDAC.
pub struct TensorDriftDirectionalDoNothing2D {
    pub rho_l: f64,
    pub split_flux: bool,
}

impl TensorDriftDirectionalDoNothing2D {
    pub fn new(rho_l: f64) -> Self {
        assert!(
            rho_l.is_finite() && rho_l > 0.0,
            "reference density must be finite and positive"
        );
        Self {
            rho_l,
            split_flux: false,
        }
    }

    pub fn with_split_flux(mut self) -> Self {
        self.split_flux = true;
        self
    }
}

impl StateTensorBoundaryIntegrator<2> for TensorDriftDirectionalDoNothing2D {
    fn nfields(&self) -> usize {
        4
    }

    fn field_names(&self) -> Option<Vec<String>> {
        drift_field_names()
    }

    /// Lane-packed directional do-nothing residual for all lanes.
    ///
    /// Each lane evaluates exactly the scalar expression with its own facet
    /// context and state lane, preserving the `split_flux` branches.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed facet solution.
    /// * `equation` - output equation index shared by all lanes.
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
        if equation == ALPHA_2D {
            if !self.split_flux {
                *out = [0.0; LANES];
                return;
            }
            let u0 = state.value(0, q);
            let u1 = state.value(1, q);
            let a = state.value(ALPHA_2D, q);
            for l in 0..LANES {
                let normal_velocity = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                out[l] = 0.5 * normal_velocity * a[l];
            }
            return;
        }
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let p = state.value(2, q);
        for l in 0..LANES {
            out[l] = directional_residual(
                ctxs[l].normal,
                [u0[l], u1[l]],
                p[l],
                self.rho_l,
                self.split_flux,
                equation,
            );
        }
    }

    /// Lane-packed directional do-nothing Jacobian action for all lanes.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor facet context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - output equation index shared by all lanes.
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
        if equation == ALPHA_2D {
            if !self.split_flux {
                *out = [0.0; LANES];
                return;
            }
            let u0 = state.value(0, q);
            let u1 = state.value(1, q);
            let a = state.value(ALPHA_2D, q);
            let du0 = direction.value(0, q);
            let du1 = direction.value(1, q);
            let da = direction.value(ALPHA_2D, q);
            for l in 0..LANES {
                let normal_velocity = ctxs[l].normal[0] * u0[l] + ctxs[l].normal[1] * u1[l];
                let direction_normal_velocity =
                    ctxs[l].normal[0] * du0[l] + ctxs[l].normal[1] * du1[l];
                out[l] = 0.5 * (direction_normal_velocity * a[l] + normal_velocity * da[l]);
            }
            return;
        }
        let u0 = state.value(0, q);
        let u1 = state.value(1, q);
        let p = state.value(2, q);
        let du0 = direction.value(0, q);
        let du1 = direction.value(1, q);
        let dp = direction.value(2, q);
        for l in 0..LANES {
            out[l] = directional_jacobian_action(
                ctxs[l].normal,
                [u0[l], u1[l]],
                [du0[l], du1[l]],
                p[l],
                dp[l],
                self.rho_l,
                self.split_flux,
                equation,
            );
        }
    }
}
