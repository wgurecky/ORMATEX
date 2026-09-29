//! Tensor-product fused EDAC Navier-Stokes kernel (`[u, v, p]`).
//!
//! Mathematics: pointwise `(f0, f1x, f1y)` triples for the fused conservative
//! momentum and pressure equations (see
//! [`KernelEdacNavierStokes2D`](crate::kernels::edac::weak::navier_stokes::KernelEdacNavierStokes2D)
//! for the PDE terms), with the directional-derivative action for the Jacobian.
//! Owns all three equations. Weak counterpart:
//! [`KernelEdacNavierStokes2D`](crate::kernels::edac::weak::navier_stokes::KernelEdacNavierStokes2D).
use crate::common::{LaneState, Lanes, TensorCtx, LANES};
use crate::kernels::common::TensorResidualKernel;
use crate::kernels::edac::config::{fluid_field_names, EdacNavierStokes2DConfig};
use crate::kernels::edac::smagorinsky_lilly::SmagorinskyLilly2D;

/// Tensor fused EDAC Navier-Stokes kernel (sum-factorized counterpart, owns all equations).
pub struct TensorKernelEdacNavierStokes2D {
    pub rho: f64,
    pub nu: f64,
    pub c0: f64,
    pub pressure_diffusion_factor: f64,
    pub smagorinsky: SmagorinskyLilly2D,
}

impl TensorKernelEdacNavierStokes2D {
    pub fn new(rho: f64, nu: f64, c0: f64, cs: f64) -> Self {
        let config = EdacNavierStokes2DConfig::new(rho, nu, c0, cs);
        Self {
            rho: config.rho,
            nu: config.nu,
            c0: config.c0,
            pressure_diffusion_factor: config.pressure_diffusion_factor,
            smagorinsky: config.smagorinsky,
        }
    }

    pub fn with_pressure_diffusion_factor(mut self, factor: f64) -> Self {
        assert!(
            factor.is_finite() && factor >= 0.0,
            "pressure diffusion factor must be finite and nonnegative"
        );
        self.pressure_diffusion_factor = factor;
        self
    }

    fn config(&self) -> EdacNavierStokes2DConfig {
        EdacNavierStokes2DConfig {
            rho: self.rho,
            nu: self.nu,
            c0: self.c0,
            pressure_diffusion_factor: self.pressure_diffusion_factor,
            smagorinsky: self.smagorinsky,
        }
    }
}

impl TensorResidualKernel<2> for TensorKernelEdacNavierStokes2D {
    #[inline]
    fn nfields(&self) -> usize {
        3
    }

    #[inline]
    fn field_names(&self) -> Option<Vec<String>> {
        fluid_field_names()
    }

    /// Lane-packed fused residual for all lanes.
    ///
    /// Computes, per lane, exactly the scalar `tensor_residual` expression
    /// with `ctx = &ctxs[l]`.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `equation` - output equation shared by all lanes.
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
        let config = self.config();
        match equation {
            0 | 1 => {
                let (row0, row1) = config.stress_tensor_row_lanes(ctxs, state, q, equation);
                let u0 = state.value(0, q);
                let u1 = state.value(1, q);
                let g0 = state.grad(equation, q, 0);
                let g1 = state.grad(equation, q, 1);
                let gp = state.grad(2, q, equation);
                for l in 0..LANES {
                    f0[l] = u0[l] * g0[l] + u1[l] * g1[l] + gp[l] / self.rho;
                    f1x[l] = row0[l];
                    f1y[l] = row1[l];
                }
            }
            2 => {
                let k = config.pressure_diffusivities_tensor_lanes(ctxs);
                let u0 = state.value(0, q);
                let u1 = state.value(1, q);
                let g00 = state.grad(0, q, 0);
                let g11 = state.grad(1, q, 1);
                let gp0 = state.grad(2, q, 0);
                let gp1 = state.grad(2, q, 1);
                let coeff = self.rho * self.c0 * self.c0;
                for l in 0..LANES {
                    f0[l] = coeff * (g00[l] + g11[l]) + u0[l] * gp0[l] + u1[l] * gp1[l];
                    f1x[l] = k[l] * gp0[l];
                    f1y[l] = k[l] * gp1[l];
                }
            }
            _ => unreachable!(),
        }
    }

    /// Lane-packed fused Jacobian action for all lanes.
    ///
    /// Computes, per lane, exactly the scalar `tensor_jacobian_action`
    /// expression with `ctx = &ctxs[l]`.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `equation` - output equation shared by all lanes.
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
        let config = self.config();
        match equation {
            0 | 1 => {
                let (row0, row1) = config.stress_tensor_row_directional_derivative_lanes(
                    ctxs, state, direction, q, equation,
                );
                let u0 = state.value(0, q);
                let u1 = state.value(1, q);
                let g0 = state.grad(equation, q, 0);
                let g1 = state.grad(equation, q, 1);
                let du0 = direction.value(0, q);
                let du1 = direction.value(1, q);
                let dg0 = direction.grad(equation, q, 0);
                let dg1 = direction.grad(equation, q, 1);
                let dgp = direction.grad(2, q, equation);
                for l in 0..LANES {
                    f0[l] = du0[l] * g0[l]
                        + du1[l] * g1[l]
                        + u0[l] * dg0[l]
                        + u1[l] * dg1[l]
                        + dgp[l] / self.rho;
                    f1x[l] = row0[l];
                    f1y[l] = row1[l];
                }
            }
            2 => {
                let k = config.pressure_diffusivities_tensor_lanes(ctxs);
                let u0 = state.value(0, q);
                let u1 = state.value(1, q);
                let gp0 = state.grad(2, q, 0);
                let gp1 = state.grad(2, q, 1);
                let dg00 = direction.grad(0, q, 0);
                let dg11 = direction.grad(1, q, 1);
                let du0 = direction.value(0, q);
                let du1 = direction.value(1, q);
                let dgp0 = direction.grad(2, q, 0);
                let dgp1 = direction.grad(2, q, 1);
                let coeff = self.rho * self.c0 * self.c0;
                for l in 0..LANES {
                    f0[l] = coeff * (dg00[l] + dg11[l])
                        + du0[l] * gp0[l]
                        + du1[l] * gp1[l]
                        + u0[l] * dgp0[l]
                        + u1[l] * dgp1[l];
                    f1x[l] = k[l] * dgp0[l];
                    f1y[l] = k[l] * dgp1[l];
                }
            }
            _ => unreachable!(),
        }
    }
}
