use crate::common::{CellState, LaneState, Lanes, LocalCtx, TensorCtx, LANES};

/// Smagorinsky-Lilly eddy viscosity for two scalar velocity fields.
///
/// The filter width is the square root of the physical cell area. The model
/// returns kinematic viscosity, so density is applied by the fluid kernel.
#[derive(Clone, Copy, Debug)]
pub struct SmagorinskyLilly2D {
    pub cs: f64,
    pub filter_width_scale: f64,
}

impl SmagorinskyLilly2D {
    pub fn new(cs: f64) -> Self {
        assert!(
            cs.is_finite() && cs >= 0.0,
            "Smagorinsky constant must be finite and nonnegative"
        );
        Self {
            cs,
            filter_width_scale: 1.0,
        }
    }

    pub fn with_filter_width_scale(mut self, scale: f64) -> Self {
        assert!(
            scale.is_finite() && scale > 0.0,
            "filter-width scale must be finite and positive"
        );
        self.filter_width_scale = scale;
        self
    }

    #[inline]
    pub fn filter_width(&self, ctx: &LocalCtx) -> f64 {
        assert_eq!(ctx.gdim, 2, "SmagorinskyLilly2D requires a 2D context");
        let area: f64 = ctx.wts.iter().zip(ctx.jdets).map(|(&w, &j)| w * j).sum();
        assert!(area.is_finite() && area > 0.0, "cell area must be positive");
        self.filter_width_scale * area.sqrt()
    }

    #[inline]
    fn strain_components(state: &CellState, q: usize) -> (f64, f64, f64) {
        (
            state.grad(0, q, 0),
            state.grad(1, q, 1),
            0.5 * (state.grad(0, q, 1) + state.grad(1, q, 0)),
        )
    }

    #[inline]
    fn strain_magnitude_from_components((sxx, syy, sxy): (f64, f64, f64)) -> f64 {
        (2.0 * (sxx * sxx + syy * syy + 2.0 * sxy * sxy)).sqrt()
    }

    #[inline]
    pub fn strain_magnitude(&self, state: &CellState, q: usize) -> f64 {
        Self::strain_magnitude_from_components(Self::strain_components(state, q))
    }

    #[inline]
    pub fn eddy_viscosity(&self, ctx: &LocalCtx, state: &CellState, q: usize) -> f64 {
        // ponytail: factor first; cs == 0 (or degenerate cell) skips the strain sqrt.
        let factor = (self.cs * self.filter_width(ctx)).powi(2);
        if factor == 0.0 {
            return 0.0;
        }
        factor * self.strain_magnitude(state, q)
    }

    /// Evaluate the eddy viscosity using tensor-product cell metadata.
    #[inline]
    pub fn eddy_viscosity_tensor(&self, ctx: &TensorCtx, state: &CellState, q: usize) -> f64 {
        // ponytail: factor first; cs == 0 skips the strain sqrt.
        let factor = (self.cs * self.filter_width_tensor(ctx)).powi(2);
        if factor == 0.0 {
            return 0.0;
        }
        factor * self.strain_magnitude(state, q)
    }

    /// Return the derivative of eddy viscosity in a complete velocity direction.
    #[inline]
    pub fn eddy_viscosity_directional_derivative(
        &self,
        ctx: &TensorCtx,
        state: &CellState,
        direction: &CellState,
        q: usize,
    ) -> f64 {
        // ponytail: factor first; cs == 0 skips strain, sqrt, and division.
        let factor = (self.cs * self.filter_width_tensor(ctx)).powi(2);
        if factor == 0.0 {
            return 0.0;
        }
        let components = Self::strain_components(state, q);
        let magnitude = Self::strain_magnitude_from_components(components);
        if magnitude <= f64::EPSILON {
            return 0.0;
        }
        let (sxx, syy, sxy) = components;
        let dsxx = direction.grad(0, q, 0);
        let dsyy = direction.grad(1, q, 1);
        let dsxy = 0.5 * (direction.grad(0, q, 1) + direction.grad(1, q, 0));
        let d_magnitude =
            (4.0 * sxx * dsxx + 4.0 * syy * dsyy + 8.0 * sxy * dsxy) / (2.0 * magnitude);
        factor * d_magnitude
    }

    /// Return the eddy viscosity and its directional derivative in one pass.
    ///
    /// Fused counterpart of [`eddy_viscosity_tensor`](Self::eddy_viscosity_tensor)
    /// and [`eddy_viscosity_directional_derivative`](Self::eddy_viscosity_directional_derivative):
    /// computes the Smagorinsky factor, strain components, and strain magnitude
    /// once and returns exactly the same two values the separate calls produce.
    /// `nu_t` is `factor * magnitude` even when `magnitude <= EPSILON` (the
    /// derivative early-out still yields `dnu_t == 0.0` there); a zero factor
    /// returns `(0.0, 0.0)` without touching the strain field.
    ///
    /// # Arguments
    /// * `ctx` - tensor cell context supplying the filter width.
    /// * `state` - linearization-point velocity state at all quadrature points.
    /// * `direction` - Gateaux velocity direction at all quadrature points.
    /// * `q` - quadrature-point index.
    ///
    /// # Returns
    /// `(nu_t, dnu_t)` pair with the same values and operation order as the
    /// two unfused methods.
    #[inline]
    pub fn eddy_viscosity_and_directional_derivative(
        &self,
        ctx: &TensorCtx,
        state: &CellState,
        direction: &CellState,
        q: usize,
    ) -> (f64, f64) {
        // ponytail: factor first; cs == 0 skips strain, sqrt, and division.
        let factor = (self.cs * self.filter_width_tensor(ctx)).powi(2);
        if factor == 0.0 {
            return (0.0, 0.0);
        }
        let components = Self::strain_components(state, q);
        let magnitude = Self::strain_magnitude_from_components(components);
        let nu_t = factor * magnitude;
        if magnitude <= f64::EPSILON {
            return (nu_t, 0.0);
        }
        let (sxx, syy, sxy) = components;
        let dsxx = direction.grad(0, q, 0);
        let dsyy = direction.grad(1, q, 1);
        let dsxy = 0.5 * (direction.grad(0, q, 1) + direction.grad(1, q, 0));
        let d_magnitude =
            (4.0 * sxx * dsxx + 4.0 * syy * dsyy + 8.0 * sxy * dsxy) / (2.0 * magnitude);
        (nu_t, factor * d_magnitude)
    }

    #[inline]
    pub(crate) fn filter_width_tensor(&self, ctx: &TensorCtx) -> f64 {
        assert!(
            ctx.cell_size.is_finite() && ctx.cell_size > 0.0,
            "cell area must be positive"
        );
        self.filter_width_scale * ctx.cell_size
    }

    /// Lane-packed filter widths for [`LANES`] cells.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    ///
    /// # Returns
    /// `filter_width_scale * cell_size[l]` per lane, asserting each
    /// `cell_size` is finite and positive exactly like the scalar path.
    #[inline]
    pub(crate) fn filter_widths_tensor_lanes(&self, ctxs: &[TensorCtx<'_>]) -> Lanes {
        debug_assert_eq!(ctxs.len(), LANES);
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            assert!(
                ctxs[l].cell_size.is_finite() && ctxs[l].cell_size > 0.0,
                "cell area must be positive"
            );
            out[l] = self.filter_width_scale * ctxs[l].cell_size;
        }
        out
    }

    /// Lane-packed Smagorinsky factors `(cs * width[l]).powi(2)`.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    ///
    /// # Returns
    /// Per-lane factor with the same operations and order as the scalar path.
    #[inline]
    pub(crate) fn factors_tensor_lanes(&self, ctxs: &[TensorCtx<'_>]) -> Lanes {
        let widths = self.filter_widths_tensor_lanes(ctxs);
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = (self.cs * widths[l]).powi(2);
        }
        out
    }

    /// Lane-packed strain magnitudes `|S|[l]` at quadrature point `q`.
    ///
    /// # Arguments
    /// * `state` - lane-packed solution.
    /// * `q` - quadrature-point index shared by all lanes.
    ///
    /// # Returns
    /// Per-lane `(2*(sxx^2+syy^2+2*sxy^2)).sqrt()` with the same operations
    /// and order as the scalar path.
    #[inline]
    pub(crate) fn strain_magnitudes_lanes(state: &LaneState<'_>, q: usize) -> Lanes {
        let sxx = state.grad(0, q, 0);
        let syy = state.grad(1, q, 1);
        let gx1 = state.grad(0, q, 1);
        let gy0 = state.grad(1, q, 0);
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            let sxy = 0.5 * (gx1[l] + gy0[l]);
            out[l] = (2.0 * (sxx[l] * sxx[l] + syy[l] * syy[l] + 2.0 * sxy * sxy)).sqrt();
        }
        out
    }

    /// Lane-packed eddy viscosities matching [`eddy_viscosity_tensor`](Self::eddy_viscosity_tensor).
    ///
    /// Per lane `nu_t[l] = 0` when `factor[l] == 0`, else `factor[l]*mag[l]`.
    /// A uniform fast path skips the strain math when every factor is zero.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed solution.
    /// * `q` - quadrature-point index shared by all lanes.
    ///
    /// # Returns
    /// Per-lane eddy viscosity, bit-identical to the scalar path lane-by-lane.
    #[inline]
    pub(crate) fn eddy_viscosities_tensor_lanes(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        q: usize,
    ) -> Lanes {
        let factors = self.factors_tensor_lanes(ctxs);
        let mut all_zero = true;
        for l in 0..LANES {
            if factors[l] != 0.0 {
                all_zero = false;
                break;
            }
        }
        if all_zero {
            return [0.0; LANES];
        }
        let mags = Self::strain_magnitudes_lanes(state, q);
        let mut out = [0.0; LANES];
        for l in 0..LANES {
            out[l] = if factors[l] == 0.0 {
                0.0
            } else {
                factors[l] * mags[l]
            };
        }
        out
    }

    /// Lane-packed `(nu_t, dnu_t)` matching [`eddy_viscosity_and_directional_derivative`](Self::eddy_viscosity_and_directional_derivative).
    ///
    /// Per lane `nu_t[l] = 0` when `factor[l] == 0` else `factor[l]*mag[l]`,
    /// and `dnu[l] = 0` when `factor[l] == 0` or `mag[l] <= EPSILON`, else
    /// `factor[l]*d_mag[l]` with
    /// `d_mag = (4*sxx*dsxx+4*syy*dsyy+8*sxy*dsxy)/(2*mag)` in scalar order.
    /// A uniform fast path returns zeros when every factor is zero.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane, length [`LANES`].
    /// * `state` - lane-packed linearization point.
    /// * `direction` - lane-packed Gateaux direction.
    /// * `q` - quadrature-point index shared by all lanes.
    ///
    /// # Returns
    /// `(nu_t, dnu_t)` lane vectors, bit-identical to the scalar path lane-by-lane.
    #[inline]
    pub(crate) fn eddy_viscosity_and_derivative_lanes(
        &self,
        ctxs: &[TensorCtx<'_>],
        state: &LaneState<'_>,
        direction: &LaneState<'_>,
        q: usize,
    ) -> (Lanes, Lanes) {
        let factors = self.factors_tensor_lanes(ctxs);
        let mut all_zero = true;
        for l in 0..LANES {
            if factors[l] != 0.0 {
                all_zero = false;
                break;
            }
        }
        if all_zero {
            return ([0.0; LANES], [0.0; LANES]);
        }
        let sxx = state.grad(0, q, 0);
        let syy = state.grad(1, q, 1);
        let gx1 = state.grad(0, q, 1);
        let gy0 = state.grad(1, q, 0);
        let dsxx = direction.grad(0, q, 0);
        let dsyy = direction.grad(1, q, 1);
        let dgx1 = direction.grad(0, q, 1);
        let dgy0 = direction.grad(1, q, 0);
        let mut nu = [0.0; LANES];
        let mut dnu = [0.0; LANES];
        for l in 0..LANES {
            if factors[l] == 0.0 {
                nu[l] = 0.0;
                dnu[l] = 0.0;
                continue;
            }
            let sxy = 0.5 * (gx1[l] + gy0[l]);
            let magnitude = (2.0 * (sxx[l] * sxx[l] + syy[l] * syy[l] + 2.0 * sxy * sxy)).sqrt();
            let nu_t = factors[l] * magnitude;
            nu[l] = nu_t;
            if magnitude <= f64::EPSILON {
                dnu[l] = 0.0;
            } else {
                let dsxy = 0.5 * (dgx1[l] + dgy0[l]);
                let d_magnitude =
                    (4.0 * sxx[l] * dsxx[l] + 4.0 * syy[l] * dsyy[l] + 8.0 * sxy * dsxy)
                        / (2.0 * magnitude);
                dnu[l] = factors[l] * d_magnitude;
            }
        }
        (nu, dnu)
    }

    /// Derivative of the eddy viscosity with respect to one velocity gradient.
    #[inline]
    pub fn eddy_viscosity_gradient_derivative(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        q: usize,
        velocity_field: usize,
        direction: usize,
    ) -> f64 {
        assert!(velocity_field < 2, "velocity field must be 0 or 1");
        assert!(direction < 2, "gradient direction must be 0 or 1");
        // ponytail: factor first; cs == 0 skips strain, sqrt, and division.
        let factor = (self.cs * self.filter_width(ctx)).powi(2);
        if factor == 0.0 {
            return 0.0;
        }
        let components = Self::strain_components(state, q);
        let magnitude = Self::strain_magnitude_from_components(components);
        if magnitude <= f64::EPSILON {
            return 0.0;
        }
        let (sxx, syy, sxy) = components;
        let shear_sum = 2.0 * sxy;
        let dq = match (velocity_field, direction) {
            (0, 0) => 4.0 * sxx,
            (0, 1) => 2.0 * shear_sum,
            (1, 0) => 2.0 * shear_sum,
            (1, 1) => 4.0 * syy,
            _ => unreachable!(),
        };
        factor * dq / (2.0 * magnitude)
    }
}
