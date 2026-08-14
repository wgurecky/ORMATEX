use crate::common::{CellState, LocalCtx};

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

    pub fn filter_width(&self, ctx: &LocalCtx) -> f64 {
        assert_eq!(ctx.gdim, 2, "SmagorinskyLilly2D requires a 2D context");
        let area: f64 = ctx
            .wts
            .iter()
            .zip(ctx.jdets)
            .map(|(&weight, &jdet)| weight * jdet)
            .sum();
        assert!(area.is_finite() && area > 0.0, "cell area must be positive");
        self.filter_width_scale * area.sqrt()
    }

    fn strain_components(state: &CellState, q: usize) -> (f64, f64, f64) {
        (
            state.grad(0, q, 0),
            state.grad(1, q, 1),
            0.5 * (state.grad(0, q, 1) + state.grad(1, q, 0)),
        )
    }

    pub fn strain_magnitude(&self, state: &CellState, q: usize) -> f64 {
        let (sxx, syy, sxy) = Self::strain_components(state, q);
        (2.0 * (sxx * sxx + syy * syy + 2.0 * sxy * sxy)).sqrt()
    }

    pub fn eddy_viscosity(&self, ctx: &LocalCtx, state: &CellState, q: usize) -> f64 {
        let delta = self.filter_width(ctx);
        (self.cs * delta).powi(2) * self.strain_magnitude(state, q)
    }

    /// Derivative of the eddy viscosity with respect to one velocity gradient.
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
        let magnitude = self.strain_magnitude(state, q);
        if magnitude <= f64::EPSILON {
            return 0.0;
        }
        let (sxx, syy, sxy) = Self::strain_components(state, q);
        let shear_sum = 2.0 * sxy;
        let dq = match (velocity_field, direction) {
            (0, 0) => 4.0 * sxx,
            (0, 1) => 2.0 * shear_sum,
            (1, 0) => 2.0 * shear_sum,
            (1, 1) => 4.0 * syy,
            _ => unreachable!(),
        };
        (self.cs * self.filter_width(ctx)).powi(2) * dq / (2.0 * magnitude)
    }
}
