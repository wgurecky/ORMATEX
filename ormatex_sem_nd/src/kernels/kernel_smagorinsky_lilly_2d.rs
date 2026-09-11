use crate::common::{CellState, LocalCtx, TensorCtx};

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
        let area: f64 = ctx.wts.iter().zip(ctx.jdets).map(|(&w, &j)| w * j).sum();
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

    fn strain_magnitude_from_components((sxx, syy, sxy): (f64, f64, f64)) -> f64 {
        (2.0 * (sxx * sxx + syy * syy + 2.0 * sxy * sxy)).sqrt()
    }

    pub fn strain_magnitude(&self, state: &CellState, q: usize) -> f64 {
        Self::strain_magnitude_from_components(Self::strain_components(state, q))
    }

    pub fn eddy_viscosity(&self, ctx: &LocalCtx, state: &CellState, q: usize) -> f64 {
        let delta = self.filter_width(ctx);
        (self.cs * delta).powi(2) * self.strain_magnitude(state, q)
    }

    /// Evaluate the eddy viscosity using tensor-product cell metadata.
    pub fn eddy_viscosity_tensor(&self, ctx: &TensorCtx, state: &CellState, q: usize) -> f64 {
        (self.cs * self.filter_width_tensor(ctx)).powi(2) * self.strain_magnitude(state, q)
    }

    /// Return the derivative of eddy viscosity in a complete velocity direction.
    pub fn eddy_viscosity_directional_derivative(
        &self,
        ctx: &TensorCtx,
        state: &CellState,
        direction: &CellState,
        q: usize,
    ) -> f64 {
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
        (self.cs * self.filter_width_tensor(ctx)).powi(2) * d_magnitude
    }

    pub(crate) fn filter_width_tensor(&self, ctx: &TensorCtx) -> f64 {
        assert!(
            ctx.cell_size.is_finite() && ctx.cell_size > 0.0,
            "cell area must be positive"
        );
        self.filter_width_scale * ctx.cell_size
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
        (self.cs * self.filter_width(ctx)).powi(2) * dq / (2.0 * magnitude)
    }
}
