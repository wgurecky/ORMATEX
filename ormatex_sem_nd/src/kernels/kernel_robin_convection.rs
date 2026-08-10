use crate::common::FacetCtx;

use super::kernel_common::BoundaryIntegrator;

/// Robin (convective) boundary: `h (T_amb - T)`.
pub struct RobinConvection {
    pub h: f64,
    pub t_amb: f64,
}

impl RobinConvection {
    pub fn new(h: f64, t_amb: f64) -> Self {
        Self { h, t_amb }
    }
}

impl BoundaryIntegrator for RobinConvection {
    fn integrand_rhs(&self, ctx: &FacetCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "RobinConvection: scalar only (ncomp==1)");
        self.h * self.t_amb * ctx.test(test_i, 0).v(q)
    }

    fn integrand_mat(
        &self,
        ctx: &FacetCtx,
        _equation: usize,
        _unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "RobinConvection: scalar only (ncomp==1)");
        self.h * ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}
