use crate::common::FacetCtx;

use crate::kernels::common::BoundaryIntegrator;

/// Constant prescribed Neumann flux `g` on a facet.
///
/// Facet RHS contribution `g*v`; no matrix term. Tensor mirror: none (boundary
/// integrals stay in weak form).
pub struct NeumannFlux {
    pub g: f64,
}

impl NeumannFlux {
    pub fn new(g: f64) -> Self {
        Self { g }
    }
}

impl BoundaryIntegrator for NeumannFlux {
    fn integrand_rhs(&self, ctx: &FacetCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "NeumannFlux: scalar only (ncomp==1)");
        self.g * ctx.test(test_i, 0).v(q)
    }
}
