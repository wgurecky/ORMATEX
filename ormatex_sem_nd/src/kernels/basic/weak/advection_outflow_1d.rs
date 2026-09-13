use crate::common::{CellState, FacetCtx};
use crate::fields::FieldRegistry;

use crate::kernels::common::StateBoundaryIntegrator;

/// Directional advective outflow for a frozen 1D velocity at one endpoint.
///
/// Mathematics: boundary flux `max(u.n, 0) * c` per transported equation,
/// i.e. the advective outflow flux that conservative volume kernels integrate
/// by parts (zero diffusive flux). Re-entering flow (`u.n < 0`) assumes zero
/// incoming concentration and contributes nothing. Tensor mirror: none (the
/// 1D tensor operator reuses this weak endpoint term).
pub struct KernelAdvectionOutflow1D {
    pub vel: f64,
    field_names: Option<Vec<String>>,
}

impl KernelAdvectionOutflow1D {
    pub fn new(vel: f64) -> Self {
        assert!(vel.is_finite(), "outflow velocity must be finite");
        Self {
            vel,
            field_names: None,
        }
    }

    /// Attach the ordered transported-field names (e.g. `["c0", "c1", "c2"]`).
    pub fn with_field_names<I, S>(vel: f64, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut kernel = Self::new(vel);
        let names = FieldRegistry::new(names);
        assert!(
            !names.is_empty(),
            "outflow field-name count must be positive"
        );
        kernel.field_names = Some(names.names().to_vec());
        kernel
    }

    fn normal_velocity(&self, normal: &[f64]) -> f64 {
        normal[0] * self.vel
    }
}

/// Advective outflow flux with zero backflow concentration.
pub(crate) fn outflow_flux_1d(normal_velocity: f64, c: f64) -> f64 {
    normal_velocity.max(0.0) * c
}

/// Gateaux derivative of [`outflow_flux_1d`] w.r.t. the transported value.
/// The frozen velocity contributes nothing; at `u.n = 0` both sides agree.
pub(crate) fn outflow_flux_action_1d(normal_velocity: f64, dc: f64) -> f64 {
    normal_velocity.max(0.0) * dc
}

impl StateBoundaryIntegrator for KernelAdvectionOutflow1D {
    fn nfields(&self) -> usize {
        self.field_names.as_ref().map_or(3, Vec::len)
    }

    fn field_names(&self) -> Option<Vec<String>> {
        self.field_names.clone()
    }

    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.gdim, 1, "KernelAdvectionOutflow1D: 1D only (gdim==1)");
        assert_eq!(ctx.ncomp, 1, "KernelAdvectionOutflow1D: scalar only (ncomp==1)");
        assert_eq!(state.nfields, self.nfields(), "kernel/state field count mismatch");
        let normal_velocity = self.normal_velocity(ctx.normal);
        outflow_flux_1d(normal_velocity, state.value(equation, q)) * ctx.test(test_i, 0).v(q)
    }

    fn jacobian_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(state.nfields, self.nfields(), "kernel/state field count mismatch");
        if unknown != equation {
            return 0.0;
        }
        let normal_velocity = self.normal_velocity(ctx.normal);
        outflow_flux_action_1d(normal_velocity, ctx.trial(trial_i, 0).v(q))
            * ctx.test(test_i, 0).v(q)
    }
}
