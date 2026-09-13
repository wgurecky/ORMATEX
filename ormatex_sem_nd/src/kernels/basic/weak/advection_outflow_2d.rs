use crate::common::{CellState, FacetCtx};
use crate::fields::FieldRegistry;
use crate::material::FrozenFacetField;

use crate::kernels::common::StateBoundaryIntegrator;

/// Directional advective outflow for a frozen 2D velocity field.
///
/// Mathematics: boundary flux `max(u.n, 0) * c` per transported equation,
/// i.e. the advective outflow flux that conservative volume kernels integrate
/// by parts (zero diffusive flux). Re-entering flow (`u.n < 0`) assumes zero
/// incoming concentration and contributes nothing. Tensor mirror:
/// [`TensorKernelAdvectionOutflow2D`](crate::kernels::basic::tensor::advection_outflow_2d::TensorKernelAdvectionOutflow2D).
pub struct KernelAdvectionOutflow2D {
    pub ux: FrozenFacetField,
    pub uy: FrozenFacetField,
    field_names: Option<Vec<String>>,
}

impl KernelAdvectionOutflow2D {
    pub fn new(ux: FrozenFacetField, uy: FrozenFacetField) -> Self {
        ux.assert_compatible(uy.facet_count(), uy.npts());
        Self {
            ux,
            uy,
            field_names: None,
        }
    }

    /// Attach the ordered transported-field names (e.g. `["c0", "c1", "c2"]`).
    pub fn with_field_names<I, S>(ux: FrozenFacetField, uy: FrozenFacetField, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut kernel = Self::new(ux, uy);
        let names = FieldRegistry::new(names);
        assert!(
            !names.is_empty(),
            "outflow field-name count must be positive"
        );
        kernel.field_names = Some(names.names().to_vec());
        kernel
    }

    fn normal_velocity(&self, facet: usize, normal: &[f64], q: usize) -> f64 {
        normal[0] * self.ux.value(facet, q) + normal[1] * self.uy.value(facet, q)
    }
}

/// Advective outflow flux with zero backflow concentration.
pub(crate) fn outflow_flux(normal_velocity: f64, c: f64) -> f64 {
    normal_velocity.max(0.0) * c
}

/// Gateaux derivative of [`outflow_flux`] w.r.t. the transported value.
/// The frozen velocity contributes nothing; at `u.n = 0` both sides agree.
pub(crate) fn outflow_flux_action(normal_velocity: f64, dc: f64) -> f64 {
    normal_velocity.max(0.0) * dc
}

impl StateBoundaryIntegrator for KernelAdvectionOutflow2D {
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
        assert_eq!(ctx.gdim, 2, "KernelAdvectionOutflow2D: 2D only (gdim==2)");
        assert_eq!(
            ctx.ncomp, 1,
            "KernelAdvectionOutflow2D: scalar only (ncomp==1)"
        );
        let normal_velocity = self.normal_velocity(ctx.facet.local_index, ctx.normal, q);
        outflow_flux(normal_velocity, state.value(equation, q)) * ctx.test(test_i, 0).v(q)
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
        assert_eq!(
            state.nfields,
            self.nfields(),
            "kernel/state field count mismatch"
        );
        if unknown != equation {
            return 0.0;
        }
        let normal_velocity = self.normal_velocity(ctx.facet.local_index, ctx.normal, q);
        outflow_flux_action(normal_velocity, ctx.trial(trial_i, 0).v(q)) * ctx.test(test_i, 0).v(q)
    }
}
