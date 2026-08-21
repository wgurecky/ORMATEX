//! Shared kernel traits and default local assembly loops.

use crate::common::{CellState, FacetCtx, LocalCtx};
use std::collections::HashMap;
use std::sync::Arc;

/// Per-cell bilinear-form kernel.
pub trait BilinearForm {
    /// Number of scalar equation/unknown fields in this form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Bare integrand for one equation/unknown block and test/trial pair.
    fn integrand(
        &self,
        ctx: &LocalCtx,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64;

    /// Assemble a field-major local block matrix.
    fn assemble_local(&self, ctx: &LocalCtx, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert!(nf > 0, "bilinear form must contain at least one field");
        assert_eq!(
            out.len(),
            local_size * local_size,
            "local matrix size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jdets[q]
                                * self.integrand(ctx, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        out[row * local_size + col] = acc;
                    }
                }
            }
        }
    }
}

/// Per-cell linear-form kernel (RHS contribution).
pub trait LinearForm {
    /// Number of scalar equation fields in this form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Bare integrand for one equation and test dof.
    fn integrand(&self, ctx: &LocalCtx, equation: usize, q: usize, test_i: usize) -> f64;

    /// Assemble a field-major local RHS.
    fn assemble_local_rhs(&self, ctx: &LocalCtx, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert!(nf > 0, "linear form must contain at least one field");
        assert_eq!(out.len(), nf * n, "local RHS size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q] * ctx.jdets[q] * self.integrand(ctx, equation, q, ti);
                }
                out[equation * n + ti] = acc;
            }
        }
    }
}

/// State-aware cell kernel for residual and Jacobian assembly.
pub trait ResidualKernel {
    /// Number of scalar PDE fields/equations in this kernel.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the kernel.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64;

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64;

    fn assemble_local_residual(&self, ctx: &LocalCtx, state: &CellState, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert_eq!(state.nfields, nf, "kernel/state field count mismatch");
        assert_eq!(out.len(), nf * n, "local residual size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q]
                        * ctx.jdets[q]
                        * self.residual_integrand(ctx, state, equation, q, ti);
                }
                out[equation * n + ti] = acc;
            }
        }
    }

    fn assemble_local_jacobian(&self, ctx: &LocalCtx, state: &CellState, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(state.nfields, nf, "kernel/state field count mismatch");
        assert_eq!(
            out.len(),
            local_size * local_size,
            "local Jacobian size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jdets[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        out[row * local_size + col] = acc;
                    }
                }
            }
        }
    }

    fn apply_local_jacobian(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        direction: &[f64],
        out: &mut [f64],
    ) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(state.nfields, nf, "kernel/state field count mismatch");
        assert_eq!(direction.len(), local_size, "local direction size mismatch");
        assert_eq!(out.len(), local_size, "local Jacobian action size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for unknown in 0..nf {
                    for si in 0..n {
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jdets[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si)
                                * direction[unknown * n + si];
                        }
                    }
                }
                out[equation * n + ti] = acc;
            }
        }
    }
}

/// Additive composition of state-aware cell kernels.
///
/// Each child must describe the same ordered system fields. The parent SEM
/// assembly still interpolates the state and scatters the local result once;
/// only the pointwise residual or Jacobian integrands are summed here.
pub struct ResidualKernelSum<'a> {
    kernels: Vec<Box<dyn ResidualKernel + Send + Sync + 'a>>,
    nfields: usize,
    field_names: Option<Vec<String>>,
}

impl<'a> ResidualKernelSum<'a> {
    /// Build a sum from heterogeneous residual kernels.
    pub fn new(kernels: Vec<Box<dyn ResidualKernel + Send + Sync + 'a>>) -> Self {
        let mut kernels = kernels.into_iter();
        let first = kernels
            .next()
            .expect("residual kernel sum must contain at least one kernel");
        let nfields = first.nfields();
        assert!(
            nfields > 0,
            "residual kernel sum must contain at least one field"
        );
        let mut field_names = first.field_names();
        Self::validate_field_names(&field_names, nfields);

        let mut sum = Self {
            kernels: vec![first],
            nfields,
            field_names: field_names.take(),
        };
        for kernel in kernels {
            sum.push(kernel);
        }
        sum
    }

    /// Start a sum with one concrete kernel.
    pub fn from_kernel<K>(kernel: K) -> Self
    where
        K: ResidualKernel + Send + Sync + 'a,
    {
        let kernel: Box<dyn ResidualKernel + Send + Sync + 'a> = Box::new(kernel);
        Self::new(vec![kernel])
    }

    /// Add one concrete kernel to this sum.
    pub fn with<K>(mut self, kernel: K) -> Self
    where
        K: ResidualKernel + Send + Sync + 'a,
    {
        let kernel: Box<dyn ResidualKernel + Send + Sync + 'a> = Box::new(kernel);
        self.push(kernel);
        self
    }

    fn push(&mut self, kernel: Box<dyn ResidualKernel + Send + Sync + 'a>) {
        assert_eq!(
            kernel.nfields(),
            self.nfields,
            "residual kernel sum field count mismatch"
        );
        let names = kernel.field_names();
        Self::validate_field_names(&names, self.nfields);
        match (&mut self.field_names, names) {
            (Some(expected), Some(actual)) => assert_eq!(
                *expected, actual,
                "residual kernel sum field names/order mismatch"
            ),
            (None, Some(actual)) => self.field_names = Some(actual),
            _ => {}
        }
        self.kernels.push(kernel);
    }

    fn validate_field_names(names: &Option<Vec<String>>, nfields: usize) {
        if let Some(names) = names {
            assert_eq!(
                names.len(),
                nfields,
                "residual kernel field-name count does not match field count"
            );
        }
    }
}

impl ResidualKernel for ResidualKernelSum<'_> {
    fn nfields(&self) -> usize {
        self.nfields
    }

    fn field_names(&self) -> Option<Vec<String>> {
        self.field_names.clone()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        self.kernels
            .iter()
            .map(|kernel| kernel.residual_integrand(ctx, state, equation, q, test_i))
            .sum()
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        self.kernels
            .iter()
            .map(|kernel| {
                kernel.jacobian_integrand(ctx, state, equation, unknown, q, test_i, trial_i)
            })
            .sum()
    }
}

/// Pointwise flux provider for one-dimensional conservation laws.
pub trait FluxKernel1D {
    fn nfields(&self) -> usize;

    /// Optional ordered names for fields whose meaning is part of the flux.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn flux(&self, ctx: &LocalCtx, state: &CellState, equation: usize, q: usize) -> f64;

    fn flux_jacobian(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
    ) -> f64;
}

/// Boundary integrator trait for Neumann and Robin forms.
pub trait BoundaryIntegrator {
    /// Number of scalar equation/unknown fields in this boundary form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the boundary form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn integrand_rhs(&self, ctx: &FacetCtx, equation: usize, q: usize, test_i: usize) -> f64;

    fn integrand_mat(
        &self,
        _ctx: &FacetCtx,
        _equation: usize,
        _unknown: usize,
        _q: usize,
        _test_i: usize,
        _trial_i: usize,
    ) -> f64 {
        0.0
    }

    fn assemble_facet_rhs(&self, ctx: &FacetCtx, rhs: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert!(
            nf > 0,
            "boundary integrator must contain at least one field"
        );
        assert_eq!(rhs.len(), nf * n, "local boundary RHS size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc +=
                        ctx.wts[q] * ctx.jfacet_det[q] * self.integrand_rhs(ctx, equation, q, ti);
                }
                rhs[equation * n + ti] = acc;
            }
        }
    }

    fn assemble_facet_mat(&self, ctx: &FacetCtx, mat: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(
            mat.len(),
            local_size * local_size,
            "local boundary matrix size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jfacet_det[q]
                                * self.integrand_mat(ctx, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        mat[row * local_size + col] = acc;
                    }
                }
            }
        }
    }
}

/// State-aware nonlinear boundary form.
///
/// Unlike [`BoundaryIntegrator`], this form receives the interpolated state at
/// facet quadrature points and can therefore represent nonlinear conditions
/// such as backflow-stabilizing outflow traction.
pub trait StateBoundaryIntegrator: Send + Sync {
    /// Number of scalar equation/unknown fields in this boundary form.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn residual_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64;

    fn jacobian_integrand(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64;

    fn apply_local_jacobian(
        &self,
        ctx: &FacetCtx,
        state: &CellState,
        direction: &[f64],
        out: &mut [f64],
    ) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(
            direction.len(),
            local_size,
            "boundary direction size mismatch"
        );
        assert_eq!(out.len(), local_size, "boundary action size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for unknown in 0..nf {
                    for si in 0..n {
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jfacet_det[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si)
                                * direction[unknown * n + si];
                        }
                    }
                }
                out[equation * n + ti] = acc;
            }
        }
    }

    fn assemble_local_residual(&self, ctx: &FacetCtx, state: &CellState, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        assert!(
            nf > 0,
            "state boundary form must contain at least one field"
        );
        assert_eq!(state.nfields, nf, "boundary/state field count mismatch");
        assert_eq!(out.len(), nf * n, "local boundary residual size mismatch");
        for equation in 0..nf {
            for ti in 0..n {
                let mut acc = 0.0;
                for q in 0..ctx.npts {
                    acc += ctx.wts[q]
                        * ctx.jfacet_det[q]
                        * self.residual_integrand(ctx, state, equation, q, ti);
                }
                out[equation * n + ti] = acc;
            }
        }
    }

    fn assemble_local_jacobian(&self, ctx: &FacetCtx, state: &CellState, out: &mut [f64]) {
        let nf = self.nfields();
        let n = ctx.ndofs;
        let local_size = nf * n;
        assert_eq!(state.nfields, nf, "boundary/state field count mismatch");
        assert_eq!(
            out.len(),
            local_size * local_size,
            "local boundary Jacobian size mismatch"
        );
        for equation in 0..nf {
            for unknown in 0..nf {
                for ti in 0..n {
                    for si in 0..n {
                        let mut acc = 0.0;
                        for q in 0..ctx.npts {
                            acc += ctx.wts[q]
                                * ctx.jfacet_det[q]
                                * self.jacobian_integrand(ctx, state, equation, unknown, q, ti, si);
                        }
                        let row = equation * n + ti;
                        let col = unknown * n + si;
                        out[row * local_size + col] = acc;
                    }
                }
            }
        }
    }
}

/// Immutable state-dependent boundary terms selected by mesh entity index.
#[derive(Clone, Default)]
pub struct StateBoundaryTerms {
    default: Option<Arc<dyn StateBoundaryIntegrator>>,
    overrides: HashMap<usize, Arc<dyn StateBoundaryIntegrator>>,
}

impl StateBoundaryTerms {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default<K>(mut self, kernel: K) -> Self
    where
        K: StateBoundaryIntegrator + 'static,
    {
        self.default = Some(Arc::new(kernel));
        self
    }

    pub fn with_entities<K, I>(mut self, entities: I, kernel: K) -> Self
    where
        K: StateBoundaryIntegrator + 'static,
        I: IntoIterator<Item = usize>,
    {
        let kernel: Arc<dyn StateBoundaryIntegrator> = Arc::new(kernel);
        for entity in entities {
            assert!(
                self.overrides.insert(entity, Arc::clone(&kernel)).is_none(),
                "state boundary entity configured more than once"
            );
        }
        self
    }

    pub(crate) fn kernel_for(&self, entity: usize) -> Option<&dyn StateBoundaryIntegrator> {
        self.overrides
            .get(&entity)
            .or(self.default.as_ref())
            .map(AsRef::as_ref)
    }
}
