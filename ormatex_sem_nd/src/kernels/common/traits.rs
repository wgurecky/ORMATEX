//! Core volume kernel traits.
//!
//! `BilinearForm` / `LinearForm` describe state-independent forms;
//! `ResidualKernel` / `TensorResidualKernel` describe state-dependent
//! residuals for weak and sum-factorized assembly. Boundary traits live
//! in [`boundary_traits`](super::boundary_traits).
use crate::common::{CellState, LocalCtx, TensorCtx};

/// Per-cell bilinear-form kernel.
pub trait BilinearForm {
    /// Number of scalar equation/unknown fields in this form.
    ///
    /// # Returns
    /// Field count; also the row/column block count of [`assemble_local`](Self::assemble_local).
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    ///
    /// # Returns
    /// `None` for anonymous single-field forms, otherwise one name per field
    /// in equation/unknown order.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Bare integrand for one equation/unknown block and test/trial pair.
    ///
    /// Returns the pointwise integrand *without* quadrature weight or Jacobian
    /// determinant; [`assemble_local`](Self::assemble_local) forms
    /// `sum_q wts[q] * jdets[q] * integrand(...)`.
    ///
    /// # Arguments
    /// * `ctx` - cell context with basis values/gradients and quadrature data.
    /// * `equation` - equation (row-block) index in `0..nfields()`.
    /// * `unknown` - unknown (column-block) index in `0..nfields()`.
    /// * `q` - quadrature-point index in `0..ctx.npts`.
    /// * `test_i` - test basis index in `0..ctx.ndofs`.
    /// * `trial_i` - trial basis index in `0..ctx.ndofs`.
    ///
    /// # Returns
    /// Scalar integrand at `q` for the `(test_i, trial_i)` pair.
    fn integrand(
        &self,
        ctx: &LocalCtx,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64;

    /// Whether this form implements tensor-product bilinear evaluation in its
    /// supported geometric dimension.
    ///
    /// # Returns
    /// `true` if [`tensor_bilinear`](Self::tensor_bilinear) is implemented for 2D.
    fn supports_tensor_bilinear(&self) -> bool {
        false
    }

    /// Whether this form implements tensor-product bilinear evaluation on an
    /// interval. This is separate from the 2D capability because a kernel may
    /// support only one geometric dimension.
    ///
    /// # Returns
    /// `true` if [`tensor_bilinear`](Self::tensor_bilinear) is implemented for 1D.
    fn supports_tensor_bilinear_1d(&self) -> bool {
        false
    }

    /// Return the pointwise action of one trial basis function.
    ///
    /// The returned values are `(f0, f1_x, f1_y)` for
    /// `f0 * test + f1 . grad(test)`. In 1D, only `f0` and `f1_x` are used.
    /// `trial_value` and `trial_grad` are the value and physical gradient of
    /// the trial basis at `q`.
    ///
    /// # Arguments
    /// * `ctx` - tensor cell context (geometry, quadrature, differentiation matrix).
    /// * `equation` - equation index in `0..nfields()`.
    /// * `unknown` - unknown index in `0..nfields()`.
    /// * `q` - quadrature-point index in `0..ctx.npts`.
    /// * `trial_value` - trial basis value at `q`.
    /// * `trial_grad` - trial basis physical gradient `[d/dx, d/dy]` at `q`
    ///   (`trial_grad[1]` is ignored in 1D).
    ///
    /// # Returns
    /// `[f0, f1_x, f1_y]` flux triple; the assembler contracts it against the
    /// test basis. No weights or Jacobian determinants included.
    fn tensor_bilinear(
        &self,
        _ctx: &TensorCtx<'_>,
        _equation: usize,
        _unknown: usize,
        _q: usize,
        _trial_value: f64,
        _trial_grad: [f64; 2],
    ) -> [f64; 3] {
        unreachable!("bilinear form does not support tensor-product evaluation")
    }

    /// Assemble a field-major local block matrix.
    ///
    /// # Arguments
    /// * `ctx` - cell context with `ndofs` basis functions and `npts` points.
    /// * `out` - local matrix of length `(nf*ndofs)^2`, row-major with
    ///   `row = equation*n + test_i`, `col = unknown*n + trial_i`. Overwritten.
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
    ///
    /// # Returns
    /// Field count; also the block count of [`assemble_local_rhs`](Self::assemble_local_rhs).
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the form.
    ///
    /// # Returns
    /// `None` for anonymous forms, otherwise one name per equation field.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Bare integrand for one equation and test dof.
    ///
    /// Returns the pointwise integrand *without* weight or Jacobian;
    /// assembly forms `sum_q wts[q] * jdets[q] * integrand(...)`.
    ///
    /// # Arguments
    /// * `ctx` - cell context with basis and quadrature data.
    /// * `equation` - equation index in `0..nfields()`.
    /// * `q` - quadrature-point index in `0..ctx.npts`.
    /// * `test_i` - test basis index in `0..ctx.ndofs`.
    ///
    /// # Returns
    /// Scalar integrand at `q` for test function `test_i`.
    fn integrand(&self, ctx: &LocalCtx, equation: usize, q: usize, test_i: usize) -> f64;

    /// Assemble a field-major local RHS.
    ///
    /// # Arguments
    /// * `ctx` - cell context with `ndofs` basis functions.
    /// * `out` - vector of length `nf*ndofs` with layout
    ///   `out[equation*n + test_i]`. Overwritten.
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
    ///
    /// # Returns
    /// Field count used when input/output counts are not overridden.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the kernel.
    ///
    /// # Returns
    /// `None` for anonymous forms, otherwise one name per field.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Number of compact state fields consumed by this term.
    ///
    /// # Returns
    /// Input field count; must match `state.nfields` at assembly.
    fn input_nfields(&self) -> usize {
        self.nfields()
    }

    /// Number of compact residual fields produced by this term.
    ///
    /// # Returns
    /// Output equation count; sets the row-block count of the local residual/Jacobian.
    fn output_nfields(&self) -> usize {
        self.nfields()
    }

    /// Optional ordered names for compact state fields.
    ///
    /// # Returns
    /// `None` for positional fields, otherwise one name per input field.
    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Optional ordered names for compact residual fields.
    ///
    /// # Returns
    /// `None` for positional fields, otherwise one name per output field.
    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Bare residual integrand for one equation and test dof.
    ///
    /// Returns the pointwise integrand *without* weight or Jacobian;
    /// assembly forms `out[equation*n + test_i] = sum_q wts[q]*jdets[q]*integrand`.
    /// This is the traditional weak form: the full `test`-weighted integrand
    /// (e.g. `nu*grad(u).grad(v)` for diffusion, `-(vel*u).grad(v)` for
    /// advection, `u*v` for mass, `-s*v` for a source moved to the LHS).
    ///
    /// # Arguments
    /// * `ctx` - cell context with test-basis values/gradients.
    /// * `state` - interpolated solution with `input_nfields()` fields.
    /// * `equation` - equation index in `0..output_nfields()`.
    /// * `q` - quadrature-point index in `0..ctx.npts`.
    /// * `test_i` - test basis index in `0..ctx.ndofs`.
    ///
    /// # Returns
    /// Scalar weak-form integrand at `q`.
    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64;

    /// Bare Jacobian integrand for one equation/unknown block and basis pair.
    ///
    /// Gateaux derivative of [`residual_integrand`](Self::residual_integrand)
    /// in the `trial_i` direction of `unknown`; assembly multiplies by
    /// `wts[q]*jdets[q]` and sums over `q`.
    ///
    /// # Arguments
    /// * `ctx` - cell context with test/trial basis data.
    /// * `state` - linearization point with `input_nfields()` fields.
    /// * `equation` - equation index in `0..output_nfields()`.
    /// * `unknown` - unknown index in `0..input_nfields()`.
    /// * `q` - quadrature-point index in `0..ctx.npts`.
    /// * `test_i` - test basis index in `0..ctx.ndofs`.
    /// * `trial_i` - trial basis index in `0..ctx.ndofs`.
    ///
    /// # Returns
    /// Scalar Jacobian integrand at `q`.
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

    /// Assemble a field-major local residual vector.
    ///
    /// # Arguments
    /// * `ctx` - cell context with `ndofs` basis functions.
    /// * `state` - interpolated solution with `input_nfields()` fields.
    /// * `out` - vector of length `output_nfields()*ndofs` with layout
    ///   `out[equation*n + test_i]`. Overwritten.
    fn assemble_local_residual(&self, ctx: &LocalCtx, state: &CellState, out: &mut [f64]) {
        let ni = self.input_nfields();
        let no = self.output_nfields();
        let n = ctx.ndofs;
        assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
        assert_eq!(out.len(), no * n, "local residual size mismatch");
        for equation in 0..no {
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

    /// Assemble a field-major local Jacobian matrix.
    ///
    /// # Arguments
    /// * `ctx` - cell context with `ndofs` basis functions.
    /// * `state` - linearization point with `input_nfields()` fields.
    /// * `out` - matrix of length `(output_nfields()*n) * (input_nfields()*n)`,
    ///   row-major with `row = equation*n + test_i`,
    ///   `col = unknown*n + trial_i`. Overwritten.
    fn assemble_local_jacobian(&self, ctx: &LocalCtx, state: &CellState, out: &mut [f64]) {
        let ni = self.input_nfields();
        let no = self.output_nfields();
        let n = ctx.ndofs;
        let row_size = no * n;
        let col_size = ni * n;
        assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
        assert_eq!(
            out.len(),
            row_size * col_size,
            "local Jacobian size mismatch"
        );
        for equation in 0..no {
            for unknown in 0..ni {
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
                        out[row * col_size + col] = acc;
                    }
                }
            }
        }
    }

    /// Apply the local Jacobian to a coefficient-space direction.
    ///
    /// Computes `out = J(state) * direction` without forming `J`.
    ///
    /// # Arguments
    /// * `ctx` - cell context with `ndofs` basis functions.
    /// * `state` - linearization point with `input_nfields()` fields.
    /// * `direction` - trial coefficients of length `input_nfields()*ndofs`
    ///   with layout `direction[unknown*n + trial_i]`.
    /// * `out` - action of length `output_nfields()*ndofs` with layout
    ///   `out[equation*n + test_i]`. Overwritten.
    fn apply_local_jacobian(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        direction: &[f64],
        out: &mut [f64],
    ) {
        let ni = self.input_nfields();
        let no = self.output_nfields();
        let n = ctx.ndofs;
        let input_size = ni * n;
        let output_size = no * n;
        assert_eq!(state.nfields, ni, "kernel/state field count mismatch");
        assert_eq!(direction.len(), input_size, "local direction size mismatch");
        assert_eq!(
            out.len(),
            output_size,
            "local Jacobian action size mismatch"
        );
        for equation in 0..no {
            for ti in 0..n {
                let mut acc = 0.0;
                for unknown in 0..ni {
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

/// Statically dispatched tensor-product residual kernel.
///
/// # Type parameters
/// * `GDIM` - geometric dimension of the physical space: `1` for the 1D
///   interval, `2` for the 2D quadrilateral. It must match
///   `TensorCtx::geometric_dimension()` and `CellState::gdim`, and it selects
///   the 1D vs 2D sum-factorized assembly path. Only `1` and `2` are supported;
///   `GDIM` is part of the type so tensor assembly never needs a capability
///   query to select its volume path.
pub trait TensorResidualKernel<const GDIM: usize>: Send + Sync {
    /// Number of scalar PDE fields/equations in this kernel.
    ///
    /// # Returns
    /// Field count used when input/output counts are not overridden.
    fn nfields(&self) -> usize {
        1
    }

    /// Optional ordered names for fields whose meaning is part of the kernel.
    ///
    /// # Returns
    /// `None` for anonymous forms, otherwise one name per field.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Number of compact state fields consumed by this term.
    ///
    /// # Returns
    /// Input field count; must match `state.nfields` at assembly.
    fn input_nfields(&self) -> usize {
        self.nfields()
    }

    /// Number of compact residual fields produced by this term.
    ///
    /// # Returns
    /// Output equation count; sets the block count of the assembled residual.
    fn output_nfields(&self) -> usize {
        self.nfields()
    }

    /// Optional ordered names for compact state fields.
    ///
    /// # Returns
    /// `None` for positional fields, otherwise one name per input field.
    fn input_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Optional ordered names for compact residual fields.
    ///
    /// # Returns
    /// `None` for positional fields, otherwise one name per output field.
    fn output_field_names(&self) -> Option<Vec<String>> {
        self.field_names()
    }

    /// Whether this kernel contributes to `equation`.
    ///
    /// Kernels that own only an equation subset (e.g. momentum-only terms of
    /// a coupled system) override this so fused sums can skip zero blocks
    /// without per-point calls. The default keeps existing kernels exact.
    ///
    /// # Arguments
    /// * `equation` - output equation index in `0..output_nfields()`.
    ///
    /// # Returns
    /// `true` if this kernel may return a nonzero triple for `equation`.
    fn owns_equation(&self, _equation: usize) -> bool {
        true
    }

    /// Pointwise weak-form flux triple for one equation at one quadrature point.
    ///
    /// Returns `[f0, f1_x, f1_y]` such that the traditional weak-form
    /// [`ResidualKernel::residual_integrand`] for a test function `v` is
    /// recovered pointwise as:
    ///
    /// ```text
    /// residual_integrand(q, test_i) = f0(q)*v(q) + f1_x(q)*dv/dx(q) + f1_y(q)*dv/dy(q)
    /// ```
    ///
    /// so `f0` is the value (mass/reaction/source) slot and `(f1_x, f1_y)` is
    /// the physical flux vector dotted with `grad(v)` (diffusion, advection,
    /// pressure, ...). The sum-factorized assembler (`tensor_assemble`)
    /// contracts this triple against the test basis: `f0` is scattered with
    /// `wdet`, while `(f1_x, f1_y)` are pulled back with `jinv` and contracted
    /// with the 1D differentiation matrix. The kernel must NOT include
    /// quadrature weights, Jacobian determinants, or basis values; those
    /// belong to the assembler.
    ///
    /// In 1D (`TensorResidualKernel<1>`) only `f0` and `f1_x` are read;
    /// `f1_y` must still be returned (conventionally `0.0`). In 2D all three
    /// slots are used.
    ///
    /// # Arguments
    /// * `ctx` - tensor cell context. Use `ctx.point(q)` or
    ///   `ctx.material_context(Some(state), q)` for coefficients; leave
    ///   `ctx.wdet`/`ctx.jinv`/`ctx.differentiation` to the assembler.
    /// * `state` - interpolated solution at all quadrature points; read the
    ///   current point with `state.value(field, q)` / `state.grad(field, q, d)`.
    /// * `equation` - output equation index in `0..output_nfields()`.
    /// * `q` - quadrature-point index in `0..ctx.npts`.
    ///
    /// # Returns
    /// `[f0, f1_x, f1_y]` physical-space flux triple described above.
    ///
    /// # Example
    /// Diffusion `-div(nu*grad(u))` returns `[0.0, nu*du/dx, nu*du/dy]`;
    /// conservative advection returns `[0.0, -velx*u, -vely*u]`; mass `u`
    /// returns `[u, 0.0, 0.0]`; a constant source `s` moved to the LHS
    /// returns `[-s, 0.0, 0.0]`.
    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3];

    /// Pointwise Gateaux derivative of [`tensor_residual`](Self::tensor_residual).
    ///
    /// Returns `[df0, df1_x, df1_y]` with the same layout and weak-form meaning
    /// as [`tensor_residual`](Self::tensor_residual): the Jacobian action for a
    /// test function `v` is `df0*v + df1_x*dv/dx + df1_y*dv/dy`. Mathematically
    /// this is `d/dε tensor_residual(state + ε*direction)|ε=0` evaluated
    /// pointwise at `q`. Linear terms simply replace `state` with `direction`
    /// (e.g. diffusion `[0.0, nu*ddu/dx, nu*ddu/dy]`); nonlinear or
    /// state-dependent-coefficient terms keep the `state` linearization point
    /// plus material-derivative products (e.g. `dnu*du*grad(state)`).
    ///
    /// The same 1D convention applies: only `df0` and `df1_x` are read by the
    /// 1D assembler, but all three slots must be returned.
    ///
    /// # Arguments
    /// * `ctx` - tensor cell context (same use as in `tensor_residual`).
    /// * `state` - linearization point; read with `state.value(_, q)` /
    ///   `state.grad(_, q, _)`.
    /// * `direction` - Gateaux direction with `input_nfields()` fields; read
    ///   with `direction.value(_, q)` / `direction.grad(_, q, _)`.
    /// * `equation` - output equation index in `0..output_nfields()`.
    /// * `q` - quadrature-point index in `0..ctx.npts`.
    ///
    /// # Returns
    /// `[df0, df1_x, df1_y]` linearized flux triple; contracted by the
    /// assembler exactly like the residual triple.
    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3];

    /// Batched pointwise residual across element lanes (SIMD-over-element hook).
    ///
    /// Default loops over lanes calling the scalar path (correct fallback).
    /// Built-in kernels may override with lane-vectorized physics.
    /// `ctxs/states` have one entry per lane; outputs are per-lane scalars.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane; `ctxs.len()` is the lane count.
    /// * `states` - one interpolated state per lane, aligned with `ctxs`.
    /// * `equation` - output equation index shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0` - per-lane value slots, length `ctxs.len()`. Overwritten.
    /// * `f1x` - per-lane x-flux slots, length `ctxs.len()`. Overwritten.
    /// * `f1y` - per-lane y-flux slots, length `ctxs.len()`. Overwritten
    ///   (unused by the 1D assembler but still written).
    fn tensor_residual_batch(
        &self,
        ctxs: &[TensorCtx<'_>],
        states: &[CellState<'_>],
        equation: usize,
        q: usize,
        f0: &mut [f64],
        f1x: &mut [f64],
        f1y: &mut [f64],
    ) {
        debug_assert_eq!(ctxs.len(), states.len());
        debug_assert_eq!(f0.len(), ctxs.len());
        for (i, (ctx, state)) in ctxs.iter().zip(states.iter()).enumerate() {
            let [a, b, c] = self.tensor_residual(ctx, state, equation, q);
            f0[i] = a;
            f1x[i] = b;
            f1y[i] = c;
        }
    }

    /// Batched pointwise Jacobian action across element lanes.
    ///
    /// Vectorized counterpart of
    /// [`tensor_jacobian_action`](Self::tensor_jacobian_action); the default
    /// loops over lanes calling the scalar path.
    ///
    /// # Arguments
    /// * `ctxs` - one tensor context per lane.
    /// * `states` - one linearization-point state per lane, aligned with `ctxs`.
    /// * `directions` - one Gateaux-direction state per lane, aligned with `ctxs`.
    /// * `equation` - output equation index shared by all lanes.
    /// * `q` - quadrature-point index shared by all lanes.
    /// * `f0` - per-lane linearized value slots, length `ctxs.len()`. Overwritten.
    /// * `f1x` - per-lane linearized x-flux slots, length `ctxs.len()`. Overwritten.
    /// * `f1y` - per-lane linearized y-flux slots, length `ctxs.len()`. Overwritten.
    fn tensor_jacobian_action_batch(
        &self,
        ctxs: &[TensorCtx<'_>],
        states: &[CellState<'_>],
        directions: &[CellState<'_>],
        equation: usize,
        q: usize,
        f0: &mut [f64],
        f1x: &mut [f64],
        f1y: &mut [f64],
    ) {
        debug_assert_eq!(ctxs.len(), states.len());
        debug_assert_eq!(ctxs.len(), directions.len());
        for (i, ((ctx, state), dir)) in ctxs
            .iter()
            .zip(states.iter())
            .zip(directions.iter())
            .enumerate()
        {
            let [a, b, c] = self.tensor_jacobian_action(ctx, state, dir, equation, q);
            f0[i] = a;
            f1x[i] = b;
            f1y[i] = c;
        }
    }
}

/// Pointwise 1D conservation-law flux kernel.
pub trait FluxKernel1D {
    /// Number of scalar equation/unknown fields in this flux.
    ///
    /// # Returns
    /// Field count of the conserved state vector.
    fn nfields(&self) -> usize;

    /// Optional ordered names for fields whose meaning is part of the flux.
    ///
    /// # Returns
    /// `None` for anonymous fluxes, otherwise one name per field.
    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    /// Physical flux for one equation at one quadrature point.
    ///
    /// # Arguments
    /// * `ctx` - cell context (time, cell metadata, physical points).
    /// * `state` - interpolated conserved state at all quadrature points.
    /// * `equation` - equation index in `0..nfields()`.
    /// * `q` - quadrature-point index in `0..ctx.npts`.
    ///
    /// # Returns
    /// Scalar physical flux `F_equation(state(q))`.
    fn flux(&self, ctx: &LocalCtx, state: &CellState, equation: usize, q: usize) -> f64;

    /// Derivative of [`flux`](Self::flux) for one equation/unknown pair.
    ///
    /// # Arguments
    /// * `ctx` - cell context.
    /// * `state` - linearization point.
    /// * `equation` - equation index in `0..nfields()`.
    /// * `unknown` - unknown index in `0..nfields()`.
    /// * `q` - quadrature-point index in `0..ctx.npts`.
    ///
    /// # Returns
    /// Scalar flux Jacobian `dF_equation/dU_unknown` at `q`.
    fn flux_jacobian(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
    ) -> f64;
}
