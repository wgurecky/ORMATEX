//! Weak / tensor / mixed residual operators for 2D problems.
use crate::jacobian::{CompleteResidualOperator, RowEpilogue};
use crate::kernels::common::{
    ResidualKernel, StateBoundaryTerms, StateTensorBoundaryTerms, TensorResidualKernel,
};
use faer::prelude::*;
use faer::sparse::SparseColMat;

use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;

use super::problem::SEM2DProblem;
use crate::common::tensor_pass::{reduce_sorted_evec_into_out, TensorPhase1Ctx};
use crate::common::batch::tensor_state_cache_limit_bytes;
use crate::common::TensorStateCache;
use crate::sem_traits::WeakResidualOps;
/// Weak-form residual operator coupling a problem, a `ResidualKernel`, and
/// optional state-dependent facet terms (see [`with_state_boundary`](SEM2DResidualOperator::with_state_boundary)).
pub struct SEM2DResidualOperator<'p, 'k, 'b, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    problem: &'p SEM2DProblem<M>,
    kernel: &'k K,
    time: f64,
    terms: Option<&'b StateBoundaryTerms>,
}

/// Statically dispatched tensor-product residual operator.
pub struct SEM2DTensorResidualOperator<'p, 'k, 'b, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    problem: &'p SEM2DProblem<M>,
    kernel: &'k K,
    time: f64,
    terms: Option<&'b StateTensorBoundaryTerms<2>>,
    state_cache: Option<TensorStateCache>,
}

impl<'p, 'k, 'b, M, K> SEM2DTensorResidualOperator<'p, 'k, 'b, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    K: TensorResidualKernel<2> + Sync,
{
    /// Retained (reduced) system size, including all fields.
    pub fn system_size(&self) -> usize {
        self.problem.system_size()
    }
    /// Residual at the operator's time, including boundary terms when set.
    ///
    /// Bit-identity: the boundary contribution may run on the serial or the
    /// color-parallel path depending on facet count and thread pool; the two
    /// paths accumulate in different orders, so bits match only for the same
    /// path and thread count (see the boundary driver note).
    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        let layout = self.problem.field_layout();
        match self.terms {
            // ponytail: overlap the serial boundary assembly with the parallel
            // volume assembly; the final add runs in the same order as before.
            Some(terms) if rayon::current_num_threads() > 1 => {
                let (mut residual, boundary) = rayon::join(
                    || {
                        self.problem.assemble_tensor_residual_with_layout(
                            self.time,
                            self.kernel,
                            state,
                            &layout,
                        )
                    },
                    || {
                        self.problem
                            .assemble_state_tensor_boundary_residual(self.time, state, terms)
                    },
                );
                for (volume, boundary) in residual.iter_mut().zip(boundary) {
                    *volume += boundary;
                }
                residual
            }
            _ => {
                let mut residual = self.problem.assemble_tensor_residual_with_layout(
                    self.time,
                    self.kernel,
                    state,
                    &layout,
                );
                if let Some(terms) = self.terms {
                    let boundary = self
                        .problem
                        .assemble_state_tensor_boundary_residual(self.time, state, terms);
                    for (volume, boundary) in residual.iter_mut().zip(boundary) {
                        *volume += boundary;
                    }
                }
                residual
            }
        }
    }
    /// Assembled Jacobian at `state`, including boundary terms when set.
    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        let layout = self.problem.field_layout();
        let mut jacobian = self.problem.assemble_tensor_jacobian_with_layout(
            self.time,
            self.kernel,
            state,
            &layout,
        );
        if let Some(terms) = self.terms {
            let boundary = self
                .problem
                .assemble_state_tensor_boundary_jacobian(self.time, state, terms);
            jacobian = jacobian.as_ref() + boundary.as_ref();
        }
        jacobian
    }
    /// Matrix-free Jacobian action on one or more direction columns.
    ///
    /// Bit-identity: the boundary contribution may run on the serial or the
    /// color-parallel path depending on facet count and thread pool; the two
    /// paths accumulate in different orders, so bits match only for the same
    /// path and thread count (see the boundary driver note).
    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        let layout = self.problem.field_layout();
        match self.terms {
            // ponytail: overlap the serial boundary action with the parallel
            // volume action; the final add runs in the same order as before.
            Some(terms) if rayon::current_num_threads() > 1 => {
                let mut out = Mat::<f64>::zeros(self.problem.system_size(), direction.ncols());
                let ((), boundary) = rayon::join(
                    || {
                        self.problem.apply_tensor_jacobian_with_layout_cached(
                            self.time,
                            self.kernel,
                            state,
                            direction,
                            &layout,
                            out.as_mut(),
                            self.state_cache.as_ref(),
                        )
                    },
                    || {
                        self.problem.apply_state_tensor_boundary_jacobian(
                            self.time, state, direction, terms,
                        )
                    },
                );
                out += boundary;
                out
            }
            _ => {
                let mut out = Mat::<f64>::zeros(self.problem.system_size(), direction.ncols());
                self.problem.apply_tensor_jacobian_with_layout_cached(
                    self.time,
                    self.kernel,
                    state,
                    direction,
                    &layout,
                    out.as_mut(),
                    self.state_cache.as_ref(),
                );
                if let Some(terms) = self.terms {
                    out += self
                        .problem
                        .apply_state_tensor_boundary_jacobian(self.time, state, direction, terms);
                }
                out
            }
        }
    }
    /// Matrix-free Jacobian action into caller-provided storage.
    ///
    /// Bit-identity: the boundary contribution may run on the serial or the
    /// color-parallel path depending on facet count and thread pool; the two
    /// paths accumulate in different orders, so bits match only for the same
    /// path and thread count (see the boundary driver note).
    pub fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        let layout = self.problem.field_layout();
        match self.terms {
            // ponytail: overlap the serial boundary action with the parallel
            // volume action; the final add runs in the same order as before.
            Some(terms) if rayon::current_num_threads() > 1 => {
                let ((), boundary) = rayon::join(
                    || {
                        self.problem.apply_tensor_jacobian_with_layout_cached(
                            self.time,
                            self.kernel,
                            state,
                            direction,
                            &layout,
                            out.rb_mut(),
                            self.state_cache.as_ref(),
                        )
                    },
                    || {
                        self.problem.apply_state_tensor_boundary_jacobian(
                            self.time, state, direction, terms,
                        )
                    },
                );
                out += boundary;
            }
            _ => {
                self.problem.apply_tensor_jacobian_with_layout_cached(
                    self.time,
                    self.kernel,
                    state,
                    direction,
                    &layout,
                    out.rb_mut(),
                    self.state_cache.as_ref(),
                );
                if let Some(terms) = self.terms {
                    out += self
                        .problem
                        .apply_state_tensor_boundary_jacobian(self.time, state, direction, terms);
                }
            }
        }
    }
    /// Return this operator at a new time.
    pub fn at_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    /// Attach state-dependent natural-boundary terms.
    pub fn with_state_boundary<'terms>(
        self,
        terms: &'terms StateTensorBoundaryTerms<2>,
    ) -> SEM2DTensorResidualOperator<'p, 'k, 'terms, M, K> {
        SEM2DTensorResidualOperator {
            problem: self.problem,
            kernel: self.kernel,
            time: self.time,
            terms: Some(terms),
            state_cache: None,
        }
    }

    /// Cache the linearization state for repeated Jacobian actions.
    ///
    /// Always stores an owned copy of `state` (needed by the prepared path,
    /// boundary terms, and to skip per-apply comparisons). The interpolated
    /// per-batch lane-packed values+grads are only built when their footprint
    /// (`≈ ncells·ninputs·3·npts·8` bytes) fits a size budget: by default
    /// 8 MiB, overridable via the `ORMATEX_TENSOR_STATE_CACHE_MB` environment
    /// variable (`0` disables the interpolated cache). Above the budget the
    /// cache holds only the owned copy and the prepared apply path
    /// re-interpolates from it (fresh gather+interpolate phase-1 mode, still
    /// no state comparison, still fused boundary/epilogue). Measurements show
    /// the interpolated cache turns slower than recomputation once it exceeds
    /// the last-level cache, hence this heuristic. Later `apply_jacobian`
    /// calls reuse the interpolated states only on full bitwise equality of
    /// the state and the input field ids (and only when they were built);
    /// otherwise they compute fresh. Results are bit-identical with and
    /// without the interpolated cache. `residual` never uses the cache.
    ///
    /// # Arguments
    /// * `state` - linearization point (`N×1`).
    pub fn prepare_linearization(&mut self, state: MatRef<f64>) {
        self.prepare_linearization_with_limit(state, tensor_state_cache_limit_bytes());
    }

    /// Cache the linearization state with an explicit interpolated-state budget.
    ///
    /// Same as [`prepare_linearization`](Self::prepare_linearization) but with
    /// the budget passed directly instead of read from the environment.
    /// Crate-internal so tests can force the threshold (e.g. `0`) without
    /// touching the process-wide env-var cache.
    ///
    /// # Arguments
    /// * `state` - linearization point (`N×1`).
    /// * `limit_bytes` - interpolated-state budget in bytes; `0` keeps only
    ///   the owned state copy.
    pub(crate) fn prepare_linearization_with_limit(
        &mut self,
        state: MatRef<f64>,
        limit_bytes: u64,
    ) {
        self.state_cache = Some(self.problem.build_tensor_state_cache_with_limit(
            self.kernel,
            state,
            limit_bytes,
        ));
    }

    /// Prepared Jacobian action from the owned linearization cache.
    ///
    /// Uses the cache's owned state copy for everything needing the state
    /// (volume via cached interpolation when built, else by fresh
    /// gather+interpolate from the owned copy; boundary via the owned copy)
    /// with no state comparison. Returns `false` when no cache is prepared.
    /// Bit-identical to `apply_jacobian_into` at the prepared state; the
    /// boundary add and `epilogue` fuse into the row-sorted phase-2 reduction.
    /// The boundary contribution itself may run on the serial or the
    /// color-parallel path depending on facet count and thread pool; bits
    /// match only for the same path and thread count (see the boundary
    /// driver note).
    ///
    /// # Arguments
    /// * `direction` - global directions (`N×ncols`).
    /// * `out` - global output (`N×ncols`). Fully overwritten.
    /// * `epilogue` - per-row post-processing fused into the reduction.
    ///
    /// # Returns
    /// `true` if the prepared apply ran, `false` to fall back.
    pub fn apply_prepared_jacobian_into(
        &self,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
        epilogue: RowEpilogue<'_>,
    ) -> bool {
        let cache = match self.state_cache.as_ref() {
            Some(cache) => cache,
            None => return false,
        };
        let layout = self.problem.field_layout();
        let selection = self.problem.fields.resolve_selection(
            self.kernel.input_nfields(),
            self.kernel.input_field_names(),
            self.kernel.output_nfields(),
            self.kernel.output_field_names(),
            "tensor residual kernel",
        );
        let ninputs = selection.inputs.len();
        let noutputs = selection.outputs.len();
        assert_eq!(
            direction.nrows(),
            layout.total_size,
            "direction size mismatch"
        );
        assert_eq!(out.nrows(), layout.total_size, "output size mismatch");
        assert_eq!(
            out.ncols(),
            direction.ncols(),
            "output column count mismatch"
        );
        let ncols = direction.ncols();
        if ncols == 0 {
            return true;
        }
        let input_offsets: Vec<usize> = selection
            .inputs
            .iter()
            .map(|&field| layout.offsets[field])
            .collect();
        let sources: Vec<usize> = selection
            .inputs
            .iter()
            .map(|&g| self.problem.restriction.field_map_source(g))
            .collect();
        let scatter = self.problem.sorted_scatter_for(&selection.outputs);
        let mut evec_sorted = self.problem.acquire_evec(scatter.nslots * ncols);
        // Owned linearization state; `ncols == 1` by construction of the cache.
        let owned_state =
            faer::MatRef::from_column_major_slice(&cache.state_copy, cache.nrows, cache.ncols);
        // Boundary action into its own buffer, concurrently with phase 1 when
        // parallel (as the unprepared path does via `rayon::join`).
        let time = self.time;
        let problem = self.problem;
        let kernel = self.kernel;
        // Interpolated states when built; otherwise fresh gather+interpolate
        // from the owned copy (same arithmetic, still no state comparison).
        let (fresh_state, cache_ref) = if cache.has_batch_states() {
            (None, Some(cache))
        } else {
            (Some(owned_state), None)
        };
        let ctx = TensorPhase1Ctx {
            time,
            ninputs,
            noutputs,
            sources: &sources,
            input_offsets: &input_offsets,
            fresh_state,
            cache: cache_ref,
            direction: Some(direction),
            scatter: &scatter,
            evec_sorted: &mut evec_sorted,
        };
        if let Some(terms) = self.terms {
            if rayon::current_num_threads() > 1 {
                let ((), boundary) = rayon::join(
                    || {
                        problem.phase1_fill_sorted_dispatch(kernel, ctx);
                    },
                    || {
                        problem.apply_state_tensor_boundary_jacobian(
                            time,
                            owned_state,
                            direction,
                            terms,
                        )
                    },
                );
                reduce_sorted_evec_into_out(
                    &scatter,
                    &evec_sorted,
                    ncols,
                    Some(boundary.as_ref()),
                    epilogue,
                    &self.problem.evec_pool,
                    out,
                );
            } else {
                problem.phase1_fill_sorted_dispatch(kernel, ctx);
                let boundary = problem.apply_state_tensor_boundary_jacobian(
                    time,
                    owned_state,
                    direction,
                    terms,
                );
                reduce_sorted_evec_into_out(
                    &scatter,
                    &evec_sorted,
                    ncols,
                    Some(boundary.as_ref()),
                    epilogue,
                    &self.problem.evec_pool,
                    out,
                );
            }
        } else {
            problem.phase1_fill_sorted_dispatch(kernel, ctx);
            reduce_sorted_evec_into_out(
                &scatter,
                &evec_sorted,
                ncols,
                None,
                epilogue,
                &self.problem.evec_pool,
                out,
            );
        }
        self.problem.release_evec(evec_sorted);
        true
    }
}

impl<M, K> CompleteResidualOperator for SEM2DTensorResidualOperator<'_, '_, '_, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    K: TensorResidualKernel<2> + Sync,
{
    fn system_size(&self) -> usize {
        SEM2DTensorResidualOperator::system_size(self)
    }
    fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        SEM2DTensorResidualOperator::residual(self, state)
    }
    fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        SEM2DTensorResidualOperator::assemble_jacobian(self, state)
    }
    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        SEM2DTensorResidualOperator::apply_jacobian(self, state, direction)
    }
    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        SEM2DTensorResidualOperator::apply_jacobian_into(self, state, direction, out);
    }
    fn prepare_linearization(&mut self, state: MatRef<f64>) {
        SEM2DTensorResidualOperator::prepare_linearization(self, state)
    }
    fn apply_prepared_jacobian_into(
        &self,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
        epilogue: RowEpilogue<'_>,
    ) -> bool {
        SEM2DTensorResidualOperator::apply_prepared_jacobian_into(self, direction, out, epilogue)
    }
}

/// Residual operator that combines one statically dispatched tensor kernel
/// with one traditional weak-form kernel.
pub struct SEM2DMixedResidualOperator<'p, 't, 'w, 'bt, 'bw, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    problem: &'p SEM2DProblem<M>,
    tensor_kernel: &'t T,
    weak_kernel: &'w W,
    time: f64,
    tensor_terms: Option<&'bt StateTensorBoundaryTerms<2>>,
    weak_terms: Option<&'bw StateBoundaryTerms>,
    state_cache: Option<TensorStateCache>,
}

impl<'p, 't, 'w, 'bt, 'bw, M, T, W> SEM2DMixedResidualOperator<'p, 't, 'w, 'bt, 'bw, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    T: TensorResidualKernel<2> + Sync,
    W: ResidualKernel + Sync,
{
    /// Retained (reduced) system size, including all fields.
    pub fn system_size(&self) -> usize {
        self.problem.system_size()
    }

    /// Residual at the operator's time, including boundary terms when set.
    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        let layout = self.problem.field_layout();
        match self.tensor_terms {
            // ponytail: overlap the serial tensor-boundary assembly with the
            // rest (same add order as the sequential path below).
            Some(terms) if rayon::current_num_threads() > 1 => {
                let (mut residual, boundary) = rayon::join(
                    || {
                        let mut residual = self.problem.assemble_tensor_residual_with_layout(
                            self.time,
                            self.tensor_kernel,
                            state,
                            &layout,
                        );
                        let weak =
                            self.problem
                                .assemble_residual(self.time, self.weak_kernel, state);
                        for (tensor, weak) in residual.iter_mut().zip(weak) {
                            *tensor += weak;
                        }
                        if let Some(terms) = self.weak_terms {
                            let boundary = self
                                .problem
                                .assemble_state_boundary_residual(self.time, state, terms);
                            for (volume, boundary) in residual.iter_mut().zip(boundary) {
                                *volume += boundary;
                            }
                        }
                        residual
                    },
                    || {
                        self.problem
                            .assemble_state_tensor_boundary_residual(self.time, state, terms)
                    },
                );
                for (volume, boundary) in residual.iter_mut().zip(boundary) {
                    *volume += boundary;
                }
                residual
            }
            _ => {
                let mut residual = self.problem.assemble_tensor_residual_with_layout(
                    self.time,
                    self.tensor_kernel,
                    state,
                    &layout,
                );
                let weak = self
                    .problem
                    .assemble_residual(self.time, self.weak_kernel, state);
                for (tensor, weak) in residual.iter_mut().zip(weak) {
                    *tensor += weak;
                }
                if let Some(terms) = self.weak_terms {
                    let boundary = self
                        .problem
                        .assemble_state_boundary_residual(self.time, state, terms);
                    for (volume, boundary) in residual.iter_mut().zip(boundary) {
                        *volume += boundary;
                    }
                }
                if let Some(terms) = self.tensor_terms {
                    let boundary = self
                        .problem
                        .assemble_state_tensor_boundary_residual(self.time, state, terms);
                    for (volume, boundary) in residual.iter_mut().zip(boundary) {
                        *volume += boundary;
                    }
                }
                residual
            }
        }
    }

    /// Assembled Jacobian at `state`, including boundary terms when set.
    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        let layout = self.problem.field_layout();
        let tensor = self.problem.assemble_tensor_jacobian_with_layout(
            self.time,
            self.tensor_kernel,
            state,
            &layout,
        );
        let weak = self
            .problem
            .assemble_residual_jacobian(self.time, self.weak_kernel, state);
        let mut jacobian = tensor.as_ref() + weak.as_ref();
        if let Some(terms) = self.weak_terms {
            let boundary = self
                .problem
                .assemble_state_boundary_jacobian(self.time, state, terms);
            jacobian = jacobian.as_ref() + boundary.as_ref();
        }
        if let Some(terms) = self.tensor_terms {
            let boundary = self
                .problem
                .assemble_state_tensor_boundary_jacobian(self.time, state, terms);
            jacobian = jacobian.as_ref() + boundary.as_ref();
        }
        jacobian
    }

    /// Matrix-free Jacobian action on one or more direction columns.
    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        let layout = self.problem.field_layout();
        match self.tensor_terms {
            // ponytail: overlap the serial tensor-boundary action with the
            // rest (same add order as the sequential path below).
            Some(terms) if rayon::current_num_threads() > 1 => {
                let mut out = Mat::<f64>::zeros(self.system_size(), direction.ncols());
                let ((), boundary) = rayon::join(
                    || {
                        self.problem.apply_tensor_jacobian_with_layout_cached(
                            self.time,
                            self.tensor_kernel,
                            state,
                            direction,
                            &layout,
                            out.as_mut(),
                            self.state_cache.as_ref(),
                        );
                        out += self.problem.apply_jacobian(
                            self.time,
                            self.weak_kernel,
                            state,
                            direction,
                        );
                        if let Some(terms) = self.weak_terms {
                            out += self
                                .problem
                                .apply_state_boundary_jacobian(self.time, state, direction, terms);
                        }
                    },
                    || {
                        self.problem.apply_state_tensor_boundary_jacobian(
                            self.time, state, direction, terms,
                        )
                    },
                );
                out += boundary;
                out
            }
            _ => {
                let mut out = Mat::<f64>::zeros(self.system_size(), direction.ncols());
                self.problem.apply_tensor_jacobian_with_layout_cached(
                    self.time,
                    self.tensor_kernel,
                    state,
                    direction,
                    &layout,
                    out.as_mut(),
                    self.state_cache.as_ref(),
                );
                out += self
                    .problem
                    .apply_jacobian(self.time, self.weak_kernel, state, direction);
                if let Some(terms) = self.weak_terms {
                    out += self
                        .problem
                        .apply_state_boundary_jacobian(self.time, state, direction, terms);
                }
                if let Some(terms) = self.tensor_terms {
                    out += self
                        .problem
                        .apply_state_tensor_boundary_jacobian(self.time, state, direction, terms);
                }
                out
            }
        }
    }

    /// Matrix-free Jacobian action into caller-provided storage.
    pub fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        let layout = self.problem.field_layout();
        match self.tensor_terms {
            // ponytail: overlap the serial tensor-boundary action with the
            // rest (same add order as the sequential path below).
            Some(terms) if rayon::current_num_threads() > 1 => {
                let ((), boundary) = rayon::join(
                    || {
                        self.problem.apply_tensor_jacobian_with_layout_cached(
                            self.time,
                            self.tensor_kernel,
                            state,
                            direction,
                            &layout,
                            out.rb_mut(),
                            self.state_cache.as_ref(),
                        );
                        out += self.problem.apply_jacobian(
                            self.time,
                            self.weak_kernel,
                            state,
                            direction,
                        );
                        if let Some(terms) = self.weak_terms {
                            out += self
                                .problem
                                .apply_state_boundary_jacobian(self.time, state, direction, terms);
                        }
                    },
                    || {
                        self.problem.apply_state_tensor_boundary_jacobian(
                            self.time, state, direction, terms,
                        )
                    },
                );
                out += boundary;
            }
            _ => {
                self.problem.apply_tensor_jacobian_with_layout_cached(
                    self.time,
                    self.tensor_kernel,
                    state,
                    direction,
                    &layout,
                    out.rb_mut(),
                    self.state_cache.as_ref(),
                );
                out += self
                    .problem
                    .apply_jacobian(self.time, self.weak_kernel, state, direction);
                if let Some(terms) = self.weak_terms {
                    out += self
                        .problem
                        .apply_state_boundary_jacobian(self.time, state, direction, terms);
                }
                if let Some(terms) = self.tensor_terms {
                    out += self
                        .problem
                        .apply_state_tensor_boundary_jacobian(self.time, state, direction, terms);
                }
            }
        }
    }

    /// Return this operator at a new time.
    pub fn at_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    /// Cache the tensor linearization state for repeated Jacobian actions.
    ///
    /// Same semantics as [`SEM2DTensorResidualOperator::prepare_linearization`],
    /// covering only the tensor volume term; the weak term always recomputes.
    /// The interpolated per-batch states are subject to the same size budget
    /// (default 8 MiB, `ORMATEX_TENSOR_STATE_CACHE_MB`); above it the prepared
    /// path re-interpolates the tensor volume from the owned copy.
    ///
    /// # Arguments
    /// * `state` - linearization point (`N×1`).
    pub fn prepare_linearization(&mut self, state: MatRef<f64>) {
        self.prepare_linearization_with_limit(state, tensor_state_cache_limit_bytes());
    }

    /// Cache the tensor linearization state with an explicit budget.
    ///
    /// Same as [`prepare_linearization`](Self::prepare_linearization) but with
    /// the interpolated-state budget passed directly. Crate-internal so tests
    /// can force the threshold without touching the process-wide env-var cache.
    ///
    /// # Arguments
    /// * `state` - linearization point (`N×1`).
    /// * `limit_bytes` - interpolated-state budget in bytes; `0` keeps only
    ///   the owned state copy.
    pub(crate) fn prepare_linearization_with_limit(
        &mut self,
        state: MatRef<f64>,
        limit_bytes: u64,
    ) {
        self.state_cache = Some(self.problem.build_tensor_state_cache_with_limit(
            self.tensor_kernel,
            state,
            limit_bytes,
        ));
    }

    /// Cached tensor-volume fill into `out` (row-sorted reduction, no epilogue).
    ///
    /// Runs phase 1 from the owned linearization cache with no state
    /// comparison (cached interpolation when built, else fresh
    /// gather+interpolate from the owned copy), then reduces the row-sorted
    /// E-vector into `out` with [`RowEpilogue::None`]. Shared by the serial
    /// and overlapped branches of
    /// [`apply_prepared_jacobian_into`](Self::apply_prepared_jacobian_into).
    ///
    /// # Arguments
    /// * `cache` - owned linearization cache from `prepare_linearization`.
    /// * `direction` - global directions (`N×ncols`).
    /// * `out` - global output (`N×ncols`). Fully overwritten.
    fn fill_cached_tensor_volume(
        &self,
        cache: &TensorStateCache,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        let layout = self.problem.field_layout();
        let selection = self.problem.fields.resolve_selection(
            self.tensor_kernel.input_nfields(),
            self.tensor_kernel.input_field_names(),
            self.tensor_kernel.output_nfields(),
            self.tensor_kernel.output_field_names(),
            "tensor residual kernel",
        );
        let ninputs = selection.inputs.len();
        let noutputs = selection.outputs.len();
        let input_offsets: Vec<usize> = selection
            .inputs
            .iter()
            .map(|&field| layout.offsets[field])
            .collect();
        let sources: Vec<usize> = selection
            .inputs
            .iter()
            .map(|&g| self.problem.restriction.field_map_source(g))
            .collect();
        let scatter = self.problem.sorted_scatter_for(&selection.outputs);
        let mut evec_sorted = self
            .problem
            .acquire_evec(scatter.nslots * direction.ncols());
        // Owned linearization state; the interpolated states are reused when
        // built, else phase 1 re-interpolates from this copy (bit-identical).
        let owned_state =
            faer::MatRef::from_column_major_slice(&cache.state_copy, cache.nrows, cache.ncols);
        let (fresh_state, cache_ref) = if cache.has_batch_states() {
            (None, Some(cache))
        } else {
            (Some(owned_state), None)
        };
        let ctx = TensorPhase1Ctx {
            time: self.time,
            ninputs,
            noutputs,
            sources: &sources,
            input_offsets: &input_offsets,
            fresh_state,
            cache: cache_ref,
            direction: Some(direction),
            scatter: &scatter,
            evec_sorted: &mut evec_sorted,
        };
        self.problem
            .phase1_fill_sorted_dispatch(self.tensor_kernel, ctx);
        reduce_sorted_evec_into_out(
            &scatter,
            &evec_sorted,
            direction.ncols(),
            None,
            RowEpilogue::None,
            &self.problem.evec_pool,
            out,
        );
        self.problem.release_evec(evec_sorted);
    }

    /// Prepared Jacobian action from the owned tensor cache.
    ///
    /// Tensor volume uses cached interpolation when built, else fresh
    /// gather+interpolate from the owned state copy; weak volume and both
    /// boundary terms recompute from the owned state copy (no state
    /// comparison). Returns `false` when no cache is prepared. Bit-identical
    /// to `apply_jacobian_into` at the prepared state: tensor volume reduces
    /// first, then `+= weak`, `+= weak boundary`, `+= tensor boundary` in the
    /// same order, then the `epilogue` scaling per row in SIMD order.
    ///
    /// # Arguments
    /// * `direction` - global directions (`N×ncols`).
    /// * `out` - global output (`N×ncols`). Fully overwritten.
    /// * `epilogue` - per-row post-processing after all adds.
    ///
    /// # Returns
    /// `true` if the prepared apply ran, `false` to fall back.
    pub fn apply_prepared_jacobian_into(
        &self,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
        epilogue: RowEpilogue<'_>,
    ) -> bool {
        let cache = match self.state_cache.as_ref() {
            Some(cache) => cache,
            None => return false,
        };
        let layout = self.problem.field_layout();
        assert_eq!(
            direction.nrows(),
            layout.total_size,
            "direction size mismatch"
        );
        assert_eq!(out.nrows(), layout.total_size, "output size mismatch");
        assert_eq!(
            out.ncols(),
            direction.ncols(),
            "output column count mismatch"
        );
        let ncols = direction.ncols();
        if ncols == 0 {
            return true;
        }
        let owned_state =
            faer::MatRef::from_column_major_slice(&cache.state_copy, cache.nrows, cache.ncols);
        // Tensor volume (cached) fills `out` first; weak volume and both
        // boundary terms then `+=` in the same order as `apply_jacobian_into`.
        // The serial tensor-boundary action overlaps the volume fill plus the
        // weak adds via `rayon::join` when parallel, as the unprepared path
        // does; the `+=` order is unchanged, hence bit-identical.
        let time = self.time;
        let problem = self.problem;
        let weak_kernel = self.weak_kernel;
        let weak_terms = self.weak_terms;
        if let Some(terms) = self.tensor_terms {
            if rayon::current_num_threads() > 1 {
                let ((), tensor_boundary) = rayon::join(
                    || {
                        self.fill_cached_tensor_volume(cache, direction, out.rb_mut());
                        out += problem.apply_jacobian(time, weak_kernel, owned_state, direction);
                        if let Some(weak) = weak_terms {
                            out += problem.apply_state_boundary_jacobian(
                                time,
                                owned_state,
                                direction,
                                weak,
                            );
                        }
                    },
                    || {
                        problem.apply_state_tensor_boundary_jacobian(
                            time,
                            owned_state,
                            direction,
                            terms,
                        )
                    },
                );
                out += tensor_boundary;
            } else {
                self.fill_cached_tensor_volume(cache, direction, out.rb_mut());
                out += problem.apply_jacobian(time, weak_kernel, owned_state, direction);
                if let Some(weak) = weak_terms {
                    out +=
                        problem.apply_state_boundary_jacobian(time, owned_state, direction, weak);
                }
                out += problem.apply_state_tensor_boundary_jacobian(
                    time,
                    owned_state,
                    direction,
                    terms,
                );
            }
        } else {
            self.fill_cached_tensor_volume(cache, direction, out.rb_mut());
            out += problem.apply_jacobian(time, weak_kernel, owned_state, direction);
            if let Some(weak) = weak_terms {
                out += problem.apply_state_boundary_jacobian(time, owned_state, direction, weak);
            }
        }
        match epilogue {
            RowEpilogue::None => {}
            RowEpilogue::NegScale(m_inv) => {
                assert_eq!(m_inv.len(), layout.total_size);
                for c in 0..ncols {
                    for r in 0..layout.total_size {
                        // SIMD order `-(v * f)`, as in the fused tensor path.
                        out[(r, c)] = -(out[(r, c)] * m_inv[r]);
                    }
                }
            }
        }
        true
    }

    /// Attach weak-form state-dependent boundary terms.
    pub fn with_weak_state_boundary<'terms>(
        self,
        terms: &'terms StateBoundaryTerms,
    ) -> SEM2DMixedResidualOperator<'p, 't, 'w, 'bt, 'terms, M, T, W> {
        SEM2DMixedResidualOperator {
            problem: self.problem,
            tensor_kernel: self.tensor_kernel,
            weak_kernel: self.weak_kernel,
            time: self.time,
            tensor_terms: self.tensor_terms,
            weak_terms: Some(terms),
            state_cache: None,
        }
    }

    /// Attach tensor state-dependent boundary terms.
    pub fn with_tensor_state_boundary<'terms>(
        self,
        terms: &'terms StateTensorBoundaryTerms<2>,
    ) -> SEM2DMixedResidualOperator<'p, 't, 'w, 'terms, 'bw, M, T, W> {
        SEM2DMixedResidualOperator {
            problem: self.problem,
            tensor_kernel: self.tensor_kernel,
            weak_kernel: self.weak_kernel,
            time: self.time,
            tensor_terms: Some(terms),
            weak_terms: self.weak_terms,
            state_cache: None,
        }
    }
}

impl<M, T, W> CompleteResidualOperator for SEM2DMixedResidualOperator<'_, '_, '_, '_, '_, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    T: TensorResidualKernel<2> + Sync,
    W: ResidualKernel + Sync,
{
    fn system_size(&self) -> usize {
        SEM2DMixedResidualOperator::system_size(self)
    }

    fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        SEM2DMixedResidualOperator::residual(self, state)
    }

    fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        SEM2DMixedResidualOperator::assemble_jacobian(self, state)
    }

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        SEM2DMixedResidualOperator::apply_jacobian(self, state, direction)
    }

    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        SEM2DMixedResidualOperator::apply_jacobian_into(self, state, direction, out);
    }

    fn prepare_linearization(&mut self, state: MatRef<f64>) {
        SEM2DMixedResidualOperator::prepare_linearization(self, state)
    }
    fn apply_prepared_jacobian_into(
        &self,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
        epilogue: RowEpilogue<'_>,
    ) -> bool {
        SEM2DMixedResidualOperator::apply_prepared_jacobian_into(self, direction, out, epilogue)
    }
}

/// Selects a weak, tensor, or mixed residual operator without erasing the
/// concrete kernel types. The match is performed once at the operator
/// boundary; each branch retains static dispatch through the SEM loops.
pub enum SEM2DResidualExecution<'p, 't, 'w, 'bt, 'bw, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
{
    Weak(SEM2DResidualOperator<'p, 'w, 'bw, M, W>),
    Tensor(SEM2DTensorResidualOperator<'p, 't, 'bt, M, T>),
    Mixed(SEM2DMixedResidualOperator<'p, 't, 'w, 'bt, 'bw, M, T, W>),
}

impl<'p, 't, 'w, 'bt, 'bw, M, T, W> SEM2DResidualExecution<'p, 't, 'w, 'bt, 'bw, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    T: TensorResidualKernel<2> + Sync,
    W: ResidualKernel + Sync,
{
    /// Retained (reduced) system size, including all fields.
    pub fn system_size(&self) -> usize {
        match self {
            Self::Weak(operator) => operator.system_size(),
            Self::Tensor(operator) => operator.system_size(),
            Self::Mixed(operator) => operator.system_size(),
        }
    }

    /// Residual at the operator's time, including boundary terms when set.
    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        match self {
            Self::Weak(operator) => operator.residual(state),
            Self::Tensor(operator) => operator.residual(state),
            Self::Mixed(operator) => operator.residual(state),
        }
    }

    /// Assembled Jacobian at `state`, including boundary terms when set.
    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        match self {
            Self::Weak(operator) => operator.assemble_jacobian(state),
            Self::Tensor(operator) => operator.assemble_jacobian(state),
            Self::Mixed(operator) => operator.assemble_jacobian(state),
        }
    }

    /// Matrix-free Jacobian action on one or more direction columns.
    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        match self {
            Self::Weak(operator) => operator.apply_jacobian(state, direction),
            Self::Tensor(operator) => operator.apply_jacobian(state, direction),
            Self::Mixed(operator) => operator.apply_jacobian(state, direction),
        }
    }

    /// Matrix-free Jacobian action into caller-provided storage.
    pub fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        match self {
            Self::Weak(operator) => {
                CompleteResidualOperator::apply_jacobian_into(operator, state, direction, out)
            }
            Self::Tensor(operator) => operator.apply_jacobian_into(state, direction, out),
            Self::Mixed(operator) => operator.apply_jacobian_into(state, direction, out),
        }
    }

    /// Return this operator at a new time.
    pub fn at_time(self, time: f64) -> Self {
        match self {
            Self::Weak(operator) => Self::Weak(operator.at_time(time)),
            Self::Tensor(operator) => Self::Tensor(operator.at_time(time)),
            Self::Mixed(operator) => Self::Mixed(operator.at_time(time)),
        }
    }

    /// Cache the tensor linearization state (no-op for `Weak`).
    ///
    /// # Arguments
    /// * `state` - linearization point (`N×1`).
    pub fn prepare_linearization(&mut self, state: MatRef<f64>) {
        match self {
            Self::Weak(_) => {}
            Self::Tensor(operator) => operator.prepare_linearization(state),
            Self::Mixed(operator) => operator.prepare_linearization(state),
        }
    }

    /// Prepared Jacobian action; `Weak` returns `false`, otherwise forwards.
    ///
    /// # Arguments
    /// * `direction` - global directions (`N×ncols`).
    /// * `out` - global output (`N×ncols`). Fully overwritten.
    /// * `epilogue` - per-row post-processing fused into the reduction.
    ///
    /// # Returns
    /// `true` if the prepared apply ran, `false` to fall back.
    pub fn apply_prepared_jacobian_into(
        &self,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
        epilogue: RowEpilogue<'_>,
    ) -> bool {
        match self {
            Self::Weak(_) => false,
            Self::Tensor(operator) => {
                operator.apply_prepared_jacobian_into(direction, out, epilogue)
            }
            Self::Mixed(operator) => {
                operator.apply_prepared_jacobian_into(direction, out, epilogue)
            }
        }
    }
}

impl<M, T, W> CompleteResidualOperator for SEM2DResidualExecution<'_, '_, '_, '_, '_, M, T, W>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    T: TensorResidualKernel<2> + Sync,
    W: ResidualKernel + Sync,
{
    fn system_size(&self) -> usize {
        SEM2DResidualExecution::system_size(self)
    }

    fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        SEM2DResidualExecution::residual(self, state)
    }

    fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        SEM2DResidualExecution::assemble_jacobian(self, state)
    }

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        SEM2DResidualExecution::apply_jacobian(self, state, direction)
    }

    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        SEM2DResidualExecution::apply_jacobian_into(self, state, direction, out);
    }

    fn prepare_linearization(&mut self, state: MatRef<f64>) {
        SEM2DResidualExecution::prepare_linearization(self, state)
    }
    fn apply_prepared_jacobian_into(
        &self,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
        epilogue: RowEpilogue<'_>,
    ) -> bool {
        SEM2DResidualExecution::apply_prepared_jacobian_into(self, direction, out, epilogue)
    }
}

impl<'p, 'k, 'b, M, K> SEM2DResidualOperator<'p, 'k, 'b, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    K: ResidualKernel + Sync,
{
    /// Retained (reduced) system size, including all fields.
    pub fn system_size(&self) -> usize {
        self.problem.system_size()
    }

    /// Residual at the operator's time, including boundary terms when set.
    pub fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        match &self.terms {
            Some(terms) => {
                self.problem
                    .assemble_complete_residual(self.time, self.kernel, state, terms)
            }
            None => self
                .problem
                .assemble_residual(self.time, self.kernel, state),
        }
    }

    /// Assembled Jacobian at `state`, including boundary terms when set.
    pub fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        match &self.terms {
            Some(terms) => {
                self.problem
                    .assemble_complete_jacobian(self.time, self.kernel, state, terms)
            }
            None => self
                .problem
                .assemble_residual_jacobian(self.time, self.kernel, state),
        }
    }

    /// Matrix-free Jacobian action on one or more direction columns.
    pub fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        match &self.terms {
            Some(terms) => self.problem.apply_complete_jacobian(
                self.time,
                self.kernel,
                state,
                direction,
                terms,
            ),
            None => self
                .problem
                .apply_jacobian(self.time, self.kernel, state, direction),
        }
    }

    /// Return this operator at a new time.
    pub fn at_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    /// Attach state-dependent natural-boundary terms.
    pub fn with_state_boundary<'terms>(
        self,
        terms: &'terms StateBoundaryTerms,
    ) -> SEM2DResidualOperator<'p, 'k, 'terms, M, K> {
        SEM2DResidualOperator {
            problem: self.problem,
            kernel: self.kernel,
            time: self.time,
            terms: Some(terms),
        }
    }
}

impl<M, K> CompleteResidualOperator for SEM2DResidualOperator<'_, '_, '_, M, K>
where
    M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync,
    K: ResidualKernel + Sync,
{
    fn system_size(&self) -> usize {
        SEM2DResidualOperator::system_size(self)
    }

    fn residual(&self, state: MatRef<f64>) -> Vec<f64> {
        SEM2DResidualOperator::residual(self, state)
    }

    fn assemble_jacobian(&self, state: MatRef<f64>) -> SparseColMat<usize, f64> {
        SEM2DResidualOperator::assemble_jacobian(self, state)
    }

    fn apply_jacobian(&self, state: MatRef<f64>, direction: MatRef<f64>) -> Mat<f64> {
        SEM2DResidualOperator::apply_jacobian(self, state, direction)
    }

    fn apply_jacobian_into(
        &self,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        mut out: MatMut<'_, f64>,
    ) {
        self.problem
            .apply_jacobian_into(self.time, self.kernel, state, direction, out.rb_mut());
        if let Some(terms) = self.terms {
            out += self
                .problem
                .apply_state_boundary_jacobian(self.time, state, direction, terms);
        }
    }
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM2DProblem<M> {
    /// Build a weak-form residual operator for `kernel`.
    ///
    /// Validates the kernel's field selection against the problem up front.
    /// Use for non-tensor physics; attach facet fluxes later with
    /// [`with_state_boundary`](SEM2DResidualOperator::with_state_boundary).
    /// For sum-factorized kernels see
    /// [`tensor_residual_operator`](Self::tensor_residual_operator).
    pub fn residual_operator<'p, 'k, K: ResidualKernel + Sync>(
        &'p self,
        kernel: &'k K,
    ) -> SEM2DResidualOperator<'p, 'k, 'static, M, K>
    where
        M: Sync,
    {
        self.resolve_residual_selection(kernel, "residual kernel");
        SEM2DResidualOperator {
            problem: self,
            kernel,
            time: 0.0,
            terms: None,
        }
    }
    /// Build a statically dispatched tensor-product residual operator.
    ///
    /// Validates tensor field selection up front; volume assembly uses the
    /// [`tensor`](super::tensor) sum-factorized path. Use when the kernel
    /// implements `TensorResidualKernel<2>`; otherwise use
    /// [`residual_operator`](Self::residual_operator), or combine both with
    /// [`mixed_residual_operator`](Self::mixed_residual_operator).
    pub fn tensor_residual_operator<'p, 'k, K: TensorResidualKernel<2> + Sync>(
        &'p self,
        kernel: &'k K,
    ) -> SEM2DTensorResidualOperator<'p, 'k, 'static, M, K>
    where
        M: Sync,
    {
        self.fields.resolve_selection(
            kernel.input_nfields(),
            kernel.input_field_names(),
            kernel.output_nfields(),
            kernel.output_field_names(),
            "tensor residual kernel",
        );
        SEM2DTensorResidualOperator {
            problem: self,
            kernel,
            time: 0.0,
            terms: None,
            state_cache: None,
        }
    }

    /// Build an operator summing one tensor and one weak-form kernel.
    ///
    /// Both kernels must agree on field count, names, and order. Residuals
    /// and Jacobian actions add volume contributions from each path, plus
    /// each path's own boundary terms when attached; use when only part of
    /// the physics has a tensor implementation.
    pub fn mixed_residual_operator<'p, 't, 'w, T, W>(
        &'p self,
        tensor_kernel: &'t T,
        weak_kernel: &'w W,
    ) -> SEM2DMixedResidualOperator<'p, 't, 'w, 'static, 'static, M, T, W>
    where
        M: Sync,
        T: TensorResidualKernel<2> + Sync,
        W: ResidualKernel + Sync,
    {
        assert_eq!(
            tensor_kernel.nfields(),
            weak_kernel.nfields(),
            "mixed residual kernel field count mismatch"
        );
        self.validate_fields(
            tensor_kernel.nfields(),
            tensor_kernel.field_names(),
            "tensor residual kernel",
        );
        self.validate_fields(
            weak_kernel.nfields(),
            weak_kernel.field_names(),
            "residual kernel",
        );
        if let (Some(tensor_names), Some(weak_names)) =
            (tensor_kernel.field_names(), weak_kernel.field_names())
        {
            assert_eq!(
                tensor_names, weak_names,
                "mixed residual kernel field names/order mismatch"
            );
        }
        SEM2DMixedResidualOperator {
            problem: self,
            tensor_kernel,
            weak_kernel,
            time: 0.0,
            tensor_terms: None,
            weak_terms: None,
            state_cache: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DofReduction2D, FieldRegistry, StateTensorBoundaryTerms};
    use crate::{TensorKernelEdacNavierStokes2D, TensorKernelEdacSplitBoundaryFlux2D};
    use ndelement::types::ReferenceCellType;
    use ndmesh::shapes::unit_square;

    type QuadMesh = ndmesh::SingleElementMesh<
        f64,
        ndelement::ciarlet::CiarletElement<f64, ndelement::map::IdentityMap, f64>,
    >;

    /// Small 3-field EDAC tensor problem for the cache-budget tests.
    ///
    /// # Returns
    /// 2×2 quad problem at degree 2 with no Dirichlet reduction.
    fn budget_test_problem() -> SEM2DProblem<QuadMesh> {
        SEM2DProblem::new(
            unit_square(2, 2, ReferenceCellType::Quadrilateral, 1),
            2,
            FieldRegistry::new(["u", "v", "p"]),
            DofReduction2D::None,
        )
    }

    /// Assert two matrices are bitwise equal.
    ///
    /// # Arguments
    /// * `a` - first matrix.
    /// * `b` - second matrix.
    /// * `what` - label for failure messages.
    fn assert_bits_eq_mat(a: MatRef<'_, f64>, b: MatRef<'_, f64>, what: &str) {
        assert_eq!(a.nrows(), b.nrows(), "{what}: row count mismatch");
        assert_eq!(a.ncols(), b.ncols(), "{what}: column count mismatch");
        for c in 0..a.ncols() {
            for r in 0..a.nrows() {
                assert_eq!(
                    a[(r, c)].to_bits(),
                    b[(r, c)].to_bits(),
                    "{what}[({r},{c})]",
                );
            }
        }
    }

    /// Prepared-without-interpolated-cache matches prepared-with-cache and unprepared.
    ///
    /// Forces the size budget to `0` via the crate-internal limit parameter
    /// (not the env var, to avoid races with parallel tests): the prepared
    /// path then re-interpolates from the owned state copy with no state
    /// comparison. Covers the plain and tensor-boundary operator variants,
    /// multi-column directions.
    #[test]
    fn prepared_without_interpolated_cache_matches_bitwise() {
        let problem = budget_test_problem();
        let k = TensorKernelEdacNavierStokes2D::new(1.0, 0.1, 10.0, 0.0);
        let terms =
            StateTensorBoundaryTerms::new().with_default(TensorKernelEdacSplitBoundaryFlux2D);
        let n = problem.system_size();
        let state = Mat::from_fn(n, 1, |i, _| 0.2 + 0.03 * i as f64);
        let direction = Mat::from_fn(n, 2, |i, c| {
            (0.17 * (i + 1) as f64 * (c as f64 + 1.0)).sin()
        });

        for with_boundary in [false, true] {
            let plain = if with_boundary {
                problem
                    .tensor_residual_operator(&k)
                    .with_state_boundary(&terms)
            } else {
                problem.tensor_residual_operator(&k)
            };
            let mut expected = Mat::zeros(n, 2);
            plain.apply_jacobian_into(state.as_ref(), direction.as_ref(), expected.as_mut());

            // Interpolated states built (huge budget).
            let mut with_cache = if with_boundary {
                problem
                    .tensor_residual_operator(&k)
                    .with_state_boundary(&terms)
            } else {
                problem.tensor_residual_operator(&k)
            };
            with_cache.prepare_linearization_with_limit(state.as_ref(), u64::MAX);
            assert!(
                with_cache
                    .state_cache
                    .as_ref()
                    .is_some_and(|c| c.has_batch_states()),
                "huge budget must build interpolated states"
            );
            let mut actual_cached = Mat::zeros(n, 2);
            assert!(with_cache.apply_prepared_jacobian_into(
                direction.as_ref(),
                actual_cached.as_mut(),
                RowEpilogue::None,
            ));
            assert_bits_eq_mat(
                actual_cached.as_ref(),
                expected.as_ref(),
                if with_boundary {
                    "prepared-with-cache boundary"
                } else {
                    "prepared-with-cache volume"
                },
            );

            // Interpolated cache disabled (budget 0): owned copy only.
            let mut without_cache = if with_boundary {
                problem
                    .tensor_residual_operator(&k)
                    .with_state_boundary(&terms)
            } else {
                problem.tensor_residual_operator(&k)
            };
            without_cache.prepare_linearization_with_limit(state.as_ref(), 0);
            assert!(
                without_cache
                    .state_cache
                    .as_ref()
                    .is_some_and(|c| !c.has_batch_states()),
                "zero budget must skip interpolated states"
            );
            let mut actual_fresh = Mat::zeros(n, 2);
            assert!(without_cache.apply_prepared_jacobian_into(
                direction.as_ref(),
                actual_fresh.as_mut(),
                RowEpilogue::None,
            ));
            assert_bits_eq_mat(
                actual_fresh.as_ref(),
                expected.as_ref(),
                if with_boundary {
                    "prepared-without-cache boundary vs unprepared"
                } else {
                    "prepared-without-cache volume vs unprepared"
                },
            );
            assert_bits_eq_mat(
                actual_fresh.as_ref(),
                actual_cached.as_ref(),
                if with_boundary {
                    "prepared-without-cache boundary vs with-cache"
                } else {
                    "prepared-without-cache volume vs with-cache"
                },
            );
        }
    }
}
