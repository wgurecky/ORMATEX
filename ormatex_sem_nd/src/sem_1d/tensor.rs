//! Sum-factorized tensor-product (`TensorResidualKernel`) paths (1D).
//!
//! Thin adapter over the shared [`crate::common::tensor_pass`] skeleton:
//! volume assembly iterates the precomputed flat `TensorBatchPlan` (natural
//! cell order, no color grouping) in a single Rayon region, scattering
//! lane-packed local actions into a row-sorted E-vector. Geometry is 1D
//! (`gdim = 1`, scalar packed `jinv`, `f1y` ignored); the sorted E-vector,
//! state cache, scratch, and phase-2 reduction are shared with the 2D path.
use std::sync::{Arc, Mutex, RwLock};

use crate::common::batch::{SortedRowScatter, TensorBatchPlan, TensorStateCache};
use crate::common::jacobian_pattern::JacobianPatternCache;
use crate::common::tensor_pass::{
    self, integrate_packed_1d, interpolate_packed_1d, TensorPhase1Ctx, TensorVolumeAdapter,
};
use crate::common::{CellData, ElementRestriction, FieldDofLayout, TensorCtx};
use crate::fields::FieldRegistry;
use crate::kernels::common::TensorResidualKernel;
use faer::prelude::*;
use faer::sparse::SparseColMat;

use ndelement::types::ReferenceCellType;
use ndmesh::traits::Mesh;

use super::problem::SEM1DProblem;

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> TensorVolumeAdapter<1>
    for SEM1DProblem<M>
{
    fn cell_data(&self) -> &CellData {
        &self.cell_data
    }
    fn restriction(&self) -> &ElementRestriction {
        &self.restriction
    }
    fn batch_plan(&self) -> &TensorBatchPlan {
        &self.tensor_batch_plan
    }
    fn evec_pool(&self) -> &Mutex<Vec<Vec<f64>>> {
        &self.evec_pool
    }
    fn scatter_cache(&self) -> &RwLock<Vec<(Vec<usize>, Arc<SortedRowScatter>)>> {
        &self.sorted_scatter_cache
    }
    fn pattern_cache(&self) -> &JacobianPatternCache {
        &self.jacobian_pattern_cache
    }
    fn fields(&self) -> &FieldRegistry {
        &self.fields
    }
    fn tensor_ctx(&self, time: f64, cell: usize) -> TensorCtx<'_> {
        self.tensor_ctx(time, cell)
    }
    fn interpolate_packed(
        cd: &CellData,
        ninputs: usize,
        coeffs: &[f64],
        values: &mut [f64],
        grads: &mut [f64],
        jinv: &[f64],
    ) {
        interpolate_packed_1d(cd, ninputs, coeffs, values, grads, jinv);
    }
    fn integrate_packed(
        cd: &CellData,
        noutputs: usize,
        f0: &[f64],
        f1x: &[f64],
        f1y: &[f64],
        out: &mut [f64],
        wdet: &[f64],
        jinv: &[f64],
    ) {
        integrate_packed_1d(cd, noutputs, f0, f1x, f1y, out, wdet, jinv);
    }
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64>> SEM1DProblem<M> {
    /// Sum-factorized tensor residual with a caller-provided field layout.
    ///
    /// Mathematics: `R(U) = Σ_e E_eᵀ r_e` with 1D sum-factorized GLL
    /// interpolation/integration (`O(p²)` per interval) and
    /// SIMD-over-element batching. `layout` avoids recomputation when the
    /// caller already holds it; external callers should use
    /// [`TensorResidualOps::assemble_tensor_residual`][crate::sem_traits::TensorResidualOps].
    pub(crate) fn assemble_tensor_residual_with_layout<K: TensorResidualKernel<1> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        layout: &FieldDofLayout,
    ) -> Vec<f64>
    where
        M: Sync,
    {
        tensor_pass::assemble_tensor_residual_with_layout(self, time, kernel, state, layout)
    }

    pub(crate) fn assemble_tensor_jacobian_with_layout<K: TensorResidualKernel<1> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        layout: &FieldDofLayout,
    ) -> SparseColMat<usize, f64>
    where
        M: Sync,
    {
        tensor_pass::assemble_tensor_jacobian_with_layout(self, time, kernel, state, layout)
    }
    /// Build an owned linearization-state cache with an explicit byte budget.
    ///
    /// Always copies the state vector; the per-batch lane-packed values+grads
    /// are only built when their footprint ([`tensor_interpolated_state_bytes`])
    /// fits `limit_bytes`. Above the budget the prepared path re-interpolates
    /// from the owned copy instead, which stays bit-identical while avoiding
    /// last-level-cache thrash. Used by `prepare_linearization_with_limit`;
    /// `apply` reuses the interpolated states only on match, and never when
    /// they were not built. The budget is passed directly instead of read
    /// from the environment. Crate-internal so tests can force the threshold
    /// (e.g. `0` to skip the interpolated states) without touching the
    /// process-wide env-var cache.
    ///
    /// # Arguments
    /// * `kernel` - tensor kernel selecting input fields.
    /// * `state` - linearization point (`N×1`).
    /// * `limit_bytes` - interpolated-state budget in bytes; `0` disables the
    ///   interpolated states (only the owned state copy is kept).
    ///
    /// # Returns
    /// Owned `TensorStateCache`, with per-batch lane-packed states only when
    /// their footprint fits `limit_bytes`.
    pub(crate) fn build_tensor_state_cache_with_limit<K: TensorResidualKernel<1> + Sync>(
        &self,
        kernel: &K,
        state: MatRef<f64>,
        limit_bytes: u64,
    ) -> TensorStateCache
    where
        M: Sync,
    {
        let layout = self.field_layout();
        let selection = self.fields.resolve_selection(
            kernel.input_nfields(),
            kernel.input_field_names(),
            kernel.output_nfields(),
            kernel.output_field_names(),
            "tensor residual kernel",
        );
        let ninputs = selection.inputs.len();
        let input_offsets: Vec<_> = selection
            .inputs
            .iter()
            .map(|&field| layout.offsets[field])
            .collect();
        let sources: Vec<usize> = selection
            .inputs
            .iter()
            .map(|&g| self.restriction.field_map_source(g))
            .collect();
        tensor_pass::build_tensor_state_cache_with_limit(
            self,
            kernel,
            state,
            limit_bytes,
            layout.total_size,
            &input_offsets,
            &sources,
            ninputs,
        )
    }

    pub(crate) fn apply_tensor_jacobian_with_layout<K: TensorResidualKernel<1> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        layout: &FieldDofLayout,
        out: MatMut<'_, f64>,
    ) where
        M: Sync,
    {
        self.apply_tensor_jacobian_with_layout_cached(
            time, kernel, state, direction, layout, out, None,
        )
    }

    pub(crate) fn apply_tensor_jacobian_with_layout_cached<K: TensorResidualKernel<1> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        layout: &FieldDofLayout,
        out: MatMut<'_, f64>,
        cache: Option<&TensorStateCache>,
    ) where
        M: Sync,
    {
        tensor_pass::apply_tensor_jacobian_with_layout_cached(
            self, time, kernel, state, direction, layout, out, cache,
        )
    }

    /// Fetch the cached row-sorted scatter table for one output selection,
    /// building and caching it on first use.
    ///
    /// # Arguments
    /// * `selection_outputs` - global field ids receiving the scatter.
    ///
    /// # Returns
    /// Shared `SortedRowScatter` (slots ordered by row, then `(color, pos, local)`).
    pub(crate) fn sorted_scatter_for(
        &self,
        selection_outputs: &[usize],
    ) -> std::sync::Arc<SortedRowScatter> {
        tensor_pass::sorted_scatter_for(self, selection_outputs)
    }

    /// Take a reusable row-sorted E-vector buffer of exactly `len` doubles.
    ///
    /// # Arguments
    /// * `len` - required buffer length (`nslots * ncols`).
    ///
    /// # Returns
    /// Pooled buffer (resized if reused at a different length), or fresh.
    pub(crate) fn acquire_evec(&self, len: usize) -> Vec<f64> {
        tensor_pass::acquire_evec(self, len)
    }

    /// Return an E-vector buffer to the pool (capped to bound memory).
    ///
    /// # Arguments
    /// * `buf` - buffer to recycle.
    pub(crate) fn release_evec(&self, buf: Vec<f64>) {
        tensor_pass::release_evec(self, buf)
    }

    /// Phase-1 entry for the prepared operator paths in `super::operators`.
    ///
    /// Runs the single lane-packed phase-1 pass over the bundled arguments.
    ///
    /// # Arguments
    /// * `kernel` - tensor residual kernel.
    /// * `ctx` - bundled phase-1 arguments (see [`TensorPhase1Ctx`]).
    pub(crate) fn phase1_fill_sorted_dispatch<K>(&self, kernel: &K, ctx: TensorPhase1Ctx<'_, '_>)
    where
        K: TensorResidualKernel<1> + Sync,
        M: Sync,
    {
        tensor_pass::phase1_fill_sorted(self, kernel, ctx);
    }
}

impl<M: Mesh<EntityDescriptor = ReferenceCellType, T = f64> + Sync>
    crate::sem_traits::TensorResidualOps<1> for SEM1DProblem<M>
{
    fn assemble_tensor_residual<K: TensorResidualKernel<1> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> Vec<f64> {
        let layout = self.field_layout();
        self.assemble_tensor_residual_with_layout(time, kernel, state, &layout)
    }

    fn assemble_tensor_jacobian<K: TensorResidualKernel<1> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
    ) -> SparseColMat<usize, f64> {
        let layout = self.field_layout();
        self.assemble_tensor_jacobian_with_layout(time, kernel, state, &layout)
    }

    fn apply_tensor_jacobian_into<K: TensorResidualKernel<1> + Sync>(
        &self,
        time: f64,
        kernel: &K,
        state: MatRef<f64>,
        direction: MatRef<f64>,
        out: MatMut<'_, f64>,
    ) {
        let layout = self.field_layout();
        self.apply_tensor_jacobian_with_layout(time, kernel, state, direction, &layout, out);
    }
}
