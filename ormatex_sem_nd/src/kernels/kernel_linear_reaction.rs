use faer::sparse::SparseColMat;

use crate::common::{CellState, LocalCtx, TensorCtx};
use crate::fields::FieldRegistry;

use super::kernel_common::{BilinearForm, ResidualKernel, TensorResidualKernel};

/// Sparse linear reaction kernel for `du/dt = rates * u`.
///
/// The matrix is indexed as `(equation, unknown)`. Missing sparse entries are
/// treated as zero, and each entry is coupled through the scalar FEM mass form.
pub struct KernelLinearReaction {
    interactions: Vec<Vec<(usize, f64)>>,
    field_names: Option<Vec<String>>,
}

impl KernelLinearReaction {
    pub fn new(rates: SparseColMat<usize, f64>) -> Self {
        assert!(rates.nrows() > 0, "reaction matrix must contain a field");
        assert_eq!(
            rates.nrows(),
            rates.ncols(),
            "reaction matrix must be square"
        );
        let nfields = rates.nrows();
        let (symbolic, values) = rates.as_ref().parts();
        let columns = symbolic.col_ptr();
        let rows = symbolic.row_idx();
        let mut interactions = vec![Vec::new(); nfields];
        for column in 0..nfields {
            for entry in columns[column]..columns[column + 1] {
                let value = values[entry];
                if value != 0.0 {
                    interactions[rows[entry]].push((column, value));
                }
            }
        }
        Self {
            interactions,
            field_names: None,
        }
    }

    /// Attach the ordered solution-field names represented by the reaction matrix.
    pub fn with_field_names<I, S>(rates: SparseColMat<usize, f64>, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut kernel = Self::new(rates);
        let names = FieldRegistry::new(names);
        assert_eq!(
            names.len(),
            kernel.interactions.len(),
            "reaction field-name count must match the matrix"
        );
        kernel.field_names = Some(names.names().to_vec());
        kernel
    }

    fn coefficient(&self, equation: usize, unknown: usize) -> f64 {
        self.interactions[equation]
            .iter()
            .find_map(|&(field, value)| (field == unknown).then_some(value))
            .unwrap_or(0.0)
    }
}

impl BilinearForm for KernelLinearReaction {
    fn nfields(&self) -> usize {
        self.interactions.len()
    }

    fn field_names(&self) -> Option<Vec<String>> {
        self.field_names.clone()
    }

    fn supports_tensor_bilinear(&self) -> bool {
        true
    }

    fn supports_tensor_bilinear_1d(&self) -> bool {
        true
    }

    fn tensor_bilinear(
        &self,
        _ctx: &TensorCtx<'_>,
        equation: usize,
        unknown: usize,
        _q: usize,
        trial_value: f64,
        _trial_grad: [f64; 2],
    ) -> [f64; 3] {
        [-self.coefficient(equation, unknown) * trial_value, 0.0, 0.0]
    }

    fn integrand(
        &self,
        ctx: &LocalCtx,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelLinearReaction: scalar fields only");
        -self.coefficient(equation, unknown) * ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}

impl ResidualKernel for KernelLinearReaction {
    fn nfields(&self) -> usize {
        self.interactions.len()
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
        assert_eq!(ctx.ncomp, 1, "KernelLinearReaction: scalar fields only");
        let source = self.interactions[equation]
            .iter()
            .map(|&(unknown, coefficient)| coefficient * state.value(unknown, q))
            .sum::<f64>();
        -source * ctx.test(test_i, 0).v(q)
    }

    fn jacobian_integrand(
        &self,
        ctx: &LocalCtx,
        _state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
        test_i: usize,
        trial_i: usize,
    ) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelLinearReaction: scalar fields only");
        -self.coefficient(equation, unknown) * ctx.test(test_i, 0).v(q) * ctx.trial(trial_i, 0).v(q)
    }
}

/// Tensor-product linear reaction kernel.
pub struct TensorKernelLinearReaction(pub KernelLinearReaction);

impl TensorKernelLinearReaction {
    pub fn new(rates: SparseColMat<usize, f64>) -> Self {
        Self(KernelLinearReaction::new(rates))
    }
    pub fn with_field_names<I, S>(rates: SparseColMat<usize, f64>, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self(KernelLinearReaction::with_field_names(rates, names))
    }
}

impl<const GDIM: usize> TensorResidualKernel<GDIM> for TensorKernelLinearReaction {
    fn nfields(&self) -> usize {
        self.0.interactions.len()
    }
    fn field_names(&self) -> Option<Vec<String>> {
        self.0.field_names.clone()
    }
    fn tensor_residual(
        &self,
        _ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        [
            -self.0.interactions[equation]
                .iter()
                .map(|&(unknown, coefficient)| coefficient * state.value(unknown, q))
                .sum::<f64>(),
            0.0,
            0.0,
        ]
    }
    fn tensor_jacobian_action(
        &self,
        _ctx: &TensorCtx<'_>,
        _state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        [
            -self.0.interactions[equation]
                .iter()
                .map(|&(unknown, coefficient)| coefficient * direction.value(unknown, q))
                .sum::<f64>(),
            0.0,
            0.0,
        ]
    }
}
