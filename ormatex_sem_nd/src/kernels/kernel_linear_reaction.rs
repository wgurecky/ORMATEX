use faer::sparse::SparseColMat;

use crate::common::{CellState, LocalCtx};

use super::kernel_common::{BilinearForm, ResidualKernel};

/// Sparse linear reaction kernel for `du/dt = rates * u`.
///
/// The matrix is indexed as `(equation, unknown)`. Missing sparse entries are
/// treated as zero, and each entry is coupled through the scalar FEM mass form.
pub struct KernelLinearReaction {
    interactions: Vec<Vec<(usize, f64)>>,
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
        Self { interactions }
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
