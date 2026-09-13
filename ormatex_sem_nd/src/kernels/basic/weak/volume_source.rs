use crate::common::{CellState, LocalCtx};
use crate::fields::FieldRegistry;

use crate::kernels::common::{LinearForm, ResidualKernel};

/// Constant volumetric source `f(x) = val`.
///
/// Sign convention: `LinearForm` contributes `+val*v` while `ResidualKernel`
/// contributes `-val*v` (sources move to the LHS); the tensor triple is
/// `(-val, 0, 0)` with zero action. Tensor mirror:
/// [`TensorKernelVolumeSource`](crate::kernels::basic::tensor::volume_source::TensorKernelVolumeSource).
pub struct KernelVolumeSource {
    pub val: f64,
    field_names: Option<Vec<String>>,
}

impl KernelVolumeSource {
    pub fn new(val: f64) -> Self {
        Self {
            val,
            field_names: None,
        }
    }

    /// Attach the single heated-field name so the source can join a named
    /// residual set (e.g. volumetric heating of `T` in a coupled solve).
    pub fn with_field_names<I, S>(val: f64, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut kernel = Self::new(val);
        let names = FieldRegistry::new(names);
        assert_eq!(
            names.len(),
            1,
            "volume-source field-name count must match the single heated field"
        );
        kernel.field_names = Some(names.names().to_vec());
        kernel
    }
}

impl LinearForm for KernelVolumeSource {
    fn integrand(&self, ctx: &LocalCtx, _equation: usize, q: usize, test_i: usize) -> f64 {
        assert_eq!(ctx.ncomp, 1, "KernelVolumeSource: scalar only (ncomp==1)");
        self.val * ctx.test(test_i, 0).v(q)
    }
}

impl ResidualKernel for KernelVolumeSource {
    fn field_names(&self) -> Option<Vec<String>> {
        self.field_names.clone()
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        _state: &CellState,
        _equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        -self.val * ctx.test(test_i, 0).v(q)
    }

    fn jacobian_integrand(
        &self,
        _ctx: &LocalCtx,
        _state: &CellState,
        _equation: usize,
        _unknown: usize,
        _q: usize,
        _test_i: usize,
        _trial_i: usize,
    ) -> f64 {
        0.0
    }
}
