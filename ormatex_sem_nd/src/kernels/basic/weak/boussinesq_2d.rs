use crate::common::{CellState, LocalCtx};

use crate::kernels::common::ResidualKernel;

/// Boussinesq buoyancy contribution from `T` to the `u` and `v` equations.
///
/// Rectangular coupling (1 input `T`, 2 outputs `u`/`v`): momentum source
/// `-buoyancy*g*(T - T_ref)`. Tensor mirror:
/// [`TensorKernelBoussinesq2D`](crate::kernels::basic::tensor::boussinesq_2d::TensorKernelBoussinesq2D).
#[derive(Clone, Copy, Debug)]
pub struct KernelBoussinesq2D {
    pub buoyancy: f64,
    pub gravity: [f64; 2],
    pub reference_temperature: f64,
}

impl KernelBoussinesq2D {
    pub fn new(buoyancy: f64, gravity: [f64; 2]) -> Self {
        assert!(buoyancy.is_finite(), "buoyancy must be finite");
        assert!(
            gravity.iter().all(|value| value.is_finite()),
            "gravity must be finite"
        );
        Self {
            buoyancy,
            gravity,
            reference_temperature: 0.0,
        }
    }

    pub fn with_reference_temperature(mut self, value: f64) -> Self {
        assert!(value.is_finite(), "reference temperature must be finite");
        self.reference_temperature = value;
        self
    }

    pub(crate) fn temperature(&self, state: &CellState, q: usize) -> f64 {
        state.value(0, q) - self.reference_temperature
    }
}

impl ResidualKernel for KernelBoussinesq2D {
    fn nfields(&self) -> usize {
        2
    }

    fn field_names(&self) -> Option<Vec<String>> {
        None
    }

    fn input_nfields(&self) -> usize {
        1
    }

    fn output_nfields(&self) -> usize {
        2
    }

    fn input_field_names(&self) -> Option<Vec<String>> {
        Some(["T"].into_iter().map(str::to_owned).collect())
    }

    fn output_field_names(&self) -> Option<Vec<String>> {
        Some(["u", "v"].into_iter().map(str::to_owned).collect())
    }

    fn residual_integrand(
        &self,
        ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        q: usize,
        test_i: usize,
    ) -> f64 {
        assert_eq!(ctx.gdim, 2, "Boussinesq terms require gdim == 2");
        assert_eq!(state.nfields, 1, "Boussinesq state must contain [T]");
        assert!(equation < 2);
        -self.buoyancy
            * self.gravity[equation]
            * self.temperature(state, q)
            * ctx.test(test_i, 0).v(q)
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
        assert_eq!(ctx.gdim, 2, "Boussinesq terms require gdim == 2");
        assert_eq!(state.nfields, 1, "Boussinesq state must contain [T]");
        if equation >= 2 || unknown != 0 {
            return 0.0;
        }
        -self.buoyancy
            * self.gravity[equation]
            * ctx.trial(trial_i, 0).v(q)
            * ctx.test(test_i, 0).v(q)
    }
}
