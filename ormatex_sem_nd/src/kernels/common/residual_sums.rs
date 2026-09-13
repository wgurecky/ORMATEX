//! Dynamically dispatched sums and sets of residual kernels.
use super::traits::{ResidualKernel, TensorResidualKernel};
use crate::common::{CellState, LocalCtx, TensorCtx};

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

/// Named composition for terms with different local field selections.
///
/// The union state is interpolated once by the SEM layer. Each child receives
/// a zero-copy local view through `CellState::field_indices`, so existing
/// compact kernels keep their original positional field contracts.
pub struct ResidualKernelSet<'a> {
    kernels: Vec<Box<dyn ResidualKernel + Send + Sync + 'a>>,
    fields: Vec<String>,
    input_maps: Vec<Vec<usize>>,
    output_maps: Vec<Vec<usize>>,
}

impl<'a> ResidualKernelSet<'a> {
    pub fn from_kernel<K>(kernel: K) -> Self
    where
        K: ResidualKernel + Send + Sync + 'a,
    {
        Self::new(vec![Box::new(kernel)])
    }

    pub fn new(kernels: Vec<Box<dyn ResidualKernel + Send + Sync + 'a>>) -> Self {
        assert!(
            !kernels.is_empty(),
            "residual kernel set must contain a kernel"
        );
        let mut set = Self {
            kernels: Vec::new(),
            fields: Vec::new(),
            input_maps: Vec::new(),
            output_maps: Vec::new(),
        };
        for kernel in kernels {
            set.push(kernel);
        }
        set
    }

    pub fn with<K>(mut self, kernel: K) -> Self
    where
        K: ResidualKernel + Send + Sync + 'a,
    {
        self.push(Box::new(kernel));
        self
    }

    fn push(&mut self, kernel: Box<dyn ResidualKernel + Send + Sync + 'a>) {
        let input_names = kernel
            .input_field_names()
            .expect("heterogeneous residual terms require named input fields");
        let output_names = kernel
            .output_field_names()
            .expect("heterogeneous residual terms require named output fields");
        assert_eq!(input_names.len(), kernel.input_nfields());
        assert_eq!(output_names.len(), kernel.output_nfields());
        let input_maps = input_names
            .iter()
            .map(|name| self.union_field(name))
            .collect();
        let output_maps = output_names
            .iter()
            .map(|name| self.union_field(name))
            .collect();
        self.kernels.push(kernel);
        self.input_maps.push(input_maps);
        self.output_maps.push(output_maps);
    }

    fn union_field(&mut self, name: &str) -> usize {
        if let Some(index) = self.fields.iter().position(|field| field == name) {
            index
        } else {
            let index = self.fields.len();
            self.fields.push(name.to_owned());
            index
        }
    }

    fn child_state<'s>(state: &'s CellState<'s>, map: &'s [usize]) -> CellState<'s> {
        CellState {
            nfields: map.len(),
            npts: state.npts,
            gdim: state.gdim,
            values: state.values,
            grads: state.grads,
            field_indices: map,
        }
    }
}

impl ResidualKernel for ResidualKernelSet<'_> {
    fn nfields(&self) -> usize {
        self.fields.len()
    }

    fn field_names(&self) -> Option<Vec<String>> {
        Some(self.fields.clone())
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
            .zip(&self.output_maps)
            .zip(&self.input_maps)
            .map(|((kernel, output_map), input_map)| {
                output_map
                    .iter()
                    .position(|&field| field == equation)
                    .map_or(0.0, |local_equation| {
                        let child = Self::child_state(state, input_map);
                        kernel.residual_integrand(ctx, &child, local_equation, q, test_i)
                    })
            })
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
            .zip(&self.output_maps)
            .zip(&self.input_maps)
            .map(|((kernel, output_map), input_map)| {
                let Some(local_equation) = output_map.iter().position(|&field| field == equation)
                else {
                    return 0.0;
                };
                let Some(local_unknown) = input_map.iter().position(|&field| field == unknown)
                else {
                    return 0.0;
                };
                let child = Self::child_state(state, input_map);
                kernel.jacobian_integrand(
                    ctx,
                    &child,
                    local_equation,
                    local_unknown,
                    q,
                    test_i,
                    trial_i,
                )
            })
            .sum()
    }
}

/// Named tensor composition for terms with different local field selections.
pub struct TensorResidualKernelSet<'a, const GDIM: usize> {
    kernels: Vec<Box<dyn TensorResidualKernel<GDIM> + 'a>>,
    fields: Vec<String>,
    input_maps: Vec<Vec<usize>>,
    output_maps: Vec<Vec<usize>>,
}

impl<'a, const GDIM: usize> TensorResidualKernelSet<'a, GDIM> {
    pub fn from_kernel<K>(kernel: K) -> Self
    where
        K: TensorResidualKernel<GDIM> + 'a,
    {
        Self::new(vec![Box::new(kernel)])
    }

    pub fn new(kernels: Vec<Box<dyn TensorResidualKernel<GDIM> + 'a>>) -> Self {
        assert!(
            !kernels.is_empty(),
            "tensor residual kernel set must contain a kernel"
        );
        let mut set = Self {
            kernels: Vec::new(),
            fields: Vec::new(),
            input_maps: Vec::new(),
            output_maps: Vec::new(),
        };
        for kernel in kernels {
            set.push(kernel);
        }
        set
    }

    pub fn with<K>(mut self, kernel: K) -> Self
    where
        K: TensorResidualKernel<GDIM> + 'a,
    {
        self.push(Box::new(kernel));
        self
    }

    fn push(&mut self, kernel: Box<dyn TensorResidualKernel<GDIM> + 'a>) {
        let input_names = kernel
            .input_field_names()
            .expect("heterogeneous tensor terms require named input fields");
        let output_names = kernel
            .output_field_names()
            .expect("heterogeneous tensor terms require named output fields");
        assert_eq!(input_names.len(), kernel.input_nfields());
        assert_eq!(output_names.len(), kernel.output_nfields());
        let input_maps = input_names
            .iter()
            .map(|name| self.union_field(name))
            .collect();
        let output_maps = output_names
            .iter()
            .map(|name| self.union_field(name))
            .collect();
        self.kernels.push(kernel);
        self.input_maps.push(input_maps);
        self.output_maps.push(output_maps);
    }

    fn union_field(&mut self, name: &str) -> usize {
        if let Some(index) = self.fields.iter().position(|field| field == name) {
            index
        } else {
            let index = self.fields.len();
            self.fields.push(name.to_owned());
            index
        }
    }
}

impl<const GDIM: usize> TensorResidualKernel<GDIM> for TensorResidualKernelSet<'_, GDIM> {
    fn nfields(&self) -> usize {
        self.fields.len()
    }

    fn field_names(&self) -> Option<Vec<String>> {
        Some(self.fields.clone())
    }

    fn tensor_residual(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let mut result = [0.0; 3];
        for ((kernel, output_map), input_map) in self
            .kernels
            .iter()
            .zip(&self.output_maps)
            .zip(&self.input_maps)
        {
            let Some(local_equation) = output_map.iter().position(|&field| field == equation)
            else {
                continue;
            };
            let child = CellState {
                nfields: input_map.len(),
                npts: state.npts,
                gdim: state.gdim,
                values: state.values,
                grads: state.grads,
                field_indices: input_map,
            };
            let contribution = kernel.tensor_residual(ctx, &child, local_equation, q);
            result[0] += contribution[0];
            result[1] += contribution[1];
            result[2] += contribution[2];
        }
        result
    }

    fn tensor_jacobian_action(
        &self,
        ctx: &TensorCtx<'_>,
        state: &CellState<'_>,
        direction: &CellState<'_>,
        equation: usize,
        q: usize,
    ) -> [f64; 3] {
        let mut result = [0.0; 3];
        for ((kernel, output_map), input_map) in self
            .kernels
            .iter()
            .zip(&self.output_maps)
            .zip(&self.input_maps)
        {
            let Some(local_equation) = output_map.iter().position(|&field| field == equation)
            else {
                continue;
            };
            let child_state = CellState {
                nfields: input_map.len(),
                npts: state.npts,
                gdim: state.gdim,
                values: state.values,
                grads: state.grads,
                field_indices: input_map,
            };
            let child_direction = CellState {
                nfields: input_map.len(),
                npts: direction.npts,
                gdim: direction.gdim,
                values: direction.values,
                grads: direction.grads,
                field_indices: input_map,
            };
            let contribution = kernel.tensor_jacobian_action(
                ctx,
                &child_state,
                &child_direction,
                local_equation,
                q,
            );
            result[0] += contribution[0];
            result[1] += contribution[1];
            result[2] += contribution[2];
        }
        result
    }
}
