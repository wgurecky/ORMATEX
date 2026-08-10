use ormatex_sem_nd::{CellState, FluxKernel1D, LocalCtx};

/// Compressible Euler equations in conservative variables
/// `[rho, momentum, total_energy]`.
#[derive(Clone, Copy, Debug)]
pub struct Euler1D {
    pub gamma: f64,
}

impl Euler1D {
    pub fn new(gamma: f64) -> Self {
        assert!(gamma > 1.0, "Euler gamma must be greater than one");
        Self { gamma }
    }

    fn primitive(&self, state: &CellState, q: usize) -> (f64, f64, f64) {
        let rho = state.value(0, q);
        let momentum = state.value(1, q);
        let energy = state.value(2, q);
        assert!(rho > 0.0, "Euler density must be positive");
        let velocity = momentum / rho;
        let pressure = (self.gamma - 1.0) * (energy - 0.5 * momentum * momentum / rho);
        (rho, velocity, pressure)
    }
}

impl FluxKernel1D for Euler1D {
    fn nfields(&self) -> usize {
        3
    }

    fn flux(&self, _ctx: &LocalCtx, state: &CellState, equation: usize, q: usize) -> f64 {
        let momentum = state.value(1, q);
        let energy = state.value(2, q);
        let (_, velocity, pressure) = self.primitive(state, q);
        match equation {
            0 => momentum,
            1 => momentum * velocity + pressure,
            2 => (energy + pressure) * velocity,
            _ => panic!("Euler equation index out of range"),
        }
    }

    fn flux_jacobian(
        &self,
        _ctx: &LocalCtx,
        state: &CellState,
        equation: usize,
        unknown: usize,
        q: usize,
    ) -> f64 {
        let rho = state.value(0, q);
        let momentum = state.value(1, q);
        let energy = state.value(2, q);
        let (_, velocity, pressure) = self.primitive(state, q);
        let gamma = self.gamma;
        match (equation, unknown) {
            (0, 0) => 0.0,
            (0, 1) => 1.0,
            (0, 2) => 0.0,
            (1, 0) => 0.5 * (gamma - 3.0) * velocity * velocity,
            (1, 1) => (3.0 - gamma) * velocity,
            (1, 2) => gamma - 1.0,
            (2, 0) => {
                (gamma - 1.0) * momentum * momentum * momentum / (2.0 * rho.powi(3))
                    - (energy + pressure) * momentum / rho.powi(2)
            }
            (2, 1) => (energy + pressure) / rho - (gamma - 1.0) * momentum * momentum / rho.powi(2),
            (2, 2) => gamma * velocity,
            _ => panic!("Euler block index out of range"),
        }
    }
}
