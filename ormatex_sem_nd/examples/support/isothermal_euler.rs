use ormatex_sem_nd::{CellState, FluxKernel1D, LocalCtx};

/// Isothermal Euler equations in primitive variables `[u, rho]`.
#[derive(Clone, Copy, Debug)]
pub struct IsothermalEuler1D {
    pub sound_speed: f64,
}

impl IsothermalEuler1D {
    pub fn new(sound_speed: f64) -> Self {
        assert!(sound_speed > 0.0, "sound speed must be positive");
        Self { sound_speed }
    }

    fn density(&self, state: &CellState, q: usize) -> f64 {
        let rho = state.value(1, q);
        assert!(rho > 0.0, "isothermal Euler density must be positive");
        rho
    }
}

impl FluxKernel1D for IsothermalEuler1D {
    fn nfields(&self) -> usize {
        2
    }

    fn flux(&self, _ctx: &LocalCtx, state: &CellState, equation: usize, q: usize) -> f64 {
        let u = state.value(0, q);
        let rho = self.density(state, q);
        let cs2 = self.sound_speed * self.sound_speed;
        match equation {
            0 => 0.5 * u * u + cs2 * rho.ln(),
            1 => rho * u,
            _ => panic!("isothermal Euler equation index out of range"),
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
        let u = state.value(0, q);
        let rho = self.density(state, q);
        let cs2 = self.sound_speed * self.sound_speed;
        match (equation, unknown) {
            (0, 0) => u,
            (0, 1) => cs2 / rho,
            (1, 0) => rho,
            (1, 1) => u,
            _ => panic!("isothermal Euler block index out of range"),
        }
    }
}
