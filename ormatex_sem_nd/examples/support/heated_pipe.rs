//! Analytic steady temperature for a 1D volumetrically heated pipe.
//!
//! Uniform velocity `u0`, thermal diffusivity `alpha`, uniform volumetric
//! source `q` (in temperature units, i.e. `Q / (rho * cp)`), fixed inlet
//! temperature and an adiabatic outlet (`dT/dx = 0` at `x = L`):
//!
//! ```text
//! u0 dT/dx = alpha d2T/dx2 + q,  T(0) = T_in,  dT/dx(L) = 0.
//! ```
//!
//! With `beta = u0 / alpha` the closed form is
//!
//! ```text
//! T(x) = T_in + (q / u0) x - (q alpha / u0^2) e^(-beta L) (e^(beta x) - 1).
//! ```

/// Parameters of the volumetrically heated 1D pipe benchmark.
#[derive(Clone, Copy, Debug)]
pub struct HeatedPipeParams {
    /// Pipe length.
    pub length: f64,
    /// Uniform inlet velocity.
    pub inlet_velocity: f64,
    /// Thermal diffusivity.
    pub thermal_diffusivity: f64,
    /// Volumetric source in temperature units.
    pub heat_source: f64,
    /// Fixed inlet temperature.
    pub inlet_temperature: f64,
}

impl HeatedPipeParams {
    pub fn analytic_temperature(&self, x: f64) -> f64 {
        let beta = self.inlet_velocity / self.thermal_diffusivity;
        let advected = self.heat_source / self.inlet_velocity * x;
        let layer = self.heat_source * self.thermal_diffusivity
            / (self.inlet_velocity * self.inlet_velocity)
            * (-beta * self.length).exp()
            * ((beta * x).exp() - 1.0);
        self.inlet_temperature + advected - layer
    }
}
