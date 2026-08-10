//! Spectral-element finite-element infrastructure built on `ndmesh`.

pub mod common;
pub mod gmsh;
pub mod jacobian;
pub mod kernels;
pub mod material;
pub mod problem_1d;
pub mod problem_2d;

pub use common::*;
pub use gmsh::*;
pub use jacobian::*;
pub use kernels::*;
pub use material::*;
pub use problem_1d::*;
pub use problem_2d::*;
