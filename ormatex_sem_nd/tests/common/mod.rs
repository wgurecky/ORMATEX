//! Shared test-support module for lane-packed (`LANES` = 8) kernel tests.
//!
//! Other integration tests include this module with:
//!
//! ```ignore
//! mod common;
//! use common::lanes as lanes;
//! ```
//!
//! (Each file directly under `tests/` is its own crate, so `mod common;`
//! resolves to this directory exactly like a `#[path]` include.)
pub mod lanes;
