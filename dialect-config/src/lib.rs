//! The dialect configuration schema: `Configuration` and all its serde
//! sub-config structs (`dialect.toml`), parsed defaults, plus the `AAType`
//! mixer selector. Sits below every driver, so the method driver stacks
//! can carry a `Configuration` without depending on the main crate.
#![allow(warnings)]

pub mod settings;
pub mod user_config;

pub use settings::Configuration;
pub use user_config::{load_dialect_config, parse_dialect_config, DIALECT_TOML_TEMPLATE};
