use crate::defaults::{EXTERNAL_DIR, USE_EXTERNAL};
use serde::{Deserialize, Serialize};

fn default_use_external_path() -> bool {
    USE_EXTERNAL
}
fn default_skf_directory() -> String {
    String::from(EXTERNAL_DIR)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ParameterizationConfig {
    #[serde(default = "default_use_external_path")]
    pub use_external_path: bool,
    #[serde(default = "default_skf_directory")]
    pub skf_directory: String,
}
