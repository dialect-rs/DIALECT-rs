pub(crate) use dialect_base::array_helper;
pub mod tests;

pub use dialect_base::array_helper::ToOwnedF;
pub use dialect_base::get_path_prefix;
// pub use zbrent::zbrent;


pub enum Calculation {
    Converged,
    NotConverged,
}
pub use dialect_base::Timer;

