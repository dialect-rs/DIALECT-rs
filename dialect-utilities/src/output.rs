//! Shared dynamics-timing log output.

use log::warn;

pub fn print_dyn_dftb(
    system_time: f32,
    energy_gradient_time: f32,
    nacme_time: f32,
    full_time: f32,
) {
    warn!("{:^85}", "");
    warn!("{: ^85}", "Electronic Structure Timings");
    warn!("{:-^85}", "");
    warn!("{:>73} {:>8.2} s", "system preparation time:", system_time);
    warn!(
        "{:>73} {:>8.2} s",
        "energy and gradient time:",
        energy_gradient_time - system_time
    );
    warn!(
        "{:>73} {:>8.2} s",
        "NACME time:",
        nacme_time - energy_gradient_time
    );
    warn!("{:>73} {:>8.2} s", "full time:", full_time);
}
