use log::warn;

pub fn print_dyn_timings_ehrenfest(
    system_time: f32,
    scf_time: f32,
    gradient_time: f32,
    exc_time: f32,
    nacme_time: f32,
    full_time: f32,
) {
    warn!("{:^85}", "");
    warn!("{: ^85}", "Electronic Structure Timings");
    warn!("{:-^85}", "");
    warn!("{:>73} {:>8.2} s", "system preparation time:", system_time);
    warn!("{:>73} {:>8.2} s", "SCF time:", scf_time - system_time);
    warn!(
        "{:>73} {:>8.2} s",
        "gradient time:",
        gradient_time - exc_time
    );
    warn!(
        "{:>73} {:>8.2} s",
        "excited state time:",
        exc_time - scf_time
    );
    warn!(
        "{:>73} {:>8.2} s",
        "NACME time:",
        nacme_time - gradient_time
    );
    warn!("{:>73} {:>8.2} s", "full time:", full_time);
}

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
