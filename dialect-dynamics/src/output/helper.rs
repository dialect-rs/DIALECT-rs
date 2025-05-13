use log::warn;

pub fn print_init_electronic_structure() {
    warn!("{: ^85}", "Electronic Structure Step");
    warn!("{:-^85}", "");
}

pub fn print_footer_electronic_structure(timing: f64) {
    warn!("{:-<85} ", "");
    warn!(
        "{:>73} {:>8.2} s",
        "Electronic Structure Step finished in", timing
    );
}

pub fn print_header_dynamics_step() {
    warn!("{:^90}", "");
    warn!("{:^90}", "");
    warn!("{: ^90}", "Molecular Dynamics Step");
    warn!("{:-^90}", "");
}

pub fn print_footer_dynamics(timing: f64) {
    warn!("{:-<90} ", "");
    warn!(
        "{:>78} {:>8.2} s",
        "Molecular Dynamics Step finished in", timing
    );
}
