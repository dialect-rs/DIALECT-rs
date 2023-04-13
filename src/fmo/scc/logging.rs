use crate::utils::Timer;
use log::info;

pub fn fmo_scc_init(max_iter: usize) {
    info!("{:^80}", "");
    info!("{: ^80}", "FMO SCC-Routine");
    info!("{:-^80}", "");
    //info!("{: <25} {}", "convergence criterium:", scf_conv);
    info!("{: <25} {}", "max. iterations:", max_iter);
    info!("{:^80}", "");
    info!("{: <45} ", "Monomer SCC Iterations:");
    info!("{:-^45} ", "");
    info!(
        "{: <5} {: >18} {: >18}",
        "Iter.", "#conv. Monomers", "#Monomers"
    );
    info!("{:-^75} ", "");
}

pub fn fmo_monomer_iteration(iter: usize, n_converged: usize, n_total: usize) {
    info!("{: >5} {:>18} {:>18}", iter + 1, n_converged, n_total);
}

pub fn fmo_scc_end(
    timer: Timer,
    e_monomer: f64,
    e_pairs: f64,
    e_emb: f64,
    e_esd: f64,
    e_disp: f64,
) {
    info!("{:-^75} ", "");
    info!("{: ^75}", "FMO SCC converged");
    info!("{:^80} ", "");
    info!(
        "{:<29} {:>24.14} Hartree",
        "sum of monomer energies:", e_monomer
    );
    info!("{:<29} {:>24.14} Hartree", "sum of pair energies:", e_pairs);
    info!(
        "{:<29} {:>24.14} Hartree",
        "sum of embedding energies:", e_emb
    );
    info!(
        "{:<29} {:>24.14} Hartree",
        "sum of ESD pair energies", e_esd
    );
    info!(
        "{:<29} {:>24.14} Hartree",
        "sum of D3 dispersion energy:", e_disp
    );
    info!("");
    info!(
        "{:<29} {:>24.14} Hartree",
        "total energy:",
        e_monomer + e_pairs + e_emb + e_esd + e_disp
    );
    info!("{:-<80} ", "");
    info!("{}", timer);
}
