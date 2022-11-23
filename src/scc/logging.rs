use crate::utils::Timer;
use log::{debug, error, info, log_enabled, trace, warn, Level};
use ndarray::ArrayView1;

pub fn print_scc_init(max_iter: usize, temperature: f64, rep_energy: f64) {
    info!("{:^80}", "");
    info!("{: ^80}", "SCC-Routine");
    info!("{:-^80}", "");
    //info!("{: <25} {}", "convergence criterium:", scf_conv);
    info!("{: <25} {}", "max. iterations:", max_iter);
    info!("{: <25} {} K", "electronic temperature:", temperature);
    info!("{: <25} {:.14} Hartree", "repulsive energy:", rep_energy);
    info!("{:^80}", "");
    info!(
        "{: <45} ",
        "SCC Iterations: all quantities are in atomic units"
    );
    info!("{:-^75} ", "");
    info!(
        "{: <5} {: >18} {: >18} {: >18} {: >12}",
        "Iter.", "SCC Energy", "Energy diff.", "dq diff.", "Lvl. shift"
    );
    info!("{:-^75} ", "");
}

pub fn print_charges(q: ArrayView1<f64>, dq: ArrayView1<f64>) {
    debug!("");
    debug!("{: <35} ", "atomic charges and partial charges");
    debug!("{:-^35}", "");
    for (idx, (qi, dqi)) in q.iter().zip(dq.iter()).enumerate() {
        debug!("Atom {: >4} q: {:>18.14} dq: {:>18.14}", idx + 1, qi, dqi);
    }
    debug!("{:-^55}", "");
}

pub fn print_energies_at_iteration(
    iter: usize,
    scf_energy: f64,
    rep_energy: f64,
    energy_old: f64,
    dq_diff_max: f64,
    ls_weight: f64,
) {
    if iter == 0 {
        info!(
            "{: >5} {:>18.10e} {:>18.13} {:>18.10e} {:>12.4}",
            iter + 1,
            scf_energy + rep_energy,
            0.0,
            dq_diff_max,
            ls_weight
        );
    } else {
        info!(
            "{: >5} {:>18.10e} {:>18.10e} {:>18.10e} {:>12.4}",
            iter + 1,
            scf_energy + rep_energy,
            energy_old - scf_energy,
            dq_diff_max,
            ls_weight
        );
    }
}

pub fn print_energies_at_iteration_unrestricted(
    iter: usize,
    scf_energy: f64,
    rep_energy: f64,
    energy_old: f64,
    dq_diff_max_alpha: f64,
    dq_diff_max_beta: f64,
) {
    if iter == 0 {
        info!(
            "{: >5} {:>18.10e} {:>18.13} {:>18.10e} {:>18.10e}",
            iter + 1,
            scf_energy + rep_energy,
            0.0,
            dq_diff_max_alpha,
            dq_diff_max_beta
        );
    } else {
        info!(
            "{: >5} {:>18.10e} {:>18.10e} {:>18.10e} {:>18.10e}",
            iter + 1,
            scf_energy + rep_energy,
            energy_old - scf_energy,
            dq_diff_max_alpha,
            dq_diff_max_beta
        );
    }
}

pub fn print_scc_end(
    timer: Timer,
    jobtype: &str,
    scf_energy: f64,
    rep_energy: f64,
    orbe: ArrayView1<f64>,
    f: &[f64],
) {
    info!("{:-^75} ", "");
    info!("{: ^75}", "SCC converged");
    info!("{:^80} ", "");
    info!("final energy: {:18.14} Hartree", scf_energy + rep_energy);
    info!("{:-<80} ", "");
    info!("{}", timer);
    if jobtype == "sp" {
        print_orbital_information(orbe.view(), &f);
    }
}

pub fn print_unrestricted_scc_end(
    timer: Timer,
    jobtype: &str,
    scf_energy: f64,
    rep_energy: f64,
    orbe_alpha: ArrayView1<f64>,
    f_alpha: &[f64],
    orbe_beta: ArrayView1<f64>,
    f_beta: &[f64],
) {
    info!("{:-^75} ", "");
    info!("{: ^75}", "SCC converged");
    info!("{:^80} ", "");
    info!("final energy: {:18.14} Hartree", scf_energy + rep_energy);
    info!("{:-<80} ", "");
    info!("{}", timer);
    if jobtype == "sp" {
        info!("Information about alpha orbitals");
        print_orbital_information(orbe_alpha.view(), &f_alpha);
        info!("Information about beta orbitals");
        print_orbital_information(orbe_beta.view(), &f_beta);
    }
}

pub fn print_orbital_information(orbe: ArrayView1<f64>, f: &[f64]) -> () {
    info!("{:^80} ", "");
    info!(
        "{:^8} {:^6} {:>18.14} | {:^8} {:^6} {:>18.14}",
        "Orb.", "Occ.", "Energy/Hartree", "Orb.", "Occ.", "Energy/Hartree"
    );
    info!("{:-^71} ", "");
    let n_orbs: usize = orbe.len();
    for i in (0..n_orbs).step_by(2) {
        if i + 1 < n_orbs {
            info!(
                "MO:{:>5} {:>6} {:>18.14} | MO:{:>5} {:>6} {:>18.14}",
                i + 1,
                f[i],
                orbe[i],
                i + 2,
                f[i + 1],
                orbe[i + 1]
            );
        } else {
            info!("MO:{:>5} {:>6} {:>18.14} |", i + 1, f[i], orbe[i]);
        }
    }
    info!("{:-^71} ", "");
}
