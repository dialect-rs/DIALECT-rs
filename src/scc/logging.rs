use crate::utils::Timer;
use log::{debug, info};
use ndarray::ArrayView1;

pub fn print_scc_init(max_iter: usize, temperature: f64, rep_energy: f64) {
    info!("{:^80}", "");
    info!("{: ^80}", "SCC-Routine");
    info!("{:-^80}", "");
    info!("{: <25} {}", "max. iterations:", max_iter);
    info!("{: <25} {} K", "electronic temperature:", temperature);
    info!("{: <25} {:.14} Hartree", "repulsive energy:", rep_energy);
    info!("{:^80}", "");
    info!(
        "{: <45} ",
        "SCC Iterations: all quantities are in atomic units"
    );
    info!("{:-^62} ", "");
    info!(
        "{: <5} {: >18} {: >18} {: >18}",
        "Iter.", "SCC Energy", "Energy diff.", "dq diff."
    );
    info!("{:-^62} ", "");
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
) {
    if iter == 0 {
        info!(
            "{: >5} {:>18.10e} {:>18.13} {:>18.10e}",
            iter + 1,
            scf_energy + rep_energy,
            0.0,
            dq_diff_max,
        );
    } else {
        info!(
            "{: >5} {:>18.10e} {:>18.10e} {:>18.10e}",
            iter + 1,
            scf_energy + rep_energy,
            energy_old - scf_energy,
            dq_diff_max,
        );
    }
}

pub fn print_scc_end(
    timer: Timer,
    jobtype: &str,
    scf_energy: f64,
    rep_energy: f64,
    e_disp: f64,
    orbe: ArrayView1<f64>,
    f: &[f64],
) {
    info!("{:-^62} ", "");
    info!("{: ^62}", "SCC converged");
    info!("{:^80} ", "");
    info!("dftb energy: {:18.14} Hartree", scf_energy + rep_energy);
    info!("dispersion energy: {:18.14} Hartree", e_disp);
    info!(
        "Total energy: {:18.14} Hartree",
        scf_energy + rep_energy + e_disp
    );
    info!("{:-<80} ", "");
    info!("{}", timer);
    if jobtype == "sp" {
        print_orbital_information(orbe.view(), f);
    }
}

pub fn print_scc_end_xtb(
    timer: Timer,
    jobtype: &str,
    scf_energy: f64,
    rep_energy: f64,
    e_disp: f64,
    e_halogen: f64,
    orbe: ArrayView1<f64>,
    f: &[f64],
) {
    info!("{:-^62} ", "");
    info!("{: ^62}", "SCC converged");
    info!("{:^80} ", "");
    info!("electronic energy: {:18.14} Hartree", scf_energy);
    info!("halogen bond correction: {:18.14} Hartree", e_halogen);
    info!("repulsive energy: {:18.14} Hartree", rep_energy);
    info!("dispersion energy: {:18.14} Hartree", e_disp);
    info!(
        "Total energy: {:18.14} Hartree",
        scf_energy + rep_energy + e_disp
    );
    info!("{:-<80} ", "");
    info!("{}", timer);
    if jobtype == "sp" {
        print_orbital_information(orbe.view(), f);
    }
}

pub fn print_orbital_information(orbe: ArrayView1<f64>, f: &[f64]) {
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
                "MO:{:>5} {:>6.2} {:>18.14} | MO:{:>5} {:>6.2} {:>18.14}",
                i + 1,
                f[i],
                orbe[i],
                i + 2,
                f[i + 1],
                orbe[i + 1]
            );
        } else {
            info!("MO:{:>5} {:>6.2} {:>18.14} |", i + 1, f[i], orbe[i]);
        }
    }
    info!("{:-^71} ", "");
}
