use crate::defaults;
use crate::initialization::parameters::RepulsivePotential;
use crate::initialization::Atom;
use itertools::Itertools;
use log::debug;
use ndarray::prelude::*;

// find indices of HOMO and LUMO orbitals (starting from 0)
pub fn get_frontier_orbitals(n_elec: usize) -> (usize, usize) {
    let homo: usize = (n_elec / 2) - 1;
    let lumo: usize = homo + 1;
    (homo, lumo)
}

// find indices of HOMO and LUMO orbitals (starting from 0)
pub fn get_frontier_orbitals_from_occ(f: &[f64]) -> (usize, usize) {
    let n_occ: usize = f
        .iter()
        .enumerate()
        .filter_map(|(idx, val)| if *val > 0.5 { Some(idx) } else { None })
        .collect::<Vec<usize>>()
        .len();
    let homo: usize = n_occ - 1;
    let lumo: usize = homo + 1;
    (homo, lumo)
}

// compute HOMO-LUMO gap in Hartree
pub fn get_homo_lumo_gap(orbe: ArrayView1<f64>, homo_lumo_idx: (usize, usize)) -> f64 {
    orbe[homo_lumo_idx.1] - orbe[homo_lumo_idx.0]
}

/// Compute energy due to core electrons and nuclear repulsion
pub fn get_repulsive_energy(atoms: &[Atom], n_atoms: usize, v_rep: &RepulsivePotential) -> f64 {
    let mut e_nuc: f64 = 0.0;
    for (i, atomi) in atoms[1..n_atoms].iter().enumerate() {
        for atomj in atoms[0..i + 1].iter() {
            let r: f64 = (atomi - atomj).norm();
            // nucleus-nucleus and core-electron repulsion
            e_nuc += v_rep.get(atomi.kind, atomj.kind).spline_eval(r);
        }
    }
    e_nuc
}

/// the repulsive potential, the dispersion correction and only depend on the nuclear
/// geometry and do not change during the SCF cycle
fn get_nuclear_energy() {}

/// Compute electronic energies
pub fn get_electronic_energy(
    p: ArrayView2<f64>,
    p0: ArrayView2<f64>,
    s: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
    g0_lr_ao: Option<ArrayView2<f64>>,
) -> f64 {
    // band structure energy
    let e_band_structure: f64 = (&p * &h0).sum();

    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq.dot(&gamma.dot(&dq));

    // electronic energy as sum of band structure energy and Coulomb energy
    let mut e_elec: f64 = e_band_structure + e_coulomb;

    // add lc exchange to electronic energy if lrc is requested
    if g0_lr_ao.is_some() {
        let e_hf_x: f64 = lc_exchange_energy(s, g0_lr_ao.unwrap(), p0, p);
        e_elec += e_hf_x;
    }

    e_elec
}

pub fn get_electronic_energy_unrestricted(
    p_alpha: ArrayView2<f64>,
    p_beta: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq_alpha: ArrayView1<f64>,
    dq_beta: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
    spin_couplings: ArrayView1<f64>,
) -> f64 {
    let dq: Array1<f64> = &dq_alpha + &dq_beta;
    let m_squared: Array1<f64> = (&dq_alpha - &dq_beta).iter().map(|x| x * x).collect();

    // band structure energy
    let e_band_structure: f64 = (&(&p_alpha + &p_beta) * &h0).sum();

    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq.dot(&gamma.dot(&dq));

    // Spin polarization energy
    let e_spin: f64 = 0.5 * m_squared.dot(&spin_couplings);

    // electronic energy as sum of band structure energy and Coulomb energy
    let e_elec: f64 = e_band_structure + e_coulomb + e_spin;

    e_elec
}

/// Construct part of the Hamiltonian corresponding to long range
/// Hartree-Fock exchange
/// H^x_mn = -1/2 sum_ab (P_ab-P0_ab) (ma|bn)_lr
/// The Coulomb potential in the electron integral is replaced by
/// 1/r ----> erf(r/R_lr)/r
pub fn lc_exact_exchange(
    s: ArrayView2<f64>,
    g0_lr_ao: ArrayView2<f64>,
    dp: ArrayView2<f64>,
) -> Array2<f64> {
    let s_dot_dp = s.dot(&dp);
    let tmp = (&g0_lr_ao * &s_dot_dp).dot(&s);
    let mut hx: Array2<f64> = &tmp + &tmp.t();
    hx = hx + &g0_lr_ao * &s_dot_dp.dot(&s);
    hx = hx + (s.dot(&(&dp * &g0_lr_ao))).dot(&s);
    hx *= -0.125;
    hx = 0.5 * (&hx + &hx.t());

    hx
}

pub fn lc_exchange_energy(
    s: ArrayView2<f64>,
    g0_lr_ao: ArrayView2<f64>,
    p0: ArrayView2<f64>,
    p: ArrayView2<f64>,
) -> f64 {
    let dp: Array2<f64> = &p - &p0;
    let mut e_hf_x: f64 = 0.0;
    e_hf_x += ((s.dot(&dp.dot(&s))) * &dp * g0_lr_ao).sum();
    e_hf_x += (s.dot(&dp) * dp.dot(&s) * g0_lr_ao).sum();
    e_hf_x *= -0.125;
    e_hf_x
}

/// Compute electronic energies
pub fn get_electronic_energy_new(
    p: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
) -> f64 {
    // band structure energy
    let e_band_structure: f64 = (&p * &h0).sum();
    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq.dot(&gamma.dot(&dq));
    // electronic energy as sum of band structure energy and Coulomb energy
    let e_elec: f64 = e_band_structure + e_coulomb;

    e_elec
}

pub fn get_electronic_energy_gamma_shell_resolved(
    p: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq_ao: ArrayView1<f64>,
    gamma_ao: ArrayView2<f64>,
) -> f64 {
    // band structure energy
    let e_band_structure: f64 = (&p * &h0).sum();
    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq_ao.dot(&gamma_ao.dot(&dq_ao));
    // electronic energy as sum of band structure energy and Coulomb energy
    let e_elec: f64 = e_band_structure + e_coulomb;

    e_elec
}

pub fn calc_coulomb_third_order(gamma_third_order: ArrayView2<f64>, dq: ArrayView1<f64>) -> f64 {
    let dq2: Array1<f64> = dq.mapv(|val| val.powi(2));
    let third_energy: f64 = 1.0 / 3.0 * dq2.dot(&gamma_third_order.dot(&dq));
    third_energy
}

pub fn calc_exchange(s: ArrayView2<f64>, g0_lr_ao: ArrayView2<f64>, dp: ArrayView2<f64>) -> f64 {
    let ex =
        ((s.dot(&dp.dot(&s))) * dp * g0_lr_ao).sum() + (s.dot(&dp) * dp.dot(&s) * g0_lr_ao).sum();
    -0.125 * ex
}

/// Construct the density matrix
/// P_mn = sum_a f_a C_ma* C_na
pub fn density_matrix(orbs: ArrayView2<f64>, f: &[f64]) -> Array2<f64> {
    let occ_indx: Vec<usize> = f.iter().positions(|&x| x > 0.0).collect();
    let occ_orbs: Array2<f64> = orbs.select(Axis(1), &occ_indx);
    let f_occ: Vec<f64> = f.iter().filter(|&&x| x > 0.0).cloned().collect();
    // THIS IS NOT AN EFFICIENT WAY TO BUILD THE LEFT HAND SIDE
    let mut f_occ_mat: Vec<f64> = Vec::new();
    for _ in 0..occ_orbs.nrows() {
        for val in f_occ.iter() {
            f_occ_mat.push(*val);
        }
    }
    let f_occ_mat: Array2<f64> = Array2::from_shape_vec(occ_orbs.raw_dim(), f_occ_mat).unwrap();
    let p: Array2<f64> = (f_occ_mat * &occ_orbs).dot(&occ_orbs.t());
    p
}

/// Construct reference density matrix
/// all atoms should be neutral
pub fn density_matrix_ref(n_orbs: usize, atoms: &[Atom]) -> Array2<f64> {
    let mut p0: Array2<f64> = Array2::zeros((n_orbs, n_orbs));
    // iterate over orbitals on center i
    let mut idx: usize = 0;
    for atomi in atoms.iter() {
        // how many electrons are put into the nl-shell
        for occ in atomi.valorbs_occupation.iter() {
            p0[[idx, idx]] = *occ;
            idx += 1;
        }
    }
    p0
}

pub fn outer_sum(vec: ArrayView1<f64>) -> Array2<f64> {
    let vec_column: Array2<f64> = vec.to_owned().insert_axis(Axis(1));
    let result: Array2<f64> = &vec_column.broadcast((vec.dim(), vec.dim())).unwrap() + &vec;
    result
}

pub fn construct_h1(
    n_orbs: usize,
    atoms: &[Atom],
    gamma: ArrayView2<f64>,
    dq: ArrayView1<f64>,
) -> Array2<f64> {
    let e_stat_pot: Array1<f64> = gamma.dot(&dq);
    let mut h1: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for _ in 0..(atomi.n_orbs) {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for _ in 0..(atomj.n_orbs) {
                    h1[[mu, nu]] = 0.5 * (e_stat_pot[i] + e_stat_pot[j]);
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    h1
}

pub fn construct_h_third_order(
    n_orbs: usize,
    atoms: &[Atom],
    gamma_third_order: ArrayView2<f64>,
    dq: ArrayView1<f64>,
) -> Array2<f64> {
    let e_stat_pot: Array1<f64> = gamma_third_order.dot(&dq);
    let e_stat_pot2: Array1<f64> = dq.map(|val| val.powi(2)).dot(&gamma_third_order);
    let mut h: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for _ in 0..(atomi.n_orbs) {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for _ in 0..(atomj.n_orbs) {
                    let contrib_1: f64 =
                        1.0 / 3.0 * (e_stat_pot[i] * dq[i] + e_stat_pot[j] * dq[j]);
                    let contrib_2: f64 = 1.0 / 6.0 * (e_stat_pot2[i] + e_stat_pot2[j]);
                    // add to h
                    h[[mu, nu]] = contrib_1 + contrib_2;
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    h
}

pub fn construct_third_order_gradient_contribution(
    n_orbs: usize,
    atoms: &[Atom],
    gamma_third_order: ArrayView2<f64>,
    dq: ArrayView1<f64>,
) -> Array2<f64> {
    let e_stat_pot: Array1<f64> = gamma_third_order.dot(&dq);
    let e_stat_pot_2: Array1<f64> = dq.map(|val| val.powi(2)).dot(&gamma_third_order);
    let mut h: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for _ in 0..(atomi.n_orbs) {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for _ in 0..(atomj.n_orbs) {
                    // add to h
                    h[[mu, nu]] = (2.0 * e_stat_pot[i] * dq[i]
                        + e_stat_pot_2[i]
                        + 2.0 * e_stat_pot[j] * dq[j]
                        + e_stat_pot_2[j])
                        / 3.0;
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    h
}

pub fn construct_third_order_gradient_contribution_test(
    n_orbs: usize,
    atoms: &[Atom],
    gamma_third_order: ArrayView2<f64>,
    dq: ArrayView1<f64>,
    grad_atom: &Atom,
    atom_idx: usize,
) -> Array2<f64> {
    let e_stat_pot: Array1<f64> = gamma_third_order.dot(&dq);
    let e_stat_pot_2: Array1<f64> = dq.map(|val| val.powi(2)).dot(&gamma_third_order);
    let mut h: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
    let mut mu: usize = 0;
    let mut nu: usize;

    let mut ao_count: usize = 0;
    let mut count: usize = 0;
    for (idx, atom) in atoms.iter().enumerate() {
        if idx == atom_idx {
            ao_count = count;
            break;
        }
        for _ in 0..(atom.n_orbs) {
            count += 1;
        }
    }

    for (i, atomi) in atoms.iter().enumerate() {
        for _ in 0..(atomi.n_orbs) {
            nu = ao_count;
            if atom_idx != i {
                for _ in 0..(grad_atom.n_orbs) {
                    // add to h
                    h[[mu, nu]] = (2.0 * e_stat_pot[i] * dq[i]
                        + e_stat_pot_2[i]
                        + 2.0 * e_stat_pot[atom_idx] * dq[atom_idx]
                        + e_stat_pot_2[atom_idx])
                        / 3.0;
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    h
}

pub fn construct_h_magnetization(
    n_orbs: usize,
    atoms: &[Atom],
    dq: ArrayView1<f64>,
    spin_couplings: ArrayView1<f64>,
) -> Array2<f64> {
    let pot: Array1<f64> = &dq * &spin_couplings;
    let mut h: Array2<f64> = Array2::zeros([n_orbs, n_orbs]);
    let mut mu: usize = 0;
    let mut nu: usize;
    for (i, atomi) in atoms.iter().enumerate() {
        for _ in 0..(atomi.n_orbs) {
            nu = 0;
            for (j, atomj) in atoms.iter().enumerate() {
                for _ in 0..(atomj.n_orbs) {
                    h[[mu, nu]] = 0.5 * (pot[i] + pot[j]);
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    h
}

pub(crate) fn enable_level_shifting(orbe: ArrayView1<f64>, n_elec: usize) -> bool {
    let hl_idxs: (usize, usize) = get_frontier_orbitals(n_elec);
    let gap: f64 = get_homo_lumo_gap(orbe.view(), hl_idxs);
    debug!("HOMO - LUMO gap:          {:>18.14}", gap);
    gap < defaults::HOMO_LUMO_TOL
}
