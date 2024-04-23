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
    return (homo, lumo);
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
    return (homo, lumo);
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
    return e_nuc;
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

    return e_elec;
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

    return e_elec;
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
    // let mut hx: Array2<f64> = (&g0_lr_ao * &s.dot(&dp)).dot(&s);
    // hx = hx + &g0_lr_ao * &(s.dot(&dp)).dot(&s);
    // hx = hx + (s.dot(&(&dp * &g0_lr_ao))).dot(&s);
    // hx = hx + s.dot(&(&g0_lr_ao * &dp.dot(&s)));
    // hx = hx * -0.125;

    // let mut hx: Array2<f64> = Array2::zeros(s.raw_dim());
    // let dim = s.dim().0;
    // for mu in 0..dim {
    //     for nu in 0..dim {
    //         if mu <= nu{
    //             for la in 0..dim {
    //                 for sig in 0..dim {
    //                     hx[[mu, nu]] += -0.125
    //                         * dp[[la, sig]]
    //                         * s[[mu, la]]
    //                         * s[[nu, sig]]
    //                         * (g0_lr_ao[[mu, sig]]
    //                         + g0_lr_ao[[mu, nu]]
    //                         + g0_lr_ao[[la, sig]]
    //                         + g0_lr_ao[[la, nu]]);
    //                 }
    //             }
    //         }
    //         else{
    //             hx[[mu,nu]] = hx[[nu,mu]];
    //         }
    //     }
    // }

    let s_dot_dp = s.dot(&dp);
    let tmp = (&g0_lr_ao * &s_dot_dp).dot(&s);
    let mut hx: Array2<f64> = &tmp + &tmp.t();
    hx = hx + &g0_lr_ao * &s_dot_dp.dot(&s);
    hx = hx + (s.dot(&(&dp * &g0_lr_ao))).dot(&s);
    hx *= -0.125;
    hx = 0.5 * (&hx + &hx.t());

    return hx;
}

pub fn lc_exchange_energy(
    s: ArrayView2<f64>,
    g0_lr_ao: ArrayView2<f64>,
    p0: ArrayView2<f64>,
    p: ArrayView2<f64>,
) -> f64 {
    let dp: Array2<f64> = &p - &p0;
    let mut e_hf_x: f64 = 0.0;
    e_hf_x += ((s.dot(&dp.dot(&s))) * &dp * &g0_lr_ao).sum();
    e_hf_x += (s.dot(&dp) * dp.dot(&s) * &g0_lr_ao).sum();
    e_hf_x *= -0.125;
    return e_hf_x;
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

    return e_elec;
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
    return p;
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
    return p0;
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
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return h1;
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
                    nu = nu + 1;
                }
            }
            mu = mu + 1;
        }
    }
    return h;
}

pub(crate) fn enable_level_shifting(orbe: ArrayView1<f64>, n_elec: usize) -> bool {
    let hl_idxs: (usize, usize) = get_frontier_orbitals(n_elec);
    let gap: f64 = get_homo_lumo_gap(orbe.view(), hl_idxs);
    debug!("HOMO - LUMO gap:          {:>18.14}", gap);
    gap < defaults::HOMO_LUMO_TOL
}
