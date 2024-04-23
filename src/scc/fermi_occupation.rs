use crate::constants;
use crate::utils::zbrent;
use ndarray::prelude::*;
use std::cmp::max;

/// Find the occupation of single-particle state a at finite temperature T
/// according to the Fermi distribution:
///     $f_a = f(en_a) = 2 /(exp(en_a - mu)/(kB*T) + 1)$
/// The chemical potential is determined from the condition that
/// sum_a f_a = Nelec
///
/// Parameters:
/// ===========
/// orbe: orbital energies
/// Nelec_paired: number of paired electrons, these electron will be placed in the same orbital
/// Nelec_unpaired: number of unpaired electrons, these electrons will sit in singly occupied
///                 orbitals (only works at T=0)
/// T: temperature in Kelvin
///
/// Returns:
/// ========
/// mu: chemical potential
/// f: list of occupations f[a] for orbitals (in the same order as the energies in orbe)
pub fn fermi_occupation(orbe: ArrayView1<f64>, n_elec_paired: usize, t: f64) -> (f64, Vec<f64>) {
    let mut fermi_occ: Vec<f64> = Vec::new();
    let mu: f64;
    if t == 0.0 {
        let result: (f64, Vec<f64>) = fermi_occupation_t0(orbe, n_elec_paired);
        mu = result.0;
        fermi_occ = result.1;
    } else {
        let n_elec: usize = n_elec_paired;
        let sort_indx: Vec<usize> = argsort(&orbe.to_vec());
        // highest doubly occupied orbital
        let h_idx: usize = max((n_elec / 2) - 1, 0);
        let homo: f64 = orbe[sort_indx[h_idx]];
        // LUMO + 1
        let lp1_idx: usize = max((n_elec / 2) + 1, sort_indx.len() - 1);
        let lumop1: f64 = orbe[sort_indx[lp1_idx]];
        // search for fermi energy in the interval [HOMO, LUMO+1]
        let func = |x: f64| -> f64 { fa_minus_nelec(x, orbe.view(), fermi, t, n_elec) };
        mu = zbrent(func, homo, lumop1, 1.0e-08, 100);
        let dn: f64 = func(mu);
        assert!(dn.abs() <= 1.0e-08);
        for en in orbe.iter() {
            fermi_occ.push(fermi(*en, mu, t));
        }
    }
    return (mu, fermi_occ);
}

/// Find the occupation of single-particle states at T=0
fn fermi_occupation_t0(orbe: ArrayView1<f64>, n_elec_paired: usize) -> (f64, Vec<f64>) {
    let mut n_elec_paired: f64 = n_elec_paired as f64;
    let sort_indx: Vec<usize> = argsort(orbe.as_slice().unwrap());
    let mut fermi_occ: Vec<f64> = vec![0.0; orbe.len()];
    for a in sort_indx.iter() {
        fermi_occ[*a] = 2.0_f64.min(n_elec_paired);
        if n_elec_paired > 1.0 {
            n_elec_paired = n_elec_paired - 2.0;
        }
    }
    return (0.0, fermi_occ);
}

// original code from from https://qiita.com/osanshouo/items/71b0272cd5e156cbf5f2
fn argsort(v: &[f64]) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
    idx
}

fn fermi(en: f64, mu: f64, t: f64) -> f64 {
    2.0 / (((en - mu) / (constants::K_BOLTZMANN * t)).exp() + 1.0)
}

fn fa_minus_nelec(
    mu: f64,
    orbe: ArrayView1<f64>,
    fermi_function: fn(f64, f64, f64) -> f64,
    t: f64,
    n_elec: usize,
) -> f64 {
    // find the root of this function to enforce sum_a f_a = Nelec
    let mut sum_fa: f64 = 0.0;
    for en_a in orbe.iter() {
        sum_fa = sum_fa + fermi_function(*en_a, mu, t)
    }
    return sum_fa - (n_elec as f64);
}
