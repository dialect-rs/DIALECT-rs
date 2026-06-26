use dialect_base::constants;
use ndarray::prelude::*;

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
    if t == 0.0 {
        return fermi_occupation_t0(orbe, n_elec_paired);
    }

    let n_elec = n_elec_paired;

    // Use the full orbital energy range plus margin for bracketing
    let e_min = orbe.iter().cloned().fold(f64::INFINITY, f64::min);
    let e_max = orbe.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Margin ensures bracket contains root even at high T
    let margin = 20.0 * constants::K_BOLTZMANN * t;
    let bracket_low = e_min - margin;
    let bracket_high = e_max + margin;

    let func = |mu: f64| -> f64 { fa_minus_nelec(mu, orbe.view(), fermi, t, n_elec) };

    // Debug: verify bracket
    let f_low = func(bracket_low);
    let f_high = func(bracket_high);
    // println!("Bracket: [{:.6}, {:.6}]", bracket_low, bracket_high);
    // println!("f(low) = {:.6}, f(high) = {:.6}", f_low, f_high);

    // f_low should be negative (sum ≈ 0 when mu << all energies)
    // f_high should be positive (sum ≈ 2*n_orb when mu >> all energies)
    assert!(
        f_low * f_high < 0.0,
        "Root not bracketed: f({:.6})={:.6}, f({:.6})={:.6}",
        bracket_low,
        f_low,
        bracket_high,
        f_high
    );

    let mu = bisect(func, bracket_low, bracket_high, 1.0e-10);
    // let mu = zbrent(func, bracket_low, bracket_high, 1.0e-10, 400);

    let dn = func(mu);
    assert!(dn.abs() <= 5.0e-04, "Electron count error: {}", dn);

    let fermi_occ: Vec<f64> = orbe.iter().map(|&en| fermi(en, mu, t)).collect();

    (mu, fermi_occ)
}

/// Find the occupation of single-particle states at T=0
fn fermi_occupation_t0(orbe: ArrayView1<f64>, n_elec_paired: usize) -> (f64, Vec<f64>) {
    let mut n_elec_paired: f64 = n_elec_paired as f64;
    let sort_indx: Vec<usize> = argsort(orbe.as_slice().unwrap());
    let mut fermi_occ: Vec<f64> = vec![0.0; orbe.len()];
    for a in sort_indx.iter() {
        fermi_occ[*a] = 2.0_f64.min(n_elec_paired);
        if n_elec_paired > 1.0 {
            n_elec_paired -= 2.0;
        } else if n_elec_paired == 1.0 {
            n_elec_paired -= 1.0;
        }
    }
    (0.0, fermi_occ)
}

/// Single-spin-channel occupation: place `n_elec` electrons (each with a
/// maximum occupation of 1) into the orbitals `orbe` at temperature `t`.
/// Returns the chemical potential and the per-orbital occupations (0..1).
///
/// This is the building block for (spin-restricted) open-shell occupations:
/// the alpha and beta channels are filled separately with `n_alpha` and
/// `n_beta` electrons over the same set of spatial orbitals, and the total
/// density occupation is `focc = focc_alpha + focc_beta`. For a closed shell
/// (`n_alpha == n_beta == n_elec/2`) the sum reproduces the paired
/// [`fermi_occupation`] result exactly.
pub fn fermi_occupation_single(orbe: ArrayView1<f64>, n_elec: usize, t: f64) -> (f64, Vec<f64>) {
    if t == 0.0 {
        // Fill the lowest `n_elec` orbitals with one electron each.
        let sort_indx: Vec<usize> = argsort(orbe.as_slice().unwrap());
        let mut occ: Vec<f64> = vec![0.0; orbe.len()];
        for (filled, &a) in sort_indx.iter().enumerate() {
            if filled < n_elec {
                occ[a] = 1.0;
            }
        }
        return (0.0, occ);
    }

    let mu = find_fermi_level_single(orbe, n_elec, t);
    let occ: Vec<f64> = orbe.iter().map(|&en| fermi_single(en, mu, t)).collect();
    (mu, occ)
}

// original code from from https://qiita.com/osanshouo/items/71b0272cd5e156cbf5f2
fn argsort(v: &[f64]) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
    idx
}

// fn fermi(en: f64, mu: f64, t: f64) -> f64 {
//     2.0 / (((en - mu) / (constants::K_BOLTZMANN * t)).exp() + 1.0)
// }

fn fermi(en: f64, mu: f64, t: f64) -> f64 {
    let x = (en - mu) / (constants::K_BOLTZMANN * t);
    if x > 100.0 {
        0.0
    } else if x < -100.0 {
        2.0
    } else {
        2.0 / (x.exp() + 1.0)
    }
}

fn bisect<F: Fn(f64) -> f64>(func: F, x1: f64, x2: f64, tol: f64) -> f64 {
    let mut a = x1;
    let mut b = x2;
    let fa = func(a);
    let fb = func(b);

    assert!(
        fa * fb <= 0.0,
        "Root not bracketed: f({})={}, f({})={}",
        a,
        fa,
        b,
        fb
    );

    // Ensure a is the side with negative (or zero) function value
    if fa > 0.0 {
        std::mem::swap(&mut a, &mut b);
    }

    const MAX_ITER: usize = 100;

    for _ in 0..MAX_ITER {
        let c = 0.5 * (a + b);

        if (b - a).abs() < tol {
            return c;
        }

        let fc = func(c);

        if fc <= 0.0 {
            a = c;
        } else {
            b = c;
        }
    }

    0.5 * (a + b)
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
        sum_fa += fermi_function(*en_a, mu, t)
    }
    sum_fa - (n_elec as f64)
}

/// Single-electron Fermi function (occupation 0-1)
fn fermi_single(en: f64, mu: f64, t: f64) -> f64 {
    let x = (en - mu) / (constants::K_BOLTZMANN * t);
    if x > 100.0 {
        0.0
    } else if x < -100.0 {
        1.0
    } else {
        1.0 / (x.exp() + 1.0)
    }
}

/// Find Fermi level for n electrons using single-electron occupations
fn find_fermi_level_single(orbe: ArrayView1<f64>, n_elec: usize, t: f64) -> f64 {
    if n_elec == 0 {
        return orbe[0] - 10.0 * constants::K_BOLTZMANN * t;
    }

    let e_min = orbe.iter().cloned().fold(f64::INFINITY, f64::min);
    let e_max = orbe.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let margin = 20.0 * constants::K_BOLTZMANN * t;
    let bracket_low = e_min - margin;
    let bracket_high = e_max + margin;

    let func = |mu: f64| -> f64 {
        let sum: f64 = orbe.iter().map(|&en| fermi_single(en, mu, t)).sum();
        sum - (n_elec as f64)
    };

    bisect(func, bracket_low, bracket_high, 1.0e-10)
}

/// Compute entropy contribution for a single spin channel (Fermi smearing)
/// Returns -T*S where S is the electronic entropy
/// orbe: orbital energies (Hartree)
/// n_elec: number of electrons in this spin channel
/// t: temperature in Kelvin
pub fn compute_channel_entropy(orbe: ArrayView1<f64>, n_elec: usize, t: f64) -> f64 {
    if t < 0.1 || n_elec == 0 {
        return 0.0;
    }

    let mu = find_fermi_level_single(orbe, n_elec, t);

    let mut entropy_sum: f64 = 0.0;
    for &en in orbe.iter() {
        let occ = fermi_single(en, mu, t);
        // Only add contribution for fractional occupations
        if occ > 1e-12 && occ < 1.0 - 1e-12 {
            entropy_sum += occ * occ.ln() + (1.0 - occ) * (1.0 - occ).ln();
        }
    }

    entropy_sum * constants::K_BOLTZMANN * t
}

/// Compute total entropy contribution for restricted calculation
/// Handles even and odd electron counts correctly by splitting into alpha/beta channels
pub fn compute_total_entropy(orbe: ArrayView1<f64>, n_elec: usize, t: f64) -> f64 {
    let n_alpha = (n_elec + 1) / 2; // Rounded up
    let n_beta = n_elec / 2; // Rounded down

    let ga = compute_channel_entropy(orbe, n_alpha, t);
    let gb = compute_channel_entropy(orbe, n_beta, t);

    ga + gb
}
