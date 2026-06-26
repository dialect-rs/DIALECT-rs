use dialect_base::constants::BOHR_TO_ANGS;
use crate::initialization::atom::XtbAtom;
use crate::parameters::{COV_RADII, HALOGEN_BOND_STRENGH, HALOGEN_DAMPING, HALOGEN_RAD_SCALE};
use crate::scc::halogen_correction::{
    check_acceptor_atoms, check_halogen_donor, get_halogen_bond_indices,
};
use ndarray::prelude::*;

/// Compute the analytical gradient of the halogen bonding energy correction.
///
/// For each halogen bond triple (H=halogen, A=acceptor, N=neighbor of halogen):
///   E = damping * bond_strength * (R^12 - HALOGEN_DAMPING * R^6) / (R^12 + 1)
/// where
///   R = HALOGEN_RAD_SCALE * (cov_h + cov_a) / dist_ha
///   damping = f^6,  f = 0.5 - 0.25 * c
///   c = (diff_h_a + diff_h_n - diff_n_a) / sqrt(diff_h_n * diff_h_a)
pub fn gradient_halogen_bonding_xtb(atoms: &[XtbAtom]) -> Array1<f64> {
    let n_atoms = atoms.len();
    let mut gradient: Array1<f64> = Array1::zeros(3 * n_atoms);

    let halogens: Vec<usize> = check_halogen_donor(atoms);
    let acceptors: Vec<usize> = check_acceptor_atoms(atoms);

    if halogens.is_empty() || acceptors.is_empty() {
        return gradient;
    }

    let bond_vec: Vec<(usize, usize, usize)> =
        get_halogen_bond_indices(atoms, &halogens, &acceptors);

    for &(h_idx, a_idx, n_idx) in bond_vec.iter() {
        let h_atom = &atoms[h_idx];
        let a_atom = &atoms[a_idx];
        let n_atom = &atoms[n_idx];
        let h_number = h_atom.number as usize - 1;
        let a_number = a_atom.number as usize - 1;

        // Covalent radii (converted from Angstrom to Bohr)
        let cov_h: f64 = COV_RADII[h_number] / BOHR_TO_ANGS;
        let cov_a: f64 = COV_RADII[a_number] / BOHR_TO_ANGS;
        let bond_strength: f64 = HALOGEN_BOND_STRENGH[h_number];

        // Position difference vectors (H-A, H-N, N-A)
        let r_ha = [
            h_atom.xyz.x - a_atom.xyz.x,
            h_atom.xyz.y - a_atom.xyz.y,
            h_atom.xyz.z - a_atom.xyz.z,
        ];
        let r_hn = [
            h_atom.xyz.x - n_atom.xyz.x,
            h_atom.xyz.y - n_atom.xyz.y,
            h_atom.xyz.z - n_atom.xyz.z,
        ];
        let r_na = [
            n_atom.xyz.x - a_atom.xyz.x,
            n_atom.xyz.y - a_atom.xyz.y,
            n_atom.xyz.z - a_atom.xyz.z,
        ];

        // Squared distances
        let diff_h_a: f64 = r_ha[0] * r_ha[0] + r_ha[1] * r_ha[1] + r_ha[2] * r_ha[2];
        let diff_h_n: f64 = r_hn[0] * r_hn[0] + r_hn[1] * r_hn[1] + r_hn[2] * r_hn[2];
        let diff_n_a: f64 = r_na[0] * r_na[0] + r_na[1] * r_na[1] + r_na[2] * r_na[2];

        let dist_ha: f64 = diff_h_a.sqrt();

        // Cosine-like angle term: c = (diff_h_a + diff_h_n - diff_n_a) / sqrt(diff_h_n * diff_h_a)
        let denom_sq: f64 = diff_h_n * diff_h_a;
        let denom: f64 = denom_sq.sqrt();
        let denom_inv: f64 = 1.0 / denom;
        let c: f64 = (diff_h_a + diff_h_n - diff_n_a) * denom_inv;

        // Angular damping: f = 0.5 - 0.25*c, damping = f^6
        let f: f64 = 0.5 - 0.25 * c;
        let f5: f64 = f.powi(5);
        let damping: f64 = f * f5; // f^6

        // Radial term: R = HALOGEN_RAD_SCALE * (cov_h + cov_a) / dist_ha
        let ratio: f64 = (cov_h + cov_a) / dist_ha;
        let r_scaled: f64 = HALOGEN_RAD_SCALE * ratio;
        let r6: f64 = r_scaled.powi(6);
        let r12: f64 = r6 * r6;

        // g = (R^12 - d*R^6) / (R^12 + 1)
        let g_num: f64 = r12 - HALOGEN_DAMPING * r6;
        let g_den: f64 = r12 + 1.0;
        let g: f64 = g_num / g_den;

        // --- Derivative of g w.r.t. R ---
        // dg/dR = (12*R^11 + 6*d*R^5*(R^12 - 1)) / (R^12 + 1)^2
        //       = (12*R^11*(R^12+1) - (R^12-d*R^6)*12*R^11) / (R^12+1)^2  <-- quotient rule
        // Using quotient rule:  dg/dR = [numerator_deriv * den - num * den_deriv] / den^2
        // num = R^12 - d*R^6,   num' = 12*R^11 - 6*d*R^5
        // den = R^12 + 1,       den' = 12*R^11
        let r5: f64 = r_scaled.powi(5);
        let r11: f64 = r5 * r6;
        let dg_dr: f64 =
            ((12.0 * r11 - 6.0 * HALOGEN_DAMPING * r5) * g_den - g_num * 12.0 * r11)
                / (g_den * g_den);

        // dR/d(dist_ha) = -R / dist_ha
        // d(dist_ha)/d(H_k) = r_ha_k / dist_ha,   d(dist_ha)/d(A_k) = -r_ha_k / dist_ha
        // So: dg/d(H_k) = dg/dR * (-R/dist_ha) * (r_ha_k/dist_ha) = dg/dR * (-R * r_ha_k / dist_ha^2)
        let dg_factor: f64 = dg_dr * (-r_scaled / (dist_ha * dist_ha));

        // dg/d(H) = dg_factor * r_ha,  dg/d(A) = -dg_factor * r_ha,  dg/d(N) = 0
        let dg_h = [dg_factor * r_ha[0], dg_factor * r_ha[1], dg_factor * r_ha[2]];

        // --- Derivative of c w.r.t. atom positions ---
        // c = (diff_h_a + diff_h_n - diff_n_a) / sqrt(diff_h_n * diff_h_a)
        //
        // dc/d(H_k) = 2*(r_ha_k + r_hn_k)*denom_inv - c*(r_hn_k/diff_h_n + r_ha_k/diff_h_a)
        // dc/d(A_k) = (-2*r_ha_k + 2*r_na_k)*denom_inv + c*r_ha_k/diff_h_a
        // dc/d(N_k) = -2*(r_hn_k + r_na_k)*denom_inv + c*r_hn_k/diff_h_n
        let inv_diff_h_a: f64 = 1.0 / diff_h_a;
        let inv_diff_h_n: f64 = 1.0 / diff_h_n;

        let mut dc_h = [0.0f64; 3];
        let mut dc_a = [0.0f64; 3];
        let mut dc_n = [0.0f64; 3];

        for k in 0..3 {
            dc_h[k] = 2.0 * (r_ha[k] + r_hn[k]) * denom_inv
                - c * (r_hn[k] * inv_diff_h_n + r_ha[k] * inv_diff_h_a);
            dc_a[k] = (-2.0 * r_ha[k] + 2.0 * r_na[k]) * denom_inv + c * r_ha[k] * inv_diff_h_a;
            dc_n[k] =
                -2.0 * (r_hn[k] + r_na[k]) * denom_inv + c * r_hn[k] * inv_diff_h_n;
        }

        // --- Full gradient ---
        // E = bond_strength * f^6 * g = bond_strength * damping * g
        // dE/dx = bond_strength * (6*f^5 * (-0.25) * dc/dx * g  +  f^6 * dg/dx)
        //       = bond_strength * (-1.5*f^5 * dc/dx * g  +  damping * dg/dx)
        let angular_prefactor: f64 = bond_strength * (-1.5) * f5 * g;
        let radial_prefactor: f64 = bond_strength * damping;

        for k in 0..3 {
            // Halogen atom (H)
            gradient[3 * h_idx + k] += angular_prefactor * dc_h[k] + radial_prefactor * dg_h[k];

            // Acceptor atom (A): dg/d(A) = -dg/d(H)
            gradient[3 * a_idx + k] += angular_prefactor * dc_a[k] - radial_prefactor * dg_h[k];

            // Neighbor atom (N): dg/d(N) = 0
            gradient[3 * n_idx + k] += angular_prefactor * dc_n[k];
        }
    }

    gradient
}
