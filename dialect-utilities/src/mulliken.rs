use dialect_dftb_core::atom::Atom;
use dialect_xtb_core::initialization::basis::Basis;
use ndarray::prelude::*;

/// Calculate Mulliken charges according to:
///       ⎲  ⎲  P   S
/// q  =  ⎳  ⎳   µν  νµ
///  A    µ∈A  ν
pub fn mulliken(p: ArrayView2<f64>, s: ArrayView2<f64>, atoms: &[Atom]) -> Array1<f64> {
    let mut q: Array1<f64> = Array1::<f64>::zeros(atoms.len());
    let q_ao: Array1<f64> = s.dot(&p).diag().to_owned();

    let mut mu = 0;
    for (q_i, atomi) in q.iter_mut().zip(atoms.iter()) {
        for _ in 0..atomi.n_orbs {
            *q_i += q_ao[mu];
            mu += 1;
        }
    }
    q
}

pub fn mulliken_atomwise(
    p: ArrayView2<f64>,
    s: ArrayView2<f64>,
    atoms: &[Atom],
    n_atoms: usize,
) -> Array1<f64> {
    let mut q: Array1<f64> = Array1::<f64>::zeros(n_atoms);

    // iterate over atoms A
    let mut mu = 0;
    // inside the loop
    for (i, atomi) in atoms.iter().enumerate() {
        // iterate over orbitals on atom A
        for _ in 0..atomi.n_orbs {
            let mut nu = 0;
            // iterate over atoms B
            for atomj in atoms.iter() {
                // iterate over orbitals on atom B
                for _ in 0..atomj.n_orbs {
                    q[i] += p[[mu, nu]] * s[[mu, nu]];
                    nu += 1;
                }
            }
            mu += 1;
        }
    }
    q
}

pub fn mulliken_atomwise_xtb(
    p: ArrayView2<f64>,
    s: ArrayView2<f64>,
    basis: &Basis,
    n_atoms: usize,
) -> Array1<f64> {
    let mut q: Array1<f64> = Array1::<f64>::zeros(n_atoms);

    for (i, funci) in basis.basis_functions.iter().enumerate() {
        let at_i: usize = funci.atom_index;
        for (j, _funcj) in basis.basis_functions.iter().enumerate() {
            q[at_i] += p[[i, j]] * s[[i, j]];
        }
    }

    q
}

/// Compute Mulliken charges per orbital: q[i] = sum_j S[i,j] * P[j,i]
/// Optimized: compute only diagonal elements of S*P (O(n^2) instead of O(n^3))
#[inline]
pub fn mulliken_aowise(p: ArrayView2<f64>, s: ArrayView2<f64>) -> Array1<f64> {
    let n = s.nrows();
    let mut q = Array1::zeros(n);

    // Compute diagonal of S*P: (S*P)[i,i] = sum_k S[i,k] * P[k,i]
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..n {
            sum += s[[i, k]] * p[[k, i]];
        }
        q[i] = sum;
    }

    q
}

/// Compute Mulliken charges from difference (p - p0) without creating intermediate array
/// q[i] = sum_k S[i,k] * (P[k,i] - P0[k,i])
#[inline]
pub fn mulliken_aowise_diff(
    p: ArrayView2<f64>,
    p0: ArrayView2<f64>,
    s: ArrayView2<f64>,
) -> Array1<f64> {
    let n = s.nrows();
    let mut q = Array1::zeros(n);

    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..n {
            sum += s[[i, k]] * (p[[k, i]] - p0[[k, i]]);
        }
        q[i] = sum;
    }

    q
}

pub fn mulliken_atomwise_from_ao_xtb(
    basis: &Basis,
    n_atoms: usize,
    dq_ao: ArrayView1<f64>,
) -> Array1<f64> {
    let mut q: Array1<f64> = Array1::<f64>::zeros(n_atoms);

    for shell in basis.shells.iter() {
        let at_i: usize = shell.atom_index;
        for i in shell.sph_start..shell.sph_end {
            q[at_i] += dq_ao[i];
        }
    }

    q
}

/// Convert AO-level charges to shell-level charges by summing orbitals within each shell
#[inline]
pub fn ao_to_shell_charges(basis: &Basis, dq_ao: ArrayView1<f64>) -> Array1<f64> {
    let n_shells = basis.shells.len();
    let mut dq_shell = Array1::<f64>::zeros(n_shells);

    for (shell_idx, shell) in basis.shells.iter().enumerate() {
        for i in shell.sph_start..shell.sph_end {
            dq_shell[shell_idx] += dq_ao[i];
        }
    }

    dq_shell
}

/// Convert shell-level charges back to AO-level by distributing equally among orbitals in each shell
/// Total charge is conserved (shell charge divided among orbitals)
#[inline]
pub fn shell_to_ao_charges(basis: &Basis, n_orbs: usize, dq_shell: ArrayView1<f64>) -> Array1<f64> {
    let mut dq_ao = Array1::<f64>::zeros(n_orbs);

    for (shell_idx, shell) in basis.shells.iter().enumerate() {
        let n_orbs_in_shell = shell.sph_end - shell.sph_start;
        let charge_per_orb = dq_shell[shell_idx] / n_orbs_in_shell as f64;
        for i in shell.sph_start..shell.sph_end {
            dq_ao[i] = charge_per_orb;
        }
    }

    dq_ao
}

/// Expand shell-level values to AO-level without dividing
/// Each orbital gets the same value as its shell (for potentials, shifts, etc.)
#[inline]
pub fn shell_to_ao_values(
    basis: &Basis,
    n_orbs: usize,
    shell_values: ArrayView1<f64>,
) -> Array1<f64> {
    let mut ao_values = Array1::<f64>::zeros(n_orbs);

    for (shell_idx, shell) in basis.shells.iter().enumerate() {
        for i in shell.sph_start..shell.sph_end {
            ao_values[i] = shell_values[shell_idx];
        }
    }

    ao_values
}

/// Scale AO-level charges so that shell totals match the mixed shell charges
/// Preserves the relative distribution of charges within each shell from the current iteration
#[inline]
pub fn scale_ao_to_match_shell(
    basis: &Basis,
    dq_ao_current: ArrayView1<f64>,
    dq_shell_mixed: ArrayView1<f64>,
) -> Array1<f64> {
    let n_orbs = dq_ao_current.len();
    let mut dq_ao = Array1::<f64>::zeros(n_orbs);

    for (shell_idx, shell) in basis.shells.iter().enumerate() {
        // Compute current shell total
        let mut shell_current_sum: f64 = 0.0;
        for i in shell.sph_start..shell.sph_end {
            shell_current_sum += dq_ao_current[i];
        }

        if shell_current_sum.abs() > 1e-14 {
            // Scale orbitals to match mixed shell total while preserving ratios
            let scale = dq_shell_mixed[shell_idx] / shell_current_sum;
            for i in shell.sph_start..shell.sph_end {
                dq_ao[i] = dq_ao_current[i] * scale;
            }
        } else {
            // Fallback to equal distribution if current shell sum is ~zero
            let n_orbs_in_shell = shell.sph_end - shell.sph_start;
            let charge_per_orb = dq_shell_mixed[shell_idx] / n_orbs_in_shell as f64;
            for i in shell.sph_start..shell.sph_end {
                dq_ao[i] = charge_per_orb;
            }
        }
    }

    dq_ao
}
