//! HOP (Hybrid Orbital Projection) gradient for FMO-xTB.
//!
//! Two terms:
//! 1. HOPSDER — overlap derivative: d/dR Tr(ρ × S^T × DD × S)
//! 2. HOPCODER — coefficient derivative: d/dR(DD) traced with density
//!
//! The HOP gradient is computed as a separate step (not fused into the shell-pair loop)
//! since it only involves the BDA atom's AO block.

use crate::hop::{
    compute_dd_matrix, compute_rotated_sp3, get_bda_ao_range, DetachedBond, HOP_SHIFT, SP3_COEFF_P,
};
use crate::initialization::basis::Basis;
use crate::integrals::{calc_overlap_derivative_d_shells, obara_saika_derivatives_all};
use nalgebra::Vector3;
use ndarray::prelude::*;

/// Compute the full HOP gradient for one detached bond.
///
/// Returns:
/// - Contributions are added to `local_gradient` (3 * n_local_atoms)
/// - `baa_gradient_3`: 3-element array for BAA atom (which may not be in the local basis)
///
/// `bda_local_atom`: local atom index of BDA in the basis
pub fn hop_gradient_single_bond(
    density: ArrayView2<f64>,
    s: ArrayView2<f64>,
    basis: &Basis,
    bda_local_atom: usize,
    bda_pos: &Vector3<f64>,
    baa_pos: &Vector3<f64>,
    local_gradient: &mut Array1<f64>,
    baa_gradient_3: &mut [f64; 3],
) {
    let bond_vec = *baa_pos - *bda_pos; // BDA→BAA direction
    let rotated_sp3 = compute_rotated_sp3(&bond_vec);
    let dd = compute_dd_matrix(rotated_sp3.view(), HOP_SHIFT);
    let (ao_start, nao) = get_bda_ao_range(basis, bda_local_atom);

    // Embed DD in nao×nao if needed (BDA may have d-orbitals too)
    let dd_full = if nao == dd.nrows() {
        dd.clone()
    } else {
        let mut dd_f = Array2::<f64>::zeros([nao, nao]);
        let sz = dd.nrows().min(nao);
        dd_f.slice_mut(s![..sz, ..sz])
            .assign(&dd.slice(s![..sz, ..sz]));
        dd_f
    };

    // === Term 1: HOPSDER (overlap derivative) ===
    hop_overlap_derivative_gradient(density, s, dd_full.view(), ao_start, nao, basis, local_gradient);

    // === Term 2: HOPCODER (coefficient derivative) ===
    let bda_grad_offset = 3 * bda_local_atom;
    hop_coefficient_derivative_gradient(
        density,
        s,
        rotated_sp3.view(),
        &bond_vec,
        ao_start,
        nao,
        bda_grad_offset,
        baa_gradient_3,
        local_gradient,
    );
}

/// General HOP gradient for one bond with explicit DD matrix, rotated_sp3, and HOPCODER sign.
///
/// For BDA atoms: pass DD = DD_bda = shift * |c><c|, coeff_sign = +1.0
/// For ghost atoms: pass DD = DD_ghost = shift * (I - |c><c|), coeff_sign = -1.0
///   (because d(DD_ghost)/d(bond) = -d(DD_bda)/d(bond))
///
/// `rotated_sp3` must use the same p-orbital ordering as the DD matrix.
/// `p_order` maps orbital index (0,1,2) to Cartesian component (x=0, y=1, z=2).
///   - Standard (s,px,py,pz): p_order = [0,1,2]
///   - xTB FMO-HOP (s,py,pz,px): p_order = [1,2,0]
pub fn hop_gradient_single_bond_general(
    density: ArrayView2<f64>,
    s: ArrayView2<f64>,
    basis: &Basis,
    bda_local_atom: usize,
    bda_pos: &Vector3<f64>,
    baa_pos: &Vector3<f64>,
    dd: ArrayView2<f64>,
    rotated_sp3: ArrayView1<f64>,
    p_order: &[usize; 3],
    coeff_sign: f64,
    local_gradient: &mut Array1<f64>,
    baa_gradient_3: &mut [f64; 3],
) {
    let bond_vec = *baa_pos - *bda_pos;
    let (ao_start, nao) = get_bda_ao_range(basis, bda_local_atom);

    // Embed DD in nao×nao if needed
    let dd_full = if nao == dd.nrows() {
        dd.to_owned()
    } else {
        let mut dd_f = Array2::<f64>::zeros([nao, nao]);
        let sz = dd.nrows().min(nao);
        dd_f.slice_mut(s![..sz, ..sz])
            .assign(&dd.slice(s![..sz, ..sz]));
        dd_f
    };

    // Term 1: HOPSDER with the given DD
    hop_overlap_derivative_gradient(density, s, dd_full.view(), ao_start, nao, basis, local_gradient);

    // Term 2: HOPCODER with sign factor
    let bda_grad_offset = 3 * bda_local_atom;
    hop_coefficient_derivative_gradient_signed(
        density,
        s,
        rotated_sp3.view(),
        &bond_vec,
        p_order,
        ao_start,
        nao,
        bda_grad_offset,
        baa_gradient_3,
        local_gradient,
        coeff_sign,
    );
}

/// HOPCODER with sign factor for BDA (+1) vs ghost (-1).
///
/// `p_order` maps orbital index j (0,1,2) to Cartesian component for b_hat.
/// Standard (s,px,py,pz): p_order = [0,1,2] → dc[j+1] = SP3_COEFF_P * d_bhat[j]
/// xTB FMO-HOP (s,py,pz,px): p_order = [1,2,0] → dc[j+1] = SP3_COEFF_P * d_bhat[p_order[j]]
fn hop_coefficient_derivative_gradient_signed(
    density: ArrayView2<f64>,
    s: ArrayView2<f64>,
    rotated_sp3: ArrayView1<f64>,
    bond_vec: &Vector3<f64>,
    p_order: &[usize; 3],
    bda_ao_start: usize,
    nao_bda: usize,
    bda_grad_offset: usize,
    baa_gradient_3: &mut [f64; 3],
    gradient: &mut Array1<f64>,
    sign: f64,
) {
    let bond_len = bond_vec.norm();
    if bond_len < 1e-14 {
        return;
    }
    let b_hat = bond_vec / bond_len;
    let ncoeff = rotated_sp3.len();
    let sz = ncoeff.min(nao_bda);

    let s_bda = s.slice(s![.., bda_ao_start..bda_ao_start + nao_bda]);
    let st_rho_s = s_bda.t().dot(&density.dot(&s_bda));

    for ic in 0..3usize {
        // d(b_hat[k])/d(bond_ic) = (delta(k,ic) - b_hat[k]*b_hat[ic]) / bond_len
        let mut d_bhat = [0.0f64; 3];
        for j in 0..3 {
            let delta = if j == ic { 1.0 } else { 0.0 };
            d_bhat[j] = (delta - b_hat[j] * b_hat[ic]) / bond_len;
        }

        // dc[j+1] = SP3_COEFF_P * d_bhat[p_order[j]] to match p-orbital ordering
        let mut dc = [0.0f64; 4];
        for j in 0..3 {
            dc[j + 1] = SP3_COEFF_P * d_bhat[p_order[j]];
        }

        let mut cont = 0.0;
        for a in 0..sz {
            for b in 0..sz {
                cont += HOP_SHIFT
                    * (dc[a] * rotated_sp3[b] + rotated_sp3[a] * dc[b])
                    * st_rho_s[[a, b]];
            }
        }

        // Apply sign factor: +1 for BDA, -1 for ghost
        gradient[bda_grad_offset + ic] -= sign * cont;
        baa_gradient_3[ic] += sign * cont;
    }
}

/// HOPSDER: overlap derivative contribution to HOP gradient.
///
/// P_HOP = S_bda × DD × S_bda^T where S_bda = S[:, bda_ao_start..bda_ao_end]
///
/// d/dR Tr(ρ × P_HOP) = 2 × sum_{μ,ν_bda} dS[μ,ν_bda]/dR × W_right[ν_bda-offset, μ]
///
/// where W_right = DD × S_bda^T × ρ
fn hop_overlap_derivative_gradient(
    density: ArrayView2<f64>,
    s: ArrayView2<f64>,
    dd: ArrayView2<f64>,
    bda_ao_start: usize,
    nao_bda: usize,
    basis: &Basis,
    gradient: &mut Array1<f64>,
) {
    let bda_ao_end = bda_ao_start + nao_bda;

    // S_bda: S[:, bda_ao_start..bda_ao_end]
    let s_bda = s.slice(s![.., bda_ao_start..bda_ao_end]);

    // W_right = DD × S_bda^T × ρ  (nao_bda × n_orbs)
    let w_right = dd.dot(&s_bda.t()).dot(&density);

    // Loop over shell pairs to accumulate dS contributions
    for shell_i in basis.shells.iter() {
        let at_i = shell_i.atom_index;
        for shell_j in basis.shells.iter() {
            let at_j = shell_j.atom_index;
            if at_i == at_j {
                continue;
            }

            // Check overlap with BDA AO range
            let j_in_bda = shell_j.sph_start < bda_ao_end && shell_j.sph_end > bda_ao_start;
            let i_in_bda = shell_i.sph_start < bda_ao_end && shell_i.sph_end > bda_ao_start;
            if !j_in_bda && !i_in_bda {
                continue;
            }

            let shell_i_has_d = shell_i.angular_momentum >= 2;
            let shell_j_has_d = shell_j.angular_momentum >= 2;

            if !shell_i_has_d && !shell_j_has_d {
                // s/p shells
                for idx_i in shell_i.sph_start..shell_i.sph_end {
                    let idx_i_local = idx_i - shell_i.sph_start;
                    for idx_j in shell_j.sph_start..shell_j.sph_end {
                        let idx_j_local = idx_j - shell_j.sph_start;

                        // Weight from both directions (S is symmetric in P_HOP)
                        let mut weight = 0.0;
                        if j_in_bda && idx_j >= bda_ao_start && idx_j < bda_ao_end {
                            weight += w_right[[idx_j - bda_ao_start, idx_i]];
                        }
                        if i_in_bda && idx_i >= bda_ao_start && idx_i < bda_ao_end {
                            weight += w_right[[idx_i - bda_ao_start, idx_j]];
                        }

                        if weight.abs() < 1e-30 {
                            continue;
                        }

                        let orbital1 = &basis.basis_functions[shell_i.start + idx_i_local];
                        let orbital2 = &basis.basis_functions[shell_j.start + idx_j_local];
                        let norm_prod = orbital1.contracted_norm * orbital2.contracted_norm;

                        let ds = obara_saika_derivatives_all(orbital1, orbital2);
                        for dir in 0..3 {
                            gradient[3 * at_i + dir] += ds[dir] * norm_prod * weight;
                            gradient[3 * at_j + dir] -= ds[dir] * norm_prod * weight;
                        }
                    }
                }
            } else {
                // d-shell handling
                let ds_d = calc_overlap_derivative_d_shells(basis, shell_i, shell_j);
                let sph_dim_i = shell_i.sph_end - shell_i.sph_start;
                let sph_dim_j = shell_j.sph_end - shell_j.sph_start;

                for sph_i in 0..sph_dim_i {
                    let idx_i = shell_i.sph_start + sph_i;
                    for sph_j in 0..sph_dim_j {
                        let idx_j = shell_j.sph_start + sph_j;

                        let mut weight = 0.0;
                        if j_in_bda && idx_j >= bda_ao_start && idx_j < bda_ao_end {
                            weight += w_right[[idx_j - bda_ao_start, idx_i]];
                        }
                        if i_in_bda && idx_i >= bda_ao_start && idx_i < bda_ao_end {
                            weight += w_right[[idx_i - bda_ao_start, idx_j]];
                        }

                        if weight.abs() < 1e-30 {
                            continue;
                        }

                        for dir in 0..3 {
                            gradient[3 * at_i + dir] += ds_d[[dir, sph_i, sph_j]] * weight;
                            gradient[3 * at_j + dir] += ds_d[[3 + dir, sph_i, sph_j]] * weight;
                        }
                    }
                }
            }
        }
    }
}

/// HOPCODER: coefficient derivative contribution to HOP gradient.
///
/// For each Cartesian direction ic:
///   d(b̂_j)/d(bond_ic) = (δ_{j,ic} - b̂_j × b̂_ic) / |bond|
///   dc/d(bond_ic) = [0, √3/2 × d(b̂)/d(bond_ic)]
///   dDD/d(bond_ic) = shift × (dc · c^T + c · dc^T)
///   cont = Tr(ρ × S_bda × dDD × S_bda^T)
///        = sum_{a,b} dDD[a,b] × (S_bda^T × ρ × S_bda)[a,b]
///
/// bond = R_baa - R_bda, so:
///   grad[bda] -= cont
///   grad[baa] += cont
fn hop_coefficient_derivative_gradient(
    density: ArrayView2<f64>,
    s: ArrayView2<f64>,
    rotated_sp3: ArrayView1<f64>,
    bond_vec: &Vector3<f64>,
    bda_ao_start: usize,
    nao_bda: usize,
    bda_grad_offset: usize,
    baa_gradient_3: &mut [f64; 3],
    gradient: &mut Array1<f64>,
) {
    let bond_len = bond_vec.norm();
    if bond_len < 1e-14 {
        return;
    }
    let b_hat = bond_vec / bond_len;
    let ncoeff = rotated_sp3.len(); // 4 for s+p
    let sz = ncoeff.min(nao_bda);

    // S_bda^T × ρ × S_bda (nao_bda × nao_bda)
    let s_bda = s.slice(s![.., bda_ao_start..bda_ao_start + nao_bda]);
    let st_rho_s = s_bda.t().dot(&density.dot(&s_bda));

    for ic in 0..3usize {
        // d(b̂_j)/d(bond_ic) = (δ_{j,ic} - b̂_j * b̂_ic) / |bond|
        let mut d_bhat = [0.0f64; 3];
        for j in 0..3 {
            let delta = if j == ic { 1.0 } else { 0.0 };
            d_bhat[j] = (delta - b_hat[j] * b_hat[ic]) / bond_len;
        }

        // dc = [0, √3/2 * d(b̂_x), √3/2 * d(b̂_y), √3/2 * d(b̂_z)]
        let mut dc = [0.0f64; 4];
        for j in 0..3 {
            dc[j + 1] = SP3_COEFF_P * d_bhat[j];
        }

        // Tr(dDD × st_rho_s) = shift × sum_{a,b} (dc[a]*c[b] + c[a]*dc[b]) × st_rho_s[a,b]
        let mut cont = 0.0;
        for a in 0..sz {
            for b in 0..sz {
                cont += HOP_SHIFT
                    * (dc[a] * rotated_sp3[b] + rotated_sp3[a] * dc[b])
                    * st_rho_s[[a, b]];
            }
        }

        // bond = R_baa - R_bda → ∂bond/∂R_bda = -I, ∂bond/∂R_baa = +I
        gradient[bda_grad_offset + ic] -= cont;
        baa_gradient_3[ic] += cont;
    }
}
