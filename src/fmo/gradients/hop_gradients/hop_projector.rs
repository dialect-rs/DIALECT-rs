//! HOP projector gradient for FMO-DFTB.

use super::helpers::build_orbital_offsets;
use crate::defaults::PROXIMITY_CUTOFF;
use crate::fmo::scc_hop::hop_data::{
    compute_bda_dd_matrix, compute_ghost_nonbond_dd, compute_rotated_sp3_dftb,
    get_bda_ao_range_dftb, HopData, DFTB_SP3_COEFF_P,
};
use crate::fmo::scc_hop::monomer::MonomerHopScc;
use crate::fmo::scc_hop::pair::PairHopScc;
use crate::fmo::Pair;
use crate::initialization::Atom;
use crate::initialization::parameters::SlaterKoster;
use crate::param::slako_transformations::{
    directional_cosines, slako_transformation_gradients_fast, SplineCache,
};
use dialect_xtb::hop::{DetachedBond, HOP_SHIFT};
use nalgebra::Vector3;
use ndarray::prelude::*;

/// Compute the HOP gradient for one detached bond (DFTB version).
///
/// Handles both BDA bonds (1 bond-pointing hybrid) and ghost bonds (3 non-bond hybrids).
///
/// Arguments:
/// - `density`: density matrix for the fragment/pair (n_ext_orbs × n_ext_orbs)
/// - `s`: overlap matrix
/// - `ext_atoms`: extended atoms (real + ghost)
/// - `dd`: the DD matrix for this bond (4×4 for s+p)
/// - `bda_local_atom`: local atom index of the BDA/ghost atom
/// - `bda_pos`, `baa_pos`: atom positions for BDA and BAA
/// - `local_gradient`: accumulated gradient (3 × n_ext_atoms)
/// - `baa_gradient_3`: 3-element array for BAA (which may not be local)
/// - `slako`: Slater-Koster parameters for SK overlap derivatives
pub fn hop_gradient_single_bond_dftb(
    density: ArrayView2<f64>,
    s: ArrayView2<f64>,
    ext_atoms: &[Atom],
    dd: ArrayView2<f64>,
    bda_local_atom: usize,
    bda_pos: &Vector3<f64>,
    baa_pos: &Vector3<f64>,
    local_gradient: &mut Array1<f64>,
    baa_gradient_3: &mut [f64; 3],
    slako: &SlaterKoster,
) {
    let (ao_start, nao) = get_bda_ao_range_dftb(ext_atoms, bda_local_atom);

    // Embed DD in nao×nao if needed
    let dd_full = if nao == dd.nrows() {
        dd.to_owned()
    } else {
        let mut dd_f = Array2::<f64>::zeros([nao, nao]);
        let sz = dd.nrows().min(nao);
        dd_f.slice_mut(s![..sz, ..sz]).assign(&dd.slice(s![..sz, ..sz]));
        dd_f
    };

    // Term 1: HOPSDER (overlap derivative)
    hop_overlap_derivative_gradient_dftb(
        density, s, dd_full.view(), ao_start, nao, ext_atoms, local_gradient, slako,
    );

    // Term 2: HOPCODER (coefficient derivative)
    let bond_vec = *baa_pos - *bda_pos;
    let rotated_sp3 = compute_rotated_sp3_dftb(&bond_vec);

    // For ghost bonds (3 non-bond hybrids), HOPCODER needs special handling.
    // The ghost DD = shift * Σ_{i=2,3,4} h_i h_i^T.
    // Its derivative w.r.t. bond direction involves the tetrahedral identity.
    // However, we can compute it as: d(ghost_DD)/d(bond) = -d(bda_DD)/d(bond)
    // because Σ_{i=1..4} h_i h_i^T = const, so d(3-hybrid sum) = -d(1-hybrid).
    //
    // But this only works if the DD is for the complementary set.
    // For the BDA DD (1 hybrid), HOPCODER is straightforward.
    // For ghost DD (3 hybrids), we use the negative of BDA HOPCODER.
    let is_ghost_dd = dd.nrows() >= 4 && {
        // Check if this is a ghost DD by checking the [0,0] element
        // BDA DD: shift * c_s^2 ≈ shift * 0.316 (for c_s=0.562)
        // Ghost DD: shift * 3 * c_s^2 ≈ shift * 0.947
        let bda_ss = HOP_SHIFT * crate::fmo::scc_hop::hop_data::DFTB_SP3_COEFF_S.powi(2);
        (dd[[0, 0]] - 3.0 * bda_ss).abs() < 1.0  // ghost has 3*c_s^2, BDA has c_s^2
    };

    let bda_grad_offset = 3 * bda_local_atom;
    if is_ghost_dd {
        // Ghost DD derivative = -BDA DD derivative (complementary set)
        hop_coefficient_derivative_gradient_dftb(
            density,
            s,
            rotated_sp3.view(),
            &bond_vec,
            ao_start,
            nao,
            bda_grad_offset,
            baa_gradient_3,
            local_gradient,
            -1.0,  // negative sign for ghost complement
        );
    } else {
        hop_coefficient_derivative_gradient_dftb(
            density,
            s,
            rotated_sp3.view(),
            &bond_vec,
            ao_start,
            nao,
            bda_grad_offset,
            baa_gradient_3,
            local_gradient,
            1.0,  // positive sign for BDA
        );
    }
}

/// HOPSDER: overlap derivative contribution to HOP gradient (DFTB version).
///
/// Uses SK-based overlap derivatives instead of xTB's Obara-Saika.
///
/// P_HOP = S_bda × DD × S_bda^T
/// d/dR Tr(ρ × P_HOP) = 2 × Σ dS[μ,ν_bda]/dR × W_right[ν_bda - offset, μ]
/// where W_right = DD × S_bda^T × ρ
pub fn hop_overlap_derivative_gradient_dftb(
    density: ArrayView2<f64>,
    s: ArrayView2<f64>,
    dd: ArrayView2<f64>,
    bda_ao_start: usize,
    nao_bda: usize,
    ext_atoms: &[Atom],
    gradient: &mut Array1<f64>,
    slako: &SlaterKoster,
) {
    let bda_ao_end = bda_ao_start + nao_bda;

    // S_bda: S[:, bda_ao_start..bda_ao_end]
    let s_bda = s.slice(s![.., bda_ao_start..bda_ao_end]);

    // W_right = DD × S_bda^T × ρ  (nao_bda × n_orbs)
    let w_right = dd.dot(&s_bda.t()).dot(&density);

    let orbital_offsets = build_orbital_offsets(ext_atoms);
    let n_atoms = ext_atoms.len();

    // Loop over atom pairs to accumulate dS contributions
    for i in 0..n_atoms {
        let atomi = &ext_atoms[i];
        let mu_start = orbital_offsets[i];
        let mu_end = orbital_offsets[i + 1];

        for j in (i + 1)..n_atoms {
            let atomj = &ext_atoms[j];
            let nu_start = orbital_offsets[j];
            let nu_end = orbital_offsets[j + 1];

            // Check overlap with BDA AO range
            let j_in_bda = nu_start < bda_ao_end && nu_end > bda_ao_start;
            let i_in_bda = mu_start < bda_ao_end && mu_end > bda_ao_start;
            if !j_in_bda && !i_in_bda {
                continue;
            }

            let r_vec = atomi - atomj;
            let dist = r_vec.norm();
            if dist >= PROXIMITY_CUTOFF {
                continue;
            }

            let (r, x, y, z) = if atomi <= atomj {
                directional_cosines(&atomi.xyz, &atomj.xyz)
            } else {
                directional_cosines(&atomj.xyz, &atomi.xyz)
            };

            let skt = slako.get(atomi.kind, atomj.kind);
            let s_cache = SplineCache::new(r, &skt.s_spline);

            // Iterate over orbital pairs
            let mut mu = mu_start;
            for orbi in atomi.valorbs.iter() {
                let mut nu = nu_start;
                for orbj in atomj.valorbs.iter() {
                    // Weight from both directions (S is symmetric in P_HOP)
                    let mut weight = 0.0;
                    if j_in_bda && nu >= bda_ao_start && nu < bda_ao_end {
                        weight += w_right[[nu - bda_ao_start, mu]];
                    }
                    if i_in_bda && mu >= bda_ao_start && mu < bda_ao_end {
                        weight += w_right[[mu - bda_ao_start, nu]];
                    }

                    if weight.abs() < 1e-30 {
                        nu += 1;
                        continue;
                    }

                    // Compute dS gradients
                    let (ds_i, ds_j) = if atomi <= atomj {
                        let s_grad = slako_transformation_gradients_fast(
                            r, x, y, z, &s_cache, orbi.l, orbi.m, orbj.l, orbj.m,
                        );
                        (
                            [-s_grad[0], -s_grad[1], -s_grad[2]],
                            s_grad,
                        )
                    } else {
                        let s_grad = slako_transformation_gradients_fast(
                            r, x, y, z, &s_cache, orbj.l, orbj.m, orbi.l, orbi.m,
                        );
                        (
                            s_grad,
                            [-s_grad[0], -s_grad[1], -s_grad[2]],
                        )
                    };

                    for dir in 0..3 {
                        gradient[3 * i + dir] += 2.0 * ds_i[dir] * weight;
                        gradient[3 * j + dir] += 2.0 * ds_j[dir] * weight;
                    }

                    nu += 1;
                }
                mu += 1;
            }
        }
    }
}

/// HOPCODER: coefficient derivative contribution to HOP gradient (DFTB version).
///
/// Uses DFTB sp3 coefficients and p-orbital ordering (py, pz, px).
///
/// For BDA (sign_factor = 1.0):
///   dDD/d(bond) = shift × (dc · c^T + c · dc^T)
///   dc = [0, c_p*d(b̂_y), c_p*d(b̂_z), c_p*d(b̂_x)]   (DFTB ordering)
///
/// For ghost (sign_factor = -1.0):
///   d(ghost_DD)/d(bond) = -d(bda_DD)/d(bond)
///   (using tetrahedral identity: Σ_all h_i h_i^T = const)
pub fn hop_coefficient_derivative_gradient_dftb(
    density: ArrayView2<f64>,
    s: ArrayView2<f64>,
    rotated_sp3: ArrayView1<f64>,
    bond_vec: &Vector3<f64>,
    bda_ao_start: usize,
    nao_bda: usize,
    bda_grad_offset: usize,
    baa_gradient_3: &mut [f64; 3],
    gradient: &mut Array1<f64>,
    sign_factor: f64,
) {
    let bond_len = bond_vec.norm();
    if bond_len < 1e-14 {
        return;
    }
    let b_hat = bond_vec / bond_len;
    let ncoeff = rotated_sp3.len();
    let sz = ncoeff.min(nao_bda);

    // S_bda^T × ρ × S_bda (nao_bda × nao_bda)
    let s_bda = s.slice(s![.., bda_ao_start..bda_ao_start + nao_bda]);
    let st_rho_s = s_bda.t().dot(&density.dot(&s_bda));

    for ic in 0..3usize {
        // d(b̂_j)/d(bond_ic) = (δ_{j,ic} - b̂_j × b̂_ic) / |bond|
        let mut d_bhat = [0.0f64; 3];
        for j in 0..3 {
            let delta = if j == ic { 1.0 } else { 0.0 };
            d_bhat[j] = (delta - b_hat[j] * b_hat[ic]) / bond_len;
        }

        // dc in DFTB ordering: [0, c_p*d(b̂_y), c_p*d(b̂_z), c_p*d(b̂_x)]
        let mut dc = [0.0f64; 4];
        dc[1] = DFTB_SP3_COEFF_P * d_bhat[1]; // d(b̂_y) for p_y
        dc[2] = DFTB_SP3_COEFF_P * d_bhat[2]; // d(b̂_z) for p_z
        dc[3] = DFTB_SP3_COEFF_P * d_bhat[0]; // d(b̂_x) for p_x

        // Tr(dDD × st_rho_s) = shift × Σ_{a,b} (dc[a]*c[b] + c[a]*dc[b]) × st_rho_s[a,b]
        let mut cont = 0.0;
        for a in 0..sz {
            for b in 0..sz {
                cont += HOP_SHIFT
                    * (dc[a] * rotated_sp3[b] + rotated_sp3[a] * dc[b])
                    * st_rho_s[[a, b]];
            }
        }

        // Apply sign factor (positive for BDA, negative for ghost complement)
        cont *= sign_factor;

        // bond = R_baa - R_bda → ∂bond/∂R_bda = -I, ∂bond/∂R_baa = +I
        gradient[bda_grad_offset + ic] -= cont;
        baa_gradient_3[ic] += cont;
    }
}

/// Compute the total HOP projector gradient for FMO-DFTB.
///
/// Follows the same delta pattern as xTB's `compute_hop_gradient_fmo_hop()`:
/// 1. Monomer HOP gradients (BDA + ghost projections)
/// 2. Pair HOP gradients (delta: pair - monomer for partial + healed bonds)
pub fn compute_hop_gradient_fmo_dftb(
    hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    pair_states: &[PairHopScc],
    pairs: &[Pair],
    atoms: &[Atom],
    slako: &SlaterKoster,
    hop_total: &mut Array1<f64>,
) {
    if hop_data.detached_bonds.is_empty() {
        return;
    }

    let detached_bonds = &hop_data.detached_bonds;

    // Step 1: Monomer HOP gradients
    for (frag_idx, mono) in mono_states.iter().enumerate() {
        let fi = &hop_data.frag_info[frag_idx];
        let ext_atoms = &mono.ext_atoms;
        let n_atoms = mono.n_ext_atoms;

        // BDA bonds (bond-pointing hybrid projection)
        let bda_bonds: Vec<&DetachedBond> = detached_bonds
            .iter()
            .filter(|b| b.bda_fragment == frag_idx)
            .collect();

        // Ghost bonds (3 non-bond hybrid projection)
        let ghost_bonds: Vec<&DetachedBond> = detached_bonds
            .iter()
            .filter(|b| b.baa_fragment == frag_idx)
            .collect();

        if bda_bonds.is_empty() && ghost_bonds.is_empty() {
            continue;
        }

        let p = mono.p.view();
        let s = mono.s.view();
        let mut local_grad = Array1::<f64>::zeros(3 * n_atoms);

        // BDA projections
        let frag_atom_start = hop_data.monomer_indices[frag_idx][0];
        for bond in &bda_bonds {
            let bda_local = bond.bda_global - frag_atom_start;
            let bda_pos = &atoms[bond.bda_global].xyz;
            let baa_pos = &atoms[bond.baa_global].xyz;
            let bond_vec = *baa_pos - *bda_pos;

            let rotated_sp3 = compute_rotated_sp3_dftb(&bond_vec);
            let dd = compute_bda_dd_matrix(rotated_sp3.view(), HOP_SHIFT);

            let mut baa_grad_3 = [0.0f64; 3];
            hop_gradient_single_bond_dftb(
                p, s, ext_atoms, dd.view(), bda_local, bda_pos, baa_pos,
                &mut local_grad, &mut baa_grad_3, slako,
            );
            for k in 0..3 {
                hop_total[3 * bond.baa_global + k] += baa_grad_3[k];
            }
        }

        // Ghost projections
        for (ghost_idx, bond) in ghost_bonds.iter().enumerate() {
            let ghost_local = fi.n_real_atoms + ghost_idx;
            let bda_pos = &atoms[bond.bda_global].xyz; // ghost is at BDA position
            let baa_pos = &atoms[bond.baa_global].xyz;
            let bond_vec = *baa_pos - *bda_pos;

            let dd_ghost = compute_ghost_nonbond_dd(&bond_vec, HOP_SHIFT);

            let mut baa_grad_3 = [0.0f64; 3];
            hop_gradient_single_bond_dftb(
                p, s, ext_atoms, dd_ghost.view(), ghost_local, bda_pos, baa_pos,
                &mut local_grad, &mut baa_grad_3, slako,
            );
            for k in 0..3 {
                hop_total[3 * bond.baa_global + k] += baa_grad_3[k];
            }
        }

        // Map local gradient to global (real atoms + ghost → BDA global)
        for (local_idx, &global_idx) in hop_data.monomer_indices[frag_idx].iter().enumerate() {
            for k in 0..3 {
                hop_total[3 * global_idx + k] += local_grad[3 * local_idx + k];
            }
        }
        // Ghost atoms → BDA global (ghost is at BDA's position)
        for (ghost_idx, bond) in ghost_bonds.iter().enumerate() {
            let local_idx = fi.n_real_atoms + ghost_idx;
            for k in 0..3 {
                hop_total[3 * bond.bda_global + k] += local_grad[3 * local_idx + k];
            }
        }
    }

    // Step 2: Pair HOP gradients (delta: pair - monomer)
    for (pair_idx, pair) in pairs.iter().enumerate() {
        let ps = &pair_states[pair_idx];
        let fi_i = &hop_data.frag_info[pair.i];
        let fi_j = &hop_data.frag_info[pair.j];
        let frag_range_i_start = hop_data.monomer_indices[pair.i][0];
        let frag_range_j_start = hop_data.monomer_indices[pair.j][0];
        let n_real_i = fi_i.n_real_atoms;
        let _n_real_j = fi_j.n_real_atoms;

        // Classify bonds for this pair
        let mut partial_bda_bonds: Vec<&DetachedBond> = Vec::new();
        let mut partial_baa_bonds: Vec<&DetachedBond> = Vec::new();
        let mut healed_bonds: Vec<&DetachedBond> = Vec::new();

        for bond in detached_bonds {
            let bda_in_pair = bond.bda_fragment == pair.i || bond.bda_fragment == pair.j;
            let baa_in_pair = bond.baa_fragment == pair.i || bond.baa_fragment == pair.j;
            if bda_in_pair && baa_in_pair {
                healed_bonds.push(bond);
            } else if bda_in_pair && !baa_in_pair {
                partial_bda_bonds.push(bond);
            } else if !bda_in_pair && baa_in_pair {
                partial_baa_bonds.push(bond);
            }
        }

        // Pair HOP gradient for partial bonds
        if !partial_bda_bonds.is_empty() || !partial_baa_bonds.is_empty() {
            let p_pair = ps.p.view();
            let s_pair = ps.s.view();
            let ext_atoms = &ps.ext_atoms;
            let n_atoms_pair = ps.n_ext_atoms;
            let mut pair_local_grad = Array1::<f64>::zeros(3 * n_atoms_pair);

            // BDA projections in pair
            for bond in &partial_bda_bonds {
                let bda_local = if bond.bda_fragment == pair.i {
                    bond.bda_global - frag_range_i_start
                } else {
                    n_real_i + (bond.bda_global - frag_range_j_start)
                };

                let bda_pos = &atoms[bond.bda_global].xyz;
                let baa_pos = &atoms[bond.baa_global].xyz;
                let bond_vec_v = *baa_pos - *bda_pos;
                let rotated_sp3 = compute_rotated_sp3_dftb(&bond_vec_v);
                let dd = compute_bda_dd_matrix(rotated_sp3.view(), HOP_SHIFT);

                let mut baa_grad_3 = [0.0f64; 3];
                hop_gradient_single_bond_dftb(
                    p_pair, s_pair, ext_atoms, dd.view(), bda_local, bda_pos, baa_pos,
                    &mut pair_local_grad, &mut baa_grad_3, slako,
                );
                for k in 0..3 {
                    hop_total[3 * bond.baa_global + k] += baa_grad_3[k];
                }
            }

            // Ghost projections in pair
            let n_real_atoms = ps.n_real_atoms;
            for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
                let ghost_local = n_real_atoms + ghost_idx;
                let bda_pos = &atoms[bond.bda_global].xyz;
                let baa_pos = &atoms[bond.baa_global].xyz;
                let bond_vec_v = *baa_pos - *bda_pos;
                let dd_ghost = compute_ghost_nonbond_dd(&bond_vec_v, HOP_SHIFT);

                let mut baa_grad_3 = [0.0f64; 3];
                hop_gradient_single_bond_dftb(
                    p_pair, s_pair, ext_atoms, dd_ghost.view(), ghost_local, bda_pos, baa_pos,
                    &mut pair_local_grad, &mut baa_grad_3, slako,
                );
                for k in 0..3 {
                    hop_total[3 * bond.baa_global + k] += baa_grad_3[k];
                }
            }

            // Map pair local gradient to global
            for (local_idx, global_idx) in hop_data.monomer_indices[pair.i].iter().enumerate() {
                for k in 0..3 {
                    hop_total[3 * global_idx + k] += pair_local_grad[3 * local_idx + k];
                }
            }
            for (local_idx, global_idx) in hop_data.monomer_indices[pair.j].iter().enumerate() {
                for k in 0..3 {
                    hop_total[3 * global_idx + k] +=
                        pair_local_grad[3 * (n_real_i + local_idx) + k];
                }
            }
            // Pair ghost atoms → BDA global (ghost is at BDA's position)
            for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
                let local_idx = n_real_atoms + ghost_idx;
                for k in 0..3 {
                    hop_total[3 * bond.bda_global + k] += pair_local_grad[3 * local_idx + k];
                }
            }

            // Subtract monomer contributions for partial bonds
            for bond in &partial_bda_bonds {
                subtract_monomer_hop_gradient(
                    bond, mono_states, hop_data, atoms, slako, detached_bonds, hop_total,
                );
            }
            for bond in &partial_baa_bonds {
                // For partial-BAA bonds, the ghost is in monomer and pair.
                // Subtract the monomer's ghost contribution.
                subtract_monomer_ghost_hop_gradient(
                    bond, mono_states, hop_data, atoms, slako, detached_bonds, hop_total,
                );
            }
        }

        // Subtract monomer contributions for healed bonds
        // (pair has no HOP for these, but monomer does → subtract both BDA and ghost)
        for bond in &healed_bonds {
            subtract_monomer_hop_gradient(
                bond, mono_states, hop_data, atoms, slako, detached_bonds, hop_total,
            );
            // Also subtract the BAA monomer's ghost projection (ghost at BDA position)
            subtract_monomer_ghost_hop_gradient(
                bond, mono_states, hop_data, atoms, slako, detached_bonds, hop_total,
            );
        }
    }
}

/// Subtract monomer BDA HOP gradient for a given bond.
///
/// This recomputes the monomer's HOP gradient for the specific BDA bond
/// and subtracts it from the global gradient.
fn subtract_monomer_hop_gradient(
    bond: &DetachedBond,
    mono_states: &[MonomerHopScc],
    hop_data: &HopData,
    atoms: &[Atom],
    slako: &SlaterKoster,
    detached_bonds: &[DetachedBond],
    hop_total: &mut Array1<f64>,
) {
    let bda_frag = bond.bda_fragment;
    let mono = &mono_states[bda_frag];
    let fi = &hop_data.frag_info[bda_frag];
    let frag_atom_start = hop_data.monomer_indices[bda_frag][0];
    let bda_local = bond.bda_global - frag_atom_start;

    let p = mono.p.view();
    let s = mono.s.view();
    let ext_atoms = &mono.ext_atoms;
    let n_atoms = mono.n_ext_atoms;

    let bda_pos = &atoms[bond.bda_global].xyz;
    let baa_pos = &atoms[bond.baa_global].xyz;
    let bond_vec = *baa_pos - *bda_pos;
    let rotated_sp3 = compute_rotated_sp3_dftb(&bond_vec);
    let dd = compute_bda_dd_matrix(rotated_sp3.view(), HOP_SHIFT);

    let mut mono_local_grad = Array1::<f64>::zeros(3 * n_atoms);
    let mut baa_grad_3 = [0.0f64; 3];

    hop_gradient_single_bond_dftb(
        p, s, ext_atoms, dd.view(), bda_local, bda_pos, baa_pos,
        &mut mono_local_grad, &mut baa_grad_3, slako,
    );

    // Subtract real atom contributions
    for (local_idx, &global_idx) in hop_data.monomer_indices[bda_frag].iter().enumerate() {
        for k in 0..3 {
            hop_total[3 * global_idx + k] -= mono_local_grad[3 * local_idx + k];
        }
    }
    // Subtract ghost atom contributions (ghost is at BDA's position → bda_global)
    let ghost_bonds: Vec<&DetachedBond> = detached_bonds
        .iter()
        .filter(|b| b.baa_fragment == bda_frag)
        .collect();
    for (ghost_idx, gbond) in ghost_bonds.iter().enumerate() {
        let local_idx = fi.n_real_atoms + ghost_idx;
        for k in 0..3 {
            hop_total[3 * gbond.bda_global + k] -= mono_local_grad[3 * local_idx + k];
        }
    }
    // Subtract BAA gradient (HOPCODER: bond direction derivative → goes to BAA)
    for k in 0..3 {
        hop_total[3 * bond.baa_global + k] -= baa_grad_3[k];
    }
}

/// Subtract monomer ghost HOP gradient for a partial-BAA bond.
///
/// The ghost exists in both monomer and pair → subtract monomer's ghost contribution.
fn subtract_monomer_ghost_hop_gradient(
    bond: &DetachedBond,
    mono_states: &[MonomerHopScc],
    hop_data: &HopData,
    atoms: &[Atom],
    slako: &SlaterKoster,
    detached_bonds: &[DetachedBond],
    hop_total: &mut Array1<f64>,
) {
    let baa_frag = bond.baa_fragment;
    let mono = &mono_states[baa_frag];
    let fi = &hop_data.frag_info[baa_frag];
    let ext_atoms = &mono.ext_atoms;
    let n_atoms = mono.n_ext_atoms;

    // Find ghost index in monomer
    let ghost_bonds: Vec<&DetachedBond> = detached_bonds
        .iter()
        .filter(|b| b.baa_fragment == baa_frag)
        .collect();
    let ghost_idx = ghost_bonds.iter().position(|b| b.bda_global == bond.bda_global && b.baa_global == bond.baa_global);
    let ghost_idx = match ghost_idx {
        Some(idx) => idx,
        None => return,
    };

    let ghost_local = fi.n_real_atoms + ghost_idx;
    let bda_pos = &atoms[bond.bda_global].xyz;
    let baa_pos = &atoms[bond.baa_global].xyz;
    let bond_vec = *baa_pos - *bda_pos;
    let dd_ghost = compute_ghost_nonbond_dd(&bond_vec, HOP_SHIFT);

    let p = mono.p.view();
    let s = mono.s.view();

    let mut mono_local_grad = Array1::<f64>::zeros(3 * n_atoms);
    let mut baa_grad_3 = [0.0f64; 3];

    hop_gradient_single_bond_dftb(
        p, s, ext_atoms, dd_ghost.view(), ghost_local, bda_pos, baa_pos,
        &mut mono_local_grad, &mut baa_grad_3, slako,
    );

    // Subtract real atom contributions
    for (local_idx, &global_idx) in hop_data.monomer_indices[baa_frag].iter().enumerate() {
        for k in 0..3 {
            hop_total[3 * global_idx + k] -= mono_local_grad[3 * local_idx + k];
        }
    }
    // Subtract ghost atom contributions (ghost is at BDA's position → bda_global)
    for (gi, gbond) in ghost_bonds.iter().enumerate() {
        let local_idx = fi.n_real_atoms + gi;
        for k in 0..3 {
            hop_total[3 * gbond.bda_global + k] -= mono_local_grad[3 * local_idx + k];
        }
    }
    // Subtract BAA gradient (HOPCODER: bond direction derivative → goes to BAA)
    for k in 0..3 {
        hop_total[3 * bond.baa_global + k] -= baa_grad_3[k];
    }
}
