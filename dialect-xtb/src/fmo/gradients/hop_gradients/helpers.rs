//! Helper functions for the xTB HOP gradient.
//!
//! Provides utility functions for ghost atom mapping, CTMUL/SHIFTCT computation,
//! ZREF/QREF-scaled repulsive gradient, and scatter operations.

use crate::fmo::scc_hop::hop_data::{get_frag_shell_range, XtbHopData, XtbHopFragInfo};
use crate::fmo::scc_hop::monomer::XtbMonomerHopScc;
use crate::fmo::scc_hop::pair::XtbPairHopScc;
use crate::fmo::scc_hop::trimer::XtbTrimerHopScc;
use crate::hop::SP3_COEFF_P;
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::Basis;
use crate::parameters::*;
use nalgebra::Vector3;
use ndarray::prelude::*;

// ============================================================================
// Local-to-global atom index mappings
// ============================================================================

/// Build local-to-global atom index mapping for a monomer (real + ghost → BDA).
///
/// Real atoms map to their global indices from `frag_atom_range`.
/// Ghost atoms map to the BDA's global index (since the ghost is physically at the BDA position).
pub fn build_monomer_local_to_global(
    frag_atom_range: std::ops::Range<usize>,
    hop_data: &XtbHopData,
    frag_idx: usize,
) -> Vec<usize> {
    let mut mapping: Vec<usize> = frag_atom_range.collect();
    // Ghost atoms: bonds where baa_fragment == frag_idx, ghost is at BDA position
    for bond in &hop_data.detached_bonds {
        if bond.baa_fragment == frag_idx {
            mapping.push(bond.bda_global);
        }
    }
    mapping
}

/// Build local-to-global mapping for a pair including ghost atoms.
///
/// Layout: [real_I, real_J, partial_ghosts]
/// Ghost atoms map to their BDA's global index.
pub fn build_pair_local_to_global(
    range_i: std::ops::Range<usize>,
    range_j: std::ops::Range<usize>,
    ghost_bda_globals: &[usize],
) -> Vec<usize> {
    let mut mapping: Vec<usize> = range_i.chain(range_j).collect();
    for &bda in ghost_bda_globals {
        mapping.push(bda);
    }
    mapping
}

/// Build local-to-global mapping for a trimer including ghost atoms.
pub fn build_trimer_local_to_global(
    range_i: std::ops::Range<usize>,
    range_j: std::ops::Range<usize>,
    range_k: std::ops::Range<usize>,
    ghost_bda_globals: &[usize],
) -> Vec<usize> {
    let mut mapping: Vec<usize> = range_i.chain(range_j).chain(range_k).collect();
    for &bda in ghost_bda_globals {
        mapping.push(bda);
    }
    mapping
}

// ============================================================================
// Scatter
// ============================================================================

/// Scatter a local gradient array to global coordinates using local_to_global mapping.
pub fn scatter_to_global(
    global: &mut Array1<f64>,
    local_grad: &Array1<f64>,
    local_to_global: &[usize],
    n_local_atoms: usize,
) {
    for local_idx in 0..n_local_atoms {
        let global_idx = local_to_global[local_idx];
        for k in 0..3 {
            global[3 * global_idx + k] += local_grad[3 * local_idx + k];
        }
    }
}

// ============================================================================
// ZREF/QREF-scaled repulsive energy gradient
// ============================================================================

/// Compute repulsive energy gradient with ZREF/QREF scaling for HOP.
///
/// Each pair's repulsive derivative is scaled by `(ZREF[i]/QREF[i]) * (ZREF[j]/QREF[j])`.
pub fn grad_repulsive_energy_xtb_scaled(
    atoms: &[XtbAtom],
    zref: ArrayView1<f64>,
    qref: ArrayView1<f64>,
) -> Array1<f64> {
    let n_atoms = atoms.len();
    let mut grad = Array1::<f64>::zeros(3 * n_atoms);

    for i in 0..n_atoms {
        let ci = if qref[i] > 1e-14 {
            zref[i] / qref[i]
        } else {
            0.0
        };
        let z_eff_i = REP_Z_EFF_PARAMS[atoms[i].number as usize - 1];
        let alpha_i = REP_ALPHA_PARAMS[atoms[i].number as usize - 1];

        for j in 0..n_atoms {
            if i == j {
                continue;
            }
            let cj = if qref[j] > 1e-14 {
                zref[j] / qref[j]
            } else {
                0.0
            };
            let z_eff_j = REP_Z_EFF_PARAMS[atoms[j].number as usize - 1];
            let alpha_j = REP_ALPHA_PARAMS[atoms[j].number as usize - 1];

            let r_vec: Vector3<f64> = &atoms[i] - &atoms[j];
            let distance = (r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z).sqrt();
            let inv_dist = 1.0 / distance;
            let e_ij = [r_vec.x * inv_dist, r_vec.y * inv_dist, r_vec.z * inv_dist];

            let sqrt_alpha = (alpha_i * alpha_j).sqrt();
            let exponential = (-sqrt_alpha * distance.powf(1.5)).exp();
            let part1 = exponential * z_eff_i * z_eff_j / distance.powi(2);
            let part2 = 1.5 * sqrt_alpha * z_eff_i * z_eff_j * exponential / distance.sqrt();

            let deriv_val = -(part1 + part2) * ci * cj;
            for k in 0..3 {
                grad[3 * i + k] += e_ij[k] * deriv_val;
            }
        }
    }
    grad
}

// ============================================================================
// Shell-per-atom counting helper
// ============================================================================

/// Count number of shells per atom for atoms in a given ext_range.
///
/// Returns vec of length `ext_range.len()` where `[i]` = number of shells for local atom `i`.
pub fn shells_per_atom_in_range(basis: &Basis, ext_range: &std::ops::Range<usize>) -> Vec<usize> {
    let n_atoms = ext_range.end - ext_range.start;
    let mut counts = vec![0usize; n_atoms];
    let shell_range = get_frag_shell_range(basis, ext_range);
    for s in shell_range.start..shell_range.end {
        let atom_local = basis.shells[s].atom_index - ext_range.start;
        counts[atom_local] += 1;
    }
    counts
}

// ============================================================================
// Full CTIJ per fragment (real + ghost shells)
// ============================================================================

/// Build the full CTIJ at shell level for fragments I and J of a pair.
///
/// Returns `(ctij_i, ctij_j)` where each is indexed by the fragment's local
/// shell position (0..n_ext_shells) — real shells from `delta_dq_shell_real`,
/// ghost shells computed from `pair.dq_shell - mono.dq_shell` (partial) or
/// `-mono.dq_shell` (healed).
pub fn build_pair_ctij_full(
    ps: &XtbPairHopScc,
    mono_states: &[XtbMonomerHopScc],
    hop_data: &XtbHopData,
) -> (Array1<f64>, Array1<f64>) {
    let fi_i = &hop_data.frag_info[ps.i];
    let fi_j = &hop_data.frag_info[ps.j];
    let sr_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
    let sr_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);
    let n_shells_i = sr_i.end - sr_i.start;
    let n_shells_j = sr_j.end - sr_j.start;
    let n_rs_i = ps.n_real_shells_i;
    let n_rs_j = ps.n_real_shells_j;

    let mut ctij_i = Array1::zeros(n_shells_i);
    let mut ctij_j = Array1::zeros(n_shells_j);

    // Real shells from delta_dq_shell_real
    ctij_i
        .slice_mut(s![..n_rs_i])
        .assign(&ps.delta_dq_shell_real.slice(s![..n_rs_i]));
    ctij_j
        .slice_mut(s![..n_rs_j])
        .assign(&ps.delta_dq_shell_real.slice(s![n_rs_i..n_rs_i + n_rs_j]));

    // Ghost shells
    let spa_i = shells_per_atom_in_range(&hop_data.ext_basis, &fi_i.ext_range);
    let spa_j = shells_per_atom_in_range(&hop_data.ext_basis, &fi_j.ext_range);
    let n_real_i = fi_i.n_real_atoms;
    let n_real_j = fi_j.n_real_atoms;
    let n_rs_pair = n_rs_i + n_rs_j;

    let mut mono_ghost_shell_i = n_rs_i;
    let mut mono_ghost_shell_j = n_rs_j;
    let mut pair_ghost_shell = n_rs_pair;
    let mut ghost_idx_i = 0usize;
    let mut ghost_idx_j = 0usize;

    for bond in &hop_data.detached_bonds {
        let bda_in_pair = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;

        if bond.baa_fragment == ps.i {
            let ngs = spa_i[n_real_i + ghost_idx_i];
            for gs in 0..ngs {
                let mono_dq = mono_states[ps.i].dq_shell[mono_ghost_shell_i + gs];
                ctij_i[mono_ghost_shell_i + gs] = if bda_in_pair {
                    // Healed: ghost absent from pair → CTIJ = -(dq + q_ref) = -population
                    let q_ref = mono_states[ps.i].q_ref_shell[mono_ghost_shell_i + gs];
                    -(mono_dq + q_ref)
                } else {
                    ps.dq_shell[pair_ghost_shell + gs] - mono_dq
                };
            }
            // Compensate BDA real shells with +q_ref (POPMAT convention)
            if bda_in_pair {
                // baa=I, bda_in_pair, bda≠baa → BDA is in ps.j
                let bda_local = bond.bda_global - hop_data.monomer_indices[ps.j][0];
                let mut k = 0;
                for (si, sh) in mono_states[ps.j].basis.shells.iter().enumerate() {
                    if sh.atom_index == bda_local && si < n_rs_j && k < ngs {
                        ctij_j[si] += mono_states[ps.i].q_ref_shell[mono_ghost_shell_i + k];
                        k += 1;
                    }
                }
            }
            if !bda_in_pair {
                pair_ghost_shell += ngs;
            }
            mono_ghost_shell_i += ngs;
            ghost_idx_i += 1;
        } else if bond.baa_fragment == ps.j {
            let ngs = spa_j[n_real_j + ghost_idx_j];
            for gs in 0..ngs {
                let mono_dq = mono_states[ps.j].dq_shell[mono_ghost_shell_j + gs];
                ctij_j[mono_ghost_shell_j + gs] = if bda_in_pair {
                    // Healed: ghost absent from pair → CTIJ = -(dq + q_ref) = -population
                    let q_ref = mono_states[ps.j].q_ref_shell[mono_ghost_shell_j + gs];
                    -(mono_dq + q_ref)
                } else {
                    ps.dq_shell[pair_ghost_shell + gs] - mono_dq
                };
            }
            // Compensate BDA real shells with +q_ref (POPMAT convention)
            if bda_in_pair {
                // baa=J, bda_in_pair, bda≠baa → BDA is in ps.i
                let bda_local = bond.bda_global - hop_data.monomer_indices[ps.i][0];
                let mut k = 0;
                for (si, sh) in mono_states[ps.i].basis.shells.iter().enumerate() {
                    if sh.atom_index == bda_local && si < n_rs_i && k < ngs {
                        ctij_i[si] += mono_states[ps.j].q_ref_shell[mono_ghost_shell_j + k];
                        k += 1;
                    }
                }
            }
            if !bda_in_pair {
                pair_ghost_shell += ngs;
            }
            mono_ghost_shell_j += ngs;
            ghost_idx_j += 1;
        }
    }

    (ctij_i, ctij_j)
}

// ============================================================================
// CTMUL computation (POPMAT-based)
// ============================================================================

/// Compute CTMUL at shell level from delta_dq_shell using POPMAT convention.
///
/// For real shells:
/// `CTMUL[shell_of_frag_I] += SCAL * delta_dq_shell_real[local_shell]`
///
/// For ghost shells (matching DFTB HOP convention):
/// - Healed bond (both BDA and BAA in pair): `CTMUL[ghost_shell] += SCAL * (-mono.dq_shell[ghost_shell])`
/// - Partial bond (BAA in pair, BDA outside): `CTMUL[ghost_shell] += SCAL * (pair.dq_shell[ghost_shell] - mono.dq_shell[ghost_shell])`
///
/// Returns array of size `n_ext_shells` (total extended shells in hop_data).
pub fn compute_ctmul_xtb_hop(
    hop_data: &XtbHopData,
    mono_states: &[XtbMonomerHopScc],
    pair_states: &[XtbPairHopScc],
    trimer_states: &[XtbTrimerHopScc],
    pair_scal: &[f64],
) -> Array1<f64> {
    let n_ext_shells = hop_data.gamma_shell_ext.nrows();
    let mut ctmul = Array1::<f64>::zeros(n_ext_shells);

    // Pair contributions
    for (pair_idx, ps) in pair_states.iter().enumerate() {
        let scal = pair_scal[pair_idx];
        if scal.abs() < 1e-14 {
            continue;
        }
        let fi_i = &hop_data.frag_info[ps.i];
        let fi_j = &hop_data.frag_info[ps.j];
        let shell_range_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
        let shell_range_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);

        // delta_dq_shell_real is [n_real_shells_i + n_real_shells_j]
        let n_rs_i = ps.n_real_shells_i;
        let n_rs_j = ps.n_real_shells_j;
        let ddq = &ps.delta_dq_shell_real;

        // Real shells
        for (local_s, global_s) in shell_range_i.clone().enumerate().take(fi_i.n_real_shells) {
            if local_s < n_rs_i {
                ctmul[global_s] += scal * ddq[local_s];
            }
        }
        for (local_s, global_s) in shell_range_j.clone().enumerate().take(fi_j.n_real_shells) {
            if local_s < n_rs_j {
                ctmul[global_s] += scal * ddq[n_rs_i + local_s];
            }
        }

        // Ghost shell contributions
        let spa_i = shells_per_atom_in_range(&hop_data.ext_basis, &fi_i.ext_range);
        let spa_j = shells_per_atom_in_range(&hop_data.ext_basis, &fi_j.ext_range);
        let mut mono_ghost_shell_offset_i = fi_i.n_real_shells;
        let mut mono_ghost_shell_offset_j = fi_j.n_real_shells;
        let mut pair_ghost_shell_offset = ps.n_real_shells_i + ps.n_real_shells_j;
        let mut ghost_idx_i = 0usize;
        let mut ghost_idx_j = 0usize;

        for bond in &hop_data.detached_bonds {
            let bda_in_pair = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;

            if bond.baa_fragment == ps.i {
                let n_ghost_shells = spa_i[fi_i.n_real_atoms + ghost_idx_i];

                if bda_in_pair {
                    // Healed: ghost CTMUL = -(dq + q_ref) = -population
                    for s in 0..n_ghost_shells {
                        let global_shell = shell_range_i.start + mono_ghost_shell_offset_i + s;
                        let mono_dq = mono_states[ps.i].dq_shell[mono_ghost_shell_offset_i + s];
                        let q_ref = mono_states[ps.i].q_ref_shell[mono_ghost_shell_offset_i + s];
                        ctmul[global_shell] += scal * (-(mono_dq + q_ref));
                    }
                    // Compensate BDA real shells with +q_ref (baa=I → BDA in J)
                    let bda_local = bond.bda_global - hop_data.monomer_indices[ps.j][0];
                    let mut k = 0;
                    for (si, sh) in mono_states[ps.j].basis.shells.iter().enumerate() {
                        if sh.atom_index == bda_local && si < n_rs_j && k < n_ghost_shells {
                            let gs = shell_range_j.start + si;
                            ctmul[gs] += scal
                                * mono_states[ps.i].q_ref_shell
                                    [mono_ghost_shell_offset_i + k];
                            k += 1;
                        }
                    }
                } else {
                    // Partial: CTMUL = pair.dq_shell[ghost] - mono.dq_shell[ghost]
                    for s in 0..n_ghost_shells {
                        let global_shell = shell_range_i.start + mono_ghost_shell_offset_i + s;
                        let pair_val = ps.dq_shell[pair_ghost_shell_offset + s];
                        let mono_val = mono_states[ps.i].dq_shell[mono_ghost_shell_offset_i + s];
                        ctmul[global_shell] += scal * (pair_val - mono_val);
                    }
                    pair_ghost_shell_offset += n_ghost_shells;
                }
                mono_ghost_shell_offset_i += n_ghost_shells;
                ghost_idx_i += 1;
            } else if bond.baa_fragment == ps.j {
                let n_ghost_shells = spa_j[fi_j.n_real_atoms + ghost_idx_j];

                if bda_in_pair {
                    // Healed: ghost CTMUL = -(dq + q_ref) = -population
                    for s in 0..n_ghost_shells {
                        let global_shell = shell_range_j.start + mono_ghost_shell_offset_j + s;
                        let mono_dq = mono_states[ps.j].dq_shell[mono_ghost_shell_offset_j + s];
                        let q_ref = mono_states[ps.j].q_ref_shell[mono_ghost_shell_offset_j + s];
                        ctmul[global_shell] += scal * (-(mono_dq + q_ref));
                    }
                    // Compensate BDA real shells with +q_ref (baa=J → BDA in I)
                    let bda_local = bond.bda_global - hop_data.monomer_indices[ps.i][0];
                    let mut k = 0;
                    for (si, sh) in mono_states[ps.i].basis.shells.iter().enumerate() {
                        if sh.atom_index == bda_local && si < n_rs_i && k < n_ghost_shells {
                            let gs = shell_range_i.start + si;
                            ctmul[gs] += scal
                                * mono_states[ps.j].q_ref_shell
                                    [mono_ghost_shell_offset_j + k];
                            k += 1;
                        }
                    }
                } else {
                    // Partial: CTMUL = pair.dq_shell[ghost] - mono.dq_shell[ghost]
                    for s in 0..n_ghost_shells {
                        let global_shell = shell_range_j.start + mono_ghost_shell_offset_j + s;
                        let pair_val = ps.dq_shell[pair_ghost_shell_offset + s];
                        let mono_val = mono_states[ps.j].dq_shell[mono_ghost_shell_offset_j + s];
                        ctmul[global_shell] += scal * (pair_val - mono_val);
                    }
                    pair_ghost_shell_offset += n_ghost_shells;
                }
                mono_ghost_shell_offset_j += n_ghost_shells;
                ghost_idx_j += 1;
            }
        }
    }

    // Trimer contributions (SCAL = 1 for trimers)
    for ts in trimer_states.iter() {
        let fi_i = &hop_data.frag_info[ts.i];
        let fi_j = &hop_data.frag_info[ts.j];
        let fi_k = &hop_data.frag_info[ts.k];
        let shell_range_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
        let shell_range_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);
        let shell_range_k = get_frag_shell_range(&hop_data.ext_basis, &fi_k.ext_range);

        let n_rs_i = ts.n_real_shells_i;
        let n_rs_j = ts.n_real_shells_j;
        let n_rs_k = ts.n_real_shells_k;
        let ddq = &ts.delta_dq_shell_real;

        // Real shells
        for (local_s, global_s) in shell_range_i.clone().enumerate().take(fi_i.n_real_shells) {
            if local_s < n_rs_i {
                ctmul[global_s] += ddq[local_s];
            }
        }
        for (local_s, global_s) in shell_range_j.clone().enumerate().take(fi_j.n_real_shells) {
            if local_s < n_rs_j {
                ctmul[global_s] += ddq[n_rs_i + local_s];
            }
        }
        for (local_s, global_s) in shell_range_k.clone().enumerate().take(fi_k.n_real_shells) {
            if local_s < n_rs_k {
                ctmul[global_s] += ddq[n_rs_i + n_rs_j + local_s];
            }
        }

        // Ghost shell contributions for trimers
        let spa_i = shells_per_atom_in_range(&hop_data.ext_basis, &fi_i.ext_range);
        let spa_j = shells_per_atom_in_range(&hop_data.ext_basis, &fi_j.ext_range);
        let spa_k = shells_per_atom_in_range(&hop_data.ext_basis, &fi_k.ext_range);
        let mut mono_ghost_shell_offset_i = fi_i.n_real_shells;
        let mut mono_ghost_shell_offset_j = fi_j.n_real_shells;
        let mut mono_ghost_shell_offset_k = fi_k.n_real_shells;
        let mut tri_ghost_shell_offset =
            ts.n_real_shells_i + ts.n_real_shells_j + ts.n_real_shells_k;
        let mut ghost_idx_i = 0usize;
        let mut ghost_idx_j = 0usize;
        let mut ghost_idx_k = 0usize;

        for bond in &hop_data.detached_bonds {
            let bda_in_tri =
                bond.bda_fragment == ts.i || bond.bda_fragment == ts.j || bond.bda_fragment == ts.k;

            if bond.baa_fragment == ts.i {
                let n_ghost_shells = spa_i[fi_i.n_real_atoms + ghost_idx_i];
                if bda_in_tri {
                    for s in 0..n_ghost_shells {
                        let global_shell = shell_range_i.start + mono_ghost_shell_offset_i + s;
                        ctmul[global_shell] +=
                            -mono_states[ts.i].dq_shell[mono_ghost_shell_offset_i + s];
                    }
                } else {
                    for s in 0..n_ghost_shells {
                        let global_shell = shell_range_i.start + mono_ghost_shell_offset_i + s;
                        let tri_val = ts.dq_shell[tri_ghost_shell_offset + s];
                        let mono_val = mono_states[ts.i].dq_shell[mono_ghost_shell_offset_i + s];
                        ctmul[global_shell] += tri_val - mono_val;
                    }
                    tri_ghost_shell_offset += n_ghost_shells;
                }
                mono_ghost_shell_offset_i += n_ghost_shells;
                ghost_idx_i += 1;
            } else if bond.baa_fragment == ts.j {
                let n_ghost_shells = spa_j[fi_j.n_real_atoms + ghost_idx_j];
                if bda_in_tri {
                    for s in 0..n_ghost_shells {
                        let global_shell = shell_range_j.start + mono_ghost_shell_offset_j + s;
                        ctmul[global_shell] +=
                            -mono_states[ts.j].dq_shell[mono_ghost_shell_offset_j + s];
                    }
                } else {
                    for s in 0..n_ghost_shells {
                        let global_shell = shell_range_j.start + mono_ghost_shell_offset_j + s;
                        let tri_val = ts.dq_shell[tri_ghost_shell_offset + s];
                        let mono_val = mono_states[ts.j].dq_shell[mono_ghost_shell_offset_j + s];
                        ctmul[global_shell] += tri_val - mono_val;
                    }
                    tri_ghost_shell_offset += n_ghost_shells;
                }
                mono_ghost_shell_offset_j += n_ghost_shells;
                ghost_idx_j += 1;
            } else if bond.baa_fragment == ts.k {
                let n_ghost_shells = spa_k[fi_k.n_real_atoms + ghost_idx_k];
                if bda_in_tri {
                    for s in 0..n_ghost_shells {
                        let global_shell = shell_range_k.start + mono_ghost_shell_offset_k + s;
                        ctmul[global_shell] +=
                            -mono_states[ts.k].dq_shell[mono_ghost_shell_offset_k + s];
                    }
                } else {
                    for s in 0..n_ghost_shells {
                        let global_shell = shell_range_k.start + mono_ghost_shell_offset_k + s;
                        let tri_val = ts.dq_shell[tri_ghost_shell_offset + s];
                        let mono_val = mono_states[ts.k].dq_shell[mono_ghost_shell_offset_k + s];
                        ctmul[global_shell] += tri_val - mono_val;
                    }
                    tri_ghost_shell_offset += n_ghost_shells;
                }
                mono_ghost_shell_offset_k += n_ghost_shells;
                ghost_idx_k += 1;
            }
        }
    }

    ctmul
}

// ============================================================================
// SHIFTCT + ESPGRAD computation
// ============================================================================

/// Compute SHIFTCT + ESPGRAD for a single monomer in extended coordinates.
///
/// SHIFTCT = gamma_shell_ext[frag_shells, :] . ctmul_ext
/// ESPGRAD subtracts pair/trimer self-interaction terms (SCAL-scaled),
/// including ghost shell contributions.
///
/// Returns shell-level values for ALL shells of this monomer (real + ghost).
pub fn compute_shiftct_espgrad_xtb_hop(
    frag_idx: usize,
    hop_data: &XtbHopData,
    mono_states: &[XtbMonomerHopScc],
    pair_states: &[XtbPairHopScc],
    trimer_states: &[XtbTrimerHopScc],
    ctmul_ext: ArrayView1<f64>,
    pair_scal: &[f64],
) -> Array1<f64> {
    let fi = &hop_data.frag_info[frag_idx];
    let shell_range = get_frag_shell_range(&hop_data.ext_basis, &fi.ext_range);

    // SHIFTCT: gamma_ext[frag_shells, :] . ctmul
    let gamma_frag = hop_data
        .gamma_shell_ext
        .slice(s![shell_range.start..shell_range.end, ..]);
    let mut shiftct = gamma_frag.dot(&ctmul_ext);

    // ESPGRAD: subtract self-interaction for each pair containing this monomer
    for (pair_idx, ps) in pair_states.iter().enumerate() {
        let scal = pair_scal[pair_idx];
        if scal.abs() < 1e-14 {
            continue;
        }
        let (is_i, is_j) = (ps.i == frag_idx, ps.j == frag_idx);
        if !is_i && !is_j {
            continue;
        }

        let fi_pi = &hop_data.frag_info[ps.i];
        let fi_pj = &hop_data.frag_info[ps.j];
        let sr_pi = get_frag_shell_range(&hop_data.ext_basis, &fi_pi.ext_range);
        let sr_pj = get_frag_shell_range(&hop_data.ext_basis, &fi_pj.ext_range);

        let ddq = &ps.delta_dq_shell_real;
        let n_rs_i = ps.n_real_shells_i;

        // Real shells: gamma_ext[frag_shells, pair_I_real_shells] . delta_dq_I_real
        let gamma_frag_pi = hop_data.gamma_shell_ext.slice(s![
            shell_range.start..shell_range.end,
            sr_pi.start..sr_pi.start + fi_pi.n_real_shells
        ]);
        let ctij_i = ddq.slice(s![..n_rs_i]);
        shiftct -= &(scal * &gamma_frag_pi.dot(&ctij_i));

        // Real shells: gamma_ext[frag_shells, pair_J_real_shells] . delta_dq_J_real
        let gamma_frag_pj = hop_data.gamma_shell_ext.slice(s![
            shell_range.start..shell_range.end,
            sr_pj.start..sr_pj.start + fi_pj.n_real_shells
        ]);
        let ctij_j = ddq.slice(s![n_rs_i..]);
        shiftct -= &(scal * &gamma_frag_pj.dot(&ctij_j));

        // Ghost shell ESPGRAD subtraction (matching DFTB compute_espgrad_shiftct_hop)
        let spa_i = shells_per_atom_in_range(&hop_data.ext_basis, &fi_pi.ext_range);
        let spa_j = shells_per_atom_in_range(&hop_data.ext_basis, &fi_pj.ext_range);
        let mut mono_ghost_shell_offset_i = fi_pi.n_real_shells;
        let mut mono_ghost_shell_offset_j = fi_pj.n_real_shells;
        let mut pair_ghost_shell_offset = ps.n_real_shells_i + ps.n_real_shells_j;
        let mut ghost_idx_i = 0usize;
        let mut ghost_idx_j = 0usize;

        for bond in &hop_data.detached_bonds {
            let bda_in_pair = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;

            if bond.baa_fragment == ps.i {
                let n_ghost_shells = spa_i[fi_pi.n_real_atoms + ghost_idx_i];
                let ghost_global_start = sr_pi.start + mono_ghost_shell_offset_i;

                // Compute ghost CTIJ: healed = -(dq + q_ref), partial = pair - mono
                let ctij_ghost: Vec<f64> = if bda_in_pair {
                    (0..n_ghost_shells)
                        .map(|s| {
                            let mono_dq = mono_states[ps.i].dq_shell[mono_ghost_shell_offset_i + s];
                            let q_ref = mono_states[ps.i].q_ref_shell[mono_ghost_shell_offset_i + s];
                            -(mono_dq + q_ref)
                        })
                        .collect()
                } else {
                    let vals: Vec<f64> = (0..n_ghost_shells)
                        .map(|s| {
                            ps.dq_shell[pair_ghost_shell_offset + s]
                                - mono_states[ps.i].dq_shell[mono_ghost_shell_offset_i + s]
                        })
                        .collect();
                    pair_ghost_shell_offset += n_ghost_shells;
                    vals
                };

                // Subtract gamma[frag_shells, ghost_shells] . ctij_ghost
                for (s, &ctij_val) in ctij_ghost.iter().enumerate() {
                    if ctij_val.abs() < 1e-30 {
                        continue;
                    }
                    let gs = ghost_global_start + s;
                    for local_j in 0..shiftct.len() {
                        shiftct[local_j] -= scal
                            * hop_data.gamma_shell_ext
                                [[shell_range.start + local_j, gs]]
                            * ctij_val;
                    }
                }
                // Compensate BDA real shells with +q_ref (baa=I → BDA in J)
                if bda_in_pair {
                    let bda_local = bond.bda_global - hop_data.monomer_indices[ps.j][0];
                    let mut k = 0;
                    for (si, sh) in mono_states[ps.j].basis.shells.iter().enumerate() {
                        if sh.atom_index == bda_local && si < fi_pj.n_real_shells && k < n_ghost_shells {
                            let bda_gs = sr_pj.start + si;
                            let q_ref_val = mono_states[ps.i].q_ref_shell[mono_ghost_shell_offset_i + k];
                            for local_j in 0..shiftct.len() {
                                shiftct[local_j] -= scal
                                    * hop_data.gamma_shell_ext
                                        [[shell_range.start + local_j, bda_gs]]
                                    * q_ref_val;
                            }
                            k += 1;
                        }
                    }
                }
                mono_ghost_shell_offset_i += n_ghost_shells;
                ghost_idx_i += 1;
            } else if bond.baa_fragment == ps.j {
                let n_ghost_shells = spa_j[fi_pj.n_real_atoms + ghost_idx_j];
                let ghost_global_start = sr_pj.start + mono_ghost_shell_offset_j;

                let ctij_ghost: Vec<f64> = if bda_in_pair {
                    (0..n_ghost_shells)
                        .map(|s| {
                            let mono_dq = mono_states[ps.j].dq_shell[mono_ghost_shell_offset_j + s];
                            let q_ref = mono_states[ps.j].q_ref_shell[mono_ghost_shell_offset_j + s];
                            -(mono_dq + q_ref)
                        })
                        .collect()
                } else {
                    let vals: Vec<f64> = (0..n_ghost_shells)
                        .map(|s| {
                            ps.dq_shell[pair_ghost_shell_offset + s]
                                - mono_states[ps.j].dq_shell[mono_ghost_shell_offset_j + s]
                        })
                        .collect();
                    pair_ghost_shell_offset += n_ghost_shells;
                    vals
                };

                for (s, &ctij_val) in ctij_ghost.iter().enumerate() {
                    if ctij_val.abs() < 1e-30 {
                        continue;
                    }
                    let gs = ghost_global_start + s;
                    for local_j in 0..shiftct.len() {
                        shiftct[local_j] -= scal
                            * hop_data.gamma_shell_ext
                                [[shell_range.start + local_j, gs]]
                            * ctij_val;
                    }
                }
                // Compensate BDA real shells with +q_ref (baa=J → BDA in I)
                if bda_in_pair {
                    let bda_local = bond.bda_global - hop_data.monomer_indices[ps.i][0];
                    let mut k = 0;
                    for (si, sh) in mono_states[ps.i].basis.shells.iter().enumerate() {
                        if sh.atom_index == bda_local && si < fi_pi.n_real_shells && k < n_ghost_shells {
                            let bda_gs = sr_pi.start + si;
                            let q_ref_val = mono_states[ps.j].q_ref_shell[mono_ghost_shell_offset_j + k];
                            for local_j in 0..shiftct.len() {
                                shiftct[local_j] -= scal
                                    * hop_data.gamma_shell_ext
                                        [[shell_range.start + local_j, bda_gs]]
                                    * q_ref_val;
                            }
                            k += 1;
                        }
                    }
                }
                mono_ghost_shell_offset_j += n_ghost_shells;
                ghost_idx_j += 1;
            }
        }
    }

    // ESPGRAD: trimers (SCAL = 1)
    for ts in trimer_states.iter() {
        let is_in = ts.i == frag_idx || ts.j == frag_idx || ts.k == frag_idx;
        if !is_in {
            continue;
        }

        let fi_ti = &hop_data.frag_info[ts.i];
        let fi_tj = &hop_data.frag_info[ts.j];
        let fi_tk = &hop_data.frag_info[ts.k];
        let sr_ti = get_frag_shell_range(&hop_data.ext_basis, &fi_ti.ext_range);
        let sr_tj = get_frag_shell_range(&hop_data.ext_basis, &fi_tj.ext_range);
        let sr_tk = get_frag_shell_range(&hop_data.ext_basis, &fi_tk.ext_range);

        let ddq = &ts.delta_dq_shell_real;
        let n_rs_i = ts.n_real_shells_i;
        let n_rs_j = ts.n_real_shells_j;

        // Real shells
        let g_fi = hop_data.gamma_shell_ext.slice(s![
            shell_range.start..shell_range.end,
            sr_ti.start..sr_ti.start + fi_ti.n_real_shells
        ]);
        shiftct -= &g_fi.dot(&ddq.slice(s![..n_rs_i]));

        let g_fj = hop_data.gamma_shell_ext.slice(s![
            shell_range.start..shell_range.end,
            sr_tj.start..sr_tj.start + fi_tj.n_real_shells
        ]);
        shiftct -= &g_fj.dot(&ddq.slice(s![n_rs_i..n_rs_i + n_rs_j]));

        let g_fk = hop_data.gamma_shell_ext.slice(s![
            shell_range.start..shell_range.end,
            sr_tk.start..sr_tk.start + fi_tk.n_real_shells
        ]);
        shiftct -= &g_fk.dot(&ddq.slice(s![n_rs_i + n_rs_j..]));

        // Ghost shell ESPGRAD for trimers
        let tri_frags = [ts.i, ts.j, ts.k];
        let tri_fi = [fi_ti, fi_tj, fi_tk];
        let tri_sr = [&sr_ti, &sr_tj, &sr_tk];
        let tri_spa: Vec<Vec<usize>> = tri_fi
            .iter()
            .map(|f| shells_per_atom_in_range(&hop_data.ext_basis, &f.ext_range))
            .collect();
        let mut mono_ghost_shell_offsets = [fi_ti.n_real_shells, fi_tj.n_real_shells, fi_tk.n_real_shells];
        let mut tri_ghost_shell_offset =
            ts.n_real_shells_i + ts.n_real_shells_j + ts.n_real_shells_k;
        let mut ghost_idxs = [0usize; 3];

        for bond in &hop_data.detached_bonds {
            let bda_in_tri = bond.bda_fragment == ts.i
                || bond.bda_fragment == ts.j
                || bond.bda_fragment == ts.k;

            for frag_pos in 0..3 {
                if bond.baa_fragment == tri_frags[frag_pos] {
                    let n_ghost_shells = tri_spa[frag_pos]
                        [tri_fi[frag_pos].n_real_atoms + ghost_idxs[frag_pos]];
                    let ghost_global_start =
                        tri_sr[frag_pos].start + mono_ghost_shell_offsets[frag_pos];

                    let ctij_ghost: Vec<f64> = if bda_in_tri {
                        (0..n_ghost_shells)
                            .map(|s| {
                                -mono_states[tri_frags[frag_pos]].dq_shell
                                    [mono_ghost_shell_offsets[frag_pos] + s]
                            })
                            .collect()
                    } else {
                        let vals: Vec<f64> = (0..n_ghost_shells)
                            .map(|s| {
                                ts.dq_shell[tri_ghost_shell_offset + s]
                                    - mono_states[tri_frags[frag_pos]].dq_shell
                                        [mono_ghost_shell_offsets[frag_pos] + s]
                            })
                            .collect();
                        tri_ghost_shell_offset += n_ghost_shells;
                        vals
                    };

                    for (s, &ctij_val) in ctij_ghost.iter().enumerate() {
                        if ctij_val.abs() < 1e-30 {
                            continue;
                        }
                        let gs = ghost_global_start + s;
                        for local_j in 0..shiftct.len() {
                            shiftct[local_j] -= hop_data.gamma_shell_ext
                                [[shell_range.start + local_j, gs]]
                                * ctij_val;
                        }
                    }
                    mono_ghost_shell_offsets[frag_pos] += n_ghost_shells;
                    ghost_idxs[frag_pos] += 1;
                    break; // Each bond belongs to exactly one baa_fragment
                }
            }
        }
    }

    shiftct
}

// ============================================================================
// ESP_Q computation for a monomer
// ============================================================================

/// Compute ESP_Q at shell level for a monomer from hop_data.
///
/// esp_q_shell = gamma_ext[frag_shells, :] . dq_shell_ext - gamma_ext[frag, frag] . dq_frag
pub fn compute_esp_q_shell_hop(
    frag_idx: usize,
    hop_data: &XtbHopData,
) -> Array1<f64> {
    let fi = &hop_data.frag_info[frag_idx];
    let shell_range = get_frag_shell_range(&hop_data.ext_basis, &fi.ext_range);

    let esp_full: Array1<f64> = hop_data
        .gamma_shell_ext
        .slice(s![shell_range.start..shell_range.end, ..])
        .dot(&hop_data.dq_shell_ext);

    let esp_self: Array1<f64> = hop_data
        .gamma_shell_ext
        .slice(s![
            shell_range.start..shell_range.end,
            shell_range.start..shell_range.end
        ])
        .dot(
            &hop_data
                .dq_shell_ext
                .slice(s![shell_range.start..shell_range.end]),
        );

    &esp_full - &esp_self
}

// ============================================================================
// Occupation-based occ/virt derivation
// ============================================================================

/// Derive occupied and virtual orbital indices from occupation numbers.
///
/// occ: indices where f[i] > 0.5
/// virt: indices where f[i] <= 0.5
pub fn compute_occ_virt_from_f(f: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let mut occ = Vec::new();
    let mut virt = Vec::new();
    for (i, &fi) in f.iter().enumerate() {
        if fi > 0.5 {
            occ.push(i);
        } else {
            virt.push(i);
        }
    }
    (occ, virt)
}

/// Get the ghost BDA globals for a pair's partial bonds.
///
/// Returns the BDA global indices for bonds that cross the pair boundary
/// (BAA is in the pair but BDA is outside → ghost placed at BDA position).
pub fn get_pair_ghost_bda_globals(
    hop_data: &XtbHopData,
    pair_i: usize,
    pair_j: usize,
) -> Vec<usize> {
    let mut ghost_bda: Vec<usize> = Vec::new();
    for bond in &hop_data.detached_bonds {
        let bda_in = bond.bda_fragment == pair_i || bond.bda_fragment == pair_j;
        let baa_in = bond.baa_fragment == pair_i || bond.baa_fragment == pair_j;
        // Partial bond: BAA is in the pair, BDA is outside → ghost at BDA
        if baa_in && !bda_in {
            if !ghost_bda.contains(&bond.bda_global) {
                ghost_bda.push(bond.bda_global);
            }
        }
    }
    ghost_bda
}

/// Get the ghost BDA globals for a trimer's partial bonds.
pub fn get_trimer_ghost_bda_globals(
    hop_data: &XtbHopData,
    tri_i: usize,
    tri_j: usize,
    tri_k: usize,
) -> Vec<usize> {
    let mut ghost_bda: Vec<usize> = Vec::new();
    for bond in &hop_data.detached_bonds {
        let bda_in = bond.bda_fragment == tri_i
            || bond.bda_fragment == tri_j
            || bond.bda_fragment == tri_k;
        let baa_in = bond.baa_fragment == tri_i
            || bond.baa_fragment == tri_j
            || bond.baa_fragment == tri_k;
        if baa_in && !bda_in {
            if !ghost_bda.contains(&bond.bda_global) {
                ghost_bda.push(bond.bda_global);
            }
        }
    }
    ghost_bda
}

/// Get detached bonds that are partial to a pair (BAA in pair, BDA outside).
pub fn get_partial_bonds_for_pair<'a>(
    hop_data: &'a XtbHopData,
    pair_i: usize,
    pair_j: usize,
) -> Vec<&'a crate::hop::DetachedBond> {
    hop_data
        .detached_bonds
        .iter()
        .filter(|bond| {
            let bda_in = bond.bda_fragment == pair_i || bond.bda_fragment == pair_j;
            let baa_in = bond.baa_fragment == pair_i || bond.baa_fragment == pair_j;
            // Partial: one end in, one end out
            (baa_in && !bda_in) || (bda_in && !baa_in)
        })
        .collect()
}

/// Get detached bonds that are fully inside a pair (both BDA and BAA in pair).
pub fn get_healed_bonds_for_pair<'a>(
    hop_data: &'a XtbHopData,
    pair_i: usize,
    pair_j: usize,
) -> Vec<&'a crate::hop::DetachedBond> {
    hop_data
        .detached_bonds
        .iter()
        .filter(|bond| {
            let bda_in = bond.bda_fragment == pair_i || bond.bda_fragment == pair_j;
            let baa_in = bond.baa_fragment == pair_i || bond.baa_fragment == pair_j;
            bda_in && baa_in
        })
        .collect()
}

// ============================================================================
// P_ref derivative gradient (d(P_ref)/dR contribution)
// ============================================================================

/// p-orbital ordering for xTB FMO-HOP: (s, py, pz, px) → p_order maps orbital
/// index j to Cartesian component of b_hat.
const PREF_P_ORDER: [usize; 3] = [1, 2, 0];

/// Compute the P_ref derivative gradient contribution for a single BDA or ghost atom.
///
/// When the bond direction rotates, the hybrid orbital |h> changes, which changes
/// P_ref (via |h><h| adjustment). In xTB with shell-resolved gamma, this changes
/// the shell-level Mulliken charges dq_shell, creating a gradient through the
/// Coulomb energy.
///
/// Formula: g[ic] = sign * 2 * Σ_μ v_eff[μ] * h[μ'] * dh_ic[μ']
///
/// where v_eff = (γ·dq_shell)[shell(μ)] - Γ_{atom(μ)}·dq²_{atom(μ)}
///
/// Returns (bda_grad_3, baa_grad_3) acting on BDA and BAA atom positions.
///
/// - sign: +1 for BDA (P_ref -= |h><h|), -1 for ghost (P_ref = |h><h|)
fn pref_derivative_gradient_single(
    v_eff_ao: &[f64],       // effective potential at AO level for the atom's AOs
    hybrid: &[f64],          // rotated sp3 coefficients [s, py, pz, px]
    bond_vec: &Vector3<f64>, // BDA → BAA vector
    sign: f64,               // +1 for BDA, -1 for ghost
) -> ([f64; 3], [f64; 3]) {
    let bond_len = bond_vec.norm();
    if bond_len < 1e-14 {
        return ([0.0; 3], [0.0; 3]);
    }
    let b_hat = bond_vec / bond_len;
    let sz = hybrid.len().min(v_eff_ao.len());

    let mut bda_grad = [0.0f64; 3];
    let mut baa_grad = [0.0f64; 3];

    for ic in 0..3usize {
        // d(b_hat[k])/d(bond_ic) = (delta(k,ic) - b_hat[k]*b_hat[ic]) / bond_len
        let mut d_bhat = [0.0f64; 3];
        for j in 0..3 {
            let delta = if j == ic { 1.0 } else { 0.0 };
            d_bhat[j] = (delta - b_hat[j] * b_hat[ic]) / bond_len;
        }

        // dh[0] = 0 (s-orbital doesn't depend on direction)
        // dh[j+1] = SP3_COEFF_P * d_bhat[PREF_P_ORDER[j]]
        let mut dh = [0.0f64; 4];
        for j in 0..3 {
            dh[j + 1] = SP3_COEFF_P * d_bhat[PREF_P_ORDER[j]];
        }

        // Contract: sign * 2 * Σ_μ v_eff[μ] * h[μ] * dh[μ]
        let mut cont = 0.0;
        for mu in 0..sz {
            cont += v_eff_ao[mu] * hybrid[mu] * dh[mu];
        }
        cont *= sign * 2.0;

        // bond_vec = BAA - BDA, so d(bond)/d(BDA) = -1, d(bond)/d(BAA) = +1
        bda_grad[ic] -= cont;
        baa_grad[ic] += cont;
    }

    (bda_grad, baa_grad)
}

/// Compute the v_eff vector at AO level for an atom in a fragment.
///
/// v_eff[μ] = shift_shell[shell(μ)] - Γ_{atom}·dq²_{atom}
///
/// where shift_shell = γ·dq_shell (own Coulomb only, no ESP).
fn compute_pref_veff_ao(
    basis: &Basis,
    gamma_shell: ArrayView2<f64>,
    dq_shell: ArrayView1<f64>,
    dq_atom: ArrayView1<f64>,
    ext_atoms: &[XtbAtom],
    atom_local: usize,
) -> Vec<f64> {
    let shift_shell = gamma_shell.dot(&dq_shell);
    let (ao_start, nao) = crate::hop::get_bda_ao_range(basis, atom_local);

    let mut v_eff = vec![0.0f64; nao];
    for (mu_local, mu_global) in (ao_start..ao_start + nao).enumerate() {
        // Find which shell this AO belongs to
        let shell_idx = basis
            .shells
            .iter()
            .position(|sh| mu_global >= sh.sph_start && mu_global < sh.sph_end)
            .unwrap();
        let at = basis.shells[shell_idx].atom_index;
        let hubb_deriv = COUL_THIRD_ORDER_ATOM[ext_atoms[at].number as usize - 1];
        v_eff[mu_local] = shift_shell[shell_idx] - hubb_deriv * dq_atom[at] * dq_atom[at];
    }
    v_eff
}

/// Compute the full P_ref derivative gradient for the FMO-xTB HOP system.
///
/// Returns the global gradient contribution from d(P_ref)/dR.
///
/// This accounts for:
/// 1. Per-monomer P_ref gradient (own Coulomb + third)
/// 2. Per-pair P_ref gradient (FMO delta)
/// 3. Embedding P_ref contribution (ESP on monomer charges)
pub fn compute_pref_gradient_fmo_xtb_hop(
    hop_data: &XtbHopData,
    mono_states: &[XtbMonomerHopScc],
    pair_states: &[XtbPairHopScc],
    atoms: &[XtbAtom],
    frag_atom_ranges: &[std::ops::Range<usize>],
    n_atoms_total: usize,
) -> Array1<f64> {
    use crate::fmo::scc_hop::hop_data::compute_rotated_sp3_xtb;
    use crate::hop::get_bda_ao_range;

    let n_grad = 3 * n_atoms_total;
    let n_frags = mono_states.len();

    // ---- Part 1: Per-monomer P_ref gradient (own Coulomb + third, no ESP) ----
    // Store per-monomer local gradients for pair delta subtraction
    let mut mono_pref_locals: Vec<Array1<f64>> = Vec::with_capacity(n_frags);
    let mut pref_total = Array1::<f64>::zeros(n_grad);

    for (frag_idx, mono) in mono_states.iter().enumerate() {
        let n_atoms = mono.n_ext_atoms;
        let mut local_grad = Array1::<f64>::zeros(3 * n_atoms);

        // BDA bonds on this monomer
        for bond in &hop_data.detached_bonds {
            if bond.bda_fragment != frag_idx {
                continue;
            }
            let bda_local = bond.bda_global - frag_atom_ranges[frag_idx].start;
            let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
            let hybrid = compute_rotated_sp3_xtb(&bond_vec);

            let v_eff = compute_pref_veff_ao(
                &mono.basis,
                mono.gamma_shell.view(),
                mono.dq_shell.view(),
                mono.dq.view(),
                &mono.ext_atoms,
                bda_local,
            );

            let (bda_g, baa_g) =
                pref_derivative_gradient_single(&v_eff, hybrid.as_slice().unwrap(), &bond_vec, 1.0);

            for k in 0..3 {
                local_grad[3 * bda_local + k] += bda_g[k];
            }
            // BAA is external (not in this monomer) — add directly to global
            for k in 0..3 {
                pref_total[3 * bond.baa_global + k] += baa_g[k];
            }
        }

        // Ghost bonds on this monomer (bonds where baa_fragment == frag_idx)
        let mut ghost_idx = 0;
        for bond in &hop_data.detached_bonds {
            if bond.baa_fragment != frag_idx {
                continue;
            }
            let ghost_local = mono.n_real_atoms + ghost_idx;
            let baa_local = bond.baa_global - frag_atom_ranges[frag_idx].start;
            let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
            let hybrid = compute_rotated_sp3_xtb(&bond_vec);

            let v_eff = compute_pref_veff_ao(
                &mono.basis,
                mono.gamma_shell.view(),
                mono.dq_shell.view(),
                mono.dq.view(),
                &mono.ext_atoms,
                ghost_local,
            );

            let (bda_g, baa_g) =
                pref_derivative_gradient_single(&v_eff, hybrid.as_slice().unwrap(), &bond_vec, -1.0);

            // Ghost is at BDA position (local ghost_local)
            for k in 0..3 {
                local_grad[3 * ghost_local + k] += bda_g[k];
            }
            // BAA is a real atom in THIS monomer
            for k in 0..3 {
                local_grad[3 * baa_local + k] += baa_g[k];
            }
            ghost_idx += 1;
        }

        // Scatter local gradient to global (including ghost→BDA mapping)
        let ltg = build_monomer_local_to_global(
            frag_atom_ranges[frag_idx].clone(),
            hop_data,
            frag_idx,
        );
        for local_idx in 0..n_atoms {
            let global_idx = ltg[local_idx];
            for k in 0..3 {
                pref_total[3 * global_idx + k] += local_grad[3 * local_idx + k];
            }
        }

        mono_pref_locals.push(local_grad);
    }

    // ---- Part 2: Per-pair P_ref gradient delta ----
    for ps in pair_states.iter() {
        let n_real_i = frag_atom_ranges[ps.i].len();
        let n_real_j = frag_atom_ranges[ps.j].len();
        let n_real = n_real_i + n_real_j;
        let n_atoms = ps.n_ext_atoms;
        let mut pair_local_grad = Array1::<f64>::zeros(3 * n_atoms);

        // Classify bonds for this pair
        for bond in &hop_data.detached_bonds {
            let bda_in = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;
            let baa_in = bond.baa_fragment == ps.i || bond.baa_fragment == ps.j;

            if bda_in && baa_in {
                // Healed bond: pair has no P_ref adjustment → pair contribution = 0
                continue;
            }

            let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
            let hybrid = compute_rotated_sp3_xtb(&bond_vec);

            if bda_in && !baa_in {
                // Partial BDA: BDA in pair, BAA outside
                let bda_local = if bond.bda_fragment == ps.i {
                    bond.bda_global - frag_atom_ranges[ps.i].start
                } else {
                    n_real_i + (bond.bda_global - frag_atom_ranges[ps.j].start)
                };

                let v_eff = compute_pref_veff_ao(
                    &ps.basis,
                    ps.gamma_shell.view(),
                    ps.dq_shell.view(),
                    ps.dq.view(),
                    &ps.ext_atoms,
                    bda_local,
                );

                let (bda_g, baa_g) = pref_derivative_gradient_single(
                    &v_eff,
                    hybrid.as_slice().unwrap(),
                    &bond_vec,
                    1.0,
                );

                for k in 0..3 {
                    pair_local_grad[3 * bda_local + k] += bda_g[k];
                }
                // BAA external
                for k in 0..3 {
                    pref_total[3 * bond.baa_global + k] += baa_g[k];
                }
            } else if baa_in && !bda_in {
                // Partial BAA: ghost in pair
                // Find ghost index
                let mut gi = 0;
                for b2 in &hop_data.detached_bonds {
                    if std::ptr::eq(b2, bond) {
                        break;
                    }
                    let b2_bda_in = b2.bda_fragment == ps.i || b2.bda_fragment == ps.j;
                    let b2_baa_in = b2.baa_fragment == ps.i || b2.baa_fragment == ps.j;
                    if b2_baa_in && !b2_bda_in {
                        gi += 1;
                    }
                }
                let ghost_local = n_real + gi;
                let baa_local = if bond.baa_fragment == ps.i {
                    bond.baa_global - frag_atom_ranges[ps.i].start
                } else {
                    n_real_i + (bond.baa_global - frag_atom_ranges[ps.j].start)
                };

                let v_eff = compute_pref_veff_ao(
                    &ps.basis,
                    ps.gamma_shell.view(),
                    ps.dq_shell.view(),
                    ps.dq.view(),
                    &ps.ext_atoms,
                    ghost_local,
                );

                let (bda_g, baa_g) = pref_derivative_gradient_single(
                    &v_eff,
                    hybrid.as_slice().unwrap(),
                    &bond_vec,
                    -1.0,
                );

                for k in 0..3 {
                    pair_local_grad[3 * ghost_local + k] += bda_g[k];
                }
                for k in 0..3 {
                    pair_local_grad[3 * baa_local + k] += baa_g[k];
                }
            }
        }

        // FMO delta: pair - mono_I - mono_J
        let mon_i_grad = &mono_pref_locals[ps.i];
        let mon_j_grad = &mono_pref_locals[ps.j];
        let range_i = &frag_atom_ranges[ps.i];
        let range_j = &frag_atom_ranges[ps.j];

        // I's real atoms
        for (local_idx, global_idx) in range_i.clone().enumerate() {
            for k in 0..3 {
                pref_total[3 * global_idx + k] +=
                    pair_local_grad[3 * local_idx + k] - mon_i_grad[3 * local_idx + k];
            }
        }
        // J's real atoms
        for (local_idx, global_idx) in range_j.clone().enumerate() {
            for k in 0..3 {
                pref_total[3 * global_idx + k] += pair_local_grad[3 * (n_real_i + local_idx) + k]
                    - mon_j_grad[3 * local_idx + k];
            }
        }
        // Pair ghost atoms → BDA global
        let ghost_bda = get_pair_ghost_bda_globals(hop_data, ps.i, ps.j);
        for (gi, &bda_global) in ghost_bda.iter().enumerate() {
            let pair_ghost_local = n_real + gi;
            for k in 0..3 {
                pref_total[3 * bda_global + k] += pair_local_grad[3 * pair_ghost_local + k];
            }
        }
        // Subtract mono ghost contributions
        for &fi in &[ps.i, ps.j] {
            let mono_grad = &mono_pref_locals[fi];
            let n_real_fi = frag_atom_ranges[fi].len();
            let mut gc = 0;
            for db in &hop_data.detached_bonds {
                if db.baa_fragment == fi {
                    let local_idx = n_real_fi + gc;
                    for k in 0..3 {
                        pref_total[3 * db.bda_global + k] -= mono_grad[3 * local_idx + k];
                    }
                    gc += 1;
                }
            }
        }

        // Subtract BAA external contributions from monomers
        // (BAA external contributions from BDA bonds were added directly to pref_total
        // for both monomer and pair, so the delta is implicit via the pair addition
        // and monomer subtraction above. But the BAA external was added for the mono
        // unconditionally — we need to subtract mono's BAA external for pair delta.)
        // For bonds where BDA is in I or J:
        for bond in &hop_data.detached_bonds {
            let bda_in = bond.bda_fragment == ps.i || bond.bda_fragment == ps.j;
            let baa_in = bond.baa_fragment == ps.i || bond.baa_fragment == ps.j;
            if bda_in {
                // The monomer already added BAA external contribution above;
                // for pair delta, we subtract it here (since we added pair's BAA
                // external above for partial bonds, or didn't add for healed bonds).
                // Actually: mono's BAA external was already in pref_total from Part 1.
                // For healed bonds: pair adds nothing (0), but mono added BAA external
                //   → delta should subtract mono's BAA external (done via real atom subtraction above)
                // Wait, BAA external means the BAA atom is NOT in the monomer,
                // so it's NOT captured by the real atom subtraction above.
                // We need to explicitly subtract mono's BAA external for pair delta.
                if bda_in && baa_in {
                    // Healed: pair has no BAA external → subtract mono's
                    let mono_frag = bond.bda_fragment;
                    let bda_local = bond.bda_global - frag_atom_ranges[mono_frag].start;
                    let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
                    let hybrid = compute_rotated_sp3_xtb(&bond_vec);
                    let mono = &mono_states[mono_frag];
                    let v_eff = compute_pref_veff_ao(
                        &mono.basis,
                        mono.gamma_shell.view(),
                        mono.dq_shell.view(),
                        mono.dq.view(),
                        &mono.ext_atoms,
                        bda_local,
                    );
                    let (_, baa_g) = pref_derivative_gradient_single(
                        &v_eff,
                        hybrid.as_slice().unwrap(),
                        &bond_vec,
                        1.0,
                    );
                    // Subtract mono's BAA external (was added in Part 1)
                    for k in 0..3 {
                        pref_total[3 * bond.baa_global + k] -= baa_g[k];
                    }
                }
                // For partial BDA bonds: pair added its own BAA external above,
                // and mono added its BAA external in Part 1 → need to subtract mono's
                if bda_in && !baa_in {
                    let mono_frag = bond.bda_fragment;
                    let bda_local = bond.bda_global - frag_atom_ranges[mono_frag].start;
                    let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
                    let hybrid = compute_rotated_sp3_xtb(&bond_vec);
                    let mono = &mono_states[mono_frag];
                    let v_eff = compute_pref_veff_ao(
                        &mono.basis,
                        mono.gamma_shell.view(),
                        mono.dq_shell.view(),
                        mono.dq.view(),
                        &mono.ext_atoms,
                        bda_local,
                    );
                    let (_, baa_g) = pref_derivative_gradient_single(
                        &v_eff,
                        hybrid.as_slice().unwrap(),
                        &bond_vec,
                        1.0,
                    );
                    for k in 0..3 {
                        pref_total[3 * bond.baa_global + k] -= baa_g[k];
                    }
                }
            }
        }

        // Similarly for ghost BAA internal contributions from monomers:
        // Ghost BAA grad goes to a local BAA atom in the monomer.
        // This is captured by the real atom subtraction above (BAA is a real atom
        // in the ghost's monomer). So no extra handling needed.
    }

    // ---- Part 3: Embedding P_ref contribution (ESP on monomer charges) ----
    for (frag_idx, mono) in mono_states.iter().enumerate() {
        let esp = compute_esp_q_shell_hop(frag_idx, hop_data);

        // BDA bonds
        for bond in &hop_data.detached_bonds {
            if bond.bda_fragment != frag_idx {
                continue;
            }
            let bda_local = bond.bda_global - frag_atom_ranges[frag_idx].start;
            let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
            let hybrid = compute_rotated_sp3_xtb(&bond_vec);

            // ESP-only v_eff (no own Coulomb, no third-order)
            let (ao_start, nao) = get_bda_ao_range(&mono.basis, bda_local);
            let mut v_eff_esp = vec![0.0f64; nao];
            for (mu_local, mu_global) in (ao_start..ao_start + nao).enumerate() {
                let shell_idx = mono
                    .basis
                    .shells
                    .iter()
                    .position(|sh| mu_global >= sh.sph_start && mu_global < sh.sph_end)
                    .unwrap();
                v_eff_esp[mu_local] = esp[shell_idx];
            }

            let (bda_g, baa_g) = pref_derivative_gradient_single(
                &v_eff_esp,
                hybrid.as_slice().unwrap(),
                &bond_vec,
                1.0,
            );

            for k in 0..3 {
                pref_total[3 * bond.bda_global + k] += bda_g[k];
            }
            for k in 0..3 {
                pref_total[3 * bond.baa_global + k] += baa_g[k];
            }
        }

        // Ghost bonds
        let mut ghost_idx = 0;
        for bond in &hop_data.detached_bonds {
            if bond.baa_fragment != frag_idx {
                continue;
            }
            let ghost_local = mono.n_real_atoms + ghost_idx;
            let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
            let hybrid = compute_rotated_sp3_xtb(&bond_vec);

            let (ao_start, nao) = get_bda_ao_range(&mono.basis, ghost_local);
            let mut v_eff_esp = vec![0.0f64; nao];
            for (mu_local, mu_global) in (ao_start..ao_start + nao).enumerate() {
                let shell_idx = mono
                    .basis
                    .shells
                    .iter()
                    .position(|sh| mu_global >= sh.sph_start && mu_global < sh.sph_end)
                    .unwrap();
                v_eff_esp[mu_local] = esp[shell_idx];
            }

            let (bda_g, baa_g) = pref_derivative_gradient_single(
                &v_eff_esp,
                hybrid.as_slice().unwrap(),
                &bond_vec,
                -1.0,
            );

            // Ghost at BDA position → global BDA
            for k in 0..3 {
                pref_total[3 * bond.bda_global + k] += bda_g[k];
            }
            // BAA in this monomer → global BAA
            for k in 0..3 {
                pref_total[3 * bond.baa_global + k] += baa_g[k];
            }
            ghost_idx += 1;
        }
    }

    pref_total
}
