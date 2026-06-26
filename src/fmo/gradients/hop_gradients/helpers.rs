//! HOP-specific helper functions for FMO-DFTB gradient computation.
//!
//! Provides utilities for:
//! - Local-to-global atom mapping with ghost atoms
//! - Ghost-to-BAA gradient scatter
//! - ZREF/QREF-scaled repulsive gradient
//! - Extended CTMUL and SHIFTCT computation using POPMAT differences

use crate::fmo::scc_hop::hop_data::HopData;
use crate::fmo::scc_hop::monomer::MonomerHopScc;
use crate::fmo::scc_hop::pair::PairHopScc;
use crate::fmo::Pair;
use crate::initialization::parameters::RepulsivePotential;
use crate::initialization::Atom;
use ndarray::prelude::*;
use std::ops::Range;

// ============================================================================
// Re-export common helpers from fmo_gradient.rs (used unchanged)
// ============================================================================

/// Build orbital-to-atom mapping.
pub fn build_orb_to_atom_map(atoms: &[Atom], n_orbs: usize) -> Vec<usize> {
    let mut orb_to_atom = vec![0usize; n_orbs];
    let mut orb_idx = 0;
    for (atom_idx, atom) in atoms.iter().enumerate() {
        for _ in 0..atom.n_orbs {
            orb_to_atom[orb_idx] = atom_idx;
            orb_idx += 1;
        }
    }
    orb_to_atom
}

/// Build orbital offsets: orbital_offsets[i] = first orbital index of atom i.
pub fn build_orbital_offsets(atoms: &[Atom]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(atoms.len() + 1);
    offsets.push(0);
    for atom in atoms {
        offsets.push(offsets.last().unwrap() + atom.n_orbs);
    }
    offsets
}

/// Build shift matrix in AO basis from atom-based shift vector.
pub fn build_shift_ao_matrix(shift: ArrayView1<f64>, atoms: &[Atom], n_orbs: usize) -> Array2<f64> {
    let orb_to_atom = build_orb_to_atom_map(atoms, n_orbs);
    let mut shift_ao = Array2::<f64>::zeros([n_orbs, n_orbs]);
    for mu in 0..n_orbs {
        let s_mu = shift[orb_to_atom[mu]];
        for nu in 0..n_orbs {
            shift_ao[[mu, nu]] = 0.5 * (s_mu + shift[orb_to_atom[nu]]);
        }
    }
    shift_ao
}

/// Compute energy-weighted density matrix W = sum_k f_k * eps_k * C_k * C_k^T
pub fn compute_w_matrix(
    orbs: ArrayView2<f64>,
    orbe: ArrayView1<f64>,
    f: &[f64],
    n_orbs: usize,
    nocc: usize,
) -> Array2<f64> {
    let mut w = Array2::<f64>::zeros([n_orbs, n_orbs]);
    for k in 0..nocc {
        let weight = f[k] * orbe[k];
        for mu in 0..n_orbs {
            for nu in 0..=mu {
                let val = weight * orbs[[mu, k]] * orbs[[nu, k]];
                w[[mu, nu]] += val;
                if mu != nu {
                    w[[nu, mu]] += val;
                }
            }
        }
    }
    w
}

// ============================================================================
// HOP-specific helpers
// ============================================================================

/// Build local-to-global atom index mapping for a monomer with ghost atoms.
///
/// Real atoms: global indices from `frag_atom_range`.
/// Ghost atoms: `bond.baa_global` for each bond where `baa_fragment == frag_idx`.
pub fn build_monomer_local_to_global_dftb(
    frag_atom_range: &Range<usize>,
    detached_bonds: &[dialect_xtb::hop::DetachedBond],
    frag_idx: usize,
) -> Vec<usize> {
    let mut mapping: Vec<usize> = frag_atom_range.clone().collect();
    // Ghost atoms are at BDA's position → scatter to bda_global
    for bond in detached_bonds {
        if bond.baa_fragment == frag_idx {
            mapping.push(bond.bda_global);
        }
    }
    mapping
}

/// Build local-to-global mapping for an extended pair.
///
/// Layout: [real_I atoms, real_J atoms, partial ghost atoms]
/// Ghost atoms map to their BAA's global index.
pub fn build_pair_local_to_global_dftb(
    frag_atom_range_i: &Range<usize>,
    frag_atom_range_j: &Range<usize>,
    pair_ghost_baa_globals: &[usize],
) -> Vec<usize> {
    let mut mapping: Vec<usize> = frag_atom_range_i.clone().collect();
    mapping.extend(frag_atom_range_j.clone());
    mapping.extend(pair_ghost_baa_globals.iter());
    mapping
}

/// Scatter a local gradient (including ghost entries) to global coordinates.
///
/// Ghost atom gradient contributions are mapped to their BAA atom's global index.
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

/// Compute the gradient of ZREF/QREF-scaled repulsive energy.
///
/// `dE/dR_a = sum_{b!=a} scale_ab * v_rep'(r_ab) * (R_a - R_b) / r_ab`
/// where `scale_ab = (zref[a]/qref[a]) * (zref[b]/qref[b])`.
///
/// The scaling factors are position-independent (ZREF/QREF are fixed).
pub fn grad_repulsive_energy_scaled(
    atoms: &[Atom],
    n_atoms: usize,
    v_rep: &RepulsivePotential,
    zref: ArrayView1<f64>,
    qref: ArrayView1<f64>,
) -> Array1<f64> {
    let mut grad = Array1::<f64>::zeros(3 * n_atoms);
    for i in 0..n_atoms {
        let ci = if qref[i] > 1e-14 { zref[i] / qref[i] } else { 0.0 };
        let atomi = &atoms[i];
        for j in (i + 1)..n_atoms {
            let cj = if qref[j] > 1e-14 { zref[j] / qref[j] } else { 0.0 };
            let scale = ci * cj;
            if scale.abs() < 1e-30 {
                continue;
            }
            let atomj = &atoms[j];
            let r_vec = atomi - atomj;
            let dist = r_vec.norm();
            let v_deriv = v_rep.get(atomi.kind, atomj.kind).spline_deriv(dist);
            let factor = scale * v_deriv / dist;
            for k in 0..3 {
                let val = factor * r_vec[k];
                grad[3 * i + k] += val;
                grad[3 * j + k] -= val;
            }
        }
    }
    grad
}

/// Compute extended CTMUL using POPMAT differences (matching `embedding_energy_hop()`).
///
/// For each pair:
///   - Real atoms of I: `ctmul[ext_a] += POPMAT_pair[a] - POPMAT_mono[a]`
///   - Real atoms of J: same
///   - Ghost atoms: healed → `-POPMAT_mono`, partial → `POPMAT_pair - POPMAT_mono`
///
/// Returns CTMUL indexed by ext_atoms (n_ext_atoms).
pub fn compute_ctmul_hop(
    hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    pair_states: &[PairHopScc],
    pairs: &[Pair],
) -> Array1<f64> {
    let mut ctmul = Array1::<f64>::zeros(hop_data.n_ext_atoms);

    for (pair_idx, pair) in pairs.iter().enumerate() {
        let fi_i = &hop_data.frag_info[pair.i];
        let fi_j = &hop_data.frag_info[pair.j];
        let n_real_i = fi_i.n_real_atoms;
        let n_real_j = fi_j.n_real_atoms;
        let ps = &pair_states[pair_idx];

        // Fragment I real atoms
        for a in 0..n_real_i {
            let popmat_pair = ps.dq[a] + ps.zref[a];
            let mono_zref = hop_data.zref[fi_i.ext_range.start + a];
            let popmat_mono = mono_states[pair.i].dq[a] + mono_zref;
            ctmul[fi_i.ext_range.start + a] += popmat_pair - popmat_mono;
        }

        // Fragment J real atoms
        for a in 0..n_real_j {
            let pair_a = n_real_i + a;
            let popmat_pair = ps.dq[pair_a] + ps.zref[pair_a];
            let mono_zref = hop_data.zref[fi_j.ext_range.start + a];
            let popmat_mono = mono_states[pair.j].dq[a] + mono_zref;
            ctmul[fi_j.ext_range.start + a] += popmat_pair - popmat_mono;
        }

        // Ghost atoms
        let mut mono_ghost_i_idx = 0usize;
        let mut mono_ghost_j_idx = 0usize;
        let mut pair_ghost_idx = ps.n_real_atoms;

        for bond in &hop_data.detached_bonds {
            let bda_in_pair = bond.bda_fragment == pair.i || bond.bda_fragment == pair.j;

            if bond.baa_fragment == pair.i {
                let mono_ghost_local = n_real_i + mono_ghost_i_idx;
                let mono_zref_g = hop_data.zref[fi_i.ext_range.start + mono_ghost_local];
                let popmat_mono_g = mono_states[pair.i].dq[mono_ghost_local] + mono_zref_g;

                let ctij = if bda_in_pair {
                    -popmat_mono_g
                } else {
                    let popmat_pair_g = ps.dq[pair_ghost_idx] + ps.zref[pair_ghost_idx];
                    pair_ghost_idx += 1;
                    popmat_pair_g - popmat_mono_g
                };
                ctmul[fi_i.ext_range.start + mono_ghost_local] += ctij;
                mono_ghost_i_idx += 1;
            } else if bond.baa_fragment == pair.j {
                let mono_ghost_local = n_real_j + mono_ghost_j_idx;
                let mono_zref_g = hop_data.zref[fi_j.ext_range.start + mono_ghost_local];
                let popmat_mono_g = mono_states[pair.j].dq[mono_ghost_local] + mono_zref_g;

                let ctij = if bda_in_pair {
                    -popmat_mono_g
                } else {
                    let popmat_pair_g = ps.dq[pair_ghost_idx] + ps.zref[pair_ghost_idx];
                    pair_ghost_idx += 1;
                    popmat_pair_g - popmat_mono_g
                };
                ctmul[fi_j.ext_range.start + mono_ghost_local] += ctij;
                mono_ghost_j_idx += 1;
            }
        }
    }

    ctmul
}

/// Compute SHIFTCT for a monomer using extended gamma matrix.
///
/// `SHIFTCT[ext_a] = sum over ALL ext atoms c: gamma_ext[ext_I_a, c] * ctmul[c]`
pub fn compute_shiftct_hop(
    frag_idx: usize,
    hop_data: &HopData,
    ctmul_ext: ArrayView1<f64>,
) -> Array1<f64> {
    let fi = &hop_data.frag_info[frag_idx];
    let ext_range = &fi.ext_range;
    let gamma_row = hop_data.gamma_ext.slice(s![ext_range.start..ext_range.end, ..]);
    gamma_row.dot(&ctmul_ext)
}

/// Compute ESPGRAD correction to SHIFTCT for a monomer.
///
/// For each pair containing this monomer, subtract the per-pair CTIJ self-interaction.
/// `SHIFTCT[ext_a] -= sum over pair atoms b: gamma_ext[ext_I_a, ext_b] * CTIJ[b]`
pub fn compute_espgrad_shiftct_hop(
    frag_idx: usize,
    hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    pair_states: &[PairHopScc],
    pairs: &[Pair],
) -> Array1<f64> {
    let fi = &hop_data.frag_info[frag_idx];
    let ext_range = &fi.ext_range;
    let n_ext_frag = ext_range.end - ext_range.start;
    let mut espgrad = Array1::<f64>::zeros(n_ext_frag);

    for (pair_idx, pair) in pairs.iter().enumerate() {
        let is_i = pair.i == frag_idx;
        let is_j = pair.j == frag_idx;
        if !is_i && !is_j {
            continue;
        }
        let ps = &pair_states[pair_idx];
        let fi_i = &hop_data.frag_info[pair.i];
        let fi_j = &hop_data.frag_info[pair.j];
        let n_real_i = fi_i.n_real_atoms;
        let n_real_j = fi_j.n_real_atoms;

        // Build extended CTIJ for both fragments' atoms in this pair
        // We need to contract gamma_ext[ext_frag, ext_pair_atoms] . ctij[ext_pair_atoms]
        // The pair covers ext_I and ext_J atoms (plus ghosts, mapped via bonds)

        // Fragment I real atoms CTIJ
        for a in 0..n_real_i {
            let popmat_pair = ps.dq[a] + ps.zref[a];
            let mono_zref = hop_data.zref[fi_i.ext_range.start + a];
            let popmat_mono = mono_states[pair.i].dq[a] + mono_zref;
            let ctij = popmat_pair - popmat_mono;
            if ctij.abs() < 1e-30 {
                continue;
            }
            let ext_a_global = fi_i.ext_range.start + a;
            for local_j in 0..n_ext_frag {
                espgrad[local_j] -=
                    hop_data.gamma_ext[[ext_range.start + local_j, ext_a_global]] * ctij;
            }
        }

        // Fragment J real atoms CTIJ
        for a in 0..n_real_j {
            let pair_a = n_real_i + a;
            let popmat_pair = ps.dq[pair_a] + ps.zref[pair_a];
            let mono_zref = hop_data.zref[fi_j.ext_range.start + a];
            let popmat_mono = mono_states[pair.j].dq[a] + mono_zref;
            let ctij = popmat_pair - popmat_mono;
            if ctij.abs() < 1e-30 {
                continue;
            }
            let ext_a_global = fi_j.ext_range.start + a;
            for local_j in 0..n_ext_frag {
                espgrad[local_j] -=
                    hop_data.gamma_ext[[ext_range.start + local_j, ext_a_global]] * ctij;
            }
        }

        // Ghost atoms CTIJ
        let mut mono_ghost_i_idx = 0usize;
        let mut mono_ghost_j_idx = 0usize;
        let mut pair_ghost_idx = ps.n_real_atoms;

        for bond in &hop_data.detached_bonds {
            let bda_in_pair = bond.bda_fragment == pair.i || bond.bda_fragment == pair.j;

            if bond.baa_fragment == pair.i {
                let mono_ghost_local = n_real_i + mono_ghost_i_idx;
                let mono_zref_g = hop_data.zref[fi_i.ext_range.start + mono_ghost_local];
                let popmat_mono_g = mono_states[pair.i].dq[mono_ghost_local] + mono_zref_g;

                let ctij = if bda_in_pair {
                    -popmat_mono_g
                } else {
                    let popmat_pair_g = ps.dq[pair_ghost_idx] + ps.zref[pair_ghost_idx];
                    pair_ghost_idx += 1;
                    popmat_pair_g - popmat_mono_g
                };

                if ctij.abs() >= 1e-30 {
                    let ext_a_global = fi_i.ext_range.start + mono_ghost_local;
                    for local_j in 0..n_ext_frag {
                        espgrad[local_j] -=
                            hop_data.gamma_ext[[ext_range.start + local_j, ext_a_global]] * ctij;
                    }
                }
                mono_ghost_i_idx += 1;
            } else if bond.baa_fragment == pair.j {
                let mono_ghost_local = n_real_j + mono_ghost_j_idx;
                let mono_zref_g = hop_data.zref[fi_j.ext_range.start + mono_ghost_local];
                let popmat_mono_g = mono_states[pair.j].dq[mono_ghost_local] + mono_zref_g;

                let ctij = if bda_in_pair {
                    -popmat_mono_g
                } else {
                    let popmat_pair_g = ps.dq[pair_ghost_idx] + ps.zref[pair_ghost_idx];
                    pair_ghost_idx += 1;
                    popmat_pair_g - popmat_mono_g
                };

                if ctij.abs() >= 1e-30 {
                    let ext_a_global = fi_j.ext_range.start + mono_ghost_local;
                    for local_j in 0..n_ext_frag {
                        espgrad[local_j] -=
                            hop_data.gamma_ext[[ext_range.start + local_j, ext_a_global]] * ctij;
                    }
                }
                mono_ghost_j_idx += 1;
            }
        }
    }

    espgrad
}

/// Compute ESP from all other fragments for a given fragment (external ESP).
///
/// `esp_q[ext_a] = gamma_ext[ext_I, :] . dq_ext - gamma_ext[ext_I, ext_I] . dq_I`
pub fn compute_esp_q_hop(
    frag_idx: usize,
    hop_data: &HopData,
) -> Array1<f64> {
    let fi = &hop_data.frag_info[frag_idx];
    let ext_range = &fi.ext_range;

    let esp_full: Array1<f64> = hop_data
        .gamma_ext
        .slice(s![ext_range.start..ext_range.end, ..])
        .dot(&hop_data.dq_ext);

    let esp_self: Array1<f64> = hop_data
        .gamma_ext
        .slice(s![ext_range.start..ext_range.end, ext_range.start..ext_range.end])
        .dot(&hop_data.dq_ext.slice(s![ext_range.start..ext_range.end]));

    &esp_full - &esp_self
}

/// Get the list of BDA global indices for pair ghost atoms (partial-BAA bonds).
///
/// These are bonds where BAA is in the pair but BDA is outside.
/// Ghost atoms are at BDA's position → scatter to bda_global.
pub fn get_pair_ghost_baa_globals(
    hop_data: &HopData,
    pair_i: usize,
    pair_j: usize,
) -> Vec<usize> {
    let mut ghost_bda = Vec::new();
    for bond in &hop_data.detached_bonds {
        let bda_in_pair = bond.bda_fragment == pair_i || bond.bda_fragment == pair_j;
        let baa_in_pair = bond.baa_fragment == pair_i || bond.baa_fragment == pair_j;
        if !bda_in_pair && baa_in_pair {
            ghost_bda.push(bond.bda_global);
        }
    }
    ghost_bda
}
