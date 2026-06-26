//! HOP projector gradient for FMO-xTB HOP.
//!
//! Computes Tr(rho * d(P_HOP)/dR) for the FMO delta: pair HOP - monomer HOP.
//! Uses hop_gradient_single_bond from the existing xTB HOP gradient code.

use super::helpers::{
    build_monomer_local_to_global, get_healed_bonds_for_pair, get_partial_bonds_for_pair,
};
use crate::fmo::scc_hop::hop_data::{
    compute_bda_dd_xtb, compute_ghost_nonbond_dd_xtb, compute_rotated_sp3_xtb, XtbHopData,
};
use crate::fmo::scc_hop::monomer::XtbMonomerHopScc;
use crate::fmo::scc_hop::pair::XtbPairHopScc;
use crate::fmo::scc_hop::trimer::XtbTrimerHopScc;
use crate::gradients::hop_gradient::hop_gradient_single_bond_general;
use crate::hop::{DetachedBond, HOP_SHIFT};
use crate::initialization::atom::XtbAtom;
use ndarray::prelude::*;
use rayon::prelude::*;

/// xTB FMO-HOP p-orbital ordering: (s, py, pz, px) → maps orbital idx to Cartesian (x=0,y=1,z=2)
const XTB_P_ORDER: [usize; 3] = [1, 2, 0];

/// Compute HOP projector gradient: monomer HOP + pair HOP delta.
///
/// Step 1: Monomer HOP gradients for all monomers with detached bonds
/// Step 2: Pair HOP delta = pair HOP - monomer HOP for partial/healed bonds
///
/// Returns (hop_total, hop_mono_only, hop_pair_delta_only) for diagnostic purposes.
pub fn compute_hop_gradient_fmo_xtb_hop(
    hop_data: &XtbHopData,
    mono_states: &[XtbMonomerHopScc],
    pair_states: &[XtbPairHopScc],
    atoms: &[XtbAtom],
    hop_total: &mut Array1<f64>,
    frag_atom_ranges: &[std::ops::Range<usize>],
    pair_scal: &[f64],
    trimer_frags: &[(usize, usize, usize)],
    trimer_states: &[XtbTrimerHopScc],
) -> (Array1<f64>, Array1<f64>, Vec<Array1<f64>>) {
    let n_grad = hop_total.len();
    let mut hop_mono_only = Array1::<f64>::zeros(n_grad);
    let mut hop_pair_delta = Array1::<f64>::zeros(n_grad);
    let mut per_pair_hop_delta: Vec<Array1<f64>> = pair_states.iter().map(|_| Array1::zeros(n_grad)).collect();
    if hop_data.detached_bonds.is_empty() {
        return (hop_mono_only, hop_pair_delta, per_pair_hop_delta);
    }

    // Step 1: Monomer HOP gradients (BDA + ghost bonds) — parallelized
    let mono_hop_results: Vec<Array1<f64>> = mono_states
        .par_iter()
        .enumerate()
        .map(|(frag_idx, mono)| {
            let bda_bonds: Vec<&DetachedBond> = hop_data
                .detached_bonds
                .iter()
                .filter(|b| b.bda_fragment == frag_idx)
                .collect();
            let ghost_bonds_mono: Vec<(usize, &DetachedBond)> = {
                let mut ghost_idx = 0usize;
                hop_data
                    .detached_bonds
                    .iter()
                    .filter_map(|b| {
                        if b.baa_fragment == frag_idx {
                            let idx = ghost_idx;
                            ghost_idx += 1;
                            Some((idx, b))
                        } else {
                            None
                        }
                    })
                    .collect()
            };

            let mut frag_global = Array1::<f64>::zeros(n_grad);
            if bda_bonds.is_empty() && ghost_bonds_mono.is_empty() {
                return frag_global;
            }

            let p: ArrayView2<f64> = mono.p.view();
            let s: ArrayView2<f64> = mono.s.view();
            let n_atoms = mono.n_ext_atoms;
            let mut local_grad = Array1::<f64>::zeros(3 * n_atoms);

            for bond in &bda_bonds {
                let bda_local = bond.bda_global - frag_atom_ranges[frag_idx].start;
                let bda_pos = &atoms[bond.bda_global].xyz;
                let baa_pos = &atoms[bond.baa_global].xyz;
                let bond_vec = *baa_pos - *bda_pos;
                let rotated_sp3 = compute_rotated_sp3_xtb(&bond_vec);
                let dd_bda = compute_bda_dd_xtb(rotated_sp3.view(), HOP_SHIFT);
                let mut baa_grad_3 = [0.0f64; 3];
                hop_gradient_single_bond_general(
                    p, s, &mono.basis, bda_local, bda_pos, baa_pos,
                    dd_bda.view(), rotated_sp3.view(), &XTB_P_ORDER,
                    1.0, &mut local_grad, &mut baa_grad_3,
                );
                for k in 0..3 { frag_global[3 * bond.baa_global + k] += baa_grad_3[k]; }
            }

            for &(gi, bond) in &ghost_bonds_mono {
                let ghost_local = mono.n_real_atoms + gi;
                let bda_pos = &atoms[bond.bda_global].xyz;
                let baa_pos = &atoms[bond.baa_global].xyz;
                let bond_vec = *baa_pos - *bda_pos;
                let rotated_sp3 = compute_rotated_sp3_xtb(&bond_vec);
                let dd_ghost = compute_ghost_nonbond_dd_xtb(&bond_vec, HOP_SHIFT);
                let mut baa_grad_3 = [0.0f64; 3];
                hop_gradient_single_bond_general(
                    p, s, &mono.basis, ghost_local, bda_pos, baa_pos,
                    dd_ghost.view(), rotated_sp3.view(), &XTB_P_ORDER,
                    -1.0, &mut local_grad, &mut baa_grad_3,
                );
                for k in 0..3 { frag_global[3 * bond.baa_global + k] += baa_grad_3[k]; }
            }

            let local_to_global =
                build_monomer_local_to_global(frag_atom_ranges[frag_idx].clone(), hop_data, frag_idx);
            for local_idx in 0..n_atoms {
                let global_idx = local_to_global[local_idx];
                for k in 0..3 { frag_global[3 * global_idx + k] += local_grad[3 * local_idx + k]; }
            }
            frag_global
        })
        .collect();
    for frag_grad in &mono_hop_results {
        *hop_total += frag_grad;
    }

    // Snapshot after step 1: this is the monomer-only HOP gradient
    hop_mono_only.assign(hop_total);

    // Step 2: Pair HOP gradients (delta: pair - mono_bda), SCAL-scaled for FMO3
    for (pair_idx, ps) in pair_states.iter().enumerate() {
        let scal = pair_scal[pair_idx];
        let hop_before_pair = hop_total.clone();
        let partial = get_partial_bonds_for_pair(hop_data, ps.i, ps.j);

        // Pair HOP gradient for partial bonds
        if !partial.is_empty() {
            let p_pair: ArrayView2<f64> = ps.p.view();
            let s_pair: ArrayView2<f64> = ps.s.view();
            let n_atoms_pair = ps.n_ext_atoms;
            let n_real_i = ps.n_real_i;
            let n_real_j = ps.n_real_j;
            let mut pair_local_grad = Array1::<f64>::zeros(3 * n_atoms_pair);

            // Split partial bonds into two types:
            // partial-BDA: BDA in pair (real atom), BAA outside
            // partial-BAA: BAA in pair, BDA outside (ghost atom)
            let partial_bda: Vec<&DetachedBond> = partial
                .iter()
                .filter(|b| b.bda_fragment == ps.i || b.bda_fragment == ps.j)
                .copied()
                .collect();
            let partial_baa: Vec<&DetachedBond> = partial
                .iter()
                .filter(|b| b.baa_fragment == ps.i || b.baa_fragment == ps.j)
                .filter(|b| b.bda_fragment != ps.i && b.bda_fragment != ps.j)
                .copied()
                .collect();

            // Process partial-BDA bonds: BDA is a real atom in the pair
            // Uses DD_bda and coeff_sign = +1
            for bond in &partial_bda {
                let bda_local = if bond.bda_fragment == ps.i {
                    bond.bda_global - frag_atom_ranges[ps.i].start
                } else {
                    n_real_i + (bond.bda_global - frag_atom_ranges[ps.j].start)
                };

                let bda_pos = &atoms[bond.bda_global].xyz;
                let baa_pos = &atoms[bond.baa_global].xyz;
                let bond_vec = *baa_pos - *bda_pos;
                let rotated_sp3 = compute_rotated_sp3_xtb(&bond_vec);
                let dd_bda = compute_bda_dd_xtb(rotated_sp3.view(), HOP_SHIFT);
                let mut baa_grad_3 = [0.0f64; 3];

                hop_gradient_single_bond_general(
                    p_pair,
                    s_pair,
                    &ps.basis,
                    bda_local,
                    bda_pos,
                    baa_pos,
                    dd_bda.view(),
                    rotated_sp3.view(),
                    &XTB_P_ORDER,
                    1.0,
                    &mut pair_local_grad,
                    &mut baa_grad_3,
                );

                for k in 0..3 {
                    hop_total[3 * bond.baa_global + k] += baa_grad_3[k];
                }
            }

            // Process partial-BAA bonds: BDA is a ghost atom in the pair
            // Uses DD_ghost and coeff_sign = -1
            // Build ghost index mapping for this pair
            let mut ghost_idx = 0;
            let ghost_bonds: Vec<(usize, &DetachedBond)> = hop_data
                .detached_bonds
                .iter()
                .filter_map(|b| {
                    let bda_in = b.bda_fragment == ps.i || b.bda_fragment == ps.j;
                    let baa_in = b.baa_fragment == ps.i || b.baa_fragment == ps.j;
                    if baa_in && !bda_in {
                        let idx = ghost_idx;
                        ghost_idx += 1;
                        Some((idx, b))
                    } else {
                        None
                    }
                })
                .collect();

            for bond in &partial_baa {
                // Find ghost index for this bond
                let gi = ghost_bonds
                    .iter()
                    .find(|(_, gb)| {
                        gb.bda_global == bond.bda_global && gb.baa_global == bond.baa_global
                    })
                    .expect("partial-BAA bond must have a corresponding ghost")
                    .0;
                let bda_local = n_real_i + n_real_j + gi;

                let bda_pos = &atoms[bond.bda_global].xyz;
                let baa_pos = &atoms[bond.baa_global].xyz;
                let bond_vec = *baa_pos - *bda_pos;
                let rotated_sp3 = compute_rotated_sp3_xtb(&bond_vec);
                let dd_ghost = compute_ghost_nonbond_dd_xtb(&bond_vec, HOP_SHIFT);
                let mut baa_grad_3 = [0.0f64; 3];

                hop_gradient_single_bond_general(
                    p_pair,
                    s_pair,
                    &ps.basis,
                    bda_local,
                    bda_pos,
                    baa_pos,
                    dd_ghost.view(),
                    rotated_sp3.view(),
                    &XTB_P_ORDER,
                    -1.0,
                    &mut pair_local_grad,
                    &mut baa_grad_3,
                );

                for k in 0..3 {
                    hop_total[3 * bond.baa_global + k] += baa_grad_3[k];
                }
            }

            // Scatter pair local gradient to global
            // Real I
            for (local_idx, global_idx) in frag_atom_ranges[ps.i].clone().enumerate() {
                for k in 0..3 {
                    hop_total[3 * global_idx + k] += pair_local_grad[3 * local_idx + k];
                }
            }
            // Real J
            for (local_idx, global_idx) in frag_atom_ranges[ps.j].clone().enumerate() {
                for k in 0..3 {
                    hop_total[3 * global_idx + k] +=
                        pair_local_grad[3 * (n_real_i + local_idx) + k];
                }
            }
            // Ghost atoms → BDA global
            for &(gi, bond) in &ghost_bonds {
                let local_idx = n_real_i + n_real_j + gi;
                for k in 0..3 {
                    hop_total[3 * bond.bda_global + k] +=
                        pair_local_grad[3 * local_idx + k];
                }
            }

            // Subtract monomer contributions for partial bonds
            // Partial-BDA: subtract only BDA contribution from bda_frag's monomer
            for bond in &partial_bda {
                subtract_bda_hop(
                    bond, mono_states, hop_data, atoms, frag_atom_ranges, hop_total, -1.0,
                );
            }
            // Partial-BAA: subtract only ghost contribution from baa_frag's monomer
            for bond in &partial_baa {
                subtract_ghost_hop(
                    bond, mono_states, hop_data, atoms, frag_atom_ranges, hop_total, -1.0,
                );
            }
        }

        // Healed bonds: subtract both BDA and ghost contributions
        let healed = get_healed_bonds_for_pair(hop_data, ps.i, ps.j);
        for bond in &healed {
            subtract_bda_hop(
                bond, mono_states, hop_data, atoms, frag_atom_ranges, hop_total, -1.0,
            );
            subtract_ghost_hop(
                bond, mono_states, hop_data, atoms, frag_atom_ranges, hop_total, -1.0,
            );
        }

        // Capture per-pair HOP delta (UNSCALED) and apply SCAL
        let this_pair_delta = &*hop_total - &hop_before_pair;
        per_pair_hop_delta[pair_idx] = this_pair_delta.clone();
        if (scal - 1.0).abs() > 1e-14 {
            *hop_total = &hop_before_pair + &(scal * &this_pair_delta);
        }
    }

    // Step 3: Trimer HOP correction (FMO3)
    // For each bond healed in at least one trimer: subtract the remaining monomer HOP
    // that wasn't already subtracted by SCAL-adjusted pair deltas.
    // SCAL = 1 - n_trimers_containing_pair. The pair delta was SCAL-scaled.
    // For a healed bond, pair_delta = -(mono_bda + mono_ghost). After SCAL:
    //   hop_total has: mono + SCAL * pair_delta = mono + SCAL * (-(mono))
    //                = mono * (1 - SCAL) = mono * n_trimers
    // We need hop_total = 0 (for complete expansion).
    // So subtract mono * n_trimers = remaining.
    // But we process per-bond, not per-trimer, to avoid double-counting.
    for bond in &hop_data.detached_bonds {
        // Find the sub-pair that heals this bond
        let heal_pair = pair_states.iter().enumerate().find(|(_, ps)| {
            (ps.i == bond.bda_fragment && ps.j == bond.baa_fragment)
                || (ps.j == bond.bda_fragment && ps.i == bond.baa_fragment)
        });
        if let Some((pidx, _)) = heal_pair {
            let scal = pair_scal[pidx];
            // Number of trimers containing this pair = 1 - SCAL
            let n_tri = 1.0 - scal;
            if n_tri.abs() > 1e-14 {
                // Subtract n_tri × mono_HOP for this bond
                subtract_bda_hop(bond, mono_states, hop_data, atoms, frag_atom_ranges, hop_total, -n_tri);
                subtract_ghost_hop(bond, mono_states, hop_data, atoms, frag_atom_ranges, hop_total, -n_tri);
            }
        }
    }

    // Pair delta = total - monomer-only
    hop_pair_delta.assign(&(hop_total as &Array1<f64>));
    hop_pair_delta -= &hop_mono_only;
    (hop_mono_only, hop_pair_delta, per_pair_hop_delta)
}

/// Subtract the BDA contribution from bda_fragment's monomer for one bond.
fn subtract_bda_hop(
    bond: &DetachedBond,
    mono_states: &[XtbMonomerHopScc],
    hop_data: &XtbHopData,
    atoms: &[XtbAtom],
    frag_atom_ranges: &[std::ops::Range<usize>],
    hop_total: &mut Array1<f64>,
    scale: f64,
) {
    let bda_frag = bond.bda_fragment;
    let mono = &mono_states[bda_frag];
    let bda_local = bond.bda_global - frag_atom_ranges[bda_frag].start;

    let mut mono_local_grad = Array1::<f64>::zeros(3 * mono.n_ext_atoms);
    let bda_pos = &atoms[bond.bda_global].xyz;
    let baa_pos = &atoms[bond.baa_global].xyz;
    let bond_vec = *baa_pos - *bda_pos;
    let rotated_sp3 = compute_rotated_sp3_xtb(&bond_vec);
    let dd_bda = compute_bda_dd_xtb(rotated_sp3.view(), HOP_SHIFT);
    let mut baa_grad_3 = [0.0f64; 3];

    hop_gradient_single_bond_general(
        mono.p.view(),
        mono.s.view(),
        &mono.basis,
        bda_local,
        bda_pos,
        baa_pos,
        dd_bda.view(),
        rotated_sp3.view(),
        &XTB_P_ORDER,
        1.0,
        &mut mono_local_grad,
        &mut baa_grad_3,
    );

    // Scatter to global: real atoms
    for (local_idx, global_idx) in frag_atom_ranges[bda_frag].clone().enumerate() {
        for k in 0..3 {
            hop_total[3 * global_idx + k] += scale * mono_local_grad[3 * local_idx + k];
        }
    }
    // Ghost atoms in bda_frag's monomer
    let mut ghost_count = 0;
    for gbond in &hop_data.detached_bonds {
        if gbond.baa_fragment == bda_frag {
            let local_idx = mono.n_real_atoms + ghost_count;
            for k in 0..3 {
                hop_total[3 * gbond.bda_global + k] +=
                    scale * mono_local_grad[3 * local_idx + k];
            }
            ghost_count += 1;
        }
    }
    // BAA gradient
    for k in 0..3 {
        hop_total[3 * bond.baa_global + k] += scale * baa_grad_3[k];
    }
}

/// Subtract the ghost contribution from baa_fragment's monomer for one bond.
fn subtract_ghost_hop(
    bond: &DetachedBond,
    mono_states: &[XtbMonomerHopScc],
    hop_data: &XtbHopData,
    atoms: &[XtbAtom],
    frag_atom_ranges: &[std::ops::Range<usize>],
    hop_total: &mut Array1<f64>,
    scale: f64,
) {
    let baa_frag = bond.baa_fragment;
    let mono_baa = &mono_states[baa_frag];
    let bda_pos = &atoms[bond.bda_global].xyz;
    let baa_pos = &atoms[bond.baa_global].xyz;
    let bond_vec = *baa_pos - *bda_pos;

    // Find ghost index in baa_frag's monomer
    let mut ghost_local = None;
    let mut gi = 0;
    for gb in &hop_data.detached_bonds {
        if gb.baa_fragment == baa_frag {
            if gb.bda_global == bond.bda_global && gb.baa_global == bond.baa_global {
                ghost_local = Some(mono_baa.n_real_atoms + gi);
                break;
            }
            gi += 1;
        }
    }

    let ghost_idx = ghost_local.expect("Ghost bond not found in baa_fragment's monomer");
    let rotated_sp3 = compute_rotated_sp3_xtb(&bond_vec);
    let dd_ghost = compute_ghost_nonbond_dd_xtb(&bond_vec, HOP_SHIFT);
    let mut mono_baa_grad = Array1::<f64>::zeros(3 * mono_baa.n_ext_atoms);
    let mut baa_grad_ghost = [0.0f64; 3];

    hop_gradient_single_bond_general(
        mono_baa.p.view(),
        mono_baa.s.view(),
        &mono_baa.basis,
        ghost_idx,
        bda_pos, // ghost is at BDA position
        baa_pos,
        dd_ghost.view(),
        rotated_sp3.view(),
        &XTB_P_ORDER,
        -1.0,
        &mut mono_baa_grad,
        &mut baa_grad_ghost,
    );

    // Scatter to global: real atoms of baa_frag
    for (local_idx, global_idx) in frag_atom_ranges[baa_frag].clone().enumerate() {
        for k in 0..3 {
            hop_total[3 * global_idx + k] += scale * mono_baa_grad[3 * local_idx + k];
        }
    }
    // Ghost atoms in baa_frag's monomer
    let mut gc = 0;
    for gb in &hop_data.detached_bonds {
        if gb.baa_fragment == baa_frag {
            let li = mono_baa.n_real_atoms + gc;
            for k in 0..3 {
                hop_total[3 * gb.bda_global + k] += scale * mono_baa_grad[3 * li + k];
            }
            gc += 1;
        }
    }
    // BAA gradient for ghost bond
    for k in 0..3 {
        hop_total[3 * bond.baa_global + k] += scale * baa_grad_ghost[k];
    }
}
