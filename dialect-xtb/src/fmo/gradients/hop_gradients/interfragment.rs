//! Inter-fragment gradient for FMO-xTB HOP.
//!
//! CTMUL embedding + ESD between fragments, using extended atoms and shell-level gamma.

use crate::fmo::scc_hop::hop_data::{get_frag_shell_range, XtbHopData};
use crate::fmo::scc_hop::monomer::XtbMonomerHopScc;
use crate::fmo::pair::XtbESDPair;
use crate::initialization::atom::XtbAtom;
use crate::scc::gamma_matrix::XtbGammaFunction;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;

/// Compute the inter-fragment gradient for HOP.
///
/// For each monomer I, loops over ALL extended shells (outer) vs monomer I's REAL shells (inner):
/// - CTMUL: ctmul_ext[a] * dq_I_shell[c] * dgamma/dR
/// - ESD: dq_ext[a] * dq_I_shell[c] * dgamma/dR (only when frag_a > frag_I)
///
/// Ghost atoms scatter to their BDA's global position via `ext_to_global` mapping.
pub fn interfragment_gradient_xtb_hop(
    atoms: &[XtbAtom],
    hop_data: &XtbHopData,
    mono_states: &[XtbMonomerHopScc],
    esd_pairs: &[XtbESDPair],
    ctmul_ext: ArrayView1<f64>,
    gammafunction: &XtbGammaFunction,
    trimer_frags: &[(usize, usize, usize)], // FMO3 trimers for self-fragment filtering
) -> Array1<f64> {
    let n_atoms_total = atoms.len();
    let n_frags = mono_states.len();

    // Build ext_atom_index → fragment mapping and ext → global mapping
    let mut ext_to_frag = vec![0usize; hop_data.n_ext_atoms];
    let mut ext_to_global = vec![0usize; hop_data.n_ext_atoms];
    for (frag_idx, fi) in hop_data.frag_info.iter().enumerate() {
        let frag_atom_range = &mono_states[frag_idx].ext_atoms;
        let n_real = fi.n_real_atoms;
        // We need the global atom range for this fragment
        // For real atoms: global index = frag_atom_start + local
        // For ghosts: global index = BDA global
        let mut ghost_bond_idx = 0;
        for (local, ext_idx) in fi.ext_range.clone().enumerate() {
            ext_to_frag[ext_idx] = frag_idx;
            if local < n_real {
                // Find the global atom range for this fragment's real atoms
                ext_to_global[ext_idx] = hop_data.monomer_indices[frag_idx][local];
            } else {
                // Ghost atom: find the corresponding bond
                let mut count = 0;
                for bond in &hop_data.detached_bonds {
                    if bond.baa_fragment == frag_idx {
                        if count == local - n_real {
                            ext_to_global[ext_idx] = bond.bda_global;
                            break;
                        }
                        count += 1;
                    }
                }
            }
        }
    }

    // ESD pair lookup
    let mut esd_lookup: HashSet<(usize, usize)> = HashSet::new();
    for esd in esd_pairs.iter() {
        esd_lookup.insert((esd.i, esd.j));
        esd_lookup.insert((esd.j, esd.i));
    }

    // Parallel fold over monomers
    let gradient: Array1<f64> = (0..n_frags)
        .into_par_iter()
        .fold(
            || Array1::<f64>::zeros(3 * n_atoms_total),
            |mut gradient, m_idx| {
                let fi = &hop_data.frag_info[m_idx];
                let shell_range_i = get_frag_shell_range(&hop_data.ext_basis, &fi.ext_range);

                // Monomer I's dq_shell (all shells including ghosts)
                let dq_i_shell = &mono_states[m_idx].dq_shell;

                // Include all shells (real + ghost) of fragment I.
                let n_shells_to_use = shell_range_i.len();
                for (s_c_local, s_c_global) in shell_range_i
                    .clone()
                    .enumerate()
                    .take(n_shells_to_use)
                {
                    let shell_c = &hop_data.ext_basis.shells[s_c_global];
                    let ext_c = shell_c.atom_index;
                    let global_c = ext_to_global[ext_c];
                    let atom_c = &hop_data.ext_atoms[ext_c];

                    // Outer loop: ALL extended shells
                    for (s_a_global, shell_a) in hop_data.ext_basis.shells.iter().enumerate() {
                        let ext_a = shell_a.atom_index;
                        let frag_a = ext_to_frag[ext_a];
                        let global_a = ext_to_global[ext_a];

                        if global_a == global_c {
                            continue;
                        }

                        let is_esd =
                            frag_a > m_idx && esd_lookup.contains(&(m_idx, frag_a));
                        let dq_esd_a = if is_esd {
                            hop_data.dq_shell_ext[s_a_global]
                        } else {
                            0.0
                        };
                        let ct_a = ctmul_ext[s_a_global];
                        if ct_a.abs() < 1e-14 && dq_esd_a.abs() < 1e-14 {
                            continue;
                        }

                        let atom_a = &hop_data.ext_atoms[ext_a];
                        let dx = atom_a.xyz[0] - atom_c.xyz[0];
                        let dy = atom_a.xyz[1] - atom_c.xyz[1];
                        let dz = atom_a.xyz[2] - atom_c.xyz[2];
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        if dist < 1e-10 {
                            continue;
                        }

                        let dgamma_dr = gammafunction.deriv(
                            dist,
                            atom_a.number,
                            shell_a.angular_momentum as u8,
                            atom_c.number,
                            shell_c.angular_momentum as u8,
                        );
                        let total_factor =
                            (ct_a + dq_esd_a) * dq_i_shell[s_c_local] * dgamma_dr / dist;

                        gradient[3 * global_a + 0] += total_factor * dx;
                        gradient[3 * global_a + 1] += total_factor * dy;
                        gradient[3 * global_a + 2] += total_factor * dz;
                        gradient[3 * global_c + 0] -= total_factor * dx;
                        gradient[3 * global_c + 1] -= total_factor * dy;
                        gradient[3 * global_c + 2] -= total_factor * dz;
                    }
                }
                gradient
            },
        )
        .reduce(
            || Array1::<f64>::zeros(3 * n_atoms_total),
            |mut a, b| {
                a += &b;
                a
            },
        );

    gradient
}
