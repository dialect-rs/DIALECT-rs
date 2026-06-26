//! Inter-fragment gradient with HOP for FMO-DFTB.
//!
//! Adapts `interfragment_gradient()` from `fmo_gradient.rs` for extended charges:
//! - CTMUL and ESD use extended atom arrays (including ghost atoms)
//! - Ghost atoms participate in gamma derivative computation
//! - Atom-to-fragment mapping accounts for ghost atoms (mapped to BAA's fragment)

use crate::fmo::scc_hop::hop_data::HopData;
use crate::fmo::scc_hop::monomer::MonomerHopScc;
use crate::fmo::ESDPair;
use crate::initialization::Atom;
use crate::scc::gamma_approximation::GammaFunction;
use ndarray::prelude::*;
use std::collections::HashSet;

/// Combined inter-fragment gradient with HOP: CTMUL embedding + ES-dimer.
///
/// For each monomer I, loops over all extended atoms a and monomer's extended atoms c:
/// - CTMUL contribution: `ctmul_ext[a] * dq_I[c] * dgamma(a,c)/dR`
/// - ESD contribution (if frag_a > m_i_index and ESD pair): `dq_ext_J[a] * dq_I[c] * dgamma(a,c)/dR`
///
/// Key HOP differences:
/// - `a` ranges over ALL ext_atoms (real + ghost) from all fragments
/// - `c` ranges over ALL ext atoms of monomer I (real + ghost)
/// - Ghost atoms use gamma between ghost and real atoms (well-defined)
/// - ESD charges use extended dq_ext including ghost charges
pub fn interfragment_gradient_hop(
    atoms: &[Atom],
    hop_data: &HopData,
    mono_states: &[MonomerHopScc],
    esd_pairs: &[ESDPair],
    ctmul_ext: ArrayView1<f64>,
    gammafunction: &GammaFunction,
) -> Array1<f64> {
    let n_atoms_total = atoms.len();
    let n_ext_atoms = hop_data.n_ext_atoms;
    let mut gradient = Array1::<f64>::zeros(3 * n_atoms_total);

    // Build ext_atom-to-fragment mapping
    let mut ext_to_frag = vec![0usize; n_ext_atoms];
    for fi in &hop_data.frag_info {
        for ext_idx in fi.ext_range.clone() {
            ext_to_frag[ext_idx] = fi.frag_idx;
        }
    }

    // Build ext_atom to global_atom mapping (for gradient assignment)
    // Real atoms: direct mapping via frag_atom_range
    // Ghost atoms: at BDA's position → mapped to BDA's global atom index
    let mut ext_to_global = vec![0usize; n_ext_atoms];
    for fi in &hop_data.frag_info {
        let frag_idx = fi.frag_idx;
        // Real atoms
        for (local_idx, ext_idx) in (fi.ext_range.start..fi.ext_range.start + fi.n_real_atoms).enumerate() {
            ext_to_global[ext_idx] = hop_data.monomer_indices[frag_idx][local_idx];
        }
        // Ghost atoms → BDA global (ghost is at BDA's position)
        let mut ghost_count = 0;
        for bond in &hop_data.detached_bonds {
            if bond.baa_fragment == frag_idx {
                let ext_idx = fi.ext_range.start + fi.n_real_atoms + ghost_count;
                ext_to_global[ext_idx] = bond.bda_global;
                ghost_count += 1;
            }
        }
    }

    // Build ESD pair lookup
    let mut esd_lookup: HashSet<(usize, usize)> = HashSet::new();
    for esd in esd_pairs.iter() {
        esd_lookup.insert((esd.i, esd.j));
        esd_lookup.insert((esd.j, esd.i));
    }

    // For each monomer I
    for (m_idx, mono) in mono_states.iter().enumerate() {
        let fi_i = &hop_data.frag_info[m_idx];
        let dq_i = &mono.dq;

        // For each ext atom a (from ALL fragments)
        for ext_a in 0..n_ext_atoms {
            let frag_a = ext_to_frag[ext_a];
            let global_a = ext_to_global[ext_a];
            let atom_a = &hop_data.ext_atoms[ext_a];

            // ESD: only count when frag_a > m_idx to avoid double-counting
            let is_esd = frag_a > m_idx && esd_lookup.contains(&(m_idx, frag_a));

            let dq_esd_a = if is_esd {
                hop_data.dq_ext[ext_a]
            } else {
                0.0
            };

            let ct_a = ctmul_ext[ext_a];

            if ct_a.abs() < 1e-14 && dq_esd_a.abs() < 1e-14 {
                continue;
            }

            // For each ext atom c in monomer I (real + ghost)
            for (local_c, ext_c) in fi_i.ext_range.clone().enumerate() {
                let global_c = ext_to_global[ext_c];
                if global_a == global_c {
                    continue;
                }

                let atom_c = &hop_data.ext_atoms[ext_c];
                let dq_c = dq_i[local_c];

                let dx = atom_a.xyz[0] - atom_c.xyz[0];
                let dy = atom_a.xyz[1] - atom_c.xyz[1];
                let dz = atom_a.xyz[2] - atom_c.xyz[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < 1e-10 {
                    continue;
                }

                let dgamma_dr = gammafunction.deriv(dist, atom_a.number, atom_c.number);
                let total_factor = (ct_a + dq_esd_a) * dq_c * dgamma_dr / dist;

                gradient[3 * global_a + 0] += total_factor * dx;
                gradient[3 * global_a + 1] += total_factor * dy;
                gradient[3 * global_a + 2] += total_factor * dz;
                gradient[3 * global_c + 0] -= total_factor * dx;
                gradient[3 * global_c + 1] -= total_factor * dy;
                gradient[3 * global_c + 2] -= total_factor * dz;
            }
        }
    }

    gradient
}
