//! Trimer SCC with HOP for FMO-xTB (FMO3).
//!
//! Trimers can have:
//! - **Healed bonds**: both BDA and BAA in trimer → no ghost, no HOP, no n_elec change
//! - **Partial-BDA bonds**: BDA in trimer, BAA outside → HOP on BDA, ZREF-=1, n_elec-=1
//! - **Partial-BAA bonds**: BAA in trimer, BDA outside → ghost at BDA position, n_elec+=1

use super::hop_data::{
    calculate_repulsive_energy_xtb_scaled, compute_bda_dd_xtb,
    compute_ghost_nonbond_dd_xtb, compute_rotated_sp3_xtb,
    create_xtb_ghost_atom, get_frag_shell_range, XtbHopData, XtbHopFragInfo,
};
use dialect_utilities::scc_helpers::aovec_to_aomat;
use dialect_utilities::fermi_occupation::compute_total_entropy;
use dialect_utilities::linalg::eigh::{compute_s_inv_sqrt, dsyevd_eigh};
use dialect_utilities::mixer::BroydenMixerNew;
use dialect_utilities::mulliken::{
    ao_to_shell_charges, mulliken_aowise_diff, mulliken_atomwise_from_ao_xtb,
    shell_to_ao_charges, shell_to_ao_values,
};
use dialect_utilities::fermi_occupation::fermi_occupation;
use dialect_utilities::scc_helpers::{density_matrix, outer_sum};
use crate::hop::{compute_hop_projector, get_bda_ao_range, DetachedBond, HOP_SHIFT};
use crate::initialization::atom::XtbAtom;
use crate::initialization::basis::{create_basis_set, Basis};
use crate::integrals::calc_overlap_matrix_parallel;
use crate::parameters::{COUL_THIRD_ORDER_ATOM, REFERENCE_OCCUPATION};
use crate::scc::gamma_matrix::gamma_matrix_shell;
use crate::scc::hamiltonian::h0_xtb1_with_cn;
use crate::scc::scc_helpers::{
    coul_third_order_hamiltonian, create_density_ref, get_electronic_energy_xtb_shell,
};
use ndarray::prelude::*;
use ndarray_stats::DeviationExt;

/// State for one trimer's HOP SCC calculation (xTB version).
#[derive(Debug, Clone)]
pub struct XtbTrimerHopScc {
    /// Monomer indices (i, j, k)
    pub i: usize,
    pub j: usize,
    pub k: usize,
    /// Extended trimer atoms: [real_I, real_J, real_K, partial_ghost_atoms]
    pub ext_atoms: Vec<XtbAtom>,
    /// Extended basis
    pub basis: Basis,
    pub n_ext_atoms: usize,
    pub n_real_i: usize,
    pub n_real_j: usize,
    pub n_real_k: usize,
    pub n_real_atoms: usize,
    pub n_ghost_atoms: usize,
    pub n_ext_orbs: usize,
    pub n_real_orbs: usize,
    pub n_real_orbs_i: usize,
    pub n_real_orbs_j: usize,
    pub n_ext_shells: usize,
    pub n_real_shells_i: usize,
    pub n_real_shells_j: usize,
    pub n_real_shells_k: usize,
    pub n_elec: usize,
    pub h0: Array2<f64>,
    pub s: Array2<f64>,
    pub x: Array2<f64>,
    pub gamma_shell: Array2<f64>,
    /// External ESP from other fragments (AO matrix, precomputed)
    pub v_esp: Array2<f64>,
    pub p_hop: Option<Array2<f64>>,
    pub p_ref: Array2<f64>,
    pub p: Array2<f64>,
    pub dq: Array1<f64>,
    pub dq_ao: Array1<f64>,
    pub dq_shell: Array1<f64>,
    pub hubbard_derivatives: Array1<f64>,
    pub last_energy: f64,
    pub orbs: Option<Array2<f64>>,
    pub orbe: Option<Array1<f64>>,
    pub f: Vec<f64>,
    pub zref: Array1<f64>,
    pub qref: Array1<f64>,
    /// Initial dq_shell from monomer charges (for delta_dq_shell_real)
    pub initial_dq_shell: Array1<f64>,
    /// delta_dq_shell truncated to real shells only (for embedding)
    pub delta_dq_shell_real: Array1<f64>,
}

/// Get QREF for an XtbAtom
fn get_qref(atom: &XtbAtom) -> f64 {
    let idx = atom.number as usize - 1;
    REFERENCE_OCCUPATION[idx][0] + REFERENCE_OCCUPATION[idx][1] + REFERENCE_OCCUPATION[idx][2]
}

/// Prepare a trimer for HOP SCC (xTB version).
#[allow(clippy::too_many_arguments)]
pub fn prepare_trimer_hop_xtb(
    tri_i: usize,
    tri_j: usize,
    tri_k: usize,
    hop_data: &XtbHopData,
    mono_dq_i: ArrayView1<f64>,
    mono_dq_j: ArrayView1<f64>,
    mono_dq_k: ArrayView1<f64>,
    mono_dq_shell_i: ArrayView1<f64>,
    mono_dq_shell_j: ArrayView1<f64>,
    mono_dq_shell_k: ArrayView1<f64>,
    esp_q_shell_i: ArrayView1<f64>,
    esp_q_shell_j: ArrayView1<f64>,
    esp_q_shell_k: ArrayView1<f64>,
    fi_i: &XtbHopFragInfo,
    fi_j: &XtbHopFragInfo,
    fi_k: &XtbHopFragInfo,
    gammafunction: &crate::scc::gamma_matrix::XtbGammaFunction,
    atoms: &[XtbAtom],
    frag_atom_range_i: &std::ops::Range<usize>,
    frag_atom_range_j: &std::ops::Range<usize>,
    frag_atom_range_k: &std::ops::Range<usize>,
    cn_ext: ArrayView1<f64>,
) -> XtbTrimerHopScc {
    let real_atoms_i: Vec<XtbAtom> = atoms[frag_atom_range_i.clone()].to_vec();
    let real_atoms_j: Vec<XtbAtom> = atoms[frag_atom_range_j.clone()].to_vec();
    let real_atoms_k: Vec<XtbAtom> = atoms[frag_atom_range_k.clone()].to_vec();
    let n_real_i = real_atoms_i.len();
    let n_real_j = real_atoms_j.len();
    let n_real_k = real_atoms_k.len();
    let n_real_atoms = n_real_i + n_real_j + n_real_k;

    let real_basis_i = create_basis_set(&real_atoms_i);
    let real_basis_j = create_basis_set(&real_atoms_j);
    let real_basis_k = create_basis_set(&real_atoms_k);
    let n_real_orbs_i = real_basis_i.nbas;
    let n_real_orbs_j = real_basis_j.nbas;
    let n_real_orbs_k = real_basis_k.nbas;
    let n_real_orbs = n_real_orbs_i + n_real_orbs_j + n_real_orbs_k;
    let n_real_shells_i = real_basis_i.shells.len();
    let n_real_shells_j = real_basis_j.shells.len();
    let n_real_shells_k = real_basis_k.shells.len();

    // Classify bonds: 3 fragments instead of 2
    let mut partial_bda_bonds: Vec<&DetachedBond> = Vec::new();
    let mut partial_baa_bonds: Vec<&DetachedBond> = Vec::new();
    let mut ghost_atoms: Vec<XtbAtom> = Vec::new();

    for bond in &hop_data.detached_bonds {
        let bda_in_trimer =
            bond.bda_fragment == tri_i || bond.bda_fragment == tri_j || bond.bda_fragment == tri_k;
        let baa_in_trimer =
            bond.baa_fragment == tri_i || bond.baa_fragment == tri_j || bond.baa_fragment == tri_k;

        if bda_in_trimer && baa_in_trimer {
            // Healed
        } else if bda_in_trimer && !baa_in_trimer {
            partial_bda_bonds.push(bond);
        } else if !bda_in_trimer && baa_in_trimer {
            partial_baa_bonds.push(bond);
            let bda_pos = atoms[bond.bda_global].xyz;
            let bda_atom = &atoms[bond.bda_global];
            ghost_atoms.push(create_xtb_ghost_atom(bda_pos, bda_atom));
        }
    }

    let n_ghost_atoms = ghost_atoms.len();

    // Build extended trimer atoms: [real_I, real_J, real_K, ghosts]
    let mut ext_atoms: Vec<XtbAtom> = Vec::with_capacity(n_real_atoms + n_ghost_atoms);
    ext_atoms.extend(real_atoms_i.iter().cloned());
    ext_atoms.extend(real_atoms_j.iter().cloned());
    ext_atoms.extend(real_atoms_k.iter().cloned());
    ext_atoms.extend(ghost_atoms.iter().cloned());

    let n_ext_atoms = ext_atoms.len();
    let basis = create_basis_set(&ext_atoms);
    let n_ext_orbs = basis.nbas;
    let n_ext_shells = basis.shells.len();

    // CN for trimer atoms
    let cn_i = cn_ext.slice(s![fi_i.ext_range.start..fi_i.ext_range.start + n_real_i]);
    let cn_j = cn_ext.slice(s![fi_j.ext_range.start..fi_j.ext_range.start + n_real_j]);
    let cn_k = cn_ext.slice(s![fi_k.ext_range.start..fi_k.ext_range.start + n_real_k]);
    let mut cn_trimer = Array1::zeros(n_ext_atoms);
    cn_trimer.slice_mut(s![..n_real_i]).assign(&cn_i);
    cn_trimer
        .slice_mut(s![n_real_i..n_real_i + n_real_j])
        .assign(&cn_j);
    cn_trimer
        .slice_mut(s![n_real_i + n_real_j..n_real_atoms])
        .assign(&cn_k);
    // Ghost atoms get CN=0

    // S, H0, X
    let s = calc_overlap_matrix_parallel(&basis);
    let h0 = h0_xtb1_with_cn(&ext_atoms, s.view(), &basis, cn_trimer.view());
    let x = compute_s_inv_sqrt(s.view());

    // gamma_shell (from scratch since ghosts may be present)
    let gamma_shell = gamma_matrix_shell(gammafunction, &ext_atoms, &basis);

    // Electron count
    let n_elec_raw: usize = real_atoms_i
        .iter()
        .chain(real_atoms_j.iter())
        .chain(real_atoms_k.iter())
        .map(|a| a.n_elec)
        .sum();
    let n_elec = n_elec_raw - partial_bda_bonds.len() + partial_baa_bonds.len();

    // Reference density with hybrid adjustments
    let mut p_ref = create_density_ref(&basis, &ext_atoms);

    // Partial-BDA: P_ref -= |hybrid><hybrid|
    for bond in &partial_bda_bonds {
        let bda_local = if bond.bda_fragment == tri_i {
            bond.bda_global - frag_atom_range_i.start
        } else if bond.bda_fragment == tri_j {
            n_real_i + (bond.bda_global - frag_atom_range_j.start)
        } else {
            n_real_i + n_real_j + (bond.bda_global - frag_atom_range_k.start)
        };
        let (ao_start, nao) = get_bda_ao_range(&basis, bda_local);
        let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
        let hybrid = compute_rotated_sp3_xtb(&bond_vec);
        let nh = nao.min(hybrid.len());
        for ii in 0..nh {
            for jj in 0..nh {
                p_ref[[ao_start + ii, ao_start + jj]] -= hybrid[ii] * hybrid[jj];
            }
        }
    }

    // Ghost atoms: P_ref = |hybrid><hybrid|
    for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
        let ghost_local = n_real_atoms + ghost_idx;
        let (ao_start, nao) = get_bda_ao_range(&basis, ghost_local);
        let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
        let hybrid = compute_rotated_sp3_xtb(&bond_vec);
        let nh = nao.min(hybrid.len());
        for ii in 0..nao {
            for jj in 0..nao {
                p_ref[[ao_start + ii, ao_start + jj]] = 0.0;
            }
        }
        for ii in 0..nh {
            for jj in 0..nh {
                p_ref[[ao_start + ii, ao_start + jj]] = hybrid[ii] * hybrid[jj];
            }
        }
    }

    // HOP projector
    let p_hop = if partial_bda_bonds.is_empty() && partial_baa_bonds.is_empty() {
        None
    } else {
        let mut p_hop_total = Array2::<f64>::zeros([n_ext_orbs, n_ext_orbs]);

        for bond in &partial_bda_bonds {
            let bda_pos = atoms[bond.bda_global].xyz;
            let baa_pos = atoms[bond.baa_global].xyz;
            let bond_vec = baa_pos - bda_pos;
            let rotated_sp3 = compute_rotated_sp3_xtb(&bond_vec);
            let dd = compute_bda_dd_xtb(rotated_sp3.view(), HOP_SHIFT);

            let bda_local = if bond.bda_fragment == tri_i {
                bond.bda_global - frag_atom_range_i.start
            } else if bond.bda_fragment == tri_j {
                n_real_i + (bond.bda_global - frag_atom_range_j.start)
            } else {
                n_real_i + n_real_j + (bond.bda_global - frag_atom_range_k.start)
            };
            let (ao_start, nao) = get_bda_ao_range(&basis, bda_local);

            let p_hop_bond = if nao == dd.nrows() {
                compute_hop_projector(s.view(), dd.view(), ao_start, nao)
            } else {
                let mut dd_full = Array2::<f64>::zeros([nao, nao]);
                let dd_size = dd.nrows().min(nao);
                dd_full
                    .slice_mut(s![..dd_size, ..dd_size])
                    .assign(&dd.slice(s![..dd_size, ..dd_size]));
                compute_hop_projector(s.view(), dd_full.view(), ao_start, nao)
            };
            p_hop_total += &p_hop_bond;
        }

        for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
            let bda_pos = atoms[bond.bda_global].xyz;
            let baa_pos = atoms[bond.baa_global].xyz;
            let bond_vec = baa_pos - bda_pos;
            let dd_ghost = compute_ghost_nonbond_dd_xtb(&bond_vec, HOP_SHIFT);

            let ghost_local = n_real_atoms + ghost_idx;
            let (ao_start, nao) = get_bda_ao_range(&basis, ghost_local);

            let p_hop_bond = if nao == dd_ghost.nrows() {
                compute_hop_projector(s.view(), dd_ghost.view(), ao_start, nao)
            } else {
                let mut dd_full = Array2::<f64>::zeros([nao, nao]);
                let dd_size = dd_ghost.nrows().min(nao);
                dd_full
                    .slice_mut(s![..dd_size, ..dd_size])
                    .assign(&dd_ghost.slice(s![..dd_size, ..dd_size]));
                compute_hop_projector(s.view(), dd_full.view(), ao_start, nao)
            };
            p_hop_total += &p_hop_bond;
        }

        Some(p_hop_total)
    };

    // Compute external ESP for trimer (shell-level).
    // For real atoms in I: esp_q_shell_I - gamma_ext[I,J]·dq_J - gamma_ext[I,K]·dq_K
    // For real atoms in J: esp_q_shell_J - gamma_ext[J,I]·dq_I - gamma_ext[J,K]·dq_K
    // For real atoms in K: esp_q_shell_K - gamma_ext[K,I]·dq_I - gamma_ext[K,J]·dq_J
    let gamma_shell_ext = &hop_data.gamma_shell_ext;
    let shell_range_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
    let shell_range_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);
    let shell_range_k = get_frag_shell_range(&hop_data.ext_basis, &fi_k.ext_range);

    let dq_shell_ext_i =
        hop_data.dq_shell_ext.slice(s![shell_range_i.start..shell_range_i.end]);
    let dq_shell_ext_j =
        hop_data.dq_shell_ext.slice(s![shell_range_j.start..shell_range_j.end]);
    let dq_shell_ext_k =
        hop_data.dq_shell_ext.slice(s![shell_range_k.start..shell_range_k.end]);

    // ESP for I: subtract J's and K's contributions
    let gamma_ij = gamma_shell_ext.slice(s![
        shell_range_i.start..shell_range_i.end,
        shell_range_j.start..shell_range_j.end
    ]);
    let gamma_ik = gamma_shell_ext.slice(s![
        shell_range_i.start..shell_range_i.end,
        shell_range_k.start..shell_range_k.end
    ]);
    let esp_shell_i: Array1<f64> =
        &esp_q_shell_i - &gamma_ij.dot(&dq_shell_ext_j) - &gamma_ik.dot(&dq_shell_ext_k);

    // ESP for J: subtract I's and K's contributions
    let gamma_ji = gamma_shell_ext.slice(s![
        shell_range_j.start..shell_range_j.end,
        shell_range_i.start..shell_range_i.end
    ]);
    let gamma_jk = gamma_shell_ext.slice(s![
        shell_range_j.start..shell_range_j.end,
        shell_range_k.start..shell_range_k.end
    ]);
    let esp_shell_j: Array1<f64> =
        &esp_q_shell_j - &gamma_ji.dot(&dq_shell_ext_i) - &gamma_jk.dot(&dq_shell_ext_k);

    // ESP for K: subtract I's and J's contributions
    let gamma_ki = gamma_shell_ext.slice(s![
        shell_range_k.start..shell_range_k.end,
        shell_range_i.start..shell_range_i.end
    ]);
    let gamma_kj = gamma_shell_ext.slice(s![
        shell_range_k.start..shell_range_k.end,
        shell_range_j.start..shell_range_j.end
    ]);
    let esp_shell_k: Array1<f64> =
        &esp_q_shell_k - &gamma_ki.dot(&dq_shell_ext_i) - &gamma_kj.dot(&dq_shell_ext_j);

    // Convert shell→AO for real atoms
    let mut esp_ao = Array1::zeros(n_ext_orbs);
    esp_ao
        .slice_mut(s![..n_real_orbs_i])
        .assign(&shell_to_ao_values(
            &real_basis_i,
            n_real_orbs_i,
            esp_shell_i.slice(s![..n_real_shells_i]),
        ));
    esp_ao
        .slice_mut(s![n_real_orbs_i..n_real_orbs_i + n_real_orbs_j])
        .assign(&shell_to_ao_values(
            &real_basis_j,
            n_real_orbs_j,
            esp_shell_j.slice(s![..n_real_shells_j]),
        ));
    esp_ao
        .slice_mut(s![n_real_orbs_i + n_real_orbs_j..n_real_orbs])
        .assign(&shell_to_ao_values(
            &real_basis_k,
            n_real_orbs_k,
            esp_shell_k.slice(s![..n_real_shells_k]),
        ));

    // Ghost atoms: ESP from all fragments NOT in the trimer
    for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
        let bda_frag = bond.bda_fragment;
        let bda_frag_atom_start = hop_data.monomer_indices[bda_frag][0];
        let bda_local_in_frag = bond.bda_global - bda_frag_atom_start;
        let bda_ext_idx = hop_data.frag_info[bda_frag].ext_range.start + bda_local_in_frag;

        let bda_shells: Vec<usize> = hop_data
            .ext_basis
            .shells
            .iter()
            .enumerate()
            .filter(|(_, sh)| sh.atom_index == bda_ext_idx)
            .map(|(idx, _)| idx)
            .collect();

        let ghost_local = n_real_atoms + ghost_idx;
        let ghost_shells: Vec<usize> = basis
            .shells
            .iter()
            .enumerate()
            .filter(|(_, sh)| sh.atom_index == ghost_local)
            .map(|(idx, _)| idx)
            .collect();

        for (&bda_sh, &ghost_sh) in bda_shells.iter().zip(ghost_shells.iter()) {
            let full_esp: f64 = gamma_shell_ext.row(bda_sh).dot(&hop_data.dq_shell_ext);
            let esp_from_i: f64 = gamma_shell_ext
                .slice(s![bda_sh, shell_range_i.start..shell_range_i.end])
                .dot(&dq_shell_ext_i);
            let esp_from_j: f64 = gamma_shell_ext
                .slice(s![bda_sh, shell_range_j.start..shell_range_j.end])
                .dot(&dq_shell_ext_j);
            let esp_from_k: f64 = gamma_shell_ext
                .slice(s![bda_sh, shell_range_k.start..shell_range_k.end])
                .dot(&dq_shell_ext_k);
            let ghost_esp = full_esp - esp_from_i - esp_from_j - esp_from_k;

            let shell_ref = &basis.shells[ghost_sh];
            for ao in shell_ref.sph_start..shell_ref.sph_end {
                esp_ao[ao] = ghost_esp;
            }
        }
    }

    // Convert ESP to AO matrix and scale by S*0.5
    let v_esp = aovec_to_aomat(esp_ao.view(), n_ext_orbs) * &s * 0.5;

    // Initial dq from monomers
    let mut dq = Array1::zeros(n_ext_atoms);
    dq.slice_mut(s![..n_real_i])
        .assign(&mono_dq_i.slice(s![..n_real_i]));
    dq.slice_mut(s![n_real_i..n_real_i + n_real_j])
        .assign(&mono_dq_j.slice(s![..n_real_j]));
    dq.slice_mut(s![n_real_i + n_real_j..n_real_atoms])
        .assign(&mono_dq_k.slice(s![..n_real_k]));

    let dq_ao = Array1::zeros(n_ext_orbs);
    let dq_shell = Array1::zeros(n_ext_shells);

    // Compute initial dq_shell from monomer shell charges
    let n_real_shells = n_real_shells_i + n_real_shells_j + n_real_shells_k;
    let mut initial_dq_shell = Array1::zeros(n_ext_shells);
    initial_dq_shell
        .slice_mut(s![..n_real_shells_i])
        .assign(&mono_dq_shell_i.slice(s![..n_real_shells_i]));
    initial_dq_shell
        .slice_mut(s![n_real_shells_i..n_real_shells_i + n_real_shells_j])
        .assign(&mono_dq_shell_j.slice(s![..n_real_shells_j]));
    initial_dq_shell
        .slice_mut(s![n_real_shells_i + n_real_shells_j..n_real_shells])
        .assign(&mono_dq_shell_k.slice(s![..n_real_shells_k]));

    // Hubbard derivatives
    let mut hubbard_derivatives = Array1::zeros(n_ext_atoms);
    for (val, atom) in hubbard_derivatives.iter_mut().zip(ext_atoms.iter()) {
        *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
    }

    // ZREF/QREF for trimer atoms
    let mut tri_zref = Array1::zeros(n_ext_atoms);
    let mut tri_qref = Array1::zeros(n_ext_atoms);

    // Copy from monomer ZREF/QREF for real atoms
    tri_zref
        .slice_mut(s![..n_real_i])
        .assign(&hop_data.zref.slice(s![fi_i.ext_range.start..fi_i.ext_range.start + n_real_i]));
    tri_qref
        .slice_mut(s![..n_real_i])
        .assign(&hop_data.qref.slice(s![fi_i.ext_range.start..fi_i.ext_range.start + n_real_i]));
    tri_zref
        .slice_mut(s![n_real_i..n_real_i + n_real_j])
        .assign(&hop_data.zref.slice(s![fi_j.ext_range.start..fi_j.ext_range.start + n_real_j]));
    tri_qref
        .slice_mut(s![n_real_i..n_real_i + n_real_j])
        .assign(&hop_data.qref.slice(s![fi_j.ext_range.start..fi_j.ext_range.start + n_real_j]));
    tri_zref
        .slice_mut(s![n_real_i + n_real_j..n_real_atoms])
        .assign(&hop_data.zref.slice(s![fi_k.ext_range.start..fi_k.ext_range.start + n_real_k]));
    tri_qref
        .slice_mut(s![n_real_i + n_real_j..n_real_atoms])
        .assign(&hop_data.qref.slice(s![fi_k.ext_range.start..fi_k.ext_range.start + n_real_k]));

    // Restore ZREF for healed bonds
    for bond in &hop_data.detached_bonds {
        let bda_in_trimer =
            bond.bda_fragment == tri_i || bond.bda_fragment == tri_j || bond.bda_fragment == tri_k;
        let baa_in_trimer =
            bond.baa_fragment == tri_i || bond.baa_fragment == tri_j || bond.baa_fragment == tri_k;
        if bda_in_trimer && baa_in_trimer {
            let bda_local = if bond.bda_fragment == tri_i {
                bond.bda_global - frag_atom_range_i.start
            } else if bond.bda_fragment == tri_j {
                n_real_i + (bond.bda_global - frag_atom_range_j.start)
            } else {
                n_real_i + n_real_j + (bond.bda_global - frag_atom_range_k.start)
            };
            tri_zref[bda_local] += 1.0;
        }
    }

    // Ghost ZREF/QREF
    for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
        let ghost_local = n_real_atoms + ghost_idx;
        tri_zref[ghost_local] = 1.0;
        tri_qref[ghost_local] = get_qref(&atoms[bond.bda_global]);
    }

    let p = p_ref.clone();

    XtbTrimerHopScc {
        i: tri_i,
        j: tri_j,
        k: tri_k,
        ext_atoms,
        basis,
        n_ext_atoms,
        n_real_i,
        n_real_j,
        n_real_k,
        n_real_atoms,
        n_ghost_atoms,
        n_ext_orbs,
        n_real_orbs,
        n_real_orbs_i,
        n_real_orbs_j,
        n_ext_shells,
        n_real_shells_i,
        n_real_shells_j,
        n_real_shells_k,
        n_elec,
        h0,
        s,
        x,
        gamma_shell,
        v_esp,
        p_hop,
        p_ref,
        p,
        dq,
        dq_ao,
        dq_shell,
        hubbard_derivatives,
        last_energy: 0.0,
        orbs: None,
        orbe: None,
        f: vec![0.0; n_ext_orbs],
        zref: tri_zref,
        qref: tri_qref,
        initial_dq_shell,
        delta_dq_shell_real: Array1::zeros(n_real_shells),
    }
}

/// Run trimer SCC to convergence (xTB version).
/// Returns total trimer energy (electronic + repulsive).
pub fn run_trimer_scc_hop_xtb(
    tri: &mut XtbTrimerHopScc,
    max_iter: usize,
    temperature: f64,
    scf_charge_conv: f64,
    scf_energy_conv: f64,
    broyden_config: &dialect_config::settings::BroydenConfig,
) -> f64 {
    let mut broyden_mixer = BroydenMixerNew::from_config(tri.n_ext_shells, broyden_config);

    let mut h_esp: Array2<f64> = &tri.h0 + &tri.v_esp;
    if let Some(ref p_hop) = tri.p_hop {
        h_esp = &h_esp + p_hop;
    }

    for iter in 0..max_iter {
        // Shell-level Coulomb
        let v_shell = tri.gamma_shell.dot(&tri.dq_shell);
        let v_ao = shell_to_ao_values(&tri.basis, tri.n_ext_orbs, v_shell.view());
        let h_coul = outer_sum(v_ao.view()) * &tri.s * 0.5;

        // Third-order
        let h_third = coul_third_order_hamiltonian(
            tri.hubbard_derivatives.view(),
            tri.dq.view(),
            &tri.basis,
        ) * &tri.s
            * 0.5;

        let h_full: Array2<f64> = &h_esp + &h_coul - &h_third;

        // Loewdin
        let h_ortho = tri.x.t().dot(&h_full).dot(&tri.x);
        let (orbe, orbs_prime) = dsyevd_eigh(h_ortho.view());
        let orbs = tri.x.dot(&orbs_prime);

        // Fermi occupation
        let (_, f) = fermi_occupation(orbe.view(), tri.n_elec, temperature);

        // Density
        let p = density_matrix(orbs.view(), &f);

        // Mulliken charges
        let dq_ao_new = mulliken_aowise_diff(p.view(), tri.p_ref.view(), tri.s.view());
        let dq_shell_new = ao_to_shell_charges(&tri.basis, dq_ao_new.view());

        let delta_dq_shell: Array1<f64> = &dq_shell_new - &tri.dq_shell;

        let dq_ao_out = shell_to_ao_charges(&tri.basis, tri.n_ext_orbs, dq_shell_new.view());
        let dq_atom_out =
            mulliken_atomwise_from_ao_xtb(&tri.basis, tri.n_ext_atoms, dq_ao_out.view());

        // Energy
        let entropy = compute_total_entropy(orbe.view(), tri.n_elec, temperature);
        let scf_energy = get_electronic_energy_xtb_shell(
            p.view(),
            tri.h0.view(),
            dq_atom_out.view(),
            dq_shell_new.view(),
            tri.gamma_shell.view(),
            tri.hubbard_derivatives.view(),
        ) + entropy;

        // Convergence
        let diff_dq = dq_atom_out.root_mean_sq_err(&tri.dq).unwrap();
        let converged = diff_dq < scf_charge_conv
            && (tri.last_energy - scf_energy).abs() < scf_energy_conv;

        tri.last_energy = scf_energy;
        tri.p = p;

        // Broyden mixing at shell level
        tri.dq_shell = broyden_mixer.next(&tri.dq_shell, &delta_dq_shell);
        tri.dq_ao = shell_to_ao_charges(&tri.basis, tri.n_ext_orbs, tri.dq_shell.view());
        tri.dq = mulliken_atomwise_from_ao_xtb(&tri.basis, tri.n_ext_atoms, tri.dq_ao.view());

        if converged {
            // Repulsive energy with ZREF/QREF scaling
            let e_rep = calculate_repulsive_energy_xtb_scaled(
                &tri.ext_atoms,
                tri.zref.view(),
                tri.qref.view(),
            );
            let total = scf_energy + e_rep;

            // Compute delta_dq_shell_real for embedding
            let n_real_shells =
                tri.n_real_shells_i + tri.n_real_shells_j + tri.n_real_shells_k;
            tri.delta_dq_shell_real = &tri.dq_shell.slice(s![..n_real_shells])
                - &tri.initial_dq_shell.slice(s![..n_real_shells]);

            tri.orbs = Some(orbs);
            tri.orbe = Some(orbe);
            tri.f = f;
            tri.last_energy = total;
            return total;
        }

        if iter == max_iter - 1 {
            panic!(
                "HOP Trimer SCC ({},{},{}) did NOT converge after {} iterations!",
                tri.i, tri.j, tri.k, max_iter
            );
        }
    }
    unreachable!()
}
