//! Pair SCC with HOP for FMO-xTB.
//!
//! Pairs can have:
//! - **Healed bonds**: both BDA and BAA in pair → no ghost, no HOP, no n_elec change
//! - **Partial-BDA bonds**: BDA in pair, BAA outside → HOP on BDA, ZREF-=1, n_elec-=1
//! - **Partial-BAA bonds**: BAA in pair, BDA outside → ghost at BDA position, n_elec+=1

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

/// State for one pair's HOP SCC calculation (xTB version).
#[derive(Debug, Clone)]
pub struct XtbPairHopScc {
    /// Monomer indices (i, j)
    pub i: usize,
    pub j: usize,
    /// Extended pair atoms: [real_I, real_J, partial_ghost_atoms]
    pub ext_atoms: Vec<XtbAtom>,
    /// Extended basis
    pub basis: Basis,
    pub n_ext_atoms: usize,
    pub n_real_i: usize,
    pub n_real_j: usize,
    pub n_real_atoms: usize,
    pub n_ghost_atoms: usize,
    pub n_ext_orbs: usize,
    pub n_real_orbs: usize,
    pub n_real_orbs_i: usize,
    pub n_ext_shells: usize,
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
    /// Initial dq_shell from monomer charges (saved at prepare time for embedding)
    pub initial_dq_shell: Array1<f64>,
    /// delta_dq_shell truncated to real shells only (for trimer embedding subtraction)
    pub delta_dq_shell_real: Array1<f64>,
    /// Number of real shells for fragment I
    pub n_real_shells_i: usize,
    /// Number of real shells for fragment J
    pub n_real_shells_j: usize,
}

/// Get QREF for an XtbAtom
fn get_qref(atom: &XtbAtom) -> f64 {
    let idx = atom.number as usize - 1;
    REFERENCE_OCCUPATION[idx][0] + REFERENCE_OCCUPATION[idx][1] + REFERENCE_OCCUPATION[idx][2]
}

/// Prepare a pair for HOP SCC (xTB version).
pub fn prepare_pair_hop_xtb(
    pair_i: usize,
    pair_j: usize,
    hop_data: &XtbHopData,
    mono_dq_i: ArrayView1<f64>,
    mono_dq_j: ArrayView1<f64>,
    mono_dq_shell_i: ArrayView1<f64>,
    mono_dq_shell_j: ArrayView1<f64>,
    esp_q_shell_i: ArrayView1<f64>,
    esp_q_shell_j: ArrayView1<f64>,
    fi_i: &XtbHopFragInfo,
    fi_j: &XtbHopFragInfo,
    gammafunction: &crate::scc::gamma_matrix::XtbGammaFunction,
    atoms: &[XtbAtom],
    frag_atom_range_i: &std::ops::Range<usize>,
    frag_atom_range_j: &std::ops::Range<usize>,
    cn_ext: ArrayView1<f64>,
) -> XtbPairHopScc {
    let real_atoms_i: Vec<XtbAtom> = atoms[frag_atom_range_i.clone()].to_vec();
    let real_atoms_j: Vec<XtbAtom> = atoms[frag_atom_range_j.clone()].to_vec();
    let n_real_i = real_atoms_i.len();
    let n_real_j = real_atoms_j.len();
    let n_real_atoms = n_real_i + n_real_j;

    let real_basis_i = create_basis_set(&real_atoms_i);
    let real_basis_j = create_basis_set(&real_atoms_j);
    let n_real_orbs_i = real_basis_i.nbas;
    let n_real_orbs_j = real_basis_j.nbas;
    let n_real_orbs = n_real_orbs_i + n_real_orbs_j;
    let n_real_shells_i = real_basis_i.shells.len();
    let n_real_shells_j = real_basis_j.shells.len();

    // Classify bonds
    let mut partial_bda_bonds: Vec<&DetachedBond> = Vec::new();
    let mut partial_baa_bonds: Vec<&DetachedBond> = Vec::new();
    let mut ghost_atoms: Vec<XtbAtom> = Vec::new();

    for bond in &hop_data.detached_bonds {
        let bda_in_pair = bond.bda_fragment == pair_i || bond.bda_fragment == pair_j;
        let baa_in_pair = bond.baa_fragment == pair_i || bond.baa_fragment == pair_j;

        if bda_in_pair && baa_in_pair {
            // Healed
        } else if bda_in_pair && !baa_in_pair {
            partial_bda_bonds.push(bond);
        } else if !bda_in_pair && baa_in_pair {
            partial_baa_bonds.push(bond);
            let bda_pos = atoms[bond.bda_global].xyz;
            let bda_atom = &atoms[bond.bda_global];
            ghost_atoms.push(create_xtb_ghost_atom(bda_pos, bda_atom));
        }
    }

    let n_ghost_atoms = ghost_atoms.len();

    // Build extended pair atoms
    let mut ext_atoms: Vec<XtbAtom> = Vec::with_capacity(n_real_atoms + n_ghost_atoms);
    ext_atoms.extend(real_atoms_i.iter().cloned());
    ext_atoms.extend(real_atoms_j.iter().cloned());
    ext_atoms.extend(ghost_atoms.iter().cloned());

    let n_ext_atoms = ext_atoms.len();
    let basis = create_basis_set(&ext_atoms);
    let n_ext_orbs = basis.nbas;
    let n_ext_shells = basis.shells.len();

    // CN for pair atoms
    let cn_i = cn_ext.slice(s![fi_i.ext_range.start..fi_i.ext_range.start + n_real_i]);
    let cn_j = cn_ext.slice(s![fi_j.ext_range.start..fi_j.ext_range.start + n_real_j]);
    let mut cn_pair = Array1::zeros(n_ext_atoms);
    cn_pair.slice_mut(s![..n_real_i]).assign(&cn_i);
    cn_pair
        .slice_mut(s![n_real_i..n_real_atoms])
        .assign(&cn_j);
    // Ghost atoms get CN=0

    // S, H0, X
    let s = calc_overlap_matrix_parallel(&basis);
    let h0 = h0_xtb1_with_cn(&ext_atoms, s.view(), &basis, cn_pair.view());
    let x = compute_s_inv_sqrt(s.view());

    // gamma_shell
    let gamma_shell = gamma_matrix_shell(gammafunction, &ext_atoms, &basis);

    // Electron count
    let n_elec_raw: usize = real_atoms_i
        .iter()
        .chain(real_atoms_j.iter())
        .map(|a| a.n_elec)
        .sum();
    let n_elec = n_elec_raw - partial_bda_bonds.len() + partial_baa_bonds.len();

    // Reference density with hybrid adjustments
    let mut p_ref = create_density_ref(&basis, &ext_atoms);

    // Partial-BDA: P_ref -= |hybrid><hybrid|
    for bond in &partial_bda_bonds {
        let bda_local = if bond.bda_fragment == pair_i {
            bond.bda_global - frag_atom_range_i.start
        } else {
            n_real_i + (bond.bda_global - frag_atom_range_j.start)
        };
        let (ao_start, nao) = get_bda_ao_range(&basis, bda_local);
        let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
        let hybrid = compute_rotated_sp3_xtb(&bond_vec);
        let nh = nao.min(hybrid.len());
        for i in 0..nh {
            for j in 0..nh {
                p_ref[[ao_start + i, ao_start + j]] -= hybrid[i] * hybrid[j];
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
        for i in 0..nao {
            for j in 0..nao {
                p_ref[[ao_start + i, ao_start + j]] = 0.0;
            }
        }
        for i in 0..nh {
            for j in 0..nh {
                p_ref[[ao_start + i, ao_start + j]] = hybrid[i] * hybrid[j];
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

            let bda_local = if bond.bda_fragment == pair_i {
                bond.bda_global - frag_atom_range_i.start
            } else {
                n_real_i + (bond.bda_global - frag_atom_range_j.start)
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

    // Compute external ESP for pair (shell-level).
    // For real atoms in I: esp_q_shell_I - gamma_shell_ext[shell_I, shell_J] . dq_shell_J
    // For real atoms in J: esp_q_shell_J - gamma_shell_ext[shell_J, shell_I] . dq_shell_I
    let gamma_shell_ext = &hop_data.gamma_shell_ext;
    let shell_range_i = get_frag_shell_range(&hop_data.ext_basis, &fi_i.ext_range);
    let shell_range_j = get_frag_shell_range(&hop_data.ext_basis, &fi_j.ext_range);

    let gamma_ij_shell = gamma_shell_ext.slice(s![
        shell_range_i.start..shell_range_i.end,
        shell_range_j.start..shell_range_j.end
    ]);
    let gamma_ji_shell = gamma_shell_ext.slice(s![
        shell_range_j.start..shell_range_j.end,
        shell_range_i.start..shell_range_i.end
    ]);

    // Real-shell dq from monomer (first n_real_shells elements)
    let dq_shell_ext_j = hop_data.dq_shell_ext.slice(s![shell_range_j.start..shell_range_j.end]);
    let dq_shell_ext_i = hop_data.dq_shell_ext.slice(s![shell_range_i.start..shell_range_i.end]);

    // ESP for I real shells: subtract J's contribution
    let esp_shell_i_full: Array1<f64> = &esp_q_shell_i - &gamma_ij_shell.dot(&dq_shell_ext_j);
    // ESP for J real shells: subtract I's contribution
    let esp_shell_j_full: Array1<f64> = &esp_q_shell_j - &gamma_ji_shell.dot(&dq_shell_ext_i);

    // Convert shell→AO for real atoms
    let mut esp_ao = Array1::zeros(n_ext_orbs);
    esp_ao
        .slice_mut(s![..n_real_orbs_i])
        .assign(&shell_to_ao_values(
            &real_basis_i,
            n_real_orbs_i,
            esp_shell_i_full.slice(s![..n_real_shells_i]),
        ));
    esp_ao
        .slice_mut(s![n_real_orbs_i..n_real_orbs])
        .assign(&shell_to_ao_values(
            &real_basis_j,
            n_real_orbs_j,
            esp_shell_j_full.slice(s![..n_real_shells_j]),
        ));

    // Ghost atoms: ESP from all fragments NOT in the pair
    for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
        // Find BDA's extended atom index (BDA is in an external fragment)
        let bda_frag = bond.bda_fragment;
        let bda_frag_atom_start = hop_data.monomer_indices[bda_frag][0];
        let bda_local_in_frag = bond.bda_global - bda_frag_atom_start;
        let bda_ext_idx = hop_data.frag_info[bda_frag].ext_range.start + bda_local_in_frag;

        // Find BDA's shell range in ext_basis for shell-level ESP
        let bda_shells: Vec<usize> = hop_data
            .ext_basis
            .shells
            .iter()
            .enumerate()
            .filter(|(_, sh)| sh.atom_index == bda_ext_idx)
            .map(|(idx, _)| idx)
            .collect();

        // Ghost shell range in pair basis
        let ghost_local = n_real_atoms + ghost_idx;
        let ghost_shells: Vec<usize> = basis
            .shells
            .iter()
            .enumerate()
            .filter(|(_, sh)| sh.atom_index == ghost_local)
            .map(|(idx, _)| idx)
            .collect();

        // For each ghost shell: ESP = gamma_ext[bda_shell, :] . dq_shell_ext
        //   - gamma_ext[bda_shell, shell_I] . dq_I - gamma_ext[bda_shell, shell_J] . dq_J
        for (&bda_sh, &ghost_sh) in bda_shells.iter().zip(ghost_shells.iter()) {
            let full_esp: f64 = gamma_shell_ext.row(bda_sh).dot(&hop_data.dq_shell_ext);
            let esp_from_i: f64 = gamma_shell_ext
                .slice(s![bda_sh, shell_range_i.start..shell_range_i.end])
                .dot(&dq_shell_ext_i);
            let esp_from_j: f64 = gamma_shell_ext
                .slice(s![bda_sh, shell_range_j.start..shell_range_j.end])
                .dot(&dq_shell_ext_j);
            let ghost_esp = full_esp - esp_from_i - esp_from_j;

            // Expand to AO
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
    dq.slice_mut(s![n_real_i..n_real_atoms])
        .assign(&mono_dq_j.slice(s![..n_real_j]));

    let mut dq_ao = Array1::zeros(n_ext_orbs);
    let mut dq_shell = Array1::zeros(n_ext_shells);

    // Compute initial dq_shell from monomer shell charges (for delta_dq_shell_real)
    let mut initial_dq_shell = Array1::zeros(n_ext_shells);
    initial_dq_shell
        .slice_mut(s![..n_real_shells_i])
        .assign(&mono_dq_shell_i.slice(s![..n_real_shells_i]));
    initial_dq_shell
        .slice_mut(s![n_real_shells_i..n_real_shells_i + n_real_shells_j])
        .assign(&mono_dq_shell_j.slice(s![..n_real_shells_j]));
    // Ghost shells remain zero

    // Hubbard derivatives
    let mut hubbard_derivatives = Array1::zeros(n_ext_atoms);
    for (val, atom) in hubbard_derivatives.iter_mut().zip(ext_atoms.iter()) {
        *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
    }

    // ZREF/QREF for pair atoms
    let mut pair_zref = Array1::zeros(n_ext_atoms);
    let mut pair_qref = Array1::zeros(n_ext_atoms);

    // Copy from monomer ZREF/QREF for real atoms
    pair_zref
        .slice_mut(s![..n_real_i])
        .assign(&hop_data.zref.slice(s![fi_i.ext_range.start..fi_i.ext_range.start + n_real_i]));
    pair_qref
        .slice_mut(s![..n_real_i])
        .assign(&hop_data.qref.slice(s![fi_i.ext_range.start..fi_i.ext_range.start + n_real_i]));
    pair_zref
        .slice_mut(s![n_real_i..n_real_atoms])
        .assign(&hop_data.zref.slice(s![fi_j.ext_range.start..fi_j.ext_range.start + n_real_j]));
    pair_qref
        .slice_mut(s![n_real_i..n_real_atoms])
        .assign(&hop_data.qref.slice(s![fi_j.ext_range.start..fi_j.ext_range.start + n_real_j]));

    // Restore ZREF for healed bonds
    for bond in &hop_data.detached_bonds {
        let bda_in_pair = bond.bda_fragment == pair_i || bond.bda_fragment == pair_j;
        let baa_in_pair = bond.baa_fragment == pair_i || bond.baa_fragment == pair_j;
        if bda_in_pair && baa_in_pair {
            let bda_local = if bond.bda_fragment == pair_i {
                bond.bda_global - frag_atom_range_i.start
            } else {
                n_real_i + (bond.bda_global - frag_atom_range_j.start)
            };
            pair_zref[bda_local] += 1.0;
        }
    }

    // Ghost ZREF/QREF
    for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
        let ghost_local = n_real_atoms + ghost_idx;
        pair_zref[ghost_local] = 1.0;
        pair_qref[ghost_local] = get_qref(&atoms[bond.bda_global]);
    }

    let p = p_ref.clone();

    let n_real_shells = n_real_shells_i + n_real_shells_j;

    XtbPairHopScc {
        i: pair_i,
        j: pair_j,
        ext_atoms,
        basis,
        n_ext_atoms,
        n_real_i,
        n_real_j,
        n_real_atoms,
        n_ghost_atoms,
        n_ext_orbs,
        n_real_orbs,
        n_real_orbs_i,
        n_ext_shells,
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
        zref: pair_zref,
        qref: pair_qref,
        initial_dq_shell,
        delta_dq_shell_real: Array1::zeros(n_real_shells),
        n_real_shells_i,
        n_real_shells_j,
    }
}

/// Run pair SCC to convergence (xTB version).
/// Returns total pair energy (electronic + repulsive).
pub fn run_pair_scc_hop_xtb(pair: &mut XtbPairHopScc, max_iter: usize, temperature: f64, scf_charge_conv: f64, scf_energy_conv: f64, broyden_config: &dialect_config::settings::BroydenConfig) -> f64 {
    let mut broyden_mixer = BroydenMixerNew::from_config(pair.n_ext_shells, broyden_config);

    let mut h_esp: Array2<f64> = &pair.h0 + &pair.v_esp;
    if let Some(ref p_hop) = pair.p_hop {
        h_esp = &h_esp + p_hop;
    }

    for iter in 0..max_iter {
        // Shell-level Coulomb
        let v_shell = pair.gamma_shell.dot(&pair.dq_shell);
        let v_ao = shell_to_ao_values(&pair.basis, pair.n_ext_orbs, v_shell.view());
        let h_coul = outer_sum(v_ao.view()) * &pair.s * 0.5;

        // Third-order
        let h_third = coul_third_order_hamiltonian(
            pair.hubbard_derivatives.view(),
            pair.dq.view(),
            &pair.basis,
        ) * &pair.s
            * 0.5;

        let h_full: Array2<f64> = &h_esp + &h_coul - &h_third;

        // Loewdin
        let h_ortho = pair.x.t().dot(&h_full).dot(&pair.x);
        let (orbe, orbs_prime) = dsyevd_eigh(h_ortho.view());
        let orbs = pair.x.dot(&orbs_prime);

        // Fermi occupation
        let (_, f) = fermi_occupation(orbe.view(), pair.n_elec, temperature);

        // Density
        let p = density_matrix(orbs.view(), &f);

        // Mulliken charges
        let dq_ao_new = mulliken_aowise_diff(p.view(), pair.p_ref.view(), pair.s.view());
        let dq_shell_new = ao_to_shell_charges(&pair.basis, dq_ao_new.view());

        let delta_dq_shell: Array1<f64> = &dq_shell_new - &pair.dq_shell;

        let dq_ao_out = shell_to_ao_charges(&pair.basis, pair.n_ext_orbs, dq_shell_new.view());
        let dq_atom_out =
            mulliken_atomwise_from_ao_xtb(&pair.basis, pair.n_ext_atoms, dq_ao_out.view());

        // Energy
        let entropy = compute_total_entropy(orbe.view(), pair.n_elec, temperature);
        let mut scf_energy = get_electronic_energy_xtb_shell(
            p.view(),
            pair.h0.view(),
            dq_atom_out.view(),
            dq_shell_new.view(),
            pair.gamma_shell.view(),
            pair.hubbard_derivatives.view(),
        ) + entropy;
        // NOTE: Do NOT add Tr(P * V_HOP) to the energy.
        // V_HOP is a constraint potential, not part of the physical energy.

        // Convergence
        let diff_dq = dq_atom_out.root_mean_sq_err(&pair.dq).unwrap();
        let converged = diff_dq < scf_charge_conv
            && (pair.last_energy - scf_energy).abs() < scf_energy_conv;

        pair.last_energy = scf_energy;
        pair.p = p;

        // Broyden mixing at shell level
        pair.dq_shell = broyden_mixer.next(&pair.dq_shell, &delta_dq_shell);
        pair.dq_ao = shell_to_ao_charges(&pair.basis, pair.n_ext_orbs, pair.dq_shell.view());
        pair.dq = mulliken_atomwise_from_ao_xtb(&pair.basis, pair.n_ext_atoms, pair.dq_ao.view());

        if converged {
            // Repulsive energy with ZREF/QREF scaling
            let e_rep = calculate_repulsive_energy_xtb_scaled(
                &pair.ext_atoms,
                pair.zref.view(),
                pair.qref.view(),
            );
            let total = scf_energy + e_rep;

            // Compute delta_dq_shell_real for trimer embedding subtraction
            let n_real_shells = pair.n_real_shells_i + pair.n_real_shells_j;
            pair.delta_dq_shell_real = &pair.dq_shell.slice(s![..n_real_shells])
                - &pair.initial_dq_shell.slice(s![..n_real_shells]);

            pair.orbs = Some(orbs);
            pair.orbe = Some(orbe);
            pair.f = f;
            pair.last_energy = total;
            return total;
        }

        if iter == max_iter - 1 {
            panic!(
                "HOP Pair SCC ({},{}) did NOT converge after {} iterations!",
                pair.i, pair.j, max_iter
            );
        }
    }
    unreachable!()
}
