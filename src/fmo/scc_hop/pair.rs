//! Pair SCC with HOP for FMO-DFTB.
//!
//! Pairs can have:
//! - **Partial bonds**: BDA in pair, BAA outside → add ghost H, add HOP, reduce n_elec
//! - **Healed bonds**: both BDA and BAA in pair → no ghost, no HOP, no n_elec change

use crate::scc::mixer::BuildMixer;
use super::hop_data::{
    create_ghost_atom, compute_rotated_sp3_dftb, compute_bda_dd_matrix,
    compute_ghost_nonbond_dd, get_bda_ao_range_dftb, HopData, HopFragInfo,
};
use crate::fmo::scc::helpers::atomvec_to_aomat;
use crate::initialization::parameters::{RepulsivePotential, SlaterKoster};
use crate::initialization::Atom;
use crate::io::settings::MixConfig;
use crate::io::SccConfig;
use crate::scc::gamma_approximation::{gamma_ao_wise_hop, gamma_atomwise_hop, GammaFunction};
use crate::scc::h0_and_s::h0_and_s;
use crate::scc::lapack_eigh::{compute_s_inv_sqrt, dsyevd_eigh};
use crate::scc::mixer::BroydenMixerNew;
use crate::scc::mulliken::mulliken_atomwise;
use super::hop_data::get_repulsive_energy_scaled;
use crate::scc::{calc_exchange, density_matrix, density_matrix_ref, get_electronic_energy_new, lc_exact_exchange};
use dialect_xtb::hop::{
    compute_hop_projector, DetachedBond, HOP_SHIFT,
};
use ndarray::prelude::*;
use ndarray_stats::DeviationExt;

/// State for one pair's HOP SCC calculation.
#[derive(Debug, Clone)]
pub struct PairHopScc {
    /// Monomer indices (i, j)
    pub i: usize,
    pub j: usize,
    /// Extended pair atoms: [real_I, real_J, partial_ghost_atoms]
    pub ext_atoms: Vec<Atom>,
    /// Number of extended atoms
    pub n_ext_atoms: usize,
    /// Number of real atoms in monomer I
    pub n_real_i: usize,
    /// Number of real atoms in monomer J
    pub n_real_j: usize,
    /// Total real atoms
    pub n_real_atoms: usize,
    /// Number of ghost atoms (from partial bonds)
    pub n_ghost_atoms: usize,
    /// Number of extended orbitals
    pub n_ext_orbs: usize,
    /// Number of real orbitals (I + J)
    pub n_real_orbs: usize,
    /// Real orbitals for I
    pub n_real_orbs_i: usize,
    /// Adjusted electron count
    pub n_elec: usize,
    /// H0 matrix
    pub h0: Array2<f64>,
    /// Overlap matrix
    pub s: Array2<f64>,
    /// S^{-1/2}
    pub x: Array2<f64>,
    /// Local gamma for pair extended atoms
    pub gamma: Array2<f64>,
    /// External ESP from other fragments (AO matrix, precomputed)
    pub v_esp: Array2<f64>,
    /// HOP projector (None if no partial bonds)
    pub p_hop: Option<Array2<f64>>,
    /// Reference density
    pub p_ref: Array2<f64>,
    /// Current density
    pub p: Array2<f64>,
    /// Charge differences (extended)
    pub dq: Array1<f64>,
    /// Occupation
    pub f: Vec<f64>,
    /// Mixer
    pub mixer: BroydenMixerNew,
    /// Last SCF energy
    pub last_energy: f64,
    /// MO coefficients
    pub orbs: Option<Array2<f64>>,
    /// MO energies
    pub orbe: Option<Array1<f64>>,
    /// ZREF for extended pair atoms (for repulsive energy scaling)
    pub zref: Array1<f64>,
    /// QREF for extended pair atoms (for repulsive energy scaling)
    pub qref: Array1<f64>,
    /// LC gamma matrix in AO basis (shell-resolved), None if non-LC
    pub gamma_lr_ao: Option<Array2<f64>>,
    /// Difference density matrix P - P_ref, used for LC mixing
    pub delta_p: Option<Array2<f64>>,
}

/// Prepare a pair for HOP SCC.
///
/// 1. Classify bonds as partial (BDA in pair, BAA outside) or healed (both in pair)
/// 2. Build extended pair atoms: real_I + real_J + partial ghost atoms
/// 3. Compute H0, S, gamma, HOP projector, reference density
/// 4. Compute external ESP from supersystem charges
pub fn prepare_pair_hop(
    pair_i: usize,
    pair_j: usize,
    hop_data: &HopData,
    mono_dq_i: ArrayView1<f64>,
    mono_dq_j: ArrayView1<f64>,
    esp_q_i: ArrayView1<f64>,
    esp_q_j: ArrayView1<f64>,
    fi_i: &HopFragInfo,
    fi_j: &HopFragInfo,
    gammafunction: &GammaFunction,
    gammafunction_lc: &Option<GammaFunction>,
    slako: &SlaterKoster,
    atoms: &[Atom],
    frag_atom_range_i: &std::ops::Range<usize>,
    frag_atom_range_j: &std::ops::Range<usize>,
    broyden_config: &crate::io::settings::BroydenConfig,
) -> PairHopScc {
    // Real atoms
    let real_atoms_i: Vec<Atom> = atoms[frag_atom_range_i.clone()].to_vec();
    let real_atoms_j: Vec<Atom> = atoms[frag_atom_range_j.clone()].to_vec();
    let n_real_i = real_atoms_i.len();
    let n_real_j = real_atoms_j.len();
    let n_real_atoms = n_real_i + n_real_j;
    let n_real_orbs_i: usize = real_atoms_i.iter().map(|a| a.n_orbs).sum();
    let n_real_orbs_j: usize = real_atoms_j.iter().map(|a| a.n_orbs).sum();
    let n_real_orbs = n_real_orbs_i + n_real_orbs_j;


    // Classify bonds into three categories (reference convention):
    // 1. Healed: both BDA and BAA in pair → no ghost, no HOP, no n_elec change
    // 2. Partial-BDA: BDA in pair, BAA outside → HOP on BDA, ZREF(BDA)-=1, n_elec-=1, NO ghost
    // 3. Partial-BAA: BAA in pair, BDA outside → ghost at BDA position, n_elec+=1, NO HOP
    let mut partial_bda_bonds: Vec<&DetachedBond> = Vec::new(); // For HOP projector
    let mut partial_baa_bonds: Vec<&DetachedBond> = Vec::new(); // For ghost atoms
    let mut ghost_atoms: Vec<Atom> = Vec::new();

    for bond in &hop_data.detached_bonds {
        let bda_in_pair = bond.bda_fragment == pair_i || bond.bda_fragment == pair_j;
        let baa_in_pair = bond.baa_fragment == pair_i || bond.baa_fragment == pair_j;

        if bda_in_pair && baa_in_pair {
            // Healed: both in pair, no action
        } else if bda_in_pair && !baa_in_pair {
            // Partial-BDA: BDA in pair, BAA outside
            // → HOP on BDA, ZREF(BDA)-=1, n_elec-=1, NO ghost
            partial_bda_bonds.push(bond);
        } else if !bda_in_pair && baa_in_pair {
            // Partial-BAA: BAA in pair, BDA outside
            // → ghost at BDA position with BDA's element type
            // (ghost IAN = BDA's atomic number, at BDA coords)
            // → n_elec += 1 (ghost has 1 electron), NO HOP
            partial_baa_bonds.push(bond);
            let bda_pos = atoms[bond.bda_global].xyz;
            let bda_atom = &atoms[bond.bda_global];
            ghost_atoms.push(create_ghost_atom(bda_pos, bda_atom));
        }
        // Neither in pair: no action
    }

    let n_ghost_atoms = ghost_atoms.len();
    let n_ghost_orbs: usize = ghost_atoms.iter().map(|a| a.n_orbs).sum();
    let n_ext_orbs = n_real_orbs + n_ghost_orbs;
    let n_ext_atoms = n_real_atoms + n_ghost_atoms;

    // Build extended pair atoms: real_I + real_J + ghosts
    let mut ext_atoms: Vec<Atom> = Vec::with_capacity(n_ext_atoms);
    ext_atoms.extend(real_atoms_i.iter().cloned());
    ext_atoms.extend(real_atoms_j.iter().cloned());
    ext_atoms.extend(ghost_atoms.iter().cloned());

    // Compute H0 and S for extended pair
    let (s, h0) = h0_and_s(n_ext_orbs, &ext_atoms, slako);
    let x = compute_s_inv_sqrt(s.view());

    // Local gamma for pair extended atoms
    let gamma = gamma_atomwise_hop(gammafunction, &ext_atoms, n_ext_atoms);

    // LC gamma in AO basis
    let gamma_lr_ao = gammafunction_lc.as_ref().map(|gf_lc| {
        let (_gamma_lr_atom, gamma_lr_ao) = gamma_ao_wise_hop(gf_lc, &ext_atoms, n_ext_atoms, n_ext_orbs);
        gamma_lr_ao
    });

    // Electron count:
    // - Subtract 1 per partial-BDA bond (BDA loses electron)
    // - Add 1 per partial-BAA bond (ghost provides electron)
    let n_elec_raw: usize = real_atoms_i.iter().chain(real_atoms_j.iter()).map(|a| a.n_elec).sum();
    let n_elec = n_elec_raw - partial_bda_bonds.len() + partial_baa_bonds.len();



    // Occupation
    let f: Vec<f64> = (0..n_ext_orbs)
        .map(|idx| if idx < n_elec / 2 { 2.0 } else { 0.0 })
        .collect();

    // Reference density matrix: neutral atom occupations on diagonal
    // Then adjust using hybrid orbital direction (matching DFTB_MAKEREFD)
    let mut p_ref = density_matrix_ref(n_ext_orbs, &ext_atoms);

    // Adjust p_ref for partial-BDA atoms: P_ref = neutral_diag - |hybrid><hybrid|
    for bond in &partial_bda_bonds {
        let bda_local = if bond.bda_fragment == pair_i {
            bond.bda_global - frag_atom_range_i.start
        } else {
            n_real_i + (bond.bda_global - frag_atom_range_j.start)
        };
        let (ao_start, nao) = get_bda_ao_range_dftb(&ext_atoms, bda_local);
        let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
        let hybrid = compute_rotated_sp3_dftb(&bond_vec);
        let nh = nao.min(hybrid.len());
        for i in 0..nh {
            for j in 0..nh {
                p_ref[[ao_start + i, ao_start + j]] -= hybrid[i] * hybrid[j];
            }
        }
    }

    // Adjust p_ref for ghost atoms: P_ref = |hybrid><hybrid|
    for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
        let ghost_local = n_real_atoms + ghost_idx;
        let (ao_start, nao) = get_bda_ao_range_dftb(&ext_atoms, ghost_local);
        let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
        let hybrid = compute_rotated_sp3_dftb(&bond_vec);
        let nh = nao.min(hybrid.len());
        for i in 0..nh {
            for j in 0..nh {
                p_ref[[ao_start + i, ao_start + j]] = hybrid[i] * hybrid[j];
            }
        }
    }

    // Compute HOP projector:
    // 1. Partial-BDA bonds: project out 1 bond-pointing hybrid on BDA
    // 2. Partial-BAA bonds: project out 3 non-bond hybrids on ghost (complement)
    let p_hop = if partial_bda_bonds.is_empty() && partial_baa_bonds.is_empty() {
        None
    } else {
        let mut p_hop_total = Array2::<f64>::zeros([n_ext_orbs, n_ext_orbs]);

        // BDA projection (1 orbital each)
        for bond in &partial_bda_bonds {
            let bda_pos = atoms[bond.bda_global].xyz;
            let baa_pos = atoms[bond.baa_global].xyz;
            let bond_vec = baa_pos - bda_pos;

            let rotated_sp3 = compute_rotated_sp3_dftb(&bond_vec);
            let dd = compute_bda_dd_matrix(rotated_sp3.view(), HOP_SHIFT);

            // BDA local index in the pair's extended atoms
            let bda_local = if bond.bda_fragment == pair_i {
                bond.bda_global - frag_atom_range_i.start
            } else {
                n_real_i + (bond.bda_global - frag_atom_range_j.start)
            };
            let (ao_start, nao) = get_bda_ao_range_dftb(&ext_atoms, bda_local);

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

        // Ghost complement projection (3 orbitals each)
        // Ghost atoms are appended after real atoms in the order of partial_baa_bonds
        for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
            let bda_pos = atoms[bond.bda_global].xyz;
            let baa_pos = atoms[bond.baa_global].xyz;
            let bond_vec = baa_pos - bda_pos;

            let dd_ghost = compute_ghost_nonbond_dd(&bond_vec, HOP_SHIFT);

            // Ghost's local index: after all real atoms
            let ghost_local = n_real_atoms + ghost_idx;
            let (ao_start, nao) = get_bda_ao_range_dftb(&ext_atoms, ghost_local);

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

    // Compute external ESP for the pair.
    // ESP for atom A in frag I within the pair:
    //   esp_pair_A = esp_q_I[A] - gamma_ext[ext_I_A, ext_J] . dq_ext_J
    // This removes the interaction with J (which is now internal to the pair).
    //
    // For the pair, we use the extended gamma to subtract I-J interaction properly.
    // esp_q_I already has: gamma_ext[ext_I, :] . dq_ext - gamma_ext[ext_I, ext_I] . dq_ext_I
    // So: esp_pair_I = esp_q_I - gamma_ext[ext_I, ext_J] . dq_ext_J
    //
    // But we only have atom-level esp_q for real atoms from the monomer SCC.
    // For the pair ESP, we compute using the extended gamma blocks.
    let gamma_ext = &hop_data.gamma_ext;
    let ext_range_i = &hop_data.frag_info[pair_i].ext_range;
    let ext_range_j = &hop_data.frag_info[pair_j].ext_range;
    let dq_ext_j = &hop_data.dq_ext.slice(s![ext_range_j.start..ext_range_j.end]);
    let dq_ext_i = &hop_data.dq_ext.slice(s![ext_range_i.start..ext_range_i.end]);

    // ESP for real atoms in I: subtract J's contribution
    let gamma_ij_block = gamma_ext.slice(s![
        ext_range_i.start..ext_range_i.end,
        ext_range_j.start..ext_range_j.end
    ]);
    let gamma_ji_block = gamma_ext.slice(s![
        ext_range_j.start..ext_range_j.end,
        ext_range_i.start..ext_range_i.end
    ]);

    // Build ESP vector for pair atoms (real I, real J, ghosts get zero)
    let mut esp_pair = Array1::zeros(n_ext_atoms);

    // For real atoms in I: esp_q_I (covers ext_I = real + ghost) minus gamma_IJ . dq_J
    let esp_i_full: Array1<f64> = &esp_q_i - &gamma_ij_block.dot(dq_ext_j);
    // Only use real atom entries for I
    esp_pair.slice_mut(s![..n_real_i]).assign(&esp_i_full.slice(s![..n_real_i]));

    // For real atoms in J: esp_q_J minus gamma_JI . dq_I
    let esp_j_full: Array1<f64> = &esp_q_j - &gamma_ji_block.dot(dq_ext_i);
    esp_pair.slice_mut(s![n_real_i..n_real_i + n_real_j])
        .assign(&esp_j_full.slice(s![..n_real_j]));

    // Ghost atoms in pair get external ESP from all fragments NOT in the pair.
    // Ghost is at BDA's position with same element, so gamma(ghost, a) ≈ gamma(BDA, a).
    // ESP_ghost = gamma_ext[bda_ext, :] . dq_ext - gamma_ext[bda_ext, ext_I] . dq_I
    //                                             - gamma_ext[bda_ext, ext_J] . dq_J
    for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
        // Find BDA's position in the supersystem ext_atoms
        // BDA is in some fragment K (not in this pair).
        // Its ext_atoms index = ext_range_K.start + local_index_in_K
        let bda_frag = bond.bda_fragment;
        let bda_ext_range = &hop_data.frag_info[bda_frag].ext_range;
        let bda_global = bond.bda_global;
        // BDA's local index within its fragment's real atoms
        let bda_frag_atom_start = if bda_frag == pair_i {
            frag_atom_range_i.start
        } else if bda_frag == pair_j {
            frag_atom_range_j.start
        } else {
            // BDA is in an external fragment
            hop_data.monomer_indices[bda_frag][0]
        };
        let bda_local_in_frag = bda_global - bda_frag_atom_start;
        let bda_ext_idx = bda_ext_range.start + bda_local_in_frag;

        // Full ESP at BDA position from ALL atoms
        let full_esp: f64 = gamma_ext.row(bda_ext_idx).dot(&hop_data.dq_ext);
        // Subtract pair I's contribution
        let esp_from_i: f64 = gamma_ext.slice(s![bda_ext_idx, ext_range_i.start..ext_range_i.end])
            .dot(dq_ext_i);
        // Subtract pair J's contribution
        let esp_from_j: f64 = gamma_ext.slice(s![bda_ext_idx, ext_range_j.start..ext_range_j.end])
            .dot(dq_ext_j);

        let ghost_esp = full_esp - esp_from_i - esp_from_j;
        esp_pair[n_real_atoms + ghost_idx] = ghost_esp;
    }

    // Convert ESP to AO matrix
    let v_esp = atomvec_to_aomat(esp_pair.view(), n_ext_orbs, &ext_atoms) * &s * 0.5;

    // Initial dq from monomers (real atoms only, ghosts zero)
    let mut dq = Array1::zeros(n_ext_atoms);
    // Use real-atom dq from monomer extended charges
    dq.slice_mut(s![..n_real_i])
        .assign(&mono_dq_i.slice(s![..n_real_i]));
    dq.slice_mut(s![n_real_i..n_real_i + n_real_j])
        .assign(&mono_dq_j.slice(s![..n_real_j]));

    let mixer = BroydenMixerNew::from_config(n_ext_atoms, broyden_config);
    let p = p_ref.clone();

    // Build ZREF/QREF for pair atoms (for repulsive energy scaling).
    // Pair atoms: [real_I, real_J, ghosts]
    // For healed bonds (both BDA and BAA in pair): BDA's ZREF is restored (+1)
    let mut pair_zref = Array1::zeros(n_ext_atoms);
    let mut pair_qref = Array1::zeros(n_ext_atoms);

    // Copy ZREF/QREF from monomer data for real atoms of I
    pair_zref.slice_mut(s![..n_real_i]).assign(
        &hop_data.zref.slice(s![fi_i.ext_range.start..fi_i.ext_range.start + n_real_i]),
    );
    pair_qref.slice_mut(s![..n_real_i]).assign(
        &hop_data.qref.slice(s![fi_i.ext_range.start..fi_i.ext_range.start + n_real_i]),
    );

    // Copy ZREF/QREF for real atoms of J
    pair_zref.slice_mut(s![n_real_i..n_real_atoms]).assign(
        &hop_data.zref.slice(s![fi_j.ext_range.start..fi_j.ext_range.start + n_real_j]),
    );
    pair_qref.slice_mut(s![n_real_atoms - n_real_j..n_real_atoms]).assign(
        &hop_data.qref.slice(s![fi_j.ext_range.start..fi_j.ext_range.start + n_real_j]),
    );

    // Restore ZREF for healed bonds: BDA's ZREF gets +1 back
    for bond in &hop_data.detached_bonds {
        let bda_in_pair = bond.bda_fragment == pair_i || bond.bda_fragment == pair_j;
        let baa_in_pair = bond.baa_fragment == pair_i || bond.baa_fragment == pair_j;
        if bda_in_pair && baa_in_pair {
            // Healed bond: restore BDA's ZREF
            let bda_local = if bond.bda_fragment == pair_i {
                bond.bda_global - frag_atom_range_i.start
            } else {
                n_real_i + (bond.bda_global - frag_atom_range_j.start)
            };
            pair_zref[bda_local] += 1.0;
        }
    }

    // Ghost atoms: ZREF=1, QREF=BDA atom's original QREF
    for (ghost_idx, bond) in partial_baa_bonds.iter().enumerate() {
        let ghost_local = n_real_atoms + ghost_idx;
        pair_zref[ghost_local] = 1.0;
        let bda_atom = &atoms[bond.bda_global];
        pair_qref[ghost_local] = bda_atom.valorbs_occupation.iter().sum();
    }

    PairHopScc {
        i: pair_i,
        j: pair_j,
        ext_atoms,
        n_ext_atoms,
        n_real_i,
        n_real_j,
        n_real_atoms,
        n_ghost_atoms,
        n_ext_orbs,
        n_real_orbs,
        n_real_orbs_i,
        n_elec,
        h0,
        s,
        x,
        gamma,
        v_esp,
        p_hop,
        p_ref,
        p,
        dq,
        f,
        mixer,
        last_energy: 0.0,
        orbs: None,
        orbe: None,
        zref: pair_zref,
        qref: pair_qref,
        gamma_lr_ao,
        delta_p: None,
    }
}

/// Run the pair SCC loop to convergence.
///
/// Returns (total_energy, delta_dq_real) where delta_dq_real is the change in
/// real-atom charges relative to monomer charges.
pub fn run_pair_scc_hop(
    pair: &mut PairHopScc,
    config: SccConfig,
    vrep: &RepulsivePotential,
    mix_config: &MixConfig,
) -> f64 {
    let max_iter = config.scf_max_cycles;
    let scf_charge_conv = config.scf_charge_conv;
    let scf_energy_conv = config.scf_energy_conv;

    // LC: create Anderson accelerator for density matrix mixing
    let mut accel: Option<crate::scc::mixer::AndersonAccel> = if pair.gamma_lr_ao.is_some() {
        let dim = pair.n_ext_orbs * pair.n_ext_orbs;
        Some(mix_config.build_mixer(dim).unwrap())
    } else {
        None
    };

    let h_esp: Array2<f64> = &pair.h0 + &pair.v_esp;

    for iter in 0..max_iter {
        // Coulomb: gamma . dq → AO basis
        let gamma_dq = pair.gamma.dot(&pair.dq);
        let v_coul = atomvec_to_aomat(gamma_dq.view(), pair.n_ext_orbs, &pair.ext_atoms);
        let h_coul = &v_coul * &pair.s * 0.5;

        let mut h: Array2<f64> = &h_coul + &h_esp;

        // LC Fock: add exchange from delta_p
        if let (Some(ref gamma_lr_ao), Some(ref delta_p)) = (&pair.gamma_lr_ao, &pair.delta_p) {
            let h_x = lc_exact_exchange(pair.s.view(), gamma_lr_ao.view(), delta_p.view());
            h = h + h_x;
        }

        if let Some(ref p_hop) = pair.p_hop {
            h = &h + p_hop;
        }

        // Loewdin
        h = pair.x.t().dot(&h).dot(&pair.x);
        let (orbe, orbs_prime) = dsyevd_eigh(h.view());
        let orbs = pair.x.dot(&orbs_prime);

        // Density
        let mut p_new: Array2<f64> = density_matrix(orbs.view(), &pair.f);
        let dp: Array2<f64> = &p_new - &pair.p_ref;

        // Density mixing: LC uses Anderson on delta_p, non-LC uses Broyden on charges
        let (dq_new, delta_p_new): (Array1<f64>, Option<Array2<f64>>) = if let Some(ref mut aa) = accel {
            let dim = pair.n_ext_orbs * pair.n_ext_orbs;
            let dp_flat: ArrayView1<f64> = dp.view().into_shape(dim).unwrap();

            let delta_p: Array2<f64> = match &pair.delta_p {
                Some(dp0) => {
                    let dp0_flat: ArrayView1<f64> = dp0.view().into_shape(dim).unwrap();
                    aa.apply(dp0_flat, dp_flat).unwrap()
                        .into_shape(p_new.raw_dim()).unwrap()
                }
                None => {
                    aa.apply(Array1::zeros(dim).view(), dp_flat).unwrap()
                        .into_shape(p_new.raw_dim()).unwrap()
                }
            };
            p_new = &delta_p + &pair.p_ref;
            let dq_temp = mulliken_atomwise(delta_p.view(), pair.s.view(), &pair.ext_atoms, pair.n_ext_atoms);
            (dq_temp, Some(delta_p))
        } else {
            let dq_raw = mulliken_atomwise(dp.view(), pair.s.view(), &pair.ext_atoms, pair.n_ext_atoms);
            let delta_dq = &dq_raw - &pair.dq;
            let dq_temp = pair.mixer.next(&pair.dq, &delta_dq);
            (dq_temp, None)
        };

        pair.p = p_new;

        // Energy
        let mut scf_energy = get_electronic_energy_new(
            pair.p.view(),
            pair.h0.view(),
            dq_new.view(),
            pair.gamma.view(),
        );

        // LC exchange energy
        if let (Some(ref gamma_lr_ao), Some(ref dp_lc)) = (&pair.gamma_lr_ao, &delta_p_new) {
            scf_energy += calc_exchange(pair.s.view(), gamma_lr_ao.view(), dp_lc.view());
        }

        let diff_dq = dq_new.root_mean_sq_err(&pair.dq).unwrap();
        let converged = diff_dq < scf_charge_conv
            && (pair.last_energy - scf_energy).abs() < scf_energy_conv;

        pair.last_energy = scf_energy;
        pair.dq = dq_new;
        pair.delta_p = delta_p_new;

        if converged {
            // Repulsive energy with ZREF/QREF scaling (matching DFTB_EREP)
            let e_rep = get_repulsive_energy_scaled(
                &pair.ext_atoms,
                pair.n_ext_atoms,
                vrep,
                pair.zref.view(),
                pair.qref.view(),
            );
            let total = scf_energy + e_rep;

            pair.orbs = Some(orbs);
            pair.orbe = Some(orbe);
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
