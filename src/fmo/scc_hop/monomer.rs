//! Monomer SCC with HOP for FMO-DFTB.
//!
//! Each monomer is computed with extended atoms (real + ghost H atoms at BAA positions).
//! The HOP projector is added to the Hamiltonian. The Mulliken charges include ghost atoms.

use super::hop_data::{
    compute_monomer_hop_projector, compute_rotated_sp3_dftb, get_bda_ao_range_dftb, HopData,
    HopFragInfo,
};
use crate::fmo::scc::helpers::atomvec_to_aomat;
use crate::initialization::Atom;
use crate::io::SccConfig;
use crate::scc::gamma_approximation::{gamma_ao_wise_hop, gamma_atomwise_hop, GammaFunction};
use crate::scc::h0_and_s::h0_and_s;
use crate::scc::lapack_eigh::{compute_s_inv_sqrt, dsyevd_eigh};
use crate::scc::mixer::{AndersonAccel, BroydenMixerNew};
use crate::scc::mulliken::mulliken_atomwise;
use crate::scc::{calc_exchange, density_matrix, density_matrix_ref, get_electronic_energy_new, get_repulsive_energy, lc_exact_exchange};
use crate::initialization::parameters::RepulsivePotential;
use ndarray::prelude::*;
use ndarray_stats::DeviationExt;

/// State for one monomer's HOP SCC calculation.
///
/// This is separate from the Monomer struct to avoid mixing with non-HOP code.
#[derive(Debug, Clone)]
pub struct MonomerHopScc {
    /// Fragment index
    pub frag_idx: usize,
    /// Extended atoms (real + ghosts)
    pub ext_atoms: Vec<Atom>,
    /// Number of extended atoms
    pub n_ext_atoms: usize,
    /// Number of real atoms
    pub n_real_atoms: usize,
    /// Number of extended orbitals
    pub n_ext_orbs: usize,
    /// Number of real orbitals
    pub n_real_orbs: usize,
    /// Adjusted electron count
    pub n_elec: usize,
    /// H0 matrix (n_ext_orbs x n_ext_orbs)
    pub h0: Array2<f64>,
    /// Overlap matrix
    pub s: Array2<f64>,
    /// S^{-1/2} for Loewdin orthogonalization
    pub x: Array2<f64>,
    /// Local gamma matrix (n_ext_atoms x n_ext_atoms)
    pub gamma: Array2<f64>,
    /// HOP projector (None if no cut bonds)
    pub p_hop: Option<Array2<f64>>,
    /// Reference density matrix
    pub p_ref: Array2<f64>,
    /// Current density matrix
    pub p: Array2<f64>,
    /// Charge differences (n_ext_atoms, including ghosts)
    pub dq: Array1<f64>,
    /// Occupation numbers
    pub f: Vec<f64>,
    /// Broyden mixer
    pub mixer: BroydenMixerNew,
    /// Last SCF energy
    pub last_energy: f64,
    /// MO coefficients (stored after convergence)
    pub orbs: Option<Array2<f64>>,
    /// MO energies
    pub orbe: Option<Array1<f64>>,
    /// LC gamma matrix in AO basis (shell-resolved), None if non-LC
    pub gamma_lr_ao: Option<Array2<f64>>,
    /// Difference density matrix P - P_ref, used for LC mixing
    pub delta_p: Option<Array2<f64>>,
}

/// Prepare a monomer for HOP SCC calculation.
///
/// Builds H0, S, gamma, reference density, HOP projector, occupation, and mixer
/// for the extended atom set (real + ghost).
pub fn prepare_monomer_hop(
    frag_idx: usize,
    frag_info: &HopFragInfo,
    hop_data: &HopData,
    gammafunction: &GammaFunction,
    gammafunction_lc: &Option<GammaFunction>,
    slako: &crate::initialization::parameters::SlaterKoster,
    atoms: &[Atom],
    frag_atom_range: &std::ops::Range<usize>,
    broyden_config: &crate::io::settings::BroydenConfig,
) -> MonomerHopScc {
    let ext_atoms: Vec<Atom> = hop_data.ext_atoms[frag_info.ext_range.clone()].to_vec();
    let n_ext_atoms = ext_atoms.len();
    let n_ext_orbs = frag_info.n_ext_orbs;
    let n_real_atoms = frag_info.n_real_atoms;
    let n_real_orbs = frag_info.n_real_orbs;
    let n_elec = frag_info.n_elec;

    // Compute H0 and S for extended atoms (including ghost)
    let (s, h0) = h0_and_s(n_ext_orbs, &ext_atoms, slako);
    let x = compute_s_inv_sqrt(s.view());

    // Local gamma for the extended fragment
    let gamma = gamma_atomwise_hop(gammafunction, &ext_atoms, n_ext_atoms);

    // LC gamma in AO basis
    let gamma_lr_ao = gammafunction_lc.as_ref().map(|gf_lc| {
        let (_gamma_lr_atom, gamma_lr_ao) = gamma_ao_wise_hop(gf_lc, &ext_atoms, n_ext_atoms, n_ext_orbs);
        gamma_lr_ao
    });

    // Reference density matrix: neutral atom occupations on diagonal
    // Then adjust using hybrid orbital direction (matching DFTB_MAKEREFD)
    let mut p_ref = density_matrix_ref(n_ext_orbs, &ext_atoms);

    // Adjust p_ref for BDA atoms: P_ref = neutral_diag - |hybrid><hybrid|
    // (DFTB_MAKEREFD for QREF != ZREF, ZREF != 1)
    // The trace is preserved (ZREF), but off-diagonal and within-atom distribution
    // differ from simple s-orbital reduction. This matters for LC exchange.
    for bond in &hop_data.detached_bonds {
        if bond.bda_fragment == frag_idx {
            let bda_local = bond.bda_global - frag_atom_range.start;
            let (ao_start, nao) = get_bda_ao_range_dftb(&ext_atoms, bda_local);
            let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
            let hybrid = compute_rotated_sp3_dftb(&bond_vec);
            // Subtract |hybrid><hybrid| from the neutral diagonal block
            let nh = nao.min(hybrid.len());
            for i in 0..nh {
                for j in 0..nh {
                    p_ref[[ao_start + i, ao_start + j]] -= hybrid[i] * hybrid[j];
                }
            }
        }
    }

    // Adjust p_ref for ghost atoms: P_ref = |hybrid><hybrid|
    // (DFTB_MAKEREFD for ZREF == 1)
    // Ghost atoms are appended after real atoms in build_hop_data order.
    let mut ghost_idx = 0;
    for bond in &hop_data.detached_bonds {
        if bond.baa_fragment == frag_idx {
            let ghost_local = n_real_atoms + ghost_idx;
            let (ao_start, nao) = get_bda_ao_range_dftb(&ext_atoms, ghost_local);
            let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
            let hybrid = compute_rotated_sp3_dftb(&bond_vec);
            // Replace diag(1,0,0,0) with |hybrid><hybrid|
            let nh = nao.min(hybrid.len());
            for i in 0..nh {
                for j in 0..nh {
                    p_ref[[ao_start + i, ao_start + j]] = hybrid[i] * hybrid[j];
                }
            }
            ghost_idx += 1;
        }
    }

    // Compute HOP projector
    let p_hop = compute_monomer_hop_projector(
        &hop_data.detached_bonds,
        frag_idx,
        &ext_atoms,
        s.view(),
        atoms,
        frag_atom_range,
        n_real_atoms,
    );

    // Occupation: Aufbau filling with adjusted electron count
    let f: Vec<f64> = (0..n_ext_orbs)
        .map(|idx| if idx < n_elec / 2 { 2.0 } else { 0.0 })
        .collect();

    // Broyden mixer for charge mixing
    let mixer = BroydenMixerNew::from_config(n_ext_atoms, broyden_config);

    // Initial dq = zeros
    let dq = Array1::zeros(n_ext_atoms);

    // Initial density = reference density
    let p = p_ref.clone();

    MonomerHopScc {
        frag_idx,
        ext_atoms,
        n_ext_atoms,
        n_real_atoms,
        n_ext_orbs,
        n_real_orbs,
        n_elec,
        h0,
        s,
        x,
        gamma,
        p_hop,
        p_ref,
        p,
        dq,
        f,
        mixer,
        last_energy: 0.0,
        orbs: None,
        orbe: None,
        gamma_lr_ao,
        delta_p: None,
    }
}

/// Perform one monomer SCC step with HOP.
///
/// The Hamiltonian is:
///   H = H0 + 0.5 * S * (V_esp_ext + gamma_local . dq) + P_HOP
///
/// where V_esp_ext is the full ESP from all other fragments' extended charges,
/// and gamma_local . dq is the intrafragment Coulomb contribution.
///
/// Returns true if converged.
pub fn monomer_scc_step_hop(
    mono: &mut MonomerHopScc,
    v_esp_ext: ArrayView1<f64>,
    config: SccConfig,
    accel: &mut Option<AndersonAccel>,
) -> bool {
    let scf_charge_conv = config.scf_charge_conv;
    let scf_energy_conv = config.scf_energy_conv;

    // Intrafragment Coulomb: gamma_local . dq
    let gamma_dq: Array1<f64> = mono.gamma.dot(&mono.dq);

    // Total ESP per atom: external + internal
    let total_esp: Array1<f64> = &v_esp_ext + &gamma_dq;

    // Convert to AO matrix: V_ao[mu,nu] = total_esp[atom_mu] + total_esp[atom_nu]
    let v_ao: Array2<f64> = atomvec_to_aomat(total_esp.view(), mono.n_ext_orbs, &mono.ext_atoms);

    // Coulomb Hamiltonian
    let h_coul: Array2<f64> = &v_ao * &mono.s * 0.5;

    // Full Hamiltonian: H0 + Coulomb + HOP
    let mut h: Array2<f64> = &mono.h0 + &h_coul;

    // LC Fock: add exchange contribution from delta_p
    if let (Some(ref gamma_lr_ao), Some(ref delta_p)) = (&mono.gamma_lr_ao, &mono.delta_p) {
        let h_x = lc_exact_exchange(mono.s.view(), gamma_lr_ao.view(), delta_p.view());
        h = h + h_x;
    }

    if let Some(ref p_hop) = mono.p_hop {
        h = &h + p_hop;
    }

    // Loewdin orthogonalization: H' = X^T . H . X
    h = mono.x.t().dot(&h).dot(&mono.x);

    // Diagonalize
    let (orbe, orbs_prime) = dsyevd_eigh(h.view());

    // Back-transform: C = X . C'
    let orbs = mono.x.dot(&orbs_prime);

    // Density matrix: P = sum_occ f_a * C_a * C_a^T
    let mut p: Array2<f64> = density_matrix(orbs.view(), &mono.f);

    // Difference density
    let dp: Array2<f64> = &p - &mono.p_ref;

    // Density mixing: LC uses Anderson on delta_p, non-LC uses Broyden on charges
    let (dq_new, delta_p_new): (Array1<f64>, Option<Array2<f64>>) = if let Some(ref mut aa) = accel {
        // LC: mix density matrix with Anderson acceleration
        let dim = mono.n_ext_orbs * mono.n_ext_orbs;
        let dp_flat: ArrayView1<f64> = dp.view().into_shape(dim).unwrap();

        let delta_p: Array2<f64> = match &mono.delta_p {
            Some(dp0) => {
                let dp0_flat: ArrayView1<f64> = dp0.view().into_shape(dim).unwrap();
                aa.apply(dp0_flat, dp_flat).unwrap()
                    .into_shape(p.raw_dim()).unwrap()
            }
            None => {
                aa.apply(Array1::zeros(dim).view(), dp_flat).unwrap()
                    .into_shape(p.raw_dim()).unwrap()
            }
        };
        let p_mixed: Array2<f64> = &delta_p + &mono.p_ref;
        let dq_temp = mulliken_atomwise(delta_p.view(), mono.s.view(), &mono.ext_atoms, mono.n_ext_atoms);
        // Override p with mixed density
        p = p_mixed;
        (dq_temp, Some(delta_p))
    } else {
        // Non-LC: Mulliken charges then Broyden mix
        let dq_raw = mulliken_atomwise(dp.view(), mono.s.view(), &mono.ext_atoms, mono.n_ext_atoms);
        let delta_dq: Array1<f64> = &dq_raw - &mono.dq;
        let dq_temp = mono.mixer.next(&mono.dq, &delta_dq);
        (dq_temp, None)
    };

    // Electronic energy: E = Tr(P . H0) + 0.5 * dq^T . gamma . dq
    let mut scf_energy = get_electronic_energy_new(
        p.view(),
        mono.h0.view(),
        dq_new.view(),
        mono.gamma.view(),
    );

    // LC exchange energy
    if let (Some(ref gamma_lr_ao), Some(ref dp_lc)) = (&mono.gamma_lr_ao, &delta_p_new) {
        scf_energy += calc_exchange(mono.s.view(), gamma_lr_ao.view(), dp_lc.view());
    }

    // Convergence check
    let diff_dq = dq_new.root_mean_sq_err(&mono.dq).unwrap();
    let conv_charge = diff_dq < scf_charge_conv;
    let conv_energy = (mono.last_energy - scf_energy).abs() < scf_energy_conv;

    // Update state
    mono.p = p;
    mono.dq = dq_new;
    mono.last_energy = scf_energy;
    mono.orbs = Some(orbs);
    mono.orbe = Some(orbe);
    mono.delta_p = delta_p_new;

    conv_charge && conv_energy
}

/// Compute the repulsive energy for a monomer's real atoms only.
///
/// Ghost atoms do not contribute to the repulsive potential (they have no
/// nuclear charge in the repulsive model), but the real-atom pairs with ghosts
/// do contribute since h0_and_s includes them. For repulsive energy, we only
/// sum over real atom pairs.
pub fn monomer_repulsive_energy_real(
    real_atoms: &[Atom],
    n_real_atoms: usize,
    vrep: &RepulsivePotential,
) -> f64 {
    get_repulsive_energy(real_atoms, n_real_atoms, vrep)
}
