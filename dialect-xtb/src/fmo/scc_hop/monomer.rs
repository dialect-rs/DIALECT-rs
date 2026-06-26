//! Monomer SCC with HOP for FMO-xTB.
//!
//! Each monomer is computed with extended atoms (real + ghost atoms at BDA positions).
//! The HOP projector is added to the Hamiltonian. Shell-resolved gamma, third-order
//! Coulomb, and Fermi occupation are used (xTB specifics).

use super::hop_data::{
    compute_monomer_hop_projector_xtb, compute_rotated_sp3_xtb,
    get_frag_shell_range, XtbHopData, XtbHopFragInfo,
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
use std::ops::Range;

/// State for one monomer's HOP SCC calculation.
#[derive(Debug, Clone)]
pub struct XtbMonomerHopScc {
    /// Fragment index
    pub frag_idx: usize,
    /// Extended atoms (real + ghosts)
    pub ext_atoms: Vec<XtbAtom>,
    /// Extended basis
    pub basis: Basis,
    /// Number of extended atoms
    pub n_ext_atoms: usize,
    /// Number of real atoms
    pub n_real_atoms: usize,
    /// Number of extended orbitals
    pub n_ext_orbs: usize,
    /// Number of real orbitals
    pub n_real_orbs: usize,
    /// Number of extended shells
    pub n_ext_shells: usize,
    /// Number of real shells
    pub n_real_shells: usize,
    /// Adjusted electron count
    pub n_elec: usize,
    /// H0 matrix
    pub h0: Array2<f64>,
    /// Overlap matrix
    pub s: Array2<f64>,
    /// S^{-1/2}
    pub x: Array2<f64>,
    /// Local gamma_shell matrix (n_ext_shells x n_ext_shells)
    pub gamma_shell: Array2<f64>,
    /// HOP projector (None if no cut bonds)
    pub p_hop: Option<Array2<f64>>,
    /// Reference density matrix
    pub p_ref: Array2<f64>,
    /// Current density matrix
    pub p: Array2<f64>,
    /// Charge differences (atom-level, n_ext_atoms including ghosts)
    pub dq: Array1<f64>,
    /// Charge differences (AO-level)
    pub dq_ao: Array1<f64>,
    /// Charge differences (shell-level)
    pub dq_shell: Array1<f64>,
    /// Reference Mulliken population per shell: q_ref_shell[s] = Σ_{μ∈s} (P_ref·S)_{μμ}
    pub q_ref_shell: Array1<f64>,
    /// Hubbard derivative parameters
    pub hubbard_derivatives: Array1<f64>,
    /// Last SCF energy
    pub last_energy: f64,
    /// MO coefficients
    pub orbs: Option<Array2<f64>>,
    /// MO energies
    pub orbe: Option<Array1<f64>>,
    /// Occupation numbers
    pub f: Vec<f64>,
    /// Broyden mixer for AO-level charges
    pub mixer: BroydenMixerNew,
}

/// Prepare a monomer for HOP SCC calculation (xTB version).
pub fn prepare_monomer_hop_xtb(
    frag_idx: usize,
    frag_info: &XtbHopFragInfo,
    hop_data: &XtbHopData,
    gammafunction: &crate::scc::gamma_matrix::XtbGammaFunction,
    atoms: &[XtbAtom],
    frag_atom_range: &Range<usize>,
    broyden_config: &dialect_config::settings::BroydenConfig,
) -> XtbMonomerHopScc {
    let ext_atoms: Vec<XtbAtom> = hop_data.ext_atoms[frag_info.ext_range.clone()].to_vec();
    let n_ext_atoms = ext_atoms.len();
    let n_real_atoms = frag_info.n_real_atoms;
    let n_elec = frag_info.n_elec;

    // Build extended basis
    let basis = create_basis_set(&ext_atoms);
    let n_ext_orbs = basis.nbas;
    let n_ext_shells = basis.shells.len();
    let n_real_orbs = frag_info.n_real_orbs;
    let n_real_shells = frag_info.n_real_shells;

    // Compute S and H0 with CN
    let cn_slice = hop_data.cn_ext.slice(s![frag_info.ext_range.start..frag_info.ext_range.end]);
    let s = calc_overlap_matrix_parallel(&basis);
    let h0 = h0_xtb1_with_cn(&ext_atoms, s.view(), &basis, cn_slice);
    let x = compute_s_inv_sqrt(s.view());

    // Local gamma_shell for extended fragment
    let gamma_shell = gamma_matrix_shell(gammafunction, &ext_atoms, &basis);

    // Reference density matrix with hybrid-based adjustments
    let mut p_ref = create_density_ref(&basis, &ext_atoms);

    // Adjust p_ref for BDA atoms: P_ref -= |hybrid><hybrid|
    for bond in &hop_data.detached_bonds {
        if bond.bda_fragment == frag_idx {
            let bda_local = bond.bda_global - frag_atom_range.start;
            let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
            let hybrid = compute_rotated_sp3_xtb(&bond_vec);
            // Find AO range for BDA atom
            let (ao_start, nao) = crate::hop::get_bda_ao_range(&basis, bda_local);
            let nh = nao.min(hybrid.len());
            for i in 0..nh {
                for j in 0..nh {
                    p_ref[[ao_start + i, ao_start + j]] -= hybrid[i] * hybrid[j];
                }
            }
        }
    }

    // Adjust p_ref for ghost atoms: P_ref = |hybrid><hybrid|
    let mut ghost_idx = 0;
    for bond in &hop_data.detached_bonds {
        if bond.baa_fragment == frag_idx {
            let ghost_local = n_real_atoms + ghost_idx;
            let bond_vec = atoms[bond.baa_global].xyz - atoms[bond.bda_global].xyz;
            let hybrid = compute_rotated_sp3_xtb(&bond_vec);
            let (ao_start, nao) = crate::hop::get_bda_ao_range(&basis, ghost_local);
            let nh = nao.min(hybrid.len());
            // Zero out the entire block first, then set hybrid
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
            ghost_idx += 1;
        }
    }

    // Compute HOP projector
    let p_hop = compute_monomer_hop_projector_xtb(
        &hop_data.detached_bonds,
        frag_idx,
        &basis,
        s.view(),
        atoms,
        frag_atom_range,
        n_real_atoms,
    );

    // Hubbard derivatives
    let mut hubbard_derivatives = Array1::zeros(n_ext_atoms);
    for (val, atom) in hubbard_derivatives.iter_mut().zip(ext_atoms.iter()) {
        *val = COUL_THIRD_ORDER_ATOM[atom.number as usize - 1];
    }

    // Reference Mulliken population per shell: q_ref_shell[s] = Σ_{μ∈s} (P_ref·S)_{μμ}
    let mut q_ref_shell = Array1::zeros(n_ext_shells);
    for (shell_idx, shell) in basis.shells.iter().enumerate() {
        for i in shell.sph_start..shell.sph_end {
            for k in 0..n_ext_orbs {
                q_ref_shell[shell_idx] += p_ref[[k, i]] * s[[i, k]];
            }
        }
    }

    // Initialize charges and density
    let dq = Array1::zeros(n_ext_atoms);
    let dq_ao = Array1::zeros(n_ext_orbs);
    let dq_shell = Array1::zeros(n_ext_shells);
    let p = p_ref.clone();

    let mixer = BroydenMixerNew::from_config(n_ext_orbs, broyden_config);

    XtbMonomerHopScc {
        frag_idx,
        ext_atoms,
        basis,
        n_ext_atoms,
        n_real_atoms,
        n_ext_orbs,
        n_real_orbs,
        n_ext_shells,
        n_real_shells,
        n_elec,
        h0,
        s,
        x,
        gamma_shell,
        p_hop,
        p_ref,
        p,
        dq,
        dq_ao,
        dq_shell,
        q_ref_shell,
        hubbard_derivatives,
        last_energy: 0.0,
        orbs: None,
        orbe: None,
        f: vec![0.0; n_ext_orbs],
        mixer,
    }
}

/// Perform one monomer SCC step with HOP (xTB version).
///
/// Hamiltonian:
///   H = H0 + H_coul_vesp - H_coul_third + P_HOP
///
/// Shell-level Coulomb: gamma_shell.dot(dq_shell) → shell_to_ao_values → outer_sum * S * 0.5
/// Third-order: coul_third_order_hamiltonian(hubbard_derivs, dq, basis) * S * 0.5
///
/// Returns true if converged.
pub fn monomer_scc_step_hop_xtb(
    mono: &mut XtbMonomerHopScc,
    v_esp: Array2<f64>,
    temperature: f64,
    scf_charge_conv: f64,
    scf_energy_conv: f64,
) -> bool {
    // Coulomb ESP contribution
    let h_coul_vesp = v_esp * &mono.s * 0.5;

    // Third-order Coulomb
    let h_coul_third = coul_third_order_hamiltonian(
        mono.hubbard_derivatives.view(),
        mono.dq.view(),
        &mono.basis,
    ) * &mono.s
        * 0.5;

    // Full Hamiltonian
    let mut h: Array2<f64> = &mono.h0 + &h_coul_vesp - &h_coul_third;
    if let Some(ref p_hop) = mono.p_hop {
        h += p_hop;
    }

    // Loewdin orthogonalization
    let h = mono.x.t().dot(&h).dot(&mono.x);
    let (orbe, orbs_prime) = dsyevd_eigh(h.view());
    let orbs = mono.x.dot(&orbs_prime);

    // Fermi occupation
    let (_, f) = fermi_occupation(orbe.view(), mono.n_elec, temperature);

    // Density matrix
    let p = density_matrix(orbs.view(), &f);
    let dp = &p - &mono.p_ref;

    // Mulliken charges (raw, unmixed)
    let dq_ao_raw = mulliken_aowise_diff(p.view(), mono.p_ref.view(), mono.s.view());

    // Broyden mixing on AO-level charges
    let delta_dq_ao: Array1<f64> = &dq_ao_raw - &mono.dq_ao;
    let dq_ao_new = mono.mixer.next(&mono.dq_ao, &delta_dq_ao);
    let dq_shell_new = ao_to_shell_charges(&mono.basis, dq_ao_new.view());

    // Atom-level charges from mixed AO charges
    let dq_atom_out =
        mulliken_atomwise_from_ao_xtb(&mono.basis, mono.n_ext_atoms, dq_ao_new.view());

    // Electronic energy + entropy
    let entropy = compute_total_entropy(orbe.view(), mono.n_elec, temperature);
    let mut scf_energy = get_electronic_energy_xtb_shell(
        p.view(),
        mono.h0.view(),
        dq_atom_out.view(),
        dq_shell_new.view(),
        mono.gamma_shell.view(),
        mono.hubbard_derivatives.view(),
    ) + entropy;
    // NOTE: Do NOT add Tr(P * V_HOP) to the energy.
    // V_HOP is a constraint potential, not part of the physical energy.
    // Including it would create spurious terms in FMO pair deltas.

    // Convergence check
    let diff_dq = dq_atom_out.root_mean_sq_err(&mono.dq).unwrap();
    let conv_charge = diff_dq < scf_charge_conv;
    let conv_energy = (mono.last_energy - scf_energy).abs() < scf_energy_conv;

    // Update state
    mono.p = p;
    mono.dq = dq_atom_out;
    mono.dq_ao = dq_ao_new;
    mono.dq_shell = dq_shell_new;
    mono.last_energy = scf_energy;
    mono.orbs = Some(orbs);
    mono.orbe = Some(orbe);
    mono.f = f;

    conv_charge && conv_energy
}
