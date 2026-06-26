use dialect_base::constants::K_BOLTZMANN;
use dialect_config::settings::DispersionConfig;
use dialect_config::Configuration;
use crate::initialization::basis::Basis;
use crate::parameters::{
    COUL_CHEMICAL_HARDNESS, PAULING_EN, REFERENCE_OCCUPATION, REP_ALPHA_PARAMS, REP_Z_EFF_PARAMS,
};
use crate::scc::hamiltonian::h0_xtb1_new;
use crate::{
    initialization::atom::XtbAtom, initialization::system::XtbSystem,
    integrals::calc_overlap_matrix_parallel,
};
use nalgebra::Point3;
use ndarray::prelude::*;
use rusty_dftd_lib::*;
use std::time::Instant;

impl XtbSystem {
    pub fn get_overlap(&mut self) {
        let s: Array2<f64> = calc_overlap_matrix_parallel(&self.basis);
        self.properties.set_s(s);
    }

    pub fn get_h0(&mut self) {
        let h0: Array2<f64> = h0_xtb1_new(&self.atoms, self.properties.s().unwrap(), &self.basis);
        self.properties.set_h0(h0);
    }

}

pub fn calculate_repulsive_energy_xtb(atoms: &[XtbAtom]) -> f64 {
    let mut erep: f64 = 0.0;

    // two loops over the atoms
    for (i, atomi) in atoms.iter().enumerate() {
        // get the z_eff and alpha values
        let z_eff_i: f64 = REP_Z_EFF_PARAMS[atomi.kind.number_usize() - 1];
        let alpha_i: f64 = REP_ALPHA_PARAMS[atomi.kind.number_usize() - 1];

        for (j, atomj) in atoms.iter().enumerate() {
            let z_eff_j: f64 = REP_Z_EFF_PARAMS[atomj.kind.number_usize() - 1];
            let alpha_j: f64 = REP_ALPHA_PARAMS[atomj.kind.number_usize() - 1];

            if i < j {
                // get the distance between the atoms
                let diff_vec = atomi - atomj;
                let distance: f64 =
                    (diff_vec.x.powi(2) + diff_vec.y.powi(2) + diff_vec.z.powi(2)).sqrt();
                let energy_val: f64 =
                    (-(alpha_i * alpha_j).sqrt() * distance.powf(1.5)).exp() * z_eff_i * z_eff_j
                        / distance;
                erep += energy_val;
            }
        }
    }
    erep
}

/// Compute third-order Coulomb Hamiltonian contribution
/// Optimized to avoid temporary allocations
#[inline]
/// Third-order Coulomb shift matrix (AO-resolved): each orbital row carries
/// its atom's `dq^2 * Γ'` value, and the result is the outer sum
/// `esp_row[i] + esp_row[j]` (later scaled by the overlap).
pub fn coul_third_order_hamiltonian(
    hubbards_derivs: ArrayView1<f64>,
    dq: ArrayView1<f64>,
    basis: &Basis,
) -> Array2<f64> {
    let n_orbs: usize = basis.nbas;

    // Build esp_ao_row: map atomic values to orbital indices
    let mut esp_ao_row: Array1<f64> = Array1::zeros(n_orbs);
    for shell in basis.shells.iter() {
        let atom_val = dq[shell.atom_index].powi(2) * hubbards_derivs[shell.atom_index];
        for i in shell.sph_start..shell.sph_end {
            esp_ao_row[i] = atom_val;
        }
    }

    // Compute outer sum directly: esp_ao[i,j] = esp_ao_row[i] + esp_ao_row[j]
    let mut esp_ao = Array2::uninit((n_orbs, n_orbs));
    for i in 0..n_orbs {
        let vi = esp_ao_row[i];
        for j in 0..n_orbs {
            esp_ao[[i, j]].write(vi + esp_ao_row[j]);
        }
    }

    unsafe { esp_ao.assume_init() }
}

/// Compute third-order Coulomb Hamiltonian contribution scaled by overlap matrix
/// result[i,j] = (esp_row[i] + esp_row[j]) * s[i,j] * scale
/// Fuses outer sum and scaling in one pass
#[inline]
pub fn coul_third_order_hamiltonian_scaled(
    hubbards_derivs: ArrayView1<f64>,
    dq: ArrayView1<f64>,
    basis: &Basis,
    s: ArrayView2<f64>,
    scale: f64,
) -> Array2<f64> {
    let n_orbs: usize = basis.nbas;

    // Build esp_ao_row: map atomic values to orbital indices
    let mut esp_ao_row: Array1<f64> = Array1::zeros(n_orbs);
    for shell in basis.shells.iter() {
        let atom_val = dq[shell.atom_index].powi(2) * hubbards_derivs[shell.atom_index];
        for i in shell.sph_start..shell.sph_end {
            esp_ao_row[i] = atom_val;
        }
    }

    // Compute outer sum and scale in one pass
    let mut result = Array2::uninit((n_orbs, n_orbs));
    for i in 0..n_orbs {
        let vi = esp_ao_row[i];
        for j in 0..n_orbs {
            result[[i, j]].write((vi + esp_ao_row[j]) * s[[i, j]] * scale);
        }
    }

    unsafe { result.assume_init() }
}

/// Compute the full SCC Hamiltonian in one pass: h0 + h_coul - h_coul_third_order
/// Avoids creating intermediate arrays
#[inline]
pub fn build_scc_hamiltonian(
    h0: ArrayView2<f64>,
    gamma_dq: ArrayView1<f64>,
    hubbards_derivs: ArrayView1<f64>,
    dq: ArrayView1<f64>,
    basis: &Basis,
    s: ArrayView2<f64>,
) -> Array2<f64> {
    let n_orbs: usize = basis.nbas;

    // Build esp_ao_row for third order: map atomic values to orbital indices
    let mut esp_ao_row: Array1<f64> = Array1::zeros(n_orbs);
    for shell in basis.shells.iter() {
        let atom_val = dq[shell.atom_index].powi(2) * hubbards_derivs[shell.atom_index];
        for i in shell.sph_start..shell.sph_end {
            esp_ao_row[i] = atom_val;
        }
    }

    // Compute h = h0 + 0.5 * (gamma_dq[i] + gamma_dq[j]) * s - 0.5 * (esp[i] + esp[j]) * s
    let mut h = Array2::uninit((n_orbs, n_orbs));
    for i in 0..n_orbs {
        let gdi = gamma_dq[i];
        let espi = esp_ao_row[i];
        for j in 0..n_orbs {
            let sij = s[[i, j]];
            let h_coul = 0.5 * (gdi + gamma_dq[j]) * sij;
            let h_third = 0.5 * (espi + esp_ao_row[j]) * sij;
            h[[i, j]].write(h0[[i, j]] + h_coul - h_third);
        }
    }

    unsafe { h.assume_init() }
}

/// Compute only the perturbation Hamiltonian: h_coul - h_coul_third_order
/// Used when H0 is precomputed in the orthogonal basis
#[inline]
pub fn build_perturbation_hamiltonian(
    gamma_dq: ArrayView1<f64>,
    hubbards_derivs: ArrayView1<f64>,
    dq: ArrayView1<f64>,
    basis: &Basis,
    s: ArrayView2<f64>,
) -> Array2<f64> {
    let n_orbs: usize = basis.nbas;

    // Build esp_ao_row for third order: map atomic values to orbital indices
    let mut esp_ao_row: Array1<f64> = Array1::zeros(n_orbs);
    for shell in basis.shells.iter() {
        let atom_val = dq[shell.atom_index].powi(2) * hubbards_derivs[shell.atom_index];
        for i in shell.sph_start..shell.sph_end {
            esp_ao_row[i] = atom_val;
        }
    }

    // Compute h_pert = 0.5 * (gamma_dq[i] + gamma_dq[j]) * s - 0.5 * (esp[i] + esp[j]) * s
    // With dialect dq = -qsh_xtb, and xtb formula H1 = -0.5*S*(jmat@qsh + qsh²*gam):
    //   = -0.5*S*(-gamma@dq + dq²*gam) = +0.5*S*gamma@dq - 0.5*S*dq²*gam
    let mut h = Array2::uninit((n_orbs, n_orbs));
    for i in 0..n_orbs {
        let gdi = gamma_dq[i];
        let espi = esp_ao_row[i];
        for j in 0..n_orbs {
            let sij = s[[i, j]];
            let h_coul = 0.5 * (gdi + gamma_dq[j]) * sij;
            let h_third = 0.5 * (espi + esp_ao_row[j]) * sij;
            h[[i, j]].write(h_coul - h_third);
        }
    }

    unsafe { h.assume_init() }
}

/// Electronic xTB energy from atomwise charges: band structure Tr(P*H0)
/// + second-order Coulomb - third-order Coulomb.
pub fn get_electronic_energy_xtb(
    p: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq: ArrayView1<f64>,
    dq_ao: ArrayView1<f64>,
    gamma: ArrayView2<f64>,
    hubbard_derivs: ArrayView1<f64>,
) -> f64 {
    // band structure energy
    let e_band_structure: f64 = (&p * &h0).sum();
    // Coulomb energy from monopoles
    let e_coulomb: f64 = 0.5 * &dq_ao.dot(&gamma.dot(&dq_ao));
    // Coulomb third order energy
    let e_coul_third: f64 = 1.0 / 3.0 * dq.map(|val| val.powi(3)).dot(&hubbard_derivs);
    let e_elec: f64 = e_band_structure + e_coulomb - e_coul_third;

    e_elec
}

/// Compute electronic energy using shell-level charges
/// This is consistent with shell-level Hamiltonian construction
pub fn get_electronic_energy_xtb_shell(
    p: ArrayView2<f64>,
    h0: ArrayView2<f64>,
    dq_atom: ArrayView1<f64>,
    dq_shell: ArrayView1<f64>,
    gamma_shell: ArrayView2<f64>,
    hubbard_derivs: ArrayView1<f64>,
) -> f64 {
    // band structure energy: Tr(P * H0)
    let e_band_structure: f64 = (&p * &h0).sum();
    // Coulomb energy from monopoles using shell-shell gamma
    let e_coulomb: f64 = 0.5 * dq_shell.dot(&gamma_shell.dot(&dq_shell));
    // Coulomb third order energy (uses atomic charges)
    let e_coul_third: f64 = 1.0 / 3.0 * dq_atom.map(|val| val.powi(3)).dot(&hubbard_derivs);
    let e_elec: f64 = e_band_structure + e_coulomb - e_coul_third;

    e_elec
}

pub fn get_entropy_energy_contribution(occupation: &[f64], t: f64) -> f64 {
    let mut energy: f64 = 0.0;
    let occ_half: Array1<f64> = 0.5 * &Array::from(occupation.to_vec());

    for (idx, val) in occ_half.iter().enumerate() {
        if *val < 1.0 && *val > 0.0 {
            energy += val * (val).ln() + (1.0 - val) * (1.0 - val).ln();
        }
        if *val == 1.0 {
            energy += val * (val).ln();
        }
        if *val == 0.0 {
            energy += (1.0 - val) * (1.0 - val).ln();
        }
    }
    energy *= 2.0 * t * K_BOLTZMANN;
    energy
}

pub fn get_dispersion_energy_xtb(atoms: &[XtbAtom], full_config: &Configuration) -> f64 {
    let config: &DispersionConfig = &full_config.dispersion;
    let positions: Vec<Point3<f64>> = atoms
        .iter()
        .map(|atom| Point3::from(atom.xyz))
        .collect::<Vec<Point3<f64>>>();
    let atomic_numbers: Vec<u8> = atoms.iter().map(|atom| atom.number).collect::<Vec<u8>>();
    let pos_an = (&positions, &atomic_numbers);
    let mut disp_mol = model::Molecule::from(pos_an);
    let disp: D3Model = D3Model::from_molecule(&disp_mol, None);
    let cutoff: RealspaceCutoff = RealspaceCutoffBuilder::new()
        .set_cn(CN_CUTOFF_D3_DEFAULT)
        .build();

    let d3param: D3Param = D3ParamBuilder::new()
        .set_s6(config.s6)
        .set_s8(config.s8)
        .set_s9(1.0)
        .set_a1(config.a1)
        .set_a2(config.a2)
        .build();
    let param: RationalDamping3Param = RationalDamping3Param::from((d3param, &disp_mol.num));
    let disp_result = get_dispersion(
        &mut disp_mol,
        &disp,
        &param,
        &cutoff,
        false,
        false,
        full_config.parallelization.number_of_cores,
    );
    disp_result.energy
}

/// Compute initial atomic charge guess using simplified electronegativity equalization.
/// This gives a better starting point than zero charges, reducing SCC iterations.
///
/// The method uses Pauling electronegativities and chemical hardness to estimate
/// partial charges: q_i ≈ (EN_mean - EN_i) * scale / hardness_i
/// where scale is adjusted to satisfy charge conservation.
pub fn compute_initial_charges(atoms: &[XtbAtom], total_charge: f64) -> Array1<f64> {
    let n_atoms = atoms.len();
    let mut charges = Array1::zeros(n_atoms);

    if n_atoms == 0 {
        return charges;
    }

    // Compute mean electronegativity
    let mut en_sum = 0.0;
    for atom in atoms.iter() {
        en_sum += PAULING_EN[atom.number as usize - 1];
    }
    let en_mean = en_sum / n_atoms as f64;

    // Compute raw charges: q_i ∝ (EN_mean - EN_i) / hardness_i
    // Atoms with higher EN than mean get negative charge (attract electrons)
    let mut sum_raw = 0.0;
    for (i, atom) in atoms.iter().enumerate() {
        let en_i = PAULING_EN[atom.number as usize - 1];
        let hardness_i = COUL_CHEMICAL_HARDNESS[atom.number as usize - 1];
        // Avoid division by very small hardness
        let h = hardness_i.max(0.1);
        let raw = (en_mean - en_i) / h;
        charges[i] = raw;
        sum_raw += raw;
    }

    // Adjust to satisfy total charge constraint
    // q_i_final = q_i_raw - (sum_raw - total_charge) / n_atoms
    let correction = (sum_raw - total_charge) / n_atoms as f64;
    for i in 0..n_atoms {
        charges[i] -= correction;
    }

    // Scale down the charges (EEQ typically gives smaller charges than raw EN differences)
    // This scaling factor is empirical; smaller values are safer but less effective
    let scale_factor = 0.05;
    for i in 0..n_atoms {
        charges[i] *= scale_factor;
    }

    // Final correction to ensure exact charge conservation
    let final_sum: f64 = charges.sum();
    let final_correction = (final_sum - total_charge) / n_atoms as f64;
    for i in 0..n_atoms {
        charges[i] -= final_correction;
    }

    charges
}

/// Convert atomic charges to orbital-resolved charges using basis information
pub fn atomic_to_orbital_charges(
    atomic_charges: ArrayView1<f64>,
    basis: &Basis,
    atoms: &[XtbAtom],
) -> Array1<f64> {
    let n_orbs = basis.nbas;
    let mut orbital_charges = Array1::zeros(n_orbs);

    // Distribute atomic charges across orbitals proportionally to reference occupations
    for shell in basis.shells.iter() {
        let atom_idx = shell.atom_index;
        let atom = &atoms[atom_idx];
        let l = shell.angular_momentum;
        let ref_occ = REFERENCE_OCCUPATION[atom.number as usize - 1][l];
        let n_orbs_in_shell = 2 * l + 1;

        // Get total valence electrons for this atom
        let mut total_ref_occ = 0.0;
        for l_check in 0..3 {
            total_ref_occ += REFERENCE_OCCUPATION[atom.number as usize - 1][l_check];
        }

        if total_ref_occ > 0.0 && !shell.polarization {
            // Distribute atomic charge proportionally to this shell's reference occupation
            let frac = ref_occ / total_ref_occ;
            let shell_charge = atomic_charges[atom_idx] * frac;
            let charge_per_orb = shell_charge / n_orbs_in_shell as f64;
            for i in shell.sph_start..shell.sph_end {
                orbital_charges[i] = charge_per_orb;
            }
        }
    }

    orbital_charges
}

pub fn create_density_ref(basis: &Basis, atoms: &[XtbAtom]) -> Array2<f64> {
    // initialize empty density matrix
    let nbas: usize = basis.nbas;
    let mut p: Array2<f64> = Array2::zeros([nbas, nbas]);

    // iterate over basis shells
    for (shell_idx, shell_i) in basis.shells.iter().enumerate() {
        let atom: &XtbAtom = &atoms[shell_i.atom_index];
        let l: usize = shell_i.angular_momentum;
        let ref_occ: f64 =
            REFERENCE_OCCUPATION[atom.number as usize - 1][l] / (2.0 * l as f64 + 1.0);
        // iterate over angular components
        for i in (shell_i.sph_start..shell_i.sph_end) {
            if shell_i.polarization {
                p[[i, i]] = 0.0;
            } else {
                p[[i, i]] = ref_occ;
            }
        }
    }
    p
}
