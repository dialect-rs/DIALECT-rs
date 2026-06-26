use crate::excited_states::{CTDavidsonWorkspace, ExcitedState};
use crate::fmo::{ExcitonStates, Monomer, PairType, SuperSystem};
use crate::initialization::Atom;
use crate::io::settings::LcmoConfig;
use crate::properties::Properties;
use crate::{initial_subspace, Davidson};
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray::concatenate;
use ndarray_linalg::{Eigh, UPLO};
use ndarray_npy::write_npy;
use rayon::prelude::*;
use std::fmt::{Display, Formatter};

impl SuperSystem<'_> {
    pub fn create_diabatic_basis(&self, n_ct: usize) -> Vec<BasisState<'_>> {
        let lcmo_config: LcmoConfig = self.config.fmo_lc_tddftb.clone();
        // Number of LE states per monomer.
        let n_le: usize = lcmo_config.n_le;
        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        let threshold_le: f64 = lcmo_config.active_space_threshold_le;
        let threshold_ct: f64 = lcmo_config.active_space_threshold_ct;

        let mut states: Vec<BasisState> = Vec::new();
        // Create all LE states.
        for mol in self.monomers.iter() {
            let homo: usize = mol.properties.homo().unwrap();
            let q_ov: ArrayView2<f64> = mol.properties.q_ov().unwrap();

            let mut le_states: Vec<BasisState> = (0..n_le)
                .into_par_iter()
                .map(|n| {
                    let tdm: ArrayView1<f64> = mol.properties.ci_coefficient(n).unwrap();
                    let tdm_dim2: ArrayView2<f64> = mol.properties.tdm(n).unwrap();

                    // determine the relevant orbital indices
                    let mut occ_indices: Vec<usize> = Vec::new();
                    let mut virt_indices: Vec<usize> = Vec::new();
                    for (idx_i, val_i) in tdm_dim2.outer_iter().enumerate() {
                        for (idx_j, val_j) in val_i.iter().enumerate() {
                            let abs_c_sqr: f64 = val_j.abs().powi(2);
                            if abs_c_sqr > threshold_le {
                                if !occ_indices.contains(&idx_i) {
                                    occ_indices.push(idx_i);
                                }
                                if !virt_indices.contains(&idx_j) {
                                    virt_indices.push(idx_j);
                                }
                            }
                        }
                    }

                    let le_state = LocallyExcited {
                        monomer: mol,
                        n,
                        atoms: &atoms[mol.slice.atom_as_range()],
                        q_trans: q_ov.dot(&tdm),
                        occs: mol.properties.orbs_slice(0, Some(homo + 1)).unwrap(),
                        virts: mol.properties.orbs_slice(homo + 1, None).unwrap(),
                        tdm,
                        tr_dipole: mol.properties.tr_dipole(n).unwrap(),
                        occ_indices,
                        virt_indices,
                    };

                    BasisState::LE(le_state)
                })
                .collect();

            states.append(&mut le_states)
        }

        if n_ct > 0 {
            // create ct indices
            let mut idx_vec: Vec<(usize, usize)> = Vec::new();
            for i in 0..self.n_mol {
                for j in 0..self.n_mol {
                    if j > i {
                        idx_vec.push((i, j));
                    }
                }
            }

            // Create all CT states.
            let mut ct_basis: Vec<_> = idx_vec
                .par_iter()
                .map(|(idx_i, idx_j)| {
                    let mut ct_basis_temp: Vec<_> = Vec::new();
                    // get the monomers
                    let m_i = &self.monomers[*idx_i];
                    let m_j = &self.monomers[*idx_j];

                    // get the PairType
                    let type_ij: PairType = self.properties.type_of_pair(m_i.index, m_j.index);

                    // create both CT states
                    let mut state_1 = ChargeTransferPreparation {
                        m_h: m_i,
                        m_l: m_j,
                        pair_type: type_ij,
                        properties: Properties::new(),
                        davidson_workspace: None,
                    };
                    let mut state_2 = ChargeTransferPreparation {
                        m_h: m_j,
                        m_l: m_i,
                        pair_type: type_ij,
                        properties: Properties::new(),
                        davidson_workspace: None,
                    };

                    // prepare the TDA calculation of both states
                    state_1.prepare_ct_tda(
                        self.properties.gamma(),
                        self.properties.gamma_lr(),
                        self.properties.gamma_ao(),
                        self.properties.gamma_lr_ao(),
                        self.properties.s().unwrap(),
                        atoms,
                        &self.config,
                    );
                    state_2.prepare_ct_tda(
                        self.properties.gamma(),
                        self.properties.gamma_lr(),
                        self.properties.gamma_ao(),
                        self.properties.gamma_lr_ao(),
                        self.properties.s().unwrap(),
                        atoms,
                        &self.config,
                    );

                    let nroots: usize = n_ct + 2;
                    state_1.run_ct_tda(
                        atoms,
                        nroots,
                        self.config.excited.davidson_iterations,
                        self.config.excited.davidson_convergence,
                        self.config.excited.davidson_subspace_multiplier,
                        &self.config,
                    );
                    state_2.run_ct_tda(
                        atoms,
                        nroots,
                        self.config.excited.davidson_iterations,
                        self.config.excited.davidson_convergence,
                        self.config.excited.davidson_subspace_multiplier,
                        &self.config,
                    );

                    let q_ov_1: ArrayView2<f64> = state_1.properties.q_ov().unwrap();
                    let q_ov_2: ArrayView2<f64> = state_2.properties.q_ov().unwrap();

                    for n in 0..n_ct {
                        let tdm_1: ArrayView1<f64> = state_1.properties.ci_coefficient(n).unwrap();
                        let tdm_dim2_1: ArrayView2<f64> = state_1.properties.tdm(n).unwrap();

                        // determine the relevant orbital indices
                        let mut occ_indices: Vec<usize> = Vec::new();
                        let mut virt_indices: Vec<usize> = Vec::new();
                        for (idx_i, val_i) in tdm_dim2_1.outer_iter().enumerate() {
                            for (idx_j, val_j) in val_i.iter().enumerate() {
                                let abs_c_sqr: f64 = val_j.abs().powi(2);
                                if abs_c_sqr > threshold_ct {
                                    if !occ_indices.contains(&idx_i) {
                                        occ_indices.push(idx_i);
                                    }
                                    if !virt_indices.contains(&idx_j) {
                                        virt_indices.push(idx_j);
                                    }
                                }
                            }
                        }

                        let ct_1 = ChargeTransferPair {
                            m_h: m_i.index,
                            m_l: m_j.index,
                            state_index: n,
                            state_energy: state_1.properties.ci_eigenvalue(n).unwrap(),
                            eigenvectors: state_1.properties.tdm(n).unwrap().to_owned(),
                            q_tr: q_ov_1.dot(&tdm_1),
                            tr_dipole: state_1.properties.tr_dipole(n).unwrap(),
                            occ_orb: m_i.slice.occ_orb,
                            virt_orb: m_j.slice.virt_orb,
                            occ_indices,
                            virt_indices,
                        };

                        let tdm_2: ArrayView1<f64> = state_2.properties.ci_coefficient(n).unwrap();
                        let tdm_dim2_2: ArrayView2<f64> = state_2.properties.tdm(n).unwrap();

                        // determine the relevant orbital indices
                        let mut occ_indices: Vec<usize> = Vec::new();
                        let mut virt_indices: Vec<usize> = Vec::new();
                        for (idx_i, val_i) in tdm_dim2_2.outer_iter().enumerate() {
                            for (idx_j, val_j) in val_i.iter().enumerate() {
                                let abs_c_sqr: f64 = val_j.abs().powi(2);
                                if abs_c_sqr > threshold_ct {
                                    if !occ_indices.contains(&idx_i) {
                                        occ_indices.push(idx_i);
                                    }
                                    if !virt_indices.contains(&idx_j) {
                                        virt_indices.push(idx_j);
                                    }
                                }
                            }
                        }

                        let ct_2 = ChargeTransferPair {
                            m_h: m_j.index,
                            m_l: m_i.index,
                            state_index: n,
                            state_energy: state_2.properties.ci_eigenvalue(n).unwrap(),
                            eigenvectors: state_2.properties.tdm(n).unwrap().to_owned(),
                            q_tr: q_ov_2.dot(&tdm_2),
                            tr_dipole: state_2.properties.tr_dipole(n).unwrap(),
                            occ_orb: m_j.slice.occ_orb,
                            virt_orb: m_i.slice.virt_orb,
                            occ_indices,
                            virt_indices,
                        };

                        ct_basis_temp.push(BasisState::PairCT(ct_1));
                        ct_basis_temp.push(BasisState::PairCT(ct_2));
                    }

                    ct_basis_temp
                })
                .collect();

            for ct_vec in ct_basis.iter_mut() {
                states.append(ct_vec);
            }
        }

        states
    }

    /// Build and diagonalize the FMO-LC-TDDFTB excitonic Hamiltonian in the
    /// reduced basis of monomer locally-excited (LE) and inter-monomer
    /// charge-transfer (CT) states, yielding the supersystem excited states.
    pub fn create_exciton_hamiltonian(&mut self) {
        // Calculate the H' matrix
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        // Number of LE states per monomer.
        let n_le: usize = self.config.fmo_lc_tddftb.n_le;
        let n_roots: usize = n_le + 3;

        let fock_matrix: ArrayView2<f64> = self.properties.lcmo_fock().unwrap();
        // Calculate the excited states of the monomers
        // Swap the orbital energies of the monomers with the elements of the H' matrix
        self.monomers.par_iter_mut().for_each(|mol| {
            mol.properties.set_orbe(
                fock_matrix
                    .slice(s![mol.slice.orb, mol.slice.orb])
                    .diag()
                    .to_owned(),
            );
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()], &self.config);
            mol.run_tda(
                &atoms[mol.slice.atom_as_range()],
                n_roots,
                self.config.excited.davidson_iterations,
                self.config.excited.davidson_convergence,
                self.config.excited.davidson_subspace_multiplier,
                true,
                &self.config,
            );
        });

        // Construct the basis states.
        let states: Vec<BasisState> = self.create_diabatic_basis(self.config.fmo_lc_tddftb.n_ct);

        let dim: usize = states.len();
        // Initialize the Exciton-Hamiltonian.
        let mut h = vec![0.0; dim * dim];

        // calculate the state couplings
        states
            .par_iter()
            .enumerate()
            .zip(h.par_chunks_exact_mut(dim))
            .for_each(|((i, state_i), h_i)| {
                states
                    .par_iter()
                    .enumerate()
                    .zip(h_i.par_iter_mut())
                    .for_each(|((j, state_j), h_ij)| {
                        if j >= i {
                            *h_ij = self.exciton_coupling(state_i, state_j);
                        }
                    });
            });
        let mut h: Array2<f64> = Array::from(h).into_shape((dim, dim)).unwrap();

        let energies: Array1<f64>;
        let eigvectors: Array2<f64>;

        // calculate all excited states
        if self.config.fmo_lc_tddftb.calculate_all_states {
            let (energies_tmp, eigvectors_tmp) = h.eigh(UPLO::Lower).unwrap();
            energies = energies_tmp;
            eigvectors = eigvectors_tmp;
        } else {
            // Use the davidson algorithm to obtain a limited number of eigenvalues
            let diag = h.diag();
            h = &h + &h.t() - Array::from_diag(&diag);
            let nroots: usize = self.config.excited.nstates;
            let guess: Array2<f64> = initial_subspace(h.diag(), nroots);
            let davidson: Davidson = Davidson::new(
                &mut h,
                guess,
                nroots,
                1e-4,
                200,
                true,
                self.config.excited.davidson_subspace_multiplier,
                false,
            )
            .unwrap();
            energies = davidson.eigenvalues;
            eigvectors = davidson.eigenvectors;
        }

        // Optionally write the adiabatic eigenvector coefficients of the
        // selected excited states to `.npy` files. Each eigenvector is a
        // column of `eigvectors` (the expansion coefficients of that
        // adiabatic state over the diabatic basis `states`); the squared
        // coefficients (weights of each diabatic basis state) are
        // orthonormal and sum to 1. State `idx` is written to
        // `adiabatic_coefficients_state_{idx}.npy`. The raw (signed)
        // coefficients are stored so they can seed an Ehrenfest run from
        // this adiabatic state with the correct relative phases.
        if self.config.fmo_lc_tddftb.save_adiabatic_coefficients {
            let n_states: usize = eigvectors.ncols();
            for &idx in self.config.fmo_lc_tddftb.selected_coefficients.iter() {
                if idx >= n_states {
                    eprintln!(
                        "Warning: selected_coefficients index {} is out of range \
                         (only {} adiabatic states available); skipping.",
                        idx, n_states
                    );
                    continue;
                }
                let coeffs: Array1<f64> = eigvectors.column(idx).to_owned();
                let filename = format!("adiabatic_coefficients_state_{}.npy", idx);
                match write_npy(&filename, &coeffs) {
                    Ok(()) => println!(
                        "Saved adiabatic coefficients of state {idx} to {filename} \
                         (Σc² = {:.6})",
                        coeffs.iter().map(|c| c * c).sum::<f64>()
                    ),
                    Err(e) => eprintln!("Failed to write {filename}: {e}"),
                }
            }
        }

        // get the number of occupied and virtual orbitals
        let n_occ: usize = self
            .monomers
            .iter()
            .map(|m| m.properties.n_occ().unwrap())
            .sum();
        let n_virt: usize = self
            .monomers
            .iter()
            .map(|m| m.properties.n_virt().unwrap())
            .sum();
        let n_orbs: usize = n_occ + n_virt;
        let mut occ_orbs: Array2<f64> = Array2::zeros([n_orbs, n_occ]);
        let mut virt_orbs: Array2<f64> = Array2::zeros([n_orbs, n_virt]);

        // get all occupide and virtual orbitals of the system
        for mol in self.monomers.iter() {
            let mol_orbs: ArrayView2<f64> = mol.properties.orbs().unwrap();
            let lumo: usize = mol.properties.lumo().unwrap();
            occ_orbs
                .slice_mut(s![mol.slice.orb, mol.slice.occ_orb])
                .assign(&mol_orbs.slice(s![.., ..lumo]));
            virt_orbs
                .slice_mut(s![mol.slice.orb, mol.slice.virt_orb])
                .assign(&mol_orbs.slice(s![.., lumo..]));
        }
        let orbs: Array2<f64> = concatenate![Axis(1), occ_orbs, virt_orbs];

        // calculate the oscillator strenghts of the excited states
        let exciton = ExcitonStates::new(
            self.properties.last_energy().unwrap(),
            (energies.clone(), eigvectors.clone()),
            states.clone(),
            (n_occ, n_virt),
            orbs,
            self.properties.s().unwrap(),
            &self.atoms,
        );
        // write the energies and oscillator strenghts to numpy files
        exciton.spectrum_to_npy("lcmo_spec.npy").unwrap();
        exciton.spectrum_to_txt("lcmo_spec.txt");
        println!("{}", exciton);

        if self.config.fmo_lc_tddftb.calculate_ntos {
            exciton.calculate_ntos_jmol(&self.config.fmo_lc_tddftb.states_to_analyse, &self.atoms);
        }
        if self.config.fmo_lc_tddftb.calculate_transition_densities {
            exciton.get_transition_densities(&self.config.fmo_lc_tddftb.states_to_analyse);
        }
        if self.config.fmo_lc_tddftb.calculate_particle_hole_densities {
            exciton.get_particle_hole_densities(
                &self.config.fmo_lc_tddftb.states_to_analyse,
                self.properties.s().unwrap(),
            );
        }
    }

    pub fn get_excitonic_matrix(&mut self) -> Array2<f64> {
        // Calculate the H' matrix
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        // Number of LE states per monomer.
        let n_le: usize = self.config.fmo_lc_tddftb.n_le;
        let n_roots: usize = n_le + 3;

        let fock_matrix: ArrayView2<f64> = self.properties.lcmo_fock().unwrap();
        // Calculate the excited states of the monomers
        // Swap the orbital energies of the monomers with the elements of the H' matrix
        self.monomers.par_iter_mut().for_each(|mol| {
            mol.properties.set_orbe(
                fock_matrix
                    .slice(s![mol.slice.orb, mol.slice.orb])
                    .diag()
                    .to_owned(),
            );
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()], &self.config);
            mol.run_tda(
                &atoms[mol.slice.atom_as_range()],
                n_roots,
                self.config.excited.davidson_iterations,
                self.config.excited.davidson_convergence,
                self.config.excited.davidson_subspace_multiplier,
                false,
                &self.config,
            );
        });

        // Construct the basis states.
        let states: Vec<BasisState> = self.create_diabatic_basis(self.config.fmo_lc_tddftb.n_ct);

        // dimension of the Hamiltonian
        let dim: usize = states.len();
        // Initialize the Exciton-Hamiltonian.
        let mut h = vec![0.0; dim * dim];

        // calculate the state couplings
        states
            .par_iter()
            .enumerate()
            .zip(h.par_chunks_exact_mut(dim))
            .for_each(|((i, state_i), h_i)| {
                states
                    .par_iter()
                    .enumerate()
                    .zip(h_i.par_iter_mut())
                    .for_each(|((j, state_j), h_ij)| {
                        if j >= i {
                            *h_ij = self.exciton_coupling(state_i, state_j);
                        }
                    });
            });

        let mut h: Array2<f64> = Array::from(h).into_shape((dim, dim)).unwrap();
        let diag = h.diag();
        h = &h + &h.t() - Array::from_diag(&diag);

        // Construct Reduced dibatic basis states
        let mut reduced_states: Vec<ReducedBasisState> = Vec::new();
        for (idx, state) in states.iter().enumerate() {
            match state {
                BasisState::LE(ref a) => {
                    // get index and the Atom vector of the monomer
                    let new_state = ReducedLE {
                        energy: h[[idx, idx]],
                        monomer_index: a.monomer.index,
                        state_index: a.n,
                        state_coefficient: 0.0,
                        homo: a.monomer.properties.homo().unwrap(),
                    };

                    reduced_states.push(ReducedBasisState::LE(new_state));
                }
                BasisState::PairCT(ref a) => {
                    reduced_states.push(ReducedBasisState::CT(a.to_owned()));
                }
            };
        }
        // save the basis in the properties
        self.properties.set_basis_states(reduced_states);
        h
    }

    pub fn get_tdm_for_ehrenfest(
        &mut self,
        coeffs: ArrayView1<f64>,
        step: usize,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // Calculate the H' matrix
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        // Number of LE states per monomer.
        let n_le: usize = self.config.fmo_lc_tddftb.n_le;
        let n_roots: usize = n_le + 3;

        let fock_matrix: ArrayView2<f64> = self.properties.lcmo_fock().unwrap();
        // Calculate the excited states of the monomers
        // Swap the orbital energies of the monomers with the elements of the H' matrix
        self.monomers.par_iter_mut().for_each(|mol| {
            mol.properties.set_orbe(
                fock_matrix
                    .slice(s![mol.slice.orb, mol.slice.orb])
                    .diag()
                    .to_owned(),
            );
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()], &self.config);
            mol.run_tda(
                &atoms[mol.slice.atom_as_range()],
                n_roots,
                self.config.excited.davidson_iterations,
                self.config.excited.davidson_convergence,
                self.config.excited.davidson_subspace_multiplier,
                false,
                &self.config,
            );
        });

        // Construct the basis states.
        let states: Vec<BasisState> = self.create_diabatic_basis(self.config.fmo_lc_tddftb.n_ct);

        // get the number of occupied and virtual orbitals
        let n_occ: usize = self
            .monomers
            .iter()
            .map(|m| m.properties.n_occ().unwrap())
            .sum();
        let n_virt: usize = self
            .monomers
            .iter()
            .map(|m| m.properties.n_virt().unwrap())
            .sum();
        let n_orbs: usize = n_occ + n_virt;
        let mut occ_orbs: Array2<f64> = Array2::zeros([n_orbs, n_occ]);
        let mut virt_orbs: Array2<f64> = Array2::zeros([n_orbs, n_virt]);

        // get all occupide and virtual orbitals of the system
        for mol in self.monomers.iter() {
            let mol_orbs: ArrayView2<f64> = mol.properties.orbs().unwrap();
            let lumo: usize = mol.properties.lumo().unwrap();
            occ_orbs
                .slice_mut(s![mol.slice.orb, mol.slice.occ_orb])
                .assign(&mol_orbs.slice(s![.., ..lumo]));
            virt_orbs
                .slice_mut(s![mol.slice.orb, mol.slice.virt_orb])
                .assign(&mol_orbs.slice(s![.., lumo..]));
        }
        let tdm: Array2<f64> =
            self.get_transition_density_matrix_from_coeffs(coeffs, (n_occ, n_virt), states);
        let tdm_ao: Array2<f64> = occ_orbs.dot(&tdm.dot(&virt_orbs.t()));

        // hole and particle densities
        let s: ArrayView2<f64> = self.properties.s().unwrap();
        let h_mat: Array2<f64> = tdm_ao.dot(&s.dot(&tdm_ao.t()));
        let p_mat: Array2<f64> = tdm_ao.t().dot(&s.dot(&tdm_ao));

        if self.config.tdm_config.store_tdm {
            let mut tmp_string: String = String::from("transition_density_");
            tmp_string.push_str(&step.to_string());
            tmp_string.push_str(".npy");
            write_npy(tmp_string, &tdm_ao).unwrap();
        }
        if self.config.tdm_config.store_hole_particle {
            let mut tmp_string: String = String::from("hole_density_");
            tmp_string.push_str(&step.to_string());
            tmp_string.push_str(".npy");
            write_npy(tmp_string, &h_mat).unwrap();

            let mut tmp_string: String = String::from("particle_density_");
            tmp_string.push_str(&step.to_string());
            tmp_string.push_str(".npy");
            write_npy(tmp_string, &p_mat).unwrap();
        }
        (tdm_ao, h_mat, p_mat)
    }

    pub fn get_tdm_for_ehrenfest_average_trajectory(&mut self, coeffs: ArrayView2<f64>) {
        // Calculate the H' matrix
        let hamiltonian = self.build_lcmo_fock_matrix();
        self.properties.set_lcmo_fock(hamiltonian);

        // Reference to the atoms of the total system.
        let atoms: &[Atom] = &self.atoms[..];
        // Number of LE states per monomer.
        let n_le: usize = self.config.fmo_lc_tddftb.n_le;
        let n_roots: usize = n_le + 3;

        let fock_matrix: ArrayView2<f64> = self.properties.lcmo_fock().unwrap();
        // Calculate the excited states of the monomers
        // Swap the orbital energies of the monomers with the elements of the H' matrix
        self.monomers.par_iter_mut().for_each(|mol| {
            mol.properties.set_orbe(
                fock_matrix
                    .slice(s![mol.slice.orb, mol.slice.orb])
                    .diag()
                    .to_owned(),
            );
            mol.prepare_tda(&atoms[mol.slice.atom_as_range()], &self.config);
            mol.run_tda(
                &atoms[mol.slice.atom_as_range()],
                n_roots,
                self.config.excited.davidson_iterations,
                self.config.excited.davidson_convergence,
                self.config.excited.davidson_subspace_multiplier,
                false,
                &self.config,
            );
        });

        // Construct the basis states.
        let states: Vec<BasisState> = self.create_diabatic_basis(self.config.fmo_lc_tddftb.n_ct);

        // get the number of occupied and virtual orbitals
        let n_occ: usize = self
            .monomers
            .iter()
            .map(|m| m.properties.n_occ().unwrap())
            .sum();
        let n_virt: usize = self
            .monomers
            .iter()
            .map(|m| m.properties.n_virt().unwrap())
            .sum();
        let n_orbs: usize = n_occ + n_virt;
        let mut occ_orbs: Array2<f64> = Array2::zeros([n_orbs, n_occ]);
        let mut virt_orbs: Array2<f64> = Array2::zeros([n_orbs, n_virt]);

        // get all occupide and virtual orbitals of the system
        for mol in self.monomers.iter() {
            let mol_orbs: ArrayView2<f64> = mol.properties.orbs().unwrap();
            let lumo: usize = mol.properties.lumo().unwrap();
            occ_orbs
                .slice_mut(s![mol.slice.orb, mol.slice.occ_orb])
                .assign(&mol_orbs.slice(s![.., ..lumo]));
            virt_orbs
                .slice_mut(s![mol.slice.orb, mol.slice.virt_orb])
                .assign(&mol_orbs.slice(s![.., lumo..]));
        }

        // create step vector
        let mut step_vec: Vec<usize> = Vec::new();
        for (count, step) in (0..self.config.tdm_config.total_steps).enumerate() {
            if count.rem_euclid(self.config.tdm_config.calculate_nth_step) == 0 {
                step_vec.push(step);
            }
        }

        if self.config.tdm_config.use_parallelization {
            step_vec.par_iter().enumerate().for_each(|(idx, step)| {
                let coeff: ArrayView1<f64> = coeffs.slice(s![idx, ..]);
                let tdm: Array2<f64> = self.get_transition_density_matrix_from_coeffs(
                    coeff,
                    (n_occ, n_virt),
                    states.clone(),
                );
                let tdm_ao: Array2<f64> = occ_orbs.dot(&tdm.dot(&virt_orbs.t()));

                // hole and particle densities
                let s: ArrayView2<f64> = self.properties.s().unwrap();
                let h_mat: Array2<f64> = tdm_ao.dot(&s.dot(&tdm_ao.t()));
                let p_mat: Array2<f64> = tdm_ao.t().dot(&s.dot(&tdm_ao));

                if self.config.tdm_config.store_tdm {
                    let mut tmp_string: String = String::from("transition_density_");
                    tmp_string.push_str(&step.to_string());
                    tmp_string.push_str(".npy");
                    write_npy(tmp_string, &tdm_ao).unwrap();
                }
                if self.config.tdm_config.store_hole_particle {
                    let mut tmp_string: String = String::from("hole_density_");
                    tmp_string.push_str(&step.to_string());
                    tmp_string.push_str(".npy");
                    write_npy(tmp_string, &h_mat).unwrap();

                    let mut tmp_string: String = String::from("particle_density_");
                    tmp_string.push_str(&step.to_string());
                    tmp_string.push_str(".npy");
                    write_npy(tmp_string, &p_mat).unwrap();
                }

                if self.config.tdm_config.calc_cube {
                    self.density_from_tdm(h_mat.view(), idx, "_h");
                    self.density_from_tdm(p_mat.view(), idx, "_p");
                }
                if self.config.tdm_config.calc_tdm_cube {
                    self.density_from_tdm(tdm.view(), idx, "_tdm");
                }
            });
        } else {
            for (idx, step) in step_vec.iter().enumerate() {
                let coeff: ArrayView1<f64> = coeffs.slice(s![idx, ..]);
                let tdm: Array2<f64> = self.get_transition_density_matrix_from_coeffs(
                    coeff,
                    (n_occ, n_virt),
                    states.clone(),
                );
                let tdm_ao: Array2<f64> = occ_orbs.dot(&tdm.dot(&virt_orbs.t()));

                // hole and particle densities
                let s: ArrayView2<f64> = self.properties.s().unwrap();
                let h_mat: Array2<f64> = tdm_ao.dot(&s.dot(&tdm_ao.t()));
                let p_mat: Array2<f64> = tdm_ao.t().dot(&s.dot(&tdm_ao));

                if self.config.tdm_config.store_tdm {
                    let mut tmp_string: String = String::from("transition_density_");
                    tmp_string.push_str(&step.to_string());
                    tmp_string.push_str(".npy");
                    write_npy(tmp_string, &tdm_ao).unwrap();
                }
                if self.config.tdm_config.store_hole_particle {
                    let mut tmp_string: String = String::from("hole_density_");
                    tmp_string.push_str(&step.to_string());
                    tmp_string.push_str(".npy");
                    write_npy(tmp_string, &h_mat).unwrap();

                    let mut tmp_string: String = String::from("particle_density_");
                    tmp_string.push_str(&step.to_string());
                    tmp_string.push_str(".npy");
                    write_npy(tmp_string, &p_mat).unwrap();
                }

                if self.config.tdm_config.calc_cube {
                    self.density_from_tdm(h_mat.view(), idx, "_h");
                    self.density_from_tdm(p_mat.view(), idx, "_p");
                }
                if self.config.tdm_config.calc_tdm_cube {
                    self.density_from_tdm(tdm.view(), idx, "_tdm");
                }
            }
        }
    }
}

/// Different types of diabatic basis states that are used for the FMO-exciton model.
#[derive(Clone, Debug)]
pub enum BasisState<'a> {
    // Locally excited state that is on one monomer.
    LE(LocallyExcited<'a>),
    // Charge transfer state between two different monomers and two MOs.
    PairCT(ChargeTransferPair),
}

impl Display for BasisState<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BasisState::LE(state) => write!(f, "{}", state),
            BasisState::PairCT(state) => write!(f, "{}", state),
        }
    }
}

/// Type that holds all the relevant data that characterize a locally excited diabatic basis state.
#[derive(Clone, Debug)]
pub struct LocallyExcited<'a> {
    // Reference to the corresponding monomer.
    pub monomer: &'a Monomer<'a>,
    // Number of excited state for the monomer. 1 -> S1, 2 -> S2, ...
    pub n: usize,
    // The atoms corresponding to the monomer of this state.
    pub atoms: &'a [Atom],
    //
    pub q_trans: Array1<f64>,
    //
    pub occs: ArrayView2<'a, f64>,
    //
    pub virts: ArrayView2<'a, f64>,
    //
    pub tdm: ArrayView1<'a, f64>,
    //
    pub tr_dipole: Vector3<f64>,
    //
    pub occ_indices: Vec<usize>,
    pub virt_indices: Vec<usize>,
}

impl Display for LocallyExcited<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LE(S{}) on Frag. {:>4}",
            self.n + 1,
            self.monomer.index + 1
        )
    }
}

impl PartialEq for LocallyExcited<'_> {
    /// Two LE states are considered equal, if it is the same excited state on the same monomer.
    fn eq(&self, other: &Self) -> bool {
        self.monomer.index == other.monomer.index && self.n == other.n
    }
}

#[derive(Clone, Debug)]
pub struct ChargeTransferPreparation<'a> {
    pub m_h: &'a Monomer<'a>,
    pub m_l: &'a Monomer<'a>,
    pub pair_type: PairType,
    pub properties: Properties,
    /// BLAS-optimized workspace for exchange integrals in Davidson iterations.
    /// Initialized by prepare_ct_tda when LC correction is enabled.
    pub davidson_workspace: Option<CTDavidsonWorkspace>,
}

pub use dialect_state::basis_states::{ChargeTransferPair, ReducedBasisState, ReducedLE};