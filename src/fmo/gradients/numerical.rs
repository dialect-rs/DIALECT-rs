#![allow(dead_code)]
#![allow(warnings)]
use crate::fmo::{ChargeTransferPair, ChargeTransferPreparation, Monomer, PairType, SuperSystem};
use crate::gradients::numerical::{
    assert_deriv, assert_deriv_5point, assert_deriv_ct_grad_full, assert_deriv_le_grad,
    assert_deriv_le_grad_full,
};
use crate::properties::Properties;
use crate::scc::scc_routine::RestrictedSCC;
use ndarray::prelude::*;
use std::time::Instant;

impl SuperSystem<'_> {
    pub fn new_fmo_ct_energy_wrapper(
        &mut self,
        geometry: Array1<f64>,
        monomer_index_i: usize,
        monomer_index_j: usize,
    ) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for pair in self.esd_pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();
        let _ = self.run_scc().unwrap();
        for mol in self.monomers.iter_mut() {
            mol.prepare_excited_gradient(&self.atoms[mol.slice.atom_as_range()]);
        }
        let m_h: &Monomer = &self.monomers[monomer_index_i];
        let m_l: &Monomer = &self.monomers[monomer_index_j];
        let type_ij: PairType = self
            .properties
            .type_of_pair(monomer_index_i, monomer_index_j);

        // create CT states
        let mut state_1 = ChargeTransferPreparation {
            m_h: m_h,
            m_l: m_l,
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
            &self.atoms,
            &self.config,
        );
        state_1.run_ct_tda(&self.atoms, 10, 150, 1.0e-4, 10, &self.config);

        let val: f64 = state_1.properties.ci_eigenvalue(0).unwrap();
        // let val = self.exciton_hamiltonian_ct_test();
        return val;
    }

    pub fn new_ct_gradient_wrapper(
        &mut self,
        monomer_index_i: usize,
        monomer_index_j: usize,
    ) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for pair in self.esd_pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let _ = self.run_scc().unwrap();
        let _hamiltonian = self.build_lcmo_fock_matrix();
        for mol in self.monomers.iter_mut() {
            mol.prepare_excited_gradient(&self.atoms[mol.slice.atom_as_range()]);
        }

        let m_h: &Monomer = &self.monomers[monomer_index_i];
        let m_l: &Monomer = &self.monomers[monomer_index_j];
        let type_ij: PairType = self
            .properties
            .type_of_pair(monomer_index_i, monomer_index_j);
        let threshold_ct: f64 = self.config.fmo_lc_tddftb.active_space_threshold_ct;

        // create CT states
        let mut state_1 = ChargeTransferPreparation {
            m_h: m_h,
            m_l: m_l,
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
            &self.atoms,
            &self.config,
        );
        state_1.run_ct_tda(&self.atoms, 5, 150, 1.0e-4, 10, &self.config);
        let q_ov_1: ArrayView2<f64> = state_1.properties.q_ov().unwrap();
        let tdm_1: ArrayView1<f64> = state_1.properties.ci_coefficient(0).unwrap();
        let tdm_dim2: ArrayView2<f64> = state_1.properties.tdm(0).unwrap();

        // determine the relevant orbital indices
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        for (idx_i, val_i) in tdm_dim2.outer_iter().enumerate() {
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
            m_h: m_h.index,
            m_l: m_l.index,
            state_index: 0,
            state_energy: state_1.properties.ci_eigenvalue(0).unwrap(),
            eigenvectors: state_1.properties.tdm(0).unwrap().to_owned(),
            q_tr: q_ov_1.dot(&tdm_1),
            tr_dipole: state_1.properties.tr_dipole(0).unwrap(),
            occ_orb: m_h.slice.occ_orb.clone(),
            virt_orb: m_l.slice.virt_orb.clone(),
            occ_indices,
            virt_indices,
        };
        drop(m_h);
        drop(m_l);
        let grad = self.charge_transfer_pair_gradient(&ct_1);
        let m_h: &Monomer = &self.monomers[monomer_index_i];
        let m_l: &Monomer = &self.monomers[monomer_index_j];

        let mut full_gradient: Array1<f64> = Array1::zeros(self.atoms.len() * 3);
        if m_h.index < m_l.index {
            full_gradient
                .slice_mut(s![m_h.slice.grad])
                .assign(&grad.slice(s![..m_h.n_atoms * 3]));
            full_gradient
                .slice_mut(s![m_l.slice.grad])
                .assign(&grad.slice(s![m_h.n_atoms * 3..]));
        } else {
            full_gradient
                .slice_mut(s![m_l.slice.grad])
                .assign(&grad.slice(s![..m_l.n_atoms * 3]));
            full_gradient
                .slice_mut(s![m_h.slice.grad])
                .assign(&grad.slice(s![m_l.n_atoms * 3..]));
        }

        return full_gradient;
    }

    pub fn new_ct_gradient_wrapper_full_system(
        &mut self,
        monomer_index_i: usize,
        monomer_index_j: usize,
    ) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for pair in self.esd_pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let _ = self.run_scc().unwrap();
        let _hamiltonian = self.build_lcmo_fock_matrix();
        for mol in self.monomers.iter_mut() {
            mol.prepare_excited_gradient(&self.atoms[mol.slice.atom_as_range()]);
        }

        let m_h: &Monomer = &self.monomers[monomer_index_i];
        let m_l: &Monomer = &self.monomers[monomer_index_j];
        let type_ij: PairType = self
            .properties
            .type_of_pair(monomer_index_i, monomer_index_j);
        let threshold_ct: f64 = self.config.fmo_lc_tddftb.active_space_threshold_ct;

        // create CT states
        let mut state_1 = ChargeTransferPreparation {
            m_h: m_h,
            m_l: m_l,
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
            &self.atoms,
            &self.config,
        );
        state_1.run_ct_tda(&self.atoms, 5, 150, 1.0e-4, 10, &self.config);
        let q_ov_1: ArrayView2<f64> = state_1.properties.q_ov().unwrap();
        let tdm_1: ArrayView1<f64> = state_1.properties.ci_coefficient(0).unwrap();
        let tdm_dim2: ArrayView2<f64> = state_1.properties.tdm(0).unwrap();

        // determine the relevant orbital indices
        let mut occ_indices: Vec<usize> = Vec::new();
        let mut virt_indices: Vec<usize> = Vec::new();
        for (idx_i, val_i) in tdm_dim2.outer_iter().enumerate() {
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
            m_h: m_h.index,
            m_l: m_l.index,
            state_index: 0,
            state_energy: state_1.properties.ci_eigenvalue(0).unwrap(),
            eigenvectors: state_1.properties.tdm(0).unwrap().to_owned(),
            q_tr: q_ov_1.dot(&tdm_1),
            tr_dipole: state_1.properties.tr_dipole(0).unwrap(),
            occ_orb: m_h.slice.occ_orb.clone(),
            virt_orb: m_l.slice.virt_orb.clone(),
            occ_indices,
            virt_indices,
        };
        drop(m_h);
        drop(m_l);
        let grad = self.charge_transfer_pair_gradient(&ct_1);
        let m_h: &Monomer = &self.monomers[monomer_index_i];
        let m_l: &Monomer = &self.monomers[monomer_index_j];

        grad
    }

    pub fn test_new_charge_transfer_gradient(
        &mut self,
        monomer_index_i: usize,
        monomer_index_j: usize,
    ) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        assert_deriv_le_grad(
            self,
            SuperSystem::new_fmo_ct_energy_wrapper,
            SuperSystem::new_ct_gradient_wrapper,
            self.get_xyz(),
            0.01,
            1e-6,
            monomer_index_i,
            monomer_index_j,
        );
    }

    pub fn test_new_charge_transfer_gradient_full_system(&mut self, monomer_index_i: usize) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        let mol_i = self.monomers[monomer_index_i].clone();

        for mol in self.monomers.clone().iter() {
            if mol.index != monomer_index_i {
                assert_deriv_ct_grad_full(
                    self,
                    SuperSystem::new_fmo_ct_energy_wrapper,
                    SuperSystem::new_ct_gradient_wrapper_full_system,
                    self.get_xyz(),
                    0.01,
                    1e-6,
                    monomer_index_i,
                    mol.index,
                    mol_i.slice.grad,
                    mol.slice.grad,
                );
            }
        }
    }

    pub fn fmo_le_energy_wrapper(
        &mut self,
        geometry: Array1<f64>,
        monomer_index: usize,
        state_index: usize,
    ) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        let val: f64 = self.exciton_le_energy(monomer_index, state_index);

        return val;
    }

    pub fn fmo_le_gradient_wrapper(
        &mut self,
        monomer_index: usize,
        state_index: usize,
    ) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        // calculate the gradient of the le_energy
        let grad: Array1<f64> = self.exciton_le_gradient(monomer_index, state_index);

        let mut full_gradient: Array1<f64> = Array1::zeros(self.atoms.len() * 3);
        let mol = &self.monomers[monomer_index];
        full_gradient.slice_mut(s![mol.slice.grad]).assign(&grad);

        return full_gradient;
    }

    pub fn test_le_gradient(&mut self, monomer_index: usize, state_index: usize) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        assert_deriv_le_grad(
            self,
            SuperSystem::fmo_le_energy_wrapper,
            SuperSystem::fmo_le_gradient_wrapper,
            self.get_xyz(),
            0.01,
            1e-6,
            monomer_index,
            state_index,
        );
    }

    pub fn test_fmo_le_energy_wrapper(
        &mut self,
        geometry: Array1<f64>,
        monomer_index: usize,
        state_index: usize,
    ) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        let val: f64 = self.exciton_le_energy(monomer_index, state_index);

        return val;
    }

    pub fn test_fmo_le_gradient_wrapper(
        &mut self,
        monomer_index: usize,
        state_index: usize,
    ) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        // calculate the gradient of the le_energy
        let grad: Array1<f64> = self.exciton_le_gradient(monomer_index, state_index);

        return grad;
    }

    pub fn test_le_gradient_all_monomers(&mut self, state_index: usize) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let _ = self.run_scc().unwrap();

        for mol in self.monomers.clone().iter() {
            assert_deriv_le_grad_full(
                self,
                SuperSystem::test_fmo_le_energy_wrapper,
                SuperSystem::test_fmo_le_gradient_wrapper,
                self.get_xyz(),
                0.01,
                1e-6,
                mol.index,
                state_index,
                mol.slice.grad,
            );
        }
    }

    pub fn total_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (monomer_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        let pair_energy: f64 = self.pair_scc(dq.view());
        self.properties.set_dq(dq);
        let emb_energy: f64 = self.embedding_energy();
        let esd_energy: f64 = self.esd_pair_energy();

        monomer_energy + pair_energy + esd_energy + emb_energy
    }

    pub fn test_total_gradient(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let maxiter: usize = self.config.scf.scf_max_cycles;
        let (monomer_energy, dq): (f64, Array1<f64>) = self.monomer_scc(maxiter);
        let pair_energy: f64 = self.pair_scc(dq.view());
        self.properties.set_dq(dq);
        let emb_energy: f64 = self.embedding_energy();
        let esd_energy: f64 = self.esd_pair_energy();
        println!("FMO MONOMER {}", monomer_energy);
        println!("FMO PAIR {}", pair_energy);
        println!("FMO ESD {}", esd_energy);
        println!("FMO EMB {}", emb_energy);
        println!(
            "FMO ENERGY WITHOUT EMBEDDING {}",
            monomer_energy + pair_energy + esd_energy
        );
        println!(
            "FMO ENERGY {}",
            monomer_energy + pair_energy + emb_energy + esd_energy
        );
        assert_deriv(
            self,
            SuperSystem::total_energy_wrapper,
            SuperSystem::ground_state_gradient,
            self.get_xyz(),
            0.01,
            1e-6,
        );
    }

    pub fn gs_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();

        gs_energy
    }

    pub fn gs_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();
        let timer = Instant::now();
        let gs_grad = self.ground_state_gradient();
        println!(
            "Time ground-state gradient: {:.5}",
            timer.elapsed().as_secs_f32()
        );
        drop(timer);
        // let timer = Instant::now();
        // let response_grad = self.solve_response_gradient();
        // println!(
        //     "Time response ground-state gradient: {:.5}",
        //     timer.elapsed().as_secs_f32()
        // );

        gs_grad // + response_grad
    }

    pub fn test_gs_gradient(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        assert_deriv_5point(
            self,
            SuperSystem::gs_energy_wrapper,
            SuperSystem::gs_gradient_wrapper,
            self.get_xyz(),
            0.01,
            1e-6,
        );
    }

    pub fn monomer_energies_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        // Do the self-consistent monomer calculations
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);

        monomer_energies
    }

    pub fn pair_energies_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        // Do the self-consistent monomer calculations
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);

        // Do the SCC-calculation for each pair individually
        let pair_energies: f64 = self.pair_scc(dq.view());

        pair_energies
    }

    pub fn es_dim_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        // Do the self-consistent monomer calculations
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);

        // Do the SCC-calculation for each pair individually
        let pair_energies: f64 = self.pair_scc(dq.view());

        // Compute the embedding energy from all pairs
        let embedding: f64 = self.embedding_energy();

        // Compute the energy from pairs that are far apart. The electrostatic dimer approximation
        // is used in this case.
        let esd_pair_energies: f64 = self.esd_pair_energy();

        esd_pair_energies
    }

    pub fn embedding_energy_wrapper(&mut self, geometry: Array1<f64>) -> f64 {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.update_xyz(geometry.view());
        self.prepare_scc();

        let max_iter: usize = self.config.scf.scf_max_cycles;
        // Do the self-consistent monomer calculations
        let (monomer_energies, dq): (f64, Array1<f64>) = self.monomer_scc(max_iter);

        // Do the SCC-calculation for each pair individually
        let pair_energies: f64 = self.pair_scc(dq.view());

        // Compute the embedding energy from all pairs
        let embedding: f64 = self.embedding_energy();

        embedding
    }

    pub fn monomer_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();

        let monomer_gradient: Array1<f64> = self.monomer_gradients();

        monomer_gradient
    }

    pub fn pair_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();

        let monomer_gradient: Array1<f64> = self.monomer_gradients();
        let pair_gradient: Array1<f64> = self.pair_gradients(monomer_gradient.view());
        // let pair_gradient: Array1<f64> = self.pair_gradients_for_testing();

        pair_gradient
    }

    pub fn es_dim_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();

        let monomer_gradient: Array1<f64> = self.monomer_gradients();
        let pair_gradient: Array1<f64> = self.pair_gradients(monomer_gradient.view());

        let esd_gradient: Array1<f64> = self.es_dimer_gradient();
        // let esd_gradient2: Array1<f64> = self.es_dimer_gradient_test();
        // let diff = &esd_gradient - &esd_gradient2;
        // println!("ESDIM diff: {:.8}", diff);

        esd_gradient
    }

    pub fn embedding_gradient_wrapper(&mut self) -> Array1<f64> {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();

        let monomer_gradient: Array1<f64> = self.monomer_gradients();
        let pair_gradient: Array1<f64> = self.pair_gradients(monomer_gradient.view());

        // let embedding_gradient: Array1<f64> = self.embedding_gradient();
        let embedding_gradient: Array1<f64> = self.embedding_gradient();

        embedding_gradient
    }

    pub fn test_gs_gradient_in_parts(&mut self) {
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        println!("\n Test the monomer energy gradient: \n");
        assert_deriv(
            self,
            SuperSystem::monomer_energies_wrapper,
            SuperSystem::monomer_gradient_wrapper,
            self.get_xyz(),
            0.01,
            1e-6,
        );

        println!("\n Test the pair energy gradient: \n");
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        assert_deriv(
            self,
            SuperSystem::pair_energies_wrapper,
            SuperSystem::pair_gradient_wrapper,
            self.get_xyz(),
            0.01,
            1e-6,
        );

        println!("\n Test the esdim energy gradient: \n");
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        assert_deriv(
            self,
            SuperSystem::es_dim_energy_wrapper,
            SuperSystem::es_dim_gradient_wrapper,
            self.get_xyz(),
            0.01,
            1e-6,
        );

        println!("\n Test the embedding energy gradient: \n");
        self.properties.reset();
        for mol in self.monomers.iter_mut() {
            mol.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        assert_deriv(
            self,
            SuperSystem::embedding_energy_wrapper,
            SuperSystem::embedding_gradient_wrapper,
            self.get_xyz(),
            0.01,
            1e-6,
        );
    }
}

#[cfg(test)]
mod fmo_dftb_gradient_tests {
    use crate::fmo::SuperSystem;
    use crate::initialization::parameter_handling::generate_parameters;
    use crate::io::Configuration;
    use dialect_base::get_path_prefix;
    use dialect_utilities::numerical::assert_deriv_5point;
    use xyz_parser::parse_xyz_file;

    /// Finite-difference step (Bohr) for the 5-point numerical gradient.
    const STEP: f64 = 1.0e-2;
    /// Largest tolerated deviation between the analytical and numerical
    /// FMO-DFTB ground-state gradient (Hartree/Bohr).
    const TOLERANCE: f64 = 1.0e-6;

    /// Non-HOP FMO-DFTB (DFTB2, no long-range correction) configuration using
    /// the ob2 Slater-Koster set, a tightly converged SCC and the same Broyden
    /// mixer settings as the FMO-xTB gradient test.
    fn fmo_dftb_config() -> Configuration {
        let mut config: Configuration = toml::from_str("").unwrap();
        config.tight_binding.use_dftb = true;
        config.tight_binding.use_xtb1 = false;
        config.tight_binding.use_gaussian_gamma = false;
        config.slater_koster.use_external_skf = true;
        config.slater_koster.skf_directory =
            format!("{}/tests/data/slako/ob2-1-1-split", get_path_prefix());
        config.lc.long_range_correction = false;
        config.scf.scf_max_cycles = 250;
        // Same SCC convergence as the FMO-xTB test; electronic temperature is
        // left at its default of 0 K for DFTB.
        config.scf.scf_charge_conv = 1.0e-13;
        config.scf.scf_energy_conv = 1.0e-13;
        config.mol.charge = 0;
        config.mol.multiplicity = 1;
        config.fmo.use_fmo = true;
        config.fmo.covalent_fragmentation = false;
        config.fmo.vdw_scaling = 2.0;
        config.broyden.alpha = 0.4;
        config.broyden.omega0 = 0.01;
        config.broyden.memory = 20;
        config.broyden.safeguard_factor = 1.0;
        config
    }

    /// The analytical FMO-DFTB ground-state gradient must agree with a 5-point
    /// numerical gradient for a 20-molecule water cluster (monomer, pair,
    /// embedding and ESD contributions across many non-covalent fragments).
    #[test]
    fn fmo_dftb_gradient_accuracy_water20() {
        let path = format!("{}/tests/data/water_20/water_20.xyz", get_path_prefix());
        let frame = parse_xyz_file(&path).unwrap();
        let config = fmo_dftb_config();
        let (slako, vrep, atoms, unique_atoms) = generate_parameters(frame, config.clone());
        let mut system = SuperSystem::from((config, &slako, &vrep, unique_atoms, atoms));

        let origin = system.get_xyz();
        assert_deriv_5point(
            &mut system,
            SuperSystem::gs_energy_wrapper,
            SuperSystem::gs_gradient_wrapper,
            origin,
            STEP,
            TOLERANCE,
        );
    }
}
