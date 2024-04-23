use crate::constants::BOHR_TO_ANGS;
use crate::fmo::{
    pair, ChargeTransferPair, ChargeTransferPreparation, GroundStateGradient, Monomer, PairType,
    SuperSystem,
};
use crate::gradients::numerical::{
    assert_deriv, assert_deriv_ct_grad_full, assert_deriv_le_grad, assert_deriv_le_grad_full,
};
use crate::initialization::Atom;
use crate::properties::Properties;
use crate::scc::scc_routine::RestrictedSCC;
use ndarray::prelude::*;
use std::net::UdpSocket;
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
        self.run_scc();
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
        };
        // prepare the TDA calculation of both states
        state_1.prepare_ct_tda(
            self.properties.gamma().unwrap(),
            self.properties.gamma_lr().unwrap(),
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
        self.run_scc();
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
        };
        // prepare the TDA calculation of both states
        state_1.prepare_ct_tda(
            self.properties.gamma().unwrap(),
            self.properties.gamma_lr().unwrap(),
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
        self.run_scc();
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
        };
        // prepare the TDA calculation of both states
        state_1.prepare_ct_tda(
            self.properties.gamma().unwrap(),
            self.properties.gamma_lr().unwrap(),
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
        self.run_scc();

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
        self.run_scc();

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
        self.run_scc();

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
        self.run_scc();

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
        self.run_scc();

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
        self.run_scc();

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
        self.run_scc();

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
        self.run_scc();

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
        assert_deriv(
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
