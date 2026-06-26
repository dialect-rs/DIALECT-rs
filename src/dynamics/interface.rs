use std::ops::AddAssign;
use super::output::{print_dyn_dftb, print_dyn_timings_ehrenfest};
use crate::fmo::{Monomer, SuperSystem};
use crate::initialization::old_system::OldSystem;
use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;
use dialect_dynamics::interface::QCInterface;
use ndarray::prelude::*;
use ndarray_linalg::c64;
use std::time::Instant;

impl QCInterface for System {
    // Return enegies, forces, non-adiabatic coupling and the transition dipole
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state: usize,
        dt: f64,
        state_coupling: bool,
        use_nacv_couplings: bool,
        gs_dynamic: bool,
        step: usize,
        nstates: usize,
    ) -> (
        Array1<f64>,
        Array2<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
        Option<Vec<Array1<f64>>>,
    ) {
        // timer
        let timer: Instant = Instant::now();
        // reset old properties
        self.properties.reset_reduced();

        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * self.n_atoms).unwrap());

        // system time
        let system_time: f32 = timer.elapsed().as_secs_f32();

        // calculate the energy and the gradient of the state
        let (energies, gradient): (Array1<f64>, Array1<f64>) =
            self.calculate_energies_and_gradient(state, state_coupling, gs_dynamic);
        let gradient: Array2<f64> = gradient.into_shape([self.n_atoms, 3]).unwrap();

        // energy and gradient
        let energy_gradient_time: f32 = timer.elapsed().as_secs_f32();
        let use_nacv: bool = use_nacv_couplings;

        // calculate the scalar couplings
        let (couplings, olap, nacv): (
            Option<Array2<f64>>,
            Option<Array2<f64>>,
            Option<Vec<Array1<f64>>>,
        ) = if state_coupling && !gs_dynamic && use_nacv {
            // prepare properties for nacv calculation
            // self.prepare_excited_grad();
            // calculate nacvs
            let (nacv, vectors): (Array2<f64>, Vec<Array1<f64>>) =
                self.get_nonadiabatic_vector_coupling(velocities, nstates);

            // set the old system
            let old_system: OldSystem = crate::initialization::old_system::new_old_system(self, None, Some(vectors.clone()));
            self.properties.set_old_system(old_system);

            // get the overlap coupling matrix
            let s_coupl: Array2<f64> = Array::eye(nstates) + &nacv * dt;

            (Some(nacv), Some(s_coupl), Some(vectors))
        } else if state_coupling && !gs_dynamic && !use_nacv {
            let (couplings, olap): (Array2<f64>, Array2<f64>) =
                self.get_scalar_coupling(dt, step);
            let mut couplings_mat: Array2<f64> = Array2::zeros(couplings.raw_dim());
            couplings_mat
                .slice_mut(s![1.., 1..])
                .assign(&couplings.slice(s![1.., 1..]));

            (Some(couplings_mat), Some(olap), None)
        } else {
            (None, None, None)
        };

        // nacme time
        let nacme_time: f32 = timer.elapsed().as_secs_f32();
        let full_time: f32 = timer.elapsed().as_secs_f32();
        print_dyn_dftb(system_time, energy_gradient_time, nacme_time, full_time);

        // slice the energies
        let energies: Array1<f64> = energies.slice(s![..nstates]).to_owned();

        (energies, gradient, couplings, olap, nacv)
    }

    fn recompute_gradient(&mut self, _coordinates: ArrayView2<f64>, state: usize) -> Array2<f64> {
        // reset old properties
        self.properties.reset_reduced();
        // calculate the energy and the gradient of the state
        let (_energies, gradient): (Array1<f64>, Array1<f64>) =
            self.calculate_energies_and_gradient(state, true, false);
        let gradient: Array2<f64> = gradient.into_shape([self.n_atoms, 3]).unwrap();

        gradient
    }

    fn compute_ehrenfest(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state_coefficients: ArrayView1<c64>,
        thresh: f64,
        _dt: f64,
        _step: usize,
        _use_state_couplings: bool,
        _use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>) {
        // timer
        let timer: Instant = Instant::now();
        // reset properties
        self.properties.reset_reduced();
        // get the number of states
        let nstates: usize = state_coefficients.len();
        let populations: Array1<f64> = state_coefficients.map(|val| val.norm_sqr());

        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * self.n_atoms).unwrap());

        // system time
        let system_time: f32 = timer.elapsed().as_secs_f32();

        // calculate the energy and the gradient of the state
        let (energies, mut gradient): (Array1<f64>, Array1<f64>) =
            self.calculate_energies_and_gradient_ehrenfest(populations.view(), thresh);
        // slice the energies
        let energies: Array1<f64> = energies.slice(s![..nstates]).to_owned();

        // energy and gradient
        let energy_gradient_time: f32 = timer.elapsed().as_secs_f32();

        let (nacv, vectors): (Array2<f64>, Vec<Array1<f64>>) =
            self.get_nonadiabatic_vector_coupling(velocities, nstates);

        // get the gradient contribution of the nacmes
        gradient = gradient
            - self.calculate_ehrenfest_gradient_nacmes(
                energies.view(),
                &vectors,
                state_coefficients,
                thresh,
            );
        let gradient: Array2<f64> = gradient.into_shape([self.n_atoms, 3]).unwrap();

        // set the old system
        let old_system: OldSystem = crate::initialization::old_system::new_old_system(self, None, Some(vectors));
        self.properties.set_old_system(old_system);

        // nacme time
        let nacme_time: f32 = timer.elapsed().as_secs_f32();

        // get 2d array from energies
        let energy_hamiltonian: Array2<f64> = Array2::from_diag(&energies);

        // full timings
        let full_time: f32 = timer.elapsed().as_secs_f32();
        print_dyn_dftb(system_time, energy_gradient_time, nacme_time, full_time);

        (energies[0], gradient, energy_hamiltonian, nacv)
    }

    fn compute_ehrenfest_tab(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state_coefficients: ArrayView1<c64>,
        thresh: f64,
        _tab_grad_threshold: f64,
        _dt: f64,
        _step: usize,
        _use_state_couplings: bool,
        _use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        // timer
        let timer: Instant = Instant::now();
        // reset properties
        self.properties.reset_reduced();
        // get the number of states
        let nstates: usize = state_coefficients.len();
        let populations: Array1<f64> = state_coefficients.map(|val| val.norm_sqr());

        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * self.n_atoms).unwrap());

        // system time
        let system_time: f32 = timer.elapsed().as_secs_f32();

        // calculate the energy and the gradient of the state
        let (energies, mut gradient, gradients): (Array1<f64>, Array1<f64>, Array2<f64>) =
            self.calculate_energies_and_gradient_ehrenfest_tab(populations.view(), thresh);
        // slice the energies
        let energies: Array1<f64> = energies.slice(s![..nstates]).to_owned();

        // energy and gradient
        let energy_gradient_time: f32 = timer.elapsed().as_secs_f32();

        // calculate the vector couplings
        let (nacv, vectors): (Array2<f64>, Vec<Array1<f64>>) =
            self.get_nonadiabatic_vector_coupling(velocities, nstates);

        // get the gradient contribution of the nacmes
        gradient = gradient
            - self.calculate_ehrenfest_gradient_nacmes(
                energies.view(),
                &vectors,
                state_coefficients,
                thresh,
            );
        let gradient: Array2<f64> = gradient.into_shape([self.n_atoms, 3]).unwrap();

        // set the old system
        let old_system: OldSystem = crate::initialization::old_system::new_old_system(self, None, Some(vectors));
        self.properties.set_old_system(old_system);

        // nacme time
        let nacme_time: f32 = timer.elapsed().as_secs_f32();

        // get 2d array from energies
        let energy_hamiltonian: Array2<f64> = Array2::from_diag(&energies);

        // full timings
        let full_time: f32 = timer.elapsed().as_secs_f32();
        print_dyn_dftb(system_time, energy_gradient_time, nacme_time, full_time);

        (energies[0], gradient, energy_hamiltonian, nacv, gradients)
    }
}

impl QCInterface for SuperSystem<'_> {
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state: usize,
        dt: f64,
        state_coupling: bool,
        use_nacv_couplings: bool,
        gs_dynamic: bool,
        _step: usize,
        nstates: usize,
    ) -> (
        Array1<f64>,
        Array2<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
        Option<Vec<Array1<f64>>>,
    ) {
        // timer
        let timer: Instant = Instant::now();

        // reset old data
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset_reduced();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_reduced();
        }
        for esd_pair in self.esd_pairs.iter_mut() {
            esd_pair.properties.reset_reduced();
        }
        self.properties.reset_reduced();

        // Return energies, forces and the nonadiabtic coupling
        let n_atoms: usize = self.atoms.len();
        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * n_atoms).unwrap());

        // update pairs and esd_pairs
        self.redefine_pairs();
        // system time
        let system_time: f32 = timer.elapsed().as_secs_f32();

        // HOP covalent fragmentation: ground-state Born-Oppenheimer
        // dynamics only -- excited states and couplings are not available.
        if self.config.fmo.covalent_fragmentation {
            if !gs_dynamic || state != 0 || state_coupling {
                panic!(
                    "FMO-DFTB dynamics with HOP covalent fragmentation is \
                     restricted to the ground state (set gs_dynamic = true; \
                     excited states and couplings are not available)"
                );
            }
            // ghost-aware HOP SCC + analytic gradient; the HOP data
            // (ghost atoms, extended gamma, ZREF) is rebuilt from the
            // current geometry inside the call
            let grad = self.run_gradient_hop().unwrap();
            let gs_energy = self.properties.last_energy().unwrap();
            let mut energies: Array1<f64> = Array1::zeros(nstates);
            energies[0] = gs_energy;
            let gradient: Array2<f64> = grad.into_shape([n_atoms, 3]).unwrap();
            let energy_gradient_time: f32 = timer.elapsed().as_secs_f32();
            let full_time: f32 = timer.elapsed().as_secs_f32();
            print_dyn_dftb(
                system_time,
                energy_gradient_time,
                energy_gradient_time,
                full_time,
            );
            return (energies, gradient, None, None, None);
        }

        // calculate the ground state energy
        self.prepare_scc();
        let gs_energy = self.run_scc().unwrap();
        let mut energies: Array1<f64> = Array1::zeros(nstates);
        energies[0] = gs_energy;

        // calculate the gs gradient
        let gs_gradient = self.ground_state_gradient();
        let mut gradient: Array1<f64> = gs_gradient.clone(); //.into_shape([n_atoms, 3]).unwrap();
        // energy and gradient
        let energy_gradient_time: f32 = timer.elapsed().as_secs_f32();

        // get the excited monomer index from the config
        // temporary way
        let excited_monomer_index: usize = self.config.tddftb.states_to_analyse[0];

        // excited state gradient
        if gs_dynamic == false && state != 0 {
            // get the monomer index for the nacv
            let mol: &mut Monomer = &mut self.monomers[excited_monomer_index];
            let monomer_atoms = &self.atoms[mol.slice.atom_as_range()];

            mol.prepare_tda(&self.atoms[mol.slice.atom_as_range()], &self.config);
            mol.run_tda(
                &self.atoms[mol.slice.atom_as_range()],
                self.config.excited.nstates,
                self.config.excited.davidson_iterations,
                self.config.excited.davidson_convergence,
                self.config.excited.davidson_subspace_multiplier,
                true,
                &self.config,
            );
            let excited_energies = mol.properties.ci_eigenvalues().unwrap();
            energies
                .slice_mut(s![1..])
                .assign(&(gs_energy + &excited_energies.slice(s![..nstates-1])));
            let exc_grad = mol.tda_gradient_lc_accumulation(monomer_atoms, state - 1);
            gradient.slice_mut(s![mol.slice.grad]).add_assign(&exc_grad);
        }
        // reshape gradient
        let gradient: Array2<f64> = gradient.into_shape([n_atoms, 3]).unwrap();

        // nonadiabatic coupling
        let (coupling, olap,nac_vecs) = if gs_dynamic == false && state_coupling == true && state != 0 {
            if use_nacv_couplings == true {
                // get the monomer index for the nacv
                let mol: &mut Monomer = &mut self.monomers[excited_monomer_index];
                // slice the velocities
                let velocities_1d: ArrayView1<f64> =
                    velocities.into_shape(3 * self.atoms.len()).unwrap();
                let velocities_monomer: ArrayView1<f64> = velocities_1d.slice(s![mol.slice.grad]);
                // get the monomer atoms
                let monomer_atoms = &self.atoms[mol.slice.atom_as_range()];
                let (coupling, vector_couplings) = mol.get_nonadiabatic_vector_coupling(
                    velocities_monomer,
                    nstates,
                    monomer_atoms,
                );

                // reshape the vector couplings
                let mut reshaped_nacvs:Vec<Array1<f64>> = Vec::new();
                for nacv_vec in vector_couplings.iter(){
                    let mut full_nacv_vec:Array1<f64> = Array1::zeros(3*self.atoms.len());
                    full_nacv_vec.slice_mut(s![mol.slice.grad]).assign(&nacv_vec);
                    reshaped_nacvs.push(full_nacv_vec);
                }

                // get the overlap coupling matrix
                let s_coupl: Array2<f64> = Array::eye(nstates) + &coupling * dt;
                (Some(coupling), Some(s_coupl),Some(reshaped_nacvs))
            } else {
                (None, None,None)
            }
        } else {
            (None, None,None)
        };

        // nacme time
        let nacme_time: f32 = timer.elapsed().as_secs_f32();
        let full_time: f32 = timer.elapsed().as_secs_f32();
        print_dyn_dftb(system_time, energy_gradient_time, nacme_time, full_time);

        (energies, gradient, coupling, olap, nac_vecs)
    }

    fn recompute_gradient(&mut self, _coordinates: ArrayView2<f64>, state: usize) -> Array2<f64> {
        // reset old data
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset_reduced();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_reduced();
        }
        for esd_pair in self.esd_pairs.iter_mut() {
            esd_pair.properties.reset_reduced();
        }
        self.properties.reset_reduced();

        // Return energies, forces and the nonadiabtic coupling
        let n_atoms: usize = self.atoms.len();

        // HOP covalent fragmentation: ground state only
        if self.config.fmo.covalent_fragmentation {
            if state != 0 {
                panic!(
                    "FMO-DFTB dynamics with HOP covalent fragmentation is \
                     restricted to the ground state"
                );
            }
            self.redefine_pairs();
            let grad = self.run_gradient_hop().unwrap();
            return grad.into_shape([n_atoms, 3]).unwrap();
        }

        // calculate the ground state energy
        self.prepare_scc();
        let _gs_energy = self.run_scc().unwrap();

        // calculate the gs gradient
        let gs_gradient = self.ground_state_gradient();
        let mut gradient: Array1<f64> = gs_gradient.clone(); //.into_shape([n_atoms, 3]).unwrap();

        // get the excited monomer index from the config
        // temporary way
        let excited_monomer_index: usize = self.config.tddftb.states_to_analyse[0];

        // excited state gradient
        if state != 0 {
            // get the monomer index for the nacv
            let mol: &mut Monomer = &mut self.monomers[excited_monomer_index];
            let monomer_atoms = &self.atoms[mol.slice.atom_as_range()];

            mol.prepare_tda(&self.atoms[mol.slice.atom_as_range()], &self.config);
            mol.run_tda(
                &self.atoms[mol.slice.atom_as_range()],
                self.config.excited.nstates,
                self.config.excited.davidson_iterations,
                self.config.excited.davidson_convergence,
                self.config.excited.davidson_subspace_multiplier,
                true,
                &self.config,
            );
            let exc_grad = mol.tda_gradient_lc_accumulation(monomer_atoms, state - 1);
            gradient.slice_mut(s![mol.slice.grad]).add_assign(&exc_grad);
        }
        // reshape gradient
        let gradient: Array2<f64> = gradient.into_shape([n_atoms, 3]).unwrap();
        gradient
    }

    fn compute_ehrenfest(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state_coefficients: ArrayView1<c64>,
        thresh: f64,
        dt: f64,
        _step: usize,
        use_state_couplings: bool,
        use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>) {
        let timer: Instant = Instant::now();
        // reset old data
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset_reduced();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_reduced();
        }
        for esd_pair in self.esd_pairs.iter_mut() {
            esd_pair.properties.reset_reduced();
        }
        self.properties.reset_reduced();
        let populations: Array1<f64> = state_coefficients.map(|val| val.norm_sqr());

        // Return energies, forces and the nonadiabtic coupling
        let n_atoms: usize = self.atoms.len();
        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * n_atoms).unwrap());
        // update pairs and esd_pairs
        self.redefine_pairs();

        // timing for the system update
        let system_time: f32 = timer.elapsed().as_secs_f32();

        // calculate the ground state energy
        self.prepare_scc();
        let gs_energy = self.run_scc().unwrap();
        let scf_time: f32 = timer.elapsed().as_secs_f32();

        // calculate diabatic hamiltonian
        let mut diabatic_hamiltonian: Array2<f64> = self.get_excitonic_matrix();
        let exc_time: f32 = timer.elapsed().as_secs_f32();

        // get the gradient
        let gradient = self.calculate_ehrenfest_gradient(populations.view(), thresh);
        let grad_time: f32 = timer.elapsed().as_secs_f32();

        let couplings: Array2<f64>;
        // calculate the nonadiabatic coupling
        if use_state_couplings {
            if use_nacv_couplings {
                // vector couplings
                let tmp =
                    self.get_nonadiabatic_vector_coupling(velocities, populations.view(), thresh);
                let coupling = tmp.0;
                let hashmap = tmp.1;

                // get the diabatic hamiltonian
                diabatic_hamiltonian =
                    self.align_signs_diabatic_hamiltonian(diabatic_hamiltonian.view());

                // store the nacv hashmap in the old system
                let mut old_system = self.properties.old_supersystem().unwrap().clone();
                old_system.nacv_storage = hashmap;
                self.properties.set_old_supersystem(old_system);

                couplings = coupling;
            } else if self.properties.old_supersystem().is_some() {
                let (coupling, diab_hamiltonian, _s, _diag, _signs): (
                    Array2<f64>,
                    Array2<f64>,
                    Array2<f64>,
                    Array1<f64>,
                    Array1<f64>,
                ) = self.nonadiabatic_scalar_coupling(diabatic_hamiltonian.view(), dt);

                // set the diabatic hamiltonian
                diabatic_hamiltonian = diab_hamiltonian;

                // set the couplings
                couplings = coupling;
            } else {
                diabatic_hamiltonian =
                    self.align_signs_diabatic_hamiltonian(diabatic_hamiltonian.view());
                couplings = Array2::zeros((1, 1));

                // set new reference
                let old_system = crate::fmo::old_supersystem::new_old_supersystem(self);
                self.properties.set_old_supersystem(old_system);
            }
        } else {
            diabatic_hamiltonian =
                self.align_signs_diabatic_hamiltonian(diabatic_hamiltonian.view());
            couplings = Array2::zeros((1, 1));

            // set new reference
            let old_system = crate::fmo::old_supersystem::new_old_supersystem(self);
            self.properties.set_ref_supersystem(old_system);
        }
        let nacme_time: f32 = timer.elapsed().as_secs_f32();

        // create diabatic hamiltonian with dimension +1
        let dim: usize = diabatic_hamiltonian.dim().0 + 1;
        let mut new_diabatic: Array2<f64> = Array2::zeros([dim, dim]);
        new_diabatic
            .slice_mut(s![1.., 1..])
            .assign(&diabatic_hamiltonian);

        for idx in 0..dim {
            new_diabatic[[idx, idx]] += gs_energy;
        }

        // reshape the gradient
        let gradient: Array2<f64> = gradient.into_shape([n_atoms, 3]).unwrap();

        // full timings
        let full_time: f32 = timer.elapsed().as_secs_f32();
        print_dyn_timings_ehrenfest(
            system_time,
            scf_time,
            grad_time,
            exc_time,
            nacme_time,
            full_time,
        );

        (gs_energy, gradient, new_diabatic, couplings)
    }

    fn compute_ehrenfest_tab(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state_coefficients: ArrayView1<c64>,
        thresh: f64,
        tab_grad_threshold: f64,
        dt: f64,
        _step: usize,
        use_state_couplings: bool,
        use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let timer: Instant = Instant::now();
        // reset old data
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset_reduced();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_reduced();
        }
        for esd_pair in self.esd_pairs.iter_mut() {
            esd_pair.properties.reset_reduced();
        }
        self.properties.reset_reduced();
        let populations: Array1<f64> = state_coefficients.map(|val| val.norm_sqr());

        // Return energies, forces and the nonadiabtic coupling
        let n_atoms: usize = self.atoms.len();
        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * n_atoms).unwrap());
        // update pairs and esd_pairs
        self.redefine_pairs();

        // timing for the system update
        let system_time: f32 = timer.elapsed().as_secs_f32();

        // calculate the ground state energy
        self.prepare_scc();
        let gs_energy = self.run_scc().unwrap();
        let scf_time: f32 = timer.elapsed().as_secs_f32();

        // calculate diabatic hamiltonian
        let mut diabatic_hamiltonian: Array2<f64> = self.get_excitonic_matrix();
        let exc_time: f32 = timer.elapsed().as_secs_f32();

        // get the gradient
        let (gradient, grad_array) =
            self.calculate_ehrenfest_gradient_tab(populations.view(), tab_grad_threshold);
        let grad_time: f32 = timer.elapsed().as_secs_f32();

        let couplings: Array2<f64>;
        // calculate the nonadiabatic coupling
        if use_state_couplings {
            if use_nacv_couplings {
                // vector couplings
                let tmp =
                    self.get_nonadiabatic_vector_coupling(velocities, populations.view(), thresh);
                let coupling = tmp.0;
                let hashmap = tmp.1;

                // get the diabatic hamiltonian
                diabatic_hamiltonian =
                    self.align_signs_diabatic_hamiltonian(diabatic_hamiltonian.view());

                // store the nacv hashmap in the old system
                let mut old_system = self.properties.old_supersystem().unwrap().clone();
                old_system.nacv_storage = hashmap;
                self.properties.set_old_supersystem(old_system);

                couplings = coupling;
            } else if self.properties.old_supersystem().is_some() {
                let (coupling, diab_hamiltonian, _s, _diag, _signs): (
                    Array2<f64>,
                    Array2<f64>,
                    Array2<f64>,
                    Array1<f64>,
                    Array1<f64>,
                ) = self.nonadiabatic_scalar_coupling(diabatic_hamiltonian.view(), dt);

                // set the diabatic hamiltonian
                diabatic_hamiltonian = diab_hamiltonian;

                // set the couplings
                couplings = coupling;
            } else {
                diabatic_hamiltonian =
                    self.align_signs_diabatic_hamiltonian(diabatic_hamiltonian.view());
                couplings = Array2::zeros((1, 1));

                // set new reference
                let old_system = crate::fmo::old_supersystem::new_old_supersystem(self);
                self.properties.set_old_supersystem(old_system);
            }
        } else {
            diabatic_hamiltonian =
                self.align_signs_diabatic_hamiltonian(diabatic_hamiltonian.view());
            couplings = Array2::zeros((1, 1));

            // set new reference
            let old_system = crate::fmo::old_supersystem::new_old_supersystem(self);
            self.properties.set_ref_supersystem(old_system);
        }
        let nacme_time: f32 = timer.elapsed().as_secs_f32();

        // create diabatic hamiltonian with dimension +1
        let dim: usize = diabatic_hamiltonian.dim().0 + 1;
        let mut new_diabatic: Array2<f64> = Array2::zeros([dim, dim]);
        new_diabatic
            .slice_mut(s![1.., 1..])
            .assign(&diabatic_hamiltonian);

        for idx in 0..dim {
            new_diabatic[[idx, idx]] += gs_energy;
        }

        // reshape the gradient
        let gradient: Array2<f64> = gradient.into_shape([n_atoms, 3]).unwrap();

        // full timings
        let full_time: f32 = timer.elapsed().as_secs_f32();
        print_dyn_timings_ehrenfest(
            system_time,
            scf_time,
            grad_time,
            exc_time,
            nacme_time,
            full_time,
        );

        (gs_energy, gradient, new_diabatic, couplings, grad_array)
    }
}


// =========================================================================


