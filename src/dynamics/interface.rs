use super::output::{print_dyn_dftb, print_dyn_timings_ehrenfest};
use crate::fmo::old_supersystem::OldSupersystem;
use crate::fmo::SuperSystem;
use crate::initialization::old_system::OldSystem;
use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;
use crate::xtb::initialization::system::XtbSystem;
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
            self.prepare_excited_grad();
            // calculate nacvs
            let (nacv, vectors): (Array2<f64>, Vec<Array1<f64>>) =
                self.get_nonadiabatic_vector_coupling(velocities, nstates);

            // set the old system
            let old_system: OldSystem = OldSystem::new(self, None, Some(vectors.clone()));
            self.properties.set_old_system(old_system);

            // get the overlap coupling matrix
            let s_coupl: Array2<f64> = Array::eye(nstates) + &nacv * dt;

            (Some(nacv), Some(s_coupl), Some(vectors))
        } else if state_coupling && !gs_dynamic && !use_nacv {
            let (couplings, olap): (Array2<f64>, Array2<f64>) =
                self.get_scalar_coupling(dt, step, nstates);
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
        let old_system: OldSystem = OldSystem::new(self, None, Some(vectors));
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
    // // calculate the scalar couplings
    // let (couplings, _olap): (Option<Array2<f64>>, Option<Array2<f64>>) = if use_state_couplings
    //     && use_nacv_couplings
    // {
    // // get the number of excited states
    // let nstates: usize = self.config.excited.nstates + 1;
    //
    // // get the overlap coupling matrix
    // let s_coupl: Array2<f64> = Array::eye(nstates) + &nacv * dt;
    //
    //     (Some(nacv), Some(s_coupl))
    // } else if use_state_couplings && !use_nacv_couplings {
    //     let (couplings, olap): (Array2<f64>, Array2<f64>) = self.get_scalar_coupling(dt, step);
    //     let mut couplings_mat: Array2<f64> = Array2::zeros(couplings.raw_dim());
    //     couplings_mat
    //         .slice_mut(s![1.., 1..])
    //         .assign(&couplings.slice(s![1.., 1..]));
    //
    //     // get filename
    //     let filename: String = format!("nacs_{}.npy", step);
    //     write_npy(filename, &couplings_mat).unwrap();
    //
    //     (Some(couplings_mat), Some(olap))
    // } else {
    //     (Some(Array2::zeros((1, 1))), None)
    // };

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
        let old_system: OldSystem = OldSystem::new(self, None, Some(vectors));
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
    // // calculate the scalar couplings
    // let (couplings, _olap): (Option<Array2<f64>>, Option<Array2<f64>>) = if use_state_couplings
    //     && use_nacv_couplings
    // {
    //     let (nacv, vectors): (Array2<f64>, Vec<Array1<f64>>) =
    //         self.get_nonadiabatic_vector_coupling(velocities, nstates);
    //
    //     // set the old system
    //     let old_system: OldSystem = OldSystem::new(self, None, Some(vectors));
    //     self.properties.set_old_system(old_system);
    //
    //     // get the number of excited states
    //     let nstates: usize = self.config.excited.nstates + 1;
    //
    //     // get the overlap coupling matrix
    //     let s_coupl: Array2<f64> = Array::eye(nstates) + &nacv * dt;
    //
    //     (Some(nacv), Some(s_coupl))
    // } else if use_state_couplings && !use_nacv_couplings {
    //     let (couplings, olap): (Array2<f64>, Array2<f64>) = self.get_scalar_coupling(dt, step);
    //     let mut couplings_mat: Array2<f64> = Array2::zeros(couplings.raw_dim());
    //     couplings_mat
    //         .slice_mut(s![1.., 1..])
    //         .assign(&couplings.slice(s![1.., 1..]));
    //
    //     // get filename
    //     let filename: String = format!("nacs_{}.npy", step);
    //     write_npy(filename, &couplings_mat).unwrap();
    //
    //     (Some(couplings_mat), Some(olap))
    // } else {
    //     (Some(Array2::zeros((1, 1))), None)
    // };
}

impl QCInterface for SuperSystem<'_> {
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        _velocities: ArrayView2<f64>,
        _state: usize,
        _dt: f64,
        _state_coupling: bool,
        _use_nacv_couplings: bool,
        _gs_dynamic: bool,
        _step: usize,
        _nstates: usize,
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
        // system time
        let system_time: f32 = timer.elapsed().as_secs_f32();

        // calculate the ground state energy
        self.prepare_scc();
        let gs_energy = self.run_scc().unwrap();

        // calculate the gs gradient
        let gs_gradient = self.ground_state_gradient();
        let gradient: Array2<f64> = gs_gradient.into_shape([n_atoms, 3]).unwrap();
        // energy and gradient
        let energy_gradient_time: f32 = timer.elapsed().as_secs_f32();

        // nacme time
        let nacme_time: f32 = timer.elapsed().as_secs_f32();
        let full_time: f32 = timer.elapsed().as_secs_f32();
        print_dyn_dftb(system_time, energy_gradient_time, nacme_time, full_time);

        (array![gs_energy], gradient, None, None, None)
    }

    fn recompute_gradient(&mut self, _coordinates: ArrayView2<f64>, _state: usize) -> Array2<f64> {
        todo!()
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
                let old_system = OldSupersystem::new(self);
                self.properties.set_old_supersystem(old_system);
            }
        } else {
            diabatic_hamiltonian =
                self.align_signs_diabatic_hamiltonian(diabatic_hamiltonian.view());
            couplings = Array2::zeros((1, 1));

            // set new reference
            let old_system = OldSupersystem::new(self);
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
                let old_system = OldSupersystem::new(self);
                self.properties.set_old_supersystem(old_system);
            }
        } else {
            diabatic_hamiltonian =
                self.align_signs_diabatic_hamiltonian(diabatic_hamiltonian.view());
            couplings = Array2::zeros((1, 1));

            // set new reference
            let old_system = OldSupersystem::new(self);
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

impl QCInterface for XtbSystem {
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        _velocities: ArrayView2<f64>,
        _state: usize,
        _dt: f64,
        _state_coupling: bool,
        _use_nacv_couplings: bool,
        _gs_dynamic: bool,
        _step: usize,
        _nstates: usize,
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
            self.calculate_energies_and_gradient();
        let gradient: Array2<f64> = gradient.into_shape([self.n_atoms, 3]).unwrap();

        // energy and gradient
        let energy_gradient_time: f32 = timer.elapsed().as_secs_f32();

        print_dyn_dftb(
            system_time,
            energy_gradient_time,
            energy_gradient_time,
            energy_gradient_time,
        );

        (energies, gradient, None, None, None)
    }

    fn compute_ehrenfest(
        &mut self,
        _coordinates: ArrayView2<f64>,
        _velocities: ArrayView2<f64>,
        _state_coefficients: ArrayView1<c64>,
        _thresh: f64,
        _dt: f64,
        _step: usize,
        _use_state_couplings: bool,
        _use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>) {
        todo!()
    }

    fn compute_ehrenfest_tab(
        &mut self,
        _coordinates: ArrayView2<f64>,
        _velocities: ArrayView2<f64>,
        _state_coefficients: ArrayView1<c64>,
        _thresh: f64,
        _tab_grad_threshold: f64,
        _dt: f64,
        _step: usize,
        _use_state_couplings: bool,
        _use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        todo!()
    }

    fn recompute_gradient(&mut self, _coordinates: ArrayView2<f64>, _state: usize) -> Array2<f64> {
        todo!()
    }
}
