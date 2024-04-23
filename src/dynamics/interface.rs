use crate::fmo::old_supersystem::OldSupersystem;
use crate::fmo::SuperSystem;
use crate::initialization::old_system::OldSystem;
use crate::initialization::System;
use crate::scc::scc_routine::RestrictedSCC;
use dialect_dynamics::initialization::velocities;
use dialect_dynamics::interface::QCInterface;
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use std::time::Instant;

impl QCInterface for System {
    // Return enegies, forces, non-adiabtic coupling and the transition dipole
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
    ) -> (
        Array1<f64>,
        Array2<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    ) {
        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * self.n_atoms).unwrap());
        // calculate the energy and the gradient of the state
        let (energies, gradient): (Array1<f64>, Array1<f64>) =
            self.calculate_energies_and_gradient(state, state_coupling, gs_dynamic);
        let gradient: Array2<f64> = gradient.into_shape([self.n_atoms, 3]).unwrap();
        let use_nacv: bool = use_nacv_couplings;

        // calculate the scalar couplings
        let (couplings, olap): (Option<Array2<f64>>, Option<Array2<f64>>) = if state_coupling
            && gs_dynamic == false
            && use_nacv
        {
            let (nacv, vectors): (Array2<f64>, Array3<f64>) =
                self.get_nonadiabatic_vector_coupling(velocities);

            // set the old system
            let old_system: OldSystem = OldSystem::new(&self, None, Some(vectors));
            self.properties.set_old_system(old_system);

            // get the number of excited states
            let nstates: usize = self.config.excited.nstates + 1;
            // add additional row and coloumn for the ground state
            let mut nacv_mat: Array2<f64> = Array2::zeros((nstates, nstates));
            nacv_mat.slice_mut(s!(1.., 1..)).assign(&nacv);

            // get the overlap coupling matrix
            let s_coupl: Array2<f64> = Array::eye(nstates) + &nacv_mat * dt;

            (Some(nacv_mat), Some(s_coupl))
        } else if state_coupling && gs_dynamic == false && use_nacv == false {
            let (couplings, olap): (Array2<f64>, Array2<f64>) = self.get_scalar_coupling(dt, step);
            let mut couplings_mat: Array2<f64> = Array2::zeros(couplings.raw_dim());
            couplings_mat
                .slice_mut(s![1.., 1..])
                .assign(&couplings.slice(s![1.., 1..]));

            (Some(couplings_mat), Some(olap))
        } else {
            (None, None)
        };

        return (energies, gradient, couplings, olap);
    }

    fn compute_ehrenfest(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state_coefficients: ArrayView1<f64>,
        thresh: f64,
        dt: f64,
        step: usize,
        use_state_couplings: bool,
        use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>) {
        self.properties.reset();
        self.properties.reset_gradient();

        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * self.n_atoms).unwrap());
        // calculate the energy and the gradient of the state
        let (energies, gradient): (Array1<f64>, Array1<f64>) =
            self.calculate_energies_and_gradient_ehrenfest(state_coefficients, thresh);
        let gradient: Array2<f64> = gradient.into_shape([self.n_atoms, 3]).unwrap();

        // calculate the scalar couplings
        let (couplings, olap): (Option<Array2<f64>>, Option<Array2<f64>>) = if use_state_couplings
            && use_nacv_couplings
        {
            let (nacv, vectors): (Array2<f64>, Array3<f64>) =
                self.get_nonadiabatic_vector_coupling(velocities);

            // set the old system
            let old_system: OldSystem = OldSystem::new(&self, None, Some(vectors));
            self.properties.set_old_system(old_system);

            // get the number of excited states
            let nstates: usize = self.config.excited.nstates + 1;
            // add additional row and coloumn for the ground state
            let mut nacv_mat: Array2<f64> = Array2::zeros((nstates, nstates));
            nacv_mat.slice_mut(s!(1.., 1..)).assign(&nacv);

            // get the overlap coupling matrix
            let s_coupl: Array2<f64> = Array::eye(nstates) + &nacv_mat * dt;

            (Some(nacv_mat), Some(s_coupl))
        } else if use_state_couplings && use_nacv_couplings == false {
            let (couplings, olap): (Array2<f64>, Array2<f64>) = self.get_scalar_coupling(dt, step);
            let mut couplings_mat: Array2<f64> = Array2::zeros(couplings.raw_dim());
            couplings_mat
                .slice_mut(s![1.., 1..])
                .assign(&couplings.slice(s![1.., 1..]));

            (Some(couplings_mat), Some(olap))
        } else {
            (Some(Array2::zeros((1, 1))), None)
        };

        // get 2d array from energies
        let mut energy_hamiltonian: Array2<f64> = Array2::from_diag(&energies);

        return (
            energies[0],
            gradient,
            energy_hamiltonian,
            couplings.unwrap(),
        );
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
        step: usize,
    ) -> (
        Array1<f64>,
        Array2<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    ) {
        // reset old data
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut() {
            esd_pair.properties.reset();
        }
        self.properties.reset();

        // Return energies, forces and the nonadiabtic coupling
        let n_atoms: usize = self.atoms.len();
        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * n_atoms).unwrap());

        // calculate the ground state energy
        self.prepare_scc();
        let gs_energy = self.run_scc().unwrap();

        // calculate the gs gradient
        let gs_gradient = self.ground_state_gradient();
        let gradient: Array2<f64> = gs_gradient.into_shape([n_atoms, 3]).unwrap();

        (array![gs_energy], gradient, None, None)
    }

    fn compute_ehrenfest(
        &mut self,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        state_coefficients: ArrayView1<f64>,
        thresh: f64,
        dt: f64,
        step: usize,
        use_state_couplings: bool,
        use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>) {
        let timer: Instant = Instant::now();
        // reset old data
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        for esd_pair in self.esd_pairs.iter_mut() {
            esd_pair.properties.reset();
        }
        self.properties.reset();

        // Return energies, forces and the nonadiabtic coupling
        let n_atoms: usize = self.atoms.len();
        // update the coordinates of the system
        self.update_xyz(coordinates.into_shape(3 * n_atoms).unwrap());

        // calculate the ground state energy
        self.prepare_scc();
        let gs_energy = self.run_scc().unwrap();
        println!("Time fmo scc routine {:.5}", timer.elapsed().as_secs_f32());
        drop(timer);
        let timer: Instant = Instant::now();

        // calculate diabatic hamiltonian
        let mut diabatic_hamiltonian: Array2<f64> = self.get_excitonic_matrix();
        println!(
            "Time FMO excitonic matrix {:.5}",
            timer.elapsed().as_secs_f32()
        );
        drop(timer);
        let timer: Instant = Instant::now();

        // get the gradient
        let gradient = self.calculate_ehrenfest_gradient(state_coefficients, thresh);

        println!("Time Gradient {:.5}", timer.elapsed().as_secs_f32());
        drop(timer);
        let timer: Instant = Instant::now();

        let mut couplings: Array2<f64>;
        // calculate the nonadiabatic coupling
        if use_state_couplings {
            if use_nacv_couplings {
                // vector couplings
                let tmp =
                    self.get_nonadiabatic_vector_coupling(velocities, state_coefficients, thresh);
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
            } else {
                if self.properties.old_supersystem().is_some() {
                    let (coupling, diab_hamiltonian, s, diag, signs): (
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
                    let old_system = OldSupersystem::new(&self);
                    self.properties.set_old_supersystem(old_system);
                }
            }
        } else {
            diabatic_hamiltonian =
                self.align_signs_diabatic_hamiltonian(diabatic_hamiltonian.view());
            couplings = Array2::zeros((1, 1));

            // set new reference
            let old_system = OldSupersystem::new(&self);
            self.properties.set_ref_supersystem(old_system);
        }
        println!("Time Couplings {:.5}", timer.elapsed().as_secs_f32());

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

        (gs_energy, gradient, new_diabatic, couplings)
    }
}
