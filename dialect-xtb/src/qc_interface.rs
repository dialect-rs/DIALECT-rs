//! QCInterface implementation and dynamics helpers for XtbSystem.

use crate::initialization::system::XtbSystem;
use dialect_dynamics::interface::QCInterface;
use dialect_utilities::output::print_dyn_dftb;
use dialect_utilities::scc_interface::RestrictedSCC;
use ndarray::prelude::*;
use ndarray_linalg::c64;
use std::time::Instant;

impl XtbSystem {
    pub fn calculate_energies_and_gradient(&mut self) -> (Array1<f64>, Array1<f64>) {
        let mut energies: Array1<f64> = Array1::zeros(1);
        // ground state energy
        self.prepare_scc();
        let gs_energy: f64 = self.run_scc().unwrap();
        energies[0] = gs_energy;
        let gradient = self.ground_state_gradient();

        (energies, gradient)
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

// =========================================================================
//
// FMO-xTB QCInterface -- ground-state Born-Oppenheimer dynamics only, for
// both the distance-based and the HOP covalent fragmentation. Excited
// states, state couplings and the Ehrenfest variants are not available.
//
// =========================================================================

use crate::fmo::supersystem::XtbSuperSystem;

impl QCInterface for XtbSuperSystem<'_> {
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        _velocities: ArrayView2<f64>,
        state: usize,
        _dt: f64,
        state_coupling: bool,
        _use_nacv_couplings: bool,
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
        let timer: Instant = Instant::now();

        if !gs_dynamic || state != 0 || state_coupling {
            panic!(
                "FMO-xTB dynamics is restricted to the ground state (set \
                 gs_dynamic = true; excited states and couplings are not \
                 available)"
            );
        }

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
        for trimer in self.trimers.iter_mut() {
            trimer.properties.reset_reduced();
        }
        self.properties.reset_reduced();

        // update the coordinates of the system
        let n_atoms: usize = self.atoms.len();
        self.update_xyz(coordinates.into_shape(3 * n_atoms).unwrap());

        // rebuild the pair / ESD-pair / trimer classification (and HOP
        // ghost positions) for the new geometry
        self.update_fragmentation();
        let system_time: f32 = timer.elapsed().as_secs_f32();

        // ground-state energy; HOP systems need the ghost-aware SCC driver
        let gs_energy: f64 = if self.config.fmo.covalent_fragmentation {
            self.run_scc_hop().unwrap()
        } else {
            self.prepare_scc();
            self.run_scc().unwrap()
        };
        let mut energies: Array1<f64> = Array1::zeros(nstates);
        energies[0] = gs_energy;

        // ground-state gradient (branches internally on HOP)
        let grad: Array1<f64> = self.ground_state_gradient();
        let gradient: Array2<f64> = grad.into_shape([n_atoms, 3]).unwrap();
        let energy_gradient_time: f32 = timer.elapsed().as_secs_f32();

        let full_time: f32 = timer.elapsed().as_secs_f32();
        print_dyn_dftb(
            system_time,
            energy_gradient_time,
            energy_gradient_time,
            full_time,
        );

        (energies, gradient, None, None, None)
    }

    fn recompute_gradient(&mut self, _coordinates: ArrayView2<f64>, _state: usize) -> Array2<f64> {
        todo!()
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
        _tab_grad_thresh: f64,
        _dt: f64,
        _step: usize,
        _use_state_couplings: bool,
        _use_nacv_couplings: bool,
    ) -> (f64, Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        todo!()
    }
}
