use crate::constants::{ATOMIC_MASSES, BOHR_TO_ANGS, HARTREE_TO_WAVENUMBERS};
use crate::initial_conditions::wigner_distribution::WignerEnsemble;
use crate::initialization::System;
use crate::optimization::helpers::{write_xyz_wigner, XYZOutput};
use log::{log_enabled, warn, Level};
use ndarray::prelude::*;
use ndarray_linalg::{c64, into_col, into_row, Eigh, UPLO};
use ndarray_npy::write_npy;
use std::fs::create_dir;

impl System {
    // create the initial geometries and velocities for the trajectories of a dynamics simulation
    pub fn create_initial_conditions(&mut self) {
        // at first, the geometry of the molecules must be optimized in the electronic ground state
        self.optimize_cartesian(0, &self.config.clone());

        // next, the hessian of the electronic ground state is calculated
        let hessian: Array2<f64> = self.calculate_num_hessian();

        // get the masses of the molecule
        let mut masses: Vec<f64> = Vec::new();
        // get the masses to the corresponding atomic numbers
        self.atoms.iter().for_each(|atom| {
            for _i in 0..3 {
                masses.push(ATOMIC_MASSES[&atom.number]);
            }
        });
        let masses: Array1<f64> = Array::from(masses);

        // get the vibrations and the modes from the mass weighted hessian
        let (omega2, _freqs, modes): (Array1<f64>, Array1<c64>, Array2<f64>) =
            calculate_vibrations_and_modes(hessian.view(), masses.view());

        // get the coordinates
        let coords: Array1<f64> = self.get_xyz();

        // create wigner ensemble
        let mut wigner_ensemble: WignerEnsemble =
            WignerEnsemble::new(self, omega2, modes.view(), masses.view(), coords.view());

        // sample the distribution and generate ensemble
        let (coord_vec, velocity_vec): (Vec<Array1<f64>>, Vec<Array1<f64>>) =
            wigner_ensemble.get_ensemble();

        // write the wigner ensemble
        self.write_wigner_ensemble(coord_vec, velocity_vec);
    }

    pub fn write_wigner_ensemble(
        &self,
        coord_vec: Vec<Array1<f64>>,
        velocity_vec: Vec<Array1<f64>>,
    ) {
        // get the atom names
        let atom_names: Vec<String> = self
            .atoms
            .iter()
            .map(|atom| String::from(atom.name))
            .collect();

        // save the coordinates and velocities
        // get the directory where the trajectories should be saved
        let path: String = if self.config.wigner_config.save_in_other_path {
            self.config.wigner_config.wigner_path.clone()
        } else {
            String::from("")
        };

        // loop over the vectors
        for (idx, (coords, velocities)) in coord_vec.iter().zip(velocity_vec.iter()).enumerate() {
            // set the path
            let dir_name: String = if self.config.wigner_config.save_in_other_path {
                path.clone() + &format!("/traj_{idx}")
            } else {
                format!("traj_{idx}")
            };
            // create the directory for the trajectory
            if !std::path::Path::new(&dir_name).exists() {
                create_dir(&dir_name).unwrap();
            }

            // create the xyz file
            let geom_name: String = dir_name.clone() + "/geom.xyz";

            let new_coords: Array2<f64> =
                BOHR_TO_ANGS * &coords.view().into_shape((self.n_atoms, 3)).unwrap();
            let xyz_out: XYZOutput = XYZOutput::new(
                atom_names.clone(),
                new_coords.clone().into_shape([self.n_atoms, 3]).unwrap(),
            );
            write_xyz_wigner(&xyz_out, geom_name);

            if self.config.wigner_config.write_velocities {
                // create the velocity file
                let velocities_name: String = dir_name + "/velocities.npy";
                let velocities_2d: Array2<f64> =
                    velocities.clone().into_shape([self.n_atoms, 3]).unwrap();
                write_npy(velocities_name, &velocities_2d).unwrap();
            }
        }
    }
}

fn calculate_vibrations_and_modes(
    hessian: ArrayView2<f64>,
    masses: ArrayView1<f64>,
) -> (Array1<f64>, Array1<c64>, Array2<f64>) {
    // calculate the outer product of the masses and take the square root
    let masses_matrix: Array2<f64> = into_col(masses)
        .dot(&into_row(masses))
        .map(|val| val.sqrt());
    // get the mass weighted hessian
    let mw_hessian: Array2<f64> = &hessian / &masses_matrix;

    // calculate the eigenvalues and eigenvectors of the mass weighted hessian
    let (omega2, vecs): (Array1<f64>, Array2<f64>) = mw_hessian.eigh(UPLO::Lower).unwrap();

    // get the complex frequencies
    let freq: Vec<c64> = omega2
        .iter()
        .map(|val| (c64::new(1.0, 0.0) * val).sqrt())
        .collect();
    // convert to array
    let freq: Array1<c64> = Array::from(freq);

    // print the frequencies
    if log_enabled!(Level::Warn) {
        warn!("{:^80}", "");
        warn!("{: ^80}", "Vibrational modes calculation");
        warn!("{:-^80}", "");

        warn!(" ");
        warn!("Vibrational frequencies:");
        for (idx, vibr) in freq.iter().enumerate() {
            if vibr.re < 0.0 {
                println!("Warning: negative frequency detected!!");
            }
            warn!("{: >5} {:>18.10}", idx, vibr.re * HARTREE_TO_WAVENUMBERS,);
        }
        warn!("");
        warn!("{:-<80} ", "");
    }

    (omega2, freq, vecs)
}
