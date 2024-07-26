use crate::fmo::polariton::polaritonic_state::ExcitonPolaritonStates;
use crate::fmo::{BasisState, SuperSystem};
use crate::initialization::Atom;
use crate::io::settings::PolaritonConfig;
use nalgebra::Vector3;
use ndarray::prelude::*;
use ndarray::{concatenate, Slice};
use ndarray_linalg::{Eigh, UPLO};
use rayon::prelude::*;
use std::f64::consts::PI;

impl SuperSystem<'_> {
    pub fn create_exciton_polariton_hamiltonian(&mut self) {
        let polariton_config: PolaritonConfig = self.config.polariton.clone();

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
        let h: Array2<f64> = Array::from(h).into_shape((dim, dim)).unwrap();
        // increase the size of the array to include the interaction with the electric field of the cavity
        let photon_dim: usize = polariton_config.photon_energy.len();
        let mut h_new: Array2<f64> = Array2::zeros([dim + photon_dim, dim + photon_dim]);
        h_new.slice_mut(s![..dim, ..dim]).assign(&h);
        drop(h);

        for (i, (((photon_energy, vol), p), e)) in polariton_config
            .photon_energy
            .iter()
            .zip(polariton_config.quantized_volume.iter())
            .zip(polariton_config.p.iter())
            .zip(polariton_config.e.iter())
            .enumerate()
        {
            let polarization: Vector3<f64> = Vector3::new(p[0], p[1], p[2]);
            let g: f64 = (4.0 * PI * *photon_energy / *vol).sqrt();
            let mut arr: Array1<f64> = Array1::zeros(dim);
            if polarization == Vector3::zeros() {
                for (i, state_i) in states.iter().enumerate() {
                    match state_i {
                        BasisState::LE(ref a) => {
                            arr[i] = -g * (a.tr_dipole.dot(&a.tr_dipole).sqrt());
                        }
                        _ => {}
                    }
                }
            } else {
                let polarization = polarization / polarization.dot(&polarization).sqrt();
                let polarization = Vector3::new(p[0] * e[0], p[1] * e[1], p[2] * e[2]);
                for (i, state_i) in states.iter().enumerate() {
                    match state_i {
                        BasisState::LE(ref a) => {
                            arr[i] = -g * (a.tr_dipole.dot(&polarization));
                        }
                        _ => {}
                    }
                }
            }
            h_new.slice_mut(s![..dim, dim + i]).assign(&arr);
            h_new[[dim + i, dim + i]] = *photon_energy;
        }
        let (energies, eigvectors): (Array1<f64>, Array2<f64>) = h_new.eigh(UPLO::Lower).unwrap();

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
        let orbs: Array2<f64> = orbs.map(|val| *val as f64);

        // calculate the oscillator strenghts of the excited states
        let polariton = ExcitonPolaritonStates::new(
            self.properties.last_energy().unwrap(),
            (energies.clone(), eigvectors.clone()),
            states.clone(),
            (n_occ, n_virt),
            orbs,
            polariton_config,
        );

        // write the excited state spectra to files
        polariton.spectrum_to_npy("lcmo_plrtn_spec.npy");
        polariton.spectrum_to_txt("lcmo_plrtn_spec.txt");

        println!("{}", polariton);
    }
}
