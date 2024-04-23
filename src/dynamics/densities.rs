use crate::{
    fmo::{build_graph, fragmentation, old_supersystem::OldSupersystem, Graph, SuperSystem},
    initialization::{parameter_handling::generate_parameters, Atom},
    io::frame_to_atoms,
    scc::scc_routine::RestrictedSCC,
};
use chemfiles::{Frame, Trajectory};
use ndarray::prelude::*;
use ndarray_npy::NpzReader;
use rayon::current_num_threads;
use rayon::prelude::*;
use std::fs::File;

impl SuperSystem<'_> {
    pub fn get_ehrenfest_densities(&mut self) {
        // load the geometries
        let mut trajectory = Trajectory::open("dynamics.xyz", 'r').unwrap();
        let mut frame = Frame::new();
        let mut count: usize = 0;
        let mut step_vec: Vec<usize> = Vec::new();

        for step in 0..self.config.tdm_config.total_steps {
            if count.rem_euclid(self.config.tdm_config.calculate_nth_step) == 0 {
                step_vec.push(step);
            }
            count += 1;
        }

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

        println!("Number of threads: {}", current_num_threads());

        if self.config.tdm_config.use_parallelization {
            step_vec.par_iter().enumerate().for_each(|(idx, step)| {
                // load the geometries
                let mut trajectory = Trajectory::open("dynamics.xyz", 'r').unwrap();
                let mut frame = Frame::new();
                trajectory.read_step(*step, &mut frame).unwrap();
                let (slako, vrep, atoms, unique_atoms) =
                    generate_parameters(frame.clone(), self.config.clone());

                // Get all [Atom]s of the SuperSystem in a sorted order that corresponds to the order of
                // the monomers
                let mut sorted_atoms: Vec<Atom> = Vec::with_capacity(atoms.len());

                // Build a connectivity graph to distinguish the individual monomers from each other
                let graph: Graph = build_graph(atoms.len(), &atoms);
                // Here does the fragmentation happens
                let monomer_indices: Vec<Vec<usize>> = fragmentation(&graph);

                for (idx, indices) in monomer_indices.into_iter().enumerate() {
                    // Clone the atoms that belong to this monomer, they will be stored in the sorted list
                    let mut monomer_atoms: Vec<Atom> =
                        indices.into_iter().map(|i| atoms[i].clone()).collect();
                    // Save the Atoms from the current Monomer
                    sorted_atoms.append(&mut monomer_atoms);
                }
                let mut system_clone: SuperSystem = self.clone();
                system_clone.atoms = sorted_atoms;
                system_clone.prepare_scc();
                system_clone.run_scc();

                // load the coefficients
                let mut coefficients =
                    NpzReader::new(File::open("coeff_abs.npz").unwrap()).unwrap();
                let coeff: Array1<f64> = coefficients.by_name(&step.to_string()).unwrap();
                let (tdm, h_mat, p_mat): (Array2<f64>, Array2<f64>, Array2<f64>) =
                    system_clone.get_tdm_for_ehrenfest(coeff.view(), idx);

                if system_clone.config.tdm_config.calc_cube {
                    system_clone.density_from_tdm(h_mat.view(), idx, "_h");
                    system_clone.density_from_tdm(p_mat.view(), idx, "_p");
                }
                if system_clone.config.tdm_config.calc_tdm_cube{
                    system_clone.density_from_tdm(tdm.view(), idx, "_tdm");
                }
            });
        } else {
            // load the geometries
            let mut trajectory = Trajectory::open("dynamics.xyz", 'r').unwrap();
            let mut frame = Frame::new();
            // load the coefficients
            let mut coefficients = NpzReader::new(File::open("coeff_abs.npz").unwrap()).unwrap();

            for (idx, step) in step_vec.iter().enumerate() {
                // get new geometry
                trajectory.read_step(*step, &mut frame).unwrap();
                let (slako, vrep, atoms, unique_atoms) =
                    generate_parameters(frame.clone(), self.config.clone());

                // Get all [Atom]s of the SuperSystem in a sorted order that corresponds to the order of
                // the monomers
                let mut sorted_atoms: Vec<Atom> = Vec::with_capacity(atoms.len());

                // Build a connectivity graph to distinguish the individual monomers from each other
                let graph: Graph = build_graph(atoms.len(), &atoms);
                // Here does the fragmentation happens
                let monomer_indices: Vec<Vec<usize>> = fragmentation(&graph);

                for (idx, indices) in monomer_indices.into_iter().enumerate() {
                    // Clone the atoms that belong to this monomer, they will be stored in the sorted list
                    let mut monomer_atoms: Vec<Atom> =
                        indices.into_iter().map(|i| atoms[i].clone()).collect();
                    // Save the Atoms from the current Monomer
                    sorted_atoms.append(&mut monomer_atoms);
                }
                self.atoms = sorted_atoms;
                self.prepare_scc();
                self.run_scc();

                let coeff: Array1<f64> = coefficients.by_name(&step.to_string()).unwrap();
                let (tdm, h_mat, p_mat): (Array2<f64>, Array2<f64>, Array2<f64>) =
                    self.get_tdm_for_ehrenfest(coeff.view(), idx);

                if self.config.tdm_config.calc_cube {
                    self.density_from_tdm(h_mat.view(), idx, "_h");
                    self.density_from_tdm(p_mat.view(), idx, "_p");
                }
                if self.config.tdm_config.calc_tdm_cube{
                    self.density_from_tdm(tdm.view(), idx, "_tdm");
                }
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
            }
        }
    }
}
