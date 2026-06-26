pub use dialect_utilities::cubes::cube::DensityToCube;

use crate::{SuperSystem, System};
use ndarray::prelude::*;
use ndarray_npy::read_npy;

impl System {
    pub fn density_to_cube(&self) {
        // load the density from the file
        let density: Array2<f64> = read_npy(self.config.density.path_to_density.clone()).unwrap();
        let density_32: Array2<f32> = density.map(|val| *val as f32);
        // create an object of DensityToCube
        let generator: DensityToCube = DensityToCube::new(
            self.config.density.points_per_bohr,
            &self.atoms,
            self.config.density.use_block_implementation,
            self.config.density.n_blocks,
            self.config.density.threshold as f32,
            &self.config.parameterization,
        );
        let output_filename: String = String::from("density.cube");
        generator.generate_density_cube_file(density_32.view(), output_filename);
    }

    pub fn density_from_tdm(&self, tdm: ArrayView2<f64>, step: usize, string: &str) {
        // load the density from the file
        let density: Array2<f32> = tdm.map(|val| *val as f32);
        // create an object of DensityToCube
        let generator: DensityToCube = DensityToCube::new(
            self.config.density.points_per_bohr,
            &self.atoms,
            self.config.density.use_block_implementation,
            self.config.density.n_blocks,
            self.config.density.threshold as f32,
            &self.config.parameterization,
        );
        let mut output_filename: String = String::from("density_");
        output_filename.push_str(&step.to_string());
        output_filename.push_str(string);
        output_filename.push_str(".cube");
        generator.generate_density_cube_file(density.view(), output_filename);
    }

    pub fn cube_from_tdm(&self, tdm: ArrayView2<f64>, state: usize) {
        // load the density from the file
        let density: Array2<f32> = tdm.map(|val| *val as f32);
        // create an object of DensityToCube
        let generator: DensityToCube = DensityToCube::new(
            self.config.density.points_per_bohr,
            &self.atoms,
            self.config.density.use_block_implementation,
            self.config.density.n_blocks,
            self.config.density.threshold as f32,
            &self.config.parameterization,
        );
        let mut output_filename: String = String::from("tdm_");
        output_filename.push_str(&state.to_string());
        output_filename.push_str(".cube");
        generator.generate_density_cube_file(density.view(), output_filename);
    }

    pub fn cube_from_orbital(&self, orbital: ArrayView1<f64>, index: usize, state: usize) {
        // load the density from the file
        let orbital_f32: Array1<f32> = orbital.map(|val| *val as f32);
        // create an object of DensityToCube
        let generator: DensityToCube = DensityToCube::new(
            self.config.density.points_per_bohr,
            &self.atoms,
            self.config.density.use_block_implementation,
            self.config.density.n_blocks,
            self.config.density.threshold as f32,
            &self.config.parameterization,
        );
        let mut output_filename: String = String::from("orbital_");
        output_filename.push_str("_state_");
        output_filename.push_str(&state.to_string());
        output_filename.push_str("_nr_");
        output_filename.push_str(&index.to_string());
        output_filename.push_str(".cube");
        generator.generate_orbital_cube_file(orbital_f32.view(), output_filename);
    }

    pub fn cube_from_orbital_arr(
        &self,
        orbital_arr: ArrayView2<f64>,
        indices: &[usize],
        state: usize,
        string: &str,
    ) {
        // create an object of DensityToCube
        let generator: DensityToCube = DensityToCube::new(
            self.config.density.points_per_bohr,
            &self.atoms,
            self.config.density.use_block_implementation,
            self.config.density.n_blocks,
            self.config.density.threshold as f32,
            &self.config.parameterization,
        );
        // convert to f32
        let orbital_array: Array2<f32> = orbital_arr.map(|val| *val as f32);
        // calculate orbitals on grid
        let orbital_vec: Vec<Array3<f32>> =
            generator.calculate_orbitals_on_grid(orbital_array.view());

        for (index, orbital) in indices.iter().zip(orbital_vec.iter()) {
            let mut output_filename: String = string.to_string();
            output_filename.push_str("_state_");
            output_filename.push_str(&state.to_string());
            output_filename.push_str("_nr_");
            output_filename.push_str(&index.to_string());
            output_filename.push_str(".cube");
            generator.write_density_to_cube(output_filename, orbital.view());
        }
    }
}

impl SuperSystem<'_> {
    pub fn density_to_cube(&self) {
        // load the density from the file
        let density: Array2<f32> = read_npy(self.config.density.path_to_density.clone()).unwrap();
        // create an object of DensityToCube
        let generator: DensityToCube = DensityToCube::new(
            self.config.density.points_per_bohr,
            &self.atoms,
            self.config.density.use_block_implementation,
            self.config.density.n_blocks,
            self.config.density.threshold as f32,
            &self.config.parameterization,
        );
        let output_filename: String = String::from("density.cube");
        generator.generate_density_cube_file(density.view(), output_filename);
    }

    pub fn density_from_tdm(&self, tdm: ArrayView2<f64>, step: usize, string: &str) {
        // load the density from the file
        let density: Array2<f32> = tdm.map(|val| *val as f32);
        // create an object of DensityToCube
        let generator: DensityToCube = DensityToCube::new(
            self.config.density.points_per_bohr,
            &self.atoms,
            self.config.density.use_block_implementation,
            self.config.density.n_blocks,
            self.config.density.threshold as f32,
            &self.config.parameterization,
        );
        let mut output_filename: String = String::from("density_");
        output_filename.push_str(&step.to_string());
        output_filename.push_str(string);
        output_filename.push_str(".cube");
        generator.generate_density_cube_file(density.view(), output_filename);
    }
}

