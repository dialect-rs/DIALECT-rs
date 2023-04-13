use crate::cubes::basisfunctions::{AtomicBasisFunction, AtomicBasisSet};
use crate::cubes::helpers::create_box_around_molecule;
use crate::initialization::Atom;
use crate::{SuperSystem, System};
use ndarray::prelude::*;
use ndarray_npy::read_npy;
use ndarray_stats::QuantileExt;
use std::fs::File;
use std::io::Write;

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
        );
        let output_filename: String = String::from("density.cube");
        generator.generate_density_cube_file(density_32.view(), output_filename);
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
        );
        let output_filename: String = String::from("density.cube");
        generator.generate_density_cube_file(density.view(), output_filename);
    }
}

pub struct DensityToCube<'a> {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
    pub zmin: f64,
    pub zmax: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub spacing: f64,
    pub atoms: &'a [Atom],
    pub basis: AtomicBasisSet,
    pub use_block_impl: bool,
    pub n_blocks: usize,
    pub threshold: f32,
}

impl<'a> DensityToCube<'a> {
    pub fn new(
        ppb: f64,
        atoms: &'a [Atom],
        use_block_impl: bool,
        n_blocks: usize,
        threshold: f32,
    ) -> Self {
        // create box around the system
        let (xmin, xmax, ymin, ymax, zmin, zmax) = create_box_around_molecule(atoms, None);
        let dx: f64 = xmax - xmin;
        let dy: f64 = ymax - ymin;
        let dz: f64 = zmax - zmin;

        let spacing: f64 = 1.0 / ppb;
        // calculate the points per axis
        let nx: usize = (dx * ppb) as usize;
        let ny: usize = (dy * ppb) as usize;
        let nz: usize = (dz * ppb) as usize;

        let basis: AtomicBasisSet = AtomicBasisSet::new(atoms);

        Self {
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            nx,
            ny,
            nz,
            spacing,
            atoms,
            basis,
            use_block_impl,
            n_blocks,
            threshold,
        }
    }

    pub fn generate_density_cube_file(&self, density: ArrayView2<f32>, filename: String) {
        let rho: Array3<f32> = self.calculate_density_on_grid(density);
        self.write_density_to_cube(filename, rho.view());
    }

    pub fn write_density_to_cube(&self, filename: String, rho_grid: ArrayView3<f32>) {
        let mut txt: String = String::from("CUBE FILE \n");
        txt.push_str("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z \n");
        txt += &format!(
            "{:5} {:11.6} {:11.6} {:11.6} \n",
            self.atoms.len(),
            self.xmin,
            self.ymin,
            self.zmin
        );
        txt += &format!(
            "{:5} {:11.6} {:11.6} {:11.6} \n",
            self.nx, self.spacing, 0, 0
        );
        txt += &format!(
            "{:5} {:11.6} {:11.6} {:11.6} \n",
            self.ny, 0, self.spacing, 0
        );
        txt += &format!(
            "{:5} {:11.6} {:11.6} {:11.6} \n",
            self.nz, 0, 0, self.spacing
        );

        for atom in self.atoms {
            let (x, y, z) = (atom.xyz[0], atom.xyz[1], atom.xyz[2]);
            txt += &format!(
                "{:5} {:11.6} {:11.6} {:11.6} {:11.6} \n",
                atom.number, atom.number as f64, x, y, z
            );
        }

        for i_val in rho_grid.outer_iter() {
            for j_val in i_val.outer_iter() {
                for (k, k_val) in j_val.iter().enumerate() {
                    let mut amp = 0.0;
                    if k_val.abs() > 1.0e-10 {
                        amp = *k_val;
                    }
                    txt += &format!(" {:11.5e} ", amp);
                    if k.rem_euclid(6) == 5 {
                        txt += &format!("\n");
                    }
                }
                txt += &format!("\n");
            }
        }
        let mut output = File::create(filename).unwrap();
        write!(output, "{}", txt);
    }

    pub fn calculate_density_on_grid(&self, density: ArrayView2<f32>) -> Array3<f32> {
        // Create the grid for each axis
        let x_grid = Array::linspace(self.xmin, self.xmax, self.nx);
        let y_grid = Array::linspace(self.ymin, self.ymax, self.ny);
        let z_grid = Array::linspace(self.zmin, self.zmax, self.nz);

        if !self.use_block_impl {
            // evaluate the basis on the grid
            let bfs_on_grid: Array4<f32> = evaluate_basis_on_grid(
                self.basis.basisfunctions.view(),
                x_grid.view(),
                y_grid.view(),
                z_grid.view(),
            );
            let rho: Array3<f32> = self.evaluate_density_on_grid_loop(density, bfs_on_grid.view());
            return rho;
        } else {
            return self.evaluate_density_on_grid_block(
                density,
                x_grid.view(),
                y_grid.view(),
                z_grid.view(),
            );
        }
    }

    pub fn evaluate_density_on_grid_loop(
        &self,
        density: ArrayView2<f32>,
        bfs_on_grid: ArrayView4<f32>,
    ) -> Array3<f32> {
        let mut rho: Array3<f32> = Array3::zeros((self.nx, self.ny, self.nz));
        let pmax = density.max().unwrap();
        let thresh: f32 = pmax * self.threshold;

        for (bfs_i, dens_i) in bfs_on_grid.outer_iter().zip(density.outer_iter()) {
            for (bfs_j, dens_ij) in bfs_on_grid.outer_iter().zip(dens_i.iter()) {
                if dens_ij.abs() > thresh {
                    rho = rho + *dens_ij * &bfs_i * &bfs_j;
                }
            }
        }
        rho
    }

    pub fn evaluate_density_on_grid_block(
        &self,
        density: ArrayView2<f32>,
        x_grid: ArrayView1<f64>,
        y_grid: ArrayView1<f64>,
        z_grid: ArrayView1<f64>,
    ) -> Array3<f32> {
        let n_block: usize = self.n_blocks;
        let basis_len: usize = density.dim().0;
        let c_len: usize = basis_len / n_block;
        let mut rho: Array3<f32> = Array3::zeros((self.nx, self.ny, self.nz));

        for i in 0..n_block {
            let mut chunck: usize = c_len * (i + 1);
            if i == (n_block - 1) {
                chunck = basis_len;
            }
            let start_i: usize = i * c_len;

            let bf_grid_i: Array4<f32> = evaluate_basis_on_grid(
                self.basis.basisfunctions.slice(s![start_i..chunck]),
                x_grid,
                y_grid,
                z_grid,
            );

            for j in 0..n_block {
                if j == i {
                    let p_i: ArrayView2<f32> = density.slice(s![start_i..chunck, start_i..chunck]);
                    // do einsum tensor product
                    // ij, jxyz -> ixyz
                    let tmp: Array4<f32> = p_i
                        .dot(
                            &bf_grid_i
                                .view()
                                .into_shape([chunck - start_i, self.nx * self.ny * self.nz])
                                .unwrap(),
                        )
                        .into_shape((chunck - start_i, self.nx, self.ny, self.nz))
                        .unwrap();

                    // iterate over first dimension
                    for (tmp_grid, bf_grid) in tmp.outer_iter().zip(bf_grid_i.outer_iter()) {
                        rho = rho + &tmp_grid * &bf_grid;
                    }
                } else {
                    let mut chunck_j: usize = c_len * (j + 1);
                    let start_j: usize = j * c_len;
                    if j == (n_block - 1) {
                        chunck_j = basis_len;
                    }
                    let p_j: ArrayView2<f32> =
                        density.slice(s![start_i..chunck, start_j..chunck_j]);
                    let bf_grid_j: Array4<f32> = evaluate_basis_on_grid(
                        self.basis.basisfunctions.slice(s![start_j..chunck_j]),
                        x_grid,
                        y_grid,
                        z_grid,
                    );

                    // do einsum tensor products
                    // ij, jxyz -> ixyz
                    let tmp: Array4<f32> = p_j
                        .dot(
                            &bf_grid_j
                                .view()
                                .into_shape([chunck_j - start_j, self.nx * self.ny * self.nz])
                                .unwrap(),
                        )
                        .into_shape((chunck - start_i, self.nx, self.ny, self.nz))
                        .unwrap();

                    // iterate over first dimension
                    for (tmp_grid, bf_grid) in tmp.outer_iter().zip(bf_grid_i.outer_iter()) {
                        rho = rho + &tmp_grid * &bf_grid;
                    }
                }
            }
        }
        rho
    }
}

pub fn evaluate_basis_on_grid(
    basis: ArrayView1<AtomicBasisFunction>,
    x_grid: ArrayView1<f64>,
    y_grid: ArrayView1<f64>,
    z_grid: ArrayView1<f64>,
) -> Array4<f32> {
    // create 4-dimensional array
    let mut bfs_on_grid: Array4<f32> =
        Array4::zeros((basis.len(), x_grid.len(), y_grid.len(), z_grid.len()));

    // iterate over the basis functions and the three axes
    for (bfs, mut bfs_arr) in basis.iter().zip(bfs_on_grid.axis_iter_mut(Axis(0))) {
        for (x_val, mut arr_x) in x_grid.iter().zip(bfs_arr.axis_iter_mut(Axis(0))) {
            for (y_val, mut arr_y) in y_grid.iter().zip(arr_x.axis_iter_mut(Axis(0))) {
                for (z_val, z_arr) in z_grid.iter().zip(arr_y.iter_mut()) {
                    *z_arr = bfs.eval(*x_val, *y_val, *z_val) as f32;
                }
            }
        }
    }
    bfs_on_grid
}
