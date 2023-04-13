use crate::constants;
use crate::initialization::Simulation;
use ndarray::prelude::*;
use ndarray::{array, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{into_col, into_row, Inverse};
use rand_distr::{Distribution, Normal};

impl Simulation {
    pub fn get_random_terms(&self) -> (Array2<f64>, Array2<f64>) {
        let n_at: usize = self.masses.len();
        let gdt: Array1<f64> = self.stepsize * &self.friction;
        let egdt: Array1<f64> = gdt.mapv(|val| (-1.0 * val).exp());
        let pterm: Array1<f64> = 2.0 * gdt - 3.0 + (4.0 - egdt.clone()) * &egdt;
        let vterm: Array1<f64> = 1.0 - egdt.mapv(|val| val.powi(2));

        let ktm: Array1<f64> =
            constants::K_BOLTZMANN * self.config.thermostat_config.temperature / &self.masses;

        let mut rho: Array1<f64> = Array1::zeros(n_at);
        let mut psig: Array1<f64> = Array1::zeros(n_at);
        let mut rhoc: Array1<f64> = Array1::zeros(n_at);

        for index in 0..n_at {
            if self.friction[index] != 0.0 {
                rho[index] = (1.0 - egdt[index]).powi(2) / (pterm[index] * vterm[index]).sqrt();
                psig[index] = (ktm[index] * pterm[index]).sqrt() / self.friction[index];
                rhoc[index] = (1.0 - rho[index].powi(2)).sqrt();
            }
        }

        let vsig: Array1<f64> = (&ktm * &vterm).mapv(|val| val.sqrt());
        let mut prand: Array2<f64> = Array2::zeros((n_at, 3));
        let mut vrand: Array2<f64> = Array2::zeros((n_at, 3));

        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..n_at {
            for j in 0..3 {
                let pnorm: f64 = normal.sample(&mut rand::thread_rng());
                let vnorm: f64 = normal.sample(&mut rand::thread_rng());
                prand[[i, j]] = psig[i] * pnorm;
                vrand[[i, j]] = vsig[i] * (rho[i] * pnorm + rhoc[i] * vnorm);
            }
        }
        (vrand, prand)
    }

    pub fn get_e_factor_langevin(&self) -> Array1<f64> {
        let efactor: Array1<f64> = 1.0 - (self.stepsize * -&self.friction).mapv(|val| val.exp());
        efactor
    }

    // Velocity Verlet routines
    pub fn get_coord_verlet(&self) -> Array2<f64> {
        let new_coords: Array2<f64> = &self.coordinates
            + &(self.stepsize * &self.velocities + 0.5 * self.stepsize.powi(2) * &self.forces);
        new_coords
    }

    pub fn get_velocities_verlet(&self, old_forces: ArrayView2<f64>) -> Array2<f64> {
        let new_velocities: Array2<f64> =
            &self.velocities + &(self.stepsize * 0.5 * &(&old_forces + &self.forces));
        new_velocities
    }

    pub fn get_coordinates_langevin(&self) -> Array2<f64> {
        let n_at: usize = self.friction.len();
        let mut new_coords: Array2<f64> = Array2::zeros(self.coordinates.raw_dim());
        for ind in 0..n_at {
            if self.friction[ind] != 0.0 {
                for j in 0..3 {
                    new_coords[[ind, j]] = self.coordinates[[ind, j]]
                        + self.saved_efactor[ind] * self.velocities[[ind, j]] / self.friction[ind]
                        + (self.friction[ind] * self.stepsize - self.saved_efactor[ind])
                            * self.forces[[ind, j]]
                            / self.friction[ind].powi(2)
                        + self.saved_p_rand[[ind, j]];
                }
            } else {
                for j in 0..3 {
                    new_coords[[ind, j]] = self.coordinates[[ind, j]]
                        + self.stepsize * self.velocities[[ind, j]]
                        + 0.5 * self.stepsize.powi(2) * self.forces[[ind, j]];
                }
            }
        }
        new_coords
    }

    pub fn get_velocities_langevin(
        &self,
        old_forces: ArrayView2<f64>,
        vrand: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n_at: usize = self.friction.len();
        let mut new_velocities: Array2<f64> = Array2::zeros(self.velocities.raw_dim());
        for ind in 0..n_at {
            if self.friction[ind] != 0.0 {
                for j in 0..3 {
                    new_velocities[[ind, j]] = self.velocities[[ind, j]]
                        * (-self.friction[ind] * self.stepsize).exp()
                        + 0.5
                            * self.saved_efactor[ind]
                            * (old_forces[[ind, j]] + self.forces[[ind, j]])
                            / self.friction[ind]
                        + vrand[[ind, j]];
                }
            } else {
                for j in 0..3 {
                    new_velocities[[ind, j]] = self.velocities[[ind, j]]
                        + self.stepsize * 0.5 * (old_forces[[ind, j]] + self.forces[[ind, j]]);
                }
            }
        }
        new_velocities
    }

    pub fn shift_to_center_of_mass(&self) -> Array2<f64> {
        let vec: Array1<f64> =
            get_center_of_mass(self.coordinates.view(), self.masses.view(), self.total_mass);
        let diff: Array2<f64> = &self.coordinates - &vec;
        diff
    }

    pub fn eliminate_translation_rotation_from_velocity(&self) -> Array2<f64> {
        let momentum: Array1<f64> = get_momentum(self.masses.view(), self.velocities.view());

        let mut new_velocities: Array2<f64> =
            eliminate_translation(self.velocities.view(), momentum.view(), self.total_mass);

        let angular_momentum: Array1<f64> = get_angular_momentum(
            self.coordinates.view(),
            self.masses.view(),
            new_velocities.view(),
        );

        let inertia: Array2<f64> =
            get_moment_of_inertia(self.coordinates.view(), self.masses.view());

        let angular_velocities: Array1<f64> =
            get_angular_velocity(angular_momentum.view(), inertia.view());

        new_velocities = eliminate_rotation(
            self.coordinates.view(),
            new_velocities.view(),
            angular_velocities.view(),
        );

        new_velocities
    }

    pub fn get_kinetic_energy(&self) -> f64 {
        let mut kinetic: f64 = 0.0;
        for index in 0..self.masses.len() {
            kinetic += self.masses[index]
                * 0.5
                * self
                    .velocities
                    .slice(s![index, ..])
                    .mapv(|val| val.powi(2))
                    .sum();
        }
        kinetic
    }
}

pub fn get_center_of_mass(
    coordinates: ArrayView2<f64>,
    masses: ArrayView1<f64>,
    total_mass: f64,
) -> Array1<f64> {
    let mut x: f64 = 0.0;
    let mut y: f64 = 0.0;
    let mut z: f64 = 0.0;

    for index in 0..masses.len() {
        x += masses[index] * coordinates[[index, 0]] / total_mass;
        y += masses[index] * coordinates[[index, 1]] / total_mass;
        z += masses[index] * coordinates[[index, 2]] / total_mass;
    }
    return array![x, y, z];
}

pub fn get_momentum(masses: ArrayView1<f64>, velocities: ArrayView2<f64>) -> Array1<f64> {
    let p: Array2<f64> = &velocities.t() * &masses;
    let arr: Array1<f64> = p.sum_axis(Axis(1));
    arr
}

pub fn eliminate_translation_rotation_from_velocity(
    velocities: ArrayView2<f64>,
    masses: ArrayView1<f64>,
    coordinates: ArrayView2<f64>,
    total_mass: f64,
) -> Array2<f64> {
    let momentum: Array1<f64> = get_momentum(masses, velocities);

    let mut new_velocities: Array2<f64> =
        eliminate_translation(velocities, momentum.view(), total_mass);

    let angular_momentum: Array1<f64> =
        get_angular_momentum(coordinates, masses, new_velocities.view());

    let inertia: Array2<f64> = get_moment_of_inertia(coordinates, masses);

    let angular_velocities: Array1<f64> =
        get_angular_velocity(angular_momentum.view(), inertia.view());

    new_velocities = eliminate_rotation(
        coordinates,
        new_velocities.view(),
        angular_velocities.view(),
    );

    new_velocities
}

pub fn eliminate_translation(
    velocities: ArrayView2<f64>,
    momentum: ArrayView1<f64>,
    total_mass: f64,
) -> Array2<f64> {
    let diff: Array2<f64> = &velocities - &(&momentum / total_mass);
    diff
}

pub fn cross_product(a: ArrayView1<f64>, b: ArrayView1<f64>) -> Array1<f64> {
    let arr: Array1<f64> = array![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
    arr
}

pub fn get_angular_momentum(
    coordinates: ArrayView2<f64>,
    masses: ArrayView1<f64>,
    velocities: ArrayView2<f64>,
) -> Array1<f64> {
    let mut angular: Array1<f64> = Array1::zeros(3);
    for index in 0..masses.len() {
        angular = angular
            + masses[index]
                * cross_product(
                    coordinates.slice(s![index, ..]),
                    velocities.slice(s![index, ..]),
                );
    }
    angular
}

pub fn get_moment_of_inertia(coordinates: ArrayView2<f64>, masses: ArrayView1<f64>) -> Array2<f64> {
    let mut i_mat: Array2<f64> = Array2::zeros((3, 3));
    let e_mat: Array2<f64> = Array::eye(3);
    for index in 0..masses.len() {
        let outer: Array2<f64> = into_col(coordinates.slice(s![index, ..]))
            .dot(&into_row(coordinates.slice(s![index, ..])));
        i_mat = i_mat
            + masses[index]
                * (coordinates
                    .slice(s![index, ..])
                    .dot(&coordinates.slice(s![index, ..]))
                    * e_mat.clone()
                    - outer);
    }
    i_mat
}

pub fn get_angular_velocity(
    angular_momentum: ArrayView1<f64>,
    inertia: ArrayView2<f64>,
) -> Array1<f64> {
    let mat: Array1<f64> = inertia.inv().unwrap().dot(&angular_momentum);
    mat
}

pub fn eliminate_rotation(
    coordinates: ArrayView2<f64>,
    velocities: ArrayView2<f64>,
    omega: ArrayView1<f64>,
) -> Array2<f64> {
    let mut new_velocities: Array2<f64> = velocities.to_owned();
    for index in 0..coordinates.dim().0 {
        new_velocities.slice_mut(s![index, ..]).assign(
            &(&velocities.slice(s![index, ..])
                - &cross_product(omega, coordinates.slice(s![index, ..]))),
        );
    }
    new_velocities
}
