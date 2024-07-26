use crate::constants;
use crate::fmo::SuperSystem;
use crate::initialization::System;
use crate::optimization::helpers::*;
use crate::scc::scc_routine::RestrictedSCC;
use log::{debug, info, log_enabled, trace, warn, Level};
use ndarray::prelude::*;
use ndarray_linalg::Norm;
use ndarray_stats::QuantileExt;

#[macro_export]
macro_rules! impl_optimize_cartesian {
    () => {
        pub fn optimize_cartesian(&mut self, state: usize) {
            // solve the following optimization problem:
            // minimize f(x) subject to  c_i(x) > 0   for  i=1,...,m
            // where f(x) is a scalar function, x is a real vector of size n
            // References
            // ----------
            // [1] J. Nocedal, S. Wright, 'Numerical Optimization', Springer, 2006
            if log_enabled!(Level::Warn) {
                warn!("{:^80}", "");
                warn!("{: ^80}", "Geometry optimization");
                warn!("{:-^80}", "");
                warn!("");
            }

            let n_atoms: usize = self.atoms.len();
            // start the optimization
            let (coordinates, gradient) = self.cartesian_optimization_loop(state);

            let new_coords: Array2<f64> =
                constants::BOHR_TO_ANGS * coordinates.into_shape((n_atoms, 3)).unwrap();
            if log_enabled!(Level::Warn) {
                warn!(" ");
                warn!("final coordinates after the optimization:");
                for (ind, atom) in self.atoms.iter().enumerate() {
                    warn!(
                        "{: >5} {:>18.10} {:>18.10} {:>18.10}",
                        atom.name,
                        new_coords[[ind, 0]],
                        new_coords[[ind, 1]],
                        new_coords[[ind, 2]]
                    );
                }
                warn!("");
            }
        }
    };
}

#[macro_export]
macro_rules! impl_cartesian_loop {
    () => {
        pub fn cartesian_optimization_loop(&mut self, state: usize) -> (Array1<f64>, Array1<f64>){
            // get coordinates
            let coords: Array1<f64> = self.get_xyz();
            let n_atoms: usize = self.atoms.len();

            let maxiter: usize = self.config.opt.geom_opt_max_cycles;
            let gtol: f64 = 0.000001 * self.config.opt.geom_opt_tol_gradient;
            let ftol: f64 = 0.000001 * self.config.opt.geom_opt_tol_energy;
            let stol: f64 = 0.000001 * self.config.opt.geom_opt_tol_displacement;
            let mut use_bfgs:bool = true;
            let mut use_line_search:bool = true;

            let n: usize = coords.len();
            let mut x_old: Array1<f64> = coords.clone();

            // variables for the storage of the energy and gradient
            let mut fk: f64 = 0.0;
            let mut grad_fk: Array1<f64> = Array::zeros(n);

            // calculate energy and gradient
            let tmp: (f64, Array1<f64>) = self.opt_energy_and_gradient(state);
            fk = tmp.0;
            grad_fk = tmp.1;

            let mut pk: Array1<f64> = Array::zeros(n);
            let mut x_kp1: Array1<f64> = Array::zeros(n);
            let mut sk: Array1<f64> = Array::zeros(n);
            let mut yk: Array1<f64> = Array::zeros(n);
            let mut inv_hk: Array2<f64> = Array::eye(n);
            let mut iterations:usize = 0;

            // vector of atom names
            let atom_names: Vec<String> = self
                .atoms
                .iter()
                .map(|atom| String::from(atom.name))
                .collect();
            let first_coords: Array2<f64> =
                constants::BOHR_TO_ANGS * &coords.view().into_shape([n_atoms, 3]).unwrap();
            let xyz_out: XYZ_Output = XYZ_Output::new(atom_names.clone(), first_coords);

            write_xyz_custom(&xyz_out,true);

            'optimization_loop: for k in 0..maxiter {
                if use_bfgs{
                    if k > 0 {
                        if yk.dot(&sk) <= 0.0 {
                            println!("yk {}", yk);
                            println!("sk {}", sk);
                            println!("Warning: positive definiteness of Hessian approximation lost in BFGS update, since yk.sk <= 0!")
                        }

                        inv_hk = bfgs_update(inv_hk.view(), sk.view(), yk.view(), k);
                    }
                    pk = inv_hk.dot(&(-&grad_fk));
                } else{
                    pk = -grad_fk.clone();
                }

                if use_line_search{
                    x_kp1 = self.line_search(x_old.view(), fk, grad_fk.view(), pk.view(), state);
                } else{
                    let amax = 1.0;
                    x_kp1 = &x_old + &(amax * &pk);
                }
                let mut f_kp1: f64 = 0.0;
                let mut grad_f_kp1: Array1<f64> = Array::zeros(n);

                // update coordinates
                self.update_xyz(x_kp1.view());
                // calculate new energy and gradient
                let tmp: (f64, Array1<f64>) = self.opt_energy_and_gradient(state);
                f_kp1 = tmp.0;
                grad_f_kp1 = tmp.1;

                // check convergence
                let f_change: f64 = (f_kp1 - fk).abs();
                let gnorm: f64 = grad_f_kp1.norm();

                let cnvg = |c| {
                    if c {
                        "Yes"
                    } else {
                        "No"
                    }
                };

                warn!("Optimization Cycle: {:5>}", k + 1);
                warn!(" ");

                // print convergence criteria
                warn!("{:>37}     {}     {}", "Maximum", "Tolerance", "Cnvgd?");
                warn!(
                    "          {:<19} {:2.6}     {:2.6}     {}",
                    "Gradient",
                    grad_fk.max().unwrap(),
                    gtol,
                    cnvg(grad_fk.max().unwrap() < &gtol),
                );
                warn!(
                    "          {:<19} {:2.6}     {:2.6}     {}",
                    "Displacement",
                    sk.max().unwrap(),
                    stol,
                    cnvg(sk.max().unwrap() < &stol),
                );
                warn!(
                    "          {:<19} {:2.6}     {:2.6}     {}",
                    "Energy change",
                    f_change,
                    ftol,
                    cnvg(f_change < ftol),
                );
                warn!(" ");

                if f_change < ftol && grad_fk.max().unwrap() < &gtol && sk.max().unwrap() < &stol {
                    // set the last coordinates and gradient
                    sk = &x_kp1 - &x_old;
                    x_old = x_kp1;
                    grad_fk = grad_f_kp1;
                    fk = f_kp1;
                    iterations = k;
                    break 'optimization_loop;
                }

                // step vector
                sk = &x_kp1 - &x_old;
                // gradient difference vector
                yk = &grad_f_kp1 - &grad_fk;
                // new variables for step k become old ones for step k+1
                x_old = x_kp1.clone();
                fk = f_kp1;
                grad_fk = grad_f_kp1;

                let new_coords: Array2<f64> =
                    constants::BOHR_TO_ANGS * &x_old.view().into_shape((n_atoms, 3)).unwrap();
                let xyz_out: XYZ_Output = XYZ_Output::new(
                    atom_names.clone(),
                    new_coords.clone().into_shape([n_atoms, 3]).unwrap(),
                );
                write_xyz_custom(&xyz_out,false);

                let opt_energy: OptEnergyOutput = OptEnergyOutput::new(k, fk);
                if k == 0{
                    write_opt_energy(&opt_energy,true);
                }
                else{
                    write_opt_energy(&opt_energy,false);
                }
            }
            let new_coords:Array2<f64> = constants::BOHR_TO_ANGS * &x_old.view().into_shape((n_atoms,3)).unwrap();
            let xyz_out:XYZ_Output =
                XYZ_Output::new(
                    atom_names.clone(),
                    new_coords.clone().into_shape([n_atoms,3]).unwrap());
            write_last_geom(&xyz_out);
            let opt_energy:OptEnergyOutput = OptEnergyOutput::new(iterations,fk);
            write_opt_energy(&opt_energy,false);

            return (x_old, grad_fk);
        }
    }
}

impl System {
    impl_cartesian_loop!();
    impl_optimize_cartesian!();

    pub fn opt_energy_and_gradient(&mut self, state: usize) -> (f64, Array1<f64>) {
        let mut energy: f64 = 0.0;
        let mut gradient: Array1<f64> = Array::zeros(3 * self.n_atoms);

        if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            energy = self.run_scc().unwrap();
            gradient = self.ground_state_gradient(false);
        } else {
            // excited state calculation
            let excited_state: usize = state - 1;
            self.prepare_scc();
            energy = self.run_scc().unwrap();

            // calculate excited states
            self.calculate_excited_states(false);
            energy += self.properties.ci_eigenvalue(excited_state).unwrap();

            gradient = self.ground_state_gradient(true);
            gradient = gradient + self.calculate_excited_state_gradient(excited_state);
        }
        self.properties.reset();

        return (energy, gradient);
    }

    pub fn calculate_energy_line_search(&mut self, state: usize) -> f64 {
        let mut energy: f64 = 0.0;

        if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            energy = self.run_scc().unwrap();
        } else {
            // excited state calculation
            let excited_state: usize = state - 1;
            self.prepare_scc();
            energy = self.run_scc().unwrap();

            // calculate excited states
            self.calculate_excited_states(false);
            energy += self.properties.ci_eigenvalue(excited_state).unwrap();
        }
        self.properties.reset();

        return (energy);
    }
}

impl SuperSystem<'_> {
    impl_cartesian_loop!();
    impl_optimize_cartesian!();

    pub fn opt_energy_and_gradient(&mut self, state: usize) -> (f64, Array1<f64>) {
        let mut energy: f64 = 0.0;
        let n_atoms: usize = self.atoms.len();
        let mut gradient: Array1<f64> = Array::zeros(3 * n_atoms);

        if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            energy = self.run_scc().unwrap();
            gradient = self.ground_state_gradient();
        } else {
            panic!(
                "The optimization procedure for the fmo systems is restricted to the ground
            state"
            );
        }
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.properties.reset();

        (energy, gradient)
    }

    pub fn calculate_energy_line_search(&mut self, state: usize) -> f64 {
        let mut energy: f64 = 0.0;

        if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            energy = self.run_scc().unwrap();
        } else {
            panic!(
                "The optimization procedure for the fmo systems is restricted to the ground
            state"
            );
        }
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset();
        }
        self.properties.reset();

        energy
    }
}
