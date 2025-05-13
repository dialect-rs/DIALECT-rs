use crate::constants;
use crate::fmo::SuperSystem;
use crate::initialization::System;
use crate::io::Configuration;
use crate::optimization::helpers::*;
use crate::scc::scc_routine::RestrictedSCC;
use crate::xtb::initialization::system::XtbSystem;
use log::{log_enabled, warn, Level};
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

#[macro_export]
macro_rules! impl_optimize_cartesian {
    () => {
        pub fn optimize_cartesian(&mut self, state: usize, config: &Configuration) {
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
            let (coordinates, _gradient) = self.cartesian_optimization_loop(state, config);

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
        pub fn cartesian_optimization_loop(&mut self, state: usize,config:&Configuration) -> (Array1<f64>, Array1<f64>){
            // get coordinates
            let coords: Array1<f64> = self.get_xyz();
            let n_atoms: usize = self.atoms.len();

            let maxiter: usize = self.config.opt.geom_opt_max_cycles;
            let gtol: f64 = 0.000001 * self.config.opt.geom_opt_tol_gradient;
            let ftol: f64 = 0.000001 * self.config.opt.geom_opt_tol_energy;
            let stol: f64 = 0.000001 * self.config.opt.geom_opt_tol_displacement;
            let use_bfgs:bool = config.opt.use_bfgs;
            let use_line_search:bool = config.opt.use_line_search;

            let n: usize = coords.len();
            let mut x_old: Array1<f64> = coords.clone();

            // calculate energy and gradient
            let tmp: (f64, Array1<f64>) = self.opt_energy_and_gradient(state);
            // variables for the storage of the energy and gradient
            let mut fk = tmp.0;
            let mut grad_fk = tmp.1;

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
            let xyz_out: XYZOutput = XYZOutput::new(atom_names.clone(), first_coords);

            write_xyz_custom(&xyz_out,true);

            'optimization_loop: for k in 0..maxiter {
                let pk: Array1<f64>;
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
                let mut f_kp1:f64 = 0.0;
                let f_change: f64 = (f_kp1 - fk).abs();

                let x_kp1:Array1<f64> = if use_line_search && f_change < 1.0e-3{
                    self.line_search(x_old.view(), fk, grad_fk.view(), pk.view(), state)
                } else{
                    let amax = 1.0;
                    &x_old + &(amax * &pk)
                };

                // update coordinates
                self.update_xyz(x_kp1.view());
                // calculate new energy and gradient
                let tmp: (f64, Array1<f64>) = self.opt_energy_and_gradient(state);
                f_kp1 = tmp.0;
                let grad_f_kp1:Array1<f64> = tmp.1;

                // check convergence
                let f_change: f64 = (f_kp1 - fk).abs();
                // let gnorm: f64 = grad_f_kp1.norm();

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
                    "          {:<19} {:2.8}     {:2.8}     {}",
                    "Gradient",
                    grad_fk.max().unwrap(),
                    gtol,
                    cnvg(grad_fk.max().unwrap() < &gtol),
                );
                warn!(
                    "          {:<19} {:2.8}     {:2.8}     {}",
                    "Displacement",
                    sk.max().unwrap(),
                    stol,
                    cnvg(sk.max().unwrap() < &stol),
                );
                warn!(
                    "          {:<19} {:2.8}     {:2.8}     {}",
                    "Energy change",
                    f_change,
                    ftol,
                    cnvg(f_change < ftol),
                );
                warn!(" ");

                if f_change < ftol && grad_fk.max().unwrap() < &gtol && sk.max().unwrap() < &stol {
                    // set the last coordinates and gradient
                    // sk = &x_kp1 - &x_old;
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
                let xyz_out: XYZOutput = XYZOutput::new(
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
            let xyz_out:XYZOutput =
                XYZOutput::new(
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
        let (energy, gradient): (f64, Array1<f64>) = if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            let tmp_energy = self.run_scc().unwrap();
            let tmp_gradient = self.ground_state_gradient(false);

            (tmp_energy, tmp_gradient)
        } else {
            // excited state calculation
            let excited_state: usize = state - 1;
            self.prepare_scc();
            let mut tmp_energy = self.run_scc().unwrap();

            // calculate excited states
            self.calculate_excited_states(false);
            tmp_energy += self.properties.ci_eigenvalue(excited_state).unwrap();

            let mut tmp_gradient = self.ground_state_gradient(true);
            tmp_gradient = tmp_gradient + self.calculate_excited_state_gradient(excited_state);

            (tmp_energy, tmp_gradient)
        };
        self.properties.reset_reduced();

        (energy, gradient)
    }

    pub fn calculate_energy_line_search(&mut self, state: usize) -> f64 {
        let energy: f64 = if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            self.run_scc().unwrap()
        } else {
            // excited state calculation
            let excited_state: usize = state - 1;
            self.prepare_scc();
            let mut tmp_energy: f64 = self.run_scc().unwrap();

            // calculate excited states
            self.calculate_excited_states(false);
            tmp_energy += self.properties.ci_eigenvalue(excited_state).unwrap();
            tmp_energy
        };
        self.properties.reset_reduced();

        energy
    }
}

impl XtbSystem {
    impl_cartesian_loop!();
    impl_optimize_cartesian!();

    pub fn opt_energy_and_gradient(&mut self, _state: usize) -> (f64, Array1<f64>) {
        // ground state energy and gradient
        self.prepare_scc();
        let energy = self.run_scc().unwrap();
        let gradient = self.ground_state_gradient();
        self.properties.reset_reduced();

        (energy, gradient)
    }

    pub fn calculate_energy_line_search(&mut self, _state: usize) -> f64 {
        // ground state energy
        self.prepare_scc();
        let energy = self.run_scc().unwrap();
        self.properties.reset_reduced();

        energy
    }
}

impl SuperSystem<'_> {
    impl_cartesian_loop!();
    impl_optimize_cartesian!();

    pub fn opt_energy_and_gradient(&mut self, state: usize) -> (f64, Array1<f64>) {
        let (energy, gradient): (f64, Array1<f64>) = if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            let tmp_energy = self.run_scc().unwrap();
            let tmp_gradient = self.ground_state_gradient();

            (tmp_energy, tmp_gradient)
        } else {
            panic!(
                "The optimization procedure for the fmo systems is restricted to the ground
            state"
            );
        };
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset_reduced();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_reduced();
        }
        self.properties.reset_reduced();

        (energy, gradient)
    }

    pub fn calculate_energy_line_search(&mut self, state: usize) -> f64 {
        let energy: f64 = if state == 0 {
            // ground state energy and gradient
            self.prepare_scc();
            self.run_scc().unwrap()
        } else {
            panic!(
                "The optimization procedure for the fmo systems is restricted to the ground
            state"
            );
        };
        for monomer in self.monomers.iter_mut() {
            monomer.properties.reset_reduced();
        }
        for pair in self.pairs.iter_mut() {
            pair.properties.reset_reduced();
        }
        self.properties.reset_reduced();

        energy
    }
}
