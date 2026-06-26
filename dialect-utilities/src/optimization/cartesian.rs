//! Cartesian geometry-optimization driver macros. Generic over the system
//! type: the macros are expanded inside `impl <System>` blocks in the
//! method driver crates. Internal paths use `$crate::optimization::helpers`.

#[macro_export]
macro_rules! impl_optimize {
    () => {
        pub fn optimize(&mut self, state: usize, config: &Configuration) {
            // choose optimizer
            if config.opt.use_advanced_optimizer {
                if config.opt.optimizer_version == 2 {
                    self.optimize_cartesian_v2(state, config);
                } else {
                    self.optimize_cartesian_v3(state, config);
                }
            } else {
                self.optimize_cartesian(state, config);
            }
        }
    };
}

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

            // Configuration parameters
            let maxiter: usize = config.opt.geom_opt_max_cycles;
            let convergence_level = &config.opt.convergence_level;
            let criteria = ConvergenceCriteria::from_level(convergence_level);
            // let gtol: f64 = 0.000001 * self.config.opt.geom_opt_tol_gradient;
            // let ftol: f64 = 0.000001 * self.config.opt.geom_opt_tol_energy;
            // let stol: f64 = 0.000001 * self.config.opt.geom_opt_tol_displacement;
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
            // Convergence helper
            let cnvg = |c: bool| if c { "Yes" } else { "No" };

            'optimization_loop: for k in 0..maxiter {
                let pk: Array1<f64>;
                if use_bfgs{
                    if k > 0 {
                        if yk.dot(&sk) <= 0.0 {
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

                // 8. Check convergence
                let (converged, status) = criteria.check(&grad_f_kp1, &sk, f_change);

                // Print convergence status
                warn!("Optimization Cycle: {:5>}", k + 1);
                warn!(" ");

                // print convergence criteria
                warn!("{:>37}     {}     {}", "Maximum", "Tolerance", "Cnvgd?");
                warn!(
                    "          {:<19} {:2.8}     {:2.8}     {}",
                    "Gradient",
                    status.grad_max,
                    criteria.grad_max,
                    cnvg(status.grad_max < criteria.grad_max),
                );
                warn!(
                    "          {:<19} {:2.8}     {:2.8}     {}",
                    "Displacement",
                    status.disp_max,
                    criteria.disp_max,
                    cnvg(status.disp_max < criteria.disp_max),
                );
                warn!(
                    "          {:<19} {:2.8}     {:2.8}     {}",
                    "Energy change",
                    status.energy_change,
                    criteria.energy,
                    cnvg(status.energy_change < criteria.energy),
                );
                warn!(" ");

                if converged {
                    iterations = k + 1;
                    x_old = x_kp1;
                    grad_fk = grad_f_kp1;
                    fk = f_kp1;
                    warn!(
                        "*** Optimization converged in {} iterations ***",
                        iterations
                    );
                    break 'optimization_loop;
                }

                // if f_change < ftol && grad_fk.max().unwrap() < &gtol && sk.max().unwrap() < &stol {
                //     // set the last coordinates and gradient
                //     // sk = &x_kp1 - &x_old;
                //     x_old = x_kp1;
                //     grad_fk = grad_f_kp1;
                //     fk = f_kp1;
                //     iterations = k;
                //     break 'optimization_loop;
                // }

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

// =============================================================================
// Advanced Cartesian Optimization (v2)
// =============================================================================

/// Macro for v2 optimization entry point with advanced optimizer selection
#[macro_export]
macro_rules! impl_optimize_cartesian_v2 {
    () => {
        /// Geometry optimization using advanced v2 algorithm with damped BFGS
        pub fn optimize_cartesian_v2(&mut self, state: usize, config: &Configuration) {
            if log_enabled!(Level::Warn) {
                warn!("{:^80}", "");
                warn!("{: ^80}", "Geometry optimization (v2 - Advanced)");
                warn!("{:-^80}", "");
                warn!("");
            }

            let n_atoms: usize = self.atoms.len();
            let (coordinates, _gradient) = self.cartesian_optimization_loop_v2(state, config);

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

/// Macro implementing Cartesian optimization loop for non-FMO systems with:
/// - L-BFGS with full model Hessian as H0 preconditioner
/// - Energy-based trust region accept/reject (reliable for non-FMO)
/// - Conservative trust adaptation to avoid accept-reject oscillation
/// - Standard convergence (all 5 criteria must be satisfied)
#[macro_export]
macro_rules! impl_cartesian_loop_v2 {
    () => {
        pub fn cartesian_optimization_loop_v2(
            &mut self,
            state: usize,
            config: &Configuration,
        ) -> (Array1<f64>, Array1<f64>) {
            use $crate::optimization::helpers::{
                ConvergenceCriteria, TrustRegion, LBFGS,
            };

            // Get initial coordinates
            let coords: Array1<f64> = self.get_xyz();
            let n_atoms: usize = self.atoms.len();

            // Configuration parameters
            let maxiter: usize = config.opt.geom_opt_max_cycles;
            let convergence_level = &config.opt.convergence_level;
            let criteria = ConvergenceCriteria::from_level(convergence_level);

            // Initialize trust region with conservative parameters.
            // Max radius 0.5 Bohr prevents overshooting that leads to
            // accept-reject oscillation common in molecular systems.
            let mut trust = TrustRegion::new(0.3);
            trust.max_radius = 0.5;

            // Build full model Hessian and create L-BFGS with it as preconditioner.
            // m=20 pairs: enough history for good curvature refinement, bad pairs
            // age out automatically after m steps.
            let h0 = self.model_hessian_full(&coords);
            let mut lbfgs = LBFGS::new_with_full_h0(20, h0);

            // Initial state
            let mut x_current: Array1<f64> = coords.clone();
            let (mut fk, mut grad) = self.opt_energy_and_gradient(state);
            let mut iterations: usize = 0;
            let mut consecutive_rejections: usize = 0;

            // Vector of atom names for output
            let atom_names: Vec<String> = self
                .atoms
                .iter()
                .map(|atom| String::from(atom.name))
                .collect();

            // Write initial coordinates
            let first_coords: Array2<f64> =
                constants::BOHR_TO_ANGS * &coords.view().into_shape([n_atoms, 3]).unwrap();
            let xyz_out: XYZOutput = XYZOutput::new(atom_names.clone(), first_coords);
            write_xyz_custom(&xyz_out, true);

            // Convergence helper
            let cnvg = |c: bool| if c { "Yes" } else { "No" };

            // Main optimization loop
            'optimization_loop: for k in 0..maxiter {
                // 1. Compute search direction from L-BFGS
                let pk = lbfgs.search_direction(&grad);

                // Ensure descent direction
                let directional_deriv = grad.dot(&pk);
                let pk = if directional_deriv >= 0.0 {
                    warn!("  L-BFGS direction not descent, resetting history");
                    lbfgs.clear();
                    -&grad
                } else {
                    pk
                };

                // 2. Apply trust region constraint.
                // When gradient criteria are already satisfied, clamp trust to
                // disp_max to ensure displacement convergence. The L-BFGS inverse
                // Hessian amplifies soft modes, producing large step norms even
                // with tiny gradients.
                let n_dof = (n_atoms as f64) * 3.0;
                let pre_grad_max = grad.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                let pre_grad_rms = (grad.dot(&grad) / n_dof).sqrt();
                let gradient_already_converged =
                    pre_grad_max < criteria.grad_max && pre_grad_rms < criteria.grad_rms;

                let effective_radius = if gradient_already_converged {
                    trust.radius.min(criteria.disp_max.max(0.0005))
                } else {
                    trust.radius
                };
                let pk_norm = pk.dot(&pk).sqrt();
                let sk = if pk_norm > effective_radius {
                    (effective_radius / pk_norm) * &pk
                } else {
                    pk
                };
                let step_norm = sk.dot(&sk).sqrt();

                // 3. Take the step and evaluate
                let x_new = &x_current + &sk;
                self.update_xyz(x_new.view());
                let (f_new, grad_new) = self.opt_energy_and_gradient(state);

                // 4. Accept/reject based on energy
                let actual_reduction = fk - f_new;
                let predicted_reduction = lbfgs.predicted_reduction(&grad, &sk);

                let rho = if predicted_reduction.abs() < 1e-14 {
                    if actual_reduction > 0.0 { 1.0 } else { 0.0 }
                } else {
                    actual_reduction / predicted_reduction
                };

                let de: f64;
                if actual_reduction > 0.0 {
                    // Accept: energy decreased
                    let yk = &grad_new - &grad;
                    lbfgs.update(sk.clone(), yk);
                    de = f_new - fk;
                    x_current = x_new;
                    fk = f_new;
                    grad = grad_new;
                    consecutive_rejections = 0;
                } else {
                    // Reject: energy increased, restore coordinates
                    de = 0.0;
                    self.update_xyz(x_current.view());
                    consecutive_rejections += 1;

                    if consecutive_rejections >= 5 {
                        warn!(
                            "  Stagnation ({} rejections), resetting L-BFGS + trust",
                            consecutive_rejections
                        );
                        lbfgs.clear();
                        let h0_new = self.model_hessian_full(&x_current);
                        lbfgs.set_h0_full(h0_new);
                        trust.reset();
                        consecutive_rejections = 0;
                    }
                }

                // 5. Trust radius adaptation (conservative for molecular systems)
                // Gentle expansion (1.5x) and moderate shrink (0.5x) to avoid
                // the accept-reject oscillation from standard 2x/0.25x updates.
                if rho > 0.75 && step_norm >= 0.8 * trust.radius {
                    trust.radius = (1.5 * trust.radius).min(trust.max_radius);
                } else if rho < 0.25 {
                    trust.radius = (0.5 * trust.radius).max(trust.min_radius);
                }

                // 6. Periodic H0 refresh
                if k > 0 && k % 50 == 0 {
                    let h0_new = self.model_hessian_full(&x_current);
                    lbfgs.set_h0_full(h0_new);
                }

                // 7. Check convergence (all 5 criteria required)
                let (converged, status) = criteria.check(&grad, &sk, de);

                // Print convergence status
                warn!("Optimization Cycle: {:5>}", k + 1);
                warn!(" ");
                warn!(
                    "{:>37}     {:>12}     {:>12}     {}",
                    "Value", "Maximum", "RMS", "Cnvgd?"
                );
                warn!(
                    "          {:<19} {:>12.8}     {:>12.8}     {}",
                    "Gradient (max/rms)",
                    status.grad_max,
                    status.grad_rms,
                    cnvg(status.grad_max_converged && status.grad_rms_converged),
                );
                warn!(
                    "          {:<19} {:>12.8}     {:>12.8}     {}",
                    "Displacement (max/rms)",
                    status.disp_max,
                    status.disp_rms,
                    cnvg(status.disp_max_converged && status.disp_rms_converged),
                );
                warn!(
                    "          {:<19} {:>12.8}     {:>12}     {}",
                    "Energy change",
                    status.energy_change,
                    "-",
                    cnvg(status.energy_converged),
                );
                warn!(
                    "          Trust radius: {:.6} Bohr (eff: {:.6}), L-BFGS pairs: {}",
                    trust.radius,
                    effective_radius,
                    lbfgs.len()
                );
                warn!(" ");

                if converged {
                    iterations = k + 1;
                    warn!(
                        "*** Optimization converged in {} iterations ***",
                        iterations
                    );
                    break 'optimization_loop;
                }

                // Write trajectory
                let new_coords: Array2<f64> =
                    constants::BOHR_TO_ANGS * &x_current.view().into_shape((n_atoms, 3)).unwrap();
                let xyz_out: XYZOutput = XYZOutput::new(
                    atom_names.clone(),
                    new_coords.clone().into_shape([n_atoms, 3]).unwrap(),
                );
                write_xyz_custom(&xyz_out, false);

                // Write energy
                let opt_energy: OptEnergyOutput = OptEnergyOutput::new(k, fk);
                write_opt_energy(&opt_energy, k == 0);

                iterations = k + 1;
            }

            // Write final geometry
            let final_coords: Array2<f64> =
                constants::BOHR_TO_ANGS * &x_current.view().into_shape((n_atoms, 3)).unwrap();
            let xyz_out: XYZOutput = XYZOutput::new(
                atom_names.clone(),
                final_coords.clone().into_shape([n_atoms, 3]).unwrap(),
            );
            write_last_geom(&xyz_out);

            let opt_energy: OptEnergyOutput = OptEnergyOutput::new(iterations, fk);
            write_opt_energy(&opt_energy, false);

            (x_current, grad)
        }
    };
}

/// Macro implementing Cartesian optimization loop for FMO systems with:
/// - L-BFGS with full model Hessian as H0 preconditioner
/// - Always-accept strategy (FMO energy noise makes energy-based reject unreliable)
/// - Gradient-based trust adaptation (energy-based rho is noisy)
/// - Gradient regression detection with L-BFGS history reset
/// - Phase-based trust clamping for displacement convergence
/// - Noise-tolerant convergence (grad+disp for 3 cycles, energy not required)
#[macro_export]
macro_rules! impl_cartesian_loop_v2_fmo {
    () => {
        pub fn cartesian_optimization_loop_v2(
            &mut self,
            state: usize,
            config: &Configuration,
        ) -> (Array1<f64>, Array1<f64>) {
            use $crate::optimization::helpers::{
                ConvergenceCriteria, TrustRegion, LBFGS,
            };

            // Get initial coordinates
            let coords: Array1<f64> = self.get_xyz();
            let n_atoms: usize = self.atoms.len();

            // Configuration parameters
            let maxiter: usize = config.opt.geom_opt_max_cycles;
            let convergence_level = &config.opt.convergence_level;
            let criteria = ConvergenceCriteria::from_level(convergence_level);

            // Initialize trust region with reasonable floor for molecular systems
            let mut trust = TrustRegion::new(0.3);
            trust.min_radius = 0.02;

            // Build full model Hessian and create L-BFGS with it as preconditioner.
            // m=20 pairs: enough history for good curvature refinement, but old
            // bad pairs age out automatically, preventing the Hessian divergence
            // that plagues full BFGS with noisy FMO gradients.
            let h0 = self.model_hessian_full(&coords);
            let mut lbfgs = LBFGS::new_with_full_h0(20, h0);

            // Initial state
            let mut x_current: Array1<f64> = coords.clone();
            let (mut fk, mut grad) = self.opt_energy_and_gradient(state);
            let mut grad_norm_sq = grad.dot(&grad);
            let mut best_grad_norm_sq = grad_norm_sq;
            let mut consecutive_good: usize = 0;
            let mut consecutive_bad: usize = 0;
            let mut consecutive_grad_disp_converged: usize = 0;
            let mut iterations: usize = 0;

            // Vector of atom names for output
            let atom_names: Vec<String> = self
                .atoms
                .iter()
                .map(|atom| String::from(atom.name))
                .collect();

            // Write initial coordinates
            let first_coords: Array2<f64> =
                constants::BOHR_TO_ANGS * &coords.view().into_shape([n_atoms, 3]).unwrap();
            let xyz_out: XYZOutput = XYZOutput::new(atom_names.clone(), first_coords);
            write_xyz_custom(&xyz_out, true);

            // Convergence helper
            let cnvg = |c: bool| if c { "Yes" } else { "No" };

            // Main optimization loop
            // L-BFGS trust-region: ONE energy+gradient evaluation per cycle.
            // No line search - trust region controls step size directly.
            'optimization_loop: for k in 0..maxiter {
                // 1. Compute search direction from L-BFGS two-loop recursion
                let pk = lbfgs.search_direction(&grad);

                // Ensure pk is a descent direction
                let directional_deriv = grad.dot(&pk);
                let pk = if directional_deriv >= 0.0 {
                    warn!("  L-BFGS direction not descent, resetting history");
                    lbfgs.clear();
                    -&grad
                } else {
                    pk
                };

                // 2. Apply trust region constraint.
                // When gradient criteria are already satisfied, switch to small
                // trust to achieve displacement convergence. This avoids the
                // problem of inverse Hessian soft modes producing large steps
                // even with small gradients. Outside this phase, use full trust
                // for maximum progress.
                let n_dof = (n_atoms as f64) * 3.0;
                let pre_grad_max = grad.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                let pre_grad_rms = (grad.dot(&grad) / n_dof).sqrt();
                let gradient_already_converged =
                    pre_grad_max < criteria.grad_max && pre_grad_rms < criteria.grad_rms;

                let effective_radius = if gradient_already_converged {
                    // Small trust to satisfy displacement convergence.
                    // Use disp_max as limit: since max(|sk_i|) <= ||sk|| <= trust,
                    // setting trust = disp_max guarantees disp_max criterion is met.
                    trust.radius.min(criteria.disp_max.max(0.0005))
                } else {
                    trust.radius
                };
                let pk_norm = pk.dot(&pk).sqrt();
                let sk = if pk_norm > effective_radius {
                    (effective_radius / pk_norm) * &pk
                } else {
                    pk
                };

                // 3. Take the step and evaluate
                let x_new = &x_current + &sk;
                self.update_xyz(x_new.view());
                let (f_new, grad_new) = self.opt_energy_and_gradient(state);

                // 4. Always accept the step (FMO energy noise makes energy-based
                // accept/reject unreliable). Always update L-BFGS - Powell's
                // damping ensures positive curvature, and limited memory
                // automatically forgets old/bad pairs after m steps.
                let grad_new_norm_sq = grad_new.dot(&grad_new);
                let yk = &grad_new - &grad;
                lbfgs.update(sk.clone(), yk);

                // Track gradient improvement for trust adaptation
                if grad_new_norm_sq <= grad_norm_sq {
                    consecutive_good += 1;
                    consecutive_bad = 0;
                } else {
                    consecutive_bad += 1;
                    consecutive_good = 0;
                }

                let de = f_new - fk;
                x_current = x_new;
                fk = f_new;
                grad = grad_new;

                // 5. Trust radius adaptation (conservative, symmetric)
                if consecutive_good >= 3 {
                    trust.radius = (1.3 * trust.radius).min(trust.max_radius);
                    consecutive_good = 0;
                }
                if consecutive_bad >= 5 {
                    trust.radius = (0.7 * trust.radius).max(trust.min_radius);
                    consecutive_bad = 0;
                }

                // 6. Track best gradient and detect regression
                grad_norm_sq = grad_new_norm_sq;
                if grad_norm_sq < best_grad_norm_sq {
                    best_grad_norm_sq = grad_norm_sq;
                }

                // Gradient regression: if 4x worse than best, accumulated
                // L-BFGS pairs are misleading. Clear history and rebuild H0.
                // Modest trust shrink (0.7x, not 0.5x) to avoid ratchet-down.
                if grad_norm_sq > 16.0 * best_grad_norm_sq {
                    warn!(
                        "  Gradient regression ({:.6} vs best {:.6}), clearing L-BFGS + rebuilding H0",
                        grad_norm_sq.sqrt(),
                        best_grad_norm_sq.sqrt()
                    );
                    lbfgs.clear();
                    let h0_new = self.model_hessian_full(&x_current);
                    lbfgs.set_h0_full(h0_new);
                    trust.radius = (0.7 * trust.radius).max(trust.min_radius);
                    best_grad_norm_sq = grad_norm_sq;
                    consecutive_good = 0;
                    consecutive_bad = 0;
                }

                // 7. Refresh model Hessian (H0) periodically.
                // The model Hessian depends on interatomic distances, so rebuild
                // it as the geometry changes to keep the preconditioner accurate.
                if k > 0 && k % 50 == 0 {
                    let h0_new = self.model_hessian_full(&x_current);
                    lbfgs.set_h0_full(h0_new);
                }

                // 8. Check convergence
                let (converged, status) = criteria.check(&grad, &sk, de);

                // Track grad+disp convergence for noise-tolerant convergence.
                // With FMO, energy fluctuates ~0.1 mH per step due to SCC noise,
                // making the energy criterion (1e-6 Hartree) impossible to satisfy.
                // If gradient AND displacement converge for 3 consecutive cycles,
                // declare convergence regardless of energy.
                let grad_disp_converged = status.grad_max_converged
                    && status.grad_rms_converged
                    && status.disp_max_converged
                    && status.disp_rms_converged;
                if grad_disp_converged {
                    consecutive_grad_disp_converged += 1;
                } else {
                    consecutive_grad_disp_converged = 0;
                }

                // Print convergence status
                warn!("Optimization Cycle: {:5>}", k + 1);
                warn!(" ");
                warn!(
                    "{:>37}     {:>12}     {:>12}     {}",
                    "Value", "Maximum", "RMS", "Cnvgd?"
                );
                warn!(
                    "          {:<19} {:>12.8}     {:>12.8}     {}",
                    "Gradient (max/rms)",
                    status.grad_max,
                    status.grad_rms,
                    cnvg(status.grad_max_converged && status.grad_rms_converged),
                );
                warn!(
                    "          {:<19} {:>12.8}     {:>12.8}     {}",
                    "Displacement (max/rms)",
                    status.disp_max,
                    status.disp_rms,
                    cnvg(status.disp_max_converged && status.disp_rms_converged),
                );
                warn!(
                    "          {:<19} {:>12.8}     {:>12}     {}",
                    "Energy change",
                    status.energy_change,
                    "-",
                    cnvg(status.energy_converged),
                );
                warn!(
                    "          Trust radius: {:.6} Bohr (eff: {:.6}), L-BFGS pairs: {}",
                    trust.radius,
                    effective_radius,
                    lbfgs.len()
                );
                warn!(" ");

                if converged || consecutive_grad_disp_converged >= 3 {
                    iterations = k + 1;
                    if converged {
                        warn!(
                            "*** Optimization converged in {} iterations ***",
                            iterations
                        );
                    } else {
                        warn!(
                            "*** Optimization converged in {} iterations (grad+disp criteria met for 3 cycles) ***",
                            iterations
                        );
                    }
                    break 'optimization_loop;
                }

                // Write trajectory
                let new_coords: Array2<f64> =
                    constants::BOHR_TO_ANGS * &x_current.view().into_shape((n_atoms, 3)).unwrap();
                let xyz_out: XYZOutput = XYZOutput::new(
                    atom_names.clone(),
                    new_coords.clone().into_shape([n_atoms, 3]).unwrap(),
                );
                write_xyz_custom(&xyz_out, false);

                // Write energy
                let opt_energy: OptEnergyOutput = OptEnergyOutput::new(k, fk);
                write_opt_energy(&opt_energy, k == 0);

                iterations = k + 1;
            }

            // Write final geometry
            let final_coords: Array2<f64> =
                constants::BOHR_TO_ANGS * &x_current.view().into_shape((n_atoms, 3)).unwrap();
            let xyz_out: XYZOutput = XYZOutput::new(
                atom_names.clone(),
                final_coords.clone().into_shape([n_atoms, 3]).unwrap(),
            );
            write_last_geom(&xyz_out);

            let opt_energy: OptEnergyOutput = OptEnergyOutput::new(iterations, fk);
            write_opt_energy(&opt_energy, false);

            (x_current, grad)
        }
    };
}

// =============================================================================
// End of Advanced Cartesian Optimization (v2 / v2_fmo)
// =============================================================================

// =============================================================================
// Advanced Cartesian Optimization (v3) - with GDIIS acceleration
// =============================================================================

/// Macro for v3 optimization entry point with GDIIS acceleration
#[macro_export]
macro_rules! impl_optimize_cartesian_v3 {
    () => {
        /// Geometry optimization using v3 algorithm with GDIIS acceleration
        pub fn optimize_cartesian_v3(&mut self, state: usize, config: &Configuration) {
            if log_enabled!(Level::Warn) {
                warn!("{:^80}", "");
                warn!("{: ^80}", "Geometry optimization (v3 - GDIIS accelerated)");
                warn!("{:-^80}", "");
                warn!("");
            }

            let n_atoms: usize = self.atoms.len();
            let (coordinates, _gradient) = self.cartesian_optimization_loop_v3(state, config);

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

/// Macro implementing v3 Cartesian optimization loop with:
/// - All v2 features (damped BFGS, Wolfe line search, trust region)
/// - GDIIS acceleration for faster convergence
/// - Model Hessian initialization
#[macro_export]
macro_rules! impl_cartesian_loop_v3 {
    () => {
        pub fn cartesian_optimization_loop_v3(
            &mut self,
            state: usize,
            config: &Configuration,
        ) -> (Array1<f64>, Array1<f64>) {
            use $crate::optimization::helpers::{
                build_model_hessian_diagonal, damped_bfgs_update, estimate_condition_number,
                solve_newton_step, ConvergenceCriteria, TrustRegion, GDIIS,
            };

            // Get initial coordinates
            let coords: Array1<f64> = self.get_xyz();
            let n_atoms: usize = self.atoms.len();

            // Get atomic numbers for model Hessian
            let atomic_numbers: Vec<u8> = self.atoms.iter().map(|a| a.number).collect();

            // Configuration parameters
            let maxiter: usize = config.opt.geom_opt_max_cycles;
            let convergence_level = &config.opt.convergence_level;
            let criteria = ConvergenceCriteria::from_level(convergence_level);

            // Initialize trust region
            let mut trust = TrustRegion::new(0.3);

            // Initialize GDIIS accelerator
            let mut gdiis = GDIIS::default();
            let mut gdiis_steps_since_reset: usize = 0;
            const GDIIS_RESET_INTERVAL: usize = 30;

            // Hessian: use model Hessian for better initial conditioning
            let mut hessian: Array2<f64> = build_model_hessian_diagonal(&atomic_numbers);
            let mut consecutive_resets: usize = 0;
            const MAX_RESETS: usize = 3;

            // Initial state
            let mut x_current: Array1<f64> = coords.clone();
            let (mut fk, mut grad) = self.opt_energy_and_gradient(state);
            let mut iterations: usize = 0;

            // Track GDIIS failures
            let mut gdiis_failures: usize = 0;

            // Vector of atom names for output
            let atom_names: Vec<String> = self
                .atoms
                .iter()
                .map(|atom| String::from(atom.name))
                .collect();

            // Write initial coordinates
            let first_coords: Array2<f64> =
                constants::BOHR_TO_ANGS * &coords.view().into_shape([n_atoms, 3]).unwrap();
            let xyz_out: XYZOutput = XYZOutput::new(atom_names.clone(), first_coords);
            write_xyz_custom(&xyz_out, true);

            // Convergence helper
            let cnvg = |c: bool| if c { "Yes" } else { "No" };

            let mut consecutive_rejections: usize = 0;

            // Main optimization loop
            // Pure trust-region approach with GDIIS acceleration.
            // No line search to avoid FMO energy drift from multiple SCC evaluations.
            'optimization_loop: for k in 0..maxiter {
                // 1. Check Hessian condition and possibly reset
                let cond_num = estimate_condition_number(&hessian);
                if cond_num > 1e12 {
                    warn!(
                        "  Hessian condition number too large ({:.2e}), resetting",
                        cond_num
                    );
                    hessian = build_model_hessian_diagonal(&atomic_numbers);
                    consecutive_resets += 1;
                    gdiis.clear();
                } else {
                    consecutive_resets = 0;
                }

                // 2. Compute search direction
                let pk: Array1<f64> = if consecutive_resets >= MAX_RESETS {
                    warn!("  Using steepest descent (too many Hessian resets)");
                    -&grad
                } else {
                    solve_newton_step(&hessian, &grad)
                };

                // Ensure pk is a descent direction
                let directional_deriv = grad.dot(&pk);
                let pk = if directional_deriv >= 0.0 {
                    warn!("  Search direction not descent, using steepest descent");
                    gdiis.clear();
                    -&grad
                } else {
                    pk
                };

                // 3. Apply trust region constraint
                let sk = trust.apply(&pk);
                let step_norm = sk.dot(&sk).sqrt();

                // 4. Take the step and evaluate (single energy+gradient evaluation)
                let x_new = &x_current + &sk;
                self.update_xyz(x_new.view());
                let (f_new, grad_new) = self.opt_energy_and_gradient(state);

                // 5. Compute quality ratio for trust region adjustment
                let actual_reduction = fk - f_new;
                let predicted_reduction = -grad.dot(&sk) - 0.5 * sk.dot(&hessian.dot(&sk));

                let rho_quality = if predicted_reduction.abs() < 1e-14 {
                    if actual_reduction > 0.0 { 1.0 } else { 0.0 }
                } else {
                    actual_reduction / predicted_reduction
                };

                // 6. Accept or reject step
                let de: f64;
                let mut used_gdiis = false;

                if actual_reduction > 0.0 {
                    // Accept: energy decreased
                    let yk = &grad_new - &grad;
                    hessian = damped_bfgs_update(hessian.view(), sk.view(), yk.view());

                    // Add to GDIIS history
                    gdiis.add(&x_new, &grad_new);
                    gdiis_steps_since_reset += 1;

                    // Try GDIIS extrapolation
                    let try_gdiis = gdiis.can_extrapolate()
                        && gdiis_failures < 5
                        && (k % 3 == 0);

                    if try_gdiis {
                        if let Some(x_gdiis) = gdiis.extrapolate() {
                            self.update_xyz(x_gdiis.view());
                            let (f_gdiis, grad_gdiis) = self.opt_energy_and_gradient(state);

                            let gdiis_grad_norm = grad_gdiis.dot(&grad_gdiis);
                            let new_grad_norm = grad_new.dot(&grad_new);

                            if f_gdiis < f_new && gdiis_grad_norm < new_grad_norm {
                                de = f_gdiis - fk;
                                x_current = x_gdiis;
                                fk = f_gdiis;
                                grad = grad_gdiis;
                                used_gdiis = true;
                                gdiis_failures = 0;
                                gdiis.add(&x_current, &grad);
                            } else {
                                gdiis_failures += 1;
                                de = f_new - fk;
                                x_current = x_new.to_owned();
                                fk = f_new;
                                grad = grad_new;
                                self.update_xyz(x_current.view());
                            }
                        } else {
                            de = f_new - fk;
                            x_current = x_new.to_owned();
                            fk = f_new;
                            grad = grad_new;
                        }
                    } else {
                        de = f_new - fk;
                        x_current = x_new.to_owned();
                        fk = f_new;
                        grad = grad_new;
                    }
                    consecutive_rejections = 0;
                } else {
                    // Reject: energy increased
                    de = 0.0;
                    self.update_xyz(x_current.view());
                    gdiis.clear();
                    consecutive_rejections += 1;

                    if consecutive_rejections >= 3 {
                        let (fk_fresh, grad_fresh) = self.opt_energy_and_gradient(state);
                        fk = fk_fresh;
                        grad = grad_fresh;
                    }

                    if consecutive_rejections >= 5 {
                        warn!(
                            "  Stagnation detected ({} rejections), resetting Hessian and trust region",
                            consecutive_rejections
                        );
                        hessian = build_model_hessian_diagonal(&atomic_numbers);
                        trust.reset();
                        consecutive_rejections = 0;
                    }
                }

                // Periodic GDIIS reset
                if gdiis_steps_since_reset > GDIIS_RESET_INTERVAL {
                    gdiis.clear();
                    gdiis_steps_since_reset = 0;
                    gdiis_failures = 0;
                }

                // 7. Update trust radius
                trust.update(rho_quality, step_norm);

                // 8. Check convergence
                let (converged, status) = criteria.check(&grad, &sk, de);

                // Print convergence status
                warn!(
                    "Optimization Cycle: {:5>}{}",
                    k + 1,
                    if used_gdiis { " [GDIIS]" } else { "" }
                );
                warn!(" ");
                warn!(
                    "{:>37}     {:>12}     {:>12}     {}",
                    "Value", "Maximum", "RMS", "Cnvgd?"
                );
                warn!(
                    "          {:<19} {:>12.8}     {:>12.8}     {}",
                    "Gradient (max/rms)",
                    status.grad_max,
                    status.grad_rms,
                    cnvg(status.grad_max_converged && status.grad_rms_converged),
                );
                warn!(
                    "          {:<19} {:>12.8}     {:>12.8}     {}",
                    "Displacement (max/rms)",
                    status.disp_max,
                    status.disp_rms,
                    cnvg(status.disp_max_converged && status.disp_rms_converged),
                );
                warn!(
                    "          {:<19} {:>12.8}     {:>12}     {}",
                    "Energy change",
                    status.energy_change,
                    "-",
                    cnvg(status.energy_converged),
                );
                warn!(
                    "          Trust radius: {:.6} Bohr, GDIIS vectors: {}",
                    trust.radius,
                    gdiis.len()
                );
                warn!(" ");

                if converged {
                    iterations = k + 1;
                    warn!(
                        "*** Optimization converged in {} iterations ***",
                        iterations
                    );
                    break 'optimization_loop;
                }

                // Write trajectory
                let new_coords: Array2<f64> =
                    constants::BOHR_TO_ANGS * &x_current.view().into_shape((n_atoms, 3)).unwrap();
                let xyz_out: XYZOutput = XYZOutput::new(
                    atom_names.clone(),
                    new_coords.clone().into_shape([n_atoms, 3]).unwrap(),
                );
                write_xyz_custom(&xyz_out, false);

                // Write energy
                let opt_energy: OptEnergyOutput = OptEnergyOutput::new(k, fk);
                write_opt_energy(&opt_energy, k == 0);

                iterations = k + 1;
            }

            // Write final geometry
            let final_coords: Array2<f64> =
                constants::BOHR_TO_ANGS * &x_current.view().into_shape((n_atoms, 3)).unwrap();
            let xyz_out: XYZOutput = XYZOutput::new(
                atom_names.clone(),
                final_coords.clone().into_shape([n_atoms, 3]).unwrap(),
            );
            write_last_geom(&xyz_out);

            let opt_energy: OptEnergyOutput = OptEnergyOutput::new(iterations, fk);
            write_opt_energy(&opt_energy, false);

            (x_current, grad)
        }
    };
}

// =============================================================================
// End of Advanced Cartesian Optimization (v3)
// =============================================================================

