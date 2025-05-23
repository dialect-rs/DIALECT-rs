use crate::initialization::{Atom, System};
use crate::param::slako_transformations::{directional_cosines, slako_transformation};
use ndarray::prelude::*;
use ndarray_linalg::Determinant;
use ndarray_stats::QuantileExt;

impl System {
    pub fn overlap_between_timesteps(&self, old_atoms: &[Atom]) -> Array2<f64> {
        // compute overlap matrix elements between two sets of atoms using
        // Slater-Koster rules
        let mut s: Array2<f64> = Array2::zeros([self.n_orbs, self.n_orbs]);

        let mut mu: usize = 0;
        // iterate over the atoms of the system
        for atom_i in old_atoms.iter() {
            // iterate over the orbitals on atom I
            for orbi in atom_i.valorbs.iter() {
                // iterate over the atoms of the old geometry
                let mut nu: usize = 0;
                for atom_j in self.atoms.iter() {
                    // iterate over the orbitals on atom J
                    for orbj in atom_j.valorbs.iter() {
                        if atom_i <= atom_j {
                            let (r, x, y, z): (f64, f64, f64, f64) =
                                directional_cosines(&atom_i.xyz, &atom_j.xyz);
                            s[[mu, nu]] = slako_transformation(
                                r,
                                x,
                                y,
                                z,
                                &self.slako.get(atom_i.kind, atom_j.kind).s_spline,
                                orbi.l,
                                orbi.m,
                                orbj.l,
                                orbj.m,
                            );
                        } else {
                            let (r, x, y, z): (f64, f64, f64, f64) =
                                directional_cosines(&atom_j.xyz, &atom_i.xyz);
                            s[[mu, nu]] = slako_transformation(
                                r,
                                x,
                                y,
                                z,
                                &self.slako.get(atom_j.kind, atom_i.kind).s_spline,
                                orbj.l,
                                orbj.m,
                                orbi.l,
                                orbi.m,
                            );
                        }
                        nu += 1;
                    }
                }
                mu += 1;
            }
        }
        s
    }

    pub fn overlap_between_timesteps_ba(&self, old_atoms: &[Atom]) -> Array2<f64> {
        // compute overlap matrix elements between two sets of atoms using
        // Slater-Koster rules
        let mut s: Array2<f64> = Array2::zeros([self.n_orbs, self.n_orbs]);

        let mut mu: usize = 0;
        // iterate over the atoms of the system
        for atom_i in self.atoms.iter() {
            // iterate over the orbitals on atom I
            for orbi in atom_i.valorbs.iter() {
                // iterate over the atoms of the old geometry
                let mut nu: usize = 0;
                for atom_j in old_atoms.iter() {
                    // iterate over the orbitals on atom J
                    for orbj in atom_j.valorbs.iter() {
                        if atom_i <= atom_j {
                            let (r, x, y, z): (f64, f64, f64, f64) =
                                directional_cosines(&atom_i.xyz, &atom_j.xyz);
                            s[[mu, nu]] = slako_transformation(
                                r,
                                x,
                                y,
                                z,
                                &self.slako.get(atom_i.kind, atom_j.kind).s_spline,
                                orbi.l,
                                orbi.m,
                                orbj.l,
                                orbj.m,
                            );
                        } else {
                            let (r, x, y, z): (f64, f64, f64, f64) =
                                directional_cosines(&atom_j.xyz, &atom_i.xyz);
                            s[[mu, nu]] = slako_transformation(
                                r,
                                x,
                                y,
                                z,
                                &self.slako.get(atom_j.kind, atom_i.kind).s_spline,
                                orbj.l,
                                orbj.m,
                                orbi.l,
                                orbi.m,
                            );
                        }
                        nu += 1;
                    }
                }
                mu += 1;
            }
        }
        s
    }

    pub fn ci_overlap_system(
        &self,
        old_atoms: &[Atom],
        old_orbs: ArrayView2<f64>,
        old_ci_coeff: ArrayView2<f64>,
        n_states: usize,
        _step: usize,
    ) -> Array2<f64> {
        // Compute CI overlap between TD-DFT 'wavefunctions'
        // Excitations i->a with coefficients |C_ia| < threshold will be neglected
        // n_states: Includes the ground state
        let threshold: f64 = 0.001;

        // calculate the overlap between the new and old geometry
        let s_ao: Array2<f64> = self.overlap_between_timesteps(old_atoms);

        let orbs_i: ArrayView2<f64> = self.properties.orbs().unwrap();
        // calculate the overlap between the molecular orbitals
        let s_mo: Array2<f64> = old_orbs.t().dot(&s_ao.dot(&orbs_i));

        // // align the phase of the MO coefficients
        // let s_mo_diag:ArrayView1<f64> = s_mo.diag();
        // let sign:Array1<f64> = get_sign_of_array(s_mo_diag);
        // println!("signs: {:.3}",sign);
        // let mut aligned_orbs:Array2<f64> = Array2::zeros(orbs_i.raw_dim());
        // for (idx, sign) in sign.iter().enumerate(){
        //     aligned_orbs.slice_mut(s![..,idx]).assign(&(*sign * &orbs_i.slice(s![..,idx])));
        // }

        // // calculate the overlap between the molecular orbitals with the newly aligned current orbs
        // let s_mo: Array2<f64> = old_orbs.t().dot(&s_ao.dot(&aligned_orbs));
        // let s_mo_diag:ArrayView1<f64> = s_mo.diag();
        // let sign:Array1<f64> = get_sign_of_array(s_mo_diag);
        // println!("signs: {:.3}",sign);
        // let filename: String = format!("s_dt_{}.npy", step);
        // write_npy(filename, &s_mo).unwrap();

        // get occupied and virtual orbitals
        let occ_indices = self.properties.occ_indices().unwrap();
        let virt_indices = self.properties.virt_indices().unwrap();
        let n_occ: usize = occ_indices.len();
        let n_virt: usize = virt_indices.len();

        // slice s_mo to get the occupied part and calculate the determinant
        let s_ij: ArrayView2<f64> = s_mo.slice(s![..n_occ, ..n_occ]);
        let det_ij = s_ij.det().unwrap();

        // scalar coupling array
        let mut s_ci: Array2<f64> = Array2::zeros((n_states, n_states));
        // get ci coefficients from properties
        let n_roots: usize = self.config.excited.nstates;
        let ci_coeff: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let ci_coeff: ArrayView3<f64> = ci_coeff.into_shape([n_occ, n_virt, n_roots]).unwrap();
        let old_ci_coeff: ArrayView3<f64> =
            old_ci_coeff.into_shape([n_occ, n_virt, n_roots]).unwrap();

        // let filename: String = format!("ci_coeff_{}.npy", step);
        // write_npy(filename, &ci_coeff).unwrap();
        // let filename: String = format!("old_ci_coeff_{}.npy", step);
        // write_npy(filename, &old_ci_coeff).unwrap();

        // overlap between ground states <Psi0|Psi0'>
        s_ci[[0, 0]] = det_ij;

        // calculate the overlap between the excited states
        // iterate over the old CI coefficients
        for i in occ_indices.iter() {
            for (a_idx, a) in virt_indices.iter().enumerate() {
                // slice old CI coefficients at the indicies i and a
                let coeffs_i = old_ci_coeff.slice(s![*i, a_idx, ..]);
                let max_coeff_i = coeffs_i.map(|val| val.abs()).max().unwrap().to_owned();

                // slice new CI coefficients at the indicies i and a
                let coeffs_new = ci_coeff.slice(s![*i, a_idx, ..]);
                let max_coeff_new = coeffs_new.map(|val| val.abs()).max().unwrap().to_owned();

                // if the value of the coefficient is smaller than the threshold,
                // exclude the excited state
                if max_coeff_new > threshold {
                    let mut s_ia: Array2<f64> = s_ij.to_owned();
                    // overlap <Psi0|PsiJ'>
                    s_ia.slice_mut(s![.., *i])
                        .assign(&s_mo.slice(s![..n_occ, *a]));
                    let det_ia: f64 = s_ia.det().unwrap();

                    // overlaps between ground state <Psi0|PsiJ'> and excited states
                    for state_j in 1..n_states {
                        let c0: f64 = coeffs_new[state_j - 1];
                        s_ci[[0, state_j]] += c0 * 2.0_f64.sqrt() * (det_ia * det_ij);
                    }
                }
                // if the value of the coefficient is smaller than the threshold,
                // exclude the excited state
                if max_coeff_i > threshold {
                    let mut s_aj: Array2<f64> = s_ij.to_owned();
                    // occupied orbitals in the configuration state function |Psi_ia>
                    // oveerlap <1,...,a,...|1,...,j,...>
                    s_aj.slice_mut(s![*i, ..])
                        .assign(&s_mo.slice(s![*a, ..n_occ]));
                    let det_aj: f64 = s_aj.det().unwrap();

                    // overlaps between ground state <PsiI|Psi0'> and excited states
                    for state_i in 1..n_states {
                        let c0: f64 = coeffs_i[state_i - 1];
                        s_ci[[state_i, 0]] += c0 * 2.0_f64.sqrt() * (det_aj * det_ij);
                    }

                    // iterate over the new CI coefficients
                    for j in occ_indices.iter() {
                        for (b_idx, b) in virt_indices.iter().enumerate() {
                            // slice the new CI coefficients at the indicies j and b
                            let coeffs_j = ci_coeff.slice(s![*j, b_idx, ..]);
                            let max_coeff_j =
                                coeffs_j.map(|val| val.abs()).max().unwrap().to_owned();
                            // if the value of the coefficient is smaller than the threshold,
                            // exclude the excited state
                            if max_coeff_j > threshold {
                                let mut s_ab: Array2<f64> = s_ij.to_owned();
                                // select part of overlap matrix for orbitals
                                // in |Psi_ia> and |Psi_jb>
                                // <1,...,a,...|1,...,b,...>
                                s_ab.slice_mut(s![*i, ..])
                                    .assign(&s_mo.slice(s![*a, ..n_occ]));
                                s_ab.slice_mut(s![.., *j])
                                    .assign(&s_mo.slice(s![..n_occ, *b]));
                                s_ab[[*i, *j]] = s_mo[[*a, *b]];
                                let det_ab: f64 = s_ab.det().unwrap();

                                let mut s_ib: Array2<f64> = s_ij.to_owned();
                                // <1,...,i,...|1,...,b,...>
                                s_ib.slice_mut(s![.., *j])
                                    .assign(&s_mo.slice(s![..n_occ, *b]));
                                let det_ib: f64 = s_ib.det().unwrap();

                                // loop over excited states
                                for state_i in 1..n_states {
                                    for state_j in 1..n_states {
                                        let cc: f64 = coeffs_i[state_i - 1] * coeffs_j[state_j - 1];
                                        // see eqn. (9.39) in A. Humeniuk, PhD thesis (2018)
                                        s_ci[[state_i, state_j]] +=
                                            cc * (det_ab * det_ij + det_aj * det_ib);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        s_ci
    }

    pub fn check_sign_ci_overlap_same_state(&self, state: usize, n_roots: usize) -> f64 {
        // check the signs of the states
        let sci_1: f64 = self.ci_overlap_system_states_2(state, state, n_roots);
        let prefac_i: f64 = if sci_1 > 0.0 { 1.0 } else { -1.0 };

        prefac_i
    }

    pub fn ci_overlap_system_states(&self, state_i: usize, state_j: usize, n_roots: usize) -> f64 {
        // Compute CI overlap between TD-DFT 'wavefunctions'
        // Excitations i->a with coefficients |C_ia| < threshold will be neglected
        // n_states: Includes the ground state
        let threshold: f64 = 0.00001;

        // get the old system
        let old_system = self.properties.old_system().unwrap();
        let old_atoms = &old_system.atoms;
        let old_orbs = old_system.orbs.view();
        let old_ci_coeff = old_system.ci_coefficients.view();

        // calculate the overlap between the new and old geometry
        let s_ao: Array2<f64> = self.overlap_between_timesteps(old_atoms);

        let orbs_i: ArrayView2<f64> = self.properties.orbs().unwrap();
        // calculate the overlap between the molecular orbitals
        let s_mo: Array2<f64> = old_orbs.t().dot(&s_ao.dot(&orbs_i));
        // println!("S in MO basis: \n {:.4}",s_mo);

        // get occupied and virtual orbitals
        let occ_indices = self.properties.occ_indices().unwrap();
        let virt_indices = self.properties.virt_indices().unwrap();
        let n_occ: usize = occ_indices.len();
        let n_virt: usize = virt_indices.len();

        // slice s_mo to get the occupied part and calculate the determinant
        let s_ij: ArrayView2<f64> = s_mo.slice(s![..n_occ, ..n_occ]);
        let det_ij = s_ij.det().unwrap();

        // check the signs of the states
        let sci_1: f64 = self.ci_overlap_system_states_2(state_i, state_i, n_roots);
        // let sci_2: f64 = self.ci_overlap_system_states_2(state_j, state_j, n_roots);
        let prefac_i: f64 = if sci_1 > 0.0 { 1.0 } else { -1.0 };
        // let prefac_j: f64 = if sci_2 > 0.0 { 1.0 } else { -1.0 };

        // scalar coupling value
        let mut s_ci: f64 = 0.0;
        // get ci coefficients from properties
        let ci_coeff: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let ci_coeff: ArrayView3<f64> = ci_coeff.into_shape([n_occ, n_virt, n_roots]).unwrap();
        let ci_coeff_slice: ArrayView2<f64> = ci_coeff.slice(s![.., .., state_i]);
        let old_ci_coeff: ArrayView3<f64> =
            old_ci_coeff.into_shape([n_occ, n_virt, n_roots]).unwrap();
        let old_ci_coeff: ArrayView2<f64> = old_ci_coeff.slice(s![.., .., state_j]);

        // calculate the overlap between the excited states
        // iterate over the old CI coefficients
        for i in occ_indices.iter() {
            for (a_idx, a) in virt_indices.iter().enumerate() {
                // slice old CI coefficients at the indicies i and a
                let coeffs_i = old_ci_coeff[[*i, a_idx]];

                // if the value of the coefficient is smaller than the threshold,
                // exclude the excited state
                if coeffs_i.abs() > threshold {
                    let mut s_aj: Array2<f64> = s_ij.to_owned();
                    // occupied orbitals in the configuration state function |Psi_ia>
                    // oveerlap <1,...,a,...|1,...,j,...>
                    s_aj.slice_mut(s![*i, ..])
                        .assign(&s_mo.slice(s![*a, ..n_occ]));
                    let det_aj: f64 = s_aj.det().unwrap();

                    // iterate over the new CI coefficients
                    for j in occ_indices.iter() {
                        for (b_idx, b) in virt_indices.iter().enumerate() {
                            // slice the new CI coefficients at the indicies j and b
                            let coeffs_j = ci_coeff_slice[[*j, b_idx]];

                            // if the value of the coefficient is smaller than the threshold,
                            // exclude the excited state
                            if coeffs_j.abs() > threshold {
                                let mut s_ab: Array2<f64> = s_ij.to_owned();
                                // select part of overlap matrix for orbitals
                                // in |Psi_ia> and |Psi_jb>
                                // <1,...,a,...|1,...,b,...>
                                s_ab.slice_mut(s![*i, ..])
                                    .assign(&s_mo.slice(s![*a, ..n_occ]));
                                s_ab.slice_mut(s![.., *j])
                                    .assign(&s_mo.slice(s![..n_occ, *b]));
                                s_ab[[*i, *j]] = s_mo[[*a, *b]];
                                let det_ab: f64 = s_ab.det().unwrap();

                                let mut s_ib: Array2<f64> = s_ij.to_owned();
                                // <1,...,i,...|1,...,b,...>
                                s_ib.slice_mut(s![.., *j])
                                    .assign(&s_mo.slice(s![..n_occ, *b]));
                                let det_ib: f64 = s_ib.det().unwrap();

                                let cc: f64 = coeffs_i * coeffs_j;
                                // see eqn. (9.39) in A. Humeniuk, PhD thesis (2018)
                                s_ci += cc * (det_ab * det_ij + det_aj * det_ib);
                            }
                        }
                    }
                }
            }
        }

        s_ci * prefac_i
    }

    pub fn ci_overlap_system_states_2(
        &self,
        state_i: usize,
        state_j: usize,
        n_roots: usize,
    ) -> f64 {
        // Compute CI overlap between TD-DFT 'wavefunctions'
        // Excitations i->a with coefficients |C_ia| < threshold will be neglected
        // n_states: Includes the ground state
        let threshold: f64 = 0.00001;

        // get the old system
        let old_system = self.properties.old_system().unwrap();
        let old_atoms = &old_system.atoms;
        let old_orbs = old_system.orbs.view();
        let old_ci_coeff = old_system.ci_coefficients.view();

        // calculate the overlap between the new and old geometry
        let s_ao: Array2<f64> = self.overlap_between_timesteps(old_atoms);

        let orbs_i: ArrayView2<f64> = self.properties.orbs().unwrap();
        // calculate the overlap between the molecular orbitals
        let s_mo: Array2<f64> = old_orbs.t().dot(&s_ao.dot(&orbs_i));
        // println!("S in MO basis: \n {:.4}",s_mo);

        // get occupied and virtual orbitals
        let occ_indices = self.properties.occ_indices().unwrap();
        let virt_indices = self.properties.virt_indices().unwrap();
        let n_occ: usize = occ_indices.len();
        let n_virt: usize = virt_indices.len();

        // slice s_mo to get the occupied part and calculate the determinant
        let s_ij: ArrayView2<f64> = s_mo.slice(s![..n_occ, ..n_occ]);
        let det_ij = s_ij.det().unwrap();

        // check the signs of the states

        // scalar coupling value
        let mut s_ci: f64 = 0.0;
        // get ci coefficients from properties
        let ci_coeff: ArrayView2<f64> = self.properties.ci_coefficients().unwrap();
        let ci_coeff: ArrayView3<f64> = ci_coeff.into_shape([n_occ, n_virt, n_roots]).unwrap();
        let ci_coeff_slice: ArrayView2<f64> = ci_coeff.slice(s![.., .., state_i]);
        let old_ci_coeff: ArrayView3<f64> =
            old_ci_coeff.into_shape([n_occ, n_virt, n_roots]).unwrap();
        let old_ci_coeff: ArrayView2<f64> = old_ci_coeff.slice(s![.., .., state_j]);

        // calculate the overlap between the excited states
        // iterate over the old CI coefficients
        for i in occ_indices.iter() {
            for (a_idx, a) in virt_indices.iter().enumerate() {
                // slice old CI coefficients at the indicies i and a
                let coeffs_i = old_ci_coeff[[*i, a_idx]];

                // if the value of the coefficient is smaller than the threshold,
                // exclude the excited state
                if coeffs_i.abs() > threshold {
                    let mut s_aj: Array2<f64> = s_ij.to_owned();
                    // occupied orbitals in the configuration state function |Psi_ia>
                    // oveerlap <1,...,a,...|1,...,j,...>
                    s_aj.slice_mut(s![*i, ..])
                        .assign(&s_mo.slice(s![*a, ..n_occ]));
                    let det_aj: f64 = s_aj.det().unwrap();

                    // iterate over the new CI coefficients
                    for j in occ_indices.iter() {
                        for (b_idx, b) in virt_indices.iter().enumerate() {
                            // slice the new CI coefficients at the indicies j and b
                            let coeffs_j = ci_coeff_slice[[*j, b_idx]];

                            // if the value of the coefficient is smaller than the threshold,
                            // exclude the excited state
                            if coeffs_j.abs() > threshold {
                                let mut s_ab: Array2<f64> = s_ij.to_owned();
                                // select part of overlap matrix for orbitals
                                // in |Psi_ia> and |Psi_jb>
                                // <1,...,a,...|1,...,b,...>
                                s_ab.slice_mut(s![*i, ..])
                                    .assign(&s_mo.slice(s![*a, ..n_occ]));
                                s_ab.slice_mut(s![.., *j])
                                    .assign(&s_mo.slice(s![..n_occ, *b]));
                                s_ab[[*i, *j]] = s_mo[[*a, *b]];
                                let det_ab: f64 = s_ab.det().unwrap();

                                let mut s_ib: Array2<f64> = s_ij.to_owned();
                                // <1,...,i,...|1,...,b,...>
                                s_ib.slice_mut(s![.., *j])
                                    .assign(&s_mo.slice(s![..n_occ, *b]));
                                let det_ib: f64 = s_ib.det().unwrap();

                                let cc: f64 = coeffs_i * coeffs_j;
                                // see eqn. (9.39) in A. Humeniuk, PhD thesis (2018)
                                s_ci += cc * (det_ab * det_ij + det_aj * det_ib);
                            }
                        }
                    }
                }
            }
        }

        s_ci
    }
}

pub fn get_sign_of_array(arr: ArrayView1<f64>) -> Array1<f64> {
    let mut sign: Array1<f64> = Array1::zeros(arr.len());
    arr.iter().enumerate().for_each(|(idx, val)| {
        if val.is_sign_positive() {
            sign[idx] = 1.0;
        } else {
            sign[idx] = -1.0;
        }
    });
    sign
}
