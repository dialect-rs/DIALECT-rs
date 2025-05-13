use crate::initialization::System;
use log::info;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_linalg::SVD;
use std::cmp::Ordering;

impl System {
    pub fn get_ntos_for_state(&self, state: usize) {
        // get the tdm, the MO coefficients and the lumo index
        let tdm: Array2<f64> = if self.config.tddftb.restrict_active_orbitals {
            let dim_o: usize = (self.occ_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            let dim_v: usize = (self.virt_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            self.properties
                .tdm_restricted(state, dim_o, dim_v)
                .unwrap()
                .to_owned()
        } else {
            self.properties.tdm(state).unwrap().to_owned()
        };

        let orbs: Array2<f64> = if self.config.tddftb.restrict_active_orbitals {
            let dim_o: usize = (self.occ_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            let dim_v: usize = (self.virt_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            let homo: usize = self.virt_indices[0] - 1;
            let lumo: usize = self.virt_indices[0];
            let orbs = self.properties.orbs().unwrap();
            orbs.slice(s![.., homo + 1 - dim_o..lumo + dim_v])
                .to_owned()
        } else {
            self.properties.orbs().unwrap().to_owned()
        };
        // calculate the ntos
        let (lambda, ntos): (Array1<f64>, Array2<f64>) =
            natural_transition_orbitals_for_cubes(tdm.view(), orbs.view());
        // get nocc
        let (nocc, nvirt): (usize, usize) = if self.config.tddftb.restrict_active_orbitals {
            let dim_o: usize = (self.occ_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            let dim_v: usize = (self.virt_indices.len() as f64
                * self.config.tddftb.active_orbital_threshold)
                as usize;
            (dim_o, dim_v)
        } else {
            (self.occ_indices.len(), self.virt_indices.len())
        };
        let honto_idx: usize = if nocc > nvirt {
            nocc - (nocc - nvirt) - 1
        } else {
            nocc - 1
        };

        // logging
        info!("{:^80}", "");
        info!(
            "{:^22} {}{} ",
            "",
            "Natural Transition Orbitals for S",
            state + 1
        );
        info!("{:-^80}", "");

        // check lambdas
        let mut index_vec: Vec<usize> = Vec::new();
        for (idx, lambda_val) in lambda.iter().enumerate() {
            if *lambda_val > 0.05 {
                index_vec.push(idx);
                if idx < honto_idx + 1 {
                    let diff_from_honto: usize = honto_idx - idx;
                    info!(
                        "Transition: {} --> {}, Amplitude: {:.4}",
                        idx + 1,
                        (honto_idx + 1 + diff_from_honto) + 1,
                        lambda_val
                    );
                }
            }
        }

        // array for orbitals
        let mut orbital_array: Array2<f64> = Array2::zeros([self.n_orbs, index_vec.len()]);

        // create cube files for the highest contributions above 10 percent
        for (idx, idx_val) in index_vec.iter().enumerate() {
            let lambda_val: f64 = lambda[*idx_val];
            if lambda_val > 0.1 {
                // create the cube files
                // self.cube_from_orbital(nto, idx, state);
                orbital_array
                    .slice_mut(s![.., idx])
                    .assign(&ntos.slice(s![.., *idx_val]));
            }
        }
        let index_vec: Vec<usize> = index_vec.iter().map(|idx| *idx + 1).collect();
        self.cube_from_orbital_arr(orbital_array.view(), &index_vec, state + 1, "nto");
    }
}

/// Compute the MO coefficients of natural transition orbitals.
///
/// If the excitation amplitudes are broadly distributed without a dominant configuration, then
/// the interpretation of the excited state can be very difficult. Additionally, when these
/// transitions are computed in the framework of TDDFT/(TDA) the physical meaning of the Kohn-
/// Sham orbitals is limited. Though, a simple interpretation where the particles and the holes of
/// an excitation are located, is important. Therefore we perform a singular value decomposition
/// on the reduced transition density matrix
/// $$  T = U \Sigma V^T, $$
/// where $\bm{\Sigma}$ is the singulary matrix,
/// $$ \Sigma_{ij} = \delta_{ij} \sqrt{\lambda_i}, $$
/// with the corresponding singular values $\lambda_i$. We can then use the U matrix to
/// construct the hole natural transition orbitals (NTOs),
/// $$ \psi_i^o = \sum_{j=1}^{N_o} U_{ji} \phi_{j}, $$
/// and the particle NTOs are defined by the $\vb{V}$ matrix
/// $$ \psi_i^v = \sum_{j=1}^{N_v} V_{ji} \phi_{j}. $$
/// The occupied $\phi_i$ and virtual $\phi_p$ orbitals are transformed by this procedure into new
/// sets of orbitals $\psi_i$ and $\psi_p$. This orbital transformation is equal to solving the
/// eigenvalue equations UTT^TU^-1 and VT^TTV^-1
/// However, it should be noted that by obtaining the eigenvectors U, V in this latter
/// way these vectors are independent from each other and do not have a unique phase relation.
/// Therefore we have used the singular value decomposition for the computation of the eigenvectors.
/// The eigenvalues $\lambda_i$ are ordered by decreasing magnitude and in total sum up to 1.
/// However in the case of TDDFT this sum will not be identically to 1 due to the presence
/// of the de-excitation operators.
///
/// References
/// ----------
/// [1] A. T. Amos and G. G. Hall, Proc. R. Soc. London, Ser. A 263, 483 (1961). <br>
/// [2] A. V. Luzanov, A. A. Sukhorukov, and V. E. Umanskii, Theor. Exp. Chem. 10, 354 (1976).<br>
/// [3] A. V. Luzanov and V. F. Pedash, Theor. Exp. Chem. 15, 338 (1979). <br>
/// [4] M. Head-Gordon, A. M. Grana, D. Maurice, and C. A. White,  J. Phys. Chem. 99, 14261 (1995). <br>
pub fn natural_transition_orbitals(
    tdm: ArrayView2<f64>,
    orbs: ArrayView2<f64>,
    lumo: usize,
) -> (Array1<f64>, Array2<f64>) {
    // Number of occupied and virtual orbitals.
    let (n_occ, n_virt): (usize, usize) = tdm.dim();

    // The number of occupied/virtual orbitals is given by the smaller of the two, due to the SVD
    let crop: CropCoeffs = CropCoeffs::new(n_occ, n_virt);

    // Singular value decomposition of the reduced transition density matrix in MO basis.
    let (u, sigma, vt): (Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>) =
        tdm.svd(true, true).unwrap();

    // Invert the NTO axis (columns) of U, since the singular values are ordered in decreasing order.
    let mut u: Array2<f64> = u.unwrap();
    u.invert_axis(Axis(1));

    // Singular values.
    let lambda: Array1<f64> = &sigma * &sigma;
    let mut lambda_rev: Array1<f64> = lambda.clone();
    lambda_rev.invert_axis(Axis(0));
    let lambda: Array1<f64> = concatenate![Axis(0), lambda_rev, lambda];

    // Array for the MO coefficients of the natural transition orbitals.
    let mut ntos: Array2<f64> = Array2::zeros(orbs.raw_dim());

    // Occupied orbitals are transformed.
    ntos.slice_mut(s![.., 0..n_occ])
        .assign(&(orbs.slice(s![.., ..lumo]).dot(&u)));

    // Virtual orbitals are transformed.
    ntos.slice_mut(s![.., n_occ..])
        .assign(&(orbs.slice(s![.., lumo..]).dot(&vt.unwrap().t())));

    // The NTO coefficients are cropped so that the length of the singular values corresponds to
    // the NTOs.
    ntos = match crop {
        CropCoeffs::None => ntos,
        CropCoeffs::Occ(n) => ntos.slice_move(s![.., n..]),
        CropCoeffs::Virt(n) => ntos.slice_move(s![.., ..n]),
    };

    let lambda = lambda
        .slice(s![sigma.len() - 20..sigma.len() + 20])
        .to_owned();
    let ntos = ntos
        .slice(s![.., sigma.len() - 20..sigma.len() + 20])
        .to_owned();

    // (singular values, coefficients of NTOs)
    (lambda, ntos)
}

pub fn natural_transition_orbitals_for_cubes(
    tdm: ArrayView2<f64>,
    orbs: ArrayView2<f64>,
) -> (Array1<f64>, Array2<f64>) {
    // Number of occupied and virtual orbitals.
    let (n_occ, n_virt): (usize, usize) = tdm.dim();

    // The number of occupied/virtual orbitals is given by the smaller of the two, due to the SVD
    let crop: CropCoeffs = CropCoeffs::new(n_occ, n_virt);

    // Singular value decomposition of the reduced transition density matrix in MO basis.
    let (u, sigma, vt): (Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>) =
        tdm.svd(true, true).unwrap();

    // Invert the NTO axis (columns) of U, since the singular values are ordered in decreasing order.
    let mut u: Array2<f64> = u.unwrap();
    u.invert_axis(Axis(1));

    // Singular values.
    let lambda: Array1<f64> = &sigma * &sigma;
    let mut lambda_rev: Array1<f64> = lambda.clone();
    lambda_rev.invert_axis(Axis(0));
    let lambda: Array1<f64> = concatenate![Axis(0), lambda_rev, lambda];

    // Array for the MO coefficients of the natural transition orbitals.
    let mut ntos: Array2<f64> = Array2::zeros(orbs.raw_dim());

    // Occupied orbitals are transformed.
    ntos.slice_mut(s![.., 0..n_occ])
        .assign(&(orbs.slice(s![.., ..n_occ]).dot(&u)));

    // Virtual orbitals are transformed.
    ntos.slice_mut(s![.., n_occ..])
        .assign(&(orbs.slice(s![.., n_occ..]).dot(&vt.unwrap().t())));

    // The NTO coefficients are cropped so that the length of the singular values corresponds to
    // the NTOs.
    ntos = match crop {
        CropCoeffs::None => ntos,
        CropCoeffs::Occ(n) => ntos.slice_move(s![.., n..]),
        CropCoeffs::Virt(n) => ntos.slice_move(s![.., ..n]),
    };

    // (singular values, coefficients of NTOs)
    (lambda, ntos)
}

/// The length of singular values from the SVD corresponds to the shorter axis of the input matrix.
/// In the case that there are more occupied orbitals than virtual ones, the MO coefficients of the
/// occupied orbitals need to be cropped. If there are less occupied orbitals then the highest
/// virtual orbitals need to be cropped. If there is an equal number of both, then nothing has to
/// be done. The enum represents the orbital space that needs to be cropped and the corresponding
/// start/end index.
enum CropCoeffs {
    Occ(usize),
    Virt(usize),
    None,
}

impl CropCoeffs {
    pub fn new(n_occ: usize, n_virt: usize) -> Self {
        match n_occ.cmp(&n_virt) {
            Ordering::Equal => Self::None,
            Ordering::Greater => Self::Occ(n_occ - n_virt),
            Ordering::Less => Self::Virt(2 * n_occ),
        }
        // if n_occ == n_virt {
        //     Self::None
        // } else if n_occ > n_virt {
        //     Self::Occ(n_occ - n_virt)
        // } else {
        //     Self::Virt(2 * n_occ)
        // }
    }
}

pub fn get_nto_singular_values(tdm: ArrayView2<f64>) -> Array1<f64> {
    // Singular value decomposition of the reduced transition density matrix in MO basis.
    let (_u, sigma, _vt): (Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>) =
        tdm.svd(false, false).unwrap();

    // Singular values.
    let lambda: Array1<f64> = &sigma * &sigma;

    lambda
}
