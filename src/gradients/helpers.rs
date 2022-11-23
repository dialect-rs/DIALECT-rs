use crate::initialization::parameters::{RepulsivePotential, RepulsivePotentialTable};
use crate::initialization::Atom;
use nalgebra::{Point3, Vector3};
use ndarray::{
    s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, Ix2, Ix3, Ix4,
};
use ndarray_einsum_beta::tensordot;
use ndarray_linalg::krylov::qr;
use rayon::iter::*;
use std::collections::HashMap;

pub fn get_outer_product(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> (Array2<f64>) {
    let mut matrix: Array2<f64> = Array::zeros((v1.len(), v2.len()));
    for (i, i_value) in v1.outer_iter().enumerate() {
        for (j, j_value) in v2.outer_iter().enumerate() {
            matrix[[i, j]] = (&i_value * &j_value).into_scalar();
        }
    }
    return matrix;
}

pub fn f_lr(
    v: ArrayView2<f64>,
    s: ArrayView2<f64>,
    grad_s: ArrayView3<f64>,
    g0_lr_a0: ArrayView2<f64>,
    g1_lr_ao: ArrayView3<f64>,
    n_atoms: usize,
    n_orb: usize,
) -> Array3<f64> {
    let sv: Array2<f64> = s.dot(&v);
    let v_t: ArrayView2<f64> = v.t();
    let sv_t: Array2<f64> = s.dot(&v_t);
    let gv: Array2<f64> = &g0_lr_a0 * &v;

    let t_sv: ArrayView2<f64> = sv.t();
    let svg_t: Array2<f64> = (&sv * &g0_lr_a0).reversed_axes();
    let sgv_t: Array2<f64> = s.dot(&gv).reversed_axes();

    let mut f_return: Array3<f64> = Array3::zeros((3 * n_atoms, n_orb, n_orb));

    for nc in 0..3 * n_atoms {
        let d_s: ArrayView2<f64> = grad_s.slice(s![nc, .., ..]);
        let d_g: ArrayView2<f64> = g1_lr_ao.slice(s![nc, .., ..]);

        let d_sv_t: Array2<f64> = d_s.dot(&v_t);
        let d_sv: Array2<f64> = d_s.dot(&v);
        let d_gv: Array2<f64> = &d_g * &v;

        let mut d_f: Array2<f64> = Array2::zeros((n_orb, n_orb));
        // 1st term
        d_f = d_f + &g0_lr_a0 * &(d_s.dot(&t_sv));
        // 2nd term
        d_f = d_f + (&d_sv_t * &g0_lr_a0).dot(&s);
        // 3rd term
        d_f = d_f + d_s.dot(&svg_t);
        // 4th term
        d_f = d_f + d_s.dot(&sgv_t);
        // 5th term
        d_f = d_f + &g0_lr_a0 * &(s.dot(&d_sv.t()));
        // 6th term
        d_f = d_f + (&sv_t * &g0_lr_a0).dot(&d_s.t());
        // 7th term
        d_f = d_f + s.dot(&(&d_sv * &g0_lr_a0).t());
        // 8th term
        d_f = d_f + s.dot(&(d_s.dot(&gv)).t());
        // 9th term
        d_f = d_f + &d_g * &(s.dot(&t_sv));
        // 10th term
        d_f = d_f + (&sv_t * &d_g).dot(&s);
        // 11th term
        d_f = d_f + s.dot(&(&sv * &d_g).t());
        // 12th term
        d_f = d_f + s.dot(&(s.dot(&d_gv)).t());
        d_f = d_f * 0.25;

        f_return.slice_mut(s![nc, .., ..]).assign(&d_f);
    }
    return f_return;
}

//  Compute the gradient of the repulsive potential
//  Parameters:
//  ===========
//  atomlist: list of tuples (Zi, [xi,yi,zi]) for each atom
//  distances: matrix with distances between atoms, distance[i,j]
//    is the distance between atoms i and j
//  directions: directions[i,j,:] is the unit vector pointing from
//    atom j to atom i
//  VREP: dictionary, VREP[(Zi,Zj)] has to be an instance of RepulsivePotential
//    for the atom pair Zi-Zj
pub fn gradient_v_rep(atoms: &[Atom], v_rep: &RepulsivePotential) -> Array1<f64> {
    let n_atoms: usize = atoms.len();
    let mut grad: Array1<f64> = Array1::zeros([3 * n_atoms]);
    for (i, atomi) in atoms.iter().enumerate() {
        let mut grad_i: Array1<f64> = Array::zeros([3]);
        for (j, atomj) in atoms.iter().enumerate() {
            if i != j {
                let mut r: Vector3<f64> = atomi - atomj;
                let r_ij: f64 = r.norm();
                r /= r_ij;
                let v_ij_deriv: f64 = v_rep.get(atomi.kind, atomj.kind).spline_deriv(r_ij);
                r *= v_ij_deriv;
                // let v: ArrayView1<f64> = unsafe {
                //     ArrayView1::from_shape_ptr(
                //         (r.shape().0, ).strides((r.strides().0, )),
                //         r.as_ptr(),
                //     )
                // };
                let v = Array1::from_iter(r.iter());
                grad_i = &grad_i + &v;
            }
        }
        grad.slice_mut(s![i * 3..i * 3 + 3]).assign(&grad_i);
    }
    return grad;
}
