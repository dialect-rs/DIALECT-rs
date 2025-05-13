use crate::initialization::Atom;
use ndarray_linalg::{c64, Scalar};
use rusty_fitpack::splrep;
use sphrs::*;
use splines::{Interpolation, Key, Spline};
use std::cmp::Ordering;

pub fn spherical_harmonics_yreal(l: i8, m: i8, r: (f64, f64, f64)) -> f64 {
    let mut yreal: c64;

    match m.cmp(&0) {
        Ordering::Greater => {
            yreal = 1.0 / 2.0_f64.sqrt()
                * (evaluate_spherical_harmonics(l, m, r)
                    + (-1.0_f64).powi(m as i32) * evaluate_spherical_harmonics(l, -m, r));
        }
        Ordering::Equal => {
            yreal = evaluate_spherical_harmonics(l, m, r);
        }
        Ordering::Less => {
            yreal = -c64::complex(0.0, 1.0) / 2.0_f64.sqrt()
                * (evaluate_spherical_harmonics(l, -m, r)
                    - (-1.0_f64).powi(m as i32) * evaluate_spherical_harmonics(l, m, r));
        }
    }
    // if m > 0 {
    //     yreal = 1.0 / 2.0_f64.sqrt()
    //         * (evaluate_spherical_harmonics(l, m, r)
    //             + (-1.0_f64).powi(m as i32) * evaluate_spherical_harmonics(l, -m, r));
    // } else if m == 0 {
    //     yreal = evaluate_spherical_harmonics(l, m, r);
    // } else {
    //     yreal = -c64::complex(0.0, 1.0) / 2.0_f64.sqrt()
    //         * (evaluate_spherical_harmonics(l, -m, r)
    //             - (-1.0_f64).powi(m as i32) * evaluate_spherical_harmonics(l, m, r));
    // }
    yreal *= (-1.0_f64).powi(m as i32);
    yreal.re()
}

pub fn evaluate_spherical_harmonics(l: i8, m: i8, r: (f64, f64, f64)) -> c64 {
    assert!(l >= m.abs());
    let x = r.0;
    let y = r.1;
    let z = r.2;
    let coords: Coordinates<f64> = Coordinates::cartesian(x, y, z);
    let sh = ComplexSHType::Spherical;
    let degree = l as i64;
    let order = m as i64;

    sh.eval(degree, order, &coords)
}

pub fn spline_radial_wavefunction(
    r: Vec<f64>,
    r_vals: Vec<f64>,
) -> (Vec<f64>, Vec<f64>, usize, f64, f64) {
    let rmin: f64 = r[0];
    let rmax: f64 = r[r.len() - 1];
    let tmp = splrep(
        r, r_vals, None, None, None, None, None, None, None, None, None, None,
    );
    let tck: Vec<f64> = tmp.0;
    let c: Vec<f64> = tmp.1;
    let k: usize = tmp.2;

    (tck, c, k, rmin, rmax)
}

pub fn spline_radial_wavefunction_v2(
    r: Vec<f64>,
    r_vals: Vec<f64>,
) -> (Spline<f64, f64>, f64, f64) {
    let rmin: f64 = r[0];
    let rmax: f64 = r[r.len() - 1];

    let mut vec: Vec<Key<f64, f64>> = Vec::new();

    for (r_i, r_val) in r.iter().zip(r_vals.iter()) {
        vec.push(Key::new(*r_i, *r_val, Interpolation::Linear));
    }
    let spline = Spline::from_vec(vec);
    (spline, rmin, rmax)
}

pub fn create_box_around_molecule(
    atoms: &[Atom],
    buffer: Option<f64>,
) -> (f64, f64, f64, f64, f64, f64) {
    // additional space around the molecule
    let mut buffer_val: f64 = 5.0;
    if buffer.is_some() {
        buffer_val = buffer.unwrap();
    }
    let mut xmin: f64 = 10000.0;
    let mut ymin: f64 = 10000.0;
    let mut zmin: f64 = 10000.0;
    let mut xmax: f64 = -10000.0;
    let mut ymax: f64 = -10000.0;
    let mut zmax: f64 = -10000.0;

    // get the maximum values of the cartesian coordinates
    for atom in atoms.iter() {
        let (x, y, z) = (atom.xyz[0], atom.xyz[1], atom.xyz[2]);
        xmin = f64::min(xmin, x);
        ymin = f64::min(ymin, y);
        zmin = f64::min(zmin, z);
        xmax = f64::max(xmax, x);
        ymax = f64::max(ymax, y);
        zmax = f64::max(zmax, z);
    }
    // add the buffer
    xmin -= buffer_val;
    ymin -= buffer_val;
    zmin -= buffer_val;
    xmax += buffer_val;
    ymax += buffer_val;
    zmax += buffer_val;

    (xmin, xmax, ymin, ymax, zmin, zmax)
}
