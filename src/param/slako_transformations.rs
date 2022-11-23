use crate::initialization::parameters::*;
use nalgebra::{Vector3, VectorSlice3};
use ndarray::prelude::*;
use rusty_fitpack::{splder_uniform, splev_uniform};
use std::collections::HashMap;

const SQRT3: f64 = 1.7320508075688772;
// pub fn get_h0_and_s_mu_nu(
//     skt: &SlaterKosterTable,
//     li: i8,
//     mi: i8,
//     posi: ArrayView1<f64>,
//     lj: i8,
//     mj: i8,
//     posj: ArrayView1<f64>,
// ) -> (f64, f64) {
//     let (r, x, y, z): (f64, f64, f64, f64) = directional_cosines(posi, posj);
//     // if the distance `r` is larger than the maximal value on the grid in the parameter files, then
//     // just a zero is returned and the splines are not evaluated at all
//     if r > skt.dmax {
//         (0.0, 0.0)
//     } else {
//         let s: f64 = slako_transformation(r, x, y, z, &skt.s_spline, li, mi, lj, mj);
//         let h: f64 = slako_transformation(r, x, y, z, &skt.h_spline, li, mi, lj, mj);
//         (s, h)
//     }
// }

/// compute directional cosines for the vector going from
/// pos1 to pos2
/// Returns:
/// ========
/// r: length of vector
/// x,y,z: directional cosines
pub fn directional_cosines(pos1: &Vector3<f64>, pos2: &Vector3<f64>) -> (f64, f64, f64, f64) {
    let xc: f64 = pos2.x - pos1.x;
    let yc: f64 = pos2.y - pos1.y;
    let zc: f64 = pos2.z - pos1.z;
    let r: f64 = (xc.powi(2) + yc.powi(2) + zc.powi(2)).sqrt();
    // directional cosines
    let x: f64;
    let y: f64;
    let z: f64;
    if r > 0.0 {
        x = xc / r;
        y = yc / r;
        z = zc / r;
    } else {
        x = 0.0;
        y = 0.0;
        z = 1.0;
    }
    return (r, x, y, z);
}

/// transformation rules for matrix elements
pub fn slako_transformation(
    r: f64,
    x: f64,
    y: f64,
    z: f64,
    s_or_h: &HashMap<u8, (Vec<f64>, Vec<f64>, usize)>,
    l1: i8,
    m1: i8,
    l2: i8,
    m2: i8,
) -> f64 {
    // x,y,z are directional cosines, r is the distance between the two centers
    // length of array sor_h
    // values of the N Slater-Koster tables for S or H0 evaluated at distance r
    // orbital qm numbers for center 1 and center 2
    // Local Variables

    // Result S(x,y,z) or H(x,y,z) after applying SK rules
    // index that encodes the tuple (l1,m1,l2,m2)

    // First we need to transform the tuple (l1,m1,l2,m2) into a unique integer
    // so that the compiler can build a branching table for each case.
    // Valid ranges for qm numbers: 0 <= l1,l2 <= lmax, -lmax <= m1,m2 <= lmax

    //transformation rules for matrix elements
    //# x,y,z are directional cosines, r is the distance between the two centers
    let value = match (l1, m1, l2, m2) {
        (0, 0, 0, 0) => splev_uniform(&s_or_h[&0].0, &s_or_h[&0].1, s_or_h[&0].2, r),
        (0, 0, 1, -1) => y * splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r),
        (0, 0, 1, 0) => z * splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r),
        (0, 0, 1, 1) => x * splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r),
        (0, 0, 2, -2) => {
            x * y * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r) * SQRT3
        }
        (0, 0, 2, -1) => {
            y * z * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r) * SQRT3
        }
        (0, 0, 2, 0) => {
            -((x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r))
                / 2.
        }
        (0, 0, 2, 1) => {
            x * z * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r) * SQRT3
        }
        (0, 0, 2, 2) => {
            ((x - y)
                * (x + y)
                * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                * SQRT3)
                / 2.
        }
        (1, -1, 0, 0) => y * splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r),
        (1, -1, 1, -1) => {
            (x.powi(2) + z.powi(2)) * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + y.powi(2) * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
        }
        (1, -1, 1, 0) => {
            y * z
                * (-splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r))
        }
        (1, -1, 1, 1) => {
            x * y
                * (-splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r))
        }
        (1, -1, 2, -2) => {
            x * ((x.powi(2) - y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + y.powi(2) * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r) * SQRT3)
        }
        (1, -1, 2, -1) => {
            z * ((x.powi(2) - y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + y.powi(2) * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r) * SQRT3)
        }
        (1, -1, 2, 0) => {
            -(y * ((x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                + 2.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    * SQRT3))
                / 2.
        }
        (1, -1, 2, 1) => {
            x * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r) * SQRT3)
        }
        (1, -1, 2, 2) => {
            -(y * (2.0
                * (2.0 * x.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + (-x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    * SQRT3))
                / 2.
        }
        (1, 0, 0, 0) => z * splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r),
        (1, 0, 1, -1) => {
            y * z
                * (-splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r))
        }
        (1, 0, 1, 0) => {
            (x.powi(2) + y.powi(2)) * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + z.powi(2) * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
        }
        (1, 0, 1, 1) => {
            x * z
                * (-splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r))
        }
        (1, 0, 2, -2) => {
            x * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r) * SQRT3)
        }
        (1, 0, 2, -1) => {
            y * ((x.powi(2) + y.powi(2) - z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + z.powi(2) * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r) * SQRT3)
        }
        (1, 0, 2, 0) => {
            z.powi(3) * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                - ((x.powi(2) + y.powi(2))
                    * z
                    * (splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                        - 2.0
                            * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                            * SQRT3))
                    / 2.
        }
        (1, 0, 2, 1) => {
            x * ((x.powi(2) + y.powi(2) - z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + z.powi(2) * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r) * SQRT3)
        }
        (1, 0, 2, 2) => {
            -((x - y)
                * (x + y)
                * z
                * (2.0 * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    - splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r) * SQRT3))
                / 2.
        }
        (1, 1, 0, 0) => x * splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r),
        (1, 1, 1, -1) => {
            x * y
                * (-splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r))
        }
        (1, 1, 1, 0) => {
            x * z
                * (-splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r))
        }
        (1, 1, 1, 1) => {
            (y.powi(2) + z.powi(2)) * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + x.powi(2) * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
        }
        (1, 1, 2, -2) => {
            y * ((-x.powi(2) + y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + x.powi(2) * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r) * SQRT3)
        }
        (1, 1, 2, -1) => {
            x * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r) * SQRT3)
        }
        (1, 1, 2, 0) => {
            -(x * ((x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                + 2.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    * SQRT3))
                / 2.
        }
        (1, 1, 2, 1) => {
            z * ((-x.powi(2) + y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + x.powi(2) * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r) * SQRT3)
        }
        (1, 1, 2, 2) => {
            x * (2.0 * y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + (x * (x - y)
                    * (x + y)
                    * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    * SQRT3)
                    / 2.
        }
        (2, -2, 0, 0) => {
            x * y * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r) * SQRT3
        }
        (2, -2, 1, -1) => {
            x * ((x.powi(2) - y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + y.powi(2)
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    * SQRT3)
        }
        (2, -2, 1, 0) => {
            x * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r) * SQRT3)
        }
        (2, -2, 1, 1) => {
            y * ((-x.powi(2) + y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + x.powi(2)
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    * SQRT3)
        }
        (2, -2, 2, -2) => {
            (x.powi(2) + z.powi(2))
                * (y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + ((x.powi(2) - y.powi(2)).powi(2) + (x.powi(2) + y.powi(2)) * z.powi(2))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
        }
        (2, -2, 2, -1) => {
            x * z
                * (-((x.powi(2) + z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r))
                    + (x.powi(2) - 3.0 * y.powi(2) + z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 3.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
        }
        (2, -2, 2, 0) => {
            (x * y
                * ((x.powi(2) + y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 4.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                * SQRT3)
                / 2.
        }
        (2, -2, 2, 1) => {
            y * z
                * (-((y.powi(2) + z.powi(2))
                    * (splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                        - splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)))
                    + 3.0
                        * x.powi(2)
                        * (-splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                            + splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)))
        }
        (2, -2, 2, 2) => {
            (x * (x - y)
                * y
                * (x + y)
                * (splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 4.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 3.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)))
                / 2.
        }
        (2, -1, 0, 0) => {
            y * z * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r) * SQRT3
        }
        (2, -1, 1, -1) => {
            z * ((x.powi(2) - y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + y.powi(2)
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    * SQRT3)
        }
        (2, -1, 1, 0) => {
            y * ((x.powi(2) + y.powi(2) - z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + z.powi(2)
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    * SQRT3)
        }
        (2, -1, 1, 1) => {
            x * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r) * SQRT3)
        }
        (2, -1, 2, -2) => {
            x * z
                * (-((x.powi(2) + z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r))
                    + (x.powi(2) - 3.0 * y.powi(2) + z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 3.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
        }
        (2, -1, 2, -1) => {
            (x.powi(2) + y.powi(2))
                * (x.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + ((y.powi(2) - z.powi(2)).powi(2) + x.powi(2) * (y.powi(2) + z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
        }
        (2, -1, 2, 0) => {
            -(y * z
                * ((x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 2.0
                        * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                * SQRT3)
                / 2.
        }
        (2, -1, 2, 1) => {
            x * y
                * (-((x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r))
                    + (x.powi(2) + y.powi(2) - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 3.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
        }
        (2, -1, 2, 2) => {
            (y * z
                * ((3.0 * x.powi(2) + y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 2.0
                        * (3.0 * x.powi(2) - y.powi(2) + z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 3.0
                        * (x - y)
                        * (x + y)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)))
                / 2.
        }
        (2, 0, 0, 0) => {
            -((x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r))
                / 2.
        }
        (2, 0, 1, -1) => {
            -(y * ((x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                + 2.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    * SQRT3))
                / 2.
        }
        (2, 0, 1, 0) => {
            z.powi(3) * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                - ((x.powi(2) + y.powi(2))
                    * z
                    * (splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                        - 2.0
                            * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                            * SQRT3))
                    / 2.
        }
        (2, 0, 1, 1) => {
            -(x * ((x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                + 2.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    * SQRT3))
                / 2.
        }
        (2, 0, 2, -2) => {
            (x * y
                * ((x.powi(2) + y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 4.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                * SQRT3)
                / 2.
        }
        (2, 0, 2, -1) => {
            -(y * z
                * ((x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 2.0
                        * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                * SQRT3)
                / 2.
        }
        (2, 0, 2, 0) => {
            (3.0 * (x.powi(2) + y.powi(2))
                * ((x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 4.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r))
                + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)).powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                / 4.
        }
        (2, 0, 2, 1) => {
            -(x * z
                * ((x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 2.0
                        * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                * SQRT3)
                / 2.
        }
        (2, 0, 2, 2) => {
            ((x - y)
                * (x + y)
                * ((x.powi(2) + y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 4.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                * SQRT3)
                / 4.
        }
        (2, 1, 0, 0) => {
            x * z * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r) * SQRT3
        }
        (2, 1, 1, -1) => {
            x * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r) * SQRT3)
        }
        (2, 1, 1, 0) => {
            x * ((x.powi(2) + y.powi(2) - z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + z.powi(2)
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    * SQRT3)
        }
        (2, 1, 1, 1) => {
            z * ((-x.powi(2) + y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + x.powi(2)
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    * SQRT3)
        }
        (2, 1, 2, -2) => {
            y * z
                * (-((y.powi(2) + z.powi(2))
                    * (splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                        - splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)))
                    + 3.0
                        * x.powi(2)
                        * (-splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                            + splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)))
        }
        (2, 1, 2, -1) => {
            x * y
                * (-((x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r))
                    + (x.powi(2) + y.powi(2) - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 3.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
        }
        (2, 1, 2, 0) => {
            -(x * z
                * ((x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 2.0
                        * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                * SQRT3)
                / 2.
        }
        (2, 1, 2, 1) => {
            (x.powi(2) + y.powi(2))
                * (y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (x.powi(4)
                    + x.powi(2) * (y.powi(2) - 2.0 * z.powi(2))
                    + z.powi(2) * (y.powi(2) + z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
        }
        (2, 1, 2, 2) => {
            -(x * z
                * ((x.powi(2) + 3.0 * y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 2.0
                        * (x.powi(2) - 3.0 * y.powi(2) - z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 3.0
                        * (-x.powi(2) + y.powi(2))
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)))
                / 2.
        }
        (2, 2, 0, 0) => {
            ((x - y)
                * (x + y)
                * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                * SQRT3)
                / 2.
        }
        (2, 2, 1, -1) => {
            -(y * (2.0
                * (2.0 * x.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + (-x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    * SQRT3))
                / 2.
        }
        (2, 2, 1, 0) => {
            -((x - y)
                * (x + y)
                * z
                * (2.0 * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    - splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r) * SQRT3))
                / 2.
        }
        (2, 2, 1, 1) => {
            x * (2.0 * y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + (x * (x - y)
                    * (x + y)
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    * SQRT3)
                    / 2.
        }
        (2, 2, 2, -2) => {
            (x * (x - y)
                * y
                * (x + y)
                * (splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 4.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 3.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)))
                / 2.
        }
        (2, 2, 2, -1) => {
            (y * z
                * ((3.0 * x.powi(2) + y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 2.0
                        * (3.0 * x.powi(2) - y.powi(2) + z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 3.0
                        * (x - y)
                        * (x + y)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)))
                / 2.
        }
        (2, 2, 2, 0) => {
            ((x - y)
                * (x + y)
                * ((x.powi(2) + y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 4.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                * SQRT3)
                / 4.
        }
        (2, 2, 2, 1) => {
            -(x * z
                * ((x.powi(2) + 3.0 * y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 2.0
                        * (x.powi(2) - 3.0 * y.powi(2) - z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 3.0
                        * (-x.powi(2) + y.powi(2))
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)))
                / 2.
        }
        (2, 2, 2, 2) => {
            (((x.powi(2) - y.powi(2)).powi(2)
                + 4.0 * (x.powi(2) + y.powi(2)) * z.powi(2)
                + 4.0 * z.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 4.0
                    * (4.0 * x.powi(2) * y.powi(2) + (x.powi(2) + y.powi(2)) * z.powi(2))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * (x.powi(2) - y.powi(2)).powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                / 4.
        }
        _ => panic!("No combination of l1, m1, l2, m2 found!"),
    };
    return value;
}

/// transformation rules for matrix elements
pub fn slako_transformation_gradients(
    r: f64,
    x: f64,
    y: f64,
    z: f64,
    s_or_h: &HashMap<u8, (Vec<f64>, Vec<f64>, usize)>,
    l1: i8,
    m1: i8,
    l2: i8,
    m2: i8,
) -> Array1<f64> {
    // TODO: The splines are evaulated multiple times at the same position. This is an unnecessary
    // load and could be implemented in a more efficient way

    // x,y,z are directional cosines, r is the distance between the two centers
    // length of array sor_h
    // values of the N Slater-Koster tables for S or H0 evaluated at distance r
    // orbital qm numbers for center 1 and center 2
    // Local Variables

    // Result S(x,y,z) or H(x,y,z) after applying SK rules
    // index that encodes the tuple (l1,m1,l2,m2)

    // First we need to transform the tuple (l1,m1,l2,m2) into a unique integer
    // so that the compiler can build a branching table for each case.
    // Valid ranges for qm numbers: 0 <= l1,l2 <= lmax, -lmax <= m1,m2 <= lmax

    //transformation rules for matrix elements
    //# x,y,z are directional cosines, r is the distance between the two centers
    let grad0: f64 = match (l1, m1, l2, m2) {
        (0, 0, 0, 0) => x * splder_uniform(&s_or_h[&0].0, &s_or_h[&0].1, s_or_h[&0].2, r, 1),
        (0, 0, 1, -1) => {
            x * y
                * (-(splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r) / r)
                    + splder_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r, 1))
        }
        (0, 0, 1, 0) => {
            x * z
                * (-(splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r) / r)
                    + splder_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r, 1))
        }
        (0, 0, 1, 1) => {
            -(((-1.0 + x.powi(2)) * splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r))
                / r)
                + x.powi(2) * splder_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r, 1)
        }
        (0, 0, 2, -2) => {
            (SQRT3
                * y
                * ((1.0 - 2.0 * x.powi(2))
                    * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * x.powi(2)
                        * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / r
        }
        (0, 0, 2, -1) => {
            (SQRT3
                * x
                * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / r
        }
        (0, 0, 2, 0) => {
            (x * (-1.0 + x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r))
                / r
                - (x * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                    * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1))
                    / 2.
        }
        (0, 0, 2, 1) => {
            (SQRT3
                * z
                * ((1.0 - 2.0 * x.powi(2))
                    * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * x.powi(2)
                        * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / r
        }
        (0, 0, 2, 2) => {
            (SQRT3
                * x
                * (2.0
                    * (1.0 - x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / (2. * r)
        }
        (1, -1, 0, 0) => {
            x * y
                * (-(splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r) / r)
                    + splder_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r, 1))
        }
        (1, -1, 1, -1) => {
            (x * (-2.0
                * (-1.0 + x.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + r * (x.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                + y.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, -1, 1, 0) => {
            (x * y
                * z
                * (2.0 * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    - 2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                    + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                        + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, -1, 1, 1) => {
            (y * ((-1.0 + 2.0 * x.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + x.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, -1, 2, -2) => {
            ((-y.powi(2) + z.powi(2)
                - 3.0 * x.powi(2) * (-1.0 + x.powi(2) - y.powi(2) + z.powi(2)))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + r * x.powi(2)
                    * (x.powi(2) - y.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                + SQRT3
                    * y.powi(2)
                    * ((1.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                        + r * x.powi(2)
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, -1, 2, -1) => {
            (x * z
                * ((2.0 - 3.0 * x.powi(2) + 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + r * (x.powi(2) - y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + SQRT3
                        * y.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                            + r * splder_uniform(
                                &s_or_h[&9].0,
                                &s_or_h[&9].1,
                                s_or_h[&9].2,
                                r,
                                1,
                            ))))
                / r
        }
        (1, -1, 2, 0) => {
            -(x * y
                * ((2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + 2.0
                        * z.powi(2)
                        * (-3.0
                            * SQRT3
                            * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                            + 3.0 * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                            + r * SQRT3
                                * splder_uniform(
                                    &s_or_h[&8].0,
                                    &s_or_h[&8].1,
                                    s_or_h[&8].2,
                                    r,
                                    1,
                                ))
                    + r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / (2. * r)
        }
        (1, -1, 2, 1) => {
            (y * z
                * ((-2.0 + 6.0 * x.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * x.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (1, -1, 2, 2) => {
            (x * y
                * (2.0
                    * (-4.0 + 6.0 * x.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (2.0 - 3.0 * x.powi(2) + 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    - 2.0
                        * r
                        * (2.0 * x.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + r * SQRT3
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / (2. * r)
        }
        (1, 0, 0, 0) => {
            x * z
                * (-(splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r) / r)
                    + splder_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r, 1))
        }
        (1, 0, 1, -1) => {
            (x * y
                * z
                * (2.0 * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    - 2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                    + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                        + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, 0, 1, 0) => {
            (x * (-2.0
                * (-1.0 + x.powi(2) + y.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + r * (x.powi(2) + y.powi(2))
                    * splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                + z.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, 0, 1, 1) => {
            (z * ((-1.0 + 2.0 * x.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + x.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, 0, 2, -2) => {
            (y * z
                * ((-2.0 + 6.0 * x.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * x.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (1, 0, 2, -1) => {
            (x * y
                * ((2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + r * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + SQRT3
                        * z.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                            + r * splder_uniform(
                                &s_or_h[&9].0,
                                &s_or_h[&9].1,
                                s_or_h[&9].2,
                                r,
                                1,
                            ))))
                / r
        }
        (1, 0, 2, 0) => {
            -(x * z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + (2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2) + 6.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * (-2.0
                        * SQRT3
                        * (x.powi(2) + y.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1))))
                / (2. * r)
        }
        (1, 0, 2, 1) => {
            (((y - z) * (y + z) - 3.0 * x.powi(2) * (-1.0 + x.powi(2) + y.powi(2) - z.powi(2)))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + r * x.powi(2)
                    * (x.powi(2) + y.powi(2) - z.powi(2))
                    * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                + SQRT3
                    * z.powi(2)
                    * ((1.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                        + r * x.powi(2)
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, 0, 2, 2) => {
            (x * z
                * ((-4.0 + 6.0 * x.powi(2) - 6.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + 2.0 * SQRT3 * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    - (x - y)
                        * (x + y)
                        * (3.0
                            * SQRT3
                            * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                            + 2.0
                                * r
                                * splder_uniform(
                                    &s_or_h[&8].0,
                                    &s_or_h[&8].1,
                                    s_or_h[&8].2,
                                    r,
                                    1,
                                )
                            - r * SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / (2. * r)
        }
        (1, 1, 0, 0) => {
            -(((-1.0 + x.powi(2)) * splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r))
                / r)
                + x.powi(2) * splder_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r, 1)
        }
        (1, 1, 1, -1) => {
            (y * ((-1.0 + 2.0 * x.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + x.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, 1, 1, 0) => {
            (z * ((-1.0 + 2.0 * x.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + x.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, 1, 1, 1) => {
            (x * (-2.0
                * (-1.0 + x.powi(2))
                * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                - (y.powi(2) + z.powi(2))
                    * (2.0 * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                        - r * splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1))
                + r * x.powi(2) * splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))
                / r
        }
        (1, 1, 2, -2) => {
            (x * y
                * ((-2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (2.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * (-x.powi(2) + y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + r * SQRT3
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, 1, 2, -1) => {
            (y * z
                * ((-2.0 + 6.0 * x.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * x.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (1, 1, 2, 0) => {
            -(2.0
                * SQRT3
                * (1.0 - 3.0 * x.powi(2))
                * z.powi(2)
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + (y.powi(2)
                    - 2.0 * z.powi(2)
                    - 3.0 * x.powi(2) * (-1.0 + x.powi(2) + y.powi(2) - 2.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                + r * x.powi(2)
                    * (2.0
                        * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / (2. * r)
        }
        (1, 1, 2, 1) => {
            (x * z
                * ((-2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (2.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * (-x.powi(2) + y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + r * SQRT3
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, 1, 2, 2) => {
            (-2.0
                * (-1.0 + 3.0 * x.powi(2))
                * (2.0 * y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                - SQRT3
                    * (3.0 * x.powi(4) + y.powi(2) - 3.0 * x.powi(2) * (1.0 + y.powi(2)))
                    * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                + r * x.powi(2)
                    * (2.0
                        * (2.0 * y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                        + SQRT3
                            * (x - y)
                            * (x + y)
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / (2. * r)
        }
        (2, -2, 0, 0) => {
            (SQRT3
                * y
                * ((1.0 - 2.0 * x.powi(2))
                    * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * x.powi(2)
                        * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / r
        }
        (2, -2, 1, -1) => {
            ((-y.powi(2) + z.powi(2)
                - 3.0 * x.powi(2) * (-1.0 + x.powi(2) - y.powi(2) + z.powi(2)))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + r * x.powi(2)
                    * (x.powi(2) - y.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                + SQRT3
                    * y.powi(2)
                    * ((1.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                        + r * x.powi(2)
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, -2, 1, 0) => {
            (y * z
                * ((-2.0 + 6.0 * x.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * x.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&12].0,
                                    &s_or_h[&12].1,
                                    s_or_h[&12].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (2, -2, 1, 1) => {
            (x * y
                * ((-2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (2.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * (-x.powi(2) + y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + r * SQRT3
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, -2, 2, -2) => {
            (x * (-2.0
                * (y.powi(2) + z.powi(2))
                * (-1.0 + 2.0 * x.powi(2) + 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (-2.0 * x.powi(4)
                        + z.powi(2)
                        + x.powi(2) * (2.0 + 4.0 * y.powi(2) - 2.0 * z.powi(2))
                        - 2.0 * y.powi(2) * (1.0 + y.powi(2) + z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 6.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -2, 2, -1) => {
            (z * ((4.0 * x.powi(4) - z.powi(2) + x.powi(2) * (-3.0 + 4.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (-3.0 * y.powi(2)
                    + z.powi(2)
                    + x.powi(2) * (3.0 - 4.0 * x.powi(2) + 12.0 * y.powi(2) - 4.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -2, 2, 0) => {
            (SQRT3
                * y
                * ((-4.0 * x.powi(4)
                    + y.powi(2)
                    + 2.0 * z.powi(2)
                    + x.powi(2) * (3.0 - 4.0 * y.powi(2) - 8.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 4.0
                        * (-1.0 + 4.0 * x.powi(2))
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 3.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - y.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 2.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * x.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + r * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 4.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * x.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, -2, 2, 1) => {
            (x * y
                * z
                * (4.0
                    * (y.powi(2) + z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 4.0
                        * (y.powi(2) + z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 6.0
                        * (-1.0 + 2.0 * x.powi(2))
                        * (splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                            - splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                    + r * (-((y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1))
                        + (-3.0 * x.powi(2) + y.powi(2) + z.powi(2))
                            * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                        + 3.0
                            * x.powi(2)
                            * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1))))
                / r
        }
        (2, -2, 2, 2) => {
            (y * (-((4.0 * x.powi(4) + y.powi(2) - x.powi(2) * (3.0 + 4.0 * y.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r))
                + 4.0
                    * (y.powi(2) + x.powi(2) * (-3.0 + 4.0 * x.powi(2) - 4.0 * y.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 9.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 4.0
                    * r
                    * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 4.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * (x - y)
                    * (x + y)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, -1, 0, 0) => {
            (SQRT3
                * x
                * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / r
        }
        (2, -1, 1, -1) => {
            (x * z
                * ((2.0 - 3.0 * x.powi(2) + 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + r * (x.powi(2) - y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + SQRT3
                        * y.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                            + r * splder_uniform(
                                &s_or_h[&12].0,
                                &s_or_h[&12].1,
                                s_or_h[&12].2,
                                r,
                                1,
                            ))))
                / r
        }
        (2, -1, 1, 0) => {
            (x * y
                * ((2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + r * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + SQRT3
                        * z.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                            + r * splder_uniform(
                                &s_or_h[&12].0,
                                &s_or_h[&12].1,
                                s_or_h[&12].2,
                                r,
                                1,
                            ))))
                / r
        }
        (2, -1, 1, 1) => {
            (y * z
                * ((-2.0 + 6.0 * x.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * x.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&12].0,
                                    &s_or_h[&12].1,
                                    s_or_h[&12].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (2, -1, 2, -2) => {
            (z * ((4.0 * x.powi(4) - z.powi(2) + x.powi(2) * (-3.0 + 4.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (-3.0 * y.powi(2)
                    + z.powi(2)
                    + x.powi(2) * (3.0 - 4.0 * x.powi(2) + 12.0 * y.powi(2) - 4.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -1, 2, -1) => {
            (x * (2.0
                * (y.powi(2) - 2.0 * x.powi(2) * (-1.0 + x.powi(2) + y.powi(2)) + z.powi(2)
                    - 2.0 * (x.powi(2) + y.powi(2)) * z.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (-2.0 * y.powi(4) + z.powi(2) - 2.0 * z.powi(2) * (x.powi(2) + z.powi(2))
                        + y.powi(2) * (1.0 - 2.0 * x.powi(2) + 4.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                - 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -1, 2, 0) => {
            (SQRT3
                * x
                * y
                * z
                * ((-2.0 + 4.0 * x.powi(2) + 4.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + (4.0 - 8.0 * x.powi(2) - 8.0 * y.powi(2) + 8.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 2.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, -1, 2, 1) => {
            (y * ((4.0 * x.powi(4) - y.powi(2) + x.powi(2) * (-3.0 + 4.0 * y.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (y.powi(2) - 3.0 * z.powi(2)
                    + x.powi(2) * (3.0 - 4.0 * x.powi(2) - 4.0 * y.powi(2) + 12.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -1, 2, 2) => {
            (x * y
                * z
                * (-2.0
                    * (-3.0 + 6.0 * x.powi(2) + 2.0 * y.powi(2) + 4.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 4.0
                        * (-3.0 + 6.0 * x.powi(2) - 2.0 * y.powi(2) + 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 6.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 12.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 12.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 3.0
                        * r
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + r * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 6.0
                        * r
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 3.0
                        * r
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 0, 0) => {
            (x * (-1.0 + x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r))
                / r
                - (x * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                    * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1))
                    / 2.
        }
        (2, 0, 1, -1) => {
            -(x * y
                * ((2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + 2.0
                        * z.powi(2)
                        * (-3.0
                            * SQRT3
                            * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                            + 3.0
                                * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                            + r * SQRT3
                                * splder_uniform(
                                    &s_or_h[&11].0,
                                    &s_or_h[&11].1,
                                    s_or_h[&11].2,
                                    r,
                                    1,
                                ))
                    + r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 1, 0) => {
            -(x * z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + (2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2) + 6.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * (-2.0
                        * SQRT3
                        * (x.powi(2) + y.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1))))
                / (2. * r)
        }
        (2, 0, 1, 1) => {
            -(2.0
                * SQRT3
                * (1.0 - 3.0 * x.powi(2))
                * z.powi(2)
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + (y.powi(2)
                    - 2.0 * z.powi(2)
                    - 3.0 * x.powi(2) * (-1.0 + x.powi(2) + y.powi(2) - 2.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                + r * x.powi(2)
                    * (2.0
                        * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, -2) => {
            (SQRT3
                * y
                * ((-4.0 * x.powi(4)
                    + y.powi(2)
                    + 2.0 * z.powi(2)
                    + x.powi(2) * (3.0 - 4.0 * y.powi(2) - 8.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 4.0
                        * (-1.0 + 4.0 * x.powi(2))
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 3.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - y.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 2.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * x.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + r * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 4.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * x.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, -1) => {
            (SQRT3
                * x
                * y
                * z
                * ((-2.0 + 4.0 * x.powi(2) + 4.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + (4.0 - 8.0 * x.powi(2) - 8.0 * y.powi(2) + 8.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 2.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, 0) => {
            (x * (-12.0
                * (-1.0 + x.powi(2) + y.powi(2))
                * (x.powi(2) + y.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                - 24.0
                    * (-1.0 + 2.0 * x.powi(2) + 2.0 * y.powi(2))
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 4.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 4.0
                    * x.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 4.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 8.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 4.0
                    * y.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 8.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 16.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 16.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 16.0
                    * z.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 3.0
                    * r
                    * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 6.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 3.0
                    * r
                    * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 12.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 12.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)).powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        (2, 0, 2, 1) => {
            (SQRT3
                * z
                * ((4.0 * x.powi(4) - y.powi(2) + x.powi(2) * (-3.0 + 4.0 * y.powi(2)))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 2.0
                        * ((y - z) * (y + z)
                            + x.powi(2)
                                * (3.0 - 4.0 * x.powi(2) - 4.0 * y.powi(2) + 4.0 * z.powi(2)))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 3.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - y.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 2.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(4)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * x.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, 2) => {
            (SQRT3
                * x
                * (4.0
                    * (-x.powi(4)
                        + y.powi(4)
                        + z.powi(2)
                        + 2.0 * y.powi(2) * z.powi(2)
                        + x.powi(2) * (1.0 - 2.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 8.0
                        * (-1.0 + 2.0 * x.powi(2) - 2.0 * y.powi(2))
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 4.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 4.0
                        * y.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 8.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * x.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 4.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 4.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * (x - y)
                        * (x + y)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        (2, 1, 0, 0) => {
            (SQRT3
                * z
                * ((1.0 - 2.0 * x.powi(2))
                    * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * x.powi(2)
                        * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / r
        }
        (2, 1, 1, -1) => {
            (y * z
                * ((-2.0 + 6.0 * x.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * x.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&12].0,
                                    &s_or_h[&12].1,
                                    s_or_h[&12].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (2, 1, 1, 0) => {
            (((y - z) * (y + z) - 3.0 * x.powi(2) * (-1.0 + x.powi(2) + y.powi(2) - z.powi(2)))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + r * x.powi(2)
                    * (x.powi(2) + y.powi(2) - z.powi(2))
                    * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                + SQRT3
                    * z.powi(2)
                    * ((1.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                        + r * x.powi(2)
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, 1, 1, 1) => {
            (x * z
                * ((-2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (2.0 - 3.0 * x.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * (-x.powi(2) + y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + r * SQRT3
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, 1, 2, -2) => {
            (x * y
                * z
                * (4.0
                    * (y.powi(2) + z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 4.0
                        * (y.powi(2) + z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 6.0
                        * (-1.0 + 2.0 * x.powi(2))
                        * (splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                            - splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                    + r * (-((y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1))
                        + (-3.0 * x.powi(2) + y.powi(2) + z.powi(2))
                            * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                        + 3.0
                            * x.powi(2)
                            * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1))))
                / r
        }
        (2, 1, 2, -1) => {
            (y * ((4.0 * x.powi(4) - y.powi(2) + x.powi(2) * (-3.0 + 4.0 * y.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (y.powi(2) - 3.0 * z.powi(2)
                    + x.powi(2) * (3.0 - 4.0 * x.powi(2) - 4.0 * y.powi(2) + 12.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, 1, 2, 0) => {
            (SQRT3
                * z
                * ((4.0 * x.powi(4) - y.powi(2) + x.powi(2) * (-3.0 + 4.0 * y.powi(2)))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 2.0
                        * ((y - z) * (y + z)
                            + x.powi(2)
                                * (3.0 - 4.0 * x.powi(2) - 4.0 * y.powi(2) + 4.0 * z.powi(2)))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 3.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - y.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 2.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(4)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * x.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 1, 2, 1) => {
            (x * (-2.0
                * (-1.0 + 2.0 * x.powi(2) + 2.0 * y.powi(2))
                * (y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (y.powi(2)
                        - 2.0 * z.powi(2)
                        - 2.0
                            * (x.powi(2) * (-1.0 + x.powi(2) + y.powi(2))
                                + (-2.0 * x.powi(2) + y.powi(2)) * z.powi(2)
                                + z.powi(4)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 6.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, 1, 2, 2) => {
            -(z * ((-4.0 * x.powi(4)
                + 3.0 * y.powi(2)
                + 2.0 * z.powi(2)
                + x.powi(2) * (3.0 - 12.0 * y.powi(2) - 8.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                - 2.0
                    * (4.0 * x.powi(4) + 3.0 * y.powi(2) + z.powi(2)
                        - x.powi(2) * (3.0 + 12.0 * y.powi(2) + 4.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                - 9.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * x.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 6.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * x.powi(2)
                    * (x - y)
                    * (x + y)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 0, 0) => {
            (SQRT3
                * x
                * (2.0
                    * (1.0 - x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 1, -1) => {
            (x * y
                * (2.0
                    * (-4.0 + 6.0 * x.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (2.0 - 3.0 * x.powi(2) + 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    - 2.0
                        * r
                        * (2.0 * x.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + r * SQRT3
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 1, 0) => {
            (x * z
                * ((-4.0 + 6.0 * x.powi(2) - 6.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (2.0 - 3.0 * x.powi(2) + 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    - r * (x - y)
                        * (x + y)
                        * (2.0
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)
                            - SQRT3
                                * splder_uniform(
                                    &s_or_h[&11].0,
                                    &s_or_h[&11].1,
                                    s_or_h[&11].2,
                                    r,
                                    1,
                                ))))
                / (2. * r)
        }
        (2, 2, 1, 1) => {
            (-2.0
                * (-1.0 + 3.0 * x.powi(2))
                * (2.0 * y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                - SQRT3
                    * (3.0 * x.powi(4) + y.powi(2) - 3.0 * x.powi(2) * (1.0 + y.powi(2)))
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                + r * x.powi(2)
                    * (2.0
                        * (2.0 * y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                        + SQRT3
                            * (x - y)
                            * (x + y)
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, -2) => {
            (y * (-((4.0 * x.powi(4) + y.powi(2) - x.powi(2) * (3.0 + 4.0 * y.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r))
                + 4.0
                    * (y.powi(2) + x.powi(2) * (-3.0 + 4.0 * x.powi(2) - 4.0 * y.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 9.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 4.0
                    * r
                    * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 4.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * (x - y)
                    * (x + y)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, -1) => {
            (x * y
                * z
                * (-2.0
                    * (-3.0 + 6.0 * x.powi(2) + 2.0 * y.powi(2) + 4.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 4.0
                        * (-3.0 + 6.0 * x.powi(2) - 2.0 * y.powi(2) + 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 6.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 12.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 12.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 3.0
                        * r
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + r * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 6.0
                        * r
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 3.0
                        * r
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, 0) => {
            (SQRT3
                * x
                * (4.0
                    * (-x.powi(4)
                        + y.powi(4)
                        + z.powi(2)
                        + 2.0 * y.powi(2) * z.powi(2)
                        + x.powi(2) * (1.0 - 2.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 8.0
                        * (-1.0 + 2.0 * x.powi(2) - 2.0 * y.powi(2))
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 4.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 4.0
                        * y.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 8.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * x.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 4.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 4.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * (x - y)
                        * (x + y)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        (2, 2, 2, 1) => {
            -(z * ((-4.0 * x.powi(4)
                + 3.0 * y.powi(2)
                + 2.0 * z.powi(2)
                + x.powi(2) * (3.0 - 12.0 * y.powi(2) - 8.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                - 2.0
                    * (4.0 * x.powi(4) + 3.0 * y.powi(2) + z.powi(2)
                        - x.powi(2) * (3.0 + 12.0 * y.powi(2) + 4.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                - 9.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * x.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 6.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * x.powi(2)
                    * (x - y)
                    * (x + y)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, 2) => {
            (x * (-4.0
                * (x.powi(4) + y.powi(2) - 2.0 * z.powi(2)
                    + (y.powi(2) + 2.0 * z.powi(2)).powi(2)
                    + x.powi(2) * (-1.0 - 2.0 * y.powi(2) + 4.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                - 8.0
                    * ((-1.0 + 2.0 * x.powi(2)) * z.powi(2)
                        + 2.0 * y.powi(2) * (-2.0 + 4.0 * x.powi(2) + z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 12.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 24.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 4.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 4.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 4.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 16.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 4.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 4.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * (x.powi(2) - y.powi(2)).powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        _ => panic!("No combination of l1, m1, l2, m2 found!"),
    };

    let grad1: f64 = match (l1, m1, l2, m2) {
        (0, 0, 0, 0) => y * splder_uniform(&s_or_h[&0].0, &s_or_h[&0].1, s_or_h[&0].2, r, 1),
        (0, 0, 1, -1) => {
            -(((-1.0 + y.powi(2)) * splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r))
                / r)
                + y.powi(2) * splder_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r, 1)
        }
        (0, 0, 1, 0) => {
            y * z
                * (-(splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r) / r)
                    + splder_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r, 1))
        }
        (0, 0, 1, 1) => {
            x * y
                * (-(splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r) / r)
                    + splder_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r, 1))
        }
        (0, 0, 2, -2) => {
            (SQRT3
                * x
                * ((1.0 - 2.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * y.powi(2)
                        * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / r
        }
        (0, 0, 2, -1) => {
            (SQRT3
                * z
                * ((1.0 - 2.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * y.powi(2)
                        * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / r
        }
        (0, 0, 2, 0) => {
            (y * (-1.0 + x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r))
                / r
                - (y * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                    * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1))
                    / 2.
        }
        (0, 0, 2, 1) => {
            (SQRT3
                * x
                * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / r
        }
        (0, 0, 2, 2) => {
            (SQRT3
                * y
                * (-2.0
                    * (1.0 + x.powi(2) - y.powi(2))
                    * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / (2. * r)
        }
        (1, -1, 0, 0) => {
            -(((-1.0 + y.powi(2)) * splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r))
                / r)
                + y.powi(2) * splder_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r, 1)
        }
        (1, -1, 1, -1) => {
            (y * (-2.0
                * (-1.0 + y.powi(2))
                * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                - (x.powi(2) + z.powi(2))
                    * (2.0 * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                        - r * splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1))
                + r * y.powi(2) * splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))
                / r
        }
        (1, -1, 1, 0) => {
            (z * ((-1.0 + 2.0 * y.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + y.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, -1, 1, 1) => {
            (x * ((-1.0 + 2.0 * y.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + y.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, -1, 2, -2) => {
            (x * y
                * (-((2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r))
                    + SQRT3
                        * (2.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * (x.powi(2) - y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + r * SQRT3
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, -1, 2, -1) => {
            (y * z
                * (-((2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r))
                    + SQRT3
                        * (2.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * (x.powi(2) - y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + r * SQRT3
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, -1, 2, 0) => {
            -(2.0
                * SQRT3
                * (1.0 - 3.0 * y.powi(2))
                * z.powi(2)
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + x.powi(2) * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                + (-2.0 * z.powi(2)
                    - 3.0 * y.powi(2) * (-1.0 + x.powi(2) + y.powi(2) - 2.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                + r * y.powi(2)
                    * (2.0
                        * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / (2. * r)
        }
        (1, -1, 2, 1) => {
            (x * z
                * ((-2.0 + 6.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * y.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (1, -1, 2, 2) => {
            (2.0 * (-1.0 + 3.0 * y.powi(2))
                * (2.0 * x.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + SQRT3
                    * (x.powi(2) - 3.0 * (1.0 + x.powi(2)) * y.powi(2) + 3.0 * y.powi(4))
                    * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                + r * y.powi(2)
                    * (-2.0
                        * (2.0 * x.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                        + SQRT3
                            * (x - y)
                            * (x + y)
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / (2. * r)
        }
        (1, 0, 0, 0) => {
            y * z
                * (-(splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r) / r)
                    + splder_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r, 1))
        }
        (1, 0, 1, -1) => {
            (z * ((-1.0 + 2.0 * y.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + y.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, 0, 1, 0) => {
            (y * (-2.0
                * (-1.0 + x.powi(2) + y.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + r * (x.powi(2) + y.powi(2))
                    * splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                + z.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, 0, 1, 1) => {
            (x * y
                * z
                * (2.0 * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    - 2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                    + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                        + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, 0, 2, -2) => {
            (x * z
                * ((-2.0 + 6.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * y.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (1, 0, 2, -1) => {
            ((x.powi(2) * (1.0 - 3.0 * y.powi(2)) - z.powi(2)
                + 3.0 * y.powi(2) * (1.0 - y.powi(2) + z.powi(2)))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + r * y.powi(2)
                    * (x.powi(2) + y.powi(2) - z.powi(2))
                    * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                + SQRT3
                    * z.powi(2)
                    * ((1.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                        + r * y.powi(2)
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, 0, 2, 0) => {
            -(y * z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + (2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2) + 6.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * (-2.0
                        * SQRT3
                        * (x.powi(2) + y.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1))))
                / (2. * r)
        }
        (1, 0, 2, 1) => {
            (x * y
                * ((2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + r * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + SQRT3
                        * z.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                            + r * splder_uniform(
                                &s_or_h[&9].0,
                                &s_or_h[&9].1,
                                s_or_h[&9].2,
                                r,
                                1,
                            ))))
                / r
        }
        (1, 0, 2, 2) => {
            (y * z
                * ((4.0 + 6.0 * x.powi(2) - 6.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    - 2.0 * SQRT3 * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    - (x - y)
                        * (x + y)
                        * (3.0
                            * SQRT3
                            * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                            + 2.0
                                * r
                                * splder_uniform(
                                    &s_or_h[&8].0,
                                    &s_or_h[&8].1,
                                    s_or_h[&8].2,
                                    r,
                                    1,
                                )
                            - r * SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / (2. * r)
        }
        (1, 1, 0, 0) => {
            x * y
                * (-(splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r) / r)
                    + splder_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r, 1))
        }
        (1, 1, 1, -1) => {
            (x * ((-1.0 + 2.0 * y.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + y.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, 1, 1, 0) => {
            (x * y
                * z
                * (2.0 * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    - 2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                    + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                        + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, 1, 1, 1) => {
            (y * (-2.0
                * (-1.0 + y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + r * (y.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                + x.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, 1, 2, -2) => {
            ((x.powi(2) * (-1.0 + 3.0 * y.powi(2)) + z.powi(2)
                - 3.0 * y.powi(2) * (-1.0 + y.powi(2) + z.powi(2)))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + r * y.powi(2)
                    * (-x.powi(2) + y.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                + SQRT3
                    * x.powi(2)
                    * ((1.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                        + r * y.powi(2)
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, 1, 2, -1) => {
            (x * z
                * ((-2.0 + 6.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * y.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (1, 1, 2, 0) => {
            -(x * y
                * ((2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + 2.0
                        * z.powi(2)
                        * (-3.0
                            * SQRT3
                            * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                            + 3.0 * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                            + r * SQRT3
                                * splder_uniform(
                                    &s_or_h[&8].0,
                                    &s_or_h[&8].1,
                                    s_or_h[&8].2,
                                    r,
                                    1,
                                ))
                    + r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / (2. * r)
        }
        (1, 1, 2, 1) => {
            (y * z
                * ((2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + r * (-x.powi(2) + y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + SQRT3
                        * x.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                            + r * splder_uniform(
                                &s_or_h[&9].0,
                                &s_or_h[&9].1,
                                s_or_h[&9].2,
                                r,
                                1,
                            ))))
                / r
        }
        (1, 1, 2, 2) => {
            (x * y
                * (-2.0
                    * (-4.0 + 6.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (-2.0 - 3.0 * x.powi(2) + 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + 2.0
                        * r
                        * (2.0 * y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + r * SQRT3
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / (2. * r)
        }
        (2, -2, 0, 0) => {
            (SQRT3
                * x
                * ((1.0 - 2.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * y.powi(2)
                        * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / r
        }
        (2, -2, 1, -1) => {
            (x * y
                * (-((2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r))
                    + SQRT3
                        * (2.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * (x.powi(2) - y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + r * SQRT3
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, -2, 1, 0) => {
            (x * z
                * ((-2.0 + 6.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * y.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&12].0,
                                    &s_or_h[&12].1,
                                    s_or_h[&12].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (2, -2, 1, 1) => {
            ((x.powi(2) * (-1.0 + 3.0 * y.powi(2)) + z.powi(2)
                - 3.0 * y.powi(2) * (-1.0 + y.powi(2) + z.powi(2)))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + r * y.powi(2)
                    * (-x.powi(2) + y.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                + SQRT3
                    * x.powi(2)
                    * ((1.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                        + r * y.powi(2)
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, -2, 2, -2) => {
            (y * (-2.0
                * (x.powi(2) + z.powi(2))
                * (-1.0 + 2.0 * y.powi(2) + 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (-2.0 * (x - y) * (x + y) * (1.0 + x.powi(2) - y.powi(2))
                        + (1.0 - 2.0 * x.powi(2) - 2.0 * y.powi(2)) * z.powi(2))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 6.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -2, 2, -1) => {
            (x * y
                * z
                * (4.0
                    * (x.powi(2) + z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 2.0
                        * (3.0 + 2.0 * x.powi(2) - 6.0 * y.powi(2) + 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 6.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 12.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * (x.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + r * x.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 3.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + r * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 3.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -2, 2, 0) => {
            (SQRT3
                * x
                * ((x.powi(2) * (1.0 - 4.0 * y.powi(2))
                    + 2.0 * z.powi(2)
                    + y.powi(2) * (3.0 - 4.0 * y.powi(2) - 8.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 4.0
                        * (-1.0 + 4.0 * y.powi(2))
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - x.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 3.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 2.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + r * y.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 4.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * y.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, -2, 2, 1) => {
            (z * ((4.0 * y.powi(4) - z.powi(2) + y.powi(2) * (-3.0 + 4.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (3.0 * x.powi(2) * (-1.0 + 4.0 * y.powi(2))
                    + z.powi(2)
                    + y.powi(2) * (3.0 - 4.0 * y.powi(2) - 4.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -2, 2, 2) => {
            (x * ((x.powi(2) - (3.0 + 4.0 * x.powi(2)) * y.powi(2) + 4.0 * y.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 4.0
                    * (-x.powi(2) + (3.0 + 4.0 * x.powi(2)) * y.powi(2) - 4.0 * y.powi(4))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 9.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * y.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 4.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 4.0
                    * r
                    * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * (x - y)
                    * y.powi(2)
                    * (x + y)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, -1, 0, 0) => {
            (SQRT3
                * z
                * ((1.0 - 2.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * y.powi(2)
                        * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / r
        }
        (2, -1, 1, -1) => {
            (y * z
                * (-((2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r))
                    + SQRT3
                        * (2.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * (x.powi(2) - y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + r * SQRT3
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, -1, 1, 0) => {
            ((x.powi(2) * (1.0 - 3.0 * y.powi(2)) - z.powi(2)
                + 3.0 * y.powi(2) * (1.0 - y.powi(2) + z.powi(2)))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + r * y.powi(2)
                    * (x.powi(2) + y.powi(2) - z.powi(2))
                    * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                + SQRT3
                    * z.powi(2)
                    * ((1.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                        + r * y.powi(2)
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, -1, 1, 1) => {
            (x * z
                * ((-2.0 + 6.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * y.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&12].0,
                                    &s_or_h[&12].1,
                                    s_or_h[&12].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (2, -1, 2, -2) => {
            (x * y
                * z
                * (4.0
                    * (x.powi(2) + z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 2.0
                        * (3.0 + 2.0 * x.powi(2) - 6.0 * y.powi(2) + 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 6.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 12.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * (x.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + r * x.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 3.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + r * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 3.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -1, 2, -1) => {
            (y * (-2.0
                * (-1.0 + 2.0 * x.powi(2) + 2.0 * y.powi(2))
                * (x.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (x.powi(2) * (1.0 - 2.0 * y.powi(2) - 2.0 * z.powi(2))
                        - 2.0 * (y - z) * (y + z) * (-1.0 + y.powi(2) - z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 6.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -1, 2, 0) => {
            (SQRT3
                * z
                * ((-x.powi(2) + (-3.0 + 4.0 * x.powi(2)) * y.powi(2) + 4.0 * y.powi(4))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 2.0
                        * (x.powi(2) + 3.0 * y.powi(2)
                            - 4.0 * x.powi(2) * y.powi(2)
                            - 4.0 * y.powi(4)
                            + (-1.0 + 4.0 * y.powi(2)) * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - x.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 3.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 2.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(4)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * y.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, -1, 2, 1) => {
            (x * ((-x.powi(2) + (-3.0 + 4.0 * x.powi(2)) * y.powi(2) + 4.0 * y.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (x.powi(2) + 3.0 * y.powi(2) - 4.0 * x.powi(2) * y.powi(2) - 4.0 * y.powi(4)
                    + 3.0 * (-1.0 + 4.0 * y.powi(2)) * z.powi(2))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -1, 2, 2) => {
            (z * ((-4.0 * y.powi(4)
                + x.powi(2) * (3.0 - 12.0 * y.powi(2))
                + 2.0 * z.powi(2)
                + y.powi(2) * (3.0 - 8.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (-8.0 * y.powi(4) + 6.0 * x.powi(2) * (-1.0 + 4.0 * y.powi(2))
                    - 2.0 * z.powi(2)
                    + y.powi(2) * (6.0 + 8.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 9.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * y.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 6.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 2.0
                    * r
                    * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * (x - y)
                    * y.powi(2)
                    * (x + y)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 0, 0) => {
            (y * (-1.0 + x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r))
                / r
                - (y * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                    * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1))
                    / 2.
        }
        (2, 0, 1, -1) => {
            -(2.0
                * SQRT3
                * (1.0 - 3.0 * y.powi(2))
                * z.powi(2)
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + (-3.0 * y.powi(4) + x.powi(2) * (1.0 - 3.0 * y.powi(2)) - 2.0 * z.powi(2)
                    + y.powi(2) * (3.0 + 6.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                + r * y.powi(2)
                    * (2.0
                        * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 1, 0) => {
            -(y * z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + (2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2) + 6.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * (-2.0
                        * SQRT3
                        * (x.powi(2) + y.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1))))
                / (2. * r)
        }
        (2, 0, 1, 1) => {
            -(x * y
                * ((2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + 2.0
                        * z.powi(2)
                        * (-3.0
                            * SQRT3
                            * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                            + 3.0
                                * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                            + r * SQRT3
                                * splder_uniform(
                                    &s_or_h[&11].0,
                                    &s_or_h[&11].1,
                                    s_or_h[&11].2,
                                    r,
                                    1,
                                ))
                    + r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, -2) => {
            (SQRT3
                * x
                * ((x.powi(2) * (1.0 - 4.0 * y.powi(2))
                    + 2.0 * z.powi(2)
                    + y.powi(2) * (3.0 - 4.0 * y.powi(2) - 8.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 4.0
                        * (-1.0 + 4.0 * y.powi(2))
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - x.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 3.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 2.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + r * y.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 4.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * y.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, -1) => {
            (SQRT3
                * z
                * ((-x.powi(2) + (-3.0 + 4.0 * x.powi(2)) * y.powi(2) + 4.0 * y.powi(4))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 2.0
                        * (x.powi(2) + 3.0 * y.powi(2)
                            - 4.0 * x.powi(2) * y.powi(2)
                            - 4.0 * y.powi(4)
                            + (-1.0 + 4.0 * y.powi(2)) * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - x.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 3.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 2.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(4)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * y.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, 0) => {
            (y * (-12.0
                * (-1.0 + x.powi(2) + y.powi(2))
                * (x.powi(2) + y.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                - 24.0
                    * (-1.0 + 2.0 * x.powi(2) + 2.0 * y.powi(2))
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 4.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 4.0
                    * x.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 4.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 8.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 4.0
                    * y.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 8.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 16.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 16.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 16.0
                    * z.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 3.0
                    * r
                    * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 6.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 3.0
                    * r
                    * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 12.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 12.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)).powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        (2, 0, 2, 1) => {
            (SQRT3
                * x
                * y
                * z
                * ((-2.0 + 4.0 * x.powi(2) + 4.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + (4.0 - 8.0 * x.powi(2) - 8.0 * y.powi(2) + 8.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 2.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, 2) => {
            (SQRT3
                * y
                * (-4.0
                    * (x.powi(4) + y.powi(2) - y.powi(4)
                        + (1.0 + 2.0 * x.powi(2) - 2.0 * y.powi(2)) * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 8.0
                        * (1.0 + 2.0 * x.powi(2) - 2.0 * y.powi(2))
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 4.0
                        * x.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 4.0
                        * y.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 4.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 8.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * x.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 4.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 4.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * (x - y)
                        * (x + y)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        (2, 1, 0, 0) => {
            (SQRT3
                * x
                * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / r
        }
        (2, 1, 1, -1) => {
            (x * z
                * ((-2.0 + 6.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * y.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&12].0,
                                    &s_or_h[&12].1,
                                    s_or_h[&12].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (2, 1, 1, 0) => {
            (x * y
                * ((2.0 - 3.0 * x.powi(2) - 3.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + r * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + SQRT3
                        * z.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                            + r * splder_uniform(
                                &s_or_h[&12].0,
                                &s_or_h[&12].1,
                                s_or_h[&12].2,
                                r,
                                1,
                            ))))
                / r
        }
        (2, 1, 1, 1) => {
            (y * z
                * ((2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + r * (-x.powi(2) + y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + SQRT3
                        * x.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                            + r * splder_uniform(
                                &s_or_h[&12].0,
                                &s_or_h[&12].1,
                                s_or_h[&12].2,
                                r,
                                1,
                            ))))
                / r
        }
        (2, 1, 2, -2) => {
            (z * ((4.0 * y.powi(4) - z.powi(2) + y.powi(2) * (-3.0 + 4.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (3.0 * x.powi(2) * (-1.0 + 4.0 * y.powi(2))
                    + z.powi(2)
                    + y.powi(2) * (3.0 - 4.0 * y.powi(2) - 4.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, 1, 2, -1) => {
            (x * ((-x.powi(2) + (-3.0 + 4.0 * x.powi(2)) * y.powi(2) + 4.0 * y.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (x.powi(2) + 3.0 * y.powi(2) - 4.0 * x.powi(2) * y.powi(2) - 4.0 * y.powi(4)
                    + 3.0 * (-1.0 + 4.0 * y.powi(2)) * z.powi(2))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, 1, 2, 0) => {
            (SQRT3
                * x
                * y
                * z
                * ((-2.0 + 4.0 * x.powi(2) + 4.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + (4.0 - 8.0 * x.powi(2) - 8.0 * y.powi(2) + 8.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 2.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 1, 2, 1) => {
            (y * (2.0
                * (z.powi(2) + x.powi(2) * (1.0 - 2.0 * y.powi(2) - 2.0 * z.powi(2))
                    - 2.0 * y.powi(2) * (-1.0 + y.powi(2) + z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (-2.0 * x.powi(4) + z.powi(2) - 2.0 * z.powi(2) * (y.powi(2) + z.powi(2))
                        + x.powi(2) * (1.0 - 2.0 * y.powi(2) + 4.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                - 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, 1, 2, 2) => {
            (x * y
                * z
                * (2.0
                    * (-3.0 + 2.0 * x.powi(2) + 6.0 * y.powi(2) + 4.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 4.0
                        * (3.0 + 2.0 * x.powi(2) - 6.0 * y.powi(2) - 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 6.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 12.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 12.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 3.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 2.0
                        * r
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 6.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 3.0
                        * r
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 0, 0) => {
            (SQRT3
                * y
                * (-2.0
                    * (1.0 + x.powi(2) - y.powi(2))
                    * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 1, -1) => {
            (2.0 * (-1.0 + 3.0 * y.powi(2))
                * (2.0 * x.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + SQRT3
                    * (x.powi(2) - 3.0 * (1.0 + x.powi(2)) * y.powi(2) + 3.0 * y.powi(4))
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                + r * y.powi(2)
                    * (-2.0
                        * (2.0 * x.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                        + SQRT3
                            * (x - y)
                            * (x + y)
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 1, 0) => {
            -(y * z
                * ((-4.0 - 6.0 * x.powi(2) + 6.0 * y.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * (x - y)
                        * (x + y)
                        * (2.0
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)
                            - SQRT3
                                * splder_uniform(
                                    &s_or_h[&11].0,
                                    &s_or_h[&11].1,
                                    s_or_h[&11].2,
                                    r,
                                    1,
                                ))))
                / (2. * r)
        }
        (2, 2, 1, 1) => {
            (x * y
                * (-2.0
                    * (-4.0 + 6.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (-2.0 - 3.0 * x.powi(2) + 3.0 * y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + 2.0
                        * r
                        * (2.0 * y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + r * SQRT3
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, -2) => {
            (x * ((x.powi(2) - (3.0 + 4.0 * x.powi(2)) * y.powi(2) + 4.0 * y.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 4.0
                    * (-x.powi(2) + (3.0 + 4.0 * x.powi(2)) * y.powi(2) - 4.0 * y.powi(4))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 9.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * y.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 4.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 4.0
                    * r
                    * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * (x - y)
                    * y.powi(2)
                    * (x + y)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, -1) => {
            (z * ((-4.0 * y.powi(4)
                + x.powi(2) * (3.0 - 12.0 * y.powi(2))
                + 2.0 * z.powi(2)
                + y.powi(2) * (3.0 - 8.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (-8.0 * y.powi(4) + 6.0 * x.powi(2) * (-1.0 + 4.0 * y.powi(2))
                    - 2.0 * z.powi(2)
                    + y.powi(2) * (6.0 + 8.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 9.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * y.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 6.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 2.0
                    * r
                    * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * (x - y)
                    * y.powi(2)
                    * (x + y)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, 0) => {
            (SQRT3
                * y
                * (-4.0
                    * (x.powi(4) + y.powi(2) - y.powi(4)
                        + (1.0 + 2.0 * x.powi(2) - 2.0 * y.powi(2)) * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 8.0
                        * (1.0 + 2.0 * x.powi(2) - 2.0 * y.powi(2))
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 4.0
                        * x.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 4.0
                        * y.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 4.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 8.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * x.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(4)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 4.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 4.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * (x - y)
                        * (x + y)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        (2, 2, 2, 1) => {
            (x * y
                * z
                * (2.0
                    * (-3.0 + 2.0 * x.powi(2) + 6.0 * y.powi(2) + 4.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 4.0
                        * (3.0 + 2.0 * x.powi(2) - 6.0 * y.powi(2) - 2.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 6.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 12.0
                        * x.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 12.0
                        * y.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 3.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - 2.0
                        * r
                        * x.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 6.0
                        * r
                        * y.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 3.0
                        * r
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, 2) => {
            (y * (-4.0
                * (x.powi(4)
                    + (-1.0 + y.powi(2) + 2.0 * z.powi(2)) * (y.powi(2) + 2.0 * z.powi(2))
                    + x.powi(2) * (1.0 - 2.0 * y.powi(2) + 4.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                - 8.0
                    * ((-1.0 + 2.0 * y.powi(2)) * z.powi(2)
                        + 2.0 * x.powi(2) * (-2.0 + 4.0 * y.powi(2) + z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                - 12.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 24.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 4.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 4.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 4.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 16.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 4.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 4.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * (x.powi(2) - y.powi(2)).powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        _ => panic!("No combination of l1, m1, l2, m2 found!"),
    };
    let grad2: f64 = match (l1, m1, l2, m2) {
        (0, 0, 0, 0) => z * splder_uniform(&s_or_h[&0].0, &s_or_h[&0].1, s_or_h[&0].2, r, 1),
        (0, 0, 1, -1) => {
            y * z
                * (-(splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r) / r)
                    + splder_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r, 1))
        }
        (0, 0, 1, 0) => {
            -(((-1.0 + z.powi(2)) * splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r))
                / r)
                + z.powi(2) * splder_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r, 1)
        }
        (0, 0, 1, 1) => {
            x * z
                * (-(splev_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r) / r)
                    + splder_uniform(&s_or_h[&2].0, &s_or_h[&2].1, s_or_h[&2].2, r, 1))
        }
        (0, 0, 2, -2) => {
            (SQRT3
                * x
                * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / r
        }
        (0, 0, 2, -1) => {
            (SQRT3
                * y
                * ((1.0 - 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * z.powi(2)
                        * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / r
        }
        (0, 0, 2, 0) => {
            (z * (2.0
                * (2.0 + x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                - r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                    * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / (2. * r)
        }
        (0, 0, 2, 1) => {
            (SQRT3
                * x
                * ((1.0 - 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * z.powi(2)
                        * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / r
        }
        (0, 0, 2, 2) => {
            (SQRT3
                * (x - y)
                * (x + y)
                * z
                * (-2.0 * splev_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r)
                    + r * splder_uniform(&s_or_h[&3].0, &s_or_h[&3].1, s_or_h[&3].2, r, 1)))
                / (2. * r)
        }
        (1, -1, 0, 0) => {
            y * z
                * (-(splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r) / r)
                    + splder_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r, 1))
        }
        (1, -1, 1, -1) => {
            (z * (-2.0
                * (-1.0 + x.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + r * (x.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                + y.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, -1, 1, 0) => {
            (y * ((-1.0 + 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + z.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, -1, 1, 1) => {
            (x * y
                * z
                * (2.0 * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    - 2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                    + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                        + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, -1, 2, -2) => {
            (x * z
                * ((2.0 - 3.0 * x.powi(2) + 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + r * (x.powi(2) - y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + SQRT3
                        * y.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                            + r * splder_uniform(
                                &s_or_h[&9].0,
                                &s_or_h[&9].1,
                                s_or_h[&9].2,
                                r,
                                1,
                            ))))
                / r
        }
        (1, -1, 2, -1) => {
            ((x.powi(2) - y.powi(2) + 3.0 * (1.0 - x.powi(2) + y.powi(2)) * z.powi(2)
                - 3.0 * z.powi(4))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + r * z.powi(2)
                    * (x.powi(2) - y.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                + SQRT3
                    * y.powi(2)
                    * ((1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                        + r * z.powi(2)
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, -1, 2, 0) => {
            (y * z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + (4.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2) - 6.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    - r * (2.0
                        * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1))))
                / (2. * r)
        }
        (1, -1, 2, 1) => {
            (x * y
                * ((-2.0 + 6.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * z.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (1, -1, 2, 2) => {
            (y * z
                * (2.0
                    * (-2.0 + 6.0 * x.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + 3.0
                        * SQRT3
                        * (-x.powi(2) + y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    - 2.0
                        * r
                        * (2.0 * x.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + r * SQRT3
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / (2. * r)
        }
        (1, 0, 0, 0) => {
            -(((-1.0 + z.powi(2)) * splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r))
                / r)
                + z.powi(2) * splder_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r, 1)
        }
        (1, 0, 1, -1) => {
            (y * ((-1.0 + 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + z.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, 0, 1, 0) => {
            (z * (-2.0
                * (-1.0 + z.powi(2))
                * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                - (x.powi(2) + y.powi(2))
                    * (2.0 * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                        - r * splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1))
                + r * z.powi(2) * splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))
                / r
        }
        (1, 0, 1, 1) => {
            (x * ((-1.0 + 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + z.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, 0, 2, -2) => {
            (x * y
                * ((-2.0 + 6.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * z.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (1, 0, 2, -1) => {
            (y * z
                * (-((2.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r))
                    + SQRT3
                        * (2.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + r * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, 0, 2, 0) => {
            (-2.0
                * SQRT3
                * (x.powi(2) + y.powi(2))
                * (-1.0 + 3.0 * z.powi(2))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + (-x.powi(2) - y.powi(2) + 3.0 * (2.0 + x.powi(2) + y.powi(2)) * z.powi(2)
                    - 6.0 * z.powi(4))
                    * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                + r * (x.powi(2) + y.powi(2))
                    * z.powi(2)
                    * (2.0
                        * SQRT3
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                        - splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1))
                + 2.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1))
                / (2. * r)
        }
        (1, 0, 2, 1) => {
            (x * z
                * (-((2.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r))
                    + SQRT3
                        * (2.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + r * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, 0, 2, 2) => {
            ((x - y)
                * (x + y)
                * ((-2.0 + 6.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * z.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / (2. * r)
        }
        (1, 1, 0, 0) => {
            x * z
                * (-(splev_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r) / r)
                    + splder_uniform(&s_or_h[&4].0, &s_or_h[&4].1, s_or_h[&4].2, r, 1))
        }
        (1, 1, 1, -1) => {
            (x * y
                * z
                * (2.0 * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                    - 2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                    + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                        + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, 1, 1, 0) => {
            (x * ((-1.0 + 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                + z.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * (-splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                            + splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1)))))
                / r
        }
        (1, 1, 1, 1) => {
            (z * (-2.0
                * (-1.0 + y.powi(2) + z.powi(2))
                * splev_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r)
                + r * (y.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&5].0, &s_or_h[&5].1, s_or_h[&5].2, r, 1)
                + x.powi(2)
                    * (-2.0 * splev_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r)
                        + r * splder_uniform(&s_or_h[&6].0, &s_or_h[&6].1, s_or_h[&6].2, r, 1))))
                / r
        }
        (1, 1, 2, -2) => {
            (y * z
                * ((2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + r * (-x.powi(2) + y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + SQRT3
                        * x.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                            + r * splder_uniform(
                                &s_or_h[&9].0,
                                &s_or_h[&9].1,
                                s_or_h[&9].2,
                                r,
                                1,
                            ))))
                / r
        }
        (1, 1, 2, -1) => {
            (x * y
                * ((-2.0 + 6.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + r * z.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&9].0,
                                    &s_or_h[&9].1,
                                    s_or_h[&9].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (1, 1, 2, 0) => {
            (x * z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + (4.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2) - 6.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    - r * (2.0
                        * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1))))
                / (2. * r)
        }
        (1, 1, 2, 1) => {
            ((-x.powi(2) + y.powi(2) + 3.0 * (1.0 + x.powi(2) - y.powi(2)) * z.powi(2)
                - 3.0 * z.powi(4))
                * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                + r * z.powi(2)
                    * (-x.powi(2) + y.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                + SQRT3
                    * x.powi(2)
                    * ((1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                        + r * z.powi(2)
                            * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / r
        }
        (1, 1, 2, 2) => {
            (x * z
                * (-2.0
                    * (-2.0 + 6.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r)
                    + 3.0
                        * SQRT3
                        * (-x.powi(2) + y.powi(2))
                        * splev_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r)
                    + 2.0
                        * r
                        * (2.0 * y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&7].0, &s_or_h[&7].1, s_or_h[&7].2, r, 1)
                    + r * SQRT3
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&8].0, &s_or_h[&8].1, s_or_h[&8].2, r, 1)))
                / (2. * r)
        }
        (2, -2, 0, 0) => {
            (SQRT3
                * x
                * y
                * z
                * (-2.0 * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / r
        }
        (2, -2, 1, -1) => {
            (x * z
                * ((2.0 - 3.0 * x.powi(2) + 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + r * (x.powi(2) - y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + SQRT3
                        * y.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                            + r * splder_uniform(
                                &s_or_h[&12].0,
                                &s_or_h[&12].1,
                                s_or_h[&12].2,
                                r,
                                1,
                            ))))
                / r
        }
        (2, -2, 1, 0) => {
            (x * y
                * ((-2.0 + 6.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * z.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&12].0,
                                    &s_or_h[&12].1,
                                    s_or_h[&12].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (2, -2, 1, 1) => {
            (y * z
                * ((2.0 + 3.0 * x.powi(2) - 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + r * (-x.powi(2) + y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + SQRT3
                        * x.powi(2)
                        * (-3.0 * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                            + r * splder_uniform(
                                &s_or_h[&12].0,
                                &s_or_h[&12].1,
                                s_or_h[&12].2,
                                r,
                                1,
                            ))))
                / r
        }
        (2, -2, 2, -2) => {
            (z * (2.0
                * (x.powi(2) + y.powi(2)
                    - 2.0 * x.powi(2) * y.powi(2)
                    - 2.0 * (-1.0 + x.powi(2) + y.powi(2)) * z.powi(2)
                    - 2.0 * z.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (x.powi(2) - 2.0 * x.powi(4) + y.powi(2) + 4.0 * x.powi(2) * y.powi(2)
                        - 2.0 * y.powi(4)
                        - 2.0 * (x.powi(2) + y.powi(2)) * z.powi(2))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                - 12.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -2, 2, -1) => {
            (x * ((-x.powi(2) + (-3.0 + 4.0 * x.powi(2)) * z.powi(2) + 4.0 * z.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (x.powi(2) - 3.0 * y.powi(2)
                    + (3.0 - 4.0 * x.powi(2) + 12.0 * y.powi(2)) * z.powi(2)
                    - 4.0 * z.powi(4))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -2, 2, 0) => {
            (SQRT3
                * x
                * y
                * z
                * (-4.0
                    * (-1.0 + x.powi(2) + y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 8.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 4.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + (x.powi(2) + y.powi(2))
                        * (4.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                            + r * splder_uniform(
                                &s_or_h[&13].0,
                                &s_or_h[&13].1,
                                s_or_h[&13].2,
                                r,
                                1,
                            ))
                    + 2.0
                        * z.powi(2)
                        * (8.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                            - 4.0
                                * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                            + r * (splder_uniform(
                                &s_or_h[&13].0,
                                &s_or_h[&13].1,
                                s_or_h[&13].2,
                                r,
                                1,
                            ) - 2.0
                                * splder_uniform(
                                    &s_or_h[&14].0,
                                    &s_or_h[&14].1,
                                    s_or_h[&14].2,
                                    r,
                                    1,
                                )))
                    - r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, -2, 2, 1) => {
            (y * ((-y.powi(2) + (-3.0 + 4.0 * y.powi(2)) * z.powi(2) + 4.0 * z.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (-3.0 * x.powi(2)
                    + y.powi(2)
                    + (3.0 + 12.0 * x.powi(2) - 4.0 * y.powi(2)) * z.powi(2)
                    - 4.0 * z.powi(4))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -2, 2, 2) => {
            (x * (x - y)
                * y
                * (x + y)
                * z
                * (-4.0 * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 16.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 12.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * (splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                        - 4.0
                            * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                        + 3.0
                            * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1))))
                / (2. * r)
        }
        (2, -1, 0, 0) => {
            (SQRT3
                * y
                * ((1.0 - 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * z.powi(2)
                        * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / r
        }
        (2, -1, 1, -1) => {
            ((x.powi(2) - y.powi(2) + 3.0 * (1.0 - x.powi(2) + y.powi(2)) * z.powi(2)
                - 3.0 * z.powi(4))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + r * z.powi(2)
                    * (x.powi(2) - y.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                + SQRT3
                    * y.powi(2)
                    * ((1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                        + r * z.powi(2)
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, -1, 1, 0) => {
            (y * z
                * (-((2.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r))
                    + SQRT3
                        * (2.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + r * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, -1, 1, 1) => {
            (x * y
                * ((-2.0 + 6.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * z.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&12].0,
                                    &s_or_h[&12].1,
                                    s_or_h[&12].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (2, -1, 2, -2) => {
            (x * ((-x.powi(2) + (-3.0 + 4.0 * x.powi(2)) * z.powi(2) + 4.0 * z.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (x.powi(2) - 3.0 * y.powi(2)
                    + (3.0 - 4.0 * x.powi(2) + 12.0 * y.powi(2)) * z.powi(2)
                    - 4.0 * z.powi(4))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -1, 2, -1) => {
            (z * (-2.0
                * (x.powi(2) + y.powi(2))
                * (-1.0 + 2.0 * x.powi(2) + 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (x.powi(2) * (1.0 - 2.0 * y.powi(2) - 2.0 * z.powi(2))
                        - 2.0 * (y - z) * (y + z) * (1.0 + y.powi(2) - z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 6.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, -1, 2, 0) => {
            (SQRT3
                * y
                * ((x.powi(2) + y.powi(2))
                    * (-1.0 + 4.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 2.0
                        * (x.powi(2) + y.powi(2)
                            - (3.0 + 4.0 * x.powi(2) + 4.0 * y.powi(2)) * z.powi(2)
                            + 4.0 * z.powi(4))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - x.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - y.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 6.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * z.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(4)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * z.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, -1, 2, 1) => {
            (x * y
                * z
                * (4.0
                    * (x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 2.0
                        * (3.0 + 2.0 * x.powi(2) + 2.0 * y.powi(2) - 6.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 6.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * (x.powi(2) + y.powi(2))
                        * (-splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                            + splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1))
                    - 3.0
                        * z.powi(2)
                        * (4.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                            + r * (splder_uniform(
                                &s_or_h[&14].0,
                                &s_or_h[&14].1,
                                s_or_h[&14].2,
                                r,
                                1,
                            ) - splder_uniform(
                                &s_or_h[&15].0,
                                &s_or_h[&15].1,
                                s_or_h[&15].2,
                                r,
                                1,
                            )))))
                / r
        }
        (2, -1, 2, 2) => {
            (y * ((3.0 * x.powi(2) + y.powi(2)
                - 2.0 * (-3.0 + 6.0 * x.powi(2) + 2.0 * y.powi(2)) * z.powi(2)
                - 8.0 * z.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (-3.0 * x.powi(2)
                        + y.powi(2)
                        + (-3.0 + 12.0 * x.powi(2) - 4.0 * y.powi(2)) * z.powi(2)
                        + 4.0 * z.powi(4))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 6.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 2.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * (x - y)
                    * (x + y)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 0, 0) => {
            (z * (2.0
                * (2.0 + x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                - r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                    * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 1, -1) => {
            (y * z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + (4.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2) - 6.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    - r * (2.0
                        * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1))))
                / (2. * r)
        }
        (2, 0, 1, 0) => {
            (-2.0
                * SQRT3
                * (x.powi(2) + y.powi(2))
                * (-1.0 + 3.0 * z.powi(2))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + (-x.powi(2) - y.powi(2) + 3.0 * (2.0 + x.powi(2) + y.powi(2)) * z.powi(2)
                    - 6.0 * z.powi(4))
                    * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                + r * (x.powi(2) + y.powi(2))
                    * z.powi(2)
                    * (2.0
                        * SQRT3
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                        - splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1))
                + 2.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1))
                / (2. * r)
        }
        (2, 0, 1, 1) => {
            (x * z
                * (2.0
                    * SQRT3
                    * (-2.0 + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + (4.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2) - 6.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    - r * (2.0
                        * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1))))
                / (2. * r)
        }
        (2, 0, 2, -2) => {
            (SQRT3
                * x
                * y
                * z
                * (-4.0
                    * (-1.0 + x.powi(2) + y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 8.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 4.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + (x.powi(2) + y.powi(2))
                        * (4.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                            + r * splder_uniform(
                                &s_or_h[&13].0,
                                &s_or_h[&13].1,
                                s_or_h[&13].2,
                                r,
                                1,
                            ))
                    + 2.0
                        * z.powi(2)
                        * (8.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                            - 4.0
                                * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                            + r * (splder_uniform(
                                &s_or_h[&13].0,
                                &s_or_h[&13].1,
                                s_or_h[&13].2,
                                r,
                                1,
                            ) - 2.0
                                * splder_uniform(
                                    &s_or_h[&14].0,
                                    &s_or_h[&14].1,
                                    s_or_h[&14].2,
                                    r,
                                    1,
                                )))
                    - r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, -1) => {
            (SQRT3
                * y
                * ((x.powi(2) + y.powi(2))
                    * (-1.0 + 4.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 2.0
                        * (x.powi(2) + y.powi(2)
                            - (3.0 + 4.0 * x.powi(2) + 4.0 * y.powi(2)) * z.powi(2)
                            + 4.0 * z.powi(4))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - x.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - y.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 6.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * z.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(4)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * z.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, 0) => {
            (z * (-8.0
                * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 4.0
                    * (3.0
                        * (x.powi(2) + y.powi(2))
                        * ((x.powi(2) + y.powi(2))
                            * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                            + 4.0
                                * z.powi(2)
                                * splev_uniform(
                                    &s_or_h[&13].0,
                                    &s_or_h[&13].1,
                                    s_or_h[&13].2,
                                    r,
                                ))
                        + (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)).powi(2)
                            * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r))
                + 3.0
                    * (x.powi(2) + y.powi(2))
                    * (8.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                        + r * (x.powi(2) + y.powi(2))
                            * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                        + 4.0
                            * r
                            * z.powi(2)
                            * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1))
                + r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2)).powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        (2, 0, 2, 1) => {
            (SQRT3
                * x
                * ((x.powi(2) + y.powi(2))
                    * (-1.0 + 4.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 2.0
                        * (x.powi(2) + y.powi(2)
                            - (3.0 + 4.0 * x.powi(2) + 4.0 * y.powi(2)) * z.powi(2)
                            + 4.0 * z.powi(4))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - x.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - y.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 6.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * z.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(4)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * z.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 0, 2, 2) => {
            (SQRT3
                * (x - y)
                * (x + y)
                * z
                * (-4.0
                    * (-1.0 + x.powi(2) + y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 8.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 4.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + (x.powi(2) + y.powi(2))
                        * (4.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                            + r * splder_uniform(
                                &s_or_h[&13].0,
                                &s_or_h[&13].1,
                                s_or_h[&13].2,
                                r,
                                1,
                            ))
                    + 2.0
                        * z.powi(2)
                        * (8.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                            - 4.0
                                * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                            + r * (splder_uniform(
                                &s_or_h[&13].0,
                                &s_or_h[&13].1,
                                s_or_h[&13].2,
                                r,
                                1,
                            ) - 2.0
                                * splder_uniform(
                                    &s_or_h[&14].0,
                                    &s_or_h[&14].1,
                                    s_or_h[&14].2,
                                    r,
                                    1,
                                )))
                    - r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        (2, 1, 0, 0) => {
            (SQRT3
                * x
                * ((1.0 - 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * z.powi(2)
                        * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / r
        }
        (2, 1, 1, -1) => {
            (x * y
                * ((-2.0 + 6.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * z.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&12].0,
                                    &s_or_h[&12].1,
                                    s_or_h[&12].2,
                                    r,
                                    1,
                                ))))
                / r
        }
        (2, 1, 1, 0) => {
            (x * z
                * (-((2.0 + 3.0 * x.powi(2) + 3.0 * y.powi(2) - 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r))
                    + SQRT3
                        * (2.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * (x.powi(2) + y.powi(2) - z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + r * SQRT3
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, 1, 1, 1) => {
            ((-x.powi(2) + y.powi(2) + 3.0 * (1.0 + x.powi(2) - y.powi(2)) * z.powi(2)
                - 3.0 * z.powi(4))
                * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                + r * z.powi(2)
                    * (-x.powi(2) + y.powi(2) + z.powi(2))
                    * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                + SQRT3
                    * x.powi(2)
                    * ((1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                        + r * z.powi(2)
                            * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / r
        }
        (2, 1, 2, -2) => {
            (y * ((-y.powi(2) + (-3.0 + 4.0 * y.powi(2)) * z.powi(2) + 4.0 * z.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + (-3.0 * x.powi(2)
                    + y.powi(2)
                    + (3.0 + 12.0 * x.powi(2) - 4.0 * y.powi(2)) * z.powi(2)
                    - 4.0 * z.powi(4))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - r * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, 1, 2, -1) => {
            (x * y
                * z
                * (4.0
                    * (x.powi(2) + y.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 2.0
                        * (3.0 + 2.0 * x.powi(2) + 2.0 * y.powi(2) - 6.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 6.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * (x.powi(2) + y.powi(2))
                        * (-splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                            + splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1))
                    - 3.0
                        * z.powi(2)
                        * (4.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                            + r * (splder_uniform(
                                &s_or_h[&14].0,
                                &s_or_h[&14].1,
                                s_or_h[&14].2,
                                r,
                                1,
                            ) - splder_uniform(
                                &s_or_h[&15].0,
                                &s_or_h[&15].1,
                                s_or_h[&15].2,
                                r,
                                1,
                            )))))
                / r
        }
        (2, 1, 2, 0) => {
            (SQRT3
                * x
                * ((x.powi(2) + y.powi(2))
                    * (-1.0 + 4.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 2.0
                        * (x.powi(2) + y.powi(2)
                            - (3.0 + 4.0 * x.powi(2) + 4.0 * y.powi(2)) * z.powi(2)
                            + 4.0 * z.powi(4))
                        * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - x.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - y.powi(2) * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 6.0
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * x.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + 4.0
                        * y.powi(2)
                        * z.powi(2)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - 8.0
                        * z.powi(4)
                        * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    - r * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    - r * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                    + 2.0
                        * r
                        * x.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    + 2.0
                        * r
                        * y.powi(2)
                        * z.powi(2)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - 2.0
                        * r
                        * z.powi(4)
                        * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                    - r * z.powi(2)
                        * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 1, 2, 1) => {
            (z * (-2.0
                * (x.powi(2) + y.powi(2))
                * (-1.0 + 2.0 * y.powi(2) + 2.0 * z.powi(2))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (y.powi(2) + 2.0 * z.powi(2)
                        - 2.0
                            * (x.powi(4)
                                + x.powi(2) * (1.0 + y.powi(2) - 2.0 * z.powi(2))
                                + z.powi(2) * (y.powi(2) + z.powi(2))))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 6.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + r * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / r
        }
        (2, 1, 2, 2) => {
            -(x * ((x.powi(2) + 3.0 * y.powi(2)
                - 2.0 * (-3.0 + 2.0 * x.powi(2) + 6.0 * y.powi(2)) * z.powi(2)
                - 8.0 * z.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (x.powi(2) - 3.0 * y.powi(2)
                        + (-3.0 - 4.0 * x.powi(2) + 12.0 * y.powi(2)) * z.powi(2)
                        + 4.0 * z.powi(4))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                - 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 6.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * (x - y)
                    * (x + y)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 0, 0) => {
            (SQRT3
                * (x - y)
                * (x + y)
                * z
                * (-2.0 * splev_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r)
                    + r * splder_uniform(&s_or_h[&9].0, &s_or_h[&9].1, s_or_h[&9].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 1, -1) => {
            (y * z
                * (2.0
                    * (-2.0 + 6.0 * x.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + 3.0
                        * SQRT3
                        * (-x.powi(2) + y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    - 2.0
                        * r
                        * (2.0 * x.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + r * SQRT3
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 1, 0) => {
            ((x - y)
                * (x + y)
                * ((-2.0 + 6.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + SQRT3
                        * (1.0 - 3.0 * z.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + r * z.powi(2)
                        * (-2.0
                            * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                            + SQRT3
                                * splder_uniform(
                                    &s_or_h[&12].0,
                                    &s_or_h[&12].1,
                                    s_or_h[&12].2,
                                    r,
                                    1,
                                ))))
                / (2. * r)
        }
        (2, 2, 1, 1) => {
            (x * z
                * (-2.0
                    * (-2.0 + 6.0 * y.powi(2) + 3.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r)
                    + 3.0
                        * SQRT3
                        * (-x.powi(2) + y.powi(2))
                        * splev_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r)
                    + 2.0
                        * r
                        * (2.0 * y.powi(2) + z.powi(2))
                        * splder_uniform(&s_or_h[&10].0, &s_or_h[&10].1, s_or_h[&10].2, r, 1)
                    + r * SQRT3
                        * (x - y)
                        * (x + y)
                        * splder_uniform(&s_or_h[&11].0, &s_or_h[&11].1, s_or_h[&11].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, -2) => {
            (x * (x - y)
                * y
                * (x + y)
                * z
                * (-4.0 * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    + 16.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    - 12.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + r * (splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                        - 4.0
                            * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                        + 3.0
                            * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1))))
                / (2. * r)
        }
        (2, 2, 2, -1) => {
            (y * ((3.0 * x.powi(2) + y.powi(2)
                - 2.0 * (-3.0 + 6.0 * x.powi(2) + 2.0 * y.powi(2)) * z.powi(2)
                - 8.0 * z.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (-3.0 * x.powi(2)
                        + y.powi(2)
                        + (-3.0 + 12.0 * x.powi(2) - 4.0 * y.powi(2)) * z.powi(2)
                        + 4.0 * z.powi(4))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                + 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 3.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 6.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 2.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * (x - y)
                    * (x + y)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, 0) => {
            (SQRT3
                * (x - y)
                * (x + y)
                * z
                * (-4.0
                    * (-1.0 + x.powi(2) + y.powi(2) + 2.0 * z.powi(2))
                    * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                    - 8.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                    + 4.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                    + (x.powi(2) + y.powi(2))
                        * (4.0 * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                            + r * splder_uniform(
                                &s_or_h[&13].0,
                                &s_or_h[&13].1,
                                s_or_h[&13].2,
                                r,
                                1,
                            ))
                    + 2.0
                        * z.powi(2)
                        * (8.0 * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                            - 4.0
                                * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                            + r * (splder_uniform(
                                &s_or_h[&13].0,
                                &s_or_h[&13].1,
                                s_or_h[&13].2,
                                r,
                                1,
                            ) - 2.0
                                * splder_uniform(
                                    &s_or_h[&14].0,
                                    &s_or_h[&14].1,
                                    s_or_h[&14].2,
                                    r,
                                    1,
                                )))
                    - r * (x.powi(2) + y.powi(2) - 2.0 * z.powi(2))
                        * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        (2, 2, 2, 1) => {
            -(x * ((x.powi(2) + 3.0 * y.powi(2)
                - 2.0 * (-3.0 + 2.0 * x.powi(2) + 6.0 * y.powi(2)) * z.powi(2)
                - 8.0 * z.powi(4))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                + 2.0
                    * (x.powi(2) - 3.0 * y.powi(2)
                        + (-3.0 - 4.0 * x.powi(2) + 12.0 * y.powi(2)) * z.powi(2)
                        + 4.0 * z.powi(4))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                - 3.0
                    * x.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 3.0
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 12.0
                    * x.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(2)
                    * z.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 3.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 2.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 6.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 2.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                - 3.0
                    * r
                    * (x - y)
                    * (x + y)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (2. * r)
        }
        (2, 2, 2, 2) => {
            (z * (-4.0
                * (x.powi(4) - 2.0 * x.powi(2) * (1.0 + y.powi(2) - 2.0 * z.powi(2))
                    + (-2.0 + y.powi(2) + 2.0 * z.powi(2)) * (y.powi(2) + 2.0 * z.powi(2)))
                * splev_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r)
                - 8.0
                    * (y.powi(2) * (-1.0 + 2.0 * z.powi(2))
                        + x.powi(2) * (-1.0 + 8.0 * y.powi(2) + 2.0 * z.powi(2)))
                    * splev_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r)
                - 12.0
                    * x.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + 24.0
                    * x.powi(2)
                    * y.powi(2)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                - 12.0
                    * y.powi(4)
                    * splev_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r)
                + r * x.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                - 2.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + r * y.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 4.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 4.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 4.0
                    * r
                    * z.powi(4)
                    * splder_uniform(&s_or_h[&12].0, &s_or_h[&12].1, s_or_h[&12].2, r, 1)
                + 16.0
                    * r
                    * x.powi(2)
                    * y.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 4.0
                    * r
                    * x.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 4.0
                    * r
                    * y.powi(2)
                    * z.powi(2)
                    * splder_uniform(&s_or_h[&13].0, &s_or_h[&13].1, s_or_h[&13].2, r, 1)
                + 3.0
                    * r
                    * (x.powi(2) - y.powi(2)).powi(2)
                    * splder_uniform(&s_or_h[&14].0, &s_or_h[&14].1, s_or_h[&14].2, r, 1)))
                / (4. * r)
        }
        _ => panic!("No combination of l1, m1, l2, m2 found!"),
    };
    return array![grad0, grad1, grad2];
}
