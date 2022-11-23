/// In the original DFTBaby code the Fermi energy is searched using the bisection
/// method, as it is necessary to find the root of (sum_a fa - n_elec).
/// Since the bisection method can be slow, we use the Brent's method.
/// Using Brent's method, find the root of a function known to lie between `x1 ` and
/// `x2`. The root will be refined until its accuracy is `tol`
///
/// The code is based on:
/// Numerical Recipes in C: The Art of Scientific Computing. W. H. Press,
/// S. A. Teukolsky, W. T. Vetterling, B. P. Flannery. Cambridge University Press 1992
pub fn zbrent<F: Fn(f64) -> f64>(func: F, x1: f64, x2: f64, tol: f64, maxiter: usize) -> f64 {
    let eps: f64 = 2.220446049250313e-016_f64.sqrt();
    let mut a: f64 = x1;
    let mut b: f64 = x2;
    let mut c: f64 = x2;
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut min1: f64;
    let mut min2: f64;

    let mut root: f64 = 0.0;

    let mut fa: f64 = func(a);
    let mut fb: f64 = func(b);
    let mut fc: f64;
    let mut p: f64;
    let mut q: f64;
    let mut r: f64;
    let mut s: f64;
    let mut tol1: f64;
    let mut xm: f64;

    assert!(
        (fa > 0.0 && fb < 0.0) || (fa < 0.0 && fb > 0.0),
        "Root must be bracketed in zbrent"
    );
    fc = fb;
    'main_loop: for _iter in 0..maxiter {
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            // rename a, b, c and adjust the bounding interval d
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        // convergence check
        tol1 = 2.0 * eps * b.abs() + 0.5 * tol;
        xm = 0.5 * (c - b);

        if xm.abs() <= tol1 || fb == 0.0 {
            root = b;
            break 'main_loop;
        }

        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            // attempt inverse quadratic interpolation
            s = fb / fa;
            if (a - c).abs() < eps {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                // check whether in bounds
                q = -q;
            }
            p = p.abs();
            min1 = 3.0 * xm * q - (tol1 * q).abs();
            min2 = (e * q).abs();
            if (2.0 * p) < min1.min(min2) {
                // accept interpolation
                e = d;
                d = p / q;
            } else {
                // interpolation failed, use bisection.
                d = xm;
                e = d;
            }
        } else {
            // bounds decreasing to slowly, use bisection.
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        if d.abs() > tol1 {
            b = b + d;
        } else if xm > 0.0 {
            b = b + tol1;
        } else {
            b = b - tol1;
        }
        fb = func(b);
    }
    return root;
}

#[cfg(test)]
mod tests {
    use super::zbrent;
    /// The tests are taken from John Burkardts Python version of the Brent's method
    /// See https://people.sc.fsu.edu/~jburkardt/py_src/brent/zero.py
    #[test]
    fn zero_test() {
        let eps: f64 = 2.220446049250313e-016;
        let maxiter: usize = 100;
        let t: f64 = 10.0 * (eps).sqrt();

        // F_01 evaluates sin ( x ) - x / 2.
        fn f_01(x: f64) -> f64 {
            (x).sin() - 0.5 * x
        }
        let a: f64 = 1.0;
        let b: f64 = 2.0;
        let x: f64 = zbrent(f_01, a, b, t, maxiter);
        assert!(f_01(x) <= t);

        // F_02 evaluates 2*x-exp(-x).
        fn f_02(x: f64) -> f64 {
            2.0 * x - (-x).exp()
        }
        let a: f64 = 0.0;
        let b: f64 = 1.0;
        let x: f64 = zbrent(f_02, a, b, t, maxiter);
        assert!(f_02(x) <= t);

        // F_03 evaluates x*exp(-x).
        fn f_03(x: f64) -> f64 {
            x * (-x).exp()
        }
        let a: f64 = -1.0;
        let b: f64 = 0.5;
        let x: f64 = zbrent(f_03, a, b, t, maxiter);
        assert!(f_03(x) <= t);

        // F_04 evaluates exp(x) - 1 / (100*x*x).
        fn f_04(x: f64) -> f64 {
            (x).exp() - 1.0 / 100.0 / x / x
        }
        let a: f64 = 0.0001;
        let b: f64 = 20.0;
        let x: f64 = zbrent(f_04, a, b, t, maxiter);
        assert!(f_04(x) <= t);

        // F_05 evaluates (x+3)*(x-1)*(x-1).
        fn f_05(x: f64) -> f64 {
            (x + 3.0) * (x - 1.0) * (x - 1.0)
        }
        let a: f64 = -5.0;
        let b: f64 = 2.0;
        let x: f64 = zbrent(f_05, a, b, t, maxiter);
        assert!(f_05(x) <= t);
    }
}
