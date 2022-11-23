use anyhow::{Context, Result};
use derive_builder::Builder;
use ndarray::prelude::*;
use ndarray_linalg::{Norm, Solve};
use serde_repr::{Deserialize_repr, Serialize_repr};
use std::cmp::Ordering;
use std::ops::AddAssign;

pub const REGULARIZATION_TYPE_I: f64 = 1e-8;
pub const REGULARIZATION_TYPE_II: f64 = 1e-10;

#[derive(Serialize_repr, Deserialize_repr, Copy, Clone, Debug, PartialEq)]
#[repr(u8)]
pub enum AAType {
    I = 1,
    II = 2,
}

/// Anderson mixing scheme may be reduced to vanilla KM mixing by setting the `memory` to zero.
#[derive(Builder, Clone, Debug)]
pub struct AndersonAccel {
    /// Dimension of the vector.
    #[allow(dead_code)]
    pub dim: usize,
    /// Number of vectors to store.
    memory: usize,
    /// Current iteration.
    #[builder(default = "0", setter(skip))]
    pub iter: usize,
    /// Type of Anderson Accelerator (can be type-I or type-II).
    aa_type: AAType,
    /// Mixing parameter β: |x_i+1 > = |x_i > + β * | F_i >
    #[builder(default = "1.0")]
    beta: f64,
    /// Regularization param, type-I: 1e-8 works well, type-II: more stable can use 1e-10 often
    #[builder(default = "self.default_regularization()?")]
    regularization: f64,
    /// Safeguard tolerance factor.
    #[builder(default = "1.0")]
    safeguard_factor: f64,
    /// Maximum norm of AA weights.
    #[builder(default = "1e8")]
    max_weight_norm: f64,
    /// Norm of AA weights.
    #[builder(default = "0.0", setter(skip))]
    norm: f64,
    /// || |x_i> - |F_i> ||_2
    #[builder(default = "0.0", setter(skip))]
    norm_g: f64,
    /// Input vector of last iteration: | x_(i-1) >
    #[builder(default = "None", setter(skip))]
    pub old_x: Option<Array1<f64>>,
    /// Output vector of last iteration: | F_(i-1) >
    #[builder(default = "None", setter(skip))]
    old_f: Option<Array1<f64>>,
    /// Matrix of stacked |Δx_i> = |x_(i+1)> - |x_i>
    #[builder(default = "self.default_array()?", setter(skip))]
    s: Array2<f64>,
    /// Matrix of stacked |ΔF_i> = |F_(i+1)> - |F_i>
    #[builder(default = "self.default_array()?", setter(skip))]
    d: Array2<f64>,
}

impl AndersonAccelBuilder {
    // Private helper method to initialize the empty arrays.
    fn default_array(&self) -> Result<Array2<f64>, String> {
        match self.dim {
            Some(d) => Ok(Array2::zeros((0, d))),
            _ => Err("Dimension has to be initialized".to_string()),
        }
    }

    // Private helper method to initialize a default regularization factor.
    fn default_regularization(&self) -> Result<f64, String> {
        if let Some(atype) = self.aa_type {
            match atype {
                AAType::I => Ok(REGULARIZATION_TYPE_I),
                AAType::II => Ok(REGULARIZATION_TYPE_II),
            }
        } else {
            Err("Anderson type has to be initialized".to_string())
        }
    }
}

impl AndersonAccel {
    /// The reset simply sets `self.iter = 0` and clears the `s` and `d` arrays.
    /// Note: In the first iteration after the restart `self.iter` is 1, as it was incremented
    /// at the end of the `apply` function.
    pub fn reset(&mut self) {
        self.iter = 0;
        self.s = Array2::zeros((0, self.s.dim().1));
        self.d = Array2::zeros((0, self.d.dim().1));
    }

    /// Check if the residual norm is sufficiently small, if not replace the AA update with
    /// a β-averaged vanilla Kranosel'skii-Mann iteration step.
    fn safeguard(&mut self, x: ArrayView1<f64>, g: ArrayView1<f64>) -> Option<Array1<f64>> {
        // Compute the L2 norm of the residual vector |g_i >.
        let norm_g1 = g.norm_l2();

        // Safeguarding check starts after the first AA iteration (self.iter == 1), as the KM
        // step in the first iteration might lead to an increase of the norm in non local convergent
        // cases.
        if norm_g1 > self.safeguard_factor * self.norm_g && self.iter > 1 {
            // In this case the previous AA step is rejected and a reset is done.
            self.reset();
            // Fall back to vanilla iteration.
            return Some(self.vanilla(x.view(), g.view()));
        }
        self.norm_g = norm_g1;
        None
    }

    /// Linear (vanilla) Kranosel'skii-Mann iteration algorithm.
    /// It updates |x_i+1> in iteration i + 1 according to
    /// |x_i+1 > = (1-β) |x_i > + β * | y_i > = |x_i > - β * |g_i >
    pub fn vanilla(&mut self, x: ArrayView1<f64>, g: ArrayView1<f64>) -> Array1<f64> {
        &x - self.beta * &g
    }

    pub fn apply(&mut self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<Array1<f64>> {
        // Compute the residual between the input vector |x_i > and the output vector | y_i >.
        let g = &x - &y;

        // In case of no memory the linear Kranosel'skii-Mann iteration algorithm is used.
        if self.memory == 0 {
            let result = self.vanilla(x.view(), g.view());
            return Ok(result);
        }

        // In the first iteration the output vector is returned and safeguarding will be skipped.
        // |x_i+1 > = | y_i >
        if self.iter == 0 {
            self.iter += 1;
            // The current input and output vectors are saved for the next iteration.
            self.old_x = Some(x.to_owned());
            self.old_f = Some(y.to_owned());
            self.norm_g = g.norm_l2();
            return Ok(y.to_owned());
        }

        // Check if residual norm has decreased and fall back to vanilla KM if not.
        if let Some(y) = self.safeguard(x.view(), g.view()) {
            return Ok(y);
        };

        // Acceleration parameters are updated.
        self.update_accel_params(x.view(), y.view())?;

        // Anderson Acceleration is done at this step.
        let y_new = self.solve(x.view(), y.view(), g.view());

        // The current input and output vectors are saved for the next iteration.
        self.old_x = Some(x.to_owned());
        self.old_f = Some(y.to_owned());

        self.iter += 1;
        Ok(y_new)
    }

    /// Anderson Acceleration update.
    /// The system of equations is solved to perform the update of | F_i >.
    pub fn solve(
        &mut self,
        x: ArrayView1<f64>,
        f: ArrayView1<f64>,
        g: ArrayView1<f64>,
    ) -> Array1<f64> {
        let y = self.compute_y();

        // S' g (Type 1) or Y' g (Type 2) is computed.
        let work = match self.aa_type {
            AAType::I => self.s.dot(&g),
            AAType::II => y.dot(&g),
        };

        // S' Y (Type 1) or Y' Y (Type 2) is computed and regularized.
        let m = self.compute_m(y.view());

        // Try to solve the system of equations.
        let gamma = m.solve_into(work);

        // Check if solve was successful, if not fall back to vanilla.
        let gamma = match gamma {
            Ok(value) => value,
            Err(_) => {
                // If matrix is singular a vanilla KM iteration and a reset is done.
                self.reset();
                return self.vanilla(x.view(), g.view());
            }
        };

        self.norm = gamma.norm_l2();

        // If the L2 norm of the matrix is larger than the threshold fall back to KM iteration and
        // reset.
        if self.norm >= self.max_weight_norm {
            self.reset();
            return self.vanilla(x.view(), g.view());
        }

        // Compute: | x_(i+1) > = | F_i > - ∑_j=0^m γ_j (| F_(j+1) > - | F_j >)
        let f = &f - self.d.t().dot(&gamma);

        // If β ≠ 1 the new | x_(i+1) > is mixed with the old ones weighted by γ.
        match (self.beta - 1.0).abs().partial_cmp(&f64::EPSILON) {
            Some(Ordering::Less) => f,
            _ => self.relax(f.view(), x.view(), gamma.view()),
        }
    }

    /// Add acceleration parameters, in particular x and f.
    /// s = | Δx_i > = | x_i > - | x_(i-1)> and
    /// d = | ΔF_i > = | F_i > - | F_(i-1)>
    fn update_accel_params(&mut self, x: ArrayView1<f64>, f: ArrayView1<f64>) -> Result<()> {
        // | Δx_i > = | x_i > - | x_(i-1)>
        let dx = &x
            - self.old_x.as_ref().context(
                "Old x not available. update_accel_params expects\
        that the input vector was set in the previous iteration.",
            )?;

        // | ΔF_i > = | F_i > - | F_(i-1)>
        let df = &f
            - self.old_f.as_ref().context(
                "Old f not available. update_accel_params expects\
        that the output vector was set in the previous iteration.",
            )?;

        // If `self.iter` is smaller/equal than memory it should be fine to increase the `s` and `d`
        // array, as their number of rows should be equal to `self.iter`. If the array has already
        // reached the size of `self.memory`, the oldest entry is replaced by the new one. The order
        // of the rows is not important, as long `s` and `d` are in the same order.
        match self.iter.cmp(&self.memory) {
            Ordering::Greater => {
                // Index that points to the oldest entry in the arrays. The first iteration
                // that this branch is used is where self.iter == self.memory + 1 and the oldest
                // entry is the zeroth one.
                let idx = (self.iter - 1) % self.memory;
                self.s.slice_mut(s![idx, ..]).assign(&dx);
                self.d.slice_mut(s![idx, ..]).assign(&df);
            }
            _ => {
                self.s.push(Axis(0), dx.view()).context(
                    "dx could not be appended to \
                the s array. The length of the vector dx has to match the number of columns of s",
                )?;
                self.d.push(Axis(0), df.view()).context(
                    "df could not be appended to \
                the s array. The length of the vector dx has to match the number of columns of s",
                )?;
            }
        }

        Ok(())
    }

    /// Calculation of Y corresponding to
    /// | Y_i > = | g_(i+1) > - | g_i > with | g_i > = | x_i > - | F_i >
    fn compute_y(&mut self) -> Array2<f64> {
        &self.s - &self.d
    }

    /// Compute the regularization term for the Anderson Acceleration.
    /// typically type-I does better with higher regularization than type-II.
    fn compute_regularization(&self, m: ArrayView2<f64>) -> f64 {
        m.norm_l2() * self.regularization
    }

    /// Set M to S'Y or Y'Y depending on type of Anderson Acceleration used.
    /// M has dimension `len` x `len` after this, where `len` is min(iter, memory).
    fn compute_m(&self, y: ArrayView2<f64>) -> Array2<f64> {
        let mut m: Array2<f64> = match self.aa_type {
            AAType::I => self.s.dot(&y.t()),
            AAType::II => y.dot(&y.t()),
        };
        if self.regularization != 0.0 {
            let r = self.compute_regularization(m.view());
            m.diag_mut().add_assign(r);
        };
        m
    }

    /// Mixing/relaxation is applied to the new | x_(i+1) > with the old ones using the weights γ.
    /// | x_(i+1)' > = β * | x_(i+1) > + (1-β) ∑_j=0^m γ_j × | x_(i-j) >
    fn relax(&self, f: ArrayView1<f64>, x: ArrayView1<f64>, gamma: ArrayView1<f64>) -> Array1<f64> {
        self.beta * &f + (1.0 - self.beta) * (&x - self.s.t().dot(&gamma))
    }
}
