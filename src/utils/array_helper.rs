use ndarray::prelude::*;
use ndarray::Data;
use rayon::prelude::*;
use std::cmp::Ordering;

pub fn parallel_matrix_multiply_2(
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    blocks: usize,
) -> Array2<f64> {
    let dim: usize = a.dim().1;
    let block_size: usize = dim / blocks;
    let f_0: usize = a.dim().0;
    let f_1: usize = b.dim().1;

    let arr_temp: Array2<f64> = (0..blocks)
        .into_par_iter()
        .map(|i| {
            let start = i * block_size;
            let mut end = (i + 1) * block_size;
            if i == (blocks - 1) {
                end = dim;
            }
            let ar: Array2<f64> = a
                .slice(s![.., start..end])
                .dot(&b.slice(s![start..end, ..]));
            ar
        })
        .reduce(|| Array2::zeros((f_0, f_1)), |a, b| a + b);

    arr_temp
}

pub fn parallel_matrix_multiply(
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    blocks: usize,
) -> Array2<f64> {
    let f_0: usize = a.dim().0;
    let f_1: usize = b.dim().1;
    let size: usize = f_0 / blocks;

    let mut arr: Array2<f64> = Array2::zeros((f_0, f_1));

    arr.axis_chunks_iter_mut(Axis(0), size)
        .into_par_iter()
        .zip(a.axis_chunks_iter(Axis(0), size).into_par_iter())
        .for_each(|(mut rows, rows_a)| {
            rows.slice_mut(s![.., ..]).assign(&rows_a.dot(&b));
        });

    arr
}

pub trait ToOwnedF<A, D> {
    fn to_owned_f(&self) -> Array<A, D>;
}
impl<A, S, D> ToOwnedF<A, D> for ArrayBase<S, D>
where
    A: Copy + Clone,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn to_owned_f(&self) -> Array<A, D> {
        let mut tmp = unsafe { Array::uninitialized(self.dim().f()) };
        tmp.assign(self);
        tmp
    }
}

pub fn argsort(v: ArrayView1<f64>) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(Ordering::Equal));
    idx
}

pub fn argsort_abs(v: ArrayView1<f64>) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| {
        v[i].abs()
            .partial_cmp(&v[j].abs())
            .unwrap_or(Ordering::Equal)
    });
    idx
}

pub fn argsort32(v: ArrayView1<f32>) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(Ordering::Equal));
    idx
}

pub fn argsort32_abs(v: ArrayView1<f32>) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| {
        v[i].abs()
            .partial_cmp(&v[j].abs())
            .unwrap_or(Ordering::Equal)
    });
    idx
}

pub fn argsort_usize(v: ArrayView1<usize>) -> Vec<usize> {
    let mut idx = (0..v.len()).collect::<Vec<_>>();
    idx.sort_unstable_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(Ordering::Equal));
    idx
}
