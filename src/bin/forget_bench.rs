extern crate ndarray_npy;
extern crate caboose_index;
extern crate average;

use numpy::ndarray::Array1;
use ndarray_npy::NpzReader;
use std::fs::File;
use std::time::Instant;
use sprs::CsMat;
use average::{Quantile, Estimate};
use caboose_index::sparse_topk_index::SparseTopKIndex;
use caboose_index::similarity::COSINE;

fn main() {
    for _ in 0..3 {
        forget_bench("tifu-instacart.npz", 30000, 28438, 900, 500);
        forget_bench("pernir-instacart.npz", 30000, 43936, 300, 500);
    }
}

fn forget_bench(
    matrix_file: &str,
    num_rows: usize,
    num_cols: usize,
    k: usize,
    num_repetitions: usize,)
{
    let mut npz = NpzReader::new(File::open(matrix_file).unwrap()).unwrap();
    let indptr: Array1<i32> = npz.by_name("indptr.npy").unwrap();
    let indices: Array1<i32> = npz.by_name("indices.npy").unwrap();
    let data: Array1<f64> = npz.by_name("data.npy").unwrap();

    let indices_copy: Vec<usize> = indices.into_raw_vec()
        .into_iter().map(|x| x as usize).collect();
    let indptr_copy: Vec<usize> = indptr.into_raw_vec()
        .into_iter().map(|x| x as usize).collect();

    let representations =
        CsMat::new((num_rows, num_cols), indptr_copy, indices_copy, data.into_raw_vec());

    let interactions_to_forget: Vec<(usize, usize)> = representations.iter()
        .step_by(100)
        .take(num_repetitions)
        .map(|(_, (row, column))| (row, column))
        .collect();

    let mut index = SparseTopKIndex::new(representations, k, COSINE);

    let mut p50 = Quantile::new(0.5);
    let mut p90 = Quantile::new(0.9);

    for (row, column) in interactions_to_forget {
        let start = Instant::now();
        index.forget(row, column);
        let end = Instant::now();
        let duration = (end - start).as_millis();
        p50.add(duration as f64);
        p90.add(duration as f64);
    }

    println!("{:?}, p50: {:?}, p90: {:?})", matrix_file, p50.quantile(), p90.quantile());

}
