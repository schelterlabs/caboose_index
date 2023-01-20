extern crate ndarray_npy;
extern crate caboose_index;

use numpy::ndarray::Array1;
use ndarray_npy::NpzReader;
use std::fs::File;
use std::time::Instant;
use sprs::CsMat;
use caboose_index::sparse_topk_index::SparseTopKIndex;

fn main() {
    //similarity_bench("yahoosongs-raw.npz", 1000991, 624962, 50, 5);
    similarity_bench("spotify-raw.npz", 1000000, 2262292, 50, 5);
    similarity_bench("lastfm-raw.npz", 993, 174078, 50, 5);
    similarity_bench("tifu-instacart.npz", 30000, 28438, 900, 5);
    similarity_bench("pernir-instacart.npz", 30000, 43936, 300, 5);
    similarity_bench("movielens10m-raw.npz", 69879, 10678, 50, 5);
}

fn similarity_bench(
    matrix_file: &str,
    num_rows: usize,
    num_cols: usize,
    k: usize,
    num_repetitions: usize)
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

    for _ in 0..num_repetitions {
        let representations_copy = representations.clone();
        let start = Instant::now();
        let index = SparseTopKIndex::new(representations_copy, k);
        let end = Instant::now();
        let duration = (end - start).as_millis();
        println!("{:?}: {:?}, {:?}", matrix_file, duration, index.neighbors(1).len());
    };
}
