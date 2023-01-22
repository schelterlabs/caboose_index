extern crate ndarray_npy;
extern crate caboose_index;

use numpy::ndarray::Array1;
use ndarray_npy::NpzReader;
use std::fs::File;
use sprs::CsMat;
use caboose_index::sparse_topk_index::SparseTopKIndex;
use caboose_index::serialize::serialize_to_file;

fn main() {
    serialize("yahoosongs-raw.npz", 1000991, 624962, 50, "yahoosongs-raw-50.bin");

    //serialize("spotify-raw.npz", 1000000, 2262292, 50, "spotify-raw-50.bin");
    //serialize("lastfm-raw.npz", 993, 174078, 50, "lastfm-raw-50.bin");
    //serialize("tifu-instacart.npz", 30000, 28438, 900, "tifu-instacart-900.bin");
    //serialize("pernir-instacart.npz", 30000, 43936, 300, "pernir-instacart-300.bin");
    //serialize("movielens10m-raw.npz", 69879, 10678, 50, "movielens10m-raw-50.bin");

}

fn serialize(
    matrix_file: &str,
    num_rows: usize,
    num_cols: usize,
    k: usize,
    output_file: &str,)
{
    println!("Reading {}", matrix_file);
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
    
    let index = SparseTopKIndex::new(representations, k);
    serialize_to_file(index, output_file);
}
