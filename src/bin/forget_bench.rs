extern crate caboose_index;
extern crate average;

use std::time::Instant;
use average::{Quantile, Estimate};
use caboose_index::sparse_topk_index::SparseTopKIndex;
use caboose_index::serialize::deserialize_from;

fn main() {

    let num_repetitions = 1;
    let num_samples_to_forget = 500;

    for _ in 0..num_repetitions {
        forget_bench("tifu-instacart-900.bin", 30000, 28438, num_samples_to_forget);
    }

    for _ in 0..num_repetitions {
        forget_bench("pernir-instacart-300.bin", 30000, 43936, num_samples_to_forget);
    }

    for _ in 0..num_repetitions {
        forget_bench("synthetic10-raw-50.bin", 10000, 50000, num_samples_to_forget);
    }

    for _ in 0..num_repetitions {
        forget_bench("movielens10m-raw-50.bin", 69879, 10678, num_samples_to_forget);
    }

    for _ in 0..num_repetitions {
        forget_bench("lastfm-raw-50.bin", 993, 174078, num_samples_to_forget);
    }

    for _ in 0..num_repetitions {
        forget_bench("synthetic50-raw-50.bin", 100000, 50000, num_samples_to_forget);
    }

    for _ in 0..num_repetitions {
        forget_bench("spotify-raw-50.bin", 1000000, 2262292, num_samples_to_forget);
    }

    for _ in 0..num_repetitions {
        forget_bench("yahoosongs-raw-50.bin", 1000991, 624962, num_samples_to_forget);
    }


}

fn forget_bench(
    index_file: &str,
    num_rows: usize,
    num_cols: usize,
    num_repetitions: usize,)
{
    let mut index: SparseTopKIndex = deserialize_from(num_rows, num_cols, index_file);

    let interactions_to_forget: Vec<(usize, usize)> = index.representations.iter()
        .step_by(100)
        .take(num_repetitions)
        .map(|(_, (row, column))| (row, column))
        .collect();

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

    //println!("{:?}, p50: {:?}, p90: {:?}", index_file, p50.quantile(), p90.quantile());
    println!("{:?},{:?},{:?}", index_file, p50.quantile(), p90.quantile());
}
