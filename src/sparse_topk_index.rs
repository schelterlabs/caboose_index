use crate::similar_row::SimilarRow;
use crate::row_accumulator::RowAccumulator;
use crate::topk::{TopK, TopkUpdate};

use std::clone::Clone;
use std::collections::binary_heap::Iter;
use std::collections::BinaryHeap;
use sprs::CsMat;
use std::marker::Sync;

use num_cpus::get_physical;
use rayon::prelude::*;
use crate::similarity::Similarity;
use crate::topk::TopkUpdate::{NeedsFullRecomputation, NoChange, Update};

use crate::utils::zero_out_entry;

pub struct SparseTopKIndex<S: Similarity> {
    representations: CsMat<f64>,
    representations_transposed: CsMat<f64>,
    topk_per_row: Vec<TopK>,
    k: usize,
    similarity: S,
    norms: Vec<f64>,
}

impl<S: Similarity + Sync> SparseTopKIndex<S> {

    pub fn neighbors(&self, row: usize) -> Iter<SimilarRow> {
        self.topk_per_row[row].iter()
    }

    pub fn new(representations: CsMat<f64>, k: usize, similarity: S) -> Self {
        let (num_rows, _) = representations.shape();

        //println!("--Creating transposed copy...");
        let mut representations_transposed: CsMat<f64> = representations.to_owned();
        representations_transposed.transpose_mut();
        representations_transposed = representations_transposed.to_csr();

        let data = representations.data();
        let indices = representations.indices();
        let indptr = representations.indptr();
        let data_t = representations_transposed.data();
        let indices_t = representations_transposed.indices();
        let indptr_t = representations_transposed.indptr();

        //println!("--Computing l2 norms...");
        //TODO is it worth to parallelize this?
        let norms: Vec<f64> = (0..num_rows)
            .map(|row| {
                let mut norm_accumulator: f64 = 0.0;
                for column_index in indptr.outer_inds_sz(row) {
                    let value = data[column_index];
                    norm_accumulator += similarity.accumulate_norm(value);
                }
                similarity.finalize_norm(norm_accumulator)
            })
            .collect();

        let num_cores = get_physical();
        let row_range = (0..num_rows).collect::<Vec<usize>>();

        let topk_partitioned: Vec<_> = row_range.par_chunks(num_cores).map(|range| {
            let mut topk_per_row: Vec<TopK> = Vec::with_capacity(range.len());
            let mut accumulator = RowAccumulator::new(num_rows.clone());
            for row in range {
                for column_index in indptr.outer_inds_sz(*row) {
                    let value = data[column_index];
                    for other_row in indptr_t.outer_inds_sz(indices[column_index]) {
                        accumulator.add_to(
                            indices_t[other_row],
                            data_t[other_row] * value.clone()
                        );
                    }
                }

                let topk = accumulator.topk_and_clear(*row, k, &similarity, &norms);
                topk_per_row.push(topk);
            }
            (range, topk_per_row)
        }).collect();

        // TODO Sort the ranges, reserve on first and append the remaining vecs
        let mut topk_per_row: Vec<TopK> = vec![TopK::new(BinaryHeap::new()); num_rows];
        for (range, topk_partition) in topk_partitioned.into_iter() {
            for (index, topk) in range.into_iter().zip(topk_partition.into_iter()) {
                topk_per_row[*index] = topk;
            }
        }

        Self {
            representations,
            representations_transposed,
            topk_per_row,
            k,
            norms,
            similarity
        }
    }


    pub fn forget(&mut self, row: usize, column: usize) {

        let (num_rows, _) = self.representations.shape();

        let old_value = self.representations.get(row, column).unwrap().clone();

        //println!("-Updating user representations");
        zero_out_entry(&mut self.representations, row, column);
        assert_eq!(*self.representations.get(row, column).unwrap(), 0.0_f64);

        zero_out_entry(&mut self.representations_transposed, column, row);
        assert_eq!(*self.representations_transposed.get(column, row).unwrap(), 0.0_f64);

        //println!("-Updating norms");
        let old_l2norm = self.norms[row];
        self.norms[row] = ((old_l2norm * old_l2norm) - (old_value * old_value)).sqrt();

        //println!("-Computing new similarities for user {}", user);
        let data = self.representations.data();
        let indices = self.representations.indices();
        let indptr = self.representations.indptr();
        let data_t = self.representations_transposed.data();
        let indices_t = self.representations_transposed.indices();
        let indptr_t = self.representations_transposed.indptr();

        let mut accumulator = RowAccumulator::new(num_rows.clone());

        for column_index in indptr.outer_inds_sz(row) {
            let value = data[column_index];
            for other_row in indptr_t.outer_inds_sz(indices[column_index]) {
                accumulator.add_to(indices_t[other_row], data_t[other_row] * value.clone());
            }
        }

        let updated_similarities = accumulator.collect_all(row, &self.similarity, &self.norms);

        let mut rows_to_fully_recompute = Vec::new();

        let changes: Vec<(usize, TopkUpdate)> = updated_similarities.par_iter().map(|similar| {

            assert_ne!(similar.row, row);

            let other_row = similar.row;
            let similarity = similar.similarity;

            let other_topk = &self.topk_per_row[other_row];
            let already_in_topk = other_topk.contains(row);

            let update = SimilarRow::new(row, similarity);

            let change = if !already_in_topk {
                if similarity != 0.0 {
                    assert_eq!(other_topk.len(), self.k);
                    other_topk.offer_non_existing_entry(update)
                } else {
                    NoChange
                }
            } else {
                if other_topk.len() < self.k {
                    if update.similarity == 0.0 {
                        other_topk.remove_existing_entry(update.row, self.k)
                    } else {
                        other_topk.update_existing_entry(update, self.k)
                    }
                } else {
                    if update.similarity == 0.0 {
                        NeedsFullRecomputation
                    } else {
                        other_topk.update_existing_entry(update, self.k)
                    }
                }
            };
            (other_row, change)
        }).collect();

        //let mut _count_nochange = 0;
        //let mut _count_update = 0;
        for (other_row, change) in changes {
            match change {
                NeedsFullRecomputation => rows_to_fully_recompute.push(other_row),
                Update(new_topk) => {
                    //_count_update += 1;
                    self.topk_per_row[other_row] = new_topk;
                },
                NoChange => (),//_count_nochange += 1,
            }
        }

        let topk = accumulator.topk_and_clear(row, self.k, &self.similarity, &self.norms);
        self.topk_per_row[row] = topk;

        // TODO is it worth to parallelize this?
        //let _count_recompute = rows_to_fully_recompute.len();
        for row_to_recompute in rows_to_fully_recompute {

            for column_index in indptr.outer_inds_sz(row_to_recompute) {
                let value = data[column_index];
                for other_row in indptr_t.outer_inds_sz(indices[column_index]) {
                    accumulator.add_to(indices_t[other_row], data_t[other_row] * value.clone());
                }
            }

            let topk = accumulator.topk_and_clear(row_to_recompute, self.k, &self.similarity,
                                                  &self.norms);

            self.topk_per_row[row_to_recompute] = topk;
        }
        //println!("NoChange={}/Update={}/Recompute={}", count_nochange, count_update, count_recompute);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;
    use crate::similarity::COSINE;
    use crate::sparse_topk_index::SparseTopKIndex;

    #[test]
    fn test_mini_example() {

        /*
        import numpy as np

        A = np.array(
                [[1, 1, 1, 0, 1],
                 [0, 1, 0, 1, 0],
                 [0, 1, 1, 0, 1],
                 [0, 0, 0, 1, 0]])

        similarity = np.dot(A, A.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag

        print(cosine)

        [[1.         0.35355339 0.8660254  0.        ]
         [0.35355339 1.         0.40824829 0.70710678]
         [0.8660254  0.40824829 1.         0.        ]
         [0.         0.70710678 0.         1.        ]]
        */

        let num_users = 4;
        let num_items = 5;

        let triplets = vec![
            (0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0),
            (1, 1, 1.0), (1, 3, 1.0),
            (2, 1, 1.0), (2, 2, 1.0), (2, 4, 1.0),
            (3, 3, 1.0),
        ];

        let mut input = TriMat::new((num_users, num_items));
        for (row, col, val) in triplets {
            input.add_triplet(row, col, val);
        }

        let user_representations = input.to_csr();
        let index = SparseTopKIndex::new(user_representations, 2, COSINE);

        let mut n0: Vec<_> = index.neighbors(0).collect();
        n0.sort();
        assert_eq!(n0.len(), 2);
        check_entry(n0[0], 2, 0.8660254);
        check_entry(n0[1], 1, 0.35355339);

        let mut n1: Vec<_> = index.neighbors(1).collect();
        n1.sort();
        assert_eq!(n1.len(), 2);
        check_entry(n1[0], 3, std::f64::consts::FRAC_1_SQRT_2);
        check_entry(n1[1], 2, 0.40824829);

        let mut n2: Vec<_> = index.neighbors(2).collect();
        n2.sort();
        assert_eq!(n2.len(), 2);
        check_entry(n2[0], 0, 0.8660254);
        check_entry(n2[1], 1, 0.40824829);

        let n3: Vec<_> = index.neighbors(3).collect();
        assert_eq!(n3.len(), 1);
        check_entry(n3[0], 1, std::f64::consts::FRAC_1_SQRT_2);
    }

    #[test]
    fn test_mini_example_with_deletion() {

        /*
        import numpy as np

        A = np.array(
                [[1, 0, 1, 0, 1],
                 [0, 1, 0, 1, 0],
                 [0, 1, 1, 0, 1],
                 [0, 0, 0, 1, 0]])

        similarity = np.dot(A, A.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag

        print(cosine)

        [[1.         0.         0.66666667 0.        ]
         [0.         1.         0.40824829 0.70710678]
         [0.66666667 0.40824829 1.         0.        ]
         [0.         0.70710678 0.         1.        ]]
        */

        let num_users = 4;
        let num_items = 5;

        let triplets = vec![
            (0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0),
            (1, 1, 1.0), (1, 3, 1.0),
            (2, 1, 1.0), (2, 2, 1.0), (2, 4, 1.0),
            (3, 3, 1.0),
        ];

        let mut input = TriMat::new((num_users, num_items));
        for (row, col, val) in triplets {
            input.add_triplet(row, col, val);
        }

        let user_representations = input.to_csr();
        let mut index = SparseTopKIndex::new(user_representations, 2, COSINE);

        index.forget(0, 1);

        let mut n0: Vec<_> = index.neighbors(0).collect();
        n0.sort();
        assert_eq!(n0.len(), 1);
        check_entry(n0[0], 2, 0.66666667);

        let mut n1: Vec<_> = index.neighbors(1).collect();
        n1.sort();
        assert_eq!(n1.len(), 2);
        check_entry(n1[0], 3, std::f64::consts::FRAC_1_SQRT_2);
        check_entry(n1[1], 2, 0.40824829);

        let mut n2: Vec<_> = index.neighbors(2).collect();
        n2.sort();
        assert_eq!(n2.len(), 2);
        check_entry(n2[0], 0, 0.66666667);
        check_entry(n2[1], 1, 0.40824829);

        let n3: Vec<_> = index.neighbors(3).collect();
        assert_eq!(n3.len(), 1);
        check_entry(n3[0], 1, std::f64::consts::FRAC_1_SQRT_2);
    }

    #[test]
    fn test_mini_example_with_double_deletion() {

        /*
        import numpy as np

        A = np.array(
                [[1, 0, 1, 0, 1],
                 [0, 1, 0, 0, 0],
                 [0, 1, 1, 0, 1],
                 [0, 0, 0, 1, 0]])

        similarity = np.dot(A, A.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag

        print(cosine)

        [[1.         0.         0.66666667 0.        ]
         [0.         1.         0.57735027 0.        ]
         [0.66666667 0.57735027 1.         0.        ]
         [0.         0.         0.         1.        ]]
        */

        let num_users = 4;
        let num_items = 5;

        let triplets = vec![
            (0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0),
            (1, 1, 1.0), (1, 3, 1.0),
            (2, 1, 1.0), (2, 2, 1.0), (2, 4, 1.0),
            (3, 3, 1.0),
        ];

        let mut input = TriMat::new((num_users, num_items));
        for (row, col, val) in triplets {
            input.add_triplet(row, col, val);
        }

        let user_representations = input.to_csr();
        let mut index = SparseTopKIndex::new(user_representations, 2, COSINE);

        index.forget(0, 1);
        index.forget(1, 3);

        let mut n0: Vec<_> = index.neighbors(0).collect();
        n0.sort();
        assert_eq!(n0.len(), 1);
        check_entry(n0[0], 2, 0.66666667);

        let mut n1: Vec<_> = index.neighbors(1).collect();
        n1.sort();
        assert_eq!(n1.len(), 1);
        check_entry(n1[0], 2, 0.57735027);


        let mut n2: Vec<_> = index.neighbors(2).collect();
        n2.sort();
        assert_eq!(n2.len(), 2);
        check_entry(n2[0], 0, 0.66666667);
        check_entry(n2[1], 1, 0.57735027);

        let n3: Vec<_> = index.neighbors(3).collect();
        dbg!(n3);
    }

    fn check_entry(entry: &SimilarRow, expected_user: usize, expected_similarity: f64) {
        assert_eq!(entry.row, expected_user);
        assert!((entry.similarity - expected_similarity).abs() < 0.0001);
    }
}

