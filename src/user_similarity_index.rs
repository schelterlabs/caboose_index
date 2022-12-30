use crate::similar_user::SimilarUser;
use crate::row_accumulator::RowAccumulator;
use crate::topk::{TopK, TopkUpdate};

use std::clone::Clone;
use std::collections::binary_heap::Iter;
use std::collections::BinaryHeap;
use sprs::CsMat;

use num_cpus::get_physical;
use rayon::prelude::*;
use crate::similarity::Similarity;
use crate::topk::TopkUpdate::{NeedsFullRecomputation, NoChange, Update};

use crate::utils::zero_out_entry;

pub struct UserSimilarityIndex<S: Similarity> {
    user_representations:  CsMat<f64>,
    user_representations_transposed: CsMat<f64>,
    topk_per_user: Vec<TopK>,
    k: usize,
    similarity: S,
    norms: Vec<f64>,
}

impl<S: Similarity + std::marker::Sync> UserSimilarityIndex<S> {

    pub fn neighbors(&self, user: usize) -> Iter<SimilarUser> {
        self.topk_per_user[user].iter()
    }

    pub fn new(user_representations: CsMat<f64>, k: usize, similarity: S) -> Self {
        let (num_users, _) = user_representations.shape();

        //println!("--Creating transposed copy...");
        let mut user_representations_transposed: CsMat<f64> = user_representations.to_owned();
        user_representations_transposed.transpose_mut();
        user_representations_transposed = user_representations_transposed.to_csr();

        let data = user_representations.data();
        let indices = user_representations.indices();
        let indptr = user_representations.indptr();
        let data_t = user_representations_transposed.data();
        let indices_t = user_representations_transposed.indices();
        let indptr_t = user_representations_transposed.indptr();

        //println!("--Computing l2 norms...");
        //TODO is it worth to parallelize this?
        let norms: Vec<f64> = (0..num_users)
            .map(|user| {
                let mut norm_accumulator: f64 = 0.0;
                for item_index in indptr.outer_inds_sz(user) {
                    let value = data[item_index];
                    norm_accumulator += similarity.accumulate_norm(value);
                }
                similarity.finalize_norm(norm_accumulator)
            })
            .collect();

        let num_cores = get_physical();
        let user_range = (0..num_users).collect::<Vec<usize>>();

        let topk_partitioned: Vec<_> = user_range.par_chunks(num_cores).map(|range| {
            let mut topk_per_user: Vec<TopK> = Vec::with_capacity(range.len());
            let mut accumulator = RowAccumulator::new(num_users.clone());
            for user in range {
                for item_index in indptr.outer_inds_sz(*user) {
                    let value = data[item_index];
                    for user_index in indptr_t.outer_inds_sz(indices[item_index]) {
                        accumulator.add_to(
                            indices_t[user_index],
                            data_t[user_index] * value.clone()
                        );
                    }
                }

                let topk = accumulator.topk_and_clear(*user, k, &similarity, &norms);
                topk_per_user.push(topk);
            }
            (range, topk_per_user)
        }).collect();

        // TODO Sort the ranges, reserve on first and append the remaining vecs
        let mut topk_per_user: Vec<TopK> = vec![TopK::new(BinaryHeap::new()); num_users];
        for (range, topk_partition) in topk_partitioned.into_iter() {
            for (index, topk) in range.into_iter().zip(topk_partition.into_iter()) {
                topk_per_user[*index] = topk;
            }
        }

        Self {
            user_representations,
            user_representations_transposed,
            topk_per_user,
            k,
            norms,
            similarity
        }
    }


    pub fn forget(&mut self, user: usize, item: usize) {

        let (num_users, _) = self.user_representations.shape();

        let old_value = self.user_representations.get(user, item).unwrap().clone();

        //println!("-Updating user representations");
        zero_out_entry(&mut self.user_representations, user, item);
        assert_eq!(*self.user_representations.get(user, item).unwrap(), 0.0_f64);

        zero_out_entry(&mut self.user_representations_transposed, item, user);
        assert_eq!(*self.user_representations_transposed.get(item, user).unwrap(), 0.0_f64);

        //println!("-Updating norms");
        let old_l2norm = self.norms[user];
        self.norms[user] = ((old_l2norm * old_l2norm) - (old_value * old_value)).sqrt();

        //println!("-Computing new similarities for user {}", user);
        let data = self.user_representations.data();
        let indices = self.user_representations.indices();
        let indptr = self.user_representations.indptr();
        let data_t = self.user_representations_transposed.data();
        let indices_t = self.user_representations_transposed.indices();
        let indptr_t = self.user_representations_transposed.indptr();

        let mut accumulator = RowAccumulator::new(num_users.clone());

        for item_index in indptr.outer_inds_sz(user) {
            let value = data[item_index];
            for user_index in indptr_t.outer_inds_sz(indices[item_index]) {
                accumulator.add_to(indices_t[user_index], data_t[user_index] * value.clone());
            }
        }


        let updated_similarities = accumulator.collect_all(user, &self.norms);

        let mut users_to_fully_recompute = Vec::new();

        let changes: Vec<(usize, TopkUpdate)> = updated_similarities.par_iter().map(|similar_user| {

            assert_ne!(similar_user.user, user);

            let other_user = similar_user.user;
            let similarity = similar_user.similarity;

            let other_topk = &self.topk_per_user[other_user];
            let already_in_topk = other_topk.contains(user);

            let similar_user_to_update = SimilarUser::new(user, similarity);

            if !already_in_topk {
                return if similarity != 0.0 {
                    assert_eq!(other_topk.len(), self.k);
                    (other_user, other_topk.offer_non_existing_entry(similar_user_to_update))
                } else {
                    (other_user, NoChange)
                }
            } else {
                if other_topk.len() < self.k {

                    if similar_user_to_update.similarity == 0.0 {
                        return (other_user, other_topk.remove_existing_entry(
                            similar_user_to_update.user, self.k));
                    } else {
                        return (other_user,
                                other_topk.update_existing_entry(similar_user_to_update, self.k));
                    }

                } else {

                    if similar_user_to_update.similarity == 0.0 {
                        return (other_user, NeedsFullRecomputation);
                    } else {
                        return (other_user, other_topk.update_existing_entry(
                            similar_user_to_update, self.k));
                    };
                }
            }
        }).collect();

        let mut count_nochange = 0;
        let mut count_update = 0;
        for (other_user, change) in changes {
            match change {
                NeedsFullRecomputation => users_to_fully_recompute.push(other_user),
                Update(new_topk) => {
                    count_update += 1;
                    self.topk_per_user[other_user] = new_topk;
                },
                NoChange => count_nochange += 1,
            }
        }

        let topk = accumulator.topk_and_clear(user, self.k, &self.similarity, &self.norms);
        self.topk_per_user[user] = topk;

        // TODO is it worth to parallelize this?
        let count_recompute = users_to_fully_recompute.len();
        for user_to_recompute in users_to_fully_recompute {

            for item_index in indptr.outer_inds_sz(user_to_recompute) {
                let value = data[item_index];
                for user_index in indptr_t.outer_inds_sz(indices[item_index]) {
                    accumulator.add_to(indices_t[user_index], data_t[user_index] * value.clone());
                }
            }

            let topk = accumulator.topk_and_clear(user_to_recompute, self.k, &self.similarity,
                                                  &self.norms);

            self.topk_per_user[user_to_recompute] = topk;
        }
        println!("NoChange={}/Update={}/Recompute={}", count_nochange, count_update, count_recompute);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;
    use crate::similarity::COSINE;
    use crate::user_similarity_index::UserSimilarityIndex;

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
        let index = UserSimilarityIndex::new(user_representations, 2, COSINE);

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
        let mut index = UserSimilarityIndex::new(user_representations, 2, COSINE);

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
        let mut index = UserSimilarityIndex::new(user_representations, 2, COSINE);

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

    fn check_entry(entry: &SimilarUser, expected_user: usize, expected_similarity: f64) {
        assert_eq!(entry.user, expected_user);
        assert!((entry.similarity - expected_similarity).abs() < 0.0001);
    }
}

