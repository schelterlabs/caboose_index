use std::collections::BinaryHeap;
use crate::topk::TopK;

use crate::similar_user::SimilarUser;
use crate::similarity::Similarity;

pub(crate) struct RowAccumulator {
    sums: Vec<f64>,
    non_zeros: Vec<isize>,
    head: isize
}

const NONE: f64 = 0.0;
const NOT_OCCUPIED: isize = -1;
const NO_HEAD: isize = -2;

impl RowAccumulator {

    pub(crate) fn new(num_items: usize) -> Self {
        RowAccumulator {
            sums: vec![NONE; num_items],
            non_zeros: vec![NOT_OCCUPIED; num_items],
            head: NO_HEAD,
        }
    }

    pub(crate) fn add_to(&mut self, column: usize, value: f64) {
        self.sums[column] += value;

        if self.non_zeros[column] == NOT_OCCUPIED {
            self.non_zeros[column] = self.head.clone();
            self.head = column as isize;
        }
    }

    pub(crate) fn collect_all(
        &self,
        user: usize,
        l2norms: &Vec<f64>,
    ) -> Vec<SimilarUser> {

        // TODO maybe this could be an iterator and not require allocation
        let mut similar_users = Vec::new();

        let mut intermediate_head = self.head;

        while intermediate_head != NO_HEAD {
            let other_user = intermediate_head as usize;

            if other_user != user {
                let similarity = self.sums[other_user] / (l2norms[user] * l2norms[other_user]);
                let scored_user = SimilarUser::new(other_user, similarity);
                similar_users.push(scored_user);
            }

            intermediate_head = self.non_zeros[other_user];
        }

        similar_users
    }

    pub(crate) fn topk_and_clear<S: Similarity>(
        &mut self,
        user: usize,
        k: usize,
        similarity: &S,
        norms: &Vec<f64>
    ) -> TopK {

        let mut topk_similar_users: BinaryHeap<SimilarUser> = BinaryHeap::with_capacity(k);

        while self.head != NO_HEAD {
            let other_user = self.head as usize;

            // We can have zero dot products after deletions
            if other_user != user && self.sums[other_user] != NONE {
                let sim = similarity.from_norms(self.sums[other_user],
                    norms[user], norms[other_user]);
                let scored_user = SimilarUser::new(other_user, sim);

                if topk_similar_users.len() < k {
                    topk_similar_users.push(scored_user);
                } else {
                    let mut top = topk_similar_users.peek_mut().unwrap();
                    if scored_user < *top {
                        *top = scored_user;
                    }
                }
            }

            self.head = self.non_zeros[other_user];
            self.sums[other_user] = NONE;
            self.non_zeros[other_user] = NOT_OCCUPIED;
        }
        self.head = NO_HEAD;

        TopK::new(topk_similar_users)
    }

}