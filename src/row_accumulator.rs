use std::collections::BinaryHeap;
use crate::topk::TopK;

use crate::similar_row::SimilarRow;
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

    pub(crate) fn collect_all<S: Similarity>(
        &self,
        row: usize,
        similarity: &S,
        norms: &Vec<f64>,
    ) -> Vec<SimilarRow> {

        // TODO maybe this could be an iterator and not require allocation
        let mut similar_users = Vec::new();

        let mut intermediate_head = self.head;

        while intermediate_head != NO_HEAD {
            let other_row = intermediate_head as usize;

            if other_row != row {
                let sim = similarity.from_norms(self.sums[other_row], norms[row], norms[other_row]);
                let scored_row = SimilarRow::new(other_row, sim);
                similar_users.push(scored_row);
            }

            intermediate_head = self.non_zeros[other_row];
        }

        similar_users
    }

    pub(crate) fn topk_and_clear<S: Similarity>(
        &mut self,
        row: usize,
        k: usize,
        similarity: &S,
        norms: &Vec<f64>
    ) -> TopK {

        let mut topk_similar_rows: BinaryHeap<SimilarRow> = BinaryHeap::with_capacity(k);

        while self.head != NO_HEAD {
            let other_row = self.head as usize;

            // We can have zero dot products after deletions
            if other_row != row && self.sums[other_row] != NONE {
                let sim = similarity.from_norms(self.sums[other_row], norms[row], norms[other_row]);
                let scored_row = SimilarRow::new(other_row, sim);

                if topk_similar_rows.len() < k {
                    topk_similar_rows.push(scored_row);
                } else {
                    let mut top = topk_similar_rows.peek_mut().unwrap();
                    if scored_row < *top {
                        *top = scored_row;
                    }
                }
            }

            self.head = self.non_zeros[other_row];
            self.sums[other_row] = NONE;
            self.non_zeros[other_row] = NOT_OCCUPIED;
        }
        self.head = NO_HEAD;

        TopK::new(topk_similar_rows)
    }

}