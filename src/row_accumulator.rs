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

    pub(crate) fn merge_and_collect_all<S: Similarity>(
        row: usize,
        similarity: &S,
        k: usize,
        norms: &Vec<f64>,
        mut accumulators: Vec<RowAccumulator>
    ) -> (Vec<SimilarRow>, TopK) {

        let last = accumulators.pop().unwrap();

        let mut sums = last.sums;
        let mut non_zero_indices = vec![0; sums.len()];
        let mut topk_similar_rows: BinaryHeap<SimilarRow> = BinaryHeap::with_capacity(k);

        for other in accumulators {
            let mut intermediate_head = other.head;
            while intermediate_head != NO_HEAD {
                let other_row = intermediate_head as usize;
                sums[other_row] += other.sums[other_row];
                non_zero_indices[other_row] = 1;
                intermediate_head = other.non_zeros[other_row];
            }
        }

        let mut similar_users = Vec::new();

        for other_row in 0..sums.len() {
            if non_zero_indices[other_row] == 1 && other_row != row {
                let sim = similarity.from_norms(sums[other_row], norms[row], norms[other_row]);
                let scored_row = SimilarRow::new(other_row, sim);
                similar_users.push(scored_row);

                if sim != 0.0 {
                    let scored_row_clone = SimilarRow::new(other_row, sim);
                    if topk_similar_rows.len() < k {
                        topk_similar_rows.push(scored_row_clone);
                    } else {
                        let mut top = topk_similar_rows.peek_mut().unwrap();
                        if scored_row_clone < *top {
                            *top = scored_row_clone;
                        }
                    }
                }
            }
        }

        (similar_users, TopK::new(topk_similar_rows))
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