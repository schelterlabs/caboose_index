use std::cmp::Ordering;

#[derive(PartialEq, Debug, Clone)]
pub struct SimilarRow {
    pub row: usize,
    pub similarity: f64,
}

impl SimilarRow {
    pub fn new(row: usize, similarity: f64) -> Self {
        SimilarRow { row, similarity }
    }
}

/// Ordering for our max-heap, not that we must use a special implementation here as there is no
/// total order on floating point numbers.
fn cmp_reverse(sim_a: &SimilarRow, sim_b: &SimilarRow) -> Ordering {
    match sim_a.similarity.partial_cmp(&sim_b.similarity) {
        Some(Ordering::Less) => Ordering::Greater,
        Some(Ordering::Greater) => Ordering::Less,
        Some(Ordering::Equal) => Ordering::Equal,
        None => Ordering::Equal
    }
}

impl Eq for SimilarRow {}

impl Ord for SimilarRow {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_reverse(self, other)
    }
}

impl PartialOrd for SimilarRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(cmp_reverse(self, other))
    }
}