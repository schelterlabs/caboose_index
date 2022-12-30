use std::cmp::Ordering;

#[derive(PartialEq, Debug, Clone)]
pub struct SimilarUser {
    pub user: usize,
    pub similarity: f64,
}

impl SimilarUser {
    pub fn new(user: usize, similarity: f64) -> Self {
        SimilarUser { user, similarity }
    }
}

/// Ordering for our max-heap, not that we must use a special implementation here as there is no
/// total order on floating point numbers.
fn cmp_reverse(sim_user_a: &SimilarUser, sim_user_b: &SimilarUser) -> Ordering {
    match sim_user_a.similarity.partial_cmp(&sim_user_b.similarity) {
        Some(Ordering::Less) => Ordering::Greater,
        Some(Ordering::Greater) => Ordering::Less,
        Some(Ordering::Equal) => Ordering::Equal,
        None => Ordering::Equal
    }
}

impl Eq for SimilarUser {}

impl Ord for SimilarUser {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_reverse(self, other)
    }
}

impl PartialOrd for SimilarUser {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(cmp_reverse(self, other))
    }
}