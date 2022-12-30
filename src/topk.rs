use std::collections::BinaryHeap;
use crate::similar_user::SimilarUser;
use std::collections::binary_heap::Iter;
use crate::topk::TopkUpdate::{NeedsFullRecomputation, NoChange, Update};

#[derive(Clone)]
pub(crate) struct TopK {
    heap: BinaryHeap<SimilarUser>,
}

pub(crate) enum TopkUpdate {
    NoChange,
    Update(TopK),
    NeedsFullRecomputation,
}

impl TopK {

    pub(crate) fn new(heap: BinaryHeap<SimilarUser>) -> Self {
        Self { heap }
    }

    pub(crate) fn len(&self) -> usize {
        self.heap.len()
    }

    pub(crate) fn iter(&self) -> Iter<SimilarUser> {
        self.heap.iter()
    }

    // TODO there must be a better way
    pub(crate) fn contains(&self, user: usize) -> bool {
        for entry in self.heap.iter() {
            if entry.user == user {
                return true;
            }
        }
        false
    }

    pub(crate) fn offer_non_existing_entry(&self, offered_entry: SimilarUser) -> TopkUpdate {
        if offered_entry < *self.heap.peek().unwrap() {
            let mut updated_heap = self.heap.clone();
            {
                let mut top = updated_heap.peek_mut().unwrap();
                *top = offered_entry;
            }
            Update(TopK::new(updated_heap))
        } else {
            NoChange
        }
    }

    pub(crate) fn remove_existing_entry(
        &self,
        user: usize,
        k: usize
    ) -> TopkUpdate {
        let mut new_heap = BinaryHeap::with_capacity(k);

        for existing_entry in self.heap.iter() {
            if existing_entry.user != user {
                new_heap.push(existing_entry.clone());
            }
        }

        Update(TopK::new(new_heap))
    }

    pub(crate) fn update_existing_entry(
        &self,
        update: SimilarUser,
        k: usize
    ) -> TopkUpdate {

        assert_ne!(update.similarity, 0.0);

        if self.heap.len() == k {
            let old_top = self.heap.peek().unwrap();
            if old_top.user == update.user && old_top.similarity > update.similarity {
                return NeedsFullRecomputation
            }
        }

        let mut new_heap = BinaryHeap::with_capacity(k);

        for existing_entry in self.heap.iter() {
            if existing_entry.user != update.user {
                new_heap.push(existing_entry.clone());
            }
        }

        if update.similarity != 0.0 {
            new_heap.push(update);
        }

        Update(TopK::new(new_heap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_not_smallest() {
        let k = 3;
        let mut original_entries = BinaryHeap::with_capacity(k);
        original_entries.push(SimilarUser::new(1, 1.0));
        original_entries.push(SimilarUser::new(2, 0.8));
        original_entries.push(SimilarUser::new(3, 0.5));

        let topk = TopK::new(original_entries);

        let update = topk.update_existing_entry(SimilarUser::new(2, 0.7), k);

        assert!(matches!(update, Update(_)));

        if let Update(new_topk) = update {
            assert_eq!(new_topk.len(), 3);
            let n = new_topk.heap.into_sorted_vec();
            check_entry(&n[0], 1, 1.0);
            check_entry(&n[1], 2, 0.7);
            check_entry(&n[2], 3, 0.5);
        }
    }

    #[test]
    fn test_update_moves() {

        let k = 3;
        let mut original_entries = BinaryHeap::with_capacity(k);
        original_entries.push(SimilarUser::new(1, 1.0));
        original_entries.push(SimilarUser::new(2, 0.8));
        original_entries.push(SimilarUser::new(3, 0.5));

        let topk = TopK::new(original_entries);

        let update = topk.update_existing_entry(SimilarUser::new(2, 1.5), k);

        assert!(matches!(update, Update(_)));
        if let Update(new_topk) = update {
            assert_eq!(new_topk.len(), 3);
            let n = new_topk.heap.into_sorted_vec();
            check_entry(&n[0], 2, 1.5);
            check_entry(&n[1], 1, 1.0);
            check_entry(&n[2], 3, 0.5);
        }
    }

    #[test]
    fn test_update_smallest_but_becomes_larger() {

        let k = 3;
        let mut original_entries = BinaryHeap::with_capacity(k);
        original_entries.push(SimilarUser::new(1, 1.0));
        original_entries.push(SimilarUser::new(2, 0.8));
        original_entries.push(SimilarUser::new(3, 0.5));

        let topk = TopK::new(original_entries);

        let update = topk.update_existing_entry(SimilarUser::new(3, 0.6), k);

        assert!(matches!(update, Update(_)));
        if let Update(new_topk) = update {
            assert_eq!(new_topk.len(), 3);
            let n = new_topk.heap.into_sorted_vec();
            check_entry(&n[0], 1, 1.0);
            check_entry(&n[1], 2, 0.8);
            check_entry(&n[2], 3, 0.6);
        }
    }

    #[test]
    fn test_update_smallest_becomes_smaller() {

        let k = 3;
        let mut original_entries = BinaryHeap::with_capacity(k);
        original_entries.push(SimilarUser::new(1, 1.0));
        original_entries.push(SimilarUser::new(2, 0.8));
        original_entries.push(SimilarUser::new(3, 0.5));

        let topk = TopK::new(original_entries);

        let update = topk.update_existing_entry(SimilarUser::new(3, 0.4), k);
        assert!(matches!(update, NeedsFullRecomputation));
    }

    #[test]
    fn test_update_smallest_becomes_smaller_but_not_full() {

        let k = 3;
        let mut original_entries = BinaryHeap::with_capacity(k);
        original_entries.push(SimilarUser::new(1, 1.0));
        original_entries.push(SimilarUser::new(3, 0.5));

        let topk = TopK::new(original_entries);

        let update = topk.update_existing_entry(SimilarUser::new(3, 0.4), k);

        assert!(matches!(update, Update(_)));
        if let Update(new_topk) = update {
            assert_eq!(new_topk.len(), 2);
            let n = new_topk.heap.into_sorted_vec();
            check_entry(&n[0], 1, 1.0);
            check_entry(&n[1], 3, 0.4);
        }
    }

    fn check_entry(entry: &SimilarUser, expected_user: usize, expected_similarity: f64) {
        assert_eq!(entry.user, expected_user);
        assert!((entry.similarity - expected_similarity).abs() < 0.0001);
    }
}