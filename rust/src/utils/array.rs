//! Array utils: by index iterators.
use ndarray::{ArrayView1};

/// Iterator of indices (`usize` elements).
pub type IndexIterator<'a> = std::slice::Iter<'a, usize>;

/// By index iterator with optional indices.
/// 
/// If indices are specified, then it iterates over array
/// by provided indices.
pub struct ArrayMaybeIndexIter<'a, 'b, 'c, D> {
    array_ref: &'a ArrayView1<'c, D>,
    index_iter: Option<IndexIterator<'b>>,
    count: usize,
}

/// By index iterator with non-optional indices.
/// 
/// It could be used to iterate over array by provided indices.
pub struct ArrayExplicitIndexIter<'a, 'b, 'c, D> {
    array_ref: &'a ArrayView1<'c, D>,
    index_iter: IndexIterator<'b>,
}

impl<'a, 'b, 'c, D> Iterator for ArrayMaybeIndexIter<'a, 'b, 'c, D>
    where D: Copy {
    type Item = D;

    fn next(&mut self) -> Option<D> {
        if let Some(ind) = &mut self.index_iter {
            let index = ind.next()?;
            Some(self.array_ref[*index])
        } else {
            if self.count >= self.array_ref.dim() {
                None
            } else {
                let index = self.count;
                self.count += 1;
                Some(self.array_ref[index])
            }
        }
    }
}

impl<'a, 'b, 'c, D> Iterator for ArrayExplicitIndexIter<'a, 'b, 'c, D>
    where D: Copy {
    type Item = D;

    fn next(&mut self) -> Option<D> {
        let index = self.index_iter.next()?;
        Some(self.array_ref[*index])
    }
}

/// Iterable by indices.
pub trait ByIndexIter<'a, 'b, 'c, D> {
    /// Iterate by optional indices.
    /// 
    /// If `None` provided instead of indices vector,
    /// iterate through all elements.
    fn iter_by_index(&'a self, indices: Option<&'b Vec<usize>>) -> ArrayMaybeIndexIter<'a, 'b, 'c, D>;

    /// Iterate by indices.
    fn iter_explicit_by_index(&'a self, indices: &'b Vec<usize>) -> ArrayExplicitIndexIter<'a, 'b, 'c, D>;
}

impl<'a, 'b, 'c, D> ByIndexIter<'a, 'b, 'c, D> for ArrayView1<'c, D>
    where D: Copy {
    fn iter_by_index(&'a self, v: Option<&'b Vec<usize>>) -> ArrayMaybeIndexIter<'a, 'b, 'c, D> {
        ArrayMaybeIndexIter {
            array_ref: self,
            index_iter: match &v {
                None => None,
                Some(it) => Some(it.iter()),
            },
            count: 0,
        }
    }

    fn iter_explicit_by_index(&'a self, v: &'b Vec<usize>) -> ArrayExplicitIndexIter<'a, 'b, 'c, D> {
        ArrayExplicitIndexIter {
            array_ref: self,
            index_iter: v.iter(),
        }
    }
}