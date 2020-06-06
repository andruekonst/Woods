use ndarray::{ArrayView1};

pub type IndexIterator<'a> = std::slice::Iter<'a, usize>;

pub struct ArrayMaybeIndexIter<'a, 'b, 'c, D> {
    array_ref: &'a ArrayView1<'c, D>,
    index_iter: Option<IndexIterator<'b>>,
    count: usize,
}

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


pub trait MaybeByIndexIter<'a, 'b, 'c, D> {
    fn iter_by_index(&'a self, v: Option<&'b Vec<usize>>) -> ArrayMaybeIndexIter<'a, 'b, 'c, D>;
}

impl<'a, 'b, 'c, D> MaybeByIndexIter<'a, 'b, 'c, D> for ArrayView1<'c, D>
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
}

pub trait ExplicitByIndexIter<'a, 'b, 'c, D> {
    fn iter_explicit_by_index(&'a self, v: &'b Vec<usize>) -> ArrayExplicitIndexIter<'a, 'b, 'c, D>;
}

impl<'a, 'b, 'c, D> ExplicitByIndexIter<'a, 'b, 'c, D> for ArrayView1<'c, D>
    where D: Copy {
    fn iter_explicit_by_index(&'a self, v: &'b Vec<usize>) -> ArrayExplicitIndexIter<'a, 'b, 'c, D> {
        ArrayExplicitIndexIter {
            array_ref: self,
            index_iter: v.iter(),
        }
    }
}