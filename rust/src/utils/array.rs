use ndarray::{ArrayView1};

pub type IndexIterator<'a> = std::slice::Iter<'a, usize>;

pub struct ArrayIndexIter<'a, 'b, 'c, D> {
    array_ref: &'a ArrayView1<'c, D>,
    index_iter: Option<IndexIterator<'b>>,
    count: usize,
}

impl<'a, 'b, 'c, D> Iterator for ArrayIndexIter<'a, 'b, 'c, D>
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

pub trait WithIndexIter<'a, 'b, 'c, D> {
    fn iter_with_index(&'a self, v: Option<&'b Vec<usize>>) -> ArrayIndexIter<'a, 'b, 'c, D>;
}

impl<'a, 'b, 'c, D> WithIndexIter<'a, 'b, 'c, D> for ArrayView1<'c, D>
    where D: Copy {
    fn iter_with_index(&'a self, v: Option<&'b Vec<usize>>) -> ArrayIndexIter<'a, 'b, 'c, D> {
        ArrayIndexIter {
            array_ref: self,
            index_iter: match &v {
                None => None,
                Some(it) => Some(it.iter()),
            },
            count: 0,
        }
    }
}