use ndarray::{ArrayView2, ArrayView1, Array1, Array};
use std::cmp::Ordering;
use rand;
use rand::Rng;
use rand::distributions::Uniform;
use average::Variance;

type D = f64;

#[derive(PartialEq,PartialOrd)]
struct NonNan(D);

impl NonNan {
    fn new(val: D) -> Option<NonNan> {
        Some(NonNan(val))
        // if val.is_nan() {
        //     None
        // } else {
        //     Some(NonNan(val))
        // }
    }
}

impl Eq for NonNan {}

impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl From<D> for NonNan {
    fn from(item: D) -> Self {
        NonNan::new(item).unwrap()
    }
}

impl Into<D> for NonNan {
    fn into(self) -> D {
        self.0
    }
}

type IndexIterator<'a> = std::slice::Iter<'a, usize>;

struct ArrayIndexIter<'a, 'b, 'c> {
    array_ref: &'a ArrayView1<'c, D>,
    index_iter: Option<IndexIterator<'b>>,
    count: usize,
}

impl<'a, 'b, 'c> Iterator for ArrayIndexIter<'a, 'b, 'c> {
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

trait WithIndexIter<'a, 'b, 'c> {
    fn iter_with_index(&'a self, v: Option<&'b Vec<usize>>) -> ArrayIndexIter<'a, 'b, 'c>;
}

impl<'a, 'b, 'c> WithIndexIter<'a, 'b, 'c> for ArrayView1<'c, D> {
    fn iter_with_index(&'a self, v: Option<&'b Vec<usize>>) -> ArrayIndexIter<'a, 'b, 'c> {
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


#[derive(Default)]
struct Split {
    feature: usize,
    threshold: D,
    impurity: D,
    values: [D; 2]
}

pub struct DecisionRuleImpl {
    split_info: Option<Split>
}

fn find_split<'a, 'b>(column: &'a ArrayView1<'_, D>, target: &ArrayView1<'_, D>, indices: Option<&'b Vec<usize>>, id: usize) -> Option<Split> {
    let min: D = column.iter_with_index(indices).map(NonNan::from).min().map(NonNan::into)?;
    let max: D = column.iter_with_index(indices).map(NonNan::from).max().map(NonNan::into)?;
    let mut rng = rand::thread_rng();
    // println!("Min: {}, max: {}", min, max);
    let threshold: D = if min < max {
        rng.gen_range(min, max)
    } else {
        min
    };

    let left: Variance = column.iter_with_index(indices).zip(target).filter(|k| {
        k.0 <= threshold
    }).map(|k| *k.1).collect();

    let right: Variance = column.iter_with_index(indices).zip(target).filter(|k| {
        k.0 > threshold
    }).map(|k| *k.1).collect();

    let impurity = left.error() * (left.len() as D) + right.error() * (right.len() as D);
    
    Some(Split {
        feature: id,
        threshold: threshold,
        impurity: impurity,
        values: [left.mean(), right.mean()]
    })
}

impl DecisionRuleImpl {
    pub fn new() -> Self {
        DecisionRuleImpl {
            split_info: None
        }
    }

    fn fit_by_indices(&mut self, columns: ArrayView2<'_, D>, target: ArrayView1<'_, D>,
                      indices: Option<&Vec<usize>>) {
        self.split_info = columns.outer_iter().enumerate().map(move |col| {
            find_split(&col.1, &target, indices, col.0)
        }).filter(|opt| {
            opt.is_some()
        }).min_by_key(|split| {
            NonNan::new(split.as_ref().unwrap().impurity).unwrap()
        }).unwrap();
    }

    pub fn fit(&mut self, columns: ArrayView2<'_, D>, target: ArrayView1<'_, D>) {
        println!("Fit:)");
        println!("Number of features: {}; number of samples: {}", columns.dim().0, target.dim());
        self.fit_by_indices(columns, target, None);
    }

    pub fn predict(&self, columns: ArrayView2<'_, D>) -> Array1<D> {
        // let predictions = Array1::zeros(columns.dim().1);
        // predictions
        columns.row(self.split_info.as_ref().unwrap().feature).iter().map(|val| {
            let cond = *val > self.split_info.as_ref().unwrap().threshold;
            let index = cond as usize;
            self.split_info.as_ref().unwrap().values[index]
        }).collect::<Array1<D>>()
    }

    pub fn test(&self) -> Result<String, ()> {
        Ok("Test text".to_owned())
    }
}
