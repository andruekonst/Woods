use ndarray::{ArrayView2, ArrayView1, Array1, Array};
use std::cmp::Ordering;
use rand;
use rand::Rng;
use rand::distributions::Uniform;
use average::Mean;

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
    fn iter_with_index(&'a self, it: Option<IndexIterator<'b>>) -> ArrayIndexIter<'a, 'b, 'c>;
}

impl<'a, 'b, 'c> WithIndexIter<'a, 'b, 'c> for ArrayView1<'c, D> {
    fn iter_with_index(&'a self, it: Option<IndexIterator<'b>>) -> ArrayIndexIter<'a, 'b, 'c> {
        ArrayIndexIter {
            array_ref: self,
            index_iter: it,
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

fn find_split<'a, 'b>(column: &'a ArrayView1<'_, D>, target: &ArrayView1<'_, D>, indices: Option<&'b Vec<usize>>, id: usize) -> Split {
    let cur_it = column.iter_with_index(Some(indices.unwrap().iter()));
    cur_it.map(|el| {
        println!("Element: {}", el);
    }).count();
    match indices {
        None => {},
        Some(ind) => {
            let sum: D = column.iter_with_index(Some(ind.iter())).sum();
            println!("Sum: {}", sum);
            panic!("Indices processing is not implemented");
        },
    }
    let min = *column.iter().min_by_key(|&k| {
        NonNan::new(*k).unwrap()
    }).unwrap();
    let max = *column.iter().max_by_key(|&k| {
        NonNan::new(*k).unwrap()
    }).unwrap();
    let mut rng = rand::thread_rng();
    // println!("Min: {}, max: {}", min, max);
    let threshold: D = if min < max {
        rng.gen_range(min, max)
    } else {
        min
    };

    let left = column.iter().zip(target).filter(|k| {
        k.0 <= &threshold
    }).map(|k| {
        k.1
    }).collect::<Mean>().mean();
    let right = column.iter().zip(target).filter(|k| {
        k.0 > &threshold
    }).map(|k| {
        k.1
    }).collect::<Mean>().mean();
    // let threshold = (*min + *max) / 2.0;
    println!("Left: {}, right: {}, threshold: {}", left, right, threshold);

    let left_impurity: D = column.iter().zip(target).filter(|k| {
        k.0 <= &threshold
    }).map(|k| {
        let t = k.1 - left;
        t * t
    }).sum();

    let right_impurity: D = column.iter().zip(target).filter(|k| {
        k.0 > &threshold
    }).map(|k| {
        let t = k.1 - right;
        t * t
    }).sum();

    let impurity = left_impurity + right_impurity;
    
    Split {
        feature: id,
        threshold: threshold,
        impurity: impurity,
        values: [left, right]
    }
}

impl DecisionRuleImpl {
    pub fn new() -> Self {
        DecisionRuleImpl {
            split_info: None
        }
    }

    fn fit_by_indices(&mut self, columns: ArrayView2<'_, D>, target: ArrayView1<'_, D>,
                      indices: Option<&Vec<usize>>) {
        // let n_features = columns.dim().0;
        // let bestSplit = Split::new();
        // for j in 0..n_features {
        //     let current = find_split(columns.slice(s![j, ..]),
        //                              target,
        //                              indices);
        // }

        let test: Vec<usize> = vec![1, 2, 3];
        find_split(&columns.row(0), &target, Some(&test), 0);
        // self.split_info = Some(columns.outer_iter().enumerate().map(move |col| {
        //     find_split(&col.1, &target, indices, col.0)
        // }).min_by_key(|split| {
        //     NonNan::new(split.impurity).unwrap()
        // }).unwrap());
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
