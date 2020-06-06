use ndarray::{ArrayView2, ArrayView1, Array1};
use rand;
use rand::Rng;
// use rand::distributions::Uniform;
use average::Variance;
// use ndarray::parallel::prelude::*;
use serde::{Serialize, Deserialize};
use crate::numerics::{D, NonNan};
use crate::estimator::Estimator;
use crate::utils::array::*;

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct Split {
    pub feature: usize,
    pub threshold: D,
    pub impurity: D,
    pub values: [D; 2]
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DecisionRuleImpl {
    pub split_info: Option<Split>
}

fn find_split<'a, 'b>(
        column: &'a ArrayView1<'_, D>,
        target: &ArrayView1<'_, D>,
        indices: Option<&'b Vec<usize>>,
        id: usize
    ) -> Option<Split> {
    let min: D;
    let max: D;
    if indices.is_some() {
        min = column.iter_explicit_by_index(indices.unwrap()).map(NonNan::from).min().map(NonNan::into)?;
        max = column.iter_explicit_by_index(indices.unwrap()).map(NonNan::from).max().map(NonNan::into)?;
    } else {
        min = column.iter().map(NonNan::from).min().map(NonNan::into)?;
        max = column.iter().map(NonNan::from).max().map(NonNan::into)?;
    }

    let mut rng = rand::thread_rng();
    let threshold: D = if min < max {
        rng.gen_range(min, max)
    } else {
        return None;
    };

    macro_rules! calc_variance {
        ($side:ident, $var:ident, $comp:expr) => {
            let $side: Variance;
            if indices.is_some() {
                $side = column.iter_explicit_by_index(indices.unwrap())
                              .zip(target.iter_explicit_by_index(indices.unwrap()))
                              .filter(|k| {
                                   let $var = k.0;
                                   $comp
                               })
                              .map(|k| k.1)
                              .collect();
            } else {
                $side = column.iter().cloned()
                              .zip(target.iter().cloned())
                              .filter(|k| {
                                   let $var = k.0;
                                   $comp
                               })
                              .map(|k| k.1)
                              .collect();
            }
        };
    }

    calc_variance!(left, it, it <= threshold);
    calc_variance!(right, it, it > threshold);

    let impurity = left.population_variance() * (left.len() as D) + right.population_variance() * (right.len() as D);
    
    Some(Split {
        feature: id,
        threshold: threshold,
        impurity: impurity,
        values: [left.mean(), right.mean()]
    })
}


type Indices = Vec<usize>;

#[derive(Default)]
pub struct SplitIndices {
    pub indices: [Indices; 2]
}

pub trait SplitRule {
    fn new() -> Self;
    fn fit_by_indices(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>,
                      indices: Option<&Vec<usize>>) -> Option<()>;
    fn split_indices(&self, columns: &ArrayView2<'_, D>, _target: &ArrayView1<'_, D>,
                         indices: Option<&Vec<usize>>) -> SplitIndices;
    fn get_split(&self) -> Option<&Split>;
}

impl SplitRule for DecisionRuleImpl {
    fn new() -> Self {
        DecisionRuleImpl {
            split_info: None
        }
    }

    fn fit_by_indices(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>,
                      indices: Option<&Vec<usize>>) -> Option<()> {
        // self.split_info = columns.outer_iter().into_par_iter().enumerate().map(move |col| {
        self.split_info = columns.outer_iter().enumerate().map(move |col| {
            find_split(&col.1, &target, indices, col.0)
        }).filter(|opt| {
            opt.is_some()
        }).min_by_key(|split| {
            NonNan::new(split.as_ref().unwrap().impurity).unwrap()
        })?;
        Some(())
    }

    fn split_indices(&self, columns: &ArrayView2<'_, D>, _target: &ArrayView1<'_, D>,
                         indices: Option<&Vec<usize>>) -> SplitIndices {
        let mut result = SplitIndices::default();
        let split_info = self.split_info.as_ref().unwrap();
        let column = columns.row(split_info.feature);
        if let Some(ind) = indices {
            for (value, id) in column.iter_by_index(indices).zip(ind) {
                let cond = value > split_info.threshold;
                result.indices[cond as usize].push(*id);
            }
        } else {
            for (value, id) in column.iter().zip(0..) {
                let cond = *value > split_info.threshold;
                result.indices[cond as usize].push(id);
            }
        }
        result
    }

    fn get_split(&self) -> Option<&Split> {
        self.split_info.as_ref()
    }
}

impl Estimator for DecisionRuleImpl {
    fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) {
        self.fit_by_indices(columns, target, None);
    }

    fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        let split_info = self.split_info.as_ref().unwrap();
        columns.row(split_info.feature).iter().map(|val| {
            let cond = *val > split_info.threshold;
            let index = cond as usize;
            split_info.values[index]
        }).collect::<Array1<D>>()
    }
}