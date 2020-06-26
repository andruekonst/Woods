//! Binary Decision Trees.
//! 
//! Each decision tree is based on [`SplitRule`].
//! 
//! Currently, only [`rule::RandomSplitRule`] is implemented.

use ndarray::{ArrayView2, ArrayView1, Array1, Axis};
// use crate::rule::{SplitRule};
use crate::estimator::{Estimator, ConstructibleWithCopyArg};
use crate::utils::numerics::D;
use serde::{Serialize, Deserialize};

pub mod rule;

use rule::SplitRule;

/// Default tree depth parameter value
const DEFAULT_TREE_DEPTH: u8 = 3u8;
/// Default tree min samples split parameter value
const DEFAULT_TREE_MIN_SAMPLES_SPLIT: usize = 2usize;

/// Decision Tree Parameters.
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct TreeParameters {
    /// Max tree depth
    pub depth: u8,
    /// Min number of samples to split node
    pub min_samples_split: usize,
}

impl TreeParameters {
    /// Make new `TreeParameters` with default or specified parameters.
    pub fn new(depth: Option<u8>, min_samples_split: Option<usize>) -> Self {
        TreeParameters {
            depth: depth.unwrap_or(DEFAULT_TREE_DEPTH),
            min_samples_split: min_samples_split.unwrap_or(DEFAULT_TREE_MIN_SAMPLES_SPLIT)
        }
    }
}

/// Decision Tree Implementation.
#[derive(Serialize, Deserialize)]
pub struct DecisionTreeImpl<Splitter> {
    /// Configuration
    params: TreeParameters,
    /// Split rules
    splitters: Vec<Splitter>,
    /// Routes from each node to left and right children
    routes: Vec<[i64; 2]>,
}

impl<S: SplitRule> DecisionTreeImpl<S> {
    pub fn new(params: TreeParameters) -> Self {
        DecisionTreeImpl {
            params,
            splitters: vec![],
            routes: vec![],
        }
    }

    fn build_tree(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>,
                  indices: Option<&Vec<usize>>, inv_depth: u8) -> i64 {
        if inv_depth == 0 || target.dim() == 0 {
            return -1;
        }
        if let Some(ind) = indices {
            if ind.len() < self.params.min_samples_split {
                return -1;
            }
        }

        let mut splitter = S::new();
        if let None = splitter.fit_by_indices(columns, target, indices) {
            return -1;
        }
        let split = splitter.split_indices(columns, target, indices);
        self.splitters.push(splitter);
        let id = self.splitters.len() - 1;
        let left_id = self.build_tree(columns, target, Some(&split.indices[0]), inv_depth - 1);
        let right_id = self.build_tree(columns, target, Some(&split.indices[1]), inv_depth - 1);

        self.routes[id][0] = left_id;
        self.routes[id][1] = right_id;

        id as i64
    }
}

impl<S: SplitRule> Estimator for DecisionTreeImpl<S> {
    fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) {
        self.splitters.clear();
        self.routes.clear();

        let n_nodes = 2usize.pow(self.params.depth.into());
        self.routes.resize(n_nodes, [-1i64; 2]);

        self.build_tree(columns, target, None, self.params.depth);
    }

    fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        columns.axis_iter(Axis(1)).map(|features| {
            let mut cur: i64 = 0;
            let mut val;
            loop {
                let split_info = self.splitters[cur as usize].get_split().unwrap();
                let cond: bool = features[split_info.feature] > split_info.threshold;
                cur = self.routes[cur as usize][cond as usize] as i64;
                val = split_info.values[cond as usize];
                if cur < 0 {
                    break;
                }
            }
            val
        }).collect::<Array1<D>>()
    }
}

impl<T: SplitRule> ConstructibleWithCopyArg for DecisionTreeImpl<T> {
    type Arg = TreeParameters;
    fn new(arg: TreeParameters) -> Self {
        DecisionTreeImpl::new(arg)
    }
}