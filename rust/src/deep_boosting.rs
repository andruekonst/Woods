use ndarray::{ArrayView2, ArrayView1, Array1, Axis, Array2, stack};
use crate::rule::{D, NonNan};
use crate::tree::TreeParameters;
use crate::boosting::{GradientBoostingParameters, TreeGBM};
use std::rc::Rc;
use serde::{Serialize, Deserialize};
use itertools::iproduct;
use rayon::prelude::*;
use rayon::iter::ParallelBridge;
use crate::estimator::*;
use crate::ensemble::*;


impl Estimator for TreeGBM {
    fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) {
        self.fit(columns, target);
    }
    fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        self.predict(columns)
    }
}

#[derive(Serialize, Deserialize)]
pub struct DeepBoostingParameters {
    pub n_estimators: u32,
    pub layer_width: u32,
    pub learning_rate: D,
}

const DEFAULT_DGBM_N_ESTIMATORS: u32 = 5u32;
const DEFAULT_DGBM_LAYER_WIDTH: u32 = 5u32;
const DEFAULT_DGBM_LEARNING_RATE: D = 0.25 as D;

impl DeepBoostingParameters {
    pub fn new(n_estimators: Option<u32>, layer_width: Option<u32>, learning_rate: Option<D>) -> Self {
        DeepBoostingParameters {
            n_estimators:   n_estimators.unwrap_or(DEFAULT_DGBM_N_ESTIMATORS),
            layer_width:     layer_width.unwrap_or(DEFAULT_DGBM_LAYER_WIDTH),
            learning_rate: learning_rate.unwrap_or(DEFAULT_DGBM_LEARNING_RATE),
        }
    }
}

type TreeGBMParams = GradientBoostingParameters<TreeParameters>;

fn cv_best_params(columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) -> TreeGBMParams {
    let depth = vec![2, 3, 5];
    let n_epochs = vec![100, 1000];
    let learning_rate = vec![0.1, 0.01];
    let best = iproduct!(depth.iter(), n_epochs.iter(), learning_rate.iter())
        .par_bridge() // compute in parallel
        .map(|p| {
        let (d, n, lr) = p;
        let tree_params = TreeParameters::new(Some(*d), None);
        let params = GradientBoostingParameters::new(tree_params, Some(*n), Some(*lr));
        let mut est = TreeGBM::new(Rc::new(params));
        let score = eval_est_cv(&mut est, 5, columns, target);
        (p, NonNan::from(score))
    }).min_by_key(|a| {
        a.1.clone()
    }).map(|a| a.0).unwrap();
    let tree_params = TreeParameters::new(Some(*best.0), None);
    let params = GradientBoostingParameters::new(tree_params, Some(*best.1), Some(*best.2));
    params
}

#[derive(Serialize, Deserialize)]
pub struct DeepBoostingImpl<EnsembleEst> {
    pub params: DeepBoostingParameters,
    estimators: Vec<EnsembleEst>,
}

impl DeepBoostingImpl<AverageEnsemble<TreeGBM>> {
    pub fn new(params: DeepBoostingParameters) -> Self {
        DeepBoostingImpl {
            params: params,
            estimators: vec![],
        }
    }

    pub fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) {
        self.estimators.clear();

        let mut acc_columns: Array2<D> = columns.to_owned(); // accumulated columns
        // copy target
        let mut cur_target: Array1<D> = target.to_owned();
        
        for it in 0..self.params.n_estimators {
            // find locally optimal GBM parameters
            let opt_params = Rc::new(cv_best_params(&acc_columns.view(), &cur_target.view()));
            let mut ensemble = AverageEnsemble::new(self.params.layer_width, opt_params);

            // fit ensemble on generated features
            ensemble.fit(&acc_columns.view(), &cur_target.view());
            // predict with ensemble
            let preds = ensemble.predict(&acc_columns.view());
            
            // append new features
            acc_columns = stack(Axis(0), &[acc_columns.view(), preds.broadcast((1, preds.dim())).unwrap()]).unwrap();

            // update target
            if it != self.params.n_estimators - 1 {
                cur_target = cur_target - preds * if it == 0 {
                    1.0 as D
                } else {
                    self.params.learning_rate
                };
            }
            self.estimators.push(ensemble);
        }
    }

    pub fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        // let mut predictions: Array1<D> = Array1::zeros(columns.dim().1);
        let mut predictions: Array1<D> = self.estimators.first().unwrap().predict(columns);
        let mut acc_columns: Array2<D> = stack(Axis(0), &[columns.to_owned().view(),
                                            predictions.broadcast((1, predictions.dim())).unwrap()]).unwrap();
        for est in self.estimators.iter().skip(1) {
            let cur_preds = est.predict(&acc_columns.view());
            // it not needed for the last iteration
            acc_columns = stack(Axis(0), &[acc_columns.view(), cur_preds.broadcast((1, cur_preds.dim())).unwrap()]).unwrap();
            // predictions = predictions + cur_preds.mapv(|v| self.params.learning_rate * v);
            predictions = predictions + cur_preds * self.params.learning_rate;
        }
        predictions
    }
}
