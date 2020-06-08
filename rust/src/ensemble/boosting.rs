use ndarray::{ArrayView2, ArrayView1, Array1};
use average::Mean;
use crate::estimator::{Estimator, ConstructibleWithCopyArg};
use crate::tree::rule::RandomSplitRule;
use crate::utils::numerics::D;
use crate::tree::{TreeParameters, DecisionTreeImpl};
use std::rc::Rc;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct GradientBoostingParameters<EstParams: Copy> {
    pub est_params: EstParams,
    pub n_estimators: u32,
    pub learning_rate: D,
}

const DEFAULT_GBM_N_ESTIMATORS: u32 = 100u32;
const DEFAULT_GBM_LEARNING_RATE: D = 0.1 as D;

impl<E: Copy> GradientBoostingParameters<E> {
    pub fn new(est_params: E, n_estimators: Option<u32>, learning_rate: Option<D>) -> Self {
        GradientBoostingParameters {
            est_params: est_params,
            n_estimators: n_estimators.unwrap_or(DEFAULT_GBM_N_ESTIMATORS),
            learning_rate: learning_rate.unwrap_or(DEFAULT_GBM_LEARNING_RATE),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct GradientBoostingImpl<Est, EstParams: Copy> {
    params: GradientBoostingParameters<EstParams>,
    estimators: Vec<Est>,
    mean: D,
}

impl<T, P: Copy> ConstructibleWithCopyArg for GradientBoostingImpl<T, P> {
    type Arg = GradientBoostingParameters<P>;
    fn new(params: Self::Arg) -> Self {
        GradientBoostingImpl {
            params: params,
            estimators: vec![],
            mean: D::default(),
        }
    }
}

impl<E, P: Copy> Estimator for GradientBoostingImpl<E, P>
    where E: Estimator + ConstructibleWithCopyArg<Arg=P> {
    fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) {
        self.estimators.clear();

        let average: Mean = target.iter().collect();
        self.mean = average.mean();
        let mut cur_target: Array1<D> = target.iter().map(|t| t - self.mean).collect();
        
        for it in 0..self.params.n_estimators {
            let mut est = E::new(self.params.est_params);
            est.fit(columns, &cur_target.view());
            let preds = est.predict(columns);
            if it != self.params.n_estimators - 1 {
                cur_target = cur_target - preds * self.params.learning_rate;
            }
            self.estimators.push(est);
        }
    }

    fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        let mut predictions: Array1<D> = Array1::from_elem(columns.dim().1, self.mean);
        for est in &self.estimators {
            let cur_preds = est.predict(columns);
            predictions = predictions + cur_preds * self.params.learning_rate;
        }
        predictions
    }
}

pub type TreeGBM = GradientBoostingImpl<DecisionTreeImpl<RandomSplitRule>, TreeParameters>;