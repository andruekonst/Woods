use ndarray::{ArrayView2, ArrayView1, Array1};
use average::Mean;
use crate::estimator::{Estimator, ConstructibleWithRcArg};
use crate::rule::DecisionRuleImpl;
use crate::numerics::D;
use crate::tree::{TreeParameters, DecisionTreeImpl};
use std::rc::Rc;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct GradientBoostingParameters<EstParams> {
    pub est_params: Rc<EstParams>,
    pub n_estimators: u32,
    pub learning_rate: D,
}

const DEFAULT_GBM_N_ESTIMATORS: u32 = 100u32;
const DEFAULT_GBM_LEARNING_RATE: D = 0.1 as D;

impl<E> GradientBoostingParameters<E> {
    pub fn new(est_params: E, n_estimators: Option<u32>, learning_rate: Option<D>) -> Self {
        GradientBoostingParameters {
            est_params: Rc::new(est_params),
            n_estimators: n_estimators.unwrap_or(DEFAULT_GBM_N_ESTIMATORS),
            learning_rate: learning_rate.unwrap_or(DEFAULT_GBM_LEARNING_RATE),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct GradientBoostingImpl<Est, EstParams> {
    params: Rc<GradientBoostingParameters<EstParams>>,
    estimators: Vec<Est>,
    mean: D,
}

// impl<T, EstParams> GradientBoostingImpl<T, EstParams> {
impl<T, EstParams> ConstructibleWithRcArg for GradientBoostingImpl<T, EstParams> {
    type Arg = GradientBoostingParameters<EstParams>;
    fn new(params: Rc<Self::Arg>) -> Self {
        GradientBoostingImpl {
            params: params,
            estimators: vec![],
            mean: D::default(),
        }
    }
}

impl<E, P> Estimator for GradientBoostingImpl<E, P>
    where E: Estimator + ConstructibleWithRcArg<Arg=P> {
    fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) {
        self.estimators.clear();

        let average: Mean = target.iter().collect();
        self.mean = average.mean();
        let mut cur_target: Array1<D> = target.iter().map(|t| t - self.mean).collect();
        
        for it in 0..self.params.n_estimators {
            let mut est = E::new(Rc::clone(&self.params.est_params));
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

pub type TreeGBM = GradientBoostingImpl<DecisionTreeImpl<DecisionRuleImpl>, TreeParameters>;