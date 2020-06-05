use ndarray::{ArrayView2, ArrayView1, Array1, Array, Axis};
use average::Mean;
use crate::rule::{DecisionRuleImpl, D};
use crate::tree::{TreeParameters, DecisionTreeImpl};
use std::rc::Rc;

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

pub struct GradientBoostingImpl<Est, EstParams> {
    params: GradientBoostingParameters<EstParams>,
    estimators: Vec<Est>,
    mean: D,
}

impl GradientBoostingImpl<DecisionTreeImpl<DecisionRuleImpl>, TreeParameters> {
    pub fn new(params: GradientBoostingParameters<TreeParameters>) -> Self {
        GradientBoostingImpl {
            params: params,
            estimators: vec![],
            mean: D::default(),
        }
    }

    pub fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) {
        self.estimators.clear();

        let average: Mean = target.iter().collect();
        self.mean = average.mean();
        let mut cur_target: Array1<D> = target.iter().map(|t| t - self.mean).collect();
        
        for it in 0..self.params.n_estimators {
            let mut est = DecisionTreeImpl::new(Rc::clone(&self.params.est_params));
            est.fit(columns, &cur_target.view());
            let preds = est.predict(columns);
            if it != self.params.n_estimators - 1 {
                // cur_target = cur_target - preds.mapv(|v| self.params.learning_rate * v);
                cur_target = cur_target - preds * self.params.learning_rate;
            }
            self.estimators.push(est);
        }
    }

    pub fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        let mut predictions: Array1<D> = Array1::from_elem(columns.dim().1, self.mean);
        for est in &self.estimators {
            let cur_preds = est.predict(columns);
            // predictions = predictions + cur_preds.mapv(|v| self.params.learning_rate * v);
            predictions = predictions + cur_preds * self.params.learning_rate;
        }
        predictions
    }
}