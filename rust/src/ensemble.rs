use serde::{Serialize, Deserialize};
use ndarray::{ArrayView2, ArrayView1, Array1};
use crate::numerics::D;
use crate::tree::TreeParameters;
use crate::boosting::{GradientBoostingImpl, GradientBoostingParameters, TreeGBM};
use crate::estimator::*;
use std::rc::Rc;

#[derive(Serialize, Deserialize)]
pub struct AverageEnsemble<Est> {
    estimators: Vec<Est>
}

impl AverageEnsemble<TreeGBM> {
    pub fn new(width: u32, params: Rc<GradientBoostingParameters<TreeParameters>>) -> Self {
        let estimators = (0..width).map(|_i| {
            GradientBoostingImpl::new(Rc::clone(&params))
        }).collect();
        AverageEnsemble {
            estimators: estimators,
        }
    }
}

impl Estimator for AverageEnsemble<TreeGBM> {
    fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) {
        for est in &mut self.estimators {
            est.fit(columns, target);
        }
    }
    fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        let preds = self.estimators.iter()
                        .map(|est| est.predict(columns));
        let mut result = Array1::zeros(columns.dim().1);
        let alpha: D = (1.0 as D) / (self.estimators.len() as D);
        for pred in preds {
            result = result + pred;
        }
        result *= alpha;
        result
    }
}