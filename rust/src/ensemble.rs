use serde::{Serialize, Deserialize};
use ndarray::{ArrayView2, ArrayView1, Array1};
use crate::numerics::D;
use crate::estimator::*;
use std::rc::Rc;

pub trait Ensemble: Estimator {
    type Arg;
    fn new(width: u32, params: Rc<Self::Arg>) -> Self;
}

#[derive(Serialize, Deserialize)]
pub struct AverageEnsemble<Est> {
    estimators: Vec<Est>
}

// impl Ensemble<GradientBoostingParameters<TreeParameters>> for AverageEnsemble<TreeGBM> {
impl<P, T: Estimator + ConstructibleWithRcArg<Arg=P>> Ensemble for AverageEnsemble<T> {
    type Arg = P;
    fn new(width: u32, params: Rc<P>) -> Self {
    // fn new(width: u32, params: Rc<GradientBoostingParameters<TreeParameters>>) -> Self {
        let estimators = (0..width).map(|_i| {
            T::new(Rc::clone(&params))
        }).collect();
        AverageEnsemble {
            estimators: estimators,
        }
    }
}

impl<T: Estimator> Estimator for AverageEnsemble<T> {
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