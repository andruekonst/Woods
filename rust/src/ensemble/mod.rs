use serde::{Serialize, Deserialize};
use ndarray::{ArrayView2, ArrayView1, Array1, Array2, stack, Axis};
use crate::utils::numerics::D;
use crate::estimator::*;
use rayon::prelude::*;
use rayon::iter::ParallelBridge;

pub mod boosting;
pub mod deep_boosting;

pub trait Ensemble: Estimator {
    type Arg;
    fn new(width: u32, params: Self::Arg) -> Self;
    fn predict_all(&self, columns: &ArrayView2<'_, D>) -> Array2<D>;
    fn predict_by_all(&self, preds: &ArrayView2<'_, D>) -> Array1<D>;
}

#[derive(Serialize, Deserialize)]
pub struct AverageEnsemble<Est> {
    estimators: Vec<Est>
}

// impl Ensemble<GradientBoostingParameters<TreeParameters>> for AverageEnsemble<TreeGBM> {
impl<P: Copy, T: Estimator + ConstructibleWithCopyArg<Arg=P>> Ensemble for AverageEnsemble<T> {
    type Arg = P;
    fn new(width: u32, params: P) -> Self {
    // fn new(width: u32, params: Rc<GradientBoostingParameters<TreeParameters>>) -> Self {
        let estimators = (0..width).map(|_i| {
            T::new(params)
        }).collect();
        AverageEnsemble {
            estimators: estimators,
        }
    }

    fn predict_all(&self, columns: &ArrayView2<'_, D>) -> Array2<D> {
        let preds: Vec<Array1<D>> = self.estimators.iter()
                        .map(|est| est.predict(columns))
                        .collect();
        let views: Vec<ArrayView2<'_, D>> = preds.iter().map(|p|
            p.broadcast((1, p.dim())).unwrap()
        ).collect();
        stack(Axis(0), &views[..]).unwrap()
    }

    fn predict_by_all(&self, preds: &ArrayView2<'_, D>) -> Array1<D> {
        preds.mean_axis(Axis(0)).unwrap()
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