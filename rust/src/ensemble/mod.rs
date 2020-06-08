use serde::{Serialize, Deserialize};
use ndarray::{ArrayView2, ArrayView1, Array1, Array2, stack, Axis};
use crate::utils::numerics::D;
use crate::estimator::*;
use rayon::prelude::*;

pub mod boosting;
pub mod deep_boosting;

/// Ensemble of estimators.
/// 
/// It can be constructed by number of estimators (`width`) and estimator parameters.
pub trait Ensemble: Estimator {
    /// Estimator parameters type.
    type Arg;
    fn new(width: u32, params: Self::Arg) -> Self;
    /// Predict with all base estimators and concatenate rows of predictions along 0 axis.
    fn predict_all(&self, columns: &ArrayView2<'_, D>) -> Array2<D>;
    /// Make ensemble prediction from base estimators predictions.
    fn predict_by_all(&self, preds: &ArrayView2<'_, D>) -> Array1<D>;

    fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        let all_preds = self.predict_all(columns);
        self.predict_by_all(&all_preds.view())
    }
}

/// Ensemble that averages all base estimators predictions.
#[derive(Serialize, Deserialize)]
pub struct AverageEnsemble<E> {
    estimators: Vec<E>
}

// impl Ensemble<GradientBoostingParameters<TreeParameters>> for AverageEnsemble<TreeGBM> {
impl<P: Copy, T: Estimator + ConstructibleWithCopyArg<Arg=P> + Send + Sync> Ensemble for AverageEnsemble<T> {
    type Arg = P;
    fn new(width: u32, params: P) -> Self {
        let estimators = (0..width).map(|_i| {
            T::new(params)
        }).collect();
        AverageEnsemble {
            estimators: estimators,
        }
    }

    fn predict_all(&self, columns: &ArrayView2<'_, D>) -> Array2<D> {
        let preds: Vec<Array1<D>> = self.estimators
                        .par_iter()
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

impl<T: Estimator + Send + Sync> Estimator for AverageEnsemble<T>
    where AverageEnsemble<T>: Ensemble {
    fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>) {
        self.estimators.par_iter_mut()
                       .for_each(|est| {
                           est.fit(columns, target)
                        });
    }

    fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        Ensemble::predict(self, columns) 
    }
}