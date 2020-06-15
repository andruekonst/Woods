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
    fn make(width: u32, params: Self::Arg) -> Self;
    /// Predict with all base estimators and concatenate rows of predictions along 0 axis.
    fn predict_all(&self, columns: &ArrayView2<'_, D>) -> Array2<D>;
    /// Make ensemble prediction from base estimators predictions.
    fn predict_by_all(&self, preds: &ArrayView2<'_, D>) -> Array1<D>;

    fn predict_ensemble(&self, columns: &ArrayView2<'_, D>) -> Array1<D> {
        let all_preds = self.predict_all(columns);
        self.predict_by_all(&all_preds.view())
    }
}

/// Iterator of estimators that can be used to make predictions.
pub trait EstimatorsIter {
    /// Make predictions by each [`Estimator`] and combine results into 2D array.
    fn make_predictions(&mut self, columns: &ArrayView2<'_, D>) -> Array2<D>;
}

/// Stack slice of predictions into 2D array.
fn stack_predictions(preds: &[Array1<D>]) -> Array2<D> {
    let views: Vec<ArrayView2<'_, D>> = preds.iter().map(|p|
        p.broadcast((1, p.dim())).unwrap()
    ).collect();
    stack(Axis(0), &views[..]).unwrap()
}

// impl<'a, T, E: 'a> EstimatorsIter for T
//     where T: Iterator<Item=&'a E>,
//           E: Estimator {
//     fn make_predictions(&mut self, columns: &ArrayView2<'_, D>) -> Array2<D> {
//         let preds: Vec<Array1<D>> = self
//                         .map(|est| est.predict(columns))
//                         .collect();
//         stack_predictions(preds)
//     }
// }

/// Collection of estimators that can be used to make predictions.
pub trait EstimatorsCollection {
    /// Make predictions with collection of estimators.
    fn make_predictions(&self, columns: &ArrayView2<'_, D>) -> Array2<D>;
}

/// Collection of `Send + Sync` estimators that can be used to make predictions.
pub trait ParEstimatorsCollection {
    /// Make predictions with collection of estimators **in parallel**.
    fn par_make_predictions(&self, columns: &ArrayView2<'_, D>) -> Array2<D>;
}

impl<E: Estimator> EstimatorsCollection for Vec<E> {
    fn make_predictions(&self, columns: &ArrayView2<'_, D>) -> Array2<D> {
        let preds: Vec<Array1<D>> = self
                        .iter()
                        .map(|est| est.predict(columns))
                        .collect();
        stack_predictions(&preds)
    }
}

impl<E: Estimator + Send + Sync> ParEstimatorsCollection for Vec<E> {
    fn par_make_predictions(&self, columns: &ArrayView2<'_, D>) -> Array2<D> {
        let preds: Vec<Array1<D>> = self
                        .par_iter()
                        .map(|est| est.predict(columns))
                        .collect();
        stack_predictions(&preds)
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
    fn make(width: u32, params: P) -> Self {
        let estimators = (0..width).map(|_i| {
            T::new(params)
        }).collect();
        AverageEnsemble {
            estimators: estimators,
        }
    }

    fn predict_all(&self, columns: &ArrayView2<'_, D>) -> Array2<D> {
        // let preds: Vec<Array1<D>> = self.estimators
        //                 .par_iter()
        //                 .map(|est| est.predict(columns))
        //                 .collect();
        // stack_predictions(preds)
        // self.estimators.par_iter().make_predictions(columns)
        self.estimators.par_make_predictions(columns)
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
        Ensemble::predict_ensemble(self, columns) 
    }
}
