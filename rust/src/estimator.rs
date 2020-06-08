use ndarray::{ArrayView2, ArrayView1, Array1, Axis, Slice};
use ndarray_stats::DeviationExt;
use crate::utils::numerics::D;
use std::rc::Rc;

/// Estimator that could be trained and used to make predictions.
pub trait Estimator {
    /// Fit estimator with training data and target.
    /// 
    /// **Important**: each `columns` row correspond to the input feature, not sample.
    fn fit(&mut self, columns: &ArrayView2<'_, D>, target: &ArrayView1<'_, D>);
    /// Predict with estimator on potentially unseed data.
    fn predict(&self, columns: &ArrayView2<'_, D>) -> Array1<D>;
}

/// Structure can be constructed with arguments of associated-type `Arg`.
pub trait ConstructibleWithArg {
    type Arg;
    fn new(arg: Self::Arg) -> Self;
}

/// Structure can be constructed with copyable arguments of associated-type `Arg`, i.e. that implement `Copy` trait.
/// 
/// It is useful for estimators which can be joined to ensemble with the same arguments.
pub trait ConstructibleWithCopyArg {
    type Arg: Copy;
    fn new(arg: Self::Arg) -> Self;
}

/// Fit estimator on training data and calculate mean squared error on validation data.
pub fn eval_est<Est: Estimator>(
    est: &mut Est,
    train_columns: &ArrayView2<'_, D>,
    train_target: &ArrayView1<'_, D>,
    val_columns: &ArrayView2<'_, D>,
    val_target: &ArrayView1<'_, D>) -> D {
    est.fit(train_columns, train_target);
    let preds = est.predict(val_columns);
    preds.mean_sq_err(val_target).unwrap() as D
}

/// Evaluate estimator ([`eval_est`]) with cross-validation.
pub fn eval_est_cv<Est: Estimator>(
        est: &mut Est,
        cv: u8,
        columns: &ArrayView2<'_, D>,
        target: &ArrayView1<'_, D>
    ) -> D {
    let n_samples: usize = target.dim();
    let fold_size: usize = n_samples / (cv as usize);
    let mut res: D = D::default();

    for i in 0..cv {
        let from = fold_size * (i as usize);
        let to = std::cmp::min(fold_size * ((i + 1) as usize), n_samples + 1);
        let fold_columns = columns.slice_axis(Axis(1), Slice::from(from..to));
        let fold_target = target.slice_axis(Axis(0), Slice::from(from..to));
        if i > 0 {
            let val_columns = columns.slice_axis(Axis(1), Slice::from(0..from));
            let val_target = target.slice_axis(Axis(0), Slice::from(0..from));
            res += eval_est(est, &fold_columns, &fold_target, &val_columns, &val_target);
        }
        if i < cv - 1 {
            let val_columns = columns.slice_axis(Axis(1), Slice::from(to..));
            let val_target = target.slice_axis(Axis(0), Slice::from(to..));
            res += eval_est(est, &fold_columns, &fold_target, &val_columns, &val_target);
        }
    }
    // res / (cv as D)
    res
}