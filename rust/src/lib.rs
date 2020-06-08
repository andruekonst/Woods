// #![deny(rust_2018_idioms)]

//! Woods - Decision Tree Ensembles implementation and a Python extension.
//! 
//! All estimators use [`ndarray`] as a tensor framework.
//! 
//! # Installation of Python extension
//! 1. Change toolchain to nightly (required by `pyo3`)
//! ```
//! > rustup toolchain install nightly
//! > rustup default nightly
//! ```
//! 2. Run `setup.py` to build / install extension
//! ```
//! > python setup.py install
//! ``` 
//! 
//! For *Windows* platform it is recommended to use [Anaconda](http://anaconda.com).
//! 
//! # Usage examples
//! 1. Train, save, load and predict with [`DeepGradientBoosting`] model.
//! ```
//! import woods
//! from sklearn.datasets import load_boston
//! from sklearn.model_selection import train_test_split
//! from sklearn.metrics import r2_score
//! 
//! # prepare data
//! X, y = load_boston(return_X_y=True)
//! X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
//! 
//! # make model
//! model = woods.DeepGradientBoosting(n_estimators=5)
//! # fit model on train data
//! model.fit(X_train, y_train)
//! 
//! # evaluate quality on test data
//! print("r^2 score:", r2_score(y_test, model.predict(X_test)))
//! 
//! # save JSON representation of model to filesystem (optional step)
//! model.save("my_deep_gbm.json", format="json")
//! # clean up memory
//! del model
//! 
//! # make empty model and load contents from JSON file
//! loaded_model = woods.DeepGradientBoosting()
//! loaded_model.load("my_deep_gbm.json", format="json")
//! 
//! # evaluate quality of loaded model on test data
//! print("r^2 score:", r2_score(y_test, loaded_model.predict(X_test)))
//! ```

pub mod estimator;
pub mod ensemble;
// pub mod rule;
pub mod tree;
// pub mod boosting;
// pub mod deep_boosting;
pub mod utils;

use crate::estimator::{Estimator, ConstructibleWithCopyArg, ConstructibleWithArg};
use tree::rule::{RandomSplitRule, SplitRule};
use crate::tree::{TreeParameters, DecisionTreeImpl};
use crate::ensemble::boosting::{GradientBoostingParameters, GradientBoostingImpl, TreeGBM};
use crate::ensemble::deep_boosting::{DeepBoostingParameters, DeepBoostingImpl};
use crate::ensemble::AverageEnsemble;
use utils::numerics::D as DType;
use utils::serialization::{load, save};

use ndarray::{ArrayView2, Array2};
use numpy::{IntoPyArray, PyArray2, PyArray1};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python, pyclass, pymethods, PyObject};


fn to_columns<D: numpy::types::TypeNum>(x: &PyArray2<D>) -> Array2<D> {
    let arr: ArrayView2<'_, D> = x.as_array();
    arr.t().to_owned()
}

#[pyclass(module="woods")]
pub struct DecisionRule {
    rule: RandomSplitRule
}

#[pymethods]
impl DecisionRule {
    #[new]
    fn new() -> Self {
        DecisionRule {
            rule: RandomSplitRule::new()
        }
    }
    
    fn fit(&mut self, x: &PyArray2<DType>, y: &PyArray1<DType>) {
        let features = to_columns(x);
        let target = y.as_array();
        assert_eq!(features.dim().1, target.dim());
        self.rule.fit(&features.view(), &target);
    }

    fn predict(&self, py: Python<'_>, x: &PyArray2<DType>) -> Py<PyArray1<DType>> {
        let features = to_columns(x);
        self.rule.predict(&features.view()).into_pyarray(py).to_owned()
    }
}

#[pyclass(module="woods")]
pub struct DecisionTree {
    tree: DecisionTreeImpl<RandomSplitRule>
}

#[pymethods]
impl DecisionTree {
    #[new]
    fn new(depth: Option<u8>, min_samples_split: Option<usize>) -> Self {
        let params = TreeParameters::new(depth, min_samples_split);
        DecisionTree {
            tree: DecisionTreeImpl::new(params)
        }
    }
    
    fn fit(&mut self, x: &PyArray2<DType>, y: &PyArray1<DType>) {
        let features = to_columns(x);
        let target = y.as_array();
        assert_eq!(features.dim().1, target.dim());
        self.tree.fit(&features.view(), &target);
    }

    fn predict(&self, py: Python<'_>, x: &PyArray2<DType>) -> Py<PyArray1<DType>> {
        let features = to_columns(x);
        self.tree.predict(&features.view()).into_pyarray(py).to_owned()
    }

    fn save(&self, filename: &str, format: Option<&str>) -> PyResult<()> {
        save(&self.tree, filename, format)
    }

    fn load(&mut self, filename: &str, format: Option<&str>) -> PyResult<()> {
        load(&mut self.tree, filename, format)
    }
}

#[pyclass(module="woods")]
pub struct GradientBoosting {
    gbm: TreeGBM
}

#[pymethods]
impl GradientBoosting {
    #[new]
    fn new(depth: Option<u8>, min_samples_split: Option<usize>, n_estimators: Option<u32>,
           learning_rate: Option<DType>) -> Self {
        let est_params = TreeParameters::new(depth, min_samples_split);
        let params = GradientBoostingParameters::new(est_params, n_estimators, learning_rate);
        GradientBoosting {
            gbm: GradientBoostingImpl::new(params)
        }
    }
    
    fn fit(&mut self, x: &PyArray2<DType>, y: &PyArray1<DType>) {
        let features = to_columns(x);
        let target = y.as_array();
        assert_eq!(features.dim().1, target.dim());
        self.gbm.fit(&features.view(), &target);
    }

    fn predict(&self, py: Python<'_>, x: &PyArray2<DType>) -> Py<PyArray1<DType>> {
        let features = to_columns(x);
        self.gbm.predict(&features.view()).into_pyarray(py).to_owned()
    }

    fn save(&self, filename: &str, format: Option<&str>) -> PyResult<()> {
        save(&self.gbm, filename, format)
    }

    fn load(&mut self, filename: &str, format: Option<&str>) -> PyResult<()> {
        load(&mut self.gbm, filename, format)
    }
}

#[pyclass(module="woods")]
pub struct DeepGradientBoosting {
    dgbm: DeepBoostingImpl<AverageEnsemble<TreeGBM>>
}

#[pymethods]
impl DeepGradientBoosting {
    #[new]
    fn new(n_estimators: Option<u32>, layer_width: Option<u32>, learning_rate: Option<DType>) -> Self {
        let params = DeepBoostingParameters::new(n_estimators, layer_width, learning_rate);
        DeepGradientBoosting {
            dgbm: DeepBoostingImpl::new(params)
        }
    }
    
    fn fit(&mut self, x: &PyArray2<DType>, y: &PyArray1<DType>) {
        let features = to_columns(x);
        let target = y.as_array();
        assert_eq!(features.dim().1, target.dim());
        self.dgbm.fit(&features.view(), &target);
    }

    fn predict(&self, py: Python<'_>, x: &PyArray2<DType>) -> Py<PyArray1<DType>> {
        let features = to_columns(x);
        self.dgbm.predict(&features.view()).into_pyarray(py).to_owned()
    }

    fn save(&self, filename: &str, format: Option<&str>) -> PyResult<()> {
        save(&self.dgbm, filename, format)
    }

    fn load(&mut self, filename: &str, format: Option<&str>) -> PyResult<()> {
        load(&mut self.dgbm, filename, format)
    }
}


#[pymodule]
fn woods(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<DecisionRule>()?;
    m.add_class::<DecisionTree>()?;
    m.add_class::<GradientBoosting>()?;
    m.add_class::<DeepGradientBoosting>()?;

    Ok(())
}