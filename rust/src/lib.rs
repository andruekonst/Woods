#![deny(rust_2018_idioms)]
mod numerics;
mod estimator;
mod ensemble;
mod rule;
mod tree;
mod boosting;
mod deep_boosting;
mod serialization;

use crate::estimator::{Estimator, ConstructibleWithRcArg, ConstructibleWithArg};
use crate::rule::{DecisionRuleImpl, SplitRule};
use crate::tree::{TreeParameters, DecisionTreeImpl};
use crate::boosting::{GradientBoostingParameters, GradientBoostingImpl, TreeGBM};
use crate::deep_boosting::{DeepBoostingParameters, DeepBoostingImpl};
use crate::ensemble::AverageEnsemble;
use crate::numerics::D as DType;
use crate::serialization::{load, save};

use ndarray::{ArrayView2, Array2};
use numpy::{IntoPyArray, PyArray2, PyArray1};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python, pyclass, pymethods, PyObject};

use std::rc::Rc;

fn to_columns<D: numpy::types::TypeNum>(x: &PyArray2<D>) -> Array2<D> {
    let arr: ArrayView2<'_, D> = x.as_array();
    arr.t().to_owned()
}

#[pyclass(module="woods")]
pub struct DecisionRule {
    rule: DecisionRuleImpl
}

#[pymethods]
impl DecisionRule {
    #[new]
    fn new() -> Self {
        DecisionRule {
            rule: DecisionRuleImpl::new()
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
    tree: DecisionTreeImpl<DecisionRuleImpl>
}

#[pymethods]
impl DecisionTree {
    #[new]
    fn new(depth: Option<u8>, min_samples_split: Option<usize>) -> Self {
        let params = TreeParameters::new(depth, min_samples_split);
        DecisionTree {
            tree: DecisionTreeImpl::new(Rc::new(params))
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
    gbm: GradientBoostingImpl<DecisionTreeImpl<DecisionRuleImpl>, TreeParameters>
}

#[pymethods]
impl GradientBoosting {
    #[new]
    fn new(depth: Option<u8>, min_samples_split: Option<usize>, n_estimators: Option<u32>,
           learning_rate: Option<DType>) -> Self {
        let est_params = TreeParameters::new(depth, min_samples_split);
        let params = GradientBoostingParameters::new(est_params, n_estimators, learning_rate);
        GradientBoosting {
            gbm: GradientBoostingImpl::new(Rc::new(params))
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