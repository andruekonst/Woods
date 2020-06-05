#![deny(rust_2018_idioms)]
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, ArrayView2, Array2};
use numpy::{IntoPyArray, PyArrayDyn, PyArray2, PyArray1};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python, pyclass, pymethods, PyObject, PyErr};
use pyo3::wrap_pyfunction;
use std::vec;

mod rule;
mod tree;
mod boosting;
mod deep_boosting;
use crate::rule::DecisionRuleImpl;
use crate::tree::{TreeParameters, DecisionTreeImpl};
use crate::boosting::{GradientBoostingParameters, GradientBoostingImpl, TreeGBM};
use crate::deep_boosting::{DeepBoostingParameters, DeepBoostingImpl, AverageEnsemble};
use std::rc::Rc;
use std::fs::File;
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;

type DType = f64;

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

fn save<T: Serialize>(what: &T, filename: &str) -> PyResult<()> {
    let mut file = File::create(filename)?;
    serde_json::to_writer(file, what).unwrap();
    Ok(())
}

fn load<T: DeserializeOwned>(what: &mut T, filename: &str) -> PyResult<()> {
    let file = File::open(filename)?;
    *what = serde_json::from_reader::<_, T>(file).unwrap();
    Ok(())
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

    fn save(&self, filename: &str) -> PyResult<()> {
        save(&self.tree, filename)
    }

    fn load(&mut self, filename: &str) -> PyResult<()> {
        load(&mut self.tree, filename)
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

    fn save(&self, filename: &str) -> PyResult<()> {
        save(&self.gbm, filename)
    }

    fn load(&mut self, filename: &str) -> PyResult<()> {
        load(&mut self.gbm, filename)
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

    // fn save(&self, filename: &str) -> PyResult<()> {
    //     save(&self.dgbm, filename)
    // }

    // fn load(&mut self, filename: &str) -> PyResult<()> {
    //     load(&mut self.dgbm, filename)
    // }
}


#[pymodule]
fn woods(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // wrapper of `axpy`
    #[pyfn(m, "axpy")]
    fn axpy_py(
        py: Python<'_>,
        a: f64,
        x: &PyArrayDyn<f64>,
        y: &PyArrayDyn<f64>,
    ) -> Py<PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        axpy(a, x, y).into_pyarray(py).to_owned()
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = x.as_array_mut();
        mult(a, x);
        Ok(())
    }

    type NumberType = f64;

    m.add_class::<DecisionRule>()?;
    m.add_class::<DecisionTree>()?;
    m.add_class::<GradientBoosting>()?;
    m.add_class::<DeepGradientBoosting>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
