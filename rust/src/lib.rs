#![deny(rust_2018_idioms)]
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, ArrayView2, Array2};
use numpy::{IntoPyArray, PyArrayDyn, PyArray2, PyArray1};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python, pyclass, pymethods, PyObject};
use pyo3::wrap_pyfunction;
use std::vec;

mod rule;
mod tree;
use crate::rule::DecisionRuleImpl;
use crate::tree::{TreeParameters, DecisionTreeImpl};

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

#[pyclass(module="woods")]
pub struct DecisionTree {
    tree: DecisionTreeImpl<DecisionRuleImpl>
}

#[pymethods]
impl DecisionTree {
    #[new]
    fn new(depth: u8, min_samples_split: usize) -> Self {
        let params = TreeParameters {
            depth: depth,
            min_samples_split: min_samples_split
        };
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

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
