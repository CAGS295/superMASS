#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

use mass::mass_batch;
use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn mass_rs<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "batched")]
    fn mass_batched<'py>(
        py: Python<'py>,
        ts: PyReadonlyArrayDyn<f64>,
        query: PyReadonlyArrayDyn<f64>,
        batch_size: usize,
        top_matches: usize,
        jobs: usize,
    ) -> (&'py PyArray1<usize>, &'py PyArray1<f64>) {
        let ts = ts.as_slice().unwrap();
        let query = query.as_slice().unwrap();
        let a = mass_batch(ts, query, batch_size, top_matches, jobs);
        let (x, y): (Vec<_>, Vec<_>) = a.iter().cloned().unzip();

        (x.into_pyarray(py), y.into_pyarray(py))
    }

    Ok(())
}

#[pymodule]
fn examples(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
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
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        axpy(a, x, y).into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
        Ok(())
    }

    Ok(())
}
