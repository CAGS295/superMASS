#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

use mass::mass_batch;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
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
    ) -> (&'py PyArray1<usize>, &'py PyArray1<f64>) {
        let ts = ts.as_slice().unwrap();
        let query = query.as_slice().unwrap();
        let a = mass_batch(ts, query, batch_size, top_matches);
        let (x, y): (Vec<_>, Vec<_>) = a.iter().cloned().unzip();

        (x.into_pyarray(py), y.into_pyarray(py))
    }

    #[pyfn(m, "mass")]
    fn mass<'py>(
        py: Python<'py>,
        ts: PyReadonlyArrayDyn<f64>,
        query: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArray1<f64> {
        let ts = ts.as_slice().unwrap();
        let query = query.as_slice().unwrap();
        let a = mass::mass(ts, query);
        a.into_pyarray(py)
    }

    Ok(())
}
