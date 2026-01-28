use pyo3::prelude::*;

pub mod kmertable;
pub mod nested;

use kmertable::*;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "align")?;
    module.add_class::<KmerTable>()?;
    Ok(module)
}
