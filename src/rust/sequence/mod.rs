use crate::add_subpackage;
use pyo3::prelude::*;

mod align;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "sequence")?;
    add_subpackage(
        &module,
        &align::module(&module)?,
        "biotite.rust.sequence.align",
    )?;
    Ok(module)
}
