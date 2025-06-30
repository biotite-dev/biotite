use pyo3::prelude::*;

mod file;

use file::*;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "structure")?;
    module.add_class::<PDBFile>()?;
    Ok(module)
}
