use crate::add_subpackage;
use pyo3::prelude::*;

mod io;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "structure")?;
    add_subpackage(&module, &io::module(&module)?, "biotite.rust.structure.io")?;
    Ok(module)
}
