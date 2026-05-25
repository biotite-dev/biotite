use crate::add_subpackage;
use pyo3::prelude::*;

pub mod pdb;
pub mod pdbx;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "io")?;
    add_subpackage(
        &module,
        &pdb::module(&module)?,
        "biotite.rust.structure.io.pdb",
    )?;
    add_subpackage(
        &module,
        &pdbx::module(&module)?,
        "biotite.rust.structure.io.pdbx",
    )?;
    Ok(module)
}
