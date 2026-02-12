use crate::add_subpackage;
use pyo3::prelude::*;

mod celllist;
mod io;

use celllist::*;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "structure")?;
    module.add_class::<CellList>()?;
    module.add_class::<CellListResult>()?;
    add_subpackage(&module, &io::module(&module)?, "biotite.rust.structure.io")?;
    Ok(module)
}
