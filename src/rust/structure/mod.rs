use crate::add_subpackage;
use pyo3::prelude::*;

pub mod celllist;
pub mod io;
pub mod sasa;
pub mod util;

use celllist::*;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "structure")?;
    module.add_class::<CellList>()?;
    module.add_class::<CellListResult>()?;
    module.add_function(pyo3::wrap_pyfunction!(sasa::sasa, &module)?)?;
    add_subpackage(&module, &io::module(&module)?, "biotite.rust.structure.io")?;
    Ok(module)
}
