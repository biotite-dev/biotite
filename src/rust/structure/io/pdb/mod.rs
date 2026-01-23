use pyo3::prelude::*;

pub mod file;
pub mod hybrid36;

use file::*;
use hybrid36::*;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "structure")?;
    module.add_class::<PDBFile>()?;
    module.add_function(wrap_pyfunction!(encode_hybrid36, &module)?)?;
    module.add_function(wrap_pyfunction!(decode_hybrid36, &module)?)?;
    module.add_function(wrap_pyfunction!(max_hybrid36_number, &module)?)?;
    Ok(module)
}
