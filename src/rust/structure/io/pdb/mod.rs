use pyo3::prelude::*;

pub mod file;
pub mod hybrid36;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "pdb")?;
    module.add_class::<file::PDBFile>()?;
    module.add_function(wrap_pyfunction!(hybrid36::encode_hybrid36, &module)?)?;
    module.add_function(wrap_pyfunction!(hybrid36::decode_hybrid36, &module)?)?;
    module.add_function(wrap_pyfunction!(hybrid36::max_hybrid36_number, &module)?)?;
    Ok(module)
}
