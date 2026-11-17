use pyo3::prelude::*;

pub mod cif;
pub mod encoding;
pub mod matching;

pub fn module<'py>(parent_module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
    let module = PyModule::new(parent_module.py(), "pdbx")?;
    module.add_function(wrap_pyfunction!(matching::find_matches, &module)?)?;
    module.add_function(wrap_pyfunction!(cif::deserialize_category, &module)?)?;
    module.add_function(wrap_pyfunction!(cif::split_blocks, &module)?)?;
    module.add_function(wrap_pyfunction!(cif::split_categories, &module)?)?;
    module.add_function(wrap_pyfunction!(encoding::run_length_encode, &module)?)?;
    module.add_function(wrap_pyfunction!(encoding::run_length_decode, &module)?)?;
    module.add_function(wrap_pyfunction!(encoding::integer_packing_encode, &module)?)?;
    module.add_function(wrap_pyfunction!(encoding::integer_packing_decode, &module)?)?;
    Ok(module)
}
