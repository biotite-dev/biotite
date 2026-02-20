use pyo3::prelude::*;
use pyo3::types::PyDict;

mod sequence;
mod structure;
pub mod util;

/// Add a submodule to a module and make it discoverable as package
fn add_subpackage(
    module: &Bound<'_, PyModule>,
    submodule: &Bound<'_, PyModule>,
    submodule_name: &str,
) -> PyResult<()> {
    module.add_submodule(submodule)?;
    let sys = PyModule::import(module.py(), "sys")?;
    let sys_modules: Bound<'_, PyDict> = sys.getattr("modules")?.cast_into()?;
    submodule.setattr("__name__", submodule_name)?;
    submodule.setattr("__package__", submodule_name)?;
    sys_modules.set_item(submodule_name, submodule)?;
    Ok(())
}

#[pymodule]
fn rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    add_subpackage(
        module,
        &structure::module(module)?,
        "biotite.rust.structure",
    )?;
    add_subpackage(module, &sequence::module(module)?, "biotite.rust.sequence")?;
    Ok(())
}
