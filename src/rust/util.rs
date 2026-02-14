use pyo3::prelude::*;

/// Check for Python interrupts periodically to allow breaking long-running loops.
///
/// This should be called inside loops that may take a long time. It checks for
/// signals every 256 iterations to avoid the overhead of checking on every iteration.
#[inline(always)]
pub fn check_signals_periodically(py: Python<'_>, iteration: usize) -> PyResult<()> {
    if iteration & 0xFF == 0 {
        py.check_signals()?;
    }
    Ok(())
}
