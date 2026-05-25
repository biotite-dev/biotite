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

/// Issue a Python warning via `warnings.warn()`.
///
/// The warning type is given as a generic parameter.
pub fn warn<T: pyo3::type_object::PyTypeInfo>(py: Python<'_>, message: &str) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    warnings.call_method1("warn", (message, py.get_type::<T>()))?;
    Ok(())
}

/// Dispatch a generic function call based on a NumPy dtype descriptor.
///
/// Accepts a custom type list.
/// Arguments are captured as a single token tree (the parenthesized group),
/// which avoids the `$T repeats N times but $arg repeats M times` error.
/// The inner function's return value is erased to `Bound<'py, PyAny>` via
/// `.into_any()` so that all branches have a uniform type.
///
/// # Usage
///
/// ```ignore
/// dispatch_dtype!(py, dtype, [i8, i16, i32, f32, f64], inner_fn(py, data))
/// ```
#[macro_export]
macro_rules! dispatch_dtype {
    ($py:expr, $dtype:expr, [$($T:ty),+], $func:ident $args:tt) => {{
        if false { unreachable!() }
        $(
            else if $dtype.is_equiv_to(&numpy::dtype::<$T>($py)) {
                $func::<$T> $args .map(|v| v.into_any())
            }
        )+
        else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported dtype: {}", $dtype
            )))
        }
    }};
}

/// Dispatch a generic function call over the cartesian product of two dtype
/// descriptors.
///
/// The inner function is called as `$func::<T1, T2>(args)` for the matching
/// pair of types. Uses an internal `@inner` arm to avoid nesting two
/// repetition levels of independently captured type lists.
///
/// # Usage
///
/// ```ignore
/// dispatch_dtypes!(
///     py,
///     in_dtype,  [i8, i16, i32],
///     out_dtype, [u8, u16, u32],
///     inner_fn(py, data, length)
/// )
/// ```
#[macro_export]
macro_rules! dispatch_dtypes {
    (
        $py:expr,
        $dtype1:expr, [$($T1:ty),+],
        $dtype2:expr, [$($T2:ty),+],
        $func:ident $args:tt
    ) => {{
        if false { unreachable!() }
        $(
            else if $dtype1.is_equiv_to(&numpy::dtype::<$T1>($py)) {
                dispatch_dtypes!(@inner $py, $dtype2, [$($T2),+], $func, [$T1] $args)
            }
        )+
        else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported dtype: {}", $dtype1
            )))
        }
    }};
    (@inner $py:expr, $dtype2:expr, [$($T2:ty),+], $func:ident, [$T1:ty] $args:tt) => {{
        if false { unreachable!() }
        $(
            else if $dtype2.is_equiv_to(&numpy::dtype::<$T2>($py)) {
                $func::<$T1, $T2> $args .map(|v| v.into_any())
            }
        )+
        else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported dtype: {}", $dtype2
            )))
        }
    }};
}
