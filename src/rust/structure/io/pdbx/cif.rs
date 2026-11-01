use numpy::{IntoPyArray, PyArrayDescr};
use pyo3::prelude::*;
use pyo3::types::PyString;
use std::borrow::Cow;

// Label as a separate module to indicate that this exception comes
// from biotite
mod biotite {
    pyo3::import_exception!(biotite.file, DeserializationError);
}

type Columns<'py> = Vec<(String, Bound<'py, PyAny>)>;

/// Split the text of a CIF file into its data blocks.
///
/// Return the name and text of each block, in the order of the file.
#[pyfunction]
pub fn split_blocks(text: &Bound<'_, PyString>) -> PyResult<Vec<(String, String)>> {
    let lines = split_lines(to_str(text)?);
    let mut block_names = Vec::new();
    let mut block_starts = Vec::new();
    let mut text_fields = TextFieldTracker::default();
    for (i, line) in lines.iter().enumerate() {
        if text_fields.is_within(line) {
            continue;
        }
        if let Some(block_name) = line.strip_prefix("data_") {
            block_starts.push(i);
            block_names.push(block_name.to_string());
        }
    }
    Ok(join_elements(&lines, block_names, &block_starts))
}

/// Split the text of a CIF block into its categories.
///
/// Return the name and text of each category, in the order of the block.
#[pyfunction]
pub fn split_categories(text: &Bound<'_, PyString>) -> PyResult<Vec<(String, String)>> {
    let lines = split_lines(to_str(text)?);
    let mut current_category_name = None;
    let mut category_names = Vec::new();
    let mut category_starts = Vec::new();
    let mut text_fields = TextFieldTracker::default();
    for (i, line) in lines.iter().enumerate() {
        if text_fields.is_within(line) || is_empty(line) {
            continue;
        }
        let category_name = if line.starts_with("loop_") {
            // In case of lines with "loop_" the category is
            // in the next data name, which may follow after comments
            let name_line = lines[i + 1..]
                .iter()
                .find(|line| !is_empty(line))
                .ok_or_else(missing_loop_data_name)?;
            parse_category_name(name_line)?.ok_or_else(missing_loop_data_name)?
        } else {
            match parse_category_name(line)? {
                Some(name) if Some(name) != current_category_name => name,
                _ => continue,
            }
        };
        // Track the new category
        current_category_name = Some(category_name);
        category_starts.push(i);
        category_names.push(category_name.to_string());
    }
    Ok(join_elements(&lines, category_names, &category_starts))
}

/// Deserialize the text of a CIF category into its name and columns.
///
/// Each column is a NumPy unicode array, as ``np.asarray()`` would create it from
/// the list of value strings.
#[pyfunction]
pub fn deserialize_category<'py>(
    py: Python<'py>,
    text: &Bound<'py, PyString>,
) -> PyResult<(String, Columns<'py>)> {
    let mut lines: Vec<&str> = split_lines(to_str(text)?)
        .into_iter()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.trim_matches(is_space))
        .filter(|line| !line.is_empty())
        .collect();
    let first_line = *lines
        .first()
        .ok_or_else(|| biotite::DeserializationError::new_err("Category is empty"))?;
    let is_looped = first_line.starts_with("loop_");
    if is_looped {
        lines.remove(0);
    }
    let name_line = lines.first().ok_or_else(missing_loop_data_name)?;
    let category_name = parse_category_name(name_line)?
        .ok_or_else(|| biotite::DeserializationError::new_err("Failed to parse category name"))?;

    let lines = to_single(&lines);
    let columns = if is_looped {
        deserialize_looped(py, &lines)?
    } else {
        deserialize_single(py, &lines)?
    };
    Ok((category_name.to_string(), columns))
}

/// Process a category where each field has a single value.
fn deserialize_single<'py>(py: Python<'py>, lines: &[Cow<'_, str>]) -> PyResult<Columns<'py>> {
    let mut columns = Vec::new();
    let mut line_i = 0;
    while line_i < lines.len() {
        let line = &lines[line_i];
        let parts: Vec<&str> = split_values(line).collect();
        let (name_part, value_part) = match parts[..] {
            // Standard case -> name and value in one line
            [name_part, value_part] => {
                line_i += 1;
                (name_part, value_part)
            }
            // Value is a multiline value on the next line
            [name_part] => {
                let next_line = lines.get(line_i + 1).ok_or_else(|| {
                    biotite::DeserializationError::new_err(format!(
                        "Missing value for data name '{name_part}'"
                    ))
                })?;
                let [value_part] = split_values(next_line).collect::<Vec<_>>()[..] else {
                    return Err(parse_error(line));
                };
                line_i += 2;
                (name_part, value_part)
            }
            [] => {
                return Err(biotite::DeserializationError::new_err(
                    "Empty line within category",
                ))
            }
            _ => return Err(parse_error(line)),
        };
        let width = value_part.chars().count().max(1);
        let mut buffer = vec![0; width];
        write_code_points(&mut buffer, value_part);
        // A repeated key is resolved by the `dict()` call on the Python side
        columns.push((
            split_data_name(name_part)?.1.to_string(),
            unicode_array(py, buffer, width)?,
        ));
    }
    Ok(columns)
}

/// Process a category where each field has multiple values
/// (category is a table).
fn deserialize_looped<'py>(py: Python<'py>, lines: &[Cow<'_, str>]) -> PyResult<Columns<'py>> {
    let mut column_names = Vec::new();
    // Maps the position of a key line to its column,
    // which differ only if a column name is repeated
    let mut column_of_key = Vec::new();
    for key_line in lines.iter().take_while(|line| line.starts_with('_')) {
        let (_, key) = split_data_name(key_line)?;
        let column_i = match column_names.iter().position(|name| *name == key) {
            Some(i) => i,
            None => {
                column_names.push(key);
                column_names.len() - 1
            }
        };
        column_of_key.push(column_i);
    }
    let data_lines = &lines[column_of_key.len()..];
    // Rows may be split over multiple lines -> do not rely on
    // row-line-alignment at all and simply cycle through columns
    let values = || {
        column_of_key
            .iter()
            .cycle()
            .zip(data_lines.iter().flat_map(|line| split_values(line)))
    };

    // First pass: count the values and find the widest value of each column
    let mut lengths = vec![0; column_names.len()];
    let mut widths = vec![1; column_names.len()];
    for (&column_i, value) in values() {
        lengths[column_i] += 1;
        widths[column_i] = widths[column_i].max(value.chars().count());
    }
    // Check if all columns have the same length
    // Otherwise, this would indicate a parsing error or an invalid CIF file
    if lengths.iter().sum::<usize>() % column_of_key.len() != 0 {
        return Err(biotite::DeserializationError::new_err(
            "Category contains columns with different lengths",
        ));
    }

    // Second pass: write each value directly into its array,
    // so the values do not need to be kept until all widths are known
    let mut buffers: Vec<Vec<u32>> = lengths
        .iter()
        .zip(&widths)
        .map(|(length, width)| vec![0; length * width])
        .collect();
    let mut row_counts = vec![0; column_names.len()];
    for (&column_i, value) in values() {
        let start = row_counts[column_i] * widths[column_i];
        write_code_points(
            &mut buffers[column_i][start..start + widths[column_i]],
            value,
        );
        row_counts[column_i] += 1;
    }

    column_names
        .into_iter()
        .zip(buffers)
        .zip(widths)
        .map(|((name, buffer), width)| Ok((name.to_string(), unicode_array(py, buffer, width)?)))
        .collect()
}

/// Iterate over the values in a line.
///
/// Supports embedded quotes (' or "), like ``'a dog's life'`` to ``a dog's life``:
/// a quote only closes a quoted value, if it is followed by whitespace or the end
/// of the line.
/// A comment, i.e. an unquoted ``#``, ends the line.
fn split_values(line: &str) -> impl Iterator<Item = &str> {
    // Special case of multiline value, where the line starts with ';'
    let mut text_field = line.strip_prefix(';');
    let mut rest = if text_field.is_some() { "" } else { line };
    std::iter::from_fn(move || {
        if let Some(value) = text_field.take() {
            return Some(value);
        }
        rest = rest.trim_start_matches(is_space);
        let value;
        match rest.chars().next()? {
            '#' => {
                rest = "";
                return None;
            }
            quote @ ('\'' | '"') => {
                let content = &rest[1..];
                // If the quoted value is not closed, it extends to the end of the line
                let end = content
                    .match_indices(quote)
                    .map(|(i, _)| i)
                    .find(|&i| content[i + 1..].chars().next().is_none_or(is_space))
                    .unwrap_or(content.len());
                value = &content[..end];
                rest = content.get(end + 1..).unwrap_or("");
            }
            _ => {
                let end = rest.find(is_space).unwrap_or(rest.len());
                value = &rest[..end];
                rest = &rest[end..];
            }
        }
        Some(value)
    })
}

/// Convert multiline values into singleline values
/// (in terms of 'lines' list elements).
/// Linebreaks are preserved as ``'\n'`` characters within a list element.
/// The initial ``';'`` character is also preserved, while the final ``';'`` character
/// is removed.
fn to_single<'a>(lines: &[&'a str]) -> Vec<Cow<'a, str>> {
    let mut processed_lines = Vec::new();
    let mut multi_line_value: Option<Vec<&str>> = None;
    for &line in lines {
        // Multiline value are enclosed by ';' at the start of the beginning and end line
        if line.starts_with(';') {
            match multi_line_value.take() {
                // Start of multiline value
                None => multi_line_value = Some(vec![line]),
                // End of multiline value
                // The current line contains only the end character ';'
                // Hence this line is not added to the processed lines
                Some(value_lines) => processed_lines.push(Cow::Owned(value_lines.join("\n"))),
            }
        } else if let Some(value_lines) = multi_line_value.as_mut() {
            value_lines.push(line);
        } else {
            processed_lines.push(Cow::Borrowed(line));
        }
    }
    processed_lines
}

/// Tracks whether a line is part of a text field, i.e. a value delimited by
/// ``;`` at the start of a line.
#[derive(Default)]
struct TextFieldTracker {
    is_open: bool,
}

impl TextFieldTracker {
    /// Return whether the line is a delimiter or the content of a text field.
    ///
    /// Must be called for each line in order.
    /// As a text field may not contain a ``;`` at the start of a line,
    /// each such line toggles between the start and the end of a text field.
    fn is_within(&mut self, line: &str) -> bool {
        if line.starts_with(';') {
            self.is_open = !self.is_open;
            return true;
        }
        self.is_open
    }
}

/// Join the lines of each element (block or category) into its text, where element `i`
/// spans from ``starts[i]`` to the start of the next element.
fn join_elements<N>(lines: &[&str], names: Vec<N>, starts: &[usize]) -> Vec<(N, String)> {
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            let stop = starts.get(i + 1).copied().unwrap_or(lines.len());
            let mut text = lines[starts[i]..stop].join("\n");
            text.push('\n');
            (name, text)
        })
        .collect()
}

/// Write the code points of `value` into the start of `row`.
/// The rest of `row` keeps its zeros, which NumPy reads as padding.
fn write_code_points(row: &mut [u32], value: &str) {
    for (code_point, c) in row.iter_mut().zip(value.chars()) {
        *code_point = c as u32;
    }
}

/// View a buffer of UCS4 code points as a NumPy unicode array with the given width,
/// without copying.
fn unicode_array<'py>(
    py: Python<'py>,
    buffer: Vec<u32>,
    width: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype = PyArrayDescr::new(py, format!("U{width}"))?;
    buffer.into_pyarray(py).call_method1("view", (dtype,))
}

/// Borrow the text of a Python string.
fn to_str<'a>(text: &'a Bound<'_, PyString>) -> PyResult<&'a str> {
    // Fails only for lone surrogates, e.g. from a file opened with
    // `errors="surrogateescape"`
    text.to_str().map_err(|_| {
        biotite::DeserializationError::new_err(
            "CIF text contains lone surrogate characters, which are not valid Unicode",
        )
    })
}

/// Equivalent to Python's ``str.isspace()`` for a single character.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// Equivalent to Python's ``str.splitlines()``.
fn split_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if matches!(
            c,
            '\n' | '\r'
                | '\x0b'
                | '\x0c'
                | '\x1c'
                | '\x1d'
                | '\x1e'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        ) {
            lines.push(&text[start..i]);
            start = i + c.len_utf8();
            if c == '\r' && chars.next_if(|&(_, next)| next == '\n').is_some() {
                start += 1;
            }
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

fn is_empty(line: &str) -> bool {
    let line = line.trim_start_matches(is_space);
    line.is_empty() || line.starts_with('#')
}

/// If the line starts with a data name, return its category name.
/// Return ``None`` otherwise.
fn parse_category_name(line: &str) -> PyResult<Option<&str>> {
    // Leading whitespace is allowed, e.g. in YASARA files
    let line = line.trim_start_matches(is_space);
    if !line.starts_with('_') {
        return Ok(None);
    }
    let data_name = line.split(is_space).next().unwrap_or(line);
    split_data_name(data_name).map(|(category_name, _)| Some(category_name))
}

/// Split a data name of the form ``_category.column`` into the category and column name.
fn split_data_name(data_name: &str) -> PyResult<(&str, &str)> {
    data_name
        .strip_prefix('_')
        .and_then(|name| name.split_once('.'))
        .ok_or_else(|| {
            biotite::DeserializationError::new_err(format!(
                "'{data_name}' is not a data name of the form '_category.column'"
            ))
        })
}

fn parse_error(line: &str) -> PyErr {
    biotite::DeserializationError::new_err(format!("Failed to parse line '{line}'"))
}

fn missing_loop_data_name() -> PyErr {
    biotite::DeserializationError::new_err("'loop_' is not followed by a data name")
}
