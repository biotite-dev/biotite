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
    for (i, line) in lines.iter().enumerate() {
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
    for (i, line) in lines.iter().enumerate() {
        if is_empty(line) {
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

/// Split a line into its values.
///
/// Supports embedded quotes (' or "), like ``'a dog's life'`` to ``a dog's life``.
#[pyfunction]
pub fn split_one_line(line: &str) -> Vec<&str> {
    let mut values = Vec::new();
    split_values(line, |value| values.push(value));
    values
}

/// Process a category where each field has a single value.
fn deserialize_single<'py>(py: Python<'py>, lines: &[Cow<'_, str>]) -> PyResult<Columns<'py>> {
    let mut columns = Vec::new();
    let mut line_i = 0;
    while line_i < lines.len() {
        let line = &lines[line_i];
        let parts = split_one_line(line);
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
                let [value_part] = split_one_line(next_line)[..] else {
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

    // First pass: count the values and find the widest value of each column
    let mut lengths = vec![0; column_names.len()];
    let mut widths = vec![1; column_names.len()];
    let key_i = for_each_value(data_lines, &column_of_key, |column_i, value| {
        lengths[column_i] += 1;
        widths[column_i] = widths[column_i].max(value.chars().count());
    });
    // Check if all columns have the same length
    // Otherwise, this would indicate a parsing error or an invalid CIF file
    if key_i != 0 {
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
    for_each_value(data_lines, &column_of_key, |column_i, value| {
        let start = row_counts[column_i] * widths[column_i];
        write_code_points(
            &mut buffers[column_i][start..start + widths[column_i]],
            value,
        );
        row_counts[column_i] += 1;
    });

    column_names
        .into_iter()
        .zip(buffers)
        .zip(widths)
        .map(|((name, buffer), width)| Ok((name.to_string(), unicode_array(py, buffer, width)?)))
        .collect()
}

/// Call `f(column_i, value)` for each value in the data lines of a looped category.
///
/// Rows may be split over multiple lines -> do not rely on
/// row-line-alignment at all and simply cycle through columns.
/// Return the key index after the last value, which is 0 if all rows are complete.
fn for_each_value<'a>(
    data_lines: &'a [Cow<'_, str>],
    column_of_key: &[usize],
    mut f: impl FnMut(usize, &'a str),
) -> usize {
    let mut key_i = 0;
    for data_line in data_lines {
        split_values(data_line, |value| {
            f(column_of_key[key_i], value);
            key_i = (key_i + 1) % column_of_key.len();
        });
    }
    key_i
}

/// Call `emit` for each value in the line.
fn split_values<'a>(line: &'a str, mut emit: impl FnMut(&'a str)) {
    // Special case of multiline value, where the line starts with ';'
    if let Some(value) = line.strip_prefix(';') {
        emit(value);
    } else if line.contains(['\'', '"']) {
        // Quoted values in the line
        let mut line = line;
        while !line.is_empty() {
            let stripped_line = line.trim_start_matches(is_space);
            // Split the line on whitespace
            let (mut word, mut rest) = partition(stripped_line, ' ');
            // Handle the case where the word starts with a quote
            if let Some(separator) = word.chars().next().filter(|&c| c == '\'' || c == '"') {
                // Handle the case of a quoted word without space
                if word.len() > 1 && word.ends_with(separator) {
                    // Emit the word without the opening and closing quotes
                    emit(&word[1..word.len() - 1]);
                    line = rest;
                    continue;
                }
                (word, rest) = partition(&stripped_line[1..], separator);
            }
            emit(word);
            line = rest;
        }
    } else {
        // No quoted values in the line -> simple whitespace split
        line.split(is_space)
            .filter(|value| !value.is_empty())
            .for_each(emit);
    }
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

/// Equivalent to Python's ``str.partition()`` with a single character separator.
fn partition(text: &str, separator: char) -> (&str, &str) {
    match text.find(separator) {
        Some(i) => (&text[..i], &text[i + separator.len_utf8()..]),
        None => (text, ""),
    }
}

fn is_empty(line: &str) -> bool {
    line.trim_matches(is_space).is_empty() || line.starts_with('#')
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
