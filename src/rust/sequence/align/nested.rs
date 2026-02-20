use std::ops::{Index, IndexMut};

pub struct NestedArray<T: Default> {
    data: Vec<T>,
    offsets: Vec<usize>,
}

impl<T: Default> NestedArray<T> {
    /// Create a new `NestedArray` with ``lengths.len()`` outer arrays, where each inner array has
    /// the length ``lengths[i]``.
    ///
    /// Parameters
    /// ----------
    /// lengths
    ///     The lengths of the inner arrays.
    ///     It is recommended that ``lengths.capacity()`` is at least as large as
    ///     ``lengths.len() + 1`` to avoid reallocations during construction.
    ///
    /// Returns
    /// -------
    /// The new `NestedArray` filled with default values.
    pub fn new(lengths: Vec<usize>) -> Self {
        // As the 'lengths' vector may become quite large (its main use case is storing k-mers),
        // reuse the 'lengths' vector to reduce memory usage
        let mut offsets = lengths;
        let mut current_offset: usize = 0;
        for offset in &mut offsets {
            let length = *offset;
            *offset = current_offset;
            current_offset += length;
        }
        // Add exclusive end index
        offsets.push(current_offset);

        let mut data = Vec::new();
        data.resize_with(current_offset, T::default);
        Self { data, offsets }
    }
}

impl<T: Default> Index<usize> for NestedArray<T> {
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.offsets[index]..self.offsets[index + 1]]
    }
}

impl<T: Default> IndexMut<usize> for NestedArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[self.offsets[index]..self.offsets[index + 1]]
    }
}
