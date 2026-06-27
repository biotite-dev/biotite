use std::ops::{Index, IndexMut};

/// A frozen container representing a nested array of fixed-length inner arrays.
///
/// All inner arrays are stored back-to-back in a single flat `data` buffer, with
/// an `offsets` array recording where each one begins and ends.
/// Use index operations to access inner arrays.
pub struct NestedArray<T: Copy> {
    /// The flat buffer holding every inner array consecutively; inner array `i`
    /// occupies ``data[offsets[i]..offsets[i + 1]]``.
    data: Vec<T>,
    /// The bounds of each inner array within `data`. Has length
    /// ``n_inner_arrays + 1``, is non-decreasing, starts at ``0`` and ends at
    /// ``data.len()``, so inner array `i` spans ``offsets[i]..offsets[i + 1]``.
    offsets: Vec<usize>,
}

impl<T: Copy> NestedArray<T> {
    /// Create a `NestedArray` directly from the flat `data` buffer and the
    /// `offsets` array, i.e. the inverse of `NestedArray::raw_parts`.
    ///
    /// Safety
    /// ------
    /// `offsets` must have length ``n_inner_arrays + 1``, be non-decreasing,
    /// start at ``0`` and end at ``data.len()``.
    pub unsafe fn from_raw_parts(data: Vec<T>, offsets: Vec<usize>) -> Self {
        debug_assert!(!offsets.is_empty());
        debug_assert_eq!(offsets[0], 0);
        debug_assert_eq!(*offsets.last().unwrap(), data.len());
        Self { data, offsets }
    }

    /// Borrow the flat data buffer and the offsets array, i.e. the inverse of
    /// `NestedArray::from_raw_parts`.
    ///
    /// `offsets` has length ``n_inner_arrays + 1``; inner array `i` occupies
    /// ``data[offsets[i]..offsets[i + 1]]``.
    /// Can be used to serialize the table.
    pub fn raw_parts(&self) -> (&[T], &[usize]) {
        (&self.data, &self.offsets)
    }
}

impl<T: Copy> Index<usize> for NestedArray<T> {
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[self.offsets[index]..self.offsets[index + 1]]
    }
}

impl<T: Copy> IndexMut<usize> for NestedArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[self.offsets[index]..self.offsets[index + 1]]
    }
}

/// Builder for a `NestedArray`: The inner array lengths are fixed up
/// front and elements are scattered into their slots via `push`.
///
/// Notes
/// -----
/// The per-slot write cursors are stored in the `offsets` array itself (no
/// separate cursor array), and the data buffer is filled while uninitialized
/// (no zero-initialization). Both are key to construction performance for large
/// tables.
///
/// ``T: Copy`` is required because the elements are written through a raw
/// pointer while the backing `Vec`'s length stays ``0`` until `build`. If the
/// builder were dropped before `build` (e.g. on a panic mid-fill), those
/// elements would not be seen by the `Vec` and thus not dropped. Forbidding
/// `Drop` (which `Copy` does) means there is nothing to leak.
pub struct NestedArrayBuilder<T: Copy> {
    /// The flat data buffer. `Vec::len` stays ``0`` (the buffer is
    /// uninitialized) until `build` sets it to `expected_total`;
    /// elements are written through the raw pointer.
    data: Vec<T>,
    /// Per-slot write cursor: ``offsets[i]`` is the next free position of slot
    /// `i`. `build` un-shifts it into the final slot-start offsets.
    offsets: Vec<usize>,
    /// The total number of elements across all slots (the data capacity).
    expected_total: usize,
    /// The number of elements added so far via `push`.
    elements_added: usize,
}

impl<T: Copy> NestedArrayBuilder<T> {
    /// Create a builder for a `NestedArray` whose inner array ``i`` holds exactly
    /// ``lengths[i]`` elements.
    pub fn new(lengths: Vec<usize>) -> Self {
        // Prefix-sum the lengths into slot start offsets, reusing the vector
        let mut offsets = lengths;
        let mut total = 0;
        for slot in &mut offsets {
            let length = *slot;
            *slot = total;
            total += length;
        }
        offsets.push(total);
        Self {
            data: Vec::with_capacity(total),
            offsets,
            expected_total: total,
            elements_added: 0,
        }
    }

    /// Append `value` to inner array `index`.
    ///
    /// Safety
    /// ------
    /// Across all calls, the number of elements pushed to a given `index` must
    /// not exceed that slot's length (as passed to `new`). Otherwise a write
    /// would spill into a neighboring slot's region and leave another slot
    /// partly uninitialized, which `build` would then expose.
    #[inline]
    pub unsafe fn push(&mut self, index: usize, value: T) {
        assert!(
            self.elements_added < self.expected_total,
            "pushed more elements than the total slot length"
        );
        let cursor = &mut self.offsets[index];
        let position = *cursor;
        *cursor = position + 1;
        // SAFETY: by the per-slot contract, `position` stays within slot
        // `index`, hence `position < expected_total == capacity`, and each
        // position is written exactly once.
        unsafe {
            self.data.as_mut_ptr().add(position).write(value);
        }
        self.elements_added += 1;
    }

    /// Finalize the builder into a `NestedArray`.
    ///
    /// Panics
    /// ------
    /// Panics if fewer elements were added than the total slot length, since
    /// that would leave part of the data buffer uninitialized.
    pub fn build(mut self) -> NestedArray<T> {
        assert_eq!(
            self.elements_added, self.expected_total,
            "fewer elements were added than the total slot length"
        );
        // SAFETY: exactly `expected_total` elements were written, one per
        // position in `0..expected_total`.
        unsafe {
            self.data.set_len(self.expected_total);
        }
        // Each `offsets[i]` has advanced to the start of slot `i + 1`;
        // shift right by one and restore `offsets[0] = 0` to recover the slot starts
        let n_slots = self.offsets.len() - 1;
        self.offsets.copy_within(0..n_slots, 1);
        self.offsets[0] = 0;
        // SAFETY: `offsets` is the prefix-sum offset array (length `n_slots + 1`,
        // non-decreasing, starting at `0` and ending at `expected_total`), and
        // `data` was just filled with exactly `expected_total` elements.
        unsafe { NestedArray::from_raw_parts(self.data, self.offsets) }
    }
}
