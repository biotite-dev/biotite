//! The [`StridedTable`] dynamic programming table.

use std::ops::{Add, AddAssign, Index, IndexMut, Sub};

/// An index into a [`StridedTable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TableIndex(usize);

/// An offset between two [`TableIndex`] values.
///
/// Moving to a neighboring cell is a cheap addition of a precomputed offset
/// (e.g. `+ 1` along a row, `+ stride` along a column), without recomputing
/// strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableOffset(isize);

impl Add<TableOffset> for TableIndex {
    type Output = TableIndex;

    #[inline(always)]
    fn add(self, offset: TableOffset) -> TableIndex {
        // `wrapping_add_signed` keeps the arithmetic in the `usize` domain; a
        // `usize -> isize -> usize` round-trip would defeat the compiler's
        // pointer-induction optimization of the table-fill loop.
        TableIndex(self.0.wrapping_add_signed(offset.0))
    }
}

impl AddAssign<TableOffset> for TableIndex {
    #[inline(always)]
    fn add_assign(&mut self, offset: TableOffset) {
        *self = *self + offset;
    }
}

impl Sub<TableIndex> for TableIndex {
    type Output = TableOffset;

    #[inline(always)]
    fn sub(self, other: TableIndex) -> TableOffset {
        TableOffset(self.0 as isize - other.0 as isize)
    }
}

/// A dense, row-major dynamic programming table holding one cell per position.
///
/// It is a container for the [`Cell`](super::cell::Cell) objects that make up
/// both the score and trace tables of an alignment.
pub struct StridedTable<T> {
    shape: (usize, usize),
    stride: usize,
    data: Vec<T>,
}

impl<T: Default + Clone> StridedTable<T> {
    /// Create a zero-initialized table of the given `(rows, columns)` shape.
    pub fn new(shape: (usize, usize)) -> Self {
        let stride = shape.1;
        let data = vec![T::default(); shape.0 * shape.1];
        StridedTable {
            shape,
            stride,
            data,
        }
    }

    /// Create a table of the given `(rows, columns)` shape with every cell set
    /// to a copy of `value`.
    pub fn filled(shape: (usize, usize), value: T) -> Self {
        let stride = shape.1;
        let data = vec![value; shape.0 * shape.1];
        StridedTable {
            shape,
            stride,
            data,
        }
    }

    /// Create a new (zero-initialized) table of the given `shape` and copy the
    /// cells of `table` into it, keeping their `(row, column)` positions.
    ///
    /// Can be used to grow the dynamic programming table.
    pub fn from_data(shape: (usize, usize), table: StridedTable<T>) -> Self {
        assert!(
            shape.0 >= table.shape.0 && shape.1 >= table.shape.1,
            "The new shape must not be smaller than the existing table"
        );
        let mut new_table = StridedTable::new(shape);
        let width = table.shape.1;
        for row in 0..table.shape.0 {
            let src = row * table.stride;
            let dst = row * new_table.stride;
            new_table.data[dst..dst + width].clone_from_slice(&table.data[src..src + width]);
        }
        new_table
    }
}

impl<T> StridedTable<T> {
    /// The `(rows, columns)` shape of the table.
    #[inline(always)]
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// The index of `position = (row, column)`.
    #[inline(always)]
    pub fn index(&self, position: (usize, usize)) -> TableIndex {
        TableIndex(position.0 * self.stride + position.1)
    }

    /// The offset corresponding to moving by `offset = (rows, columns)`.
    #[inline(always)]
    pub fn index_offset(&self, offset: (isize, isize)) -> TableOffset {
        TableOffset(offset.0 * self.stride as isize + offset.1)
    }

    /// The `(row, column)` position of `index`, i.e. the inverse of
    /// [`StridedTable::index`].
    #[inline(always)]
    pub fn unindex(&self, index: TableIndex) -> (usize, usize) {
        (index.0 / self.stride, index.0 % self.stride)
    }

    /// Iterate over all cells in row-major (memory) order, paired with their
    /// index. Iterating the backing buffer avoids per-cell bounds checks.
    #[inline]
    pub fn indexed_iter(&self) -> impl Iterator<Item = (TableIndex, &T)> {
        self.data
            .iter()
            .enumerate()
            .map(|(i, cell)| (TableIndex(i), cell))
    }
}

impl<T> Index<TableIndex> for StridedTable<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: TableIndex) -> &T {
        &self.data[index.0]
    }
}

impl<T> IndexMut<TableIndex> for StridedTable<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: TableIndex) -> &mut T {
        &mut self.data[index.0]
    }
}
