# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdbx"
__author__ = "Patrick Kunzmann"
__all__ = ["CIFFile", "CIFBlock", "CIFCategory", "CIFColumn", "CIFData"]

import itertools
from collections.abc import MutableMapping, Sequence
import numpy as np
from biotite.file import (
    DeserializationError,
    File,
    SerializationError,
    is_open_compatible,
    is_text,
)
from biotite.structure.io.pdbx.component import MaskValue, _Component

UNICODE_CHAR_SIZE = 4


# Small class without much functionality
# It exists merely for consistency with BinaryCIFFile
class CIFData:
    """
    This class represents the data in a :class:`CIFColumn`.

    Parameters
    ----------
    array : array_like or int or float or str
        The data array to be stored.
        If a single item is given, it is converted into an array.
    dtype : dtype-like, optional
        If given, the *dtype* the stored array should be converted to.

    Attributes
    ----------
    array : ndarray
        The stored data array.

    Notes
    -----
    When a :class:`CIFFile` is written, the data type is automatically
    converted to string.
    The other way around, when a :class:`CIFFile` is read, the data type
    is always a string type.

    Examples
    --------

    >>> data = CIFData([1, 2, 3])
    >>> print(data.array)
    [1 2 3]
    >>> print(len(data))
    3
    >>> # A single item is converted into an array
    >>> data = CIFData("apple")
    >>> print(data.array)
    ['apple']
    """

    def __init__(self, array, dtype=None):
        self._array = _arrayfy(array)
        if np.issubdtype(self._array.dtype, np.object_):
            raise ValueError("Object arrays are not supported")
        if dtype is not None:
            self._array = self._array.astype(dtype)

    @property
    def array(self):
        return self._array

    @staticmethod
    def subcomponent_class():
        return None

    @staticmethod
    def supercomponent_class():
        return CIFColumn

    def __len__(self):
        return len(self._array)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return np.array_equal(self._array, other._array)


class CIFColumn:
    """
    This class represents a single column in a :class:`CIFCategory`.

    Parameters
    ----------
    data : CIFData or array_like or int or float or str
        The data to be stored.
        If no :class:`CIFData` is given, the passed argument is
        coerced into such an object.
    mask : CIFData or array_like, dtype=int or int
        The mask to be stored.
        If given, the mask indicates whether the `data` is
        inapplicable (``.``) or missing (``?``) in some rows.
        The data presence is indicated by values from the
        :class:`MaskValue` enum.
        If no :class:`CIFData` is given, the passed argument is
        coerced into such an object.
        By default, no mask is created.

    Attributes
    ----------
    data : CIFData
        The stored data.
    mask : CIFData
        The mask that indicates whether certain data elements are
        inapplicable or missing.
        If no mask is present, this attribute is ``None``.

    Examples
    --------

    >>> print(CIFColumn([1, 2, 3]).as_array())
    ['1' '2' '3']
    >>> mask = [MaskValue.PRESENT, MaskValue.INAPPLICABLE, MaskValue.MISSING]
    >>> print(CIFColumn([1, 2, 3], mask).as_array())
    ['1' '.' '?']
    >>> print(CIFColumn([1]).as_item())
    1
    >>> print(CIFColumn([1], mask=[MaskValue.MISSING]).as_item())
    ?
    """

    def __init__(self, data, mask=None):
        if not isinstance(data, CIFData):
            data = CIFData(data, str)
        if mask is None:
            mask = np.full(len(data), MaskValue.PRESENT, dtype=np.uint8)
            mask[data.array == "."] = MaskValue.INAPPLICABLE
            mask[data.array == "?"] = MaskValue.MISSING
            if np.all(mask == MaskValue.PRESENT):
                # No mask required
                mask = None
            else:
                mask = CIFData(mask)
        else:
            if not isinstance(mask, CIFData):
                mask = CIFData(mask, np.uint8)
            if len(mask) != len(data):
                raise IndexError(
                    f"Data has length {len(data)}, but mask has length {len(mask)}"
                )
        self._data = data
        self._mask = mask

    @property
    def data(self):
        return self._data

    @property
    def mask(self):
        return self._mask

    @staticmethod
    def subcomponent_class():
        return CIFData

    @staticmethod
    def supercomponent_class():
        return CIFCategory

    def as_item(self):
        """
        Get the only item in the data of this column.

        If the data is masked as inapplicable or missing, ``'.'`` or
        ``'?'`` is returned, respectively.
        If the data contains more than one item, an exception is raised.

        Returns
        -------
        item : str
            The item in the data.
        """
        if self._mask is None:
            return self._data.array.item()
        mask = self._mask.array.item()
        if self._mask is None or mask == MaskValue.PRESENT:
            item = self._data.array.item()
            # Limit float precision to 3 decimals
            if isinstance(item, float):
                return f"{item:.3f}"
            else:
                return str(item)
        elif mask == MaskValue.INAPPLICABLE:
            return "."
        elif mask == MaskValue.MISSING:
            return "?"

    def as_array(self, dtype=str, masked_value=None):
        """
        Get the data of this column as an :class:`ndarray`.

        This is a shortcut to get ``CIFColumn.data.array``.
        Furthermore, the mask is applied to the data.

        Parameters
        ----------
        dtype : dtype-like, optional
            The data type the array should be converted to.
            By default, a string type is used.
        masked_value : str, optional
            The value that should be used for masked elements, i.e.
            ``MaskValue.INAPPLICABLE`` or ``MaskValue.MISSING``.
            By default, masked elements are converted to ``'.'`` or
            ``'?'`` depending on the :class:`MaskValue`.

        Returns
        -------
        array : ndarray
            The column data as array.
        """
        if self._mask is None:
            return self._data.array.astype(dtype, copy=False)

        elif np.issubdtype(dtype, np.str_):
            # Limit float precision to 3 decimals
            if np.issubdtype(self._data.array.dtype, np.floating):
                array = np.array([f"{e:.3f}" for e in self._data.array], type=dtype)
            else:
                # Copy, as otherwise original data would be overwritten
                # with mask values
                array = self._data.array.astype(dtype, copy=True)
            if masked_value is None:
                array[self._mask.array == MaskValue.INAPPLICABLE] = "."
                array[self._mask.array == MaskValue.MISSING] = "?"
            else:
                array[self._mask.array == MaskValue.INAPPLICABLE] = masked_value
                array[self._mask.array == MaskValue.MISSING] = masked_value
            return array

        else:
            # Array needs to be converted, but masked values are
            # not necessarily convertible
            # (e.g. '' cannot be converted to int)
            if masked_value is None:
                array = np.zeros(len(self._data), dtype=dtype)
            else:
                array = np.full(len(self._data), masked_value, dtype=dtype)

            present_mask = self._mask.array == MaskValue.PRESENT
            array[present_mask] = self._data.array[present_mask].astype(dtype)
            return array

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self._data != other._data:
            return False
        if self._mask != other._mask:
            return False
        return True


class CIFCategory(_Component, MutableMapping):
    """
    This class represents a category in a :class:`CIFBlock`.

    Columns can be accessed and modified like a dictionary.
    The values are :class:`CIFColumn` objects.

    Parameters
    ----------
    columns : dict, optional
        The columns of the category.
        The keys are the column names and the values are the
        :class:`CIFColumn` objects (or objects that can be coerced into
        a :class:`CIFColumn`).
        By default, an empty category is created.
        Each column must have the same length.
    name : str, optional
        The name of the category.
        This is only used for serialization and is automatically set,
        when the :class:`CIFCategory` is added to a :class:`CIFBlock`.
        It only needs to be set manually, when the category is directly
        serialized.

    Attributes
    ----------
    name : str
        The name of the category.
    row_count : int
        The number of rows in the category, i.e. the length of each
        column.

    Notes
    -----
    When a column containing strings with line breaks are added, these
    strings are written as multiline strings to the CIF file.

    Examples
    --------

    >>> # Add column on creation
    >>> category = CIFCategory({"fruit": ["apple", "banana"]}, name="fruits")
    >>> # Add column later on
    >>> category["taste"] = ["delicious", "tasty"]
    >>> # Add column the formal way
    >>> category["color"] = CIFColumn(CIFData(["red", "yellow"]))
    >>> # Access a column
    >>> print(category["fruit"].as_array())
    ['apple' 'banana']
    >>> print(category.serialize())
    loop_
    _fruits.fruit
    _fruits.taste
    _fruits.color
    apple  delicious red
    banana tasty     yellow
    """

    def __init__(self, columns=None, name=None):
        self._name = name
        if columns is None:
            columns = {}
        else:
            columns = {
                key: CIFColumn(col) if not isinstance(col, CIFColumn) else col
                for key, col in columns.items()
            }

        self._row_count = None
        self._columns = columns

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def row_count(self):
        if self._row_count is None:
            # Row count is not determined yet
            # -> check the length of the first column
            self._row_count = len(next(iter(self.values())))
        return self._row_count

    @staticmethod
    def subcomponent_class():
        return CIFColumn

    @staticmethod
    def supercomponent_class():
        return CIFBlock

    @staticmethod
    def deserialize(text):
        lines = [line.strip() for line in text.splitlines() if not _is_empty(line)]

        if _is_loop_start(lines[0]):
            is_looped = True
            lines.pop(0)
        else:
            is_looped = False

        category_name = _parse_category_name(lines[0])
        if category_name is None:
            raise DeserializationError("Failed to parse category name")

        lines = _to_single(lines)
        if is_looped:
            category_dict = CIFCategory._deserialize_looped(lines)
        else:
            category_dict = CIFCategory._deserialize_single(lines)
        return CIFCategory(category_dict, category_name)

    def serialize(self):
        if self._name is None:
            raise SerializationError("Category name is required")
        if not self._columns:
            raise ValueError("At least one column is required")

        for column_name, column in self.items():
            if self._row_count is None:
                self._row_count = len(column)
            elif len(column) != self._row_count:
                raise SerializationError(
                    f"All columns must have the same length, "
                    f"but '{column_name}' has length {len(column)}, "
                    f"while the first column has row_count {self._row_count}"
                )

        if self._row_count == 0:
            raise ValueError("At least one row is required")
        elif self._row_count == 1:
            lines = self._serialize_single()
        else:
            lines = self._serialize_looped()
        # Enforce terminal line break
        lines.append("")
        return "\n".join(lines)

    def __getitem__(self, key):
        return self._columns[key]

    def __setitem__(self, key, column):
        if not isinstance(column, CIFColumn):
            column = CIFColumn(column)
        self._columns[key] = column

    def __delitem__(self, key):
        if len(self._columns) == 1:
            raise ValueError("At least one column must remain")
        del self._columns[key]

    def __contains__(self, key):
        return key in self._columns

    def __iter__(self):
        return iter(self._columns)

    def __len__(self):
        return len(self._columns)

    def __eq__(self, other):
        # Row count can be omitted here, as it is based on the columns
        if not isinstance(other, type(self)):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for col_name in self.keys():
            if self[col_name] != other[col_name]:
                return False
        return True

    @staticmethod
    def _deserialize_single(lines):
        """
        Process a category where each field has a single value.
        """
        category_dict = {}
        line_i = 0
        while line_i < len(lines):
            line = lines[line_i]
            parts = list(_split_one_line(line))
            if len(parts) == 2:
                # Standard case -> name and value in one line
                name_part, value_part = parts
                line_i += 1
            elif len(parts) == 1:
                # Value is a multiline value on the next line
                name_part = parts[0]
                parts = list(_split_one_line(lines[line_i + 1]))
                if len(parts) == 1:
                    value_part = parts[0]
                else:
                    raise DeserializationError(f"Failed to parse line '{line}'")
                line_i += 2
            elif len(parts) == 0:
                raise DeserializationError("Empty line within category")
            else:
                raise DeserializationError(f"Failed to parse line '{line}'")
            category_dict[name_part.split(".")[1]] = CIFColumn(value_part)
        return category_dict

    @staticmethod
    def _deserialize_looped(lines):
        """
        Process a category where each field has multiple values
        (category is a table).
        """
        category_dict = {}
        column_names = []
        i = 0
        for key_line in lines:
            if key_line[0] == "_":
                # Key line
                key = key_line.split(".")[1]
                column_names.append(key)
                category_dict[key] = []
                i += 1
            else:
                break

        data_lines = lines[i:]
        # Rows may be split over multiple lines -> do not rely on
        # row-line-alignment at all and simply cycle through columns
        column_indices = itertools.cycle(range(len(column_names)))
        for data_line in data_lines:
            values = _split_one_line(data_line)
            for val in values:
                column_index = next(column_indices)
                column_name = column_names[column_index]
                category_dict[column_name].append(val)

        # Check if all columns have the same length
        # Otherwise, this would indicate a parsing error or an invalid CIF file
        column_index = next(column_indices)
        if column_index != 0:
            raise DeserializationError(
                "Category contains columns with different lengths"
            )

        return category_dict

    def _serialize_single(self):
        keys = ["_" + self._name + "." + name for name in self.keys()]
        max_len = max(len(key) for key in keys)
        # "+3" Because of three whitespace chars after longest key
        req_len = max_len + 3
        return [
            # Remove potential terminal newlines from multiline values
            (key.ljust(req_len) + _escape(column.as_item())).strip()
            for key, column in zip(keys, self.values())
        ]

    def _serialize_looped(self):
        key_lines = ["_" + self._name + "." + key + " " for key in self.keys()]

        column_arrays = []
        for column in self.values():
            array = column.as_array(str)
            # Quote before measuring the number of chars,
            # as the quote characters modify the length
            array = np.array([_escape(element) for element in array])
            column_arrays.append(array)

        # Number of characters the longest string in the column needs
        # This can be deduced from the dtype
        # The "+1" is for the small whitespace column
        column_n_chars = [
            array.dtype.itemsize // UNICODE_CHAR_SIZE + 1 for array in column_arrays
        ]
        value_lines = [""] * self._row_count
        for i in range(self._row_count):
            for j, array in enumerate(column_arrays):
                value_lines[i] += array[i].ljust(column_n_chars[j])
            # Remove trailing justification of last column
            # and potential terminal newlines from multiline values
            value_lines[i] = value_lines[i].strip()

        return ["loop_"] + key_lines + value_lines


class CIFBlock(_Component, MutableMapping):
    """
    This class represents a block in a :class:`CIFFile`.

    Categories can be accessed and modified like a dictionary.
    The values are :class:`CIFCategory` objects.

    Parameters
    ----------
    categories : dict, optional
        The categories of the block.
        The keys are the category names and the values are the
        :class:`CIFCategory` objects.
        By default, an empty block is created.
    name : str, optional
        The name of the block.
        This is only used for serialization and is automatically set,
        when the :class:`CIFBlock` is added to a :class:`CIFFile`.
        It only needs to be set manually, when the block is directly
        serialized.

    Attributes
    ----------
    name : str
        The name of the block.

    Notes
    -----
    The category names do not include the leading underscore character.
    This character is automatically added when the category is
    serialized.

    Examples
    --------

    >>> # Add category on creation
    >>> block = CIFBlock({"foo": CIFCategory({"some_column": 1})}, name="baz")
    >>> # Add category later on
    >>> block["bar"] = CIFCategory({"another_column": [2, 3]})
    >>> # Access a column
    >>> print(block["bar"]["another_column"].as_array())
    ['2' '3']
    >>> print(block.serialize())
    data_baz
    #
    _foo.some_column   1
    #
    loop_
    _bar.another_column
    2
    3
    #
    """

    def __init__(self, categories=None, name=None):
        self._name = name
        if categories is None:
            categories = {}
        self._categories = categories

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @staticmethod
    def subcomponent_class():
        return CIFCategory

    @staticmethod
    def supercomponent_class():
        return CIFFile

    @staticmethod
    def deserialize(text):
        lines = text.splitlines()
        current_category_name = None
        category_starts = []
        category_names = []
        for i, line in enumerate(lines):
            if not _is_empty(line):
                is_loop_in_line = _is_loop_start(line)
                category_name_in_line = _parse_category_name(line)
                if is_loop_in_line or (
                    category_name_in_line != current_category_name
                    and category_name_in_line is not None
                ):
                    # Track the new category
                    if is_loop_in_line:
                        # In case of lines with "loop_" the category is
                        # in the next line
                        category_name_in_line = _parse_category_name(lines[i + 1])
                    current_category_name = category_name_in_line
                    category_starts.append(i)
                    category_names.append(current_category_name)
        return CIFBlock(_create_element_dict(lines, category_names, category_starts))

    def serialize(self):
        if self._name is None:
            raise SerializationError("Block name is required")
        # The block starts with the black name line followed by a comment line
        text_blocks = ["data_" + self._name + "\n#\n"]
        for category_name, category in self._categories.items():
            if isinstance(category, str):
                # Category is already stored as lines
                text_blocks.append(category)
            else:
                try:
                    category.name = category_name
                    text_blocks.append(category.serialize())
                except Exception:
                    raise SerializationError(
                        f"Failed to serialize category '{category_name}'"
                    )
                # A comment line is set after each category
                text_blocks.append("#\n")
        return "".join(text_blocks)

    def __getitem__(self, key):
        category = self._categories[key]
        if isinstance(category, str):
            # Element is stored in serialized form
            # -> must be deserialized first
            try:
                category = CIFCategory.deserialize(category)
            except Exception:
                raise DeserializationError(f"Failed to deserialize category '{key}'")
            # Update with deserialized object
            self._categories[key] = category
        return category

    def __setitem__(self, key, category):
        if not isinstance(category, CIFCategory):
            raise TypeError(
                f"Expected 'CIFCategory', but got '{type(category).__name__}'"
            )
        category.name = key
        self._categories[key] = category

    def __delitem__(self, key):
        del self._categories[key]

    def __contains__(self, key):
        return key in self._categories

    def __iter__(self):
        return iter(self._categories)

    def __len__(self):
        return len(self._categories)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for cat_name in self.keys():
            if self[cat_name] != other[cat_name]:
                return False
        return True


class CIFFile(_Component, File, MutableMapping):
    """
    This class represents a CIF file.

    The categories of the file can be accessed and modified like a
    dictionary.
    The values are :class:`CIFBlock` objects.

    To parse or write a structure from/to a :class:`CIFFile` object,
    use the high-level :func:`get_structure()` or
    :func:`set_structure()` function respectively.

    Parameters
    ----------
    blocks : dict (str -> CIFBlock), optional
        The initial blocks of the file.
        Maps the block names to the corresponding :class:`CIFBlock` objects.
        By default no initial blocks are added.

    Attributes
    ----------
    block : CIFBlock
        The sole block of the file.
        If the file contains multiple blocks, an exception is raised.

    Notes
    -----
    The content of CIF files are lazily deserialized:
    When reading the file only the line positions of all blocks are
    indexed.
    The time consuming deserialization of a block/category is only
    performed when accessed.
    The deserialized :class:`CIFBlock`/:class:`CIFCategory` objects
    are cached for subsequent accesses.

    Examples
    --------
    Read a CIF file and access its content:

    >>> import os.path
    >>> file = CIFFile.read(os.path.join(path_to_structures, "1l2y.cif"))
    >>> print(file["1L2Y"]["citation_author"]["name"].as_array())
    ['Neidigh, J.W.' 'Fesinmeyer, R.M.' 'Andersen, N.H.']
    >>> # Access the only block in the file
    >>> print(file.block["entity"]["pdbx_description"].as_item())
    TC5b

    Create a CIF file and write it to disk:

    >>> category = CIFCategory(
    ...     {"some_column": "some_value", "another_column": "another_value"}
    ... )
    >>> block = CIFBlock({"some_category": category, "another_category": category})
    >>> file = CIFFile({"some_block": block, "another_block": block})
    >>> print(file.serialize())
    data_some_block
    #
    _some_category.some_column      some_value
    _some_category.another_column   another_value
    #
    _another_category.some_column      some_value
    _another_category.another_column   another_value
    #
    data_another_block
    #
    _some_category.some_column      some_value
    _some_category.another_column   another_value
    #
    _another_category.some_column      some_value
    _another_category.another_column   another_value
    #
    >>> file.write(os.path.join(path_to_directory, "some_file.cif"))
    """

    def __init__(self, blocks=None):
        if blocks is None:
            blocks = {}
        self._blocks = blocks

    @property
    def lines(self):
        return self.serialize().splitlines()

    @property
    def block(self):
        if len(self) != 1:
            raise ValueError("There are multiple blocks in the file")
        return self[next(iter(self))]

    @staticmethod
    def subcomponent_class():
        return CIFBlock

    @staticmethod
    def supercomponent_class():
        return None

    @staticmethod
    def deserialize(text):
        lines = text.splitlines()
        block_starts = []
        block_names = []
        for i, line in enumerate(lines):
            if not _is_empty(line):
                data_block_name = _parse_data_block_name(line)
                if data_block_name is not None:
                    block_starts.append(i)
                    block_names.append(data_block_name)
        return CIFFile(_create_element_dict(lines, block_names, block_starts))

    def serialize(self):
        text_blocks = []
        for block_name, block in self._blocks.items():
            if isinstance(block, str):
                # Block is already stored as text
                text_blocks.append(block)
            else:
                try:
                    block.name = block_name
                    text_blocks.append(block.serialize())
                except Exception:
                    raise SerializationError(
                        f"Failed to serialize block '{block_name}'"
                    )
        # Enforce terminal line break
        text_blocks.append("")
        return "".join(text_blocks)

    @classmethod
    def read(cls, file):
        """
        Read a CIF file.

        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.

        Returns
        -------
        file_object : CIFFile
            The parsed file.
        """
        # File name
        if is_open_compatible(file):
            with open(file, "r") as f:
                text = f.read()
        # File object
        else:
            if not is_text(file):
                raise TypeError("A file opened in 'text' mode is required")
            text = file.read()
        return CIFFile.deserialize(text)

    def write(self, file):
        """
        Write the contents of this object into a CIF file.

        Parameters
        ----------
        file : file-like object or str
            The file to be written to.
            Alternatively a file path can be supplied.
        """
        if is_open_compatible(file):
            with open(file, "w") as f:
                f.write(self.serialize())
        else:
            if not is_text(file):
                raise TypeError("A file opened in 'text' mode is required")
            file.write(self.serialize())

    def __getitem__(self, key):
        block = self._blocks[key]
        if isinstance(block, str):
            # Element is stored in serialized form
            # -> must be deserialized first
            try:
                block = CIFBlock.deserialize(block)
            except Exception:
                raise DeserializationError(f"Failed to deserialize block '{key}'")
            block.name = key
            # Update with deserialized object
            self._blocks[key] = block
        return block

    def __setitem__(self, key, block):
        if not isinstance(block, CIFBlock):
            raise TypeError(f"Expected 'CIFBlock', but got '{type(block).__name__}'")
        block.name = key
        self._blocks[key] = block

    def __delitem__(self, key):
        del self._blocks[key]

    def __contains__(self, key):
        return key in self._blocks

    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return len(self._blocks)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for block_name in self.keys():
            if self[block_name] != other[block_name]:
                return False
        return True


def _is_empty(line):
    return len(line.strip()) == 0 or line[0] == "#"


def _create_element_dict(lines, element_names, element_starts):
    """
    Create a dict mapping the `element_names` to the corresponding
    `lines`, which are located between ``element_starts[i]`` and
    ``element_starts[i+1]``.
    """
    # Add exclusive stop to indices for easier slicing
    element_starts.append(len(lines))
    # Lazy deserialization
    # -> keep as text for now and deserialize later if needed
    return {
        element_name: "\n".join(lines[element_starts[i] : element_starts[i + 1]]) + "\n"
        for i, element_name in enumerate(element_names)
    }


def _parse_data_block_name(line):
    """
    If the line defines a data block, return this name.
    Return ``None`` otherwise.
    """
    if line.startswith("data_"):
        return line[5:]
    else:
        return None


def _parse_category_name(line):
    """
    If the line defines a category, return this name.
    Return ``None`` otherwise.
    """
    if line[0] != "_":
        return None
    else:
        return line[1 : line.find(".")]


def _is_loop_start(line):
    """
    Return whether the line starts a looped category.
    """
    return line.startswith("loop_")


def _to_single(lines):
    r"""
    Convert multiline values into singleline values
    (in terms of 'lines' list elements).
    Linebreaks are preserved as ``'\n'`` characters within a list element.
    The initial ``';'`` character is also preserved, while the final ``';'`` character
    is removed.
    """
    processed_lines = []
    in_multi_line = False
    mutli_line_value = []
    for line in lines:
        # Multiline value are enclosed by ';' at the start of the beginning and end line
        if line[0] == ";":
            if not in_multi_line:
                # Start of multiline value
                in_multi_line = True
                mutli_line_value.append(line)
            else:
                # End of multiline value
                in_multi_line = False
                # The current line contains only the end character ';'
                # Hence this line is not added to the processed lines
                processed_lines.append("\n".join(mutli_line_value))
                mutli_line_value = []
        else:
            if in_multi_line:
                mutli_line_value.append(line)
            else:
                processed_lines.append(line)
    return processed_lines


def _escape(value):
    """
    Escape special characters in a value to make it compatible with CIF.
    """
    if "\n" in value:
        # A value with linebreaks must be represented as multiline value
        return _multiline(value)
    elif "'" in value and '"' in value:
        # If both quote types are present, you cannot use them for escaping
        return _multiline(value)
    elif len(value) == 0:
        return "''"
    elif value[0] == "_":
        return "'" + value + "'"
    elif "'" in value:
        return '"' + value + '"'
    elif '"' in value:
        return "'" + value + "'"
    elif " " in value:
        return "'" + value + "'"
    elif "\t" in value:
        return "'" + value + "'"
    else:
        return value


def _multiline(value):
    """
    Convert a string that may contain linebreaks into CIF-compatible
    multiline string.
    """
    return "\n;" + value + "\n;\n"


def _split_one_line(line):
    """
    Split a line into its fields.
    Supporting embedded quotes (' or "), like `'a dog's life'` to  `a dog's life`
    """
    # Special case of multiline value, where the line starts with ';'
    if line[0] == ";":
        yield line[1:]
    elif "'" in line or '"' in line:
        # Quoted values in the line
        while line:
            # Strip leading whitespace(s)
            stripped_line = line.lstrip()
            # Split the line on whitespace
            word, _, line = stripped_line.partition(" ")
            # Handle the case where the word start with a quote
            if word.startswith(("'", '"')):
                # Set the separator to the quote found
                separator = word[0]
                # Handle the case of a quoted word without space
                if word.endswith(separator) and len(word) > 1:
                    # Yield the word without the opening and closing quotes
                    yield word[1:-1]
                    continue
                # split the word on the separator
                word, _, line = stripped_line[1:].partition(separator)

            yield word
    else:
        # No quoted values in the line -> simple whitespace split
        for line in line.split():
            yield line


def _arrayfy(data):
    if not isinstance(data, (Sequence, np.ndarray)) or isinstance(data, str):
        data = [data]
    elif len(data) == 0:
        raise ValueError("Array must contain at least one element")
    return np.asarray(data)
