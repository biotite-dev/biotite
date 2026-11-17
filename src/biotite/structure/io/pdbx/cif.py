from __future__ import annotations

__name__ = "biotite.structure.io.pdbx"
__author__ = "Patrick Kunzmann"
__all__ = ["CIFFile", "CIFBlock", "CIFCategory", "CIFColumn", "CIFData"]

from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from os import PathLike
from typing import IO, Any, Self, TypeAlias, overload
import numpy as np
from biotite.file import (
    DeserializationError,
    File,
    SerializationError,
    is_open_compatible,
    is_text,
)
from biotite.rust.structure.io.pdbx import (
    deserialize_category,
    split_blocks,
    split_categories,
)
from biotite.structure.io.pdbx.component import MaskValue, _Component
from biotite.typing import NDArray1

UNICODE_CHAR_SIZE = 4


# Used to indicate elements of a CIF file that are not yet deserialized,
# as deserialization is performed lazily
_SerializedElement: TypeAlias = str


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

    def __init__(
        self,
        array: NDArray1[Any, Any] | Sequence[Any] | int | float | str,
        dtype: np.dtype | type | str | None = None,
    ) -> None:
        self._array = _arrayfy(array)
        if np.issubdtype(self._array.dtype, np.object_):
            raise ValueError("Object arrays are not supported")
        if dtype is not None:
            self._array = self._array.astype(dtype, copy=False)

    @property
    def array(self) -> NDArray1[Any, Any]:
        return self._array

    @staticmethod
    def subcomponent_class() -> None:
        return None

    @staticmethod
    def supercomponent_class() -> type[CIFColumn]:
        return CIFColumn

    def __len__(self) -> int:
        return len(self._array)

    def __eq__(self, other: object) -> bool:
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

    def __init__(
        self,
        data: CIFData | NDArray1[Any, Any] | Sequence[Any] | int | float | str,
        mask: CIFData | NDArray1[Any, Any] | Sequence[int] | None = None,
    ) -> None:
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
    def data(self) -> CIFData:
        return self._data

    @property
    def mask(self) -> CIFData | None:
        return self._mask

    @staticmethod
    def subcomponent_class() -> type[CIFData]:
        return CIFData

    @staticmethod
    def supercomponent_class() -> type[CIFCategory]:
        return CIFCategory

    def as_item(self) -> str:
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
        if mask == MaskValue.PRESENT:
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
        raise ValueError(f"Unknown mask value: {mask}")

    def as_array(
        self,
        dtype: np.dtype | type | str = str,
        masked_value: Any = None,
    ) -> NDArray1[Any, Any]:
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
                array = np.array([f"{e:.3f}" for e in self._data.array], dtype=dtype)
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
            # (e.g. '.' cannot be converted to int)
            if masked_value is None:
                array = np.zeros(len(self._data), dtype=dtype)
            else:
                array = np.full(len(self._data), masked_value, dtype=dtype)

            present_mask = self._mask.array == MaskValue.PRESENT
            array[present_mask] = self._data.array[present_mask].astype(dtype)
            return array

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self._data != other._data:
            return False
        if self._mask != other._mask:
            return False
        return True


class CIFCategory(_Component, MutableMapping[str, CIFColumn]):
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

    def __init__(
        self,
        columns: Mapping[str, CIFColumn | NDArray1[Any, Any] | Sequence[Any]]
        | None = None,
        name: str | None = None,
    ) -> None:
        self._name = name
        self._columns: dict[str, CIFColumn]
        if columns is None:
            self._columns = {}
        else:
            self._columns = {
                key: CIFColumn(col) if not isinstance(col, CIFColumn) else col
                for key, col in columns.items()
            }

        self._row_count: int | None = None

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def row_count(self) -> int:
        if self._row_count is None:
            # Row count is not determined yet
            # -> check the length of the first column
            self._row_count = len(next(iter(self.values())))
        return self._row_count

    @staticmethod
    def subcomponent_class() -> type[CIFColumn]:
        return CIFColumn

    @staticmethod
    def supercomponent_class() -> type[CIFBlock]:
        return CIFBlock

    @classmethod
    def deserialize(cls, text: str) -> Self:
        category_name, columns = deserialize_category(text)
        return cls(dict(columns), category_name)

    def serialize(self) -> str:
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

    def __getitem__(self, key: str) -> CIFColumn:
        return self._columns[key]

    def __setitem__(
        self,
        key: str,
        column: CIFColumn | NDArray1[Any, Any] | Sequence[Any],
    ) -> None:
        if not isinstance(column, CIFColumn):
            column = CIFColumn(column)
        self._columns[key] = column

    def __delitem__(self, key: str) -> None:
        if len(self._columns) == 1:
            raise ValueError("At least one column must remain")
        del self._columns[key]

    def __contains__(self, key: object) -> bool:
        return key in self._columns

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def __eq__(self, other: object) -> bool:
        # Row count can be omitted here, as it is based on the columns
        if not isinstance(other, type(self)):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for col_name in self.keys():
            if self[col_name] != other[col_name]:
                return False
        return True

    def _serialize_single(self) -> list[str]:
        if self._name is None:
            raise SerializationError("Category name is required")
        keys = ["_" + self._name + "." + name for name in self.keys()]
        max_len = max(len(key) for key in keys)
        # "+3" Because of three whitespace chars after longest key
        req_len = max_len + 3
        values = _escape(np.array([column.as_item() for column in self.values()]))
        return [
            # Remove potential terminal newlines from multiline values
            (key.ljust(req_len) + value).strip()
            for key, value in zip(keys, values.tolist())
        ]

    def _serialize_looped(self) -> list[str]:
        if self._name is None:
            raise SerializationError("Category name is required")
        if self._row_count is None:
            raise SerializationError("Category row count is required")
        key_lines = ["_" + self._name + "." + key + " " for key in self.keys()]

        value_lines = None
        for column in self.values():
            # Quote before measuring the number of chars,
            # as the quote characters modify the length
            array = _escape(column.as_array(str))
            # Number of characters the longest string in the column needs
            # The "+1" is for the small whitespace column
            n_chars = int(np.strings.str_len(array).max(initial=0)) + 1
            padded = np.strings.ljust(array, n_chars)
            value_lines = padded if value_lines is None else value_lines + padded
        if value_lines is None:
            raise SerializationError("Category contains no columns")
        # Remove trailing justification of last column
        # and potential terminal newlines from multiline values
        value_lines = np.strings.strip(value_lines)

        return ["loop_"] + key_lines + value_lines.tolist()


class CIFBlock(_Component, MutableMapping[str, CIFCategory]):
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

    def __init__(
        self,
        categories: dict[str, CIFCategory | _SerializedElement] | None = None,
        name: str | None = None,
    ) -> None:
        self._name = name
        # Categories may be stored in not yet deserialized text form alongside
        # already deserialized `CIFCategory` instances
        self._categories: dict[str, CIFCategory | str] = (
            {} if categories is None else dict(categories)
        )

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @staticmethod
    def subcomponent_class() -> type[CIFCategory]:
        return CIFCategory

    @staticmethod
    def supercomponent_class() -> type[CIFFile]:
        return CIFFile

    @classmethod
    def deserialize(cls, text: str) -> Self:
        return cls(dict(split_categories(text)))

    def serialize(self) -> str:
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

    def __getitem__(self, key: str) -> CIFCategory:
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

    def __setitem__(self, key: str, category: CIFCategory) -> None:
        if not isinstance(category, CIFCategory):
            raise TypeError(
                f"Expected 'CIFCategory', but got '{type(category).__name__}'"
            )
        category.name = key
        self._categories[key] = category

    def __delitem__(self, key: str) -> None:
        del self._categories[key]

    def __contains__(self, key: object) -> bool:
        return key in self._categories

    def __iter__(self) -> Iterator[str]:
        return iter(self._categories)

    def __len__(self) -> int:
        return len(self._categories)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for cat_name in self.keys():
            if self[cat_name] != other[cat_name]:
                return False
        return True


class CIFFile(_Component, File, MutableMapping[str, CIFBlock]):
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

    def __init__(
        self, blocks: dict[str, CIFBlock | _SerializedElement] | None = None
    ) -> None:
        # Blocks may be stored in not yet deserialized text form alongside
        # already deserialized `CIFBlock` instances
        self._blocks: dict[str, CIFBlock | str] = {} if blocks is None else dict(blocks)

    @property
    def lines(self) -> list[str]:
        return self.serialize().splitlines()

    @property
    def block(self) -> CIFBlock:
        if len(self) == 0:
            raise ValueError("There are no blocks in the file")
        elif len(self) > 1:
            raise ValueError("There are multiple blocks in the file")
        else:
            return self[next(iter(self))]

    @staticmethod
    def subcomponent_class() -> type[CIFBlock]:
        return CIFBlock

    @staticmethod
    def supercomponent_class() -> None:
        return None

    @classmethod
    def deserialize(cls, text: str) -> Self:
        return cls(dict(split_blocks(text)))

    def serialize(self) -> str:
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
    def read(cls, file: PathLike[str] | str | IO[str]) -> CIFFile:
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

    def write(self, file: PathLike[str] | str | IO[str]) -> None:
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

    def __getitem__(self, key: str) -> CIFBlock:
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

    def __setitem__(self, key: str, block: CIFBlock) -> None:
        if not isinstance(block, CIFBlock):
            raise TypeError(f"Expected 'CIFBlock', but got '{type(block).__name__}'")
        block.name = key
        self._blocks[key] = block

    def __delitem__(self, key: str) -> None:
        del self._blocks[key]

    def __contains__(self, key: object) -> bool:
        return key in self._blocks

    def __iter__(self) -> Iterator[str]:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for block_name in self.keys():
            if self[block_name] != other[block_name]:
                return False
        return True


def _escape(array: NDArray1[Any, np.str_]) -> NDArray1[Any, np.str_]:
    """
    Escape special characters in values to make them compatible with CIF.
    """
    has_linebreak = np.strings.find(array, "\n") != -1
    has_single_quote = np.strings.find(array, "'") != -1
    has_double_quote = np.strings.find(array, '"') != -1
    has_whitespace = (np.strings.find(array, " ") != -1) | (
        np.strings.find(array, "\t") != -1
    )
    is_empty = np.strings.str_len(array) == 0
    starts_with_underscore = np.strings.startswith(array, "_")

    # A value with linebreaks or with both quote types must be represented
    # as multiline value, as neither quote type can be used for escaping
    multiline_mask = has_linebreak | (has_single_quote & has_double_quote)
    empty_mask = ~multiline_mask & is_empty
    # Values with an embedded single quote are escaped with double quotes,
    # all other values that require escaping use single quotes
    double_quote_mask = (
        ~multiline_mask & ~is_empty & ~starts_with_underscore & has_single_quote
    )
    single_quote_mask = (
        ~multiline_mask
        & ~is_empty
        & (
            starts_with_underscore
            | (~has_single_quote & (has_double_quote | has_whitespace))
        )
    )

    # Quoting adds two characters
    escaped = array.astype(f"U{array.dtype.itemsize // UNICODE_CHAR_SIZE + 2}")
    escaped[empty_mask] = "''"
    escaped[single_quote_mask] = "'" + array[single_quote_mask] + "'"
    escaped[double_quote_mask] = '"' + array[double_quote_mask] + '"'
    if multiline_mask.any():
        multiline_values = _multiline(array[multiline_mask])
        # Multiline values need more characters than the quoted ones
        n_chars = max(escaped.dtype.itemsize, multiline_values.dtype.itemsize)
        escaped = escaped.astype(f"U{n_chars // UNICODE_CHAR_SIZE}")
        escaped[multiline_mask] = multiline_values
    return escaped


@overload
def _multiline(value: str) -> str: ...
@overload
def _multiline(value: NDArray1[Any, np.str_]) -> NDArray1[Any, np.str_]: ...
def _multiline(
    value: str | NDArray1[Any, np.str_],
) -> str | NDArray1[Any, np.str_]:
    """
    Convert a string (or an array of strings) that may contain linebreaks
    into CIF-compatible multiline string.
    """
    return "\n;" + value + "\n;\n"


def _arrayfy(data: Any) -> NDArray1[Any, Any]:
    if not isinstance(data, (Sequence, np.ndarray)) or isinstance(data, str):
        data = [data]
    elif len(data) == 0:
        raise ValueError("Array must contain at least one element")
    return np.asarray(data)
