# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdbx"
__author__ = "Patrick Kunzmann"
__all__ = [
    "BinaryCIFFile",
    "BinaryCIFBlock",
    "BinaryCIFCategory",
    "BinaryCIFColumn",
    "BinaryCIFData",
]

from collections.abc import Sequence
import msgpack
import numpy as np
from biotite.file import File, SerializationError, is_binary, is_open_compatible
from biotite.structure.io.pdbx.component import (
    MaskValue,
    _Component,
    _HierarchicalContainer,
)
from biotite.structure.io.pdbx.encoding import (
    create_uncompressed_encoding,
    decode_stepwise,
    deserialize_encoding,
    encode_stepwise,
)


class BinaryCIFData(_Component):
    r"""
    This class represents the data in a :class:`BinaryCIFColumn`.

    Parameters
    ----------
    array : array_like or int or float or str
        The data array to be stored.
        If a single item is given, it is converted into an array.
    encoding : list of Encoding , optional
        The encoding steps that are successively applied to the data.
        By default, the data is stored uncompressed directly as bytes.

    Attributes
    ----------
    array : ndarray
        The stored data array.
    encoding : list of Encoding
        The encoding steps.

    Examples
    --------

    >>> data = BinaryCIFData([1, 2, 3])
    >>> print(data.array)
    [1 2 3]
    >>> print(len(data))
    3
    >>> # A single item is converted into an array
    >>> data = BinaryCIFData("apple")
    >>> print(data.array)
    ['apple']

    A well-chosen encoding can significantly reduce the serialized data
    size:

    >>> # Default uncompressed encoding
    >>> array = np.arange(100)
    >>> uncompressed_bytes = BinaryCIFData(array).serialize()["data"]
    >>> print(len(uncompressed_bytes))
    400
    >>> # Delta encoding followed by run-length encoding
    >>> # [0, 1, 2, ...] -> [0, 1, 1, ...] -> [0, 1, 1, 99]
    >>> compressed_bytes = BinaryCIFData(
    ...     array,
    ...     encoding = [
    ...         # [0, 1, 2, ...] -> [0, 1, 1, ...]
    ...         DeltaEncoding(),
    ...         # [0, 1, 1, ...] -> [0, 1, 1, 99]
    ...         RunLengthEncoding(),
    ...         # [0, 1, 1, 99] -> b"\x00\x00..."
    ...         ByteArrayEncoding()
    ...     ]
    ... ).serialize()["data"]
    >>> print(len(compressed_bytes))
    16
    """

    def __init__(self, array, encoding=None):
        if not isinstance(array, (Sequence, np.ndarray)) or isinstance(array, str):
            array = [array]
        array = np.asarray(array)
        if np.issubdtype(array.dtype, np.object_):
            raise ValueError("Object arrays are not supported")

        self._array = array
        if encoding is None:
            self._encoding = create_uncompressed_encoding(array)
        else:
            self._encoding = list(encoding)

    @property
    def array(self):
        return self._array

    @property
    def encoding(self):
        return self._encoding

    @staticmethod
    def subcomponent_class():
        return None

    @staticmethod
    def supercomponent_class():
        return BinaryCIFColumn

    @staticmethod
    def deserialize(content):
        encoding = [deserialize_encoding(enc) for enc in content["encoding"]]
        return BinaryCIFData(decode_stepwise(content["data"], encoding), encoding)

    def serialize(self):
        serialized_data = encode_stepwise(self._array, self._encoding)
        if not isinstance(serialized_data, bytes):
            raise SerializationError("Final encoding must return 'bytes'")
        serialized_encoding = [enc.serialize() for enc in self._encoding]
        return {"data": serialized_data, "encoding": serialized_encoding}

    def __len__(self):
        return len(self._array)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if not np.array_equal(self._array, other._array):
            return False
        if self._encoding != other._encoding:
            return False
        return True


class BinaryCIFColumn(_Component):
    """
    This class represents a single column in a :class:`CIFCategory`.

    Parameters
    ----------
    data : BinaryCIFData or array_like or int or float or str
        The data to be stored.
        If no :class:`BinaryCIFData` is given, the passed argument is
        coerced into such an object.
    mask : BinaryCIFData or array_like, dtype=int or int
        The mask to be stored.
        If given, the mask indicates whether the `data` is
        inapplicable (``.``) or missing (``?``) in some rows.
        The data presence is indicated by values from the
        :class:`MaskValue` enum.
        If no :class:`BinaryCIFData` is given, the passed argument is
        coerced into such an object.
        By default, no mask is created.

    Attributes
    ----------
    data : BinaryCIFData
        The stored data.
    mask : BinaryCIFData
        The mask that indicates whether certain data elements are
        inapplicable or missing.
        If no mask is present, this attribute is ``None``.

    Examples
    --------

    >>> print(BinaryCIFColumn([1, 2, 3]).as_array())
    [1 2 3]
    >>> mask = [MaskValue.PRESENT, MaskValue.INAPPLICABLE, MaskValue.MISSING]
    >>> # Mask values are only inserted into string arrays
    >>> print(BinaryCIFColumn([1, 2, 3], mask).as_array(int))
    [1 2 3]
    >>> print(BinaryCIFColumn([1, 2, 3], mask).as_array(str))
    ['1' '.' '?']
    >>> print(BinaryCIFColumn([1]).as_item())
    1
    >>> print(BinaryCIFColumn([1], mask=[MaskValue.MISSING]).as_item())
    ?
    """

    def __init__(self, data, mask=None):
        if not isinstance(data, BinaryCIFData):
            data = BinaryCIFData(data)
        if mask is not None:
            if not isinstance(mask, BinaryCIFData):
                mask = BinaryCIFData(mask)
            if len(data) != len(mask):
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
        return BinaryCIFData

    @staticmethod
    def supercomponent_class():
        return BinaryCIFCategory

    def as_item(self):
        """
        Get the only item in the data of this column.

        If the data is masked as inapplicable or missing, ``'.'`` or
        ``'?'`` is returned, respectively.
        If the data contains more than one item, an exception is raised.

        Returns
        -------
        item : str or int or float
            The item in the data.
        """
        if self._mask is None:
            return self._data.array.item()
        mask = self._mask.array.item()
        if mask is None or mask == MaskValue.PRESENT:
            return self._data.array.item()
        elif mask == MaskValue.INAPPLICABLE:
            return "."
        elif mask == MaskValue.MISSING:
            return "?"

    def as_array(self, dtype=None, masked_value=None):
        """
        Get the data of this column as an :class:`ndarray`.

        This is a shortcut to get ``BinaryCIFColumn.data.array``.
        Furthermore, the mask is applied to the data.

        Parameters
        ----------
        dtype : dtype-like, optional
            The data type the array should be converted to.
            By default, the original type is used.
        masked_value : str or int or float, optional
            The value that should be used for masked elements, i.e.
            ``MaskValue.INAPPLICABLE`` or ``MaskValue.MISSING``.
            By default, masked elements are converted to ``'.'`` or
            ``'?'`` depending on the :class:`MaskValue`.

        Returns
        -------
        array : ndarray
            The column data as array.
        """
        if dtype is None:
            dtype = self._data.array.dtype

        if self._mask is None:
            return self._data.array.astype(dtype, copy=False)

        elif np.issubdtype(dtype, np.str_):
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

        elif np.dtype(dtype).kind == self._data.array.dtype.kind:
            if masked_value is None:
                return self._data.array.astype(dtype, copy=False)
            else:
                array = self._data.array.astype(dtype, copy=True)
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

    @staticmethod
    def deserialize(content):
        return BinaryCIFColumn(
            BinaryCIFData.deserialize(content["data"]),
            BinaryCIFData.deserialize(content["mask"])
            if content["mask"] is not None
            else None,
        )

    def serialize(self):
        return {
            "data": self._data.serialize(),
            "mask": self._mask.serialize() if self._mask is not None else None,
        }

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


class BinaryCIFCategory(_HierarchicalContainer):
    """
    This class represents a category in a :class:`BinaryCIFBlock`.

    Columns can be accessed and modified like a dictionary.
    The values are :class:`BinaryCIFColumn` objects.

    Parameters
    ----------
    columns : dict, optional
        The columns of the category.
        The keys are the column names and the values are the
        :class:`BinaryCIFColumn` objects (or objects that can be coerced
        into a :class:`BinaryCIFColumn`).
        By default, an empty category is created.
        Each column must have the same length.
    row_count : int, optional
        The number of rows in the category.

    Attributes
    ----------
    row_count : int
        The number of rows in the category, i.e. the length of each
        column.
        By default, the row count is determined when the first column is added.

    Examples
    --------

    >>> # Add column on creation
    >>> category = BinaryCIFCategory({"fruit": ["apple", "banana"]})
    >>> # Add column later on
    >>> category["taste"] = ["delicious", "tasty"]
    >>> # Add column the formal way
    >>> category["color"] = BinaryCIFColumn(BinaryCIFData(["red", "yellow"]))
    >>> # Access a column
    >>> print(category["fruit"].as_array())
    ['apple' 'banana']
    """

    def __init__(self, columns=None, row_count=None):
        if columns is None:
            columns = {}
        else:
            columns = {
                key: BinaryCIFColumn(col)
                if not isinstance(col, (BinaryCIFColumn, dict))
                else col
                for key, col in columns.items()
            }

        self._row_count = row_count
        super().__init__(columns)

    @property
    def row_count(self):
        if self._row_count is None:
            # Row count is not determined yet
            # -> check the length of the first column
            self._row_count = len(next(iter(self.values())))
        return self._row_count

    @staticmethod
    def subcomponent_class():
        return BinaryCIFColumn

    @staticmethod
    def supercomponent_class():
        return BinaryCIFBlock

    @staticmethod
    def deserialize(content):
        return BinaryCIFCategory(
            BinaryCIFCategory._deserialize_elements(content["columns"], "name"),
            content["rowCount"],
        )

    def serialize(self):
        if len(self) == 0:
            raise SerializationError("At least one column is required")

        for column_name, column in self.items():
            if self._row_count is None:
                self._row_count = len(column)
            elif len(column) != self._row_count:
                raise SerializationError(
                    f"All columns must have the same length, "
                    f"but '{column_name}' has length {len(column)}, "
                    f"while the first column has row_count {self._row_count}"
                )

        return {
            "rowCount": self.row_count,
            "columns": self._serialize_elements("name"),
        }

    def __setitem__(self, key, element):
        if not isinstance(element, (BinaryCIFColumn, dict)):
            element = BinaryCIFColumn(element)
        super().__setitem__(key, element)


class BinaryCIFBlock(_HierarchicalContainer):
    """
    This class represents a block in a :class:`BinaryCIFFile`.

    Categories can be accessed and modified like a dictionary.
    The values are :class:`BinaryCIFCategory` objects.

    Parameters
    ----------
    categories : dict, optional
        The categories of the block.
        The keys are the category names and the values are the
        :class:`BinaryCIFCategory` objects.
        By default, an empty block is created.

    Notes
    -----
    The category names do not include the leading underscore character.
    This character is automatically added when the category is
    serialized.

    Examples
    --------

    >>> # Add category on creation
    >>> block = BinaryCIFBlock({"foo": BinaryCIFCategory({"some_column": 1})})
    >>> # Add category later on
    >>> block["bar"] = BinaryCIFCategory({"another_column": [2, 3]})
    >>> # Access a column
    >>> print(block["bar"]["another_column"].as_array())
    [2 3]
    """

    def __init__(self, categories=None):
        if categories is None:
            categories = {}
        super().__init__(
            # Actual bcif files use leading '_' as category names
            {"_" + name: category for name, category in categories.items()}
        )

    @staticmethod
    def subcomponent_class():
        return BinaryCIFCategory

    @staticmethod
    def supercomponent_class():
        return BinaryCIFFile

    @staticmethod
    def deserialize(content):
        return BinaryCIFBlock(
            {
                # The superclass uses leading '_' in category names,
                # but on the level of this class, the leading '_' is omitted
                name.lstrip("_"): category
                for name, category in BinaryCIFBlock._deserialize_elements(
                    content["categories"], "name"
                ).items()
            }
        )

    def serialize(self):
        return {"categories": self._serialize_elements("name")}

    def __getitem__(self, key):
        try:
            return super().__getitem__("_" + key)
        except KeyError:
            raise KeyError(key)

    def __setitem__(self, key, element):
        try:
            return super().__setitem__("_" + key, element)
        except KeyError:
            raise KeyError(key)

    def __delitem__(self, key):
        try:
            return super().__setitem__("_" + key)
        except KeyError:
            raise KeyError(key)

    def __iter__(self):
        return (key.lstrip("_") for key in super().__iter__())

    def __contains__(self, key):
        return super().__contains__("_" + key)


class BinaryCIFFile(File, _HierarchicalContainer):
    """
    This class represents a *BinaryCIF* file.

    The categories of the file can be accessed and modified like a
    dictionary.
    The values are :class:`BinaryCIFBlock` objects.

    To parse or write a structure from/to a :class:`BinaryCIFFile`
    object, use the high-level :func:`get_structure()` or
    :func:`set_structure()` function respectively.

    Parameters
    ----------
    blocks : dict (str -> BinaryCIFBlock), optional
        The initial blocks of the file.
        Maps the block names to the corresponding :class:`BinaryCIFBlock` objects.
        By default no initial blocks are added.

    Attributes
    ----------
    block : BinaryCIFBlock
        The sole block of the file.
        If the file contains multiple blocks, an exception is raised.

    Notes
    -----
    The content of *BinaryCIF* files are lazily deserialized:
    Only when a column is accessed, the time consuming data decoding
    is performed.
    The decoded :class:`BinaryCIFBlock`/:class:`BinaryCIFCategory`
    objects are cached for subsequent accesses.

    Examples
    --------
    Read a *BinaryCIF* file and access its content:

    >>> import os.path
    >>> file = BinaryCIFFile.read(os.path.join(path_to_structures, "1l2y.bcif"))
    >>> print(file["1L2Y"]["citation_author"]["name"].as_array())
    ['Neidigh, J.W.' 'Fesinmeyer, R.M.' 'Andersen, N.H.']
    >>> # Access the only block in the file
    >>> print(file.block["entity"]["pdbx_description"].as_item())
    TC5b

    Create a *BinaryCIF* file and write it to disk:

    >>> category = BinaryCIFCategory({"some_column": "some_value"})
    >>> block = BinaryCIFBlock({"some_category": category})
    >>> file = BinaryCIFFile({"some_block": block})
    >>> file.write(os.path.join(path_to_directory, "some_file.bcif"))
    """

    def __init__(self, blocks=None):
        File.__init__(self)
        _HierarchicalContainer.__init__(self, blocks)

    @property
    def block(self):
        if len(self) != 1:
            raise ValueError("There are multiple blocks in the file")
        return self[next(iter(self))]

    @staticmethod
    def subcomponent_class():
        return BinaryCIFBlock

    @staticmethod
    def supercomponent_class():
        return None

    @staticmethod
    def deserialize(content):
        return BinaryCIFFile(
            BinaryCIFFile._deserialize_elements(content["dataBlocks"], "header")
        )

    def serialize(self):
        return {"dataBlocks": self._serialize_elements("header")}

    @classmethod
    def read(cls, file):
        """
        Read a *BinaryCIF* file.

        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.

        Returns
        -------
        file_object : BinaryCIFFile
            The parsed file.
        """
        # File name
        if is_open_compatible(file):
            with open(file, "rb") as f:
                return BinaryCIFFile.deserialize(
                    msgpack.unpackb(f.read(), use_list=True, raw=False)
                )
        # File object
        else:
            if not is_binary(file):
                raise TypeError("A file opened in 'binary' mode is required")
            return BinaryCIFFile.deserialize(
                msgpack.unpackb(file.read(), use_list=True, raw=False)
            )

    def write(self, file):
        """
        Write contents into a *BinaryCIF* file.

        Parameters
        ----------
        file : file-like object or str
            The file to be written to.
            Alternatively, a file path can be supplied.
        """
        serialized_content = self.serialize()
        serialized_content["encoder"] = "biotite"
        serialized_content["version"] = "0.3.0"
        packed_bytes = msgpack.packb(
            serialized_content, use_bin_type=True, default=_encode_numpy
        )
        if is_open_compatible(file):
            with open(file, "wb") as f:
                f.write(packed_bytes)
        else:
            if not is_binary(file):
                raise TypeError("A file opened in 'binary' mode is required")
            file.write(packed_bytes)


def _encode_numpy(item):
    """
    Convert NumPy scalar types to native Python types,
    as *Msgpack* cannot handle NumPy types (e.g. float32).

    The function is given to the Msgpack packer as value for the
    `default` parameter.
    """
    if isinstance(item, np.generic):
        return item.item()
    else:
        raise TypeError(f"can not serialize '{type(item).__name__}' object")
