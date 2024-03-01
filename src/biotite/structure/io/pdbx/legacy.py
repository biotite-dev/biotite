# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdbx"
__author__ = "Patrick Kunzmann"
__all__ = ["PDBxFile"]

import copy
from collections.abc import MutableMapping
import warnings
from .cif import CIFFile, CIFBlock, CIFCategory, CIFColumn
from ....file import File, InvalidFileError


class PDBxFile(File, MutableMapping):
    """
    This class represents the legacy interface to CIF files.

    The categories of the file can be accessed using the
    :meth:`get_category()`/:meth:`set_category()` methods.
    The content of each category is represented by a dictionary.
    The dictionary contains the entry
    (e.g. *label_entity_id* in *atom_site*) as key.
    The corresponding values are either strings in *non-looped*
    categories, or 1-D numpy arrays of string objects in case of
    *looped* categories.

    A category can be changed or added using :meth:`set_category()`:
    If a string-valued dictionary is provided, a *non-looped* category
    will be created; if an array-valued dictionary is given, a
    *looped* category will be created. In case of arrays, it is
    important that all arrays have the same size.

    Alternatively, The content of this file can also be read/write
    accessed using dictionary-like indexing:
    You can either provide a data block and a category or only a
    category, in which case the first data block is taken.

    DEPRECATED: Use :class:`CIFFile` instead.

    Notes
    -----
    This class is also able to detect and parse multiline entries in the
    file. However, when writing a category no multiline values are used.
    This could lead to long lines.

    This class uses a lazy category dictionary creation: When reading
    the file only the line positions of all categories are checked. The
    time consuming task of dictionary creation is done when
    :meth:`get_category()` is called.

    Examples
    --------
    Read the file and get author names:

    >>> import os.path
    >>> file = PDBxFile.read(os.path.join(path_to_structures, "1l2y.cif"))
    >>> author_dict = file.get_category("citation_author", block="1L2Y")
    >>> print(author_dict["name"])
    ['Neidigh, J.W.' 'Fesinmeyer, R.M.' 'Andersen, N.H.']

    Dictionary style indexing, no specification of data block:

    >>> print(file["citation_author"]["name"])
    ['Neidigh, J.W.' 'Fesinmeyer, R.M.' 'Andersen, N.H.']

    Get the structure from the file:

    >>> arr = get_structure(file)
    >>> print(type(arr).__name__)
    AtomArrayStack
    >>> arr = get_structure(file, model=1)
    >>> print(type(arr).__name__)
    AtomArray

    Modify atom array and write it back into the file:

    >>> arr_mod = rotate(arr, [1,2,3])
    >>> set_structure(file, arr_mod)
    >>> file.write(os.path.join(path_to_directory, "1l2y_mod.cif"))
    """

    def __init__(self):
        warnings.warn(
            "'PDBxFile' is deprecated, use 'CIFFile' instead",
            DeprecationWarning
        )
        super().__init__()
        self._cif_file = CIFFile()

    @property
    def cif_file(self):
        return self._cif_file

    @property
    def lines(self):
        return self._cif_file.lines

    @classmethod
    def read(cls, file):
        """
        Read a PDBx/mmCIF file.

        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.

        Returns
        -------
        file_object : PDBxFile
            The parsed file.
        """
        pdbx_file = PDBxFile()
        pdbx_file._cif_file = CIFFile.read(file)
        return pdbx_file

    def write(self, file):
        self._cif_file.write(file)


    def get_block_names(self):
        """
        Get the names of all data blocks in the file.

        Returns
        -------
        blocks : list
            List of data block names.
        """
        return sorted(self._cif_file.keys())

    def get_category(self, category, block=None, expect_looped=False):
        """
        Get the dictionary for a given category.

        Parameters
        ----------
        category : string
            The name of the category. The leading underscore is omitted.
        block : string, optional
            The name of the data block. Default is the first
            (and most times only) data block of the file.
        expect_looped : bool, optional
            If set to true, the returned dictionary will always contain
            arrays (only if the category exists):
            If the category is *non-looped*, each array will contain
            only one element.

        Returns
        -------
        category_dict : dict of (str or ndarray, dtype=str) or None
            A entry keyed dictionary. The corresponding values are
            strings or array of strings for *non-looped* and
            *looped* categories, respectively.
            Returns None, if the data block does not contain the given
            category.
        """
        if block is None:
            try:
                block = self.get_block_names()[0]
            except IndexError:
                raise InvalidFileError("File is empty")

        if category not in self._cif_file[block]:
            return None

        category_dict = {}
        for column_name, column in self._cif_file[block][category].items():
            if not expect_looped and len(column) == 1:
                category_dict[column_name] = column.as_item()
            else:
                category_dict[column_name] = column.as_array()
        return category_dict

    def set_category(self, category, category_dict, block=None):
        """
        Set the content of a category.

        If the category is already existing, all lines corresponding
        to the category are replaced. Otherwise a new category is
        created and the lines are appended at the end of the data block.

        Parameters
        ----------
        category : string
            The name of the category. The leading underscore is omitted.
        category_dict : dict
            The category content. The dictionary must have strings
            (subcategories) as keys and strings or :class:`ndarray`
            objects as values.
        block : string, optional
            The name of the data block. Default is the first
            (and most times only) data block of the file. If the
            block is not contained in the file yet, a new block is
            appended at the end of the file.
        """
        if block is None:
            try:
                block = self.get_block_names()[0]
            except IndexError:
                raise InvalidFileError(
                    "File is empty, give an explicit data block"
                )

        if block not in self._cif_file:
            self._cif_file = CIFBlock()
        self._cif_file[block][category] = CIFCategory({
            column_name: CIFColumn(array)
            for column_name, array in category_dict.items()
        })

    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        clone._cif_file = copy.deepcopy(self._cif_file)

    def __setitem__(self, index, item):
        block, category_name = self._full_index(index)
        self.set_category(category_name, item, block=block)

    def __getitem__(self, index):
        block, category_name = self._full_index(index)
        return self.get_category(category_name, block=block)

    def __delitem__(self, index):
        block, category_name = self._full_index(index)
        del self._cif_file[block][category_name]

    def __contains__(self, index):
        block, category_name = self._full_index(index)
        return (block, category_name) in self._categories

    def __iter__(self):
        try:
            block = self.get_block_names()[0]
        except IndexError:
            raise InvalidFileError(
                "File is empty, give an explicit data block"
            )

        return iter(self._cif_file[block])

    def __len__(self):
        try:
            block = self.get_block_names()[0]
        except IndexError:
            raise InvalidFileError(
                "File is empty, give an explicit data block"
            )

        return len(self._cif_file[block])

    def _full_index(self, index):
        """
        Converts a an integer or tuple index into a block and a category
        name.
        """
        if isinstance(index, tuple):
            return index[0], index[1]
        elif isinstance(index, str):
            return self.get_block_names()[0], index
        else:
            raise TypeError(
                f"'{type(index).__name__}' is an invalid index type"
            )