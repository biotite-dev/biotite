# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mol"
__author__ = "Patrick Kunzmann, Benjamin Mayer"
__all__ = ["SDFile", "SDRecord", "Metadata"]

import re
import warnings
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
import numpy as np
from biotite.file import (
    DeserializationError,
    File,
    InvalidFileError,
    SerializationError,
    is_open_compatible,
    is_text,
)
from biotite.structure.atoms import AtomArray
from biotite.structure.bonds import BondList, BondType
from biotite.structure.io.mol.ctab import (
    read_structure_from_ctab,
    write_structure_to_ctab,
)
from biotite.structure.io.mol.header import Header

_N_HEADER = 3
# Number of header lines
_RECORD_DELIMITER = "$$$$"


class Metadata(MutableMapping):
    r"""
    Additional nonstructural data in an SD record.

    The metadata is stored as key-value pairs.
    As SDF allows multiple different identifiers for keys,
    the keys are represented by :class:`Metadata.Key`.

    Parameters
    ----------
    metadata : dict, optional
        The metadata as key-value pairs.
        Keys are instances of :class:`Metadata.Key`.
        Alternatively, keys can be given as strings, in which case the
        string is used as the :attr:`Metadata.Key.name`.
        Values are strings.
        Line breaks in values are allowed.

    Notes
    -----
    Key names may only contain alphanumeric characters, underscores and
    periods.

    Examples
    --------

    >>> metadata = Metadata({
    ...     "foo": "Lorem ipsum",
    ...     Metadata.Key(number=42, name="bar"): "dolor sit amet,\nconsectetur"
    ... })
    >>> print(metadata)
    > <foo>
    Lorem ipsum
    <BLANKLINE>
    > DT42 <bar>
    dolor sit amet,
    consectetur
    <BLANKLINE>
    >>> print(metadata["foo"])
    Lorem ipsum
    >>> # Strings can be only used for access, if the key contains only a name
    >>> print("bar" in metadata)
    False
    >>> print(metadata[Metadata.Key(number=42, name="bar")])
    dolor sit amet,
    consectetur
    """

    @dataclass(frozen=True, kw_only=True)
    class Key:
        """
        A metadata key.

        Parameters
        ----------
        number : int, optional
            number of the field in the database.
        name : str, optional
            Name of the field.
            May only contain alphanumeric characters, underscores and
            periods.
        registry_internal : int, optional
            Internal registry number.
        registry_external : str, optional
            External registry number.

        Attributes
        ----------
        number, name, registry_internal, registry_external
            The same as the parameters.
        """

        # The characters that can be given as input to `name`
        # First character must be alphanumeric,
        # following characters may include underscores and periods
        # Although the V3000 specification forbids the use of periods,
        # they are still used in practice and therefore allowed here
        _NAME_INPUT_REGEX = re.compile(r"^[a-zA-Z0-9][\w.]*$")
        # These regexes are used to parse the key from a line
        _COMPONENT_REGEX = {
            "number": re.compile(r"^DT(\d+)$"),
            "name": re.compile(r"^<([a-zA-Z0-9][\w.]*)>$"),
            "registry_internal": re.compile(r"^(\d+)$"),
            "registry_external": re.compile(r"^\(([\w.-]*)\)$"),
        }

        number: ... = None
        name: ... = None
        registry_internal: ... = None
        registry_external: ... = None

        def __post_init__(self):
            if self.name is None and self.number is None:
                raise ValueError("At least the field number or name must be set")
            if self.name is not None:
                if not Metadata.Key._NAME_INPUT_REGEX.match(self.name):
                    raise ValueError(
                        f"Invalid name '{self.name}', must only contains "
                        "alphanumeric characters, underscores and periods"
                    )
            if self.number is not None:
                # Cannot set field directly as 'frozen=True'
                object.__setattr__(self, "number", int(self.number))
            if self.registry_internal is not None:
                object.__setattr__(
                    self, "registry_internal", int(self.registry_internal)
                )

        @staticmethod
        def deserialize(text):
            """
            Create a :class:`Metadata.Key` object by deserializing the given text
            content.

            Parameters
            ----------
            text : str
                The content to be deserialized.

            Returns
            -------
            key : Metadata.Key
                The parsed key.
            """
            # Omit the leading '>'
            key_components = text[1:].split()
            parsed_component_dict = {}
            for component in key_components:
                # For each component in each the key,
                # try to match it with each of the regexes
                for attr_name, regex in Metadata.Key._COMPONENT_REGEX.items():
                    pattern_match = regex.match(component)
                    if pattern_match is None:
                        # Try next pattern
                        continue
                    if attr_name in parsed_component_dict:
                        raise DeserializationError(
                            f"Duplicate key component for '{attr_name}'"
                        )
                    value = pattern_match.group(1)
                    parsed_component_dict[attr_name] = value
                    break
                else:
                    # There is no matching pattern
                    raise DeserializationError(f"Invalid key component '{component}'")
            return Metadata.Key(**parsed_component_dict)

        def serialize(self):
            """
            Convert this object into text content.

            Returns
            -------
            content : str
                The serialized content.
            """
            key_string = "> "
            if self.number is not None:
                key_string += f"DT{self.number} "
            if self.name is not None:
                key_string += f"<{self.name}> "
            if self.registry_internal is not None:
                key_string += f"{self.registry_internal} "
            if self.registry_external is not None:
                key_string += f"({self.registry_external}) "
            return key_string

        def __str__(self):
            return self.serialize()

    def __init__(self, metadata=None):
        if metadata is None:
            metadata = {}
        self._metadata = {}
        for key, value in metadata.items():
            self._metadata[_to_metadata_key(key)] = value

    @staticmethod
    def deserialize(text):
        """
        Create a :class:`Metadata` objtect by deserializing the given text content.

        Parameters
        ----------
        text : str
            The content to be deserialized.

        Returns
        -------
        metadata : Metadata
            The parsed metadata.
        """
        metadata = {}
        current_key = None
        current_value = None
        for line in text.splitlines():
            line = line.strip()
            if len(line) == 0:
                # Skip empty lines
                continue
            if line.startswith(">"):
                _add_key_value_pair(metadata, current_key, current_value)
                current_key = Metadata.Key.deserialize(line)
                current_value = None
            else:
                if current_key is None:
                    raise DeserializationError("Value found before metadata key")
                if current_value is None:
                    current_value = line
                else:
                    current_value += "\n" + line
        # Add final pair
        _add_key_value_pair(metadata, current_key, current_value)
        return Metadata(metadata)

    def serialize(self):
        """
        Convert this object into text content.

        Returns
        -------
        content : str
            The serialized content.
        """
        text_blocks = []
        for key, value in self._metadata.items():
            text_blocks.append(key.serialize())
            # Add empty line after value
            text_blocks.append(value + "\n")
        return _join_with_terminal_newline(text_blocks)

    def __getitem__(self, key):
        return self._metadata[_to_metadata_key(key)]

    def __setitem__(self, key, value):
        if len(value) == 0:
            raise ValueError("Metadata value must not be empty")
        self._metadata[_to_metadata_key(key)] = value

    def __delitem__(self, key):
        del self._metadata[_to_metadata_key(key)]

    def __iter__(self):
        return iter(self._metadata)

    def __len__(self):
        return len(self._metadata)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for key in self.keys():
            if self[key] != other[key]:
                return False
        return True

    def __str__(self):
        return self.serialize()


class SDRecord:
    """
    A record in a SD file.

    Parameters
    ----------
    header : Header, optional
        The header of the record.
        By default, an empty header is created.
    ctab : str, optional
        The connection table (atoms and bonds) in the record.
        By default, an empty structure is created.
    metadata : Metadata, Mapping or str, optional
        The metadata of the record.
        Can be given as dictionary mapping :attr:`Metadata.Key.name`
        to the respective metadata value.
        By default, no metadata is appended to the record.

    Attributes
    ----------
    header, ctab, metadata
        The same as the parameters.

    Examples
    --------

    >>> atoms = residue("ALA")
    >>> record = SDRecord(header=Header(mol_name="ALA", dimensions="3D"))
    >>> record.set_structure(atoms)
    >>> print(record.get_structure())
                0             N        -0.966    0.493    1.500
                0             C         0.257    0.418    0.692
                0             C        -0.094    0.017   -0.716
                0             O        -1.056   -0.682   -0.923
                0             C         1.204   -0.620    1.296
                0             O         0.661    0.439   -1.742
                0             H        -1.383   -0.425    1.482
                0             H        -0.676    0.661    2.452
                0             H         0.746    1.392    0.682
                0             H         1.459   -0.330    2.316
                0             H         0.715   -1.594    1.307
                0             H         2.113   -0.676    0.697
                0             H         0.435    0.182   -2.647
    >>> # Add the record to an SD file
    >>> file = SDFile()
    >>> file["ALA"] = record
    >>> print(file)
    ALA
                        3D
    <BLANKLINE>
     13 12  0     0  0  0  0  0  0  1 V2000
       -0.9660    0.4930    1.5000 N   0  0  0  0  0  0  0  0  0  0  0  0
        0.2570    0.4180    0.6920 C   0  0  0  0  0  0  0  0  0  0  0  0
       -0.0940    0.0170   -0.7160 C   0  0  0  0  0  0  0  0  0  0  0  0
       -1.0560   -0.6820   -0.9230 O   0  0  0  0  0  0  0  0  0  0  0  0
        1.2040   -0.6200    1.2960 C   0  0  0  0  0  0  0  0  0  0  0  0
        0.6610    0.4390   -1.7420 O   0  0  0  0  0  0  0  0  0  0  0  0
       -1.3830   -0.4250    1.4820 H   0  0  0  0  0  0  0  0  0  0  0  0
       -0.6760    0.6610    2.4520 H   0  0  0  0  0  0  0  0  0  0  0  0
        0.7460    1.3920    0.6820 H   0  0  0  0  0  0  0  0  0  0  0  0
        1.4590   -0.3300    2.3160 H   0  0  0  0  0  0  0  0  0  0  0  0
        0.7150   -1.5940    1.3070 H   0  0  0  0  0  0  0  0  0  0  0  0
        2.1130   -0.6760    0.6970 H   0  0  0  0  0  0  0  0  0  0  0  0
        0.4350    0.1820   -2.6470 H   0  0  0  0  0  0  0  0  0  0  0  0
      1  2  1  0  0  0  0
      1  7  1  0  0  0  0
      1  8  1  0  0  0  0
      2  3  1  0  0  0  0
      2  5  1  0  0  0  0
      2  9  1  0  0  0  0
      3  4  2  0  0  0  0
      3  6  1  0  0  0  0
      5 10  1  0  0  0  0
      5 11  1  0  0  0  0
      5 12  1  0  0  0  0
      6 13  1  0  0  0  0
    M  END
    $$$$
    <BLANKLINE>
    """

    def __init__(self, header=None, ctab=None, metadata=None):
        if header is None:
            self._header = Header()
        else:
            self._header = header

        self._ctab = ctab

        if metadata is None:
            self._metadata = Metadata()
        elif isinstance(metadata, Metadata):
            self._metadata = metadata
        elif isinstance(metadata, Mapping):
            self._metadata = Metadata(metadata)
        elif isinstance(metadata, str):
            # Serialized form -> will be lazily deserialized
            self._metadata = metadata
        else:
            raise TypeError(
                "Expected 'Metadata', Mapping or str, "
                f"but got '{type(metadata).__name__}'"
            )

    @property
    def header(self):
        if isinstance(self._header, str):
            try:
                self._header = Header.deserialize(self._header)
            except Exception:
                raise DeserializationError("Failed to deserialize header")
        return self._header

    @header.setter
    def header(self, header):
        self._header = header

    @property
    def ctab(self):
        # CTAB string cannot be changed directly -> no setter
        return self._ctab

    @property
    def metadata(self):
        if isinstance(self._metadata, str):
            try:
                self._metadata = Metadata.deserialize(self._metadata)
            except Exception:
                raise DeserializationError("Failed to deserialize metadata")
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        if isinstance(metadata, Metadata):
            self._metadata = metadata
        elif isinstance(metadata, Mapping):
            self._metadata = Metadata(metadata)
        else:
            raise TypeError(
                f"Expected 'Metadata' or Mapping, but got '{type(metadata).__name__}'"
            )

    @staticmethod
    def deserialize(text):
        """
        Create an :class:`SDRecord` by deserializing the given text content.

        Parameters
        ----------
        text : str
            The content to be deserialized.

        Returns
        -------
        record : SDRecord
            The parsed record.
        """
        lines = text.splitlines()
        ctab_end = _get_ctab_stop(lines)

        header = _join_with_terminal_newline(lines[:_N_HEADER])
        ctab = _join_with_terminal_newline(lines[_N_HEADER:ctab_end])
        metadata = _join_with_terminal_newline(lines[ctab_end:])
        return SDRecord(header, ctab, metadata)

    def serialize(self):
        """
        Convert this object into text content.

        Returns
        -------
        content : str
            The serialized content.
        """
        if isinstance(self._header, str):
            header_string = self._header
        else:
            header_string = self._header.serialize()

        if self._ctab is None:
            ctab_string = _empty_ctab()
        else:
            ctab_string = self._ctab

        if isinstance(self._metadata, str):
            metadata_string = self._metadata
        else:
            metadata_string = self._metadata.serialize()

        return header_string + ctab_string + metadata_string

    def get_structure(self):
        """
        Parse the structural data in the SD record.

        Returns
        -------
        array : AtomArray
            This :class:`AtomArray` contains the optional ``charge``
            annotation and has an associated :class:`BondList`.
            All other annotation categories, except ``element`` are
            empty.
        """
        ctab_lines = self._ctab.splitlines()
        if len(ctab_lines) == 0:
            raise InvalidFileError("File does not contain structure data")
        return read_structure_from_ctab(ctab_lines)

    def set_structure(self, atoms, default_bond_type=BondType.ANY, version=None):
        """
        Set the structural data in the SD record.

        Parameters
        ----------
        atoms : AtomArray
            The array to be saved into this file.
            Must have an associated :class:`BondList`.
        default_bond_type : BondType, optional
            Bond type fallback for the *Bond block*, if a
            :class:`BondType` has no CTAB counterpart.
            By default, each such bond is treated as
            :attr:`BondType.ANY`.
        version : {"V2000", "V3000"}, optional
            The version of the CTAB format.
            ``"V2000"`` uses the *Atom* and *Bond* block, while
            ``"V3000"`` uses the *Properties* block.
            By default, ``"V2000"`` is used, unless the number of atoms
            or bonds exceeds 999, in which case ``"V3000"`` is used.
        """
        self._ctab = _join_with_terminal_newline(
            write_structure_to_ctab(atoms, default_bond_type, version)
        )

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if not self.header == other.header:
            return False
        if not self.ctab == other.ctab:
            return False
        if not self.metadata == other.metadata:
            return False
        return True

    def __str__(self):
        return self.serialize()


class SDFile(File, MutableMapping):
    """
    This class represents an SD file for storing small molecule
    structures.

    The records for each molecule in the file can be accessed and
    modified like a dictionary.
    The structures can be parsed and written from/to each
    :class:`SDRecord` object via :func:`get_structure()` or
    :func:`set_structure()`, respectively.

    Parameters
    ----------
    records : dict (str -> SDRecord), optional
        The initial records of the file.
        Maps the record names to the corresponding :class:`SDRecord` objects.
        By default no initial records are added.

    Attributes
    ----------
    record : CIFBlock
        The sole record of the file.
        If the file contains multiple records, an exception is raised.

    Examples
    --------
    Read a SD file and parse the molecular structure:

    >>> import os.path
    >>> file = SDFile.read(os.path.join(path_to_structures, "molecules", "TYR.sdf"))
    >>> molecule = file.record.get_structure()
    >>> print(molecule)
            0             N         1.320    0.952    1.428
            0             C        -0.018    0.429    1.734
            0             C        -0.103    0.094    3.201
            0             O         0.886   -0.254    3.799
            0             C        -0.274   -0.831    0.907
            0             C        -0.189   -0.496   -0.559
            0             C         1.022   -0.589   -1.219
            0             C        -1.324   -0.102   -1.244
            0             C         1.103   -0.282   -2.563
            0             C        -1.247    0.210   -2.587
            0             C        -0.032    0.118   -3.252
            0             O         0.044    0.420   -4.574
            0             O        -1.279    0.184    3.842
            0             H         1.977    0.225    1.669
            0             H         1.365    1.063    0.426
            0             H        -0.767    1.183    1.489
            0             H         0.473   -1.585    1.152
            0             H        -1.268   -1.219    1.134
            0             H         1.905   -0.902   -0.683
            0             H        -2.269   -0.031   -0.727
            0             H         2.049   -0.354   -3.078
            0             H        -2.132    0.523   -3.121
            0             H        -0.123   -0.399   -5.059
            0             H        -1.333   -0.030    4.784

    Note that important atom annotations may be missing.
    These can be set afterwards:

    >>> molecule.res_name[:] = "TYR"
    >>> molecule.atom_name[:] = create_atom_names(molecule)
    >>> print(molecule)
            0  TYR N1     N         1.320    0.952    1.428
            0  TYR C1     C        -0.018    0.429    1.734
            0  TYR C2     C        -0.103    0.094    3.201
            0  TYR O1     O         0.886   -0.254    3.799
            0  TYR C3     C        -0.274   -0.831    0.907
            0  TYR C4     C        -0.189   -0.496   -0.559
            0  TYR C5     C         1.022   -0.589   -1.219
            0  TYR C6     C        -1.324   -0.102   -1.244
            0  TYR C7     C         1.103   -0.282   -2.563
            0  TYR C8     C        -1.247    0.210   -2.587
            0  TYR C9     C        -0.032    0.118   -3.252
            0  TYR O2     O         0.044    0.420   -4.574
            0  TYR O3     O        -1.279    0.184    3.842
            0  TYR H1     H         1.977    0.225    1.669
            0  TYR H2     H         1.365    1.063    0.426
            0  TYR H3     H        -0.767    1.183    1.489
            0  TYR H4     H         0.473   -1.585    1.152
            0  TYR H5     H        -1.268   -1.219    1.134
            0  TYR H6     H         1.905   -0.902   -0.683
            0  TYR H7     H        -2.269   -0.031   -0.727
            0  TYR H8     H         2.049   -0.354   -3.078
            0  TYR H9     H        -2.132    0.523   -3.121
            0  TYR H10    H        -0.123   -0.399   -5.059
            0  TYR H11    H        -1.333   -0.030    4.784

    Create a SD file and write it to disk:

    >>> another_molecule = residue("ALA")
    >>> file = SDFile()
    >>> record = SDRecord()
    >>> record.set_structure(molecule)
    >>> file["TYR"] = record
    >>> record = SDRecord()
    >>> record.set_structure(another_molecule)
    >>> file["ALA"] = record
    >>> file.write(os.path.join(path_to_directory, "some_file.cif"))
    >>> print(file)
    TYR
    <BLANKLINE>
    <BLANKLINE>
     24 24  0     0  0  0  0  0  0  1 V2000
        1.3200    0.9520    1.4280 N   0  0  0  0  0  0  0  0  0  0  0  0
       -0.0180    0.4290    1.7340 C   0  0  0  0  0  0  0  0  0  0  0  0
       -0.1030    0.0940    3.2010 C   0  0  0  0  0  0  0  0  0  0  0  0
        0.8860   -0.2540    3.7990 O   0  0  0  0  0  0  0  0  0  0  0  0
       -0.2740   -0.8310    0.9070 C   0  0  0  0  0  0  0  0  0  0  0  0
       -0.1890   -0.4960   -0.5590 C   0  0  0  0  0  0  0  0  0  0  0  0
        1.0220   -0.5890   -1.2190 C   0  0  0  0  0  0  0  0  0  0  0  0
       -1.3240   -0.1020   -1.2440 C   0  0  0  0  0  0  0  0  0  0  0  0
        1.1030   -0.2820   -2.5630 C   0  0  0  0  0  0  0  0  0  0  0  0
       -1.2470    0.2100   -2.5870 C   0  0  0  0  0  0  0  0  0  0  0  0
       -0.0320    0.1180   -3.2520 C   0  0  0  0  0  0  0  0  0  0  0  0
        0.0440    0.4200   -4.5740 O   0  0  0  0  0  0  0  0  0  0  0  0
       -1.2790    0.1840    3.8420 O   0  0  0  0  0  0  0  0  0  0  0  0
        1.9770    0.2250    1.6690 H   0  0  0  0  0  0  0  0  0  0  0  0
        1.3650    1.0630    0.4260 H   0  0  0  0  0  0  0  0  0  0  0  0
       -0.7670    1.1830    1.4890 H   0  0  0  0  0  0  0  0  0  0  0  0
        0.4730   -1.5850    1.1520 H   0  0  0  0  0  0  0  0  0  0  0  0
       -1.2680   -1.2190    1.1340 H   0  0  0  0  0  0  0  0  0  0  0  0
        1.9050   -0.9020   -0.6830 H   0  0  0  0  0  0  0  0  0  0  0  0
       -2.2690   -0.0310   -0.7270 H   0  0  0  0  0  0  0  0  0  0  0  0
        2.0490   -0.3540   -3.0780 H   0  0  0  0  0  0  0  0  0  0  0  0
       -2.1320    0.5230   -3.1210 H   0  0  0  0  0  0  0  0  0  0  0  0
       -0.1230   -0.3990   -5.0590 H   0  0  0  0  0  0  0  0  0  0  0  0
       -1.3330   -0.0300    4.7840 H   0  0  0  0  0  0  0  0  0  0  0  0
      1  2  1  0  0  0  0
      1 14  1  0  0  0  0
      1 15  1  0  0  0  0
      2  3  1  0  0  0  0
      2  5  1  0  0  0  0
      2 16  1  0  0  0  0
      3  4  2  0  0  0  0
      3 13  1  0  0  0  0
      5  6  1  0  0  0  0
      5 17  1  0  0  0  0
      5 18  1  0  0  0  0
      6  7  2  0  0  0  0
      6  8  1  0  0  0  0
      7  9  1  0  0  0  0
      7 19  1  0  0  0  0
      8 10  2  0  0  0  0
      8 20  1  0  0  0  0
      9 11  2  0  0  0  0
      9 21  1  0  0  0  0
     10 11  1  0  0  0  0
     10 22  1  0  0  0  0
     11 12  1  0  0  0  0
     12 23  1  0  0  0  0
     13 24  1  0  0  0  0
    M  END
    $$$$
    ALA
    <BLANKLINE>
    <BLANKLINE>
     13 12  0     0  0  0  0  0  0  1 V2000
       -0.9660    0.4930    1.5000 N   0  0  0  0  0  0  0  0  0  0  0  0
        0.2570    0.4180    0.6920 C   0  0  0  0  0  0  0  0  0  0  0  0
       -0.0940    0.0170   -0.7160 C   0  0  0  0  0  0  0  0  0  0  0  0
       -1.0560   -0.6820   -0.9230 O   0  0  0  0  0  0  0  0  0  0  0  0
        1.2040   -0.6200    1.2960 C   0  0  0  0  0  0  0  0  0  0  0  0
        0.6610    0.4390   -1.7420 O   0  0  0  0  0  0  0  0  0  0  0  0
       -1.3830   -0.4250    1.4820 H   0  0  0  0  0  0  0  0  0  0  0  0
       -0.6760    0.6610    2.4520 H   0  0  0  0  0  0  0  0  0  0  0  0
        0.7460    1.3920    0.6820 H   0  0  0  0  0  0  0  0  0  0  0  0
        1.4590   -0.3300    2.3160 H   0  0  0  0  0  0  0  0  0  0  0  0
        0.7150   -1.5940    1.3070 H   0  0  0  0  0  0  0  0  0  0  0  0
        2.1130   -0.6760    0.6970 H   0  0  0  0  0  0  0  0  0  0  0  0
        0.4350    0.1820   -2.6470 H   0  0  0  0  0  0  0  0  0  0  0  0
      1  2  1  0  0  0  0
      1  7  1  0  0  0  0
      1  8  1  0  0  0  0
      2  3  1  0  0  0  0
      2  5  1  0  0  0  0
      2  9  1  0  0  0  0
      3  4  2  0  0  0  0
      3  6  1  0  0  0  0
      5 10  1  0  0  0  0
      5 11  1  0  0  0  0
      5 12  1  0  0  0  0
      6 13  1  0  0  0  0
    M  END
    $$$$
    <BLANKLINE>
    """

    def __init__(self, records=None):
        self._records = {}
        if records is not None:
            for mol_name, record in records.items():
                if isinstance(record, SDRecord):
                    record.header.mol_name = mol_name
                self._records[mol_name] = record

    @property
    def lines(self):
        return self.serialize().splitlines()

    @property
    def record(self):
        if len(self) == 0:
            raise ValueError("There are no records in the file")
        if len(self) > 1:
            raise ValueError("There are multiple records in the file")
        return self[next(iter(self))]

    @staticmethod
    def deserialize(text):
        """
        Create an :class:`SDFile` by deserializing the given text content.

        Parameters
        ----------
        text : str
            The content to be deserialized.

        Returns
        -------
        file_object : SDFile
            The parsed file.
        """
        lines = text.splitlines()
        record_ends = np.array(
            [i for i, line in enumerate(lines) if line.startswith(_RECORD_DELIMITER)],
            dtype=int,
        )
        if len(record_ends) == 0:
            warnings.warn(
                "Final record delimiter missing, "
                "maybe this is a MOL file instead of a SD file"
            )
            record_ends = np.array([len(lines) - 1], dtype=int)
        # The first record starts at the first line and the last
        # delimiter is at the end of the file
        # Records in the middle start directly after the delimiter
        record_starts = np.concatenate(([0], record_ends[:-1] + 1), dtype=int)
        record_names = [lines[start].strip() for start in record_starts]
        return SDFile(
            {
                # Do not include the delimiter
                # -> stop at end (instead of end + 1)
                name: _join_with_terminal_newline(lines[start:end])
                for name, start, end in zip(record_names, record_starts, record_ends)
            }
        )

    def serialize(self):
        """
        Convert this object into text content.

        Returns
        -------
        content : str
            The serialized content.
        """
        text_blocks = []
        for record_name, record in self._records.items():
            if isinstance(record, str):
                # Record is already stored as text
                text_blocks.append(record)
            else:
                try:
                    text_blocks.append(record.serialize())
                except Exception:
                    raise SerializationError(
                        f"Failed to serialize record '{record_name}'"
                    )
            text_blocks.append(_RECORD_DELIMITER + "\n")
        return "".join(text_blocks)

    @classmethod
    def read(cls, file):
        """
        Read a SD file.

        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.

        Returns
        -------
        file_object : SDFile
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
        return SDFile.deserialize(text)

    def write(self, file):
        """
        Write the contents of this object into a SD file.

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
        record = self._records[key]
        if isinstance(record, str):
            # Element is stored in serialized form
            # -> must be deserialized first
            try:
                record = SDRecord.deserialize(record)
            except Exception:
                raise DeserializationError(f"Failed to deserialize record '{key}'")
            # Update with deserialized object
            self._records[key] = record
        return record

    def __setitem__(self, key, record):
        if not isinstance(record, SDRecord):
            raise TypeError(f"Expected 'SDRecord', but got '{type(record).__name__}'")
        # The molecule name in the header is unique across the file
        record.header.mol_name = key
        self._records[key] = record

    def __delitem__(self, key):
        del self._records[key]

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for record_name in self.keys():
            if self[record_name] != other[record_name]:
                return False
        return True

    def __str__(self):
        return self.serialize()


def _join_with_terminal_newline(text_blocks):
    if len(text_blocks) == 0:
        return ""
    else:
        return "\n".join(text_blocks) + "\n"


def _empty_ctab():
    empty_atoms = AtomArray(0)
    empty_atoms.bonds = BondList(0)
    return _join_with_terminal_newline(write_structure_to_ctab(empty_atoms))


def _to_metadata_key(key):
    if isinstance(key, Metadata.Key):
        return key
    elif isinstance(key, str):
        return Metadata.Key(name=key)
    else:
        raise TypeError(
            f"Expected 'Metadata.Key' or str, but got '{type(key).__name__}'"
        )


def _add_key_value_pair(metadata, key, value):
    if key is not None:
        if value is None:
            raise DeserializationError(f"No value found for metadata key {key}")
        metadata[key] = value


def _get_ctab_stop(lines):
    for i in range(_N_HEADER, len(lines)):
        if lines[i].startswith("M  END"):
            return i + 1
    return len(lines)
