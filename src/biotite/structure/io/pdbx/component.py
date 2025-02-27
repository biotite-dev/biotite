# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module contains internally abstract classes for representing parts
of CIF/BinaryCIF files, such as categories and columns.
"""

__name__ = "biotite.structure.io.pdbx"
__author__ = "Patrick Kunzmann"
__all__ = ["MaskValue"]

from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping
from enum import IntEnum
from biotite.file import DeserializationError, SerializationError


class MaskValue(IntEnum):
    """
    This enum type represents the possible values of a mask array.

    - `PRESENT` : A value is present.
    - `INAPPLICABLE` : For this row no value is applicable or
      inappropriate (``.`` in *CIF*).
      In some cases it may also refer to a default value for the
      respective column.
    - `MISSING` : For this row the value is missing or unknown
      (``?`` in *CIF*).
    """

    PRESENT = 0
    INAPPLICABLE = 1
    MISSING = 2


class _Component(metaclass=ABCMeta):
    """
    Base class for all components in a CIF/BinaryCIF file.
    """

    @staticmethod
    def subcomponent_class():
        """
        Get the class of the components that are stored in this component.

        Returns
        -------
        subcomponent_class : type
            The class of the subcomponent.
            If this component already represents the lowest level, i.e.
            it does not contain subcomponents, ``None`` is
            returned.
        """
        return None

    @staticmethod
    def supercomponent_class():
        """
        Get the class of the component that contains this component.

        Returns
        -------
        supercomponent_class : type
            The class of the supercomponent.
            If this component present already the highest level, i.e.
            it is not contained in another component, ``None`` is
            returned.
        """
        return None

    @staticmethod
    @abstractmethod
    def deserialize(content):
        """
        Create this component by deserializing the given content.

        Parameters
        ----------
        content : str or dict
            The content to be deserialized.
            The type of this parameter depends on the file format.
            In case of *CIF* files, this is the text of the lines
            that represent this component.
            In case of *BinaryCIF* files, this is a dictionary
            parsed from the *MessagePack* data.
        """
        raise NotImplementedError()

    @abstractmethod
    def serialize(self):
        """
        Convert this component into a Python object that can be written
        to a file.

        Returns
        -------
        content : str or dict
            The content to be serialized.
            The type of this return value depends on the file format.
            In case of *CIF* files, this is the text of the lines
            that represent this component.
            In case of *BinaryCIF* files, this is a dictionary
            that can be encoded into *MessagePack*.
        """
        raise NotImplementedError()

    def __str__(self):
        return str(self.serialize())


class _HierarchicalContainer(_Component, MutableMapping, metaclass=ABCMeta):
    """
    A container for hierarchical data in BinaryCIF files.
    For example, the file contains multiple blocks, each block contains
    multiple categories and each category contains multiple columns.

    It uses lazy deserialization:
    A component is only deserialized from the serialized data, if it
    is accessed.
    The deserialized component is then cached in the container.

    Parameters
    ----------
    elements : dict, optional
        The initial elements of the container.
        By default no initial elements are added.
    """

    def __init__(self, elements=None):
        if elements is None:
            elements = {}
        for element in elements.values():
            if not isinstance(element, (dict, self.subcomponent_class())):
                raise TypeError(
                    f"Expected '{self.subcomponent_class().__name__}', "
                    f"but got '{type(element).__name__}'"
                )
        self._elements = elements

    @staticmethod
    def _deserialize_elements(content, take_key_from):
        """
        Lazily deserialize the elements of this container.

        Parameters
        ----------
        content : dict
            The serialized content describing the elements for this
            container.
        take_key_from : str
            The key in each element of `content`, whose value is used as
            the key for the respective element.

        Returns
        -------
        elements : dict
            The elements that should be stored in this container.
            This return value can be given to the constructor.
        """
        elements = {}
        for serialized_element in content:
            key = serialized_element[take_key_from]
            # Lazy deserialization
            # -> keep serialized for now and deserialize later if needed
            elements[key] = serialized_element
        return elements

    def _serialize_elements(self, store_key_in=None):
        """
        Serialize the elements that are stored in this container.

        Each element that is still serialized (due to lazy
        deserialization), is kept as it is.

        Parameters
        ----------
        store_key_in: str, optional
            If given, the key of each element is stored as value in the
            serialized element.
            This is basically the reverse operation of `take_key_from` in
            :meth:`_deserialize_elements()`.
        """
        serialized_elements = []
        for key, element in self._elements.items():
            if isinstance(element, self.subcomponent_class()):
                try:
                    serialized_element = element.serialize()
                except Exception:
                    raise SerializationError(f"Failed to serialize element '{key}'")
            else:
                # Element is already stored in serialized form
                serialized_element = element
            if store_key_in is not None:
                serialized_element[store_key_in] = key
                serialized_elements.append(serialized_element)
        return serialized_elements

    def __getitem__(self, key):
        element = self._elements[key]
        if not isinstance(element, self.subcomponent_class()):
            # Element is stored in serialized form
            # -> must be deserialized first
            try:
                element = self.subcomponent_class().deserialize(element)
            except Exception:
                raise DeserializationError(f"Failed to deserialize element '{key}'")
            # Update container with deserialized object
            self._elements[key] = element
        return element

    def __setitem__(self, key, element):
        if isinstance(element, self.subcomponent_class()):
            pass
        elif isinstance(element, _HierarchicalContainer):
            # A common mistake may be to use the wrong container type
            raise TypeError(
                f"Expected '{self.subcomponent_class().__name__}', "
                f"but got '{type(element).__name__}'"
            )
        else:
            try:
                element = self.subcomponent_class().deserialize(element)
            except Exception:
                raise DeserializationError("Failed to deserialize given value")
        self._elements[key] = element

    def __delitem__(self, key):
        del self._elements[key]

    # Implement `__contains__()` explicitly,
    # because the mixin method unnecessarily deserializes the value, if available
    def __contains__(self, key):
        return key in self._elements

    def __iter__(self):
        return iter(self._elements)

    def __len__(self):
        return len(self._elements)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for key in self.keys():
            if self[key] != other[key]:
                return False
        return True
