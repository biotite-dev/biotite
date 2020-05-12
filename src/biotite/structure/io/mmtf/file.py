# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mmtf"
__author__ = "Patrick Kunzmann"
__all__ = ["MMTFFile"]

import io
from collections.abc import MutableMapping
import struct
import copy
import numpy as np
import msgpack
from ....file import File, is_binary
from ...error import BadStructureError
from .decode import decode_array
from .encode import encode_array


class MMTFFile(File, MutableMapping):
    """
    This class represents a MMTF file.
    
    When reading a file, the *MessagePack* unpacker is used to create
    a dictionary of the file content.
    This dictionary is accessed by indexing the :class:`MMTFFile`
    instance directly with the dictionary keys.
    If the dictionary value is an encoded array, the value automatically
    decoded.
    Decoded arrays are always returned as :class:`ndarray` instances.
    
    Examples
    --------
    
    >>> import os.path
    >>> mmtf_file = MMTFFile.read(os.path.join(path_to_structures, "1l2y.mmtf"))
    >>> print(mmtf_file["title"])
    NMR Structure of Trp-Cage Miniprotein Construct TC5b
    >>> print(mmtf_file["chainNameList"])
    ['A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A'
     'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A' 'A'
     'A' 'A']
    """
    
    def __init__(self):
        super().__init__()
        self._content = {}
        self._content["mmtfVersion"] = "1.0.0"
        self._content["mmtfProducer"] = "UNKNOWN"
    
    @classmethod
    def read(self, file):
        """
        Read a MMTF file.
        
        Parameters
        ----------
        file : file-like object or str
            The file to be read.
            Alternatively a file path can be supplied.
        
        Returns
        -------
        file_object : MMTFFile
            The parsed file.
        """
        mmtf_file = MMTFFile()
        # File name
        if isinstance(file, str):
            with open(file, "rb") as f:
                mmtf_file._content = msgpack.unpackb(
                    f.read(), use_list=True, raw=False
                )
        # File object
        else:
            if not is_binary(file):
                raise TypeError("A file opened in 'binary' mode is required")
            mmtf_file._content = msgpack.unpackb(
                file.read(), use_list=True, raw=False
            )
        return mmtf_file
    
    def write(self, file):
        """
        Write contents into a MMTF file.
        
        Parameters
        ----------
        file : file-like object or str
            The file to be written to.
            Alternatively, a file path can be supplied.
        """
        packed_bytes = msgpack.packb(
            self._content, use_bin_type=True, default=_encode_numpy
        )
        if isinstance(file, str):
            with open(file, "wb") as f:
                f.write(packed_bytes)
        else:
            if not is_binary(file):
                raise TypeError("A file opened in 'binary' mode is required")
            file.write(packed_bytes)
    
    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        clone._content = copy.deepcopy(self._content)
    
    def get_codec(self, key):
        """
        Obtain the codec ID of an MMTF encoded value.
        
        Parameters
        ----------
        key : str
            The key for the potentially encoded value.
        
        Returns
        -------
        codec : int or None
            The codec ID. `None` if the value is not encoded.
        """
        data = self._content[key]
        if isinstance(data, bytes) and data[0] == 0:
            codec = struct.unpack(">i", data[0:4])[0]
            return codec
        else:
            return None
    
    def get_length(self, key):
        """
        Obtain the length of an MMTF encoded value.
        
        Parameters
        ----------
        key : str
            The key for the potentially encoded value.
        
        Returns
        -------
        codec : int or None
            The length of the `bytes` array.
            `None` if the value is not encoded.
        """
        data = self._content[key]
        if isinstance(data, bytes) and data[0] == 0:
            length = struct.unpack(">i", data[4:8])[0]
            return length
        else:
            return None
    
    def get_param(self, key):
        """
        Obtain the parameter of an MMTF encoded value.
        
        Parameters
        ----------
        key : str
            The key for the potentially encoded value.
        
        Returns
        -------
        codec : int or None
            The parameter of the encoded value.
            `None` if the value is not encoded.
        """
        data = self._content[key]
        if isinstance(data, bytes) and data[0] == 0:
            param = struct.unpack(">i", data[8:12])[0]
            return param
        else:
            return None
    
    def set_array(self, key, array, codec, param=0):
        length = len(array)
        raw_bytes = encode_array(array, codec, param)
        data = struct.pack(">i", codec) \
             + struct.pack(">i", length) \
             + struct.pack(">i", param) \
             + raw_bytes
        self._content[key] = data
    
    def __getitem__(self, key):
        data = self._content[key]
        if isinstance(data, bytes) and data[0] == 0:
            # MMTF specific format -> requires decoding
            codec     = struct.unpack(">i", data[0:4 ])[0]
            length    = struct.unpack(">i", data[4:8 ])[0]
            param     = struct.unpack(">i", data[8:12])[0]
            raw_bytes = data[12:]
            return decode_array(codec, raw_bytes, param)
        else:
            return data
    
    def __setitem__(self, key, item):
        if isinstance(item, np.ndarray):
            raise TypeError("Arrays that need to be encoded must be addeed "
                            "via 'set_array()'")
        self._content[key] = item
    
    def __delitem__(self, key):
        del self._content[key]
    
    def __iter__(self):
        return self._content.__iter__()
    
    def __len__(self):
        return len(self._content)
    
    def __contains__(self, item):
        return item in self._content


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
