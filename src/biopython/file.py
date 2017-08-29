# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import abc

__all__ = ["File", "TextFile", "register_suffix", "read"]


_suffix_dict = {}

class File(metaclass=abc.ABCMeta):
    
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def read(self, file_name):
        pass
    
    @abc.abstractmethod
    def write(self, file_name):
        pass
    
    @abc.abstractmethod
    def copy():
        pass


def register_suffix(suffixes):
    def decorator(file_cls):
        for suffix in suffixes:
            _suffix_dict[suffix.lower()] = file_cls
        return file_cls
    return decorator


def read(file_name):
    suffix = file_name.split(".")[-1].lower()
    try:
        file_cls = _suffix_dict[suffix]
    except KeyError:
        raise ValueError("'{:}' suffix is not supported".format(suffix))
    file = file_cls()
    file.read(file_name)
    return file
        

class TextFile(File, metaclass=abc.ABCMeta):
    
    def __init__(self):
        self._lines = []
    
    def read(self, file_name):
        with open(file_name, "r") as f:
            str_data = f.read()
        self._lines = str_data.split("\n")
    
    def write(self, file_name):
        with open(file_name, "w") as f:
            f.writelines([line+"\n" for line in self._lines])