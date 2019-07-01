# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["GFFFile"]

from collections.abc import MutableSequence
from ....file import TextFile


class GFFFile(TextFile, MutableSequence):
    """
    This class represents a file in GFF3 format.
    """
    
    def __init__(self, chars_per_line=80):
    
    def read(self, file):
        super().read(file)
        self._index_comments_and_directives()
    
    def insert(self, item):
        pass
        
    def __setitem__(self, index, item):
        pass
    
    def __getitem__(self, index):
        pass
        return seq_string
    
    def __delitem__(self, index):
        pass
    
    def __len__(self):
        return len(self._entries)
    
    def _index_comments_and_directives(self):
        """
        Parse the file for comment and directive lines.
        Count these lines cumulatively, so that entry indices can be
        mapped onto line indices.
        Additionally track the line index of directive lines.
        """    