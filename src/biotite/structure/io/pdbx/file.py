# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["PDBxFile"]

import shlex
import numpy as np
from ....file import TextFile


class PDBxFile(TextFile):
    """
    This class represents a PDBx/mmCIF file.
    
    The categories of the file can be accessed using the
    `get_category()`/`set_category()` methods. The content of each
    category is represented by a dictionary. The dictionary contains
    the entry (e.g. *label_entity_id* in *atom_site*) as key. The
    corresponding values are either strings in *non-looped* categories,
    or 1-D numpy arrays of string objects in case of *looped*
    categories.
    
    A category can be changed or added using `set_category()`:
    If a string-valued dictionary is provided, a *non-looped* category
    will be created; if an array-valued dictionary is given, a
    *looped* category will be created. In case of arrays, it is
    important that all arrays have the same size.
    
    Alternatively, The content of this file can also be read/write
    accessed using dictionary-like indexing:
    You can either provide a data block and a category or only a
    category, in which case the first data block is taken.
    
    Notes
    -----
    This class is also able to detect and parse multiline entries in the
    file. However, when writing a category no multiline values are used.
    This could lead to long lines.
    
    This class uses a lazy category dictionary creation: When reading
    the file only the line positions of all categories are checked. The
    time consuming task of dictionary creation is done when
    `get_category()` is called.
    
    Examples
    --------
    Read the file and get author names

    >>> file = PDBxFile()
    >>> file.read("1l2y.cif")
    >>> author_dict = file.get_category("citation_author", block="1L2Y")
    >>> print(author_dict["name"])
    ['Neidigh, J.W.' 'Fesinmeyer, R.M.' 'Andersen, N.H.']
    
    Dictionary style indexing, no specification of data block:
    
    >>> print(file["citation_author"]["name"])
    ['Neidigh, J.W.' 'Fesinmeyer, R.M.' 'Andersen, N.H.']
    
    Get the structure from the file:
    
    >>> arr = get_structure(file)
    >>> print(type(arr))
    <class 'biotite.structure.atoms.AtomArrayStack'>
    >>> arr = get_structure(file, model=1)
    >>> print(type(arr))
    <class 'biotite.structure.atoms.AtomArray'>
    
    Modify atom array and write it back into the file:
    
    >>> arr_mod = rotate(arr, [1,2,3])
    >>> set_structure(file, arr_mod)
    >>> file.write("1l2y_mod.cif")
    """
    
    def __init__(self):
        super().__init__()
        # This dictionary saves the PDBx category names,
        # together with its line position in the file
        # and the data_block it is in
        self._categories = {}
    
    
    def read(self, file):
        super().read(file)
        # Remove emptyline at then end of file, if present
        if self.lines[-1] == "":
            del self.lines[-1]
        
        current_data_block = ""
        current_category = None
        start = -1
        stop = -1
        is_loop = False
        has_multiline_values = False
        for i, line in enumerate(self.lines):
            # Ignore empty and comment lines
            if not _is_empty(line):
                data_block_name = _data_block_name(line)
                if data_block_name is not None:
                    data_block = data_block_name
                    # If new data block begins, reset category data
                    current_category = None
                    start = -1
                    stop = -1
                    is_loop = False
                    has_multiline_values = False
                
                is_loop_in_line = _is_loop_start(line)
                category_in_line = _get_category_name(line)
                if is_loop_in_line or (category_in_line != current_category
                                       and category_in_line is not None):
                    # Start of a new category
                    # Add an entry into the dictionary with the old category
                    stop = i
                    self._add_category(data_block, current_category, start,
                                       stop, is_loop, has_multiline_values)
                    # Track the new category
                    if is_loop_in_line:
                        # In case of lines with "loop_" the category is in the
                        # next line
                        category_in_line = _get_category_name(self.lines[i+1])
                    is_loop = is_loop_in_line
                    current_category = category_in_line
                    start = i
                    has_multiline_values = False
                
                multiline = _is_multi(line, is_loop)
                if multiline:
                    has_multiline_values = True
        # Add the entry for the final category
        # Since at the end of the file the end of the category
        # is not determined by the start of a new one,
        # this needs to be handled separately
        stop = len(self.lines)
        self._add_category(data_block, current_category, start,
                           stop, is_loop, has_multiline_values)
    
    
    def get_block_names(self):
        """
        Get the names of all data blocks in the file.
        
        Returns
        -------
        blocks : list
            List of data block names.
        """
        blocks = []
        for category_tuple in self._categories.keys():
            block = category_tuple[0]
            if block not in blocks:
                blocks.append(block)
        return blocks
    
    
    def get_category(self, category, block=None):
        """
        Get the dictionary for a given category.
        
        Parameters
        ----------
        category : string
            The name of the category. The leading underscore is omitted.
        block : string, optional
            The name of the data block. Default is the first
            (and most times only) data block of the file.
        
        This function optionally uses C-extensions.
            
        Returns
        -------
        category_dict : dict
            A entry keyed dictionary. The corresponding values are
            strings or `ndarrays` of strings for *non-looped* and
            *looped* categories, respectively.
        """
        if block is None:
            block = self.get_block_names()[0]
        category_info = self._categories[(block, category)]
        start = category_info["start"]
        stop = category_info["stop"]
        is_loop = category_info["loop"]
        is_multilined = category_info["multiline"]
        
        if is_multilined:
            # Convert multiline values into singleline values
            prelines = [line.strip() for line in self.lines[start:stop]
                         if not _is_empty(line) and not _is_loop_start(line)]
            lines = (len(prelines)) * [None]
            # lines index
            k = 0
            # prelines index
            i = 0
            while i < len(prelines):
                if prelines[i][0] == ";":
                    # multiline values
                    lines[k-1] += " '" + prelines[i][1:]
                    j = i+1
                    while prelines[j] != ";":
                        lines[k-1] += prelines[j]
                        j += 1
                    lines[k-1] += "'"
                    i = j+1
                elif not is_loop and prelines[i][0] in ["'",'"']:
                    # Singleline values where value is in the line
                    # after the corresponding key
                    lines[k-1] += " " + prelines[i]
                    i += 1
                else:    
                    # Normal singleline value in the same row as the key
                    lines[k] = prelines[i]
                    i += 1
                    k += 1
            lines = [line for line in lines if line is not None]
            
        else:
            lines = [line.strip() for line in self.lines[start:stop]
                     if not _is_empty(line) and not _is_loop_start(line)]
        
        if is_loop:
            # Special optimization for "atom_site":
            # Even if the values are quote protected,
            # no whitespace is expected in escaped values
            # Therefore slow shlex.split() call is not necessary
            if category == "atom_site":
                whitespace_values = False
            else:
                whitespace_values = True
            category_dict = _process_looped(lines, whitespace_values)
        else:
            category_dict = _process_singlevalued(lines)
        
        return category_dict
            
    
    def set_category(self, category, category_dict, block=None):
        """
        Set the content of a category.
        
        If the category is already exisiting, all lines corresponding
        to the category are replaced. Otherwise a new category is
        created and the lines are appended at the end of the data block.
        
        Parameters
        ----------
        category : string
            The name of the category. The leading underscore is omitted.
        category_dict : dict
            The category content. The dictionary must have strings
            (subcategories) as keys and strings or `ndarrays` as
            values.
        block : string, optional
            The name of the data block. Default is the first
            (and most times only) data block of the file. If the
            block is not contained in the file yet, a new block is
            appended at the end of the file..
        """
        if block is None:
            block = self.get_block_names()[0]
        
        sample_category_value = list(category_dict.values())[0]
        if (isinstance(sample_category_value, (np.ndarray, list))):
            is_looped = True
            arr_len = len(list(category_dict.values())[0])
            # Check whether all arrays have the same length
            for subcat, array in category_dict.items():
                if len(array) != arr_len:
                    raise ValueError(
                        f"Length of Subcategory '{subcat}' is {len(array)}, "
                        f" but {arr_len} was expected"
                    )
        else:
            is_looped = False
        
        
        # Value arrays (looped categories) can be modified (e.g. quoted)
        # Hence make a copy to avoid unwaned side effects
        # due to modification of input values
        if is_looped:
            category_dict = {key : val.copy() for key, val
                             in category_dict.items()}

        # Enclose values with quotes if required
        for key, value in category_dict.items():
            if is_looped:
                for i in range(len(value)):
                    value[i] = _quote(value[i])
            else:
                category_dict[key] = _quote(value)
        
        if is_looped:
            keylines = ["_" + category + "." + key
                         for key in category_dict.keys()]
            value_arr = list(category_dict.values())
            # Array containing the number of characters + whitespace
            # of each column 
            col_lens = np.zeros(len(value_arr), dtype=int)
            for i, column in enumerate(value_arr):
                col_len = 0
                for value in column:
                    if len(value) > col_len:
                        col_len = len(value)
                # Length of column is max value length 
                # +1 whitespace character as separator 
                col_lens[i] = col_len+1
            valuelines = [""] * arr_len
            for i in range(arr_len):
                for j, arr in enumerate(value_arr):
                    valuelines[i] += arr[i] + " "*(col_lens[j] - len(arr[i]))
            newlines = ["loop_"] + keylines + valuelines
            
        else:
            # For better readability, not only one space is inserted
            # after each key, but as much spaces that every value starts
            # at the same position in the line
            max_len = 0
            for key in category_dict.keys():
                if len(key) > max_len:
                    max_len = len(key)
            # "+3" Because of three whitespace chars after longest key
            req_len = max_len + 3
            newlines = ["_" + category + "." + key
                         + " " * (req_len-len(key)) + value
                         for key, value in category_dict.items()]
            
        # A comment line is set after every category
        newlines += ["#"]
        
        if (block,category) in self._categories:
            # Category already exists in data block
            category_info = self._categories[(block, category)]
            # Insertion point of new lines
            old_category_start = category_info["start"]
            old_category_stop = category_info["stop"]
            category_start = old_category_start 
            # Difference between number of lines of the old and new category
            len_diff = len(newlines) - (old_category_stop-old_category_start)
            # Remove old category content
            del self.lines[old_category_start : old_category_stop]
            # Insert new lines at category start
            self.lines[category_start:category_start] = newlines
            # Update category info
            category_info["start"] = category_start
            category_info["stop"] = category_start + len(newlines)
            # When writing a category no multiline values are used
            category_info["multiline"] = False
            category_info["loop"] = is_looped
        elif block in self.get_block_names():
            # Data block exists but not the category
            # Find last category in the block
            # and set start of new category to stop of last category
            last_stop = 0
            for category_tuple, category_info in self._categories.items():
                if block == category_tuple[0]:
                    if last_stop < category_info["stop"]:
                        last_stop = category_info["stop"]
            category_start = last_stop
            category_stop = category_start + len(newlines)
            len_diff = len(newlines)
            self.lines[category_start:category_start] = newlines
            self._add_category(block, category, category_start, category_stop,
                               is_looped, is_multilined=False)
        else:
            # The data block does not exist
            # Put the begin of data block in front of newlines
            newlines = ["data_"+block, "#"] + newlines
            # Find last category in the file
            # and set start of new data_block with new category
            # to stop of last category
            last_stop = 0
            for category_info in self._categories.values():
                if last_stop < category_info["stop"]:
                    last_stop = category_info["stop"]
            category_start = last_stop + 2
            category_stop = last_stop + len(newlines)
            len_diff = len(newlines)-2
            self.lines[last_stop:last_stop] = newlines
            self._add_category(block, category, category_start, category_stop,
                               is_looped, is_multilined=False)
        # Update start and stop of all categories appearing after the
        # changed/added category
        for category_info in self._categories.values():
            if category_info["start"] > category_start:
                category_info["start"] += len_diff
                category_info["stop"] += len_diff
    
    
    def __copy_fill__(self, clone):
        super().__copy_fill__(clone)
        clone._categories = copy.deepcopy(self._categories)
        
        
    def __setitem__(self, index, item):
        if isinstance(index, tuple):
            self.set_category(index[1], item, block=index[0])
        else:
            self.set_category(index, item)
    
    
    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self.get_category(index[1], block=index[0])
        else:
            return self.get_category(index)
    
    
    def _add_category(self, block, category_name,
                      start, stop, is_loop, is_multilined):
        # Before the first category starts,
        # the current_category is None
        # This is checked before adding an entry
        if category_name is not None:
            self._categories[
                (block, category_name)] = {"start"     : start,
                                           "stop"      : stop,
                                           "loop"      : is_loop,
                                           "multiline" : is_multilined}
    
    
def _process_singlevalued(lines):
    category_dict = {}
    for line in lines:
        parts = shlex.split(line)
        key = parts[0].split(".")[1]
        value = parts[1]
        category_dict[key] = value
    return category_dict


def _process_looped(lines, whitepace_values):
    category_dict = {}
    keys = []
    # Array index
    i = 0
    # Dictionary key index
    j = 0
    for line in lines:
        if line[0] == "_":
            # Key line
            key = line.split(".")[1]
            keys.append(key)
            # Pessimistic array allocation
            # numpy array filled with strings
            category_dict[key] = np.zeros(len(lines), dtype=object)
            keys_length = len(keys)
        else:
            # If whitespace is expected in quote protected values,
            # use standard shlex split
            # Otherwise use much more faster whitespace split
            # and quote removal if applicable,
            # bypassing the slow shlex module 
            if whitepace_values:
                values = shlex.split(line)
            else:
                values = line.split()
                for k in range(len(values)):
                    # Remove quotes
                    if ((values[k][0] == '"' and values[k][-1] == '"') or
                        (values[k][0] == "'" and values[k][-1] == "'")):
                            values[k] = values[k][1:-1]
            for value in values:
                category_dict[keys[j]][i] = value
                j += 1
                if j == keys_length:
                    # If all keys have been filled with a value,
                    # restart with first key with incremented index
                    j = 0
                    i += 1
    for key in category_dict.keys():
        # Trim to correct size
        category_dict[key] = category_dict[key][:i]
    return category_dict
    

def _is_empty(line):
    return len(line) == 0 or line[0] == "#"


def _data_block_name(line):
    if line.startswith("data_"):
        return line[5:]
    else:
        return None


def _is_loop_start(line):
    return line.startswith("loop_")


def _is_multi(line, is_loop):
    if is_loop:
        return line[0] == ";"
    else:
        return line[0] in [";","'",'"']


def _get_category_name(line):
    if line[0] != "_":
        return None
    else:
        return line[1:line.find(".")]


def _quote(value):
    if "'" in value:
        return('"' + value + '"')
    elif '"' in value:
        return("'" + value + "'")
    elif " " in value:
        return("'" + value + "'")
    else:
        return value
