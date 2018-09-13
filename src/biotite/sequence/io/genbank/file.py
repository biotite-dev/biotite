# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

from ....file import TextFile, InvalidFileError
from ...annotation import Location, Feature, Annotation, AnnotatedSequence
from ...seqtypes import NucleotideSequence, ProteinSequence 
import textwrap
import copy
import re
import io

__all__ = ["GenBankFile", "GenPeptFile", "MultiFile"]


class GenBankFile(TextFile):
    """
    This class represents a file in GenBank format.
    
    A GenBank file provides 3 kinds of information:
    At first it contains some general information about the file, like
    IDs, database relations and source organism.
    Secondly it contains sequence annotations, i.e. the positions of
    a reference sequence, that fulfill certain roles, like promoters or
    coding sequences.
    At last the file contains optionally the reference sequence.
    
    As of now, GenBank files can only be parsed, writing GenBank files
    is not supported at this point.
    
    Examples
    --------
    
    >>> file = gb.GenBankFile()
    >>> file.read("path/to/ec_bl21.gb")
    >>> print(file.get_definition())
    Escherichia coli BL21(DE3), complete genome.
    >>> for f in file.get_annotation(include_only=["CDS"]):
    ...     if "gene" in f.qual and "lac" in f.qual["gene"]:
    ...         for loc in f.locs:
    ...             print(f.qual["gene"], loc.strand, loc.first, loc.last)
    lacA Strand.REVERSE 330784 331395
    lacY Strand.REVERSE 331461 332714
    lacZ Strand.REVERSE 332766 335840
    lacI Strand.REVERSE 335963 337045
    lacI Strand.FORWARD 748736 749818
    """
    
    def __init__(self):
        super().__init__()
        # Field start and stop indices in list of lines
        # and names of categories
        self._fields = []
    
    def read(self, file):
        super().read(file)
        start = -1
        stop = -1
        name = ""
        self._fields = []
        for i, line in enumerate(self.lines):
            # Check if line contains a new major field
            # (Header beginning from first column)
            if len(line) != 0 and line[0] != " " and line[:2] != "//":
                stop = i
                if start != -1:
                    # Store previous field
                    self._fields.append((start, stop, name))
                start = i
                name = line[0:12].strip()
        # Store last field
        stop = i
        self._fields.append((start, stop, name))
    
    def write(self, file):
        """
        Not implemented yet.
        """
        raise NotImplementedError()
    
    def get_locus(self):
        """
        Parse the *LOCUS* field of the file.
        
        Returns
        ----------
        locus_dict : dict
            A dictionary storing the locus *name*, *length*, *type*,
            *division* and *date*.
        
        Examples
        --------
        
        >>> file = gb.GenBankFile()
        >>> file.read("path/to/ec_bl21.gb")
        >>> for key, val in file.get_locus().items():
        ...     print(key, ":", val)
        name : CP001509
        length : 4558953
        type : DNA circular
        division : BCT
        date : 16-FEB-2017
        """
        locus_dict = {}
        starts, stops = self._get_field_indices("LOCUS")
        locus_info = self.lines[starts[0]].split()
        locus_dict["name"] = locus_info[1]
        locus_dict["length"] = locus_info[2]
        locus_dict["type"] = locus_info[4]
        if locus_info[5] in ["circular", "linear"]:
            locus_dict["type"] += " " + locus_info[5]
        locus_dict["division"] = locus_info[-2]
        locus_dict["date"] = locus_info[-1]
        return locus_dict
    
    def get_definition(self):
        """
        Parse the *DEFINITION* field of the file.
        
        Returns
        ----------
        definition : str
            Content of the *DEFINITION* field.
        """
        starts, stops = self._get_field_indices("DEFINITION")
        return self.lines[starts[0]][12:].strip()
    
    def get_accession(self):
        """
        Parse the *ACCESSION* field of the file.
        
        Returns
        ----------
        accession : str
            Content of the *ACCESSION* field.
        """
        starts, stops = self._get_field_indices("ACCESSION")
        return self.lines[starts[0]][12:].strip()
    
    def get_version(self):
        """
        Parse the *VERSION* field of the file.
        
        Returns
        ----------
        version : str
            Content of the *VERSION* field. Does not include GI.
        """
        starts, stops = self._get_field_indices("VERSION")
        return self.lines[starts[0]][12:].split()[0]
    
    def get_gi(self):
        """
        Get the GI of the file.
        
        Returns
        ----------
        gi : str
            The GI of the file.
        """
        starts, stops = self._get_field_indices("VERSION")
        version_info = self.lines[starts[0]][12:].split()
        if len(version_info) < 2 or "GI" not in version_info[1]:
            raise InvalidFileError("File does not contain GI")
        # Truncate GI
        return version_info[1][3:]
    
    def get_db_link(self):
        """
        Parse the *DBLINK* field of the file.
        
        Returns
        ----------
        link_dict : dict
            A dictionary storing the database links, with the database
            name as key, and the corresponding ID as value.
        
        Examples
        --------
        
        >>> file = gb.GenBankFile()
        >>> file.read("path/to/ec_bl21.gb")
        >>> for key, val in file.get_db_link().items():
        ...     print(key, ":", val)
        BioProject : PRJNA20713
        BioSample : SAMN02603478
        """
        starts, stops = self._get_field_indices("DBLINK")
        link_dict = {}
        for i in range(starts[0], stops[0]):
            line = self.lines[i]
            content = line[12:].split(":")
            link_dict[content[0].strip()] = content[1].strip()
        return link_dict
    
    def get_source(self):
        """
        Parse the *SOURCE* field of the file.
        
        Returns
        ----------
        source : str
            Organism name corresponding to this file.
        """
        starts, stops = self._get_field_indices("SOURCE")
        return self.lines[starts[0]][12:].strip()
    
    def get_references(self):
        """
        Parse the *REFERENCE* fields of the file.
        
        Returns
        ----------
        ref_list : list
            A list, where each element is a dictionary storing
            the reference information for one reference.
        """
        references = []
        starts, stops = self._get_field_indices("REFERENCE")
        for i in range(len(starts)):
            start = starts[i]
            stop = stops[i]
            ref_dict_raw = self._get_minor_fields(start, stop)
            ref_dict = {}
            for key, val in ref_dict_raw.items():
                if key == "REFERENCE":
                    loc_info = val[val.index("(")+1 : val.index(")")].split()
                    first_base = int(loc_info[loc_info.index("to") -1])
                    last_base  = int(loc_info[loc_info.index("to") +1])
                    ref_dict["location"] = (first_base, last_base)
                elif key == "AUTHORS":
                    ref_dict["authors"] = val
                elif key == "TITLE":
                    ref_dict["title"] = val
                elif key == "JOURNAL":
                    ref_dict["journal"] = val
                elif key == "PUBMED":
                    ref_dict["pubmed"] = val
                elif key == "REMARK":
                    ref_dict["remark"] = val
            references.append(ref_dict)
        return references
    
    def get_comment(self):
        """
        Parse the *COMMENT* field of the file.
        
        Returns
        ----------
        comment : str
            Content of the *COMMENT* field.
        """
        starts, stops = self._get_field_indices("COMMENT")
        comment = ""
        for line in self.lines[starts[0] : stops[0]]:
            comment += line[12:].strip() + " "
        return comment.strip()
    
    def get_annotation(self, include_only=None):
        """
        Get the sequence annotation from the *ANNOTATION* field.
        
        Parameters
        ----------
        include_only : iterable object, optional
            List of names of feature keys (`str`), which should included
            in the annotation. By default all features are included.
        
        Returns
        ----------
        annotation : Annotation
            Sequence annotation from the file.
        """
        starts, stops = self._get_field_indices("FEATURES")
        # Remove the first line,
        # because it contains the "FEATURES" string itself.
        start = starts[0] +1
        stop = stops[0]
        
        feature_list = []
        feature_key = None
        feature_value = ""
        for i in range(start, stop):
            line = self.lines[i]
            # Check if line contains feature key
            if line[5] != " ":
                if feature_key is not None:
                    # Store old feature key and value
                    feature_list.append((feature_key, feature_value))
                # Track new key
                feature_key = line[5:20].strip()
                feature_value = ""
            feature_value += line[21:] + " "
        # Store last feature key and value (loop already exited)
        feature_list.append((feature_key, feature_value))
        
        # Process only relevant features and put them into Annotation
        annotation = Annotation()
        regex = re.compile(r"""(".*?"|/.*?=)""")
        for key, val in feature_list:
            if include_only is None or key in include_only:
                qual_dict = {}
                qualifiers = [s.strip() for s in regex.split(val)]
                # Remove empty quals
                qualifiers = [s for s in qualifiers if s]
                # First string is location identifier
                locs = _parse_locs(qualifiers.pop(0).strip())
                qual_key = None
                qual_val = None
                for qual in qualifiers:
                    if qual[0] == "/":
                        # Store previous qualifier pair
                        if qual_key is not None:
                            # In case of e.g. '/pseudo'
                            # 'qual_val' is 'None'
                            qual_dict[qual_key] = qual_val
                            qual_key = None
                            qual_val = None
                        # Qualifier key
                        # -> remove "/" and "="
                        qual_key = qual[1:-1]
                    else:
                        # Qualifier value
                        # -> remove potential quotes
                        if qual[0] == '"':
                            qual_val = qual[1:-1]
                        else:
                            qual_val = qual
                # Store final qualifier pair
                if qual_key is not None:
                    qual_dict[qual_key] = qual_val
                annotation.add_feature(Feature(key, locs, qual_dict))
        return annotation
    
    def get_sequence(self):
        """
        Get the sequence from the *ORIGIN* field.
        
        Returns
        ----------
        sequence : NucleotideSequence
            The reference sequence in the file.
        """
        seq_str = self._get_seq_string()
        if len(seq_str) == 0:
            raise InvalidFileError("The file does not contain "
                                   "sequence information")
        return NucleotideSequence(seq_str)
    
    def get_annotated_sequence(self, include_only=None):
        """
        Get an annotated sequence by combining the *ANNOTATION* and
        *ORIGIN* fields.
        
        Parameters
        ----------
        include_only : iterable object, optional
            List of names of feature keys (`str`), which should included
            in the annotation. By default all features are included.
        
        Returns
        ----------
        annot_seq : AnnotatedSequence
            The annotated sequence.
        """
        sequence = self.get_sequence()
        annotation = self.get_annotation(include_only)
        return AnnotatedSequence(annotation, sequence)
    
    def _get_seq_string(self):
        starts, stops = self._get_field_indices("ORIGIN")
        seq_str = ""
        # Remove numbers, emtpy spaces and the '//' at end of file
        regex = re.compile("[0-9]| |/")
        for i in range(starts[0]+1, stops[0]):
            seq_str += regex.sub("", self.lines[i])
        return seq_str
            

    def _get_field_indices(self, name):
        starts = []
        stops = []
        for field in self._fields:
            if field[2] == name:
                starts.append(field[0])
                stops.append(field[1])
        if len(starts) == 0:
            raise InvalidFileError(f"File does not contain '{name}' category")
        return starts, stops
    
    def _get_minor_fields(self, field_start, field_stop):
        header = ""
        content = ""
        minor_field_dict = {}
        for i in range(field_start, field_stop):
            line = self.lines[i]
            # Check if line contains a new minor field
            # (Header beginning from first column)
            if len(line) != 0 and line[:12].strip() != "":
                # Store previous minor field
                if content != "":
                    minor_field_dict[header] = content
                header = line[:12].strip()
                content = line[12:].strip()
            else:
                content += " " + line[12:].strip()
        # Store last minor field
        minor_field_dict[header] = content
        return minor_field_dict


def _parse_locs(loc_str):
    locs = []
    if loc_str.startswith(("join", "order")):
        str_list = loc_str[loc_str.index("(")+1:loc_str.rindex(")")].split(",")
        for s in str_list:
            locs.extend(_parse_locs(s))
    elif loc_str.startswith("complement"):
        compl_str = loc_str[loc_str.index("(")+1:loc_str.rindex(")")]
        compl_locs = _parse_locs(compl_str)
        for loc in compl_locs:
            loc.strand = Location.Strand.REVERSE
        locs.extend(compl_locs)
    else:
        locs = [_parse_single_loc(loc_str)]
    return locs


def _parse_single_loc(loc_str):
    if ".." in loc_str:
        split_char = ".."
        defect = Location.Defect(0)
    elif "." in loc_str:
        split_char = "."
        defect = Location.Defect.UNK_LOC
    elif "^" in loc_str:
        split_char = "^"
        loc_str_split = loc_str.split("..")
        defect = Location.Defect.BETWEEN
    else:
        first_and_last = int(loc_str)
        return Location(first_and_last, first_and_last)
    loc_str_split = loc_str.split(split_char)
    first_str = loc_str_split[0]
    last_str = loc_str_split[1]
    if first_str[0] == "<":
        first = int(first_str[1:])
        defect |= Location.Defect.BEYOND_LEFT
    else:
        first = int(first_str)
    if last_str[0] == ">":
        last = int(last_str[1:])
        defect |= Location.Defect.BEYOND_RIGHT
    else:
        last = int(last_str)
    return Location(first, last, defect=defect)



class GenPeptFile(GenBankFile):
    """
    This class represents a file in GenBank related GenPept format.
    
    See also
    --------
    GenBankFile
    """
    
    def get_locus(self):
        """
        Parse the *LOCUS* field of the file.
        
        Returns
        ----------
        locus_dict : dict
            A dictionary storing the locus *name*, *length*,
            *division* and *date*.
        
        Examples
        --------
        
        >>> file = gb.GenPeptFile()
        >>> file.read("path/to/bt_lysozyme.gp")
        >>> for key, val in file.get_locus().items():
        ...     print(key, ":", val)
        name : AAC37312
        length : 147
        division : MAM
        date : 27-APR-1993
        """
        locus_dict = {}
        starts, stops = self._get_field_indices("LOCUS")
        locus_info = self.lines[starts[0]].split()
        locus_dict["name"] = locus_info[1]
        locus_dict["length"] = locus_info[2]
        locus_dict["division"] = locus_info[-2]
        locus_dict["date"] = locus_info[-1]
        return locus_dict
    
    def get_db_source(self):
        """
        Parse the *DBSOURCE* field of the file.
        
        Returns
        ----------
        accession : str
            Content of the *DBSOURCE* field.
        """
        starts, stops = self._get_field_indices("DBSOURCE")
        return self.lines[starts[0]][12:].strip()
    
    def get_sequence(self):
        """
        Get the sequence from the *ORIGIN* field.
        
        Returns
        ----------
        sequence : ProteinSequence
            The reference sequence in the file.
        """
        seq_str = self._get_seq_string()
        if len(seq_str) == 0:
            raise InvalidFileError("The file does not contain "
                                   "sequence information")
        return ProteinSequence(seq_str)


class MultiFile(TextFile):
    """
    This class represents a file in GenBank or GenPept format,
    that contains multiple entries, for more than one UID.
    
    The information for each UID are appended to each other in such a
    file.
    Objects of this class can be iterated to obtain a `GenBankFile`
    or `GenPeptFile` for each entry in the file.

    Parameters
    ----------
    file_type : {'gb', 'gp'}
        Determines whether the objects should be used for GenBank or
        GenPept files.
    
    Examples
    --------
    
    >>> fetch_single_file(
    ...     ["1L2Y_A", "3O5R_A", "5UGO_A"], "multifile.gp", "protein", "gp")
    >>> multi_file = MultiFile(file_type="gp")
    >>> multi_file.read("multifile.gp")
    >>> for gp_file in multi_file:
    ...     print(gp_file.get_accession())
    1L2Y_A
    3O5R_A
    5UGO_A
    """

    def __init__(self, file_type):
        super().__init__()
        if file_type == "gb":
            self._file_class = GenBankFile
        elif file_type == "gp":
            self._file_class = GenPeptFile
        else:
            raise ValueError(f"'{file_type}' is an invalid file type")

    def __iter__(self):
        start_i = 0
        for i in range(len(self.lines)):
            line = self.lines[i]
            if line.strip() == "//":
                # Create file with lines corresponding to that file
                file_content = "\n".join(self.lines[start_i : i])
                file = self._file_class()
                file.read(io.StringIO(file_content))
                # Reset file start index
                start_i = i
                yield file