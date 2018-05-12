# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["Query", "SimpleQuery", "CompositeQuery", "search"]

import requests
import abc
from xml.etree import ElementTree


_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

_search_url = ("esearch.fcgi?db={:}"
              "&term={:}"
              "&retmax={:}")

class Query(metaclass=abc.ABCMeta):
    """
    Base class for a wrapper around a search term
    for the NCBI Entrez search service.
    """
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def __str__(self):
        pass
    
    def __or__(self, operand):
        if not isinstance(operand, Query):
            operand = SimpleQuery(operand)
        return CompositeQuery("OR", self, operand)
    
    def __and__(self, operand):
        if not isinstance(operand, Query):
            operand = SimpleQuery(operand)
        return CompositeQuery("AND", self, operand)
    
    def __xor__(self, operand):
        if not isinstance(operand, Query):
            operand = SimpleQuery(operand)
        return CompositeQuery("NOT", self, operand)


class CompositeQuery(Query):
    """
    A representation of an composite query
    for the NCBI Entrez search service.
    
    A composite query is a combination of two other queries,
    combined either with an 'AND', 'OR' or 'NOT' operator.

    Usually the user does not create insatnces of this class directly,
    but he combines `Query` instances with ``|``, ``&`` or ``^``.
    
    Parameters
    ----------
    operator: str, {"AND", "OR", "NOT"}
        The combination operator.
    queries : iterable object of SimpleQuery
        The queries to be combined.
    """
    
    def __init__(self, operator, query1, query2):
        super().__init__()
        self._op = operator
        self._q1 = query1
        self._q2 = query2
    
    def __str__(self):
        return "({:}) {:} ({:})".format(str(self._q1), self._op, self._q2)
        


class SimpleQuery(Query):
    
    # Field identifiers are taken from
    # https://www.ncbi.nlm.nih.gov/books/NBK49540/
    _fields = [
        "Accession", "All Fields", "Author", "EC/RN Number", "Feature Key",
        "Filter", "Gene Name", "Genome Project", "Issue", "Journal", "Keyword",
        "Modification Date", "Molecular Weight", "Organism", "Page Number",
        "Primary Accession", "Properties", "Protein Name", "Publication Date",
        "SeqID String", "Sequence Length", "Substance Name", "Text Word",
        "Title", "Volume"
    ]

    def __init__(self, term, field=None):
        super().__init__()
        if field is not None:
            if field not in SimpleQuery._fields:
                raise ValueError("'{:}' is an unknown field identifier"
                                 .format(field))
        if any([(invalid_string in term) for invalid_string
                in ['"', "AND", "OR", "NOT", "[", "]", "(", ")"]]):
            raise ValueError("Search term contains invalid content")
        if " " in term:
            # Encapsulate in quotes if spaces are in search term
            term = '"' + term + '"'
        self._term = term
        self._field = field
    
    def __str__(self):
        string = self._term
        if self._field is not None:
            string += "[" + self._field + "]"
        return string


def search(query, db_name, number=20):
    """
    Get all PDB IDs that meet the given query requirements,
    via the RCSB SEARCH service.
    
    This function requires an internet connection.
    
    Parameters
    ----------
    query : Query
        Index of the atom array.
    
    Returns
    -------
    ids : list of str
        A list of strings containing all PDB IDs that meet the query
        requirements.
    
    Examples
    --------
    
    >>> query = ResolutionQuery(0.0, 0.6)
    >>> ids = search(query)
    >>> print(ids)
    ['1EJG', '1I0T', '3NIR', '3P4J', '5D8V', '5NW3']
    """ 
    r = requests.get(
        (_base_url + _search_url).format(db_name, str(query), str(number))
    )
    xml_response = r.text
    root = ElementTree.fromstring(xml_response)
    xpath = ".//IdList/Id"
    uids = [element.text for element in root.findall(xpath)]
    return uids
    