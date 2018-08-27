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

    Usually the user does not create instances of this class directly,
    but `Query` instances are combined with
    ``|`` (OR), ``&`` (AND) or ``^`` (NOT).
    
    Parameters
    ----------
    operator: str, {"AND", "OR", "NOT"}
        The combination operator.
    queries : iterable object of SimpleQuery
        The queries to be combined.
    
    Examples
    --------

    >>> query = SimpleQuery("Escherichia coli", "Organism") & \\
    ... SimpleQuery("90:100", "Sequence Length")
    >>> print(type(query).__name__)
    CompositeQuery
    >>> print(query)
    ("Escherichia coli"[Organism]) AND (90:100[Sequence Length])
    """
    
    def __init__(self, operator, query1, query2):
        super().__init__()
        self._op = operator
        self._q1 = query1
        self._q2 = query2
    
    def __str__(self):
        return "({:}) {:} ({:})".format(str(self._q1), self._op, self._q2)
        


class SimpleQuery(Query):
    """
    A simple query for the NCBI Entrez search service without
    combination via 'AND', 'OR' or 'NOT'. A query consists of a search
    term and an optional field.
    
    Parameters
    ----------
    term: str
        The search term.
    field : str, optional
        The field to search the term in.
        The list of possible fields and the required search term
        formatting can be found
        `here <https://www.ncbi.nlm.nih.gov/books/NBK49540/>`_.
        By default the field is omitted and all fields are searched in
        for the term, implicitly.
    
    Examples
    --------
    
    >>> query = SimpleQuery("Escherichia coli")
    >>> print(query)
    "Escherichia coli"
    >>> query = SimpleQuery("Escherichia coli", "Organism")
    >>> print(query)
    "Escherichia coli"[Organism]
    """

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
                raise ValueError(f"Unknown field identifier '{field}'")
        for invalid_string in \
            ['"', "AND", "OR", "NOT", "[", "]", "(", ")", "\t", "\n"]:
                if invalid_string in term:
                    raise ValueError(
                        f"Query contains illegal term {invalid_string}"
                    )
        if " " in term:
            # Encapsulate in quotes if spaces are in search term
            term = f'"{term}"'
        self._term = term
        self._field = field
    
    def __str__(self):
        string = self._term
        if self._field is not None:
            string += f"[{self._field}]"
        return string


def search(query, db_name, number=20):
    """
    Get all PDB IDs that meet the given query requirements,
    via the NCBI ESearch service.
    
    This function requires an internet connection.
    
    Parameters
    ----------
    query : Query
        The search query.
    
    Returns
    -------
    ids : list of str
        A list of strings containing all NCBI UIDs (accession number)
        that meet the query requirements.
    
    Examples
    --------
    >>> query = SimpleQuery("Escherichia coli", "Organism") & \\
    ...         SimpleQuery("90:100", "Sequence Length")
    >>> ids = search(query, "nuccore", number=5)
    >>> print(ids)
    ['1447562463', '1447562199', '1447561837', '1447561836', '1447561835']
    """ 
    r = requests.get(
        (_base_url + _search_url).format(db_name, str(query), str(number))
    )
    xml_response = r.text
    root = ElementTree.fromstring(xml_response)
    xpath = ".//IdList/Id"
    uids = [element.text for element in root.findall(xpath)]
    return uids
    