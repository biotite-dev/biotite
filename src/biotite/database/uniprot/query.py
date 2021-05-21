# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.uniprot"
__author__ = "Maximilian Greil"
__all__ = ["SimpleQuery", "CompositeQuery", "search"]

import requests
import abc

_base_url = "https://www.uniprot.org/uniprot/"

_search_url = ("?query={:}"
               "&format=list"
               "&limit={:}")


class Query(metaclass=abc.ABCMeta):
    """
    Base class for a wrapper around a search term
    for the UniProt search service.
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
    for the UniProt search service.

    A composite query is a combination of two other queries,
    combined either with an 'AND', 'OR' or 'NOT' operator.

    Usually the user does not create instances of this class directly,
    but :class:`Query` instances are combined with
    ``|`` (OR), ``&`` (AND) or ``^`` (NOT).

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
        return "{:}+{:}+{:}".format(str(self._q1), self._op, str(self._q2))


def _check_brackets(term):
    """
    Check if term contains correct number of round - and square brackets.

    Parameters
    ----------
    term: str
        The search term.

    Returns
    -------
    bool
        True if term contains correct number of round - and square brackets, otherwise False.
    """
    square_count = 0
    round_count = 0
    for i in term:
        if i == "[":
            square_count += 1
        elif i == "]":
            square_count -= 1
        if i == "(":
            round_count += 1
        elif i == ")":
            round_count -= 1
        if square_count < 0:
            return False
        if round_count < 0:
            return False
    return square_count == 0 and round_count == 0


class SimpleQuery(Query):
    """
    A simple query for the UniProt search service without
    combination via 'AND', 'OR' or 'NOT'. A query consists of a search
    term and an optional field.

    A list of available search fields with description can be found
    `here <https://www.uniprot.org/help/query-fields>`_.

    Parameters
    ----------
    field : str
        The field to search the term in.
       The list of possible fields and the required search term
       formatting can be found
       `here <https://www.uniprot.org/help/query-fields>`_.
    term: str
       The search term.
    """

    # Field identifiers are taken from
    # https://www.uniprot.org/help/query-fields
    _fields = [
        "accession", "active", "annotation", "author", "cdantigen", "chebi", "citation", "cluster", "count", "created",
        "database", "ec", "evidence", "existence", "family", "fragment", "gene", "gene_exact", "goa", "host", "id",
        "inchikey", "inn", "interactor", "keyword", "length", "lineage", "mass", "method", "mnemonic", "modified",
        "name", "organelle", "organism", "plasmid", "proteome", "proteomecomponent", "replaces", "reviewed", "scope",
        "sequence", "sequence_modified", "strain", "taxonomy", "tissue", "web"
    ]

    def __init__(self, field, term):
        super().__init__()
        if field not in SimpleQuery._fields:
            raise ValueError(f"Unknown field identifier '{field}'")
        if not _check_brackets(term):
            raise ValueError(
                f"Query term contains illegal number of round brackets ( ) and/or square brackets [ ]"
            )
        for invalid_string in \
                ['"', "AND", "OR", "NOT", "\t", "\n"]:
            if invalid_string in term:
                raise ValueError(
                    f"Query contains illegal term {invalid_string}"
                )
        if " " in term:
            term = f'"{term}"'
        self._field = field
        self._term = term

    def __str__(self):
        return f"{self._field}:{self._term}"


def search(query, number=10):
    """
    Get all UniProt IDs that meet the given query requirements,
    via the UniProt search service.

    This function requires an internet connection.

    Parameters
    ----------
    query : Query
        The search query.
    number : int
        The maximum number of IDs that are obtained.

    Returns
    -------
    ids : list of str
        A list of strings containing all UniProt IDs
        that meet the query requirements.

    Warnings
    --------
    Even if you give valid input to this function, in rare cases the
    database might return no or malformed data to you.
    In these cases the request should be retried.
    When the issue occurs repeatedly, the error is probably in your
    input.

    Notes
    -----
    A list of available search fields with description can be found
    `here <https://www.uniprot.org/help/query-fields>`_.
    """

    r = requests.get(
        (_base_url + _search_url).format(str(query), str(number))
    )
    content = r.text
    return content.split('\n')[:-1]
