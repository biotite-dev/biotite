# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.rcsb"
__author__ = "Patrick Kunzmann, Maximilian Dombrowsky"
__all__ = [
    "Query",
    "SingleQuery",
    "CompositeQuery",
    "BasicQuery",
    "FieldQuery",
    "SequenceQuery",
    "StructureQuery",
    "MotifQuery",
    "Sorting",
    "Grouping",
    "DepositGrouping",
    "IdentityGrouping",
    "UniprotGrouping",
    "search",
    "count",
]

import abc
import copy
import json
from datetime import datetime
import numpy as np
import requests
from biotite.database.error import RequestError
from biotite.sequence.seqtypes import NucleotideSequence

_search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
_scope_to_target = {
    "protein": "pdb_protein_sequence",
    "rna": "pdb_rna_sequence",
    "dna": "pdb_dna_sequence",
}


class Query(metaclass=abc.ABCMeta):
    """
    A representation of a JSON query for the RCSB search API.

    This is the abstract base class for all queries.
    """

    @abc.abstractmethod
    def get_content(self):
        """
        Get the query content, i.e. the data belonging to the
        ``'query'`` attribute in the RCSB search API.

        This content is converted into JSON by the :func:`search`
        and :func:`count` functions.

        Returns
        -------
        content : dict
            The content dictionary for the ``'query'`` attributes.
        """
        pass

    def __and__(self, query):
        return CompositeQuery([self, query], "and")

    def __or__(self, query):
        return CompositeQuery([self, query], "or")


class SingleQuery(Query, metaclass=abc.ABCMeta):
    """
    A terminal query node for the RCSB search API.

    Multiple :class:`SingleQuery` objects can be combined to
    :class:`CompositeQuery`objects using the ``|`` and ``&`` operators.

    This is the abstract base class for all queries that are
    terminal nodes.
    """

    @abc.abstractmethod
    def get_content(self):
        return {"parameters": {}}


class CompositeQuery(Query):
    """
    A group query node for the RCSB search API.

    A composite query is an combination of other queries, combined
    either with the `'and'` or `'or'` operator.
    Usually, a :class:`CompositeQuery` will not be created by calling
    its constructor, but by combining queries using the ``|`` or ``&``
    operator.

    Parameters
    ----------
    queries : iterable object of Query
        The queries to be combined.
    operator : {'or', 'and'}
        The type of combination.
    """

    def __init__(self, queries, operator):
        self._queries = queries
        if operator not in ("or", "and"):
            raise ValueError(f"Operator must be 'or' or 'and', not '{operator}'")
        self._operator = operator

    def get_content(self):
        """
        A dictionary representation of the query.
        This dictionary is the content of the ``'query'`` key in the
        JSON query.

        Returns
        -------
        content : dict
            The dictionary representation of the query.
        """
        content = {
            "type": "group",
            "logical_operator": self._operator,
            "nodes": [query.get_content() for query in self._queries],
        }
        return content


class BasicQuery(SingleQuery):
    """
    A text query for searching for a given term across all available
    fields.

    Parameters
    ----------
    term : str
        The search term.
        If the term contains multiple words, the query will return
        results where the entire term is present.
        The matching is not case-sensitive.
        Logic combinations of terms is described
        `here <https://search.rcsb.org/#basic-queries>`_.

    Examples
    --------

    >>> query = BasicQuery("Miniprotein Construct")
    >>> print(sorted(search(query)))
    ['1L2Y']
    """

    def __init__(self, term):
        super().__init__()
        self._term = term

    def get_content(self):
        content = super().get_content()
        content["type"] = "terminal"
        content["service"] = "full_text"
        content["parameters"]["value"] = f'"{self._term}"'
        return content


class FieldQuery(SingleQuery):
    """
    A text query for searching for values in a given field using the
    given operator.

    The operators are keyword arguments of this function and the search
    value is the value given to the respective parameter.
    The operators are mutually exclusive.
    If none is given, the search will return results where the given
    field exists.

    A :class:`FieldQuery` is negated using the ``~`` operator.

    Parameters
    ----------
    field : str
        The field to search in.
    molecular_definition : bool, optional
        If set true, this query searches in fields
        associated with
        `molecular definitions <https://search.rcsb.org/chemical-search-attributes.html>`_.
        If false (default), this query searches in fields
        associated with `PDB structures <https://search.rcsb.org/structure-search-attributes.html>`_.
    case_sensitive : bool, optional
        If set to true, searches are case sensitive.
        By default matching is case-insensitive.
    exact_match : str, optional
        Operator for returning results whose field exactly matches the
        given value.
    contains_words, contains_phrase : str, optional
        Operator for returning results whose field matches
        individual words from the given value or the value as exact
        phrase, respectively.
    greater, less, greater_or_equal, less_or_equal, equals : int or float or datetime, optional
        Operator for returning results whose field values are larger,
        smaller or equal to the given value.
    range, range_closed : tuple(int, int) or tuple(float, float) or tuple(datetime, datetime), optional
        Operator for returning results whose field matches values within
        the given range.
        `range_closed` includes the interval limits.
    is_in : tuple of str or list of str, optional
        Operator for returning results whose field matches any of the
        values in the given list.

    Notes
    -----
    A complete list of the available fields and its supported operators
    is documented at
    `<https://search.rcsb.org/structure-search-attributes.html>`_
    and
    `<https://search.rcsb.org/chemical-search-attributes.html>`_.

    Examples
    --------

    >>> query = FieldQuery("reflns.d_resolution_high", less_or_equal=0.6)
    >>> print(sorted(search(query)))
    ['1EJG', '1I0T', '3NIR', '3P4J', '4JLJ', '5D8V', '5NW3', '7ATG', '7R0H']
    """

    def __init__(
        self, field, molecular_definition=False, case_sensitive=False, **kwargs
    ):
        super().__init__()
        self._negation = False
        self._field = field
        self._mol_definition = molecular_definition
        self._case_sensitive = case_sensitive

        if len(kwargs) > 1:
            raise TypeError("Only one operator must be given")
        elif len(kwargs) == 1:
            self._operator = list(kwargs.keys())[0]
            self._value = list(kwargs.values())[0]
        else:
            # No operator is given
            self._operator = "exists"
            self._value = None

        if self._operator not in [
            "exact_match",
            "contains_words",
            "contains_phrase",
            "greater",
            "less",
            "greater_or_equal",
            "less_or_equal",
            "equals",
            "range",
            "range_closed",
            "is_in",
            "exists",
        ]:
            raise TypeError(
                f"Constructor got an unexpected keyword argument '{self._operator}'"
            )

        # Convert dates into ISO 8601
        if isinstance(self._value, datetime):
            self._value = _to_isoformat(self._value)
        elif isinstance(self._value, (tuple, list, np.ndarray)):
            self._value = [
                _to_isoformat(val) if isinstance(val, datetime) else val
                for val in self._value
            ]

        # Create dictionary for 'range' operator
        if self._operator == "range":
            self._value = {
                "from": self._value[0],
                "include_lower": False,
                "to": self._value[1],
                "include_upper": False,
            }
        elif self._operator == "range_closed":
            self._value = {
                "from": self._value[0],
                "include_lower": True,
                "to": self._value[1],
                "include_upper": True,
            }

        # Rename operators to names used in API
        if self._operator == "is_in":
            # 'in' is not an available parameter name in Python
            self._operator = "in"
        elif self._operator == "range_closed":
            # For backwards compatibility
            self._operator = "range"

    def get_content(self):
        content = super().get_content()
        content["type"] = "terminal"
        if self._mol_definition:
            content["service"] = "text_chem"
        else:
            content["service"] = "text"
        content["parameters"]["attribute"] = self._field
        content["parameters"]["operator"] = self._operator
        content["parameters"]["negation"] = self._negation
        content["parameters"]["case_sensitive"] = self._case_sensitive
        if self._value is not None:
            content["parameters"]["value"] = self._value
        return content

    def __invert__(self):
        clone = copy.deepcopy(self)
        clone._negation = not clone._negation
        return clone


class SequenceQuery(SingleQuery):
    """
    A query for protein/DNA/RNA molecules with a sequence similar to a
    given input sequence using
    `MMseqs2 <https://github.com/soedinglab/mmseqs2>`_.

    Parameters
    ----------
    sequence : Sequence or str
        The input sequence.
        If `sequence` is a :class:`NucleotideSequence` and the `scope`
        is ``'rna'``, ``'T'`` is automatically replaced by ``'U'``.
    scope : {'protein', 'dna', 'rna'}
        The type of molecule to find.
    min_identity : float, optional
        A match is only returned, if the sequence identity between
        the match and the input sequence exceeds this value.
        Must be between 0 and 1.
        By default, the sequence identity is ignored.
    max_expect_value : float, optional
        A match is only returned, if the *expect value* (E-value) does
        not exceed this value.
        By default, the value is effectively ignored.

    Notes
    -----
    *MMseqs2* is run on the RCSB servers.

    Examples
    --------

    >>> sequence = "NLYIQWLKDGGPSSGRPPPS"
    >>> query = SequenceQuery(sequence, scope="protein", min_identity=0.95)
    >>> print(sorted(search(query)))
    ['1L2Y', '2LDJ', '9G22', '9G2N', '9G2O', '9G31', '9G32', '9GDL', '9GDN', '9GDT', '9GDU', '9GE1']
    """

    def __init__(self, sequence, scope, min_identity=0.0, max_expect_value=10000000.0):
        super().__init__()
        self._target = _scope_to_target.get(scope.lower())
        if self._target is None:
            raise ValueError(f"'{scope}' is an invalid scope")

        if isinstance(sequence, NucleotideSequence) and scope.lower() == "rna":
            self._sequence = str(sequence).replace("T", "U")
        else:
            self._sequence = str(sequence)

        self._min_identity = min_identity
        self._max_expect_value = max_expect_value

    def get_content(self):
        content = super().get_content()
        content["type"] = "terminal"
        content["service"] = "sequence"
        content["parameters"]["value"] = self._sequence
        content["parameters"]["target"] = self._target
        content["parameters"]["identity_cutoff"] = self._min_identity
        content["parameters"]["evalue_cutoff"] = self._max_expect_value
        return content


class MotifQuery(SingleQuery):
    """
    A query for protein/DNA/RNA molecules containing the given sequence
    motif.

    Parameters
    ----------
    pattern : str
        The sequence pattern.
    pattern_type : {'simple', 'prosite', 'regex'}
        The type of the pattern.
    scope : {'protein', 'dna', 'rna'}
        The type of molecule to find.

    Examples
    --------

    >>> query = MotifQuery(
    ...     "C-x(2,4)-C-x(3)-[LIVMFYWC]-x(8)-H-x(3,5)-H.",
    ...     "prosite",
    ...     "protein"
    ... )
    """

    def __init__(self, pattern, pattern_type, scope):
        super().__init__()
        self._pattern = pattern
        self._pattern_type = pattern_type
        self._target = _scope_to_target.get(scope.lower())

    def get_content(self):
        content = super().get_content()
        content["type"] = "terminal"
        content["service"] = "seqmotif"
        content["parameters"]["value"] = self._pattern
        content["parameters"]["pattern_type"] = self._pattern_type
        content["parameters"]["target"] = self._target
        return content


class StructureQuery(SingleQuery):
    """
    A query for protein/DNA/RNA molecules with structural similarity
    to the query structure.

    Either the chain or assembly ID of the query structure must be
    specified.

    Parameters
    ----------
    pdb_id : str
        The PDB ID of the query structure.
    chain : str, optional
        The chain ID (more exactly ``asym_id``) of the query structure.
    assembly : str, optional
        The assembly ID (``assembly_id``) of the query structure.
    strict : bool, optional
        If true, structure comparison is strict, otherwise it is
        relaxed.

    Examples
    --------

    >>> query = StructureQuery("1L2Y", chain="A")
    >>> print(sorted(search(query)))
    ['1L2Y', '1RIJ', '2JOF', '2LDJ', '2M7D', '7MQS', '9DPF']
    """

    def __init__(self, pdb_id, chain=None, assembly=None, strict=True):
        super().__init__()

        if (chain is None and assembly is None) or (
            chain is not None and assembly is not None
        ):
            raise TypeError("Either the chain ID or assembly ID must be set")
        elif chain is None:
            self._value = {"entry_id": pdb_id, "asssembly_id": assembly}
        else:
            self._value = {"entry_id": pdb_id, "asym_id": chain}

        self._operator = "strict_shape_match" if strict else "relaxed_shape_match"

    def get_content(self):
        content = super().get_content()
        content["type"] = "terminal"
        content["service"] = "structure"
        content["parameters"]["value"] = self._value
        content["parameters"]["operator"] = self._operator
        return content


class Sorting:
    def __init__(self, field, descending=True):
        self._field = field
        self._descending = descending

    @property
    def field(self):
        return self._field

    @property
    def descending(self):
        return self._descending

    def get_content(self):
        """
        Get the sorting content, i.e. the data belonging to the
        ``'sort'`` and ``'ranking_criteria_type'`` attributes in the
        RCSB search API.

        This content is converted into JSON by the :func:`search`
        function.

        Returns
        -------
        content : dict
            The content dictionary for the ``'sort'`` and
            ``'ranking_criteria_type'`` attributes.
        """
        direction = "desc" if self._descending else "asc"
        return {"sort_by": self._field, "direction": direction}


class Grouping(metaclass=abc.ABCMeta):
    """
    A representation of the JSON grouping options of the RCSB search
    API.

    Parameters
    ----------
    sort_by : str or Sorting, optional
        If specified, the returned PDB IDs within each group are sorted
        by the values of the given field name.
        A complete list of the available fields is documented at
        `<https://search.rcsb.org/structure-search-attributes.html>`_.
        and
        `<https://search.rcsb.org/chemical-search-attributes.html>`_.
        If a string is given, sorting is performed in descending order.
        To choose the order a :class:`Sorting` object needs to be
        provided.
    """

    def __init__(self, sort_by=None):
        if sort_by is None:
            self._sorting = None
        elif isinstance(sort_by, Sorting):
            self._sorting = sort_by
        else:
            self._sorting = Sorting(sort_by)

    @abc.abstractmethod
    def get_content(self):
        """
        Get the grouping content, i.e. the data belonging to the
        ``'group_by'`` attribute in the RCSB search API.

        This content is converted into JSON by the :func:`search`
        and :func:`count` functions.

        ABSTRACT: Override when inheriting.

        Returns
        -------
        content : dict
            The content dictionary for the ``'group_by'`` attributes.
        """
        if self._sorting is not None:
            return {"ranking_criteria_type": self._sorting.get_content()}
        else:
            return {}

    @abc.abstractmethod
    def is_compatible_return_type(self, return_type):
        """
        Check whether this :class:`Group` is compatible with the
        RCSB search API ``return_type``.

        ABSTRACT: Override when inheriting.

        Parameters
        ----------
        return_type : str
            The ``return_type`` attribute to be checked.

        Returns
        -------
        is_compatible : bool
            True, if this :class:`Group` is compatible with the
            `return_type`, false otherwise.
        """
        pass


class DepositGrouping(Grouping):
    """
    This class groups PDB entries if they were deposited as a
    collection.
    Such a group usually contain the same protein with e.g. a different
    bound molecule.

    This :class:`Grouping` is only applicable, if the
    :func:`count()`/:func:`search()` return type is set to ``entry``.

    Parameters
    ----------
    sort_by : str or Sorting, optional
        If specified, the returned PDB IDs within each group are sorted
        by the values of the given field name.
        A complete list of the available fields is documented at
        `<https://search.rcsb.org/structure-search-attributes.html>`_.
        and
        `<https://search.rcsb.org/chemical-search-attributes.html>`_.
        If a string is given, sorting is performed in descending order.
        To choose the order a :class:`Sorting` object needs to be
        provided.
    """

    def get_content(self):
        content = super().get_content()
        content["aggregation_method"] = "matching_deposit_group_id"
        return content

    def is_compatible_return_type(self, return_type):
        return return_type == "entry"


class IdentityGrouping(Grouping):
    """
    This class groups protein chains with a given sequence identity
    with each other.

    This :class:`Grouping` is only applicable, if the
    :func:`count()`/:func:`search()` return type is set to
    ``polymer_entity``.

    Parameters
    ----------
    similarity_cutoff : {100, 95, 90, 70, 50, 30}
        The sequence identity in percent at which the structures are
        grouped.
        In other words, a returned group contains sequences that have
        `similarity_cutoff` sequence identity with each other.
        Since the PDB uses precalculated clusters, only certain values
        are available.
    sort_by : str or Sorting, optional
        If specified, the returned PDB IDs within each group are sorted
        by the values of the given field name.
        A complete list of the available fields is documented at
        `<https://search.rcsb.org/structure-search-attributes.html>`_.
        and
        `<https://search.rcsb.org/chemical-search-attributes.html>`_.
        If a string is given, sorting is performed in descending order.
        To choose the order a :class:`Sorting` object needs to be
        provided.
    """

    def __init__(self, similarity_cutoff, sort_by=None):
        super().__init__(sort_by)
        if similarity_cutoff not in (100, 95, 90, 70, 50, 30):
            raise ValueError(
                f"A similarity cutoff of {similarity_cutoff}% is not supported"
            )
        self._similarity_cutoff = similarity_cutoff

    def get_content(self):
        content = super().get_content()
        content["aggregation_method"] = "sequence_identity"
        content["similarity_cutoff"] = self._similarity_cutoff
        return content

    def is_compatible_return_type(self, return_type):
        return return_type == "polymer_entity"


class UniprotGrouping(Grouping):
    """
    This class groups protein chains that point to the same *Uniprot*
    accession ID.

    This :class:`Grouping` is only applicable, if the
    :func:`count()`/:func:`search()` return type is set to
    ``polymer_entity``.

    Parameters
    ----------
    sort_by : str or Sorting, optional
        If specified, the returned PDB IDs within each group are sorted
        by the values of the given field name.
        A complete list of the available fields is documented at
        `<https://search.rcsb.org/structure-search-attributes.html>`_.
        and
        `<https://search.rcsb.org/chemical-search-attributes.html>`_.
        If a string is given, sorting is performed in descending order.
        To choose the order a :class:`Sorting` object needs to be
        provided.
    """

    def get_content(self):
        content = super().get_content()
        content["aggregation_method"] = "matching_uniprot_accession"
        return content

    def is_compatible_return_type(self, return_type):
        return return_type == "polymer_entity"


def count(query, return_type="entry", group_by=None, content_types=("experimental",)):
    """
    Count PDB entries that meet the given query requirements,
    via the RCSB search API.

    This function requires an internet connection.

    Parameters
    ----------
    query : Query
        The search query.
    return_type : {'entry', 'assembly', 'polymer_entity', 'non_polymer_entity', 'polymer_instance'}, optional
        The type of the counted identifiers:

        - ``'entry'``: All macthing PDB entries are counted.
        - ``'assembly'``: All matching assemblies are counted.
        - ``'polymer_entity'``: All matching polymeric entities are
          counted.
        - ``'non_polymer_entity'``: All matching non-polymeric entities
          are counted.
        - ``'polymer_instance'``: All matching chains are counted.
    group_by : Grouping
        If this parameter is set, the number of groups is returned
        instead.
    content_types : iterable of {"experimental", "computational"}, optional
        Specify whether experimental and computational structures should
        be included.
        At least one of them needs to be specified.
        By default only experimental structures are included.
        Note, that identifiers for computational structures cannot be
        downloaded via :func:`biotite.database.rcsb.fetch()` as they
        point to *AlphaFold DB* and *ModelArchive*.

    Returns
    -------
    count : int
        The total number of PDB IDs (or groups) that would be returned
        by calling :func:`search()` using the same parameters.

    Notes
    -----
    If `group_by` is set, the number of results may be lower than in an
    ungrouped query, as grouping is not applicable to all structures.
    For example a DNA structure has no associated *Uniprot* accession
    and hence is omitted by :class:`UniprotGrouping`.

    Examples
    --------

    >>> query = FieldQuery("reflns.d_resolution_high", less_or_equal=0.6)
    >>> print(count(query))
    9
    >>> ids = search(query)
    >>> print(sorted(ids))
    ['1EJG', '1I0T', '3NIR', '3P4J', '4JLJ', '5D8V', '5NW3', '7ATG', '7R0H']
    """
    query_dict = _initialize_query_dict(query, return_type, group_by, content_types)

    query_dict["request_options"]["return_counts"] = True

    r = requests.get(_search_url, params={"json": json.dumps(query_dict)})

    if r.status_code == 200:
        if group_by is None:
            return r.json()["total_count"]
        else:
            return r.json()["group_by_count"]
    elif r.status_code == 204:
        # Search did not return any results
        return 0
    else:
        try:
            raise RequestError(f"Error {r.status_code}: {r.json()['message']}")
        except json.decoder.JSONDecodeError:
            # In case there an error response without message
            raise RequestError(f"Error {r.status_code}")


def search(
    query,
    return_type="entry",
    range=None,
    sort_by=None,
    group_by=None,
    return_groups=False,
    content_types=("experimental",),
):
    """
    Get all PDB IDs that meet the given query requirements,
    via the RCSB search API.

    This function requires an internet connection.

    Parameters
    ----------
    query : Query
        The search query.
    return_type : {'entry', 'assembly', 'polymer_entity', 'non_polymer_entity', 'polymer_instance'}, optional
        The type of the returned identifiers:

        - ``'entry'``: Only the PDB ID is returned (e.g. ``'XXXX'``).
          These can be used directly a input to :func:`fetch()`.
        - ``'assembly'``: The PDB ID appended with assembly ID is
          returned (e.g. ``'XXXX-1'``).
        - ``'polymer_entity'``: The PDB ID appended with entity ID of
          polymers is returned (e.g. ``'XXXX_1'``).
        - ``'non_polymer_entity'``: The PDB ID appended with entity ID
          of non-polymeric entities is returned (e.g. ``'XXXX_1'``).
        - ``'polymer_instance'``: The PDB ID appended with chain ID
          (more exactly ``'asym_id'``) is returned (e.g. ``'XXXX.A'``).

    range : tuple(int, int), optional
        If this parameter is specified, only PDB IDs in this range
        are selected from all matching PDB IDs and returned
        (pagination).
        The range is zero-indexed and the stop value is exclusive.
    sort_by : str or Sorting, optional
        If specified, the returned PDB IDs are sorted by the values
        of the given field name.
        A complete list of the available fields is documented at
        `<https://search.rcsb.org/structure-search-attributes.html>`_.
        and
        `<https://search.rcsb.org/chemical-search-attributes.html>`_.
        If a string is given sorting is performed in descending order.
        To choose the order, a :class:`Sorting` object needs to be
        provided.
    group_by : Grouping
        If this parameter is set, the PDB IDs that meet the query
        requirements, are grouped according to the given criterion.
    return_groups : boolean, optional
        Only has effect, if `group_by` is set.
        By default the representative with the highest rank in each
        group is returned.
        The rank is determined by the `sort_by` parameter of
        :class:`Grouping` provided in `group_by`.
        If set to true, groups containing all structures belonging to
        the group are returned instead.
    content_types : iterable of {"experimental", "computational"}, optional
        Specify whether experimental and computational structures should
        be included.
        At least one of them needs to be specified.
        By default only experimental structures are included.
        Note, that identifiers for computational structures cannot be
        downloaded via :func:`biotite.database.rcsb.fetch()` as they
        point to *AlphaFold DB* and *ModelArchive*.

    Returns
    -------
    ids : list of str or dict (str -> list of str)
        If `return_groups` is false (default case), a list of strings
        containing all PDB IDs that meet the query requirements is
        returned.
        If `return_groups` is set to true a dictionary of groups is
        returned.
        This dictionary maps group identifiers to a list of all PDB IDs
        belonging to this group.

    Notes
    -----
    If `group_by` is set, the number of results may be lower than in an
    ungrouped query, as grouping is not applicable to all structures.
    For example a DNA structure has no associated *Uniprot* accession
    and hence is omitted by :class:`UniprotGrouping`.

    Also note that `sort_by` does not affect the order within a group.
    This order is determined by the `sort_by` parameter of the
    :class:`Grouping`.

    Examples
    --------

    >>> query = FieldQuery("reflns.d_resolution_high", less_or_equal=0.6)
    >>> print(sorted(search(query)))
    ['1EJG', '1I0T', '3NIR', '3P4J', '4JLJ', '5D8V', '5NW3', '7ATG', '7R0H']
    >>> print(search(query, sort_by="rcsb_accession_info.initial_release_date"))
    ['7R0H', '7ATG', '5NW3', '5D8V', '4JLJ', '3P4J', '3NIR', '1I0T', '1EJG']
    >>> print(search(
    ...     query, range=(1,4), sort_by="rcsb_accession_info.initial_release_date"
    ... ))
    ['7ATG', '5NW3', '5D8V']
    >>> print(sorted(search(query, return_type="polymer_instance")))
    ['1EJG.A', '1I0T.A', '1I0T.B', '3NIR.A', '3P4J.A', '3P4J.B', '4JLJ.A', '4JLJ.B', '5D8V.A', '5NW3.A', '7ATG.A', '7ATG.B', '7R0H.A']
    >>> print(search(
    ...     query, return_type="polymer_entity", return_groups=True,
    ...     group_by=UniprotGrouping(sort_by="rcsb_accession_info.initial_release_date"),
    ... ))
    {'P24297': ['5NW3_1'], 'P27707': ['4JLJ_1'], 'P80176': ['5D8V_1'], 'O29777': ['7R0H_1'], 'P01542': ['3NIR_1', '1EJG_1']}
    """
    query_dict = _initialize_query_dict(query, return_type, group_by, content_types)

    if group_by is not None:
        if return_groups:
            query_dict["request_options"]["group_by_return_type"] = "groups"
        else:
            query_dict["request_options"]["group_by_return_type"] = "representatives"

    if sort_by is not None:
        if isinstance(sort_by, Sorting):
            sorting = sort_by
        else:
            sorting = Sorting(sort_by)
        query_dict["request_options"]["sort"] = [sorting.get_content()]

    if range is None:
        query_dict["request_options"]["return_all_hits"] = True
    elif range[1] <= range[0]:
        raise ValueError("Range stop must be greater than range start")
    else:
        query_dict["request_options"]["paginate"] = {
            "start": int(range[0]),
            "rows": int(range[1]) - int(range[0]),
        }

    r = requests.get(_search_url, params={"json": json.dumps(query_dict)})

    if r.status_code == 200:
        if group_by is None or not return_groups:
            return [result["identifier"] for result in r.json()["result_set"]]
        else:
            return {
                group["identifier"]: [
                    result["identifier"] for result in group["result_set"]
                ]
                for group in r.json()["group_set"]
            }
    elif r.status_code == 204:
        # Search did not return any results
        return []
    else:
        try:
            raise RequestError(f"Error {r.status_code}: {r.json()['message']}")
        except json.decoder.JSONDecodeError:
            # In case there an error response without message
            raise RequestError(f"Error {r.status_code}")


def _initialize_query_dict(query, return_type, group_by, content_types):
    """
    Initialize the request parameter dictionary with attributes that
    `count()` and `search()` have in common.
    """
    if return_type not in [
        "entry",
        "polymer_instance",
        "assembly",
        "polymer_entity",
        "non_polymer_entity",
    ]:
        raise ValueError(f"'{return_type}' is an invalid return type")

    request_options = {}

    if len(content_types) == 0:
        raise ValueError("At least one content type must be specified")
    for content_type in content_types:
        if content_type not in ("experimental", "computational"):
            raise ValueError(f"Unknown content type '{content_type}'")
    request_options["results_content_type"] = content_types

    if group_by is not None:
        if not group_by.is_compatible_return_type(return_type):
            raise ValueError(
                f"Return type '{return_type}' is not compatible with the given Grouping"
            )
        request_options["group_by"] = group_by.get_content()

    query_dict = {
        "query": query.get_content(),
        "return_type": return_type,
        "request_options": request_options,
    }
    return query_dict


def _to_isoformat(object):
    """
    Convert a datetime into the specifc ISO 8601 format required by the RCSB.
    """
    return object.strftime("%Y-%m-%dT%H:%M:%SZ")
