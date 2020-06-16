# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.rcsb"
__author__ = "Patrick Kunzmann, Maximilian Dombrowsky"
__all__ = ["Query", "SingleQuery", "CompositeQuery",
           "BasicQuery", "FieldQuery",
           "SequenceQuery", "StructureQuery", "MotifQuery",
           "search", "count"]

import abc
import json
import copy
from datetime import datetime
import numpy as np
import requests
from ...sequence.seqtypes import ProteinSequence, NucleotideSequence
from ..error import RequestError


_search_url = "http://search.rcsb.org/rcsbsearch/v1/query"
_scope_to_target = {
    "protein": "pdb_protein_sequence",
    "rna":     "pdb_rna_sequence",
    "dna":     "pdb_dna_sequence"
}

_node_id = 0


class Query(metaclass=abc.ABCMeta):
    """
    A representation for a JSON query for the RCSB search API.
    
    This class is the abstract base class for all queries.
    """
    @abc.abstractmethod
    def get_content(self):
        pass


class SingleQuery(Query, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_content(self):
        global _node_id
        content = {
            "node_id": _node_id,
            "parameters": {}
        }
        _node_id += 1
        return content
    
    def __and__(self, query):
        return CompositeQuery([self, query], "and")
    
    def __or__(self, query):
        return CompositeQuery([self, query], "or") 


class CompositeQuery(Query):
    """
    A representation of an composite query for the RCSB search API.
    
    A composite query is an accumulation of other queries, combined
    either with an 'and' or 'or' operator.
    """
    def __init__(self, queries, operator):
        self._queries = queries
        if operator not in ("or", "and"):
            raise ValueError(
                f"Operator must be 'or' or 'and', not '{operator}'"
            )
        self._operator = operator
    
    def get_content(self):
        content = {
            "type": "group",
            "logical_operator": self._operator,
            "nodes": [query.get_content() for query in self._queries]
        }
        return content



class BasicQuery(SingleQuery):
    def __init__(self, term):
        super().__init__()
        self._term = term

    def get_content(self):
        content = super().get_content()
        content["type"] = "terminal"
        content["service"] = "text"
        content["parameters"]["value"] = self._term
        return content


class FieldQuery(SingleQuery):
    """
    https://search.rcsb.org/search-attributes.html
    """
    def __init__(self, field, **kwargs):
        super().__init__()
        self._negation = False
        self._field = field
        
        if len(kwargs) > 1:
            raise TypeError("Only one operator must be given")
        elif len(kwargs) == 1:
            self._operator = list(kwargs.keys())[0]
            self._value = list(kwargs.values())[0]
            if self._operator == "is_in":
                self._operator = "in"
        else:
            # No operator is given
            self._operator = "exists"
            self._value = None
        
        if self._operator not in [
            "exact_match",
            "contains_words", "contains_phrase",
            "greater", "less", "greater_or_equal", "less_or_equal", "equals",
            "range", "range_closed",
            "in",
            "exists"
        ]:
            raise TypeError(
                f"Constructor got an unexpected keyword argument "
                f"'{self._operator}'"
            )
        
        # Convert dates into ISO 8601
        if isinstance(self._value, datetime):
             self._value = _to_isoformat(self._value)
        elif isinstance(self._value, (tuple, list, np.ndarray)):
            self._value = [
                _to_isoformat(val) if isinstance(val, datetime) else val
                for val in self._value
            ]

    def negate(self):
        clone = copy.deepcopy(self)
        clone._negation = True
        return clone
    
    def is_negated(self):
        return self._negation

    def get_content(self):
        content = super().get_content()
        content["type"] = "terminal"
        content["service"] = "text"
        content["parameters"]["attribute"] = self._field
        content["parameters"]["operator"] = self._operator
        content["parameters"]["negation"] = self._negation
        if self._value is not None:
            content["parameters"]["value"] = self._value
        return content

    def __invert__(self):
        return self.negate()


class SequenceQuery(SingleQuery):
    def __init__(self, sequence, scope,
                 min_identity=0.0, max_expect_value=10000000.0):
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
    def __init__(self, pdb_id, chain=None, assembly=None, strict=True):
        super().__init__()

        if (chain is None and assembly is None) \
           or (chain is not None and assembly is not None):
                raise TypeError(
                    "Either the chain ID or assembly ID must be set"
                )
        elif chain is None:
            self._value = {
                "entry_id": pdb_id,
                "asssembly_id": assembly
            }
        else:
            self._value = {
                "entry_id": pdb_id,
                "asym_id": chain
            }
        
        self._operator = "strict_shape_match" if strict \
                         else "relaxed_shape_match"

    def get_content(self):
        content = super().get_content()
        content["type"] = "terminal"
        content["service"] = "structure"
        content["parameters"]["value"] = self._value
        content["parameters"]["operator"] = self._operator
        return content


def count(query, return_type="entry"):
    if return_type not in [
        "entry", "polymer_instance", "assembly",
        "polymer_entity", "non_polymer_entity",
    ]:
        raise ValueError(f"'{return_type}' is an invalid return type")
    
    query_dict = {
        "query": query.get_content(),
        "return_type": return_type,
        "request_options": {
            # Do not return any IDs,
            # as we are only interested in the 'total_count' attribute
            "pager": {
                "start": 0,
                "rows": 0
            }
        }
    }
    r = requests.get(_search_url, params={"json": json.dumps(query_dict)})
    
    if r.status_code == 200:
        return r.json()["total_count"]
    elif r.status_code == 204:
        # Search did not return any results
        return 0
    else:
        try:
            raise RequestError(f"Error {r.status_code}: {r.json()['message']}")
        except json.decoder.JSONDecodeError:
            # In case there an error response without message
            raise RequestError(f"Error {r.status_code}")


def search(query, return_type="entry", range=None, sort_by=None):
    """
    Get all PDB IDs that meet the given query requirements,
    via the RCSB search API service.
    
    This function requires an internet connection.
    
    Parameters
    ----------
    query : Query
        The search query.
    
    Returns
    -------
    ids : list of str
        A list of strings containing all PDB IDs that meet the query
        requirements

    
    Examples
    --------
    
    >>> query = ResolutionQuery(max=0.6)
    >>> ids = search(query)
    >>> print(ids)
    ['1EJG', '1I0T', '3NIR', '3P4J', '5D8V', '5NW3']
    """
    if return_type not in [
        "entry", "polymer_instance", "assembly",
        "polymer_entity", "non_polymer_entity",
    ]:
        raise ValueError(f"'{return_type}' is an invalid return type")
    
    if sort_by is None:
        sort_by = "score"

    if range is None:
        start = 0
        rows = count(query)
    elif range[1] <= range[0]:
        raise ValueError("Range stop must be greater than range start")
    else:
        start = range[0]
        rows = range[1] - start

    query_dict = {
        "query": query.get_content(),
        "return_type": return_type,
        "request_options": {
            "pager": {
                "start": start,
                "rows": rows
            },
            "sort": [
                {
                    "sort_by": sort_by,
                }
            ]
        }
    }
    r = requests.get(_search_url, params={"json": json.dumps(query_dict)})
    
    if r.status_code == 200:
        return [result["identifier"] for result in r.json()["result_set"]]
    elif r.status_code == 204:
        # Search did not return any results
        return []
    else:
        try:
            raise RequestError(f"Error {r.status_code}: {r.json()['message']}")
        except json.decoder.JSONDecodeError:
            # In case there an error response without message
            raise RequestError(f"Error {r.status_code}")


def _to_isoformat(object):
    """
    Convert a datetime into the specifc ISO 8601 format required by the RCSB.
    """
    return object.strftime("%Y-%m-%dT%H:%M:%SZ")