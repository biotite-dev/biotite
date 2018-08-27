# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["Query", "CompositeQuery", "SimpleQuery", "MethodQuery",
           "ResolutionQuery", "BFactorQuery", "MolecularWeightQuery",
           "search"]

import requests
import abc
from xml.etree.ElementTree import Element, SubElement, tostring


_search_url = "https://www.rcsb.org/pdb/rest/search"

class Query(metaclass=abc.ABCMeta):
    """
    A representation for an XML query for the RCSB SEARCH service.
    
    This class is the abstract base class for all queries.
    """
    
    def __init__(self):
        self.query = None
    
    def get_query(self):
        """
        Get the root XML `Element` representing the query.
        
        Returns
        -------
        query : Element
            The root element of the query.
        """
        return self.query
    
    def __str__(self):
        return tostring(self.query, encoding="unicode")


class CompositeQuery(Query):
    """
    A representation of an composite XML query.
    
    A composite query is an accumulation of other queries, combined
    either with an 'and' or 'or' operator.
    
    A combination of `CompositeQuery` instances is not possible.
    
    Parameters
    ----------
    operator: str, 'or' or 'and'
        The combination operator.
    queries : iterable object of SimpleQuery
        The queries to be combined.
    """
    
    def __init__(self, operator, queries):
        super().__init__()
        self.query = Element("orgPdbCompositeQuery")
        for i, q in enumerate(queries):
            refinement = SubElement(self.query, "queryRefinement")
            ref_level = SubElement(refinement, "queryRefinementLevel")
            ref_level.text = str(i)
            if i != 0:
                conj_type = SubElement(refinement, "conjunctionType")
                conj_type.text = operator
            refinement.append(q.query)


class SimpleQuery(Query, metaclass=abc.ABCMeta):
    """
    The abstract base class for all non-composite queries.
    
    Offers the convenient `add_param()` method for simple creation
    of custom queries.
    
    Parameters
    ----------
    query_type: str
        The name of the query type. This is the suffix for the
        'QueryType' XML tag.
    parameter_class : optional
        If specifed, this string is the prefix for all parameters
        (XML tags) of the query.
    """
    
    def __init__(self, query_type, parameter_class=""):
        super().__init__()
        self.query = Element("orgPdbQuery")
        self._param_cls = parameter_class
        type = SubElement(self.query, "queryType")
        type.text = "org.pdb.query.simple." + query_type
    
    def add_param(self, param, content):
        """
        Add a parameter (XML tag/text pair) to the query.
        
        Parameters
        ----------
        param: str
            The XML tag name for the parameter.
        content : str
            The text content for the parameter.
        """
        if self._param_cls == "":
            child = SubElement(self.query, param)
        else:
            child = SubElement(self.query, self._param_cls + "." + param)
        child.text = content


class MethodQuery(SimpleQuery):
    """
    Query that filters structures, that were elucidated with a certain
    method.
    
    Parameters
    ----------
    method: str
        Structures of the given method are filtered. Possible values
        are:
        'X-RAY', 'SOLUTION_NMR', 'SOLID-STATE NMR',
        'ELECTRON MICROSCOPY', 'ELECTRON CRYSTALLOGRAPHY',
        'FIBER DIFFRACTION', 'NEUTRON DIFFRACTION',
        'SOLUTION SCATTERING', 'HYBRID' and 'OTHER'.
    has_data: bool, optional
        If specified, the query additionally filters structures, that
        store or do not store experimental data, respectively.
    """
    
    def __init__(self, method, has_data=None):
        super().__init__("ExpTypeQuery", "mvStructure")
        self.add_param("expMethod.value", method.upper())
        if has_data == True:
            self.add_param("hasExperimentalData.value", "Y")
        elif has_data == False:
            self.add_param("hasExperimentalData.value", "N")

class ResolutionQuery(SimpleQuery):
    """
    Query that filters X-ray elucidated structures within a defined
    resolution range.
    
    Parameters
    ----------
    min: float
        The minimum resolution value.
    max: float
        The maximum resolution value.
    """
    
    def __init__(self, min, max):
        super().__init__("ResolutionQuery", "refine.ls_d_res_high")
        self.add_param("comparator", "between")
        self.add_param("min", f"{min:.2f}")
        self.add_param("max", f"{max:.2f}")
    
class BFactorQuery(SimpleQuery):
    """
    Query that filters structures within a defined B-factor range.
    
    
    Parameters
    ----------
    min: float
        The minimum resolution value.
    max: float
        The maximum resolution value.
    """
    
    def __init__(self, min, max):
        super().__init__("ResolutionQuery", "refine.B_iso_mean")
        self.add_param("comparator", "between")
        self.add_param("min", f"{min:.2f}")
        self.add_param("max", f"{max:.2f}")

class MolecularWeightQuery(SimpleQuery):
    """
    Query that filters structures within a molecular weight range.
    
    
    Parameters
    ----------
    min: float
        The minimum molecular weight (g/mol).
    max: float
        The maximum molecular weight (g/mol).
    """
    
    def __init__(self, min, max):
        super().__init__("MolecularWeightQuery",
                         "mvStructure.structureMolecularWeight")
        self.add_param("min", f"{min:.1f}")
        self.add_param("max", f"{max:.1f}")


def search(query):
    """
    Get all PDB IDs that meet the given query requirements,
    via the RCSB SEARCH service.
    
    This function requires an internet connection.
    
    Parameters
    ----------
    query : Query
        The search query.
    
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
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(_search_url, data=str(query), headers=headers)
    if r.text.startswith("Problem creating Query from XML"):
        raise ValueError(r.text)
    return r.text.split()
    