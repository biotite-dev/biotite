# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann, Maximilian Dombrowsky"
__all__ = ["Query", "CompositeQuery", "RangeQuery", "SimpleQuery",
           "ResolutionQuery", "BFactorQuery", "MolecularWeightQuery",
           "MoleculeTypeQuery", "MethodQuery",
           "PubMedIDQuery", "UniProtIDQuery", "PfamIDQuery",
           "TextSearchQuery",
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


class RangeQuery(SimpleQuery, metaclass=abc.ABCMeta):
    """
    The abstract base class for all non-composite queries, that allow
    a minimum and a maximum value
    (comparator ``between`` in the XML query).
    
    Parameters
    ----------
    query_type: str
        The name of the query type. This is the suffix for the
        'QueryType' XML tag.
    parameter_class : optional
        If specifed, this string is the prefix for all parameters
        (XML tags) of the query.
    """
    def __init__(self, query_type, parameter_class, min, max):
        super().__init__(query_type, parameter_class)
        self.add_param("comparator", "between")
        if min is not None:
            self.add_param("min", f"{min:.5f}")
        if max is not None:
            self.add_param("max", f"{max:.5f}")
    
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


class ResolutionQuery(RangeQuery):
    """
    Query that filters X-ray elucidated structures within a defined
    resolution range.
    
    Parameters
    ----------
    min: float, optional
        The minimum resolution value.
    max: float, optional
        The maximum resolution value.
    """
    def __init__(self, min=None, max=None):
        super().__init__(
            "ResolutionQuery", "refine.ls_d_res_high",
            min, max
        )
    
class BFactorQuery(RangeQuery):
    """
    Query that filters structures within a defined average B-factor
    range.
    
    Parameters
    ----------
    min: float, optional
        The minimum resolution value.
    max: float, optional
        The maximum resolution value.
    """
    def __init__(self, min=None, max=None):
        super().__init__("AverageBFactorQuery", "refine.B_iso_mean", min, max)

class MolecularWeightQuery(RangeQuery):
    """
    Query that filters structures within a molecular weight range.
    Water molecules are excluded from the molecular weight.
    
    Parameters
    ----------
    min: float, optional
        The minimum molecular weight (g/mol).
    max: float, optional
        The maximum molecular weight (g/mol).
    """
    def __init__(self, min=None, max=None):
        super().__init__(
            "MolecularWeightQuery", "mvStructure.structureMolecularWeight",
            min, max
        )


class MoleculeTypeQuery(SimpleQuery):
    """
    Query that filters structures with a defined molecular type.

    Parameters
    ----------
    rna: bool, optional
        If true, RNA structures are selected, otherwise excluded.
        By default, the occurrence of this molecule type is ignored.
    dna: bool, optional
        If true, DNA structures are selected, otherwise excluded.
        By default, the occurrence of this molecule type is ignored.
    hyrbid: bool, optional
        If true, DNA/RNA hybrid structures are selected,
        otherwise excluded.
        By default, the occurrence of this molecule type is ignored.
    protein: bool, optional
        If true, protein structures are selected, otherwise excluded.
        selected.
        By default, the occurrence of this molecule type is ignored.
    """
    def __init__(self, rna=None, dna=None, hybrid=None, protein=None):
        super().__init__("ChainTypeQuery","")
        
        if rna is None:
            self.add_param("containsRna","?")
        elif rna:
            self.add_param("containsRna","Y")
        else:
            self.add_param("containsRna","N")
        
        if dna is None:
            self.add_param("containsDna","?")
        elif dna:
            self.add_param("containsDna","Y")
        else:
            self.add_param("containsDna","N")
        
        if hybrid is None:
            self.add_param("containsHybrid","?")
        elif hybrid:
            self.add_param("containsHybrid","Y")
        else:
            self.add_param("containsHybrid","N")
        
        if protein is None:
            self.add_param("containsProtein","?")
        elif protein:
            self.add_param("containsProtein","Y")
        else:
            self.add_param("containsProtein","N")

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

class PubMedIDQuery(SimpleQuery):
    """
    Query that filters structures, that are published by any article
    in the given list of PubMed IDs.
    
    Parameters
    ----------
    ids: iterable object of str
        A list of PubMed IDs.
    """
    def __init__(self, ids):
        super().__init__("PubmedIdQuery")
        self.add_param("pubMedIdList", ", ".join(ids))

class UniProtIDQuery(SimpleQuery):
    """
    Query that filters structures, that are referenced by any entry
    in the given list of UniProtKB IDs.
    
    Parameters
    ----------
    ids: iterable object of str
        A list of UniProtKB IDs.
    """
    def __init__(self, ids):
        super().__init__("UpAccessionIdQuery")
        self.add_param("accessionIdList", ", ".join(ids))

class PfamIDQuery(SimpleQuery):
    """
    Query that filters structures, that are referenced by any entry
    in the given list of Pfam family IDs.
    
    Parameters
    ----------
    ids: iterable object of str
        A list of Pfam family IDs.
    """
    def __init__(self, ids):
        super().__init__("PfamIdQuery")
        self.add_param("pfamID", ", ".join(ids))

class TextSearchQuery(SimpleQuery):
    """
    Query that filters structures, that have the given text in its
    corresponding *mmCIF* coordinate file.
    
    Parameters
    ----------
    tex: str
        The text to search.
    """
    def __init__(self, text):
        super().__init__("AdvancedKeywordQuery")
        self.add_param("keywords", text)




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
    
    Warnings
    --------
    Even if you give valid input to this function, in rare cases the
    database might return no or malformed data to you.
    In these cases the request should be retried.
    When the issue occurs repeatedly, the error is probably in your
    input.

    
    Examples
    --------
    
    >>> query = ResolutionQuery(0.6)
    >>> ids = search(query)
    >>> print(ids)
    ['1EJG', '1I0T', '3NIR', '3P4J', '5D8V', '5NW3']
    """
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(_search_url, data=str(query), headers=headers)
    if r.text.startswith("Problem creating Query from XML"):
        raise ValueError(r.text)
    return r.text.split()
    