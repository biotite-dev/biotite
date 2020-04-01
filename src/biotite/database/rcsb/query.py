# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.rcsb"
__author__ = "Patrick Kunzmann, Maximilian Dombrowsky"
__all__ = ["Query", "CompositeQuery", "RangeQuery", "SimpleQuery",
           "ResolutionQuery", "BFactorQuery", "MolecularWeightQuery",
           "ChainCountQuery", "EntityCountQuery", "ModelCountQuery",
           "ChainLengthQuery",
           "MoleculeTypeQuery", "MethodQuery", "SoftwareQuery",
           "PubMedIDQuery", "UniProtIDQuery", "PfamIDQuery",
           "SequenceClusterQuery",
           "TextSearchQuery", "KeywordQuery", "TitleQuery",
           "DecriptionQuery", "MacromoleculeNameQuery",
           "ExpressionOrganismQuery", "AuthorQuery",
           "DateQuery",
           "search"]

from xml.etree.ElementTree import Element, SubElement, tostring
import datetime
import abc
import requests
from ..error import RequestError


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
    
    A combination of :class:`CompositeQuery` instances is not possible.
    
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

        PROTECTED: Do not call from outside.
        
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
    min, max: float or int or date
        The value range.
    """
    def __init__(self, query_type, parameter_class, min, max):
        super().__init__(query_type, parameter_class)
        self.add_param("comparator", "between")
        if min is not None:
            if isinstance(min, float):
                self.add_param("min", f"{min:.5f}")
            else:
                self.add_param("min", str(min))
        if max is not None:
            if isinstance(max, float):
                self.add_param("max", f"{max:.5f}")
            else:
                self.add_param("max", str(max))
    
    def add_param(self, param, content, omit_prefix=False):
        """
        Add a parameter (XML tag/text pair) to the query.

        PROTECTED: Do not call from outside.
        
        Parameters
        ----------
        param: str
            The XML tag name for the parameter.
        content : str
            The text content for the parameter.
        omit_prefix : bool, optional
            If true, the parameter prefix, specified with
            `parameter_class` in the constructor, is omitted.
        """
        if self._param_cls == "" or omit_prefix:
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
        The minimum B-factor value.
    max: float, optional
        The maximum B-factor value.
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

class ChainCountQuery(SimpleQuery):
    """
    Query that filters structures within a given range of number of
    chains.
    
    Parameters
    ----------
    min, max: int, optional
        The minimum and maximum number of chains.
    bio_assembly: bool, optional
        If set to true, the number of chains in a
        `biological assembly <https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/biological-assemblies>`_
        (oligomer) is counted.
        Otherwise, the number of chains in the asymmetric subunit, that
        is equal to the amount of chains in the file, is counted.
    """
    def __init__(self, min=None, max=None, bio_assembly=False):
        if bio_assembly:
            super().__init__("BiolUnitQuery")
            if min is not None:
                self.add_param("oligomeric_statemin", str(min))
            if max is not None:
                self.add_param("oligomeric_statemax", str(max))
        else:
            super().__init__("NumberOfChainsQuery", "struct_asym.numChains")
            if min is not None:
                self.add_param("min", str(min))
            if max is not None:
                self.add_param("max", str(max))

class EntityCountQuery(RangeQuery):
    """
    Query that filters structures, that have given number of entities,
    i.e. different chemical compounds.
    
    Parameters
    ----------
    min, max: int, optional
        The minimum and maximum number of entities.
    entity_type: {'protein', 'rna', 'dna', 'ligand', 'other'}, optional
        If set, only entities of the given type are considered.
    """
    _entity_type_dict = {
        "protein": "p",
        "rna": "r",
        "dna": "d",
        "ligand": "n",
        "other": "?",
    }
    
    def __init__(self, min=None, max=None, entity_type=None):
        super().__init__(
            "NumberOfEntitiesQuery", "struct_asym.numEntities", min, max
        )
        if entity_type is not None:
            self.add_param(
                "entity.type.",
                EntityCountQuery._entity_type_dict[entity_type.lower()],
                omit_prefix=True
            )

class ModelCountQuery(RangeQuery):
    """
    Query that filters structures, that have given number of models.
    
    Parameters
    ----------
    min, max: int, optional
        The minimum and maximum number of models.
    """
    def __init__(self, min=None, max=None):
        super().__init__(
            "ModelCountQuery", "mvStructure.modelCount", min, max
        )

class ChainLengthQuery(RangeQuery):
    """
    Query that filters structures with chains in the given chain length
    range.
    
    Parameters
    ----------
    min, max: int, optional
        The minimum and maximum number of chains.
    """
    def __init__(self, min=None, max=None):
        super().__init__(
            "SequenceLengthQuery", "v_sequence.chainLength", min, max
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

class SoftwareQuery(SimpleQuery):
    """
    Query that filters structures, that were elucidated with the help
    of the given software.
    
    Parameters
    ----------
    name: str
        Name of the software.
    """
    def __init__(self, name):
        super().__init__("SoftwareQuery", "VSoftware.name")
        self.add_param("value", name)

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

class SequenceClusterQuery(SimpleQuery):
    """
    Query that filters structures, that are in part of a
    `PDB sequence cluster <http://www.rcsb.org/pdb/statistics/clusterStatistics.do>`_
    with the given ID.
    
    Parameters
    ----------
    cluster_id: int
        The sequence cluster ID.
    """
    def __init__(self, cluster_id):
        super().__init__("SequenceClusterQuery")
        self.add_param("sequenceClusterName", cluster_id)

class TextSearchQuery(SimpleQuery):
    """
    Query that filters structures, that have the given text in their
    corresponding *mmCIF* coordinate file.
    
    Parameters
    ----------
    tex: str
        The text to search.
    """
    def __init__(self, text):
        super().__init__("AdvancedKeywordQuery")
        self.add_param("keywords", text)

class KeywordQuery(SimpleQuery):
    """
    Query that filters structures, that have the given keyword in their
    corresponding *mmCIF* field ``_struct_keywords.pdbx_keywords``.
    
    Parameters
    ----------
    keyword: str
        The text to search.
    """
    def __init__(self, keyword):
        super().__init__("TokenKeywordQuery", "struct_keywords.pdbx_keywords")
        self.add_param("value", keyword)

class TitleQuery(SimpleQuery):
    """
    Query that filters structures, that have the given text in their
    tile (*mmCIF* field ``_struct.title``).
    
    Parameters
    ----------
    keyword: str
        The text to search.
    """
    def __init__(self, text):
        super().__init__("StructTitleQuery", "struct.title")
        self.add_param("comparator", "contains")
        self.add_param("value", text)

class DecriptionQuery(SimpleQuery):
    """
    Query that filters structures, that have the given text in their
    description (*mmCIF* field ``_entity.pdbx_description``).
    
    Parameters
    ----------
    keyword: str
        The text to search.
    """
    def __init__(self, text):
        super().__init__("StructDescQuery", "entity.pdbx_description")
        self.add_param("comparator", "contains")
        self.add_param("value", text)

class MacromoleculeNameQuery(SimpleQuery):
    """
    Query that filters structures, that contain macromolecules with the
    given name
    (*mmCIF* fields
    ``_entity.pdbx_description`` or ``_entity_name_com.name``).
    
    Parameters
    ----------
    name: str
        The name of the macromolecule.
    """
    def __init__(self, name):
        super().__init__("MoleculeNameQuery")
        self.add_param("macromoleculeName", name)

class ExpressionOrganismQuery(SimpleQuery):
    """
    Query that filters structures, of which the protein was expressed
    in the specified host organism
    (*mmCIF* field ``_entity_src_gen.pdbx_host_org_scientific_name``).

    The unabbreviated scientific name is required,
    e.g. ``Escherichia coli``.
    Capitalization is not required.
    
    Parameters
    ----------
    name: str
        The scientific name of the host organism.
    """
    def __init__(self, name):
        super().__init__(
            "ExpressionOrganismQuery",
            "entity_src_gen.pdbx_host_org_scientific_name"
        )
        self.add_param("value", name)

class AuthorQuery(SimpleQuery):
    """
    Query that filters structures from a given author.
    
    Parameters
    ----------
    name: str
        The text to search.
    exact : bool, optional
        If true, the author name must completely match the given query
        name.
        If false, the author name merely must contain the given query
        name.
    """
    def __init__(self, name, exact=False):
        super().__init__("AdvancedAuthorQuery")
        if exact:
            self.add_param("exactMatch", "true")
        else:
            self.add_param("exactMatch", "false")
        self.add_param("audit_author.name", name)

class DateQuery(RangeQuery):
    """
    Query that filters structures that were deposited, released or
    revised in a given time interval
    
    Parameters
    ----------
    min_date, max_date: date or str
        The time interval, represented by two dates.
    event : {'deposition', 'release', 'revision'}
        The event to look for: Either the structure deposition, release
        or revision.
    """
    def __init__(self, min_date, max_date, event="deposition"):
        if event == "deposition":
            super().__init__(
                "DepositDateQuery",
                "pdbx_database_status.recvd_initial_deposition_date",
                min_date, max_date
            )
        elif event == "release":
            super().__init__(
                "ReleaseDateQuery",
                "pdbx_audit_revision_history.revision_date",
                min_date, max_date
            )
        elif event == "revision":
            super().__init__(
                "ReviseDateQuery",
                "pdbx_audit_revision_history.revision_date",
                min_date, max_date
            )
        else:
            raise ValueError(f"'{event}' is not a valid event")




def search(query, omit_chain=True):
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
    omit_chain: bool, optional
        If true, the chain information is removed from the IDs,
        e.g. '1L2Y:1' is converted into '1L2Y'.
        Only the ID without the chain information can be used as input
        to :func:`fetch()`.
    
    Warnings
    --------
    Even if you give valid input to this function, in rare cases the
    database might return no or malformed data to you.
    In these cases the request should be retried.
    When the issue occurs repeatedly, the error is probably in your
    input.

    
    Examples
    --------
    
    >>> query = ResolutionQuery(max=0.6)
    >>> ids = search(query)
    >>> print(ids)
    ['1EJG', '1I0T', '3NIR', '3P4J', '5D8V', '5NW3']
    """
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(_search_url, data=str(query), headers=headers)
    if r.text.startswith("Problem creating Query from XML"):
        raise RequestError(r.text)
    if "<html>" in r.text:
        # Response should contain plain PDB IDs,
        # a HTML tag indicates an error
        raise RequestError(r.text.replace("\n", " "))
    ids = r.text.split()
    if omit_chain:
        for i, id in enumerate(ids):
            if ":" in id:
                ids[i] = id.split(":")[0]
    return ids
    