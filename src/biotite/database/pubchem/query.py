# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.pubchem"
__author__ = "Patrick Kunzmann"
__all__ = [
    "Query",
    "NameQuery",
    "SmilesQuery",
    "InchiQuery",
    "InchiKeyQuery",
    "FormulaQuery",
    "SuperstructureQuery",
    "SubstructureQuery",
    "SimilarityQuery",
    "IdentityQuery",
    "search",
]

import abc
import collections
import copy
import requests
from biotite.database.error import RequestError
from biotite.database.pubchem.error import parse_error_details
from biotite.database.pubchem.throttle import ThrottleStatus
from biotite.structure.io.mol.mol import MOLFile

_base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"


class Query(metaclass=abc.ABCMeta):
    """
    A search query for the *PubChem* REST API.
    Unlike most other database interfaces in *Biotite*, multiple queries
    cannot be combined using logical operators.

    This is the abstract base class for all queries.
    """

    @abc.abstractmethod
    def get_input_url_path(self):
        """
        Get the *input* part of the request URL.

        Returns
        -------
        get_input_url_path : str
            The *input* part of the request URL.
            Must not contain slash characters at the beginning and end
            of the string.
        """
        pass

    def get_params(self):
        """
        Get the POST payload for this query.

        Returns
        -------
        params : dict (str -> object)
            The payload.
        """
        return {}

    def get_files(self):
        """
        Get the POST file payload for this query.

        Returns
        -------
        params : dict (str -> object)
            The file payload.
        """
        return {}


class NameQuery(Query):
    """
    A query that searches for compounds with the given name.

    The name of the compound must match the given name completely,
    but synonyms of the compound name are also considered.

    Parameters
    ----------
    name : str
        The compound name to be searched.

    Examples
    --------

    >>> print(search(NameQuery("Alanine")))
    [5950, ..., ...]
    """

    def __init__(self, name):
        self._name = name

    def get_input_url_path(self):
        return "compound/name"

    def get_params(self):
        return {"name": self._name}


class SmilesQuery(Query):
    """
    A query that searches for compounds with a given
    *Simplified Molecular Input Line Entry Specification* (*SMILES*)
    string.

    Parameters
    ----------
    smiles : str
        The *SMILES* string.

    Examples
    --------

    >>> print(search(SmilesQuery("CCCC")))
    [7843]
    """

    def __init__(self, smiles):
        self._smiles = smiles

    def get_input_url_path(self):
        return "compound/smiles"

    def get_params(self):
        return {"smiles": self._smiles}


class InchiQuery(Query):
    """
    A query that searches for compounds with a given
    *International Chemical Identifier* (*InChI*) string.

    Parameters
    ----------
    inchi : str
        The *InChI* string.

    Examples
    --------

    >>> print(search(InchiQuery("InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3")))
    [7843]
    """

    def __init__(self, inchi):
        self._inchi = inchi

    def get_input_url_path(self):
        return "compound/inchi"

    def get_params(self):
        return {"inchi": self._inchi}


class InchiKeyQuery(Query):
    """
    A query that searches for compounds with a given
    *International Chemical Identifier* (*InChI*) key.

    Parameters
    ----------
    inchi_key : str
        The *InChI* key.

    Examples
    --------

    >>> print(search(InchiKeyQuery("IJDNQMDRQITEOD-UHFFFAOYSA-N")))
    [7843]
    """

    def __init__(self, inchi_key):
        self._inchi_key = inchi_key

    def get_input_url_path(self):
        return "compound/inchikey"

    def get_params(self):
        return {"inchikey": self._inchi_key}


class FormulaQuery(Query):
    """
    A query that searches for compounds with the given molecular
    formula.

    The formula can also be created from an :class:`AtomArray` using
    the :meth:`from_atoms()` method.

    Parameters
    ----------
    formula : str
        The molecular formula, i.e. each capitalized element with its
        count in the compound concatenated into a single string.
    allow_other_elements : bool, optional
        If set to true, compounds with additional elements, not present
        in the molecular formula, will also match.
    number : int, optional
        The maximum number of matches that this query may return.
        By default, the *PubChem* default value is used, which can be
        considered unlimited.

    Examples
    --------

    >>> print(search(FormulaQuery("C4H10", number=5)))
    [..., ..., ..., ..., ...]
    >>> atom_array = residue("ALA")
    >>> print(search(FormulaQuery.from_atoms(atom_array, number=5)))
    [..., ..., ..., ..., ...]
    """

    def __init__(self, formula, allow_other_elements=False, number=None):
        self._formula = formula
        self._allow_other_elements = allow_other_elements
        self._number = number

    @staticmethod
    def from_atoms(atoms, allow_other_elements=False, number=None):
        """
        Create the query from an the given structure by using its
        molecular formula.

        Parameters
        ----------
        atoms : AtomArray or AtomArrayStack
            The structure to take the molecular formula from.
        allow_other_elements : bool, optional
            If set to true, compounds with additional elements, not
            present in the molecular formula, will also match.
        number : int, optional
            The maximum number of matches that this query may return.
            By default, the *PubChem* default value is used, which can
            be considered unlimited.

        Returns
        -------
        query : FormulaQuery
            The query.
        """
        element_counter = collections.Counter(atoms.element)
        formula = ""
        # C and H come first in molecular formula
        if "C" in element_counter:
            formula += _format_element("C", element_counter["C"])
            del element_counter["C"]
        if "H" in element_counter:
            formula += _format_element("H", element_counter["H"])
            del element_counter["H"]
        # All other elements follow in alphabetical order
        sorted_elements = sorted(element_counter.keys())
        for element in sorted_elements:
            formula += _format_element(element, element_counter[element])
        return FormulaQuery(formula, allow_other_elements, number)

    def get_input_url_path(self):
        # The 'fastformula' service seems not to accept the formula
        # in the parameter section of the request
        return f"compound/fastformula/{self._formula}"

    def get_params(self):
        params = {"AllowOtherElements": self._allow_other_elements}
        # Only set maximum number, if provided by the user
        # The PubChem default value for this might change over time
        if self._number is not None:
            params["MaxRecords"] = self._number
        return params


def _format_element(element, count):
    if count == 1:
        return element.capitalize()
    else:
        return element.capitalize() + str(count)


class StructureQuery(Query, metaclass=abc.ABCMeta):
    """
    Abstract superclass for all structure based searches.
    This class handles structure inputs and option formatting.

    Exactly one of the input structure parameters `smiles`, `smarts`,
    `inchi`, `sdf` or `cid` must be given.

    Parameters
    ----------
    smiles : str, optional
        The query *SMILES* string.
    smarts : str, optional
        The query *SMARTS* pattern.
    inchi : str, optional
        The query *InChI* string.
    sdf : str, optional
        A query structure as SDF formatted string.
        Usually :meth:`from_atoms()` is used to create the SDF from an
        :class:`AtomArray`.
    cid : int, optional
        The query structure given as CID.
    number : int, optional
        The maximum number of matches that this query may return.
        By default, the *PubChem* default value is used, which can
        be considered unlimited.
    """

    _query_keys = ("smiles", "smarts", "inchi", "sdf", "cid")

    def __init__(self, **kwargs):
        query_key_found = False
        for query_key in StructureQuery._query_keys:
            if query_key in kwargs:
                if not query_key_found:
                    self._query_key = query_key
                    self._query_val = kwargs[query_key]
                    # Delete parameter from kwargs for later check for
                    # unused (invalid) parameters
                    del kwargs[query_key]
                    query_key_found = True
                else:
                    # A query key was already found,
                    # duplicates are not allowed
                    raise TypeError(
                        "Only one of 'smiles', 'smarts', 'inchi', 'sdf' or "
                        "'cid' may be given"
                    )
        if not query_key_found:
            raise TypeError(
                "Expected exactly one of 'smiles', 'smarts', 'inchi', 'sdf' or 'cid'"
            )
        if "number" in kwargs:
            self._number = kwargs["number"]
            del kwargs["number"]
        else:
            self._number = None
        # If there are still remaining parameters that were not handled
        # by this superclass or the inheriting class, they are invalid
        for key in kwargs:
            raise TypeError(f"'{key}' is an invalid keyword argument")

    @classmethod
    def from_atoms(cls, atoms, *args, **kwargs):
        """
        Create a query using the given query structure.

        Parameters
        ----------
        atoms : AtomArray or AtomArrayStack
            The query structure.
        *args, **kwargs
            See the constructor for additional options.

        Returns
        -------
        query : StructureQuery
            The query object.
        """
        mol_file = MOLFile()
        mol_file.set_structure(atoms)
        # Every MOL string with "$$$$" is a valid SDF string
        # Important: USE MS-style new lines
        return cls(*args, sdf="\r\n".join(mol_file.lines) + "\r\n$$$$\r\n", **kwargs)

    def get_input_url_path(self):
        input_string = f"compound/{self.search_type()}/{self._query_key}"
        if self._query_key == "cid":
            # Put CID in URL and not in POST payload,
            # as PubChem is confused otherwise
            input_string += "/" + str(self._query_val)
        return input_string

    def get_params(self):
        if self._query_key not in ("cid", "sdf"):
            # CID is in URL
            # SDF is given as file
            params = {self._query_key: self._query_val}
        else:
            params = {}
        # Only set maximum number, if provided by the user
        # The PubChem default value for this might change over time
        if self._number is not None:
            params["MaxRecords"] = self._number
        for key, val in self.search_options().items():
            # Convert 'snake case' Python parameters
            # to 'camel case' request parameters
            key = "".join([word.capitalize() for word in key.split("_")])
            params[key] = val
        return params

    def get_files(self):
        # Multi-line SDF string requires payload as file
        if self._query_key == "sdf":
            return {"sdf": self._query_val}
        else:
            return {}

    @abc.abstractmethod
    def search_type(self):
        """
        Get the type of performed search for the request input part.

        PROTECTED: Override when inheriting.

        Returns
        -------
        search_type : str
            The search type for the input part, i.e. the part directly
            after ``compound/``.
        """
        pass

    def search_options(self):
        """
        Get additional options for the POST options.

        PROTECTED: Override when inheriting.

        Returns
        -------
        options : dict (str -> object)
            They keys are automatically converted from *snake case* to
            *camel case* required by the request parameters.
        """
        return {}


class SuperOrSubstructureQuery(StructureQuery, metaclass=abc.ABCMeta):
    """
    Abstract superclass for super- and substructure searches.
    This class handles specific options for these searches.

    Exactly one of the input structure parameters `smiles`, `smarts`,
    `inchi`, `sdf` or `cid` must be given.

    Parameters
    ----------
    smiles : str, optional
        The query *SMILES* string.
    smarts : str, optional
        The query *SMARTS* pattern.
    inchi : str, optional
        The query *InChI* string.
    sdf : str, optional
        A query structure as SDF formatted string.
        Usually :meth:`from_atoms()` is used to create the SDF from an
        :class:`AtomArray`.
    cid : int, optional
        The query structure given as CID.
    number : int, optional
        The maximum number of matches that this query may return.
        By default, the *PubChem* default value is used, which can
        be considered unlimited.
    match_charges : bool, optional
        If set to true, atoms must match the specified charge.
    match_tautomers : bool, optional
        If set to true, allow match to tautomers of the given structure.
    rings_not_embedded : bool, optional
        If set to true, rings may not be embedded in a larger system.
    single_double_bonds_match : bool, optional
        If set to true, single or double bonds match aromatic bonds.
    chains_match_rings : bool, optional
        If set to true, chain bonds in the query may match rings in
        hits.
    strip_hydrogen : bool, optional
        If set to true, remove any explicit hydrogens before searching.
    stereo : {'ignore', 'exact', 'relative', 'nonconflicting'}, optional
        How to handle stereo.

    Notes
    -----
    Optional parameter descriptions are taken from the *PubChem* REST
    API
    `documentation <https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=Substructure-Superstructure>`_.
    """

    _option_defaults = {
        "match_charges": False,
        "match_tautomers": False,
        "rings_not_embedded": False,
        "single_double_bonds_match": True,
        "chains_match_rings": True,
        "strip_hydrogen": False,
        "stereo": "ignore",
    }

    def __init__(self, **kwargs):
        self._options = copy.copy(SuperOrSubstructureQuery._option_defaults)
        for option, value in kwargs.items():
            if option in SuperOrSubstructureQuery._option_defaults.keys():
                self._options[option] = value
                del kwargs[option]
        super().__init__(**kwargs)

    def search_options(self):
        return self._options


class SuperstructureQuery(SuperOrSubstructureQuery):
    """
    A query that searches for all structures, where the given
    input structure is a superstructure.
    In other words, this query matches substructures of the input
    structure.

    Exactly one of the input structure parameters `smiles`, `smarts`,
    `inchi`, `sdf` or `cid` must be given.

    Parameters
    ----------
    smiles : str, optional
        The query *SMILES* string.
    smarts : str, optional
        The query *SMARTS* pattern.
    inchi : str, optional
        The query *InChI* string.
    sdf : str, optional
        A query structure as SDF formatted string.
        Usually :meth:`from_atoms()` is used to create the SDF from an
        :class:`AtomArray`.
    cid : int, optional
        The query structure given as CID.
    number : int, optional
        The maximum number of matches that this query may return.
        By default, the *PubChem* default value is used, which can
        be considered unlimited.
    match_charges : bool, optional
        If set to true, atoms must match the specified charge.
    match_tautomers : bool, optional
        If set to true, allow match to tautomers of the given structure.
    rings_not_embedded : bool, optional
        If set to true, rings may not be embedded in a larger system.
    single_double_bonds_match : bool, optional
        If set to true, single or double bonds match aromatic bonds.
    chains_match_rings : bool, optional
        If set to true, chain bonds in the query may match rings in
        hits.
    strip_hydrogen : bool, optional
        If set to true, remove any explicit hydrogens before searching.
    stereo : {'ignore', 'exact', 'relative', 'nonconflicting'}, optional
        How to handle stereo.

    Notes
    -----
    Optional parameter descriptions are taken from the *PubChem* REST
    API
    `documentation <https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=Substructure-Superstructure>`_.

    Examples
    --------

    >>> # CID of alanine
    >>> print(search(SuperstructureQuery(cid=5950, number=5)))
    [..., ..., ..., ..., ...]
    >>> # AtomArray of alanine
    >>> atom_array = residue("ALA")
    >>> print(search(SuperstructureQuery.from_atoms(atom_array, number=5)))
    [..., ..., ..., ..., ...]
    """

    def search_type(self):
        return "fastsuperstructure"


class SubstructureQuery(SuperOrSubstructureQuery):
    """
    A query that searches for all structures, where the given
    input structure is a substructure.
    In other words, this query matches superstructures of the input
    structure.

    Exactly one of the input structure parameters `smiles`, `smarts`,
    `inchi`, `sdf` or `cid` must be given.

    Parameters
    ----------
    smiles : str, optional
        The query *SMILES* string.
    smarts : str, optional
        The query *SMARTS* pattern.
    inchi : str, optional
        The query *InChI* string.
    sdf : str, optional
        A query structure as SDF formatted string.
        Usually :meth:`from_atoms()` is used to create the SDF from an
        :class:`AtomArray`.
    cid : int, optional
        The query structure given as CID.
    number : int, optional
        The maximum number of matches that this query may return.
        By default, the *PubChem* default value is used, which can
        be considered unlimited.
    match_charges : bool, optional
        If set to true, atoms must match the specified charge.
    match_tautomers : bool, optional
        If set to true, allow match to tautomers of the given structure.
    rings_not_embedded : bool, optional
        If set to true, rings may not be embedded in a larger system.
    single_double_bonds_match : bool, optional
        If set to true, single or double bonds match aromatic bonds.
    chains_match_rings : bool, optional
        If set to true, chain bonds in the query may match rings in
        hits.
    strip_hydrogen : bool, optional
        If set to true, remove any explicit hydrogens before searching.
    stereo : {'ignore', 'exact', 'relative', 'nonconflicting'}, optional
        How to handle stereo.

    Notes
    -----
    Optional parameter descriptions are taken from the *PubChem* REST
    API
    `documentation <https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=Substructure-Superstructure>`_.

    Examples
    --------

    >>> # CID of alanine
    >>> print(search(SubstructureQuery(cid=5950, number=5)))
    [5950, ..., ..., ..., ...]
    >>> # AtomArray of alanine
    >>> atom_array = residue("ALA")
    >>> print(search(SubstructureQuery.from_atoms(atom_array, number=5)))
    [5950, ..., ..., ..., ...]
    """

    def search_type(self):
        return "fastsubstructure"


class SimilarityQuery(StructureQuery):
    """
    A query that searches for all structures similar to the given
    input structure.

    Exactly one of the input structure parameters `smiles`, `smarts`,
    `inchi`, `sdf` or `cid` must be given.

    Parameters
    ----------
    threshold : float, optional
        The minimum required *Tanimoto* similarity for a match.
        Must be between 0 (no similarity) and 1 (complete match).
    conformation_based : bool, optional
        If set to true, the similarity is computed based on the
        3D conformation.
        By default, only the elements and bonds between the atoms are
        considered for similarity computation.
    smiles : str, optional
        The query *SMILES* string.
    smarts : str, optional
        The query *SMARTS* pattern.
    inchi : str, optional
        The query *InChI* string.
    sdf : str, optional
        A query structure as SDF formatted string.
        Usually :meth:`from_atoms()` is used to create the SDF from an
        :class:`AtomArray`.
    cid : int, optional
        The query structure given as CID.
    number : int, optional
        The maximum number of matches that this query may return.
        By default, the *PubChem* default value is used, which can
        be considered unlimited.

    Notes
    -----
    The conformation based similarity measure uses *shape-Tanimoto* and
    *color-Tanimoto* scores :footcite:`Kim2018`.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> # CID of alanine
    >>> print(search(SimilarityQuery(cid=5950, threshold=1.0, number=5)))
    [5950, ..., ..., ..., ...]
    >>> # AtomArray of alanine
    >>> atom_array = residue("ALA")
    >>> print(search(SimilarityQuery.from_atoms(atom_array, threshold=1.0, number=5)))
    [5950, ..., ..., ..., ...]
    """

    def __init__(self, threshold=0.9, conformation_based=False, **kwargs):
        self._threshold = threshold
        self._conformation_based = conformation_based
        super().__init__(**kwargs)

    def search_type(self):
        dim = "3d" if self._conformation_based else "2d"
        return f"fastsimilarity_{dim}"

    def search_options(self):
        return {"threshold": int(round(self._threshold * 100))}


class IdentityQuery(StructureQuery):
    """
    A query that searches for all structures that are identical to the
    given input structure.

    Exactly one of the input structure parameters `smiles`, `smarts`, `inchi`,
    `sdf` or `cid` must be given.

    Parameters
    ----------
    identity_type : {'same_connectivity', 'same_tautomer', 'same_stereo', 'same_isotope', 'same_stereo_isotope', 'nonconflicting_stereo', 'same_isotope_nonconflicting_stereo'}, optional
        The type of identity search.
    smiles : str, optional
        The query *SMILES* string.
    smarts : str, optional
        The query *SMARTS* pattern.
    inchi : str, optional
        The query *InChI* string.
    sdf : str, optional
        A query structure as SDF formatted string.
        Usually :meth:`from_atoms()` is used to create the SDF from an
        :class:`AtomArray`.
    cid : int, optional
        The query structure given as CID.
    number : int, optional
        The maximum number of matches that this query may return.
        By default, the *PubChem* default value is used, which can
        be considered unlimited.

    Examples
    --------

    >>> # CID of alanine
    >>> print(search(IdentityQuery(cid=5950)))
    [5950]
    >>> # AtomArray of alanine
    >>> atom_array = residue("ALA")
    >>> print(search(IdentityQuery.from_atoms(atom_array)))
    [5950]
    """

    def __init__(self, identity_type="same_stereo_isotope", **kwargs):
        self._identity_type = identity_type
        super().__init__(**kwargs)

    def search_type(self):
        return "fastidentity"

    def get_params(self):
        # Use 'get_params()' instead of 'search_options()', since the
        # parameter 'identity_type' in the REST API is *snake case*
        # -> Conversion to *camel case* is undesirable
        params = super().get_params()
        params["identity_type"] = self._identity_type
        return params


def search(query, throttle_threshold=0.5, return_throttle_status=False):
    """
    Get all CIDs that meet the given query requirements,
    via the PubChem REST API.

    This function requires an internet connection.

    Parameters
    ----------
    query : Query
        The search query.
    throttle_threshold : float or None, optional
        A value between 0 and 1.
        If the load of either the request time or count exceeds this
        value the execution is halted.
        See :class:`ThrottleStatus` for more information.
        If ``None`` is given, the execution is never halted.
    return_throttle_status : float, optional
        If set to true, the :class:`ThrottleStatus` is also returned.

    Returns
    -------
    ids : list of int
        List of all compound IDs (CIDs) that meet the query requirement.
    throttle_status : ThrottleStatus
        The :class:`ThrottleStatus` obtained from the server response.
        This can be used for custom request throttling, for example.
        Only returned, if `return_throttle_status` is set to true.

    Examples
    --------

    >>> print(search(NameQuery("Alanine")))
    [5950, ..., ...]
    """
    # Use POST to be compatible with the larger payloads
    # of structure searches
    if query.get_files():
        files = {key: file for key, file in query.get_files().items()}
    else:
        files = None
    r = requests.post(
        _base_url + query.get_input_url_path() + "/cids/TXT",
        data=query.get_params(),
        files=files,
    )
    if not r.ok:
        raise RequestError(parse_error_details(r.text))
    throttle_status = ThrottleStatus.from_response(r)
    if throttle_threshold is not None:
        throttle_status.wait_if_busy(throttle_threshold)

    cids = [int(cid) for cid in r.text.splitlines()]
    if return_throttle_status:
        return cids, throttle_status
    else:
        return cids
