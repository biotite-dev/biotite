# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.pubchem"
__author__ = "Patrick Kunzmann"
__all__ = ["Query", "NameQuery", "SmilesQuery", "InchiQuery", "InchiKeyQuery",
           "FormulaQuery", 
           "search"]

import abc
import collections
import requests
from .error import parse_error_details
from .throttle import ThrottleStatus
from ..error import RequestError


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
        Get the URL parameters for this query.

        Returns
        -------
        params : dict
            The URL parameters.
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
    [5950, 449619, 7311724, 155817681]
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
    [7843, 6360, 71309065, 16213391, 16213390]
    >>> atom_array = residue("ALA")
    >>> print(search(FormulaQuery.from_atoms(atom_array, number=5)))
    [5950, 5641, 1088, 398, 239]
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
    [5950, 449619, 7311724, 155817681]
    """
    r = requests.get(
        _base_url + query.get_input_url_path() + "/cids/TXT",
        params=query.get_params()
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
