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

    @abc.abstractmethod
    def get_input_url_path(self):
        pass

    def get_params(self):
        return {}


class NameQuery(Query):

    def __init__(self, name):
        self._name = name
    
    def get_input_url_path(self):
        return "compound/name"

    def get_params(self):
        return {"name": self._name}


class SmilesQuery(Query):

    def __init__(self, smiles):
        self._smiles = smiles
    
    def get_input_url_path(self):
        return "compound/smiles"

    def get_params(self):
        return {"smiles": self._smiles}


class InchiQuery(Query):

    def __init__(self, inchi):
        self._inchi = inchi
    
    def get_input_url_path(self):
        return "compound/inchi"

    def get_params(self):
        return {"inchi": self._inchi}


class InchiKeyQuery(Query):

    def __init__(self, inchi_key):
        self._inchi_key = inchi_key
    
    def get_input_url_path(self):
        return "compound/inchikey"

    def get_params(self):
        return {"inchikey": self._inchi_key}


class FormulaQuery(Query):
    """
    Notes
    -----
    As this query can be time-consuming, request timeouts may be
    expected.
    """

    def __init__(self, formula, allow_other_elements=False, number=None):
        self._formula = formula
        self._allow_other_elements = allow_other_elements
        self._number = number
    
    @staticmethod
    def from_atoms(atoms, allow_other_elements=False, number=None):
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


def search(query, return_throttle_status=False):
    """
    Get all CIDs that meet the given query requirements,
    via the PubChem REST API.
    
    This function requires an internet connection.
    
    Parameters
    ----------
    query : Query
        The search query.
    
    Returns
    -------
    ids : list of int
        List of all CIDs that meet the query requirement.
    """
    r = requests.get(
        _base_url + query.get_input_url_path() + "/cids/TXT",
        params=query.get_params()
    )
    if not r.ok:
        raise RequestError(parse_error_details(r.text))
    throttle_status = ThrottleStatus.from_response(r)
    cids = [int(cid) for cid in r.text.splitlines()]

    if return_throttle_status:
        return cids, throttle_status
    else:
        return cids
