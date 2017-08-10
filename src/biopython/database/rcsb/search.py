# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import requests
import abc
from xml.etree.ElementTree import Element, SubElement, tostring

_search_url = "https://www.rcsb.org/pdb/rest/search"

class Query(metaclass=abc.ABCMeta):
    
    def __init__(self):
        pass
    
    def get_query(self):
        return self.query
    
    def __str__(self):
        return tostring(self.query, encoding="unicode")


class CompositeQuery(Query):
    
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
    
    def __init__(self, query_type, parameter_class=""):
        super().__init__()
        self.query = Element("orgPdbQuery")
        self._param_cls = parameter_class
        type = SubElement(self.query, "queryType")
        type.text = "org.pdb.query.simple." + query_type
        
    def add_param(self, param, content):
        if self._param_cls == "":
            child = SubElement(self.query, param)
        else:
            child = SubElement(self.query, self._param_cls + "." + param)
        child.text = content


class MethodQuery(SimpleQuery):
    
    def __init__(self, method, has_data=None):
        super().__init__("ExpTypeQuery", "mvStructure")
        self.add_param("expMethod.value", method.upper())
        if has_data == True:
            self.add_param("hasExperimentalData.value", "Y")
        elif has_data == False:
            self.add_param("hasExperimentalData.value", "N")
            
class ResolutionQuery(SimpleQuery):
    
    def __init__(self, min, max):
        super().__init__("ResolutionQuery", "refine.ls_d_res_high")
        self.add_param("comparator", "between")
        self.add_param("min", "{:.2f}".format(min))
        self.add_param("max", "{:.2f}".format(max))
        
class BFactorQuery(SimpleQuery):
    
    def __init__(self, min, max):
        super().__init__("ResolutionQuery", "refine.B_iso_mean")
        self.add_param("comparator", "between")
        self.add_param("min", "{:.2f}".format(min))
        self.add_param("max", "{:.2f}".format(max))
    
    
    
def search(query):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(_search_url, data=str(query), headers=headers)
    return r.text.split()
    