# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from typing import Iterable, List, Union, Optional
from abc import abstractmethod
from xml.etree.ElementTree import Element


class Query:
    def __init__(self) -> None: ...
    def get_query(self) -> Element: ...
    def __str__(self) -> str: ...

class CompositeQuery(Query):
    def __init__(
        self, operator: str, queries: Iterable[SimpleQuery]
    ) -> None: ...

class SimpleQuery(Query):
    def __init__(self, query_type: str, parameter_class: str = "") -> None: ...
    def add_param(self, param: str, content: str) -> None: ...

class MethodQuery(SimpleQuery):
    def __init__(
        self, method: str, has_data: Optional[bool] = None
    ) -> None: ...

class ResolutionQuery(SimpleQuery):
    def __init__(self, min: float, max: float) -> None: ...

class BFactorQuery(SimpleQuery):
    def __init__(self, min: float, max: float) -> None: ...

class MolecularWeightQuery(SimpleQuery):
    def __init__(self, min: float, max: float) -> None: ...


def search(query: Query) -> List[str]: ...