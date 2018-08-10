from typing import (
    List,
    Union,
)


def search(
    query: Union[CompositeQuery, MolecularWeightQuery, ResolutionQuery]
) -> List[str]: ...


class CompositeQuery:
    def __init__(
        self,
        operator: str,
        queries: List[Union[ResolutionQuery, MolecularWeightQuery]]
    ) -> None: ...


class MolecularWeightQuery:
    def __init__(self, min: int, max: int) -> None: ...


class Query:
    def __init__(self) -> None: ...
    def __str__(self) -> str: ...


class ResolutionQuery:
    def __init__(self, min: float, max: float) -> None: ...


class SimpleQuery:
    def __init__(self, query_type: str, parameter_class: str = '') -> None: ...
    def add_param(self, param: str, content: str) -> None: ...
