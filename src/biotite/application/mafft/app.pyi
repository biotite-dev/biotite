from biotite.sequence.seqtypes import ProteinSequence
from typing import List


class MafftApp:
    def __init__(
        self,
        sequences: List[ProteinSequence],
        bin_path: None = None,
        mute: bool = True
    ) -> None: ...
    def get_cli_arguments(self) -> List[str]: ...
    @staticmethod
    def get_default_bin_path() -> str: ...
