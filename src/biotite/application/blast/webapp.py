# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application.blast"
__author__ = "Patrick Kunzmann"
__all__ = ["BlastWebApp"]

import time
from xml.etree import ElementTree
import requests
from biotite.application.application import AppState, requires_state
from biotite.application.blast.alignment import BlastAlignment
from biotite.application.webapp import WebApp
from biotite.database.entrez.query import Query
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.io.fasta.convert import get_sequence
from biotite.sequence.io.fasta.file import FastaFile
from biotite.sequence.seqtypes import NucleotideSequence, ProteinSequence
from biotite.sequence.sequence import Sequence

_ncbi_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"


class BlastWebApp(WebApp):
    """
    Perform a local alignment against a large sequence database using
    using the web-based BLAST application (by default NCBI BLAST).

    Parameters
    ----------
    program : str
        The specific BLAST program. One of 'blastn', 'megablast',
        'blastp', 'blastx', 'tblastn' and 'tblastx'.
    query : Sequence or str
        The query sequence. If a string is provided, it is interpreted
        as path to a FASTA file, if the string contains a valid FASTA
        file extension, otherwise it is interpreted as a single letter
        string representation of a sequence.
    database : str, optional
        The NCBI sequence database to blast against. By default it
        contains all sequences (`database`='nr'`).
    app_url : str, optional
        URL of the BLAST web app. By default NCBI BLAST is used.
        This can be changed to a private server or another cloud
        provider.
    obey_rules : bool, optional
        If true, the application raises an :class:`RuleViolationError`,
        if the server is contacted too often, based on the NCBI BLAST
        usage rules.
    mail : str, optional
        If a mail address is provided, it will be appended in the
        HTTP request. This allows the NCBI to contact you in case
        your application sends too many requests.
    """

    _last_contact = 0
    _last_request = 0
    _contact_delay = 3
    _request_delay = 60

    def __init__(
        self,
        program: str,
        query: Sequence | str,
        database: str = "nr",
        app_url: str = _ncbi_url,
        obey_rules: bool = True,
        mail: str = "padix.key@gmail.com",
    ) -> None:
        super().__init__(app_url, obey_rules)

        # 'megablast' is somehow not working
        # When entering the corresponding HTTPS request into a browser
        # you are redirected onto the blast mainpage
        if program not in ["blastn", "blastp", "blastx", "tblastn", "tblastx"]:
            raise ValueError(f"'{program}' is not a valid BLAST program")
        self._program: str = program

        requires_protein = program in ["blastp", "tblastn"]
        if isinstance(query, str) and query.endswith((".fa", ".fst", ".fasta")):
            # If string has a file extension, it is interpreted as
            # FASTA file from which the sequence is taken
            file = FastaFile.read(query)
            # Get first entry in file and take the sequence
            # (rather than header)
            self._query: str = str(get_sequence(file))
        elif isinstance(query, Sequence):
            self._query = str(query)
        else:
            self._query = str(query)

        # Check for unsuitable symbols in query string
        if requires_protein:
            ref_alphabet = ProteinSequence.alphabet
        else:
            ref_alphabet = NucleotideSequence.alphabet_amb
        for symbol in self._query:
            if symbol.upper() not in ref_alphabet:
                raise ValueError(f"Query sequence contains unsuitable symbol {symbol}")

        self._database: str = database

        self._gap_openining: float | None = None
        self._gap_extension: float | None = None
        self._word_size: int | None = None

        self._expect_value: float | None = None
        self._max_results: int | None = None
        self._entrez_query: str | None = None

        self._reward: int | None = None
        self._penalty: int | None = None

        self._matrix: str | None = None
        self._threshold: int | None = None

        self._mail: str = mail
        self._rid: str | None = None

    @requires_state(AppState.CREATED)
    def set_entrez_query(self, query: Query) -> None:
        """
        Limit the size of the database.
        Only sequences that match the query are searched.

        Parameters
        ----------
        query : Query
            An NCBI Entrez query.
        """
        self._entrez_query = str(query)

    @requires_state(AppState.CREATED)
    def set_max_results(self, number: int) -> None:
        """
        Limit the maximum number of results.

        Parameters
        ----------
        number : int
            The maximum number of results.
        """
        self._max_results = number

    @requires_state(AppState.CREATED)
    def set_max_expect_value(self, value: float) -> None:
        """
        Set the threshold expectation value (E-value).
        No alignments with an E-value above this threshold will be
        considered.

        The E-Value is the expectation value for the number of random
        sequences of a similar sized database getting an equal or higher
        score by change when aligned with the query sequence.

        Parameters
        ----------
        value : float
            The threshold E-value.
        """
        self._expect_value = value

    @requires_state(AppState.CREATED)
    def set_gap_penalty(self, opening: float, extension: float) -> None:
        """
        Set the affine gap penalty for the alignment.

        Parameters
        ----------
        opening : float
            The penalty for gap opening.
        extension : float
            The penalty for gap extension.
        """
        self._gap_openining = opening
        self._gap_extension = extension

    @requires_state(AppState.CREATED)
    def set_word_size(self, size: int) -> None:
        """
        Set the word size for alignment seeds.

        Parameters
        ----------
        size : int
            Word size.
        """
        self._word_size = size

    @requires_state(AppState.CREATED)
    def set_match_reward(self, reward: int) -> None:
        """
        Set the score of a symbol match in the alignment.

        Used only in 'blastn' and 'megablast'.

        Parameters
        ----------
        reward : int
            Match reward. Must be positive.
        """
        self._reward = reward

    @requires_state(AppState.CREATED)
    def set_mismatch_penalty(self, penalty: int) -> None:
        """
        Set the penalty of a symbol mismatch in the alignment.

        Used only in 'blastn' and 'megablast'.

        Parameters
        ----------
        penalty : int
            Mismatch penalty. Must be negative.
        """
        self._penalty = penalty

    @requires_state(AppState.CREATED)
    def set_substitution_matrix(self, matrix_name: str) -> None:
        """
        Set the penalty of a symbol mismatch in the alignment.

        Used only in 'blastp', "blastx', 'tblastn' and 'tblastx'.

        Parameters
        ----------
        matrix_name : str
            Name of the substitution matrix. Default is 'BLOSUM62'.
        """
        self._matrix = matrix_name.upper()

    @requires_state(AppState.CREATED)
    def set_threshold(self, threshold: int) -> None:
        """
        Set the threshold neighboring score for initial words.

        Used only in 'blastp', "blastx', 'tblastn' and 'tblastx'.

        Parameters
        ----------
        threshold : int
            Threshold value. Must be positve.
        """
        self._threshold = threshold

    def run(self) -> None:
        param_dict = {}
        param_dict["tool"] = "Biotite"
        param_dict["email"] = self._mail
        param_dict["CMD"] = "Put"
        param_dict["PROGRAM"] = self._program
        param_dict["QUERY"] = str(self._query)
        param_dict["DATABASE"] = self._database
        if self._entrez_query is not None:
            param_dict["ENTREZ_QUERY"] = self._entrez_query
        if self._max_results is not None:
            param_dict["HITLIST_SIZE"] = str(self._max_results)
        if self._expect_value is not None:
            param_dict["EXPECT"] = self._expect_value
        if self._gap_openining is not None and self._gap_extension is not None:
            param_dict["GAPCOSTS"] = "{:d} {:d}".format(
                self._gap_openining, self._gap_extension
            )
        if self._word_size is not None:
            param_dict["WORD_SIZE"] = self._word_size

        if self._program in ["blastn", "megablast"]:
            if self._reward is not None:
                param_dict["NUCL_REWARD"] = self._reward
            if self._penalty is not None:
                param_dict["NUCL_PENALTY"] = self._penalty

        if self._program in ["blastp", "blastx", "tblastn", "tblastx"]:
            if self._matrix is not None:
                param_dict["MATRIX"] = self._matrix
            if self._threshold is not None:
                param_dict["THRESHOLD"] = self._threshold

        request = requests.get(self.app_url(), params=param_dict)
        if "Submitted URI too large" in request.text:
            raise ValueError("The URI is too large, try a shorter sequence")
        self._contact()
        self._request()
        info_dict = BlastWebApp._get_info(request.text)
        self._rid = info_dict["RID"]

    def is_finished(self) -> bool:
        data_dict = {"FORMAT_OBJECT": "SearchInfo", "RID": self._rid, "CMD": "Get"}
        request = requests.get(self.app_url(), params=data_dict)
        self._contact()
        info_dict = BlastWebApp._get_info(request.text)
        if info_dict["Status"] == "UNKNOWN":
            # Indicates invalid query input values
            raise ValueError(
                "The input values seem to be invalid "
                "(Server responsed status 'UNKNOWN')"
            )
        return info_dict["Status"] == "READY"

    def wait_interval(self) -> float:
        # NCBI requires a 3 second delay between server contacts
        return BlastWebApp._contact_delay

    def clean_up(self) -> None:
        param_dict = {}
        param_dict["CMD"] = "Delete"
        param_dict["RID"] = self._rid
        requests.get(self.app_url(), params=param_dict)

    def evaluate(self) -> None:
        param_dict = {}
        param_dict["tool"] = "BiotiteClient"
        if self._mail is not None:
            param_dict["email"] = self._mail
        param_dict["CMD"] = "Get"
        param_dict["RID"] = self._rid
        param_dict["FORMAT_TYPE"] = "XML"
        param_dict["NCBI_GI"] = "T"
        request = requests.get(self.app_url(), params=param_dict)
        self._contact()

        self._alignments = []
        self._xml_response = request.text
        root = ElementTree.fromstring(self._xml_response)
        # Extract BlastAlignment objects from <Hit> tags
        hit_xpath = "./BlastOutput_iterations/Iteration/Iteration_hits/Hit"
        hits = root.findall(hit_xpath)
        for hit in hits:
            hsp = hit.find(".Hit_hsps/Hsp")
            if hsp is None:
                raise ValueError("BLAST XML response misses required 'Hsp' element")
            hit_definition = _xml_text(hit, "Hit_def")
            hit_id = _xml_text(hit, "Hit_accession")
            score = int(_xml_text(hsp, "Hsp_score"))
            e_value = float(_xml_text(hsp, "Hsp_evalue"))
            query_begin = int(_xml_text(hsp, "Hsp_query-from"))
            query_end = int(_xml_text(hsp, "Hsp_query-to"))
            hit_begin = int(_xml_text(hsp, "Hsp_hit-from"))
            hit_end = int(_xml_text(hsp, "Hsp_hit-to"))

            seq1_str = _xml_text(hsp, "Hsp_qseq")
            seq2_str = _xml_text(hsp, "Hsp_hseq")
            base_alignment = Alignment.from_strings(
                [seq1_str, seq2_str],
                sequence_factory=(
                    NucleotideSequence
                    if self._program in ["blastn", "megablast"]
                    else lambda seq_str: ProteinSequence(
                        seq_str.replace("U", "C").replace("O", "K")
                    )
                ),
            )

            alignment = BlastAlignment(
                base_alignment.sequences,
                base_alignment.trace,
                score,
                e_value,
                (query_begin, query_end),
                (hit_begin, hit_end),
                hit_id,
                hit_definition,
            )
            self._alignments.append(alignment)

    @requires_state(AppState.JOINED)
    def get_xml_response(self) -> str:
        """
        Get the raw XML response.

        Returns
        -------
        response : str
            The raw XML response.
        """
        return self._xml_response

    @requires_state(AppState.JOINED)
    def get_alignments(self) -> list[BlastAlignment]:
        """
        Get the resulting local sequence alignments.

        Returns
        -------
        alignment : list of BlastAlignment
            The local sequence alignments.
        """
        return self._alignments

    @staticmethod
    def _get_info(text: str) -> dict[str, str]:
        """
        Get the *QBlastInfo* block of the response HTML as dictionary
        """
        lines = [line for line in text.split("\n")]
        info_dict = {}
        in_info_block = False
        for line in lines:
            if "QBlastInfoBegin" in line:
                in_info_block = True
                continue
            if "QBlastInfoEnd" in line:
                in_info_block = False
                continue
            if in_info_block:
                pair = line.split("=")
                info_dict[pair[0].strip()] = pair[1].strip()
        return info_dict

    def _contact(self) -> None:
        """
        Resets the time since the last server contact. Used for
        detecting server rule violation.
        """
        contact = time.time()
        if (contact - BlastWebApp._last_contact) < BlastWebApp._contact_delay:
            self.violate_rule("The server was contacted too often")
        BlastWebApp._last_contact = contact

    def _request(self) -> None:
        """
        Resets the time since the last new alignment request. Used for
        detecting server rule violation.
        """
        request = time.time()
        if (request - BlastWebApp._last_request) < BlastWebApp._request_delay:
            self.violate_rule("Too frequent BLAST requests")
        BlastWebApp._last_request = request


def _xml_text(element: ElementTree.Element, child_tag: str) -> str:
    """
    Find ``child_tag`` under ``element`` and return its text content,
    raising a clear error if either the child is missing or its body is empty.
    """
    child = element.find(child_tag)
    if child is None or child.text is None:
        raise ValueError(f"BLAST XML response misses required '{child_tag}' content")
    return child.text
