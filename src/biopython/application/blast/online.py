# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from .alignment import BlastAlignment
from ..application import Application, requires_state, AppState
from ..webapp import WebApp, RuleViolationError
from ...sequence.sequence import Sequence
from ...sequence.seqtypes import DNASequence, ProteinSequence
from ...sequence.io.fasta.file import FastaFile
from ...sequence.align.align import Alignment
import time
import requests
from xml.etree import ElementTree

__all__ = ["BlastWebApp"]


_ncbi_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

class BlastWebApp(WebApp):
    
    _last_contact = 0
    _last_request = 0
    _contact_delay = 3
    _request_delay = 60
    
    def __init__(self, program, query, database="nr",
                 app_url=_ncbi_url, obey_rules=True, mail=None):
        super().__init__(app_url, obey_rules)
        
        if program not in ["blastn", "megablast", "blastp",
                           "blastx", "tblastn", "tblastx"]:
            raise ValueError("'{:}' is not a valid BLAST program"
                             .format(program))
        self._program = program
        
        requires_protein = (program in ["blastn", "megablast",
                                        "blastx", "blastx"])
        if isinstance(query, str) and query.endswith((".fa",".fst",".fasta")):
                file = FastaFile()
                file.read(self._query)
                sequence = file.get_sequence()
                if isinstance(sequence, ProteinSequence) != requires_protein:
                    raise ValueError("Query type is not suitable for program")
        elif isinstance(query, Sequence):
            self._query = query.copy()
        else:
            if requires_protein:
                self._query = ProteinSequence(str(query))
            else:
                self._query = ProteinSequence(str(query))
        
        self._database = database
        self._gap_openining = None
        self._gap_extension = None
        self._word_size = None
        self._expect_value = None
        
        self._reward = None
        self._penalty = None
        
        self._matrix = None
        self._threshold = None
        
        self._mail=mail
        self._rid = None
    
    @requires_state(AppState.CREATED)
    def set_max_expect_value(self, value):
        self._expect_value = value
    
    @requires_state(AppState.CREATED)
    def set_gap_penalty(self, opening, extension):
        self._gap_openining = opening
        self._gap_extension = extension
    
    @requires_state(AppState.CREATED)
    def set_word_size(self, size):
        self._word_size = size
    
    @requires_state(AppState.CREATED)
    def set_match_reward(self, reward):
        self._reward = score
    
    @requires_state(AppState.CREATED)
    def set_mismatch_penalty(self, penalty):
        self._penalty = penalty
    
    @requires_state(AppState.CREATED)
    def set_substitution_matrix(self, matrix_name):
        self._matrix = matrix_name.upper()
    
    @requires_state(AppState.CREATED)
    def set_threshold(self, threshold):
        self._threshold = threshold
    
    def run(self):
        param_dict = {}
        param_dict["tool"] = "BiopythonClient"
        if self._mail is not None:
            param_dict["email"] = self._mail
        param_dict["CMD"] = "Put"
        param_dict["PROGRAM"] = self._program
        param_dict["QUERY"] = str(self._query)
        param_dict["DATABASE"] = self._database
        if self._expect_value is not None:
            param_dict["EXPECT"] = self._expect_value
        if self._gap_openining is not None and self._gap_extension is not None:
            param_dict["GAPCOSTS"] = "{:d} {:d}".format(self._gap_openining,
                                                      self._gap_extension)
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
        self._contact()
        self._request()
        info_dict = BlastOnline._get_info(request.text)
        self._rid = info_dict["RID"]
    
    def is_finished(self):
        data_dict = {"FORMAT_OBJECT" : "SearchInfo",
                     "RID"           : self._rid,
                     "CMD"           : "Get"}
        request = requests.get(self.app_url(), params=data_dict)
        self._contact()
        info_dict = BlastOnline._get_info(request.text)
        return info_dict["Status"] == "READY"
        
    def wait_interval(self):
        # NCBI requires a 3 second delay between server contacts
        return BlastOnline._contact_delay
    
    def clean_up(self):
        param_dict = {}
        param_dict["CMD"] = "Delete"
        param_dict["RID"] = self._rid
        request = requests.get(self.app_url(), params=param_dict)
    
    def evaluate(self):
        param_dict = {}
        param_dict["tool"] = "BiopythonClient"
        if self._mail is not None:
            param_dict["email"] = self._mail
        param_dict["CMD"] = "Get"
        param_dict["RID"] = self._rid
        param_dict["FORMAT_TYPE"] = "XML"
        param_dict["NCBI_GI"] = "T"
        request = requests.get(self.app_url(), params=param_dict)
        self._contact()
        
        self._alignments = []
        root = ElementTree.fromstring(request.text)
        # Extract BlastAlignment objects from <Hit> tags
        hit_xpath = "./BlastOutput_iterations/Iteration/Iteration_hits/Hit"
        hits = root.findall(hit_xpath)
        for hit in hits:
            hit_definition = hit.find("Hit_def").text
            hit_id = hit.find("Hit_accession").text
            hsp = hit.find(".Hit_hsps/Hsp")
            score = int(hsp.find("Hsp_score").text)
            e_value = float(hsp.find("Hsp_evalue").text)
            query_begin = int(hsp.find("Hsp_query-from").text)
            query_end = int(hsp.find("Hsp_query-to").text)
            hit_begin = int(hsp.find("Hsp_hit-from").text)
            hit_end = int(hsp.find("Hsp_hit-to").text)
            
            seq1_str = hsp.find("Hsp_qseq").text
            seq2_str = hsp.find("Hsp_hseq").text
            if self._program in ["blastn", "megablast"]:
                # DNASequence/ProteinSequence do ignore gaps
                # Gaps are represented by the trace
                seq1 = DNASequence(seq1_str.replace("-", ""))
                seq2 = DNASequence(seq2_str.replace("-", ""))
            else:
                seq1 = ProteinSequence(seq1_str.replace("-", ""))
                seq2 = ProteinSequence(seq2_str.replace("-", ""))
            trace = Alignment.trace_from_string(seq1_str, seq2_str)
            
            alignment = BlastAlignment( seq1 ,seq2, trace, score, e_value,
                                        (query_begin, query_end),
                                        (query_begin, query_end),
                                        hit_id, hit_definition )
            self._alignments.append(alignment)
    
    @requires_state(AppState.JOINED)
    def get_alignments(self):
        return self._alignments
    
    @staticmethod
    def _get_info(text):
        lines = [line for line in text.split("\n")]
        info_dict = {}
        in_info_block = False
        for i, line in enumerate(lines):
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
    
    def _contact(self):
        contact = time.time()
        if (contact - BlastOnline._last_contact) < BlastOnline._contact_delay:
            self.violate_rule("The server was contacted too often")
        BlastOnline._last_contact = contact
    
    def _request(self):
        request = time.time()
        if (request - BlastOnline._last_request) < BlastOnline._request_delay:
            self.violate_rule("Too frequent BLAST requests")
        BlastOnline._last_request = request
