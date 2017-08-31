# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ..application import Application, evaluation
from ..webapp import WebApp
from ...sequence.io.fasta import FastaFile
import abc
import time
import requests
from abc import abstractmethod

__all__ = ["BlastOnline", "NucleotideBlastOnline", "ProteinBlastOnline"]


contact_delay = 3
request_delay = 60

ncbi_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
#ncbi_url = "http://www.ebi.ac.uk/Tools/services/rest/ncbiblast"

class BlastOnline(WebApp, metaclass=abc.ABCMeta):
    
    def __init__(self, program, query, database="nr",
                 app_url=ncbi_url, obey_rules=True, mail=None):
        super().__init__(app_url, obey_rules)
        self._program = program
        if isinstance(query, str) and query.endswith((".fa",".fst",".fasta")):
                file = FastaFile()
                file.read(self._query)
                param["QUERY"] = str(file.get_sequence())
        else:
            param["QUERY"] = str(self._query)
        self._query = query
        self._database = database
        self._gap_openining = None
        self._gap_extension = None
        self._word_size = None
        self._expect_value = None
        
        self._mail=mail
        self._rid = None
        self._is_finished = False
    
    def _set_max_evalue(self, value):
        self._expect_value = value
    
    def set_gap_penalty(self, opening, extension):
        self._gap_openining = opening
        self._gap_extension = extension
        
    def set_word_size(self, size):
        self._word_size = size
    
    def run(self):
        super().run()
        param_dict = self._get_param_dict()
        request = requests.get(self.app_url(), params=param_dict)
        info_dict = BlastOnline._get_info(request.text)
        self._rid = info_dict["RID"]
    
    def join(self):
        if not self.is_started():
            raise ApplicationError("The application run has not been "
                                   "not started yet")
        while not self.is_finished():
            # NCBI requires a 3 second delay between requests
            time.sleep(contact_delay)
        super().join()
    
    def is_finished(self):
        if not self.is_started():
            raise ApplicationError("The application run has not been "
                                   "not started yet")
        if self._is_finished:
            # If it is already known that the application finished,
            # return true
            return True
        else:
            # Otherwise check again
            data_dict = {"FORMAT_OBJECT" : "SearchInfo",
                         "RID"           : self._rid,
                         "CMD"           : "Get"}
            request = requests.get(self.app_url(), params=data_dict)
            info_dict = BlastOnline._get_info(request.text)
            print(info_dict)
            if info_dict["Status"] == "READY":
                self._is_finished = True
            return self._is_finished
    
    @evaluation
    def get_result(self, format):
        param_dict = {}
        param_dict["tool"] = "BiopythonClient"
        if self._mail is not None:
            param_dict["email"] = self._mail
        param_dict["CMD"] = "Get"
        param_dict["RID"] = self._rid
        param_dict["FORMAT_TYPE"] = format
        request = requests.get(self.app_url(), params=param_dict)
        print(request.text)
    
    @abc.abstractmethod
    def _get_param_dict(self):
        param = {}
        param["tool"] = "BiopythonClient"
        if self._mail is not None:
            param["email"] = self._mail
        param["CMD"] = "Put"
        param["PROGRAM"] = self._program
        if ( isinstance(self._query, str)
             and self._query.endswith((".fa",".fst",".fasta")) ):
                file = FastaFile()
                file.read(self._query)
                param["QUERY"] = str(file.get_sequence())
        else:
            param["QUERY"] = str(self._query)
        param["DATABASE"] = self._database
        if self._expect_value is not None:
            param["EXPECT"] = self._expect_value
        if self._gap_openining is not None and self._gap_extension is not None:
            param["GAPCOSTS"] = "{:d} {:d}".format(self._gap_openining,
                                                      self._gap_extension)
        if self._word_size is not None:
            param["WORD_SIZE"] = self._word_size
        return param
    
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
    

class NucleotideBlastOnline(BlastOnline):
    
    def __init__(self, program, query, database="nr",
                 app_url=ncbi_url, obey_rules=True, mail=None):
        if program not in ["blastn", "megablast"]:
            raise ValueError("'{:}' is not a valid nucleotide BLAST program"
                             .format(program))
        super().__init__(program, query, database, app_url, obey_rules, mail)
        self._reward = None
        self._penalty = None
    
    def set_match_reward(self, reward):
        self._reward = score
    
    def set_mismatch_penalty(self, penalty):
        self._penalty = penalty
    
    def _get_param_dict(self):
        put_dict = super()._get_param_dict()
        if self._reward is not None:
            put_dict["NUCL_REWARD"] = self._reward
        if self._penalty is not None:
            put_dict["NUCL_PENALTY"] = self._penalty
        return put_dict


class ProteinBlastOnline(BlastOnline):
    
    def __init__(self, program, query, database="nr",
                 app_url=ncbi_url, obey_rules=True, mail=None):
        if program not in ["blastp", "blastx", "tblastn", "tblastx"]:
            raise ValueError("'{:}' is not a valid protein BLAST program"
                             .format(program))
        super().__init__(program, query, database, app_url, obey_rules, mail)
        self._matrix = None
        self._threshold = None
    
    def set_substitution_matrix(self, matrix_name):
        self._matrix = matrix_name.upper()
    
    def set_threshold(self, threshold):
        self._threshold = threshold
    
    def _get_param_dict(self):
        put_dict = super()._get_param_dict()
        if self._matrix is not None:
            put_dict["MATRIX"] = self._matrix
        if self._threshold is not None:
            put_dict["THRESHOLD"] = self._threshold
        return put_dict
    
