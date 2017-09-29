# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ...sequence.align.align import Alignment

__all__ = ["BlastAlignment"]

class BlastAlignment(Alignment):
    
    def __init__(self, sequences, trace, score, e_value,
                 query_interval, hit_interval, hit_id, hit_definition):
        super().__init__(sequences, trace, score)
        self.e_value = e_value
        self.query_interval = query_interval
        self.hit_interval = hit_interval
        self.hit_id = hit_id
        self.hit_definition = hit_definition