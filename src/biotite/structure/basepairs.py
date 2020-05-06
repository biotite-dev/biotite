import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack, coord, stack
from .filter import filter_nucleotides, _filter_atom_type
from .celllist import CellList
from .hbond import hbond
from .util import distance
from itertools import chain

def get_proximate_basepair_candidates(array, max_cutoff = 15, min_cutoff = 9):
#TODO: Make Cutoffs Argument
    array = array[filter_nucleotides(array) 
                    & _filter_atom_type(array, ["C1'", "C1*"])]
    
    cell_list = CellList(array, 6.0)
    basepair_candidates = []
    
    for atom in array:
        candidates = cell_list.get_atoms(atom.coord, max_cutoff)
        atom_id = [atom.res_id, atom.chain_id]
        
        for candidate in candidates:
            partner_id = [array[int(candidate)].res_id,
                        array[int(candidate)].chain_id]
            if ( (distance(array[int(candidate)].coord, atom.coord) > min_cutoff) & 
                    ((partner_id + atom_id) not in basepair_candidates)):
                basepair_candidates.append(atom_id + partner_id)
    
    return basepair_candidates
