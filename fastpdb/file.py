import biotite
from .fastpdb import parse_coord_single_model, parse_coord_multi_model

class PDBFile(biotite.TextFile):

    def get_coord(self, model=None):
        if model is None:
            coord = parse_coord_multi_model(self.lines)
        else:
            coord = parse_coord_single_model(self.lines, model)
        return coord