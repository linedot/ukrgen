from .tile import tile

class operation:
    def __init__(self,
                 tiles : list[tile],
                 tile_offsets : list[list[int]],
                 subindices : list[int],
                 opstr : str):

        self.tiles = tiles
        self.tile_offsets = tile_offsets
        self.subindices = subindices
        self.opstr = opstr
