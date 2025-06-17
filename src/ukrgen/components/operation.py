# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .tile import tile,dimension_type

class operation:
    def __init__(self,
                 tiles : list[tile],
                 tile_offsets : list[list[int]],
                 subindices : list[int],
                 opstr : str,
                 tile_strs : list[str] = ['A','B','C']):

        self.tiles = tiles
        self.tile_offsets = tile_offsets
        self.subindices = subindices
        self.opstr = opstr
        self.tile_strs = tile_strs

    def __str__(self) -> str:

        idxstr = lambda dima,dimb,sidx : "" if dima.size==dimb.size else f".el[{sidx}]" if dima.size > dimb.size else f"+{sidx}"
        vlensuf = lambda d : "*VLEN" if d.dt == dimension_type.vla else ""

        num_tiles = len(self.tiles)

        idx_str_prefixes = []
        for i in range(num_tiles):
            offsets = self.tile_offsets[i]
            tile = self.tiles[i]
            idx_str_prefixes.append(f"{offsets[0]}{vlensuf(tile.dima)}")
            idx_str_prefixes.append(f"{offsets[1]}{vlensuf(tile.dimb)}")
                
        first_idx_str = f"{idx_str_prefixes[0]}{idxstr(self.tiles[0].dima,self.tiles[-1].dima,self.subindices[0])}"
        last_idx_str = f"{idx_str_prefixes[-3]}{idxstr(self.tiles[-2].dimb,self.tiles[-1].dimb,self.subindices[1])}"

        dest_idx_str_full = (f"({idx_str_prefixes[-2]}{idxstr(self.tiles[-1].dima,self.tiles[0].dima,self.subindices[0])},"
                             f"{idx_str_prefixes[-1]}{idxstr(self.tiles[-1].dimb,self.tiles[-2].dimb,self.subindices[1])})")

        idx_strs = [first_idx_str]
        num_middle_tiles = num_tiles-2
        for i in range(num_middle_tiles*2):
            this_dim = self.tiles[0+i//2].dimb
            next_dim = self.tiles[1+i//2].dima
            if 1 == (i % 2):
                this_dim,next_dim = next_dim,this_dim
            subidx = self.subindices[2+i//2]
            idx_strs.append(f"{idx_str_prefixes[i+1]}{idxstr(this_dim,next_dim,subidx)}")

        idx_strs.append(last_idx_str)

        idx_strs = [f"{ts}({off1},{off2})" for ts,off1,off2 in zip(self.tile_strs[:-1],idx_strs[0::2],idx_strs[1::2])]

        result = f"{self.tile_strs[-1]}{dest_idx_str_full} <- {self.opstr}({','.join(idx_strs)},{self.tile_strs[-1]}{dest_idx_str_full})"

        return result
