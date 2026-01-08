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

        fits_in = lambda t1,t2 : 0 not in t1.tiling_of(t2)

        idxstr = lambda dima,dimb,sidx : "" if dima.size==dimb.size else f".el[{sidx}]" if dima.size > dimb.size else f"+{sidx}"
        vlensuf = lambda d : "*VLEN" if d.dt == dimension_type.vla else ""

        
        # First index of first tile shares dimension with the first index of last tile
        # Last index of second-to-last tile shares dimension with the last index
        # of last tile
        
        # some cases that need to be mapped out correctly (either v > 1 or v = VLA):
        # FMA  A(v,1) B(1,1) C(v,1) -> no subidx or add
        # FMA  A(1,1) B(1,v) C(1,v) -> no subidx or add
        # FMA  A(v,1) B(1,v) C(v,1) -> subindex in B, add to second dim in C
        # FMA  A(v,1) B(1,v) C(1,v) -> subindex in A, add to first dim in C
        # DOTA A(1,v) B(v,1) C(1,1) -> no subidx or add
        # DOTA A(1,v) B(v,1) C(v,1) -> subindex in C, add to second dim of B
        #          "         C(1,v) -> subindex in C, add to first dim of A
        # OPA  A(v,1) B(1,v) C(v,v) -> no subidx or add
        # MMA  A(v,v) B(v,v) C(v,v) -> no subidx or add
        #
        # TODO: more than one value > 1 or VLA vlen, multiples of VLA vlens.
        #
        # visualize subindex with ".el[subidx]", for addition just increment index

        first_input_tiling = self.tiles[0].tiling_of(self.tiles[-1])
        last_input_tiling = self.tiles[-2].tiling_of(self.tiles[-1])

        # expected values:
        # FMA  A(v,1) B(1,1) C(v,1) -> first (1,1), last (v,1)
        # FMA  A(1,1) B(1,v) C(1,v) -> first (1,v), last (1,1)
        # FMA  A(v,1) B(1,v) C(v,1) -> first (1,1), last (v,0)
        # FMA  A(v,1) B(1,v) C(1,v) -> first (0,v), last (1,1)
        # DOTA A(1,v) B(v,1) C(1,1) -> first (1,0), last (0,1)
        # DOTA A(1,v) B(v,1) C(v,1) -> first (v,0), last (1,1)
        #       "            C(1,v) -> first (1,1), last (0,v) 
        # OPA  A(v,1) B(1,v) C(v,v) -> first (1,v), last (v,1)
        # MMA  A(v,v) B(v,v) C(v,v) -> first (1,1), last (1,1)


        # Initial thoughts on > 3 components:
        # indices:
        # [m,n,k,l,o,p,r,s,t]
        # A(m,k) B(k,n) + C(m,n)
        # A(m,k) B(k,l) C(l,n) + D(m,n)
        # [...]
        # A(m,k) B(k,l) C(l,o) D(o,p) E(p,r) F(r,s) G(s,t) H(t,n) + I(m,n)

        subindices = [["",""] for _ in self.tiles]
        offset_incs = [[0,0] for _ in self.tiles]

        if 0 in first_input_tiling and 0 not in last_input_tiling:
            if 0 == first_input_tiling[0] and 0 != first_input_tiling[1]:
                # A subidx, C add
                subindices[0][0] = f".el[{self.subindices[0]}]"
                offset_incs[-1][0] = self.subindices[0]
                pass
            elif 0 == first_input_tiling[1] and 0 != first_input_tiling[0]:
                # A add, C subidx
                subindices[-1][0] = f".el[{self.subindices[0]}]"
                offset_incs[0][0] = self.subindices[0]
                pass
            else:
                raise NotImplementedError("Can't handle this tiling")
        elif 0 not in first_input_tiling and 0 in last_input_tiling:
            if 0 == last_input_tiling[0] and 0 != last_input_tiling[1]:
                # C subidx, B add
                subindices[-1][1] = f".el[{self.subindices[1]}]"
                offset_incs[-2][1] = self.subindices[1]
                pass
            elif 0 == last_input_tiling[1] and 0 != last_input_tiling[0]:
                # B subidx, C add
                subindices[-2][1] = f".el[{self.subindices[1]}]"
                offset_incs[-1][1] = self.subindices[1]
                pass
            else:
                raise NotImplementedError("Can't handle this tiling")
        elif (0 not in first_input_tiling and 0 not in last_input_tiling) or \
             (0 in first_input_tiling and 0 in last_input_tiling):
            pass
        else:
            raise NotImplementedError("Can't handle this tiling")


        # build main indices
        #idx_str_prefixes = []
        #for i in range(num_tiles):
        #    offsets = self.tile_offsets[i]
        #    tile = self.tiles[i]
        #    idx_str_prefixes.append(f"{offsets[0]}{vlensuf(tile.dima)}")
        #    idx_str_prefixes.append(f"{offsets[1]}{vlensuf(tile.dimb)}")

        #first_idx_str = f"{idx_str_prefixes[0]}{idxstr(self.tiles[0].dima,self.tiles[-1].dima,self.subindices[0])}"
        #last_idx_str = f"{idx_str_prefixes[-3]}{idxstr(self.tiles[-2].dimb,self.tiles[-1].dimb,self.subindices[1])}"

        #dest_idx_str_full = (f"({idx_str_prefixes[-2]}{idxstr(self.tiles[-1].dima,self.tiles[0].dima,self.subindices[0])},"
        #                     f"{idx_str_prefixes[-1]}{idxstr(self.tiles[-1].dimb,self.tiles[-2].dimb,self.subindices[1])})")

        #idx_strs = [first_idx_str]
        #num_middle_tiles = num_tiles-2
        #for i in range(num_middle_tiles*2):
        #    this_dim = self.tiles[0+i//2].dimb
        #    next_dim = self.tiles[1+i//2].dima
        #    if 1 == (i % 2):
        #        this_dim,next_dim = next_dim,this_dim
        #    subidx = self.subindices[2+i//2]
        #    idx_strs.append(f"{idx_str_prefixes[i+1]}{idxstr(this_dim,next_dim,subidx)}")

        #idx_strs.append(last_idx_str)

        num_tiles = len(self.tiles)

        idx_strs = []
        for i in range(num_tiles):
            offsets = self.tile_offsets[i]
            offsets[0] += offset_incs[i][0]
            offsets[1] += offset_incs[i][1]
            tile = self.tiles[i]
            idx_strs.append(
                    f"{offsets[0]}{vlensuf(tile.dima)}{subindices[i][0]}")
            idx_strs.append(
                    f"{offsets[1]}{vlensuf(tile.dimb)}{subindices[i][1]}")

        idx_strs = [f"{ts}({off1},{off2})" for ts,off1,off2 in zip(self.tile_strs,idx_strs[0::2],idx_strs[1::2])]

        result = f"{idx_strs[-1]} <- {self.opstr}({','.join(idx_strs)})"

        return result
