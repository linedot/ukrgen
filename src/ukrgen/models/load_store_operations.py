# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from typing import Self,Callable
from enum import Enum,auto
from ..components.tile import tile,scalar_tile
from ..components.operation import dimension_type

class lsc_reg_type(Enum):
    address = auto()
    data = auto()

class lsc_offset:
    """Offset for addresses

    :param reg_strides: for each GP register containing a stride, offset in 
                        multiples of the value stored in it
    :type reg_strides: list[int]
    :param vlen_strides: for each power of VLEN, offset in multiples of that value,
                         i.e [a*VLEN, b*VLEN*VLEN, c*VLEN*VLEN*VLEN, ...]
    :type vlen_strides: list[int]
    :param immoff: immediate offset in elements
    :type immoff: int

    """
    def __init__(self,
                 reg_strides : list[int],
                 vlen_strides : list[int],
                 immoff : int):
        self.reg_strides = reg_strides
        self.vlen_strides = vlen_strides
        self.immoff = immoff

    @classmethod
    def adjust_offlists(cls, one, other):
        l1 = len(one.reg_strides)
        l2 = len(other.reg_strides)
        if l1 > l2:
            other.reg_strides.extend([0 for i in range(l2,l1)])
        if l2 > l1:
            one.reg_strides.extend([0 for i in range(l1,l2)])

        l1 = len(one.vlen_strides)
        l2 = len(other.vlen_strides)
        if l1 > l2:
            other.vlen_strides.extend([0 for i in range(l2,l1)])
        if l2 > l1:
            one.vlen_strides.extend([0 for i in range(l1,l2)])

    def __add__(self, other : Self):
        if not isinstance(other, lsc_offset):
            raise NotImplementedError(f"can't add lsc_offset and {type(other)}")

        lsc_offset.adjust_offlists(self,other)

        return lsc_offset(
            reg_strides=[s+o for s,o in zip(self.reg_strides,other.reg_strides)],
            vlen_strides=[s+o for s,o in zip(self.vlen_strides,other.vlen_strides)],
            immoff=self.immoff+other.immoff)


    def __sub__(self, other : Self):
        if not isinstance(other, lsc_offset):
            raise NotImplementedError(f"can't subtract {type(other)} from lsc_offset")

        lsc_offset.adjust_offlists(self,other)

        return lsc_offset(
            reg_strides=[s-o for s,o in zip(self.reg_strides,other.reg_strides)],
            vlen_strides=[s-o for s,o in zip(self.vlen_strides,other.vlen_strides)],
            immoff=self.immoff-other.immoff)

    def allcompare(self, other : Self,
                   comparison : Callable[[Self,Self],bool]):
        
        lsc_offset.adjust_offlists(self,other)

        return all([comparison(s,o) for s,o in \
                   zip(self.reg_strides,other.reg_strides)]) and \
               all([comparison(s,o) for s,o in \
                   zip(self.vlen_strides, other.vlen_strides)]) and \
               (comparison(self.immoff,other.immoff))

    def anycompare(self, other : Self,
                   comparison : Callable[[Self,Self],bool]):
        
        lsc_offset.adjust_offlists(self,other)

        return any([comparison(s,o) for s,o in \
                   zip(self.reg_strides,other.reg_strides)]) or \
               any([comparison(s,o) for s,o in \
                   zip(self.vlen_strides, other.vlen_strides)]) or \
               (comparison(self.immoff,other.immoff))


    def __lt__(self, other : Self):
        # NOTE: this is sketchy. It's used in addr_resolver,
        #       perhaps this should be removed and a more elaborate check that
        #       takes the required offset to target into account should be 
        #       implemented?
        return self.allcompare(other, lambda s,o : s <= o) and \
               self.anycompare(other, lambda s,o : s < o)

    def __eq__(self, other : Self):
        return self.allcompare(other, lambda s,o : s == o)

    def __str__(self):
        result = ""
        if self.reg_strides:
            rstr = "+".join([f"{o}*stride{i}" for i,o in enumerate(self.reg_strides) if o != 0])
            if rstr:
                result += f"({rstr})+"
        if self.vlen_strides:
            vstr = "+".join([f"{o}*VLEN^{i+1}" for i,o in enumerate(self.vlen_strides) if o != 0])
            if vstr:
                result += f"({vstr})+"
        
        return f"{result}{self.immoff}"


class lsc_operation:
    def __init__(self,
                 tiles : list[tile],
                 indices: list[list[int]],
                 reads: list[int],
                 writes: list[int],
                 reg_types : list[lsc_reg_type]):
        self.tiles = tiles
        self.indices = indices
        self.reads = reads
        self.writes = writes
        self.reg_types = reg_types

class lsc_load(lsc_operation):
    def __init__(self, rtype_idx : int, res_idx : int, addr_idx : int, off : int, t : tile):
        self.off = off

        tiles = [scalar_tile, t]
        indices = [[rtype_idx, addr_idx], [rtype_idx, res_idx]]
        # read address
        reads = [0]
        # write resource
        writes = [1]

        reg_types = [lsc_reg_type.address, lsc_reg_type.data]

        super(lsc_load, self).__init__(tiles=tiles, indices=indices,
                                       reads=reads, writes=writes,
                                       reg_types=reg_types)

    @property
    def rtype_idx(self):
        return self.indices[0][0]

    @property
    def res_idx(self):
        return self.indices[1][1]

    @property
    def addr_idx(self):
        return self.indices[0][1]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        reg_chars = ['a','b','c']
        i = self.rtype_idx
        vladims = sum([1 if (d.dt == dimension_type.vla) else 0 for d in [self.t.dima,self.t.dimb] ])
        vlenstr = ""
        if 0 < vladims:
            vlenstr = "*VLEN"*vladims
        return f"{reg_chars[i]}{self.res_idx} <- LOAD {reg_chars[i]}a{self.addr_idx} + {self.off}{vlenstr}"

class lsc_store(lsc_operation):
    def __init__(self, rtype_idx : int, res_idx : int, addr_idx : int, off : int, t : tile):
        self.off = off

        tiles = [scalar_tile, t]
        indices = [[rtype_idx, addr_idx], [rtype_idx, res_idx]]
        # read resource
        reads = [0,1]
        # we write memory, not the address register
        writes = []

        reg_types = [lsc_reg_type.address, lsc_reg_type.data]

        super(lsc_store, self).__init__(tiles=tiles, indices=indices,
                                        reads=reads, writes=writes,
                                        reg_types=reg_types)

    @property
    def rtype_idx(self):
        return self.indices[0][0]

    @property
    def res_idx(self):
        return self.indices[1][1]

    @property
    def addr_idx(self):
        return self.indices[0][1]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        reg_chars = ['a','b','c']
        i = self.rtype_idx
        vladims = sum([1 if (d.dt == dimension_type.vla) else 0 for d in [self.t.dima,self.t.dimb] ])
        vlenstr = ""
        if 0 < vladims:
            vlenstr = "*VLEN"*vladims
        return f"{reg_chars[i]}a{self.addr_idx} + {self.off}{vlenstr} <- STORE {reg_chars[i]}{self.res_idx}"

class lsc_zero(lsc_operation):
    def __init__(self, rtype_idx : int, res_idx : int, t : tile):

        tiles = [t]
        indices = [[rtype_idx, res_idx]]
        reads = []
        # write resource
        writes = [0]

        reg_types = [lsc_reg_type.data]

        super(lsc_zero, self).__init__(tiles=tiles, indices=indices,
                                       reads=reads, writes=writes,
                                       reg_types=reg_types)
    @property
    def rtype_idx(self):
        return self.indices[0][0]

    @property
    def res_idx(self):
        return self.indices[0][1]

    @property
    def t(self):
        return self.tiles[0]

    def __str__(self):
        reg_chars = ['a','b','c']
        return f"{reg_chars[self.rtype_idx]}{self.res_idx} <- 0"

class lsc_addr_add(lsc_operation):
    def __init__(self, rtype_idx : int, addr_idx : int, off : int, t : tile):
        self.off = off

        # t is used for calculating address, isn't the tile the address resides in
        tiles = [scalar_tile, t]
        indices = [[rtype_idx, addr_idx]]
        # read addr
        reads = [0]
        # write addr
        writes = [0]
        
        reg_types = [lsc_reg_type.address]


        super(lsc_addr_add, self).__init__(tiles=tiles, indices=indices,
                                           reads=reads, writes=writes,
                                           reg_types=reg_types)

    @property
    def rtype_idx(self):
        return self.indices[0][0]

    @property
    def addr_idx(self):
        return self.indices[0][1]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        reg_chars = ['a','b','c']
        vladims = sum([1 if (d.dt == dimension_type.vla) else 0 for d in [self.t.dima,self.t.dimb] ])
        vlenstr = ""
        if 0 < vladims:
            vlenstr = "*VLEN"*vladims
        ar_str = f"{reg_chars[self.rtype_idx]}a{self.addr_idx}"
        return f"{ar_str} <- {ar_str} + {self.off}{vlenstr}"

class lsc_debugmsg(lsc_operation):
    def __init__(self, msg : str):
        super(lsc_debugmsg, self).__init__(tiles=[], indices=[], reads=[], writes=[], reg_types=[])
        self.msg = msg
    def __str__(self):
        return self.msg
    
class lsc_transformation(lsc_operation):
    def __init__(self, op : str,
                 res_indices : list[int],
                 sub_indices : list[int],
                 tiles : list[tile],
                 reads : list[int] = [0,1,2],
                 writes : list[int] = [2]):
        self.op = op
        indices = [[i,r,s] for i,(r,s) in enumerate(zip(res_indices,sub_indices))]
        reg_types = [lsc_reg_type.data for i in range(len(res_indices))]
        super(lsc_transformation, self).__init__(tiles=tiles, indices=indices,
                                                 reads=reads, writes=writes,
                                                 reg_types=reg_types)

    @property
    def res_indices(self):
        return [idxlist[1] for idxlist in self.indices]

    @property
    def sub_indices(self):
        return [idxlist[2] for idxlist in self.indices]

    def __str__(self):
        reg_chars = ['a','b','c']
        elsuf = lambda subidx : f".el[{subidx}]" if None != subidx else ""
        #TODO: subindices
        regstrs = [f"{reg_chars[i]}{r}{elsuf(subidx)}" for i,(r,subidx) in\
                enumerate(zip(self.res_indices,self.sub_indices))]
        out = f"{regstrs[-1]} <- {self.op}("
        out += ", ".join(regstrs)
        out += ")"
        return out
