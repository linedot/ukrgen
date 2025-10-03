# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import string
from typing import Self,Callable
from enum import Enum,auto
from ..components.tile import tile,scalar_tile
from ..components.operation import dimension_type

class lsc_reg_type(Enum):
    address = auto()
    data = auto()
    offset = auto()
    value = auto()

# TODO: fractional offsets
class fraction:
    def __init__(self, nom : int, div : int = 1):
        self.nom = nom
        self.div = div


class stridexvlen:
    """
    encodes a multiplicative chain of strides and vlens
    """
    def __init__(self, stride_ids : set[int], vlen_ids : set[int]):
        self.stride_ids = stride_ids
        self.vlen_ids = vlen_ids

    def __eq__(self, other):
        return self.stride_ids == other.stride_ids and \
               self.vlen_ids == other.vlen_ids
    def __str__(self):
        sstr = ""
        vstr = ""
        if self.stride_ids:
            sstr = "*".join([f"stride{id}" for id in self.stride_ids])
            sstr += "*"
        if self.vlen_ids:
            vstr =  "*".join([f"VLEN{id}" for id in self.vlen_ids])


        return f"{sstr}{vstr}"
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return hash(str(self))

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
                 sxv_strides : dict[stridexvlen,int],
                 reg_strides : list[int],
                 vlen_strides : list[int],
                 immoff : int):
        self.sxv_strides = sxv_strides
        self.reg_strides = reg_strides
        self.vlen_strides = vlen_strides
        self.immoff = immoff

    @classmethod
    def adjust_offlists(cls, one, other):

        for key in one.sxv_strides.keys():
            if key not in other.sxv_strides:
                other.sxv_strides[key] = 0

        for key in other.sxv_strides.keys():
            if key not in one.sxv_strides:
                one.sxv_strides[key] = 0


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


    def is_scalar(self) -> bool:
        result = False
        if self.immoff != 0:
            result = True
        if any([v != 0 for v in self.vlen_strides]):
            result = False
        if any([v != 0 for k,v in self.sxv_strides.items()]):
            result = False
        if any([v != 0 for v in self.reg_strides]):
            result = False

        return result

    def is_vector(self) -> bool:
        result = False
        if 1 == sum([1 for v in self.vlen_strides if 0 != v]):
            result = True

        if any([v != 0 for k,v in self.sxv_strides.items()]):
            result = False
        if any([v != 0 for v in self.reg_strides]):
            result = False
        if self.immoff != 0:
            result = False

        return result

    def __add__(self, other : Self):
        if not isinstance(other, lsc_offset):
            raise NotImplementedError(f"can't add lsc_offset and {type(other)}")

        lsc_offset.adjust_offlists(self,other)

        return lsc_offset(
            sxv_strides={key : self.sxv_strides[key]+other.sxv_strides[key]\
                    for key in self.sxv_strides.keys()},
            reg_strides=[s+o for s,o in zip(self.reg_strides,other.reg_strides)],
            vlen_strides=[s+o for s,o in zip(self.vlen_strides,other.vlen_strides)],
            immoff=self.immoff+other.immoff)


    def __sub__(self, other : Self):
        if not isinstance(other, lsc_offset):
            raise NotImplementedError(f"can't subtract {type(other)} from lsc_offset")

        lsc_offset.adjust_offlists(self,other)

        return lsc_offset(
            sxv_strides={key : self.sxv_strides[key]-other.sxv_strides[key]\
                    for key in self.sxv_strides.keys()},
            reg_strides=[s-o for s,o in zip(self.reg_strides,other.reg_strides)],
            vlen_strides=[s-o for s,o in zip(self.vlen_strides,other.vlen_strides)],
            immoff=self.immoff-other.immoff)

    def colin(self, other : Self):
        """
        Return the "colinear" part of this offset with the other offset
        All different strides/vlens/strides*vlens are treated as independent
        For example if this offset is 2*stride0*vlen1 + 3*vlen1 and
        the other offset is 1*vlen1, this returns 3*vlen1

        :param other: The other offset
        :return: Part of this offset that is colinear with other
        """
        lsc_offset.adjust_offlists(self,other)

        result = lsc_offset.zero_offset()

        for key in self.sxv_strides:
            if (0 != self.sxv_strides[key]) and (0 != other.sxv_strides[key]):
                result.sxv_strides[key] = self.sxv_strides[key]

        result.reg_strides = [s if (s!=0 and o!=0) else 0 for s,o in \
                zip(self.reg_strides,other.reg_strides)]

        result.vlen_strides = [s if (s!=0 and o!=0) else 0 for s,o in \
                zip(self.vlen_strides,other.vlen_strides)]

        if (self.immoff != 0) and (other.immoff != 0):
            result.immoff = self.immoff

        return result

    def allcompare(self, other : Self,
                   comparison : Callable[[Self,Self],bool]):
        
        lsc_offset.adjust_offlists(self,other)

        return all([comparison(self.sxv_strides[key],other.sxv_strides[key]) \
                for key in self.sxv_strides]) and \
               all([comparison(s,o) for s,o in \
                   zip(self.reg_strides,other.reg_strides)]) and \
               all([comparison(s,o) for s,o in \
                   zip(self.vlen_strides, other.vlen_strides)]) and \
               (comparison(self.immoff,other.immoff))

    def anycompare(self, other : Self,
                   comparison : Callable[[Self,Self],bool]):
        
        lsc_offset.adjust_offlists(self,other)

        return any([comparison(self.sxv_strides[key],other.sxv_strides[key]) \
                for key in self.sxv_strides]) or \
               any([comparison(s,o) for s,o in \
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
        if self.sxv_strides:
            rstr = "+".join([f"{val}*{key}" for key,val \
                    in self.sxv_strides.items() if val != 0])
            if rstr:
                result += f"({rstr})+"
        if self.reg_strides:
            rstr = "+".join([f"{o}*stride{i}" for i,o in enumerate(self.reg_strides) if o != 0])
            if rstr:
                result += f"({rstr})+"
        if self.vlen_strides:
            vstr = "+".join([f"{o}*VLEN{i+1}" for i,o in enumerate(self.vlen_strides) if o != 0])
            if vstr:
                result += f"({vstr})+"
        
        return f"{result}{self.immoff}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.__str__())

    @classmethod
    def zero_offset(cls):
        return cls(dict(),[],[],0)

    @classmethod
    def vo(cls, vid : int, value : int):
        """
        1D Vector offset
        """
        voffs = [0 for i in range(vid+1)]
        voffs[vid] = value
        return cls(dict(),[],voffs,0)

    @classmethod
    def to(cls, vid1 : int, vid2 : int, value1 : int, value2):
        """
        2D Tile offset
        """
        voffs = [0 for i in range(max(vid1,vid2)+1)]
        voffs[vid1] = value1
        voffs[vid2] = value2
        return cls(dict(),[],voffs,0)

    @classmethod
    def so(cls, value : int):
        """
        scalar offset
        """
        return cls(dict(),[],[],value)

class lsc_operation:
    def __init__(self,
                 tiles : list[tile],
                 indices: list[tuple[str,list[int]]],
                 reads: list[int],
                 writes: list[int],
                 reg_types : list[lsc_reg_type]):
        for t in tiles:
            if not isinstance(t, tile):
                raise ValueError("non-tile in list of tiles")
        self.tiles = tiles
        for idx in indices:
            if not isinstance(idx[0],str) or \
               not isinstance(idx[1],list):
               raise ValueError(f"invalid index: {idx}")
            for i in idx[1]:
               if not isinstance(i,int):
                   raise ValueError(f"invalid index: {idx}")
        self.indices = indices
        self.reads = reads
        self.writes = writes
        self.reg_types = reg_types


class ldst_modifier(Enum):
    bcast1 = auto()

class lsc_load(lsc_operation):
    def __init__(self,
                 component : str,
                 res_idx : int,
                 addr_idx : int,
                 off : lsc_offset,
                 stride : lsc_offset,
                 t : tile,
                 mods : set[ldst_modifier]):
        self.off = off
        self.stride = stride
        self.mods = mods

        tiles = [scalar_tile, t]
        indices = [(component, [addr_idx]), (component, [res_idx])]
        # read address
        reads = [0]
        # write resource
        writes = [1]

        reg_types = [lsc_reg_type.address, lsc_reg_type.data]

        super().__init__(tiles=tiles, indices=indices,
                                       reads=reads, writes=writes,
                                       reg_types=reg_types)

    @property
    def component(self):
        return self.indices[0][0]

    @property
    def res_idx(self):
        return self.indices[1][1][0]

    @property
    def addr_idx(self):
        return self.indices[0][1][0]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        cr = self.indices[1][0]
        ca = self.indices[0][0]
        return f"{cr}{self.res_idx} <- LOAD {ca}a{self.addr_idx} + {self.off}"

class lsc_store(lsc_operation):
    def __init__(self,
                 component : str,
                 res_idx : int,
                 addr_idx : int,
                 off : lsc_offset,
                 stride : lsc_offset,
                 t : tile,
                 mods : set[ldst_modifier]):
        self.off = off
        self.stride = stride
        self.mods = mods

        tiles = [scalar_tile, t]
        indices = [(component, [addr_idx]), (component, [res_idx])]
        # read resource
        reads = [0,1]
        # we write memory, not the address register
        writes = []

        reg_types = [lsc_reg_type.address, lsc_reg_type.data]

        super().__init__(tiles=tiles, indices=indices,
                                        reads=reads, writes=writes,
                                        reg_types=reg_types)

    @property
    def component(self):
        return self.indices[0][0]

    @property
    def res_idx(self):
        return self.indices[1][1][0]

    @property
    def addr_idx(self):
        return self.indices[0][1][0]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        cr = self.indices[1][0]
        ca = self.indices[0][0]
        return f"{ca}a{self.addr_idx} + {self.off} <- STORE {cr}{self.res_idx}"

class lsc_zero(lsc_operation):
    def __init__(self, component : str, res_idx : int, t : tile):

        tiles = [t]
        indices = [(component, [res_idx])]
        reads = []
        # write resource
        writes = [0]

        reg_types = [lsc_reg_type.data]

        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)
    @property
    def component(self):
        return self.indices[0][0]

    @property
    def res_idx(self):
        return self.indices[0][1][0]

    @property
    def t(self):
        return self.tiles[0]

    def __str__(self):
        return f"{self.component}{self.res_idx} <- 0"

class lsc_addr_add(lsc_operation):
    def __init__(self, component : str, addr_idx : int, off : lsc_offset, t : tile):
        if not isinstance(off, lsc_offset):
            raise ValueError(f"offset {off} is not an lsc_offset")
        self.off = off

        # t is used for calculating address, isn't the tile the address resides in
        tiles = [scalar_tile, t]
        indices = [(component, [addr_idx])]
        # read addr
        reads = [0]
        # write addr
        writes = [0]
        
        reg_types = [lsc_reg_type.address]


        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)

    @property
    def component(self):
        return self.indices[0][0]

    @property
    def addr_idx(self):
        return self.indices[0][1][0]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        ar_str = f"{self.component}a{self.addr_idx}"
        return f"{ar_str} <- {ar_str} + {self.off}"

class lsc_add_val_off(lsc_operation):
    def __init__(self, valname : str, off : lsc_offset):
        if not isinstance(off, lsc_offset):
            raise ValueError(f"offset {off} is not an lsc_offset")
        self.off = off
        self.valname = valname

        # t is used for calculating address, isn't the tile the address resides in
        tiles = [scalar_tile]
        indices = []
        # read addr
        reads = [0]
        # write addr
        writes = [0]
        
        reg_types = [lsc_reg_type.value]


        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)

    def __str__(self):
        return f"{self.valname} <- {self.valname} + {self.off}"

class lsc_debugmsg(lsc_operation):
    def __init__(self, msg : str):
        super(lsc_debugmsg, self).__init__(tiles=[], indices=[], reads=[], writes=[], reg_types=[])
        self.msg = msg
    def __str__(self):
        return self.msg


class tr_modifier(Enum):
    np = auto()
    
class lsc_transformation(lsc_operation):
    def __init__(self, op : str,
                 res_indices : list[tuple[str,int]],
                 sub_indices : list[int],
                 tiles : list[tile],
                 reads : list[int] = [0,1,2],
                 writes : list[int] = [2]):

        opsplit = op.split(":")
        opstr = opsplit[0]
        self.op = opstr
        self.mods = []
        if 1 < len(opsplit):
            modlist = opsplit[1].split(",")        
            self.mods = [tr_modifier[m] for m in modlist]
        indices = [(key,[r,s]) if s is not None else (key,[r]) for (key,r),s in zip(res_indices,sub_indices)]
        reg_types = [lsc_reg_type.data for i in range(len(res_indices))]
        super(lsc_transformation, self).__init__(tiles=tiles, indices=indices,
                                                 reads=reads, writes=writes,
                                                 reg_types=reg_types)


    @property
    def modifiers(self):
        return self.mods

    @property 
    def components(self):
        return [index[0] for index in self.indices]

    @property
    def res_indices(self):
        return [index[1][0] for index in self.indices]

    @property
    def sub_indices(self):
        return [None if len(index[1]) == 1 else index[1][1] for index in self.indices]

    def __str__(self):
        elsuf = lambda subidx : f".el[{subidx}]" if None != subidx else ""
        #TODO: subindices
        regstrs = [f"{c}{ridx}{elsuf(subidx)}" for (c,_),ridx,subidx in\
                zip(self.indices,self.res_indices,self.sub_indices)]

        modstr = [str(m).replace('tr_modifier.','') for m in self.mods]
        if modstr:
            modstr = ":"+f"{{{','.join(modstr)}}}"
        else:
            modstr = ""
        out = f"{regstrs[-1]} <- {self.op}{modstr}("
        out += ", ".join(regstrs)
        out += ")"
        return out
