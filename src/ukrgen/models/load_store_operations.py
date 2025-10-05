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

from .lsc.offset import lsc_offset
from .lsc.index import lsc_reg_index 

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

class lsc_operation:
    def __init__(self,
                 tiles : list[tile],
                 indices: list[lsc_reg_index],
                 reads: list[int],
                 writes: list[int],
                 reg_types : list[lsc_reg_type]):
        for t in tiles:
            if not isinstance(t, tile):
                raise ValueError("non-tile in list of tiles")
        self.tiles = tiles
        for idx in indices:
            if not isinstance(idx,lsc_reg_index):
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
        indices = [lsc_reg_index(component, [addr_idx]),
                   lsc_reg_index(component, [res_idx])]
        # read address
        reads = [0]
        # write resource
        writes = [1]

        reg_types = [lsc_reg_type.address, lsc_reg_type.data]

        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)

    @property
    def res_idx(self):
        return self.indices[1]

    @property
    def addr_idx(self):
        return self.indices[0]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        return f"RES:{self.res_idx} <- LOAD ADDR:{self.addr_idx} + {self.off}"

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
        indices = [lsc_reg_index(component, [addr_idx]),
                   lsc_reg_index(component, [res_idx])]
        # read resource
        reads = [0,1]
        # we write memory, not the address register
        writes = []

        reg_types = [lsc_reg_type.address, lsc_reg_type.data]

        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)
    @property
    def res_idx(self):
        return self.indices[1]

    @property
    def addr_idx(self):
        return self.indices[0]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        return f"ADDR:{self.addr_idx} + {self.off} <- STORE RES:{self.res_idx}"

class lsc_zero(lsc_operation):
    def __init__(self, component : str, res_idx : int, t : tile):

        tiles = [t]
        indices = [lsc_reg_index(component, [res_idx])]
        reads = []
        # write resource
        writes = [0]

        reg_types = [lsc_reg_type.data]

        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)

    @property
    def res_idx(self):
        return self.indices[0]

    @property
    def t(self):
        return self.tiles[0]

    def __str__(self):
        return f"RES:{self.res_idx} <- 0"

class lsc_addr_add(lsc_operation):
    def __init__(self, component : str, addr_idx : int, off : lsc_offset, t : tile):
        if not isinstance(off, lsc_offset):
            raise ValueError(f"offset {off} is not an lsc_offset")
        self.off = off

        # t is used for calculating address, isn't the tile the address resides in
        tiles = [scalar_tile, t]
        indices = [lsc_reg_index(component, [addr_idx])]
        # read addr
        reads = [0]
        # write addr
        writes = [0]
        
        reg_types = [lsc_reg_type.address]


        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)

    @property
    def addr_idx(self):
        return self.indices[0]

    @property
    def t(self):
        return self.tiles[1]

    def __str__(self):
        ar_str = f"ADDR:{self.addr_idx}"
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
                 res_indices : list[lsc_reg_index],
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
        reg_types = [lsc_reg_type.data for i in range(len(res_indices))]
        super(lsc_transformation, self).__init__(tiles=tiles, indices=res_indices,
                                                 reads=reads, writes=writes,
                                                 reg_types=reg_types)


    @property
    def modifiers(self):
        return self.mods

    def __str__(self):
        regstrs = [f"RES:{idx}" for idx in self.indices]

        modstr = [str(m).replace('tr_modifier.','') for m in self.mods]
        if modstr:
            modstr = ":"+f"{{{','.join(modstr)}}}"
        else:
            modstr = ""
        out = f"{regstrs[-1]} <- {self.op}{modstr}("
        out += ", ".join(regstrs)
        out += ")"
        return out
