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
from .lsc.reg import lsc_reg_type


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

        self.properties : dict[str,int|str] = dict()

    def add_property(self, name : str, value : int|str):
        self.properties[name] = value


def can_reorder(first : lsc_operation, second : lsc_operation) -> bool:
    """
    Checks if two lsc ops have data accesses that require them to be in
    order [first,second] and returns False if the order is required and
    True if the order is not required
    """

    # RAW
    if any([(first.indices[widx] == second.indices[ridx]) \
            and (first.reg_types[widx] == second.reg_types[ridx])\
            for widx in first.writes \
            for ridx in second.reads]):
        return False
    # WAR
    if any([(first.indices[ridx] == second.indices[widx]) \
            and (first.reg_types[ridx] == second.reg_types[widx])\
            for ridx in first.reads \
            for widx in second.writes]):
        return False

    # RAR is not a data hazard
    # WAW is probably also not a hazard - there should be a RAW or WAR somewhere
    # between two ops or the code is malformed

    return True


class ldst_modifier(Enum):
    bcast1 = auto()
    lane = auto()
    postinc = auto()

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
        offsetstr = f" + {self.off}"
        resstr = f"RES:{self.res_idx}"
        opstr = "LOAD"
        if ldst_modifier.bcast1 in self.mods:
            opstr = "BCAST"
        if ldst_modifier.lane in self.mods:
            resstr += f"[lane:{self.properties['lane']}]"
        if ldst_modifier.postinc in self.mods:
            offsetstr = f"; ADDR:{self.addr_idx} <- ADDR:{self.addr_idx} + {self.off}"
        return f"{resstr} <- {opstr} ADDR:{self.addr_idx}{offsetstr}"

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
        offsetstr = f" + {self.off}"
        poststr = ""
        resstr = f"RES:{self.res_idx}"
        if ldst_modifier.lane in self.mods:
            resstr += f"[lane:{self.properties['lane']}]"
        if ldst_modifier.postinc in self.mods:
            offsetstr = ""
            poststr = f"; ADDR:{self.addr_idx} <- ADDR:{self.addr_idx} + {self.off}"
        return f"ADDR:{self.addr_idx}{offsetstr} <- STORE {resstr}{poststr}"

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

        #TODO: This is needed because registers might get replaced (for example by the
        #      minreguse_scheduler) and we need component assignments to actual data.
        #      In other I/O ops we can just use component of addr_idx. Possibly come up
        #      with a better system?
        self.data_components = [idx.component for idx in res_indices]

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
