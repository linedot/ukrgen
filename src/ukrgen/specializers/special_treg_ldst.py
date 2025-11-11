# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

"""
Structures for abstracting storing/loading tile registers through
inserting/extracting vector registers. While this is an abstraction,
this is very SME specific.
"""

from copy import deepcopy

from asmgen.registers import asm_data_type as adt, adt_size

from ..components.tile import tile, scalar_tile, scalar_dp, dimension_type
from ..models.load_store_operations import lsc_operation, lsc_load, lsc_store, lsc_addr_add, lsc_reg_type

class lsc_treg_row_extract(lsc_operation):
    def __init__(self,
                 treg_tile : tile,
                 vreg_tile : tile,
                 component : str,
                 treg_id : int,
                 vreg_id : int
                 ):

        tiles = [treg_tile, vreg_tile]
        indices = [(component,[treg_id]), (component,[vreg_id])]
        reg_types = [lsc_reg_type.data, lsc_reg_type.data]
        reads = [0]
        writes = [1]
        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)

class lsc_treg_rc_ldst(lsc_operation):
    @property
    def res_idx(self):
        return self.indices[1][1]

    @property
    def addr_idx(self):
        return self.indices[0][1]

    @property
    def component(self):
        return self.indices[0][0]

    @property
    def t(self):
        return self.tiles[1]

class lsc_treg_row_store(lsc_treg_rc_ldst):
    def __init__(self,
                 treg_slice_tile : tile,
                 component : str,
                 addr_idx : int,
                 treg_id : int,
                 aoff : int,
                 roff : int
                 ):

        self.aoff = aoff
        self.roff = roff

        tiles = [scalar_tile, treg_slice_tile]
        indices = [(component,[addr_idx]), (component,[treg_id])]
        reg_types = [lsc_reg_type.address, lsc_reg_type.data]
        reads = [0,1]
        writes = []
        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)

    def __str__(self):
        vladims = min(1,sum([1 if (d.dt == dimension_type.vla) else 0 for d in [self.t.dima,self.t.dimb] ]))
        vlenstr = ""
        if 0 < vladims:
            vlenstr = "*VLEN"*vladims
        return f"{self.component}a{self.addr_idx} + {self.aoff}{vlenstr} <- STORE {self.component}{self.res_idx}.row[{self.roff}]"


class lsc_treg_row_insert(lsc_operation):
    def __init__(self,
                 treg_tile : tile,
                 vreg_tile : tile,
                 component : str,
                 treg_id : int,
                 vreg_id : int
                 ):

        tiles = [treg_tile, vreg_tile]
        indices = [(component,[treg_id]), (component,[vreg_id])]
        reg_types = [lsc_reg_type.data, lsc_reg_type.data]
        reads = [1]
        writes = [0]
        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)

class lsc_treg_row_load(lsc_treg_rc_ldst):
    def __init__(self,
                 treg_slice_tile : tile,
                 component : str,
                 addr_idx : int,
                 treg_id : int,
                 aoff : int,
                 roff : int
                 ):

        self.aoff = aoff
        self.roff = roff

        tiles = [scalar_tile, treg_slice_tile]
        indices = [(component,[addr_idx]), (component,[treg_id])]
        reg_types = [lsc_reg_type.address, lsc_reg_type.data]
        reads  = [0]
        writes = [1]
        super().__init__(tiles=tiles, indices=indices,
                         reads=reads, writes=writes,
                         reg_types=reg_types)

class special_treg_ldst:

    def __init__(self):
        self.load_queue = []
        self.store_queue = []

    def check_load_flush(self, op : lsc_operation):
        for read_idx in op.reads:
            if lsc_reg_type.data != op.reg_types[read_idx]:
                continue
            component = op.indices[read_idx][0]
            res_idx = op.indices[read_idx]
            for op in self.load_queue:
                if op.component == component and \
                   op.res_idx == res_idx:
                    return True
        return False

    def check_store_flush(self, op : lsc_operation):
        for write_idx in op.writes:
            if lsc_reg_type.data != op.reg_types[write_idx]:
                continue
            component = op.indices[write_idx][0]
            res_idx = op.indices[write_idx]
            for op in self.load_queue:
                if op.component == component and \
                   op.res_idx == res_idx:
                    return True
        return False

    def add_treg_load(self, op: lsc_load, dt : adt):
        # 1 load ->
        # vla loop until reg off < vlen:
        #   fixed loop:
        #     load_row regoff+immoff from aoff
        #     inc immoff
        #     inc aoff by vlen
        #   inc reg off by immoff
        rows_per_iter = 16//adt_size(dt)

        slice_tile = tile(scalar_dp, op.t.dimb)
        for i in range(rows_per_iter):
            self.load_queue.append(
                    lsc_treg_row_load(treg_slice_tile=slice_tile,
                                      component=op.component,
                                      addr_idx=op.addr_idx,
                                      treg_id=op.res_idx,
                                      aoff=i+op.off*rows_per_iter,
                                      roff=i))

    def add_treg_store(self, op : lsc_store, dt : adt):
        rows_per_iter = 16//adt_size(dt)

        slice_tile = tile(op.t.dima, scalar_dp)
        for i in range(rows_per_iter):
            self.store_queue.append(
                    lsc_treg_row_store(treg_slice_tile=slice_tile,
                                      component=op.component,
                                      addr_idx=op.addr_idx,
                                      treg_id=op.res_idx,
                                      aoff=i+op.off*rows_per_iter,
                                      roff=i))

    def flush_load(self) -> list[lsc_operation]:

        loads = deepcopy(self.load_queue)
        self.load_queue.clear()
        return loads
        
    def flush_store(self) -> list[lsc_operation]:

        stores = deepcopy(self.store_queue)
        self.store_queue.clear()
        return stores
