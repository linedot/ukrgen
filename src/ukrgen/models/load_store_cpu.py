# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import copy
import string
from abc import abstractmethod
from enum import Enum,auto

from ..generators import mm_op,tile,dimension_type
from ..components.operation import operation
from ..components import scalar_tile

from .load_store_operations import (
    lsc_addr_add,
    lsc_debugmsg,
    lsc_load,
    lsc_offset,
    lsc_operation,
    lsc_store,
    lsc_transformation,
    lsc_zero,
)

from .tile_offset_mapper import tile_offset_mapper

from .addr_resolver import addr_resolver

class lsc_state(Enum):
    invalid = auto()
    clean = auto()
    loaded = auto()
    modified = auto()



class load_store_cpu:
    def __init__(self,
                 res_counts : list[int],
                 res_steps : list[int],
                 ar : addr_resolver,
                 preload_counts : list[int],
                 offset_mappers : list[tile_offset_mapper],
                 resolve_order : list[int] = [0,1,2],
                 op : str = "fma"):

        for rc,pc in zip(res_counts,preload_counts):
            assert pc <= rc, "Can't preload more than max. specified number of resources"

        self.res_counts = res_counts
        self.res_steps = res_steps
        self.preload_counts = preload_counts
        self.offset_mappers = offset_mappers
        self.resolve_order = resolve_order
        self.op = op
        self.ar = ar

        self.reset()

    def reset(self):

        self.res_indices = [0]*len(self.res_counts)
        self.res_subindices = [0]*len(self.res_counts)

        self.cdos = [
            { i : None for i in range(res_count)}\
                    for res_count,pre_count in\
                    zip(self.res_counts,self.preload_counts)
        ]

        self.states = [
            { i : lsc_state.invalid for i in range(res_count)}\
                    for res_count,pre_count in\
                    zip(self.res_counts,self.preload_counts)
        ]

        # Track last tile used per res to determine offset to next tile given
        # last offset used to load data for this res
        # NOTE: This is currently only used by preload(add_current_offsets=True)
        self.last_tile_used = [None]*len(self.res_counts)

    def resolve_data(self, t : tile,
                     res_idx : int,
                     toff : lsc_offset,
                     rtype_idx : int) -> list[lsc_operation]:
        # check if the required data is already in the resource
        corg = self.cdos[rtype_idx][res_idx]
        result = []
        # At this point corg should be toff if the reg is preloaded
        #if corg == -2:
        #    self.cdos[rtype_idx][res_idx] = toff
        #    corg = toff
        rtype_char = string.ascii_lowercase[rtype_idx]
        if (not corg is None) and \
                (not lsc_state.invalid == self.states[rtype_idx][res_idx]):
            if corg == toff:
                #print(f"{rtype_char}{res_idx} already has offset {toff}")
                return result

        # store if dirty
        if lsc_state.modified == self.states[rtype_idx][res_idx]:
            addr_adds,idx,off = self.ar.resolve_addr(rtype_idx=rtype_idx, toff=corg)
            for add in addr_adds:
                result.append(lsc_addr_add(rtype_idx=add.rtype_idx,
                                           addr_idx=add.addr_idx,
                                           off=add.offset,
                                           t=t))

            result.append(lsc_store(rtype_idx=rtype_idx, res_idx=res_idx,
                                    addr_idx=idx,
                                    off=off,
                                    t=t))

        # load
        addr_adds,idx,off = self.ar.resolve_addr(rtype_idx=rtype_idx, toff=toff)
        for add in addr_adds:
            result.append(lsc_addr_add(rtype_idx=add.rtype_idx,
                                       addr_idx=add.addr_idx,
                                       off=add.offset,
                                       t=t))

        #print(f"Updating {rtype_char}{res_idx} off to {toff}")
        self.cdos[rtype_idx][res_idx] = toff
        self.states[rtype_idx][res_idx] = lsc_state.loaded

        assert(idx is not None)
        result.append(lsc_load(rtype_idx=rtype_idx, res_idx=res_idx,
                               addr_idx=idx,
                               off=off,
                               t=t))
        self.last_tile_used[rtype_idx] = t

        return result

    def store_modified(self) -> list[lsc_operation]:
        result = []
        for rtype_idx,res_count in enumerate(self.res_counts):
            for res_idx in range(res_count):
                if lsc_state.modified == self.states[rtype_idx][res_idx]:
                    corg = self.cdos[rtype_idx][res_idx]
                    # NOTE: are different tiles for different res indices feasible?
                    t = self.last_tile_used[rtype_idx]
                    addr_adds,idx,off = self.ar.resolve_addr(
                            rtype_idx=rtype_idx, toff=corg)
                    for add in addr_adds:
                        result.append(lsc_addr_add(rtype_idx=add.rtype_idx,
                                                   addr_idx=add.addr_idx,
                                                   off=add.offset,
                                                   t=t))
                    result.append(lsc_store(rtype_idx=rtype_idx, res_idx=res_idx,
                                            addr_idx=idx,
                                            off=off,
                                            t=t))
                    self.states[rtype_idx][res_idx] = lsc_state.clean
        return result

        
    def map_one(self,
                a_tile : tile, b_tile : tile, c_tile : tile,
                a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                m_subidx : int, n_subidx : int, k_subidx : int) -> list[lsc_operation]:

        tiles = [a_tile,b_tile,c_tile]
        a_subidx = None
        if a_tile.dima.size > c_tile.dima.size:
            a_subidx = m_subidx
            c_idx = (c_idx[0]+m_subidx, c_idx[1])
        if a_tile.dimb.size > b_tile.dima.size:
            a_subidx = k_subidx
            b_idx = (b_idx[0]+k_subidx, b_idx[1])

        b_subidx = None
        if b_tile.dima.size > a_tile.dimb.size:
            b_subidx = k_subidx
            a_idx = (a_idx[0], a_idx[1]+k_subidx)
        if b_tile.dimb.size > c_tile.dimb.size:
            b_subidx = n_subidx
            c_idx = (c_idx[0], c_idx[1]+n_subidx)
            

        c_subidx = None
        if c_tile.dima.size > a_tile.dima.size:
            c_subidx = m_subidx
            a_idx = (a_idx[0]+m_subidx, a_idx[1])
        if c_tile.dimb.size > b_tile.dimb.size:
            c_subidx = n_subidx
            b_idx = (b_idx[0], b_idx[1]+n_subidx)

        indices = [a_idx,b_idx,c_idx]
        target_offsets = [ mapper(tile,idx) for  mapper,tile,idx in\
                zip(self.offset_mappers,tiles,indices)]

        result = []
        res_indices = [None for i in self.res_indices]




        
        #for i,(toff,dos,res_count,t) in enumerate(
        #        zip(target_offsets,self.cdos,self.res_counts,[a_tile,b_tile,c_tile])):
        for i in self.resolve_order[:len(self.res_counts)]:
            toff = target_offsets[i]
            dos = self.cdos[i]
            res_count = self.res_counts[i]
            t = tiles[i]
            res_idx = None

            #result.append(lsc_debugmsg(f"idx {i}:{indices[i]} translated to offset {toff}"))

            # check if any of the registers have the data
            for j in range(res_count):
                if dos[j] is None:
                    continue
                if (toff == dos[j]):
                    res_idx = j
                    break

            # use the current index for this input, use the next next time
            if res_idx is None:
                res_idx = self.res_indices[i]
                self.res_indices[i] = (res_idx+self.res_steps[i]) \
                                          % self.res_counts[i]

            res_indices[i] = res_idx
            result.extend(self.resolve_data(t=t,
                                            res_idx=res_idx,
                                            toff=toff,
                                            rtype_idx=i))

        areg = res_indices[0]
        breg = res_indices[1]
        creg = res_indices[2]
        #result.append(f"c{creg} <- {self.op}(a{areg},b{breg},c{creg})")


        self.states[2][creg] = lsc_state.modified
        result.append(lsc_transformation(op=self.op, res_indices=res_indices,
                                  sub_indices=[a_subidx,b_subidx,c_subidx],
                                  tiles=[a_tile,b_tile,c_tile]))
        self.last_tile_used[0] = a_tile
        self.last_tile_used[1] = b_tile
        self.last_tile_used[2] = c_tile
        return result

    def preload(self, ops : list[mm_op],
                next_ops : list[mm_op],
                zero_addrs : bool = False,
                update_addrs : bool = True,
                zero_dims : list[int] = [2],
                ignore_dims : list[int] = [],
                add_current_offsets : bool = False,
                ) -> list[lsc_operation]:
        results = []
        preloads_done = [0]*len(self.res_counts)
        preload_states = copy.deepcopy(self.states)
        for dim in ignore_dims:
            # treat ignored as already preloaded
            preloads_done[dim] = self.preload_counts[dim]+1
            # ignores modified state
            self.states[dim] = {i : lsc_state.loaded \
                    for i in range(self.res_counts[dim])}



        # Trick: We save the tracked data offsets before the preload
        #        and then restore them, modifying only those that
        #        the preload should affect
        initial_cdos = copy.deepcopy(self.cdos)
        self.cdos = [{ i : None for i in range(res_count)} for \
                res_count in self.res_counts]
        if zero_addrs:
            self.ar.zero_current_offsets()

        # NOTE: Another pythonism, the commented out line and the next one
        #       are not equivalent and the former will result in partly
        #       overlapping dicts
        # preload_dos = [{}]*len(self.res_counts)
        preload_dos = [{} for i in range(len(self.res_counts))]
        # NOTE: gotta copy the dicts explicitly, otherwise they'll be references
        #preload_addr_reg_offsets = [d.copy() for d in self.addr_reg_offsets]
        preload_addr_reg_offsets = copy.deepcopy(self.ar.current_offsets)

        # for each rtype_idx, what offsets contains the data necessary for the next op
        # after the preload
        preload_next_offsets = [None for i in range(len(self.res_counts))]

        preload_addr_reg_last_used_tile = [ [None for i in reglist] for\
                reglist in self.ar.indices]
        i = 0
        combined_ops = ops+next_ops
        is_next_preload = lambda rtype_idx : \
                (self.preload_counts[rtype_idx]) == preloads_done[rtype_idx]
        while any([p+1 > d for p,d in zip(self.preload_counts,preloads_done)])\
              and i < len(combined_ops):
            op = combined_ops[i]
            subresults = []
            mapresults = self.map_one(
                a_tile = op.a_tile, b_tile = op.b_tile, c_tile = op.c_tile,
                a_idx = op.a_idx, b_idx = op.b_idx, c_idx = op.c_idx,
                m_subidx = op.m_subidx, n_subidx = op.n_subidx, k_subidx = op.k_subidx
                )
            for op in mapresults:
                if isinstance(op, lsc_transformation):
                    continue
                if isinstance(op, lsc_debugmsg):
                    subresults.append(op)
                    continue
                rtype_char = string.ascii_lowercase[op.rtype_idx]
                pc = self.preload_counts[op.rtype_idx]
                pd = preloads_done[op.rtype_idx]
                #print(f"Preloads for {rtype_char}: {pd}/{pc}")
                if pc+1 <= pd:
                    continue
                if isinstance(op, lsc_store):
                    # Haven't thought about what this would mean, so error out for now
                    raise RuntimeError("store op encountered while processing preload")
                if isinstance(op, lsc_addr_add):
                    #caoff = self.addr_reg_offsets[op.rtype_idx][op.addr_idx]
                    caoff = self.ar.current_offsets[op.rtype_idx][op.addr_idx]

                    if is_next_preload(op.rtype_idx):
                        preload_next_offsets[op.rtype_idx] = caoff
                        continue
                    #preload_addr_reg_offsets[op.rtype_idx][op.addr_idx] = caoff + op.off
                    preload_addr_reg_last_used_tile[op.rtype_idx][op.addr_idx] = op.t

                    if op.rtype_idx not in zero_dims:
                        preload_addr_reg_offsets[op.rtype_idx][op.addr_idx] = caoff
                        subresults.append(op)
                if isinstance(op, lsc_load):
                    #caoff = self.addr_reg_offsets[op.rtype_idx][op.addr_idx]
                    caoff = self.ar.current_offsets[op.rtype_idx][op.addr_idx]
                    # Some updates to offsets are implicit, therefore update
                    new_do = caoff + op.off
                    if is_next_preload(op.rtype_idx):
                        #print(f"Next offset for {rtype_char}: {new_do}")
                        preload_next_offsets[op.rtype_idx] = new_do
                        preloads_done[op.rtype_idx] += 1
                        continue
                    preload_dos[op.rtype_idx][op.res_idx] = new_do
                    tsize = op.t.dima.size*op.t.dimb.size
                    preload_addr_reg_last_used_tile[op.rtype_idx][op.addr_idx] = op.t

                    preloads_done[op.rtype_idx] += 1
                    if op.rtype_idx in zero_dims:
                        subresults.append(lsc_zero(
                            rtype_idx=op.rtype_idx, res_idx=op.res_idx, t=op.t))
                    else:
                        preload_addr_reg_offsets[op.rtype_idx][op.addr_idx] = caoff
                        subresults.append(op)

                    preload_states[op.rtype_idx][op.res_idx] = lsc_state.loaded

            results.extend(subresults)
            i+= 1

        preload_next_offsets = [lsc_offset.zero_offset() if no is None else no \
                for no in preload_next_offsets]
        # only the changes that were part of the preload should be propagated
        self.ar.current_offsets = copy.deepcopy(preload_addr_reg_offsets)

        if update_addrs:
            addr_adds = self.ar.get_addr_adds_for_new_offsets(preload_next_offsets)

            for add in addr_adds:
                # No address updates if data was just zeroed out
                if add.rtype_idx in zero_dims:
                    continue
                t = preload_addr_reg_last_used_tile[add.rtype_idx][add.addr_idx]

                # no loads happaned, therefor no tile used for a load
                # Tile will have been set for a tranform, use it
                if None == t:
                    t = self.last_tile_used[add.rtype_idx]
                results.append(lsc_addr_add(rtype_idx=add.rtype_idx,
                                            addr_idx=add.addr_idx,
                                            off=add.offset,
                                            t=t))
                # Add to tracked offset
                self.ar.current_offsets[add.rtype_idx][add.addr_idx] += add.offset

        # use the original dos and assign the preloaded data
        self.cdos = copy.deepcopy(initial_cdos)
        for i,dos in enumerate(preload_dos):
            for res_idx,orig in dos.items():
                self.cdos[i][res_idx] = orig

        #print("dos after preload:")
        #for i,dos in enumerate(self.cdos):
        #    rtype_char = string.ascii_lowercase[i]
        #    for res_idx,do in dos.items():
        #        print(f"  {rtype_char}{res_idx} : {do}")

        #print("tracked pre offsets after preload:")
        #for i,idx_list in enumerate(self.ar.indices):
        #    rtype_char = string.ascii_lowercase[i]
        #    for idx in idx_list:
        #        off = preload_addr_reg_offsets[i][idx]
        #        print(f"  {rtype_char}a{idx} : {off}")

        #print("addr offsets after preload:")
        #for i,idx_list in enumerate(self.ar.indices):
        #    rtype_char = string.ascii_lowercase[i]
        #    for idx in idx_list:
        #        off = self.ar.current_offsets[i][idx]
        #        print(f"  {rtype_char}a{idx} : {off}")

        #print("next offsets after preload:")
        #for i,idx_list in enumerate(self.ar.indices):
        #    rtype_char = string.ascii_lowercase[i]
        #    off = preload_next_offsets[i]
        #    print(f"  {rtype_char} : {off}")

        # set the reg states
        self.states = copy.deepcopy(preload_states)

        # reset the indices
        # self.res_indices = [ 0 for i in self.res_indices]
        self.res_indices = [ d%count for d,count in\
                zip(self.preload_counts,self.res_counts)]
        return results

    def __call__(self, ops : list[mm_op]) -> list[str]:

        #print("dos before main call:")
        #for i,dos in enumerate(self.cdos):
        #    rtype_char = string.ascii_lowercase[i]
        #    for res_idx,do in dos.items():
        #        print(f"  {rtype_char}{res_idx} : {do}")
        result = []
        for op in ops:
            result.extend(self.map_one(
                a_tile = op.a_tile, b_tile = op.b_tile, c_tile = op.c_tile,
                a_idx = op.a_idx, b_idx = op.b_idx, c_idx = op.c_idx,
                m_subidx = op.m_subidx, n_subidx = op.n_subidx, k_subidx = op.k_subidx
                ))
        return result
