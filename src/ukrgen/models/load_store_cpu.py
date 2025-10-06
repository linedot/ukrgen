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
    lsc_operation,
    lsc_store,
    lsc_transformation,
    lsc_zero,
)

from .lsc.index import lsc_reg_index
from .lsc.offset import lsc_offset

from .offset_mapper import offset_mapper

from .addr_resolver import addr_resolver

class lsc_state(Enum):
    invalid = auto()
    clean = auto()
    loaded = auto()
    modified = auto()



class load_store_cpu:
    def __init__(self,
                 res_counts : dict[str,int],
                 res_steps : dict[str,int],
                 ar : addr_resolver,
                 preload_counts : dict[str,int],
                 offset_mappers : dict[str,offset_mapper],
                 resolve_order : list[str]):

        for c in res_counts.keys():
            assert preload_counts[c] <= res_counts[c], "Can't preload more than max. specified number of resources"


        self.res_counts = res_counts
        self.res_steps = res_steps
        self.preload_counts = preload_counts
        self.offset_mappers = offset_mappers
        self.resolve_order = resolve_order
        self.ar = ar

        self.reset()

    def reset(self):

        self.res_indices = {c : 0 for c in self.res_counts.keys()}
        self.res_subindices = {c : 0 for c in self.res_counts.keys()}

        self.cdos = {
            c: { i : None for i in range(res_count)}\
                    for c,res_count in\
                    self.res_counts.items()
        }

        self.states = {
            c: { i : lsc_state.invalid for i in range(res_count)}\
                    for c,res_count in\
                    self.res_counts.items()
        }

        # Track last tile used per res to determine offset to next tile given
        # last offset used to load data for this res
        # NOTE: This is currently only used by preload(add_current_offsets=True)
        self.last_tile_used = { c : None for c in self.res_counts.keys() }

    def resolve_data(self, t : tile,
                     res_idx : int,
                     toff : lsc_offset,
                     component : str) -> list[lsc_operation]:

        if not isinstance(res_idx, int):
            raise ValueError(f"Invalid res_idx: {res_idx}")

        # check if the required data is already in the resource
        corg = self.cdos[component][res_idx]
        result = []

        #print(f"Resolving {toff} into {component}{res_idx}")
        if (not corg is None) and \
                (not lsc_state.invalid == self.states[component][res_idx]):
            if corg == toff:
                #print(f"{component}{res_idx} already has offset {toff}")
                return result

        # store if dirty
        if lsc_state.modified == self.states[component][res_idx]:
            addr_adds,idx,off = self.ar.resolve_addr(component=component, toff=corg)
            for add in addr_adds:
                result.append(lsc_addr_add(component=add.component,
                                           addr_idx=add.addr_idx,
                                           off=add.offset,
                                           t=t))

            result.append(lsc_store(component=component, res_idx=res_idx,
                                    addr_idx=idx,
                                    off=off,
                                    stride=None,
                                    t=t,
                                    mods=set()))

        # load
        addr_adds,idx,off = self.ar.resolve_addr(component=component, toff=toff)
        for add in addr_adds:
            result.append(lsc_addr_add(component=add.component,
                                       addr_idx=add.addr_idx,
                                       off=add.offset,
                                       t=t))

        #print(f"cdos for component {component}:")
        #print(f"\t{self.cdos[component]}")
        #print(f"Updating {component}{res_idx} off to {toff}")
        self.cdos[component][res_idx] = toff
        self.states[component][res_idx] = lsc_state.loaded

        assert(idx is not None)
        result.append(lsc_load(component=component, res_idx=res_idx,
                               addr_idx=idx,
                               off=off,
                               stride=None,
                               t=t,
                               mods=set()))
        self.last_tile_used[component] = t

        return result

    def store_modified(self, ignore_components : list[str]) -> list[lsc_operation]:
        result = []
        for component,res_count in self.res_counts.items():
            if component in ignore_components:
                continue
            for res_idx in range(res_count):
                if lsc_state.modified == self.states[component][res_idx]:
                    corg = self.cdos[component][res_idx]
                    # NOTE: are different tiles for different res indices feasible?
                    t = self.last_tile_used[component]
                    addr_adds,idx,off = self.ar.resolve_addr(
                            component=component, toff=corg)
                    for add in addr_adds:
                        result.append(lsc_addr_add(component=add.component,
                                                   addr_idx=add.addr_idx,
                                                   off=add.offset,
                                                   t=t))
                    result.append(lsc_store(component=component, res_idx=res_idx,
                                            addr_idx=idx,
                                            off=off,
                                            stride=None,
                                            t=t,mods=set()))
                    self.states[component][res_idx] = lsc_state.clean
        return result

        
    def map_one(self,
                a_tile : tile, b_tile : tile, c_tile : tile,
                a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                m_subidx : int, n_subidx : int, k_subidx : int,
                op : str, cnames : list[str]) -> list[lsc_operation]:

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

        target_offsets = { cname : self.offset_mappers[cname].map_tile_idx(tile,idx) for  cname,tile,idx in\
                zip(cnames,tiles,indices)}

        tiles = {c : t for c,t in zip(cnames,tiles)}

        result = []
        res_indices = {c : None for c in cnames}




        
        ordered_components = [c for c in self.resolve_order if c in cnames]
        for component in ordered_components:

            toff = target_offsets[component]
            dos = self.cdos[component]
            res_count = self.res_counts[component]
            t = tiles[component]
            res_idx = None

            #result.append(lsc_debugmsg(f"idx {i}:{indices[i]} translated to offset {toff}"))

            # check if any of the registers have the data
            for j in range(res_count):
                #print(f"Looking for {toff}, {component}{j} has {dos[j]}")
                if dos[j] is None:
                    continue
                #print(f"toff-caoff ={toff-dos[j]}")
                #print(f"{dos[j].sxv_strides.keys()}")
                if (toff == dos[j]):
                    res_idx = j
                    break

            # use the current index for this input, use the next next time
            if res_idx is None:
                res_idx = self.res_indices[component]
                self.res_indices[component] = (res_idx+self.res_steps[component]) \
                                          % self.res_counts[component]

            res_indices[component] = res_idx
            result.extend(self.resolve_data(t=t,
                                            res_idx=res_idx,
                                            toff=toff,
                                            component=component))

        areg = res_indices[cnames[0]]
        breg = res_indices[cnames[1]]
        creg = res_indices[cnames[2]]


        lsc_indices = [
                lsc_reg_index(c, [res_indices[c], subidx] if \
                        subidx is not None else [res_indices[c]])
                for c,subidx in zip(cnames,
                                        [a_subidx,b_subidx,c_subidx])
                ]

        self.states[cnames[2]][creg] = lsc_state.modified
        result.append(
                lsc_transformation(op=op,
                    res_indices=lsc_indices,
                    tiles=[a_tile,b_tile,c_tile]))
        self.last_tile_used[cnames[0]] = a_tile
        self.last_tile_used[cnames[1]] = b_tile
        self.last_tile_used[cnames[2]] = c_tile
        return result

    def preload(self, ops : list[mm_op],
                next_ops : list[mm_op],
                zero_components : list[str],
                ignore_components : list[str],
                zero_addrs : bool = False,
                update_addrs : bool = True,
                add_current_offsets : bool = False,
                ) -> list[lsc_operation]:
        results = []
        preloads_done = {c : 0 for c in self.res_counts.keys()}
        preload_states = copy.deepcopy(self.states)
        for c in ignore_components:
            if c not in self.preload_counts:
                continue
            # treat ignored as already preloaded
            preloads_done[c] = self.preload_counts[c]+1
            # ignores modified state
            self.states[c] = {i : lsc_state.loaded \
                    for i in range(self.res_counts[c])}



        # Trick: We save the tracked data offsets before the preload
        #        and then restore them, modifying only those that
        #        the preload should affect
        initial_cdos = copy.deepcopy(self.cdos)
        self.cdos = { c : { i : None for i in range(res_count)} for \
                c,res_count in self.res_counts.items()}
        if zero_addrs:
            self.ar.zero_current_offsets()

        # NOTE: Another pythonism, the commented out line and the next one
        #       are not equivalent and the former will result in partly
        #       overlapping dicts
        # preload_dos = [{}]*len(self.res_counts)
        preload_dos = {c : dict() for c in self.res_counts.keys()}
        # NOTE: gotta copy the dicts explicitly, otherwise they'll be references
        #preload_addr_reg_offsets = [d.copy() for d in self.addr_reg_offsets]
        preload_addr_reg_offsets = copy.deepcopy(self.ar.current_offsets)

        # for each component, what offsets contains the data necessary for the next op
        # after the preload
        preload_next_offsets = { c : None for c in self.res_counts.keys()}

        preload_addr_reg_last_used_tile = { c : [None for i in reglist] for\
                c,reglist in self.ar.indices.items()}
        i = 0
        combined_ops = ops+next_ops
        is_next_preload = lambda component : \
                (self.preload_counts[component]) == preloads_done[component]
        while any([self.preload_counts[c]+1 > d for c,d in preloads_done.items()])\
              and i < len(combined_ops):
            op = combined_ops[i]
            subresults = []
            mapresults = self.map_one(
                a_tile = op.a_tile, b_tile = op.b_tile, c_tile = op.c_tile,
                a_idx = op.a_idx, b_idx = op.b_idx, c_idx = op.c_idx,
                m_subidx = op.m_subidx, n_subidx = op.n_subidx, k_subidx = op.k_subidx,
                op=op.opstr, cnames=op.tile_strs
                )
            for op in mapresults:
                if isinstance(op, lsc_transformation):
                    continue
                if isinstance(op, lsc_debugmsg):
                    subresults.append(op)
                    continue
                component = op.indices[0].component
                pc = self.preload_counts[component]
                pd = preloads_done[component]
                #print(f"Preloads for {component}: {pd}/{pc}")
                if pc+1 <= pd:
                    continue
                if isinstance(op, lsc_store):
                    # Haven't thought about what this would mean, so error out for now
                    raise RuntimeError("store op encountered while processing preload")
                if isinstance(op, lsc_addr_add):
                    #caoff = self.addr_reg_offsets[component][op.addr_idx]
                    caoff = self.ar.current_offsets[component][op.addr_idx.indices[0]]

                    if is_next_preload(component):
                        preload_next_offsets[component] = caoff
                        continue
                    #preload_addr_reg_offsets[component][op.addr_idx] = caoff + op.off
                    preload_addr_reg_last_used_tile[component][op.addr_idx.indices[0]] = op.t

                    if component not in zero_components:
                        preload_addr_reg_offsets[component][op.addr_idx.indices[0]] = caoff
                        subresults.append(op)
                if isinstance(op, lsc_load):
                    #caoff = self.addr_reg_offsets[component][op.addr_idx]
                    caoff = self.ar.current_offsets[component][op.addr_idx.indices[0]]
                    # Some updates to offsets are implicit, therefore update
                    new_do = caoff + op.off

                    if is_next_preload(component):
                        #print(f"Next offset for {component}: {new_do}")
                        preload_next_offsets[component] = new_do
                        preloads_done[component] += 1
                        continue
                    preload_dos[component][op.res_idx.indices[0]] = new_do
                    tsize = op.t.dima.size*op.t.dimb.size
                    preload_addr_reg_last_used_tile[component][op.addr_idx.indices[0]] = op.t

                    preloads_done[component] += 1
                    if component in zero_components:
                        subresults.append(lsc_zero(
                            component=component, res_idx=op.res_idx.indices[0], t=op.t))
                    else:
                        preload_addr_reg_offsets[component][op.addr_idx] = caoff
                        subresults.append(op)

                    preload_states[component][op.res_idx.indices[0]] = lsc_state.loaded

            results.extend(subresults)
            i+= 1

        preload_next_offsets = { c : lsc_offset.zero_offset() if no is None else no \
                for c,no in preload_next_offsets.items()}
        #print(f"pno: {",".join([str(no) for no in preload_next_offsets])}")
        # only the changes that were part of the preload should be propagated
        self.ar.current_offsets = copy.deepcopy(preload_addr_reg_offsets)

        if update_addrs:
            addr_adds = self.ar.get_addr_adds_for_new_offsets(preload_next_offsets)

            for add in addr_adds:
                # No address updates if data was just zeroed out
                if add.component in zero_components:
                    continue
                t = preload_addr_reg_last_used_tile[add.component][add.addr_idx]

                # no loads happaned, therefor no tile used for a load
                # Tile will have been set for a tranform, use it
                if None == t:
                    t = self.last_tile_used[add.component]
                results.append(lsc_addr_add(component=add.component,
                                            addr_idx=add.addr_idx,
                                            off=add.offset,
                                            t=t))
                # Add to tracked offset
                self.ar.current_offsets[add.component][add.addr_idx] += add.offset

        # use the original dos and assign the preloaded data
        self.cdos = copy.deepcopy(initial_cdos)
        for c,dos in preload_dos.items():
            for res_idx,orig in dos.items():
                self.cdos[c][res_idx] = orig



        # set the reg states
        self.states = copy.deepcopy(preload_states)

        # reset the indices
        self.res_indices = { c : d % self.res_counts[c] for c,d in\
                self.preload_counts.items()}
        return results

    def __call__(self, ops : list[mm_op]) -> list[str]:

        result = []
        for op in ops:
            result.extend(self.map_one(
                a_tile = op.a_tile, b_tile = op.b_tile, c_tile = op.c_tile,
                a_idx = op.a_idx, b_idx = op.b_idx, c_idx = op.c_idx,
                m_subidx = op.m_subidx, n_subidx = op.n_subidx, k_subidx = op.k_subidx,
                op=op.opstr, cnames=op.tile_strs
                ))
        return result
