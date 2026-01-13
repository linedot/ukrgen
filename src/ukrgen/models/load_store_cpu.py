# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging
import copy
import string
from abc import abstractmethod
from enum import Enum,auto

from dataclasses import dataclass

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


@dataclass
class lsc_model_state:
    res_indices : dict[str,int]
    res_subindices : dict[str,int]
    cdos : dict[str,dict[int,lsc_offset|None]]
    acdos : dict[str,list[lsc_offset|None]]
    states : dict[str,dict[int,lsc_state]]
    last_tile_used : dict[str,tile|None]


class load_store_cpu:
    def __init__(self,
                 res_counts : dict[str,int],
                 res_steps : dict[str,int],
                 ar : addr_resolver,
                 offset_mappers : dict[str,offset_mapper],
                 resolve_order : list[str]):



        self.res_counts = res_counts
        self.res_steps = res_steps
        self.offset_mappers = offset_mappers
        self.resolve_order = resolve_order
        self.ar = ar

        self.debug = logging.getLogger("LSC").debug

        self.states : dict[str,lsc_model_state] = dict()

        self.new_state("default")
        self.current_state : str = "default"

    def new_state(self, name : str,
                  copyfrom : str|None = None):

        if name in self.states:
            raise ValueError(f"state {name} already exists")

        if copyfrom is not None:
            if copyfrom not in self.states:
                raise ValueError(f"state {copyfrom} doesn't exist")


        self.states[name] = lsc_model_state(
            res_indices={c : 0 for c in self.res_counts.keys()},
            res_subindices={c : 0 for c in self.res_counts.keys()},
            cdos={
                c: { i : None for i in range(res_count)}\
                    for c,res_count in\
                    self.res_counts.items()
            },
            acdos=copy.deepcopy(self.ar.starting_offsets),
            states={
                c: { i : lsc_state.invalid for i in range(res_count)}\
                    for c,res_count in\
                    self.res_counts.items()
            },
            last_tile_used={ c : None for c in self.res_counts.keys() })

    def resolve_data(self, t : tile,
                     res_idx : int,
                     toff : lsc_offset,
                     component : str) -> list[lsc_operation]:

        if not isinstance(res_idx, int):
            raise ValueError(f"Invalid res_idx: {res_idx}")

        state = self.states[self.current_state]
        self.ar.current_offsets = state.acdos

        # check if the required data is already in the resource
        corg = state.cdos[component][res_idx]
        result = []

        self.debug(f"Requiring {toff} in {component}{res_idx}")
        if (not corg is None) and \
                (not lsc_state.invalid == state.states[component][res_idx]):
            if corg == toff:
                #print(f"{component}{res_idx} already has offset {toff}")
                return result

        # store if dirty
        if lsc_state.modified == state.states[component][res_idx]:
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

        self.debug(f"Resolving {toff} for {component}{res_idx}")
        # load
        addr_adds,idx,off = self.ar.resolve_addr(component=component, toff=toff)
        for add in addr_adds:
            result.append(lsc_addr_add(component=add.component,
                                       addr_idx=add.addr_idx,
                                       off=add.offset,
                                       t=t))

        #print(f"cdos for component {component}:")
        #print(f"\t{state.cdos[component]}")
        #print(f"Updating {component}{res_idx} off to {toff} by loading from {idx} with {off}")
        state.cdos[component][res_idx] = toff
        state.states[component][res_idx] = lsc_state.loaded

        assert(idx is not None)
        result.append(lsc_load(component=component, res_idx=res_idx,
                               addr_idx=idx,
                               off=off,
                               stride=None,
                               t=t,
                               mods=set()))
        state.last_tile_used[component] = t

        return result

    def store_modified(self, ignore_components : list[str]) -> list[lsc_operation]:

        state = self.states[self.current_state]
        self.ar.current_offsets = state.acdos

        result = []
        for component,res_count in self.res_counts.items():
            if component in ignore_components:
                continue
            for res_idx in range(res_count):
                if lsc_state.modified == state.states[component][res_idx]:
                    corg = state.cdos[component][res_idx]
                    # NOTE: are different tiles for different res indices feasible?
                    t = state.last_tile_used[component]
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
                    state.states[component][res_idx] = lsc_state.clean
        return result

        
    def map_one(self,
                a_tile : tile, b_tile : tile, c_tile : tile,
                a_idx : (int,int), b_idx : (int,int), c_idx : (int,int),
                m_subidx : int, n_subidx : int, k_subidx : int,
                op : str, cnames : list[str]) -> list[lsc_operation]:

        tiles = [a_tile,b_tile,c_tile]

        a_subidx = None
        b_subidx = None
        c_subidx = None

        state = self.states[self.current_state]
        self.ar.current_offsets = state.acdos

        # TODO: deduplicate this code and what's in operation.py

        first_input_tiling = a_tile.tiling_of(c_tile)
        last_input_tiling = b_tile.tiling_of(c_tile)

        if 0 in first_input_tiling and 0 not in last_input_tiling:
            if 0 == first_input_tiling[0] and 0 != first_input_tiling[1]:
                # A subidx, C add
                a_subidx = m_subidx
                c_idx = (c_idx[0]+m_subidx, c_idx[1])
                pass
            elif 0 == first_input_tiling[1] and 0 != first_input_tiling[0]:
                # A add, C subidx
                c_subidx = m_subidx
                a_idx = (a_idx[0]+m_subidx, a_idx[1])
                pass
            else:
                raise NotImplementedError("Can't handle this tiling")
        elif 0 not in first_input_tiling and 0 in last_input_tiling:
            if 0 == last_input_tiling[0] and 0 != last_input_tiling[1]:
                # C subidx, B add
                c_subidx = n_subidx
                b_idx = (b_idx[0], b_idx[1]+n_subidx)
                pass
            elif 0 == last_input_tiling[1] and 0 != last_input_tiling[0]:
                # B subidx, C add
                b_subidx = n_subidx
                c_idx (c_idx[0], c_idx[1]+n_subidx)
                pass
            else:
                raise NotImplementedError("Can't handle this tiling")
        elif (0 not in first_input_tiling and 0 not in last_input_tiling) or \
             (0 in first_input_tiling and 0 in last_input_tiling):
            pass
        else:
            raise NotImplementedError("Can't handle this tiling")

        indices = [a_idx,b_idx,c_idx]

        target_offsets = { cname : self.offset_mappers[cname].map_tile_idx(tile,idx) for  cname,tile,idx in\
                zip(cnames,tiles,indices)}

        tiles = {c : t for c,t in zip(cnames,tiles)}

        result = []
        res_indices = {c : None for c in cnames}




        
        ordered_components = [c for c in self.resolve_order if c in cnames]
        for component in ordered_components:

            toff = target_offsets[component]
            dos = state.cdos[component]
            res_count = self.res_counts[component]
            t = tiles[component]
            res_idx = None

            #result.append(lsc_debugmsg(f"idx {i}:{indices[i]} translated to offset {toff}"))

            self.debug(f"Looking for {toff} for component {component}")
            # check if any of the registers have the data
            for j in range(res_count):
                self.debug(f"RES:{component}{j} has {dos[j]}")
                if dos[j] is None:
                    continue
                self.debug(f"toff-caoff ={toff-dos[j]}")
                if (toff == dos[j]):
                    self.debug(f"RES:{component}{j} has target {toff}")
                    res_idx = j
                    #break

            # use the current index for this input, use the next next time
            if res_idx is None:
                res_idx = state.res_indices[component]
                state.res_indices[component] = (res_idx+self.res_steps[component]) \
                                          % self.res_counts[component]

            res_indices[component] = res_idx
            #print(f"resolving {toff} into ADDR:{component}{res_idx}")
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

        state.states[cnames[2]][creg] = lsc_state.modified
        result.append(
                lsc_transformation(op=op,
                    res_indices=lsc_indices,
                    tiles=[a_tile,b_tile,c_tile]))
        state.last_tile_used[cnames[0]] = a_tile
        state.last_tile_used[cnames[1]] = b_tile
        state.last_tile_used[cnames[2]] = c_tile
        return result

    def preload(self, ops : list[mm_op],
                next_ops : list[mm_op],
                preload_counts : dict[str,int],
                zero_components : list[str],
                ignore_components : list[str],
                zero_addrs : bool = False,
                update_addrs : bool = True,
                add_current_offsets : bool = False,
                ) -> list[lsc_operation]:

        for c in self.res_counts.keys():
            if preload_counts[c] > self.res_counts[c]:
                raise ValueError(
                    f"{preload_counts[c]} preloads into {self.res_counts[c]} resources for {c}")

        state = self.states[self.current_state]
        self.ar.current_offsets = state.acdos

        results = []
        preloads_done = {c : 0 for c in self.res_counts.keys()}
        preload_states = copy.deepcopy(state.states)
        for c in ignore_components:
            if c not in preload_counts:
                continue
            # treat ignored as already preloaded
            preloads_done[c] = preload_counts[c]+1
            # ignores modified state
            state.states[c] = {i : lsc_state.loaded \
                    for i in range(self.res_counts[c])}



        # Trick: We save the tracked data offsets before the preload
        #        and then restore them, modifying only those that
        #        the preload should affect
        initial_cdos = copy.deepcopy(state.cdos)
        state.cdos = { c : { i : None for i in range(res_count)} for \
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
                (preload_counts[component]) == preloads_done[component]
        while any([preload_counts[c]+1 > d for c,d in preloads_done.items()])\
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
                pc = preload_counts[component]
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
                    self.debug(f"caoff for ADDR:{component}{op.addr_idx.indices[0]}: {caoff}")
                    # Some updates to offsets are implicit, therefore update
                    new_do = caoff + op.off

                    if is_next_preload(component):
                        #print(f"Next offset for {component}: {new_do}")
                        preload_next_offsets[component] = new_do
                        preloads_done[component] += 1
                        continue
                    self.debug(f"New do for RES:{component}{op.res_idx.indices[0]}: {new_do}")
                    preload_dos[component][op.res_idx.indices[0]] = new_do
                    preload_addr_reg_last_used_tile[component][op.addr_idx.indices[0]] = op.t

                    preloads_done[component] += 1
                    if component in zero_components:
                        subresults.append(lsc_zero(
                            component=component, res_idx=op.res_idx.indices[0], t=op.t))
                    else:
                        preload_addr_reg_offsets[component][op.addr_idx.indices[0]] = caoff
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
                    t = state.last_tile_used[add.component]
                results.append(lsc_addr_add(component=add.component,
                                            addr_idx=add.addr_idx,
                                            off=add.offset,
                                            t=t))
                # Add to tracked offset
                self.ar.current_offsets[add.component][add.addr_idx] += add.offset

        def print_cdos(dos):
            for c,do in dos.items():
                self.debug(f"{c}:")
                for idx,off in do.items():
                    self.debug(f"  {idx}:{off}")
        self.debug("cdos")
        print_cdos(state.cdos)
        self.debug("initial cdos")
        print_cdos(initial_cdos)
        self.debug("preload cdos")
        print_cdos(preload_dos)

        # use the original dos and assign the preloaded data
        state.cdos = copy.deepcopy(initial_cdos)
        for c,dos in preload_dos.items():
            for res_idx,orig in dos.items():
                state.cdos[c][res_idx] = orig



        # set the reg states
        state.states = copy.deepcopy(preload_states)

        # reset the indices
        state.res_indices = { c : d % self.res_counts[c] for c,d in\
                preload_counts.items()}
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
