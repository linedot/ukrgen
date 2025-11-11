# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

# See: addr_resolver.md

import string
from typing import Self
from copy import deepcopy

from .load_store_operations import lsc_offset
from ..components.tile import tile

from .addr.reg_selectors import interleaving_selector,phasing_selector

class addr_add:
    def __init__(self, component : str, addr_idx : int,
                 offset : lsc_offset):
        self.component = component
        self.addr_idx = addr_idx
        self.offset = offset

    def __eq__(self, other : Self):
        return self.component == other.component and \
               self.addr_idx == other.addr_idx and \
               self.offset == other.offset

    def __str__(self):

        return f"{self.component}a{self.addr_idx} += {self.offset}"

    def __repr__(self):
        return self.__str__()

class addr_resolver:
    def __init__(self,
                 indices : dict[str,list[int]],
                 starting_offsets : dict[str,list[lsc_offset]],
                 offset_ranges : dict[str,list[tuple[lsc_offset,lsc_offset]]],
                 steps : dict[str,list[lsc_offset]],
                 phasing_components : list[str] = ["AB","C"],
                 max_incs : int = 2
                 ):
        for c,slist in starting_offsets.items():
            for off in slist:
                if not isinstance(off, lsc_offset):
                    raise ValueError(f"starting offset for {c} not lsc_offset")

        for c,rlist in offset_ranges.items():
            for minoff,maxoff in rlist:
                if not isinstance(minoff, lsc_offset)\
                   or not isinstance(maxoff, lsc_offset):
                    raise ValueError(f"offset ranges for {c} not lsc_offset")

        for c,slist in steps.items():
            for off in slist:
                if not isinstance(off, lsc_offset):
                    raise ValueError("step offsets for {c} not lsc_offset")

        self.indices=indices
        self.starting_offsets=starting_offsets
        self.current_offsets=deepcopy(self.starting_offsets) 
        self.last_resolved_offsets={ c : { i : None \
                for i in idx_list} \
                for c,idx_list in self.indices.items()}
        self.offset_ranges=offset_ranges
        self.steps = steps
        self.max_incs = max_incs


        self.candidate_selectors = {
                c : phasing_selector() if c in phasing_components \
                        else interleaving_selector() for c in \
                        self.indices.keys()}

    def zero_current_offsets(self):
        zo = lsc_offset.zero_offset()
        self.current_offsets = { c : [zo for i in offlist ] for \
                c,offlist in self.current_offsets.items()}

    def get_addr_adds_for_new_offsets(self,
                                      start_list : dict[str,lsc_offset] = None):
        if None == start_list:
            new_offsets=deepcopy(self.starting_offsets)
        else:

            # start_list determines the offset for the first addr reg for each component
            # self.starting_offsets are added to that value for the corresponding register
            
            new_offsets = { c :
                [first+offset if offset else first for \
                        offset in self.starting_offsets[c]] \
                    for c,first in start_list.items()}

        result = []

        for component in self.current_offsets.keys():
            newlist = new_offsets[component]
            oldlist = self.current_offsets[component]

            for idx,coff,toff in zip(self.indices[component],oldlist,newlist):
                if toff != coff:
                    result.append(addr_add(component=component,
                                           addr_idx=idx,
                                           offset=toff-coff))

        return result


    def toff_in_range(self, caoff : lsc_offset, toff : lsc_offset,
                      offset_range : tuple[lsc_offset,lsc_offset]):

        lsc_offset.adjust_offlists(caoff,offset_range[0])
        lsc_offset.adjust_offlists(caoff,offset_range[1])
        lsc_offset.adjust_offlists(caoff,toff)

        # sxv offsets have to be equal
        for sxv in caoff.sxv_strides.keys():
            if caoff.sxv_strides[sxv] != toff.sxv_strides[sxv]:
                return False

        immoffset_in_range = \
            (toff.immoff >=(caoff.immoff-offset_range[0].immoff)) and \
            (toff.immoff <=(caoff.immoff+offset_range[1].immoff))

        voffset_in_range = all([(toff >= (coff - minoff)) and \
                                (toff <= (coff + maxoff)) for \
            toff,coff,minoff,maxoff in \
                zip(toff.vlen_strides,
                    caoff.vlen_strides,
                    offset_range[0].vlen_strides,
                    offset_range[1].vlen_strides)])

        roffset_in_range = all([(toff >= (coff - minoff)) and \
                                (toff <= (coff + maxoff)) for \
            toff,coff,minoff,maxoff in \
                zip(toff.reg_strides,
                    caoff.reg_strides,
                    offset_range[0].reg_strides,
                    offset_range[1].reg_strides)])

        return immoffset_in_range and voffset_in_range and roffset_in_range
        

    def resolve_addr(self, 
                     component : str,
                     toff : lsc_offset) -> tuple[list[addr_add], int, lsc_offset]:
        """
        :rtype: tuple[list[addr_add],int,lsc_offset]
        :return: tuple consisting of a list of addr_add structure for address 
                 registers that need to be incremented, the index of the addr
                 register to use in the self.indices[component] list and the
                 offset to use it with
        """

        addr_adds = []

        addr_reg_count = len(self.indices[component])
        addr_idx_to_use = None
        incs_to_do = self.max_incs
        off = lsc_offset.zero_offset()
        
        #print(f"Resolving {toff} for {component}")
        
        while addr_idx_to_use is None and 0 < incs_to_do:

            # Start with first reg
            best_candidate_idx = 0
            current_offsets = self.current_offsets[component]
            offset_min = current_offsets[best_candidate_idx]

            self.candidate_selectors[component].reset(
                            current_offset=offset_min,
                            target_offset=toff,
                            offset_range=self.offset_ranges[component][best_candidate_idx])

            # if the first one is also in range, we can use it
            if self.toff_in_range(caoff=offset_min,
                                  toff=toff,
                                  offset_range=self.offset_ranges[component][best_candidate_idx]):
                addr_idx_to_use = best_candidate_idx
                off = toff-offset_min
                #print(f"will use ADDR:{component}{addr_idx_to_use} (first)")

            # check if we can address the data with one of the address registers
            # + immediate/vector offset
            for addr_list_idx in range(addr_reg_count):
                caoff = current_offsets[addr_list_idx]

                # ignore registers containing the same offset
                if addr_list_idx != best_candidate_idx and \
                   caoff == current_offsets[best_candidate_idx]:
                    continue
                offset_range = self.offset_ranges[component][addr_list_idx]

                #NOTE: This causes the interleaving behaviour
                #if caoff < offset_min:
                #    offset_min = caoff
                #    best_candidate_idx = addr_list_idx
                #    print(f"Smallest offset on ADDR:{component}{addr_list_idx}")
                if(self.candidate_selectors[component](
                            current_offset=caoff,
                            target_offset=toff,
                            offset_range=offset_range)):
                    best_candidate_idx = addr_list_idx
                    #print(f"Selected candidate ADDR:{component}{addr_list_idx}: {caoff}")
                    if self.toff_in_range(
                            caoff=caoff,
                            toff=toff,
                            offset_range=offset_range):
                        #print(f"Candidate in range of {toff}: ADDR:{component}{addr_list_idx}: {caoff}")
                        addr_idx_to_use = best_candidate_idx
                        off = toff-caoff
                #else:
                    #print(f"Equal or worse candidate ADDR:{component}{addr_list_idx}: {caoff}")
            if not addr_idx_to_use is None:
                break

            incs_to_do -= 1
            if incs_to_do <= 0:
                print("current offsets:")
                for addr_list_idx in range(addr_reg_count):
                    caoff = current_offsets[addr_list_idx]
                    print(f"  {component}a{addr_list_idx}: {caoff}")

                print(f"Target offset: {toff}")
                print(f"offset ranges:")
                for addr_list_idx in range(addr_reg_count):
                    minoff = self.offset_ranges[component][addr_list_idx][0]
                    maxoff = self.offset_ranges[component][addr_list_idx][1]
                    print(f"  {component}a{addr_list_idx} : [{minoff},{maxoff}]")
                raise RuntimeError(f"current offsets not in range after {self.max_incs} address adds")

            # TODO: Unmessify this (perhaps a set of allowed steps?)
            if toff.sxv_strides or \
               toff.reg_strides or \
               (toff == lsc_offset.zero_offset()):
                add_value = toff - self.current_offsets[component][best_candidate_idx]
            else:
                add_value = self.steps[component][best_candidate_idx]
            #print(f"Adding {add_value} to {component}a{best_candidate_idx}")
            new_add = addr_add(component=component,
                               addr_idx=self.indices[component][best_candidate_idx],
                               offset=add_value)
            addr_adds.append(new_add)

            self.current_offsets[component][best_candidate_idx] += add_value


        self.last_resolved_offsets[component][addr_idx_to_use] = off

        return addr_adds,addr_idx_to_use,off
