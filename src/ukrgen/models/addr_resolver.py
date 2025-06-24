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

class addr_add:
    def __init__(self, rtype_idx : int, addr_idx : int,
                 offset : lsc_offset):
        self.rtype_idx = rtype_idx
        self.addr_idx = addr_idx
        self.offset = offset

    def __eq__(self, other : Self):
        return self.rtype_idx == other.rtype_idx and \
               self.addr_idx == other.addr_idx and \
               self.offset == other.offset

    def __str__(self):

        rtype_char = string.ascii_lowercase[self.rtype_idx]
        return f"{rtype_char}a{self.addr_idx} += {self.offset}"

    def __repr__(self):
        return self.__str__()

class addr_resolver:
    def __init__(self,
                 indices : list[list[int]],
                 starting_offsets : list[list[lsc_offset]],
                 offset_ranges : list[list[tuple[lsc_offset,lsc_offset]]],
                 steps : list[list[lsc_offset]],
                 max_incs : int = 2
                 ):
        for slist in starting_offsets:
            for off in slist:
                if not isinstance(off, lsc_offset):
                    raise ValueError("starting offsets are not lsc_offset")

        for rlist in offset_ranges:
            for minoff,maxoff in rlist:
                if not isinstance(minoff, lsc_offset)\
                   or not isinstance(maxoff, lsc_offset):
                    raise ValueError("offset ranges are not lsc_offset")

        for slist in steps:
            for off in slist:
                if not isinstance(off, lsc_offset):
                    raise ValueError("step offsets are not lsc_offset")

        self.indices=indices
        self.starting_offsets=starting_offsets
        self.current_offsets=deepcopy(self.starting_offsets) 
        self.offset_ranges=offset_ranges
        self.steps = steps
        self.max_incs = 2


    def toff_in_range(self, caoff : lsc_offset, toff : lsc_offset,
                      offset_range : tuple[lsc_offset,lsc_offset]):


        lsc_offset.adjust_offlists(caoff,toff)
        lsc_offset.adjust_offlists(caoff,offset_range[0])
        lsc_offset.adjust_offlists(caoff,offset_range[1])

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
                     rtype_idx : int,
                     toff : lsc_offset) -> tuple[list[addr_add], int, lsc_offset]:
        """
        :rtype: tuple[list[addr_add],int,lsc_offset]
        :return: tuple consisting of a list of addr_add structure for address 
                 registers that need to be incremented, the index of the addr
                 register to use in the self.indices[rtype_idx] list and the
                 offset to use it with
        """

        addr_adds = []

        addr_reg_count = len(self.indices[rtype_idx])
        addr_idx_to_use = -1
        incs_to_do = self.max_incs
        off = lsc_offset(reg_strides=[],
                         vlen_strides=[],
                         immoff=0)
        while -1 == addr_idx_to_use and 0 < incs_to_do:

            # Start with first reg
            caoff_min_list_idx = 0
            current_offsets = self.current_offsets[rtype_idx]
            offset_min = current_offsets[caoff_min_list_idx]

            # check if we can address the data with one of the address registers
            # + immediate/vector offset
            for addr_list_idx in range(addr_reg_count):
                caoff = current_offsets[addr_list_idx]
                offset_range = self.offset_ranges[rtype_idx][addr_list_idx]

                
                if caoff < offset_min:
                    offset_min = caoff
                    caoff_min_list_idx = addr_list_idx
                
                if self.toff_in_range(caoff=caoff,
                                      toff=toff,
                                      offset_range=offset_range):
                    off = toff-caoff
                    addr_idx_to_use = addr_list_idx
                    break


            if -1 != addr_idx_to_use:
                break


            if incs_to_do <= 0:
                raise RuntimeError(f"Target offset not in range after {self.max_incs} address adds")

            # this could be an alternative cotnrolled by a parameter?
            # add_value = toff - offset_min

            add_value = self.steps[rtype_idx][caoff_min_list_idx]
            new_add = addr_add(rtype_idx=rtype_idx,
                               addr_idx=self.indices[rtype_idx][caoff_min_list_idx],
                               offset=add_value)
            addr_adds.append(new_add)

            self.current_offsets[rtype_idx][caoff_min_list_idx] += add_value
            incs_to_do -= 1

        return addr_adds,addr_idx_to_use,off
