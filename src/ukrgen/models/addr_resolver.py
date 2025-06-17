# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

# See: addr_resolver.md

from copy import deepcopy

from ..components.tile import tile

class addr_add:
    def __init__(self, rtype_idx, addr_idx, offset):
        self.rtype_idx = rtype_idx
        self.addr_idx = addr_idx,
        self.offset = offset

class addr_resolver:
    def __init__(self,
                 indices : list[list[int]],
                 starting_offsets : list[list[int]],
                 offset_ranges : list[list[tuple[int,int]]],
                 steps : list[list[int]]
                 ):
        self.indices=indices
        self.starting_offsets=starting_offsets
        self.current_offsets=deepcopy(self.starting_offsets) 
        self.offset_ranges=offset_ranges
        self.steps = steps

    def toff_in_range(self, caoff : int, toff : int,
                      offset_range : tuple[int,int]):
        return toff >= (caoff-offset_range[0]) and \
               toff <= (caoff+offset_range[1])

    def resolve_addr(self, 
                     rtype_idx : int,
                     toff : int,
                     t : tile) -> tuple[list[addr_add], int]:
        """
        :rtype: tuple[list[addr_add],int]
        :return: tuple consisting of a list of addr_add structure for address 
                 registers that need to be incremented and an int, indicating
                 index of the addr register to use in the self.indices list
        """

        addr_adds = []

        addr_reg_count = len(self.indices[rtype_idx])
        current_offsets = self.current_offsets[rtype_idx]
        addr_idx_to_use = -1
        while -1 == addr_idx_to_use:
            # TODO: better starting value
            offset_min = 99999999
            caoff_min_list_idx = -1
            # check if we can address the data with one of the address registers + immediate offset
            for addr_list_idx in range(addr_reg_count):
                caoff = current_offsets[addr_list_idx]
                offset_range = self.offset_ranges[rtype_idx][addr_list_idx]

                
                if caoff < offset_min:
                    offset_min = caoff
                    caoff_min_list_idx = addr_list_idx
                
                if self.is_in_range(caoff=caoff,
                               toff=toff,
                               offset_range=offset_range):
                    off = toff-caoff
                    addr_idx_to_use = addr_idx
            if -1 != addr_idx_to_use:
                break


            # this could be an alternative cotnrolled by a parameter?
            # add_value = toff - offset_min

            add_value = self.steps[rtype_idx][caoff_min_list_idx]
            new_add = addr_add(rtype_idx=rtype_idx,
                               addr_idx=self.indices[rtype_idx][caoff_min_list_idx],
                               offset=add_value)
            addr_adds.append(new_add)

            self.current_offsets[rtype_idx][caoff_min_list_idx] += add_value

        return addr_idx_to_use
