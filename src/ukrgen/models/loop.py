# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from enum import Enum

from typing import Type,Callable

from .load_store_operations import lsc_operation


class lsc_comparison:

    def __init__(self, cmp : comparison):
        self.cmp = cmp

    cmp_strings = {
            'nz' : "!= 0",
            'ez' : "== 0",
            'eq' : "==",
            'le' : "<=",
            'ge' : ">=",
            'lt' : "<",
            'gt' : ">"
            }
    def __str__(self):
        return lsc_comparison.cmp_strings[self.cmp.name]

    def __repr__(self):
        return self.__str__()

class lsc_condition:
    def __init__(self, first : str, second : str,
                 comparison : lsc_comparison):
        self.first = first
        self.second = second
        self.comparison = comparison

    def __str__(self):
        result = f"{self.first} {self.comparison}"
        if self.second is not None:
            result += f" {self.second}"
        return result

    def __repr__(self):
        return self.__str__()

class lsc_loop_divergence:
    def __init__(self, name : str, ops : list[lsc_operation], condition : lsc_condition):
        self.name = name
        self.ops = ops
        self.condition = condition

class lsc_loop(lsc_operation):
    def __init__(self, name : str, condition : lsc_condition, level=1):
        self.name = name
        self.condition = condition
        self.level = level

        self.block = []
        self.divergences = []

        #TODO: populating this depending on the block (+ conditional branches) might be useful
        super().__init__(tiles=[], indices=[], reads=[], writes=[], reg_types=[])

    def add_block(self, ops : list[lsc_operation]):
        self.block.extend(ops)

    def add_singleshot_divergence(self, 
                                  name : str,
                                  ops : list[lsc_operation],
                                  condition : lsc_condition):

        self.divergences.append(
                lsc_loop_divergence(name=name, ops=ops, condition=condition))


    def __str__(self):
        result = ""

        indent = lambda i : (i+self.level)*"  "

        for div in reversed(self.divergences):
            result += f"{indent(0)}<-LABEL:{self.name}_{div.name}"
            result += f"\n{indent(1)}" + f"\n{indent(1)}".join(
                    [str(op) for op in div.ops])
        
        result += f"{indent(0)}<-LABEL:{self.name}"
        result += f"\n{indent(1)}" + f"\n{indent(1)}".join([str(op) for op in self.block])

        for div in self.divergences:
            result += f"\n{indent(1)}if {div.condition} -->LABEL:{self.name}_{div.name}"
        result += f"\n{indent(1)}if {self.condition} -->LABEL:{self.name}"

        return result

    def __repr__(self):
        return self.__str__()
