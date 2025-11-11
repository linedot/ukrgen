# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from enum import Enum

from typing import Type,Callable

from .load_store_operations import lsc_operation

from asmgen.registers import adt_triple, reg_tracker, asm_data_type as adt
from asmgen.asmblocks.noarch import asmgen, comparison


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


    # TODO: explore alternative: expand number of lsc_operation child classes to include
    #       comparisons and branches and handle this differently
    def transform(self,
                  gen : asmgen,
                  rt : reg_tracker,
                  subtransformers : dict[Type[lsc_operation],
                                    Callable[[lsc_operation,dict[str,adt]],str]],
                  component_dts : dict[str,adt]):

        indent = lambda i : (i+self.level)*"  "
        asmblock = ""
        for div in reversed(self.divergences):
            asmblock += indent(0) + gen.label(label = f"{self.name}_{div.name}")
            for op in div.ops:
                asmblock += indent(1) + subtransformers[type(op)](op,component_dts)

        
        cntr_idx = rt.aliased_regs["greg"][self.condition.first]
        asmblock += indent(0) + gen.label(label=self.name)

        for op in self.block:
            asmblock += indent(1) + subtransformers[type(op)](op,component_dts)

        for div in self.divergences:
            label = f"{self.name}_{div.name}"
            reg1 = gen.greg(rt.aliased_regs["greg"][div.condition.first])
            reg2 = None
            if div.condition.second is not None:
                reg2 = gen.greg(rt.aliased_regs["greg"][div.condition.second])
            asmblock += indent(1) + gen.cb(reg1 = reg1,
                                           reg2 = reg2,
                                           cmp=div.condition.comparison.cmp,
                                           label=label)


        reg1 = gen.greg(rt.aliased_regs["greg"][self.condition.first])
        reg2 = None
        if self.condition.second is not None:
            reg2 = gen.greg(rt.aliased_regs["greg"][self.condition.second])

        asmblock += indent(1) + gen.cb(reg1 = reg1,
                                       reg2 = reg2,
                                       cmp=self.condition.comparison.cmp,
                                       label=self.name)

        return asmblock


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
