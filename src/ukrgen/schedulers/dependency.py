# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from ..models.load_store_operations import lsc_operation

from enum import Enum,auto


class dependency_type(Enum):
    RAR = auto()
    RAW = auto()
    WAR = auto()
    WAW = auto()

def get_dependencies(op1 : lsc_operation, op2: lsc_operation) ->\
        dict[dependency_type,set[reg_compare]]:


    op2_reads = {
        reg_compare(ttype=op2.reg_types[i],
                    index=op2.indices[i],
                    t=op2.tiles[i]) for i in op2.reads}
    op2_writes = {
        reg_compare(ttype=op2.reg_types[i],
                    index=op2.indices[i],
                    t=op2.tiles[i]) for i in op2.writes}

    op1_reads = {
        reg_compare(ttype=op1.reg_types[i],
                    index=op1.indices[i],
                    t=op1.tiles[i]) for i in op1.reads}
    op1_writes = {
        reg_compare(ttype=op1.reg_types[i],
                    index=op1.indices[i],
                    t=op1.tiles[i]) for i in op1.writes}


    return {
       dependency_type.RAR : set(op2_reads) & set(op1_reads),
       dependency_type.RAW : set(op2_reads) & set(op1_writes),
       dependency_type.WAR : set(op2_writes) & set(op1_reads),
       dependency_type.WAW : set(op2_writes) & set(op1_writes),
    }
