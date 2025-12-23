# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from asmgen.registers import asm_data_type as adt
from ..models.load_store_cpu import load_store_cpu
from ..models.load_store_operations import lsc_operation
from ..models.offset_mapper import offset_mapper
from ..specializers.asm import lsc_specializer,op_support
from ..generators.mm import mm_op

from .stage_param import stage_param

class gemm_context:
    """
    Contains data, parameters and structures involved in the generation of a gemm kernel
    as well as it's state through the different composition stages
    """
    def __init__(self):
        self.gen             : asmgen = None
        self.rt              : register_tracker = None
        self.params          : dict[str,stage_param] = dict()

        self.model           : load_store_cpu = None
        self.specializer     : lsc_specializer = None
        self.op_support_list : list[op_support] = list()
        self.component_dts   : dict[str,adt] = dict()
        self.sup             : op_support = None
        self.mappers         : dict[str,offset_mapper] = dict()
        self.strides         : dict[str,tuple[int|None,int|None]] = dict()
        self.mru_map         : dict[str,tuple[list[str],list[str]]] = dict()
        self.sched_map       : dict[str,tuple[list[str],bool]] = dict()
        self.speciazation_order : list[str] = list()

        self.tifs            : dict[str,list[mm_op]] = dict()
        self.irs             : dict[str,list[lsc_operation]] = dict()
        self.asmblocks       : dict[str,str] = dict()

