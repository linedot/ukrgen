# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from dataclasses import dataclass
from asmgen.registers import asm_data_type as adt
from ..specializers.asm import lsc_specializer,op_support


from ..generators.mm import mm_op


class gemm_context:
    def __init__(self):
        self.gen : asmgen = None
        self.rt : register_tracker = None
        self.params : dict[str,str] = dict()
        self.specializer : lsc_specializer = None
        self.op_support_list : list[op_support] = list()
        self.sup : op_support = None

        self.tifs : dict[str,list[mm_op]] = dict()


class blis_composer:
    pass
