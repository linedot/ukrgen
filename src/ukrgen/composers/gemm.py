from dataclasses import dataclass
from asmgen.registers import asm_data_type as adt
from ..specializers.asm import lsc_specializer,op_support


class gemm_context:
    def __init__(self):
        self.gen : asmgen = None
        self.rt : register_tracker = None
        self.params : dict[str,str] = dict()
        self.specializer : lsc_specializer = None
        self.op_support_list : list[op_support] = list()
        self.sup : op_support = None
        self.needs_unvec : bool = False


class blis_composer:
    pass
