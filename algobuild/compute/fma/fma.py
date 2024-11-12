from asmgen.asmblocks.noarch import asmgen,asm_data_type

import abc

from dataclasses import dataclass

@dataclass
class fma_info:
    a_data_register : int = 0,
    b_data_register : int = 1,
    c_data_register : int = 2,
    datatype          : asm_data_type = asm_data_type.FP64

class fma:
    def __init__(self, asmgen : asmgen, fma_info : fma_info):
        self.asmgen = asmgen
        self.fma_info = fma_info
    @abc.abstractmethod
    def process(self):
        raise NotImplementedError("Attempted to call base class fma.process() method")
