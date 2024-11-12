from asmgen.asmblocks.noarch import asmgen,asm_data_type

from dataclasses import dataclass

import abc

@dataclass
class loadinfo:
    address_register : int = 0
    address_offset   : int = 0
    offset_step      : int = 8
    target_register  : int = 0
    datatype         : asm_data_type = asm_data_type.FP64

class load_advancer:
    def __init__(self, 
        asmgen : asmgen,
        loadinfo : loadinfo):
        
        self.loadinfo = loadinfo
        self.asmgen = asmgen

    @abc.abstractmethod
    def advance(self):
        raise NotImplementedError("Tried to call abstract method")
