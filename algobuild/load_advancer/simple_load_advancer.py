from . import load_advancer,loadinfo
from ..offset_advancer import simple_offset_advancer

from asmgen.asmblocks.noarch import asmgen

class simple_load_advancer(load_advancer):
    def __init__(self, asmgen: asmgen,
        loadinfo : loadinfo,
        vector_registers : list[int],
        max_offset : int,
        offset_advancer : simple_offset_advancer):
        
        super().__init__(asmgen=asmgen, loadinfo=loadinfo)

        self.vector_list_position = 0
        self.loadinfo.vector_register = vector_registers[self.vector_list_position]
        self.vector_registers = vector_registers
        self.max_offset = max_offset
        self.offset_advancer = offset_advancer

    def advance(self) -> str:
        asmblock = ""
        self.loadinfo.address_offset += self.loadinfo.offset_step
        if (self.loadinfo.address_offset + self.loadinfo.offset_step) > self.max_offset:
            advance_by = self.loadinfo.address_offset
            self.loadinfo.address_offset = 0
            asmblock += self.offset_advancer.advance(
                    asmgen=self.asmgen,
                    address_register=self.loadinfo.address_register,
                    offset=advance_by)
        self.vector_list_position = (self.vector_list_position+1) % len(self.vector_registers)
        self.loadinfo.vector_register = self.vector_registers[self.vector_list_position]
        return asmblock
