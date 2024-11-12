from . import load_advancer,loadinfo
from ..offset_advancer import simple_offset_advancer

from asmgen.asmblocks.noarch import asmgen

class simple_load_advancer(load_advancer):
    def __init__(self, asmgen: asmgen,
        loadinfo : loadinfo,
        target_registers : list[int],
        max_offset : int,
        offset_advancer : simple_offset_advancer):
        
        super().__init__(asmgen=asmgen, loadinfo=loadinfo)

        self.target_list_position = 0
        self.loadinfo.target_register = target_registers[self.target_list_position]
        self.target_registers = target_registers
        self.max_offset = max_offset
        self.offset_advancer = offset_advancer

    def advance(self) -> str:
        asmblock = ""
        self.loadinfo.address_offset += self.loadinfo.offset_step
        if self.loadinfo.address_offset > self.max_offset:
            advance_by = self.loadinfo.address_offset
            self.loadinfo.address_offset = 0
            asmblock += self.offset_advancer.advance(
                    asmgen=self.asmgen,
                    address_register=self.loadinfo.address_register,
                    offset=advance_by)
        self.target_list_position = (self.target_list_position+1) % len(self.target_registers)
        self.loadinfo.target_register = self.target_registers[self.target_list_position]
        return asmblock
