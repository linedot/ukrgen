from asmgen.asmblocks.noarch import asmgen,asm_data_type,reg_tracker

import abc

class offset_advancer:
    @abc.abstractmethod
    def advance(self, asmgen : asmgen, address_register : int, offset : int) -> str:
        raise NotImplementedError("Trying to call base class method")

class simple_offset_advancer(offset_advancer):
    def advance(self, asmgen : asmgen, address_register : int, offset: int) -> str:
        asmblock = asmgen.add_greg_imm(asmgen.greg(address_register), offset)
        return asmblock

class vector_offset_advancer(offset_advancer):
    def __init__(self, datatype : asm_data_type):
        self.datatype = datatype
    def advance(self, asmgen : asmgen, address_register : int, offset: int) -> str:
        asmblock = asmgen.add_greg_voff(asmgen.greg(address_register), offset, self.datatype)
        return asmblock

class greg_mul_offset_advancer(offset_advancer):
    def __init__(self, multiplier_register : int, register_tracker : reg_tracker):
        self.multiplier_register = multiplier_register
        self.register_tracker = register_tracker
    def advance(self, asmgen : asmgen, address_register : int, offset: int) -> str:
        spare_reg = self.register_tracker.reserve_any_greg()

        asmblock  = asmgen.mul_greg_imm(asmgen.greg(spare_reg),asmgen.greg(self.multiplier_register), offset)
        asmblock += asmgen.add_greg_greg(asmgen.greg(address_register),
                                         asmgen.greg(address_register),
                                         asmgen.greg(self.multiplier_register))

        self.register_tracker.unuse_greg(spare_reg)
        return asmblock

class greg_offset_advancer(offset_advancer):
    def __init__(self, offset_register : int):
        self.offset_register = offset_register
    def advance(self, asmgen : asmgen, address_register : int, offset: int) -> str:
        # Offset must be 1 (as in 1xwhat's int the register)
        assert(1 == offset, "non-1 offset with greg_offset_advancer, check your code")

        return asmgen.add_greg_greg(asmgen.greg(address_register),
                                         asmgen.greg(address_register),
                                         asmgen.greg(self.offset_register))
