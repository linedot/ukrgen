import abc

from asmgen.asmblocks.noarch import asmgen

from ..load_advancer import load_advancer

class loader:
    
    @abc.abstractmethod
    def process(self) -> str:
        raise NotImplementedError("Attempted to call base class method")
    

class loader_vector_voff(loader):
    def __init__(self, 
            asmgen       : asmgen,
            load_advancer: load_advancer):
        self.load_advancer = load_advancer

    def process(self) -> str:

        asmgen = self.load_advancer.asmgen

        li = self.load_advancer.loadinfo

        asmblock  = asmgen.load_vector_voff(
            asmgen.greg(li.address_register), 
            li.address_offset, 
            asmgen.vreg(li.vector_register),
            li.datatype)
        asmblock += self.load_advancer.advance()

        return asmblock

