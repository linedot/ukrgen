import unittest

from asmgen.asmblocks.noarch import asm_data_type

from algobuild.load_advancer import simple_load_advancer,loadinfo
from algobuild.offset_advancer import simple_offset_advancer

class test_simple_load_advancer(unittest.TestCase):
    def test_5steps(self):
        from asmgen.asmblocks.avx_fma import fma256
        li = loadinfo(address_register=0, address_offset=0, offset_step=8, vector_register=0, datatype=asm_data_type.FP64)
        sla = simple_load_advancer(fma256(), li, [0,1], 16, 
                                   simple_offset_advancer())

        asmcommands = "".join([sla.advance() for i in range(5)])
        self.assertEqual(asmcommands, "\"addq $16,%%r8\\n\\t\"\n\"addq $16,%%r8\\n\\t\"\n")
