import unittest

from asmgen.asmblocks.noarch import asm_data_type

from algobuild.offset_advancer import vector_offset_advancer
from algobuild.load_advancer import simple_load_advancer,loadinfo
from algobuild.loader import loader_vector_voff

class test_loader_vector_voff_fma256(unittest.TestCase):
    def test_32steps(self):
        from asmgen.asmblocks.avx_fma import fma256
        li = loadinfo(address_register=0, address_offset=0, offset_step=1, vector_register=0, datatype=asm_data_type.FP64)
        asmgen = fma256()
        sla = simple_load_advancer(asmgen, li, [0,1], 16, 
                                   vector_offset_advancer(li.datatype))

        loader = loader_vector_voff(
                asmgen=asmgen,
                load_advancer=sla)

        asmcommands = "".join([loader.process() for i in range(32)])
        self.assertEqual(asmcommands, 
           ("\"vmovupd (%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 32(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 64(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 96(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 128(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 160(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 192(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 224(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 256(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 288(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 320(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 352(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 384(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 416(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 448(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 480(%%r8),%%ymm1\\n\\t\"\n"
            "\"addq $512,%%r8\\n\\t\"\n"
            "\"vmovupd (%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 32(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 64(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 96(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 128(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 160(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 192(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 224(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 256(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 288(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 320(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 352(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 384(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 416(%%r8),%%ymm1\\n\\t\"\n"
            "\"vmovupd 448(%%r8),%%ymm0\\n\\t\"\n"
            "\"vmovupd 480(%%r8),%%ymm1\\n\\t\"\n"
            "\"addq $512,%%r8\\n\\t\"\n"))
