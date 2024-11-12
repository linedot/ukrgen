import unittest

from asmgen.asmblocks.noarch import asm_data_type,reg_tracker

from algobuild.offset_advancer import simple_offset_advancer
from algobuild.load_advancer import simple_load_advancer,loadinfo
from algobuild.loader import loader_fp_immoff

class test_loader_fp_immoff_rvv(unittest.TestCase):
    def test_8steps(self):
        from asmgen.asmblocks.rvv import rvv

        rt = reg_tracker(max_greg=20, max_vreg=32, max_freg=32)
        address_register = rt.reserve_specific_greg(0)
        li = loadinfo(address_register=0,
                      address_offset=0,
                      offset_step=8,
                      target_register=0,
                      datatype=asm_data_type.FP64)
        asmgen = rvv()

        sla = simple_load_advancer(asmgen, li, [0,1],
                                   asmgen.max_fload_immoff(li.datatype), 
                                   simple_offset_advancer())


        loader = loader_fp_immoff(
                asmgen=asmgen,
                load_advancer=sla)

        asmblock = "".join([loader.process() for i in range(8)])

        rt.unuse_greg(0)
        self.assertEqual(asmblock,(
            "\"fld f0, 0(t0)\\n\\t\"\n"
            "\"fld f1, 8(t0)\\n\\t\"\n"
            "\"fld f0, 16(t0)\\n\\t\"\n"
            "\"fld f1, 24(t0)\\n\\t\"\n"
            "\"fld f0, 32(t0)\\n\\t\"\n"
            "\"fld f1, 40(t0)\\n\\t\"\n"
            "\"fld f0, 48(t0)\\n\\t\"\n"
            "\"fld f1, 56(t0)\\n\\t\"\n"))

    def test_16steps_strided(self):
        from asmgen.asmblocks.rvv import rvv

        rt = reg_tracker(max_greg=20, max_vreg=32, max_freg=32)
        address_register = rt.reserve_specific_greg(0)
        li = loadinfo(address_register=0,
                      address_offset=0,
                      offset_step=512,
                      target_register=0,
                      datatype=asm_data_type.FP64)
        asmgen = rvv()

        sla = simple_load_advancer(asmgen, li, [0,1],
                                   asmgen.max_fload_immoff(li.datatype), 
                                   simple_offset_advancer())


        loader = loader_fp_immoff(
                asmgen=asmgen,
                load_advancer=sla)

        asmblock = "".join([loader.process() for i in range(16)])

        rt.unuse_greg(0)
        self.assertEqual(asmblock,(
            "\"fld f0, 0(t0)\\n\\t\"\n"
            "\"fld f1, 512(t0)\\n\\t\"\n"
            "\"fld f0, 1024(t0)\\n\\t\"\n"
            "\"fld f1, 1536(t0)\\n\\t\"\n"
            "\"add t0,t0,2048\\n\\t\"\n"
            "\"fld f0, 0(t0)\\n\\t\"\n"
            "\"fld f1, 512(t0)\\n\\t\"\n"
            "\"fld f0, 1024(t0)\\n\\t\"\n"
            "\"fld f1, 1536(t0)\\n\\t\"\n"
            "\"add t0,t0,2048\\n\\t\"\n"
            "\"fld f0, 0(t0)\\n\\t\"\n"
            "\"fld f1, 512(t0)\\n\\t\"\n"
            "\"fld f0, 1024(t0)\\n\\t\"\n"
            "\"fld f1, 1536(t0)\\n\\t\"\n"
            "\"add t0,t0,2048\\n\\t\"\n"
            "\"fld f0, 0(t0)\\n\\t\"\n"
            "\"fld f1, 512(t0)\\n\\t\"\n"
            "\"fld f0, 1024(t0)\\n\\t\"\n"
            "\"fld f1, 1536(t0)\\n\\t\"\n"
            "\"add t0,t0,2048\\n\\t\"\n"))


