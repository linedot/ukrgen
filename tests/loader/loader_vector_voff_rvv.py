import unittest

from asmgen.asmblocks.noarch import asm_data_type,reg_tracker

from algobuild.offset_advancer import greg_offset_advancer
from algobuild.load_advancer import simple_load_advancer,loadinfo
from algobuild.loader import loader_vector_voff

class test_loader_vector_voff_rvv(unittest.TestCase):
    def test_32steps(self):
        from asmgen.asmblocks.rvv import rvv

        rt = reg_tracker(max_greg=20, max_vreg=32, max_freg=32)
        address_register = rt.reserve_specific_greg(0)
        vlen_register = rt.reserve_any_greg()
        li = loadinfo(address_register=0, address_offset=0, offset_step=1, vector_register=0, datatype=asm_data_type.FP64)
        asmgen = rvv()

        asmblock = asmgen.simd_size_to_greg(asmgen.greg(vlen_register), li.datatype)

        sla = simple_load_advancer(asmgen, li, [0,1], asmgen.max_load_voff, 
                                   greg_offset_advancer(vlen_register))


        loader = loader_vector_voff(
                asmgen=asmgen,
                load_advancer=sla)

        asmblock += "".join([loader.process() for i in range(32)])

        rt.unuse_greg(vlen_register)
        rt.unuse_greg(0)
        self.assertEqual(asmblock,
           ("\"csrr t1, vlenb\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v0, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"
            "\"vle64.v v1, (t0)\\n\\t\"\n"
            "\"add t0,t0,t1\\n\\t\"\n"))


