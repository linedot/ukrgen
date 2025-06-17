import unittest

from ukrgen.components import (
        dimension_properties,
        dimension_type,
        simple_ukr_tile,
        storage_type,
        tile,
        scalar_dp,
        scalar_tile,
        vla_vector,
        x4_vector
        )
from ukrgen.generators import mm,order2D
from ukrgen.models import load_store_cpu

from ukrgen.specializers.asm import lsc_specializer


from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve

from asmgen.registers import reg_tracker

class test_lsc_specializer(unittest.TestCase):
    def test_addr_add_analysis(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(vla_vector,scalar_dp))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(scalar_dp,scalar_dp))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(vla_vector,scalar_dp))

        mmgen = mm(a_tile, b_tile, c_tile, opstr="fma")


        mm_ops = mmgen.generate()
        ac_mapper = lambda tile, idx : m*idx[1]+idx[0]
        b_mapper = lambda tile, idx : n*idx[0]+idx[1]
        

        machine = load_store_cpu(
                     res_counts=[4,4,8],
                     res_steps=[1,1,1],
                     addr_counts=[2,1,1],
                     addr_offset_ranges=[[(0,0),(0,0)],[(0,7)],[(0,0)]],
                     addr_starts=[[0,1],[0],[0]],
                     preload_counts=[2,3,8],
                     offset_mappers = [ac_mapper,b_mapper,ac_mapper])

        mm_ops_next = mmgen.generate(add_dims=[0,0,0,0,0,k])

        preload = machine.preload(mm_ops)
        mainblock = machine(mm_ops)
        storeblock = machine.store_modified()
        preload_mb = machine.preload(mm_ops_next,
                                       zero_addrs=False,
                                       ignore_dims=[2])

        gen = rvv()

        rt = reg_tracker(
                reg_type_init_list=[
                    ('greg', gen.max_gregs),
                    ('freg', gen.max_fregs),
                    ('vreg', gen.max_vregs)
            ])

        specializer = lsc_specializer(model=machine, gen=gen, rt=rt)

        #print("Generator supports following operations:")
        #for k,v in specializer.op_support_map.items():
        #    print(f"Operation {k}:")
        #    for sup in v:
        #        print(f"{sup}")

        specializer.analyse(ops=preload+mainblock+storeblock+preload_mb)

        self.assertEqual(specializer.byteadds, set([8,3]))
        self.assertEqual(specializer.vlenadds, set([2]))
