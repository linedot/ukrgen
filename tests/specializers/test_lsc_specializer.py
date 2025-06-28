# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

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
from ukrgen.models import load_store_cpu,addr_resolver
from ukrgen.models.load_store_operations import lsc_offset
from ukrgen.models.tile_offset_mapper import flat_mapper

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
        # Everything contiguous
        ac_mapper = flat_mapper(lambda t,idx : t.dima.size*m*idx[1]+idx[0])
        b_mapper = flat_mapper(lambda t,idx : t.dima.size*n*idx[0]+idx[1])


        zo = lsc_offset.zero_offset()
        o1v = lsc_offset([],[1],0)
        o2v = lsc_offset([],[2],0)
        o7i = lsc_offset([],[],7)
        o8i = lsc_offset([],[],8)
                
        ar = addr_resolver(indices=[[0,1],[0],[0]],
                           starting_offsets=[[zo,o1v],[zo],[zo]],
                           offset_ranges=[
                               [(zo,zo),(zo,zo)],
                               [(zo,o7i)],
                               [(zo,zo)]
                           ],
                           steps=[[o2v,o2v],
                                  [o8i],
                                  [o1v]
                           ])

        machine = load_store_cpu(
                     res_counts=[4,4,8],
                     res_steps=[1,1,1],
                     ar=ar,
                     preload_counts=[2,3,8],
                     offset_mappers = [ac_mapper,b_mapper,ac_mapper])

        mm_ops_p1k = mmgen.generate(add_dims=[0,0,0,0,0,k])
        mm_ops_p2k = mmgen.generate(add_dims=[0,0,0,0,0,k*2])

        preload = machine.preload(mm_ops,mm_ops_p1k)
        mainblock = machine(mm_ops)
        preload_mb = machine.preload(mm_ops_p1k,
                                     mm_ops_p2k,
                                     zero_addrs=False,
                                     ignore_dims=[2])
        storeblock = machine.store_modified()

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

        self.assertEqual(specializer.byteadds, 
                         set([
                             o8i,
                             lsc_offset([],[],3)]))
        self.assertEqual(specializer.vlenadds,
                         set([
                             o2v]))
