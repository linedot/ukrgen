import unittest


from algobuild.components import (
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
from algobuild.generators import mm,order2D
from algobuild.generators.mm import string_mapper
from algobuild.models import load_store_cpu

class test_load_store_cpu(unittest.TestCase):
    def test_resolve_order(self):
        m = 2
        n = 4
        unroll_factor = 2
        k = unroll_factor
        use_fma_vf = True
        
        a_tile = simple_ukr_tile(a_size=m, b_size=k, subdims=(vla_vector,scalar_dp))
        b_tile = simple_ukr_tile(a_size=k, b_size=n, subdims=(scalar_dp,scalar_dp))
        c_tile = simple_ukr_tile(a_size=m, b_size=n, subdims=(vla_vector,scalar_dp))

        mmgen = mm(a_tile, b_tile, c_tile)
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
                     offset_mappers = [ac_mapper,b_mapper,ac_mapper],
                     resolve_order=[1,0,2])

        mm_ops_next = mmgen.generate(add_dims=[0,0,0,0,0,k])
        #print("\n".join(map(str,inspector(mm_ops_next))))

        preload = machine.preload(mm_ops)
        mainblock = machine(mm_ops)
        storeblock = machine.store_modified()
        preload_mb = machine.preload(mm_ops_next,
                                       zero_addrs=False,
                                       ignore_dims=[2])

        print("\n".join(map(str,preload)))
        print("MAIN LOOP -------------------------------")
        print("  "+"\n  ".join(map(str,mainblock)))
        print("PRELOAD NEXT ----------------------------")
        print("  "+"\n  ".join(map(str,preload_mb)))
        print("END MAIN LOOP ---------------------------")
        print("STOREBLOCK ------------------------------")
        print("\n".join(map(str,storeblock)))
        print("ENDSTOREBLOCK ---------------------------")
