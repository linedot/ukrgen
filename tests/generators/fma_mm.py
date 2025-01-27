import unittest

from algobuild.load import (
        vector_load_generator,
        lrsg_simple,
        register_state_tracker,
        register_state as rs,
        simple_scheduler,
        greg_offset_rollover_tracker)

from algobuild.compute.matmul import mm_rsg_afirst
from algobuild.generators import fma_mm as fma_mm_gen

from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.noarch import asm_data_type as adt
from asmgen.asmblocks.noarch import reg_tracker

class test_fma_mm_2vx4(unittest.TestCase):
    def test_fp32_allvregs(self):
        dt = adt.SINGLE
        gen = rvv()
        address_register_count = 2
        a_data_register_count = 4
        a_preload_count = 2
        b_data_register_count = 6
        b_preload_count = 4
        c_data_register_count = 8
        m = 2
        n = 4
        unroll_factor = 6
        use_fma_vf = True

        mmgen = fma_mm_gen(dt=dt, gen=gen,
                       abc_address_register_counts=[2, 2, 2],
                       address_offset_splits=[1,31,1],
                       abc_data_register_counts=[4,8,8],
                       abc_preload_counts=[4,4,8],
                       compute_rsg=mm_rsg_afirst,
                       abc_lrsg_types=[lrsg_simple for i in range(3)],
                       m=2, n=4, unroll_factor=6,
                       use_fma_vf = True)

        mmgen.generate()

