# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest
from copy import deepcopy
from typing import Type


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
from ukrgen.models import load_store_cpu
from ukrgen.models.loop import lsc_loop,lsc_condition,lsc_comparison
from ukrgen.models.load_store_operations import (
        lsc_transformation,
        lsc_add_val_off,
        lsc_offset
        )
from ukrgen.models.load_store_cpu import load_store_cpu

from ukrgen.specializers.asm import lsc_specializer

from asmgen.asmblocks.noarch import comparison,asmgen
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.registers import reg_tracker,asm_data_type as adt,adt_triple


vla_vector_tile = tile(vla_vector, scalar_dp)

fp64_triple=adt_triple(
    adt.FP64,
    adt.FP64,
    adt.FP64)

class test_lsc_loop(unittest.TestCase):
    def setUp(self):
        self.fma_ops =[
            lsc_transformation(op="fma", 
                               res_indices=[0,1,0],
                               sub_indices=[None,None,None],
                               tiles=[vla_vector_tile,
                                      vla_vector_tile,
                                      vla_vector_tile]),
            lsc_transformation(op="fma", 
                               res_indices=[0,1,1],
                               sub_indices=[None,None,None],
                               tiles=[vla_vector_tile,
                                      vla_vector_tile,
                                      vla_vector_tile]),
            lsc_transformation(op="fma", 
                               res_indices=[0,1,2],
                               sub_indices=[None,None,None],
                               tiles=[vla_vector_tile,
                                      vla_vector_tile,
                                      vla_vector_tile]),
            lsc_transformation(op="fma", 
                               res_indices=[0,1,3],
                               sub_indices=[None,None,None],
                               tiles=[vla_vector_tile,
                                      vla_vector_tile,
                                      vla_vector_tile]),
        ]

        condition = lsc_condition(first="counter", second=None, 
                                  comparison=lsc_comparison(comparison.nz))
        self.simple_loop = lsc_loop(name="testloop", condition=condition)

    def setup_grs(self, gen : Type[asmgen]):

        generator = gen()
        rt = reg_tracker(reg_type_init_list=
                         [('greg', generator.max_gregs),
                          ('freg', generator.max_fregs),
                          ('vreg', generator.max_vregs),
                          ('treg', generator.max_tregs(dt=adt.FP64))])
        specializer = lsc_specializer(model=None, gen=generator, rt=rt)

        return generator,rt,specializer


    def test_fmablock(self):

        loop = deepcopy(self.simple_loop)
        loop.add_block(ops=self.fma_ops)
        loop.add_block(ops=[
            lsc_add_val_off("counter", off=lsc_offset({},[],[],-1))
            ])
        print(loop)

        gen,rt,specializer = self.setup_grs(rvv)

        counter_idx = rt.reserve_any_reg("greg")
        rt.alias_reg("greg", "counter", counter_idx)

        specializer.analyse([loop])
        sloop = specializer.pre_specialize([loop],triple=fp64_triple)

        asmblock  = specializer.code_init(triple=fp64_triple)
        asmblock += "\n".join(specializer.specialize(sloop,triple=fp64_triple))
        asmblock += specializer.code_fini(triple=fp64_triple)

        rt.unuse_reg("greg", counter_idx)

        print(asmblock)
        
    
    def test_divergence(self):
        loop = deepcopy(self.simple_loop)

        condition = lsc_condition(first="counter", second="pfdist",
                                  comparison=lsc_comparison(comparison.eq))
        loop.add_singleshot_divergence(name="prefetch", ops=[], condition=condition)

        loop.add_block(ops=[
            lsc_add_val_off("counter", off=lsc_offset({},[],[],-1))
            ])
        print(loop)

        gen,rt,specializer = self.setup_grs(rvv)

        counter_idx = rt.reserve_any_reg("greg")
        rt.alias_reg("greg", "counter", counter_idx)
        pfdist_idx = rt.reserve_any_reg("greg")
        rt.alias_reg("greg", "pfdist", pfdist_idx)

        specializer.analyse([loop])
        sloop = specializer.pre_specialize([loop],triple=fp64_triple)

        asmblock  = specializer.code_init(triple=fp64_triple)
        asmblock += "\n".join(specializer.specialize(sloop,triple=fp64_triple))
        asmblock += specializer.code_fini(triple=fp64_triple)

        rt.unuse_reg("greg", counter_idx)
        rt.unuse_reg("greg", pfdist_idx)

        print(asmblock)

