# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.composers.stages.support import support_stage
from ukrgen.composers.stages.datatype import datatype_stage
from ukrgen.composers.stages.variant import variant_stage
from ukrgen.composers.gemm import gemm_context

from ukrgen.components.tile import vla_vector,tile,scalar_dp
from ukrgen.specializers.asm import op_support

from asmgen.registers import adt_triple,asm_data_type as adt

class test_variant_stage(unittest.TestCase):
    def test_rvv_fma(self):

        ukr_ctx = gemm_context()

        stage1 = support_stage(context=ukr_ctx)

        stage1.set_param("isa", "rvv")
        stage1.set_param("op", "fma")

        stage1.progress()

        stage2 = datatype_stage(context=ukr_ctx)

        stage2.set_param("AB-data-type", "SINGLE")
        stage2.set_param("C-data-type", "SINGLE")

        stage2.progress()

        self.assertEqual(ukr_ctx.sup, None)

        stage3 = variant_stage(context=ukr_ctx,
                               op_support_list=ukr_ctx.op_support_list)

        self.assertEqual(stage3.get_param_choices("variant"),[0,1])

        stage3.set_param("variant", 1)

        stage3.progress()

        vector_tile = tile(dima=vla_vector,dimb=scalar_dp)

        expected_sup = op_support(
                adt_triple(adt.FP32,adt.FP32,adt.FP32),
                a_tile=vector_tile,
                b_tile=vector_tile,
                c_tile=vector_tile,
                target_op="fma")

        # TODO: implement __eq__ in tile for proper comparison 
        #       (might also be useful in other places)
        self.assertEqual(str(ukr_ctx.sup), str(expected_sup))
