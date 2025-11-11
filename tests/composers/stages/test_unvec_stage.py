# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.composers.stages.support import support_stage
from ukrgen.composers.stages.datatype import datatype_stage
from ukrgen.composers.stages.variant import variant_stage
from ukrgen.composers.stages.dimension import dimension_stage
from ukrgen.composers.stages.unvec import unvec_stage
from ukrgen.composers.gemm import gemm_context

from ukrgen.components.tile import vla_vector,tile,scalar_dp,dimension_type
from ukrgen.specializers.asm import op_support

from asmgen.registers import adt_triple,asm_data_type as adt

class test_unvec_stage(unittest.TestCase):
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

        stage4 = dimension_stage(context=ukr_ctx)

        stage4.set_param("vecdir", "M")
        stage4.set_param("m", 2)
        stage4.set_param("n", 12)
        stage4.set_param("k", 8)
        stage4.set_param("order", "mnkMNK")

        stage4.progress()

        self.assertTrue(ukr_ctx.needs_unvec)

        stage5 = unvec_stage(context=ukr_ctx)

        stage5.set_param("unvec-method", "load_bcast")

        stage5.progress()

        self.assertEqual(ukr_ctx.sup.b_tile.dima.size, 1)
        self.assertEqual(ukr_ctx.sup.b_tile.dima.dt, dimension_type.fixed)
        self.assertEqual(ukr_ctx.sup.b_tile.dimb.size, 1)
        self.assertEqual(ukr_ctx.sup.b_tile.dimb.dt, dimension_type.fixed)
