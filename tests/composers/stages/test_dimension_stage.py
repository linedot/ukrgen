# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.composers.stages.composition import composition_stage
from ukrgen.composers.stages.support import support_stage
from ukrgen.composers.stages.datatype import datatype_stage
from ukrgen.composers.stages.dimension import dimension_stage

from ukrgen.composers.gemm import gemm_context
from ukrgen.composers.stage_engine import stage_engine

from ukrgen.components.tile import scalar_dp,vla_vector

from .inject_params import inject_params

class test_dimension_stage(unittest.TestCase):
    def test_rvv_fma(self):        

        ukr_ctx = gemm_context()

        params = {
            "isa" : "rvv",
            "op" : "fma",
            "AB-data-type" : "SINGLE",
            "C-data-type" : "SINGLE",
            "variant" : 1,
            "m" : 2,
            "n" : 12,
            "k" : 8
        }


        stages = [support_stage,datatype_stage,dimension_stage]

        se = stage_engine(stages=stages,
                          ctx=ukr_ctx,
                          prolog=lambda s : inject_params(s, params))

        se.run()

        self.assertEqual(ukr_ctx.params["ma"].value, 2)
        self.assertEqual(ukr_ctx.params["mc"].value, 2)
        self.assertEqual(ukr_ctx.params["nb"].value, 12)
        self.assertEqual(ukr_ctx.params["nc"].value, 12)


        self.assertTrue(ukr_ctx.sup.a_tile.is_vla_vector)
        self.assertTrue(ukr_ctx.sup.b_tile.is_scalar)
        self.assertTrue(ukr_ctx.sup.c_tile.is_vla_vector)
