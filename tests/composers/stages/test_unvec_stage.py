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

class test_unvec_stage(unittest.TestCase):
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
            "k" : 8,
            "unvec-method" : "lane_bcast"
        }

        def get_param(stage: composition_stage, name : str) -> str:
            if name in params:
                return params[name]
            else:
                return stage.get_default_value(name)

        stages = [support_stage,datatype_stage,dimension_stage]

        se = stage_engine(stages=stages,
                          ctx=ukr_ctx,
                          get_param_callback=get_param)

        # TODO: change test after implementing other unvec methods
        with self.assertRaises(NotImplementedError) as exctx:
            se.run()
