# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging
import unittest

from ukrgen.composers.stages.composition import composition_stage
from ukrgen.composers.stages.support import support_stage
from ukrgen.composers.stages.datatype import datatype_stage
from ukrgen.composers.stages.dimension import dimension_stage
from ukrgen.composers.stages.tif import mm_tif_stage
from ukrgen.composers.stages.model import lsc_model_stage
from ukrgen.composers.stages.mru import lsc_mru_stage
from ukrgen.composers.stages.schedule import lsc_schedule_stage
from ukrgen.composers.stages.specialize import specialize_lsc_stage
from ukrgen.composers.stages.codegen import blis_ukr_codegen_stage

from ukrgen.composers.gemm import gemm_context
from ukrgen.composers.stage_engine import stage_engine

from ukrgen.components.tile import scalar_dp,vla_vector

class test_codegen_stage(unittest.TestCase):
    def test_rvv_fma(self):

        ukr_ctx = gemm_context()

        params = {
            "isa" : "rvv",
            "op" : "fma",
            "AB-data-type" : "SINGLE",
            "C-data-type" : "SINGLE",
            "variant" : 0,
            "m" : 2,
            "n" : 12,
            "k" : 8,
            "A-data-regs" : 4,
            "B-data-regs" : 8,
            "AB-data-regs" : 24,
            "C-data-regs" : 24,
            "A-addr-regs" : 2,
            "B-addr-regs" : 2,
            "C-addr-regs" : 2,
            "A-preload" : 4,
            "B-preload" : 8,
        }

        def get_param(stage: composition_stage, name : str) -> str:
            if name in params:
                return params[name]
            else:
                return stage.get_default_value(name)

        stages = [
            support_stage,
            datatype_stage,
            dimension_stage,
            mm_tif_stage,
            lsc_model_stage,
            specialize_lsc_stage,
            lsc_mru_stage,
            lsc_schedule_stage,
            blis_ukr_codegen_stage]

        se = stage_engine(stages=stages,
                          ctx=ukr_ctx,
                          get_param_callback=get_param)

        se.run()


        self.assertIn("full_function", ukr_ctx.asmblocks)
