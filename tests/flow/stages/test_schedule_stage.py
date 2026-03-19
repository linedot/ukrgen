# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.flow.stages.support import support_stage
from ukrgen.flow.stages.datatype import datatype_stage
from ukrgen.flow.stages.dimension import dimension_stage
from ukrgen.flow.stages.tif import mm_tif_stage
from ukrgen.flow.stages.model import lsc_model_stage
from ukrgen.flow.stages.specialize import specialize_lsc_stage
from ukrgen.flow.stages.mru import lsc_mru_stage
from ukrgen.flow.stages.schedule import lsc_schedule_stage

from ukrgen.flow.ukr_context import ukr_context
from ukrgen.flow.stage_engine import stage_engine

from ukrgen.components.tile import scalar_dp,vla_vector

from .inject_params import inject_params

class test_schedule_stage(unittest.TestCase):
    def test_rvv_fma(self):        

        ukr_ctx = ukr_context()

        params = {
            "isa" : "rvv",
            "op" : "fma",
            "AB-data-type" : "SINGLE",
            "C-data-type" : "SINGLE",
            "variant" : 1,
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

        stages = [
            support_stage,
            datatype_stage,
            dimension_stage,
            mm_tif_stage,
            lsc_model_stage,
            specialize_lsc_stage,
            lsc_mru_stage,
            lsc_schedule_stage]

        se = stage_engine(stages=stages,
                          ctx=ukr_ctx,
                          prolog=lambda s : inject_params(s, params))

        se.run()

        self.assertIn("preload", ukr_ctx.irs)
        self.assertIn("main", ukr_ctx.irs)
        self.assertIn("store", ukr_ctx.irs)

