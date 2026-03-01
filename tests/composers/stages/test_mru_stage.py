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
from ukrgen.composers.stages.tif import mm_tif_stage
from ukrgen.composers.stages.model import lsc_model_stage
from ukrgen.composers.stages.specialize import specialize_lsc_stage
from ukrgen.composers.stages.mru import lsc_mru_stage

from ukrgen.composers.ukr_context import ukr_context
from ukrgen.composers.stage_engine import stage_engine

from ukrgen.components.tile import scalar_dp,vla_vector

from .inject_params import inject_params

class test_mru_stage(unittest.TestCase):
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
            lsc_mru_stage]

        se = stage_engine(stages=stages,
                          ctx=ukr_ctx,
                          prolog=lambda s : inject_params(s, params))

        se.run()


        self.assertIn("preload", ukr_ctx.irs)
        self.assertIn("main", ukr_ctx.irs)
        self.assertIn("preload_next", ukr_ctx.irs)
        self.assertIn("store", ukr_ctx.irs)
