# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.composers.stages.composition import composition_stage
from ukrgen.composers.stages.support import support_stage
from ukrgen.composers.stages.datatype import datatype_stage

from ukrgen.composers.gemm import gemm_context
from ukrgen.composers.stage_engine import stage_engine

from .inject_params import inject_params

class test_datatype_stage(unittest.TestCase):
    def test_rvv_fma(self):        

        ukr_ctx = gemm_context()

        params = {
            "isa" : "rvv",
            "op" : "fma",
            "AB-data-type" : "SINGLE",
            "C-data-type" : "SINGLE",
        }

        stages = [support_stage,datatype_stage]

        se = stage_engine(stages=stages,
                          ctx=ukr_ctx,
                          prolog=lambda s : inject_params(s, params))

        se.run()
