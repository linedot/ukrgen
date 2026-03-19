# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.flow.stages.stage import stage
from ukrgen.flow.stages.support import support_stage
from ukrgen.flow.stages.datatype import datatype_stage

from ukrgen.flow.ukr_context import ukr_context
from ukrgen.flow.stage_engine import stage_engine

from .inject_params import inject_params

class test_datatype_stage(unittest.TestCase):
    def test_rvv_fma(self):        

        ukr_ctx = ukr_context()

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
