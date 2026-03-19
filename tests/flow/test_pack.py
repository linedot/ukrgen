# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging
import unittest

from ukrgen.flow.stages.stage import stage
from ukrgen.flow.stages.support import support_stage
from ukrgen.flow.stages.datatype import datatype_stage
from ukrgen.flow.stages.dimension import dimension_stage
from ukrgen.flow.stages.tif import pack_tif_stage
from ukrgen.flow.stages.model import lsc_model_stage
from ukrgen.flow.stages.mru import lsc_mru_stage
from ukrgen.flow.stages.schedule import lsc_schedule_stage
from ukrgen.flow.stages.specialize import specialize_lsc_stage
from ukrgen.flow.stages.codegen import blis_ukr_codegen_stage

from ukrgen.flow.ukr_context import ukr_context
from ukrgen.flow.stage_engine import stage_engine

from .stages.inject_params import inject_params

class test_pack(unittest.TestCase):
    def test_rvv_fma(self):

        ukr_ctx = ukr_context()

        params = {
            "ukr" : "pack",
            "isa" : "rvv",
            "op" : "fmul",
            "X-data-type" : "SINGLE",
            "Y-data-type" : "SINGLE",
            "variant" : 0,
            "m" : 2,
            "n" : 4,
            "k" : 4,
            "X-data-regs" : 4,
            "Y-data-regs" : 4,
            "X-addr-regs" : 2,
            "Y-addr-regs" : 2,
            "X-preload" : 2,
        }

        stages = [
            support_stage,
            datatype_stage,
            dimension_stage,
            pack_tif_stage,
            #lsc_model_stage,
            #specialize_lsc_stage,
            #lsc_mru_stage,
            #lsc_schedule_stage,
            #blis_ukr_codegen_stage
            ]

        se = stage_engine(stages=stages,
                          ctx=ukr_ctx,
                          prolog=lambda s : inject_params(s, params))

        se.run()

        print("\n".join(map(str,ukr_ctx.tifs["pack"])))

