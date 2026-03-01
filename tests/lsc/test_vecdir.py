
# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest
import logging

from ukrgen.composers.stages.composition import composition_stage
from ukrgen.composers.stages.support import support_stage
from ukrgen.composers.stages.datatype import datatype_stage
from ukrgen.composers.stages.dimension import dimension_stage
from ukrgen.composers.stages.tif import mm_tif_stage
from ukrgen.composers.stages.model import lsc_model_stage

from ukrgen.composers.ukr_context import ukr_context
from ukrgen.composers.stage_engine import stage_engine

from ukrgen.components.tile import scalar_dp,vla_vector

from ..composers.stages.inject_params import inject_params

class test_lsc_vecdir(unittest.TestCase):
    def test_n(self):        

        ukr_ctx = ukr_context()

        params = {
            "isa" : "rvv",
            "op" : "fma",
            "AB-data-type" : "SINGLE",
            "C-data-type" : "SINGLE",
            "variant" : 1,
            "m" : 2,
            "n" : 4,
            "k" : 2,
            "order" : "nmkNMK",
            "vecdir" : "N",
            "A-data-regs" : 2,
            "B-data-regs" : 4,
            "AB-data-regs" : 8,
            "C-data-regs" : 8,
            "A-addr-regs" : 1,
            "B-addr-regs" : 1,
            "C-addr-regs" : 2,
            "A-preload" : 2,
            "B-preload" : 4,
        }

        logging.basicConfig(level=logging.DEBUG)


        stages = [
            support_stage,
            datatype_stage,
            dimension_stage,
            mm_tif_stage,
            lsc_model_stage]

        se = stage_engine(stages=stages,
                          ctx=ukr_ctx,
                          prolog=lambda s : inject_params(s, params))

        se.run()

        for name in ["preload","main","preload_next"]:
            print(f"==== LSC for {name} ====")
            for op in ukr_ctx.irs[name]:
                print(op)
