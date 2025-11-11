# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.composers.stages.support import support_stage

from ukrgen.composers.gemm import gemm_context

class test_support_stage(unittest.TestCase):
    def test_rvv_fma(self):

        ukr_ctx = gemm_context()

        stage = support_stage(context=ukr_ctx)

        stage.set_param("isa", "rvv")
        stage.set_param("op", "fma")

        stage.progress()

