# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from ukrgen.composers.stages.support import support_stage
from ukrgen.composers.stages.datatype import datatype_stage

from ukrgen.composers.gemm import gemm_context

class test_datatype_stage(unittest.TestCase):
    def test_rvv_fma(self):

        ukr_ctx = gemm_context()

        stage1 = support_stage(context=ukr_ctx)

        stage1.set_param("isa", "rvv")
        stage1.set_param("op", "fma")

        stage1.progress()

        stage2 = datatype_stage(context=ukr_ctx)

        expected_choices = [
                'SINGLE',
                'HALF',
                'SINT16',
                'DOUBLE',
                'SINT32',
                'UINT16',
                'SINT64',
                'SINT8',
                'UINT32',
                'UINT8']

        self.assertEqual(set(expected_choices),
                         set(stage2.get_param_choices("AB-data-type")))

        
