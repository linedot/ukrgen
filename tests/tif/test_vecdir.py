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

from ukrgen.composers.gemm import gemm_context
from ukrgen.composers.stage_engine import stage_engine

from ukrgen.components.tile import scalar_dp,vla_vector

from ..composers.stages.inject_params import inject_params

class test_tif_vecdir(unittest.TestCase):
    def test_n(self):        

        ukr_ctx = gemm_context()

        params = {
            "isa" : "rvv",
            "op" : "fma",
            "AB-data-type" : "SINGLE",
            "C-data-type" : "SINGLE",
            "variant" : 1,
            "m" : 4,
            "n" : 4,
            "k" : 2,
            "order" : "nmkNMK",
            "vecdir" : "N"
        }

        stages = [support_stage,datatype_stage,dimension_stage,mm_tif_stage]

        se = stage_engine(stages=stages,
                          ctx=ukr_ctx,
                          prolog=lambda s : inject_params(s, params))

        se.run()

        for name,tif in ukr_ctx.tifs.items():
            print(f"==== TIF for {name} ====")
            for op in tif:
                print(op)

        # AB(0,VLEN*0) <- fma(A(0,0), B(0,0*VLEN), AB(0,VLEN*0))
        # AB(0,VLEN*1) <- fma(A(0,0), B(0,1*VLEN), AB(0,VLEN*1))
        # AB(0,VLEN*2) <- fma(A(0,0), B(0,2*VLEN), AB(0,VLEN*2))
        # AB(0,VLEN*3) <- fma(A(0,0), B(0,3*VLEN), AB(0,VLEN*3))

        # AB(1,VLEN*0) <- fma(A(1,0), B(0,0*VLEN), AB(1,VLEN*0))
        # AB(1,VLEN*1) <- fma(A(1,0), B(0,1*VLEN), AB(1,VLEN*1))
        # AB(1,VLEN*2) <- fma(A(1,0), B(0,2*VLEN), AB(1,VLEN*2))
        # AB(1,VLEN*3) <- fma(A(1,0), B(0,3*VLEN), AB(1,VLEN*3))

        # AB(2,VLEN*0) <- fma(A(2,0), B(0,0*VLEN), AB(2,VLEN*0))
        # AB(2,VLEN*1) <- fma(A(2,0), B(0,1*VLEN), AB(2,VLEN*1))
        # AB(2,VLEN*2) <- fma(A(2,0), B(0,2*VLEN), AB(2,VLEN*2))
        # AB(2,VLEN*3) <- fma(A(2,0), B(0,3*VLEN), AB(2,VLEN*3))

        # AB(3,VLEN*0) <- fma(A(3,0), B(0,0*VLEN), AB(3,VLEN*0))
        # AB(3,VLEN*1) <- fma(A(3,0), B(0,1*VLEN), AB(3,VLEN*1))
        # AB(3,VLEN*2) <- fma(A(3,0), B(0,2*VLEN), AB(3,VLEN*2))
        # AB(3,VLEN*3) <- fma(A(3,0), B(0,3*VLEN), AB(3,VLEN*3))


        # AB(0,VLEN*0) <- fma(A(0,1), B(1,0*VLEN), AB(0,VLEN*0))
        # AB(0,VLEN*1) <- fma(A(0,1), B(1,1*VLEN), AB(0,VLEN*1))
        # AB(0,VLEN*2) <- fma(A(0,1), B(1,2*VLEN), AB(0,VLEN*2))
        # AB(0,VLEN*3) <- fma(A(0,1), B(1,3*VLEN), AB(0,VLEN*3))

        # AB(1,VLEN*0) <- fma(A(1,1), B(1,0*VLEN), AB(1,VLEN*0))
        # AB(1,VLEN*1) <- fma(A(1,1), B(1,1*VLEN), AB(1,VLEN*1))
        # AB(1,VLEN*2) <- fma(A(1,1), B(1,2*VLEN), AB(1,VLEN*2))
        # AB(1,VLEN*3) <- fma(A(1,1), B(1,3*VLEN), AB(1,VLEN*3))

        # AB(2,VLEN*0) <- fma(A(2,1), B(1,0*VLEN), AB(2,VLEN*0))
        # AB(2,VLEN*1) <- fma(A(2,1), B(1,1*VLEN), AB(2,VLEN*1))
        # AB(2,VLEN*2) <- fma(A(2,1), B(1,2*VLEN), AB(2,VLEN*2))
        # AB(2,VLEN*3) <- fma(A(2,1), B(1,3*VLEN), AB(2,VLEN*3))

        # AB(3,VLEN*0) <- fma(A(3,1), B(1,0*VLEN), AB(3,VLEN*0))
        # AB(3,VLEN*1) <- fma(A(3,1), B(1,1*VLEN), AB(3,VLEN*1))
        # AB(3,VLEN*2) <- fma(A(3,1), B(1,2*VLEN), AB(3,VLEN*2))
        # AB(3,VLEN*3) <- fma(A(3,1), B(1,3*VLEN), AB(3,VLEN*3))
