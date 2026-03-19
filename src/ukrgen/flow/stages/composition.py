# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from __future__ import annotations

from ..ukr_context import ukr_context

from ..stage_param import stage_param

from .stage import stage

class composition_stage(stage):

    def __init__(self, context : ukr_context):
        pass

    def progress(self) -> list[composition_stage]:

        # components:
        # - mm:
        #   - A,B,C
        # - gemm:
        #   - A,B,C,AB,alpha,beta
        # - pack:
        #   - X,Y,kappa


        # STOs:
        # - mm:
        #   - mm (A,B,C)
        #     - m k x k n + m n
        #     - preload
        #     - tail
        #     - loop
        #   - store (C)
        # - gemm:
        #   - mm (A,B,AB) 
        #     - m k x k n + m n
        #     - preload (0,0,0,0,0,+k)
        #     - tail (0,0,0,0,0,+1)
        #     - loop
        #   - mm (C, beta, C)
        #     - m n x n n + m n
        #     - fmul
        #     - bands(0,0)
        #   - mm (AB, alpha, C)
        #     - m n x n n + m n
        #     - bands(0,0)
        #   - store (C)
        # - pack:
        #   - mm (X, kappa, Y)
        #     - m n x n n + m n
        #     - preload (0,0,0,0,+n,+n)
        #     - tail (0,0,0,0,+1,+1)
        #   - store (Y)
        #   - combine mm+ store, loop
        #

        pass
