# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging

from .composition import composition_stage
from ..gemm import gemm_context

from ...schedulers.minreguse_scheduler import minreguse_scheduler


class lsc_mru_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        self.debug = logging.getLogger("LSC").debug


    def progress(self) -> list[composition_stage]:
        # register reuse 
        mru_scheduler = minreguse_scheduler()

        for dst,(preceeding,targets) in self.context.mru_map.items():
            src = []
            pre = []
            for bname in targets:
                src.extend(self.context.irs[bname])
            for bname in preceeding:
                pre.extend(self.context.irs[bname])

            mru_scheduler.analyze_preceeding(pre)
            mru_scheduler.analyze(src)
            src_reo = mru_scheduler.reorder(src)

            self.debug(f"################### MRU {dst.upper()} REORDERED PSEUDO-ASM ###################")
            self.debug("\n".join(map(str,src_reo)))

            mru_scheduler.analyze(src_reo)
            src_rep = mru_scheduler.replace(
                src_reo,
                self.context.specializer.data_registers
            )
            self.debug(f"################### MRU {dst.upper()} RENAMED PSEUDO-ASM ###################")
            self.debug("\n".join(map(str,src_rep)))

            self.context.irs[dst] = src_rep

        
        self.context.params.update(self.params)

        return list()
