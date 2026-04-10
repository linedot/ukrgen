# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging

from .stage import stage
from ..ukr_context import ukr_context


class specialize_lsc_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)

        self.debug = logging.getLogger("LSC").debug

    def progress(self) -> list[stage]:

        blockorder = [
            "preload",
            "main",
            "store",
            "preload_next"
        ]
        if 1 != self.context.params["k"].value:
            blockorder = [
            "preload",
            "main",
            "1k_preload",
            "1k_main",
            "store",
            "preload_next",
            "1k_preload_next"
        ]

        specializer = self.context.specializer
        component_dts = self.context.component_dts

        for block_name in blockorder:
            if not block_name in self.context.irs:
                continue
            specializer.analyse(self.context.irs[block_name])

        for block_name in blockorder:
            if not block_name in self.context.irs:
                continue
            self.context.irs[block_name] = specializer.pre_specialize(
                ops=self.context.irs[block_name],
                component_dts=component_dts)

        self.debug("################### SPECIALIZED PSEUDO-ASM ###################")
        self.debug("\n".join(map(str,self.context.irs["preload"])))
        self.debug("MAIN LOOP -------------------------------")
        self.debug("  "+"\n  ".join(map(str,self.context.irs["main"])))
        self.debug("PRELOAD NEXT ----------------------------")
        self.debug("  "+"\n  ".join(map(str,self.context.irs["preload_next"])))
        self.debug("END MAIN LOOP ---------------------------")
        self.debug("STOREBLOCK ------------------------------")
        self.debug("\n".join(map(str,self.context.irs["store"])))
        self.debug("ENDSTOREBLOCK ---------------------------")

        
        self.context.params.update(self.params)

        return list()
