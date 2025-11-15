# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging

from .composition import composition_stage
from ..gemm import gemm_context


class specialize_lsc_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        self.debug = logging.getLogger("LSC").debug

    def progress(self) -> list[composition_stage]:

        blockorder = [
            "preload",
            "main",
            "betascale",
            "alphascale",
            "store",
            "preload_next"
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
        if "gemm" == self.context.params["ukr"]:
            self.debug("BETASCALE BLOCK -------------------------")
            self.debug("\n".join(map(str,self.context.irs["betascale"])))
            self.debug("END BETASCALE BLOCK ---------------------")
            self.debug("ALPHASCALE BLOCK ------------------------")
            self.debug("\n".join(map(str,self.context.irs["alphascale"])))
            self.debug("END ALPHASCALE BLOCK --------------------")
        self.debug("STOREBLOCK ------------------------------")
        self.debug("\n".join(map(str,self.context.irs["store"])))
        self.debug("ENDSTOREBLOCK ---------------------------")

        
        self.context.params.update(self.params)

        return list()
