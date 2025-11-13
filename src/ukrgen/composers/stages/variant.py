# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .composition import composition_stage
from ..gemm import gemm_context

from ...specializers.asm import op_support

class variant_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)


        self.params["variant"] = 0
        self.default_values["variant"] = 0
        self.choices["variant"] = list(range(len(context.op_support_list)))

    def progress(self) -> list[composition_stage]:
        self.context.sup = self.context.op_support_list[self.params["variant"]]

        self.context.params.update(self.params)

        return list()
