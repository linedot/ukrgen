# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .composition import composition_stage
from ..gemm import gemm_context
from ..stage_param import stage_param

from ...specializers.asm import op_support

class variant_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)


        self.params["variant"] = stage_param(
                value=0, description=0,
                default=0,
                choices=list(range(len(context.op_support_list))),
                required=False)

    def progress(self) -> list[composition_stage]:
        suplist = self.context.op_support_list
        index = int(self.params["variant"].value)
        self.context.sup = suplist[index]

        self.context.params.update(self.params)

        return list()
