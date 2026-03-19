# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .stage import stage
from ..ukr_context import ukr_context
from ..stage_param import stage_param

from ...specializers.asm import op_support

class variant_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)

        indices = list(range(len(context.op_support_list)))
        suplist = context.op_support_list

        opsupblock = "; ".join([f"{i} ==> \n{suplist[i]}" for i in indices])

        self.params["variant"] = stage_param(
                value="0", 
                description=f"Variant of the supported instruction. {opsupblock}",
                default="0",
                choices=[str(i) for i in indices],
                required=False)

    def progress(self) -> list[stage]:
        suplist = self.context.op_support_list
        index = int(self.params["variant"].value)
        self.context.sup = suplist[index]

        self.context.params.update(self.params)

        return list()
