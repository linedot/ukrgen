# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .stage import stage
from .unvec import unvec_stage
from .ukr import ukr_composition_map
from ..ukr_context import ukr_context
from ..stage_param import stage_param
from ...specializers.asm import op_support
from ...components.tile import copy_with_vecdir

class dimension_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)
        

        ukr = self.context.params["ukr"].value

        composition = ukr_composition_map[ukr]

        dimensions = set()
        # get dimensions by investigating all STO descriptions
        for sto in composition.get_sto_descriptions():
            for dims in sto.dimensions.values():
                for d in dims:
                    if not d.is_dynamic:
                        dimensions.add(str(d))

        dimensions = sorted(list(dimensions))

        for dim in dimensions:
            self.params[dim] = stage_param(
                    value=None,
                    description=f"Microkernel dimension {dim}")


        default_order = "".join(sorted(dimensions))
        default_order = default_order+default_order.upper()
        self.params["order"] = stage_param(
                value=default_order,
                default=default_order,
                description="Order in which to tile the kernel",
                required=False
                )

    def progress(self) -> list[stage]:


        self.context.params.update(self.params)


        # Do we need to unvec?
        #if self.context.params["op"].value == "fma" and \
        #        self.context.sup.b_tile.is_vector and \
        #        self.context.sup.a_tile.is_vector:

        #    return [unvec_stage]
        #else:
        #    return list()
        return []
