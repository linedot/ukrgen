# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .composition import composition_stage
from ..gemm import gemm_context

from ...specializers.asm import op_support
from ...models.load_store_operations import lsc_load,lsc_transformation,ldst_modifier

class unvec_lsc_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

    def progress(self) -> list[composition_stage]:

        sup = self.context.sup
        unvec_components = ["B","alpha","beta"]
        if "load_bcast" == self.context.params["unvec-method"]:
            vec_tile = sup.a_tile
            def mod_load(op : lsc_load) -> lsc_load:
                if not isinstance(op,lsc_load):
                    return op
                if op.addr_idx.component in unvec_components:
                    op.mods.add(ldst_modifier.bcast1)
                    op.tiles[1] = vec_tile

                return op
            def mod_transform(op : lsc_transformation) -> lsc_transformation:
                if not isinstance(op,lsc_transformation):
                    return op
                for i,idx in enumerate(op.indices):
                    c = idx.component
                    if c in unvec_components:
                        op.tiles[i] = vec_tile

                return op

            for mod in [mod_load,mod_transform]:
                for key,ir in self.context.irs.items():
                    self.context.irs[key] = [mod(op) for op in self.context.irs[key]]
        
        self.context.params.update(self.params)

        return list()
