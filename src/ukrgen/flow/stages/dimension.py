# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .stage import stage
from .unvec import unvec_stage

from ..ukr_context import ukr_context
from ..stage_param import stage_param

from ...specializers.asm import op_support

from ...components.tile import copy_with_vecdir

class dimension_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)

        #TODO: dims are ukr-specific, this is gemm/mm specific, decouple and generalize
        for dim in ["m","n","k"]:
            self.params[dim] = stage_param(
                    value=None,
                    description=f"Microkernel dimension {dim}")

        # TODO: This is op-specific, decouple and generalize
        self.params["vecdir"] = stage_param(
                value="M",
                default="M",
                description="Microkernel dimension along which to vectorize",
                choices=["M","N"],
                required=False
                )

        self.params["order"] = stage_param(
                value="mnkMNK",
                default="mnkMNK",
                description="Order in which to tile the kernel",
                required=False
                )

    def progress(self) -> list[stage]:

        self.context.params["ma"] = self.params["m"]
        self.context.params["mc"] = self.params["m"]
        self.context.params["nb"] = self.params["n"]
        self.context.params["nc"] = self.params["n"]


        self.context.params.update(self.params)


        vecdir = self.context.params["vecdir"].value
        assert vecdir in ["M","N"], f"Invalid vecdir: {vecdir}"


        # TODO: There should be some kind of generalization
        # TODO: real/data tile are already used in some places,
        #       maybe there should be support/component tiles?
        b_map = {"M" : -1, "N" : 1}
        a_map = {"M" : 0, "N" : -1}
        c_map = {"M" : 0, "N" : 1}

        if vecdir == "N":
            self.context.sup.b_tile,self.context.sup.a_tile = \
              self.context.sup.a_tile,self.context.sup.b_tile
    
        if -1 != b_map[vecdir]:
            self.context.sup.b_tile = copy_with_vecdir(
                    t=self.context.sup.b_tile,
                    vectorized_dimension=b_map[vecdir])
        if -1 != a_map[vecdir]:
            self.context.sup.a_tile = copy_with_vecdir(
                    t=self.context.sup.a_tile,
                    vectorized_dimension=a_map[vecdir])
        if -1 != c_map[vecdir]:
            self.context.sup.c_tile = copy_with_vecdir(
                    t=self.context.sup.c_tile,
                    vectorized_dimension=c_map[vecdir])

        # Do we need to unvec?
        if self.context.params["op"].value == "fma" and \
                self.context.sup.b_tile.is_vector and \
                self.context.sup.a_tile.is_vector:

            return [unvec_stage]
        else:
            return list()
