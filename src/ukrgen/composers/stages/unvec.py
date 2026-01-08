# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .composition import composition_stage
from ..gemm import gemm_context
from ..stage_param import stage_param
from ...components.tile import (
        dimension_properties,
        dimension_type,
        copy_with_vecdir
)

class unvec_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        self.params["unvec-method"] = stage_param(
                value="load_bcast", 
                default="load_bcast",
                description="Method to handle scalar data when using vector registers",
                choices=["load_bcast","lane_select","lane_bcast"])

    def progress(self) -> list[composition_stage]:

        vecdir = self.context.params["vecdir"].value 

        assert vecdir in ["M","N"], f"Invalid vecdir: {vecdir}"


        # TODO: There should be some kind of generalization
        # TODO: real/data tile are already used in some places,
        #       maybe there should be support/component tiles?
        b_map = {"M" : -1, "N" : 1}
        a_map = {"M" : 0, "N" : -1}
        c_map = {"M" : 0, "N" : 1}
    
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


        if vecdir == "M":
            modified_tile = self.context.sup.b_tile
            mod_dt = self.context.sup.triple.b
            mod_dim = self.context.params["n"].value
        elif vecdir == "N":
            modified_tile = self.context.sup.a_tile
            mod_dt = self.context.sup.triple.a
            mod_dim = self.context.params["m"].value

        
        if self.params["unvec-method"].value in ['load_bcast']:
            # Ensure tile is scalar
            modified_tile.dima = dimension_properties(
                    dt=dimension_type.fixed, size=1,
                    sdt=dimension_type.fixed, sd_size=1)
            modified_tile.dimb = dimension_properties(
                    dt=dimension_type.fixed, size=1,
                    sdt=dimension_type.fixed, sd_size=1)

        elif self.params["unvec-method"].value in ['lane_select','lane_bcast']:
            # TODO: needs more involved changes, crash for now
            raise NotImplementedError("lane_select and lane_bcast not yet implemented")

            into_size = gen.indexable_elements(mod_dt)
            mod_dim //= into_size

            modified_tile.dima = dimension_properties(
                    dt=dimension_type.fixed, size=1,
                    sdt=dimension_type.fixed, sd_size=1)
            modified_tile.dimb = dimension_properties(
                    dt=dimension_type.fixed, size=into_size,
                    sdt=dimension_type.fixed, sd_size=into_size)

        if vecdir == "M":
            self.context.sup.b_tile = modified_tile
            self.context.params["nb"].value = mod_dim

        elif vecdir == "N":
            self.context.sup.a_tile = modified_tile
            self.context.params["ma"].value = mod_dim

        self.context.params.update(self.params)

        return list()
