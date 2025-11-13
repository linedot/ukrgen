# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from .composition import composition_stage
from ..gemm import gemm_context
from ...components.tile import dimension_properties,dimension_type

class unvec_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        self.params["unvec-method"] = "load_bcast"
        self.default_values["unvec-method"] = "load_bcast"
        self.choices["unvec-method"] = ["load_bcast","lane_select","lane_bcast"]

    def progress(self) -> list[composition_stage]:

        if self.context.params["vecdir"] == "M":
            modified_tile = self.context.sup.b_tile
            mod_dt = self.context.sup.triple.b
            mod_dim = self.context.params["n"]
        elif self.context.params["vecdir"] == "N":
            modified_tile = self.context.sup.a_tile
            mod_dt = self.context.sup.triple.a
            mod_dim = self.context.params["m"]
        
        if self.params["unvec-method"] in ['load_bcast']:
            modified_tile.dima = dimension_properties(
                    dt=dimension_type.fixed, size=1,
                    sdt=dimension_type.fixed, sd_size=1)
        elif self.params["unvec-method"] in ['lane_select','lane_bcast']:
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

        if self.context.params["vecdir"] == "M":
            self.context.sup.b_tile = modified_tile
            self.context.params["nb"] = mod_dim

        elif self.context.params["vecdir"] == "N":
            self.context.sup.a_tile = modified_tile
            self.context.params["ma"] = mod_dim

        return list()
