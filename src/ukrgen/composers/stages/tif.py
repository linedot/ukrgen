# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging

from .composition import composition_stage
from ..gemm import gemm_context

from ...specializers.asm import op_support
from ...components.tile import simple_ukr_tile,scalar_dp
from ...generators.mm import mm,order2D

class mm_tif_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)


        self.debug = logging.getLogger("TIF").debug

        sup = context.sup

        ma = context.params["ma"]
        nb = context.params["nb"]
        mc = context.params["mc"]
        nc = context.params["nc"]
        k = context.params["k"]

        self.a_tile = simple_ukr_tile(
                a_size=ma, b_size=k,
                subdims=(sup.a_tile.dima, sup.a_tile.dimb))
        self.b_tile = simple_ukr_tile(
                a_size=k, b_size=nb,
                subdims=(sup.b_tile.dima, sup.b_tile.dimb))
        self.c_tile = simple_ukr_tile(
                a_size=mc, b_size=nc,
                subdims=(sup.c_tile.dima, sup.c_tile.dimb))

    def progress(self):

        order = order2D(self.context.params["order"])
        m = self.context.params["m"]
        n = self.context.params["n"]
        k = self.context.params["k"]
        sup = self.context.sup

        if "mm" == self.context.params["ukr"]:
            genmm = mm(a=self.a_tile, b=self.b_tile, c=self.c_tile,
                       lo=order, opstr=self.context.params["op"],
                       tile_strs=["A","B","C"])
        if "gemm" == self.context.params["ukr"]:
            genmm = mm(a=self.a_tile, b=self.b_tile, c=self.c_tile,
                       lo=order, opstr=self.context.params["op"],
                       tile_strs=["A","B","AB"])

            scale_tile = simple_ukr_tile(a_size=m,
                                         b_size=n,
                                         subdims=(sup.c_tile.dima,
                                                  sup.c_tile.dimb))
            alphabeta_tile = simple_ukr_tile(a_size=n, b_size=n,
                                        subdims=(scalar_dp,scalar_dp),
                                        bands=(0,0))
            genbetascale = mm(scale_tile, alphabeta_tile, scale_tile,
                              opstr="fmul", tile_strs=["C","beta","C"])
            genalphascale = mm(scale_tile, alphabeta_tile, scale_tile,
                            opstr="fma", tile_strs=["AB","alpha","C"])
            self.context.tifs["betascale"] = genbetascale.generate()
            self.context.tifs["alphascale"] = genalphascale.generate()

        self.context.tifs["mm"]     = genmm.generate()
        self.context.tifs["mm_p1k"] = genmm.generate(add_dims=[0,0,0,0,0,k])
        self.context.tifs["mm_p2k"] = genmm.generate(add_dims=[0,0,0,0,0,2*k])

        self.context.params.update(self.params)

        self.debug("################### TILE-INSTRUCTION-FORMAT ###################")
        self.debug("### MAIN ###")
        for op in self.context.tifs["mm"]:
            self.debug(str(op))
        self.debug("### BETA SCALE ###")

        if "gemm" == self.context.params["ukr"]:
            for op in self.context.tifs["betascale"]:
                self.debug(str(op))
            self.debug("### ALPHA SCALE ###")
            for op in self.context.tifs["alphascale"]:
                self.debug(str(op))

        return list()
