# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging

from .composition import composition_stage
from ..ukr_context import ukr_context

from ...specializers.asm import op_support
from ...components.tile import simple_ukr_tile,scalar_dp
from ...generators.mm import mm,order2D

class mm_tif_stage(composition_stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)


        self.debug = logging.getLogger("TIF").debug

        sup = context.sup

        ma = int(context.params["ma"].value)
        nb = int(context.params["nb"].value)
        mc = int(context.params["mc"].value)
        nc = int(context.params["nc"].value)
        k  = int(context.params["k"].value)

        self.a_tile = simple_ukr_tile(
                a_size=ma, b_size=k,
                subdims=(sup.a_tile.dima, sup.a_tile.dimb))
        self.b_tile = simple_ukr_tile(
                a_size=k, b_size=nb,
                subdims=(sup.b_tile.dima, sup.b_tile.dimb))
        self.c_tile = simple_ukr_tile(
                a_size=mc, b_size=nc,
                subdims=(sup.c_tile.dima, sup.c_tile.dimb))

        if k > 1:
            self.a_tile_k1 = simple_ukr_tile(
                    a_size=ma, b_size=1,
                    subdims=(sup.a_tile.dima, sup.a_tile.dimb)
                    )
            self.b_tile_k1 = simple_ukr_tile(
                    a_size=1, b_size=nb,
                    subdims=(sup.b_tile.dima, sup.b_tile.dimb)
                    )

    def progress(self):

        order = order2D(self.context.params["order"].value)
        m = int(self.context.params["m"].value)
        n = int(self.context.params["n"].value)
        k = int(self.context.params["k"].value)
        sup = self.context.sup

        tile_strs = []
        ukr = self.context.params["ukr"].value
        if "mm" == ukr:
            tile_strs = ["A","B","C"]
        elif "gemm" == ukr:
            tile_strs = ["A","B","AB"]
        else:
            raise ValueError(f"Tile names for {ukr} unknown")

        genmm = mm(a=self.a_tile, b=self.b_tile, c=self.c_tile,
                   lo=order, opstr=self.context.params["op"].value,
                   tile_strs=tile_strs)

        if "gemm" == ukr:

            # TODO: Haven't brained through this, but this is necessary
            #       for correct order when vecdir=N. Investigate and
            #       understand why
            # NOTE: I'm pulling the Ks in front of the order, i.e
            #       mnkMNK -> kKmnMN; nmkNMK -> kKmnMN
            #       It might be more correct to do
            #       mnkMNK -> kmnKMN, etc...
            scaleorder_str = self.context.params["order"].value
            scaleorder_str = scaleorder_str.replace("k","")
            scaleorder_str = scaleorder_str.replace("K", "")
            scaleorder_str = "kK"+scaleorder_str

            self.debug(f"SCALE ORDER: {scaleorder_str}")

            scaleorder = order2D(scaleorder_str)

            scale_tile = simple_ukr_tile(a_size=m,
                                         b_size=n,
                                         subdims=(sup.c_tile.dima,
                                                  sup.c_tile.dimb))
            alphabeta_tile = simple_ukr_tile(a_size=n, b_size=n,
                                        subdims=(scalar_dp,scalar_dp),
                                        bands=(0,0))
            genbetascale = mm(scale_tile, alphabeta_tile, scale_tile,
                              lo=scaleorder,
                              opstr="fmul", tile_strs=["C","beta","C"])
            genalphascale = mm(scale_tile, alphabeta_tile, scale_tile,
                               lo=scaleorder,
                               opstr="fma", tile_strs=["AB","alpha","C"])
            self.context.tifs["betascale"] = genbetascale.generate()
            self.context.tifs["alphascale"] = genalphascale.generate()

        self.context.tifs["mm"]     = genmm.generate()
        self.context.tifs["mm_p1k"] = genmm.generate(add_dims=[0,0,0,0,0,k])
        self.context.tifs["mm_p2k"] = genmm.generate(add_dims=[0,0,0,0,0,2*k])

        # We additionally need k = 1 for tail handling
        if k > 1:
            genmm1k = mm(a=self.a_tile_k1, b=self.b_tile_k1, c=self.c_tile,
                         lo=order, opstr=self.context.params["op"].value,
                         tile_strs=tile_strs)
            self.context.tifs["mm1k"] = genmm1k.generate()
            self.context.tifs["mm1k_p1"] = genmm1k.generate(
                    add_dims=[0,0,0,0,0,1])
            self.context.tifs["mm1k_p2"] = genmm1k.generate(
                    add_dims=[0,0,0,0,0,2])


        self.context.params.update(self.params)

        self.debug("################### TILE-INSTRUCTION-FORMAT ###################")
        self.debug("### MAIN ###")
        for op in self.context.tifs["mm"]:
            self.debug(str(op))
        self.debug("### MAIN +1k ###")
        for op in self.context.tifs["mm_p1k"]:
            self.debug(str(op))
        self.debug("### MAIN +2k ###")
        for op in self.context.tifs["mm_p2k"]:
            self.debug(str(op))

        if k > 1:
            self.debug("### k1 MAIN ###")
            for op in self.context.tifs["mm1k"]:
                self.debug(str(op))
            self.debug("### k1 MAIN +1 ###")
            for op in self.context.tifs["mm1k_p1"]:
                self.debug(str(op))
            self.debug("### k1 MAIN +2 ###")
            for op in self.context.tifs["mm1k_p2"]:
                self.debug(str(op))


        if "gemm" == ukr:
            self.debug("### BETA SCALE ###")
            for op in self.context.tifs["betascale"]:
                self.debug(str(op))
            self.debug("### ALPHA SCALE ###")
            for op in self.context.tifs["alphascale"]:
                self.debug(str(op))

        return list()
