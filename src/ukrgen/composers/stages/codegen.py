# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging

from asmgen.asmblocks.noarch import comparison

from .composition import composition_stage
from ..gemm import gemm_context
from ..stage_param import stage_param

from ...codegen.fngen import fngen
from ...codegen.blis import get_blis_gemm_cc

from ...models.loop import lsc_condition,lsc_loop,lsc_comparison
from ...models.lsc.offset import lsc_offset
from ...models.load_store_operations import lsc_add_val_off

class blis_ukr_codegen_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)
        
        self.debug = logging.getLogger("CODEGEN").debug


        self.params["function-name"] = stage_param(
                value=None, 
                description="Override ASM Symbol/function name with a custom one",
                required=False)


    def progress(self) -> list[composition_stage]:

        gemm_fngen = fngen(gen=self.context.gen, rt=self.context.rt)

        strides = self.context.strides
        stride_map : dict[str,str] = dict()
        # gotta map {r,c}s_{a,b,c} onto strideN
        for component,rcs in strides.items():
            if rcs[0] is not None:
                stride_map[f"rs_{component}"] = f"stride{rcs[0]}"
            if rcs[1] is not None:
                stride_map[f"cs_{component}"] = f"stride{rcs[1]}"

        # TODO: decouple BLIS-specific code
        cc = get_blis_gemm_cc(gen=self.context.gen)

        gemm_fngen.init_cc(cc=cc,
                           reverse_alias_map=stride_map)

        # TODO: decouple loopification
        condition = lsc_condition(first="k", second=None, 
                                  comparison=lsc_comparison(comparison.nz))
        mainloop = lsc_loop(name="knloop", condition=condition, level=2)

        mainloop.add_block(self.context.irs["main"])
        mainloop.add_block(ops=[
            lsc_add_val_off("k", off=lsc_offset({},[],[],-1))
            ])

        # TODO: figure out prefetching
        #if distance_pfc:
        #    vecdim=0
        #    count = [m,n][vecdim]
        #    for i in m:
        #        asmblock += ""
        #    loop.add_singleshot_divergence(name="pfc", block="")

        self.context.irs["main"] = [mainloop]


        specializer = self.context.specializer

        self.context.asmblocks["init"] = specializer.code_init(
                component_dts=self.context.component_dts)


        for bname in self.context.specialization_order:
            self.context.asmblocks[bname] = specializer.specialize(
                    ops = self.context.irs[bname], 
                    component_dts=self.context.component_dts)



        self.context.asmblocks["fini"] = specializer.code_fini(
                component_dts=self.context.component_dts)


        fnsave,fnload,fnrestore = gemm_fngen.get_boilerplate(cc=cc)


        gen = self.context.gen

        asmblock = gen.asmwrap(
            "# FUNC INTRO ---------------------------------")
        asmblock += fnsave
        asmblock += fnload
        asmblock += gen.asmwrap(
            "# INIT ---------------------------------------")
        asmblock += self.context.asmblocks["init"]
        asmblock += gen.asmwrap(
            "# PRELOAD ------------------------------------")
        asmblock += "".join(self.context.asmblocks["preload"])
        asmblock += gen.asmwrap(
            "# MAIN LOOP ----------------------------------")
        asmblock += "  "+"  ".join(self.context.asmblocks["main"])
        asmblock += gen.asmwrap(
            "# END MAIN LOOP ------------------------------")
        asmblock += gen.asmwrap(
            "# LAST ITERATION -----------------------------")
        asmblock += "  "+"  ".join(self.context.asmblocks["lastiter"])
        if "gemm" == self.context.params["ukr"].value:
            asmblock += gen.asmwrap(
                "# SCALE+STOREBLOCK ---------------------------")
        else:
            asmblock += gen.asmwrap(
                "# STOREBLOCK ---------------------------------")
        asmblock += "".join(self.context.asmblocks["store"])
        asmblock += gen.asmwrap(
            "# END STOREBLOCK -----------------------------")
        asmblock += gen.asmwrap(
            "# FINALIZE -----------------------------------")
        asmblock += self.context.asmblocks["fini"]
        asmblock += gen.asmwrap(
            "# FUNC OUTRO ---------------------------------")
        asmblock += fnrestore
        asmblock += gen.asmwrap("ret") # TODO: add ret() to asmgen


        self.debug("################### ASM ###################")
        self.debug("FUNC INTRO ------------------------------")
        self.debug(fnsave)
        self.debug(fnload)
        self.debug("INIT ------------------------------------")
        self.debug(self.context.asmblocks["init"])
        self.debug("PRELOAD ---------------------------------")
        self.debug("".join(self.context.asmblocks["preload"]))
        self.debug("MAIN LOOP -------------------------------")
        self.debug("  "+"  ".join(self.context.asmblocks["main"]))
        self.debug("END MAIN LOOP ---------------------------")
        self.debug("LAST ITERATION --------------------------")
        self.debug("  "+"  ".join(self.context.asmblocks["lastiter"]))
        if "gemm" == self.context.params["ukr"].value:
            self.debug("SCALE+STOREBLOCK ------------------------")
        else:
            self.debug("STOREBLOCK ------------------------------")
        self.debug("".join(self.context.asmblocks["store"]))
        self.debug("ENDSTOREBLOCK ---------------------------")
        self.debug("FINALIZE --------------------------------")
        self.debug(self.context.asmblocks["fini"])
        self.debug("FUNC OUTRO ------------------------------")
        self.debug(fnrestore)




        #for key in rt.aliased_regs.keys():
        #    moreargs = dict()
        #    if key in {'freg','treg'}:
        #        moreargs = {"dt":self.context.component_dts["A"]}
        #    genlog.debug(f"Aliased {key}s in the end:")
        #    for alias,regidx in rt.aliased_regs[key].items():
        #        reg = getattr(gen, key)(regidx,**moreargs)
        #        genlog.debug(f"  {alias:30} : {reg}")


        strides = self.context.strides

        suffix = ""
        if (strides["C"][0] is None) and (strides["C"][1] is not None):
            suffix = "_cs"
        elif (strides["C"][1] is None) and (strides["C"][0] is not None):
            suffix = "_rs"
        elif (strides["C"][1] is not None) and (strides["C"][0] is not None):
            suffix = "_csrs"

        if not suffix:
            suffix = "_nostride"

        isa = self.context.params["isa"].value
        ukr = self.context.params["ukr"].value
        m = self.context.params["m"].value
        n = self.context.params["n"].value

        fnname = f"ukrgen_{ukr}_{isa}_{m}Vx{n}"
        if self.params["function-name"].value is not None:
            fnname = self.params["function-name"].value
        

        asmheader = (
             ".section .text\n"
            f".global {fnname}{suffix}\n"
            f"{fnname}{suffix}:\n  "
        )
        
        # TODO: some kind of fancy formatting
        asmblock = asmblock.replace("\n","\n  ")


        self.context.asmblocks["full_function"] = asmheader+asmblock
        
        self.context.params.update(self.params)

        return list()
