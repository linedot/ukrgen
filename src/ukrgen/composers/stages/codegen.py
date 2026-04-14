# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import logging

from asmgen.asmblocks.noarch import comparison
from asmgen.registers import reg_tracker

from .composition import composition_stage
from ..ukr_context import ukr_context
from ..stage_param import stage_param

from asmgen.callconv.fngen import fngen
from ...codegen.blis import get_blis_gemm_cc

from ...models.loop import lsc_condition,lsc_loop,lsc_comparison
from ...models.lsc.offset import lsc_offset
from ...models.load_store_operations import lsc_add_val_off


# NOTE: Currently not using m,n,data,cntx inside the kernel. rs_C and cs_C
#       should get replaced with stride0 or stride1 if they are used, so
#       add them to unused list as well
blis_unused_parameters = {'m', 'n', 'rs_C', 'cs_C', 'data', 'cntx'}

class blis_ukr_codegen_stage(composition_stage):
    def __init__(self, context : ukr_context):
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

        k = int(self.context.params["k"].value)

        gemm_fngen.init_cc(cc=cc,
                           reverse_alias_map=stride_map,
                           unused_parameters=blis_unused_parameters)

        # TODO: decouple loopification
        condition = lsc_condition(first="k", second=None, 
                                  comparison=lsc_comparison(comparison.NZ))
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


        if 1 != k:
            k1condition = lsc_condition(first="kleft", second=None, 
                                        comparison=lsc_comparison(comparison.NZ))
            k1loop = lsc_loop(name="k1loop", condition=k1condition, level=2)
            k1loop.add_block(self.context.irs["1k_main"]+self.context.irs["1k_preload_next"])
            k1loop.add_block(ops=[
                lsc_add_val_off("kleft", off=lsc_offset({},[],[],-1))
            ])
            self.context.irs["1k_main"] = [k1loop]


        specializer = self.context.specializer

        self.context.asmblocks["init"] = specializer.code_init(
                component_dts=self.context.component_dts)

        #TODO: More generalized system for the k1 loop
        if 1 != k:
            kleftidx = self.context.rt.reserve_any_reg("greg")
            self.context.rt.alias_reg("greg", "kleft", kleftidx)
            kidx = self.context.rt.aliased_regs["greg"]["k"]

            kreg = self.context.gen.greg(kidx)
            kleftreg = self.context.gen.greg(kleftidx)
            self.context.asmblocks["init"] += \
                self.context.gen.asmwrap(f"# {kreg} <- unrolled iterations")
            self.context.asmblocks["init"] += \
                self.context.gen.asmwrap(f"# {kleftreg} <- tail iterations")

            tmpidx = self.context.rt.reserve_any_reg("greg")
            tmpreg = self.context.gen.greg(tmpidx)

            self.context.asmblocks["init"] += \
                    self.context.gen.kiterkleft(
                            kreg=kreg,
                            kleftreg=kleftreg,
                            tmpreg=tmpreg,
                            unroll=k
                            )
            self.context.rt.unuse_reg("greg",tmpidx)


        for bname in self.context.specialization_order:
            self.context.asmblocks[bname] = specializer.specialize(
                    ops = self.context.irs[bname], 
                    component_dts=self.context.component_dts)



        self.context.asmblocks["fini"] = specializer.code_fini(
                component_dts=self.context.component_dts)


        fnsave,fnload,fnrestore = gemm_fngen.get_boilerplate(
                cc=cc, unused_parameters=blis_unused_parameters)


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

        #TODO: non-hacky way to handle k checks
        asmblock += gen.asmwrap(
            "# KN CHECK 0 ---------------------------------")
        kregidx = self.context.rt.aliased_regs["greg"]["k"]
        asmblock += gen.cb(reg1=gen.greg(kregidx), reg2=None,
                           cmp=comparison.EZ, label="kndone")
        asmblock += gen.asmwrap(
            "# KN CHECK 1 ---------------------------------")
        asmblock += gen.add_greg_imm(reg=gen.greg(kregidx), imm=-1)
        asmblock += gen.cb(reg1=gen.greg(kregidx), reg2=None,
                           cmp=comparison.EZ, label="knlastiter")

        asmblock += gen.asmwrap(
            "# MAIN LOOP ----------------------------------")
        asmblock += "  "+"  ".join(self.context.asmblocks["main"])
        asmblock += gen.asmwrap(
            "# END MAIN LOOP ------------------------------")
        asmblock += gen.asmwrap(
            "# LAST ITERATION -----------------------------")
        asmblock += gen.label(label="knlastiter")
        asmblock += "  "+"  ".join(self.context.asmblocks["lastiter"])
        asmblock += gen.label(label="kndone")
        if 1 != self.context.params["k"].value:
            asmblock += gen.asmwrap(
                "# K1 CHECK 0 --------------------------------")
            kleftregidx = self.context.rt.aliased_regs["greg"]["kleft"]
            asmblock += gen.cb(reg1=gen.greg(kleftregidx), reg2=None,
                               cmp=comparison.EZ, label="loopsdone")
            asmblock += gen.asmwrap(
                "# K1 PRELOAD --------------------------------")
            asmblock += "".join(self.context.asmblocks["1k_preload"])
            #TODO: non-hacky way to handle k checks
            asmblock += gen.asmwrap(
                "# K1 CHECK 1 --------------------------------")
            asmblock += gen.add_greg_imm(reg=gen.greg(kleftregidx), imm=-1)
            asmblock += gen.cb(reg1=gen.greg(kleftregidx), reg2=None,
                               cmp=comparison.EZ, label="k1lastiter")
            asmblock += gen.asmwrap(
                "# K1 LOOP ---------------------------")
            asmblock += "  "+"  ".join(self.context.asmblocks["1k_main"])
            asmblock += gen.asmwrap(
                "# K1 LAST ITERATION -----------------")
            asmblock += gen.label(label="k1lastiter")
            asmblock += "  "+"  ".join(self.context.asmblocks["1k_lastiter"])
            
        asmblock += gen.label(label="loopsdone")
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
        asmblock += gen.isadata()


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
