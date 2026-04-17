# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import importlib

from .stage import stage

from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import reg_tracker,asm_data_type as adt

from ..ukr_context import ukr_context
from ..stage_param import stage_param
from ...specializers.asm import lsc_specializer

from .isaparam import isaparam_stage

asmgen_modules = {
    'avx128' : 'avx_fma',
    'avx256' : 'avx_fma',
    'avx512' : 'avx_fma',
    'rvv'    : 'rvv',
    'rvv071' : 'rvv071',
    'neon'   : 'neon',
    'sve'    : 'sve',
    'sme'    : 'sme',
}

asmgen_map = {
    'avx128' : 'fma128',
    'avx256' : 'fma256',
    'avx512' : 'avx512',
    'rvv'    : 'rvv',
    'rvv071' : 'rvv071',
    'neon'   : 'neon',
    'sve'    : 'sve',
    'sme'    : 'sme',
}

def get_ukr_components(ukr : str) -> list[str]:
    if "gemm" == ukr:
        return ["A","B","AB","C"]
    if "mm" == ukr:
        return ["A","B","C"]
    if "pack" == ukr:
        return ["X","Y"]

    raise ValueError(f"Invalid microkernel {ukr}")

def get_ukr_mru_map(ukr : str) -> dict[str,tuple[list[str],list[str]]]:
    if ukr in {"gemm","mm"}:
        return { "store" : (
            ["preload","main","preload_next"],
            ["store"])
        }
    # we don't need to MRU anything
    if "pack" == ukr:
        return {}

    raise ValueError(f"Invalid microkernel {ukr}")

def get_ukr_sched_map(ukr : str) -> dict[str,tuple[list[str],bool]]:
    if ukr in ["gemm","mm"]:
        return { 
            "preload" : (["preload"],False),
            "main" : (["main","preload_next"],True),
            "lastiter" : (["lastiter"],False),
            #"store" : (["store"],False) # Don't reschedule store for now (it will fail)
        }

    # There is probably not enough room to schedule anyway
    if "pack" == ukr:
        return {}

    raise ValueError(f"Invalid microkernel {ukr}")

def get_ukr_specialization_order(ukr : str) -> list[str]:
    if ukr in ["gemm","mm"]:
        return ["preload","main","lastiter","store"]

    # Store is in main
    if "pack" == ukr:
        return ["preload","main","lastiter"]

    raise ValueError(f"Invalid microkernel {ukr}")

class support_stage(stage):
    def __init__(self, context : ukr_context):
        super().__init__(context)

        supported_isas = list(asmgen_modules.keys())
        supported_instructions = ["fma","dota","fopa","mma"]
        supported_ukrs = ["gemm","mm"]

        self.params["isa"] = stage_param(
                value=None,
                description="Instruction set to use",
                choices=supported_isas)
        self.params["op"] = stage_param(
                value=None,
                description="Arithmentic instruction to base the kernel on",
                choices=supported_instructions
                )
        self.params["ukr"] = stage_param(
                value="gemm",
                default="gemm",
                description="Type of Microkernel to generate",
                choices=supported_ukrs,
                required=False
                )

    def progress(self):

        isa = self.params["isa"].value

        module = importlib.import_module(f"asmgen.asmblocks.{asmgen_modules[isa]}")
        self.context.gen = module.__dict__[asmgen_map[isa]]()
        
        self.context.gen.set_output_inline(yesno=False)


        self.context.ukr_components = get_ukr_components(self.params["ukr"].value)
        self.context.mru_map = get_ukr_mru_map(self.params["ukr"].value)
        self.context.sched_map = get_ukr_sched_map(self.params["ukr"].value)
        self.context.specialization_order = get_ukr_specialization_order(self.params["ukr"].value)


        self.context.params.update(self.params)

        isaparams = self.context.gen.get_parameters()
        if len(isaparams) > 0:
            return [isaparam_stage]

        self.context.rt = reg_tracker([
            ('greg',self.context.gen.max_gregs),
            ('freg',self.context.gen.max_fregs),
            ('vreg',self.context.gen.max_vregs),
            ('treg',self.context.gen.max_tregs(adt.FP64))])

        self.context.specializer = lsc_specializer(
                model=None,
                gen=self.context.gen)

        return []
