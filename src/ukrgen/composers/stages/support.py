# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import importlib

from .composition import composition_stage

from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import reg_tracker,asm_data_type as adt

from ..gemm import gemm_context
from ..stage_param import stage_param
from ...specializers.asm import lsc_specializer

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

    raise ValueError(f"Invalid microkernel {ukr}")

def get_ukr_mru_map(ukr : str) -> dict[str,tuple[list[str],list[str]]]:
    if ukr in {"gemm","mm"}:
        return { "store" : (
            ["preload","main","preload_next"],
            ["store"])
        }

    raise ValueError(f"Invalid microkernel {ukr}")

def get_ukr_sched_map(ukr : str) -> dict[str,tuple[list[str],bool]]:
    if ukr in ["gemm","mm"]:
        return { 
            "preload" : (["preload"],False),
            "main" : (["main","preload_next"],True),
            "lastiter" : (["lastiter"],False),
            #"store" : (["store"],False) # Don't reschedule store for now (it will fail)
        }

    raise ValueError(f"Invalid microkernel {ukr}")

def get_ukr_specialization_order(ukr : str) -> list[str]:
    if ukr in ["gemm","mm"]:
        return ["preload","main","lastiter","store"]

    raise ValueError(f"Invalid microkernel {ukr}")

class support_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        supported_isas = ["rvv","rvv071","sve","neon","avx128","avx256","avx512"]
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

        self.context.rt = reg_tracker([
            ('greg',self.context.gen.max_gregs),
            ('freg',self.context.gen.max_fregs),
            ('vreg',self.context.gen.max_vregs),
            ('treg',self.context.gen.max_tregs(adt.FP64))])

        self.context.ukr_components = get_ukr_components(self.params["ukr"].value)
        self.context.mru_map = get_ukr_mru_map(self.params["ukr"].value)
        self.context.sched_map = get_ukr_sched_map(self.params["ukr"].value)
        self.context.specialization_order = get_ukr_specialization_order(self.params["ukr"].value)

        self.context.specializer = lsc_specializer(
                model=None,
                gen=self.context.gen,
                rt=self.context.rt)

        self.context.params.update(self.params)
        return []
