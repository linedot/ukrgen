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


class support_stage(composition_stage):
    def __init__(self, context : gemm_context):
        super().__init__(context)

        self.params["isa"] = None
        self.default_values["isa"] = None
        self.choices["isa"] = ["rvv","rvv071","sve","neon","avx128","avx256","avx512"]

        self.params["op"] = "fma"
        self.default_values["op"] = "fma"
        self.choices["op"] = ["fma","dota","fopa","mma"]

        self.params["ukr"] = "gemm"
        self.default_values["ukr"] = "gemm"
        self.choices["ukr"] = ["gemm","mm"]

    def progress(self):

        assert self.params["isa"] is not None

        isa = self.params["isa"]

        module = importlib.import_module(f"asmgen.asmblocks.{asmgen_modules[isa]}")
        self.context.gen = module.__dict__[asmgen_map[isa]]()
        
        self.context.gen.set_output_inline(yesno=False)

        self.context.rt = reg_tracker([
            ('greg',self.context.gen.max_gregs),
            ('freg',self.context.gen.max_fregs),
            ('vreg',self.context.gen.max_vregs),
            ('treg',self.context.gen.max_tregs(adt.FP64))])

        self.context.params.update(self.params)

        self.context.specializer = lsc_specializer(
                model=None,
                gen=self.context.gen,
                rt=self.context.rt)

        return []
