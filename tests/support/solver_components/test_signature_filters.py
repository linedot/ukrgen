# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests filtering out operation signatures when narrowing down hw support
"""

import unittest


from asmgen.asmblocks.op import (
    operation_modifier as opmod,
    operand_modifier as opdmod,
    opd3_modifier as d3mod,
    opdna1_modifier as dna1mod,
    register_type as rgt
)
from asmgen.registers import asm_data_type as adt

from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.avx_fma import avx512
from asmgen.asmblocks.neon import neon


from ukrgen.support.data_move import (
    filter_by_op_mods,
    filter_by_operand_count,
    filter_by_operand_mods,
    filter_by_step_requirements,
    dm_step,
    orig_ref,
    temp_ref
)


class test_signature_filters(unittest.TestCase):
    """
    Testsuite for functions that filter out operation signatures based on requirements to
    the operation or it's operands
    """

    def setUp(self):

        self.gensme = sme()
        self.gensve = sve()
        self.genrvv = rvv()
        self.genavx512 = avx512()
        self.genneon = neon()

    def test_sve_fp64_fadd_blocklane_filtered_by_mask(self):
        """
        Tests that the BLOCKLANE version of the FP64 SVE fadd gets filtered out if we require a masked operation
        """

        sigs = self.gensve.fadd.get_signatures()

        sigs = [sig for sig in sigs if sig.operands['adreg'].dt == adt.FP64]

        self.assertEqual(2, len(sigs))
        self.assertTrue(any(opdmod.BLOCKLANE in sig.operands['bdreg'].modifiers for sig in sigs))

        sigs = filter_by_op_mods(sigs, {d3mod.MASK})

        self.assertEqual(1, len(sigs))
        self.assertFalse(any(opdmod.BLOCKLANE in sig.operands['bdreg'].modifiers for sig in sigs))


    def test_rvv_fp64_fma_vv_and_widening_filtered_by_fp64vf(self):
        """
        Tests that both the .vv form and widening version of RVV fma gets filtered out by fp64 .vf requirement
        """

        sigs = self.genrvv.fma.get_signatures()

        sigs = [sig for sig in sigs if sig.operands['cdreg'].dt == adt.FP64]

        self.assertTrue(len(sigs) > 0)
        self.assertTrue(any(sig.operands['bdreg'].rtype == rgt.VEC for sig in sigs))
        self.assertTrue(any(sig.operands['bdreg'].dt == adt.FP32 for sig in sigs))

        sigs = filter_by_operand_mods(sigs,{opdmod.VF},'bdreg',adt.FP64,rgt.FP)

        self.assertTrue(len(sigs) > 0)
        self.assertFalse(any(sig.operands['bdreg'].rtype == rgt.VEC for sig in sigs))
        self.assertFalse(any(sig.operands['bdreg'].dt == adt.FP32 for sig in sigs))

    def test_neon_fp32_nonstruct_filtered_by_4outs(self):
        """
        Tests that neon fp32 loads that write to 4 operands are all structured loads
        """

        sigs = self.genneon.load.get_signatures()

        sigs = [sig for sig in sigs if sig.operands['adreg'].dt == adt.FP32]

        self.assertTrue(len(sigs) > 0)
        self.assertTrue(any(dna1mod.STRUCT not in sig.modifiers for sig in sigs))

        sigs = filter_by_operand_count(sigs, 'load', 0, 4)

        self.assertTrue(len(sigs) > 0)
        self.assertTrue(all(dna1mod.STRUCT in sig.modifiers for sig in sigs))


    def test_sme_fp16_row_move_step_filters_vec_to_vec(self):
        """
        Tests that a data move step that moves a row into a tile filters out vec-to-vec moves
        """

        sigs = self.gensme.move.get_signatures()

        step = dm_step(
                op='move',
                dest=orig_ref(),
                dest_rtype=rgt.TILE,
                src=[temp_ref(tag="T0")],
                src_rtypes=[rgt.VEC],
                op_mod_reqs=set(),
                opd_mod_reqs={orig_ref():{opdmod.ROW}}
                )

        self.assertTrue(len(sigs) > 0)
        self.assertTrue(any(all(opd.rtype == rgt.VEC for _,opd in sig.operands.items()) for sig in sigs))

        sigs = filter_by_step_requirements(sigs, step, adt.FP16)

        self.assertTrue(len(sigs) > 0)
        self.assertFalse(any(all(opd.rtype == rgt.VEC for _,opd in sig.operands.items()) for sig in sigs))
