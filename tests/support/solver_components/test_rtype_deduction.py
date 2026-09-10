# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest

from asmgen.registers import (
    asm_data_type as adt
)
from asmgen.asmblocks.op import (
    register_type as rgt
)

from ukrgen.matching.math import (
    HW_FADD_AST,
    HW_FOPA_AST,
    operand_ref,
    expression_node,
    operation
)

from ukrgen.support.dm_solver import (
    deduce_operand_rtype
)


class test_rtype_deduction(unittest.TestCase):
    """
    Testsuite for solver components
    """
    def setUp(self):
        
        self.scalar_move = expression_node(
                op=operation.MOVE,
                left=operand_ref(name="dst", indices=(None,None)),
                right=operand_ref(name="src", indices=(None,None)))


    def test_deduce_fadd_fp64_a_is_vreg(self):
        """
        Tests that the register type of the first operand to an FP64 FADD is vector
        """
        
        rtype = deduce_operand_rtype(HW_FADD_AST, 'adreg', adt.FP64)

        self.assertEqual(rtype, rgt.VEC)

    def test_deduce_fadd_u8_a_is_vreg(self):
        """
        Tests that the register type of the first operand to an U8 ADD is vector
        """
        
        rtype = deduce_operand_rtype(HW_FADD_AST, 'adreg', adt.UINT8)

        self.assertEqual(rtype, rgt.VEC)

    def test_deduce_fopa_fp64_c_is_treg(self):
        """
        Tests that the register type of the last operand to an FP64 FOPA is tile
        """

        rtype = deduce_operand_rtype(HW_FOPA_AST, 'cdreg', adt.FP64)

        self.assertEqual(rtype, rgt.TILE)

    def test_deduce_move_i32_is_greg(self):
        """
        Test that the register type of a I32 MOVE is GP
        """

        rtype = deduce_operand_rtype(self.scalar_move, 'dst', adt.SINT32)
        self.assertEqual(rtype, rgt.GP)
        rtype = deduce_operand_rtype(self.scalar_move, 'src', adt.SINT32)
        self.assertEqual(rtype, rgt.GP)

    def test_deduce_move_fp32_is_freg(self):
        """
        Test that the register type of a FP32 MOVE is FP
        """

        rtype = deduce_operand_rtype(self.scalar_move, 'dst', adt.FP32)
        self.assertEqual(rtype, rgt.FP)
        rtype = deduce_operand_rtype(self.scalar_move, 'src', adt.FP32)
        self.assertEqual(rtype, rgt.FP)
