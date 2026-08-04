# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

"""
Testsuite for structures and methods involved in determining op support in the
generator
"""

import logging
import unittest
from typing import Any


from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.sme import sme
from ukrgen.matching.math import (
    transformation as tf,
    expression_node,
    operation,
    operand_ref,
    ast_node,
    HW_FMUL_AST,
    HW_FADD_AST,
    HW_FMA_AST,
    HW_FDOTA_AST,
    HW_FOPA_AST,
    HW_MMA_AST,
    extract_deepest_operation,
    solve_requirement,
    generate_variants,
    map_and_match,
    transform_ast,
    simplify_ast,
    get_operands,
    transform_operand,
    decimate_index
)


from ukrgen.support.op import op_support_builder

def print_unified_signatures(sigs):
    """
    Nicely formats instruction signatures, handling both RVV (RISC-V Vector)
    and SME (Scalable Matrix Extension) properties, constraints, and operands.
    """
    def format_enum(e):
        if hasattr(e, 'name'):
            return e.name
        
        e_str = str(e)
        if '<' in e_str and ':' in e_str:
            try:
                return e_str.split('.')[1].split(':')[0]
            except IndexError:
                pass
        return e_str

    def format_constraint(c):
        # 1. SME: min/max value constraints
        if hasattr(c, 'minval') and hasattr(c, 'maxval'):
            c_type = getattr(c, 'what', 'val')
            return f"{c_type} in [{c.minval}..{c.maxval}]"
        
        # 2. RVV: modulo/multiple constraints (e.g., LMUL alignment)
        if hasattr(c, 'what') and hasattr(c, 'multiple'):
            return f"{c.what} % {c.multiple} == 0"
        
        # 3. Fallback: parameter dictionaries
        if hasattr(c, 'params') and c.params:
            if 'minval' in c.params and 'maxval' in c.params:
                return f"[{c.params['minval']}..{c.params['maxval']}]"
            
            c_type = getattr(c, 'what', 'Constraint')
            params_str = ", ".join(f"{k}={v}" for k, v in c.params.items())
            return f"{c_type}({params_str})"
            
        return "constrained"

    for op, sig_list in sigs.items():
        print(f"\n{'='*70}")
        print(f"Operation: {op.upper()} ({len(sig_list)} signatures)")
        print(f"{'='*70}")

        for i, sig in enumerate(sig_list, 1):
            print(f"\n  [{i}] Signature:")

            # Extract Global Signature Modifiers (e.g., MASK, PART, NP)
            if getattr(sig, 'modifiers', None):
                mods = [format_enum(m) for m in sig.modifiers]
                print(f"      Modifiers     : {', '.join(mods)}")

            # Extract Structural Parameters (e.g., bdreg_blocksize, widening_method)
            if getattr(sig, 'structural_params', None):
                params = [f"{k}={format_enum(v)}" for k, v in sig.structural_params.items()]
                print(f"      Struct Params : {', '.join(params)}")

            # Extract and Format Operands
            print("      Operands      :")
            for name, shape in sig.operands.items():
                details = []

                if getattr(shape, 'otype', None):
                    details.append(f"Type: {format_enum(shape.otype)}")
                if getattr(shape, 'rtype', None):
                    details.append(f"Reg: {format_enum(shape.rtype)}")
                if getattr(shape, 'dt', None):
                    details.append(f"DT: {format_enum(shape.dt)}")

                # Operand specific modifiers (e.g., BLOCKLANE)
                if getattr(shape, 'modifiers', None):
                    op_mods = [format_enum(m) for m in shape.modifiers]
                    details.append(f"Mods: [{', '.join(op_mods)}]")

                # Constraints (Lane indexing, LMUL multiples, etc.)
                if getattr(shape, 'value_constraints', None):
                    c_strings = [format_constraint(c) for c in shape.value_constraints]
                    details.append(f"Limit: {', '.join(c_strings)}")

                print(f"        - {name:<12}: {', '.join(details)}")

class test_op_support(unittest.TestCase):

    def test_rvv_arith_ops(self):
        osb = op_support_builder(gen=rvv())

        osb.determine_base_support()

        sigs = {op : getattr(osb.gen, op).get_signatures()
                for op in osb.arith_ops}

        print_unified_signatures(sigs)

    def test_sme_arith_ops(self):
        osb = op_support_builder(gen=sme())

        osb.determine_base_support()

        sigs = {op : getattr(osb.gen, op).get_signatures()
                for op in osb.arith_ops}

        print_unified_signatures(sigs)
        
    def test_sme_ldst_ops(self):
        osb = op_support_builder(gen=sme())

        osb.determine_base_support()

        sigs = {op : getattr(osb.gen, op).get_signatures()
                for op in osb.ldst_ops}

        print_unified_signatures(sigs)
