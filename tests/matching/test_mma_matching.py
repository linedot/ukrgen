# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

import unittest
from typing import Any

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
    solve_requirement
)

def print_solution(solutions : list[list[dict[str,Any]]]):
    
    for i,s in enumerate(solutions):
        print(f"Solution {i}:")
        for j,op in enumerate(s):
            print(f"  Operation {j}")
            print(f"    AST: {op['hw_ast']}")
            print( "    Transformations:")
            for opd, tfs in op['transformations'].items():
                print(f"      {opd}:{'->'.join(t.name for t in tfs)}")
            print( "    Name mapping:")
            for hw_name, req_name in op['name_mapping'].items():
                print(f"      {hw_name}->{req_name}")
            print( "    Index mapping:")
            for hw_idx, req_idx in op['index_mapping'].items():
                print(f"      {hw_idx}->{req_idx}")

class test_mm_matching(unittest.TestCase):

    def setUp(self):

        self.mm_req = expression_node(
            op=operation.MOVE,
            left=operand_ref(name="C", indices=('m', 'n')),
            right=expression_node(
                op=operation.ADD,
                left=operand_ref(name="C", indices=('m','n')),
                right=expression_node(
                    op=operation.REDUCE_SUM,
                    left=expression_node(
                        op=operation.MUL,
                        left=operand_ref(name="A", indices=('m','k')),
                        right=operand_ref(name="B", indices=('k','n'))),
                    reduce_dim='k')
                )
            )

    def test_match_fma(self):
        solutions = solve_requirement(self.mm_req, [HW_FMA_AST])

        self.assertEqual(3, len(solutions))
    
        expected_transformations = {
            (tf.NONE, tf.SCALAR_REDUCE, tf.NONE), # scalar B
            (tf.SCALAR_REDUCE, tf.TRANSPOSE, tf.TRANSPOSE), # scalar A
            (tf.SCALAR_REDUCE, tf.SCALAR_REDUCE, tf.SCALAR_REDUCE), # degen. to Scalar
        }

        actual_transformations = set()
        for s in solutions:
            self.assertEqual(1, len(s))
            tfs = s[0]['transformations']
            self.assertListEqual([1,1,1], [len(tfs[opd])
                                           for opd in {'adreg','bdreg','cdreg'}])
            actual_transformations.add(
                (tfs['adreg'][0],tfs['bdreg'][0],tfs['cdreg'][0])
            )            

        self.assertEqual(expected_transformations, actual_transformations)

    def test_match_fmulfadd(self):
        solutions = solve_requirement(self.mm_req, [HW_FMUL_AST,HW_FADD_AST])

        print_solution(solutions)

    def test_match_fdota(self):
        solutions = solve_requirement(self.mm_req, [HW_FDOTA_AST])

        print_solution(solutions)

    def test_match_fopa(self):
        solutions = solve_requirement(self.mm_req, [HW_FOPA_AST])

        print_solution(solutions)
        

    def test_extract_deepest(self):

        split_ast, dependencies = extract_deepest_operation(
                self.mm_req,
                temp_counter=0)

        self.assertEqual("MOVE(C[m,n],ADD(C[m,n],REDUCE_SUM(T0[m,k,n],k)))", 
                         str(split_ast))
        self.assertEqual(1,len(dependencies))
        self.assertEqual("MOVE(T0[m,k,n],MUL(A[m,k],B[k,n]))", 
                         str(dependencies[0]))

        split_ast, dependencies = extract_deepest_operation(
                split_ast,
                temp_counter=0)

        self.assertEqual("MOVE(C[m,n],ADD(C[m,n],T0[m,n]))", 
                         str(split_ast))
        self.assertEqual(1,len(dependencies))
        self.assertEqual("MOVE(T0[m,n],REDUCE_SUM(T0[m,k,n],k))", 
                         str(dependencies[0]))

        split_ast, dependencies = extract_deepest_operation(
                split_ast,
                temp_counter=0)

        self.assertEqual("MOVE(C[m,n],ADD(C[m,n],T0[m,n]))", 
                         str(split_ast))
        self.assertEqual(0,len(dependencies))


        split_ast, dependencies = extract_deepest_operation(
                split_ast,
                temp_counter=0)

        self.assertEqual("MOVE(C[m,n],ADD(C[m,n],T0[m,n]))", 
                         str(split_ast))
        self.assertEqual(0,len(dependencies))

