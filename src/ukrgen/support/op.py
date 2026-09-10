# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

"""
Classes and datatypes for querying if a capability is
supported in an asmgen generator
"""

import itertools

from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import (
    asm_data_type as adt,
    adt_size
)
from asmgen.asmblocks.op import (
    operation,
    operand_shape,
    operand_type as otype,
    register_type as rtype,
    operation_signature as opsig
)

from ..matching.math import (
    ast_node,
    solve_requirement,
    req_solution_step as rsstep,
    operation as math_op,
    get_operands,
    for_each_expression,
    operand_ref,
    HW_FADD_AST,
    HW_FMUL_AST,
    HW_FMA_AST,
    HW_FDOTA_AST,
    HW_FOPA_AST,
    HW_MMA_AST
)

from .dm_solver import resolved_operation_chain,resolve_ast_solution

from ..components.tile import (
    dimension_type as dimt,
    dimension_properties as dimp,
    tile
)

def infer_hw_tile(gen : asmgen,
                  shape : operand_shape,
                  dt : adt) -> tile:
    """
    Determine operand hw tile from operand shape and

    :param gen: generator to query for additional information
    :param shape: shape of the operand
    :param dt: element data type

    :return: 2D tile representation of the operand
    """


    vla_vec_dp = dimp(dt=dimt.vla, size=1,
                      sdt=dimt.vla, sd_size=1)
    fixed_scalar_dp = dimp(dt=dimt.fixed, size=1,
                           sdt=dimt.fixed, sd_size=1)

    # an immediate is a scalar tile
    if otype.IMMEDIATE == shape.otype:
        return tile(dima=fixed_scalar_dp,
                    dimb=fixed_scalar_dp)

    if otype.REGISTER == shape.otype and \
            shape.rtype in {rtype.GP,rtype.FP,rtype.MASK}:
        return tile(dima=fixed_scalar_dp,
                    dimb=fixed_scalar_dp)

    is_vla = gen.is_vla

    vec_dp = vla_vec_dp

    if not is_vla:
        element_count = gen.simd_size//adt_size(dt)
        #TODO: figure out from signatures how many elements are
        #      subindexable (AVX can only do the first 128bits)
        vec_dp = dimp(dt=dimt.fixed, size=element_count,
                      sdt=dimt.fixed, sd_size=element_count)

    #TODO: figure out how to support BLOCKIDX (SVE), is there a need to
    #      encode more info in 'tile'? Or should that be handled some
    #      other way?

    if otype.REGISTER == shape.otype and \
            shape.rtype in {rtype.VEC}:
        return tile(vec_dp,fixed_scalar_dp)


    # Just VLA in both dimensions. Potentially different actual sizes
    tile_tile = tile(vla_vec_dp,vla_vec_dp)

    # If dims are VLA but a ratio is known, could be something like this?
    # tile_1r2c = tile(dimp(dimt.vla, 1, dimt.vla, 1),
    #                  dimp(dimt.vla, 2, dimt.vla, 2)

    # it might make sense to encode whether or not the 2 dimensions are
    # equal or not in the 'tile' structures when both are VLA

    if not is_vla:
        # TODO: figure out how and where to incode information about rows
        #       and columns
        nrows = gen.tile_rows
        ncols = gen.tile_cols
        tile_tile = tile(dima=dimp(dt=dimt.fixed, size=nrows,
                                   sdt=dimt.fixed, sd_size=nrows),
                         dimb=dimp(dt=dimt.fixed, size=ncols,
                                   sdt=dimt.fixed, sd_size=ncols),
                         )

    # TODO: One dim being VLA and the other being fixed is also possible,
    #       figure out how to handle it

    if otype.REGISTER == shape.otype and \
            shape.rtype in {rtype.TILE}:
        return tile_tile

    raise NotImplementedError("Operand shape not supported")

class op_support:
    """
    Support for an operation with specific operands
    """
    def __init__(self, signature : opsig):

        self.signature = signature

        # data_tile
        self.data_tiles : dict[str, tile] = {}
        self.hw_tiles : dict[str,tile] = {}

        #for name,shape in self.signature.operands.items():
            #self.hw_tiles[name] = infer_hw_tile(shape)


def generate_op_supports(gen : asmgen,
                         signature : opsig) -> list[op_support]:
    """
    Generate op support structs from an op signature
    """
    hw_tiles = {
        name : infer_hw_tile(gen,shape,shape.dt) \
            for name,shape in signature.operands.items()
    }
    supports = []



class op_support_builder:

    ast_op_map = {
        HW_FADD_AST:  'fadd',
        HW_FMUL_AST:  'fmul',
        HW_FMA_AST:   'fma',
        HW_FDOTA_AST: 'fdota',
        HW_FOPA_AST:  'fopa',
        HW_MMA_AST:   'mma'
    }

    def __init__(self, gen : asmgen):
        self.gen = gen

        self.is_vla : bool = False
        self.fregs_in_vregs : bool = False
        self.mregs_are_vregs : bool = False
        self.arith_ops : list[operation]
        self.ldst_ops : list[operation]
        self.move_ops : list[operation]



    def determine_base_support(self):

        # Is this VLA
        self.is_vla = self.gen.is_vla
        self.fregs_in_vregs = self.gen.are_fregs_in_vregs

        # TODO: figure out through signatures
        self.mregs_are_vregs = False

        arith_ops = self.ast_op_map.values()

        self.arith_ops = [op for op in arith_ops if hasattr(self.gen,op)]

        # The generator is guaranteed to have those, but for consistencies sake...
        ldst_ops = ['load','store']
        self.ldst_ops = [op for op in ldst_ops if hasattr(self.gen,op)]

        # Might have multiple different ones at some point, for now all are
        # 'move'
        move_ops = ['move']
        self.move_ops = [op for op in move_ops if hasattr(self.gen,op)]


    def create_base_solutions(self, req : ast_node) -> list[list[rsstep]]:
        available_asts = [ast for ast,op in self.ast_op_map.items() if op in self.arith_ops]

        return solve_requirement(req, available_asts)


    def get_all_ldst_opd_dts(self) -> set[adt]:
        """
        Queries all load and store operations for supported data types and returns a set
        containing all of them
        """

        dts = set()
        for opname in self.ldst_ops:
            sigs : list[opsig] = getattr(self.gen, opname).get_signatures()
            for sig in sigs:
                for opd_name,osh in sig.operands.items():
                    if osh.dt is not None:
                        dts.add(osh.dt)

        return dts

    def match_temporary_dts(self,
                            sstep : rsstep,
                            io_dts : dict[str,adt],
                            ) -> dict[str,adt|None]:
        """
        Extracts all operands from the solution step AST and matches the data
        types of temporary operands to the data types of input/output operands

        :param sstep: solution step with the AST to query for operands
        :param io_dts: Contains data types mapped onto mathematical operands (A,B,C,...)
        :return 
        """
        operands = [sstep.name_mapping[opd] for opd in get_operands(sstep.hw_ast)]
        matched_dts = io_dts.copy()

        for opd in operands:
            if opd not in matched_dts:
                matched_dts[opd] = None

        changed = True
        while changed:
            changed = False

            for lr in for_each_expression(sstep.hw_ast, lambda op : [op.left,op.right]):

                known_dt = None
                expr_opd_names = []

                for expr_opd in lr:
                    if expr_opd is None or not isinstance(expr_opd, operand_ref):
                        continue

                    mapped_name = sstep.name_mapping[expr_opd.name]
                    expr_opd_names.append(mapped_name)

                    if matched_dts.get(mapped_name) is not None:
                        known_dt = matched_dts[mapped_name]

                if known_dt is not None:
                    for name in expr_opd_names:
                        if matched_dts[name] is None:
                            matched_dts[name] = known_dt
                            changed = True

        return matched_dts


    def get_hw_dts(sstep : rsstep,
                   dts : dict[str,adt]
                   ) -> dict[str,adt|None]:
        """
        Get the hw operand (adreg,cdreg,...) data type map for this step using the mathematical
        operand (A,B,C,...) map
        """

    def find_hw_implementations(
            self,
            req : ast_node,
            registry: resolution_registry,
            io_dts : dict[str,adt],
            #) -> dict[frozenset[tuple[str,adt]],list[resolved_operation_chain]]:
            ) -> list[resolved_operation_chain]:
        """
        Find all possible ways of performing the specified chain of mathematical operations with
        all required data movements and computations

        :param gen: Generator to query the available operations of
        :param solution_steps: A solution to the mathematical requirement as determined by 
                               `ukrgen.matching.math.solve_requirement`
        :param dts: data types mapped to operand names as used in the solution
        :param registry: Operand transformation resolution registry
        :return: list of resolved operation chains
        """
        

        solutions = self.create_base_solutions(req)

        #dt_options = self.get_all_ldst_opd_dts()
        #io_operands = list(get_operands(req))


        #resolved_solutions : dict[frozenset[tuple[str,adt]],resolved_operation_chain] = dict()
        resolved_solutions : list[resolved_operation_chain] = []
        
        #for dt_combo in itertools.product(dt_options, repeat=len(io_operands)):
        #    io_dts = dict(zip(io_operands,dt_combo))

        #key = frozenset(zip(io_operands,dt_combo))

        for solution in solutions:
            resolved_solution_steps = []

            dts = io_dts.copy()

            for sstep in solution:
                dts = self.match_temporary_dts(sstep, dts)


            step_invalid = False
            for sstep in solution:

                #hw_dts = [dts[sstep.name_mapping[opd]] for opd in get_operands(sstep.hw_ast)]
                
                step_candidates = resolve_ast_solution(
                        self.gen, sstep, 
                        dts, self.ast_op_map[sstep.hw_ast],
                        registry)
                if not step_candidates:
                    step_invalid = True
                    break

                resolved_solution_steps.append(step_candidates)


            if not step_invalid:
                #if key not in resolved_solutions:
                #    resolved_solutions[key] = []
                for step_resolution_choice in itertools.product(*resolved_solution_steps):
                    #resolved_solutions[key].append(resolved_operation_chain(
                    resolved_solutions.append(resolved_operation_chain(
                            math_chain=solution,
                            resolved_chain=step_resolution_choice))

        return resolved_solutions

